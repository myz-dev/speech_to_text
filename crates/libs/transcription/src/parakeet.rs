//! Parakeet TDT 0.6B v3 (NVIDIA NeMo) exported to ONNX, run with ONNX Runtime.
//!
//! The model is a directory of ONNX graphs run in sequence: `nemo128.onnx` turns audio
//! into log-mel features, the encoder turns features into acoustic vectors, and the
//! decoder turns those into tokens.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::AutoDevicePolicy;
use ort::value::TensorRef;

/// Mel frequency bands per feature frame.
pub const MEL_BINS: usize = 128;

/// Samples between two consecutive feature frames: 10 ms at 16 kHz. Each frame itself
/// analyses a 25 ms window, so neighbouring frames overlap.
pub const HOP_SAMPLES: usize = 160;

/// Shortest input the preprocessor accepts. The graph extends each edge of the audio by
/// mirroring 256 samples (half its 512-sample FFT frame); the bundled ONNX Runtime 1.31
/// rejects that unless the input is longer than 256 samples. Older runtimes accept
/// shorter input.
pub const MIN_SAMPLES: usize = 257;

/// Numbers per encoder output vector.
pub const ENCODER_DIM: usize = 1024;

/// Feature frames per encoder frame: the encoder outputs one vector per 80 ms.
pub const SUBSAMPLING: usize = 8;

const PREPROCESSOR_FILE: &str = "nemo128.onnx";
const ENCODER_FILE: &str = "encoder-model.int8.onnx";

/// The process's ONNX Runtime environment, shared by all sessions. ort allows only one
/// environment per process, and never a second one, even after the first is dropped.
static ENVIRONMENT: Mutex<Option<Environment>> = Mutex::new(None);

/// Returns the shared environment, creating it on first use.
fn environment() -> Result<Environment, ort::Error> {
    // The lock only guards an `Option` that is written in one step, so a panic while
    // holding it cannot leave it inconsistent; ignore poisoning.
    let mut shared = ENVIRONMENT.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(environment) = shared.as_ref() {
        return Ok(environment.clone());
    }
    let environment = ort::init().build()?;
    *shared = Some(environment.clone());
    Ok(environment)
}

#[derive(Debug, thiserror::Error)]
#[error("cannot load parakeet model {}", path.display())]
pub struct LoadModelError {
    path: PathBuf,
    #[source]
    source: ort::Error,
}

#[derive(Debug, thiserror::Error)]
pub enum TranscribeError {
    #[error("audio too short: {samples} samples, need at least {MIN_SAMPLES}")]
    TooShort { samples: usize },
    #[error("parakeet inference failed")]
    Inference(#[source] ort::Error),
    #[error("model output {0:?} is missing")]
    MissingOutput(&'static str),
}

/// The Parakeet ONNX graphs loaded into ONNX Runtime sessions, running on the CPU.
pub struct ParakeetModel {
    preprocessor: Session,
    encoder: Session,
}

/// Log-mel spectrogram: how much energy each of [`MEL_BINS`] frequency bands holds, one
/// frame every 10 ms.
///
/// `data` is row-major, one row of `frames` values per mel bin. Each bin is normalized to
/// mean 0 and standard deviation 1 over the clip, as the encoder expects.
#[derive(Debug, Clone, PartialEq)]
pub struct Features {
    pub data: Vec<f32>,
    pub frames: usize,
}

/// Encoder output: one vector of [`ENCODER_DIM`] numbers per 80 ms of audio, describing
/// what is being said there in a form the decoder can read.
///
/// `data` is row-major as the graph returns it: one row of `frames` values per vector
/// component, so the vector of frame `t` is every `frames`-th value starting at `t`.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    pub data: Vec<f32>,
    pub frames: usize,
}

impl ParakeetModel {
    /// Loads the model files from `dir`, e.g. `models/parakeet-tdt-0.6b-v3`.
    pub fn load(dir: &Path) -> Result<Self, LoadModelError> {
        Ok(Self {
            preprocessor: session(&dir.join(PREPROCESSOR_FILE))?,
            encoder: session(&dir.join(ENCODER_FILE))?,
        })
    }

    /// Computes the log-mel features of 16 kHz mono samples.
    ///
    /// Takes `&mut self` because ONNX Runtime sessions require it to run; no state is
    /// kept between calls.
    pub fn features(&mut self, samples: &[f32]) -> Result<Features, TranscribeError> {
        if samples.len() < MIN_SAMPLES {
            return Err(TranscribeError::TooShort {
                samples: samples.len(),
            });
        }

        // The graph takes a batch of clips; we pass a batch of one.
        let lengths = [samples.len() as i64];
        let inputs = ort::inputs![
            "waveforms" => TensorRef::from_array_view(([1, samples.len()], samples))
                .map_err(TranscribeError::Inference)?,
            "waveforms_lens" => TensorRef::from_array_view(([1], lengths.as_slice()))
                .map_err(TranscribeError::Inference)?,
        ];
        let outputs = self
            .preprocessor
            .run(inputs)
            .map_err(TranscribeError::Inference)?;

        // Shape [batch, MEL_BINS, frames].
        let (shape, data) = outputs
            .get("features")
            .ok_or(TranscribeError::MissingOutput("features"))?
            .try_extract_tensor::<f32>()
            .map_err(TranscribeError::Inference)?;
        Ok(Features {
            data: data.to_vec(),
            frames: shape[2] as usize,
        })
    }

    /// Runs the encoder over the whole clip. Produces
    /// `features.frames.div_ceil(SUBSAMPLING)` frames.
    ///
    /// Takes `&mut self` because ONNX Runtime sessions require it to run; no state is
    /// kept between calls.
    pub fn encode(&mut self, features: &Features) -> Result<Encoded, TranscribeError> {
        let lengths = [features.frames as i64];
        let inputs = ort::inputs![
            "audio_signal" => TensorRef::from_array_view(
                ([1, MEL_BINS, features.frames], features.data.as_slice()),
            )
            .map_err(TranscribeError::Inference)?,
            "length" => TensorRef::from_array_view(([1], lengths.as_slice()))
                .map_err(TranscribeError::Inference)?,
        ];
        let outputs = self
            .encoder
            .run(inputs)
            .map_err(TranscribeError::Inference)?;

        // Shape [batch, ENCODER_DIM, frames].
        let (shape, data) = outputs
            .get("outputs")
            .ok_or(TranscribeError::MissingOutput("outputs"))?
            .try_extract_tensor::<f32>()
            .map_err(TranscribeError::Inference)?;
        Ok(Encoded {
            data: data.to_vec(),
            frames: shape[2] as usize,
        })
    }
}

/// Loads one ONNX graph into a CPU session of the shared environment.
///
/// ort's default device policy hands parts of the encoder to CoreML on macOS, even
/// without its `coreml` feature. That makes session creation ~4× slower, encoding ~1.8×
/// slower, and changes the output; `PreferCPU` keeps the whole graph on the CPU.
fn session(path: &Path) -> Result<Session, LoadModelError> {
    let build = || -> Result<Session, ort::Error> {
        Session::builder(&environment()?)?
            .with_auto_device(AutoDevicePolicy::PreferCPU)?
            .commit_from_file(path)
    };
    build().map_err(|source| LoadModelError {
        path: path.to_path_buf(),
        source,
    })
}
