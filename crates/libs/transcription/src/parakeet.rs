//! Parakeet TDT 0.6B v3 (NVIDIA NeMo) exported to ONNX, run with ONNX Runtime.
//!
//! The model is a directory of ONNX graphs run in sequence: `nemo128.onnx` turns audio
//! into log-mel features, the encoder turns features into acoustic vectors, and the
//! decoder turns those into tokens, which `vocab.txt` maps to text.

use std::io;
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
const DECODER_FILE: &str = "decoder_joint-model.int8.onnx";
const VOCABULARY_FILE: &str = "vocab.txt";

/// The decoder remembers what it has written so far in an LSTM with 2 layers of 640
/// numbers; the graph takes and returns that memory as two tensors of this shape
/// (layers, batch, numbers).
const DECODER_STATE_SHAPE: [usize; 3] = [2, 1, 640];
const DECODER_STATE_LEN: usize = DECODER_STATE_SHAPE[0] * DECODER_STATE_SHAPE[2];

/// How many frames the decoder can choose to skip after each step: 0 to 4.
const SKIP_CHOICES: usize = 5;

/// Most tokens written for one encoder frame before moving on. The model can keep
/// predicting "stay on this frame"; this bounds that loop. onnx-asr reads it from
/// `config.json` with the same default; the shipped `config.json` does not set it.
const MAX_TOKENS_PER_FRAME: usize = 10;

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
pub enum LoadModelError {
    #[error("cannot load parakeet model {}", path.display())]
    Graph {
        path: PathBuf,
        #[source]
        source: ort::Error,
    },
    #[error("cannot read parakeet vocabulary {}", path.display())]
    ReadVocabulary {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("malformed parakeet vocabulary {} at line {line}", path.display())]
    MalformedVocabulary { path: PathBuf, line: usize },
    #[error("parakeet vocabulary {} does not end with <blk>", path.display())]
    MissingBlank { path: PathBuf },
}

#[derive(Debug, thiserror::Error)]
pub enum TranscribeError {
    #[error("audio too short: {samples} samples, need at least {MIN_SAMPLES}")]
    TooShort { samples: usize },
    #[error("parakeet inference failed")]
    Inference(#[source] ort::Error),
    #[error("model output {0:?} is missing")]
    MissingOutput(&'static str),
    #[error(
        "decoder returned {scores} scores, expected {vocabulary} tokens and {SKIP_CHOICES} \
         skips; vocab.txt does not match the decoder"
    )]
    VocabularyMismatch { vocabulary: usize, scores: usize },
}

/// The Parakeet ONNX graphs loaded into ONNX Runtime sessions, running on the CPU.
pub struct ParakeetModel {
    preprocessor: Session,
    encoder: Session,
    decoder: Session,
    /// Token pieces by id, e.g. `"▁Kin"`; the last one is the blank `<blk>`.
    vocabulary: Vec<String>,
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
            vocabulary: read_vocabulary(&dir.join(VOCABULARY_FILE))?,
            preprocessor: session(&dir.join(PREPROCESSOR_FILE))?,
            encoder: session(&dir.join(ENCODER_FILE))?,
            decoder: session(&dir.join(DECODER_FILE))?,
        })
    }

    /// Transcribes 16 kHz mono samples. Parakeet detects the language itself and takes
    /// no hint.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, TranscribeError> {
        let features = self.features(samples)?;
        let encoded = self.encode(&features)?;
        let tokens = self.decode(&encoded)?;
        let pieces: String = tokens
            .iter()
            .map(|&token| self.vocabulary[token].as_str())
            .collect();
        Ok(text_from_pieces(&pieces))
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

    /// Turns encoder frames into token ids with greedy TDT decoding.
    ///
    /// TDT ("token and duration transducer") predicts two things at each step: the next
    /// token, or blank for "nothing new here", and how many encoder frames to skip
    /// before the next step (0 to 4). Skipping over frames where nothing new is said is
    /// what makes it faster than decoders that visit every frame.
    ///
    /// Follows onnx-asr's `_AsrWithTransducerDecoding._decoding` (Ilya Stupakov, MIT),
    /// which drives these ONNX graphs and implements NeMo's greedy TDT decoding.
    fn decode(&mut self, encoded: &Encoded) -> Result<Vec<usize>, TranscribeError> {
        let blank = self.vocabulary.len() - 1;
        let mut tokens = Vec::new();
        let mut state = DecoderState::initial();
        let mut vector = vec![0.0; ENCODER_DIM];
        let mut frame = 0;
        let mut tokens_on_frame = 0;

        while frame < encoded.frames {
            for (component, value) in vector.iter_mut().enumerate() {
                *value = encoded.data[component * encoded.frames + frame];
            }
            // The decoder reads the last written token; blank stands for "none yet".
            let previous = tokens.last().copied().unwrap_or(blank);
            let step = self.decode_step(&vector, previous, &state)?;

            // The new state includes `previous`, which only counts once a real token
            // follows it; after a blank the decoder keeps its old state.
            if step.token != blank {
                tokens.push(step.token);
                state = step.state;
                tokens_on_frame += 1;
            }
            if step.skip > 0 {
                frame += step.skip;
                tokens_on_frame = 0;
            } else if step.token == blank || tokens_on_frame == MAX_TOKENS_PER_FRAME {
                frame += 1;
                tokens_on_frame = 0;
            }
        }
        Ok(tokens)
    }

    /// Runs the decoder once on one encoder vector and the previously written token.
    fn decode_step(
        &mut self,
        vector: &[f32],
        previous: usize,
        state: &DecoderState,
    ) -> Result<Step, TranscribeError> {
        let targets = [previous as i32];
        let target_length = [1_i32];
        let inputs = ort::inputs![
            "encoder_outputs" => TensorRef::from_array_view(([1, ENCODER_DIM, 1], vector))
                .map_err(TranscribeError::Inference)?,
            "targets" => TensorRef::from_array_view(([1, 1], targets.as_slice()))
                .map_err(TranscribeError::Inference)?,
            "target_length" => TensorRef::from_array_view(([1], target_length.as_slice()))
                .map_err(TranscribeError::Inference)?,
            "input_states_1" => TensorRef::from_array_view((DECODER_STATE_SHAPE, state.first.as_slice()))
                .map_err(TranscribeError::Inference)?,
            "input_states_2" => TensorRef::from_array_view((DECODER_STATE_SHAPE, state.second.as_slice()))
                .map_err(TranscribeError::Inference)?,
        ];
        let outputs = self
            .decoder
            .run(inputs)
            .map_err(TranscribeError::Inference)?;

        let extract = |name: &'static str| -> Result<Vec<f32>, TranscribeError> {
            let (_, data) = outputs
                .get(name)
                .ok_or(TranscribeError::MissingOutput(name))?
                .try_extract_tensor::<f32>()
                .map_err(TranscribeError::Inference)?;
            Ok(data.to_vec())
        };
        // One score per vocabulary token, followed by one score per skip (0, 1, 2, ...).
        let scores = extract("outputs")?;
        if scores.len() != self.vocabulary.len() + SKIP_CHOICES {
            return Err(TranscribeError::VocabularyMismatch {
                vocabulary: self.vocabulary.len(),
                scores: scores.len(),
            });
        }
        let (token_scores, skip_scores) = scores.split_at(self.vocabulary.len());
        Ok(Step {
            token: argmax(token_scores),
            skip: argmax(skip_scores),
            state: DecoderState {
                first: extract("output_states_1")?,
                second: extract("output_states_2")?,
            },
        })
    }
}

/// The decoder's memory of the tokens written so far.
struct DecoderState {
    first: Vec<f32>,
    second: Vec<f32>,
}

impl DecoderState {
    fn initial() -> Self {
        Self {
            first: vec![0.0; DECODER_STATE_LEN],
            second: vec![0.0; DECODER_STATE_LEN],
        }
    }
}

/// What one decoder step predicts.
struct Step {
    token: usize,
    /// Encoder frames to move forward; 0 means another token may follow on this frame.
    skip: usize,
    /// Decoder state after reading the previous token.
    state: DecoderState,
}

/// Index of the highest score. `scores` must not be empty.
fn argmax(scores: &[f32]) -> usize {
    let mut best = 0;
    for (index, &score) in scores.iter().enumerate() {
        if score > scores[best] {
            best = index;
        }
    }
    best
}

/// Turns joined token pieces into text, as onnx-asr does. `▁` marks the start of a word
/// and becomes a space, but a space is only kept between a character and a following
/// letter, digit or `_`: stray `▁` pieces before punctuation or another space vanish.
fn text_from_pieces(pieces: &str) -> String {
    let chars: Vec<char> = pieces.replace('▁', " ").chars().collect();
    let mut text = String::new();
    for (index, &char) in chars.iter().enumerate() {
        let before_word = chars
            .get(index + 1)
            .is_some_and(|next| next.is_alphanumeric() || *next == '_');
        if char == ' ' && (index == 0 || !before_word) {
            continue;
        }
        text.push(char);
    }
    text
}

/// Reads `vocab.txt`: one `piece id` pair per line, ids counting up from 0, ending with
/// the blank token `<blk>`.
fn read_vocabulary(path: &Path) -> Result<Vec<String>, LoadModelError> {
    let content =
        std::fs::read_to_string(path).map_err(|source| LoadModelError::ReadVocabulary {
            path: path.to_path_buf(),
            source,
        })?;
    let malformed = |line: usize| LoadModelError::MalformedVocabulary {
        path: path.to_path_buf(),
        line,
    };

    let mut vocabulary = Vec::new();
    for (index, line) in content.lines().enumerate() {
        let (piece, id) = line.rsplit_once(' ').ok_or_else(|| malformed(index + 1))?;
        if id.parse::<usize>() != Ok(index) {
            return Err(malformed(index + 1));
        }
        vocabulary.push(piece.to_string());
    }
    if vocabulary.last().map(String::as_str) != Some("<blk>") {
        return Err(LoadModelError::MissingBlank {
            path: path.to_path_buf(),
        });
    }
    Ok(vocabulary)
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
    build().map_err(|source| LoadModelError::Graph {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::text_from_pieces;

    #[test]
    fn word_starts_become_spaces_except_at_the_beginning() {
        assert_eq!(text_from_pieces("▁Heute▁Abend"), "Heute Abend");
        assert_eq!(text_from_pieces("▁Kin▁o"), "Kin o");
    }

    #[test]
    fn spaces_before_punctuation_and_at_the_end_vanish() {
        assert_eq!(text_from_pieces("▁Kino▁."), "Kino.");
        assert_eq!(text_from_pieces("▁it▁'s"), "it's");
        assert_eq!(text_from_pieces("▁Ende▁"), "Ende");
    }

    #[test]
    fn a_standalone_word_start_does_not_double_the_space() {
        assert_eq!(text_from_pieces("▁Kino▁▁ins"), "Kino ins");
    }

    #[test]
    fn empty_input_gives_empty_text() {
        assert_eq!(text_from_pieces(""), "");
    }
}
