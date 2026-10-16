//! One interface over both engines, for callers that let the user choose.

use std::path::{Path, PathBuf};

use crate::Device;
use crate::parakeet::{self, ParakeetModel};
use crate::whisper::{self, WhisperModel};

/// Whisper model file inside a models directory.
pub const WHISPER_MODEL_FILE: &str = "ggml-large-v3-turbo.bin";

/// Parakeet model directory inside a models directory.
pub const PARAKEET_MODEL_DIR: &str = "parakeet-tdt-0.6b-v3";

/// A speech-to-text engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// whisper.cpp; detects the language and takes a hint.
    Whisper,
    /// Parakeet TDT via ONNX Runtime; faster, but reports no language.
    Parakeet,
}

impl Engine {
    /// Where this engine's model lives inside the `models` directory.
    pub fn model_path(self, models: &Path) -> PathBuf {
        match self {
            Engine::Whisper => models.join(WHISPER_MODEL_FILE),
            Engine::Parakeet => models.join(PARAKEET_MODEL_DIR),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoadModelError {
    #[error(transparent)]
    Whisper(#[from] whisper::LoadModelError),
    #[error(transparent)]
    Parakeet(#[from] parakeet::LoadModelError),
}

#[derive(Debug, thiserror::Error)]
pub enum TranscribeError {
    #[error(transparent)]
    Whisper(#[from] whisper::TranscribeError),
    #[error(transparent)]
    Parakeet(#[from] parakeet::TranscribeError),
}

/// A loaded model of either engine.
pub enum Model {
    Whisper(WhisperModel),
    Parakeet(ParakeetModel),
}

/// Text of one transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    pub text: String,
    /// Language code the engine transcribed in, e.g. `"de"`; `None` for Parakeet.
    pub language: Option<&'static str>,
}

impl Model {
    /// Loads `engine`'s model from the `models` directory. `device` applies to
    /// Whisper; Parakeet always runs on the CPU.
    pub fn load(engine: Engine, models: &Path, device: Device) -> Result<Self, LoadModelError> {
        let path = engine.model_path(models);
        Ok(match engine {
            Engine::Whisper => Model::Whisper(WhisperModel::load(&path, device)?),
            Engine::Parakeet => Model::Parakeet(ParakeetModel::load(&path)?),
        })
    }

    /// Transcribes 16 kHz mono `samples`. `language` (a code like `"de"`) is a hint
    /// for Whisper, which detects the language without one; Parakeet ignores it.
    ///
    /// Takes `&mut self` because Parakeet's ONNX Runtime sessions require it to run; no
    /// state is kept between calls.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<Transcript, TranscribeError> {
        Ok(match self {
            Model::Whisper(model) => {
                let transcript = model.transcribe(samples, language)?;
                Transcript {
                    text: transcript.text,
                    language: Some(transcript.language),
                }
            }
            Model::Parakeet(model) => Transcript {
                text: model.transcribe(samples)?,
                language: None,
            },
        })
    }
}
