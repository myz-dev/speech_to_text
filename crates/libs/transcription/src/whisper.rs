use std::path::{Path, PathBuf};

use crate::Device;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Debug, thiserror::Error)]
#[error("cannot load whisper model {}", path.display())]
pub struct LoadModelError {
    path: PathBuf,
    #[source]
    source: whisper_rs::WhisperError,
}

#[derive(Debug, thiserror::Error)]
pub enum TranscribeError {
    #[error("unknown language {0:?}")]
    UnknownLanguage(String),
    #[error("no audio samples")]
    NoSamples,
    #[error("whisper inference failed")]
    Inference(#[source] whisper_rs::WhisperError),
}

/// A whisper.cpp model loaded for inference on a [`Device`]; GPU means Metal
/// (whisper-rs feature `metal`).
///
/// Loading takes seconds and gigabytes; load once and reuse for every transcription.
pub struct WhisperModel {
    context: WhisperContext,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    pub text: String,
    /// Language whisper decoded with, always as a code like `"de"`.
    ///
    /// - When a hint is passed during transcription, this field equals the hint as a code
    ///   (`"german"` becomes `"de"`); whisper does not auto-detect a language in this case.
    /// - When no hint is passed, this field is the language that whisper auto-detected.
    ///
    /// English-only (`.en`) models always report `"en"`.
    pub language: &'static str,
}

impl WhisperModel {
    /// Loads a ggml model file, e.g. `ggml-large-v3-turbo.bin`, onto `device`.
    pub fn load(path: &Path, device: Device) -> Result<Self, LoadModelError> {
        // whisper.cpp logs every loading and inference step to stderr; discard them.
        whisper_rs::install_logging_hooks();

        let mut params = WhisperContextParameters::default();
        params.use_gpu(device == Device::Gpu);
        let context =
            WhisperContext::new_with_params(path, params).map_err(|source| LoadModelError {
                path: path.to_path_buf(),
                source,
            })?;
        Ok(Self { context })
    }

    /// Transcribes 16 kHz mono samples.
    ///
    /// `language` is a code like `"de"` or a name like `"german"`; `None` lets whisper
    /// detect it from the first 30 seconds.
    pub fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<Transcript, TranscribeError> {
        if samples.is_empty() {
            return Err(TranscribeError::NoSamples);
        }
        if let Some(language) = language {
            // `get_lang_id` panics on interior NUL bytes, so reject those first.
            if language.contains('\0') || whisper_rs::get_lang_id(language).is_none() {
                return Err(TranscribeError::UnknownLanguage(language.to_string()));
            }
        }

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(language);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);

        // A state holds the working memory of one inference run; the model weights are shared.
        let mut state = self
            .context
            .create_state()
            .map_err(TranscribeError::Inference)?;
        state
            .full(params, samples)
            .map_err(TranscribeError::Inference)?;

        let mut text = String::new();
        for segment in state.as_iter() {
            let segment_text = segment.to_str_lossy().map_err(TranscribeError::Inference)?;
            text.push_str(&segment_text);
        }

        // Invariant: whisper sets a valid language id after `full`, hint or detected.
        let language = whisper_rs::get_lang_str(state.full_lang_id_from_state())
            .expect("whisper reports a known language id after transcription");

        Ok(Transcript {
            text: text.trim().to_string(),
            language,
        })
    }
}
