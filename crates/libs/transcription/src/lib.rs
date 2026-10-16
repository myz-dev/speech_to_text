//! Speech-to-text from 16 kHz mono samples.
//! Backends: whisper.cpp via whisper-rs (Metal, M1) and
//! Parakeet TDT via ONNX Runtime (M2). See [PLAN.md](../../../PLAN.md).
//! [`Model`] wraps either one for callers that let the user choose.

mod model;
pub mod parakeet;
pub mod whisper;

pub use model::{
    Engine, LoadModelError, Model, PARAKEET_MODEL_DIR, TranscribeError, Transcript,
    WHISPER_MODEL_FILE,
};

/// Where a model runs its inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// The GPU backend the crate is built with (Metal on macOS).
    Gpu,
    Cpu,
}
