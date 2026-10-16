//! Audio input for the speech models: loading WAV files, recording from the microphone,
//! converting audio to 16 kHz mono `f32`, and detecting speech in it.
//!
//! See [PLAN.md](../../../PLAN.md).

mod convert;
mod microphone;
mod wav;

pub mod silence;

pub use convert::{ResampleError, resample, to_mono};
pub use microphone::{RecordError, Recording, record};
pub use wav::{LoadWavError, SaveWavError, load_wav, save_wav};

/// Sample rate expected by both Whisper and Parakeet.
pub const SAMPLE_RATE: u32 = 16_000;
