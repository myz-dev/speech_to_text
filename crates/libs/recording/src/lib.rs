//! Audio input for the speech models: loading WAV files and converting audio to
//! 16 kHz mono `f32`. Microphone capture and macOS permission handling follow in M3.
//!
//! See [PLAN.md](../../../PLAN.md).

mod convert;
mod wav;

pub use convert::{ResampleError, resample, to_mono};
pub use wav::{LoadWavError, load_wav};

/// Sample rate expected by both Whisper and Parakeet.
pub const SAMPLE_RATE: u32 = 16_000;
