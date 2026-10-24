//! Audio input for the speech models: loading WAV files, recording from the microphone,
//! converting audio to 16 kHz mono `f32`, and detecting speech in it: by loudness
//! ([`silence`]) or with a speech classifier ([`vad`]).
//!
//! See [PLAN.md](../../../PLAN.md).

mod convert;
mod microphone;
mod wav;

pub mod silence;
pub mod vad;

pub use convert::{ResampleError, resample, to_mono};
pub use microphone::{Flow, RecordError, Recording, record, record_streaming};
pub use wav::{LoadWavError, SaveWavError, load_wav, save_wav};

/// Sample rate expected by both Whisper and Parakeet.
pub const SAMPLE_RATE: u32 = 16_000;
