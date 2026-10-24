//! Speech detection with Silero VAD, a small neural network that tells speech from
//! other sounds (coughs, typing, music) where an energy detector ([`crate::silence`])
//! only sees loudness.
//!
//! The model reads 16 kHz mono audio in chunks of [`CHUNK_SAMPLES`] (32 ms) and returns
//! for each chunk the probability that it holds speech. It is recurrent: each chunk is
//! judged together with what came before, which the model returns as a [`State`] that
//! is passed into the next [`Vad::step`].

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::TensorRef;

use crate::SAMPLE_RATE;

/// Samples judged per step: 32 ms at 16 kHz, the chunk length the model is trained on.
pub const CHUNK_SAMPLES: usize = 512;

/// Samples of the previous chunk put before each chunk, so the graph reads 576 samples
/// per step. It needs them: given a chunk of 512 samples alone, it returns a probability
/// near 0 for every chunk, speech included, without an error.
const CONTEXT_SAMPLES: usize = 64;

/// Shape of the recurrent memory (layers, batch, numbers) the model takes and returns.
const MEMORY_SHAPE: [usize; 3] = [2, 1, 128];
const MEMORY_LEN: usize = MEMORY_SHAPE[0] * MEMORY_SHAPE[2];

#[derive(Debug, thiserror::Error)]
#[error("cannot load silero vad model {}", path.display())]
pub struct LoadVadError {
    pub path: PathBuf,
    #[source]
    pub source: ort::Error,
}

#[derive(Debug, thiserror::Error)]
pub enum VadError {
    #[error("silero vad inference failed")]
    Inference(#[source] ort::Error),
    #[error("model output {0:?} is missing")]
    MissingOutput(&'static str),
    #[error("model output {output:?} has {len} values, expected {expected}")]
    WrongOutputLen {
        output: &'static str,
        len: usize,
        expected: usize,
    },
}

/// The Silero VAD graph loaded into an ONNX Runtime session, running on the CPU.
pub struct Vad {
    session: Session,
}

/// What the model remembers from the chunks before the next one. Starts empty
/// ([`State::default`]) at the beginning of a recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    /// The model's recurrent memory, as it returned it.
    memory: [f32; MEMORY_LEN],
    /// The last [`CONTEXT_SAMPLES`] samples of the previous chunk.
    context: [f32; CONTEXT_SAMPLES],
}

impl Default for State {
    fn default() -> Self {
        Self {
            memory: [0.0; MEMORY_LEN],
            context: [0.0; CONTEXT_SAMPLES],
        }
    }
}

impl Vad {
    /// Loads the model file, e.g. `models/silero-vad/silero_vad.onnx`.
    pub fn open(path: &Path) -> Result<Self, LoadVadError> {
        let session = onnx::session(path).map_err(|source| LoadVadError {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self { session })
    }

    /// The probability (0 to 1) that `chunk` holds speech, given the chunks before it in
    /// `state`, and the state for the next chunk.
    ///
    /// Takes `&mut self` because ONNX Runtime sessions require it to run; everything the
    /// model remembers is in `state`.
    pub fn step(
        &mut self,
        state: &State,
        chunk: &[f32; CHUNK_SAMPLES],
    ) -> Result<(State, f32), VadError> {
        let mut input = [0.0; CONTEXT_SAMPLES + CHUNK_SAMPLES];
        input[..CONTEXT_SAMPLES].copy_from_slice(&state.context);
        input[CONTEXT_SAMPLES..].copy_from_slice(chunk);
        let sample_rate = [i64::from(SAMPLE_RATE)];

        let inputs = ort::inputs![
            "input" => TensorRef::from_array_view(([1, input.len()], input.as_slice()))
                .map_err(VadError::Inference)?,
            "state" => TensorRef::from_array_view((MEMORY_SHAPE, state.memory.as_slice()))
                .map_err(VadError::Inference)?,
            "sr" => TensorRef::from_array_view(([0usize; 0], sample_rate.as_slice()))
                .map_err(VadError::Inference)?,
        ];
        let outputs = self.session.run(inputs).map_err(VadError::Inference)?;

        let [probability] = output::<1>(&outputs, "output")?;
        let memory = output::<MEMORY_LEN>(&outputs, "stateN")?;
        let next = State {
            memory,
            context: next_context(chunk),
        };
        Ok((next, probability))
    }

    /// The speech probability of every whole chunk of `audio` (16 kHz mono), in order.
    /// A last chunk shorter than [`CHUNK_SAMPLES`] is not judged.
    pub fn probabilities(&mut self, audio: &[f32]) -> Result<Vec<f32>, VadError> {
        let mut state = State::default();
        let mut probabilities = Vec::with_capacity(audio.len() / CHUNK_SAMPLES);
        for chunk in audio.chunks_exact(CHUNK_SAMPLES) {
            // `chunks_exact` only yields chunks of exactly CHUNK_SAMPLES.
            let chunk = chunk.try_into().expect("chunk has CHUNK_SAMPLES samples");
            let (next, probability) = self.step(&state, chunk)?;
            state = next;
            probabilities.push(probability);
        }
        Ok(probabilities)
    }
}

/// The context for the chunk after `chunk`: its last [`CONTEXT_SAMPLES`] samples.
fn next_context(chunk: &[f32; CHUNK_SAMPLES]) -> [f32; CONTEXT_SAMPLES] {
    let mut context = [0.0; CONTEXT_SAMPLES];
    context.copy_from_slice(&chunk[CHUNK_SAMPLES - CONTEXT_SAMPLES..]);
    context
}

/// The values of the model output `name`, which must hold exactly `N` of them.
fn output<const N: usize>(
    outputs: &ort::session::SessionOutputs,
    name: &'static str,
) -> Result<[f32; N], VadError> {
    let (_, values) = outputs
        .get(name)
        .ok_or(VadError::MissingOutput(name))?
        .try_extract_tensor::<f32>()
        .map_err(VadError::Inference)?;
    values.try_into().map_err(|_| VadError::WrongOutputLen {
        output: name,
        len: values.len(),
        expected: N,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_next_context_is_the_end_of_the_chunk() {
        let chunk: [f32; CHUNK_SAMPLES] = std::array::from_fn(|i| i as f32);

        let context = next_context(&chunk);

        let expected: [f32; CONTEXT_SAMPLES] =
            std::array::from_fn(|i| (CHUNK_SAMPLES - CONTEXT_SAMPLES + i) as f32);
        assert_eq!(context, expected);
    }
}
