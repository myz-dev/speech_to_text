//! Speech detection with Silero VAD, a small neural network that tells speech from
//! other sounds (coughs, typing, music) where an energy detector ([`crate::silence`])
//! only sees loudness.
//!
//! The model reads 16 kHz mono audio in chunks of [`CHUNK_SAMPLES`] (32 ms) and returns
//! for each chunk the probability that it holds speech. It is recurrent: each chunk is
//! judged together with what came before, which the model returns as a [`State`] that
//! is passed into the next [`Vad::step`].
//!
//! [`decide`] turns the probabilities into the events of the energy detector
//! ([`Event`]): speech starts after a few chunks of high probability and ends after a
//! hangover of lower ones. The thresholds come from recordings of speech, coughs,
//! throat-clearing and typing (FINDINGS.md, "Silero VAD against coughs and typing").

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::TensorRef;

use crate::SAMPLE_RATE;
use crate::silence::Event;

/// Samples judged per step: 32 ms at 16 kHz, the chunk length the model is trained on.
pub const CHUNK_SAMPLES: usize = 512;

/// Length of one chunk in milliseconds.
pub const CHUNK_MS: u32 = 32;

/// Whole chunks in `ms` milliseconds.
pub const fn chunks(ms: u32) -> u32 {
    ms / CHUNK_MS
}

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

/// When the speech probabilities count as speech. Durations are in chunks of
/// [`CHUNK_MS`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// A chunk above this probability may start speech. High, because a cough scores
    /// up to 0.9 for a few chunks; speech scores above it for hundreds of milliseconds.
    pub start_probability: f32,
    /// Speech continues while chunks stay at or above this probability. Lower than the
    /// start, so a probability near the start does not toggle between speech and
    /// silence, and quiet word endings do not run down the hangover.
    pub end_probability: f32,
    /// Chunks in a row above [`Params::start_probability`] before speech counts. Coughs
    /// stay above it for at most 2 chunks, a single short word ("Ja.") for 7.
    pub start_chunks: u32,
    /// Chunks in a row below [`Params::end_probability`] before speech has ended.
    pub hangover_chunks: u32,
    /// Chunks without any speech before [`Event::NoSpeechTimeout`].
    pub no_speech_chunks: u32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            start_probability: 0.9,
            end_probability: 0.5,
            start_chunks: 4,
            hangover_chunks: chunks(1_200),
            no_speech_chunks: chunks(10_000),
        }
    }
}

/// State of the speech decision, passed into [`decide`] and returned by it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Detector {
    phase: Phase,
    chunks_seen: u32,
    heard_speech: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    /// `likely_run` chunks in a row were above the start probability.
    Silence { likely_run: u32 },
    /// `unlikely_run` chunks in a row were below the end probability.
    Speech { unlikely_run: u32 },
}

impl Default for Phase {
    fn default() -> Self {
        Phase::Silence { likely_run: 0 }
    }
}

impl Detector {
    /// Whether [`Event::SpeechStarted`] has been reported.
    pub fn heard_speech(&self) -> bool {
        self.heard_speech
    }
}

/// Feeds the speech probability of the next chunk to the detector, and reports when
/// speech starts and ends.
pub fn decide(detector: Detector, probability: f32, params: &Params) -> (Detector, Option<Event>) {
    let (phase, mut event) = match detector.phase {
        Phase::Silence { likely_run } => {
            let likely_run = if probability > params.start_probability {
                likely_run + 1
            } else {
                0
            };
            if likely_run >= params.start_chunks {
                (
                    Phase::Speech { unlikely_run: 0 },
                    Some(Event::SpeechStarted),
                )
            } else {
                (Phase::Silence { likely_run }, None)
            }
        }
        Phase::Speech { unlikely_run } => {
            let unlikely_run = if probability < params.end_probability {
                unlikely_run + 1
            } else {
                0
            };
            if unlikely_run >= params.hangover_chunks {
                (Phase::Silence { likely_run: 0 }, Some(Event::SpeechEnded))
            } else {
                (Phase::Speech { unlikely_run }, None)
            }
        }
    };

    let chunks_seen = detector.chunks_seen + 1;
    let heard_speech = detector.heard_speech || event == Some(Event::SpeechStarted);
    if !heard_speech && chunks_seen == params.no_speech_chunks {
        event = Some(Event::NoSpeechTimeout);
    }
    let next = Detector {
        phase,
        chunks_seen,
        heard_speech,
    };
    (next, event)
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

    /// Events reported for `probabilities`, with the chunk index each one fell on.
    fn decisions(probabilities: &[f32], params: &Params) -> Vec<(usize, Event)> {
        let mut detector = Detector::default();
        let mut events = Vec::new();
        for (index, &probability) in probabilities.iter().enumerate() {
            let event;
            (detector, event) = decide(detector, probability, params);
            events.extend(event.map(|event| (index, event)));
        }
        events
    }

    const PARAMS: Params = Params {
        start_probability: 0.9,
        end_probability: 0.5,
        start_chunks: 3,
        hangover_chunks: 4,
        no_speech_chunks: 20,
    };

    #[test]
    fn speech_starts_on_the_last_of_start_chunks_above_the_start_probability() {
        let probabilities = [0.1, 0.95, 0.95, 0.95, 0.95];

        assert_eq!(
            decisions(&probabilities, &PARAMS),
            [(3, Event::SpeechStarted)]
        );
    }

    #[test]
    fn a_dip_below_the_start_probability_restarts_the_count() {
        let probabilities = [0.95, 0.95, 0.8, 0.95, 0.95];

        assert_eq!(decisions(&probabilities, &PARAMS), []);
    }

    #[test]
    fn speech_continues_between_the_end_and_the_start_probability() {
        let mut probabilities = vec![0.95; 3];
        probabilities.extend([0.6; 10]);

        assert_eq!(
            decisions(&probabilities, &PARAMS),
            [(2, Event::SpeechStarted)]
        );
    }

    #[test]
    fn speech_ends_after_hangover_chunks_below_the_end_probability() {
        let probabilities = [0.95, 0.95, 0.95, 0.1, 0.1, 0.1, 0.1, 0.1];

        assert_eq!(
            decisions(&probabilities, &PARAMS),
            [(2, Event::SpeechStarted), (6, Event::SpeechEnded)]
        );
    }

    #[test]
    fn a_likely_chunk_in_the_hangover_restarts_it() {
        let probabilities = [0.95, 0.95, 0.95, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1];

        assert_eq!(
            decisions(&probabilities, &PARAMS),
            [(2, Event::SpeechStarted), (10, Event::SpeechEnded)]
        );
    }

    #[test]
    fn no_speech_times_out_once() {
        let probabilities = [0.1; 50];

        assert_eq!(
            decisions(&probabilities, &PARAMS),
            [(19, Event::NoSpeechTimeout)]
        );
    }

    #[test]
    fn heard_speech_stays_true_after_speech_ends() {
        let mut detector = Detector::default();
        for probability in [0.95, 0.95, 0.95, 0.1, 0.1, 0.1, 0.1] {
            detector = decide(detector, probability, &PARAMS).0;
        }

        assert!(detector.heard_speech());
    }
}
