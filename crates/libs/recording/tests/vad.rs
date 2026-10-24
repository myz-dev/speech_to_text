//! Tests against the real Silero VAD model. They skip (pass with a note) when the model
//! file is missing; the noise tests also skip when their recording in
//! `fixtures/noise/` is missing. Speech comes from the macOS `say` command.

use std::path::{Path, PathBuf};
use std::process::Command;

use recording::silence::Event;
use recording::vad::{self, CHUNK_SAMPLES, Detector, Params, State, Vad};

/// Probability above which a chunk counts as speech in these tests.
const SPEECH: f32 = 0.5;

fn workspace_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn open_vad() -> Option<Vad> {
    let path = workspace_dir().join("models").join(vad::MODEL_FILE);
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return None;
    }
    Some(Vad::open(&path).unwrap())
}

/// A recording from `fixtures/noise/`, as 16 kHz mono samples.
fn noise_fixture(name: &str) -> Option<Vec<f32>> {
    let path = workspace_dir().join("fixtures/noise").join(name);
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return None;
    }
    Some(recording::load_wav(&path).unwrap())
}

/// Speaks `text` with a macOS voice and returns 16 kHz mono samples.
fn speak(voice: &str, text: &str) -> Vec<f32> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("speech.wav");
    let status = Command::new("say")
        .args(["-v", voice, "--data-format=LEI16@16000", "-o"])
        .arg(&path)
        .arg(text)
        .status()
        .unwrap();
    assert!(status.success(), "say failed for voice {voice}");
    recording::load_wav(&path).unwrap()
}

/// Share of chunks above [`SPEECH`].
fn speech_share(probabilities: &[f32]) -> f32 {
    let speech = probabilities.iter().filter(|&&p| p > SPEECH).count();
    speech as f32 / probabilities.len() as f32
}

/// Most chunks in a row above [`SPEECH`].
fn longest_speech_run(probabilities: &[f32]) -> usize {
    let mut longest = 0;
    let mut run = 0;
    for &p in probabilities {
        run = if p > SPEECH { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    longest
}

/// Uniform white noise at about -30 dBFS (RMS), from a fixed seed.
fn white_noise(samples: usize) -> Vec<f32> {
    let mut seed: u32 = 0x9e37_79b9;
    (0..samples)
        .map(|_| {
            // xorshift32: enough randomness for noise, and the same on every run.
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32 * 2.0 - 1.0) * 0.055
        })
        .collect()
}

#[test]
fn german_speech_is_mostly_speech() {
    let Some(mut vad) = open_vad() else {
        return;
    };
    let speech = speak("Anna", "Heute Abend gehen wir zusammen ins Kino.");

    let probabilities = vad.probabilities(&speech).unwrap();

    assert!(speech_share(&probabilities) > 0.5, "{probabilities:?}");
}

#[test]
fn english_speech_is_mostly_speech() {
    let Some(mut vad) = open_vad() else {
        return;
    };
    let speech = speak("Daniel", "The quick brown fox jumps over the lazy dog.");

    let probabilities = vad.probabilities(&speech).unwrap();

    assert!(speech_share(&probabilities) > 0.5, "{probabilities:?}");
}

#[test]
fn digital_silence_is_never_speech() {
    let Some(mut vad) = open_vad() else {
        return;
    };

    let probabilities = vad.probabilities(&[0.0; 16_000]).unwrap();

    assert_eq!(longest_speech_run(&probabilities), 0, "{probabilities:?}");
}

#[test]
fn loud_white_noise_is_never_speech() {
    let Some(mut vad) = open_vad() else {
        return;
    };

    // Loud enough that the energy detector takes it for speech.
    let probabilities = vad.probabilities(&white_noise(32_000)).unwrap();

    assert_eq!(longest_speech_run(&probabilities), 0, "{probabilities:?}");
}

/// What the speech decision with the default [`Params`] reports for `audio`, in order.
fn events(vad: &mut Vad, audio: &[f32]) -> Vec<Event> {
    let mut detector = Detector::default();
    let mut events = Vec::new();
    for probability in vad.probabilities(audio).unwrap() {
        let event;
        (detector, event) = vad::decide(detector, probability, &Params::default());
        events.extend(event);
    }
    events
}

/// Whether the speech decision reports that speech started anywhere in `audio`.
fn starts_speech(vad: &mut Vad, audio: &[f32]) -> bool {
    events(vad, audio).contains(&Event::SpeechStarted)
}

#[test]
fn coughing_never_starts_speech() {
    let Some(mut vad) = open_vad() else {
        return;
    };
    for name in ["cough.wav", "cough-2.wav", "cough-3.wav"] {
        let Some(cough) = noise_fixture(name) else {
            continue;
        };

        assert!(!starts_speech(&mut vad, &cough), "{name}");
    }
}

#[test]
fn typing_never_starts_speech() {
    let (Some(mut vad), Some(typing)) = (open_vad(), noise_fixture("typing.wav")) else {
        return;
    };

    assert!(!starts_speech(&mut vad, &typing));
}

#[test]
fn speech_while_typing_starts_speech() {
    let (Some(mut vad), Some(audio)) = (open_vad(), noise_fixture("typing-speech.wav")) else {
        return;
    };

    assert!(starts_speech(&mut vad, &audio));
}

#[test]
fn a_single_short_word_starts_speech() {
    let Some(mut vad) = open_vad() else {
        return;
    };
    for name in ["word-ja.wav", "word-nein.wav", "word-okay.wav"] {
        let Some(word) = noise_fixture(name) else {
            continue;
        };

        assert!(starts_speech(&mut vad, &word), "{name}");
    }
}

#[test]
fn a_cough_after_speech_does_not_start_it_again() {
    let (Some(mut vad), Some(audio)) = (open_vad(), noise_fixture("speech-cough.wav")) else {
        return;
    };

    assert_eq!(
        events(&mut vad, &audio),
        [Event::SpeechStarted, Event::SpeechEnded]
    );
}

/// A known limit: clearing the throat is voiced, and Silero takes it for speech.
#[test]
fn clearing_the_throat_starts_speech() {
    let (Some(mut vad), Some(audio)) = (open_vad(), noise_fixture("throat-clear.wav")) else {
        return;
    };

    assert!(starts_speech(&mut vad, &audio));
}

#[test]
fn every_dictation_fixture_starts_speech() {
    let Some(mut vad) = open_vad() else {
        return;
    };
    for language in ["de", "en"] {
        let Ok(entries) = std::fs::read_dir(workspace_dir().join("fixtures").join(language)) else {
            eprintln!("skipping: fixtures/{language} not found");
            continue;
        };
        for path in entries.map(|entry| entry.unwrap().path()) {
            if path.extension().is_none_or(|extension| extension != "wav") {
                continue;
            }
            let audio = recording::load_wav(&path).unwrap();

            assert!(starts_speech(&mut vad, &audio), "{}", path.display());
        }
    }
}

#[test]
fn a_last_partial_chunk_is_not_judged() {
    let Some(mut vad) = open_vad() else {
        return;
    };

    let probabilities = vad.probabilities(&[0.0; 3 * CHUNK_SAMPLES - 1]).unwrap();

    assert_eq!(probabilities.len(), 2);
}

#[test]
fn stepping_chunk_by_chunk_gives_the_same_probabilities() {
    let Some(mut vad) = open_vad() else {
        return;
    };
    let speech = speak("Anna", "Guten Morgen.");

    let mut state = State::default();
    let mut stepped = Vec::new();
    for chunk in speech.chunks_exact(CHUNK_SAMPLES) {
        let (next, probability) = vad.step(&state, chunk.try_into().unwrap()).unwrap();
        state = next;
        stepped.push(probability);
    }

    assert_eq!(stepped, vad.probabilities(&speech).unwrap());
}

#[test]
fn the_state_carries_what_came_before() {
    let Some(mut vad) = open_vad() else {
        return;
    };
    let speech = speak("Anna", "Guten Morgen.");
    let mut state = State::default();
    for chunk in speech.chunks_exact(CHUNK_SAMPLES) {
        state = vad.step(&state, chunk.try_into().unwrap()).unwrap().0;
    }
    let quiet = [0.0; CHUNK_SAMPLES];

    let (_, after_speech) = vad.step(&state, &quiet).unwrap();
    let (_, from_start) = vad.step(&State::default(), &quiet).unwrap();

    assert_ne!(after_speech, from_start);
}

#[test]
fn a_missing_model_names_its_path() {
    let error = Vad::open(Path::new("models/missing.onnx"))
        .err()
        .expect("a missing model file must not load");

    assert_eq!(
        error.to_string(),
        "cannot load silero vad model models/missing.onnx"
    );
}

/// The events of [`vad::Live`] fed `audio`, converted to `sample_rate`, in pieces of
/// 10 ms, like microphone buffers.
fn live_events(vad: &mut Vad, audio: &[f32], sample_rate: u32) -> Vec<Event> {
    let audio = recording::resample(audio.to_vec(), recording::SAMPLE_RATE, sample_rate).unwrap();
    let mut live = vad::Live::new(vad, Params::default(), sample_rate).unwrap();
    let mut events = Vec::new();
    for piece in audio.chunks(sample_rate as usize / 100) {
        events.extend(live.push(piece).unwrap());
    }
    events
}

#[test]
fn live_reports_the_events_of_the_whole_recording_at_any_rate() {
    let Some(mut vad) = open_vad() else {
        return;
    };
    for name in ["speech-cough.wav", "cough.wav", "word-ja.wav", "typing.wav"] {
        let Some(audio) = noise_fixture(name) else {
            continue;
        };
        let expected = events(&mut vad, &audio);
        for sample_rate in [16_000, 44_100, 48_000] {
            assert_eq!(
                live_events(&mut vad, &audio, sample_rate),
                expected,
                "{name} at {sample_rate} Hz"
            );
        }
    }
}

#[test]
fn live_fed_one_sample_at_a_time_reports_the_events_of_the_whole_recording() {
    let Some(mut vad) = open_vad() else {
        return;
    };
    let speech = speak("Anna", "Heute Abend gehen wir zusammen ins Kino.");
    let expected = events(&mut vad, &speech);
    assert!(expected.contains(&Event::SpeechStarted), "{expected:?}");

    for sample_rate in [16_000, 44_100] {
        let audio =
            recording::resample(speech.clone(), recording::SAMPLE_RATE, sample_rate).unwrap();
        let mut live = vad::Live::new(&mut vad, Params::default(), sample_rate).unwrap();
        let mut events = Vec::new();
        for sample in &audio {
            events.extend(live.push(std::slice::from_ref(sample)).unwrap());
        }

        assert_eq!(events, expected, "at {sample_rate} Hz");
    }
}
