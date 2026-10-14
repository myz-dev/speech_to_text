//! Tests against the real model. They skip (pass with a note) when the model file is
//! missing, and generate speech with the macOS `say` command.
//! Transcription is deterministic for a given input (greedy decoding, fixed seeds), so exact
//! assertions are stable on one machine; macOS voice or whisper.cpp updates can change them.

mod common;

use std::path::{Path, PathBuf};

use common::speak;
use transcription::Device;
use transcription::whisper::{TranscribeError, WhisperModel};

const MODEL_FILE: &str = "ggml-large-v3-turbo.bin";

fn model_path() -> PathBuf {
    common::models_dir().join(MODEL_FILE)
}

fn load_model(device: Device) -> Option<WhisperModel> {
    let path = model_path();
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return None;
    }
    Some(WhisperModel::load(&path, device).unwrap())
}

/// Lowercase words without punctuation, for robust comparison.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .collect()
}

#[test]
fn transcribes_english_and_detects_language() {
    let Some(model) = load_model(Device::Gpu) else {
        return;
    };
    let samples = speak("Daniel", "The quick brown fox jumps over the lazy dog.");

    let transcript = model.transcribe(&samples, None).unwrap();

    assert_eq!(transcript.language, "en");
    assert_eq!(
        words(&transcript.text),
        words("The quick brown fox jumps over the lazy dog.")
    );
}

#[test]
fn transcribes_german_and_detects_language() {
    let Some(model) = load_model(Device::Gpu) else {
        return;
    };
    let samples = speak("Anna", "Heute Abend gehen wir zusammen ins Kino.");

    let transcript = model.transcribe(&samples, None).unwrap();

    assert_eq!(transcript.language, "de");
    assert_eq!(
        words(&transcript.text),
        words("Heute Abend gehen wir zusammen ins Kino.")
    );
}

#[test]
fn language_hint_overrides_detection_and_is_reported_as_code() {
    let Some(model) = load_model(Device::Gpu) else {
        return;
    };
    // English speech, so detection alone would report "en".
    let samples = speak("Daniel", "Good morning, how are you today?");

    let by_code = model.transcribe(&samples, Some("de")).unwrap();
    let by_name = model.transcribe(&samples, Some("german")).unwrap();

    assert_eq!(by_code.language, "de");
    assert_eq!(by_name.language, "de");
}

#[test]
fn transcribes_on_cpu() {
    let Some(model) = load_model(Device::Cpu) else {
        return;
    };
    let samples = speak("Anna", "Guten Morgen.");

    let transcript = model.transcribe(&samples, None).unwrap();

    assert_eq!(transcript.language, "de");
    assert_eq!(words(&transcript.text), words("Guten Morgen."));
}

#[test]
fn rejects_invalid_input_before_inference() {
    let Some(model) = load_model(Device::Gpu) else {
        return;
    };

    let unknown = model.transcribe(&[0.0; 16_000], Some("klingon"));
    assert!(matches!(unknown, Err(TranscribeError::UnknownLanguage(l)) if l == "klingon"));

    let nul = model.transcribe(&[0.0; 16_000], Some("d\0e"));
    assert!(matches!(nul, Err(TranscribeError::UnknownLanguage(_))));

    let empty = model.transcribe(&[], None);
    assert!(matches!(empty, Err(TranscribeError::NoSamples)));
}

#[test]
fn missing_model_file_reports_its_path() {
    let err = WhisperModel::load(Path::new("models/missing.bin"), Device::Gpu)
        .err()
        .unwrap();
    assert_eq!(
        err.to_string(),
        "cannot load whisper model models/missing.bin"
    );
}
