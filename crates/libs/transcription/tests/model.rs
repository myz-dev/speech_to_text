//! `Model` over both engines. Tests that need model files skip (pass with a note) when
//! they are missing.

mod common;

use std::path::Path;

use transcription::{Device, Engine, LoadModelError, Model};

fn load(engine: Engine) -> Option<Model> {
    let models = common::models_dir();
    if !engine.model_path(&models).exists() {
        eprintln!(
            "skipping: {} not found",
            engine.model_path(&models).display()
        );
        return None;
    }
    Some(Model::load(engine, &models, Device::Gpu).unwrap())
}

#[test]
fn whisper_reports_the_language_it_detected() {
    let Some(mut model) = load(Engine::Whisper) else {
        return;
    };
    let speech = common::speak("Anna", "Heute Abend gehen wir zusammen ins Kino.");

    let transcript = model.transcribe(&speech, None).unwrap();

    assert_eq!(transcript.language, Some("de"));
    assert_eq!(
        common::words(&transcript.text),
        common::words("Heute Abend gehen wir zusammen ins Kino.")
    );
}

#[test]
fn parakeet_reports_no_language_and_ignores_the_hint() {
    let Some(mut model) = load(Engine::Parakeet) else {
        return;
    };
    let speech = common::speak("Anna", "Heute Abend gehen wir zusammen ins Kino.");

    let transcript = model.transcribe(&speech, Some("en")).unwrap();

    assert_eq!(transcript.language, None);
    assert_eq!(
        common::words(&transcript.text),
        common::words("Heute Abend gehen wir zusammen ins Kino.")
    );
}

#[test]
fn each_engine_has_its_own_path_in_the_models_directory() {
    let models = Path::new("models");
    assert_eq!(
        Engine::Whisper.model_path(models),
        Path::new("models/ggml-large-v3-turbo.bin")
    );
    assert_eq!(
        Engine::Parakeet.model_path(models),
        Path::new("models/parakeet-tdt-0.6b-v3")
    );
}

#[test]
fn load_errors_keep_the_engines_message() {
    let err = Model::load(Engine::Whisper, Path::new("nowhere"), Device::Cpu)
        .err()
        .unwrap();
    assert!(matches!(err, LoadModelError::Whisper(_)), "{err:?}");
    assert_eq!(
        err.to_string(),
        "cannot load whisper model nowhere/ggml-large-v3-turbo.bin"
    );

    let err = Model::load(Engine::Parakeet, Path::new("nowhere"), Device::Cpu)
        .err()
        .unwrap();
    assert!(matches!(err, LoadModelError::Parakeet(_)), "{err:?}");
    assert_eq!(
        err.to_string(),
        "cannot read parakeet vocabulary nowhere/parakeet-tdt-0.6b-v3/vocab.txt"
    );
}
