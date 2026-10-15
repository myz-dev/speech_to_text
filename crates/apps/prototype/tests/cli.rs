//! Make sure arguments are parsed/used as expected.

use clap::Parser;
use prototype::{Cli, Command, Engine};

#[test]
fn transcribe_defaults_to_whisper_without_language() {
    let cli = Cli::try_parse_from(["stt", "transcribe", "clip.wav"]).unwrap();
    let Command::Transcribe { file, stt } = cli.command else {
        panic!("expected transcribe, got {:?}", cli.command);
    };
    assert_eq!(file.to_str(), Some("clip.wav"));
    assert_eq!(stt.engine, Engine::Whisper);
    assert_eq!(stt.language, None);
}

#[test]
fn transcribe_accepts_engine_and_language() {
    let cli = Cli::try_parse_from([
        "stt",
        "transcribe",
        "clip.wav",
        "-e",
        "parakeet",
        "-l",
        "de",
    ])
    .unwrap();
    let Command::Transcribe { stt, .. } = cli.command else {
        panic!("expected transcribe, got {:?}", cli.command);
    };
    assert_eq!(stt.engine, Engine::Parakeet);
    assert_eq!(stt.language.as_deref(), Some("de"));
}

#[test]
fn unknown_engine_is_rejected() {
    assert!(Cli::try_parse_from(["stt", "transcribe", "clip.wav", "-e", "vosk"]).is_err());
}

#[test]
fn record_saves_nothing_by_default() {
    let cli = Cli::try_parse_from(["stt", "record", "-e", "parakeet"]).unwrap();
    let Command::Record { stt, save } = cli.command else {
        panic!("expected record, got {:?}", cli.command);
    };
    assert_eq!(stt.engine, Engine::Parakeet);
    assert_eq!(save, None);
}

#[test]
fn record_accepts_save_path() {
    let cli = Cli::try_parse_from(["stt", "record", "--save", "fixtures/de/note.wav"]).unwrap();
    let Command::Record { save, .. } = cli.command else {
        panic!("expected record, got {:?}", cli.command);
    };
    assert_eq!(
        save.as_deref().and_then(|p| p.to_str()),
        Some("fixtures/de/note.wav")
    );
}

#[test]
fn unimplemented_commands_report_their_milestone() {
    let cli = Cli::try_parse_from(["stt", "bench"]).unwrap();
    let err = prototype::run(cli, &mut Vec::new()).unwrap_err();
    assert_eq!(err.to_string(), "not implemented yet (M5)");
}
