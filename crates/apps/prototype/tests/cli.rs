//! Make sure arguments are parsed/used as expected.

use clap::Parser;
use prototype::{Cli, Command, DEFAULT_CLEANUP_MODEL, Engine, Lang};

#[test]
fn transcribe_defaults_to_whisper_without_language() {
    let cli = Cli::try_parse_from(["stt", "transcribe", "clip.wav"]).unwrap();
    let Command::Transcribe(args) = cli.command else {
        panic!("expected transcribe, got {:?}", cli.command);
    };
    assert_eq!(args.file.to_str(), Some("clip.wav"));
    assert_eq!(args.stt.engine, Engine::Whisper);
    assert_eq!(args.stt.language, None);
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
    let Command::Transcribe(args) = cli.command else {
        panic!("expected transcribe, got {:?}", cli.command);
    };
    assert_eq!(args.stt.engine, Engine::Parakeet);
    assert_eq!(args.stt.language, Some(Lang::De));
}

#[test]
fn unknown_engine_is_rejected() {
    assert!(Cli::try_parse_from(["stt", "transcribe", "clip.wav", "-e", "vosk"]).is_err());
}

#[test]
fn record_saves_nothing_by_default() {
    let cli = Cli::try_parse_from(["stt", "record", "-e", "parakeet"]).unwrap();
    let Command::Record(args) = cli.command else {
        panic!("expected record, got {:?}", cli.command);
    };
    assert_eq!(args.stt.engine, Engine::Parakeet);
    assert_eq!(args.save, None);
}

#[test]
fn record_accepts_save_path() {
    let cli = Cli::try_parse_from(["stt", "record", "--save", "fixtures/de/note.wav"]).unwrap();
    let Command::Record(args) = cli.command else {
        panic!("expected record, got {:?}", cli.command);
    };
    assert_eq!(
        args.save.as_deref().and_then(|p| p.to_str()),
        Some("fixtures/de/note.wav")
    );
}

#[test]
fn clean_requires_a_language() {
    assert!(Cli::try_parse_from(["stt", "clean", "äh hallo"]).is_err());
}

#[test]
fn clean_defaults_to_the_default_model() {
    let cli = Cli::try_parse_from(["stt", "clean", "äh hallo", "-l", "de"]).unwrap();
    let Command::Clean(args) = cli.command else {
        panic!("expected clean, got {:?}", cli.command);
    };
    assert_eq!(args.text, "äh hallo");
    assert_eq!(args.language, Lang::De);
    assert_eq!(args.model, DEFAULT_CLEANUP_MODEL);
}

#[test]
fn clean_rejects_unsupported_language() {
    assert!(Cli::try_parse_from(["stt", "clean", "bonjour", "-l", "fr"]).is_err());
}

#[test]
fn whisper_language_hint_is_limited_to_de_and_en() {
    assert!(Cli::try_parse_from(["stt", "transcribe", "clip.wav", "-l", "fr"]).is_err());
}

#[test]
fn unimplemented_commands_report_their_milestone() {
    let cli = Cli::try_parse_from(["stt", "bench"]).unwrap();
    let err = prototype::run(cli, &mut Vec::new()).unwrap_err();
    assert_eq!(err.to_string(), "not implemented yet (M5)");
}
