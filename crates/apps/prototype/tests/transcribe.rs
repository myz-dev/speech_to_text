//! `stt transcribe` end to end. Tests needing the Whisper model skip (pass with a note)
//! when it is missing; speech is generated with the macOS `say` command.

use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser;
use prototype::{Cli, WHISPER_MODEL_FILE};

fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models")
}

/// Runs `stt` with `args` and returns what it printed.
fn run(args: &[&str]) -> anyhow::Result<String> {
    let cli = Cli::try_parse_from(["stt"].iter().chain(args))?;
    let mut out = Vec::new();
    prototype::run(cli, &mut out)?;
    Ok(String::from_utf8(out)?)
}

#[test]
fn prints_transcript_then_language_and_timings() {
    let models = models_dir();
    if !models.join(WHISPER_MODEL_FILE).exists() {
        eprintln!("skipping: whisper model not found in {}", models.display());
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("speech.wav");
    let status = Command::new("say")
        .args(["-v", "Anna", "--data-format=LEI16@16000", "-o"])
        .arg(&wav)
        .arg("Guten Morgen.")
        .status()
        .unwrap();
    assert!(status.success());

    let output = run(&[
        "transcribe",
        wav.to_str().unwrap(),
        "--models",
        models.to_str().unwrap(),
    ])
    .unwrap();

    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines[0], "Guten Morgen.");
    assert!(lines[1].starts_with("language de | audio "), "{}", lines[1]);
    assert!(lines[1].contains("real-time factor"), "{}", lines[1]);
}

#[test]
fn missing_wav_file_is_reported_before_loading_the_model() {
    let err = run(&["transcribe", "does/not/exist.wav", "--models", "nowhere"]).unwrap_err();
    assert_eq!(err.to_string(), "cannot read does/not/exist.wav");
}

#[test]
fn missing_model_is_reported_with_its_path() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("silence.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    hound::WavWriter::create(&wav, spec)
        .unwrap()
        .finalize()
        .unwrap();

    let err = run(&["transcribe", wav.to_str().unwrap(), "--models", "nowhere"]).unwrap_err();
    assert_eq!(
        err.to_string(),
        format!("cannot load whisper model nowhere/{WHISPER_MODEL_FILE}")
    );
}
