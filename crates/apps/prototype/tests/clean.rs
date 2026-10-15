//! `stt clean` end to end. Tests skip (pass with a note) when Ollama is not running or
//! the default model is not pulled.

use clap::Parser;
use cleanup::OLLAMA_URL;
use prototype::{Cli, DEFAULT_CLEANUP_MODEL};

/// True if Ollama runs and has the default model; prints why a test is skipped otherwise.
fn default_model_available() -> bool {
    let tags = ureq::get(format!("{OLLAMA_URL}/api/tags"))
        .call()
        .and_then(|mut response| response.body_mut().read_to_string());
    let available =
        tags.is_ok_and(|tags| tags.contains(&format!("\"name\":\"{DEFAULT_CLEANUP_MODEL}\"")));
    if !available {
        eprintln!("skipping: ollama with {DEFAULT_CLEANUP_MODEL} not available");
    }
    available
}

/// Runs `stt` with `args` and returns what it printed.
fn run(args: &[&str]) -> anyhow::Result<String> {
    let cli = Cli::try_parse_from(["stt"].iter().chain(args))?;
    let mut out = Vec::new();
    prototype::run(cli, &mut out)?;
    Ok(String::from_utf8(out)?)
}

#[test]
fn prints_cleaned_text_then_model_and_timing() {
    if !default_model_available() {
        return;
    }
    let output = run(&["clean", "äh wie geht es dir", "-l", "de"]).unwrap();

    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines[0], "Wie geht es dir?");
    assert!(
        lines[1].starts_with(&format!("model {DEFAULT_CLEANUP_MODEL} | cleanup ")),
        "{}",
        lines[1]
    );
}

#[test]
fn unknown_model_is_reported_with_ollamas_message() {
    if !default_model_available() {
        return;
    }
    let err = run(&["clean", "hallo", "-l", "de", "-m", "no-such-model:1b"]).unwrap_err();
    assert_eq!(
        err.to_string(),
        "ollama rejected the request for no-such-model:1b (http 404): \
         model 'no-such-model:1b' not found"
    );
}
