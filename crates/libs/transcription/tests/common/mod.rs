//! Helpers shared by the integration tests.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The workspace's `models/` directory.
pub fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models")
}

/// Speaks `text` with a macOS voice and returns 16 kHz mono samples.
pub fn speak(voice: &str, text: &str) -> Vec<f32> {
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

/// Lowercase words without punctuation, for robust comparison.
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .collect()
}
