//! Tests against the real model files. They skip (pass with a note) when the model
//! directory is missing, and generate speech with the macOS `say` command.

mod common;

use std::path::Path;

use transcription::parakeet::{HOP_SAMPLES, MEL_BINS, MIN_SAMPLES, ParakeetModel, TranscribeError};

fn load_model() -> Option<ParakeetModel> {
    let dir = common::models_dir().join("parakeet-tdt-0.6b-v3");
    if !dir.exists() {
        eprintln!("skipping: {} not found", dir.display());
        return None;
    }
    Some(ParakeetModel::load(&dir).unwrap())
}

#[test]
fn features_have_one_frame_per_10_ms_plus_one() {
    let Some(mut model) = load_model() else {
        return;
    };
    let speech = common::speak("Anna", "Heute Abend gehen wir zusammen ins Kino.");

    // One frame per completed hop of 160 samples, plus one: 16159 samples still give
    // 101 frames, 16160 give 102.
    for (len, frames) in [(16_000, 101), (16_159, 101), (16_160, 102)] {
        let features = model.features(&speech[..len]).unwrap();
        assert_eq!(features.frames, frames, "{len} samples");
        assert_eq!(features.frames, len / HOP_SAMPLES + 1);
        assert_eq!(features.data.len(), MEL_BINS * frames);
    }
}

#[test]
fn each_mel_bin_is_normalized_over_the_clip() {
    let Some(mut model) = load_model() else {
        return;
    };
    let speech = common::speak("Daniel", "The quick brown fox jumps over the lazy dog.");

    let features = model.features(&speech).unwrap();

    for (bin, row) in features.data.chunks_exact(features.frames).enumerate() {
        let n = row.len() as f32;
        let mean = row.iter().sum::<f32>() / n;
        let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1.0);
        assert!(mean.abs() < 1e-3, "bin {bin}: mean {mean}");
        assert!(
            (variance.sqrt() - 1.0).abs() < 1e-2,
            "bin {bin}: std {}",
            variance.sqrt()
        );
    }
}

#[test]
fn rejects_input_shorter_than_the_analysis_window() {
    let Some(mut model) = load_model() else {
        return;
    };

    let short = model.features(&[0.0; MIN_SAMPLES - 1]);
    assert!(matches!(
        short,
        Err(TranscribeError::TooShort { samples: 256 })
    ));

    let empty = model.features(&[]);
    assert!(matches!(
        empty,
        Err(TranscribeError::TooShort { samples: 0 })
    ));

    let shortest = model.features(&[0.0; MIN_SAMPLES]).unwrap();
    assert_eq!(shortest.frames, 2);
}

#[test]
fn loads_more_than_once_per_process() {
    let Some(first) = load_model() else { return };
    let second = load_model().unwrap();
    drop((first, second));

    load_model().unwrap();
}

#[test]
fn missing_model_directory_reports_the_file_path() {
    let err = ParakeetModel::load(Path::new("models/missing"))
        .err()
        .unwrap();
    assert_eq!(
        err.to_string(),
        "cannot load parakeet model models/missing/nemo128.onnx"
    );
}
