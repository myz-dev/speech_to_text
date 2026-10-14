//! Tests against the real model files. They skip (pass with a note) when the model
//! directory is missing, and generate speech with the macOS `say` command.

mod common;

use std::path::Path;

use transcription::parakeet::{
    ENCODER_DIM, Features, HOP_SAMPLES, MEL_BINS, MIN_SAMPLES, ParakeetModel, SUBSAMPLING,
    TranscribeError,
};

fn load_model() -> Option<ParakeetModel> {
    let dir = common::models_dir().join("parakeet-tdt-0.6b-v3");
    if !dir.exists() {
        eprintln!("skipping: {} not found", dir.display());
        return None;
    }
    Some(ParakeetModel::load(&dir).unwrap())
}

#[test]
fn transcribes_german() {
    let Some(mut model) = load_model() else {
        return;
    };
    let speech = common::speak("Anna", "Heute Abend gehen wir zusammen ins Kino.");

    let text = model.transcribe(&speech).unwrap();

    assert_eq!(
        common::words(&text),
        common::words("Heute Abend gehen wir zusammen ins Kino.")
    );
}

#[test]
fn transcribes_english() {
    let Some(mut model) = load_model() else {
        return;
    };
    let speech = common::speak("Daniel", "The quick brown fox jumps over the lazy dog.");

    let text = model.transcribe(&speech).unwrap();

    assert_eq!(
        common::words(&text),
        common::words("The quick brown fox jumps over the lazy dog.")
    );
}

#[test]
fn silence_gives_no_text() {
    let Some(mut model) = load_model() else {
        return;
    };

    let text = model.transcribe(&[0.0; 32_000]).unwrap();

    assert_eq!(text, "");
}

#[test]
fn noise_gives_no_text() {
    let Some(mut model) = load_model() else {
        return;
    };
    // 2 s of white-ish noise from a fixed pseudo-random sequence, so the test is
    // deterministic.
    let mut seed: u32 = 1;
    let noise: Vec<f32> = (0..32_000)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32 * 0.2 - 0.1
        })
        .collect();

    let text = model.transcribe(&noise).unwrap();

    assert_eq!(text, "");
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
    assert_eq!(features.data.len(), MEL_BINS * features.frames);

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
    let Some(_first) = load_model() else { return };
    load_model().unwrap();
}

#[test]
fn encoder_emits_one_vector_per_8_feature_frames() {
    let Some(mut model) = load_model() else {
        return;
    };
    let speech = common::speak("Anna", "Heute Abend gehen wir zusammen ins Kino.");
    let features = model.features(&speech).unwrap();

    let encoded = model.encode(&features).unwrap();

    assert_eq!(encoded.frames, features.frames.div_ceil(SUBSAMPLING));
    assert_eq!(encoded.data.len(), ENCODER_DIM * encoded.frames);

    // A partial last group of feature frames still produces a frame.
    for (feature_frames, encoded_frames) in [(1, 1), (8, 1), (9, 2), (16, 2), (17, 3)] {
        let silence = Features {
            data: vec![0.0; MEL_BINS * feature_frames],
            frames: feature_frames,
        };
        assert_eq!(model.encode(&silence).unwrap().frames, encoded_frames);
    }
}

#[test]
fn encoder_runs_entirely_on_the_cpu() {
    let Some(mut model) = load_model() else {
        return;
    };
    let empty = Features {
        data: Vec::new(),
        frames: 0,
    };

    // Empty input fails inside the first node, and the error names that node: a CoreML
    // partition would show up as `CoreMLExecutionProvider_…`.
    let Err(TranscribeError::Inference(err)) = model.encode(&empty) else {
        panic!("expected an inference error for empty input");
    };
    assert!(!err.to_string().contains("CoreML"), "{err}");
}

#[test]
fn missing_model_directory_reports_the_file_path() {
    let err = ParakeetModel::load(Path::new("models/missing"))
        .err()
        .unwrap();
    assert_eq!(
        err.to_string(),
        "cannot read parakeet vocabulary models/missing/vocab.txt"
    );
}

#[test]
fn vocabulary_that_does_not_match_the_decoder_is_an_error() {
    let models = common::models_dir().join("parakeet-tdt-0.6b-v3");
    if !models.exists() {
        eprintln!("skipping: {} not found", models.display());
        return;
    }
    // The real graphs with a vocabulary of 3 instead of 8193 tokens.
    let dir = tempfile::tempdir().unwrap();
    for file in [
        "nemo128.onnx",
        "encoder-model.int8.onnx",
        "decoder_joint-model.int8.onnx",
    ] {
        std::os::unix::fs::symlink(models.join(file), dir.path().join(file)).unwrap();
    }
    std::fs::write(dir.path().join("vocab.txt"), "<unk> 0\n▁a 1\n<blk> 2\n").unwrap();
    let mut model = ParakeetModel::load(dir.path()).unwrap();
    let speech = common::speak("Anna", "Guten Morgen.");

    let result = model.transcribe(&speech);

    assert!(
        matches!(
            result,
            Err(TranscribeError::VocabularyMismatch {
                vocabulary: 3,
                scores: 8198
            })
        ),
        "{result:?}"
    );
}

#[test]
fn malformed_vocabulary_is_reported_with_its_line() {
    for (content, line) in [
        ("<unk>0\n<blk> 1\n", 1),        // no space between piece and id
        ("<unk> 0\n▁a 2\n<blk> 2\n", 2), // id does not match the line
        ("<unk> 0\n▁a x\n<blk> 2\n", 2), // id is not a number
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.txt");
        std::fs::write(&path, content).unwrap();

        let err = ParakeetModel::load(dir.path()).err().unwrap();

        assert_eq!(
            err.to_string(),
            format!(
                "malformed parakeet vocabulary {} at line {line}",
                path.display()
            ),
            "{content:?}"
        );
    }
}

#[test]
fn vocabulary_without_blank_is_rejected() {
    for content in ["<unk> 0\n▁a 1\n", ""] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.txt");
        std::fs::write(&path, content).unwrap();

        let err = ParakeetModel::load(dir.path()).err().unwrap();

        assert_eq!(
            err.to_string(),
            format!(
                "parakeet vocabulary {} does not end with <blk>",
                path.display()
            ),
            "{content:?}"
        );
    }
}

#[test]
fn missing_graph_file_reports_its_path() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("vocab.txt"), "<unk> 0\n<blk> 1\n").unwrap();

    let err = ParakeetModel::load(dir.path()).err().unwrap();

    assert_eq!(
        err.to_string(),
        format!(
            "cannot load parakeet model {}",
            dir.path().join("nemo128.onnx").display()
        )
    );
}
