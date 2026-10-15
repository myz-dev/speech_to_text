use recording::{load_wav, save_wav};

#[test]
fn saved_samples_load_back_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recording.wav");
    let samples = vec![0.0, 0.5, -0.25, 1.0e-6, -1.0];

    save_wav(&path, &samples).unwrap();

    assert_eq!(load_wav(&path).unwrap(), samples);
}

#[test]
fn saves_16k_mono_float() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recording.wav");

    save_wav(&path, &[0.0; 160]).unwrap();

    let reader = hound::WavReader::open(&path).unwrap();
    let spec = reader.spec();
    assert_eq!(
        (spec.channels, spec.sample_rate, spec.bits_per_sample),
        (1, 16_000, 32)
    );
    assert_eq!(spec.sample_format, hound::SampleFormat::Float);
    assert_eq!(reader.len(), 160);
}

#[test]
fn unwritable_path_is_reported() {
    let err = save_wav("does/not/exist/out.wav".as_ref(), &[0.0]).unwrap_err();
    assert_eq!(err.to_string(), "cannot write does/not/exist/out.wav");
}
