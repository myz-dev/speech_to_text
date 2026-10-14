use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavSpec, WavSpecEx, WavWriter};
use recording::{LoadWavError, SAMPLE_RATE, load_wav};

fn spec(channels: u16, sample_rate: u32, bits: u16, format: SampleFormat) -> WavSpec {
    WavSpec {
        channels,
        sample_rate,
        bits_per_sample: bits,
        sample_format: format,
    }
}

fn write_int(dir: &Path, spec: WavSpec, samples: &[i32]) -> PathBuf {
    let path = dir.join("int.wav");
    let mut writer = WavWriter::create(&path, spec).unwrap();
    for &s in samples {
        if spec.bits_per_sample == 8 {
            writer.write_sample(s as i8).unwrap();
        } else {
            writer.write_sample(s).unwrap();
        }
    }
    writer.finalize().unwrap();
    path
}

/// WAVE_FORMAT_EXTENSIBLE header with a 16-bit container whose "valid bits" field is
/// `valid_bits`, followed by two zero samples. hound copies the field into
/// `bits_per_sample` without checking it.
fn extensible_header_bytes(valid_bits: u16) -> Vec<u8> {
    const PCM_SUBFORMAT: [u8; 16] = [
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B,
        0x71,
    ];
    let mut fmt = Vec::new();
    fmt.extend(0xFFFE_u16.to_le_bytes()); // format tag: extensible
    fmt.extend(1_u16.to_le_bytes()); // channels
    fmt.extend(SAMPLE_RATE.to_le_bytes());
    fmt.extend((SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    fmt.extend(2_u16.to_le_bytes()); // block align
    fmt.extend(16_u16.to_le_bytes()); // container bits
    fmt.extend(22_u16.to_le_bytes()); // extension size
    fmt.extend(valid_bits.to_le_bytes());
    fmt.extend(4_u32.to_le_bytes()); // channel mask
    fmt.extend(PCM_SUBFORMAT);
    let data = [0_u8; 4];

    let mut bytes = Vec::new();
    bytes.extend(b"RIFF");
    bytes.extend((4 + 8 + fmt.len() as u32 + 8 + data.len() as u32).to_le_bytes());
    bytes.extend(b"WAVE");
    bytes.extend(b"fmt ");
    bytes.extend((fmt.len() as u32).to_le_bytes());
    bytes.extend(fmt);
    bytes.extend(b"data");
    bytes.extend((data.len() as u32).to_le_bytes());
    bytes.extend(data);
    bytes
}

fn write_float(dir: &Path, spec: WavSpec, samples: &[f32]) -> PathBuf {
    let path = dir.join("float.wav");
    let mut writer = WavWriter::create(&path, spec).unwrap();
    for &s in samples {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
    path
}

#[test]
fn float_mono_16k_is_returned_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let samples = [0.0, 0.5, -0.5, 1.0, -1.0];
    let path = write_float(
        dir.path(),
        spec(1, SAMPLE_RATE, 32, SampleFormat::Float),
        &samples,
    );
    assert_eq!(load_wav(&path).unwrap(), samples);
}

#[test]
fn int16_is_scaled_to_unit_range() {
    let dir = tempfile::tempdir().unwrap();
    let samples = [0, 16_384, -16_384, i16::MIN as i32];
    let path = write_int(
        dir.path(),
        spec(1, SAMPLE_RATE, 16, SampleFormat::Int),
        &samples,
    );
    assert_eq!(load_wav(&path).unwrap(), [0.0, 0.5, -0.5, -1.0]);
}

#[test]
fn int24_is_scaled_to_unit_range() {
    let dir = tempfile::tempdir().unwrap();
    let samples = [0, 1 << 22, -(1 << 23)];
    let path = write_int(
        dir.path(),
        spec(1, SAMPLE_RATE, 24, SampleFormat::Int),
        &samples,
    );
    assert_eq!(load_wav(&path).unwrap(), [0.0, 0.5, -1.0]);
}

#[test]
fn int8_is_scaled_to_unit_range() {
    let dir = tempfile::tempdir().unwrap();
    let samples = [0, 64, -128];
    let path = write_int(
        dir.path(),
        spec(1, SAMPLE_RATE, 8, SampleFormat::Int),
        &samples,
    );
    assert_eq!(load_wav(&path).unwrap(), [0.0, 0.5, -1.0]);
}

#[test]
fn int32_is_scaled_to_unit_range() {
    let dir = tempfile::tempdir().unwrap();
    let samples = [0, 1 << 30, i32::MIN];
    let path = write_int(
        dir.path(),
        spec(1, SAMPLE_RATE, 32, SampleFormat::Int),
        &samples,
    );
    assert_eq!(load_wav(&path).unwrap(), [0.0, 0.5, -1.0]);
}

#[test]
fn int24_in_4_byte_container_is_scaled_to_unit_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("int24in32.wav");
    let spec_ex = WavSpecEx {
        spec: spec(1, SAMPLE_RATE, 24, SampleFormat::Int),
        bytes_per_sample: 4,
    };
    let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    let mut writer = WavWriter::new_with_spec_ex(file, spec_ex).unwrap();
    for s in [0, 1 << 22, -(1 << 23)] {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
    assert_eq!(load_wav(&path).unwrap(), [0.0, 0.5, -1.0]);
}

#[test]
fn out_of_range_bit_depth_in_header_is_a_read_error() {
    let dir = tempfile::tempdir().unwrap();
    for valid_bits in [12, 64, 100] {
        let path = dir.path().join(format!("valid_bits_{valid_bits}.wav"));
        std::fs::write(&path, extensible_header_bytes(valid_bits)).unwrap();
        let result = load_wav(&path);
        assert!(
            matches!(result, Err(LoadWavError::Read { .. })),
            "valid bits {valid_bits}: {result:?}"
        );
    }
}

#[test]
fn extensible_header_with_matching_bits_loads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("extensible.wav");
    std::fs::write(&path, extensible_header_bytes(16)).unwrap();
    assert_eq!(load_wav(&path).unwrap(), [0.0, 0.0]);
}

#[test]
fn stereo_is_averaged_to_mono() {
    let dir = tempfile::tempdir().unwrap();
    // Frames: (0.5, -0.5), (0.5, 0.0)
    let samples = [16_384, -16_384, 16_384, 0];
    let path = write_int(
        dir.path(),
        spec(2, SAMPLE_RATE, 16, SampleFormat::Int),
        &samples,
    );
    assert_eq!(load_wav(&path).unwrap(), [0.0, 0.25]);
}

#[test]
fn stereo_48k_becomes_one_second_of_16k_mono() {
    let dir = tempfile::tempdir().unwrap();
    let samples = vec![0; 2 * 48_000];
    let path = write_int(dir.path(), spec(2, 48_000, 16, SampleFormat::Int), &samples);
    assert_eq!(load_wav(&path).unwrap().len(), SAMPLE_RATE as usize);
}

#[test]
fn missing_file_reports_its_path() {
    let err = load_wav(Path::new("does/not/exist.wav")).unwrap_err();
    assert!(matches!(err, LoadWavError::Read { .. }));
    assert_eq!(err.to_string(), "cannot read does/not/exist.wav");
}

#[test]
fn non_wav_file_is_a_read_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("notes.wav");
    std::fs::write(&path, "not audio").unwrap();
    assert!(matches!(load_wav(&path), Err(LoadWavError::Read { .. })));
}
