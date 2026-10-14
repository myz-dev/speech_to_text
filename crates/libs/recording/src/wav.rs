use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavReader};

use crate::SAMPLE_RATE;
use crate::convert::{ResampleError, resample, to_mono};

#[derive(Debug, thiserror::Error)]
pub enum LoadWavError {
    #[error("cannot read {}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },
    #[error("cannot convert {} to {SAMPLE_RATE} Hz", path.display())]
    Resample {
        path: PathBuf,
        #[source]
        source: ResampleError,
    },
}

/// Loads a WAV file as 16 kHz mono samples, the input format of the speech models.
///
/// Accepts integer PCM (8–32 bit) and 32-bit float, any channel count and sample rate.
/// Channels are averaged; other sample rates are resampled.
pub fn load_wav(path: &Path) -> Result<Vec<f32>, LoadWavError> {
    let read_error = |source| LoadWavError::Read {
        path: path.to_path_buf(),
        source,
    };

    let reader = WavReader::open(path).map_err(read_error)?;
    let spec = reader.spec();
    let interleaved = read_samples(reader).map_err(read_error)?;

    let mono = to_mono(interleaved, spec.channels as usize);
    resample(mono, spec.sample_rate, SAMPLE_RATE).map_err(|source| LoadWavError::Resample {
        path: path.to_path_buf(),
        source,
    })
}

/// Reads all samples as `f32`, still interleaved. Integers are scaled to `-1.0..1.0`;
/// floats are returned as stored.
fn read_samples<R: std::io::Read>(reader: WavReader<R>) -> Result<Vec<f32>, hound::Error> {
    let spec = reader.spec();
    match spec.sample_format {
        SampleFormat::Float => reader.into_samples::<f32>().collect(),
        SampleFormat::Int => {
            // Signed n-bit integers span -2^(n-1)..2^(n-1); dividing by 2^(n-1) maps them
            // to -1.0..1.0. hound returns 8-bit WAV (unsigned on disk) already signed.
            // `bits_per_sample` comes from the file header and can hold any value; hound
            // only decodes these four widths.
            let scale: f32 = match spec.bits_per_sample {
                8 => 128.0,
                16 => 32_768.0,
                24 => 8_388_608.0,
                32 => 2_147_483_648.0,
                _ => return Err(hound::Error::Unsupported),
            };
            reader
                .into_samples::<i32>()
                .map(|sample| sample.map(|s| s as f32 / scale))
                .collect()
        }
    }
}
