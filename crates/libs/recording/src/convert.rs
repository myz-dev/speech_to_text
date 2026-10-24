use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

/// Frames per FFT chunk. Only affects speed and latency, not the result length.
const RESAMPLER_CHUNK_SIZE: usize = 1024;

#[derive(Debug, thiserror::Error)]
pub enum ResampleError {
    #[error("cannot create resampler from {from_rate} Hz to {to_rate} Hz")]
    Setup {
        from_rate: u32,
        to_rate: u32,
        #[source]
        source: rubato::ResamplerConstructionError,
    },
    #[error("resampling failed")]
    Process(#[source] rubato::ResampleError),
}

/// Averages interleaved multi-channel samples into one channel.
///
/// An incomplete trailing frame is dropped.
///
/// # Panics
/// If `channels` is 0. WAV files with 0 channels are rejected by the reader already.
pub fn to_mono(interleaved: Vec<f32>, channels: usize) -> Vec<f32> {
    if channels == 1 {
        return interleaved;
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Converts mono samples from one sample rate to another.
///
/// Uses an FFT resampler with an anti-aliasing filter: when downsampling, content above
/// the new Nyquist frequency (half of `to_rate`) is removed instead of folding back into
/// the audible range. The output has `ceil(len * to_rate / from_rate)` samples.
pub fn resample(
    samples: Vec<f32>,
    from_rate: u32,
    to_rate: u32,
) -> Result<Vec<f32>, ResampleError> {
    if from_rate == to_rate || samples.is_empty() {
        return Ok(samples);
    }

    let mut resampler = resampler(from_rate, to_rate)?;
    // Invariant: one channel with `samples.len()` frames always matches the slice length.
    let input = InterleavedSlice::new(&samples, 1, samples.len())
        .expect("mono slice length equals its frame count");
    let output = resampler
        .process_all(&input, samples.len(), None)
        .map_err(ResampleError::Process)?;
    Ok(output.take_data())
}

/// A mono FFT resampler from `from_rate` to `to_rate` that takes chunks of a fixed
/// number of input samples ([`Resampler::input_frames_next`]).
pub(crate) fn resampler(from_rate: u32, to_rate: u32) -> Result<Fft<f32>, ResampleError> {
    Fft::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        RESAMPLER_CHUNK_SIZE,
        1,
        FixedSync::Input,
    )
    .map_err(|source| ResampleError::Setup {
        from_rate,
        to_rate,
        source,
    })
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use super::*;

    fn sine(frequency: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
        let len = (sample_rate as f32 * seconds) as usize;
        (0..len)
            .map(|i| (TAU * frequency * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    /// Positive-going zero crossings, i.e. full periods of a sine.
    fn periods(samples: &[f32]) -> usize {
        samples
            .windows(2)
            .filter(|w| w[0] < 0.0 && w[1] >= 0.0)
            .count()
    }

    #[test]
    fn to_mono_averages_each_frame() {
        let stereo = [1.0, -1.0, 0.5, 0.25, 0.0, 1.0];
        assert_eq!(to_mono(stereo.to_vec(), 2), vec![0.0, 0.375, 0.5]);
    }

    #[test]
    fn to_mono_passes_mono_through() {
        let mono = [0.1, 0.2, 0.3];
        assert_eq!(to_mono(mono.to_vec(), 1), mono.to_vec());
    }

    #[test]
    fn to_mono_drops_incomplete_trailing_frame() {
        assert_eq!(to_mono(vec![1.0, 1.0, 0.5], 2), vec![1.0]);
    }

    #[test]
    fn resample_same_rate_is_identity() {
        let samples = sine(440.0, 16_000, 0.1);
        assert_eq!(resample(samples.clone(), 16_000, 16_000).unwrap(), samples);
    }

    #[test]
    fn resample_empty_input_is_empty() {
        assert!(resample(vec![], 48_000, 16_000).unwrap().is_empty());
    }

    #[test]
    fn resample_48k_to_16k_keeps_duration_and_pitch() {
        let output = resample(sine(440.0, 48_000, 1.0), 48_000, 16_000).unwrap();
        assert_eq!(output.len(), 16_000);
        // 440 periods in one second; allow one off at the edges.
        assert!(periods(&output).abs_diff(440) <= 1, "{}", periods(&output));
    }

    #[test]
    fn resample_keeps_events_at_the_same_time() {
        // A click at 0.5 s must still be at 0.5 s, i.e. no filter delay in the output.
        let mut input = vec![0.0; 48_000];
        input[24_000] = 1.0;
        let output = resample(input, 48_000, 16_000).unwrap();
        let peak = (0..output.len())
            .max_by(|&a, &b| output[a].total_cmp(&output[b]))
            .unwrap();
        assert!(peak.abs_diff(8_000) <= 1, "peak at {peak}");
    }

    #[test]
    fn resample_44_1k_to_16k_keeps_duration() {
        let output = resample(sine(440.0, 44_100, 1.0), 44_100, 16_000).unwrap();
        assert_eq!(output.len(), 16_000);
    }

    #[test]
    fn resample_keeps_loudness_of_audible_tone() {
        let output = resample(sine(1_000.0, 48_000, 1.0), 48_000, 16_000).unwrap();
        // A full-scale sine has RMS 1/sqrt(2); skip the edges where the filter ramps in.
        let middle = &output[1_000..15_000];
        assert!((rms(middle) - 0.707).abs() < 0.02, "rms {}", rms(middle));
    }

    #[test]
    fn resample_removes_tones_above_new_nyquist() {
        // 12 kHz cannot be represented at 16 kHz (Nyquist 8 kHz). Without anti-aliasing it
        // would fold back to a loud 4 kHz tone.
        let output = resample(sine(12_000.0, 48_000, 1.0), 48_000, 16_000).unwrap();
        let middle = &output[1_000..15_000];
        assert!(rms(middle) < 0.01, "rms {}", rms(middle));
    }

    #[test]
    fn resample_rejects_zero_rate() {
        let err = resample(vec![0.0; 100], 0, 16_000).unwrap_err();
        assert!(matches!(err, ResampleError::Setup { from_rate: 0, .. }));
    }
}
