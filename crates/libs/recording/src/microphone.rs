use std::sync::mpsc::{self, Sender};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, ErrorKind, FromSample, SampleFormat, SizedSample, Stream, StreamConfig};

use crate::SAMPLE_RATE;
use crate::convert::{ResampleError, resample, to_mono};

#[derive(Debug, thiserror::Error)]
pub enum RecordError {
    #[error("no microphone found")]
    NoInputDevice,
    #[error("cannot read the input format of {device}")]
    Config {
        device: String,
        #[source]
        source: cpal::Error,
    },
    #[error("{device} delivers {format} samples, which are not supported")]
    UnsupportedSampleFormat {
        device: String,
        format: SampleFormat,
    },
    #[error("cannot start recording from {device}")]
    Start {
        device: String,
        #[source]
        source: cpal::Error,
    },
    #[error("recording from {device} failed")]
    Stream {
        device: String,
        #[source]
        source: cpal::Error,
    },
    #[error(
        "{device} delivered only digital silence; check that this app may use \
         the microphone (System Settings → Privacy & Security → Microphone)"
    )]
    Silent { device: String },
    #[error("cannot convert the recording to {SAMPLE_RATE} Hz")]
    Resample(#[source] ResampleError),
}

/// A microphone recording, converted to 16 kHz mono samples.
#[derive(Debug)]
pub struct Recording {
    /// 16 kHz mono samples.
    pub samples: Vec<f32>,
    /// Name of the input device, to tell which microphone was used.
    pub device: String,
    /// How often the device reported an overload while recording. Each one may have
    /// left a gap in the audio.
    /// macOS reports overloads for the whole device, not per stream, e.g. when the
    /// stream starts while the microphone permission dialog is open.
    pub overruns: usize,
}

/// Records from the default input device until `wait` returns.
///
/// `wait` runs on the calling thread while the device captures audio on its own thread,
/// e.g. a closure that blocks until the user presses Enter. The device's native format
/// is converted to 16 kHz mono afterwards.
///
/// macOS does not report a denied microphone permission as an error; the device then
/// delivers exact zeros. A recording that consists only of zeros is therefore rejected
/// as [`RecordError::Silent`]. A real microphone always picks up some noise.
pub fn record(wait: impl FnOnce()) -> Result<Recording, RecordError> {
    let device = cpal::default_host()
        .default_input_device()
        .ok_or(RecordError::NoInputDevice)?;
    let name = device_name(&device);

    let supported = device
        .default_input_config()
        .map_err(|source| RecordError::Config {
            device: name.clone(),
            source,
        })?;
    let config = supported.config();

    let (sample_sender, sample_receiver) = mpsc::channel();
    let (error_sender, error_receiver) = mpsc::channel();
    let stream = match supported.sample_format() {
        SampleFormat::F32 => build_stream::<f32>(&device, config, sample_sender, error_sender),
        SampleFormat::I16 => build_stream::<i16>(&device, config, sample_sender, error_sender),
        SampleFormat::I32 => build_stream::<i32>(&device, config, sample_sender, error_sender),
        format => {
            return Err(RecordError::UnsupportedSampleFormat {
                device: name,
                format,
            });
        }
    };
    let start_error = |source| RecordError::Start {
        device: name.clone(),
        source,
    };
    let stream = stream.map_err(start_error)?;
    stream.play().map_err(start_error)?;

    wait();
    // After the drop no callback runs anymore, so the channels hold all captured audio.
    // `try_iter` drains them without waiting for the senders, which cpal may keep alive.
    drop(stream);

    let errors: Vec<cpal::Error> = error_receiver.try_iter().collect();
    if let Some(source) = first_fatal_error(&errors) {
        return Err(RecordError::Stream {
            device: name,
            source: source.clone(),
        });
    }
    let interleaved: Vec<f32> = sample_receiver.try_iter().flatten().collect();
    if is_digital_silence(&interleaved) {
        return Err(RecordError::Silent { device: name });
    }

    let mono = to_mono(interleaved, config.channels as usize);
    let samples = resample(mono, config.sample_rate, SAMPLE_RATE).map_err(RecordError::Resample)?;
    Ok(Recording {
        samples,
        device: name,
        overruns: count_overruns(&errors),
    })
}

/// Opens an input stream that sends each buffer, converted to `f32`, to `samples`.
fn build_stream<T>(
    device: &Device,
    config: StreamConfig,
    samples: Sender<Vec<f32>>,
    errors: Sender<cpal::Error>,
) -> Result<Stream, cpal::Error>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            // The receivers live until `record` returns, after the stream is dropped, so a
            // failed send cannot lose audio that anyone would read.
            let _ = samples.send(data.iter().map(|&s| s.to_sample::<f32>()).collect());
        },
        move |error| {
            let _ = errors.send(error);
        },
        None,
    )
}

/// The first stream error that ends the recording. Not fatal: a rerouted default device
/// keeps recording, missing real-time scheduling only risks glitches, and glitches
/// themselves are counted by [`count_overruns`].
fn first_fatal_error(errors: &[cpal::Error]) -> Option<&cpal::Error> {
    errors.iter().find(|error| {
        !matches!(
            error.kind(),
            ErrorKind::DeviceChanged | ErrorKind::RealtimeDenied | ErrorKind::Xrun
        )
    })
}

fn count_overruns(errors: &[cpal::Error]) -> usize {
    errors
        .iter()
        .filter(|error| error.kind() == ErrorKind::Xrun)
        .count()
}

/// True if a non-empty recording contains only exact zeros. An empty recording (stopped
/// before the first buffer arrived) is not silence, just short.
fn is_digital_silence(samples: &[f32]) -> bool {
    !samples.is_empty() && samples.iter().all(|&s| s == 0.0)
}

fn device_name(device: &Device) -> String {
    device
        .description()
        .map(|description| description.name().to_owned())
        .unwrap_or_else(|_| "unnamed input device".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_zeros_is_digital_silence() {
        assert!(is_digital_silence(&[0.0; 480]));
    }

    #[test]
    fn faint_noise_is_not_digital_silence() {
        let mut samples = vec![0.0; 480];
        samples[100] = 1.0e-6;
        assert!(!is_digital_silence(&samples));
    }

    #[test]
    fn empty_recording_is_not_digital_silence() {
        assert!(!is_digital_silence(&[]));
    }

    #[test]
    fn rerouted_device_denied_realtime_and_overrun_are_not_fatal() {
        let errors = [
            ErrorKind::DeviceChanged.into(),
            ErrorKind::RealtimeDenied.into(),
            ErrorKind::Xrun.into(),
        ];
        assert!(first_fatal_error(&errors).is_none());
    }

    #[test]
    fn disconnected_device_is_fatal() {
        let errors = [ErrorKind::Xrun.into(), ErrorKind::DeviceNotAvailable.into()];
        let error = first_fatal_error(&errors).unwrap();
        assert_eq!(error.kind(), ErrorKind::DeviceNotAvailable);
    }

    #[test]
    fn counts_only_overruns() {
        let errors = [
            ErrorKind::Xrun.into(),
            ErrorKind::DeviceChanged.into(),
            ErrorKind::Xrun.into(),
        ];
        assert_eq!(count_overruns(&errors), 2);
    }
}
