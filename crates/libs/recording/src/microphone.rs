use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

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
    ///
    /// macOS reports overloads for the whole device, not per stream, e.g. when the
    /// stream starts while the microphone permission dialog is open.
    pub overruns: usize,
}

/// Whether [`record_streaming`] keeps recording, returned by its callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

/// How long [`record_streaming`] waits for a buffer before it calls back without audio.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Exact zeros for this long mean the microphone permission is denied, not a slow start.
const SILENT_LIMIT_SECS: usize = 1;

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
    let input = Input::open()?;
    wait();
    // After the drop no callback runs anymore, so the channels hold all captured audio.
    // `try_iter` drains them without waiting for the senders, which cpal may keep alive.
    drop(input.stream);

    let errors: Vec<cpal::Error> = input.errors.try_iter().collect();
    let interleaved: Vec<f32> = input.samples.try_iter().flatten().collect();
    let mono = to_mono(interleaved, input.channels);
    finish(input.device, mono, input.sample_rate, &errors)
}

/// Records from the default input device and hands the audio to `on_audio` as it
/// arrives, until `on_audio` returns [`Flow::Stop`].
///
/// `on_audio` runs on the calling thread with the new mono samples at the device's
/// native rate, and that rate. It is called at least every 20 ms, with an empty slice
/// when no audio arrived, so it can stop on other events too. The whole recording is
/// converted to 16 kHz once, after the stop.
///
/// Like [`record`], a recording of only zeros is [`RecordError::Silent`]; here the
/// recording stops after a second of them instead of waiting for `on_audio`. If
/// `on_audio` stops before the first buffer arrives, the recording is empty.
pub fn record_streaming(
    on_audio: impl FnMut(&[f32], u32) -> Flow,
) -> Result<Recording, RecordError> {
    let input = Input::open()?;
    let (mut mono, mut errors) = stream_mono(
        &input.samples,
        &input.errors,
        input.channels,
        input.sample_rate,
        on_audio,
    );
    drop(input.stream);

    // Audio captured between the stop and the drop belongs to the recording too.
    for buffer in input.samples.try_iter() {
        mono.extend(to_mono(buffer, input.channels));
    }
    errors.extend(input.errors.try_iter());
    finish(input.device, mono, input.sample_rate, &errors)
}

/// The default input device, started and capturing into two channels: one for buffers
/// of interleaved `f32` samples at its native rate, one for stream errors.
struct Input {
    stream: Stream,
    device: String,
    channels: usize,
    sample_rate: u32,
    samples: Receiver<Vec<f32>>,
    errors: Receiver<cpal::Error>,
}

impl Input {
    fn open() -> Result<Self, RecordError> {
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

        let (sample_sender, samples) = mpsc::channel();
        let (error_sender, errors) = mpsc::channel();
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

        Ok(Self {
            stream,
            device: name,
            channels: usize::from(config.channels),
            sample_rate: config.sample_rate,
            samples,
            errors,
        })
    }
}

/// Receives buffers, downmixes each to mono and passes it to `on_audio`, until
/// `on_audio` stops, a fatal stream error arrives, the senders are gone, or a second of
/// exact zeros has arrived. Returns the mono audio and the errors seen.
fn stream_mono(
    samples: &Receiver<Vec<f32>>,
    errors: &Receiver<cpal::Error>,
    channels: usize,
    sample_rate: u32,
    mut on_audio: impl FnMut(&[f32], u32) -> Flow,
) -> (Vec<f32>, Vec<cpal::Error>) {
    let silent_limit = sample_rate as usize * SILENT_LIMIT_SECS;
    let mut mono = Vec::new();
    let mut seen_errors = Vec::new();
    let mut heard_signal = false;
    loop {
        seen_errors.extend(errors.try_iter());
        if seen_errors.iter().any(is_fatal) {
            break;
        }
        let new = match samples.recv_timeout(POLL_INTERVAL) {
            Ok(buffer) => to_mono(buffer, channels),
            Err(RecvTimeoutError::Timeout) => Vec::new(),
            Err(RecvTimeoutError::Disconnected) => break,
        };
        heard_signal = heard_signal || new.iter().any(|&sample| sample != 0.0);
        let start = mono.len();
        mono.extend(new);
        if !heard_signal && mono.len() >= silent_limit {
            break;
        }
        if on_audio(&mono[start..], sample_rate) == Flow::Stop {
            break;
        }
    }
    (mono, seen_errors)
}

/// Turns captured mono audio into a [`Recording`], or the error that ended it.
fn finish(
    device: String,
    mono: Vec<f32>,
    sample_rate: u32,
    errors: &[cpal::Error],
) -> Result<Recording, RecordError> {
    if let Some(source) = errors.iter().find(|error| is_fatal(error)) {
        return Err(RecordError::Stream {
            device,
            source: source.clone(),
        });
    }
    if is_digital_silence(&mono) {
        return Err(RecordError::Silent { device });
    }
    let samples = resample(mono, sample_rate, SAMPLE_RATE).map_err(RecordError::Resample)?;
    Ok(Recording {
        samples,
        device,
        overruns: count_overruns(errors),
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
            // The receivers live until the recording function returns, after the stream is
            // dropped, so a failed send cannot lose audio that anyone would read.
            let _ = samples.send(data.iter().map(|&s| s.to_sample::<f32>()).collect());
        },
        move |error| {
            let _ = errors.send(error);
        },
        None,
    )
}

/// Whether a stream error ends the recording.
///
/// Recoverable, the recording continues:
/// - `DeviceChanged`: the default device was rerouted. cpal documents that the stream
///   stays active on backends that reroute. On macOS input streams never reroute: they
///   keep recording from the device they started with, even if a newly plugged-in
///   headset becomes the default, and no error is raised.
/// - `RealtimeDenied`: the audio thread runs without real-time priority. It only risks
///   glitches, and those are reported as `Xrun` themselves.
/// - `Xrun`: a device overload that may have left a short gap in the audio. Counted in
///   [`Recording::overruns`] instead, because macOS reports it for the whole device, not
///   for this stream.
///
/// Fatal: every other kind, which means the stream stopped delivering audio or can no
/// longer be trusted, e.g. `DeviceNotAvailable` (microphone unplugged) or
/// `StreamInvalidated` (device sample rate changed, so the samples no longer match
/// the rate we resample from). `ErrorKind` is non-exhaustive, so kinds added by future
/// cpal versions count as fatal until classified here.
fn is_fatal(error: &cpal::Error) -> bool {
    !matches!(
        error.kind(),
        ErrorKind::DeviceChanged | ErrorKind::RealtimeDenied | ErrorKind::Xrun
    )
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

    const RATE: u32 = 1_000;

    /// Runs `stream_mono` on `buffers`, already queued, with the senders kept alive so
    /// only `on_audio`, an error or digital silence ends it.
    fn stream(
        buffers: &[Vec<f32>],
        stream_errors: &[ErrorKind],
        channels: usize,
        on_audio: impl FnMut(&[f32], u32) -> Flow,
    ) -> (Vec<f32>, Vec<cpal::Error>) {
        let (sample_sender, samples) = mpsc::channel();
        let (error_sender, errors) = mpsc::channel();
        for buffer in buffers {
            sample_sender.send(buffer.clone()).unwrap();
        }
        for &kind in stream_errors {
            error_sender.send(kind.into()).unwrap();
        }
        stream_mono(&samples, &errors, channels, RATE, on_audio)
    }

    /// Stops once `on_audio` has seen `limit` calls.
    fn stop_after(limit: usize) -> impl FnMut(&[f32], u32) -> Flow {
        let mut calls = 0;
        move |_, _| {
            calls += 1;
            if calls >= limit {
                Flow::Stop
            } else {
                Flow::Continue
            }
        }
    }

    #[test]
    fn stream_hands_each_buffer_downmixed_with_the_native_rate() {
        let buffers = [vec![0.1, 0.3, 0.5, 0.7], vec![-0.2, 0.2]];
        let mut seen = Vec::new();
        let (mono, _) = stream(&buffers, &[], 2, |audio, rate| {
            seen.push((audio.to_vec(), rate));
            if seen.len() == 2 {
                Flow::Stop
            } else {
                Flow::Continue
            }
        });
        assert_eq!(seen, [(vec![0.2, 0.6], RATE), (vec![0.0], RATE)]);
        assert_eq!(mono, [0.2, 0.6, 0.0]);
    }

    #[test]
    fn stream_stops_when_on_audio_says_so() {
        let buffers = [vec![0.1; 10], vec![0.2; 10], vec![0.3; 10]];
        let (mono, _) = stream(&buffers, &[], 1, stop_after(2));
        assert_eq!(mono.len(), 20);
    }

    #[test]
    fn stream_calls_on_audio_without_audio_every_poll_interval() {
        let started = std::time::Instant::now();
        let mut calls = Vec::new();
        let (mono, _) = stream(&[], &[], 1, |audio, _| {
            calls.push(audio.len());
            if calls.len() == 3 {
                Flow::Stop
            } else {
                Flow::Continue
            }
        });
        assert_eq!(calls, [0, 0, 0]);
        assert!(mono.is_empty());
        assert!(started.elapsed() >= 3 * POLL_INTERVAL);
    }

    #[test]
    fn stream_ends_when_the_device_stops_sending() {
        let (sample_sender, samples) = mpsc::channel();
        let (_error_sender, errors) = mpsc::channel();
        sample_sender.send(vec![0.1; 10]).unwrap();
        drop(sample_sender);
        let (mono, _) = stream_mono(&samples, &errors, 1, RATE, |_, _| Flow::Continue);
        assert_eq!(mono.len(), 10);
    }

    #[test]
    fn stream_stops_at_a_fatal_error_and_reports_it() {
        let buffers = [vec![0.1; 10], vec![0.1; 10]];
        let (mono, errors) = stream(&buffers, &[ErrorKind::DeviceNotAvailable], 1, |_, _| {
            Flow::Continue
        });
        assert!(mono.is_empty());
        assert!(errors.iter().any(is_fatal));
    }

    #[test]
    fn stream_keeps_recording_through_an_overrun() {
        let buffers = [vec![0.1; 10], vec![0.1; 10]];
        let (mono, errors) = stream(&buffers, &[ErrorKind::Xrun], 1, stop_after(2));
        assert_eq!(mono.len(), 20);
        assert_eq!(count_overruns(&errors), 1);
    }

    #[test]
    fn stream_stops_after_a_second_of_digital_silence() {
        let buffers = vec![vec![0.0; 100]; 15];
        let (mono, _) = stream(&buffers, &[], 1, |_, _| Flow::Continue);
        assert_eq!(mono.len(), RATE as usize);
        assert!(is_digital_silence(&mono));
    }

    #[test]
    fn stream_keeps_recording_when_signal_follows_a_short_silent_start() {
        let mut buffers = vec![vec![0.0; 100]; 5];
        buffers.extend(vec![vec![0.01; 100]; 10]);
        let (mono, _) = stream(&buffers, &[], 1, stop_after(15));
        assert_eq!(mono.len(), 1_500);
    }

    #[test]
    fn fatal_error_wins_over_digital_silence() {
        let errors = [ErrorKind::DeviceNotAvailable.into()];
        let result = finish("mic".to_owned(), vec![0.0; 480], RATE, &errors);
        assert!(matches!(result, Err(RecordError::Stream { .. })));
    }

    #[test]
    fn finish_counts_overruns_in_a_good_recording() {
        let errors = [ErrorKind::Xrun.into()];
        let recording = finish("mic".to_owned(), vec![0.1; 480], RATE, &errors).unwrap();
        assert_eq!(recording.overruns, 1);
        assert!(!recording.samples.is_empty());
    }

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
    fn rerouted_device_denied_realtime_and_overrun_are_recoverable() {
        for kind in [
            ErrorKind::DeviceChanged,
            ErrorKind::RealtimeDenied,
            ErrorKind::Xrun,
        ] {
            assert!(!is_fatal(&kind.into()), "{kind:?}");
        }
    }

    #[test]
    fn disconnected_device_and_changed_sample_rate_are_fatal() {
        for kind in [ErrorKind::DeviceNotAvailable, ErrorKind::StreamInvalidated] {
            assert!(is_fatal(&kind.into()), "{kind:?}");
        }
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
