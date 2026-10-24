use std::ops::Range;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, ErrorKind, FromSample, SampleFormat, SizedSample, Stream, StreamConfig};

use crate::SAMPLE_RATE;
use crate::convert::{ResampleError, resample, to_mono};
use crate::silence::{self, Params};
use crate::vad::{self, Event, Vad, VadError};

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
    #[error("speech detection failed")]
    Vad(#[source] VadError),
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
    /// Range of `samples` to transcribe: from the first to the last speech, padded.
    /// `None` if Silero VAD heard no speech, e.g. only silence, coughs or typing.
    pub speech: Option<Range<usize>>,
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
/// is converted to 16 kHz mono afterwards. Speech is detected with `vad` and
/// `vad_params`, and trimmed with `params`, as in [`record_streaming`], after the
/// recording instead of during it.
///
/// macOS does not report a denied microphone permission as an error; the device then
/// delivers exact zeros. A recording that consists only of zeros is therefore rejected
/// as [`RecordError::Silent`]. A real microphone always picks up some noise.
pub fn record(
    params: &Params,
    vad: &mut Vad,
    vad_params: &vad::Params,
    wait: impl FnOnce(),
) -> Result<Recording, RecordError> {
    let input = Input::open()?;
    wait();
    // After the drop no callback runs anymore, so the channels hold all captured audio.
    // `try_iter` drains them without waiting for the senders, which cpal may keep alive.
    drop(input.stream);

    let errors: Vec<cpal::Error> = input.errors.try_iter().collect();
    let interleaved: Vec<f32> = input.samples.try_iter().flatten().collect();
    let mono = to_mono(interleaved, input.channels);
    let levels = mono
        .chunks_exact(silence::frame_len(input.sample_rate))
        .map(silence::level_db)
        .collect();
    let mut live = vad::Live::new(vad, *vad_params, input.sample_rate).map_err(RecordError::Vad)?;
    live.push(&mono).map_err(RecordError::Vad)?;
    let streamed = Streamed {
        mono,
        errors,
        levels,
        heard_speech: live.heard_speech(),
    };
    finish(input.device, streamed, input.sample_rate, params)
}

/// Records from the default input device, measuring it in 30 ms frames and running
/// Silero VAD on it as it arrives, until `on_frames` returns [`Flow::Stop`].
///
/// `on_frames` runs on the calling thread with the levels (dBFS) of the frames completed
/// since its last call and the speech events of the 32 ms chunks completed since then
/// ([`vad::Live`] with `vad_params`). It is called at least every
/// 20 ms, with no frames when no audio arrived, so it can stop on other events too. The
/// whole recording is converted to 16 kHz once more after the stop, for the speech
/// models; `params` trims it to the speech.
///
/// Like [`record`], a recording of only zeros is [`RecordError::Silent`]; here the
/// recording stops after a second of them instead of waiting for `on_frames`. If
/// `on_frames` stops before the first buffer arrives, the recording is empty.
pub fn record_streaming(
    params: &Params,
    vad: &mut Vad,
    vad_params: &vad::Params,
    on_frames: impl FnMut(&[f32], &[Event]) -> Flow,
) -> Result<Recording, RecordError> {
    let input = Input::open()?;
    let mut live = vad::Live::new(vad, *vad_params, input.sample_rate).map_err(RecordError::Vad)?;
    let streamed = stream_mono(
        &input.samples,
        &input.errors,
        input.channels,
        input.sample_rate,
        |mono| live.push(mono),
        on_frames,
    );
    drop(input.stream);
    // A failed speech detector ends the recording with its error, even if the device
    // also failed: the stream errors are only looked at in `finish`.
    let mut streamed = streamed.map_err(RecordError::Vad)?;

    // Audio captured between the stop and the drop belongs to the recording too.
    for buffer in input.samples.try_iter() {
        streamed.mono.extend(to_mono(buffer, input.channels));
    }
    streamed.errors.extend(input.errors.try_iter());
    finish(input.device, streamed, input.sample_rate, params)
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

/// What [`stream_mono`] captured.
struct Streamed {
    /// The recording, mono at the device's native rate.
    mono: Vec<f32>,
    errors: Vec<cpal::Error>,
    /// Level of every measured 30 ms frame of `mono`, in dBFS. Audio after the last
    /// whole frame, or received after the stop, is not measured.
    levels: Vec<f32>,
    /// Whether the speech detector reported [`Event::SpeechStarted`].
    heard_speech: bool,
}

/// Receives buffers, downmixes each to mono, passes it to `detect_speech`, measures
/// every completed 30 ms frame and passes the new levels and speech events to
/// `on_frames`, until `on_frames` stops, a fatal stream error arrives, the senders are
/// gone, or a second of exact zeros has arrived. An error of `detect_speech` ends it.
fn stream_mono(
    samples: &Receiver<Vec<f32>>,
    errors: &Receiver<cpal::Error>,
    channels: usize,
    sample_rate: u32,
    mut detect_speech: impl FnMut(&[f32]) -> Result<Vec<Event>, VadError>,
    mut on_frames: impl FnMut(&[f32], &[Event]) -> Flow,
) -> Result<Streamed, VadError> {
    let silent_limit = sample_rate as usize * SILENT_LIMIT_SECS;
    let frame_len = silence::frame_len(sample_rate);
    let mut mono = Vec::new();
    let mut seen_errors = Vec::new();
    let mut heard_signal = false;
    let mut heard_speech = false;
    let mut levels = Vec::new();
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
        let events = detect_speech(&new)?;
        heard_speech = heard_speech || events.contains(&Event::SpeechStarted);
        mono.extend(new);
        if !heard_signal && mono.len() >= silent_limit {
            break;
        }

        let measured_frames = levels.len();
        levels.extend(
            mono[measured_frames * frame_len..]
                .chunks_exact(frame_len)
                .map(silence::level_db),
        );
        if on_frames(&levels[measured_frames..], &events) == Flow::Stop {
            break;
        }
    }
    Ok(Streamed {
        mono,
        errors: seen_errors,
        levels,
        heard_speech,
    })
}

/// Turns captured audio into a [`Recording`], or the error that ended it.
fn finish(
    device: String,
    streamed: Streamed,
    sample_rate: u32,
    params: &Params,
) -> Result<Recording, RecordError> {
    let Streamed {
        mono,
        errors,
        levels,
        heard_speech,
    } = streamed;
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
    // Silero VAD decides whether there is speech: by loudness alone, coughs and typing
    // are speech. `speech_span` only trims, by loudness. Speech too quiet for it
    // (below `Params::min_start_db`) keeps the whole recording: audio is kept, not
    // words lost. Known limit: quiet speech next to a louder cough or typing is cut,
    // because the span then covers only the loud noise.
    let to_sample = |frame| frame_start_at_16k(frame, sample_rate).min(samples.len());
    let speech = heard_speech.then(|| match silence::speech_span(&levels, params) {
        Some(frames) => to_sample(frames.start)..to_sample(frames.end),
        None => 0..samples.len(),
    });
    Ok(Recording {
        samples,
        device,
        overruns: count_overruns(&errors),
        speech,
    })
}

/// The 16 kHz sample at which frame `frame`, measured at `sample_rate`, starts.
///
/// The frame starts at native sample `frame * frame_len`. A frame is not exactly 30 ms
/// at every rate (661 samples, 29.98 ms, at 22.05 kHz), so the time comes from the
/// native samples; `frame * 30 ms` would drift by 0.75 ms per second of recording.
fn frame_start_at_16k(frame: usize, sample_rate: u32) -> usize {
    frame * silence::frame_len(sample_rate) * SAMPLE_RATE as usize / sample_rate as usize
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

    /// 30-sample frames, so short buffers span frame boundaries.
    const RATE: u32 = 1_000;
    const FRAME_LEN: usize = 30;

    /// Constant samples at `db` dBFS: their RMS level is exactly `db`.
    fn level(db: f32, len: usize) -> Vec<f32> {
        vec![10_f32.powf(db / 20.0); len]
    }

    /// Runs `stream_mono` on `buffers`, already queued, with the senders kept alive so
    /// only `on_frames`, an error or digital silence ends it. No speech is detected.
    fn stream(
        buffers: &[Vec<f32>],
        stream_errors: &[ErrorKind],
        channels: usize,
        on_frames: impl FnMut(&[f32], &[Event]) -> Flow,
    ) -> Streamed {
        stream_detecting(
            buffers,
            stream_errors,
            channels,
            |_| Ok(Vec::new()),
            on_frames,
        )
        .unwrap()
    }

    /// Like [`stream`], with `detect_speech` as the speech detector.
    fn stream_detecting(
        buffers: &[Vec<f32>],
        stream_errors: &[ErrorKind],
        channels: usize,
        detect_speech: impl FnMut(&[f32]) -> Result<Vec<Event>, VadError>,
        on_frames: impl FnMut(&[f32], &[Event]) -> Flow,
    ) -> Result<Streamed, VadError> {
        let (sample_sender, samples) = mpsc::channel();
        let (error_sender, errors) = mpsc::channel();
        for buffer in buffers {
            sample_sender.send(buffer.clone()).unwrap();
        }
        for &kind in stream_errors {
            error_sender.send(kind.into()).unwrap();
        }
        stream_mono(&samples, &errors, channels, RATE, detect_speech, on_frames)
    }

    /// Stops once `on_frames` has seen `limit` calls.
    fn stop_after(limit: usize) -> impl FnMut(&[f32], &[Event]) -> Flow {
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
    fn stream_downmixes_each_buffer_in_order() {
        let buffers = [vec![0.1, 0.3, 0.5, 0.7], vec![-0.2, 0.2]];
        let streamed = stream(&buffers, &[], 2, stop_after(2));
        assert_eq!(streamed.mono, [0.2, 0.6, 0.0]);
    }

    #[test]
    fn stream_measures_frames_that_span_buffers() {
        // Four buffers of 20 samples: frames end after the 2nd and 3rd buffer.
        let buffers = vec![level(-20.0, 20); 4];
        let mut seen = Vec::new();
        let mut on_frames = stop_after(4);
        stream(&buffers, &[], 1, |levels, events| {
            seen.push(levels.to_vec());
            on_frames(levels, events)
        });
        let counts: Vec<usize> = seen.iter().map(Vec::len).collect();
        assert_eq!(counts, [0, 1, 1, 0]);
        for level_db in seen.concat() {
            assert!((level_db + 20.0).abs() < 0.01, "{level_db}");
        }
    }

    /// Reports [`Event::SpeechStarted`] for the first buffer that holds a sample of 0.5.
    fn detect_half_as_speech() -> impl FnMut(&[f32]) -> Result<Vec<Event>, VadError> {
        let mut started = false;
        move |mono| {
            if !started && mono.contains(&0.5) {
                started = true;
                return Ok(vec![Event::SpeechStarted]);
            }
            Ok(Vec::new())
        }
    }

    #[test]
    fn stream_passes_each_buffer_to_the_detector_and_its_events_to_on_frames() {
        let buffers = [vec![0.1; 10], vec![0.5; 10], vec![0.1; 10]];
        let mut calls = Vec::new();
        let mut on_frames = stop_after(buffers.len());
        let streamed = stream_detecting(
            &buffers,
            &[],
            1,
            detect_half_as_speech(),
            |frames, events| {
                calls.push(events.to_vec());
                on_frames(frames, events)
            },
        )
        .unwrap();
        assert_eq!(calls, [vec![], vec![Event::SpeechStarted], vec![]]);
        assert!(streamed.heard_speech);
    }

    #[test]
    fn stream_ends_with_the_error_of_the_detector() {
        let buffers = [vec![0.1; 10]];
        let result = stream_detecting(
            &buffers,
            &[],
            1,
            |_| Err(VadError::MissingOutput("output")),
            |_, _| Flow::Continue,
        );
        assert!(matches!(result, Err(VadError::MissingOutput("output"))));
    }

    #[test]
    fn stream_keeps_the_level_of_every_frame() {
        let buffers = vec![level(-20.0, 20); 4];
        let streamed = stream(&buffers, &[], 1, stop_after(4));
        // 80 samples: two whole frames of 30, the rest is not measured.
        assert_eq!(streamed.levels.len(), 2);
    }

    #[test]
    fn stream_without_speech_events_hears_no_speech() {
        // Loud enough that the energy detector would take it for speech.
        let buffers = vec![level(-20.0, FRAME_LEN); 10];
        let streamed = stream(&buffers, &[], 1, stop_after(10));
        assert!(!streamed.heard_speech);
    }

    #[test]
    fn stream_stops_when_on_frames_says_so() {
        let buffers = [vec![0.1; 10], vec![0.2; 10], vec![0.3; 10]];
        let streamed = stream(&buffers, &[], 1, stop_after(2));
        assert_eq!(streamed.mono.len(), 20);
    }

    #[test]
    fn stream_calls_on_frames_without_audio_every_poll_interval() {
        let started = std::time::Instant::now();
        let mut calls = Vec::new();
        let streamed = stream(&[], &[], 1, |frames, _| {
            calls.push(frames.len());
            if calls.len() == 3 {
                Flow::Stop
            } else {
                Flow::Continue
            }
        });
        assert_eq!(calls, [0, 0, 0]);
        assert!(streamed.mono.is_empty());
        assert!(started.elapsed() >= 3 * POLL_INTERVAL);
    }

    #[test]
    fn stream_ends_when_the_device_stops_sending() {
        let (sample_sender, samples) = mpsc::channel();
        let (_error_sender, errors) = mpsc::channel();
        sample_sender.send(vec![0.1; 10]).unwrap();
        drop(sample_sender);
        let streamed = stream_mono(
            &samples,
            &errors,
            1,
            RATE,
            |_| Ok(Vec::new()),
            |_, _| Flow::Continue,
        )
        .unwrap();
        assert_eq!(streamed.mono.len(), 10);
    }

    #[test]
    fn stream_stops_at_a_fatal_error_and_reports_it() {
        let buffers = [vec![0.1; 10], vec![0.1; 10]];
        let streamed = stream(&buffers, &[ErrorKind::DeviceNotAvailable], 1, |_, _| {
            Flow::Continue
        });
        assert!(streamed.mono.is_empty());
        assert!(streamed.errors.iter().any(is_fatal));
    }

    #[test]
    fn stream_keeps_recording_through_an_overrun() {
        let buffers = [vec![0.1; 10], vec![0.1; 10]];
        let streamed = stream(&buffers, &[ErrorKind::Xrun], 1, stop_after(2));
        assert_eq!(streamed.mono.len(), 20);
        assert_eq!(count_overruns(&streamed.errors), 1);
    }

    #[test]
    fn stream_stops_after_a_second_of_digital_silence() {
        let buffers = vec![vec![0.0; 100]; 15];
        let streamed = stream(&buffers, &[], 1, |_, _| Flow::Continue);
        assert_eq!(streamed.mono.len(), RATE as usize);
        assert!(is_digital_silence(&streamed.mono));
    }

    #[test]
    fn stream_keeps_recording_when_signal_follows_a_short_silent_start() {
        let mut buffers = vec![vec![0.0; 100]; 5];
        buffers.extend(vec![vec![0.01; 100]; 10]);
        let streamed = stream(&buffers, &[], 1, stop_after(15));
        assert_eq!(streamed.mono.len(), 1_500);
    }

    /// Captured audio for `finish`, measured in frames, without stream errors.
    fn streamed(mono: Vec<f32>, heard_speech: bool) -> Streamed {
        let levels = mono
            .chunks_exact(silence::frame_len(RATE))
            .map(silence::level_db)
            .collect();
        Streamed {
            mono,
            errors: Vec::new(),
            levels,
            heard_speech,
        }
    }

    #[test]
    fn fatal_error_wins_over_digital_silence() {
        let mut captured = streamed(vec![0.0; 480], false);
        captured.errors.push(ErrorKind::DeviceNotAvailable.into());
        let result = finish("mic".to_owned(), captured, RATE, &Params::default());
        assert!(matches!(result, Err(RecordError::Stream { .. })));
    }

    #[test]
    fn finish_rejects_digital_silence() {
        let captured = streamed(vec![0.0; 480], false);
        let result = finish("mic".to_owned(), captured, RATE, &Params::default());
        assert!(matches!(result, Err(RecordError::Silent { .. })));
    }

    #[test]
    fn finish_counts_overruns_in_a_good_recording() {
        let mut captured = streamed(vec![0.1; 480], false);
        captured.errors.push(ErrorKind::Xrun.into());
        let recording = finish("mic".to_owned(), captured, RATE, &Params::default()).unwrap();
        assert_eq!(recording.overruns, 1);
        assert!(!recording.samples.is_empty());
    }

    /// One second of quiet room, one of speech, two of quiet room.
    fn quiet_loud_quiet() -> Vec<f32> {
        let mut mono = level(-70.0, RATE as usize);
        mono.extend(level(-30.0, RATE as usize));
        mono.extend(level(-70.0, 2 * RATE as usize));
        mono
    }

    #[test]
    fn finish_trims_to_the_speech_the_detector_heard() {
        let captured = streamed(quiet_loud_quiet(), true);
        let recording = finish("mic".to_owned(), captured, RATE, &Params::default()).unwrap();
        let speech = recording.speech.expect("speech range");
        let seconds = |sample: usize| sample as f32 / SAMPLE_RATE as f32;
        assert!(seconds(speech.start) < 1.0 && seconds(speech.start) > 0.5);
        assert!(seconds(speech.end) > 2.0 && seconds(speech.end) < 2.5);
    }

    #[test]
    fn frame_start_follows_the_native_frame_length() {
        // 48 kHz: 1440 samples per frame, exactly 30 ms, 480 samples at 16 kHz.
        assert_eq!(frame_start_at_16k(10_000, 48_000), 4_800_000);
        // 22.05 kHz: 661 samples per frame; 10_000 frames are 299.77 s, not 300 s.
        assert_eq!(frame_start_at_16k(10_000, 22_050), 4_796_371);
    }

    #[test]
    fn finish_has_no_speech_when_the_detector_heard_none() {
        let captured = streamed(quiet_loud_quiet(), false);
        let recording = finish("mic".to_owned(), captured, RATE, &Params::default()).unwrap();
        assert_eq!(recording.speech, None);
    }

    #[test]
    fn finish_keeps_everything_when_the_detector_heard_speech_too_quiet_to_trim() {
        let captured = streamed(level(-70.0, 2 * RATE as usize), true);
        let recording = finish("mic".to_owned(), captured, RATE, &Params::default()).unwrap();
        assert_eq!(recording.speech, Some(0..recording.samples.len()));
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
