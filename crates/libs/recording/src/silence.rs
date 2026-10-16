//! Energy-based speech detection on 30 ms frames.
//!
//! Two uses with the same threshold:
//! - [`step`] runs live while recording, one frame at a time, and reports when speech
//!   starts and ends. It only knows the past, so it estimates the noise floor as it goes.
//! - [`speech_span`] runs on a finished recording and finds the range from the first to
//!   the last speech, for trimming. It sees the whole recording, so the quietest frame
//!   is its noise floor.
//!
//! Pauses inside the speech are never removed: the speech models handle them.
//!
//! Input is mono audio at any sample rate. Interleaved multi-channel audio must be
//! downmixed first ([`crate::to_mono`]); otherwise a frame holds less than 30 ms. Frames
//! are cut by time and levels are in dB, neither depends on the rate, so the same
//! [`Params`] work for the microphone's native rate and for 16 kHz.

use std::ops::Range;

/// Length of one analysis frame.
pub const FRAME_MS: u32 = 30;

/// Level reported for digital silence, where the logarithm would be minus infinity.
pub const MIN_DB: f32 = -100.0;

/// Thresholds and durations of the detector. Durations are in frames of [`FRAME_MS`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Speech must be this much louder than the noise floor.
    pub start_above_floor_db: f32,
    /// ... and at least this loud, so a very quiet room does not turn breathing into
    /// speech.
    pub min_start_db: f32,
    /// The noise floor is never taken to be louder than this. Without the cap, a
    /// recording that is speech from its first frame to its last would have the speech
    /// as its floor, and no speech at all. A background louder than this counts as
    /// speech instead: audio is kept, not words lost.
    pub max_floor_db: f32,
    /// Speech ends this far below the start threshold. The gap keeps a level near the
    /// threshold from toggling between speech and silence.
    pub hysteresis_db: f32,
    /// How fast the live noise floor follows a louder background, per frame. It follows
    /// a quieter one at once.
    pub floor_rise_db: f32,
    /// Frames above the threshold before speech counts; shorter bursts are clicks.
    pub start_frames: u32,
    /// Frames below the end threshold before speech has ended.
    pub hangover_frames: u32,
    /// Frames without any speech before [`Event::NoSpeechTimeout`].
    pub no_speech_frames: u32,
    /// Frames kept before the first speech when trimming.
    pub pad_before_frames: u32,
    /// Frames kept after the last speech when trimming.
    pub pad_after_frames: u32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            start_above_floor_db: 12.0,
            min_start_db: -55.0,
            max_floor_db: -40.0,
            hysteresis_db: 6.0,
            floor_rise_db: 0.05,
            start_frames: frames(90),
            hangover_frames: frames(1_200),
            no_speech_frames: frames(10_000),
            pad_before_frames: frames(250),
            pad_after_frames: frames(400),
        }
    }
}

/// Whole frames in `ms` milliseconds.
pub const fn frames(ms: u32) -> u32 {
    ms / FRAME_MS
}

/// Samples in one frame at `sample_rate`. Zero below 34 Hz; audio runs at 8 kHz or more.
pub fn frame_len(sample_rate: u32) -> usize {
    (sample_rate * FRAME_MS / 1_000) as usize
}

/// Root mean square level of `frame` in dB relative to full scale: 0 dB is a square wave
/// at ±1, a full-scale sine is -3 dB. Silence and an empty frame give [`MIN_DB`].
pub fn level_db(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return MIN_DB;
    }
    let mean_square = frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32;
    // 10·log10 of the power is 20·log10 of the RMS. `max` turns -inf into MIN_DB.
    (10.0 * mean_square.log10()).max(MIN_DB)
}

/// Level a frame must reach to count as speech, given the noise floor.
fn start_threshold_db(floor_db: f32, params: &Params) -> f32 {
    (floor_db.min(params.max_floor_db) + params.start_above_floor_db).max(params.min_start_db)
}

/// What [`step`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    SpeechStarted,
    SpeechEnded,
    /// No speech in the first [`Params::no_speech_frames`] frames. Reported once.
    NoSpeechTimeout,
}

/// State of the live detector, passed into [`step`] and returned by it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detector {
    /// Estimated background level. Starts at +inf, so the first frame sets it.
    floor_db: f32,
    phase: Phase,
    frames_seen: u32,
    heard_speech: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    /// `loud_run` frames in a row were above the start threshold.
    Silence { loud_run: u32 },
    /// `quiet_run` frames in a row were below the end threshold.
    Speech { quiet_run: u32 },
}

impl Default for Detector {
    fn default() -> Self {
        Self {
            floor_db: f32::INFINITY,
            phase: Phase::Silence { loud_run: 0 },
            frames_seen: 0,
            heard_speech: false,
        }
    }
}

/// Feeds the level of the next frame to the detector.
///
/// The frame is compared with the noise floor of the frames before it; only then does
/// it update the floor. Otherwise a loud frame would raise the floor it is measured
/// against.
pub fn step(detector: Detector, level_db: f32, params: &Params) -> (Detector, Option<Event>) {
    let start_db = start_threshold_db(detector.floor_db, params);
    let end_db = start_db - params.hysteresis_db;

    let (phase, mut event) = match detector.phase {
        Phase::Silence { loud_run } => {
            let loud_run = if level_db >= start_db {
                loud_run + 1
            } else {
                0
            };
            if loud_run >= params.start_frames {
                (Phase::Speech { quiet_run: 0 }, Some(Event::SpeechStarted))
            } else {
                (Phase::Silence { loud_run }, None)
            }
        }
        Phase::Speech { quiet_run } => {
            let quiet_run = if level_db < end_db { quiet_run + 1 } else { 0 };
            if quiet_run >= params.hangover_frames {
                (Phase::Silence { loud_run: 0 }, Some(Event::SpeechEnded))
            } else {
                (Phase::Speech { quiet_run }, None)
            }
        }
    };

    let frames_seen = detector.frames_seen + 1;
    let heard_speech = detector.heard_speech || event == Some(Event::SpeechStarted);
    if !heard_speech && frames_seen == params.no_speech_frames {
        event = Some(Event::NoSpeechTimeout);
    }

    let next = Detector {
        floor_db: level_db.min(detector.floor_db + params.floor_rise_db),
        phase,
        frames_seen,
        heard_speech,
    };
    (next, event)
}

/// Range of `samples` from the first to the last speech, padded by
/// [`Params::pad_before_frames`] and [`Params::pad_after_frames`]. `None` if nothing in
/// the recording counts as speech.
///
/// `samples` is mono at `sample_rate`.
///
/// The noise floor is the quietest frame of the recording. An unusually quiet frame
/// lowers the threshold, which keeps more audio: errors cost speed, not words.
///
/// # Panics
/// If `sample_rate` is below 34 Hz, where a frame has no samples.
pub fn speech_span(samples: &[f32], sample_rate: u32, params: &Params) -> Option<Range<usize>> {
    let frame_len = frame_len(sample_rate);
    let levels: Vec<f32> = samples.chunks(frame_len).map(level_db).collect();
    let floor_db = levels.iter().copied().fold(f32::INFINITY, f32::min);
    let start_db = start_threshold_db(floor_db, params);

    // Frame ranges of the first and last speech: runs of at least `start_frames` loud
    // frames.
    let mut first: Option<usize> = None;
    let mut last_end = 0;
    let mut loud_run = 0;
    for (i, &level) in levels.iter().enumerate() {
        loud_run = if level >= start_db { loud_run + 1 } else { 0 };
        if loud_run >= params.start_frames as usize {
            first.get_or_insert(i + 1 - loud_run);
            last_end = i + 1;
        }
    }

    let start_frame = first?.saturating_sub(params.pad_before_frames as usize);
    let end_frame = last_end + params.pad_after_frames as usize;
    Some(start_frame * frame_len..(end_frame * frame_len).min(samples.len()))
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{SQRT_2, TAU};

    use super::*;

    const RATE: u32 = 16_000;

    fn amplitude(db: f32) -> f32 {
        10_f32.powf(db / 20.0)
    }

    /// 440 Hz sine whose RMS level is `db`; stands in for speech.
    fn tone(db: f32, seconds: f32, rate: u32) -> Vec<f32> {
        let peak = amplitude(db) * SQRT_2;
        let len = (rate as f32 * seconds) as usize;
        (0..len)
            .map(|i| peak * (TAU * 440.0 * i as f32 / rate as f32).sin())
            .collect()
    }

    /// Uniform white noise whose RMS level is `db`; stands in for the room. A fixed-seed
    /// xorshift keeps the tests deterministic without a dependency.
    fn noise(db: f32, seconds: f32, rate: u32) -> Vec<f32> {
        // A uniform distribution on [-a, a] has RMS a/sqrt(3).
        let peak = amplitude(db) * 3_f32.sqrt();
        let len = (rate as f32 * seconds) as usize;
        let mut state = 0x2545_f491_u32;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                peak * (state as f32 / u32::MAX as f32 * 2.0 - 1.0)
            })
            .collect()
    }

    fn mix(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    /// Runs the live detector over `samples` and returns each event with the time, in
    /// ms, at the end of the frame that produced it.
    fn events(samples: &[f32], rate: u32) -> Vec<(u32, Event)> {
        let params = Params::default();
        let mut detector = Detector::default();
        let mut found = Vec::new();
        for (i, frame) in samples.chunks_exact(frame_len(rate)).enumerate() {
            let (next, event) = step(detector, level_db(frame), &params);
            detector = next;
            if let Some(event) = event {
                found.push((((i + 1) as u32) * FRAME_MS, event));
            }
        }
        found
    }

    fn kinds(events: &[(u32, Event)]) -> Vec<Event> {
        events.iter().map(|&(_, event)| event).collect()
    }

    #[test]
    fn level_db_of_full_scale_sine_is_minus_3_db() {
        let level = level_db(&tone(-3.01, 0.03, RATE));
        let full_scale: Vec<f32> = (0..480).map(|i| (TAU * i as f32 / 48.0).sin()).collect();
        assert!((level_db(&full_scale) + 3.01).abs() < 0.05);
        assert!((level + 3.01).abs() < 0.05, "{level}");
    }

    #[test]
    fn level_db_drops_6_db_when_amplitude_halves() {
        let loud = tone(-10.0, 0.03, RATE);
        let quiet: Vec<f32> = loud.iter().map(|s| s / 2.0).collect();
        let drop = level_db(&loud) - level_db(&quiet);
        assert!((drop - 6.02).abs() < 0.01, "{drop}");
    }

    #[test]
    fn level_db_of_silence_and_of_empty_frame_is_min_db() {
        assert_eq!(level_db(&[0.0; 480]), MIN_DB);
        assert_eq!(level_db(&[]), MIN_DB);
    }

    #[test]
    fn quiet_background_alone_starts_no_speech_and_times_out_once() {
        let found = events(&noise(-60.0, 12.0, RATE), RATE);
        assert_eq!(found, vec![(9_990, Event::NoSpeechTimeout)]);
    }

    #[test]
    fn speech_starts_after_90_ms_and_ends_after_the_hangover() {
        let samples = [
            noise(-60.0, 1.0, RATE),
            tone(-20.0, 2.0, RATE),
            noise(-60.0, 3.0, RATE),
        ]
        .concat();
        let found = events(&samples, RATE);
        assert_eq!(kinds(&found), [Event::SpeechStarted, Event::SpeechEnded]);
        let (started, ended) = (found[0].0, found[1].0);
        assert!(
            (1_060..=1_120).contains(&started),
            "started at {started} ms"
        );
        assert!((4_170..=4_230).contains(&ended), "ended at {ended} ms");
    }

    #[test]
    fn short_click_does_not_start_speech() {
        // 40 ms touches two frames; speech needs three.
        let samples = [
            noise(-60.0, 1.0, RATE),
            tone(-10.0, 0.04, RATE),
            noise(-60.0, 1.0, RATE),
        ]
        .concat();
        assert_eq!(events(&samples, RATE), []);
    }

    #[test]
    fn pause_shorter_than_the_hangover_keeps_one_speech_segment() {
        let samples = [
            noise(-60.0, 0.5, RATE),
            tone(-20.0, 1.0, RATE),
            noise(-60.0, 0.8, RATE),
            tone(-20.0, 1.0, RATE),
            noise(-60.0, 2.0, RATE),
        ]
        .concat();
        let found = events(&samples, RATE);
        assert_eq!(kinds(&found), [Event::SpeechStarted, Event::SpeechEnded]);
    }

    #[test]
    fn pause_longer_than_the_hangover_ends_speech_and_the_next_speech_starts_again() {
        let samples = [
            noise(-60.0, 0.5, RATE),
            tone(-20.0, 1.0, RATE),
            noise(-60.0, 2.0, RATE),
            tone(-20.0, 1.0, RATE),
            noise(-60.0, 2.0, RATE),
        ]
        .concat();
        let found = events(&samples, RATE);
        assert_eq!(
            kinds(&found),
            [
                Event::SpeechStarted,
                Event::SpeechEnded,
                Event::SpeechStarted,
                Event::SpeechEnded
            ]
        );
    }

    #[test]
    fn speech_over_a_loud_background_is_detected() {
        let background = noise(-45.0, 4.0, RATE);
        let speech = [
            vec![0.0; RATE as usize],
            tone(-20.0, 1.0, RATE),
            vec![0.0; 2 * RATE as usize],
        ]
        .concat();
        let found = events(&mix(&background, &speech), RATE);
        assert_eq!(kinds(&found), [Event::SpeechStarted, Event::SpeechEnded]);
    }

    #[test]
    fn steady_loud_background_from_the_first_frame_is_not_speech() {
        assert_eq!(events(&noise(-45.0, 3.0, RATE), RATE), []);
    }

    #[test]
    fn speech_from_the_first_frame_is_detected() {
        let samples = [tone(-20.0, 1.0, RATE), noise(-60.0, 2.0, RATE)].concat();
        let found = events(&samples, RATE);
        assert_eq!(kinds(&found), [Event::SpeechStarted, Event::SpeechEnded]);
        assert_eq!(found[0].0, 90);
    }

    #[test]
    fn sound_below_min_start_is_not_speech_even_in_a_silent_room() {
        // 30 dB above the floor, but quieter than -55 dBFS.
        let samples = [noise(-90.0, 1.0, RATE), tone(-60.0, 2.0, RATE)].concat();
        assert_eq!(events(&samples, RATE), []);
    }

    #[test]
    fn speech_span_covers_the_speech_with_padding() {
        let samples = [
            noise(-60.0, 1.0, RATE),
            tone(-20.0, 1.0, RATE),
            noise(-60.0, 2.0, RATE),
        ]
        .concat();
        let span = speech_span(&samples, RATE, &Params::default()).unwrap();
        let frame = frame_len(RATE);
        // Speech is 16_000..32_000; 8 frames of padding before, 13 after.
        assert!(span.start.abs_diff(16_000 - 8 * frame) <= frame, "{span:?}");
        assert!(span.end.abs_diff(32_000 + 13 * frame) <= frame, "{span:?}");
    }

    #[test]
    fn speech_span_keeps_pauses_between_speech() {
        let samples = [
            noise(-60.0, 1.0, RATE),
            tone(-20.0, 1.0, RATE),
            noise(-60.0, 3.0, RATE),
            tone(-20.0, 1.0, RATE),
            noise(-60.0, 1.0, RATE),
        ]
        .concat();
        let span = speech_span(&samples, RATE, &Params::default()).unwrap();
        assert!(span.start < 16_000 && span.end > 96_000, "{span:?}");
    }

    #[test]
    fn speech_span_keeps_a_recording_that_is_speech_throughout() {
        let samples = tone(-20.0, 1.0, RATE);
        let span = speech_span(&samples, RATE, &Params::default()).unwrap();
        assert_eq!(span, 0..samples.len());
    }

    #[test]
    fn speech_span_is_none_without_speech() {
        let params = Params::default();
        let click = [
            noise(-60.0, 1.0, RATE),
            tone(-10.0, 0.04, RATE),
            noise(-60.0, 1.0, RATE),
        ]
        .concat();
        assert_eq!(speech_span(&noise(-60.0, 2.0, RATE), RATE, &params), None);
        assert_eq!(speech_span(&click, RATE, &params), None);
        assert_eq!(speech_span(&[], RATE, &params), None);
    }

    #[test]
    fn speech_span_finds_the_same_time_at_44_1_khz() {
        let rate = 44_100;
        let samples = [
            noise(-60.0, 1.0, rate),
            tone(-20.0, 1.0, rate),
            noise(-60.0, 2.0, rate),
        ]
        .concat();
        let span = speech_span(&samples, rate, &Params::default()).unwrap();
        let ms = |sample: usize| sample as u32 * 1_000 / rate;
        // Speech is 1000..2000 ms; 240 ms of padding before, 390 ms after.
        assert!(ms(span.start).abs_diff(760) <= FRAME_MS, "{span:?}");
        assert!(ms(span.end).abs_diff(2_390) <= FRAME_MS, "{span:?}");
    }
}
