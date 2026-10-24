//! Trimming by loudness: finds the speech in a finished recording from the levels of
//! its 30 ms frames, so the silence before and after it is not transcribed
//! ([`crate::Recording::speech`]). Whether a recording holds speech at all is decided by
//! Silero VAD ([`crate::vad`]); by loudness alone, coughs and typing are speech.
//!
//! The noise floor is the quietest window of [`FLOOR_FRAMES`] frames: the loudest frame
//! of a window is its level, and windows holding digital silence ([`MIN_DB`]) are
//! skipped. Digital silence is a missing signal (the microphone starting, a dropped
//! buffer), not a quiet room; and a quiet stretch shorter than a window, such as the edge
//! of a dropout, is not the room either. Taking either as the floor would lower the
//! threshold until a moderately loud background counts as speech.
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

/// Frames in one window of the noise floor estimate.
pub const FLOOR_FRAMES: usize = 3;

/// Thresholds and durations of the trimming. Durations are in frames of [`FRAME_MS`].
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
    /// Frames above the threshold before speech counts; shorter bursts are clicks.
    pub start_frames: u32,
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
            start_frames: frames(90),
            pad_before_frames: frames(240),
            pad_after_frames: frames(390),
        }
    }
}

/// Whole frames in `ms` milliseconds.
pub const fn frames(ms: u32) -> u32 {
    ms / FRAME_MS
}

/// Samples in one frame at `sample_rate`. Zero below 34 Hz; audio runs at 8 kHz or more.
pub(crate) fn frame_len(sample_rate: u32) -> usize {
    (sample_rate * FRAME_MS / 1_000) as usize
}

/// Root mean square level of `frame` in dB relative to full scale: 0 dB is a square wave
/// at ±1, a full-scale sine is -3 dB. Silence and an empty frame give [`MIN_DB`].
pub(crate) fn level_db(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return MIN_DB;
    }
    let mean_square = frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32;
    // 10·log10 of the power is 20·log10 of the RMS. `max` turns -inf into MIN_DB.
    (10.0 * mean_square.log10()).max(MIN_DB)
}

/// Noise floor of one window of frame levels: its loudest level. `None` if the window
/// holds digital silence.
fn window_floor_db(levels: &[f32]) -> Option<f32> {
    if levels.iter().any(|&level| level <= MIN_DB) {
        return None;
    }
    levels.iter().copied().reduce(f32::max)
}

/// Level a frame must reach to count as speech, given the noise floor.
fn start_threshold_db(floor_db: f32, params: &Params) -> f32 {
    (floor_db.min(params.max_floor_db) + params.start_above_floor_db).max(params.min_start_db)
}

/// Frames from the first to the last speech in `levels` (one per frame, in dBFS),
/// padded by [`Params::pad_before_frames`] and [`Params::pad_after_frames`]. `None` if
/// nothing counts as speech. The end can lie past the last level: the padding may reach
/// into audio after the last whole frame.
///
/// The noise floor is the quietest window of the recording. An unusually quiet window
/// lowers the threshold, which keeps more audio: errors cost speed, not words.
pub(crate) fn speech_span(levels: &[f32], params: &Params) -> Option<Range<usize>> {
    let floor_db = levels
        .windows(FLOOR_FRAMES)
        .filter_map(window_floor_db)
        .fold(f32::INFINITY, f32::min);
    let start_db = start_threshold_db(floor_db, params);

    // The first and last speech: runs of at least `start_frames` loud frames.
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

    let start = first?.saturating_sub(params.pad_before_frames as usize);
    Some(start..last_end + params.pad_after_frames as usize)
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

    /// A -45 dBFS fan, interrupted by digital silence:
    /// 0.5 s at the start (as a microphone may deliver) and dropouts of 20 ms and 0.5 s.
    /// No zero run is aligned to a frame, so frames at its edges are partly silent.
    fn fan_with_digital_silence(rate: u32) -> Vec<f32> {
        let zeros = |seconds: f32| vec![0.0; (rate as f32 * seconds) as usize + 7];
        [
            zeros(0.5),
            noise(-45.0, 3.0, rate),
            zeros(0.02),
            noise(-45.0, 3.0, rate),
            zeros(0.5),
            noise(-45.0, 6.0, rate),
        ]
        .concat()
    }

    /// Level of every whole frame of `samples`, as recording measures them.
    fn levels(samples: &[f32], rate: u32) -> Vec<f32> {
        samples
            .chunks_exact(frame_len(rate))
            .map(level_db)
            .collect()
    }

    /// Start of `frame` in milliseconds.
    fn ms(frame: usize) -> u32 {
        frame as u32 * FRAME_MS
    }

    #[test]
    fn speech_span_covers_the_speech_with_padding() {
        let samples = [
            noise(-60.0, 1.0, RATE),
            tone(-20.0, 1.0, RATE),
            noise(-60.0, 2.0, RATE),
        ]
        .concat();
        let span = speech_span(&levels(&samples, RATE), &Params::default()).unwrap();
        // Speech is 1000..2000 ms; 240 ms of padding before, 390 ms after.
        assert!(ms(span.start).abs_diff(760) <= FRAME_MS, "{span:?}");
        assert!(ms(span.end).abs_diff(2_390) <= FRAME_MS, "{span:?}");
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
        let span = speech_span(&levels(&samples, RATE), &Params::default()).unwrap();
        assert!(ms(span.start) < 1_000 && ms(span.end) > 6_000, "{span:?}");
    }

    #[test]
    fn speech_span_keeps_a_recording_that_is_speech_throughout() {
        let levels = levels(&tone(-20.0, 1.0, RATE), RATE);
        let span = speech_span(&levels, &Params::default()).unwrap();
        assert_eq!(span.start, 0);
        assert!(span.end >= levels.len(), "{span:?}");
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
        assert_eq!(
            speech_span(&levels(&noise(-60.0, 2.0, RATE), RATE), &params),
            None
        );
        assert_eq!(speech_span(&levels(&click, RATE), &params), None);
        assert_eq!(speech_span(&[], &params), None);
    }

    #[test]
    fn speech_span_ignores_digital_silence_and_a_single_quiet_frame() {
        let mut levels = levels(&fan_with_digital_silence(RATE), RATE);
        // One frame at -60 dBFS: quieter than the fan, but not digital silence, so only
        // the window keeps it from becoming the floor.
        levels.push(-60.0);
        assert_eq!(speech_span(&levels, &Params::default()), None);
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
        let span = speech_span(&levels(&samples, rate), &Params::default()).unwrap();
        assert!(ms(span.start).abs_diff(760) <= FRAME_MS, "{span:?}");
        assert!(ms(span.end).abs_diff(2_390) <= FRAME_MS, "{span:?}");
    }

    #[test]
    fn speech_span_starts_at_the_first_speech() {
        let params = Params::default();
        let quiet = |frames| vec![-60.0; frames];
        let loud = |db, frames| vec![db; frames];
        let pad = params.pad_before_frames as usize;
        // Each case with the frame where its speech starts.
        let cases: [(Vec<f32>, usize); 3] = [
            // Speech at -32 dBFS from the first frame: the threshold comes from the
            // pause after it (-48 dBFS), so the first burst is kept.
            (
                [loud(-32.0, 30), quiet(20), loud(-32.0, 30), quiet(50)].concat(),
                0,
            ),
            // A burst of exactly `start_frames` loud frames.
            ([quiet(40), loud(-20.0, 3), quiet(40)].concat(), 40),
            // Speech over a loud background with digital silence in it.
            (
                {
                    let mut levels = levels(&fan_with_digital_silence(RATE), RATE);
                    levels[200..240].fill(-20.0);
                    levels
                },
                200,
            ),
        ];
        for (levels, speech_start) in cases {
            let span = speech_span(&levels, &params).expect("a speech span");
            assert_eq!(span.start, speech_start.saturating_sub(pad), "{levels:?}");
        }
    }
}
