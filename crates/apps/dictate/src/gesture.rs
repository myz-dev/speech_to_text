//! How the hotkey controls a recording. A recording starts when the key goes down.
//! Holding the key records until it is released (push-to-talk). A short tap instead
//! keeps the recording going until the next press, or until the speech detector
//! reports that speech ended or never started.

use std::time::{Duration, Instant};

use recording::Flow;
use recording::vad::Event;

/// Presses released sooner than this are taps; longer ones are holds.
pub const TAP_MAX: Duration = Duration::from_millis(300);

/// A hotkey state change, as macOS reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Down,
    Up,
}

/// Something that can end a running recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    Key(Key),
    Speech(Event),
}

/// How the running recording is controlled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The key went down at `at` and is still held; tap or hold is not decided yet.
    Pressed { at: Instant },
    /// Started by a tap: runs until the next press, the end of speech, or no speech.
    Toggled,
}

/// Feeds the next input, received at `now`, to a running recording.
pub fn on_input(mode: Mode, input: Input, now: Instant) -> (Mode, Flow) {
    match (mode, input) {
        (Mode::Pressed { at }, Input::Key(Key::Up)) => {
            if now.duration_since(at) < TAP_MAX {
                (Mode::Toggled, Flow::Continue)
            } else {
                (mode, Flow::Stop)
            }
        }
        (Mode::Toggled, Input::Key(Key::Down)) => (mode, Flow::Stop),
        (Mode::Toggled, Input::Speech(Event::SpeechEnded | Event::NoSpeechTimeout)) => {
            (mode, Flow::Stop)
        }
        // While the key is held, the speech detector does not end the recording: a
        // pause must not cut off push-to-talk.
        _ => (mode, Flow::Continue),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY_UP: Input = Input::Key(Key::Up);
    const KEY_DOWN: Input = Input::Key(Key::Down);

    /// A recording whose key went down at `start`.
    fn pressed(start: Instant) -> Mode {
        Mode::Pressed { at: start }
    }

    fn ms(millis: u64) -> Duration {
        Duration::from_millis(millis)
    }

    #[test]
    fn short_press_toggles_and_keeps_recording() {
        let start = Instant::now();
        let result = on_input(pressed(start), KEY_UP, start + ms(120));
        assert_eq!(result, (Mode::Toggled, Flow::Continue));
    }

    #[test]
    fn release_after_holding_stops() {
        let start = Instant::now();
        let (_, flow) = on_input(pressed(start), KEY_UP, start + ms(2_000));
        assert_eq!(flow, Flow::Stop);
    }

    #[test]
    fn release_at_exactly_the_tap_limit_is_a_hold() {
        let start = Instant::now();
        let (_, flow) = on_input(pressed(start), KEY_UP, start + TAP_MAX);
        assert_eq!(flow, Flow::Stop);
    }

    #[test]
    fn holding_ignores_the_speech_detector() {
        let start = Instant::now();
        for event in [
            Event::SpeechStarted,
            Event::SpeechEnded,
            Event::NoSpeechTimeout,
        ] {
            let result = on_input(pressed(start), Input::Speech(event), start + ms(5_000));
            assert_eq!(result, (pressed(start), Flow::Continue), "{event:?}");
        }
    }

    #[test]
    fn second_press_stops_a_toggled_recording() {
        let (_, flow) = on_input(Mode::Toggled, KEY_DOWN, Instant::now());
        assert_eq!(flow, Flow::Stop);
    }

    #[test]
    fn end_of_speech_or_no_speech_stops_a_toggled_recording() {
        for event in [Event::SpeechEnded, Event::NoSpeechTimeout] {
            let (_, flow) = on_input(Mode::Toggled, Input::Speech(event), Instant::now());
            assert_eq!(flow, Flow::Stop, "{event:?}");
        }
    }

    #[test]
    fn toggled_recording_continues_through_speech_start_and_key_up() {
        for input in [Input::Speech(Event::SpeechStarted), KEY_UP] {
            let result = on_input(Mode::Toggled, input, Instant::now());
            assert_eq!(result, (Mode::Toggled, Flow::Continue), "{input:?}");
        }
    }

    #[test]
    fn tap_then_tap_stops() {
        let start = Instant::now();
        let inputs = [(KEY_UP, start + ms(100)), (KEY_DOWN, start + ms(900))];
        let mut mode = pressed(start);
        let mut flows = Vec::new();
        for (input, now) in inputs {
            let flow;
            (mode, flow) = on_input(mode, input, now);
            flows.push(flow);
        }
        assert_eq!(flows, [Flow::Continue, Flow::Stop]);
    }
}
