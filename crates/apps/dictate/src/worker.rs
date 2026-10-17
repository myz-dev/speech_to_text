//! The worker thread: owns the speech model, records while the hotkey controls it (see
//! [`gesture`]) and transcribes afterwards. Everything slow happens here, never on the
//! main thread, which runs the UI and the hotkey handler.

use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Instant;

use anyhow::Context;
use recording::Flow;
use recording::silence::Params;
use tauri::{AppHandle, Manager};
use transcription::{Device, Engine, Model};

use crate::gesture::{self, Input, Key, Mode};
use crate::paste;
use crate::status::{self, Status, TranscriptEvent};

/// Engine loaded at start. Parakeet is the faster one (see FINDINGS.md).
const ENGINE: Engine = Engine::Parakeet;

/// Speech shorter than this is not transcribed; Parakeet rejects very short input.
const MIN_SPEECH_SECS: f64 = 0.3;

/// Starts the worker thread and returns the sender that drives it with hotkey changes.
/// The thread runs until every sender is dropped; the app keeps one in its state until
/// it exits.
pub fn spawn(app: AppHandle) -> Sender<Key> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || run(&app, &receiver));
    sender
}

fn run(app: &AppHandle, keys: &Receiver<Key>) {
    status::publish(app, Status::Loading);
    let mut model = match load_model(app) {
        Ok(model) => model,
        Err(error) => {
            // Without a model there is nothing to do; keep the error visible.
            status::publish(app, Status::error(&error));
            return;
        }
    };
    status::publish(app, Status::Idle);

    for key in keys {
        // A release here belongs to a press that was ignored or already handled.
        if key != Key::Down {
            continue;
        }
        let pressed = Mode::Pressed { at: Instant::now() };
        match dictate(app, &mut model, keys, pressed) {
            Ok(()) => status::publish(app, Status::Idle),
            Err(error) => status::publish(app, Status::error(&error)),
        }
        // Presses made while transcribing were not meant to start a new recording.
        keys.try_iter().for_each(drop);
    }
}

/// Loads [`ENGINE`]'s model from the app's models directory.
fn load_model(app: &AppHandle) -> anyhow::Result<Model> {
    let models = app
        .path()
        .app_data_dir()
        .context("cannot find the app data directory")?
        .join("models");
    Model::load(ENGINE, &models, Device::Gpu).with_context(|| missing_model_hint(&models))
}

fn missing_model_hint(models: &Path) -> String {
    format!(
        "no usable {ENGINE:?} model in {}; see SETUP.md",
        models.display()
    )
}

/// Records until the hotkey or the speech detector ends the recording (see [`gesture`]),
/// transcribes the speech, shows the transcript and pastes it into the focused app.
/// A recording without speech ends here.
fn dictate(
    app: &AppHandle,
    model: &mut Model,
    keys: &Receiver<Key>,
    mut mode: Mode,
) -> anyhow::Result<()> {
    status::publish(app, Status::Recording);
    let mut stopped_at = Instant::now();
    let recording = recording::record_streaming(&Params::default(), |frames| {
        // Runs after every microphone buffer (~10 ms), or every 20 ms without one, but
        // only once the microphone has started. Keys are timed here, when read: a
        // release during a slow microphone start (hundreds of milliseconds for some
        // Bluetooth headsets) is timed late and turns a tap into a hold.
        let keys = keys.try_iter().map(Input::Key);
        let events = frames.iter().filter_map(|frame| frame.event);
        for input in keys.chain(events.map(Input::Speech)) {
            let flow;
            (mode, flow) = gesture::on_input(mode, input, Instant::now());
            if flow == Flow::Stop {
                stopped_at = Instant::now();
                return Flow::Stop;
            }
        }
        Flow::Continue
    })?;

    let Some(speech) = recording.speech else {
        return Ok(());
    };
    let samples = &recording.samples[speech];
    let audio_secs = samples.len() as f64 / f64::from(recording::SAMPLE_RATE);
    if audio_secs < MIN_SPEECH_SECS {
        return Ok(());
    }

    status::publish(app, Status::Transcribing);
    let transcript = model.transcribe(samples, None)?;
    // Shown before pasting, so the text is visible even if pasting fails.
    status::publish_transcript(
        app,
        TranscriptEvent {
            text: transcript.text.clone(),
            audio_secs,
            stop_to_text_secs: stopped_at.elapsed().as_secs_f64(),
        },
    );
    if !transcript.text.is_empty() {
        paste::paste(&transcript.text)?;
    }
    Ok(())
}
