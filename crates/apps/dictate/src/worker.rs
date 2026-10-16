//! The worker thread: owns the speech model, records while the hotkey is held and
//! transcribes afterwards. Everything slow happens here, never on the main thread,
//! which runs the UI and the hotkey handler.

use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Instant;

use anyhow::Context;
use recording::silence::Params;
use tauri::{AppHandle, Manager};
use transcription::{Device, Engine, Model};

use crate::paste;
use crate::status::{self, Status, TranscriptEvent};

/// Engine loaded at start. Parakeet is the faster one (see FINDINGS.md).
const ENGINE: Engine = Engine::Parakeet;

/// Recordings shorter than this are taken as an accidental tap and dropped. Parakeet
/// rejects very short input anyway.
const MIN_RECORDING_SECS: f64 = 0.3;

/// What the hotkey handler tells the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    KeyDown,
    KeyUp,
}

/// Starts the worker thread and returns the sender that drives it. The thread runs
/// until every sender is dropped; the app keeps one in its state until it exits.
pub fn spawn(app: AppHandle) -> Sender<Command> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || run(&app, &receiver));
    sender
}

fn run(app: &AppHandle, commands: &Receiver<Command>) {
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

    for command in commands {
        // A KeyUp here belongs to a press that was ignored or already handled.
        if command != Command::KeyDown {
            continue;
        }
        match dictate(app, &mut model, commands) {
            Ok(()) => status::publish(app, Status::Idle),
            Err(error) => status::publish(app, Status::error(&error)),
        }
        // Presses made while transcribing were not meant to start a new recording.
        commands.try_iter().for_each(drop);
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

/// Records until the key is released, transcribes, shows the transcript and pastes it
/// into the focused app.
fn dictate(app: &AppHandle, model: &mut Model, commands: &Receiver<Command>) -> anyhow::Result<()> {
    status::publish(app, Status::Recording);
    let mut stopped_at = Instant::now();
    let recording = recording::record(&Params::default(), || {
        wait_for_key_up(commands);
        stopped_at = Instant::now();
    })?;

    let audio_secs = recording.samples.len() as f64 / f64::from(recording::SAMPLE_RATE);
    if audio_secs < MIN_RECORDING_SECS {
        return Ok(());
    }

    status::publish(app, Status::Transcribing);
    let transcript = model.transcribe(&recording.samples, None)?;
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

/// Blocks until the hotkey is released. Also returns when the channel closes (the
/// app is quitting), so the recording ends either way.
fn wait_for_key_up(commands: &Receiver<Command>) {
    for command in commands {
        if command == Command::KeyUp {
            return;
        }
    }
}
