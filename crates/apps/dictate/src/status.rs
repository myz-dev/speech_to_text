//! What the app is doing, reported to the window and the menu-bar icon.
//!
//! The window may load after the first status was emitted, so the latest status is
//! also kept in the app state, where the window reads it on start
//! ([`current_status`]).

use std::sync::{Mutex, PoisonError};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::tray;

/// Serialized as `{"state": "recording"}` or
/// `{"state": "error", "message": "…"}`, mirrored by `Status` in `ui/src/main.ts`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum Status {
    Loading,
    Idle,
    Recording,
    Transcribing,
    CleaningUp,
    /// Idle after a dictation that worked but had a problem worth showing.
    Warning {
        message: String,
    },
    Error {
        message: String,
    },
}

impl Status {
    /// An error status with the whole cause chain, e.g. "no usable Parakeet model in
    /// …: cannot read parakeet vocabulary …: No such file or directory".
    pub fn error(error: &anyhow::Error) -> Self {
        Status::Error {
            message: format!("{error:#}"),
        }
    }

    /// A warning status with the whole cause chain, like [`Status::error`].
    pub fn warning(error: &anyhow::Error) -> Self {
        Status::Warning {
            message: format!("{error:#}"),
        }
    }
}

/// A finished dictation, emitted as the `transcript` event.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptEvent {
    /// The transcript as the speech model produced it.
    pub text: String,
    /// The text after the cleanup, if it ran and succeeded.
    pub cleaned: Option<String>,
    pub audio_secs: f64,
    /// From the end of the recording to the text being ready, including the cleanup.
    pub stop_to_text_secs: f64,
    /// Whether the text is pasted; otherwise it is only on the clipboard.
    pub pasted: bool,
}

/// The latest published status, kept in the app state.
#[derive(Default)]
pub struct Latest(Mutex<Option<Status>>);

/// Records `status` as the latest, emits it to the window as the `status` event and
/// shows it on the menu-bar icon.
///
/// Stored before it is emitted: a window that reads [`current_status`] after an
/// event then never gets a value older than that event.
pub fn publish(app: &AppHandle, status: Status) {
    // The lock only guards a value replaced in one step, so a panic while holding
    // it cannot leave it inconsistent; ignore poisoning.
    let latest = app.state::<Latest>();
    *latest.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(status.clone());
    if let Err(error) = app.emit("status", &status) {
        eprintln!("cannot emit the status: {error}");
    }
    tray::show_status(app, &status);
}

pub fn publish_transcript(app: &AppHandle, transcript: TranscriptEvent) {
    if let Err(error) = app.emit("transcript", &transcript) {
        eprintln!("cannot emit the transcript: {error}");
    }
}

/// The latest status, for a window that just loaded.
#[tauri::command]
pub fn current_status(latest: tauri::State<Latest>) -> Option<Status> {
    latest
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}
