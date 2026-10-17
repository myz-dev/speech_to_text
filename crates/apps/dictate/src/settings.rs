//! User facing settings.
//! The settings are kept in the app state; the window only shows them, and every change
//! is emitted to it as the `settings` event.
//! The worker reads them when a dictation starts, so a change applies from the next one.
//! They are kept in memory: restarting the app restores the defaults.

use std::sync::{Mutex, PoisonError};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

/// Mirrored by `Settings` in `ui/src/main.ts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    /// Paste the transcript into the focused app. Otherwise it is only put on the
    /// clipboard, which needs no Accessibility permission.
    pub paste: bool,
    /// Clean up the transcript with a local LLM served by Ollama.
    pub cleanup: bool,
    /// The language dictated in. Only the cleanup needs it: Parakeet transcribes any
    /// language without being told, but does not report which one it heard.
    pub language: Language,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            paste: false,
            cleanup: false,
            language: Language::De,
        }
    }
}

/// A language the cleanup supports, serialized as its ISO 639-1 code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    De,
    En,
}

impl From<Language> for cleanup::Language {
    fn from(language: Language) -> Self {
        match language {
            Language::De => cleanup::Language::German,
            Language::En => cleanup::Language::English,
        }
    }
}

/// The current settings, kept in the app state.
#[derive(Default)]
pub struct Current(Mutex<Settings>);

impl Current {
    fn lock(&self) -> std::sync::MutexGuard<'_, Settings> {
        // The lock only guards a value replaced in one step, so a panic while holding it
        // cannot leave it inconsistent; ignore poisoning.
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The current settings, for the worker.
pub fn current(app: &AppHandle) -> Settings {
    *app.state::<Current>().lock()
}

/// The current settings, for the window.
#[tauri::command]
pub fn settings(current: tauri::State<Current>) -> Settings {
    *current.lock()
}

/// Replaces the settings and emits them as the `settings` event.
///
/// Stored before it is emitted, like [`crate::status::publish`]: a window that reads
/// [`settings`] after an event never gets a value older than that event.
#[tauri::command]
pub fn set_settings(settings: Settings, app: AppHandle, current: tauri::State<Current>) {
    *current.lock() = settings;
    if let Err(error) = app.emit("settings", settings) {
        eprintln!("cannot emit the settings: {error}");
    }
}
