//! What the user can change in the window. The worker reads the settings when a
//! dictation starts, so a change applies from the next one. They are kept in memory:
//! restarting the app restores the defaults.

use std::sync::{Mutex, PoisonError};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

/// Mirrored by `Settings` in `ui/src/main.ts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    /// Paste the transcript into the focused app. Otherwise it is only put on the
    /// clipboard, which needs no Accessibility permission.
    pub paste: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { paste: true }
    }
}

/// The current settings, kept in the app state.
#[derive(Default)]
pub struct Current(Mutex<Settings>);

/// The current settings, for the worker.
pub fn current(app: &AppHandle) -> Settings {
    *lock(&app.state::<Current>())
}

/// The current settings, for the window.
#[tauri::command]
pub fn settings(current: tauri::State<Current>) -> Settings {
    *lock(&current)
}

/// Replaces the settings with the window's.
#[tauri::command]
pub fn set_settings(settings: Settings, current: tauri::State<Current>) {
    *lock(&current) = settings;
}

fn lock(current: &Current) -> std::sync::MutexGuard<'_, Settings> {
    // The lock only guards a value replaced in one step, so a panic while holding it
    // cannot leave it inconsistent; ignore poisoning.
    current.0.lock().unwrap_or_else(PoisonError::into_inner)
}
