//! The global hotkey. macOS reports both the press and the release (Carbon hotkeys,
//! without key repeat), which hold-to-talk needs. Registering it needs no permission.
//!
//! macOS lets several apps register the same combination without an error, and then
//! delivers it to all of them, so a clash with another app (e.g. Raycast on ⌥Space)
//! cannot be detected here. The hotkey is chosen to be rarely taken.

use std::sync::mpsc::Sender;

use tauri::{AppHandle, Manager};
use tauri_plugin_global_shortcut::{
    Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutEvent, ShortcutState,
};

use crate::gesture::Key;

/// The hotkey as shown to the user; the window's hint repeats it.
pub const HOTKEY_LABEL: &str = "⌃⌥Space";

/// Registers ⌃⌥Space. Fails only if this app registered it already or macOS reserves
/// the combination.
pub fn register(app: &AppHandle) -> Result<(), tauri_plugin_global_shortcut::Error> {
    let hotkey = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::Space);
    app.global_shortcut().register(hotkey)
}

/// Handles every registered shortcut. It runs on the main thread, so it only passes
/// the key state on to the worker.
pub fn on_shortcut(app: &AppHandle, _shortcut: &Shortcut, event: ShortcutEvent) {
    let key = match event.state {
        ShortcutState::Pressed => Key::Down,
        ShortcutState::Released => Key::Up,
    };
    // Sending fails only if the worker ended, which it does when no model could be
    // loaded; that error is already shown.
    let _ = app.state::<Sender<Key>>().send(key);
}
