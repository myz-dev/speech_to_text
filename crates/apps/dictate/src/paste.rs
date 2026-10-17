//! Pastes text into the focused app: puts it on the clipboard and posts ⌘V, as if the
//! user pressed it. The user's clipboard text is restored afterwards
//! ([`restore_clipboard`]).
//!
//! Posting keystrokes to other apps needs the Accessibility permission (System
//! Settings → Privacy & Security → Accessibility). In `cargo tauri dev` macOS asks
//! for it on behalf of the terminal that started the app.

use std::process::Command;
use std::thread;
use std::time::Duration;

use anyhow::{Context, bail};
use arboard::{Clipboard, SetExtApple};
use objc2_core_graphics::{
    CGEvent, CGEventFlags, CGEventSource, CGEventSourceStateID, CGEventTapLocation,
    CGPreflightPostEventAccess, CGRequestPostEventAccess,
};

/// Virtual key code of the V key (`kVK_ANSI_V`).
/// The code is the key's position on the keyboard so it depends on the keyboard layout.
/// We could look up the code on runtime, but this works for QWERTY, QWERTZ and AZERTY,
/// which is enough for now.
const KEY_V: u16 = 9;

/// How long the target app gets to read the clipboard after ⌘V before the previous
/// text is put back. Apps read it asynchronously; restoring too early pastes the old
/// text instead.
const PASTE_SETTLE: Duration = Duration::from_millis(250);

/// Pastes `text` into the focused app and returns the clipboard text it replaced, for
/// [`restore_clipboard`].
///
/// Without the Accessibility permission, macOS is asked to show its permission
/// prompt, `text` stays on the clipboard for a manual ⌘V, and an error says so.
pub fn paste(text: &str) -> anyhow::Result<Option<String>> {
    let mut clipboard = Clipboard::new().context("cannot open the clipboard")?;
    // Only text can be restored. If the clipboard held an image or files, they are
    // replaced by the dictated text.
    let previous = clipboard.get_text().ok();
    clipboard
        .set()
        .exclude_from_history()
        .text(text)
        .context("cannot put the text on the clipboard")?;

    if !CGPreflightPostEventAccess() {
        // Lists the app in the Accessibility settings and shows the system prompt,
        // but only the first time: after a denial, macOS stays silent. Opening the
        // settings page covers that case; the permission applies from the next paste on.
        CGRequestPostEventAccess();
        open_accessibility_settings();
        bail!(
            "pasting needs the Accessibility permission: switch it on in the opened \
             System Settings page (Privacy & Security → Accessibility); the text is on \
             the clipboard, paste it with ⌘V"
        );
    }

    press_command_v()?;
    Ok(previous)
}

/// Puts `text` on the clipboard for the user to paste with ⌘V. It stays in clipboard
/// managers' history, unlike the text [`paste`] puts there briefly.
pub fn copy(text: &str) -> anyhow::Result<()> {
    Clipboard::new()
        .and_then(|mut clipboard| clipboard.set_text(text))
        .context("cannot put the text on the clipboard")
}

/// Puts `previous` back on the clipboard once the target app has read the pasted text.
/// Failing here does not undo the paste, so callers report it as a warning.
pub fn restore_clipboard(previous: Option<String>) -> anyhow::Result<()> {
    let Some(previous) = previous else {
        return Ok(());
    };
    thread::sleep(PASTE_SETTLE);
    Clipboard::new()
        .and_then(|mut clipboard| clipboard.set_text(previous))
        .context("the text was pasted, but the previous clipboard text could not be restored")
}

/// Opens System Settings at Privacy & Security → Accessibility. Failing to open it is
/// only logged: the error message names the page anyway.
fn open_accessibility_settings() {
    let page = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility";
    if let Err(error) = Command::new("open").arg(page).spawn() {
        eprintln!("cannot open the Accessibility settings: {error}");
    }
}

/// Posts ⌘V key-down and key-up events, as the keyboard would.
fn press_command_v() -> anyhow::Result<()> {
    // A private source keeps the event independent of the modifier keys the user is
    // still holding from the hotkey (⌃⌥); the flags below are the only ones it has.
    let source = CGEventSource::new(CGEventSourceStateID::Private)
        .context("cannot create a keyboard event source")?;
    for key_down in [true, false] {
        let event = CGEvent::new_keyboard_event(Some(&source), KEY_V, key_down)
            .context("cannot create the ⌘V keyboard event")?;
        // Apple docs about the MaskCommand flag:
        // Indicates that the Command key is down for a keyboard, mouse, or flag-changed event.
        CGEvent::set_flags(Some(&event), CGEventFlags::MaskCommand);
        CGEvent::post(CGEventTapLocation::HIDEventTap, Some(&event));
    }
    Ok(())
}
