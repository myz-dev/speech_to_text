//! Pastes text into the focused app: puts it on the clipboard and posts ⌘V, as if the
//! user pressed it. The user's clipboard text is restored afterwards.
//!
//! Posting keystrokes to other apps needs the Accessibility permission (System
//! Settings → Privacy & Security → Accessibility). In `cargo tauri dev` macOS asks
//! for it on behalf of the terminal that started the app.

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

/// Pastes `text` into the focused app.
///
/// Without the Accessibility permission, macOS is asked to show its permission
/// prompt, `text` stays on the clipboard for a manual ⌘V, and an error says so.
pub fn paste(text: &str) -> anyhow::Result<()> {
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
        // Shows the system prompt once and lists the app in the Accessibility
        // settings; the permission applies from the next paste on.
        CGRequestPostEventAccess();
        bail!(
            "pasting needs the Accessibility permission (System Settings → Privacy & \
             Security → Accessibility); the text is on the clipboard, paste it with ⌘V"
        );
    }

    press_command_v()?;
    thread::sleep(PASTE_SETTLE);
    if let Some(previous) = previous {
        clipboard
            .set_text(previous)
            .context("cannot restore the previous clipboard text")?;
    }
    Ok(())
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
