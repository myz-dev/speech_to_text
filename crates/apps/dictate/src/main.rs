//! Dictate: a macOS menu-bar app that records speech, transcribes it locally and
//! pastes the text into the focused app.

mod tray;

use tauri::{ActivationPolicy, WindowEvent};

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // A menu-bar app: no Dock icon and no menu bar of its own.
            app.set_activation_policy(ActivationPolicy::Accessory);
            tray::create(app.handle())?;
            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the window only hides it; the app keeps running in the menu bar
            // and quits from its menu.
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                if let Err(error) = window.hide() {
                    eprintln!("cannot hide the window: {error}");
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("the Tauri event loop failed to start");
}
