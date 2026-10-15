//! Dictate: a macOS menu-bar app that records speech, transcribes it locally and
//! pastes the text into the focused app.

mod shortcut;
mod status;
mod tray;
mod worker;

use tauri::{ActivationPolicy, Manager, WindowEvent};

use crate::status::Status;

fn main() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(shortcut::on_shortcut)
                .build(),
        )
        .manage(status::Latest::default())
        .invoke_handler(tauri::generate_handler![status::current_status])
        .setup(|app| {
            // A menu-bar app: no Dock icon and no menu bar of its own.
            app.set_activation_policy(ActivationPolicy::Accessory);
            tray::create(app.handle())?;
            match shortcut::register(app.handle()) {
                Ok(()) => {
                    app.manage(worker::spawn(app.handle().clone()));
                }
                Err(error) => {
                    let error = anyhow::Error::new(error).context(format!(
                        "cannot register the hotkey {}",
                        shortcut::HOTKEY_LABEL
                    ));
                    status::publish(app.handle(), Status::error(&error));
                }
            }
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
