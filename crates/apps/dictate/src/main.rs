//! Dictate: a macOS menu-bar app that records speech, transcribes it locally and
//! pastes the text into the focused app.

mod gesture;
mod paste;
mod settings;
mod shortcut;
mod status;
mod tray;
mod worker;

use tauri::{ActivationPolicy, Manager, RunEvent, WindowEvent};

use crate::status::Status;

/// Restarts Dictate, e.g. so a newly granted Accessibility permission applies.
#[tauri::command]
fn restart(app: tauri::AppHandle) {
    app.restart();
}

fn main() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(shortcut::on_shortcut)
                .build(),
        )
        .manage(status::Latest::default())
        .manage(settings::Current::default())
        .invoke_handler(tauri::generate_handler![
            status::current_status,
            settings::settings,
            settings::set_settings,
            paste::can_paste,
            paste::request_paste_permission,
            restart,
        ])
        .setup(|app| {
            // A menu-bar app: no Dock icon and no menu bar of its own.
            app.set_activation_policy(ActivationPolicy::Accessory);
            tray::create(app.handle())?;
            // The window starts hidden and is shown here: a menu-bar app is not brought
            // to the front when it starts, so a window shown by the config alone ends up
            // behind the app that launched it.
            tray::show_main_window(app.handle());
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
        .build(tauri::generate_context!())
        .expect("the Tauri app failed to build")
        .run(|app, event| {
            // Launching the app again while it runs (Finder, Spotlight, Raycast) only
            // sends it this event; show the window, as a Mac app does.
            if let RunEvent::Reopen { .. } = event {
                tray::show_main_window(app);
            }
        });
}
