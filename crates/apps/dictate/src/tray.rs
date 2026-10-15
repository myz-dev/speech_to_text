//! The menu-bar icon and its menu, the app's only permanent UI.

use tauri::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager};

/// Menu item IDs, matched in [`on_menu_event`].
const SHOW: &str = "show";
const QUIT: &str = "quit";

/// Adds the icon to the menu bar. Tauri keeps it alive until the app exits.
pub fn create(app: &AppHandle) -> tauri::Result<()> {
    let menu = Menu::with_items(
        app,
        &[
            &MenuItem::with_id(app, SHOW, "Show Dictate", true, None::<&str>)?,
            &PredefinedMenuItem::separator(app)?,
            &MenuItem::with_id(app, QUIT, "Quit Dictate", true, None::<&str>)?,
        ],
    )?;
    TrayIconBuilder::new()
        // A template image is drawn by macOS in the menu bar's color, so the black
        // glyph stays visible on light and dark menu bars.
        .icon(tauri::include_image!("icons/tray.png"))
        .icon_as_template(true)
        .tooltip("Dictate")
        .menu(&menu)
        .on_menu_event(on_menu_event)
        .build(app)?;
    Ok(())
}

fn on_menu_event(app: &AppHandle, event: MenuEvent) {
    match event.id().as_ref() {
        SHOW => show_main_window(app),
        QUIT => app.exit(0),
        _ => {}
    }
}

fn show_main_window(app: &AppHandle) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    // Showing a window does not activate a menu-bar app, so focus it explicitly.
    if let Err(error) = window.show().and_then(|()| window.set_focus()) {
        eprintln!("cannot show the window: {error}");
    }
}
