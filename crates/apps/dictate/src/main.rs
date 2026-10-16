//! Dictate: a macOS menu-bar app that records speech, transcribes it locally and
//! pastes the text into the focused app.

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("the Tauri event loop failed to start");
}
