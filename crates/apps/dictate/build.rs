fn main() {
    // Reads tauri.conf.json and capabilities/, and prepares the app context that
    // `tauri::generate_context!` embeds into the binary.
    tauri_build::build()
}
