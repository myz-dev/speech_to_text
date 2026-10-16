import { defineConfig } from "vite";

// Tauri loads the dev server from the fixed devUrl in tauri.conf.json, so the port
// must not move to the next free one.
export default defineConfig({
  clearScreen: false,
  server: { port: 5173, strictPort: true },
});
