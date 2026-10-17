// The window only displays what the Rust side reports; recording, transcription and
// pasting all happen in Rust.

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

/** Mirrors `Status` in `src/status.rs`. */
type Status =
  | { state: "loading" | "idle" | "recording" | "transcribing" }
  | { state: "warning" | "error"; message: string };

/** Mirrors `TranscriptEvent` in `src/status.rs`. */
interface Transcript {
  text: string;
  audioSecs: number;
  stopToTextSecs: number;
  pasted: boolean;
}

/** Mirrors `Settings` in `src/settings.rs`. */
interface Settings {
  paste: boolean;
}

const LABELS = {
  loading: "Loading the speech model…",
  idle: "Ready",
  recording: "Recording…",
  transcribing: "Transcribing…",
} as const;

function element(id: string): HTMLElement {
  const found = document.getElementById(id);
  if (!found) throw new Error(`#${id} is missing from index.html`);
  return found;
}

const statusLine = element("status");
const transcriptText = element("transcript");
const timings = element("timings");
const pasteSetting = element("paste") as HTMLInputElement;

function showStatus(status: Status) {
  statusLine.textContent =
    status.state === "error"
      ? `Error: ${status.message}`
      : status.state === "warning"
        ? `Warning: ${status.message}`
        : LABELS[status.state];
  statusLine.className = status.state;
}

function showTranscript(transcript: Transcript) {
  transcriptText.textContent = transcript.text || "(no speech recognized)";
  timings.textContent =
    `${transcript.audioSecs.toFixed(1)} s of audio, ` +
    `text ready ${transcript.stopToTextSecs.toFixed(2)} s after release` +
    (transcript.pasted ? "" : ", on the clipboard: paste it with ⌘V");
}

function showSettings(settings: Settings) {
  pasteSetting.checked = settings.paste;
}

// Rust keeps the settings: the window sends all of them on every change and shows
// what comes back in the `settings` event. If sending fails, it shows Rust's again.
function saveSettings() {
  const settings: Settings = { paste: pasteSetting.checked };
  invoke("set_settings", { settings }).catch(async (error) => {
    console.error("cannot save the settings", error);
    showSettings(await invoke<Settings>("settings"));
  });
}

// Listen first, then fetch the current status and settings, so no update falls in
// between.
await listen<Status>("status", (event) => showStatus(event.payload));
await listen<Transcript>("transcript", (event) => showTranscript(event.payload));
await listen<Settings>("settings", (event) => showSettings(event.payload));
const current = await invoke<Status | null>("current_status");
if (current) showStatus(current);
showSettings(await invoke<Settings>("settings"));
pasteSetting.addEventListener("change", saveSettings);
