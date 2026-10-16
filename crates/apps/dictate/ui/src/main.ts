// The window only displays what the Rust side reports; recording, transcription and
// pasting all happen in Rust.

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

/** Mirrors `Status` in `src/status.rs`. */
type Status =
  | { state: "loading" | "idle" | "recording" | "transcribing" }
  | { state: "error"; message: string };

/** Mirrors `TranscriptEvent` in `src/status.rs`. */
interface Transcript {
  text: string;
  audioSecs: number;
  stopToTextSecs: number;
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

function showStatus(status: Status) {
  statusLine.textContent =
    status.state === "error" ? `Error: ${status.message}` : LABELS[status.state];
  statusLine.className = status.state;
}

function showTranscript(transcript: Transcript) {
  transcriptText.textContent = transcript.text || "(no speech recognized)";
  timings.textContent =
    `${transcript.audioSecs.toFixed(1)} s of audio, ` +
    `text ready ${transcript.stopToTextSecs.toFixed(2)} s after release`;
}

// Listen first, then fetch the current status, so no update falls in between.
await listen<Status>("status", (event) => showStatus(event.payload));
await listen<Transcript>("transcript", (event) => showTranscript(event.payload));
const current = await invoke<Status | null>("current_status");
if (current) showStatus(current);
