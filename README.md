# Dictate

"Dictate" is a desktop (menu-bar) application that records speech and transcribes it to text.
It is an educational project with the goal to learn how a local speech-to-text 
pipeline functions and which AI models run well on Apple silicon. 

To install it, follow [SETUP.md](SETUP.md). It was tested by installing the app on a
second Mac following only that document.

## Feature set

Audio is recorded and transcribed by a local speech-to-text model, and optionally
cleaned up by a local LLM. Everything runs on-device; no cloud services are involved.

Record by holding `⌃⌥Space` and releasing it when you are done. Or tap `⌃⌥Space` to
start recording, then either tap again or simply stop speaking: the recording ends
about a second later by itself.

The resulting text is saved in the clipboard, so you can paste it into any text field. 
Or opt in to automatic pasting: the text is then pasted to where your cursor currently sits.

## Architecture

The Rust side registers the hotkey, records, transcribes and pastes.

After exploring alternatives in the `stt` [prototype](crates/apps/prototype), we have
settled on:

- **Parakeet** (`parakeet-tdt-0.6b-v3`) for transcription. It was chosen over Whisper
  because it is about 25× faster on dictation-length clips, returns empty text for
  silence and noise, writes punctuated text, and needs no language hint
  ([FINDINGS M2](FINDINGS.md#m2-parakeet-end-to-end-vs-whisper)).
  Whisper remains in the `transcription` crate and the prototype.
- **Silero VAD** (a small speech classifier, MIT) to decide whether and when there is
  speech: unlike a loudness threshold, it does not take coughs or typing for speech
  ([FINDINGS](FINDINGS.md#app-silero-vad-against-coughs-and-typing)). Loudness is
  still used to trim the silence before and after the speech.
- **`ministral-3:8b`** via Ollama for the optional cleanup.

## Processing pipeline

A recording starts when the hotkey goes down. It ends when the key is released (hold),
or, after a tap, on a second tap, when speech has ended, or when no speech was heard
for 10 s.

```
hotkey (main thread) ──Key──▶ worker thread (owns the model)
                                │ record_streaming(on_frames)
                                │   ├─ downmix each buffer, measure 30 ms frames → level (dBFS)
                                │   ├─ vad::Live: resample → 16 kHz, Silero per 32 ms chunk
                                │   │    → vad::decide → SpeechEnded ⇒ stop (after a tap)
                                │   └─ keys ⇒ stop (release / second tap)
                                │ no speech heard ⇒ done, nothing pasted
                                │ resample once → 16 kHz
                                │ silence::speech_span on the measured levels → trim
                                │ transcribe (Parakeet) → [cleanup] → paste or copy
                                └─ app.emit: status, transcript
UI (vanilla TS): status, last raw/cleaned text, settings and setup instructions
```

### Audio processing

**Input:** The microphone is recorded at its native rate (48 kHz on a MacBook Pro M4).
The audio callback sends each buffer (about 10 ms) through a channel to the worker,
which waits for one at most 20 ms. Each buffer is downmixed to mono and appended, and
every complete 30 ms frame is measured as its RMS level in dBFS; a frame may span two
buffers. For the speech detector, each buffer is also resampled to 16 kHz as it
arrives. Once the recording ends, the whole recording is resampled again to the 16 kHz
Parakeet expects (about 20 ms of work).

**Live detection:** Silero VAD scores every 32 ms chunk of 16 kHz audio with the
probability that it holds speech (0.09 ms per chunk on the CPU). Speech starts when 4
chunks in a row (128 ms) score above 0.9, and ends after 1.2 s below 0.5, so short
pauses do not end a recording. A probability of 0.5 is not enough to start: a cough
scored above it for 320 ms, longer than a spoken "Ja". Above 0.9, coughs last at most 2
chunks and a single word at least 7; typing never got above 0.08
([FINDINGS](FINDINGS.md#app-silero-vad-against-coughs-and-typing)).

**Trimming:** If Silero VAD heard no speech, nothing is transcribed. Otherwise the
measured levels are scanned with hindsight: speech is 12 dB above the quietest part of the recording, and at least at
-55 dBFS, and the silence before and after it is cut, keeping a short margin. Speech
too quiet for that keeps the whole recording. Measured on the built-in microphone:
room -71 to -65 dBFS, speech peaks -36 to -32 dBFS
([FINDINGS](FINDINGS.md#app-silence-detection-on-the-built-in-microphone)).

**Speech to text:** Parakeet runs with ONNX Runtime on the CPU. The decoding follows
[onnx-asr](https://github.com/istupakov/onnx-asr) by Ilya Stupakov (MIT), whose ONNX
export of the model the app uses.

**Cleanup:** If enabled, Ollama is asked to load the model when a recording starts, so
it is ready when the transcript is. The transcript is then sent with a system prompt
that has the LLM remove hesitations ("uh", "äh"), apply self-corrections ("let's meet
on Tuesday, no, actually Thursday"), write enumerations as lists, and write numbers,
phone numbers and email addresses as they are written
([FINDINGS](FINDINGS.md#cleanup-rules-in-the-prompt-and-ministral-38b)). If Ollama
fails, the raw transcript displayed to the user as a fallback.

## Known limits

- Clearing the throat counts as speech: it is voiced, and Silero VAD scores it like a
  short word.
- A pause of more than 1.2 s ends a tapped recording.
- With slow microphones (some Bluetooth headsets), a tap can count as a hold.
- Each new build needs the Accessibility permission again (ad-hoc signing).

## Trade-offs

This project targets Apple silicon only (the build is aarch64); Intel Macs and other
platforms are not supported. The cleanup model needs about 6 GB of memory, which is
tight on an 8 GB Mac. Parakeet transcribes about 25 European languages, but the
cleanup is tuned for German and English only. The UI is minimal: the focus of this
exercise was a local processing pipeline. UX would profit from a setup wizard (setting
permissions, downloading models); to save time it is absent.

## Development and AI use

Almost all code and documentation in this repository was written by an LLM,
under rules in [AGENTS.md](AGENTS.md) and checked by a human:

- **Small, reviewed changes.** One branch per change, merged with `--no-ff`, so every
  step is visible in `git log --graph`. Commits written with the AI carry a
  `Co-Authored-By` trailer.
- **Independent review.** Each change comes with review notes that list its design
  decisions, claims to verify and risky spots. They serve as the prompt for a separate
  review agent and for my own review. Findings become follow-up commits, never amends,
  so the review stays in the history.
- **Measured, not assumed.** Claims about speed, accuracy and thresholds were checked
  with the prototype and recorded in [FINDINGS.md](FINDINGS.md).
- **Corrections.** For example:
  - The AI first trimmed with its own speech analysis, which turned typing into "Vielen
    Dank". Now one set of measured levels serves both decisions (2373096).
  - It read the AGENTS.md rule against stateful methods as "only free functions"; I
    rewrote the rule to prefer idiomatic Rust (c7fd278).
  - A review found that mapping frames to 16 kHz drifted at 22.05 kHz.

## Repository

| Path | Contents |
|---|---|
| `crates/apps/dictate` | The Tauri app |
| `crates/apps/dictate/ui` | The minimal frontend |
| `crates/apps/prototype` | The `stt` CLI prototype (excluded from root commands, use `-p prototype`) |
| `crates/libs/` | Reusable libraries (recording, transcription, cleanup, eval, onnx) |
| `fixtures/` | Test recordings (local only) and reference transcripts |
| `models/` | Model weights (local only, see [PLAN.md](PLAN.md#model-files-models-gitignored)) |

## Development requirements

- Apple silicon Mac, Rust (edition 2024)
- [cargo-nextest](https://nexte.st): `cargo install cargo-nextest --locked`
- [Ollama](https://ollama.com) for the cleanup step (optional)
- For the app: Node.js and the Tauri 2 CLI: `cargo install tauri-cli --version "^2.12" --locked`
- Model files: `models/` as listed in PLAN.md for the tests and the prototype; the app
  reads from `~/Library/Application Support/local.dictate/models/` (see SETUP.md)

## Commands

```bash
(cd crates/apps/dictate/ui && npm install)  # once: frontend dependencies
(cd crates/apps/dictate && cargo tauri dev) # run the app
cargo build                                 # build the app and libraries
cargo nextest run                           # run their tests
cargo nextest run -p transcription          # test one crate
cargo test --doc                            # doctests (nextest doesn't run these)
cargo doc --workspace --no-deps --open      # browse the docs
cargo run -p prototype -- --help            # the stt prototype CLI
```

## Roadmap

The most obvious ways forward for this project would be:
- detecting the spoken language, so the cleanup no longer needs the language setting
- refining the cleanup: tuning the prompt has shown big improvements already; rule-based
  steps could improve it further and in some cases skip the LLM call
- UI/UX improvements
- user-facing error messages with an action ("start the Ollama app") instead of the
  raw cause chain, and a log file, so errors outlive the next status

## Related work

[Handy](https://github.com/cjpais/Handy) is a mature, MIT-licensed Tauri app built
around the same idea, with more features.
