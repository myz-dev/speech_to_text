# speech_to_text

An educational project: learning how local speech-to-text works and which
models run well on Apple silicon. Audio is recorded, transcribed and cleaned
up by a local LLM entirely on-device, with no cloud services involved.

**Trade-off:** we target Apple silicon (Metal GPU, fast CPU cores) on purpose.
Weaker or non-Apple machines are not supported for now.

Goals, milestones and model candidates are in [PLAN.md](PLAN.md); results go
into [FINDINGS.md](FINDINGS.md).
Development rules (for humans and agents) are in [AGENTS.md](AGENTS.md).

## Repository

| Path | Contents |
|---|---|
| `crates/apps/` | Executables: the `stt` CLI prototype and the `dictate` Tauri app |
| `crates/libs/` | Reusable libraries (recording, transcription, cleanup, eval) |
| `fixtures/` | Test recordings (local only) and reference transcripts |
| `models/` | Model weights (local only, see [PLAN.md](PLAN.md#model-files-models-gitignored)) |

## Requirements

- Apple silicon Mac, Rust (edition 2024)
- [cargo-nextest](https://nexte.st): `cargo install cargo-nextest --locked`
- [Ollama](https://ollama.com) for the cleanup step
- For the app: Node.js and the Tauri 2 CLI: `cargo install tauri-cli --version "^2.12" --locked`
- Model files in `models/` as listed in PLAN.md

## Commands

```bash
cargo build                                                # build everything
cargo nextest run                                          # run all tests
cargo nextest run -p transcription                         # test one crate
cargo test --doc                                           # doctests (nextest doesn't run these)
cargo run -p prototype -- transcribe clip.wav -e parakeet  # run the CLI
cargo run -p prototype -- --help                           # list commands
(cd crates/apps/dictate/ui && npm install)                 # once, for the app
(cd crates/apps/dictate && cargo tauri dev)                # run the app
cargo doc --workspace --no-deps --open                     # browse the docs
```
