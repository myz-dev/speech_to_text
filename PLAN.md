# CLI Prototype Plan

## Goal
Answer with numbers: **is local STT + LLM cleanup on an M4 Pro (24 GB) good
and fast enough for daily dictation in German and English?**

The prototype code is disposable; the learnings (`FINDINGS.md`), the fixture set
and the eval harness carry over to the real app.

## Questions to answer
1. **Accuracy** – WER on my voice, per language and per STT engine
2. **Latency** – key release → cleaned text, for 10 s / 30 s / 60 s recordings
3. **Engine choice** – Whisper (autoregressive, whisper.cpp/Metal) vs. Parakeet
   (TDT transducer, ONNX Runtime): speed, accuracy, robustness to silence/noise
4. **Cleanup reliability** – does the LLM clean without answering, changing meaning,
   dropping content, or translating German into English?
5. **Resources** – RAM, and whether both models can stay resident

## Out of scope
GUI, global hotkey, paste-at-cursor, streaming, packaging/signing.

## Model candidates
| Stage | Candidates | Runtime |
|---|---|---|
| STT | whisper large-v3-turbo (f16, q5_0), whisper small; large-v3 as reference | whisper-rs (Metal) |
| STT | parakeet-tdt-0.6b-v3 (int8; fp32 as reference) | ort (ONNX Runtime, CPU vs. CoreML EP) |
| Cleanup | qwen3:4b-instruct, gemma3:4b; step-up: qwen3:8b, gemma3:12b; baseline llama3.2:3b | Ollama |

## Model files (`models/`, gitignored)
| Path | Source |
|---|---|
| `ggml-large-v3-turbo.bin` | huggingface.co/ggerganov/whisper.cpp |
| `parakeet-tdt-0.6b-v3/{nemo128.onnx, encoder-model.int8.onnx, decoder_joint-model.int8.onnx, vocab.txt, config.json}` | huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx |
| `silero-vad/silero_vad.onnx` (v6.2, MIT) | github.com/snakers4/silero-vad/raw/v6.2/src/silero_vad/data/silero_vad.onnx |

## Milestones
| # | Deliverable | Done when |
|---|---|---|
| M0 | Workspace, git, findings log | `cargo build` passes, models on disk |
| M1 | `stt transcribe file.wav --engine whisper` | Prints text + timing |
| M2 | `--engine parakeet`: ONNX preprocessor → encoder → TDT greedy decode | Same file, both engines, comparable output |
| M3 | `stt record` (cpal, Enter to start/stop, 16 kHz mono) | Mic → transcript end-to-end |
| M4 | `stt clean`, `stt run` (Ollama, `think: false`) | Cleaned text + per-stage timing |
| M5 | `stt bench` over `fixtures/` | Table: 2 engines × ≥2 LLMs × 2 languages |
| M6 | Write-up in `FINDINGS.md` | Go/no-go + recommended combo |

M2 is the technically interesting one: Parakeet has no Rust wrapper with
Whisper's maturity, so we drive the ONNX graphs and implement the TDT decoding
loop ourselves. (`transcribe-rs`, used by Handy, is a reference to compare against.)

## Exit criteria (tentative)
- WER < ~5 % on clean speech, both languages
- < 3 s from end of speech to cleaned text for 30 s of audio
- LLM answers, alters meaning or switches language in < 1 of 20 fixtures

## Risks to watch
- **Code-switching** – English tech terms inside German sentences; Parakeet has no
  language hint, Whisper's `--language de` may mangle English words.
- **Long audio with Parakeet** – encoder attention memory grows with length;
  may need chunking beyond ~a few minutes (not an issue for dictation-length clips).
- **Whisper hallucinations** on silence/trailing noise ("Vielen Dank fürs Zuschauen").
- **CoreML EP** may be slower than CPU for some graphs, or fail on int8 ops; measure both.

## Workspace layout
```
crates/apps/prototype      CLI binary `stt`
crates/libs/recording      mic capture, resampling, permissions
crates/libs/transcription  STT backends (whisper-rs, Parakeet via ort)
crates/libs/cleanup        LLM backends (Ollama, later embedded llama.cpp)
crates/libs/eval           WER + latency harness
```
The app is `crates/apps/dictate`; `default-members` leaves the prototype out of root
commands. To retire the prototype, delete it.
