# Findings

Running log of what was tried, the numbers, and the conclusion.
Entries are ordered by milestone; no dates or times.

---

## M1: Whisper large-v3-turbo, GPU vs CPU

Setup: `stt transcribe` (release build) on German speech generated with macOS `say`
(voice Anna), 16 kHz mono; greedy decoding, no language hint; M4 Pro, whisper-rs 0.16.
Single runs; model already in the OS file cache.

| Audio  | GPU (Metal) inference | CPU inference | Real-time factor GPU / CPU |
|-------:|----------------------:|--------------:|---------------------------:|
| 1.9 s  | 1.03 s                | 4.44 s        | 0.54 / 2.33                |
| 6.1 s  | 1.07 s                | 4.50 s        | 0.17 / 0.73                |
| 20.0 s | 1.34 s                | 4.69 s        | 0.07 / 0.23                |
| 40.1 s | 2.03 s                | 7.19 s        | 0.05 / 0.18                |

- Whisper processes audio in 30 s windows and pads shorter input to 30 s, so every
  window costs a fixed ~1.0 s on GPU and ~4.4 s on CPU; text length adds a little on top
  (more tokens to decode). The real-time factor therefore depends on clip length and is
  only comparable between clips of similar length.
- All transcripts are correct and identical on GPU and CPU; language detected as `de`.
- Model load takes ~0.35 s on both devices. The first run of a newly built binary
  takes 5.5–6.3 s to load on GPU; the cause is not isolated (Metal shader compilation
  cached per binary is the likely candidate).
- Conclusion: for dictation-length clips (a few seconds) the floor dominates: ~1 s on
  GPU, which is acceptable; ~4.4 s on CPU, which is too slow to feel instant. GPU it is.

## M2: Parakeet encoder, CPU vs CoreML

Setup: `encoder-model.int8.onnx` (622 MB) through ort 2.0.0-rc.14 / ONNX Runtime 1.31,
release build, same German `say` clips as M1; fastest of three runs per clip. "CPU" is
a session with `AutoDevicePolicy::PreferCPU`; "ort default" is a session without any
device setting, which ort configures with its `MaxEfficiency` policy.

| Audio  | Encoder frames | CPU     | ort default (CPU + CoreML) |
|-------:|---------------:|--------:|---------------------------:|
| 1.9 s  | 24             | 0.038 s | 0.064 s                    |
| 6.1 s  | 77             | 0.089 s | 0.141 s                    |
| 20.0 s | 250            | 0.239 s | 0.409 s                    |
| 40.1 s | 502            | 0.490 s | 0.888 s                    |
| session creation | —    | 0.68 s  | 3.01 s                     |

- ort's default policy hands part of the encoder to CoreML on macOS, even without ort's
  `coreml` feature: on empty input the error comes from a node named
  `CoreMLExecutionProvider_…`. With `PreferCPU` it comes from a CPU node (`ConvInteger`).
- Why CoreML loses: ONNX Runtime's log for the default session reports 3,249 nodes
  (after optimization), 1,528 of them in 319 separate CoreML partitions and 1,721 on
  the CPU. All 217 `MatMulInteger` and 77 `ConvInteger` nodes, the int8 matrix
  multiplications and convolutions that carry the computation, stay on the CPU; CoreML
  only gets the elementwise and reshaping nodes between them. Every run crosses
  between CPU and CoreML hundreds of times, and loading compiles 319 CoreML models.
- The CoreML-assisted output differs from the CPU output by up to 0.135, on values
  that peak at 0.15–0.23. The CPU output is the reference; the effect on transcripts is
  measured once decoding works.
- Requesting the CoreML execution provider explicitly (ort feature `coreml`,
  NeuralNetwork format) gives output byte-identical to ort's default session. With the
  MLProgram format, session creation fails: CoreML rejects the encoder's unbounded time
  dimension.
- Unlike Whisper there is no fixed 30 s window: encode time grows with clip length.
- Conclusion: Parakeet runs on the CPU with `PreferCPU`. It is faster at every length,
  loads ~4× faster, and computes the reference output. The preprocessor's output is
  identical under both policies. CoreML could only help with an unquantized encoder
  whose matrix multiplications it can run itself; not tested.

## M2: Parakeet end to end vs Whisper

Setup: `ParakeetModel::transcribe` (preprocessor, encoder, greedy TDT decoding; CPU,
`PreferCPU`) vs `stt transcribe` with Whisper large-v3-turbo on Metal (M1 entry); same
German `say` clips, release build, fastest of three runs.

| Audio  | Parakeet total | of which encoder | Whisper inference |
|-------:|---------------:|-----------------:|------------------:|
| 1.9 s  | 0.042 s        | 0.038 s          | 1.03 s            |
| 6.1 s  | 0.101 s        | 0.093 s          | 1.07 s            |
| 20.0 s | 0.278 s        | 0.243 s          | 1.34 s            |
| 40.1 s | 0.563 s        | 0.490 s          | 2.03 s            |

- Parakeet is ~25× faster than Whisper on dictation-length clips and ~4× on 40 s;
  decoding adds 9–15 % to the encoder time. Model load: Parakeet 0.74 s, Whisper 0.4 s.
- Transcripts: 1.9 s, 6.1 s, 20.0 s and the English fox sentence are correct apart from
  punctuation. In the 40.1 s clip Parakeet drops "ins Kino" in the repeated sentence;
  Whisper transcribes it correctly. 2 s of silence and of noise give empty text.
- Our TDT loop is checked against onnx-asr 0.12 (the reference implementation by the
  author of the ONNX export): given the same encoder output it makes the same 240
  decoder calls on the 40.1 s clip, and the final texts are identical on the other five
  test clips.
- onnx-asr transcribes the 40.1 s clip correctly because its preprocessing differs from
  `nemo128.onnx` in the model download: its NumPy preprocessor (default on CPU) and its
  `nemo128_conv.onnx` both zero-pad the audio edges (the downloaded graph mirrors them)
  and count `samples / 160` valid frames, excluding the last one from normalization (the
  downloaded graph counts `samples / 160 + 1`). The resulting small feature
  differences flip one greedy decision. Which variant is more accurate overall is open
  until the fixture benchmark (M5).

## M3: Microphone to transcript

Setup: `stt record` (release build), built-in MacBook Pro microphone (48 kHz mono f32);
the German sentence "Guten Morgen, das ist ein Test der Spracherkennung." spoken by
`say` (voice Anna) through the built-in speakers. Single runs, model warm.

| Engine   | Audio | Inference | Stop to text |
|----------|------:|----------:|-------------:|
| Parakeet | 5.5 s | 0.11 s    | 0.13 s       |
| Whisper  | 5.6 s | 1.05 s    | 1.07 s       |

- Both transcripts are exact, including through speaker and room acoustics.
- Stop to text adds ~0.02 s to inference: draining the capture channel and resampling
  48 kHz to 16 kHz. Recording itself adds no meaningful latency.
- A recording saved with `--save` transcribes to the same text with `stt transcribe`.
- macOS can report a device overload (cpal `Xrun`) while the microphone permission
  dialog is open; it is counted as a warning, not an error. Later recordings showed none.

## M4: Ollama's qwen3:4b cannot turn thinking off

Setup: Ollama 0.32, `/api/chat` with `"think": false`.

- Ollama's `qwen3:4b` tag is Qwen3-4B-Thinking-2507 (`general.finetune: Thinking`,
  `general.version: 2507`). Its chat template always opens a `<think>` block, so
  `think: false` has no effect: the reasoning lands in `message.content`, followed by
  `</think>` and the answer. "Say hi" took 195 tokens and 2.6 s with the model loaded;
  cleaning "hallo" took 57 s including the model load.
- `cleanup::clean` therefore rejects any response that contains `</think>`.
- In Ollama's registry `qwen3:4b` and `qwen3:4b-thinking` share one model blob, as do
  `qwen3:4b-instruct` and `qwen3:4b-instruct-2507-q4_K_M`; there is no tag
  `qwen3:4b-instruct-2507`.
- Conclusion: the qwen3 candidate is `qwen3:4b-instruct`, the non-thinking variant.

## M4: The cleanup prompt must name the language

Setup: `/api/chat`, temperature 0, `think: false`; 14 short probe sentences (10 German,
4 English, with hesitations, a dictated question and a dictated instruction) with
gemma3:4b and qwen3:4b-instruct (both Q4_K_M), models loaded.

| System prompt | gemma3:4b | qwen3:4b-instruct |
|---|---|---|
| "Keep the language of the transcript. Never translate." | 2 of 10 German translated to English | 10 of 10 German translated; dictated email written out |
| Stronger "same language as the transcript" rule | 4 of 4 English translated to German | 0 translated; dictated instruction dropped |
| Same, plus one German and one English example pair | 4 of 4 English translated to German (one partly) | 0 translated; dictated email written out |
| "The transcript is in {language}. Write your reply in {language}." + `<transcript>` tags | 0 | 0 |

- Small models translate whenever they have to infer the language, in either
  direction. Naming the language stops it, so `clean` takes a `Language`; Whisper
  detects it, Parakeet needs it from the user.
- Listing "also" and "like" as fillers makes models delete them where they carry
  meaning ("I also like…"); the prompt lists only hesitation sounds.
- "Delete every hesitation sound …, wherever it occurs, including at the start" makes
  gemma3:4b drop a leading "Äh," that it keeps with "Remove hesitation sounds".
- Open for M5: gemma3:4b keeps a leading English "Um,"; qwen3:4b-instruct changed
  "auf drei Uhr" to "um drei Uhr" (meaning change) and drops a leading "Also".
- 0.5–1.3 s per sentence with the model loaded.

## M4: Stop to cleaned text with `stt run`

Setup: `stt run` (release build), German sentences spoken by `say` (voice Anna) through
the built-in speakers into the built-in microphone; models loaded before recording;
single runs.

| Engine + LLM | Audio | Transcribe | Clean | Stop to text |
|---|---:|---:|---:|---:|
| Parakeet + gemma3:4b | 7.5 s | 0.15 s | 0.81 s | 0.99 s |
| Whisper + gemma3:4b | 7.5 s | 1.07 s | 0.87 s | 1.97 s |
| Whisper + qwen3:4b-instruct | 7.6 s | 1.08 s | 0.46 s | 1.57 s |
| Parakeet + gemma3:4b | 32.1 s | 0.47 s | 1.60 s | 2.13 s |
| Whisper + gemma3:4b | 32.1 s | 1.96 s | 2.03 s | 4.05 s |

- With 32 s of audio, Parakeet + gemma3:4b meets the < 3 s target; Whisper + gemma3:4b
  misses it. Cleanup time grows with text length (~70 words: 1.6–2.0 s), because the
  LLM writes out the whole text.
- Preloading the LLM before recording keeps its load (0.1–0.3 s warm, ~1.7 s cold)
  out of the time from stop to text.
- The cleanup can change meaning while fixing recognition errors: Whisper heard "auf
  3 Uhr" as "in 3 Uhr", and gemma3:4b turned it into "um 15:00 Uhr". It also fixed
  "Staging-Söger" to "Staging-Server" correctly.
- Speaker-to-microphone recordings are too quiet to judge accuracy: one 32 s recording
  peaked at -23 dBFS with an RMS of -50 dBFS (the `say` file itself: 0 / -18 dBFS).
  On it Parakeet produced English nonsense and Whisper garbled several phrases; on the
  `say` file Parakeet made one error ("Buckend"). On poor audio Parakeet, which has no
  language hint, can switch to English. Accuracy needs real dictation (M5).
- `say` speech contains no real hesitations: both engines drop "äh"/"ähm" from it
  themselves, so it does not exercise the cleanup's filler removal.

## Conclusion: prototype close-out

The prototype stops after M4; the fixture benchmark (M5) was not run. Speed and the
cleanup's failure modes are answered; accuracy on real dictation is not.

**Answered**
- **Latency target is reachable with Parakeet.** Stop to cleaned text: Parakeet +
  gemma3:4b 0.99 s (7.5 s audio) and 2.13 s (32 s audio); Whisper + gemma3:4b 1.97 s
  and 4.05 s. The < 3 s target for 30 s of audio holds only with Parakeet.
- **Parakeet runs on the CPU, Whisper on Metal.** Parakeet transcribes 2–40 s in
  0.04–0.56 s; ort's default CoreML partitioning is ~1.8× slower and must be avoided
  with `PreferCPU`. Whisper has a ~1 s floor per 30 s window on GPU, ~4.4 s on CPU.
- **The cleanup is the slowest stage** for longer dictation (1.6–2.0 s for ~70 words),
  because the LLM rewrites the whole text; it grows with text length.
- **Models must be resident before recording starts.** Loads: Whisper 0.4 s,
  Parakeet 0.7 s, LLM ~1.7 s cold, 0.1–0.3 s warm. A freshly built binary's first
  Whisper load takes ~6 s.
- **The cleanup prompt must name the language.** Otherwise 4B models translate in
  either direction; with the language named and the transcript in `<transcript>` tags,
  0 of 14 probes were translated. So the pipeline needs the
  language: Whisper detects it, Parakeet needs it from the user.
- **The cleanup can change meaning** while fixing recognition errors ("in 3 Uhr" →
  "um 15:00 Uhr"). It is not safe to apply silently without a way to see the raw text.
- **Thinking models do not fit.** `qwen3:4b` reasons even with `think: false`; use
  `qwen3:4b-instruct`.
- **macOS microphone:** a denied permission yields digital silence, not an error; an
  input stream keeps its device when the default changes; overloads while the
  permission dialog is open are harmless.

**Open**
- WER on real dictation in German and English, per engine, including code-switching.
- Whether prompt tuning or a larger model (8–12B) stops meaning changes, and at what
  latency.
- RAM with both models resident.

**Recommendation for the app:** Parakeet (CPU) + gemma3:4b or qwen3:4b-instruct via
Ollama, both loaded at startup, an explicit de/en setting, and the raw transcript kept
alongside the cleaned text. The app should be able to save dictations as fixtures, so
accuracy is measured on real use.

## App: silence detection on the built-in microphone

Setup: `stt record --auto-stop -l de` (release build, Whisper), built-in MacBook Pro
microphone (48 kHz), quiet room, default `silence::Params`. One take per case, spoken
by a person. Levels are 30 ms RMS in dBFS.

| Take | Content                          | Stop                     | Result                         |
|------|----------------------------------|--------------------------|--------------------------------|
| 1    | one sentence                     | speech ended, 1.2 s late | exact transcript, 3.7 of 5.3 s kept |
| 2    | sentence, ~1 s pause, sentence   | speech ended, 1.2 s late | one transcript, pause bridged  |
| 3    | silence, distant music at the end | no speech after 10 s    | not transcribed                |
| 4, 5 | typing only                      | no speech after 10 s     | not transcribed                |
| 6    | heavy typing and a cough         | speech ended at 24.6 s   | transcribed (false positive)   |

- **Levels:** room -71 to -65 dBFS, speech peaks -36 to -32, music -61, typing up to
  -45, a cough -37 to -27. In a quiet room the start threshold sits at its lower limit
  (`min_start_db`, -55 dBFS); speech clears it by about 20 dB.
- **Whether there is speech must come from the live detector.** Its noise floor follows
  steady noise: during typing it rose to -58 dBFS (threshold -46) and never started.
  The quietest window of the whole recording stays at -70 dBFS (threshold -55); against
  it, fast typing makes runs of 3–5 loud frames, which count as speech, and Whisper
  turned them into "Vielen Dank." The trimming uses the whole-recording floor only to
  place the start and end of speech the live detector heard.
- **A cough is speech to an energy detector:** 300 ms at -37 to -27 dBFS, like a
  syllable. Telling them apart needs a speech classifier (VAD model).
- **Steady loud noise keeps a recording open.** After speech started, keystrokes above
  the end threshold (start − 6 dB) reset the 1.2 s hangover; take 6 ran until the
  typing stopped. It costs time, not words.
- **The microphone starts with digital silence:** one faint frame, then 90–120 ms of
  exact zeros, then the room. Noise floor windows that contain digital silence must be
  skipped, or the floor drops to -100 dBFS.
- Auto-stop adds the 1.2 s hangover to the time from the last word to text; stop to
  text itself was 0.57 s with Whisper.

## App: the Accessibility permission applies only to a new process

Setup: the dmg build, started without the permission, which was then switched on in
System Settings while Dictate ran.

- `CGPreflightPostEventAccess` kept returning false in the running process: unticking
  and ticking "Paste automatically" (which checks again) did not change it.
- After **Restart Dictate** (`AppHandle::restart`) it returned true and pasting worked.
- So the window has to offer a restart after the permission is granted; checking again
  in the same process is not enough.

## Cleanup: rules in the prompt, and ministral-3:8b

Setup: trials with dictated German and English samples covering hesitations,
self-corrections, spoken lists, amounts and phone numbers, legal sections, email
addresses, letters and dictated questions; local models via Ollama at temperature 0 on
an M4 Pro (24 GB).

- **"Keep every word that carries meaning" blocks the useful edits.** With it alone,
  models keep the retracted part of a self-correction ("Dienstag, nein, warte,
  Mittwoch") and never write a spoken enumeration as a list. The prompt names each edit
  as a rule: self-corrections, enumerations, writing numbers and units, spoken email
  addresses, greeting and sign-off lines.
- **4B models do not follow the longer prompt reliably.** qwen3:4b-instruct changed
  digits of a dictated phone number, with and without the rule on numbers; gemma3:4b
  answered dictated questions on some transcripts. ministral-3:8b followed the rules and
  changed no number, so it is the default model (6 GB download, Apache 2.0).
- **Speed is that of the 4B models.** About 1 s for a short dictation and 1.8 s for 70
  words with the model loaded; loading takes ~1 s from the disk cache.
- **ministral-3:8b sometimes writes Markdown bold or code markers** despite "plain text
  without Markdown"; `clean` removes `**` and `` ` `` from its reply.
- **Open:** it sometimes tightens wording beyond the rules: "äh also ich wollte ähm nur
  kurz sagen dass das meeting morgen äh auf drei uhr verschoben ist" became "Das Meeting
  ist morgen um drei Uhr verschoben." The cleanup also cannot restore words the
  speech-to-text dropped.

## App: Silero VAD against coughs and typing

Setup: Silero VAD v6.2 (`recording::vad`, CPU, 32 ms chunks at 16 kHz) and the energy
detector (`silence`, default `Params`) on the same files: the 18 dictation fixtures and
10 noise recordings, built-in MacBook Pro microphone, quiet room, one speaker. Each file
starts with about 1 s of room silence. "Run" is the most chunks in a row above a
probability.

| Recording | Energy detector | Silero: peak, longest run at 0.5 / 0.9 | Parakeet / Whisper |
|---|---|---|---|
| `cough` (several coughs) | speech started twice | 0.90, 10 / 1 | "" / "*hums*" |
| `cough-2` (single coughs, 2 s apart) | started 6 times | 0.56, 2 / 0 | "" / "Ahem. Ahem. …" |
| `cough-3` (two coughing fits) | started twice | 0.65, 3 / 0 | "" / "*hums*" |
| `throat-clear` (4 times "ähem") | started 4 times | 0.98, 30 / 11 | "" / "\*ahem\*" |
| `typing` (fast typing) | started at 0.6 s | 0.08, 0 / 0 | "" / "Vielen Dank." |
| `typing-speech` (typing, then a sentence while typing) | started, never ended | 1.00, 141 / 137 | – |
| `speech-cough` (sentence, 2 s, cough) | started again at the cough | the cough: 0.93, 14 / 2 | – |
| `word-ja`, `word-nein`, `word-okay` | started | 1.00, 9–13 / 7–10 | – |
| 18 dictation fixtures | started | 1.00, 18–188 / ≥ 17 | – |

- **A probability threshold of 0.5 does not reject coughs:** one cough stayed above it
  for 10 chunks (320 ms), longer than the 9 chunks of "Ja." At 0.9 the longest cough
  run is 2 chunks and the shortest word's 7. `vad::Params` starts speech after 4 chunks
  above 0.9 (128 ms) and ends it after 1.2 s below 0.5. On all 28 files this rejects every
  cough and the typing, starts on every word and dictation, and does not restart at the
  cough after speech. An end threshold of 0.75 gave the same events.
- **Clearing the throat is speech to Silero** (voiced, 11 chunks above 0.9): it still
  starts speech. Rejecting it would need a start run longer than the shortest word.
- **Typing alone started the energy detector** at 0.6 s; in the take of "silence
  detection on the built-in microphone" it did not. It never started Silero (peak 0.08).
- **Parakeet writes no text for coughs, throat-clearing or typing; Whisper invents
  some.** With Parakeet, a false start costs timing, not text: in tap mode, a cough
  followed by more than 1.2 s of quiet ends the recording before the speech, and typing
  keeps it open.
- **Starting later costs no words:** Silero starts 0.06–0.16 s after the energy detector
  on the dictations (1.0 s on one), but the recording runs from the key press and the
  trimming keeps padding before the first speech.
- Silero takes 0.09 ms per 32 ms chunk on the CPU.
- **Open:** one speaker, one room, a few takes of each noise. A pause of more than 1.2 s
  inside a dictation ends speech with both detectors (`zahlen`).
