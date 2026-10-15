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

## M4: gemma3:4b cleanup, first impressions

Setup: `cleanup::clean` with `gemma3:4b` (Q4_K_M), temperature 0, model loaded; the
five sentences of `crates/libs/cleanup/tests/ollama.rs`.

- 0.5–0.9 s per sentence.
- It removes fillers and fixes punctuation and capitalization, keeps German German and
  English terms inside German sentences English, and does not answer a dictated
  question or write a dictated email.
- Without an explicit "questions end with a question mark" rule it left a dictated
  question unpunctuated; with the rule it adds the comma and question mark.
- It keeps "Also," at the start of a German sentence (filler or not is ambiguous) and
  drops "so basically" in English.
