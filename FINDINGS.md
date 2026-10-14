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
