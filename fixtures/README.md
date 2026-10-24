# Fixtures

One folder per language (`de`, `en`). Each sample is a pair:

```
de/email-to-team.wav   # recording (gitignored – stays local)
de/email-to-team.txt   # exact reference transcript of what was said (committed)
```

- WAV, any sample rate (resampled to 16 kHz mono by the harness)
- 10–60 s each, ~10 per language, dictated the way you'd actually use it
- Include some code-switching (English terms inside German sentences)
- Reference = verbatim speech, including fillers you want the cleanup to remove;
  optionally add `<name>.expected.txt` with the ideal cleaned output

`noise/` holds recordings for the speech detector (gitignored, no transcripts), each with
about 1 s of room silence at both ends, recorded with
`cargo run -p prototype --release -- record -e parakeet --save fixtures/noise/<name>.wav`:

| File | Content |
|---|---|
| `cough.wav`, `cough-2.wav`, `cough-3.wav` | coughs: several, single ones 2 s apart, two fits |
| `throat-clear.wav` | clearing the throat ("ähem") 4 times |
| `typing.wav` | fast typing, no speech |
| `typing-speech.wav` | typing, then a sentence spoken while typing |
| `speech-cough.wav` | a sentence, 2 s pause, one cough |
| `word-ja.wav`, `word-nein.wav`, `word-okay.wav` | one short word each |

The tests in `crates/libs/recording/tests/vad.rs` skip a file that is missing.
