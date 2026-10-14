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
