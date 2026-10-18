# Setting up Dictate

Dictate lives in the menu bar. You hold a hotkey and speak, and the text is typed into
the app you are using. Speech recognition runs entirely on your Mac; nothing is sent to
the internet.

You need:
- a Mac with Apple silicon (M1 or newer); tested on macOS 26
- about 1 GB of free disk space (7 GB if you enable cleanup and download ministral-3:8b)
- the file `Dictate_0.1.0_aarch64.dmg`

Setup takes about 10 minutes, most of it the model download.

## 1. Install the app

1. Open `Dictate_0.1.0_aarch64.dmg` and drag **Dictate** into **Applications**.
2. Eject the disk image.

## 2. Allow the app to open

Dictate is not signed by a registered Apple developer, so macOS blocks it the first
time:

1. Open **Dictate** from Applications. macOS says it "could not verify" the app.
   Click **Done** (not "Move to Trash").
2. Open **System Settings → Privacy & Security** and scroll down to **Security**.
   Next to "Dictate was blocked…", click **Open Anyway**, then confirm with your
   password or Touch ID.
3. Click **Open** in the dialog that follows.

If macOS says the app "is damaged", or **Open Anyway** does not appear (it shows only
for about an hour after a blocked launch), remove the download mark in Terminal and
open Dictate again:

```bash
xattr -dr com.apple.quarantine /Applications/Dictate.app
```

This affects only Dictate; macOS keeps checking every other app.

macOS remembers this; later launches open normally. Dictate opens its window and puts
an icon in the menu bar. Closing the window only hides it: Dictate keeps running in the
menu bar. Opening Dictate again (from Applications, Spotlight or a launcher) or
icon → **Show Dictate** brings the window back; icon → **Quit Dictate** ends it.
Until the model is downloaded, the window shows "no usable Parakeet model"; that is
expected.

## 3. Download the speech model

Dictate uses NVIDIA's Parakeet model (about 640 MB). Open **Terminal**
(Applications → Utilities), paste this block and press Return:

```bash
dir="$HOME/Library/Application Support/local.dictate/models/parakeet-tdt-0.6b-v3"
base="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main"
for f in nemo128.onnx encoder-model.int8.onnx decoder_joint-model.int8.onnx vocab.txt; do
  curl -L --fail --create-dirs -o "$dir/$f" "$base/$f" || echo "FAILED: $f"
done
```

curl shows its progress per file. When the prompt returns without "FAILED", the model
is in place. If a file failed, run the block again.

Then quit Dictate (menu bar icon → **Quit Dictate**) and open it again, so it loads the
model.

## 4. First dictation

1. The window says **Ready** once the model is loaded (a few seconds after Dictate
   starts).
2. Click into any text field, e.g. in Notes.
3. **Hold ⌃⌥Space** (Control + Option + Space), say a sentence, release the keys.
4. The first time, macOS asks whether Dictate may use the microphone. Click **Allow**,
   then dictate again.

The text appears in the window and on the clipboard: paste it with ⌘V. The window
says "on the clipboard" because automatic pasting is off at first (step 5).

Two ways to use the hotkey:
- **Hold** it while you speak; releasing it ends the recording.
- **Tap** it once and speak; the recording ends by itself about a second after you
  stop speaking, or when you tap again.

Short pauses do not end a recording. If no speech is heard, nothing is pasted.

## 5. Optional: paste automatically

To have the text typed into the app you are using, tick **Paste automatically** in the
window. macOS then needs to allow Dictate to press ⌘V for you (the **Accessibility**
permission). The window shows the steps:

1. Click **Open Accessibility settings**. If macOS first shows a dialog, choose
   **Open System Settings**.
2. In **System Settings → Privacy & Security → Accessibility**, switch **Dictate** on.
3. Click **Restart Dictate** in the window: macOS applies the permission only after a
   restart.

From then on the text is typed where your cursor is, and your previous clipboard text
is put back afterwards. Until the permission applies, the text goes to the clipboard.

## 6. Optional: clean up the text with a local LLM

The speech model already writes punctuated sentences. The cleanup additionally removes
hesitations ("ähm", "uh"), false starts and retracted words ("Dienstag, nein, Mittwoch"),
and writes spoken lists, amounts and email addresses the way they are written, using a
language model on your Mac. It adds about 1–2 s per dictation.

1. Install [Ollama](https://ollama.com) (the app, or `brew install ollama`) and start it.
2. In Terminal: `ollama pull ministral-3:8b` (about 6 GB).
3. In the Dictate window, tick **Clean up with a local LLM** and choose the language
   you dictate in (Deutsch or English).

The window shows the cleaned text with the original below it. If Ollama is not
running, the original text is used and the window says why.

Settings are not saved: after restarting Dictate they are back to their defaults.

## Troubleshooting

| What you see | What to do |
|---|---|
| "no usable Parakeet model in …" | Step 3 did not finish; run the download again, then restart Dictate. |
| "… delivered only digital silence; check that this app may use the microphone" | System Settings → Privacy & Security → **Microphone**: switch Dictate on. |
| "Dictate may not paste yet" | Step 5. The text is on the clipboard; paste it with ⌘V meanwhile. |
| Nothing happens on ⌃⌥Space | Another app may use the same hotkey (e.g. a launcher). Quit it and try again. |
| A tap stops at once with a Bluetooth headset | The headset's microphone starts slowly, so the tap counts as a hold. Hold the key while you speak instead. |
| Pasting stopped working after installing a newer version | System Settings → Privacy & Security → Accessibility: remove Dictate (−) and switch it on again. Each build is a new app to macOS. |

## Uninstall

Quit Dictate, delete it from Applications, then delete the model and the permissions
you gave it (microphone, Accessibility):

```bash
rm -rf "$HOME/Library/Application Support/local.dictate"
tccutil reset All local.dictate
```

If you downloaded the cleanup model only for Dictate, remove it too (about 6 GB):

```bash
ollama rm ministral-3:8b
```
