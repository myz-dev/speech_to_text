//! `stt record`: record from the microphone, then transcribe.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Instant;

use anyhow::bail;
use recording::Flow;
use recording::silence::{self, Params};
use recording::vad::{self, Event, Vad};

use crate::{
    Dictation, Lang, SttArgs, check_recording, load_model, load_vad, record_dictation, timed,
    timings_line, wait_for_enter,
};

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub stt: SttArgs,
    /// Also save the recording as a 16 kHz mono WAV file, e.g. as a fixture.
    /// With --auto-stop, the untrimmed recording is saved
    #[arg(long)]
    pub save: Option<PathBuf>,
    /// Stop when speech ends instead of at the second Enter, print the level of every
    /// 30 ms frame, and trim silence before transcribing
    #[arg(long)]
    pub auto_stop: bool,
}

/// Loads the model, records from the default microphone between two presses of Enter
/// (or until speech ends, with `--auto-stop`), then prints the transcript and the
/// timings line of `transcribe`, extended by the time from the stop to the transcript.
/// Prompts go to stderr.
pub(crate) fn execute(args: &Args, out: &mut dyn Write) -> anyhow::Result<()> {
    let stt = &args.stt;
    // Loading first keeps it out of the time from stop to text.
    let (mut model, load_secs) = timed(|| load_model(stt))?;
    let mut vad = load_vad(stt)?;
    let dictation = if args.auto_stop {
        record_until_silence(&mut vad, args.save.as_deref())?
    } else {
        record_dictation(&mut vad, args.save.as_deref())?
    };
    let (transcript, inference_secs) =
        timed(|| Ok(model.transcribe(&dictation.samples, stt.language.map(Lang::code))?))?;

    writeln!(out, "{}", transcript.text)?;
    writeln!(
        out,
        "{} | stop to text {:.2} s",
        timings_line(&transcript, &dictation.samples, load_secs, inference_secs),
        dictation.conversion_secs + inference_secs,
    )?;
    Ok(())
}

/// Records from the default microphone until the speech detector reports the end of
/// speech (or no speech at all), or Enter is pressed. Prints every frame and every
/// speech event, then keeps only the detected speech.
fn record_until_silence(vad: &mut Vad, save: Option<&Path>) -> anyhow::Result<Dictation> {
    eprint!("Press Enter to start recording.");
    wait_for_enter()?;
    eprintln!("Recording until you stop speaking, or press Enter.");
    eprintln!("  time   level");
    let enter = enter_presses();
    let mut frame_count = 0_u32;
    let mut silent_start_frames = 0_u32;
    let mut stopped_at = Instant::now();

    let vad_params = vad::Params::default();
    let recording =
        recording::record_streaming(&Params::default(), vad, &vad_params, |levels, events| {
            for &level_db in levels {
                frame_count += 1;
                if frame_count <= silence::frames(1_000) && level_db <= silence::MIN_DB {
                    silent_start_frames += 1;
                }
                print_frame(frame_count, level_db);
            }
            for &event in events {
                eprintln!("  <- {event:?}");
                if matches!(event, Event::SpeechEnded | Event::NoSpeechTimeout) {
                    stopped_at = Instant::now();
                    return Flow::Stop;
                }
            }
            if enter.try_recv().is_ok() {
                stopped_at = Instant::now();
                return Flow::Stop;
            }
            Flow::Continue
        })?;

    eprintln!(
        "{} ms of digital silence in the first second.",
        silent_start_frames * silence::FRAME_MS
    );
    check_recording(&recording, save)?;
    let Some(speech) = recording.speech else {
        bail!("no speech detected");
    };
    let seconds = |samples: usize| samples as f64 / f64::from(recording::SAMPLE_RATE);
    eprintln!(
        "Speech from {:.2} s to {:.2} s, {:.2} s of {:.2} s kept.",
        seconds(speech.start),
        seconds(speech.end),
        seconds(speech.len()),
        seconds(recording.samples.len()),
    );
    Ok(Dictation {
        samples: recording.samples[speech].to_vec(),
        conversion_secs: stopped_at.elapsed().as_secs_f64(),
    })
}

/// Sends a message for every line read from stdin. The thread is not joined: after an
/// automatic stop it stays blocked on stdin until the process exits.
fn enter_presses() -> Receiver<()> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(
        move || {
            while wait_for_enter().is_ok() && sender.send(()).is_ok() {}
        },
    );
    receiver
}

/// Prints one frame: its end time, its level in dBFS, and a level bar from -80 dBFS
/// (2 dB per character).
fn print_frame(index: u32, level_db: f32) {
    let secs = f64::from(index * silence::FRAME_MS) / 1_000.0;
    let bar = "#".repeat(((level_db + 80.0) / 2.0).max(0.0) as usize);
    eprintln!("{secs:6.2} {level_db:7.1}  {bar}");
}
