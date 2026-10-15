//! `stt record`: record from the microphone, then transcribe.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, bail};

use crate::{SttArgs, duration_secs, load_model, timed, timings_line, transcribe_samples};

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub stt: SttArgs,
    /// Also save the recording as a 16 kHz mono WAV file, e.g. as a fixture
    #[arg(long)]
    pub save: Option<PathBuf>,
}

/// Loads the model, records from the default microphone between two presses of Enter,
/// then prints the transcript and the timings line of `transcribe`, extended by the
/// time from the second Enter to the transcript. Prompts go to stderr.
pub(crate) fn execute(args: &Args, out: &mut dyn Write) -> anyhow::Result<()> {
    let stt = &args.stt;
    // Loading first keeps it out of the time from stop to text.
    let (mut model, load_secs) = timed(|| load_model(stt))?;

    eprint!("Press Enter to start recording.");
    wait_for_enter()?;
    eprint!("Recording, press Enter to stop.");
    let mut enter = Ok(());
    let mut stopped_at = Instant::now();
    let recording = recording::record(|| {
        enter = wait_for_enter();
        stopped_at = Instant::now();
    })?;
    enter?;
    // Converting the recording to 16 kHz counts towards the time from stop to text.
    let conversion_secs = stopped_at.elapsed().as_secs_f64();

    let audio_secs = duration_secs(&recording.samples);
    eprintln!("Recorded {audio_secs:.1} s from {}.", recording.device);
    if recording.overruns > 0 {
        eprintln!(
            "warning: the device reported {} overload(s); the audio may have gaps",
            recording.overruns
        );
    }
    if recording.samples.is_empty() {
        bail!("no audio recorded");
    }
    if let Some(path) = &args.save {
        recording::save_wav(path, &recording.samples)?;
    }

    let (transcript, inference_secs) =
        timed(|| transcribe_samples(&mut model, &recording.samples, stt.language.as_deref()))?;

    writeln!(out, "{}", transcript.text)?;
    writeln!(
        out,
        "{} | stop to text {:.2} s",
        timings_line(&transcript, &recording.samples, load_secs, inference_secs),
        conversion_secs + inference_secs,
    )?;
    Ok(())
}

/// Blocks until a line is read from stdin. At the end of input it returns at once.
fn wait_for_enter() -> anyhow::Result<()> {
    std::io::stdin()
        .read_line(&mut String::new())
        .context("cannot read from stdin")?;
    Ok(())
}
