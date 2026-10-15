//! `stt record`: record from the microphone, then transcribe.

use std::io::Write;
use std::path::PathBuf;

use crate::{SttArgs, load_model, record_dictation, timed, timings_line, transcribe_samples};

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
    let dictation = record_dictation(args.save.as_deref())?;
    let (transcript, inference_secs) =
        timed(|| transcribe_samples(&mut model, &dictation.samples, stt.language))?;

    writeln!(out, "{}", transcript.text)?;
    writeln!(
        out,
        "{} | stop to text {:.2} s",
        timings_line(&transcript, &dictation.samples, load_secs, inference_secs),
        dictation.conversion_secs + inference_secs,
    )?;
    Ok(())
}
