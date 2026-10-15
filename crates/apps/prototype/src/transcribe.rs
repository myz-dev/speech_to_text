//! `stt transcribe`: transcribe a WAV file.

use std::io::Write;
use std::path::PathBuf;

use crate::{Lang, SttArgs, load_model, timed, timings_line};

#[derive(Debug, clap::Args)]
pub struct Args {
    pub file: PathBuf,
    #[command(flatten)]
    pub stt: SttArgs,
}

/// Prints the transcript of `args.file`, then one line with the language (if the engine
/// reports one) and timings.
pub(crate) fn execute(args: &Args, out: &mut dyn Write) -> anyhow::Result<()> {
    let stt = &args.stt;
    let samples = recording::load_wav(&args.file)?;
    let (mut model, load_secs) = timed(|| load_model(stt))?;
    let (transcript, inference_secs) =
        timed(|| Ok(model.transcribe(&samples, stt.language.map(Lang::code))?))?;

    writeln!(out, "{}", transcript.text)?;
    writeln!(
        out,
        "{}",
        timings_line(&transcript, &samples, load_secs, inference_secs)
    )?;
    Ok(())
}
