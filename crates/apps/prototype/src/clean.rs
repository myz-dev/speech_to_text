//! `stt clean`: clean up a transcript with the local LLM.

use std::io::Write;

use crate::{CleanupArgs, timed};

#[derive(Debug, clap::Args)]
pub struct Args {
    pub text: String,
    #[command(flatten)]
    pub cleanup: CleanupArgs,
}

/// Prints the cleaned `args.text`, then one line with the model and how long cleanup
/// took.
pub(crate) fn execute(args: &Args, out: &mut dyn Write) -> anyhow::Result<()> {
    let cleanup = &args.cleanup;
    let (cleaned, cleanup_secs) = timed(|| {
        Ok(cleanup::clean(
            &cleanup.model,
            &args.text,
            cleanup.language.into(),
        )?)
    })?;
    writeln!(out, "{cleaned}")?;
    writeln!(out, "model {} | cleanup {cleanup_secs:.2} s", cleanup.model)?;
    Ok(())
}
