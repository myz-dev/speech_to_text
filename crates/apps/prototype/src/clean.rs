//! `stt clean`: clean up a transcript with the local LLM.

use std::io::Write;

use crate::{DEFAULT_CLEANUP_MODEL, Lang, timed};

#[derive(Debug, clap::Args)]
pub struct Args {
    pub text: String,
    /// Language of the transcript; the cleaned text stays in it
    #[arg(short, long, value_enum)]
    pub language: Lang,
    /// Ollama model for the cleanup
    #[arg(short, long, default_value = DEFAULT_CLEANUP_MODEL)]
    pub model: String,
}

/// Prints the cleaned `args.text`, then one line with the model and how long cleanup
/// took.
pub(crate) fn execute(args: &Args, out: &mut dyn Write) -> anyhow::Result<()> {
    let (cleaned, cleanup_secs) = timed(|| {
        Ok(cleanup::clean(
            &args.model,
            &args.text,
            args.language.into(),
        )?)
    })?;
    writeln!(out, "{cleaned}")?;
    writeln!(out, "model {} | cleanup {cleanup_secs:.2} s", args.model)?;
    Ok(())
}
