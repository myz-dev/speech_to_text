//! `stt bench`: run all fixtures and report error rates and latency.

use std::io::Write;
use std::path::PathBuf;

use anyhow::bail;

#[derive(Debug, clap::Args)]
pub struct Args {
    #[arg(long, default_value = "fixtures")]
    pub fixtures: PathBuf,
}

pub(crate) fn execute(_args: &Args, _out: &mut dyn Write) -> anyhow::Result<()> {
    bail!("not implemented yet (M5)")
}
