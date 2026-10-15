//! `stt run`: record, transcribe and clean in one go.

use std::io::Write;

use anyhow::bail;

use crate::SttArgs;

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub stt: SttArgs,
}

pub(crate) fn execute(_args: &Args, _out: &mut dyn Write) -> anyhow::Result<()> {
    bail!("not implemented yet (M4)")
}
