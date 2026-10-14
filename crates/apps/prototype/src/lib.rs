//! Prototype app clearing milestones defined in [PLAN.md](../../../PLAN.md).

use std::path::PathBuf;

use anyhow::bail;
use clap::{Parser, Subcommand, ValueEnum};

/// Prototype for local voice-to-text: record, transcribe, clean up, benchmark.
#[derive(Debug, Parser)]
#[command(name = "stt", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Transcribe a WAV file (M1 whisper, M2 parakeet)
    Transcribe {
        file: PathBuf,
        #[command(flatten)]
        stt: SttArgs,
    },
    /// Record from the microphone until Enter is pressed, then transcribe (M3)
    Record {
        #[command(flatten)]
        stt: SttArgs,
    },
    /// Clean up a raw transcript with the local LLM (M4)
    Clean { text: String },
    /// Record, transcribe and clean in one go, with per-stage timing (M4)
    Run {
        #[command(flatten)]
        stt: SttArgs,
    },
    /// Run all fixtures and report error rates and latency (M5)
    Bench {
        #[arg(long, default_value = "fixtures")]
        fixtures: PathBuf,
    },
}

#[derive(Debug, clap::Args)]
pub struct SttArgs {
    #[arg(short, long, value_enum, default_value_t = Engine::Whisper)]
    pub engine: Engine,
    /// Language hint for Whisper (de, en); auto-detect if omitted. Parakeet ignores it.
    #[arg(short, long)]
    pub language: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Engine {
    Whisper,
    Parakeet,
}

/// Executes the parsed command.
pub fn run(cli: Cli) -> anyhow::Result<()> {
    let milestone = match cli.command {
        Command::Transcribe { stt, .. } => match stt.engine {
            Engine::Whisper => "M1",
            Engine::Parakeet => "M2",
        },
        Command::Record { .. } => "M3",
        Command::Clean { .. } | Command::Run { .. } => "M4",
        Command::Bench { .. } => "M5",
    };
    bail!("not implemented yet ({milestone})")
}
