//! Prototype app clearing milestones defined in [PLAN.md](../../../PLAN.md).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::bail;
use clap::{Parser, Subcommand, ValueEnum};
use transcription::Device;
use transcription::whisper::WhisperModel;

/// Whisper model file inside the models directory.
pub const WHISPER_MODEL_FILE: &str = "ggml-large-v3-turbo.bin";

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
    /// Directory containing the model files
    #[arg(long, default_value = "models")]
    pub models: PathBuf,
    /// Run inference on the CPU instead of the GPU
    #[arg(long)]
    pub cpu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Engine {
    Whisper,
    Parakeet,
}

/// Executes the parsed command, writing its results to `out`.
pub fn run(cli: Cli, out: &mut dyn Write) -> anyhow::Result<()> {
    match cli.command {
        Command::Transcribe { file, stt } => transcribe(&file, &stt, out),
        Command::Record { .. } => bail!("not implemented yet (M3)"),
        Command::Clean { .. } | Command::Run { .. } => bail!("not implemented yet (M4)"),
        Command::Bench { .. } => bail!("not implemented yet (M5)"),
    }
}

/// Prints the transcript of `file`, then one line with language and timings.
fn transcribe(file: &Path, stt: &SttArgs, out: &mut dyn Write) -> anyhow::Result<()> {
    let samples = recording::load_wav(file)?;
    let device = if stt.cpu { Device::Cpu } else { Device::Gpu };

    let start = Instant::now();
    let model = match stt.engine {
        Engine::Whisper => WhisperModel::load(&stt.models.join(WHISPER_MODEL_FILE), device)?,
        Engine::Parakeet => bail!("not implemented yet (M2)"),
    };
    let load_secs = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let transcript = model.transcribe(&samples, stt.language.as_deref())?;
    let inference_secs = start.elapsed().as_secs_f64();

    let audio_secs = samples.len() as f64 / f64::from(recording::SAMPLE_RATE);
    // Real-time factor: inference time per second of audio; below 1 is faster than real time.
    let rtf = inference_secs / audio_secs;
    writeln!(out, "{}", transcript.text)?;
    writeln!(
        out,
        "language {} | audio {audio_secs:.1} s | load {load_secs:.2} s | \
         inference {inference_secs:.2} s | real-time factor {rtf:.2}",
        transcript.language,
    )?;
    Ok(())
}
