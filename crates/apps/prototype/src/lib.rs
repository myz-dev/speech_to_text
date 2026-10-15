//! Prototype app clearing milestones defined in [PLAN.md](../../../PLAN.md).

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use cleanup::Language;
use transcription::Device;
use transcription::parakeet::ParakeetModel;
use transcription::whisper::WhisperModel;

pub mod bench;
pub mod clean;
pub mod record;
pub mod run;
pub mod transcribe;

/// Whisper model file inside the models directory.
pub const WHISPER_MODEL_FILE: &str = "ggml-large-v3-turbo.bin";

/// Parakeet model directory inside the models directory.
pub const PARAKEET_MODEL_DIR: &str = "parakeet-tdt-0.6b-v3";

/// Ollama model used for cleanup unless `--model` names another.
pub const DEFAULT_CLEANUP_MODEL: &str = "gemma3:4b";

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
    Transcribe(transcribe::Args),
    /// Record from the microphone between two presses of Enter, then transcribe (M3)
    Record(record::Args),
    /// Clean up a raw transcript with the local LLM (M4)
    Clean(clean::Args),
    /// Record, transcribe and clean in one go, with per-stage timing (M4)
    Run(run::Args),
    /// Run all fixtures and report error rates and latency (M5)
    Bench(bench::Args),
}

#[derive(Debug, clap::Args)]
pub struct SttArgs {
    #[arg(short, long, value_enum, default_value_t = Engine::Whisper)]
    pub engine: Engine,
    /// Language hint for Whisper (de, en); auto-detect if omitted. Parakeet ignores it
    #[arg(short, long)]
    pub language: Option<String>,
    /// Directory containing the model files
    #[arg(long, default_value = "models")]
    pub models: PathBuf,
    /// Run Whisper on the CPU instead of the GPU.
    /// Ignored for Parakeet, it always runs on the CPU
    #[arg(long)]
    pub cpu: bool,
}

#[derive(Debug, clap::Args)]
pub struct CleanupArgs {
    /// Language of the transcript; the cleaned text stays in it
    #[arg(short, long, value_enum)]
    pub language: Lang,
    /// Ollama model for the cleanup
    #[arg(short, long, default_value = DEFAULT_CLEANUP_MODEL)]
    pub model: String,
}

/// A language the cleanup supports, as given on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Lang {
    De,
    En,
}

impl From<Lang> for Language {
    fn from(lang: Lang) -> Self {
        match lang {
            Lang::De => Language::German,
            Lang::En => Language::English,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Engine {
    Whisper,
    Parakeet,
}

/// Executes the parsed command, writing its results to `out`.
pub fn run(cli: Cli, out: &mut dyn Write) -> anyhow::Result<()> {
    match &cli.command {
        Command::Transcribe(args) => transcribe::execute(args, out),
        Command::Record(args) => record::execute(args, out),
        Command::Clean(args) => clean::execute(args, out),
        Command::Run(args) => run::execute(args, out),
        Command::Bench(args) => bench::execute(args, out),
    }
}

/// A loaded speech model of the engine chosen on the command line.
enum Model {
    Whisper(WhisperModel),
    Parakeet(ParakeetModel),
}

/// Loads the model for `stt.engine`.
fn load_model(stt: &SttArgs) -> anyhow::Result<Model> {
    let model = match stt.engine {
        Engine::Whisper => {
            let device = if stt.cpu { Device::Cpu } else { Device::Gpu };
            let path = stt.models.join(WHISPER_MODEL_FILE);
            Model::Whisper(WhisperModel::load(&path, device)?)
        }
        Engine::Parakeet => {
            let path = stt.models.join(PARAKEET_MODEL_DIR);
            Model::Parakeet(ParakeetModel::load(&path)?)
        }
    };
    Ok(model)
}

struct Transcript {
    text: String,
    /// Language code the engine transcribed in, if it reports one.
    language: Option<&'static str>,
}

/// Transcribes 16 kHz mono `samples`. `language` is a hint for Whisper; Parakeet
/// ignores it.
fn transcribe_samples(
    model: &mut Model,
    samples: &[f32],
    language: Option<&str>,
) -> anyhow::Result<Transcript> {
    Ok(match model {
        Model::Whisper(model) => {
            let transcript = model.transcribe(samples, language)?;
            Transcript {
                text: transcript.text,
                language: Some(transcript.language),
            }
        }
        Model::Parakeet(model) => Transcript {
            text: model.transcribe(samples)?,
            language: None,
        },
    })
}

/// The summary line after a transcript: language (if the engine reports one), audio
/// length, load and inference time, and real-time factor.
fn timings_line(
    transcript: &Transcript,
    samples: &[f32],
    load_secs: f64,
    inference_secs: f64,
) -> String {
    let audio_secs = duration_secs(samples);
    // Real-time factor: inference time per second of audio; below 1 is faster than real time.
    let rtf = inference_secs / audio_secs;
    // Parakeet does not report the language it heard.
    let language = transcript
        .language
        .map(|code| format!("language {code} | "))
        .unwrap_or_default();
    format!(
        "{language}audio {audio_secs:.1} s | load {load_secs:.2} s | \
         inference {inference_secs:.2} s | real-time factor {rtf:.2}"
    )
}

/// Length of 16 kHz `samples` in seconds.
fn duration_secs(samples: &[f32]) -> f64 {
    samples.len() as f64 / f64::from(recording::SAMPLE_RATE)
}

/// Runs `f` and returns its result with the elapsed seconds.
fn timed<T>(f: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<(T, f64)> {
    let start = Instant::now();
    let result = f()?;
    Ok((result, start.elapsed().as_secs_f64()))
}
