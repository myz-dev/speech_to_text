//! Prototype app clearing milestones defined in [PLAN.md](../../../PLAN.md).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::bail;
use clap::{Parser, Subcommand, ValueEnum};
use transcription::Device;
use transcription::parakeet::ParakeetModel;
use transcription::whisper::WhisperModel;

/// Whisper model file inside the models directory.
pub const WHISPER_MODEL_FILE: &str = "ggml-large-v3-turbo.bin";

/// Parakeet model directory inside the models directory.
pub const PARAKEET_MODEL_DIR: &str = "parakeet-tdt-0.6b-v3";

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

/// Prints the transcript of `file`, then one line with the language (if the engine
/// reports one) and timings.
fn transcribe(file: &Path, stt: &SttArgs, out: &mut dyn Write) -> anyhow::Result<()> {
    let samples = recording::load_wav(file)?;
    let (mut model, load_secs) = load_model(stt)?;
    let (transcript, inference_secs) =
        timed(|| transcribe_samples(&mut model, &samples, stt.language.as_deref()));
    let transcript = transcript?;

    writeln!(out, "{}", transcript.text)?;
    writeln!(
        out,
        "{}",
        timings_line(&transcript, &samples, load_secs, inference_secs)
    )?;
    Ok(())
}

/// A loaded speech model of the engine chosen on the command line.
enum Model {
    Whisper(WhisperModel),
    Parakeet(ParakeetModel),
}

/// Loads the model for `stt.engine` and returns it with the elapsed seconds.
fn load_model(stt: &SttArgs) -> anyhow::Result<(Model, f64)> {
    let (model, load_secs) = timed(|| -> anyhow::Result<Model> {
        Ok(match stt.engine {
            Engine::Whisper => {
                let device = if stt.cpu { Device::Cpu } else { Device::Gpu };
                let path = stt.models.join(WHISPER_MODEL_FILE);
                Model::Whisper(WhisperModel::load(&path, device)?)
            }
            Engine::Parakeet => {
                let path = stt.models.join(PARAKEET_MODEL_DIR);
                Model::Parakeet(ParakeetModel::load(&path)?)
            }
        })
    });
    Ok((model?, load_secs))
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
    let audio_secs = samples.len() as f64 / f64::from(recording::SAMPLE_RATE);
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

/// Runs `f` and returns its result with the elapsed seconds.
fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let result = f();
    (result, start.elapsed().as_secs_f64())
}
