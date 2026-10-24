//! Prototype app clearing milestones defined in [PLAN.md](../../../PLAN.md).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, bail};
use clap::{Parser, Subcommand, ValueEnum};
use cleanup::Language;
use recording::Recording;
use recording::silence::Params;
use recording::vad::{self, Vad};
use transcription::{Device, Model, Transcript};

pub mod bench;
pub mod clean;
pub mod record;
pub mod run;
pub mod transcribe;

pub use transcription::{PARAKEET_MODEL_DIR, WHISPER_MODEL_FILE};

/// Ollama model used for cleanup unless `--model` names another.
pub const DEFAULT_CLEANUP_MODEL: &str = "ministral-3:8b";

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
    /// Language spoken. Whisper uses it as a hint and detects the language if it is
    /// omitted; Parakeet ignores it
    #[arg(short, long, value_enum)]
    pub language: Option<Lang>,
    /// Directory containing the model files
    #[arg(long, default_value = "models")]
    pub models: PathBuf,
    /// Run Whisper on the CPU instead of the GPU.
    /// Ignored for Parakeet, it always runs on the CPU
    #[arg(long)]
    pub cpu: bool,
}

/// A language the prototype supports, as given on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Lang {
    De,
    En,
}

impl Lang {
    /// ISO 639-1 code, as Whisper expects it.
    pub fn code(self) -> &'static str {
        match self {
            Lang::De => "de",
            Lang::En => "en",
        }
    }
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

impl From<Engine> for transcription::Engine {
    fn from(engine: Engine) -> Self {
        match engine {
            Engine::Whisper => transcription::Engine::Whisper,
            Engine::Parakeet => transcription::Engine::Parakeet,
        }
    }
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

/// Audio dictated between two presses of Enter.
struct Dictation {
    /// 16 kHz mono samples.
    samples: Vec<f32>,
    /// Seconds from the second Enter until `samples` were ready (converting the device
    /// audio to 16 kHz). Counts towards the time from stop to text.
    conversion_secs: f64,
}

/// Records from the default microphone between two presses of Enter and saves the
/// recording to `save` if given. Prompts and warnings go to stderr.
fn record_dictation(vad: &mut Vad, save: Option<&Path>) -> anyhow::Result<Dictation> {
    eprint!("Press Enter to start recording.");
    wait_for_enter()?;
    eprint!("Recording, press Enter to stop.");
    let mut enter = Ok(());
    let mut stopped_at = Instant::now();
    let recording = recording::record(&Params::default(), vad, &vad::Params::default(), || {
        enter = wait_for_enter();
        stopped_at = Instant::now();
    })?;
    enter?;
    let conversion_secs = stopped_at.elapsed().as_secs_f64();

    check_recording(&recording, save)?;
    Ok(Dictation {
        samples: recording.samples,
        conversion_secs,
    })
}

/// Reports the length and overloads of `recording`, rejects an empty one and saves it
/// to `save` if given.
fn check_recording(recording: &Recording, save: Option<&Path>) -> anyhow::Result<()> {
    let audio_secs = duration_secs(&recording.samples);
    eprintln!("Recorded {audio_secs:.1} s from {}.", recording.device);
    if recording.overruns > 0 {
        eprintln!(
            "warning: the device reported {} overload(s); the audio may have gaps",
            recording.overruns
        );
    }
    if recording.samples.is_empty() {
        bail!("no audio recorded");
    }
    if let Some(path) = save {
        recording::save_wav(path, &recording.samples)?;
    }
    Ok(())
}

/// Blocks until a line is read from stdin. At the end of input it returns at once.
fn wait_for_enter() -> anyhow::Result<()> {
    std::io::stdin()
        .read_line(&mut String::new())
        .context("cannot read from stdin")?;
    Ok(())
}

/// Loads the model for the engine and device chosen on the command line.
fn load_model(stt: &SttArgs) -> anyhow::Result<Model> {
    let device = if stt.cpu { Device::Cpu } else { Device::Gpu };
    Ok(Model::load(stt.engine.into(), &stt.models, device)?)
}

/// Loads Silero VAD from the models directory, for the speech detection of a recording.
fn load_vad(stt: &SttArgs) -> anyhow::Result<Vad> {
    Ok(Vad::open(&stt.models.join(vad::MODEL_FILE))?)
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
