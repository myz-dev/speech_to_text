//! Prototype app clearing milestones defined in [PLAN.md](../../../PLAN.md).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, bail};
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
    /// Record from the microphone between two presses of Enter, then transcribe (M3)
    Record {
        #[command(flatten)]
        stt: SttArgs,
        /// Also save the recording as a 16 kHz mono WAV file, e.g. as a fixture
        #[arg(long)]
        save: Option<PathBuf>,
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
        Command::Record { stt, save } => record(&stt, save.as_deref(), out),
        Command::Clean { .. } | Command::Run { .. } => bail!("not implemented yet (M4)"),
        Command::Bench { .. } => bail!("not implemented yet (M5)"),
    }
}

/// Prints the transcript of `file`, then one line with the language (if the engine
/// reports one) and timings.
fn transcribe(file: &Path, stt: &SttArgs, out: &mut dyn Write) -> anyhow::Result<()> {
    let samples = recording::load_wav(file)?;
    let (mut model, load_secs) = timed(|| load_model(stt))?;
    let (transcript, inference_secs) =
        timed(|| transcribe_samples(&mut model, &samples, stt.language.as_deref()))?;

    writeln!(out, "{}", transcript.text)?;
    writeln!(
        out,
        "{}",
        timings_line(&transcript, &samples, load_secs, inference_secs)
    )?;
    Ok(())
}

/// Loads the model, records from the default microphone between two presses of Enter,
/// then prints the transcript and the timings line of `transcribe`, extended by the
/// time from the second Enter to the transcript. Prompts go to stderr.
fn record(stt: &SttArgs, save: Option<&Path>, out: &mut dyn Write) -> anyhow::Result<()> {
    // Loading first keeps it out of the time from stop to text.
    let (mut model, load_secs) = timed(|| load_model(stt))?;

    eprint!("Press Enter to start recording.");
    wait_for_enter()?;
    eprint!("Recording, press Enter to stop.");
    let mut enter = Ok(());
    let mut stopped_at = Instant::now();
    let recording = recording::record(|| {
        enter = wait_for_enter();
        stopped_at = Instant::now();
    })?;
    enter?;
    // Converting the recording to 16 kHz counts towards the time from stop to text.
    let conversion_secs = stopped_at.elapsed().as_secs_f64();

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

    let (transcript, inference_secs) =
        timed(|| transcribe_samples(&mut model, &recording.samples, stt.language.as_deref()))?;

    writeln!(out, "{}", transcript.text)?;
    writeln!(
        out,
        "{} | stop to text {:.2} s",
        timings_line(&transcript, &recording.samples, load_secs, inference_secs),
        conversion_secs + inference_secs,
    )?;
    Ok(())
}

/// Blocks until a line is read from stdin. At the end of input it returns at once.
fn wait_for_enter() -> anyhow::Result<()> {
    std::io::stdin()
        .read_line(&mut String::new())
        .context("cannot read from stdin")?;
    Ok(())
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
