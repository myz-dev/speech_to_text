//! `stt run`: record, transcribe and clean in one go.

use std::io::Write;
use std::path::PathBuf;

use anyhow::bail;

use crate::{
    DEFAULT_CLEANUP_MODEL, Engine, Lang, SttArgs, duration_secs, load_model, record_dictation,
    timed,
};

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub stt: SttArgs,
    /// Ollama model for the cleanup
    #[arg(short, long, default_value = DEFAULT_CLEANUP_MODEL)]
    pub model: String,
    /// Also save the recording as a 16 kHz mono WAV file, e.g. as a fixture
    #[arg(long)]
    pub save: Option<PathBuf>,
}

/// Loads both models, records between two presses of Enter, then prints the raw
/// transcript, the cleaned text, and one line with the language and per-stage timings.
/// Prompts and load times go to stderr.
pub(crate) fn execute(args: &Args, out: &mut dyn Write) -> anyhow::Result<()> {
    let stt = &args.stt;
    // Checked before loading and recording, so a missing -l costs nothing.
    if stt.engine == Engine::Parakeet && stt.language.is_none() {
        bail!("parakeet does not detect the language; pass -l de or -l en");
    }

    // Loading first keeps it out of the time from stop to text.
    let (mut model, stt_load_secs) = timed(|| load_model(stt))?;
    let ((), llm_load_secs) = timed(|| Ok(cleanup::load(&args.model)?))?;
    eprintln!(
        "Loaded {:?} in {stt_load_secs:.2} s and {} in {llm_load_secs:.2} s.",
        stt.engine, args.model
    );

    let dictation = record_dictation(args.save.as_deref())?;
    let (transcript, transcribe_secs) =
        timed(|| Ok(model.transcribe(&dictation.samples, stt.language.map(Lang::code))?))?;
    // An LLM given an empty transcript tends to invent text.
    if transcript.text.trim().is_empty() {
        bail!("no speech recognized");
    }
    let language = cleanup_language(stt.language, transcript.language)?;
    let (cleaned, clean_secs) = timed(|| {
        Ok(cleanup::clean(
            &args.model,
            &transcript.text,
            language.into(),
        )?)
    })?;

    let stop_to_text = dictation.conversion_secs + transcribe_secs + clean_secs;
    writeln!(out, "{}", transcript.text)?;
    writeln!(out, "{cleaned}")?;
    writeln!(
        out,
        "language {} | audio {:.1} s | transcribe {transcribe_secs:.2} s | \
         clean {clean_secs:.2} s | stop to text {stop_to_text:.2} s",
        language.code(),
        duration_secs(&dictation.samples),
    )?;
    Ok(())
}

/// The language to clean in: the one given with `-l`, which Whisper also transcribed
/// in, or else the one Whisper detected. Parakeet detects none.
fn cleanup_language(given: Option<Lang>, detected: Option<&str>) -> anyhow::Result<Lang> {
    match (given, detected) {
        (Some(lang), _) => Ok(lang),
        (None, Some("de")) => Ok(Lang::De),
        (None, Some("en")) => Ok(Lang::En),
        (None, Some(code)) => {
            bail!("whisper detected language {code}; cleanup supports de and en, pass -l")
        }
        (None, None) => bail!("the language is unknown; pass -l de or -l en"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_language_wins_over_detected() {
        assert_eq!(
            cleanup_language(Some(Lang::En), Some("de")).unwrap(),
            Lang::En
        );
    }

    #[test]
    fn detected_german_and_english_are_used() {
        assert_eq!(cleanup_language(None, Some("de")).unwrap(), Lang::De);
        assert_eq!(cleanup_language(None, Some("en")).unwrap(), Lang::En);
    }

    #[test]
    fn other_detected_language_is_rejected() {
        let err = cleanup_language(None, Some("nl")).unwrap_err();
        assert_eq!(
            err.to_string(),
            "whisper detected language nl; cleanup supports de and en, pass -l"
        );
    }

    #[test]
    fn no_language_at_all_is_rejected() {
        assert!(cleanup_language(None, None).is_err());
    }
}
