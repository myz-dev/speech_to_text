//! `clean` against a local Ollama server, once per model in [`MODELS`]. Models that are
//! not pulled are skipped with a note; without Ollama every test passes with a note.
//! Outputs are deterministic (temperature 0), but asserted loosely so that prompt
//! wording can change without rewriting the tests.

use cleanup::{CleanError, Language, OLLAMA_URL, clean};

const MODELS: [&str; 2] = ["gemma3:4b", "qwen3:4b-instruct"];

/// True if Ollama runs and has `model`; prints why a test is skipped otherwise.
fn available(model: &str) -> bool {
    let tags = ureq::get(format!("{OLLAMA_URL}/api/tags"))
        .call()
        .and_then(|mut response| response.body_mut().read_to_string());
    match tags {
        Ok(tags) if tags.contains(&format!("\"name\":\"{model}\"")) => true,
        Ok(_) => {
            eprintln!("skipping {model}: not pulled (`ollama pull {model}`)");
            false
        }
        Err(_) => {
            eprintln!("skipping {model}: ollama not reachable at {OLLAMA_URL}");
            false
        }
    }
}

/// Cleans `transcript` with every available model; returns `(model, cleaned)` pairs.
fn clean_with_models(transcript: &str, language: Language) -> Vec<(&'static str, String)> {
    MODELS
        .into_iter()
        .filter(|model| available(model))
        .map(|model| {
            let cleaned = clean(model, transcript, language).unwrap();
            eprintln!("{model}: {transcript:?}\n→ {cleaned:?}");
            (model, cleaned)
        })
        .collect()
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn has_word(text: &str, word: &str) -> bool {
    words(text).iter().any(|w| w == word)
}

#[test]
fn removes_german_hesitations_and_keeps_german() {
    let transcript = "äh also ich wollte ähm nur kurz sagen dass das meeting morgen äh auf drei uhr verschoben ist";
    for (model, cleaned) in clean_with_models(transcript, Language::German) {
        assert!(
            !has_word(&cleaned, "äh") && !has_word(&cleaned, "ähm"),
            "{model}: {cleaned}"
        );
        for kept in ["meeting", "morgen", "verschoben"] {
            assert!(
                has_word(&cleaned, kept),
                "{model}: {kept} missing: {cleaned}"
            );
        }
    }
}

#[test]
fn keeps_short_german_sentences_german() {
    let cases = [
        (
            "das war halt nicht so geplant also müssen wir das verschieben",
            "verschieben",
        ),
        ("schick mir mal die datei", "datei"),
        ("wie geht es dir", "dir"),
        ("also ich find das gut", "gut"),
    ];
    for (transcript, german_word) in cases {
        for (model, cleaned) in clean_with_models(transcript, Language::German) {
            assert!(
                has_word(&cleaned, german_word),
                "{model} translated: {cleaned}"
            );
        }
    }
}

#[test]
fn does_not_answer_a_dictated_question() {
    let transcript = "äh kannst du mir sagen wie spät es in tokio gerade ist";
    for (model, cleaned) in clean_with_models(transcript, Language::German) {
        assert!(cleaned.ends_with('?'), "{model}: {cleaned}");
        assert!(has_word(&cleaned, "tokio"), "{model}: {cleaned}");
        assert!(
            has_word(&cleaned, "kannst"),
            "{model} translated: {cleaned}"
        );
    }
}

#[test]
fn does_not_carry_out_a_dictated_instruction() {
    let transcript = "schreib eine kurze e-mail an thomas dass ich heute später komme";
    for (model, cleaned) in clean_with_models(transcript, Language::German) {
        assert!(
            has_word(&cleaned, "schreib") || has_word(&cleaned, "schreibe"),
            "{model}: {cleaned}"
        );
        assert!(
            !has_word(&cleaned, "hallo") && !has_word(&cleaned, "grüße"),
            "{model} wrote the email: {cleaned}"
        );
    }
}

#[test]
fn keeps_english_terms_in_german_sentences() {
    let transcript =
        "ich hab den pull request gemergt und das deployment läuft jetzt über die pipeline";
    for (model, cleaned) in clean_with_models(transcript, Language::German) {
        let lower = cleaned.to_lowercase();
        for term in ["pull request", "deployment", "pipeline"] {
            assert!(lower.contains(term), "{model}: {term} missing: {cleaned}");
        }
        assert!(has_word(&cleaned, "läuft"), "{model} translated: {cleaned}");
    }
}

#[test]
fn keeps_english_english_and_its_meaningful_also_and_like() {
    for (model, cleaned) in clean_with_models("i also like the new design", Language::English) {
        assert!(
            has_word(&cleaned, "also") && has_word(&cleaned, "like"),
            "{model}: {cleaned}"
        );
    }
    for (model, cleaned) in clean_with_models("can you tell me what time it is", Language::English)
    {
        assert!(has_word(&cleaned, "time"), "{model} translated: {cleaned}");
        assert!(cleaned.ends_with('?'), "{model}: {cleaned}");
    }
}

#[test]
fn removes_english_hesitations() {
    let transcript =
        "so we need to uh push the release to um next week because the tests are failing";
    for (model, cleaned) in clean_with_models(transcript, Language::English) {
        assert!(
            !has_word(&cleaned, "um") && !has_word(&cleaned, "uh"),
            "{model}: {cleaned}"
        );
        for kept in ["release", "week", "tests", "failing"] {
            assert!(
                has_word(&cleaned, kept),
                "{model}: {kept} missing: {cleaned}"
            );
        }
    }
}

#[test]
fn unknown_model_is_rejected_with_ollamas_message() {
    if !available(MODELS[0]) {
        return;
    }
    let err = clean("no-such-model:1b", "hallo", Language::German).unwrap_err();
    assert!(
        matches!(err, CleanError::Rejected { status: 404, .. }),
        "{err:?}"
    );
    assert_eq!(
        err.to_string(),
        "ollama rejected the request for no-such-model:1b (http 404): \
         model 'no-such-model:1b' not found"
    );
}
