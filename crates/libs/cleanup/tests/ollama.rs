//! `clean` against a local Ollama server. Tests skip (pass with a note) when Ollama is
//! not running or the model is not pulled. Outputs are deterministic (temperature 0),
//! but asserted loosely so that prompt wording can change without rewriting the tests.

use cleanup::{CleanError, OLLAMA_URL, clean};

const MODEL: &str = "gemma3:4b";

/// True if Ollama runs and has `model`; prints why a test is skipped otherwise.
fn available(model: &str) -> bool {
    let tags = ureq::get(format!("{OLLAMA_URL}/api/tags"))
        .call()
        .and_then(|mut response| response.body_mut().read_to_string());
    match tags {
        Ok(tags) if tags.contains(&format!("\"name\":\"{model}\"")) => true,
        Ok(_) => {
            eprintln!("skipping: {model} not pulled (`ollama pull {model}`)");
            false
        }
        Err(_) => {
            eprintln!("skipping: ollama not reachable at {OLLAMA_URL}");
            false
        }
    }
}

fn clean_with_model(transcript: &str) -> Option<String> {
    if !available(MODEL) {
        return None;
    }
    let cleaned = clean(MODEL, transcript).unwrap();
    eprintln!("{transcript:?}\n→ {cleaned:?}");
    Some(cleaned)
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[test]
fn removes_german_fillers_and_keeps_german() {
    let Some(cleaned) = clean_with_model(
        "äh also ich wollte ähm nur kurz sagen dass das meeting morgen äh auf drei uhr verschoben ist",
    ) else {
        return;
    };
    let words = words(&cleaned);
    assert!(!words.iter().any(|w| w == "äh" || w == "ähm"), "{cleaned}");
    for kept in ["meeting", "morgen", "verschoben"] {
        assert!(words.iter().any(|w| w == kept), "{kept} missing: {cleaned}");
    }
}

#[test]
fn does_not_answer_a_dictated_question() {
    let Some(cleaned) = clean_with_model("kannst du mir sagen wie spät es in tokio gerade ist")
    else {
        return;
    };
    assert!(cleaned.ends_with('?'), "{cleaned}");
    assert!(words(&cleaned).iter().any(|w| w == "tokio"), "{cleaned}");
    assert!(
        cleaned.len() < 80,
        "answered instead of cleaning: {cleaned}"
    );
}

#[test]
fn does_not_carry_out_a_dictated_instruction() {
    let Some(cleaned) =
        clean_with_model("schreib eine kurze e-mail an thomas dass ich heute später komme")
    else {
        return;
    };
    let words = words(&cleaned);
    assert!(
        words.iter().any(|w| w == "schreib" || w == "schreibe"),
        "{cleaned}"
    );
    assert!(
        !words.iter().any(|w| w == "hallo" || w == "grüße"),
        "wrote the email: {cleaned}"
    );
    assert!(cleaned.len() < 100, "wrote the email: {cleaned}");
}

#[test]
fn keeps_english_terms_in_german_sentences() {
    let Some(cleaned) = clean_with_model(
        "ich hab den pull request gemergt und das deployment läuft jetzt über die pipeline",
    ) else {
        return;
    };
    let lower = cleaned.to_lowercase();
    for term in ["pull request", "deployment", "pipeline"] {
        assert!(lower.contains(term), "{term} missing: {cleaned}");
    }
    assert!(
        words(&cleaned).iter().any(|w| w == "läuft"),
        "translated: {cleaned}"
    );
}

#[test]
fn keeps_english_english() {
    let Some(cleaned) = clean_with_model(
        "um so basically we need to uh push the release to next week because the tests are failing",
    ) else {
        return;
    };
    let words = words(&cleaned);
    assert!(!words.iter().any(|w| w == "um" || w == "uh"), "{cleaned}");
    for kept in ["release", "week", "tests", "failing"] {
        assert!(words.iter().any(|w| w == kept), "{kept} missing: {cleaned}");
    }
}

#[test]
fn unknown_model_is_rejected_with_ollamas_message() {
    if !available(MODEL) {
        return;
    }
    let err = clean("no-such-model:1b", "hallo").unwrap_err();
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
