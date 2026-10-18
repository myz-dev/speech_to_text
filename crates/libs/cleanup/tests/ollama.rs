//! `clean` against a local Ollama server, once per model in [`MODELS`]. Models that are
//! not pulled are skipped with a note; without Ollama every test passes with a note.
//! Outputs are deterministic (temperature 0), but asserted loosely so that prompt
//! wording can change without rewriting the tests.

use cleanup::{CleanError, Language, OLLAMA_URL, clean, load};

/// The model the app and the prototype use by default.
const MODELS: [&str; 1] = ["ministral-3:8b"];

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

/// The output is pasted into other apps, where Markdown shows up as literal characters.
fn assert_plain_text(model: &str, cleaned: &str) {
    assert!(
        !cleaned.contains("**") && !cleaned.contains('`'),
        "{model} wrote Markdown: {cleaned}"
    );
}

fn digits(text: &str) -> String {
    text.chars().filter(char::is_ascii_digit).collect()
}

#[test]
fn keeps_only_the_corrected_version_of_a_self_correction() {
    let transcript = "können wir das meeting auf dienstag um drei verschieben nein warte \
                      dienstag geht nicht lieber mittwoch um drei";
    for (model, cleaned) in clean_with_models(transcript, Language::German) {
        assert!(!has_word(&cleaned, "dienstag"), "{model}: {cleaned}");
        assert!(has_word(&cleaned, "mittwoch"), "{model}: {cleaned}");
    }
}

#[test]
fn writes_a_spoken_enumeration_as_a_numbered_list() {
    let transcript = "für morgen habe ich drei aufgaben erstens die rechnung an die firma \
                      schneider schicken zweitens den zahnarzttermin absagen und drittens \
                      die präsentation fertig machen";
    for (model, cleaned) in clean_with_models(transcript, Language::German) {
        let items: Vec<&str> = cleaned
            .lines()
            .filter(|line| line.trim_start().starts_with(['1', '2', '3']))
            .collect();
        assert_eq!(items.len(), 3, "{model}: {cleaned}");
        assert!(items[2].contains("Präsentation"), "{model}: {cleaned}");
        assert_plain_text(model, &cleaned);
    }
}

#[test]
fn writes_amounts_the_german_way_without_changing_digits() {
    let transcript = "die rechnung über 1250 euro und 80 cent ist am 3. november fällig \
                      ruf mich bei fragen unter 0171 234 5678 an";
    for (model, cleaned) in clean_with_models(transcript, Language::German) {
        assert!(cleaned.contains("1.250,80"), "{model}: {cleaned}");
        assert!(
            digits(&cleaned).contains("01712345678"),
            "{model}: {cleaned}"
        );
        assert_plain_text(model, &cleaned);
    }
}

#[test]
fn writes_a_spoken_email_address() {
    let transcript = "seine e-mail-adresse ist t punkt schmid at kanzlei minus berg punkt de";
    for (model, cleaned) in clean_with_models(transcript, Language::German) {
        assert!(
            cleaned.contains("t.schmid@kanzlei-berg.de"),
            "{model}: {cleaned}"
        );
        assert_plain_text(model, &cleaned);
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

#[test]
fn loads_a_model_without_generating() {
    for model in MODELS.into_iter().filter(|model| available(model)) {
        load(model).unwrap();
    }
}

#[test]
fn loading_an_unknown_model_is_rejected() {
    if !available(MODELS[0]) {
        return;
    }
    let err = load("no-such-model:1b").unwrap_err();
    assert!(
        matches!(err, CleanError::Rejected { status: 404, .. }),
        "{err:?}"
    );
}
