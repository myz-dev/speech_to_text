# Development Rules

## Workflow: human-sized changes
- One concern per change. Aim for ≤ ~300 lines of non-test diff, small enough for a
  human to review in one sitting. Split larger work into steps that each build and pass.
- Test code is exempt from the size limit, but must stay skimmable: test names state the
  behavior (`rejects_stereo_without_downmix`), so reviewing starts with reading names.
- After each change: commit it on its branch, write the review notes (below), summarize
  the change, and stop for the human's review.
- Review findings become follow-up commits on the same branch, never amends, so the
  review stays in the history. Type them by what they change (`test:`, `docs:`,
  `refactor:`); `fix:` is for bugs that reached `main`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org)
  (`feat(recording): …`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`; scope = crate
  name), so a changelog can be generated with git-cliff.
- Never commit to `main` directly: one short-lived branch per change (`m1/whisper`),
  merged with `git merge --no-ff` only after the human approves the review.

## Review notes
For every change, write `.review/NOTES.md` (gitignored, overwritten per change). It doubles
as the prompt for an independent review agent, so write it for a skeptical reader:
- **Goal:** what was added and to what end (2–3 sentences)
- **Diff:** how to see it (`git show <sha>` for the change, `git diff main...HEAD` for
  the branch including review follow-ups)
- **Design decisions:** each choice with the alternative that was rejected and why
- **Claims to verify:** concrete statements the reviewer should check, not take on trust
  (e.g. "resampling preserves duration within 1 sample")
- **Risky spots:** `file:line` locations most likely to hide bugs
- **Verify with:** commands to run
- **Out of scope:** what was deliberately not done
End with: *"Review the diff for correctness and simplicity against AGENTS.md. Verify the
claims above independently; do not assume they hold."*

## Principles
- Simple over clever. Code is read far more often than written; optimize for a reader
  scanning it once.
- Prefer free functions whose arguments carry all relevant state. Structs hold data or
  expensive resources (a loaded model), not behavior hierarchies.
- Methods on such structs are fine when they take `&self` and every other input is an
  argument; avoid `&mut self` methods that carry hidden state from one call to the next.
- No factories, builders-for-the-sake-of-it, or layers of indirection. A code path
  should be followable with "go to definition".

## Documentation
- Docs and comments describe the system as it is now, not how it got there. No
  "initially we…", "changed to…", "now uses…", "turned out that…".
- History belongs in commit messages; that's where "why did this change" is answered.
- Exception: a pitfall worth not repeating. State it as a present-tense constraint with
  its reason: "Input must be 16 kHz; Whisper produces garbage on other rates without
  an error", not "we first passed 44.1 kHz and got garbage".
- FINDINGS.md is the one place for experiment results, but each entry states the
  outcome (setup, numbers, conclusion), not the journey to it.

## Abstractions
- Closed set of variants (STT engines, LLM backends) → enum + `match`, not a trait.
- Introduce a trait only for open extension or a dependency that can't run in tests.
- No generics unless there are ≥2 concrete uses today.

## Errors
- Libs: `thiserror`. Enum variants when the cause is identified by the variant
  (`MissingPermission`, `CorruptFile { path }`); a struct when details matter more
  than kind. One error type per crate or per major operation, not one global enum.
- Attach context where it is known: carry `path`, `model`, etc. in the variant and
  keep the underlying error in `#[source]`. Avoid bare `#[from]` that loses context.
- Display messages: lowercase, no trailing period, don't repeat the source's message.
- Apps: `anyhow` with `.context()` / `.with_context()`.
- No `unwrap`/`expect` in lib code except for true invariants, with a comment why.

## Testing
- `cargo nextest run`; pure logic gets unit tests with real data, no mocks.
- Tests needing model weights skip when `models/` is missing.

## Before finishing a change
- `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo nextest run`, all clean.
- New external dependencies go into `[workspace.dependencies]`.
- Results/learnings go into FINDINGS.md: no dates or times.
