//! The process's ONNX Runtime environment and the sessions built on it, shared by every
//! crate that runs an ONNX graph (Parakeet in `transcription`, Silero VAD in
//! `recording`).

use std::path::Path;
use std::sync::{Mutex, PoisonError};

use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::AutoDevicePolicy;

/// The process's ONNX Runtime environment, shared by all sessions. ort allows only one
/// environment per process, and never a second one, even after the first is dropped.
static ENVIRONMENT: Mutex<Option<Environment>> = Mutex::new(None);

/// Returns the shared environment, creating it on first use.
fn environment() -> Result<Environment, ort::Error> {
    // The lock only guards an `Option` that is written in one step, so a panic while
    // holding it cannot leave it inconsistent; ignore poisoning.
    let mut shared = ENVIRONMENT.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(environment) = shared.as_ref() {
        return Ok(environment.clone());
    }
    let environment = ort::init().build()?;
    *shared = Some(environment.clone());
    Ok(environment)
}

/// Loads the ONNX graph at `path` into a CPU session of the shared environment.
///
/// ort's default device policy hands parts of a graph to CoreML on macOS, even without
/// its `coreml` feature. For the Parakeet encoder that makes session creation ~4× slower,
/// encoding ~1.8× slower, and changes the output; `PreferCPU` keeps the whole graph on
/// the CPU.
pub fn session(path: &Path) -> Result<Session, ort::Error> {
    Session::builder(&environment()?)?
        .with_auto_device(AutoDevicePolicy::PreferCPU)?
        .commit_from_file(path)
}
