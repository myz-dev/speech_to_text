//! User facing settings.
//! The settings are kept in the app state; the window only shows them, and every change
//! is emitted to it as the `settings` event.
//! The worker reads them when a dictation starts, so a change applies from the next one.
//! Every change is saved to `settings.json` in the app's config directory, which is read
//! again when the app starts.

use std::{
    io,
    path::{Path, PathBuf},
    sync::{Mutex, PoisonError},
};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

/// Why the settings file could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("cannot read the settings file '{path}'")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("the settings file '{path}' is invalid")]
    Invalid {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("cannot create the settings directory '{path}'")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("cannot write the settings file '{path}'")]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// Mirrored by `Settings` in `ui/src/main.ts`.
///
/// A field missing from the settings file takes its default, so a file saved before a
/// field was added still loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    /// Paste the transcript into the focused app. Otherwise it is only put on the
    /// clipboard, which needs no Accessibility permission.
    pub paste: bool,
    /// Clean up the transcript with a local LLM served by Ollama.
    pub cleanup: bool,
    /// The language dictated in. Only the cleanup needs it: Parakeet transcribes any
    /// language without being told, but does not report which one it heard.
    pub language: Language,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            paste: false,
            cleanup: false,
            language: Language::De,
        }
    }
}

impl Settings {
    /// Reads the settings saved at `path`, or `None` if none were saved yet.
    pub fn load(path: &Path) -> Result<Option<Self>, SettingsError> {
        let content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(SettingsError::Read {
                    path: path.into(),
                    source,
                });
            }
        };
        serde_json::from_str(&content)
            .map(Some)
            .map_err(|source| SettingsError::Invalid {
                path: path.into(),
                source,
            })
    }

    /// Saves the settings to `path`, creating its directory if needed.
    ///
    /// Writes a temporary file next to it and renames it over `path`, so if the app
    /// crashes while writing, the previous file stays intact instead of a truncated one.
    /// Errors name `path`, even when writing the temporary file failed.
    pub fn save(&self, path: &Path) -> Result<(), SettingsError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|source| SettingsError::CreateDir {
                path: dir.into(),
                source,
            })?;
        }
        // Booleans and a unit enum always serialize.
        let content = serde_json::to_string_pretty(self).expect("settings serialize to JSON");
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, content)
            .and_then(|()| std::fs::rename(&temporary, path))
            .map_err(|source| SettingsError::Write {
                path: path.into(),
                source,
            })
    }
}

/// The settings file in the app's config directory
/// (`~/Library/Application Support/<identifier>` on macOS).
pub fn path(app: &AppHandle) -> tauri::Result<PathBuf> {
    Ok(app.path().app_config_dir()?.join("settings.json"))
}

/// A language the cleanup supports, serialized as its ISO 639-1 code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    De,
    En,
}

impl From<Language> for cleanup::Language {
    fn from(language: Language) -> Self {
        match language {
            Language::De => cleanup::Language::German,
            Language::En => cleanup::Language::English,
        }
    }
}

/// The current settings and the file they are saved to, kept in the app state.
pub struct Current {
    path: PathBuf,
    settings: Mutex<Settings>,
}

impl Current {
    /// `settings`, saved to `path` on every change.
    pub fn new(path: PathBuf, settings: Settings) -> Self {
        Self {
            path,
            settings: Mutex::new(settings),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Settings> {
        // The lock only guards a value replaced in one step, so a panic while holding it
        // cannot leave it inconsistent; ignore poisoning.
        self.settings.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The current settings, for the worker.
pub fn current(app: &AppHandle) -> Settings {
    *app.state::<Current>().lock()
}

/// The current settings, for the window.
#[tauri::command]
pub fn settings(current: tauri::State<Current>) -> Settings {
    *current.lock()
}

/// Saves the settings, replaces the current ones and emits them as the `settings` event.
///
/// Saved first: if saving fails, the current settings stay as they were and the error,
/// with its cause, goes to the window, which then shows the current settings again.
/// The lock is held while saving, so the file always holds the latest settings.
///
/// Stored before it is emitted, like [`crate::status::publish`]: a window that reads
/// [`settings`] after an event never gets a value older than that event.
#[tauri::command]
pub fn set_settings(
    settings: Settings,
    app: AppHandle,
    current: tauri::State<Current>,
) -> Result<(), String> {
    let mut stored = current.lock();
    settings
        .save(&current.path)
        .map_err(|error| format!("{:#}", anyhow::Error::new(error)))?;
    *stored = settings;
    drop(stored);
    if let Err(error) = app.emit("settings", settings) {
        eprintln!("cannot emit the settings: {error}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings_path() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config").join("settings.json");
        (dir, path)
    }

    #[test]
    fn saved_settings_load_back_unchanged() {
        let (_dir, path) = settings_path();
        let settings = Settings {
            paste: true,
            cleanup: true,
            language: Language::En,
        };
        settings.save(&path).unwrap();
        assert_eq!(Settings::load(&path).unwrap(), Some(settings));
    }

    #[test]
    fn save_creates_the_missing_directory() {
        let (_dir, path) = settings_path();
        Settings::default().save(&path).unwrap();
        assert!(path.is_file());
    }

    #[test]
    fn save_replaces_the_previous_file_and_leaves_no_temporary_file() {
        let (_dir, path) = settings_path();
        Settings::default().save(&path).unwrap();
        let changed = Settings {
            paste: true,
            ..Settings::default()
        };
        changed.save(&path).unwrap();
        assert_eq!(Settings::load(&path).unwrap(), Some(changed));
        let files = std::fs::read_dir(path.parent().unwrap()).unwrap().count();
        assert_eq!(files, 1);
    }

    #[test]
    fn missing_file_loads_as_none() {
        let (_dir, path) = settings_path();
        assert_eq!(Settings::load(&path).unwrap(), None);
    }

    #[test]
    fn invalid_file_is_an_error_naming_the_path() {
        let (_dir, path) = settings_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{ not json").unwrap();
        let error = Settings::load(&path).unwrap_err();
        assert!(matches!(error, SettingsError::Invalid { .. }));
        assert!(error.to_string().contains("settings.json"));
    }

    #[test]
    fn missing_fields_take_their_defaults() {
        let (_dir, path) = settings_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, r#"{ "paste": true }"#).unwrap();
        let expected = Settings {
            paste: true,
            ..Settings::default()
        };
        assert_eq!(Settings::load(&path).unwrap(), Some(expected));
    }

    #[test]
    fn file_format_uses_the_ui_field_names() {
        let (_dir, path) = settings_path();
        Settings::default().save(&path).unwrap();
        let saved: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let expected = serde_json::json!({ "paste": false, "cleanup": false, "language": "de" });
        assert_eq!(saved, expected);
    }
}
