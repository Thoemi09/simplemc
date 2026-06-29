//! Checkpoint and restart helpers.
//!
//! The MVP format is a versioned JSON envelope around any serde-serializable payload. Simulation
//! code decides what belongs in the payload: typically state, RNG, kernel/update set, measurements,
//! and any user metadata needed to resume.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const CHECKPOINT_VERSION: u32 = 1;

pub type Result<T> = std::result::Result<T, IoError>;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported checkpoint version {found}, expected {expected}")]
    UnsupportedVersion { expected: u32, found: u32 },
}

/// Versioned checkpoint envelope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint<T> {
    pub version: u32,
    pub payload: T,
}

impl<T> Checkpoint<T> {
    pub fn new(payload: T) -> Self {
        Self {
            version: CHECKPOINT_VERSION,
            payload,
        }
    }

    pub fn payload(&self) -> &T {
        &self.payload
    }

    pub fn payload_mut(&mut self) -> &mut T {
        &mut self.payload
    }

    pub fn into_payload(self) -> T {
        self.payload
    }

    pub fn validate_version(&self) -> Result<()> {
        validate_version(self.version)
    }
}

pub fn to_json_string<T: Serialize>(checkpoint: &Checkpoint<T>) -> Result<String> {
    Ok(serde_json::to_string_pretty(checkpoint)?)
}

pub fn to_json_vec<T: Serialize>(checkpoint: &Checkpoint<T>) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec_pretty(checkpoint)?)
}

pub fn from_json_str<T: DeserializeOwned>(json: &str) -> Result<Checkpoint<T>> {
    let checkpoint: Checkpoint<T> = serde_json::from_str(json)?;
    checkpoint.validate_version()?;
    Ok(checkpoint)
}

pub fn from_json_slice<T: DeserializeOwned>(json: &[u8]) -> Result<Checkpoint<T>> {
    let checkpoint: Checkpoint<T> = serde_json::from_slice(json)?;
    checkpoint.validate_version()?;
    Ok(checkpoint)
}

pub fn save_json<T: Serialize>(path: impl AsRef<Path>, checkpoint: &Checkpoint<T>) -> Result<()> {
    fs::write(path, to_json_vec(checkpoint)?)?;
    Ok(())
}

pub fn load_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Checkpoint<T>> {
    from_json_slice(&fs::read(path)?)
}

/// Save a checkpoint by writing a temporary sibling file and renaming it into place.
pub fn save_json_atomic<T: Serialize>(
    path: impl AsRef<Path>,
    checkpoint: &Checkpoint<T>,
) -> Result<()> {
    let path = path.as_ref();
    let tmp_path = temporary_path(path);

    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(&to_json_vec(checkpoint)?)?;
        file.sync_all()?;
    }

    fs::rename(&tmp_path, path)?;
    Ok(())
}

pub fn save_payload_json<T: Serialize>(path: impl AsRef<Path>, payload: &T) -> Result<()> {
    save_json(path, &Checkpoint::new(payload))
}

pub fn load_payload_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    Ok(load_json(path)?.into_payload())
}

fn validate_version(version: u32) -> Result<()> {
    if version != CHECKPOINT_VERSION {
        return Err(IoError::UnsupportedVersion {
            expected: CHECKPOINT_VERSION,
            found: version,
        });
    }
    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut tmp_path = path.to_path_buf();
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map_or_else(|| "tmp".to_string(), |extension| format!("{extension}.tmp"));
    tmp_path.set_extension(extension);
    tmp_path
}
