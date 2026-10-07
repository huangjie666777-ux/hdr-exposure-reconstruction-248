//! hdr_fusion248 — HDR radiance reconstruction from bracketed 8-bit PNG exposures.
//!
//! The crate is split across cooperating modules:
//!
//! - [`config`]: JSON input description and validation.
//! - [`io`]: PNG decoding and atomic PFM / mask-PNG / report-JSON writing.
//! - [`response`]: Debevec & Malik logarithmic inverse-response recovery.
//! - [`fusion`]: per-pixel weighted radiance fusion and the validity mask.

pub mod config;
pub mod fusion;
pub mod io;
pub mod response;

use std::fmt;

/// Crate-wide error type.
#[derive(Debug)]
pub enum Error {
    /// A semantic/validation failure with a human readable message.
    Msg(String),
    /// An underlying I/O failure.
    Io(std::io::Error),
}

impl Error {
    pub fn msg<S: Into<String>>(s: S) -> Self {
        Error::Msg(s.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Msg(s) => f.write_str(s),
            Error::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<String> for Error {
    fn from(e: String) -> Self {
        Error::Msg(e)
    }
}

impl From<&str> for Error {
    fn from(e: &str) -> Self {
        Error::Msg(e.to_string())
    }
}

/// Convenience `Result` alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A decoded 8-bit RGB exposure image plus its exposure time in seconds.
#[derive(Clone, Debug)]
pub struct ExposureImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row-major RGB triples, length `3 * width * height`, values `0..=255`.
    pub rgb: Vec<u8>,
    /// Exposure duration in seconds (finite, strictly positive).
    pub exposure_seconds: f64,
    /// The path it was loaded from, retained for provenance reporting.
    pub source: String,
}

/// The reconstructed linear radiance result.
#[derive(Clone, Debug)]
pub struct RadianceMap {
    pub width: u32,
    pub height: u32,
    /// Row-major RGB triples of relative linear radiance (not normalized/clamped).
    pub rgb: Vec<f32>,
    /// Row-major RGB validity mask; each byte is `255` (valid) or `0` (invalid).
    pub valid: Vec<u8>,
}
