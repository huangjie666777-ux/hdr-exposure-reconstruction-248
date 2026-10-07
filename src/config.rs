//! Input description: a JSON file listing bracketed PNG exposures.

use serde::Deserialize;
use std::path::Path;

use crate::{Error, ExposureImage, Result};

/// Maximum number of exposures accepted by the engine.
pub const MAX_EXPOSURES: usize = 8;
/// Minimum number of exposures accepted by the engine.
pub const MIN_EXPOSURES: usize = 3;
/// Maximum image edge length in pixels.
pub const MAX_EDGE: u32 = 512;

/// One entry of the input JSON: an 8-bit RGB PNG and its exposure in seconds.
#[derive(Debug, Clone, Deserialize)]
pub struct ExposureEntry {
    /// Path to the 8-bit RGB (or RGBA) PNG; relative paths resolve against
    /// the directory of the input JSON file.
    pub path: String,
    /// Exposure duration in seconds (finite, strictly positive).
    #[serde(alias = "exposure")]
    pub exposure_seconds: f64,
}

/// Root object of the input JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct InputManifest {
    /// Three to eight bracketed exposures of the same static scene.
    pub exposures: Vec<ExposureEntry>,
    /// Optional output paths. Missing entries use defaults next to the JSON.
    #[serde(default)]
    pub output: OutputPaths,
}

/// Optional output overrides.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OutputPaths {
    /// RGB 32-bit float PFM radiance map.
    #[serde(default)]
    pub pfm: Option<String>,
    /// RGB validity mask PNG (255 valid / 0 invalid per channel).
    #[serde(default)]
    pub mask: Option<String>,
    /// JSON report with response curves, samples, residuals and provenance.
    #[serde(default)]
    pub report: Option<String>,
}

/// Validate the manifest and load every referenced PNG.
///
/// `manifest_path` is the path of the input JSON, used to resolve relative
/// image paths. Original image files are only read, never modified.
pub fn load_manifest(manifest_path: &Path) -> Result<(InputManifest, Vec<ExposureImage>)> {
    let text = std::fs::read_to_string(manifest_path)?;
    let manifest: InputManifest = serde_json::from_str(&text)
        .map_err(|e| Error::msg(format!("invalid input JSON {manifest_path:?}: {e}")))?;
    validate_manifest(&manifest)?;

    let base_dir = manifest_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| Path::new(".").to_path_buf());

    let mut images = Vec::with_capacity(manifest.exposures.len());
    for entry in &manifest.exposures {
        let image_path = resolve_path(&base_dir, &entry.path);
        let image = crate::io::read_png_rgb8(&image_path)?;
        images.push(ExposureImage {
            width: image.width,
            height: image.height,
            rgb: image.rgb,
            exposure_seconds: entry.exposure_seconds,
            source: entry.path.clone(),
        });
    }
    validate_images(&images)?;
    Ok((manifest, images))
}

/// Resolve a JSON-provided path: absolute paths win, relative ones join the
/// JSON's directory.
fn resolve_path(base_dir: &Path, path: &str) -> std::path::PathBuf {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        base_dir.join(candidate)
    }
}

/// Validate exposure count, finiteness/positivity and value diversity.
pub fn validate_manifest(manifest: &InputManifest) -> Result<()> {
    let n = manifest.exposures.len();
    if !(MIN_EXPOSURES..=MAX_EXPOSURES).contains(&n) {
        return Err(Error::msg(format!(
            "expected {MIN_EXPOSURES}..={MAX_EXPOSURES} exposures, found {n}"
        )));
    }
    for (i, entry) in manifest.exposures.iter().enumerate() {
        let t = entry.exposure_seconds;
        if !t.is_finite() || t <= 0.0 {
            return Err(Error::msg(format!(
                "exposure {i} ({}) must be a finite positive number of seconds, got {t}",
                entry.path
            )));
        }
    }
    let distinct = {
        let mut times: Vec<f64> = manifest
            .exposures
            .iter()
            .map(|e| e.exposure_seconds)
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON * a.abs().max(*b).max(1.0));
        times.len()
    };
    if distinct < 2 {
        return Err(Error::msg(
            "at least two distinct exposure times are required to recover a response curve",
        ));
    }
    Ok(())
}

/// Validate common dimensions and the 512px edge limit.
pub fn validate_images(images: &[ExposureImage]) -> Result<()> {
    let first = &images[0];
    if first.width == 0 || first.height == 0 {
        return Err(Error::msg("images must have non-zero dimensions"));
    }
    if first.width > MAX_EDGE || first.height > MAX_EDGE {
        return Err(Error::msg(format!(
            "image edge exceeds the {MAX_EDGE}px limit ({}x{})",
            first.width, first.height
        )));
    }
    for (i, image) in images.iter().enumerate().skip(1) {
        if image.width != first.width || image.height != first.height {
            return Err(Error::msg(format!(
                "image {} ({}) is {}x{}, expected {}x{}",
                i,
                image.source,
                image.width,
                image.height,
                first.width,
                first.height
            )));
        }
    }
    Ok(())
}
