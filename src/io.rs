//! Image decoding and atomic PFM / PNG / JSON output.
//!
//! Outputs are fully rendered in memory first, written to sibling temporary
//! files, and renamed into place only once every deliverable is ready, so a
//! failure never leaves a partial delivery on disk.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use png::ColorType;

use crate::{Error, RadianceMap, Result};

/// Decoded 8-bit image (alpha, if present, is dropped).
pub struct Rgb8Image {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// Decode an 8-bit RGB or RGBA PNG. Other formats are rejected explicitly.
pub fn read_png_rgb8(path: &Path) -> Result<Rgb8Image> {
    let bytes = std::fs::read(path)
        .map_err(|e| Error::msg(format!("cannot read image {path:?}: {e}")))?;
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| Error::msg(format!("failed to decode PNG {path:?}: {e}")))?;
    let info = reader.info().clone();
    if info.bit_depth != png::BitDepth::Eight {
        return Err(Error::msg(format!(
            "{path:?}: only 8-bit PNGs are supported (found {:?})",
            info.bit_depth
        )));
    }
    let mut raw = vec![0u8; reader.output_buffer_size()];
    let frame = reader
        .next_frame(&mut raw)
        .map_err(|e| Error::msg(format!("failed to read PNG frame {path:?}: {e}")))?;
    let width = frame.width;
    let height = frame.height;
    let source = &raw[..frame.line_size * frame.height as usize];

    let rgb = match info.color_type {
        ColorType::Rgb => {
            if source.len() != (width as usize) * (height as usize) * 3 {
                return Err(Error::msg(format!("{path:?}: unexpected RGB data length")));
            }
            source.to_vec()
        }
        ColorType::Rgba => source
            .chunks_exact(4)
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect(),
        other => {
            return Err(Error::msg(format!(
                "{path:?}: only 8-bit RGB/RGBA PNGs are supported (found {other:?})"
            )))
        }
    };

    Ok(Rgb8Image {
        width,
        height,
        rgb,
    })
}

/// Encode the per-channel validity mask as an 8-bit RGB PNG.
pub fn encode_mask_png(map: &RadianceMap) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    {
        let mut encoder = png::Encoder::new(
            Cursor::new(&mut buffer),
            map.width,
            map.height,
        );
        encoder.set_color(ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| Error::msg(format!("PNG header failure: {e}")))?;
        writer
            .write_image_data(&map.valid)
            .map_err(|e| Error::msg(format!("PNG write failure: {e}")))?;
    }
    Ok(buffer)
}

/// Encode the radiance map as a color (`PF`) little-endian 32-bit float PFM.
///
/// Rows are emitted bottom-to-top, which is the conventional PFM raster
/// order (matching Netpbm/HDRShop readers).
pub fn encode_pfm(map: &RadianceMap) -> Result<Vec<u8>> {
    let width = map.width as usize;
    let height = map.height as usize;
    let mut out = Vec::with_capacity(64 + map.rgb.len() * 4);
    out.extend_from_slice(b"PF\n");
    out.extend_from_slice(format!("{width} {height}\n").as_bytes());
    // Negative scale means little-endian IEEE-754; magnitude is ignored.
    out.extend_from_slice(b"-1.0\n");
    for row in (0..height).rev() {
        let start = row * width * 3;
        let end = start + width * 3;
        for &value in &map.rgb[start..end] {
            if !value.is_finite() {
                return Err(Error::msg("refusing to write non-finite PFM value"));
            }
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(out)
}

/// A completed deliverable staged beside its final destination.
pub struct StagedFile {
    temp: PathBuf,
    destination: PathBuf,
}

/// Stage raw bytes at `destination.with_extension(<ext>.part-<pid>)`.
pub fn stage_bytes(destination: &Path, bytes: &[u8]) -> Result<StagedFile> {
    let pid = std::process::id();
    let mut file_name = destination
        .file_name()
        .map(|v| v.to_os_string())
        .ok_or_else(|| Error::msg(format!("invalid output path {destination:?}")))?;
    file_name.push(format!(".hdr_fusion248.{pid}.part"));
    let temp = destination.with_file_name(file_name);
    std::fs::write(&temp, bytes)?;
    Ok(StagedFile {
        temp,
        destination: destination.to_path_buf(),
    })
}

impl StagedFile {
    /// Atomically move the staged file to its final name.
    pub fn commit(self) -> Result<()> {
        std::fs::rename(&self.temp, &self.destination)?;
        std::mem::forget(self);
        Ok(())
    }
}

impl Drop for StagedFile {
    fn drop(&mut self) {
        // Only invoked for uncommitted files: removes the staged partial.
        let _ = std::fs::remove_file(&self.temp);
    }
}

/// Write every deliverable: all bytes are supplied up front, all parts are
/// staged first, and renames happen only after every stage succeeded.
pub fn commit_outputs(files: Vec<(PathBuf, Vec<u8>)>) -> Result<()> {
    let mut staged = Vec::with_capacity(files.len());
    for (destination, bytes) in &files {
        staged.push(stage_bytes(destination, bytes)?);
    }
    for entry in staged {
        entry.commit()?;
    }
    Ok(())
}
