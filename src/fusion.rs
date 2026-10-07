//! Per-pixel weighted radiance fusion.
//!
//! For every pixel and channel the per-exposure log-radiance estimates
//! `g[z] - ln(dt)` are combined with the same triangular weight used during
//! response recovery, exponentiated, and written as relative linear radiance.
//! Values are neither normalized to a maximum nor clamped to `[0, 1]`.

use crate::response::{weight, ChannelRecovery};
use crate::{Error, ExposureImage, RadianceMap, Result};

/// Fuse the exposure stack into a [`RadianceMap`] using recovered curves.
pub fn fuse(
    images: &[ExposureImage],
    channels: &[ChannelRecovery; 3],
) -> Result<RadianceMap> {
    let width = images[0].width;
    let height = images[0].height;
    let pixel_count = (width as usize) * (height as usize);
    let log_exposures: Vec<f64> = images
        .iter()
        .map(|img| img.exposure_seconds.ln())
        .collect();

    let mut rgb = vec![0.0f32; pixel_count * 3];
    let mut valid = vec![0u8; pixel_count * 3];

    for channel in 0..3 {
        let g = &channels[channel].g;
        for pixel in 0..pixel_count {
            let base = pixel * 3 + channel;
            let mut weighted_sum = 0.0f64;
            let mut weight_sum = 0.0f64;
            for (j, img) in images.iter().enumerate() {
                let z = img.rgb[base];
                let w = weight(z);
                if w <= 0.0 {
                    continue;
                }
                weighted_sum += w * (g[z as usize] - log_exposures[j]);
                weight_sum += w;
            }
            if weight_sum > 0.0 {
                let log_radiance = weighted_sum / weight_sum;
                let radiance = log_radiance.exp();
                if !radiance.is_finite() {
                    let (x, y) = ((pixel % width as usize) as u32, pixel as u32 / width);
                    return Err(Error::msg(format!(
                        "non-finite radiance at pixel ({x},{y}) channel {channel}: {radiance}"
                    )));
                }
                rgb[base] = radiance as f32;
                if !rgb[base].is_finite() {
                    return Err(Error::msg(format!(
                        "radiance does not fit in f32 at pixel channel {channel}: {radiance}"
                    )));
                }
                valid[base] = 255;
            }
            // No usable exposure: radiance stays 0 and the channel is invalid.
        }
    }

    Ok(RadianceMap {
        width,
        height,
        rgb,
        valid,
    })
}

/// Count valid channels per kind for quick reporting.
pub fn validity_counts(valid: &[u8]) -> ValiditySummary {
    let mut channels_valid = 0usize;
    let mut pixels_all_valid = 0usize;
    let pixel_count = valid.len() / 3;
    for p in 0..pixel_count {
        let mut all = true;
        for c in 0..3 {
            if valid[p * 3 + c] == 255 {
                channels_valid += 1;
            } else {
                all = false;
            }
        }
        if all {
            pixels_all_valid += 1;
        }
    }
    ValiditySummary {
        pixels: pixel_count,
        channels_valid,
        pixels_all_valid,
    }
}

/// Aggregate validity statistics embedded in the report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ValiditySummary {
    pub pixels: usize,
    pub channels_valid: usize,
    pub pixels_all_valid: usize,
}
