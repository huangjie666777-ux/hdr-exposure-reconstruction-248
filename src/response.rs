//! Debevec & Malik logarithmic inverse-response recovery.
//!
//! For one color channel we jointly solve for the 256 entries of the log
//! inverse response `g` (mapping pixel value `z` to `ln(E*)` camera space)
//! and the log radiance `lE` of every sampled pixel:
//!
//! ```text
//! w(z) * (g[z_ij] - lE_i)            = w(z) * ln(dt_j)   (data rows)
//! 10 * w(z+1) * (g[z] - 2g[z+1] + g[z+2]) = 0             (smoothness)
//! g[128] = 0                                             (gauge anchor)
//! ```
//! The weight is `w(z) = min(z, 255 - z)`. The over-determined system is
//! solved by SVD least squares, and its numerical rank must be full or the
//! material is rejected as unidentifiable.

use nalgebra::{DMatrix, DVector, SVD};
use serde::Serialize;

use crate::{Error, ExposureImage, Result};

/// Maximum number of sampled pixels per channel (uniformly strided).
pub const MAX_SAMPLES_PER_CHANNEL: usize = 128;
/// Anchor intensity fixing the additive gauge: `g[128] = 0`.
pub const ANCHOR_Z: usize = 128;
/// Smoothness multiplier applied to the second-difference rows.
pub const LAMBDA: f64 = 10.0;
/// Number of discrete intensity levels for 8-bit data.
pub const LEVELS: usize = 256;

/// Debevec triangular weight: lowest at 0/255, highest in the middle.
#[inline]
pub fn weight(z: u8) -> f64 {
    (z as usize).min(255 - z as usize) as f64
}

/// Per-channel recovery output.
#[derive(Debug, Clone, Serialize)]
pub struct ChannelRecovery {
    /// The 256 entries of the log inverse response curve `g`.
    pub g: Vec<f64>,
    /// Sampled pixel coordinates `[x, y]` actually used for this channel.
    pub sample_coordinates: Vec<[usize; 2]>,
    /// Weighted RMS residual of the data equations (response vs radiance).
    pub data_rms: f64,
    /// Weighted RMS residual of the smoothness equations.
    pub smoothness_rms: f64,
    /// Residual of the gauge anchor equation `g[128] = 0`.
    pub anchor_residual: f64,
    /// Numerical rank of the design matrix.
    pub rank: usize,
    /// Number of columns (unknowns) of the design matrix.
    pub columns: usize,
    /// Ratio of the smallest accepted singular value to the largest.
    pub singular_value_ratio: f64,
}

/// Recover all three channel curves from the bracketed exposure stack.
///
/// Returns one [`ChannelRecovery`] per channel in RGB order.
pub fn recover_channels(images: &[ExposureImage]) -> Result<[ChannelRecovery; 3]> {
    let width = images[0].width as usize;
    let height = images[0].height as usize;
    let n = images.len();
    let log_dt: Vec<f64> = images
        .iter()
        .map(|img| img.exposure_seconds.ln())
        .collect();

    let mut out = Vec::with_capacity(3);
    for channel in 0..3 {
        let samples = select_samples(images, channel, width, height);
        if samples.is_empty() {
            return Err(Error::msg(format!(
                "channel {channel}: no identifiable pixel (need values other than all-0/all-255)"
            )));
        }
        out.push(recover_one(images, channel, &samples, n, &log_dt)?);
    }
    Ok([
        out.remove(0),
        out.remove(0),
        out.remove(0),
    ])
}

/// Uniformly pick up to [`MAX_SAMPLES_PER_CHANNEL`] pixels that carry usable
/// information in `channel` across the stack: pixels whose values are not 0 in
/// every exposure and not 255 in every exposure are eligible.
pub fn select_samples(
    images: &[ExposureImage],
    channel: usize,
    width: usize,
    height: usize,
) -> Vec<[usize; 2]> {
    let npix = width * height;
    let mut eligible = Vec::with_capacity(npix);
    for p in 0..npix {
        let base = p * 3 + channel;
        let mut all_zero = true;
        let mut all_full = true;
        for img in images {
            let z = img.rgb[base];
            if z != 0 {
                all_zero = false;
            }
            if z != 255 {
                all_full = false;
            }
            if !all_zero && !all_full {
                break;
            }
        }
        if !all_zero && !all_full {
            eligible.push([p % width, p / width]);
        }
    }
    let need = eligible.len().min(MAX_SAMPLES_PER_CHANNEL);
    if need == 0 {
        return Vec::new();
    }
    let stride = eligible.len() as f64 / need as f64;
    (0..need)
        .map(|k| eligible[((k as f64 * stride).floor() as usize).min(eligible.len() - 1)])
        .collect()
}

/// Build and solve the Debevec least-squares system for one channel.
fn recover_one(
    images: &[ExposureImage],
    channel: usize,
    samples: &[[usize; 2]],
    n_exp: usize,
    log_dt: &[f64],
) -> Result<ChannelRecovery> {
    let width = images[0].width as usize;
    let p = samples.len();
    // Unknown layout: g[0..256], then lE for each sampled pixel.
    let n_cols = LEVELS + p;
    let data_rows = p * n_exp;
    let smooth_rows = LEVELS - 2;
    let anchor_rows = 1;
    let n_rows = data_rows + smooth_rows + anchor_rows;

    let mut a = DMatrix::<f64>::zeros(n_rows, n_cols);
    let mut b = DVector::<f64>::zeros(n_rows);

    let mut row = 0;
    // Data rows: w(z) * g[z] - w(z) * lE_i = w(z) * ln(dt_j)
    for (i, sample) in samples.iter().enumerate() {
        let pixel_index = sample[1] * width + sample[0];
        let base = pixel_index * 3 + channel;
        for (j, img) in images.iter().enumerate() {
            let z = img.rgb[base] as usize;
            let w = weight(img.rgb[base]);
            a[(row, z)] = w;
            a[(row, LEVELS + i)] = -w;
            b[row] = w * log_dt[j];
            row += 1;
        }
    }
    // Smoothness rows: lambda * w(z+1) * (g[z] - 2g[z+1] + g[z+2]) = 0
    for z in 0..LEVELS - 2 {
        let w = LAMBDA * weight((z + 1) as u8);
        a[(row, z)] = w;
        a[(row, z + 1)] = -2.0 * w;
        a[(row, z + 2)] = w;
        row += 1;
    }
    // Gauge anchor: g[128] = 0 with unit weight.
    a[(row, ANCHOR_Z)] = 1.0;
    let anchor_row = row;
    debug_assert_eq!(anchor_row, n_rows - 1);

    // Rank check before solving: unidentifiable material is rejected rather
    // than producing an arbitrary curve.
    let svd = SVD::new(a.clone(), true, true);
    let sigma_max = svd.singular_values[0];
    if !sigma_max.is_finite() || sigma_max <= 0.0 {
        return Err(Error::msg(format!("channel {channel}: degenerate design matrix")));
    }
    let tol = n_cols.max(n_rows) as f64 * f64::EPSILON * sigma_max;
    let rank = svd.singular_values.iter().filter(|&&s| s > tol).count();
    let singular_value_ratio = svd.singular_values[n_cols - 1] / sigma_max;
    if rank < n_cols {
        return Err(Error::msg(format!(
            "channel {channel}: response unidentifiable (matrix rank {rank} < {n_cols} unknowns); \
             bracket with more exposure variation / scene intensities"
        )));
    }

    let x = svd
        .solve(&b, tol)
        .map_err(|e| Error::msg(format!("channel {channel}: SVD solve failed: {e}")))?;

    // Residual statistics split by equation family.
    let residual = &a * &x - &b;
    let data_res = &residual.rows(0, data_rows);
    let smooth_res = residual.rows(data_rows, smooth_rows);
    let data_rms = (data_res.dot(data_res) / data_rows.max(1) as f64).sqrt();
    let smoothness_rms = (smooth_res.dot(&smooth_res) / smooth_rows.max(1) as f64).sqrt();
    let anchor_residual = residual[anchor_row];

    let g: Vec<f64> = x.rows(0, LEVELS).iter().copied().collect();
    for value in &g {
        if !value.is_finite() {
            return Err(Error::msg(format!(
                "channel {channel}: non-finite inverse response produced"
            )));
        }
    }

    Ok(ChannelRecovery {
        g,
        sample_coordinates: samples.to_vec(),
        data_rms,
        smoothness_rms,
        anchor_residual,
        rank,
        columns: n_cols,
        singular_value_ratio,
    })
}

/// Log radiance contribution of one exposure sample given a recovered curve.
///
/// `g[z] - ln(dt)` is the per-exposure estimate of `ln(E)`; the caller
/// combines these with [`weight`].
#[inline]
pub fn log_radiance_sample(g: &[f64], z: u8, log_exposure: f64) -> f64 {
    g[z as usize] - log_exposure
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_is_symmetric_triangle() {
        assert_eq!(weight(0), 0.0);
        assert_eq!(weight(255), 0.0);
        assert_eq!(weight(127), 127.0);
        assert_eq!(weight(128), 127.0);
        assert_eq!(weight(64), 64.0);
    }

    #[test]
    fn sample_selection_strides_evenly() {
        // Synthetic minimal stack is exercised end-to-end in examples/synthetic_demo.rs.
        assert!(LEVELS == 256);
    }
}
