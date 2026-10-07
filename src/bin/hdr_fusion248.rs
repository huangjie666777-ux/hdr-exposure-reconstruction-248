//! Command-line entry point for hdr_fusion248.
//!
//! Usage: `hdr_fusion248 <manifest.json>`.
//! Relative image/output paths resolve against the manifest's directory.

use std::path::{Path, PathBuf};

use serde::Serialize;

use hdr_fusion248::config::load_manifest;
use hdr_fusion248::fusion::{fuse, validity_counts};
use hdr_fusion248::io::{commit_outputs, encode_mask_png, encode_pfm};
use hdr_fusion248::response::recover_channels;
use hdr_fusion248::{Error, Result};

#[derive(Debug, Serialize)]
struct ExposureProvenance {
    path: String,
    exposure_seconds: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    width: u32,
    height: u32,
    smoothness_lambda: f64,
    weight: &'static str,
    anchor: String,
    channels: [ChannelReport; 3],
    exposures: Vec<ExposureProvenance>,
    outputs: OutputReport,
    validity: hdr_fusion248::fusion::ValiditySummary,
}

#[derive(Debug, Serialize)]
struct ChannelReport {
    channel: &'static str,
    /// Log inverse response g[z] for z = 0..=255.
    log_inverse_response: Vec<f64>,
    /// Sampled pixel coordinates actually fitted: [x, y].
    sample_coordinates: Vec<[usize; 2]>,
    sample_count: usize,
    data_rms: f64,
    smoothness_rms: f64,
    anchor_residual: f64,
    rank: usize,
    design_columns: usize,
    singular_value_ratio: f64,
}

#[derive(Debug, Serialize)]
struct OutputReport {
    pfm: String,
    mask_png: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: {} <manifest.json>", args.get(0).map(String::as_str).unwrap_or("hdr_fusion248"));
        std::process::exit(2);
    }
    let manifest_path = PathBuf::from(&args[1]);
    match run(&manifest_path) {
        Ok(report) => {
            println!(
                "hdr_fusion248: reconstructed {}x{} radiance -> {}",
                report.width, report.height, report.outputs.pfm
            );
            for channel in report.channels.iter() {
                println!(
                    "  channel {}: samples={:3} data_rms={:.6e} smooth_rms={:.6e} rank={}/{}",
                    channel.channel,
                    channel.sample_count,
                    channel.data_rms,
                    channel.smoothness_rms,
                    channel.rank,
                    channel.design_columns
                );
            }
            println!(
                "  valid channels: {}/{} (fully valid pixels: {})",
                report.validity.channels_valid,
                report.validity.pixels * 3,
                report.validity.pixels_all_valid
            );
        }
        Err(err) => {
            eprintln!("hdr_fusion248: {err}");
            std::process::exit(1);
        }
    }
}

fn run(manifest_path: &Path) -> Result<Report> {
    let (manifest, images) = load_manifest(manifest_path)?;
    let recovered = recover_channels(&images)?;
    let radiance = fuse(&images, &recovered)?;

    let base_dir = manifest_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| Path::new(".").to_path_buf());
    let stem = manifest_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("hdr");
    let resolve = |value: &Option<String>, suffix: &str| -> PathBuf {
        match value {
            Some(v) => {
                let path = PathBuf::from(v);
                if path.is_absolute() {
                    path
                } else {
                    base_dir.join(path)
                }
            }
            None => base_dir.join(format!("{stem}.{suffix}")),
        }
    };
    let pfm_path = resolve(&manifest.output.pfm, "pfm");
    let mask_path = resolve(&manifest.output.mask, "mask.png");
    let report_path = resolve(&manifest.output.report, "report.json");

    let names = ["red", "green", "blue"];
    let channels = [0, 1, 2].map(|i| ChannelReport {
        channel: names[i],
        log_inverse_response: recovered[i].g.clone(),
        sample_coordinates: recovered[i].sample_coordinates.clone(),
        sample_count: recovered[i].sample_coordinates.len(),
        data_rms: recovered[i].data_rms,
        smoothness_rms: recovered[i].smoothness_rms,
        anchor_residual: recovered[i].anchor_residual,
        rank: recovered[i].rank,
        design_columns: recovered[i].columns,
        singular_value_ratio: recovered[i].singular_value_ratio,
    });

    let exposures = images
        .iter()
        .map(|img| ExposureProvenance {
            path: img.source.clone(),
            exposure_seconds: img.exposure_seconds,
        })
        .collect();

    let validity = validity_counts(&radiance.valid);
    let report = Report {
        width: radiance.width,
        height: radiance.height,
        smoothness_lambda: hdr_fusion248::response::LAMBDA,
        weight: "w(z) = min(z, 255 - z)",
        anchor: "g[128] = 0".to_string(),
        channels,
        exposures,
        outputs: OutputReport {
            pfm: pfm_path.display().to_string(),
            mask_png: mask_path.display().to_string(),
        },
        validity,
    };

    // Render everything first so any encoding error aborts before touching
    // existing deliverables; then stage all files and atomically commit.
    let pfm_bytes = encode_pfm(&radiance)?;
    let mask_bytes = encode_mask_png(&radiance)?;
    let report_bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| Error::msg(format!("failed to serialize report: {e}")))?;
    commit_outputs(vec![
        (pfm_path, pfm_bytes),
        (mask_path, mask_bytes),
        (report_path, report_bytes),
    ])?;

    Ok(report)
}
