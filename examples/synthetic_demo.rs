//! Synthetic known-response demo and self-test for hdr_fusion248.
//!
//! It builds a small linear radiance image with a wide intensity sweep,
//! applies a known *non-linear* camera response (`z = 255 * (E*dt)^(1/2.2)`,
//! i.e. gamma 2.2 encoding) under five bracketed exposures, writes the input
//! stack plus a manifest, runs the real recovery pipeline, and checks the
//! reconstruction numerically.
//!
//! Run with: `cargo run --example synthetic_demo -- --check`
//!
//! Without `--check`, demo artifacts are written into `demo_out/`.

use std::io::Cursor;
use std::path::Path;

use hdr_fusion248::config::validate_images;
use hdr_fusion248::fusion::fuse;
use hdr_fusion248::io::{encode_mask_png, encode_pfm, read_png_rgb8};
use hdr_fusion248::response::recover_channels;
use hdr_fusion248::ExposureImage;

const W: u32 = 64;
const H: u32 = 64;
const GAMMA: f64 = 2.2;
const EXPOSURES: [f64; 5] = [1.0 / 32.0, 1.0 / 8.0, 1.0 / 2.0, 2.0, 8.0];

/// Ground-truth relative linear radiance per pixel (same in every channel,
/// plus a chromatic gradient) spanning roughly five orders of magnitude.
fn ground_truth(x: usize, y: usize) -> [f64; 3] {
    let t = x as f64 / (W as f64 - 1.0);
    // 10^-2.0 .. 10^2.0 across the width.
    let e = 10f64.powf(-2.0 + 4.0 * t);
    let shade = 0.55 + 0.45 * ((y as f64 / (H as f64 - 1.0)) * std::f64::consts::PI).sin();
    let v = e * shade;
    let tint = 0.85 + 0.15 * (y as f64 / (H as f64 - 1.0));
    [v, v * shade, v * tint]
}

/// Apply the known non-linear response and 8-bit quantization/clamping.
fn encode_value(radiance: f64, exposure: f64) -> u8 {
    let linear = radiance * exposure;
    if linear <= 0.0 {
        return 0;
    }
    let encoded = 255.0 * linear.powf(1.0 / GAMMA);
    if encoded <= 0.0 {
        0
    } else if encoded >= 255.0 {
        255
    } else {
        encoded.round() as u8
    }
}

fn build_stack() -> Vec<ExposureImage> {
    EXPOSURES
        .iter()
        .map(|&dt| {
            let mut rgb = Vec::with_capacity((W * H * 3) as usize);
            for y in 0..H as usize {
                for x in 0..W as usize {
                    for c in 0..3 {
                        rgb.push(encode_value(ground_truth(x, y)[c], dt));
                    }
                }
            }
            ExposureImage {
                width: W,
                height: H,
                rgb,
                exposure_seconds: dt,
                source: format!("synthetic_dt={dt:.4}s.png"),
            }
        })
        .collect()
}

/// Round-trip a stack through real PNG encoding and decoding to exercise the
/// production image path.
fn png_roundtrip_stack(stack: &[ExposureImage]) -> Vec<ExposureImage> {
    stack
        .iter()
        .map(|img| {
            let mut bytes = Vec::new();
            {
                let mut encoder = png::Encoder::new(Cursor::new(&mut bytes), W, H);
                encoder.set_color(png::ColorType::Rgb);
                encoder.set_depth(png::BitDepth::Eight);
                let mut writer = encoder.write_header().unwrap();
                writer.write_image_data(&img.rgb).unwrap();
            }
            let decoded = read_png_rgb8_from_bytes(&bytes);
            assert_eq!(decoded.rgb, img.rgb, "PNG round-trip must be lossless");
            ExposureImage {
                width: decoded.width,
                height: decoded.height,
                rgb: decoded.rgb,
                exposure_seconds: img.exposure_seconds,
                source: img.source.clone(),
            }
        })
        .collect()
}

fn read_png_rgb8_from_bytes(bytes: &[u8]) -> hdr_fusion248::io::Rgb8Image {
    let tmp = std::env::temp_dir().join(format!("hdr248_demo_{}.png", std::process::id()));
    std::fs::write(&tmp, bytes).unwrap();
    let image = read_png_rgb8(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    image
}

/// Linear regression of `y` on `x` with an intercept: returns (slope, rms).
fn linear_fit(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        cov += (x - mean_x) * (y - mean_y);
        var += (x - mean_x).powi(2);
    }
    let slope = cov / var;
    let intercept = mean_y - slope * mean_x;
    let sse: f64 = ys
        .iter()
        .zip(xs)
        .map(|(y, x)| (y - (slope * x + intercept)).powi(2))
        .sum();
    let sy = (ys.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>() / n).sqrt();
    let rms = (sse / n).sqrt();
    (slope, rms / sy, intercept)
}

fn main() {
    let check_only = std::env::args().nth(1).as_deref() == Some("--check");
    let out_dir = Path::new("demo_out");

    let stack = build_stack();
    let stack = png_roundtrip_stack(&stack);
    validate_images(&stack).expect("synthetic stack must pass validation");

    let recovered = recover_channels(&stack).expect("response recovery");
    let map = fuse(&stack, &recovered).expect("fusion");

    // The recovered image is proportional to ground truth in log space with a
    // constant offset (the gauge g[128]=0 fixes only an additive constant).
    let mut failures = Vec::new();
    let names = ["R", "G", "B"];
    for c in 0..3 {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut valid_pixels = 0usize;
        for y in 0..H as usize {
            for x in 0..W as usize {
                let idx = (y * W as usize + x) * 3 + c;
                if map.valid[idx] == 255 {
                    valid_pixels += 1;
                    xs.push(ground_truth(x, y)[c].ln());
                    ys.push((map.rgb[idx] as f64).ln());
                }
            }
        }
        assert_eq!(valid_pixels, (W * H) as usize, "all pixels valid: {}", names[c]);
        let (slope, normalized_rms, _intercept) = linear_fit(&xs, &ys);
        println!(
            "channel {}: slope={slope:.6} normalized_log_rms={normalized_rms:.3e} samples={}",
            names[c], recovered[c].sample_coordinates.len()
        );
        if (slope - 1.0).abs() > 0.02 {
            failures.push(format!("{} slope {slope} not ~1", names[c]));
        }
        if normalized_rms > 0.01 {
            failures.push(format!("{} normalized log rms {normalized_rms} > 0.01", names[c]));
        }

        // The inverse response must match the known gamma curve up to a shift:
        // g(z) = GAMMA * ln(z/255) + const. Compare pairwise differences,
        // which cancel the unknown constant.
        let z1 = 40u8;
        let z2 = 210u8;
        let recovered_delta = recovered[c].g[z2 as usize] - recovered[c].g[z1 as usize];
        let truth_delta = GAMMA * ((z2 as f64 / 255.0) / (z1 as f64 / 255.0)).ln();
        let curve_err = (recovered_delta - truth_delta).abs();
        println!(
            "channel {}: g({z2})-g({z1}) recovered={recovered_delta:.6} truth={truth_delta:.6} |err|={curve_err:.3e}",
            names[c]
        );
        if curve_err > 0.05 {
            failures.push(format!("{} response curve error {curve_err}", names[c]));
        }
        println!(
            "channel {}: data_rms={:.6e} smoothness_rms={:.6e} rank={}/{}",
            names[c], recovered[c].data_rms, recovered[c].smoothness_rms,
            recovered[c].rank, recovered[c].columns
        );
    }

    // Center pixel reconstruction vs ground truth (ratio is constant per channel).
    for c in 0..3 {
        let idx = ((H as usize / 2) * W as usize + W as usize / 2) * 3 + c;
        let truth = ground_truth(W as usize / 2, H as usize / 2)[c];
        let got = map.rgb[idx] as f64;
        println!(
            "center {}: truth={truth:.6} reconstructed={got:.6} scale={:.6}",
            names[c], got / truth
        );
        assert!(got.is_finite());
    }

    if !check_only {
        std::fs::create_dir_all(out_dir).unwrap();
        for (i, img) in stack.iter().enumerate() {
            let mut bytes = Vec::new();
            {
                let mut encoder = png::Encoder::new(Cursor::new(&mut bytes), W, H);
                encoder.set_color(png::ColorType::Rgb);
                encoder.set_depth(png::BitDepth::Eight);
                let mut writer = encoder.write_header().unwrap();
                writer.write_image_data(&img.rgb).unwrap();
            }
            std::fs::write(out_dir.join(format!("exposure_{i}.png")), bytes).unwrap();
        }
        std::fs::write(out_dir.join("radiance.pfm"), encode_pfm(&map).unwrap()).unwrap();
        std::fs::write(out_dir.join("mask.png"), encode_mask_png(&map).unwrap()).unwrap();
        println!("demo artifacts written to {out_dir:?}");
    }

    if failures.is_empty() {
        println!("SELF-TEST PASSED: recovered linear radiance tracks ground truth");
    } else {
        eprintln!("SELF-TEST FAILED: {failures:?}");
        std::process::exit(1);
    }
}
