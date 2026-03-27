use chrono::Utc;
use ed25519_dalek::SigningKey;
use hex::ToHex;
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use crate::Preset;

const SCORE_SCALE: f64 = 1_000_000.0;

pub fn compute_score(h: u8, l: u8, preset: &Preset, wh: f32, wl: f32) -> f64 {
    match preset {
        Preset::Height => h as f64,
        Preset::Length => -(l as f64),
        Preset::HeightLength => (h as f64 * wh as f64) - (l as f64 * wl as f64),
    }
}


#[inline]
pub fn score_to_fixed(score: f64) -> i64 {
    (score * SCORE_SCALE) as i64
}

#[inline]
pub fn fixed_to_score(val: i64) -> f64 {
    val as f64 / SCORE_SCALE
}

/// Called from the GPU path: score already computed on GPU.
/// We just check threshold, update best, and output.
pub fn handle_candidate(
    seed: &[u8],
    pk: &[u8],
    gpu_score: i64,
    best_score: &Arc<AtomicI64>,
    csv_file: &Option<Mutex<fs::File>>,
    capture_range: f64,
    app_start: std::time::Instant,
) {
    let current_best = best_score.load(Ordering::Relaxed);
    let threshold = if current_best == i64::MIN {
        i64::MIN
    } else {
        score_to_fixed(fixed_to_score(current_best) - capture_range)
    };
    if gpu_score < threshold {
        return;
    }

    let prev_best = best_score.fetch_max(gpu_score, Ordering::SeqCst);
    let is_record = gpu_score > prev_best;

    if capture_range == 0.0 && !is_record {
        return;
    }

    let updated_best = best_score.load(Ordering::Relaxed);
    let threshold2 = if updated_best == i64::MIN {
        i64::MIN
    } else {
        score_to_fixed(fixed_to_score(updated_best) - capture_range)
    };
    if gpu_score < threshold2 {
        return;
    }

    // Recompute display values.
    let height = leading_zeros_of_pubkey(pk);
    let subnet_str = subnet_string(pk, height);
    let length = subnet_str.len() as u8;
    let addr_str = address_for_pubkey_str(pk, height);
    let score = fixed_to_score(gpu_score);

    let privkey_str = build_privkey(seed, pk);

    let is_calibrating = app_start.elapsed().as_secs() < 10;
    if !is_calibrating {
        print_result(
            is_record,
            score,
            &subnet_str,
            &addr_str,
            height,
            length,
            &privkey_str,
        );

        if let Some(csv) = csv_file {
            let mut f = csv.lock().unwrap();
            writeln!(
                f,
                "{},{subnet_str},{addr_str},{height},{length},{score:.6},{privkey_str}",
                Utc::now().timestamp()
            )
            .unwrap();
        }
    }
}

/// Called from the CPU path: compute score here.
pub fn handle_keypair(
    seed: &[u8],
    pk: &[u8],
    preset: &Preset,
    weight_h: f32,
    weight_l: f32,
    capture_range: f64,
    best_score: &Arc<AtomicI64>,
    csv_file: &Option<Mutex<fs::File>>,
    app_start: std::time::Instant,
) {
    let height = leading_zeros_of_pubkey(pk);
    let subnet_str = subnet_string(pk, height);
    let length = subnet_str.len() as u8;

    let score = compute_score(height, length, preset, weight_h, weight_l);
    let score_fixed = score_to_fixed(score);

    // Fast pre-check before atomic ops.
    let current_best = best_score.load(Ordering::Relaxed);
    let threshold = if current_best == i64::MIN {
        i64::MIN
    } else {
        score_to_fixed(fixed_to_score(current_best) - capture_range)
    };
    if score_fixed < threshold {
        return;
    }

    let prev_best = best_score.fetch_max(score_fixed, Ordering::SeqCst);
    let is_record = score_fixed > prev_best;

    if capture_range == 0.0 && !is_record {
        return;
    }

    // Re-check with updated best.
    let updated_best = best_score.load(Ordering::Relaxed);
    let threshold2 = if updated_best == i64::MIN {
        i64::MIN
    } else {
        score_to_fixed(fixed_to_score(updated_best) - capture_range)
    };
    if score_fixed < threshold2 {
        return;
    }

    let addr_str = address_for_pubkey_str(pk, height);
    let privkey_str = build_privkey(seed, pk);

    let is_calibrating = app_start.elapsed().as_secs() < 10;
    if !is_calibrating {
        print_result(
            is_record,
            score,
            &subnet_str,
            &addr_str,
            height,
            length,
            &privkey_str,
        );

        if let Some(csv) = csv_file {
            let mut f = csv.lock().unwrap();
            writeln!(
                f,
                "{},{subnet_str},{addr_str},{height},{length},{score:.6},{privkey_str}",
                Utc::now().timestamp()
            )
            .unwrap();
        }
    }
}

// ------------------------------------------------------------------ //
// Internal helpers                                                     //
// ------------------------------------------------------------------ //

fn leading_zeros_of_pubkey(pk: &[u8]) -> u8 {
    let mut zeros = 0u8;
    for b in pk {
        let z = b.leading_zeros() as u8;
        zeros += z;
        if z != 8 {
            break;
        }
    }
    zeros
}

/// Build the /64 subnet string (lower 64 bits zeroed, bit 0 of byte 0 set).
fn subnet_string(pk: &[u8], height: u8) -> String {
    let mut buf = address_for_pubkey_octets(pk, height);
    buf[0] |= 0x01;
    buf[8..16].fill(0);
    std::net::Ipv6Addr::from(buf).to_string()
}

fn address_for_pubkey_str(pk: &[u8], height: u8) -> String {
    std::net::Ipv6Addr::from(address_for_pubkey_octets(pk, height)).to_string()
}

fn address_for_pubkey_octets(pk: &[u8], height: u8) -> [u8; 16] {
    let mut buf = [0u8; 16];
    buf[0] = 0x02;
    buf[1] = height;
    let shift = ((height + 1) % 8) as u32;
    for (src, trg) in pk[(height / 8) as usize..]
        .windows(2)
        .zip(buf[2..].iter_mut())
    {
        let low_bits = if shift == 0 {
            0
        } else {
            src[1].wrapping_shr(8 - shift)
        };
        *trg = src[0].wrapping_shl(shift) ^ low_bits ^ 0xFF;
    }
    buf
}

fn build_privkey(seed: &[u8], pk: &[u8]) -> String {
    // Verify in debug builds only.
    debug_assert!({
        let mut fixed = [0u8; 32];
        fixed.copy_from_slice(seed);
        SigningKey::from_bytes(&fixed).verifying_key().to_bytes() == pk
    });

    let mut sk = [0u8; 64];
    sk[..32].copy_from_slice(seed);
    sk[32..].copy_from_slice(pk);
    sk.encode_hex()
}

fn print_result(
    is_record: bool,
    score: f64,
    subnet_str: &str,
    addr_str: &str,
    height: u8,
    length: u8,
    privkey_str: &str,
) {
    let mut lock = std::io::stdout().lock();
    if is_record {
        writeln!(lock, "=======================================").unwrap();
        writeln!(lock, "NEW RECORD  score: {:.6}", score).unwrap();
    } else {
        writeln!(lock, "---------------------------------------").unwrap();
        writeln!(lock, "Near record  score: {:.6}", score).unwrap();
    }
    writeln!(lock, "Subnet:     {}/64", subnet_str).unwrap();
    writeln!(lock, "Address:    {}", addr_str).unwrap();
    writeln!(lock, "Height:     {}", height).unwrap();
    writeln!(lock, "Length:     {}", length).unwrap();
    writeln!(lock, "PrivateKey: {}", privkey_str).unwrap();
    if is_record {
        writeln!(lock, "=======================================").unwrap();
    } else {
        writeln!(lock, "---------------------------------------").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::{address_for_pubkey_octets, leading_zeros_of_pubkey};
    use std::str::FromStr;

    #[test]
    fn test_address_for_pubkey() {
        let pk = hex::decode("000000000c4f58e09d19592f242951e6aa3185bd5ec6b95c0d56c93ae1268cbd")
            .unwrap();
        let h = leading_zeros_of_pubkey(&pk);
        assert_eq!(
            std::net::Ipv6Addr::from(address_for_pubkey_octets(&pk, h)),
            std::net::Ipv6Addr::from_str("224:7614:e3ec:5cd4:da1b:7ad5:c32a:b9cf").unwrap()
        );
    }
}
