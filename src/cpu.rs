use std::{
    fs,
    sync::{
        atomic::{AtomicI64, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread, time,
};

use chrono::{SecondsFormat, Utc};
use rand::RngCore;
use rayon::prelude::*;

use crate::{
    
    handler::{fixed_to_score, handle_keypair},
    Args,
};

pub fn run_cpu(
    args: &Args,
    csv_file: &Option<Mutex<fs::File>>,
    app_start: std::time::Instant,
) {
    let preset = &args.preset;
    let capture_range = args.capture_range;
    let wh = args.weight_h;
    let wl = args.weight_l;
    let generated = Arc::new(AtomicU64::new(0));
    let best_score = Arc::new(AtomicI64::new(i64::MIN));

    let stats_thread = {
        let log_interval = args.log_interval;
        let generated = generated.clone();
        let best_score = best_score.clone();

        thread::spawn(move || {
            let mut start = time::Instant::now();
            loop {
                thread::sleep(time::Duration::from_secs(log_interval));

                let hashrate = generated.swap(0, Ordering::AcqRel) as f64
                    / start.elapsed().as_secs_f64()
                    / 1_000_000.0;
                start = time::Instant::now();

                let best_raw = best_score.load(Ordering::Relaxed);
                let best_display = if best_raw == i64::MIN {
                    "none".to_string()
                } else {
                    format!("{:.6}", fixed_to_score(best_raw))
                };

                eprintln!(
                    "{} Hashrate: {:.3} MH/s  BestScore: {}",
                    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
                    hashrate,
                    best_display,
                );
            }
        })
    };

    rayon::iter::repeat(()).for_each(|_| {
        let mut rng = rand::thread_rng();
        for _ in 0..64 * 1024 {
            let mut seed = [0u8; 32];
            rng.fill_bytes(&mut seed);

            let kp = ed25519_dalek::SigningKey::from_bytes(&seed);
            let pk = kp.verifying_key();
            handle_keypair(
                &seed,
                pk.as_bytes(),
                preset,
                wh,
                wl,
                capture_range,
                &best_score,
                csv_file,
                app_start,
            );
        }
        generated.fetch_add(64 * 1024, Ordering::AcqRel);
    });

    stats_thread.join().unwrap();
}
