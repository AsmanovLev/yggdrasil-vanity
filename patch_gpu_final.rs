use std::fs;
pub fn main() {
    let mut s = fs::read_to_string("src/gpu.rs").unwrap();
    
    // Fix struct
    s = s.replace("pub device_idx: usize,", "pub device_idx: Vec<usize>,");
    
    // Fix new()
    s = s.replace("DeviceSpecifier::Indices(vec![opts.device_idx])", "DeviceSpecifier::Indices(opts.device_idx)");

    // Need to handle the run_opencl loop, it is large.
    // I will replace run_opencl entirely.
    let run_opencl_start = s.find("pub fn run_opencl").unwrap();
    let end_of_file = s.len();
    let new_run_opencl = r#"pub fn run_opencl(args: &Args, csv_file: &Option<Mutex<fs::File>>, app_start: std::time::Instant) {
    let platform = Platform::list().get(args.platform_idx).expect("Invalid platform index");
    let all_devices = ocl::Device::list_all(platform).unwrap_or_default();

    let device_indices = if args.device_idx == "all" {
        (0..all_devices.len()).collect::<Vec<_>>()
    } else {
        args.device_idx
            .split(',')
            .map(|s| s.trim().parse::<usize>().expect("Invalid device index"))
            .collect::<Vec<_>>()
    };

    let mut threads = Vec::new();
    let best_score = Arc::new(AtomicI64::new(i64::MIN));
    let total_iters = Arc::new(AtomicU64::new(0));
    let lifetime_candidates = Arc::new(AtomicU64::new(0));

    let (cand_tx, cand_rx) = mpsc::sync_channel(device_indices.len() * 2);

    for device_idx in device_indices {
        let args = args.clone();
        let best_score = best_score.clone();
        let total_iters = total_iters.clone();
        let lifetime_candidates = lifetime_candidates.clone();
        let cand_tx = cand_tx.clone();

        let thread = thread::spawn(move || {
            run_gpu_thread(
                &args,
                device_idx,
                best_score,
                total_iters,
                lifetime_candidates,
                cand_tx,
                app_start,
            );
        });
        threads.push(thread);
    }
    drop(cand_tx);

    let stats_thread = {
        let best_score = best_score.clone();
        let total_iters = total_iters.clone();
        let lifetime_candidates = lifetime_candidates.clone();
        let log_interval = args.log_interval;

        thread::spawn(move || {
            let mut start = time::Instant::now();
            let mut stats_start: Option<time::Instant> = None;
            loop {
                thread::sleep(time::Duration::from_secs(log_interval));

                if app_start.elapsed().as_secs() >= 10 && stats_start.is_none() {
                    stats_start = Some(time::Instant::now());
                }

                let current_iters = total_iters.swap(0, Ordering::Relaxed);
                let elapsed_secs = start.elapsed().as_secs_f64();
                let hashrate = current_iters as f64 / elapsed_secs / 1_000_000.0;
                start = time::Instant::now();

                let best_raw = best_score.load(Ordering::Relaxed);
                let best_display = if best_raw == i64::MIN {
                    "none".to_string()
                } else {
                    format!("{:.6}", fixed_to_score(best_raw))
                };
                
                let (cands_per_hour, eta_str) = if let Some(st) = stats_start {
                    let active_secs = st.elapsed().as_secs_f64();
                    let lc = lifetime_candidates.load(Ordering::Relaxed);
                    if active_secs > 0.0 && lc > 0 {
                        let cps = lc as f64 / active_secs;
                        (cps * 3600.0, format_eta(1.0 / cps))
                    } else {
                        (0.0, "∞".to_string())
                    }
                } else {
                    (0.0, "Calibrating...".to_string())
                };

                eprintln!(
                    "{} Hashrate: {:.2} MH/s  BestScore: {}  Candidates/h: {:.5}  ETA: {}",
                    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
                    hashrate,
                    best_display,
                    cands_per_hour,
                    eta_str
                );
            }
        })
    };
    
    let csv_file = csv_file.clone();
    let capture_range = args.capture_range;
    let cpu_thread = thread::spawn(move || {
        let csv = csv_file;
        for (buf, count) in cand_rx {
            process_candidates(&buf, count, &best_score, &csv, capture_range, app_start);
        }
    });

    for thread in threads {
        thread.join().unwrap();
    }
    stats_thread.join().unwrap();
    cpu_thread.join().unwrap();
}

fn run_gpu_thread(
    args: &Args,
    device_idx: usize,
    best_score: Arc<AtomicI64>,
    total_iters: Arc<AtomicU64>,
    lifetime_candidates: Arc<AtomicU64>,
    cand_tx: mpsc::SyncSender<(Vec<u8>, usize)>,
    app_start: std::time::Instant,
) {
    let capture_range = args.capture_range;
    let mode = match args.preset {
        Preset::Height => 0,
        Preset::Length => 1,
        Preset::HeightLength => 2,
    };
    let fw = args.weight_h;
    let lw = args.weight_l;
    let mh = args.minimal_height;
    let threads_per_gpu = args.threads;

    let mut gpu = Gpu::new(
        GpuOptions {
            platform_idx: args.platform_idx,
            device_idx: vec![device_idx],
            threads: threads_per_gpu,
            local_work_size: args.local_work_size,
        },
        mode,
        fw,
        lw,
        mh,
    )
    .unwrap();

    let mut global_offset = 0u64;

    loop {
        let threshold = score_threshold(best_score.load(Ordering::Relaxed), capture_range);
        gpu.write_threshold(threshold).unwrap();
        gpu.reset_counter().unwrap();
        
        gpu.kernel.set_arg(6, global_offset).unwrap();

        gpu.compute().unwrap();

        let mut candidate_buf = Vec::with_capacity(CANDIDATES_BUF_BYTES);
        let count = gpu.read_candidates(&mut candidate_buf).unwrap();

        if count > 0 {
            let payload_end = CANDIDATES_HEADER + count * CANDIDATE_STRIDE;
            let payload = candidate_buf[..payload_end].to_vec();
            if cand_tx.send((payload, count)).is_err() {
                break; 
            }
            if app_start.elapsed().as_secs() >= 10 {
                lifetime_candidates.fetch_add(count as u64, Ordering::Relaxed);
            }
        }

        global_offset += threads_per_gpu as u64;
        total_iters.fetch_add(threads_per_gpu as u64, Ordering::Relaxed);
    }
}

fn process_candidates(
    buf: &[u8],
    count: usize,
    best_score: &Arc<AtomicI64>,
    csv_file: &Option<Mutex<fs::File>>,
    capture_range: f64,
    app_start: std::time::Instant,
) {
    let data = &buf[CANDIDATES_HEADER..];
    data.par_chunks_exact(CANDIDATE_STRIDE)
        .take(count)
        .for_each(|rec| {
            let seed = &rec[0..32];
            let pk = &rec[32..64];
            let gpu_score = i64::from_le_bytes(rec[64..72].try_into().unwrap());
            handle_candidate(seed, pk, gpu_score, best_score, csv_file, capture_range, app_start);
        });
}

fn format_eta(eta_secs: f64) -> String {
    if eta_secs > 86400.0 {
        format!(\"{:.1}d\", eta_secs / 86400.0)
    } else if eta_secs > 3600.0 {
        format!(\"{:.1}h\", eta_secs / 3600.0)
    } else if eta_secs > 60.0 {
        format!(\"{:.1}m\", eta_secs / 60.0)
    } else {
        format!(\"{:.0}s\", eta_secs)
    }
}
"#;
    let final_s = format!("{}{}", &s[..run_opencl_start], new_run_opencl);
    fs::write("src/gpu.rs", final_s).unwrap();
}
