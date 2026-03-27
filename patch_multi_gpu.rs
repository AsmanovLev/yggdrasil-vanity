use std::fs;
pub fn main() {
    let mut s = fs::read_to_string("src/gpu.rs").unwrap();
    
    // Replace run_opencl with a version that accepts &Args and runs loop per device
    let start = s.find("pub fn run_opencl").unwrap();
    let end = s.find("fn process_candidates").unwrap();
    
    let new_run_opencl = r#"pub fn run_opencl(args: &Args, csv_file: &Option<Mutex<fs::File>>, app_start: std::time::Instant) {
    let platforms = Platform::list();
    let platform = platforms.get(args.platform_idx).expect("Invalid platform index");
    let all_devices = ocl::Device::list_all(platform).unwrap_or_default();
    
    let device_indices = if args.device_idx == "all" {
        (0..all_devices.len()).collect::<Vec<_>>()
    } else {
        args.device_idx.split(',').map(|s| s.trim().parse::<usize>().expect("Invalid device index")).collect::<Vec<_>>()
    };

    let best_score = Arc::new(AtomicI64::new(i64::MIN));
    let (cand_tx, cand_rx) = mpsc::sync_channel(device_indices.len() * 2);

    let mut gpu_threads = Vec::new();
    for dev_idx in device_indices {
        let args = args.clone();
        let best_score = best_score.clone();
        let cand_tx = cand_tx.clone();
        gpu_threads.push(thread::spawn(move || {
            run_gpu_instance(&args, dev_idx, best_score, cand_tx, app_start);
        }));
    }
    drop(cand_tx);

    let csv_file = Arc::new(csv_file.clone());
    let cpu_thread = thread::spawn(move || {
        for (buf, count) in cand_rx {
            process_candidates(&buf, count, &best_score, &csv_file, args.capture_range, app_start);
        }
    });

    for t in gpu_threads { t.join().unwrap(); }
    cpu_thread.join().unwrap();
}

fn run_gpu_instance(args: &Args, device_idx: usize, best_score: Arc<AtomicI64>, cand_tx: mpsc::SyncSender<(Vec<u8>, usize)>, app_start: std::time::Instant) {
    let mode = match args.preset { Preset::Height => 0, Preset::Length => 1, Preset::HeightLength => 2 };
    let mut gpu = Gpu::new(GpuOptions { platform_idx: args.platform_idx, device_idx: vec![device_idx], threads: args.threads, local_work_size: args.local_work_size }, mode, args.weight_h, args.weight_l, args.minimal_height).unwrap();
    
    let mut global_offset = 0u64;
    loop {
        let threshold = score_threshold(best_score.load(Ordering::Relaxed), args.capture_range);
        gpu.write_threshold(threshold).unwrap();
        gpu.reset_counter().unwrap();
        gpu.kernel.set_arg(6, global_offset).unwrap();
        gpu.compute().unwrap();
        
        let mut candidate_buf = Vec::with_capacity(CANDIDATES_BUF_BYTES);
        if let Ok(count) = gpu.read_candidates(&mut candidate_buf) {
            if count > 0 {
                let _ = cand_tx.send((candidate_buf[..CANDIDATES_HEADER + count * CANDIDATE_STRIDE].to_vec(), count));
            }
        }
        global_offset += args.threads as u64;
    }
}
"#;
    s.replace_range(start..end, new_run_opencl);
    fs::write("src/gpu.rs", s).unwrap();
}
