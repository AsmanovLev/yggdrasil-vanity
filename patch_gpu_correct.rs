use std::fs;
pub fn main() {
    let mut s = fs::read_to_string("src/gpu.rs").unwrap();
    
    let start = s.find("pub fn run_opencl").unwrap();
    let end = s.find("fn run_gpu_instance").unwrap();
    
    let new_run_opencl = r#"pub fn run_opencl(args: &Args, csv_file: &Option<Mutex<fs::File>>, app_start: std::time::Instant) {
    let platform = Platform::list().get(args.platform_idx).expect("Invalid platform index");
    let all_devices = ocl::Device::list_all(platform).unwrap_or_default();
    
    let device_indices = if args.device_idx == "all" {
        (0..all_devices.len()).collect::<Vec<_>>()
    } else {
        args.device_idx.split(',').map(|s| s.trim().parse::<usize>().expect("Invalid device index")).collect::<Vec<_>>()
    };

    let best_score = Arc::new(AtomicI64::new(i64::MIN));
    let (cand_tx, cand_rx) = mpsc::sync_channel(device_indices.len() * 2);

    let mut gpu_threads = Vec::new();
    let args_arc = Arc::new(args.clone());
    for dev_idx in device_indices {
        let args = args_arc.clone();
        let best_score = best_score.clone();
        let cand_tx = cand_tx.clone();
        gpu_threads.push(thread::spawn(move || {
            run_gpu_instance(&args, dev_idx, best_score, cand_tx, app_start);
        }));
    }
    drop(cand_tx);

    let csv_file = Arc::new(csv_file.clone());
    let capture_range = args.capture_range;
    let cpu_thread = thread::spawn(move || {
        for (buf, count) in cand_rx {
            process_candidates(&buf, count, &best_score, &csv_file, capture_range, app_start);
        }
    });

    for t in gpu_threads { t.join().unwrap(); }
    cpu_thread.join().unwrap();
}
"#;
    s.replace_range(start..end, new_run_opencl);
    fs::write("src/gpu.rs", s).unwrap();
}
