use std::fs;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time;

use chrono::SecondsFormat;
use chrono::Utc;
use ocl::builders::DeviceSpecifier;
use ocl::builders::ProgramBuilder;
use ocl::flags::MemFlags;
use ocl::Buffer;
use ocl::Platform;
use ocl::ProQue;
use ocl::Result;
use rand::RngCore;
use rayon::prelude::*;

use crate::handler::{fixed_to_score, handle_candidate};
use crate::Args;
use crate::Preset;

const MAX_CANDIDATES: usize = 65536;
const CANDIDATE_STRIDE: usize = 72; // seed[32] + pubkey[32] + score_i64[8]
const CANDIDATES_HEADER: usize = 4; // u32 counter at offset 0
const CANDIDATES_BUF_BYTES: usize = CANDIDATES_HEADER + MAX_CANDIDATES * CANDIDATE_STRIDE;

struct GpuOptions {
    pub platform_idx: usize,
    pub device_idx: usize,
    pub threads: usize,
    pub local_work_size: Option<usize>,
}

struct Gpu {
    kernel: ocl::Kernel,
    candidates: Buffer<u8>,
    #[allow(dead_code)]
    base_seed: Buffer<u8>,
    threshold_buf: Buffer<u8>,
}

impl Gpu {
    pub fn new(opts: GpuOptions, mode: u32, fw: f32, lw: f32, mh: u8) -> Result<Gpu> {
        let mut prog_bldr = ProgramBuilder::new();
        prog_bldr
            .src(include_str!("../kernel/sha512.cl"))
            .src(include_str!("../kernel/curve25519-constants.cl"))
            .src(include_str!("../kernel/curve25519-constants2.cl"))
            .src(include_str!("../kernel/curve25519.cl"))
            .src(include_str!("../kernel/entry.cl"));

        let platforms = Platform::list();
        if platforms.is_empty() {
            return Err("No OpenCL platforms exist (check your drivers and OpenCL setup)".into());
        }
        if opts.platform_idx >= platforms.len() {
            return Err(format!(
                "Platform index {} too large (max {})",
                opts.platform_idx,
                platforms.len() - 1
            )
            .into());
        }

        let pro_que = ProQue::builder()
            .prog_bldr(prog_bldr)
            .platform(platforms[opts.platform_idx])
            .device(DeviceSpecifier::Indices(vec![opts.device_idx]))
            .dims(opts.threads)
            .build()?;

        let device = pro_que.device();
        eprintln!("Initializing GPU {} {}", device.vendor()?, device.name()?);

        let candidates = pro_que
            .buffer_builder::<u8>()
            .len(CANDIDATES_BUF_BYTES)
            .flags(MemFlags::new().read_write())
            .build()?;

        let base_seed = pro_que
            .buffer_builder::<u8>()
            .len(32)
            .flags(MemFlags::new().read_only().host_write_only())
            .build()?;

        let mut bs = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut bs);
        base_seed.write(&bs[..]).enq()?;

        let threshold_buf = pro_que
            .buffer_builder::<u8>()
            .len(8)
            .flags(MemFlags::new().read_only().host_write_only())
            .build()?;

        let kernel = {
            let mut kb = pro_que.kernel_builder("generate_pubkey");
            kb.global_work_size(opts.threads)
                .arg(&candidates)
                .arg(&base_seed)
                .arg(&threshold_buf)
                .arg(mode)
                .arg(fw)
                .arg(lw)
                .arg(0u64)
                .arg(mh);
            if let Some(lws) = opts.local_work_size {
                kb.local_work_size(lws);
            }
            kb.build()?
        };

        Ok(Gpu {
            kernel,
            candidates,
            base_seed,
            threshold_buf,
        })
    }

    pub fn write_threshold(&mut self, threshold: i64) -> Result<()> {
        let bytes = threshold.to_le_bytes();
        self.threshold_buf.write(bytes.as_slice()).enq()
    }

    pub fn reset_counter(&mut self) -> Result<()> {
        self.candidates.write(&[0u8, 0, 0, 0][..]).offset(0).enq()
    }

    pub fn compute(&mut self) -> Result<()> {
        unsafe { self.kernel.enq() }
    }

    pub fn read_candidates(&mut self, out: &mut Vec<u8>) -> Result<usize> {
        out.resize(CANDIDATES_BUF_BYTES, 0u8);
        self.candidates.read(out.as_mut_slice()).enq()?;
        let count = u32::from_le_bytes(out[0..4].try_into().unwrap()) as usize;
        Ok(count.min(MAX_CANDIDATES))
    }
}

fn score_threshold(best: i64, capture_range: f64) -> i64 {
    if best == i64::MIN {
        i64::MIN
    } else {
        ((fixed_to_score(best) - capture_range) * 1_000_000.0) as i64
    }
}

pub fn run_opencl(args: &Args, csv_file: &Option<Mutex<fs::File>>, app_start: std::time::Instant) {
    let capture_range = args.capture_range;
    let mode = match args.preset {
        Preset::Height => 0,
        Preset::Length => 1,
        Preset::HeightLength => 2,
    };
    let fw = args.weight_h;
    let lw = args.weight_l;
    let mh = args.minimal_height;

    let threads = args.threads;

    let mut gpu = Gpu::new(
        GpuOptions {
            platform_idx: args.platform_idx,
            device_idx: args.device_idx,
            threads,
            local_work_size: args.local_work_size,
        },
        mode,
        fw,
        lw,
        mh,
    )
    .unwrap();

    let best_score = Arc::new(AtomicI64::new(i64::MIN));
    let (cand_tx, cand_rx) = mpsc::sync_channel::<(Vec<u8>, usize)>(1);

    let _cpu_thread = {
        let best_score = best_score.clone();
        let csv_ptr: usize = csv_file as *const Option<Mutex<fs::File>> as usize;
        thread::spawn(move || {
            let csv = unsafe { &*(csv_ptr as *const Option<Mutex<fs::File>>) };
            for (buf, count) in cand_rx {
                process_candidates(&buf, count, &best_score, csv, capture_range, app_start);
            }
        })
    };

    let mut iters = 0u64;
    let mut global_offset = 0u64;
    let mut lifetime_candidates = 0u64;
    let mut stats_start: Option<time::Instant> = None;
    let mut start = time::Instant::now();

    loop {
        let threshold = score_threshold(best_score.load(Ordering::Relaxed), capture_range);
        gpu.write_threshold(threshold).unwrap();
        gpu.reset_counter().unwrap();

        // Update GPU global_offset
        gpu.kernel.set_arg(6, global_offset).unwrap();

        // Dispatch
        gpu.compute().unwrap();

        let mut candidate_buf = Vec::with_capacity(CANDIDATES_BUF_BYTES);
        let count = gpu.read_candidates(&mut candidate_buf).unwrap();

        if count > 0 {
            let payload_end = CANDIDATES_HEADER + count * CANDIDATE_STRIDE;
            let payload = candidate_buf[..payload_end].to_vec();
            let _ = cand_tx.send((payload, count));

            if app_start.elapsed().as_secs() >= 10 {
                lifetime_candidates += count as u64;
            }
        }

        global_offset += threads as u64;
        iters += threads as u64;

        let elapsed = start.elapsed();
        if elapsed.as_secs() >= args.log_interval {
            if app_start.elapsed().as_secs() >= 10 && stats_start.is_none() {
                stats_start = Some(time::Instant::now());
            }

            let hashrate = iters as f64 / elapsed.as_secs_f64() / 1_000_000.0;
            let best_raw = best_score.load(Ordering::Relaxed);
            let best_display = if best_raw == i64::MIN {
                "none".to_string()
            } else {
                format!("{:.6}", fixed_to_score(best_raw))
            };

            let (cands_per_hour, eta_str) = if let Some(st) = stats_start {
                let active_secs = st.elapsed().as_secs_f64();
                if active_secs > 0.0 && lifetime_candidates > 0 {
                    let cps = lifetime_candidates as f64 / active_secs;
                    let cph = cps * 3600.0;
                    let eta_secs = 1.0 / cps;
                    let eta_s = if eta_secs > 86400.0 {
                        format!("{:.1}d", eta_secs / 86400.0)
                    } else if eta_secs > 3600.0 {
                        format!("{:.1}h", eta_secs / 3600.0)
                    } else if eta_secs > 60.0 {
                        format!("{:.1}m", eta_secs / 60.0)
                    } else {
                        format!("{:.0}s", eta_secs)
                    };
                    (cph, eta_s)
                } else {
                    (0.0, "N/A".to_string())
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
            start = time::Instant::now();
            iters = 0;
        }
    }

    #[allow(unreachable_code)]
    {
        drop(cand_tx);
        _cpu_thread.join().unwrap();
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
            handle_candidate(
                seed,
                pk,
                gpu_score,
                best_score,
                csv_file,
                capture_range,
                app_start,
            );
        });
}
