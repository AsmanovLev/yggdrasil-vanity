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

pub fn run_opencl(args: Args, csv_file: Option<Mutex<fs::File>>, app_start: std::time::Instant) {
    let platforms = Platform::list();
    let platform = platforms
        .get(args.platform_idx)
        .expect("Invalid platform index");
    let all_devices = ocl::Device::list_all(platform).unwrap_or_default();

    let device_indices = if args.device_idx == "all" {
        (0..all_devices.len()).collect::<Vec<_>>()
    } else {
        args.device_idx
            .split(',')
            .map(|s| s.trim().parse::<usize>().expect("Invalid device index"))
            .collect::<Vec<_>>()
    };

    let best_score = Arc::new(AtomicI64::new(i64::MIN));
    let (cand_tx, cand_rx) = mpsc::sync_channel(device_indices.len() * 2);

    let mut gpu_threads = Vec::new();
    let args_arc = Arc::new(args);
    for dev_idx in device_indices {
        let args = args_arc.clone();
        let best_score = best_score.clone();
        let cand_tx = cand_tx.clone();
        gpu_threads.push(thread::spawn(move || {
            run_gpu_instance(&args, dev_idx, best_score, cand_tx, app_start);
        }));
    }
    drop(cand_tx);

    let csv_arc = Arc::new(csv_file);
    let capture_range = args_arc.capture_range;
    let cpu_thread = thread::spawn(move || {
        for (buf, count) in cand_rx {
            process_candidates(&buf, count, &best_score, &csv_arc, capture_range, app_start);
        }
    });

    for t in gpu_threads {
        t.join().unwrap();
    }
    cpu_thread.join().unwrap();
}
fn run_gpu_instance(
    args: &Args,
    device_idx: usize,
    best_score: Arc<AtomicI64>,
    cand_tx: mpsc::SyncSender<(Vec<u8>, usize)>,
    app_start: std::time::Instant,
) {
    let mode = match args.preset {
        Preset::Height => 0,
        Preset::Length => 1,
        Preset::HeightLength => 2,
    };
    let mut gpu = Gpu::new(
        GpuOptions {
            platform_idx: args.platform_idx,
            device_idx: device_idx,
            threads: args.threads,
            local_work_size: args.local_work_size,
        },
        mode,
        args.weight_h,
        args.weight_l,
        args.minimal_height,
    )
    .unwrap();

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
                let _ = cand_tx.send((
                    candidate_buf[..CANDIDATES_HEADER + count * CANDIDATE_STRIDE].to_vec(),
                    count,
                ));
            }
        }
        global_offset += args.threads as u64;
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
