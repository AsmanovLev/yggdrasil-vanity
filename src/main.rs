use std::{fs, io::Write, sync::Mutex};

use clap::{Parser, ValueEnum};

mod cpu;
mod gpu;
mod handler;

use cpu::run_cpu;
use gpu::run_opencl;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
pub enum Preset {
    /// Optimise for height only (more leading zero bits).
    Height,
    /// Optimise for length only (shorter IPv6 string representation).
    Length,
    /// Optimise for both (formula: weight_h * height - weight_l * length).
    HeightLength,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Disable OpenCL (use CPU instead)
    #[arg(short = 'c', long, default_value_t = false)]
    pub disable_opencl: bool,

    /// OpenCL thread count.
    /// Default 4M is tuned for RTX 3060 Laptop (6 GB VRAM).
    /// Seeds: 128 MB, candidates buf: ~4.5 MB. Raise if GPU utilisation <90%.
    #[arg(short, long, default_value_t = 4 * 1024 * 1024)]
    pub threads: usize,

    /// OpenCL local work size (leave unset to let driver choose)
    #[arg(short, long)]
    pub local_work_size: Option<usize>,

    /// OpenCL platform index
    #[arg(short, long, default_value_t = 0)]
    pub platform_idx: usize,

    /// OpenCL device indices (comma-separated, e.g., "0,1"), or "all".
    #[arg(short, long, default_value = "all")]
    pub device_idx: String,

    /// List all available OpenCL devices and exit.
    #[arg(long, default_value_t = false)]
    pub list_devices: bool,

    /// Log hashrate every N seconds
    #[arg(short = 'i', long, default_value_t = 10)]
    pub log_interval: u64,

    /// CSV file to write results to
    #[arg(short = 'f', long)]
    pub csv_file: Option<String>,

    /// Scoring preset.
    /// 'height': Maximize leading zeros (original behavior)
    /// 'length': Minimize IPv6 string length
    /// 'height-length': Maximize weighted combination of both
    #[arg(long, value_enum, default_value_t = Preset::Height)]
    pub preset: Preset,

    /// Weight for height in 'height-length' preset (score = h*weight_h - l*weight_l)
    #[arg(long, default_value_t = 1.0)]
    pub weight_h: f32,

    /// Weight for length in 'height-length' preset (score = h*weight_h - l*weight_l)
    #[arg(long, default_value_t = 1.0)]
    pub weight_l: f32,

    /// Save results within this many score points below the current best.
    /// 0.0 = only new records. Higher = also save near-records.
    #[arg(long, default_value_t = 0.0)]
    pub capture_range: f64,

    /// Minimal height to consider (GPU-side filter).
    #[arg(long, default_value_t = 0)]
    pub minimal_height: u8,
}

fn main() {
    let child = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(main_)
        .unwrap();
    child.join().unwrap();
}

fn main_() {
    let args = Args::parse();

    eprintln!(
        "Preset: {:?}  |  capture_range: {}  |  threads: {}",
        args.preset, args.capture_range, args.threads,
    );

    let csv_file = args.csv_file.clone().map(|path| {
        let mut f = fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .unwrap();
        if f.metadata().unwrap().len() == 0 {
            f.write_all(b"timestamp,subnet,address,height,length,score,privatekey\n")
                .unwrap();
        }
        Mutex::new(f)
    });

    let app_start = std::time::Instant::now();
    if !args.disable_opencl {
        run_opencl(args, csv_file, app_start)
    } else {
        run_cpu(&args, &csv_file, app_start)
    }
}
