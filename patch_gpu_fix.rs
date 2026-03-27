use std::fs;
pub fn main() {
    let mut s = fs::read_to_string("src/gpu.rs").unwrap();
    // Revert struct to usize
    s = s.replace("pub device_idx: Vec<usize>,", "pub device_idx: usize,");
    
    // Fix Gpu::new to use vec![opts.device_idx] again
    s = s.replace(".device(DeviceSpecifier::Indices(opts.device_idx))", ".device(DeviceSpecifier::Indices(vec![opts.device_idx]))");
    
    // Fix run_gpu_instance call: it should just pass device_idx (usize)
    s = s.replace("device_idx: vec![device_idx],", "device_idx: device_idx,");
    
    fs::write("src/gpu.rs", s).unwrap();
}
