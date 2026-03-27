use std::fs;

pub fn main() {
    let mut s = fs::read_to_string("src/gpu.rs").unwrap();
    // Re-insert the closing brace for the struct impl and main run_opencl
    // Actually, it seems I messed up the run_opencl function end.
    // The easiest is to revert and re-apply.
    
    // Just appending the missing closing braces
    s.push_str("\n}\nfn format_eta(eta_secs: f64) -> String {\n    if eta_secs > 86400.0 {\n        format!(\"{:.1}d\", eta_secs / 86400.0)\n    } else if eta_secs > 3600.0 {\n        format!(\"{:.1}h\", eta_secs / 3600.0)\n    } else if eta_secs > 60.0 {\n        format!(\"{:.1}m\", eta_secs / 60.0)\n    } else {\n        format!(\"{:.0}s\", eta_secs)\n    }\n}\n");
    fs::write("src/gpu.rs", s).unwrap();
}
