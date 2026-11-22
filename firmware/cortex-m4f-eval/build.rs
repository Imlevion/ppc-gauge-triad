use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR missing"));
    let src = if env::var_os("CARGO_FEATURE_QEMU").is_some() {
        "memory_mps2.x"
    } else {
        "memory_stm32f411.x"
    };

    fs::copy(src, out.join("memory.x")).expect("failed to copy linker memory script");
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory_stm32f411.x");
    println!("cargo:rerun-if-changed=memory_mps2.x");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_QEMU");
}
