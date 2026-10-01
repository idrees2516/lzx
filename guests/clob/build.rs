// Host-side build script: pass the RISC-V linker script (and keep it simple
// on host builds so `cargo test --target x86_64-unknown-linux-gnu` works).
use std::env;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    if target == "riscv64imac-unknown-none-elf" {
        let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let ld = manifest.join("riscv.ld");
        println!("cargo:rustc-link-arg=-T{}", ld.display());
    }
    println!("cargo:rerun-if-changed=riscv.ld");
    println!("cargo:rerun-if-changed=build.rs");
}
