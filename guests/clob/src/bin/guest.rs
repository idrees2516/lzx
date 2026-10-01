//! lzx zkVM guest binary — Guest A (perp CLOB kernel), RV64IMAC, no_std.
//!
//! Memory map (see riscv.ld): .data/.bss at 0x0000, stack at 0x0C00
//! (sp := 0x0FC0 from the entry stub), .text at 0x1000 (ENTRY(_start)),
//! INPUT region at 0x1C00 (64 u64 words, read with volatile loads),
//! OUTPUT region at 0x1E00 (64 u64 words, written with volatile stores).
//! The VM halts on the final `ecall`.
//!
//! The real guest is built for `riscv64imac-unknown-none-elf`. When this bin
//! is compiled for the host (cargo test builds every target), it degenerates
//! to a std stub so host test runs stay green — the riscv64 build path is the
//! one that produces the zkVM image and is 100% `core`-only.

#![cfg_attr(target_arch = "riscv64", no_std)]
#![cfg_attr(target_arch = "riscv64", no_main)]

#[cfg(target_arch = "riscv64")]
mod guest {
    use clob_guest::{run_kernel, ClobState, Stats, INPUT_WORDS, OUTPUT_WORDS};

    const INPUT_BASE: u64 = 0x1C00;
    const OUTPUT_BASE: u64 = 0x1E00;

    // .bss state (the flat loader zero-fills; ClobState::zeros() == all-zero).
    static mut INPUT_BUF: [u64; INPUT_WORDS] = [0; INPUT_WORDS];
    static mut OUTPUT_BUF: [u64; OUTPUT_WORDS] = [0; OUTPUT_WORDS];
    static mut STATE: ClobState = ClobState::zeros();
    static mut STATS: Stats = Stats::zeros();

    // Entry stub: the FIRST instruction at 0x1000 sets sp; gp is anchored
    // before any small-data access; then the Rust kernel runs; `ecall` halts
    // the VM. `.option norelax` keeps the linker from rewriting gp bootstrap.
    core::arch::global_asm!(
        ".section .text._start, \"ax\"",
        ".globl _start",
        ".type _start, @function",
        "_start:",
        "   li      sp, 0xfc0",
        ".option push",
        ".option norelax",
        "   la      gp, __global_pointer$",
        ".option pop",
        "   call    {kernel}",
        "   ecall",
        ".size _start, . - _start",
        kernel = sym kernel_main,
    );

    /// Volatile-copy the INPUT region, run the kernel, volatile-dump the
    /// OUTPUT region. All accesses are 8-byte aligned u64 loads/stores.
    #[no_mangle]
    pub extern "C" fn kernel_main() {
        unsafe {
            let mut i: u64 = 0;
            while i < INPUT_WORDS as u64 {
                let addr = (INPUT_BASE + i * 8) as *const u64;
                INPUT_BUF[i as usize] = core::ptr::read_volatile(addr);
                i += 1;
            }

            let input: &[u64] = core::slice::from_raw_parts(
                core::ptr::addr_of!(INPUT_BUF) as *const u64,
                INPUT_WORDS,
            );
            let output: &mut [u64] = core::slice::from_raw_parts_mut(
                core::ptr::addr_of_mut!(OUTPUT_BUF) as *mut u64,
                OUTPUT_WORDS,
            );
            run_kernel(
                input,
                output,
                &mut *core::ptr::addr_of_mut!(STATE),
                &mut *core::ptr::addr_of_mut!(STATS),
            );

            let mut j: u64 = 0;
            while j < OUTPUT_WORDS as u64 {
                let addr = (OUTPUT_BASE + j * 8) as *mut u64;
                core::ptr::write_volatile(addr, OUTPUT_BUF[j as usize]);
                j += 1;
            }
        }
    }

    /// panic = abort (no unwinding, no formatting — the kernel is written to
    /// be panic-free: all array accesses are bounded by validated counts).
    #[panic_handler]
    fn panic(_info: &core::panic::PanicInfo) -> ! {
        loop {}
    }
}

#[cfg(not(target_arch = "riscv64"))]
fn main() {
    println!(
        "clob guest: host stub — the zkVM image is built with \
         `cargo build --release --target riscv64imac-unknown-none-elf`"
    );
}
