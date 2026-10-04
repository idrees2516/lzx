//! # lattice-guest
//!
//! Jolt-style guest programs for the `lattice-vm` RV64IM kernel, plus the
//! assembler that builds them and the pure-std Rust reference models that
//! pin their semantics.
//!
//! Three layers:
//!
//! * [`asm`] — a two-pass RV64IM assembler (labels, `.word`/`.byte`/
//!   `.data`/`.org`, every instruction the VM decodes, GAS-ish text
//!   front-end `assemble_str`). Every encoding round-trips through
//!   `lattice_vm::decode` (see `asm::tests::decode_table`).
//! * [`programs`] — benchmark guest programs written in that assembly,
//!   each one a real algorithm that terminates with `ecall` and leaves
//!   its result in `a0`/`a1` (hash-shaped programs spread the extra
//!   words across `a2..a5` and a digest area in memory).
//! * [`mod@reference`] — independent pure-std Rust models of every program
//!   (u64 wrapping arithmetic mirroring the VM exactly) with NIST KATs
//!   for SHA-256 and Keccak/SHA3.
//!
//! # a16z/jolt lineage
//!
//! The program set models the guest benchmarks of the a16z Jolt zkVM
//! (jolt-guests / the `jolt-core` benchmark suite):
//!
//! | program | models |
//! |---|---|
//! | `fibonacci` | jolt `fibonacci` guest (u64 wrapping add loop) |
//! | `sha2_256` | jolt `sha2` guest — full SHA-256, own padding, K in data |
//! | `sha3_keccak` | jolt `keccak`/`sha3` guest — keccak-f\[1600\] permutation |
//! | `matrix_mul` | jolt `matrix_multiplication` guest (i64, checksummed) |
//! | `sorting` | jolt `batched_insertion_sort` guest |
//! | `merkle_tree` | jolt `merkle_tree` guest (SHA-2 compression as the 2-to-1 hash) |
//! | `modinv` | jolt `batched_modular_inverse` guest (here mod 2^61-1, binary ext-Euclid) |
//! | `muldiv` | jolt `mul_div128`-style 128-bit mul / 128÷64 divide via `mulh`/`div` |
//! | `collatz` | the classic Collatz-range zkVM benchmark |
//! | `memory_ops` | pointer-chase + strided-sweep memory benchmark |
//! | `regex` | jolt `regex` guest — hand-coded DFA over the public input |
//! | `modexp` | the secp256k1-ecdsa-verify analog: batched square-and-multiply |
//! |   | modular exponentiation (the dominant cost of ECDSA verify) |
//!
//! # Running
//!
//! `suite()` returns every program at its default size;
//! `programs::run_program` loads code at 0, public input at `0x1000`,
//! runs to `ecall`, and checks the result against the reference.
//! `cycle_estimates()` exposes the measured step counts (the "cycles"
//! a proof system would pay for each guest).
//!
//! Cycle budget note: the small/medium programs sit well under 2^14
//! steps; the large hash-heavy variants (sha2-256 8 blocks, sha3 10
//! permutations, 64-leaf merkle, 64×modinv/modexp, collatz-1000-range)
//! are intrinsically above it — 512 SHA-2 rounds alone cost ~24k
//! instructions — and their exact measured counts are reported by
//! `cycle_estimates` rather than clamped.
//!
//! NOTE (GUEST-1 state): `asm` is fully implemented and tested (every
//! encoding round-trips `lattice_vm::decode`); `programs` and
//! `reference` are placeholder stubs to be filled by the next agent.
//! `lattice-vm` is currently a *dev*-dependency (the assembler itself is
//! pure std); a public program runner will need it promoted to a real
//! dependency.

#![forbid(unsafe_code)]
#![allow(
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::double_parens
)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod asm;
pub mod programs;
pub mod reference;

/// Address at which the public input is loaded (VM convention).
pub const PUBLIC_INPUT_BASE: u64 = 0x1000;
