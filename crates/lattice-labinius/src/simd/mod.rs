//! The AVX-512 backend of the ported labinius PCS — upstream's kernel *designs*, ported as pure
//! `std` intrinsics (no `asm!` blocks, no external crates), gated on runtime CPU detection with
//! the exact scalar reference paths as fallback.
//!
//! What is here, and where it comes from:
//! * [`transpose`] — the GFNI bit-slicing front end (`BinaryIndex32`): 128 `F162` (= 32 ring
//!   elements) turned into the `vpermb` byte-index rows the binary kernels consume, in four
//!   phases (qword transposes, GFNI bit transposes + 8-way interleave, plane gather, one
//!   two-row affine per pair).
//! * [`ntt_small`] — the split-tree forward NTT for binary inputs, primes `3889`/`9721`
//!   (vertical batch-of-32 layout, byte-split 16-entry lookup tables fusing levels 0-2 and the
//!   level-3 twiddles, signed-Montgomery 3-uop twiddle multiplies, radix-3 butterflies, fused
//!   levels 5+6 in registers, lazy reduction with a per-level bound table).
//! * [`ntt_quad`] — the quadratic-slot tree, primes `2917`/`4861`/`12637` (the splitting tree
//!   with its second radix-2 level removed, ending in 324 `Z_q[X]/(X^2 - psi'^u)` leaves; the
//!   level-3 twiddles fold into the lookup tables for 2917/4861, 12637 keeps the unfolded
//!   phase 1 with lookup-Barrett reductions).
//! * [`commit`] — the Ajtai commitment MAC: `vpmaddwd` raw accumulation of slot products that
//!   are **never individually reduced**, exact compile-time fold-back periods, a packed
//!   two-slot accumulator, the `hsum8` three-stage lane fold and an f64-exact `mod_q` finish;
//!   the quadratic leaves carry three sums per leaf combined with a Karatsuba trick.
//!
//! Every kernel is verified against the scalar reference (`tests/simd.rs`): outputs agree
//! modulo `q` with `scalar::ntt` / `scalar::ntt_quad` of the same lift, and every lane respects
//! the declared bound. On a CPU without the feature set, or on a non-x86-64 target, none of
//! this code runs and the scalar paths answer.
//!
//! Not ported from upstream's SIMD layer (documented gaps, mapped in the crate README):
//! `bin_asm` (hand-scheduled `asm!` transform — the pure-intrinsics reference here is the same
//! tree at a less aggressive schedule) and the `bd`/`norm` SIMD helpers.

pub mod commit;
pub mod gen_large;
pub mod gen_quad;
pub mod gen_small;
pub mod ntt_large;
pub mod ntt_quad;
pub mod ntt_small;
pub mod slots;
pub mod transpose;

use crate::params::N;

/// One batch of 32 ring elements in the vertical layout the kernels write and read:
/// `v[j][p]` is slot `j` of ring element `p`, one 64-byte vector per slot.
///
/// This is upstream's `ring::element::Batch32` minus the representation tag (the port's
/// transforms are explicit about which domain they are in).
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Batch32 {
    pub v: [[i16; 32]; N],
}

impl Batch32 {
    pub fn zero() -> Self {
        Batch32 { v: [[0; 32]; N] }
    }
}

/// Is the AVX-512 PCS backend usable on this machine?
pub fn available() -> bool {
    crate::hw::avx512_pcs()
}
