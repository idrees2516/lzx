//! # lattice-core
//!
//! Foundation crate for the LZX lattice-based post-quantum zkVM:
//! * `field` — 64-bit Goldilocks prime field with batch inversion.
//! * `keccak` — in-house Keccak-f\[1600\] permutation + SHA3/SHAKE sponge.
//! * `transcript` — Fiat–Shamir transcript with domain separation.
//! * `mle` — dense multilinear extensions (evaluations over the boolean hypercube).
//! * `decomposition` — balanced gadget digit decomposition used by lattice
//!   commitments and folding schemes to control norm growth.
//! * `challenge_set` — challenge sampling with rejection (audit report §6.4:
//!   bounded, counted, domain-separated challenge generation).
//! * `short_challenge` — paper-calibrated short **ring-element** challenge
//!   distributions (fixed-weight ternary / biased ternary / small sets)
//!   with certified operator-norm bounds Γ_C and rejection (Wave 6 §2.1).
//! * `norm_budget` — symbolic folding norm budgets with hard wraparound
//!   gates `β < min(q/2, β*)` (Wave 6 §2.4).
//! * `extension` — F_{p²} arithmetic over Goldilocks with transcript
//!   sampling (Wave 6 §2.3: the F_{q^e} sumcheck substrate).
//! * `field_simd` — packed Goldilocks AVX-512 kernels (8 lanes per
//!   `__m512i`, Plonky2-style packed mul + lazy reduction), runtime-gated
//!   with the exact scalar paths as fallback (`LZX_NO_SIMD=1` disables).

// `field_simd` holds the crate's only `unsafe`: core::arch intrinsics behind
// runtime CPU detection, exactly the lattice-labinius `hw.rs` doctrine. Every
// other module keeps the forbid-level guarantee; the deny here exists so the
// carve-out can be scoped to that one module (an inner `#![allow]` cannot
// relax a `forbid`).
#![deny(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// Tests may use unwrap/expect/panic freely.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod challenge_set;
pub mod decomposition;
pub mod extension;
pub mod field;
pub mod field_simd;
pub mod keccak;
pub mod mle;
pub mod norm_budget;
pub mod short_challenge;
pub mod transcript;

pub use field::Goldilocks;
pub use mle::DenseMle;
pub use transcript::Transcript;
