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

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// Tests may use unwrap/expect/panic freely.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod challenge_set;
pub mod decomposition;
pub mod field;
pub mod keccak;
pub mod mle;
pub mod transcript;

pub use field::Goldilocks;
pub use mle::DenseMle;
pub use transcript::Transcript;
