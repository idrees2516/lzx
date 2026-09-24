//! # lattice-ring
//!
//! Negacyclic polynomial rings `R_q = Z_q[X] / (X^n + 1)` with a complete
//! number-theoretic transform, the arithmetic substrate of every lattice
//! scheme in the workspace (Akita commitments, folding schemes, SALSA,
//! RoKoko projections).
//!
//! * `modulus` — u32 prime moduli supporting negacyclic NTTs, with
//!   Barrett-reduced hot-path arithmetic (Wave 6).
//! * `ring` — `RingElement` with schoolbook and NTT multiplication.
//! * `ntt` — complete negacyclic NTT (psi = 2n-th root of unity) with
//!   precomputed tables, forward/inverse, plus RoKoko-style *incomplete*
//!   NTT levels.
//! * `module` — module vectors/matrices over R_q (Module-SIS instances).
//! * `packing` — CRT / coefficient packing helpers shared by tensor
//!   embeddings (Akita tensor.rs lineage).
//! * `extension` — the quadratic extension ring `R_q[Y]/(Y² + 1)` with
//!   Karatsuba multiplication and short-challenge sampling (Wave 6 §2.3).
//! * `modulus50` — the ~2^50 prime modulus class (q ≡ 129 mod 256) with
//!   incomplete-NTT quadratic slots (Wave 6 §2.2).

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod extension;
pub mod modulus;
pub mod modulus50;
pub mod ntt;
pub mod packing;

pub use packing::PackingError;
pub mod ring;

pub use modulus::Modulus32;
pub use ring::{RingConfig, RingElement, RingError};
