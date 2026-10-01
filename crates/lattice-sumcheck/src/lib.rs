//! # lattice-sumcheck
//!
//! The generic multilinear sumcheck engine shared by every protocol layer
//! (Jolt-style claim DAG stages, Akita evaluation proofs, SALSA norm
//! sumchecks, RoKoko projections, folding-scheme cross terms).
//!
//! * `virtual_poly` — virtual polynomials: sums of products of MLEs with
//!   coefficients (the product structure every staged relation compiles to).
//! * `sumcheck` — the prover/verifier pair with Fiat–Shamir challenges,
//!   compressed round polynomials, and final-claim binding to PCS
//!   evaluations (the opening layer owns actual polynomial commitments).
//! * `zerocheck` — Spartan-style zero test: prove P ≡ 0 over the hypercube
//!   via an eq-multiplied sumcheck.
//! * `batch` — random-linear-combination batching of multiple sumcheck
//!   claims into one invocation (SALSA batched-verification lineage).

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod batch;
pub mod extrapolate;
pub mod fastprover;
pub mod multiproduct;
pub mod sumcheck;
pub mod virtual_poly;
pub mod zerocheck;

pub use sumcheck::{SumcheckError, SumcheckProof, SumcheckVerifier};
pub use virtual_poly::{VirtualPolyError, VirtualPolynomial};
