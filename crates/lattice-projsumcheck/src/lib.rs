//! # lattice-projsumcheck
//!
//! **The Sum-Check Protocol over the Monomial Basis** — a full Rust
//! implementation of ePrint 2026/762 (Dao, Biswas, Eagen, Milson,
//! Papini, Thaler).
//!
//! The paper's thesis: the Boolean hypercube `{0,1}^n` is not the optimal
//! interpolating set for sum-check. Switching to the **infinity hypercube
//! `{0,∞}^n`** — where evaluating a multilinear polynomial at `∞` means
//! extracting a monomial coefficient (Proposition 3.1) — yields a
//! near-drop-in variant with:
//!
//! * **subtraction-free binding** — `p(r, x') = p(0, x') + r·p(∞, x')`
//!   (Corollary 3.2), saving `d·(2^n − 1)` field subtractions per proof
//!   (Proposition 4.1) and roughly 10% end-to-end prover time on prime
//!   fields (§6.6), ~33% of binding cost in-circuit (§4.4);
//! * **cheaper structured polynomials** — `eq` collapses to
//!   `Π(1 + X_iY_i)`, `LT`'s decisive factor becomes `Y_j`, XOR becomes
//!   multiplication-free, and every `(1−Z)` factor turns into `1` or an
//!   ignored-variable factor `(1+Z)` (§4.2, Appendix A);
//! * **faster full-domain tables** — the `eq` doubling `(e, e·r)` with a
//!   free left half (1.94× on BN254) and a subtraction-free `LT`
//!   recurrence (§6.4);
//! * **claim-preserving dummy rounds** in batched sum-check — no
//!   `2^{n_max−n}` pre-scaling, no renormalization (§B.1/§B.2);
//! * **one-element-per-round proof compression** — the prover sends
//!   `{s(∞), s(1), …, s(d−1)}` and the verifier derives `s(0)` from the
//!   round identity `s(0) + s(∞) = C`;
//! * **PCS alignment** — monomial coefficients are what WHIR-style (and
//!   this codebase's compact Ajtai) commitments consume, eliminating the
//!   Möbius basis conversion entirely (§4.3);
//! * **upper-limb Montgomery challenges** for 256-bit fields — sampled
//!   from a 2^125 subset whose Montgomery form has zero low limbs,
//!   halving the native multiplications of CIOS (1.92× chained, §5),
//!   with grinding to close the security gap (§5.3).
//!
//! ## Module map
//!
//! * [`proj_mle`] — monomial-coefficient MLEs: coefficient extraction,
//!   projective binding (the SIMD kernel), Möbius transforms, the
//!   projective `eq` table.
//! * [`proj_sumcheck`] — the protocol (Figure 3): prove/verify, the
//!   `Ū_d = {∞} ∪ {1..d−1}` message points, Lemma 2.2 interpolation.
//! * [`tables`] — the structured-polynomial library: per-pair dictionary,
//!   `LT`/`eq`/`shift` closed forms and doubling recurrences, the Jolt
//!   table families (bitwise, comparisons, range/alignment, guards,
//!   XOR-rotations, byte reversal, `MulUNoOverflow`, `Pow2`) — every
//!   closed form Möbius-verified against its discrete table.
//! * [`dummy`] — batched projective sum-check with claim-preserving
//!   dummy rounds (§B).
//! * [`fp256`] — the 4-limb Montgomery CIOS field with the upper-limb
//!   challenge short-circuit and grinding (§5).
//!
//! ## Conventions
//!
//! Variable 0 is the most significant index bit (matching `DenseMle`);
//! coefficient arrays share the truth-table slot layout, so
//! `MonomialMle::from_truth_table` is an array identity.

#![deny(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod dummy;
pub mod fastprover;
pub mod fp256;
pub mod proj_mle;
pub mod proj_sumcheck;
pub mod tables;

pub use proj_mle::{MonomialMle, ProjMleError};
pub use proj_sumcheck::{
    ProjSumcheckError, ProjSumcheckOutput, ProjSumcheckProof, ProjSumcheckVerifier,
    ProjVirtualPolynomial,
};
