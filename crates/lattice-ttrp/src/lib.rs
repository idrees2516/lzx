//! # lattice-ttrp
//!
//! **Succinct Shortness Check Under a Few Kilobytes via Tensor Train
//! Random Projections** — ePrint 2026/2146 (Geng & Plançon).
//!
//! TTRP is a structured Johnson–Lindenstrauss projection whose matrix is
//! a tensor train: k rows, each the chain product of µ small core tensors
//! of internal rank c with entries from `D_ghl`. It gives:
//!
//! * a **succinct representation** — `O(k·d·µ·c²)` core entries instead
//!   of the `O(λ·m)` unstructured JL matrix of LaBRADOR (Table 1's
//!   6492 MB → 0.2 MB);
//! * a **projection length linear in λ** (k = O(λ) rows), unlike RoKoko's
//!   structured JL whose projection must itself be committed;
//! * a **slack factor** `B̃/B = √(k(c/2)^{µ−1}/θ)` (polynomial, Theorem 4).
//!
//! The crate implements, in depth:
//!
//! * [`cores`] — core tensors, the `χ_TT` sampler, the TT row
//!   materialisation and its mixed-product chain (Fact 1);
//! * [`projection`] — both computation paths of the projection: the
//!   integer contraction `y0 = M_Z·cf(v) mod q` (Lemma 6's prover
//!   algorithm) and the ring view `y = M·v̄` via the spatial/coefficient
//!   chain split, tied together by the power-of-two cyclotomic identity
//!   `ct(a·b̄) = ⟨cf(a), cf(b)⟩`;
//! * [`bounds`] — the moment bounds (Lemma 3), modular overflow bounds
//!   (Lemma 4 / Theorem 2), Cantelli ℓ₂ concentration (Lemma 5 /
//!   Theorem 3), the completeness bound B̂, the slack and no-overflow
//!   ceilings, and the §7.2 parameter search;
//! * [`protocol`] — the `Π_TTRP` reduction of knowledge (Figure 1 +
//!   Corollary 1): the projection, the `Γ`-aggregation with the
//!   constant-term identity, the γ-batched degree-2 ring sumcheck, the
//!   terminal evaluation claim, and the verifier's O(k·µ₁·c²·d)
//!   tensor-structured MLE evaluation.
//!
//! Integration posture: the output evaluation claim `(conj(r), w_r)` is
//! appended to the caller's `Ξ_poly` relation (the claim-ledger pattern
//! shared with the zkvm pipeline); the Ajtai commitment layer stays with
//! the caller.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod bounds;
pub mod cores;
pub mod projection;
pub mod protocol;

pub use cores::{sample_cores, CoreTensor, TtrpCoreError, TtrpParams};
pub use protocol::{
    mle_eval_ring, prove, verify, TtrpError, TtrpProof, TtrpStatement, TtrpVerified,
};
