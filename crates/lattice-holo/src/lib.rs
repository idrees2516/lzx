//! # lattice-holo — Proof-Carrying Data via Holography Accumulation
//! (ePrint 2026/538)
//!
//! From-scratch implementation of Paslis–Ràfols–Zacharakis:
//!
//! * **`poly`** — the two polynomial representations the paper's protocols
//!   run over (`ν = 1` univariate on the roots-of-unity domain `H` with the
//!   vanishing polynomial `u_H`, Lagrange bases `λ_h`, and the identity
//!   polynomial `Λ(X,Y) = (u_H(X)Y − u_H(Y)X)/(n(X−Y))`; `ν = log n`
//!   multivariate on the boolean cube with the eq bases and
//!   `Λ(X,Y) = ∏(XᵢYᵢ + (1−Xᵢ)(1−Yᵢ))`), plus matrix polynomials
//!   `M(X,Y) = λ(Y)ᵀ M λ(X)`.
//! * **`relations`** — the paper's relation family with satisfaction
//!   checks: `R_PCE` (polynomial commitment evaluation), `R_PCEP`
//!   (evaluation proofs), `R_hbPCE` (holographic bivariate evaluations),
//!   `R_CCS`, and the central **`R_GBF`** generalized bilinear forms
//!   relation with its `R_GBF,α` / `R_GBF,α,β` specializations
//!   (Definitions 4–8).
//! * **`pc`** — the polynomial commitment instantiation: vector Pedersen
//!   over BN254 G1 (homomorphic, statistically hiding), with evaluation
//!   proofs as **linear openings** (the transparent long-opening regime —
//!   the paper's PC abstraction is a drop-in boundary; swapping in an
//!   IPA/KZG-style short opening is local to `pc::open`/`verify`), plus
//!   the batch evaluation proof scheme (`Π_provePCE`, `Π_batchPCEP`).
//! * **`sumcheck`** — the batched sum-check machinery for both
//!   representations: multivariate rounds over the boolean cube (per-round
//!   degree `dl + dr`), and the univariate domain sum-check via the paper's
//!   `h₁/h₂` decomposition `q(X) = s/n + X·h₁(X) + u_H(X)·h₂(X)` for
//!   degree-`> n` polynomials (§2/§4's Figures).
//! * **`gbf1`** — `Π_GBF1`: the Marlin-style reduction (commit the
//!   intermediate vectors `d_{jM,jv} = M_{jM} v_{jv}`, one sum-check).
//! * **`gbf2`** — `Π_GBF2`: the Spartan-style reduction (two sum-checks),
//!   plus the early-stopping variant (the HyperNova-style linearized
//!   committed CCS).
//! * **`batch`** — `Π_batchM`: the linear-combination reduction
//!   `R_hbPCE → R_GBF,α,β` (Lemma 1).
//! * **`collapse`** — `Π_Collapse`: `R_CCS → R_GBF,α` (Lemma 2).
//! * **`barebones`** — the composed argument
//!   `(Π_provePCE × ID) ∘ (ID × Π_batchM) ∘ Π_GBF,α ∘ Π_Collapse`
//!   (Theorem 7) — recovering SuperMarlin (ν=1) and SuperSpartan (ν=log n)
//!   as instantiations.
//! * **`fold`** — `Π_Fold`: the many-to-one holography accumulation
//!   `R*_Acc → R_Acc` (Theorem 8) — the paper's core contribution.
//! * **`decider`** — the non-uniform PCD decider (§5.3): one linear
//!   combination of the `ℓ·t_M` matrix commitments and a single
//!   evaluation check.
//! * **`pcd`** — the PCD construction: NI-Barebones as the argument and
//!   NI-Π_Fold as the accumulation scheme (Corollary 3), for
//!   constant-depth compliance predicates.
//!
//! The interpretation notes and honest deviations are collected in
//! `docs/papers/implemented/holography-pcd.md`.

#![deny(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
// Index-arithmetic loops (domain/coordinate bookkeeping) read clearer with
// explicit indices — the fp256 doctrine.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod batch;
pub mod barebones;
pub mod collapse;
pub mod decider;
pub mod fold;
pub mod gbf1;
pub mod gbf2;
pub mod pcd;
pub mod pc;
pub mod poly;
pub mod relations;
pub mod sumcheck;

/// The protocol field: BN254 `F_r`.
pub use lattice_pcd::Fp256;
