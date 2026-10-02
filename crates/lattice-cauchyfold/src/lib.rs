//! # lattice-cauchyfold — CauchyFold (ePrint 2026/2011), fully implemented
//! at the scaled profile
//!
//! **Paper**: Wang, *"CauchyFold: Residue-Optimal High-Arity Lattice
//! Folding via Scaled Cauchy Challenges"*.
//!
//! ## What is implemented
//!
//! * **The Cauchy carrier algebra** (§4 + Appendix A.3): poles/scales,
//!   the challenge family `a_i(c) = λ_i/(c − ξ_i)` with the
//!   partial-fraction identity, the denominator-cleared fold `W(T)`, the
//!   carrier `H_src(T)` with its `k` output-valued coefficients, the
//!   carrier identity (Prop 4.4), the fixed-carrier discrepancy `F(T)`
//!   with its `≤ 2k/|C|` soundness (Lemma 4.5), and the fast carrier
//!   construction (product trees + multipoint evaluation +
//!   interpolation) — differential-tested against the direct
//!   pair-processing form.
//! * **The exact-width boundary theory** (§4.1–4.4), executable: `Va`,
//!   the separation condition, `dim Va = k` (Theorem 4.1 / Lemma 4.2 /
//!   Corollary 4.3) pinned by exact `K`-linear algebra on the concrete
//!   challenge families — including the negative control (the power
//!   family does not attain the width).
//! * **The node protocol** (§5, Figure 1): the transcript order
//!   (source/carrier commitments before `c`, the output commitment
//!   after), the fold `(z*, E*)`, the field-level checks (random
//!   aggregation + one `K`-valued sum-check over the 19 root objects'
//!   digit cubes — Booleanity legs `x(x−1)=0`, the output's linear
//!   bindings to the claimed `Az*/Bz*/Cz*`, the carrier-evaluation leg,
//!   the public residual update), the root reduction (the level-2 digit
//!   witness `W`, the `ΓW = Y` system with the ring-structured
//!   commitment rows, the `R16` range polynomial, the §5.4 fingerprint
//!   sum-check), the finite linear reduction chain (§5.5 mechanics:
//!   projection with retries, symmetric `h_ij` before the challenge, the
//!   certified `D46` short challenges, the response identities
//!   (28)–(30), the radix-split children), and the §5.6 terminal with
//!   the fail-closed codec.
//! * **The extraction machinery** (§6 + Appendix C): the integral
//!   comparison (Lemma 6.2 — compare before clearing), the
//!   coordinate-replay extraction with the honest/cheating harness, and
//!   the statistical loss accounting (`ε_i`, `Λ_i`, `κ_node`, the
//!   fixed-vector projection bound).
//! * **The wire layer** (Appendix D.7): 6-byte `F_q` coefficients,
//!   24-byte `K` elements, 384-byte `R_{q,64}` elements, the terminal
//!   codec with every fail-closed decoder check.
//! * **The parameter profiles** (§7 + Appendix D): both paper profiles
//!   recorded declaratively with their exact numbers (not executed —
//!   57.5M-coefficient witnesses, ~15 GiB, 2.6 h runs), plus the
//!   executed scaled profile.
//!
//! ## The honest-deviation ledger (summary; the full note lives in
//! `docs/papers/implemented/cauchyfold.md`)
//!
//! 1. **Scale**: the executed profile is `k = 16` with `s = 4, y = 2`
//!    (~2.5K level-1 digits, ~7.5K level-2) versus the paper's
//!    57.5M-coefficient root witness. The paper's two profiles are
//!    recorded with their exact tables.
//! 2. **The root matrix**: one shared Ajtai root matrix for all 19
//!    objects (the paper samples per-object matrices); the
//!    γ-combination then rides one homomorphic system.
//! 3. **The chain**: 2 nonterminal layers + terminal (a profile
//!    parameter; the paper's 5+1 is its scale); the auxiliary digit
//!    commitments `u1 = B·t̃`, `u2 = D·h̃` are replaced by direct `h`
//!    transmission checked through identity (29); the inter-layer
//!    recomposition is verified at the terminal.
//! 4. **D46 certification**: float DFT with a conservative margin
//!    replaces the paper's exact rational interval test (the unit
//!    property — distinct challenges differ by units — holds exactly via
//!    the `‖Δ‖∞ ≤ 4 < √(q/2)` criterion).
//! 5. **The terminal codec**: fixed-width quotients replace the
//!    arithmetic-coded high frame; all fail-closed checks kept.
//!
//! Everything else — the carrier algebra, the boundary theory, the
//! protocol order, the layer mechanics, the extraction discipline — is
//! the paper's own construction.

// The lattice/cube kernels index tables by explicit bit arithmetic
// (MSB-first layouts, negacyclic folds); iterator rewrites cannot express
// the indexing — the same convention as lattice-labrador.
#![allow(clippy::needless_range_loop)]

pub mod boundary;
pub mod cauchy;
pub mod commit;
pub mod extract;
pub mod field_k;
pub mod node;
pub mod params;
pub mod reduce_chain;
pub mod sumcheck_k;
pub mod wire;

pub use boundary::{analyze, k4_rank, BoundaryAnalysis};
pub use cauchy::{
    carrier_identity_holds, discrepancy_poly, w_poly, Carrier, CauchyParams, QuadraticMap,
};
pub use commit::{
    AjtaiKey, Level1Encoding, Level2Encoding, split_radix16,
};
pub use field_k::{Fq48, K4, KPoly, Q48};
pub use node::{honest_witness, prove, verify, NodeError, NodeInstance, NodeParams, NodeProof, NodeWitness};
pub use reduce_chain::{
    digit_energy, radix_recompose, radix_split, sample_short_challenge, ChainError, ChainProof,
};
pub use wire::{decode_terminal, encode_terminal};
