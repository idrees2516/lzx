//! # lattice-pcd — Zero-Knowledge PCD from Accumulation Schemes (ePrint 2026/289)
//!
//! From-scratch implementation of Zheng–Gao–Liu (PolyU HK), "Zero-Knowledge
//! Proof-Carrying Data from Accumulation Schemes":
//!
//! * **`fp_base`** — the BN254 *base* field `F_p` (CIOS Montgomery, 4×u64),
//!   the coordinate field of the commitment group. The *protocol* field is
//!   BN254 `F_r`, reused from `lattice_projsumcheck::fp256::Fp256`.
//! * **`g1`** — BN254 G1 (`y² = x³ + 3`, Jacobian coordinates, windowed
//!   scalar multiplication, try-and-increment hashing; cofactor 1 so the
//!   on-curve check is a subgroup check).
//! * **`pedersen`** — vector Pedersen commitments `Com(v; r) = Σ vᵢ·Gᵢ + r·H`:
//!   homomorphic over `F_r`-scalars, statistically hiding (random `r`),
//!   binding via discrete log. This is the instantiation the paper itself
//!   assumes ("conducted over cyclic groups for simplicity", §5 complexity).
//! * **`sps`** — the special-sound-protocol framework of §5.1: the
//!   committed-message NARK `FS[Π_sps^cm]` with the homogeneous algebraic
//!   verifier map `V_sps = Σ_k f_k^V`, instantiated for (i) R1CS (d=2, µ=1),
//!   (ii) CCS (d=q, µ=1 — the high-degree headline case), and (iii) the
//!   grand-product permutation check (d=n, µ=2 — a challenge-dependent map).
//! * **`zk_sumcheck`** — the CFS17/XZZ+19 zero-knowledge sum-check (the
//!   `O(d·m)` mask `g = r₀ + Σ rᵢ(Xᵢ)` of Eq. (3)) plus the [KS24]
//!   point-update sum-check that re-randomizes old `G(β)` claims to a fresh
//!   point, and the simulator for the ZK property.
//! * **`accum`** — §5.2's zk-Protogalaxy: the zero-knowledge accumulation
//!   scheme for special-sound NARKs (masking vector for the accumulator
//!   witness, eq-interpolated `F(X)` over `2m+1` parties, the masked batched
//!   sum-check, the error commitment `E`, and the decider).
//! * **`nark`** — the non-interactive argument `FS[Π_sps^cm]` itself
//!   (`G, I, P, V`).
//! * **`pcd`** — §4.2's ZK-PCD construction: the recursive circuit split
//!   `R^(0) = R_φ` (predicate, witness-carrying) and `R^(1) = R_V`
//!   (accumulation verification, public), the two accumulators
//!   `acc = (acc^(0), acc^(1))`, and the final verifier `b₀ ∧ b₁ ∧ b₂`.
//!
//! The interpretation notes and honest deviations are collected in
//! `docs/papers/implemented/zk-pcd-accumulation.md`.

// Index-arithmetic loops (party/coordinate bookkeeping through the
// interpolation lattice) read clearer with explicit indices.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![deny(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod accum;
pub mod ajtai_fr;
pub mod fp_base;
pub mod g1;
pub mod nark;
pub mod pcd;
pub mod pedersen;
pub mod pq;
pub mod sps;
pub mod util;
pub mod zk_sumcheck;

pub use fp_base::FpBase;
pub use g1::{G1Affine, G1Point};
pub use pedersen::{PedersenCommitment, PedersenKey};

/// BN254 scalar field (the protocol field), re-exported for convenience.
pub use lattice_projsumcheck::fp256::Fp256;
