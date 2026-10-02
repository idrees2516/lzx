//! # lattice-greyhound
//!
//! The **LaBRADOR** proof system (Beullens–Seiler, ePrint 2022/1341, CRYPTO
//! 2023) and the **Greyhound** polynomial commitment scheme (Nguyen–Seiler,
//! ePrint 2024/1293) implemented paper-faithfully over
//! `R_q = Z_q[X]/(X^64+1)`, `q = 2^32 − 99`.
//!
//! Module map (paper → code):
//! * `ring` — R_q arithmetic, digit decomposition, σ^{-1}, inverses (§2 of
//!   both papers).
//! * `challenge` — the challenge space C (32×±1 + 8×±2, op-norm rejection),
//!   quarternary/Z_q challenge distributions (LaBRADOR §2, Greyhound §5).
//! * `sis` — the Module-SIS estimator, the shared Ajtai key, the commitment
//!   parameter search (`comparams`), LIFTS, the JL norm bound q/125 (§5.4–§7,
//!   Theorem 5.1's rank conditions, Lemma 4.2).
//! * `transcript` — the Fiat-Shamir 16-byte hash chain.
//! * `relation` — the principal relation R: the F / F' families of quadratic
//!   dot-product constraints (LaBRADOR §5.1, Greyhound §2.4).
//! * `jl` — the modular Johnson–Lindenstrauss projection and its collapse
//!   (LaBRADOR §4, Lemma 4.1/4.2).
//! * `protocol` — the main protocol: Figure 2 prover / Figure 3 verifier, the
//!   LIFTS aggregation, amortization with g/h garbage, the §5.3 target
//!   relation E1–E6, the §5.6 tail with 2r−1 interleaved garbage.
//! * `recursion` — the recursive composition driver + the proof-size model
//!   (§5.7, the paper's Table 3).
//! * `greyhound` — the polynomial commitment: Setup/Commit/Open/Eval with the
//!   Z_q→R_q translation (Greyhound §4, Figure 4) and the R1 relation compile
//!   (§4.3).
//! * `r1cs` — LaBRADOR's R1CS reductions: binary R1CS (§6, Figure 4) and R1CS
//!   mod 2^64+1 with NAF encodings (§6, Figure 5), mixed.
//! * `batch` — Greyhound's multi-point/multi-poly batching (§3.2, Figure 2).
//! * `cwss` — the coordinate-wise special-soundness extractor (Lemma 3.2) and
//!   the weak-binding reduction (Lemma 2.11).
//! * `zk` — Greyhound's hiding commitment and HVZK evaluation variant (§4.5).
//! * `sizes` — the concrete parameter tables (Table 4) and the proof-size
//!   accounting reproducing the 53KB@2^30 claim.
//!
//! The honest deviation ledger lives in
//! `docs/papers/implemented/{labrador,greyhound}.md`.

// Upstream kernel structure: the digit layouts index several structures in
// lockstep (r[i], t[j], dec[rho], aug[row]) with position-dependent strides —
// the range-loop lint's iterator suggestions cannot express them. The
// patterns follow the ported reference (cf. lattice-labrador's identical
// allow).
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]

pub mod batch;
pub mod challenge;
pub mod cwss;
pub mod greyhound;
pub mod jl;
pub mod protocol;
pub mod r1cs;
pub mod recursion;
pub mod relation;
pub mod ring;
pub mod sis;
pub mod sizes;
pub mod transcript;
pub mod zk;

pub use relation::{DotCnst, PrincipalStatement, PrincipalWitness, Term, VectorSpec};
pub use ring::{Poly, Q, N};
