//! # lattice-blindfold — LatticeBlindFold (ePrint 2026/1857)
//!
//! **Paper**: Dall'Ava, *"LatticeBlindFold: A Lattice-Based Analogue of
//! NovaBlindFold"*, ePrint 2026/1857 (ICME Labs).
//!
//! ## What the paper does
//!
//! Every lattice folding scheme beyond Nova itself (SuperNeo, LatticeFold(+),
//! Cyclo, ...) is only **randomizing**, not **blinding**: the folding
//! transcript leaks information about the witnesses being folded.
//! LatticeBlindFold is the first lattice-based, plausibly post-quantum
//! analogue of the NovaBlindFold protocol: it makes SuperNeo **blinding**
//! (honest-verifier zero-knowledge) by three devices:
//!
//! 1. **Libra-style polynomial masking** of the Sum-Check transcript
//!    (`p = a0 + Σ p̃_i(X_i)`, §3.4): unlike Libra, the mask is *never
//!    opened* — the final check happens entirely at the level of the
//!    ABDLOP commitments, so `p(r')` and `Q(r')` never appear in the
//!    verifier's view as field elements.
//! 2. **ABDLOP commit-and-prove** replacing SuperNeo's plaintext evaluation
//!    hints: equality checks are performed homomorphically at the level of
//!    the commitments. Since ABDLOP is only secure over the base ring R_F
//!    and not over R_K where the Sum-Check lives, a **componentwise
//!    instantiation** (§3.3) commits to the {1,Y}-coordinates of R_K
//!    elements as R_F elements (the rank-doubling embedding) and translates
//!    every R_K-relation into a pair of R_F-relations once, at parameter
//!    generation.
//! 3. **Rejection sampling** so the randomized folded instance-witness pair
//!    is simulatable — forcing the decomposition depth k = Θ(log n_F).
//!
//! The protocol is `Π_LBF = Π'_DEC ∘ Π'_RLC ∘ Π'_R1CS` (Protocol 12): a
//! single interactive folding step taking uncommitted R1CS instances in
//! (over the blinded layout + unpadded relation) and outputting `k`
//! committed evaluation claims `CE_com(b, L, B̃, Commit, T, K)` together
//! with the ABDLOP openings certifying them. It is complete
//! (statistically, ≤ 9·2^−λ), knowledge-sound (MSIS/MLWE over cyclotomic
//! rings + Extended-MLWE), and blinding for K = 1 (the verifier's entire
//! view, aborted rejection-sampling attempts included, is simulatable from
//! the public instances alone).
//!
//! ## What this crate implements
//!
//! Every required part of the paper, over the repo's lattices (Ajtai/SIS
//! stack), with the honest-deviation ledger in
//! `docs/papers/implemented/latticeblindfold.md`:
//!
//! * the field/ring tower `F_q → K = F_q2 → R_F → R_K` with the Solinas
//!   prime q = 2^64 − 59 (q ≡ 5 mod 8, ν = 2 the fixed non-residue),
//!   negacyclic arithmetic, the τ_ℓ rotation basis (Lemma 3.12), degree-0
//!   elements (Lemma 2.20), strong sampling sets (Def 2.13) and the
//!   b-ary decomposition `split_b`;
//! * the **blinded R1CS layout** (Def 3.1) with the statistical-hiding
//!   analysis of the compact Ajtai commitment `L` on the blinding block
//!   (Lemma 3.3, both regimes);
//! * the **ABDLOP commitment** (§2.5.2) with messages in BDLOP slots and
//!   R_K messages committed componentwise (§3.3: linear relations, the
//!   quadratic rank-doubling matrices R̂^(a), R̂^(b) of Eq (3.9), the norm
//!   convention of §3.3.4, and the K-valued challenge-masking soundness of
//!   §3.3.5 with its norm-form determinant);
//! * the **zk-fying stack** (§3): hiding for Ajtai (§3.1), the companion
//!   procedures from the ABDLOP proofs of knowledge (§3.2) — Π_many^(1)
//!   (Lemma 3.6), Π_many^(2) (Lemma 3.7), the Π_many^(ct) wrapper
//!   (Protocol 4, Lemma 3.8), the salt/message substitution (Lemma 3.10),
//!   the concatenated instance Π_anc (Def 3.15, Lemma 3.16) — Libra-style
//!   masking (§3.4), the ABDLOP-route PCS replacement (§3.5) and the
//!   rejection sampling recall (§3.6, Lemma 3.20 with Rej1/Rej2 and the
//!   width calibration of Eq (4.18));
//! * the **masked Sum-Check** over K with degree-Dmax rounds (Protocol 5's
//!   shape, §4.1.1.1's perfect-masking property);
//! * the **three reductions** (§4.1): Π'_R1CS (Protocol 6, all 18 steps),
//!   Π'_RLC (Protocol 7, the rejection-sampled mask loop with the
//!   Cy,0/Cy,j commitments and the Wmax attempt budget), Π'_DEC
//!   (Protocol 8, fresh-salt decomposition with the batched proof of
//!   knowledge);
//! * **Π_LBF** (Protocol 12) with the samplers of Protocols 9/10/11, the
//!   ι_bl precomposition (Def 4.6), the **accumulator-free variant**
//!   Π°_LBF (Corollary 4.24) and the folding blueprint (Protocol 13);
//! * the **simulators** (Protocols 1-3 hybrid chain S0 → S1 → S_ABDLOP and
//!   the transcript-level simulator for the blinding property);
//! * the **error-budget calculator** (Propositions 4.15/4.17, Theorem
//!   4.13's ε_LBF-blind) and the parameter tables of §4.3.4;
//! * **adversarial harnesses**: two-transcript extraction with the MSIS
//!   kernel recovery, tampering rejection at every protocol layer, and
//!   statistical tests of the blinding claims.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
// The repo convention (cf. lattice-accordion): index-based loops read
// better for the cube/ring-coordinate arithmetic.
#![allow(clippy::needless_range_loop)]

pub mod abdlop;
pub mod ajtai;
pub mod embed;
pub mod fp;
pub mod fq2;
pub mod gauss;
pub mod params;
pub mod pok;
pub mod protocol;
pub mod ring;
pub mod rk;
pub mod sumcheck;

pub use params::{Params, SecurityBudget};
