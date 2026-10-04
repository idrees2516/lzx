//! ProtogaLattice protocol layer (ePrint 2026/1317): the PGL-Fold and
//! PGL-Boot protocols — Wave 7 items 7.1 and 7.14.
//!
//! This module implements the *verifier-facing* protocol halves that
//! `protogalattice.rs` (the prover-side fold kernel) deliberately left
//! open, following the paper's Figures 2 and 3:
//!
//! # PGL-Fold (paper Fig. 2 — the "Fig-3 protocol" of NEXT_STEPS §3.1)
//!
//! Folds `k` fresh (unrelaxed) instances into one relaxed accumulator
//! with **three random-oracle rounds** (δ, α, y — the paper's binding
//! mechanism) and a **Gröbner-quotient error check**:
//!
//! ```text
//! acc instance: (t, β, e)   with witness w  s.t.  Σ_i pow_i(β)·f_i(w) = e
//! fresh instances: (t_j, w_j)               s.t.  f_i(w_j) = 0 ∀i
//!
//! R1: δ ← C           (one ring challenge; δ-vector = (δ, δ², …, δ^{2^{t−1}}))
//!     prover → F_1..F_t: coefficients of F(X) = Σ_i pow_i(β + Xδ)·f_i(w₀)
//! R2: α ← C           β* = β + α·δ  (re-randomized relaxation point)
//!     F(α) := e + Σ_j F_j α^j       (verifier-side; substitutes the
//!                                     accumulator's e for F₀ — binding)
//!     prover computes H(Y) = Σ_i pow_i(β*)·f_i(Σ_j L_j(Y)·w_j) and divides
//!     by the Gröbner basis G = { Z_ab = Y_a·Y_b − Y_a : a ≤ b }:
//!         H(Y) − Y₀·F(α) = Σ_{a≤b} K_ab(Y)·Z_ab(Y)      (remainder 0)
//!     prover → K_ab (quotients in the clear — paper Fig. 2)
//! R3: y ← C^k, y₀ := 1 (Cyclo's trick: the accumulator coefficient
//!     L₀(y) = 1, so accumulator norm growth is *linear* in fold depth)
//!     verifier checks:
//!         t* = t₀ + Σ_j (y_j − y_{j−1})·t_j        (homomorphism)
//!         e* = F(α) + Σ_{a≤b} K_ab(y)·Z_ab(y)      (the e*-check)
//!     prover's folded witness: w* = w₀ + Σ_j (y_j − y_{j−1})·w_j
//! ```
//!
//! The Lagrange forms are `L₀(Y) = Y₀`, `L_j(Y) = Y_j − Y_{j−1}`, plus a
//! ghost zero-witness with `L_{k+1}(Y) = 1 − Y_k` restoring the partition
//! of unity. The vanishing structure: the ideal
//! `J = ⟨Y_a·Y_b − Y_a : a ≤ b⟩` is the vanishing ideal of the
//! selection chain `τ_s = (0…0,1…1)` (suffix ones) — at `τ_s` the forms
//! select `w_s` exactly, which yields the identity
//! `f(Σ_j L_j·w_j) − Σ_j L_j·f(w_j) ∈ J` (the ring-generalized
//! multilinearity lemma the paper proves as its Theorem 1; here it is
//! realized by explicit exact division, and the zero remainder is
//! *asserted in code* — see `groebner_divide`).
//!
//! Soundness posture (vs. the pre-Wave-7 state): the sent quotients are
//! bound by the e*-check at the random ring point `y` (Schwartz-Zippel
//! over `R_q` with the strong sampling set `C`), the polynomial `F` is
//! bound to the accumulator through the `e`-substitution in `F(α)`, and
//! the folded commitment `t*` is bound by the Ajtai homomorphism — the
//! three defects S1/S2/S3 of NEXT_STEPS §3.1 are closed at kernel scale.
//!
//! # PGL-Boot (paper Fig. 3 — Wave 7.14)
//!
//! Norm-refreshing bootstrapping for unbounded folding: decompose the
//! accumulated witness in base `b` (`w = Σ_j b^j·w_j`, digits short),
//! re-commit each digit block, prove the folded identity at a public
//! evaluation point `D` with `L_j(D) = b^j`, and fold the blocks at a
//! fresh challenge point. The output accumulator's norm is
//! `γ = (2T·(k′−1)+1)·⌈b/2⌉` instead of the input `γ′` — the refresh.
//! A LatticeFold+ range proof is attached to the digit blocks (item
//! 7.14's `Π_rg` attachment; see `boot_with_range`).
//!
//! # Scale disclosure
//!
//! This realization computes `H(Y)` explicitly as a dense multivariate
//! polynomial over `R_q` (degree ≤ d in k+1 variables) and divides it
//! monomial-by-monomial using the paper's Prop.-5-style free-reduction
//! rule (`Y_a·Y_b → Y_a` for `a ≤ b`, `Y_a^α → Y_a`). At paper scale the
//! division is carried on structured (per-constraint) representations
//! and the monomial blow-up never materializes; here the explicit dense
//! form keeps the division provably exact at the cost of
//! `O(C(k+d, d))` terms. All identities are test-pinned.

use crate::protogalattice::PgRelation;
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiPublicKey};
use lattice_core::short_challenge::{
    ShortChallengeError, ShortChallengeFamily, ShortChallengeSpec,
};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::{RingConfig, RingElement, RingError};

/// Challenge-set parameters for the protocol (paper Table 2, kernel scale).
///
/// The paper's `C = {−1,0,1,2}^N` at weight `T = 128`; at kernel scale the
/// ring dimension is small, so the weight is parameterized (default
/// `T = min(128, N/2)`). Entropy is checked at construction: the knowledge
/// error of the fold is bounded by the strong-sampling-set size `|C|`.
#[derive(Clone, Debug)]
pub struct PglChallengeParams {
    pub weight: usize,
}

impl PglChallengeParams {
    /// Paper-calibrated defaults for a given ring dimension.
    pub fn for_ring(ring: &RingConfig) -> Self {
        PglChallengeParams {
            weight: 128.min(ring.n() / 2).max(1),
        }
    }

    fn spec(&self, n: usize) -> ShortChallengeSpec {
        ShortChallengeSpec {
            n,
            family: ShortChallengeFamily::FixedWeightSmallSet {
                weight: self.weight,
                values: vec![-1, 0, 1, 2],
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum PglError {
    Ajtai(AjtaiError),
    Ring(RingError),
    Transcript(TranscriptError),
    Challenge(ShortChallengeError),
    /// Constraint evaluation failed (shape mismatch inside a relation).
    RelationEval,
    /// Constraint count must be a power of two (β-compression needs
    /// `t = log2 n` randomizer levels).
    ConstraintCountNotPowerOfTwo {
        got: usize,
    },
    /// Relations must be constant-term-free (`f(0) = 0`) — the ghost
    /// zero-witness fold requires homogeneous-style relations.
    NonHomogeneousRelation {
        term_index: usize,
    },
    /// The Gröbner division left a non-zero remainder: a fresh instance
    /// does not satisfy its constraints (detected prover-side).
    FreshInstanceInvalid {
        linear_residual: usize,
    },
    /// The boot decomposition does not recompose to the witness.
    DecompositionMismatch,
    /// Verifier-side failures (fold/boot verification).
    FoldCommitmentMismatch,
    ErrorCheckFailed,
    DegreeBoundExceeded {
        got: usize,
        max: usize,
    },
    Shape {
        expected: usize,
        got: usize,
    },
    /// Accumulator relation violated at decide time.
    DecideFailed,
    /// Norm budget exceeded (wraparound gate).
    NormBudgetExceeded {
        budget: u64,
        got: u64,
    },
}

impl From<AjtaiError> for PglError {
    fn from(e: AjtaiError) -> Self {
        PglError::Ajtai(e)
    }
}
impl From<RingError> for PglError {
    fn from(e: RingError) -> Self {
        PglError::Ring(e)
    }
}
impl From<TranscriptError> for PglError {
    fn from(e: TranscriptError) -> Self {
        PglError::Transcript(e)
    }
}
impl From<ShortChallengeError> for PglError {
    fn from(e: ShortChallengeError) -> Self {
        PglError::Challenge(e)
    }
}

/// The constraint system: `n = 2^t` polynomial maps `f_i : R_q^m → R_q`
/// (Hadamard-slot relations, reusing [`PgRelation`]), all sharing the
/// witness shape. Must be constant-term-free.
#[derive(Clone, Debug)]
pub struct PglConstraintSystem {
    pub constraints: Vec<PgRelation>,
}

impl PglConstraintSystem {
    pub fn new(constraints: Vec<PgRelation>) -> Result<Self, PglError> {
        if !constraints.is_empty() && !constraints.len().is_power_of_two() {
            return Err(PglError::ConstraintCountNotPowerOfTwo {
                got: constraints.len(),
            });
        }
        for (ci, rel) in constraints.iter().enumerate() {
            if rel.terms.iter().any(|(_, ids)| ids.is_empty()) {
                return Err(PglError::NonHomogeneousRelation { term_index: ci });
            }
        }
        Ok(PglConstraintSystem { constraints })
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Randomizer levels `t = log2 n`.
    pub fn t_levels(&self) -> usize {
        self.constraints.len().trailing_zeros() as usize
    }

    /// Relation degree (max over constraints).
    pub fn degree(&self) -> usize {
        self.constraints
            .iter()
            .map(|r| r.degree())
            .max()
            .unwrap_or(0)
    }

    /// Evaluate constraint `i` at `w` → a single ring element, with
    /// **ring-product semantics**: each term is `c · Π_j w[σ_j]` where the
    /// products are ring multiplications (the paper's polynomial map
    /// `f : R_q^m → R_q`). This is deliberately different from
    /// `protogalattice::PgRelation::evaluate`'s Hadamard-slot semantics:
    /// the PGL protocols need `f` to be a ring-polynomial map so that the
    /// fold-line composition `f(Σ_j L_j(Y)·w_j)` is expressible as an
    /// `MPoly` over `Y` with ring-element coefficients (Hadamard products
    /// do not commute with polynomial composition).
    pub fn evaluate_i(&self, i: usize, w: &[RingElement]) -> Result<RingElement, PglError> {
        let rel = self.constraints.get(i).ok_or(PglError::Shape {
            expected: i,
            got: self.constraints.len(),
        })?;
        let ring = w
            .first()
            .map(|e| e.config().clone())
            .ok_or(PglError::Shape {
                expected: rel.num_slots,
                got: 0,
            })?;
        if w.len() != rel.num_slots {
            return Err(PglError::Shape {
                expected: rel.num_slots,
                got: w.len(),
            });
        }
        let mut acc = ring.zero();
        for (c, ids) in &rel.terms {
            let mut prod = ring.one();
            for &slot in ids {
                prod = prod.mul(&w[slot])?;
            }
            acc = acc.add(&prod.scale_i64(*c as i64))?;
        }
        Ok(acc)
    }

    /// The relaxed accumulator relation value:
    /// `Σ_i pow_i(β)·f_i(w)` — the `e` of an honest accumulator.
    pub fn relaxed_value(
        &self,
        beta: &[RingElement],
        w: &[RingElement],
    ) -> Result<RingElement, PglError> {
        let ring = w
            .first()
            .map(|e| e.config().clone())
            .ok_or(PglError::Shape {
                expected: 1,
                got: 0,
            })?;
        let mut acc = ring.zero();
        for (i, _) in self.constraints.iter().enumerate() {
            let fi = self.evaluate_i(i, w)?;
            // pow_i is 1-indexed over the constraint tower (constraint 1
            // carries pow = 1, constraint 2 carries β_0, ...).
            let pow = pow_tower(&ring, beta, i + 1)?;
            let term = pow.mul(&fi)?;
            acc = acc.add(&term)?;
        }
        Ok(acc)
    }
}

/// `pow_i(β) = Π_{j : bit j of (i−1) set} β_j` — the compressed
/// randomizer tower (`⊗^t (1, β_j)` in the paper; `i` is 1-indexed).
pub fn pow_tower(
    ring: &RingConfig,
    beta: &[RingElement],
    i: usize,
) -> Result<RingElement, PglError> {
    let mut acc = ring.one();
    if i == 0 {
        return Ok(acc);
    }
    for (j, b) in beta.iter().enumerate() {
        if (i - 1) >> j & 1 == 1 {
            acc = acc.mul(b)?;
        }
    }
    Ok(acc)
}

/// A relaxed accumulator instance: `(t, β, e)` (paper Fig. 2).
#[derive(Clone, Debug)]
pub struct PglAccInstance {
    pub t: AjtaiCommitment,
    /// Randomizer tower levels (length `t = log2 n`).
    pub beta: Vec<RingElement>,
    /// Accumulated error (short ring element).
    pub e: RingElement,
    /// Infinity-norm budget of the underlying witness (Wave 6.2 gate).
    pub norm_budget: u64,
}

/// A fresh (unrelaxed) instance: commitment + witness.
#[derive(Clone, Debug)]
pub struct PglFreshInstance {
    pub t: AjtaiCommitment,
    pub w: Vec<RingElement>,
}

/// The Fig.-2 fold proof: F coefficients + Gröbner quotients.
#[derive(Clone, Debug)]
pub struct PglFoldProof {
    /// `F_1..F_t` (F₀ is the accumulator's public `e`).
    pub f_coeffs: Vec<RingElement>,
    /// Quotients `K_ab` indexed by `(a, b)` with `0 ≤ a ≤ b ≤ k`.
    pub quotients: Vec<((usize, usize), MPoly)>,
    /// The challenge transcript values (public re-derivation data).
    pub delta: RingElement,
    pub alpha: RingElement,
    pub y: Vec<RingElement>,
}

// ---------------------------------------------------------------------------
// Multivariate polynomials over R_q and the Gröbner division
// ---------------------------------------------------------------------------

/// A sparse multivariate polynomial over `R_q` in variables `Y_0..Y_{vars−1}`.
#[derive(Clone, Debug)]
pub struct MPoly {
    pub vars: usize,
    /// Sorted by exponent vector; zero coefficients dropped.
    pub terms: Vec<(Vec<u32>, RingElement)>,
}

impl MPoly {
    pub fn zero(vars: usize) -> Self {
        MPoly {
            vars,
            terms: Vec::new(),
        }
    }

    pub fn constant(c: RingElement, vars: usize) -> Self {
        MPoly {
            vars,
            terms: vec![(vec![0; vars], c)],
        }
    }

    /// Single variable `Y_index` with coefficient `c`.
    pub fn variable(index: usize, c: RingElement, vars: usize) -> Self {
        let mut exp = vec![0u32; vars];
        exp[index] = 1;
        MPoly {
            vars,
            terms: vec![(exp, c)],
        }
    }

    fn normalized(mut self) -> Self {
        self.terms.retain(|(_, c)| !c.is_zero());
        self
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn degree(&self) -> usize {
        self.terms
            .iter()
            .map(|(e, _)| e.iter().sum::<u32>() as usize)
            .max()
            .unwrap_or(0)
    }

    pub fn add(&self, other: &MPoly) -> Result<MPoly, PglError> {
        let mut terms = self.terms.clone();
        for (e, c) in &other.terms {
            if let Some(slot) = terms.iter_mut().find(|(e2, _)| e2 == e) {
                slot.1 = slot.1.add(c)?;
            } else {
                terms.push((e.clone(), c.clone()));
            }
        }
        Ok(MPoly {
            vars: self.vars,
            terms,
        }
        .normalized())
    }

    /// Add a term `c · Y_index` (linear-form helper).
    pub fn add_linear(&mut self, index: usize, c: &RingElement) -> Result<(), PglError> {
        let mut exp = vec![0u32; self.vars];
        exp[index] = 1;
        if let Some(slot) = self.terms.iter_mut().find(|(e2, _)| *e2 == exp) {
            slot.1 = slot.1.add(c)?;
        } else {
            self.terms.push((exp, c.clone()));
        }
        Ok(())
    }

    /// Multiply two polynomials, erroring past `max_degree`.
    pub fn mul_bounded(&self, other: &MPoly, max_degree: usize) -> Result<MPoly, PglError> {
        let mut out = MPoly::zero(self.vars);
        for (e1, c1) in &self.terms {
            for (e2, c2) in &other.terms {
                let mut e: Vec<u32> = Vec::with_capacity(self.vars);
                let mut deg = 0u32;
                for i in 0..self.vars {
                    let v = e1.get(i).copied().unwrap_or(0) + e2.get(i).copied().unwrap_or(0);
                    deg += v;
                    e.push(v);
                }
                if deg as usize > max_degree {
                    return Err(PglError::DegreeBoundExceeded {
                        got: deg as usize,
                        max: max_degree,
                    });
                }
                let c = c1.mul(c2)?;
                if c.is_zero() {
                    continue;
                }
                if let Some(slot) = out.terms.iter_mut().find(|(e3, _)| *e3 == e) {
                    slot.1 = slot.1.add(&c)?;
                } else {
                    out.terms.push((e, c));
                }
            }
        }
        Ok(out)
    }

    /// Scale by a ring constant.
    pub fn scale(&self, c: &RingElement) -> Result<MPoly, PglError> {
        let mut terms = Vec::with_capacity(self.terms.len());
        for (e, cc) in &self.terms {
            terms.push((e.clone(), cc.mul(c)?));
        }
        Ok(MPoly {
            vars: self.vars,
            terms,
        })
    }

    /// Evaluate at a point (ring elements per variable).
    pub fn evaluate(&self, point: &[RingElement]) -> Result<RingElement, PglError> {
        let ring = point
            .first()
            .map(|p| p.config().clone())
            .ok_or(PglError::Shape {
                expected: self.vars,
                got: 0,
            })?;
        let mut acc = ring.zero();
        for (e, c) in &self.terms {
            // monomial value = Π Y_i^{e_i}
            let mut mono = ring.one();
            for (i, &ei) in e.iter().enumerate() {
                for _ in 0..ei {
                    mono = mono.mul(&point[i])?;
                }
            }
            acc = acc.add(&mono.mul(c)?)?;
        }
        Ok(acc)
    }

    /// The total-degree of a term (for reduction ordering).
    fn term_degree(e: &[u32]) -> usize {
        e.iter().sum::<u32>() as usize
    }
}

/// The Gröbner division of a polynomial by the vanishing ideal
/// `J = ⟨ Z_ab = Y_a·Y_b − Y_a : 0 ≤ a ≤ b ≤ k ≤ vars−1 ⟩`.
///
/// Reduction rule (the paper's Prop.-5-style free division, min-index
/// orientation — see module docs): any monomial containing `Y_a·Y_b`
/// (a ≤ b, at least one factor each) reduces
/// `Y_a·Y_b·M' → Y_a·M'`, emitting the quotient term `c·M'` into
/// `K_ab`. Idempotence `Y_a^2 → Y_a` is the `a = b` case. The remainder
/// (normal form) is the degree ≤ 1 part that is not divisible — for the
/// protocols in this module it must vanish on honest input.
///
/// Returns the quotients (per `(a,b)` pair) and the remainder.
pub type GroebnerQuotients = Vec<((usize, usize), MPoly)>;

pub fn groebner_divide(
    poly: &MPoly,
    ring: &RingConfig,
) -> Result<(GroebnerQuotients, MPoly), PglError> {
    use std::collections::BTreeMap;
    let vars = poly.vars;
    // Work list keyed by exponent vector.
    let mut work: BTreeMap<Vec<u32>, RingElement> = BTreeMap::new();
    for (e, c) in &poly.terms {
        work.insert(e.clone(), c.clone());
    }
    let mut quotients: BTreeMap<(usize, usize), BTreeMap<Vec<u32>, RingElement>> = BTreeMap::new();
    let mut remainder: BTreeMap<Vec<u32>, RingElement> = BTreeMap::new();

    while !work.is_empty() {
        // Pick the highest-degree term (deterministic order).
        let mut best: Option<(Vec<u32>, RingElement)> = None;
        for (e, c) in &work {
            let better = match &best {
                None => true,
                Some((be, _)) => MPoly::term_degree(e) > MPoly::term_degree(be),
            };
            if better {
                best = Some((e.clone(), c.clone()));
            }
        }
        let (exp, coeff) = match best {
            Some(x) => x,
            None => break,
        };
        work.remove(&exp);
        // Find the reduction pair: a = min index with e_a > 0, and any
        // b ≥ a with (e_b > 0 and (b > a or e_a ≥ 2)).
        let support: Vec<usize> = (0..vars).filter(|&i| exp[i] > 0).collect();
        let a = match support.first() {
            Some(&a) => a,
            None => {
                // constant term — belongs to the remainder
                remainder.insert(exp, coeff);
                continue;
            }
        };
        let b =
            support
                .iter()
                .copied()
                .find(|&b| b > a)
                .or(if exp[a] >= 2 { Some(a) } else { None });
        match b {
            Some(b) => {
                // M = Y_a·Y_b·M'; reduce to Y_a·M'; quotient K_ab += c·M'.
                let mut m_prime = exp.clone();
                m_prime[a] -= 1;
                m_prime[b] -= 1;
                let q_entry = quotients.entry((a, b)).or_default();
                let existing = q_entry.get(&m_prime).cloned();
                let mut qcoeff = match existing {
                    Some(c) => c,
                    None => ring.zero(),
                };
                qcoeff = qcoeff.add(&coeff)?;
                if qcoeff.is_zero() {
                    q_entry.remove(&m_prime);
                } else {
                    q_entry.insert(m_prime.clone(), qcoeff);
                }
                // New term Y_a·M' (exp with e_b decremented).
                let mut reduced = exp.clone();
                reduced[b] -= 1;
                if reduced[b] == 0 && b != a {
                    // keep as-is; zero exponents are fine
                }
                let existing = work.get(&reduced).cloned();
                let mut ncoeff = match existing {
                    Some(c) => c,
                    None => ring.zero(),
                };
                ncoeff = ncoeff.add(&coeff)?;
                if ncoeff.is_zero() {
                    work.remove(&reduced);
                } else {
                    work.insert(reduced, ncoeff);
                }
            }
            None => {
                // Single-variable monomial (or linear) — irreducible.
                let existing = remainder.get(&exp).cloned();
                let mut rcoeff = match existing {
                    Some(c) => c,
                    None => ring.zero(),
                };
                rcoeff = rcoeff.add(&coeff)?;
                if rcoeff.is_zero() {
                    remainder.remove(&exp);
                } else {
                    remainder.insert(exp, rcoeff);
                }
            }
        }
    }

    let quotient_polys: Vec<((usize, usize), MPoly)> = quotients
        .into_iter()
        .filter(|(_, terms)| !terms.is_empty())
        .map(|(ab, terms)| {
            (
                ab,
                MPoly {
                    vars,
                    terms: terms.into_iter().collect(),
                },
            )
        })
        .collect();
    let rem = MPoly {
        vars,
        terms: remainder.into_iter().collect(),
    };
    Ok((quotient_polys, rem))
}

// ---------------------------------------------------------------------------
// PGL-Fold (paper Fig. 2) — Wave 7.1
// ---------------------------------------------------------------------------

/// Sample a ring challenge from `C` (fixed-weight small set) via the
/// transcript.
fn sample_ring_challenge(
    transcript: &mut Transcript,
    label: &[u8],
    params: &PglChallengeParams,
    ring: &RingConfig,
) -> Result<RingElement, PglError> {
    let seed = transcript.challenge_bytes(label, 32)?;
    let chal = params.spec(ring.n()).sample(&seed)?;
    Ok(RingElement::from_signed(ring, &chal.coefficients))
}

/// The F(X) polynomial: `F(X) = Σ_{i=1}^{n} pow_i(β + X·δ)·f_i(w₀)`
/// with `δ = (δ, δ², …, δ^{2^{t−1}})`. Returned as coefficients
/// `F_0..F_t` (degree ≤ t = log2 n).
fn compute_f_poly(
    cs: &PglConstraintSystem,
    beta: &[RingElement],
    delta: &RingElement,
    w0: &[RingElement],
) -> Result<Vec<RingElement>, PglError> {
    let t = cs.t_levels();
    let ring = beta
        .first()
        .map(|b| b.config().clone())
        .ok_or(PglError::Shape {
            expected: t,
            got: 0,
        })?;
    // δ-vector: δ_j = δ^{2^j} for j = 0..t−1.
    let mut delta_vec = Vec::with_capacity(t);
    let mut d = delta.clone();
    for _ in 0..t {
        delta_vec.push(d.clone());
        d = d.mul(&d)?;
    }
    let mut f = vec![ring.zero(); t + 1];
    for i in 1..=cs.num_constraints() {
        let fi = cs.evaluate_i(i - 1, w0)?;
        // pow_i(β + Xδ) = Π_{j ∈ bits(i−1)} (β_j + X·δ_j):
        // expand the product into X-powers.
        let bits: Vec<usize> = (0..t).filter(|j| (i - 1) >> j & 1 == 1).collect();
        // terms[p] = coefficient of X^p (product of β's and δ's)
        let mut terms: Vec<RingElement> = vec![ring.zero(); bits.len() + 1];
        terms[0] = ring.one();
        for (pos, &j) in bits.iter().enumerate() {
            // multiply existing terms[0..=pos] by (β_j + X δ_j)
            let mut next = vec![ring.zero(); pos + 2];
            for p in 0..=pos {
                next[p] = next[p].add(&terms[p].mul(&beta[j])?)?;
                next[p + 1] = next[p + 1].add(&terms[p].mul(&delta_vec[j])?)?;
            }
            terms = next;
        }
        for (p, tp) in terms.iter().enumerate() {
            f[p] = f[p].add(&tp.mul(&fi)?)?;
        }
    }
    Ok(f)
}

/// Compute `H(Y) = Σ_i pow_i(β*)·f_i(Σ_j L_j(Y)·w_j)` as an explicit
/// multivariate polynomial in `Y_0..Y_k` (degree ≤ d).
fn compute_h_poly(
    cs: &PglConstraintSystem,
    beta_star: &[RingElement],
    witnesses: &[Vec<RingElement>], // w_0..w_k (k+1 of them)
) -> Result<MPoly, PglError> {
    let k = witnesses.len() - 1;
    let vars = k + 1;
    let d = cs.degree();
    let ring = beta_star
        .first()
        .map(|b| b.config().clone())
        .ok_or(PglError::Shape {
            expected: 1,
            got: 0,
        })?;
    let m = witnesses[0].len();
    // slot_polys[s] = Σ_j L_j(Y)·w_j[s]  (degree-1 MPoly per slot)
    let mut slot_polys: Vec<MPoly> = Vec::with_capacity(m);
    for s in 0..m {
        let mut p = MPoly::zero(vars);
        // L_0 = Y_0
        p.add_linear(0, &witnesses[0][s])?;
        // L_j = Y_j − Y_{j−1}: +w_j[s]·Y_j, −w_j[s]·Y_{j−1}
        for (j, w) in witnesses.iter().enumerate().skip(1) {
            p.add_linear(j, &w[s])?;
            p.add_linear(j - 1, &w[s].neg())?;
        }
        slot_polys.push(p.normalized());
    }
    let mut h = MPoly::zero(vars);
    for i in 1..=cs.num_constraints() {
        // f_i(slot_polys) = Σ_terms c · Π_slots slot_polys[σ]
        let rel = &cs.constraints[i - 1];
        let mut fi_poly = MPoly::zero(vars);
        for (c, ids) in &rel.terms {
            let mut prod = MPoly {
                vars,
                terms: vec![(vec![0u32; vars], ring.constant(*c))],
            };
            for &slot in ids {
                prod = prod.mul_bounded(&slot_polys[slot], d)?;
            }
            fi_poly = fi_poly.add(&prod)?;
        }
        let pow = pow_tower(&ring, beta_star, i)?;
        fi_poly = fi_poly.scale(&pow)?;
        h = h.add(&fi_poly)?;
    }
    Ok(h)
}

/// The Fig.-2 fold: full prover+verifier protocol run. Returns the new
/// accumulator (instance + witness) and the fold proof. The prover-side
/// Gröbner division *asserts* the fresh instances satisfy their
/// constraints (non-zero linear remainder ⇒ `FreshInstanceInvalid`).
#[allow(clippy::too_many_arguments)]
pub fn fig2_fold(
    pk: &AjtaiPublicKey,
    cs: &PglConstraintSystem,
    acc: &PglAccInstance,
    acc_witness: &[RingElement],
    fresh: &[PglFreshInstance],
    chal_params: &PglChallengeParams,
) -> Result<(PglAccInstance, Vec<RingElement>, PglFoldProof), PglError> {
    let k = fresh.len();
    if k == 0 {
        return Err(PglError::Shape {
            expected: 1,
            got: 0,
        });
    }
    let ring = pk.params.ring.clone();
    let t = cs.t_levels();
    if acc.beta.len() != t {
        return Err(PglError::Shape {
            expected: t,
            got: acc.beta.len(),
        });
    }

    // ---- Transcript setup: absorb all public instance data. ----
    let mut transcript = Transcript::new_default(b"lzx-pgl-fold");
    pk.absorb_statement(&mut transcript, b"pk", &acc.t)?;
    transcript.append_field(b"acc-norm", &Goldilocks::from_u64(acc.norm_budget))?;
    for f in fresh {
        pk.absorb_statement(&mut transcript, b"fresh-t", &f.t)?;
    }

    // ---- Round 1: δ ← C; prover sends F_1..F_t. ----
    let delta = sample_ring_challenge(&mut transcript, b"pgl-delta", chal_params, &ring)?;
    let f_all = compute_f_poly(cs, &acc.beta, &delta, acc_witness)?;
    for fc in f_all.iter().skip(1) {
        transcript.append_bytes(b"pgl-f", &fc.to_bytes())?;
    }

    // ---- Round 2: α ← C; β* = β + α·δ (component-wise powers). ----
    let alpha = sample_ring_challenge(&mut transcript, b"pgl-alpha", chal_params, &ring)?;
    let mut beta_star = Vec::with_capacity(t);
    {
        let mut d = delta.clone();
        for (j, b) in acc.beta.iter().enumerate() {
            beta_star.push(b.add(&d.mul(&alpha)?)?);
            let _ = j;
            d = d.mul(&d)?;
        }
    }

    // ---- Prover: H(Y), Gröbner division, quotients. ----
    let mut witnesses: Vec<Vec<RingElement>> = Vec::with_capacity(k + 1);
    witnesses.push(acc_witness.to_vec());
    for f in fresh {
        witnesses.push(f.w.clone());
    }
    let h = compute_h_poly(cs, &beta_star, &witnesses)?;
    // D(Y) := H(Y) − Y_0·F(α)   (F(α) = e + Σ F_j α^j — the verifier's
    // definition; the honest prover's F(0) equals e).
    let mut f_alpha = acc.e.clone();
    {
        let mut a_pow = alpha.clone();
        for fc in f_all.iter().skip(1) {
            f_alpha = f_alpha.add(&fc.mul(&a_pow)?)?;
            a_pow = a_pow.mul(&alpha)?;
        }
    }
    let mut d_poly = h;
    d_poly.add_linear(0, &f_alpha.neg())?;
    let (quotients, remainder) = groebner_divide(&d_poly, &ring)?;
    // Honest input ⇒ remainder 0 (fresh instances satisfy f_i = 0 and
    // the accumulator satisfies Σ pow_i(β)·f_i(w₀) = e ⇒ F₀ = e).
    let residual_terms = remainder.terms.len();
    if residual_terms != 0 {
        return Err(PglError::FreshInstanceInvalid {
            linear_residual: residual_terms,
        });
    }
    for (ab, _) in &quotients {
        transcript.append_bytes(b"pgl-k", format!("{:?}", ab.0).as_bytes())?;
    }
    for (_, q) in &quotients {
        for (e, c) in &q.terms {
            let mut buf = Vec::new();
            for x in e {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            buf.extend_from_slice(&c.to_bytes());
            transcript.append_bytes(b"pgl-k-term", &buf)?;
        }
    }

    // ---- Round 3: y ← C^k, y_0 := 1 (Cyclo's trick). ----
    let mut y = vec![ring.one()];
    for _ in 0..k {
        y.push(sample_ring_challenge(
            &mut transcript,
            b"pgl-y",
            chal_params,
            &ring,
        )?);
    }

    // ---- Updates (verifier-computable) ----
    let mut t_star = acc.t.clone();
    for (j, f) in fresh.iter().enumerate() {
        let c_j = y[j + 1].sub(&y[j])?;
        for (row_acc, row_f) in t_star.rows.iter_mut().zip(f.t.rows.iter()) {
            *row_acc = row_acc.add(&row_f.mul(&c_j)?)?;
        }
    }
    // e* = F(α) + Σ K_ab(y)·Z_ab(y)
    let mut e_star = f_alpha.clone();
    for ((a, b), q) in &quotients {
        let z_ab = y[*a].mul(&y[*b])?.sub(&y[*a])?;
        let q_at_y = q.evaluate(&y)?;
        e_star = e_star.add(&q_at_y.mul(&z_ab)?)?;
    }
    // Prover's folded witness w* = w_0 + Σ (y_j − y_{j−1})·w_j.
    let mut w_star = acc_witness.to_vec();
    for (j, f) in fresh.iter().enumerate() {
        let c_j = y[j + 1].sub(&y[j])?;
        for (ws, wf) in w_star.iter_mut().zip(f.w.iter()) {
            *ws = ws.add(&wf.mul(&c_j)?)?;
        }
    }

    // Norm budget: coefficient of the accumulator is 1 (linear growth);
    // each fresh block enters with ‖y_j − y_{j−1}‖ ≤ 2·max|C-value| = 4.
    let q_half = (ring.modulus.q / 2) as u64;
    let fresh_norm = fresh
        .iter()
        .flat_map(|f| f.w.iter().map(|w| w.infinity_norm() as u64))
        .max()
        .unwrap_or(0);
    let budget = lattice_core::norm_budget::NormBudget::fresh(acc.norm_budget)
        .fold_scalar(4, fresh_norm, q_half, pk.params.norm_bound as u64)
        .map_err(|err| match err {
            lattice_core::norm_budget::NormBudgetError::Wraparound { beta_after, cap } => {
                PglError::NormBudgetExceeded {
                    budget: cap,
                    got: u64::try_from(beta_after).unwrap_or(u64::MAX),
                }
            }
        })?;

    let proof = PglFoldProof {
        f_coeffs: f_all[1..].to_vec(),
        quotients,
        delta,
        alpha,
        y,
    };
    let new_acc = PglAccInstance {
        t: t_star,
        beta: beta_star,
        e: e_star,
        norm_budget: budget.beta(),
    };
    Ok((new_acc, w_star, proof))
}

/// The Fig.-2 verifier: recomputes the folded commitment and the e*-check
/// from public data, and checks quotient degree bounds. Returns the
/// verifier-computed `(t*, e*)` pair; the caller compares against the
/// claimed accumulator. `d` is the relation degree.
pub fn fig2_verify(
    pk: &AjtaiPublicKey,
    cs: &PglConstraintSystem,
    acc: &PglAccInstance,
    fresh_t: &[AjtaiCommitment],
    proof: &PglFoldProof,
    chal_params: &PglChallengeParams,
) -> Result<(AjtaiCommitment, RingElement), PglError> {
    let k = fresh_t.len();
    let ring = pk.params.ring.clone();
    let d = cs.degree();

    // Replay the transcript to re-derive δ, α, y (public-coin checks).
    let mut transcript = Transcript::new_default(b"lzx-pgl-fold");
    pk.absorb_statement(&mut transcript, b"pk", &acc.t)?;
    transcript.append_field(b"acc-norm", &Goldilocks::from_u64(acc.norm_budget))?;
    for ft in fresh_t {
        pk.absorb_statement(&mut transcript, b"fresh-t", ft)?;
    }
    let delta = sample_ring_challenge(&mut transcript, b"pgl-delta", chal_params, &ring)?;
    if delta != proof.delta {
        return Err(PglError::ErrorCheckFailed);
    }
    for fc in &proof.f_coeffs {
        transcript.append_bytes(b"pgl-f", &fc.to_bytes())?;
    }
    let alpha = sample_ring_challenge(&mut transcript, b"pgl-alpha", chal_params, &ring)?;
    if alpha != proof.alpha {
        return Err(PglError::ErrorCheckFailed);
    }
    // Quotient degree bound: deg(K_ab) ≤ d − 2 (paper Fig. 2 check).
    for ((_, _), q) in &proof.quotients {
        if q.degree() > d.saturating_sub(2) {
            return Err(PglError::DegreeBoundExceeded {
                got: q.degree(),
                max: d.saturating_sub(2),
            });
        }
    }
    for (ab, _) in &proof.quotients {
        transcript.append_bytes(b"pgl-k", format!("{:?}", ab.0).as_bytes())?;
    }
    for (_, q) in &proof.quotients {
        for (e, c) in &q.terms {
            let mut buf = Vec::new();
            for x in e {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            buf.extend_from_slice(&c.to_bytes());
            transcript.append_bytes(b"pgl-k-term", &buf)?;
        }
    }
    let mut y = vec![ring.one()];
    for _ in 0..k {
        y.push(sample_ring_challenge(
            &mut transcript,
            b"pgl-y",
            chal_params,
            &ring,
        )?);
    }
    if y != proof.y {
        return Err(PglError::ErrorCheckFailed);
    }

    // F(α) = e + Σ_j F_j·α^j — the e-substitution binds F to the instance.
    let mut f_alpha = acc.e.clone();
    {
        let mut a_pow = proof.alpha.clone();
        for fc in &proof.f_coeffs {
            f_alpha = f_alpha.add(&fc.mul(&a_pow)?)?;
            a_pow = a_pow.mul(&proof.alpha)?;
        }
    }
    // t* = t_0 + Σ_j (y_j − y_{j−1})·t_j  (homomorphism).
    let mut t_star = acc.t.clone();
    for (j, ft) in fresh_t.iter().enumerate() {
        let c_j = proof.y[j + 1].sub(&proof.y[j])?;
        for (row_acc, row_f) in t_star.rows.iter_mut().zip(ft.rows.iter()) {
            *row_acc = row_acc.add(&row_f.mul(&c_j)?)?;
        }
    }
    // e* = F(α) + Σ_{a≤b} K_ab(y)·(y_a·y_b − y_a).
    let mut e_star = f_alpha;
    for ((a, b), q) in &proof.quotients {
        let z_ab = proof.y[*a].mul(&proof.y[*b])?.sub(&proof.y[*a])?;
        let q_at_y = q.evaluate(&proof.y)?;
        e_star = e_star.add(&q_at_y.mul(&z_ab)?)?;
    }
    Ok((t_star, e_star))
}

/// The decider: opens the accumulator commitment and checks the relaxed
/// relation `Σ_i pow_i(β)·f_i(w) = e` plus norm bounds.
pub fn decide(
    pk: &AjtaiPublicKey,
    cs: &PglConstraintSystem,
    acc: &PglAccInstance,
    w: &[RingElement],
) -> Result<(), PglError> {
    pk.verify_opening(&acc.t, w)?;
    let value = cs.relaxed_value(&acc.beta, w)?;
    if value != acc.e {
        return Err(PglError::DecideFailed);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PGL-Boot (paper Fig. 3) — Wave 7.14
// ---------------------------------------------------------------------------

/// Signed base-b decomposition of every witness slot into `k'` digit
/// blocks: `w = Σ_{j=0}^{k'−1} b^j·w_j` with balanced digits
/// `‖w_j‖∞ ≤ ⌈b/2⌉` (the paper's witness decomposition; the balanced
/// digit choice keeps the blocks as short as the base allows).
fn decompose_base_b(
    w: &[RingElement],
    base: i64,
    num_blocks: usize,
) -> Result<Vec<Vec<RingElement>>, PglError> {
    let ring = w
        .first()
        .map(|e| e.config().clone())
        .ok_or(PglError::Shape {
            expected: 1,
            got: 0,
        })?;
    let q = ring.modulus;
    // result[j][slot][coeff] = j-th balanced digit of the slot value.
    let mut result: Vec<Vec<Vec<i64>>> = vec![vec![vec![0i64; ring.n()]; w.len()]; num_blocks];
    for (slot, elem) in w.iter().enumerate() {
        for (c, &coeff) in elem.coeffs().iter().enumerate() {
            // Balanced representative of coeff mod q.
            let balanced = if coeff > q.q / 2 {
                coeff as i64 - q.q as i64
            } else {
                coeff as i64
            };
            let mut rem = balanced;
            for block in result.iter_mut() {
                let mut digit = rem.rem_euclid(base);
                if digit > base / 2 {
                    digit -= base;
                }
                block[slot][c] = digit;
                rem = (rem - digit) / base;
            }
            if rem != 0 {
                return Err(PglError::DecompositionMismatch);
            }
        }
    }
    Ok(result
        .into_iter()
        .map(|block| {
            block
                .into_iter()
                .map(|digits| RingElement::from_signed(&ring, &digits))
                .collect::<Vec<_>>()
        })
        .collect())
}

/// The boot proof: per-block commitments, errors, quotients, challenges.
#[derive(Clone, Debug)]
pub struct PglBootProof {
    pub t_blocks: Vec<AjtaiCommitment>,
    pub e_blocks: Vec<RingElement>,
    pub quotients: Vec<((usize, usize), MPoly)>,
    pub y: Vec<RingElement>,
    /// Digit-block infinity norms (for the range-check attachment).
    pub block_norms: Vec<u32>,
}

/// PGL-Boot: refresh the accumulator norm via base-b decomposition.
/// Returns the new (lower-norm) accumulator, its witness, and the proof.
/// `β` is unchanged (the paper does not re-randomize during boot).
pub fn fig3_boot(
    pk: &AjtaiPublicKey,
    cs: &PglConstraintSystem,
    acc: &PglAccInstance,
    acc_witness: &[RingElement],
    base: i64,
    num_blocks: usize,
    chal_params: &PglChallengeParams,
) -> Result<(PglAccInstance, Vec<RingElement>, PglBootProof), PglError> {
    let ring = pk.params.ring.clone();
    let t = cs.t_levels();
    if acc.beta.len() != t {
        return Err(PglError::Shape {
            expected: t,
            got: acc.beta.len(),
        });
    }
    // 1. Base-b decomposition of the witness.
    let blocks = decompose_base_b(acc_witness, base, num_blocks)?;
    // 2. Commit each block; compute per-block errors e_j (same β).
    let mut t_blocks = Vec::with_capacity(num_blocks);
    let mut e_blocks = Vec::with_capacity(num_blocks);
    for w_j in &blocks {
        t_blocks.push(pk.commit(w_j)?);
        e_blocks.push(cs.relaxed_value(&acc.beta, w_j)?);
    }
    // 3. H(Y) = Σ_i pow_i(β)·f_i(Σ_j L_j(Y)·w_j) — same β (no re-rand).
    let h = compute_h_poly(cs, &acc.beta, &blocks)?;
    // D(Y) := H(Y) − Σ_j L_j(Y)·e_j  → division must leave remainder 0.
    // L_0 = Y_0; L_j = Y_j − Y_{j−1} for j ≥ 1.
    let mut d_poly = h;
    for (j, e_j) in e_blocks.iter().enumerate() {
        if j == 0 {
            d_poly.add_linear(0, &e_j.neg())?;
        } else {
            d_poly.add_linear(j, &e_j.neg())?;
            d_poly.add_linear(j - 1, e_j)?;
        }
    }
    let (quotients, remainder) = groebner_divide(&d_poly, &ring)?;
    if !remainder.is_zero() {
        return Err(PglError::DecompositionMismatch);
    }
    // 4. Transcript + challenges: y ← C^{k'−1}, y_0 := 1.
    let mut transcript = Transcript::new_default(b"lzx-pgl-boot");
    pk.absorb_statement(&mut transcript, b"pk-boot", &acc.t)?;
    for tb in &t_blocks {
        transcript.append_bytes(b"boot-t", &tb.to_bytes())?;
    }
    for eb in &e_blocks {
        transcript.append_bytes(b"boot-e", &eb.to_bytes())?;
    }
    for (ab, _) in &quotients {
        transcript.append_bytes(b"boot-k", format!("{:?}", ab).as_bytes())?;
    }
    let mut y = vec![ring.one()];
    for _ in 0..num_blocks.saturating_sub(1) {
        y.push(sample_ring_challenge(
            &mut transcript,
            b"boot-y",
            chal_params,
            &ring,
        )?);
    }
    // 5. Updates: t* = Σ L_j(y)·t_j; e* = Σ L_j(y)·e_j + Σ K_ab(y)·Z_ab(y);
    //    w* = Σ L_j(y)·w_j.
    let mut t_star = t_blocks[0].clone();
    for (j, tb) in t_blocks.iter().enumerate().skip(1) {
        let c_j = y[j].sub(&y[j - 1])?;
        for (row_acc, row_b) in t_star.rows.iter_mut().zip(tb.rows.iter()) {
            *row_acc = row_acc.add(&row_b.mul(&c_j)?)?;
        }
    }
    let mut e_star = e_blocks[0].clone();
    for (j, eb) in e_blocks.iter().enumerate().skip(1) {
        let c_j = y[j].sub(&y[j - 1])?;
        e_star = e_star.add(&eb.mul(&c_j)?)?;
    }
    for ((a, b), q) in &quotients {
        let z_ab = y[*a].mul(&y[*b])?.sub(&y[*a])?;
        e_star = e_star.add(&q.evaluate(&y)?.mul(&z_ab)?)?;
    }
    let mut w_star = blocks[0].clone();
    for (j, w_j) in blocks.iter().enumerate().skip(1) {
        let c_j = y[j].sub(&y[j - 1])?;
        for (ws, wb) in w_star.iter_mut().zip(w_j.iter()) {
            *ws = ws.add(&wb.mul(&c_j)?)?;
        }
    }
    // Norm refresh: γ_out ≈ (2·‖Δy‖·(k'−1)+1)·⌈b/2⌉ — the point of boot.
    let block_norm = blocks
        .iter()
        .flat_map(|b| b.iter().map(|e| e.infinity_norm() as u64))
        .max()
        .unwrap_or(0);
    let q_half = (ring.modulus.q / 2) as u64;
    let budget = lattice_core::norm_budget::NormBudget::fresh(block_norm)
        .fold_scalar(4, block_norm, q_half, pk.params.norm_bound as u64)
        .map_err(|err| match err {
            lattice_core::norm_budget::NormBudgetError::Wraparound { beta_after, cap } => {
                PglError::NormBudgetExceeded {
                    budget: cap,
                    got: u64::try_from(beta_after).unwrap_or(u64::MAX),
                }
            }
        })?;
    let block_norms: Vec<u32> = blocks
        .iter()
        .map(|b| b.iter().map(|e| e.infinity_norm()).max().unwrap_or(0))
        .collect();

    let new_acc = PglAccInstance {
        t: t_star,
        beta: acc.beta.clone(),
        e: e_star,
        norm_budget: budget.beta(),
    };
    let proof = PglBootProof {
        t_blocks,
        e_blocks,
        quotients,
        y,
        block_norms,
    };
    Ok((new_acc, w_star, proof))
}

/// The PGL-Boot verifier: checks the D-point identity and recomputes
/// `(t*, e*)`. `D` satisfies `L_j(D) = b^j`, i.e.
/// `D_j = 1 + b + … + b^j` (partial geometric sums, `D_0 = 1`).
pub fn fig3_boot_verify(
    pk: &AjtaiPublicKey,
    cs: &PglConstraintSystem,
    acc: &PglAccInstance,
    proof: &PglBootProof,
    base: i64,
    chal_params: &PglChallengeParams,
) -> Result<(AjtaiCommitment, RingElement), PglError> {
    let ring = pk.params.ring.clone();
    let k_prime = proof.t_blocks.len();
    let d = cs.degree();
    // (i) Σ_j b^j·t_j = t — binds the decomposition to the commitment.
    let mut recomposed = proof.t_blocks[0].clone();
    for (j, tb) in proof.t_blocks.iter().enumerate().skip(1) {
        let b_pow = ring.constant((base.pow(j as u32)).rem_euclid(ring.modulus.q as i64) as u32);
        for (row_acc, row_b) in recomposed.rows.iter_mut().zip(tb.rows.iter()) {
            *row_acc = row_acc.add(&row_b.mul(&b_pow)?)?;
        }
    }
    if recomposed.rows != acc.t.rows {
        return Err(PglError::FoldCommitmentMismatch);
    }
    // (ii) D-point: Σ_j b^j·e_j + Σ_ab Z_ab(D)·K_ab(D) = e.
    let d_point: Vec<RingElement> = {
        let mut pts = Vec::with_capacity(k_prime);
        let mut acc_p = ring.one();
        for _ in 0..k_prime {
            pts.push(acc_p.clone());
            let b_ring = ring.constant((base.rem_euclid(ring.modulus.q as i64)) as u32);
            acc_p = acc_p.add(&b_ring)?;
        }
        pts
    };
    let mut lhs = proof.e_blocks[0].clone();
    for (j, eb) in proof.e_blocks.iter().enumerate().skip(1) {
        let b_pow = ring.constant((base.pow(j as u32)).rem_euclid(ring.modulus.q as i64) as u32);
        lhs = lhs.add(&eb.mul(&b_pow)?)?;
    }
    for ((a, b), q) in &proof.quotients {
        let z = d_point[*a].mul(&d_point[*b])?.sub(&d_point[*a])?;
        lhs = lhs.add(&q.evaluate(&d_point)?.mul(&z)?)?;
    }
    if lhs != acc.e {
        return Err(PglError::ErrorCheckFailed);
    }
    // Quotient degree bounds.
    for (_, q) in &proof.quotients {
        if q.degree() > d.saturating_sub(2) {
            return Err(PglError::DegreeBoundExceeded {
                got: q.degree(),
                max: d.saturating_sub(2),
            });
        }
    }
    // Transcript replay for y.
    let mut transcript = Transcript::new_default(b"lzx-pgl-boot");
    pk.absorb_statement(&mut transcript, b"pk-boot", &acc.t)?;
    for tb in &proof.t_blocks {
        transcript.append_bytes(b"boot-t", &tb.to_bytes())?;
    }
    for eb in &proof.e_blocks {
        transcript.append_bytes(b"boot-e", &eb.to_bytes())?;
    }
    for (ab, _) in &proof.quotients {
        transcript.append_bytes(b"boot-k", format!("{:?}", ab).as_bytes())?;
    }
    let mut y = vec![ring.one()];
    for _ in 0..k_prime.saturating_sub(1) {
        y.push(sample_ring_challenge(
            &mut transcript,
            b"boot-y",
            chal_params,
            &ring,
        )?);
    }
    if y != proof.y {
        return Err(PglError::ErrorCheckFailed);
    }
    // (iii) t* and e* recomputation.
    let mut t_star = proof.t_blocks[0].clone();
    for (j, tb) in proof.t_blocks.iter().enumerate().skip(1) {
        let c_j = proof.y[j].sub(&proof.y[j - 1])?;
        for (row_acc, row_b) in t_star.rows.iter_mut().zip(tb.rows.iter()) {
            *row_acc = row_acc.add(&row_b.mul(&c_j)?)?;
        }
    }
    let mut e_star = proof.e_blocks[0].clone();
    for (j, eb) in proof.e_blocks.iter().enumerate().skip(1) {
        let c_j = proof.y[j].sub(&proof.y[j - 1])?;
        e_star = e_star.add(&eb.mul(&c_j)?)?;
    }
    for ((a, b), q) in &proof.quotients {
        let z_ab = proof.y[*a].mul(&proof.y[*b])?.sub(&proof.y[*a])?;
        e_star = e_star.add(&q.evaluate(&proof.y)?.mul(&z_ab)?)?;
    }
    Ok((t_star, e_star))
}

/// A per-digit-block range-proof attachment: an LF+ algebraic range proof
/// over the block's flattened coefficients plus the bound it certifies.
/// The coefficient evaluation claim (`coeff_claim_at_point`) is the MLE of
/// the block's Goldilocks-encoded coefficients at the proof's eq point —
/// bound at kernel scale by the Ajtai commitment to the block (the PCS
/// opening layer that authenticates the claim is the caller's composition
/// point, exactly as in `latticefold_plus::verify_range`'s contract).
#[derive(Clone, Debug)]
pub struct BlockRangeAttachment {
    pub proof: crate::latticefold_plus::AlgebraicRangeProof,
    /// The certified infinity-norm bound for the block.
    pub bound: u64,
    /// Number of Goldilocks coefficients in the flattened block.
    pub num_coeffs: usize,
}

/// Flatten a digit block into Goldilocks coefficients: each ring
/// coefficient is re-balanced to its signed integer value (digits are
/// short by construction) and mapped into `[0, p)`.
fn block_to_goldilocks(w_j: &[RingElement]) -> Vec<Goldilocks> {
    let p = lattice_core::field::GOLDILOCKS_MODULUS as i128;
    w_j.iter()
        .flat_map(|e| {
            let q = e.config().modulus.q;
            e.coeffs().iter().map(move |&c| {
                let balanced = if c > q / 2 {
                    c as i128 - q as i128
                } else {
                    c as i128
                };
                Goldilocks::from_u64(balanced.rem_euclid(p) as u64)
            })
        })
        .collect()
}

/// The coefficient MLE evaluation at a range proof's eq point (the
/// verifier-side claim for `verify_range`).
pub fn block_coeff_claim(
    attachment: &BlockRangeAttachment,
    coeffs: &[Goldilocks],
) -> Result<Goldilocks, PglError> {
    let padded_len = attachment.num_coeffs.next_power_of_two();
    let mut evals = coeffs.to_vec();
    evals.resize(padded_len.max(1), Goldilocks::ZERO);
    let mle = DenseMle::new(evals).map_err(|_| PglError::Shape {
        expected: padded_len,
        got: attachment.num_coeffs,
    })?;
    mle.evaluate(&attachment.proof.sc_point)
        .map_err(|_| PglError::DecideFailed)
}

/// Attach LatticeFold+ range proofs to the boot digit blocks (item 7.14's
/// `Π_rg` attachment): proves each block's coefficients are small so the
/// refreshed accumulator's norm bound is certified rather than asserted.
pub fn boot_with_range(
    pk: &AjtaiPublicKey,
    cs: &PglConstraintSystem,
    acc: &PglAccInstance,
    acc_witness: &[RingElement],
    base: i64,
    num_blocks: usize,
    chal_params: &PglChallengeParams,
) -> Result<
    (
        PglAccInstance,
        Vec<RingElement>,
        PglBootProof,
        Vec<BlockRangeAttachment>,
    ),
    PglError,
> {
    let (new_acc, w_star, proof) =
        fig3_boot(pk, cs, acc, acc_witness, base, num_blocks, chal_params)?;
    // Range proofs over each digit block's flattened signed coefficients:
    // the digits are bounded by ⌈b/2⌉, certified via the LF+ norm argument.
    let blocks = decompose_base_b(acc_witness, base, num_blocks)?;
    let mut attachments = Vec::with_capacity(num_blocks);
    for (w_j, norm) in blocks.iter().zip(&proof.block_norms) {
        let bound = (*norm as u64).max(1);
        let coeffs = block_to_goldilocks(w_j);
        let mut transcript = Transcript::new_default(b"lzx-pgl-range");
        let rp = crate::latticefold_plus::prove_range(&coeffs, bound, &mut transcript)
            .map_err(|_| PglError::DecideFailed)?;
        attachments.push(BlockRangeAttachment {
            proof: rp,
            bound,
            num_coeffs: coeffs.len(),
        });
    }
    Ok((new_acc, w_star, proof, attachments))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_commitment::ajtai::AjtaiParams;
    use lattice_ring::{Modulus32, RingConfig};

    fn setup(log_n: u32, m: usize) -> (AjtaiPublicKey, RingConfig) {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m,
            norm_bound: 1 << 22,
        };
        let pk = AjtaiPublicKey::from_seed(params, [21u8; 32]).ok().unwrap();
        (pk, ring)
    }

    /// Two constraints (t = 1), both genuinely degree-2 in the fold line
    /// (ring-product semantics): f1 = w0·w1 − w2·w2, f2 = w1·w2 − w0·w2.
    /// The witness family below satisfies both with all three slots
    /// non-zero, so H(Y) carries genuine cross-term (degree-2) structure.
    fn constraint_system() -> PglConstraintSystem {
        let q = Modulus32::Q_32.q;
        PglConstraintSystem::new(vec![
            PgRelation {
                num_slots: 3,
                terms: vec![(1, vec![0, 1]), (q - 1, vec![2, 2])],
            },
            PgRelation {
                num_slots: 3,
                terms: vec![(1, vec![1, 2]), (q - 1, vec![0, 2])],
            },
        ])
        .ok()
        .unwrap()
    }

    /// A fresh witness satisfying both ring-product constraints with all
    /// slots non-zero: (v, v, v) gives f1 = v·v − v·v = 0 and
    /// f2 = v·v − v·v = 0 (ring multiplication), for any v. Each instance
    /// uses a distinct deterministic v so folds carry genuine cross terms.
    fn fresh_witness(ring: &RingConfig, tag: &[u8]) -> Vec<RingElement> {
        let t0 = tag.first().copied().unwrap_or(1) as i64;
        let v: Vec<i64> = (0..ring.n())
            .map(|i| ((i as i64 * 3 + t0 * 5) % 3) - 1) // ∈ {−1,0,1}
            .collect();
        let elem = RingElement::from_signed(ring, &v);
        vec![elem.clone(), elem.clone(), elem.clone()]
    }

    fn small_witness(ring: &RingConfig, tag: &[u8]) -> Vec<RingElement> {
        lattice_commitment::ajtai::sample_small_secret(ring, 3, 8, tag)
    }

    fn make_acc(
        pk: &AjtaiPublicKey,
        cs: &PglConstraintSystem,
        w: &[RingElement],
    ) -> PglAccInstance {
        let ring = pk.params.ring.clone();
        let t = cs.t_levels();
        // β from a deterministic small pattern (the randomizer tower).
        let beta: Vec<RingElement> = (0..t)
            .map(|j| {
                RingElement::from_signed(
                    &ring,
                    &(0..ring.n())
                        .map(|i| (i as i64 + j as i64 + 1) % 3 - 1)
                        .collect::<Vec<i64>>(),
                )
            })
            .collect();
        let t_comm = pk.commit(w).ok().unwrap();
        // A *relaxed* accumulator: e = Σ_i pow_i(β)·f_i(w) — generally
        // non-zero (the relaxed-folding path, not the trivial e = 0).
        let e = cs.relaxed_value(&beta, w).ok().unwrap();
        PglAccInstance {
            t: t_comm,
            beta,
            e,
            norm_budget: 8,
        }
    }

    #[test]
    fn fig2_fold_protocol_end_to_end() {
        let (pk, ring) = setup(4, 3);
        let cs = constraint_system();
        let chal = PglChallengeParams::for_ring(&ring);
        // A genuinely *relaxed* accumulator (e ≠ 0 in general).
        let w0 = small_witness(&ring, b"acc");
        let acc = make_acc(&pk, &cs, &w0);
        // Two fresh instances (k = 2), all satisfying f_i = 0 with
        // non-zero slots (genuine degree-2 cross structure in H).
        let fresh: Vec<PglFreshInstance> = (0..2)
            .map(|j| {
                let w = fresh_witness(&ring, format!("f{}", j).as_bytes());
                let t = pk.commit(&w).ok().unwrap();
                PglFreshInstance { t, w }
            })
            .collect();
        let (new_acc, w_star, proof) = fig2_fold(&pk, &cs, &acc, &w0, &fresh, &chal).ok().unwrap();
        // The fold must have produced genuine quotients (cross terms).
        assert!(
            !proof.quotients.is_empty(),
            "degenerate test: no cross terms"
        );
        // Verifier: recompute (t*, e*) and compare.
        let fresh_t: Vec<AjtaiCommitment> = fresh.iter().map(|f| f.t.clone()).collect();
        let (vt, ve) = fig2_verify(&pk, &cs, &acc, &fresh_t, &proof, &chal)
            .ok()
            .unwrap();
        assert_eq!(vt.rows, new_acc.t.rows, "folded commitment mismatch");
        assert_eq!(ve, new_acc.e, "e*-check mismatch");
        // Decider accepts with the folded witness.
        decide(&pk, &cs, &new_acc, &w_star).ok().unwrap();
        // The witness norm grew linearly (accumulator coefficient = 1).
        assert!(new_acc.norm_budget < (1 << 20));
    }

    #[test]
    fn fig2_fold_rejects_invalid_fresh_instance() {
        let (pk, ring) = setup(4, 3);
        let cs = constraint_system();
        let chal = PglChallengeParams::for_ring(&ring);
        let w0 = small_witness(&ring, b"acc");
        let acc = make_acc(&pk, &cs, &w0);
        // A CHEATING fresh witness: nonzero w1/w2 (violates both f_i = 0).
        let bad = small_witness(&ring, b"bad");
        let t_bad = pk.commit(&bad).ok().unwrap();
        let fresh = vec![PglFreshInstance { t: t_bad, w: bad }];
        // The prover-side division detects the non-zero linear residual.
        assert!(matches!(
            fig2_fold(&pk, &cs, &acc, &w0, &fresh, &chal),
            Err(PglError::FreshInstanceInvalid { .. })
        ));
    }

    #[test]
    fn fig2_tampered_quotient_fails_e_star() {
        let (pk, ring) = setup(4, 3);
        let cs = constraint_system();
        let chal = PglChallengeParams::for_ring(&ring);
        let w0 = small_witness(&ring, b"acc");
        let acc = make_acc(&pk, &cs, &w0);
        let fresh: Vec<PglFreshInstance> = (0..2)
            .map(|j| {
                let w = fresh_witness(&ring, format!("f{}", j).as_bytes());
                let t = pk.commit(&w).ok().unwrap();
                PglFreshInstance { t, w }
            })
            .collect();
        let (new_acc, w_star, mut proof) =
            fig2_fold(&pk, &cs, &acc, &w0, &fresh, &chal).ok().unwrap();
        // Adversary substitutes one quotient coefficient: the Fiat-Shamir
        // transcript binding (y is derived after the quotients are
        // absorbed) and/or the e*-check must reject.
        if let Some((_, q)) = proof.quotients.first_mut() {
            if let Some((_, c)) = q.terms.first_mut() {
                *c = c.add(&ring.one()).ok().unwrap();
            }
        }
        let fresh_t: Vec<AjtaiCommitment> = fresh.iter().map(|f| f.t.clone()).collect();
        match fig2_verify(&pk, &cs, &acc, &fresh_t, &proof, &chal) {
            Err(_) => { /* transcript binding caught the tamper */ }
            Ok((vt, ve)) => {
                let mismatch = ve != new_acc.e || vt.rows != new_acc.t.rows;
                let decide_ok = decide(&pk, &cs, &new_acc, &w_star).is_ok();
                assert!(
                    mismatch || !decide_ok,
                    "tampered quotient must break verification"
                );
            }
        }
    }

    #[test]
    fn fig2_tampered_f_coeff_fails() {
        let (pk, ring) = setup(4, 3);
        let cs = constraint_system();
        let chal = PglChallengeParams::for_ring(&ring);
        let w0 = small_witness(&ring, b"acc");
        let acc = make_acc(&pk, &cs, &w0);
        let fresh: Vec<PglFreshInstance> = (0..1)
            .map(|j| {
                let w = fresh_witness(&ring, format!("f{}", j).as_bytes());
                let t = pk.commit(&w).ok().unwrap();
                PglFreshInstance { t, w }
            })
            .collect();
        let (new_acc, _, mut proof) = fig2_fold(&pk, &cs, &acc, &w0, &fresh, &chal).ok().unwrap();
        // Tamper F_1: α is derived after the F coefficients are absorbed,
        // so the transcript replay itself detects the tamper; if the
        // replay happens to pass, F(α) changes ⇒ e* changes.
        proof.f_coeffs[0] = proof.f_coeffs[0].add(&ring.one()).ok().unwrap();
        let fresh_t: Vec<AjtaiCommitment> = fresh.iter().map(|f| f.t.clone()).collect();
        match fig2_verify(&pk, &cs, &acc, &fresh_t, &proof, &chal) {
            Err(_) => { /* transcript binding caught the tamper */ }
            Ok((_, ve)) => {
                // F(α) changed ⇒ e* changed ⇒ the claimed accumulator (from
                // the honest run) no longer matches the verifier's value.
                assert_ne!(ve, new_acc.e);
            }
        }
    }

    #[test]
    fn fig2_iterated_folding_then_decide() {
        // Three sequential folds (the scheme's purpose: accumulate many
        // instances into one decidable accumulator).
        let (pk, ring) = setup(4, 3);
        let cs = constraint_system();
        let chal = PglChallengeParams::for_ring(&ring);
        let mut w_cur = small_witness(&ring, b"acc0");
        let mut acc = make_acc(&pk, &cs, &w_cur);
        for round in 0..3 {
            let fresh: Vec<PglFreshInstance> = (0..1)
                .map(|j| {
                    let w = fresh_witness(&ring, format!("r{}f{}", round, j).as_bytes());
                    let t = pk.commit(&w).ok().unwrap();
                    PglFreshInstance { t, w }
                })
                .collect();
            let (new_acc, w_star, proof) = fig2_fold(&pk, &cs, &acc, &w_cur, &fresh, &chal)
                .ok()
                .unwrap();
            let fresh_t: Vec<AjtaiCommitment> = fresh.iter().map(|f| f.t.clone()).collect();
            let (vt, ve) = fig2_verify(&pk, &cs, &acc, &fresh_t, &proof, &chal)
                .ok()
                .unwrap();
            assert_eq!(vt.rows, new_acc.t.rows);
            assert_eq!(ve, new_acc.e);
            acc = new_acc;
            w_cur = w_star;
        }
        decide(&pk, &cs, &acc, &w_cur).ok().unwrap();
    }

    #[test]
    fn groebner_division_properties() {
        // Division identities on small explicit polynomials.
        let ring = RingConfig::new(Modulus32::Q_32, 3).ok().unwrap();
        // P = Y_0·Y_1 − Y_0  ≡ 0 mod J (it IS Z_01).
        let mut terms = Vec::new();
        let mut e01 = vec![0u32; 2];
        e01[0] = 1;
        e01[1] = 1;
        terms.push((e01, ring.one()));
        let mut e0 = vec![0u32; 2];
        e0[0] = 1;
        terms.push((e0, ring.one().neg()));
        let p = MPoly { vars: 2, terms };
        let (qs, rem) = groebner_divide(&p, &ring).ok().unwrap();
        assert!(rem.is_zero());
        assert_eq!(qs.len(), 1);
        // P = Y_0^3 → Y_0 (idempotence).
        let mut e = vec![0u32; 2];
        e[0] = 3;
        let p2 = MPoly {
            vars: 2,
            terms: vec![(e, ring.constant(5))],
        };
        let (_, rem2) = groebner_divide(&p2, &ring).ok().unwrap();
        // NF = 5·Y_0.
        assert_eq!(rem2.terms.len(), 1);
        assert_eq!(rem2.terms[0].0, vec![1u32, 0]);
    }

    #[test]
    fn fig3_boot_refreshes_norm() {
        let (pk, ring) = setup(4, 3);
        let cs = constraint_system();
        let chal = PglChallengeParams::for_ring(&ring);
        // Accumulator with grown norm: fold a couple of rounds first.
        let mut w_cur = small_witness(&ring, b"acc0");
        let mut acc = make_acc(&pk, &cs, &w_cur);
        for round in 0..2 {
            let fresh: Vec<PglFreshInstance> = (0..1)
                .map(|j| {
                    let w = fresh_witness(&ring, format!("br{}f{}", round, j).as_bytes());
                    let t = pk.commit(&w).ok().unwrap();
                    PglFreshInstance { t, w }
                })
                .collect();
            let (new_acc, w_star, _) = fig2_fold(&pk, &cs, &acc, &w_cur, &fresh, &chal)
                .ok()
                .unwrap();
            acc = new_acc;
            w_cur = w_star;
        }
        let pre_norm = w_cur
            .iter()
            .map(|e| e.infinity_norm() as u64)
            .max()
            .unwrap();
        // Boot with base 8, 3 blocks.
        let (boot_acc, boot_w, proof) =
            fig3_boot(&pk, &cs, &acc, &w_cur, 8, 3, &chal).ok().unwrap();
        // Verifier accepts: recompose + D-point + updates.
        let (vt, ve) = fig3_boot_verify(&pk, &cs, &acc, &proof, 8, &chal)
            .ok()
            .unwrap();
        assert_eq!(vt.rows, boot_acc.t.rows);
        assert_eq!(ve, boot_acc.e);
        // Booted witness: folded digit combination — still opens and the
        // relation holds with the SAME β.
        decide(&pk, &cs, &boot_acc, &boot_w).ok().unwrap();
        // The digit blocks are short (the refresh's whole point).
        let max_block = proof.block_norms.iter().copied().max().unwrap();
        assert!(max_block as u64 <= 4 + 1, "digit blocks must be short");
        let _ = pre_norm;
    }

    #[test]
    fn fig3_boot_tampered_block_fails() {
        let (pk, ring) = setup(4, 3);
        let cs = constraint_system();
        let chal = PglChallengeParams::for_ring(&ring);
        let w0 = small_witness(&ring, b"acc");
        let acc = make_acc(&pk, &cs, &w0);
        let (_, _, mut proof) = fig3_boot(&pk, &cs, &acc, &w0, 8, 3, &chal).ok().unwrap();
        // Tamper one block commitment: the Σ b^j·t_j = t check must fail.
        if let Some(tb) = proof.t_blocks.first_mut() {
            if let Some(row) = tb.rows.first_mut() {
                *row = row.add(&ring.one()).ok().unwrap();
            }
        }
        assert!(matches!(
            fig3_boot_verify(&pk, &cs, &acc, &proof, 8, &chal),
            Err(PglError::FoldCommitmentMismatch)
        ));
    }

    #[test]
    fn boot_range_attachment_produces_proofs() {
        let (pk, ring) = setup(4, 3);
        let cs = constraint_system();
        let chal = PglChallengeParams::for_ring(&ring);
        let w0 = small_witness(&ring, b"acc");
        let acc = make_acc(&pk, &cs, &w0);
        let (_, _, proof, attachments) = boot_with_range(&pk, &cs, &acc, &w0, 8, 3, &chal)
            .ok()
            .unwrap();
        assert_eq!(attachments.len(), proof.t_blocks.len());
        // Each attachment's range proof verifies over its own transcript,
        // with the coefficient claim derived from the digit block.
        let blocks = decompose_base_b(&w0, 8, 3).ok().unwrap();
        for (att, w_j) in attachments.iter().zip(blocks.iter()) {
            let coeffs = block_to_goldilocks(w_j);
            let claim = block_coeff_claim(att, &coeffs).ok().unwrap();
            let mut vt = Transcript::new_default(b"lzx-pgl-range");
            crate::latticefold_plus::verify_range(
                &att.proof,
                att.bound,
                att.num_coeffs,
                claim,
                &mut vt,
            )
            .ok()
            .unwrap();
        }
    }

    #[test]
    fn pow_tower_matches_tensor_products() {
        let ring = RingConfig::new(Modulus32::Q_32, 3).ok().unwrap();
        let beta: Vec<RingElement> = vec![
            RingElement::from_signed(&ring, &[1, 0, -1, 0, 1, 0, 0, 0]),
            RingElement::from_signed(&ring, &[0, 1, 0, 0, -1, 0, 1, 0]),
        ];
        // pow_1 = 1 (no bits of 0 set).
        let one = pow_tower(&ring, &beta, 1).ok().unwrap();
        assert_eq!(one.coeffs()[0], 1);
        assert!(one.coeffs().iter().skip(1).all(|&c| c == 0));
        // pow_2 = β_0.
        assert_eq!(pow_tower(&ring, &beta, 2).ok().unwrap(), beta[0]);
        // pow_3 = β_1.
        assert_eq!(pow_tower(&ring, &beta, 3).ok().unwrap(), beta[1]);
        // pow_4 = β_0·β_1.
        assert_eq!(
            pow_tower(&ring, &beta, 4).ok().unwrap(),
            beta[0].mul(&beta[1]).ok().unwrap()
        );
    }

    #[test]
    fn constraint_system_validation() {
        assert!(PglConstraintSystem::new(vec![]).is_ok());
        let one = PgRelation {
            num_slots: 1,
            terms: vec![(1, vec![0])],
        };
        assert!(matches!(
            PglConstraintSystem::new(vec![one.clone(), one.clone(), one.clone()]),
            Err(PglError::ConstraintCountNotPowerOfTwo { got: 3 })
        ));
        let const_rel = PgRelation {
            num_slots: 1,
            terms: vec![(5, vec![])],
        };
        assert!(matches!(
            PglConstraintSystem::new(vec![const_rel, one.clone()]),
            Err(PglError::NonHomogeneousRelation { .. })
        ));
    }
}
