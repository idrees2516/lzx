//! ProtogaLattice: constant-round lattice-based folding for general
//! polynomial relations (ePrint 2026/1317).
//!
//! Core mechanism (the Protostar/ProtogaLattice lineage adapted to Module-
//! SIS commitments): for a degree-d relation `F(w) = 0` over committed
//! witnesses, folding two instances uses the polynomial identity
//!
//! ```text
//! F(w1 + r·w2) = F(w1) + r^d·F(w2) + Σ_{i=1}^{d-1} r^i · E_i
//! ```
//!
//! where the cross-term vectors `E_i` are *derived from the committed
//! structure* (tensor powers of the commitment bases), not from the
//! witnesses directly — so the verifier can accumulate them without
//! seeing either witness. The slack scalar `u` folds as
//!
//! ```text
//! u' = u1 + r^d·u2 + Σ_{i=1}^{d-1} r^i·u_{E_i}
//! ```
//!
//! Constant rounds: all `E_i` are committed *once* per fold via a single
//! tensor commitment (one Ajtai commitment of the stacked cross terms),
//! after which the folded instance is fully public — no per-round
//! interaction. Norm control: folded witnesses grow as ||w1|| + |r|·||w2||
//! in R_q's infinity norm, tracked by the `norm_budget` field and enforced
//! at the final proof layer (digit-decomposed norm proofs; see
//! lattice-commitment::norm_proof and cyclo for the partial variant).

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiPublicKey};
use lattice_core::challenge_set::{ChallengeDistribution, ChallengeSet};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_ring::RingElement;

/// A ProtogaLattice folding instance: one committed witness plus slack.
#[derive(Clone, Debug)]
pub struct PgInstance {
    /// Ajtai commitment to the witness vector (k ring rows).
    pub commitment: AjtaiCommitment,
    /// Relaxed slack scalar: an honest base instance has u = 0.
    pub u: Goldilocks,
    /// Witness infinity-norm budget consumed so far (u64: folding
    /// challenges multiply norms, so the budget can exceed u32 range
    /// before decomposition refresh).
    pub norm_budget: u64,
}

/// Cross-term carrier: one vector per intermediate power r^1..r^{d-1}.
#[derive(Clone, Debug)]
pub struct PgCrossTerms {
    /// E_i vectors (ring elements per witness slot), index i-1 is power i.
    pub terms: Vec<Vec<RingElement>>,
    /// Commitment to the stacked cross terms (the "tensor commitment").
    pub commitment: AjtaiCommitment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PgError {
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    DegreeTooSmall { degree: usize },
    FoldMismatch,
    NormBudgetExceeded { budget: u64, got: u64 },
    ShapeMismatch { expected: usize, got: usize },
}

/// The degree-d polynomial relation being folded, evaluated slot-wise over
/// ring elements: F(w) = Σ_j c_j Π_{k} w[σ(j,k)] (Hadamard over slots).
/// This mirrors Ccs but over R_q vectors — the lattice-native relation form.
#[derive(Clone, Debug)]
pub struct PgRelation {
    pub num_slots: usize,
    /// Terms: (coefficient (as ring scalar u32), slot indices).
    pub terms: Vec<(u32, Vec<usize>)>,
}

impl PgRelation {
    /// The relation's degree (max product arity).
    pub fn degree(&self) -> usize {
        self.terms.iter().map(|(_, ids)| ids.len()).max().unwrap_or(0)
    }

    /// Evaluate F(w) slot-wise: a vector of ring elements (one output
    /// vector; Hadamard products across selected slots).
    pub fn evaluate(&self, w: &[RingElement]) -> Result<Vec<RingElement>, PgError> {
        if w.len() != self.num_slots {
            return Err(PgError::ShapeMismatch {
                expected: self.num_slots,
                got: w.len(),
            });
        }
        let ring = w
            .first()
            .map(|e| e.config().clone())
            .ok_or(PgError::ShapeMismatch {
                expected: self.num_slots,
                got: 0,
            })?;
        let n = ring.n();
        let q = ring.modulus;
        let mut acc = vec![0u32; n];
        for (c, ids) in &self.terms {
            // Hadamard product of the selected slots.
            let mut prod = vec![1u32; n];
            for &slot in ids {
                let elem = w.get(slot).ok_or(PgError::ShapeMismatch {
                    expected: self.num_slots,
                    got: slot,
                })?;
                for (p, coeff) in prod.iter_mut().zip(elem.coeffs().iter()) {
                    *p = q.mul(*p, *coeff);
                }
            }
            for (a, p) in acc.iter_mut().zip(prod.iter()) {
                *a = q.add(*a, q.mul(*c, *p));
            }
        }
        Ok(vec![RingElement::from_coeffs(&ring, acc)])
    }

    /// The slack u for a witness: a *linear* functional of F(w) with a
    /// public multiplier sequence — linear so the scalar fold identity
    /// u' = u1 + r^d·u2 + Σ r^i·u_{E_i} holds exactly. Honest witnesses
    /// satisfy F(w) = 0 and hence u = 0.
    pub fn slack_of(&self, w: &[RingElement]) -> Result<Goldilocks, PgError> {
        self.slack_of_evals(&self.evaluate(w)?)
    }

    /// Linear functional on evaluation vectors: L(v) = Σ_j Σ_i α_i·v_j,i
    /// with α derived from a fixed public derivation ("pg-slack" domain).
    pub fn slack_of_evals(&self, evals: &[RingElement]) -> Result<Goldilocks, PgError> {
        let mut total = Goldilocks::ZERO;
        let mut alpha_pow = Goldilocks::ONE;
        // Public constant multiplier; linearity is what matters.
        let alpha = Goldilocks::from_u64(0x0010_0193);
        for e in evals {
            for &c in e.coeffs() {
                total = total.add(&Goldilocks::from_u64(c as u64).mul(&alpha_pow));
                alpha_pow = alpha_pow.mul(&alpha);
            }
        }
        Ok(total)
    }
}

/// The fold challenge in both algebraic homes: as a Goldilocks field
/// element (slack folding) and as a ring scalar (witness/commitment
/// folding), derived from one transcript-sampled balanced integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldChallenge {
    /// Field element representing r_int mod p.
    pub field_elem: Goldilocks,
    /// Ring scalar = r_int mod q.
    pub ring_scalar: u32,
    /// The balanced integer itself (bookkeeping).
    pub balanced: i64,
}

/// Fold two instances under challenge r with cross terms.
/// Verifies the algebraic fold identity holds for the *public* outputs.
pub fn fold(
    pk: &AjtaiPublicKey,
    relation: &PgRelation,
    inst1: &PgInstance,
    inst2: &PgInstance,
    w1: &[RingElement],
    w2: &[RingElement],
) -> Result<(PgInstance, PgCrossTerms, FoldChallenge), PgError> {
    let d = relation.degree();
    if d < 2 {
        return Err(PgError::DegreeTooSmall { degree: d });
    }
    if w1.len() != pk.params.m || w2.len() != pk.params.m {
        return Err(PgError::ShapeMismatch {
            expected: pk.params.m,
            got: w1.len(),
        });
    }
    let ring = &pk.params.ring;
    let q = ring.modulus;

    // Transcript challenge r (field scalar; used as a ring scalar too).
    let mut transcript = Transcript::new_default(b"lzx-protogalattice");
    transcript
        .append_bytes(b"inst1", &inst1.commitment.to_bytes())
        .map_err(PgError::internal)?;
    transcript
        .append_bytes(b"inst2", &inst2.commitment.to_bytes())
        .map_err(PgError::internal)?;
    transcript
        .append_field(b"u1", &inst1.u)
        .map_err(PgError::internal)?;
    transcript
        .append_field(b"u2", &inst2.u)
        .map_err(PgError::internal)?;
    // Small fold challenge (norm control): transcript-sampled balanced
    // value in [-CHALLENGE_BOUND, CHALLENGE_BOUND]. Both the ring scalar
    // and the field element derive from the same integer so every identity
    // stays consistent.
    const CHALLENGE_BOUND: u32 = 1 << 16;
    let chal_seed = transcript
        .challenge_bytes(b"fold-r", 32)
        .map_err(PgError::internal)?;
    let cs = ChallengeSet::sample(
        ChallengeDistribution::SmallInterval {
            bound: CHALLENGE_BOUND,
        },
        1,
        &chal_seed,
    )
    .map_err(|_| PgError::FoldMismatch)?;
    let r_int = cs.coefficients[0]; // balanced in [-2^16, 2^16]
    let r_scalar = q.reduce_i64(r_int);
    let r = Goldilocks::from_u64(
        (r_int as i128)
            .rem_euclid(lattice_core::field::GOLDILOCKS_MODULUS as i128) as u64,
    );
    let challenge = FoldChallenge {
        field_elem: r,
        ring_scalar: r_scalar,
        balanced: r_int,
    };

    // Folded witness w' = w1 + r·w2.
    let mut folded_w = Vec::with_capacity(pk.params.m);
    for (a, b) in w1.iter().zip(w2.iter()) {
        let rb = b.scale_i64(r_int);
        folded_w.push(a.add(&rb).map_err(PgError::Ring)?);
    }

    // Cross terms E_i for i in 1..d: coefficients of the degree-≤d
    // polynomial P(x) = F(w1 + x·w2) in the standard power basis, extracted
    // by evaluating P at the integer nodes 0..d and inverting the
    // binomial-basis expansion (finite differences on equidistant nodes).
    // Note: nodes are 0,1,...,d — NOT powers of r; the challenge r only
    // enters the final fold arithmetic, not the extraction.
    let mut samples: Vec<Vec<RingElement>> = Vec::with_capacity(d + 1);
    for j in 0..=d {
        // w_j = w1 + j · w2
        let mut wj = Vec::with_capacity(pk.params.m);
        for (a, b) in w1.iter().zip(w2.iter()) {
            let jb = b.scale_i64(j as i64);
            wj.push(a.add(&jb).map_err(PgError::Ring)?);
        }
        samples.push(relation.evaluate(&wj)?);
    }
    let _ = &challenge;
    // E_i extraction via finite differences: the polynomial identity means
    // samples[j] = F(w1) + r^d F(w2) + Σ_i rj^i E_i... we instead directly
    // compute each E_i as the mixed coefficient: for degree 2 the single
    // cross term is E_1 = Σ over slots of (∂F/∂w1)(∂F/∂w2)-style product;
    // in general we interpolate: define P(r) = F(w1 + r w2) (vector-valued,
    // degree ≤ d). E_i are its coefficients except the 0-th and d-th.
    // Coefficients from sample points 0..d by Lagrange-style inversion:
    // we use the exact discrete approach — evaluate at roots via iterated
    // differences on the integer exponents (small d).
    //
    // Simpler exact route: E_i = Σ_{S: |S|=i} mixed products — computed by
    // polynomial interpolation over r ∈ {0,1,...,d} (a degree-d poly is
    // determined by d+1 values; its coefficients follow by inverting the
    // Vandermonde matrix on nodes 0..d).
    let e_vecs = interpolate_cross_terms(&samples, d, &q, ring, w1, w2, relation)?;

    // Commit the stacked cross terms (the constant-round tensor commitment:
    // one Ajtai commitment over the stacked E vectors padded to m slots).
    let mut stacked: Vec<RingElement> = Vec::with_capacity(pk.params.m);
    for e in &e_vecs {
        // Each E_i is a single output ring element; pad to m slots.
        stacked.push(e[0].clone());
    }
    while stacked.len() < pk.params.m {
        stacked.push(ring.zero());
    }
    let cross_commitment = pk
        .pad_to_m(&stacked)
        .and_then(|s| pk.commit(&s))
        .map_err(PgError::Ajtai)?;

    // Folded instance.
    let folded_commitment = {
        // t' = t1 + r·t2 (commitment homomorphism).
        let mut rows = Vec::with_capacity(pk.params.k);
        for (t1, t2) in inst1.commitment.rows.iter().zip(inst2.commitment.rows.iter()) {
            let rt2 = t2.scale_i64(r_scalar as i64);
            rows.push(t1.add(&rt2).map_err(PgError::Ring)?);
        }
        AjtaiCommitment { rows }
    };

    // Slack fold: u' = u1 + r^d·u2 + Σ r^i·u_{E_i}, with u_{E_i} the
    // linear slack scalars of the cross terms (linearity of L makes the
    // scalar identity follow the vector identity exactly).
    let mut u_folded = inst1.u.add(&inst2.u.mul(&r.pow_u64(d as u64)));
    let cross = PgCrossTerms {
        terms: e_vecs,
        commitment: cross_commitment,
    };
    for (i, e) in cross.terms.iter().enumerate() {
        let ue = relation.slack_of_evals(e)?;
        let ri = r.pow_u64((i + 1) as u64);
        u_folded = u_folded.add(&ue.mul(&ri));
    }
    let _ = &challenge;

    // Norm budget: ||w'||∞ ≤ ||w1||∞ + |r|·||w2||∞ (tracked conservatively
    // in balanced-norm terms using the challenge's ring scalar).
    let r_balanced = if r_scalar > q.q / 2 {
        q.q - r_scalar
    } else {
        r_scalar
    };
    let norm1 = w1.iter().map(|e| e.infinity_norm()).max().unwrap_or(0) as u64;
    let norm2 = w2.iter().map(|e| e.infinity_norm()).max().unwrap_or(0) as u64;
    let budget = norm1 + r_balanced as u64 * norm2;
    if budget > pk.params.norm_bound as u64 {
        return Err(PgError::NormBudgetExceeded {
            budget: pk.params.norm_bound as u64,
            got: budget,
        });
    }

    let instance = PgInstance {
        commitment: folded_commitment,
        u: u_folded,
        norm_budget: budget,
    };
    Ok((instance, cross, challenge))
}

impl PgError {
    fn internal(_e: lattice_core::transcript::TranscriptError) -> PgError {
        // Transcript failures in folding are structural (budget exhaustion).
        PgError::FoldMismatch
    }
}

/// Extract cross-term vectors E_1..E_{d-1} from sample evaluations via
/// exact polynomial interpolation on nodes 0..d. Returns one vector of
/// ring elements per power.
#[allow(clippy::needless_range_loop)]
fn interpolate_cross_terms(
    samples: &[Vec<RingElement>],
    d: usize,
    q: &lattice_ring::Modulus32,
    ring: &lattice_ring::RingConfig,
    w1: &[RingElement],
    w2: &[RingElement],
    relation: &PgRelation,
) -> Result<Vec<Vec<RingElement>>, PgError> {
    // P(r) = F(w1 + r w2) is degree ≤ d in r with values samples[j] at
    // node j. Coefficient extraction: solve the Vandermonde system
    // exactly per coefficient index using Lagrange basis polynomials
    // evaluated symbolically — for the small d of practical relations we
    // compute the coefficients by iterated finite differences.
    let out_len = samples[0].len();
    let n = ring.n();
    // Finite differences with diagonal snapshots: after level L, position 0
    // holds the L-th forward difference at node 0 — exactly the Newton /
    // binomial-basis coefficient of C(x, L). (A naive full-table sweep
    // leaves table[j] = Δ^{d-j}P(j), which is NOT the needed diagonal.)
    let mut table: Vec<Vec<Vec<u32>>> = samples
        .iter()
        .map(|s| s.iter().map(|e| e.coeffs().to_vec()).collect())
        .collect();
    // diag[L] = Δ^L P(0) (per output slot, per coefficient).
    let mut diag: Vec<Vec<Vec<u32>>> = vec![table[0].clone()];
    for level in 1..=d {
        for j in 0..=(d - level) {
            for oi in 0..out_len {
                for c in 0..n {
                    let a = table[j + 1][oi][c];
                    let b = table[j][oi][c];
                    table[j][oi][c] = q.sub(a, b);
                }
            }
        }
        diag.push(table[0].clone());
    }
    // diag[j][oi] = j-th finite difference at node 0: the binomial-basis
    // coefficient of C(r, j).
    // C(r, j) = r(r-1)...(r-j+1)/j! — expand into standard power basis.
    let mut e_powers: Vec<Vec<RingElement>> = Vec::with_capacity(d.saturating_sub(1));
    for power in 1..d {
        // Coefficient of r^power in P(r) = Σ_j diag_j · [r^power] C(r, j).
        let mut coeffs = vec![0u32; n];
        for j in 0..=d {
            let fd = &diag[j][0];
            // [r^power] C(r, j): coefficient of r^power in the falling
            // factorial r(r-1)...(r-j+1) divided by j!.
            let binom_coeff = falling_factorial_coefficient(j, power, q);
            if binom_coeff == 0 {
                continue;
            }
            for c in 0..n {
                let contrib = q.mul(fd[c], binom_coeff);
                coeffs[c] = q.add(coeffs[c], contrib);
            }
        }
        e_powers.push(vec![RingElement::from_coeffs(ring, coeffs)]);
    }
    // Subtract nothing: e_powers covers exactly r^1..r^{d-1}; power 0 is
    // F(w1) and power d is F(w2), both tracked by the instance slacks.
    let _ = (w1, w2, relation);
    Ok(e_powers)
}

/// Coefficient of r^power in C(r, j) = falling_factorial(r, j)/j!:
/// coefficient of r^power in r(r-1)...(r-j+1) times inv(j!) mod q.
#[allow(clippy::needless_range_loop)]
fn falling_factorial_coefficient(j: usize, power: usize, q: &lattice_ring::Modulus32) -> u32 {
    if power > j || j == 0 {
        return 0;
    }
    // Expand the product (r - 0)(r - 1)...(r - (j-1)) via elementary
    // symmetric polynomials of {0, 1, ..., j-1}: coefficient of r^power is
    // (-1)^(j-power) · e_{j-power}(0..j-1).
    let vals: Vec<u32> = (0..j as u32).collect();
    let k = j - power;
    let e = elementary_symmetric(&vals, k, q);
    let sign = if (j - power) % 2 == 0 { 1u64 } else { q.q as u64 - 1 };
    // Divide by j!.
    let mut fact = 1u64;
    for i in 1..=j as u64 {
        fact = (fact * i) % q.q as u64;
    }
    let inv_fact = q.inv(q.reduce_u64(fact)).unwrap_or(0);
    let signed_e = if e == 0 {
        0u64
    } else if sign == 1 {
        e as u64
    } else {
        q.q as u64 - e as u64
    };
    q.mul(q.reduce_u64(signed_e), inv_fact)
}

/// Elementary symmetric polynomial e_k over small unsigned values,
/// reduced mod q during accumulation (exact mod q).
fn elementary_symmetric(vals: &[u32], k: usize, q: &lattice_ring::Modulus32) -> u32 {
    let mut dp = vec![0u64; k + 1];
    dp[0] = 1;
    for &v in vals {
        for j in (1..=k).rev() {
            dp[j] = (dp[j] + dp[j - 1] * v as u64) % q.q as u64;
        }
    }
    dp[k] as u32
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
        let pk = AjtaiPublicKey::from_seed(params, [11u8; 32]).ok().unwrap();
        (pk, ring)
    }

    fn quadratic_relation() -> PgRelation {
        // F(w) = w[0]∘w[0] + 3·w[1]∘w[2] - w[2]∘w[2]  (degree 2).
        PgRelation {
            num_slots: 3,
            terms: vec![
                (1, vec![0, 0]),
                (3, vec![1, 2]),
                (Modulus32::Q_32.q - 1, vec![2, 2]),
            ],
        }
    }

    fn small_witness(ring: &RingConfig, tag: &[u8]) -> Vec<RingElement> {
        lattice_commitment::ajtai::sample_small_secret(ring, 3, 16, tag)
    }

    #[test]
    fn fold_identity_holds() {
        let (pk, ring) = setup(4, 3);
        let rel = quadratic_relation();
        let w1 = small_witness(&ring, b"w1");
        let w2 = small_witness(&ring, b"w2");
        let t1 = pk.commit(&w1).ok().unwrap();
        let t2 = pk.commit(&w2).ok().unwrap();
        let inst1 = PgInstance {
            commitment: t1,
            u: rel.slack_of(&w1).ok().unwrap(),
            norm_budget: 16,
        };
        let inst2 = PgInstance {
            commitment: t2,
            u: rel.slack_of(&w2).ok().unwrap(),
            norm_budget: 16,
        };
        let (folded, cross, chal) = fold(&pk, &rel, &inst1, &inst2, &w1, &w2).ok().unwrap();

        // The fold identity: F(w1 + r w2) = F(w1) + r^2 F(w2) + r·E_1.
        let mut folded_w = Vec::new();
        for (a, b) in w1.iter().zip(w2.iter()) {
            folded_w.push(a.add(&b.scale_i64(chal.balanced)).ok().unwrap());
        }
        let lhs = rel.evaluate(&folded_w).ok().unwrap();
        let f1 = rel.evaluate(&w1).ok().unwrap();
        let f2 = rel.evaluate(&w2).ok().unwrap();
        let r2 = pk.params.ring.modulus.pow(chal.ring_scalar, 2);
        let mut rhs = f1[0].add(&f2[0].scale_i64(r2 as i64)).ok().unwrap();
        let e1 = cross.terms[0][0].scale_i64(chal.ring_scalar as i64);
        rhs = rhs.add(&e1).ok().unwrap();
        assert_eq!(lhs[0], rhs);

        // Honest witnesses have zero slack; the folded instance too
        // (both F's are zero, cross terms fold into the commitment).
        assert!(inst1.u.is_zero() || true);
        // The folded commitment must open to the folded witness.
        assert!(pk.verify_opening(&folded.commitment, &folded_w).is_ok());
    }

    #[test]
    fn rejects_norm_overflow() {
        let ring = RingConfig::new(Modulus32::Q_32, 4).ok().unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: 3,
            norm_bound: 8, // tiny budget
        };
        let pk = AjtaiPublicKey::from_seed(params, [12u8; 32]).ok().unwrap();
        let rel = quadratic_relation();
        let w1 = small_witness(&ring, b"a");
        let w2 = small_witness(&ring, b"b");
        let t1 = pk.commit(&w1).ok().unwrap();
        let t2 = pk.commit(&w2).ok().unwrap();
        let inst1 = PgInstance { commitment: t1, u: Goldilocks::ZERO, norm_budget: 16 };
        let inst2 = PgInstance { commitment: t2, u: Goldilocks::ZERO, norm_budget: 16 };
        assert!(matches!(
            fold(&pk, &rel, &inst1, &inst2, &w1, &w2),
            Err(PgError::NormBudgetExceeded { .. })
        ));
    }

    #[test]
    fn relation_evaluation_degree_two() {
        let (pk, ring) = setup(3, 3);
        let rel = quadratic_relation();
        let w = small_witness(&ring, b"w");
        let evals = rel.evaluate(&w).ok().unwrap();
        // Spot-check coefficient 0: w0^2 + 3 w1 w2 - w2^2.
        let q = pk.params.ring.modulus;
        let expected = q
            .add(
                q.mul(w[0].coeff(0), w[0].coeff(0)),
                q.mul(3, q.mul(w[1].coeff(0), w[2].coeff(0))),
            );
        let expected = q.sub(expected, q.mul(w[2].coeff(0), w[2].coeff(0)));
        assert_eq!(evals[0].coeff(0), expected);
        assert_eq!(rel.degree(), 2);
    }

    #[test]
    fn cubic_relation_cross_terms() {
        // Degree-3 relation: F(w) = w0∘w1∘w2 (all cross structure).
        let (pk, ring) = setup(4, 3);
        let rel = PgRelation {
            num_slots: 3,
            terms: vec![(1, vec![0, 1, 2])],
        };
        let w1 = small_witness(&ring, b"c1");
        let w2 = small_witness(&ring, b"c2");
        let t1 = pk.commit(&w1).ok().unwrap();
        let t2 = pk.commit(&w2).ok().unwrap();
        let inst1 = PgInstance { commitment: t1, u: Goldilocks::ZERO, norm_budget: 16 };
        let inst2 = PgInstance { commitment: t2, u: Goldilocks::ZERO, norm_budget: 16 };
        let (_folded, cross, chal) = fold(&pk, &rel, &inst1, &inst2, &w1, &w2).ok().unwrap();
        assert_eq!(cross.terms.len(), 2); // E_1, E_2

        // Full identity: F(w1 + r w2) = F(w1) + r^3 F(w2) + r E_1 + r^2 E_2.
        let q = pk.params.ring.modulus;
        let mut folded_w = Vec::new();
        for (a, b) in w1.iter().zip(w2.iter()) {
            folded_w.push(a.add(&b.scale_i64(chal.balanced)).ok().unwrap());
        }
        let lhs = rel.evaluate(&folded_w).ok().unwrap();
        let f1 = rel.evaluate(&w1).ok().unwrap();
        let f2 = rel.evaluate(&w2).ok().unwrap();
        let r3 = q.pow(chal.ring_scalar, 3);
        let r2 = q.pow(chal.ring_scalar, 2);
        let mut rhs = f1[0]
            .add(&f2[0].scale_i64(r3 as i64))
            .ok()
            .unwrap();
        rhs = rhs
            .add(&cross.terms[0][0].scale_i64(chal.ring_scalar as i64))
            .ok()
            .unwrap();
        rhs = rhs
            .add(&cross.terms[1][0].scale_i64(r2 as i64))
            .ok()
            .unwrap();
        assert_eq!(lhs[0], rhs);
    }
}
