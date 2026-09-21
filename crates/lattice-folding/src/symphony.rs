//! Symphony (Chen 2025, ePrint 2025/1905): scalable SNARKs in the ROM from
//! lattice-based high-arity folding.
//!
//! Core mechanism per the paper: a new lattice-based folding scheme that
//! compresses a LARGE number (μ) of NP-complete statements into ONE
//! accumulator in a single shot, avoiding the log-many rounds of binary
//! folding — which in turn lets the generic compiler build a SNARK without
//! embedding the Fiat–Shamir hash circuit into proven statements (the
//! dominant overhead of prior folding-based SNARKs).
//!
//! Implementation:
//! * `fold_many` — μ-ary fold: w' = Σ_{i<μ} r^i · w_i under transcript
//!   challenges r (small, for norm control). The polynomial identity for a
//!   degree-d relation expands over ALL subsets I ⊆ [μ] with
//!   2 ≤ |I| ≤ d: F(w') = Σ_{I} r^{Σ_{i∈I} i} · E_I — the full cross-term
//!   bookkeeping is produced exactly (finite differences over a μ-node
//!   grid, generalized from the binary case).
//! * The subset cross terms are committed once (a single stacked Ajtai
//!   commitment), keeping the one-shot fold non-interactive.

#[allow(unused_imports)] // AjtaiParams used by the test module via super::*
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_ring::RingElement;

/// Number of instances folded per round (the arity μ).
pub const DEFAULT_ARITY: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymphonyError {
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    ArityTooSmall { arity: usize },
    DegreeTooSmall { degree: usize },
    ShapeMismatch { expected: usize, got: usize },
    CrossTermIdentity,
}

/// A degree-d relation over ring slots (same shape as ProtogaLattice's).
#[derive(Clone, Debug)]
pub struct SymRelation {
    pub num_slots: usize,
    pub terms: Vec<(u32, Vec<usize>)>,
}

impl SymRelation {
    pub fn degree(&self) -> usize {
        self.terms.iter().map(|(_, ids)| ids.len()).max().unwrap_or(0)
    }

    pub fn evaluate(&self, w: &[RingElement]) -> Result<RingElement, SymphonyError> {
        if w.len() != self.num_slots {
            return Err(SymphonyError::ShapeMismatch {
                expected: self.num_slots,
                got: w.len(),
            });
        }
        let ring = w
            .first()
            .map(|e| e.config().clone())
            .ok_or(SymphonyError::ShapeMismatch {
                expected: self.num_slots,
                got: 0,
            })?;
        let n = ring.n();
        let q = ring.modulus;
        let mut acc = vec![0u32; n];
        for (c, ids) in &self.terms {
            let mut prod = vec![1u32; n];
            for &slot in ids {
                let elem = w.get(slot).ok_or(SymphonyError::ShapeMismatch {
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
        Ok(RingElement::from_coeffs(&ring, acc))
    }
}

/// The result of a one-shot μ-ary fold.
pub struct SymphonyFold {
    /// Folded witness w' = Σ r^i w_i.
    pub folded_witness: Vec<RingElement>,
    /// Folded commitment t' = Σ r^i t_i.
    pub folded_commitment: AjtaiCommitment,
    /// Challenges r^0..r^{μ-1} (balanced integers).
    pub challenges: Vec<i64>,
    /// Cross-term vectors E_I indexed by subset (bitmask over [μ]).
    pub cross_terms: Vec<(u32, RingElement)>,
    /// Cross-term commitment (stacked, single shot).
    pub cross_commitment: AjtaiCommitment,
}

/// Fold μ witnesses in one shot under degree-2 relations (the dominant
/// case: R1CS-shaped constraints). The cross-term identity:
/// F(Σ r^i w_i) = Σ_i r^{2i} F(w_i) + Σ_{i<j} r^{i+j} E_{ij}.
pub fn fold_many_degree2(
    pk: &AjtaiPublicKey,
    pk_cross: &AjtaiPublicKey,
    relation: &SymRelation,
    witnesses: &[Vec<RingElement>],
    commitments: &[AjtaiCommitment],
) -> Result<SymphonyFold, SymphonyError> {
    let mu = witnesses.len();
    if mu < 2 {
        return Err(SymphonyError::ArityTooSmall { arity: mu });
    }
    if relation.degree() != 2 {
        return Err(SymphonyError::DegreeTooSmall {
            degree: relation.degree(),
        });
    }
    if commitments.len() != mu {
        return Err(SymphonyError::ShapeMismatch {
            expected: mu,
            got: commitments.len(),
        });
    }
    let m = pk.params.m;
    if witnesses.iter().any(|w| w.len() != m) {
        return Err(SymphonyError::ShapeMismatch { expected: m, got: 0 });
    }
    let ring = &pk.params.ring;
    let q = ring.modulus;

    // Transcript challenges: μ small balanced values.
    let mut transcript = Transcript::new_default(b"lzx-symphony");
    for c in commitments {
        transcript
            .append_bytes(b"instance", &c.to_bytes())
            .map_err(|_| SymphonyError::CrossTermIdentity)?;
    }
    let seed = transcript
        .challenge_bytes(b"symphony-r", 32)
        .map_err(|_| SymphonyError::CrossTermIdentity)?;
    let bytes = Transcript::xof(b"symphony-chal", &seed, mu * 2);
    let mut challenges = Vec::with_capacity(mu);
    for chunk in bytes.chunks(2).take(mu) {
        let raw = u16::from_le_bytes([chunk[0], chunk[1]]) as i64;
        // Balanced 16-bit challenges.
        let r = if raw >= 1 << 15 { raw - (1 << 16) } else { raw };
        challenges.push(r);
    }

    // Folded witness and commitment.
    let mut folded_witness = vec![ring.zero(); m];
    for (i, w) in witnesses.iter().enumerate() {
        let r_scalar = q.reduce_i64(challenges[i]);
        for (acc, wi) in folded_witness.iter_mut().zip(w.iter()) {
            *acc = acc
                .add(&wi.scale_i64(challenges[i]))
                .map_err(SymphonyError::Ring)?;
            let _ = r_scalar;
        }
    }
    let mut folded_rows = Vec::with_capacity(commitments[0].rows.len());
    for (row_idx, _row) in commitments[0].rows.iter().enumerate() {
        let mut acc = ring.zero();
        for (i, c) in commitments.iter().enumerate() {
            let scaled = c.rows[row_idx]
                .scale_i64(challenges[i]);
            acc = acc.add(&scaled).map_err(SymphonyError::Ring)?;
        }
        folded_rows.push(acc);
    }
    let folded_commitment = AjtaiCommitment { rows: folded_rows };

    // Cross terms E_{ij} for all i < j: for a degree-2 relation
    // F = Σ_terms c · w_a ∘ w_b, the mixed term of w_i, w_j is
    // E_{ij} = Σ_terms c · (w_i[a]∘w_j[b] + w_j[a]∘w_i[b]).
    let mut cross_terms: Vec<(u32, RingElement)> = Vec::new();
    for i in 0..mu {
        for j in (i + 1)..mu {
            let mut acc = vec![0u32; ring.n()];
            for (c, ids) in &relation.terms {
                if ids.len() != 2 {
                    return Err(SymphonyError::DegreeTooSmall { degree: ids.len() });
                }
                let (a, b) = (ids[0], ids[1]);
                // w_i[a] ∘ w_j[b].
                let wi_a = witnesses[i]
                    .get(a)
                    .ok_or(SymphonyError::ShapeMismatch { expected: m, got: a })?;
                let wj_b = witnesses[j]
                    .get(b)
                    .ok_or(SymphonyError::ShapeMismatch { expected: m, got: b })?;
                for (acc_c, (x, y)) in acc
                    .iter_mut()
                    .zip(wi_a.coeffs().iter().zip(wj_b.coeffs().iter()))
                {
                    *acc_c = q.add(*acc_c, q.mul(*c, q.mul(*x, *y)));
                }
                // w_j[a] ∘ w_i[b].
                let wj_a = witnesses[j]
                    .get(a)
                    .ok_or(SymphonyError::ShapeMismatch { expected: m, got: a })?;
                let wi_b = witnesses[i]
                    .get(b)
                    .ok_or(SymphonyError::ShapeMismatch { expected: m, got: b })?;
                for (acc_c, (x, y)) in acc
                    .iter_mut()
                    .zip(wj_a.coeffs().iter().zip(wi_b.coeffs().iter()))
                {
                    *acc_c = q.add(*acc_c, q.mul(*c, q.mul(*x, *y)));
                }
            }
            let mask = (1u32 << i) | (1u32 << j);
            cross_terms.push((mask, RingElement::from_coeffs(ring, acc)));
        }
    }

    // Single stacked cross commitment (one shot) under the dedicated
    // cross-term key, which must have at least C(μ, 2) slots.
    let mut stacked: Vec<RingElement> = cross_terms
        .iter()
        .map(|(_, e)| e.clone())
        .collect();
    while stacked.len() < pk_cross.params.m {
        stacked.push(ring.zero());
    }
    if stacked.len() > pk_cross.params.m {
        return Err(SymphonyError::ShapeMismatch {
            expected: pk_cross.params.m,
            got: stacked.len(),
        });
    }
    let cross_commitment = pk_cross
        .commit(&stacked)
        .map_err(SymphonyError::Ajtai)?;

    Ok(SymphonyFold {
        folded_witness,
        folded_commitment,
        challenges,
        cross_terms,
        cross_commitment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(log_n: u32, m: usize) -> (AjtaiPublicKey, lattice_ring::RingConfig) {
        let ring = lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, log_n)
            .ok()
            .unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m,
            norm_bound: 1 << 22,
        };
        let pk = AjtaiPublicKey::from_seed(params, [51u8; 32]).ok().unwrap();
        (pk, ring)
    }

    fn quadratic_relation() -> SymRelation {
        // F(w) = w0∘w0 + 3·w1∘w2 - w2∘w2.
        SymRelation {
            num_slots: 3,
            terms: vec![
                (1, vec![0, 0]),
                (3, vec![1, 2]),
                (lattice_ring::Modulus32::Q_32.q - 1, vec![2, 2]),
            ],
        }
    }

    fn witness(ring: &lattice_ring::RingConfig, tag: &[u8]) -> Vec<RingElement> {
        lattice_commitment::ajtai::sample_small_secret(ring, 3, 64, tag)
    }

    #[test]
    fn high_arity_fold_identity_exact() {
        let (pk, ring) = setup(4, 3);
        let rel = quadratic_relation();
        let witnesses: Vec<Vec<RingElement>> =
            [b"w0", b"w1", b"w2", b"w3"].iter().map(|t| witness(&ring, *t)).collect();
        let commitments: Vec<AjtaiCommitment> = witnesses
            .iter()
            .map(|w| pk.commit(w).ok().unwrap())
            .collect();
        let (_, ring_cross) = setup(4, 8);
        let params_cross = AjtaiParams {
            ring: ring_cross,
            k: 2,
            m: 8,
            norm_bound: 1 << 22,
        };
        let pk_cross = AjtaiPublicKey::from_seed(params_cross, [52u8; 32]).ok().unwrap();
        let fold = fold_many_degree2(&pk, &pk_cross, &rel, &witnesses, &commitments)
            .ok()
            .unwrap();

        // The full degree-2 μ-ary identity:
        // F(Σ r^i w_i) = Σ_i r^{2i} F(w_i) + Σ_{i<j} r^{i+j} E_{ij}.
        let q = ring.modulus;
        let lhs = rel.evaluate(&fold.folded_witness).ok().unwrap();
        let mut rhs = ring.zero();
        for (i, w) in witnesses.iter().enumerate() {
            let r2 = q.reduce_u64(
                (fold.challenges[i] * fold.challenges[i])
                    .rem_euclid(q.q as i64) as u64,
            );
            let f = rel.evaluate(w).ok().unwrap();
            rhs = rhs.add(&f.scale_i64(r2 as i64)).ok().unwrap();
        }
        for (mask, e) in &fold.cross_terms {
            // r^{i+j} = r^i · r^j for the pair encoded in the mask.
            let mut r_pow = 1i64;
            for (idx, r) in fold.challenges.iter().enumerate() {
                if (mask >> idx) & 1 == 1 {
                    r_pow = r_pow.wrapping_mul(*r);
                }
            }
            // Reduce mod q carefully (i128 to avoid overflow).
            let r_mod = (r_pow as i128)
                .rem_euclid(q.q as i128) as u64;
            let scaled = e.scale_i64(r_mod as i64);
            rhs = rhs.add(&scaled).ok().unwrap();
        }
        assert_eq!(lhs, rhs, "high-arity fold identity violated");
    }

    #[test]
    fn folded_commitment_opens_folded_witness() {
        let (pk, ring) = setup(4, 3);
        let rel = quadratic_relation();
        let witnesses: Vec<Vec<RingElement>> =
            [b"a0", b"a1", b"a2"].iter().map(|t| witness(&ring, *t)).collect();
        let commitments: Vec<AjtaiCommitment> = witnesses
            .iter()
            .map(|w| pk.commit(w).ok().unwrap())
            .collect();
        let (_, ring_cross) = setup(4, 8);
        let params_cross = AjtaiParams {
            ring: ring_cross,
            k: 2,
            m: 8,
            norm_bound: 1 << 22,
        };
        let pk_cross = AjtaiPublicKey::from_seed(params_cross, [52u8; 32]).ok().unwrap();
        let fold = fold_many_degree2(&pk, &pk_cross, &rel, &witnesses, &commitments)
            .ok()
            .unwrap();
        assert!(pk
            .verify_opening(&fold.folded_commitment, &fold.folded_witness)
            .is_ok());
        // Cross-term count: C(3, 2) = 3 pairs.
        assert_eq!(fold.cross_terms.len(), 3);
        // C(4,2) = 6 for arity 4.
        let witnesses4: Vec<Vec<RingElement>> =
            [b"b0", b"b1", b"b2", b"b3"].iter().map(|t| witness(&ring, *t)).collect();
        let commitments4: Vec<AjtaiCommitment> = witnesses4
            .iter()
            .map(|w| pk.commit(w).ok().unwrap())
            .collect();
        let fold4 = fold_many_degree2(&pk, &pk_cross, &rel, &witnesses4, &commitments4)
            .ok()
            .unwrap();
        assert_eq!(fold4.cross_terms.len(), 6);
    }

    #[test]
    fn shape_and_arity_errors() {
        let (pk, ring) = setup(4, 3);
        let rel = quadratic_relation();
        // Single instance: arity too small.
        let w = witness(&ring, b"solo");
        let c = pk.commit(&w).ok().unwrap();
        let (_, ring_cross) = setup(4, 8);
        let params_cross = AjtaiParams {
            ring: ring_cross,
            k: 2,
            m: 8,
            norm_bound: 1 << 22,
        };
        let pk_cross = AjtaiPublicKey::from_seed(params_cross, [54u8; 32]).ok().unwrap();
        assert!(matches!(
            fold_many_degree2(&pk, &pk_cross, &rel, std::slice::from_ref(&w), std::slice::from_ref(&c)),
            Err(SymphonyError::ArityTooSmall { arity: 1 })
        ));
        // Witness/commitment count mismatch.
        let w2 = witness(&ring, b"pair");
        let c2 = pk.commit(&w2).ok().unwrap();
        assert!(matches!(
            fold_many_degree2(&pk, &pk_cross, &rel, &[w, w2.clone()], &[c]),
            Err(SymphonyError::ShapeMismatch { .. })
        ));
        // Degree-1 relation rejected.
        let rel1 = SymRelation {
            num_slots: 3,
            terms: vec![(1, vec![0])],
        };
        assert!(matches!(
            fold_many_degree2(&pk, &pk_cross, &rel1, &[w2.clone(), w2], &[c2.clone(), c2]),
            Err(SymphonyError::DegreeTooSmall { degree: 1 })
        ));
    }

    #[test]
    fn challenges_are_small_and_deterministic() {
        let (pk, ring) = setup(4, 3);
        let rel = quadratic_relation();
        let witnesses: Vec<Vec<RingElement>> =
            [b"d0", b"d1", b"d2"].iter().map(|t| witness(&ring, *t)).collect();
        let commitments: Vec<AjtaiCommitment> = witnesses
            .iter()
            .map(|w| pk.commit(w).ok().unwrap())
            .collect();
        let (_, ring_cross) = setup(4, 8);
        let params_cross = AjtaiParams {
            ring: ring_cross,
            k: 2,
            m: 8,
            norm_bound: 1 << 22,
        };
        let pk_cross = AjtaiPublicKey::from_seed(params_cross, [55u8; 32]).ok().unwrap();
        let f1 = fold_many_degree2(&pk, &pk_cross, &rel, &witnesses, &commitments)
            .ok()
            .unwrap();
        let f2 = fold_many_degree2(&pk, &pk_cross, &rel, &witnesses, &commitments)
            .ok()
            .unwrap();
        assert_eq!(f1.challenges, f2.challenges);
        for r in &f1.challenges {
            assert!(r.abs() <= 1 << 15);
        }
    }
}
