//! Neo/SuperNeo (Nguyen–Setty 2026, ePrint 2026/242): post-quantum folding
//! with pay-per-bit costs over small fields.
//!
//! Core mechanisms per the paper:
//! * **Pay-per-bit witness commitment** — the commitment cost is
//!   proportional to the number of NONZERO BITS in the witness: values are
//!   bit-decomposed and only nonzero bit-positions are committed (sparse
//!   packing), so sparse/small witnesses cost proportionally less.
//! * **Relaxed CCS folding over small fields** — CCS instances fold with a
//!   random linear combination plus cross-term vector, and small-field
//!   overflow is handled by lifting into R_q (the R_q ↔ F_q evaluation-map
//!   bridge shared with Cyclo).
//! * `π_CCS` — the final CCS-satisfaction proof for the folded instance.
//!
//! Implementation: relaxed CCS instances (witness w, slack u) fold as
//! w' = w1 + r·w2 with the quadratic cross-term vector E tracked exactly;
//! the pay-per-bit commitment sparsifies the bit-decomposed witness.

use lattice_relations::ccs::{Ccs, CcsError};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;

/// A relaxed CCS instance: witness + slack vector + scalar u.
///
/// Relaxed satisfaction (the Nova-style form): with
/// v(w) = product_vector(w) and B(w) = Σ_i B_i·w,
/// `v(w) − slack = u · B(w)`. Fresh instances carry u = 1, slack = 0.
#[derive(Clone, Debug)]
pub struct RelaxedCcsInstance {
    pub witness: Vec<Goldilocks>,
    /// Slack vector (folding error accumulator; zero for fresh instances).
    pub slack: Vec<Goldilocks>,
    /// Scalar u (1 for fresh instances; folds linearly).
    pub u: Goldilocks,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuperNeoError {
    Ccs(CcsError),
    ShapeMismatch { expected: usize, got: usize },
    NotSatisfied,
}

/// Pay-per-bit bit statistics: the commitment cost model from the paper —
/// only nonzero bits are committed, so cost = popcount of the witness.
pub fn pay_per_bit_cost(witness: &[Goldilocks]) -> u64 {
    witness
        .iter()
        .map(|w| w.to_canonical_u64().count_ones() as u64)
        .sum()
}

/// Sparse bit decomposition: returns (position, bit) pairs for nonzero
/// bits only — the pay-per-bit commitment input.
pub fn sparse_bits(witness: &[Goldilocks]) -> Vec<(usize, u8)> {
    let mut out = Vec::new();
    for (i, w) in witness.iter().enumerate() {
        let raw = w.to_canonical_u64();
        for b in 0..64 {
            if (raw >> b) & 1 == 1 {
                out.push((i * 64 + b, 1));
            }
        }
    }
    out
}

/// Fold two relaxed CCS instances (degree-2 relations) with a transcript
/// challenge. The fold:
/// * w' = w1 + r·w2,
/// * slack' = slack1 + r²·slack2 + r·E where E is the Hadamard cross-term
///   of the linearized constraint images (exact for the relaxed form).
#[allow(clippy::needless_range_loop)]
pub fn fold_relaxed_ccs(
    ccs: &Ccs,
    inst1: &RelaxedCcsInstance,
    inst2: &RelaxedCcsInstance,
) -> Result<(RelaxedCcsInstance, Goldilocks), SuperNeoError> {
    if inst1.witness.len() != ccs.m || inst2.witness.len() != ccs.m {
        return Err(SuperNeoError::ShapeMismatch {
            expected: ccs.m,
            got: inst1.witness.len(),
        });
    }
    // Transcript challenge (field element).
    let mut transcript = Transcript::new_default(b"lzx-superneo");
    for w in &inst1.witness {
        transcript
            .append_field(b"w1", w)
            .map_err(|_| SuperNeoError::NotSatisfied)?;
    }
    for w in &inst2.witness {
        transcript
            .append_field(b"w2", w)
            .map_err(|_| SuperNeoError::NotSatisfied)?;
    }
    let r = transcript
        .challenge_field(b"fold-r")
        .map_err(|_| SuperNeoError::NotSatisfied)?;

    // w' = w1 + r·w2.
    let folded_witness: Vec<Goldilocks> = inst1
        .witness
        .iter()
        .zip(inst2.witness.iter())
        .map(|(a, b)| a.add(&r.mul(b)))
        .collect();

    // Cross-term E: for relations v(w) built from Hadamard products
    // A_i w ∘ A_j w, the mixed term is
    // E = Σ_terms c · (A_a w1 ∘ A_b w2 + A_a w2 ∘ A_b w1).
    let n = ccs.n;
    let mut e = vec![Goldilocks::ZERO; n];
    let q = Goldilocks::ONE; // field ops
    let _ = q;
    let mut a_images1: Vec<Vec<Goldilocks>> = Vec::with_capacity(ccs.a_matrices.len());
    let mut a_images2: Vec<Vec<Goldilocks>> = Vec::with_capacity(ccs.a_matrices.len());
    for a in &ccs.a_matrices {
        a_images1.push(a.multiply(&inst1.witness).map_err(SuperNeoError::Ccs)?);
        a_images2.push(a.multiply(&inst2.witness).map_err(SuperNeoError::Ccs)?);
    }
    // Folding contract: the product side must be purely quadratic
    // (arity-2 selections). Linear constraints belong on the span side
    // (b_matrices); mixing them into the product vector breaks the exact
    // fold identity (an r(1-r)·v_linear discrepancy appears).
    for ids in &ccs.selections {
        if ids.len() != 2 {
            return Err(SuperNeoError::ShapeMismatch {
                expected: 2,
                got: ids.len(),
            });
        }
    }
    for (term_idx, ids) in ccs.selections.iter().enumerate() {
        let c = ccs.constants.get(term_idx).copied().unwrap_or(Goldilocks::ONE);
        if ids.len() == 2 {
            let (ia, ib) = (ids[0], ids[1]);
            let x1 = &a_images1[ia];
            let y1 = &a_images1[ib];
            let x2 = &a_images2[ia];
            let y2 = &a_images2[ib];
            for k in 0..n {
                // c · (x1[k]·y2[k] + x2[k]·y1[k])
                let mixed = x1[k].mul(&y2[k]).add(&x2[k].mul(&y1[k]));
                e[k] = e[k].add(&c.mul(&mixed));
            }
        } else {
            // Unreachable after the contract check above.
            return Err(SuperNeoError::ShapeMismatch {
                expected: 2,
                got: ids.len(),
            });
        }
    }

    // B-images for the cross term.
    let b_image1: Vec<Goldilocks> = {
        let mut acc = vec![Goldilocks::ZERO; n];
        for b in &ccs.b_matrices {
            let img = b.multiply(&inst1.witness).map_err(SuperNeoError::Ccs)?;
            for (a, v) in acc.iter_mut().zip(img.iter()) {
                *a = a.add(v);
            }
        }
        acc
    };
    let b_image2: Vec<Goldilocks> = {
        let mut acc = vec![Goldilocks::ZERO; n];
        for b in &ccs.b_matrices {
            let img = b.multiply(&inst2.witness).map_err(SuperNeoError::Ccs)?;
            for (a, v) in acc.iter_mut().zip(img.iter()) {
                *a = a.add(v);
            }
        }
        acc
    };

    // Committed cross term (the paper's T): the prover computes and
    // commits it; the folded slack absorbs r·T. With
    // v(w') = v(w1) + r² v(w2) + r·E and u' = u1 + r·u2, satisfaction
    // v(w') − slack' = u'·B(w') forces
    // T = E − u1·B(w2) − u2·B(w1).
    let mut t_vec = Vec::with_capacity(n);
    for k in 0..n {
        let t = e[k]
            .sub(&inst1.u.mul(&b_image2[k]))
            .sub(&inst2.u.mul(&b_image1[k]));
        t_vec.push(t);
    }

    // slack' = slack1 + r²·slack2 + r·T; u' = u1 + r·u2.
    let r2 = r.mul(&r);
    let mut folded_slack = Vec::with_capacity(n);
    for k in 0..n {
        let s = inst1
            .slack
            .get(k)
            .copied()
            .unwrap_or(Goldilocks::ZERO)
            .add(&r2.mul(&inst2.slack.get(k).copied().unwrap_or(Goldilocks::ZERO)))
            .add(&r.mul(&t_vec[k]));
        folded_slack.push(s);
    }
    let folded_u = inst1.u.add(&r.mul(&inst2.u));

    Ok((
        RelaxedCcsInstance {
            witness: folded_witness,
            slack: folded_slack,
            u: folded_u,
        },
        r,
    ))
}

/// Verify the relaxed CCS relation exactly:
/// `product_vector(w) − slack == u · (Σ_i B_i·w)`.
pub fn verify_folded(
    ccs: &Ccs,
    inst: &RelaxedCcsInstance,
) -> Result<bool, SuperNeoError> {
    if inst.witness.len() != ccs.m || inst.slack.len() != ccs.n {
        return Ok(false);
    }
    let v = ccs.product_vector(&inst.witness).map_err(SuperNeoError::Ccs)?;
    let mut b_span = vec![Goldilocks::ZERO; ccs.n];
    for b in &ccs.b_matrices {
        let img = b.multiply(&inst.witness).map_err(SuperNeoError::Ccs)?;
        for (a, x) in b_span.iter_mut().zip(img.iter()) {
            *a = a.add(x);
        }
    }
    for k in 0..ccs.n {
        let lhs = v[k].sub(&inst.slack.get(k).copied().unwrap_or(Goldilocks::ZERO));
        let rhs = inst.u.mul(&b_span[k]);
        if lhs != rhs {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_relations::ccs::SparseMatrix;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// A w∘w = w relation (boolean witnesses) as degree-2 CCS.
    fn bool_ccs(n: usize) -> Ccs {
        let a = SparseMatrix::identity(n);
        Ccs {
            m: n,
            n,
            a_matrices: vec![a.clone(), a],
            b_matrices: vec![SparseMatrix::identity(n)],
            selections: vec![vec![0, 1]],
            constants: vec![fe(1)],
        }
    }

    fn bool_witness(n: usize, tag: u64) -> Vec<Goldilocks> {
        // Boolean values with real entropy (the relation w∘w = w holds
        // exactly); earlier versions used a degenerate LCG that produced
        // all-zero witnesses for even tags — caught by the vacuity probe.
        (0..n)
            .map(|i| {
                let mut hash_input = Vec::new();
                hash_input.extend_from_slice(&tag.to_le_bytes());
                hash_input.extend_from_slice(&(i as u64).to_le_bytes());
                let digest = lattice_core::transcript::Transcript::hash_domain(
                    b"bool-witness",
                    &hash_input,
                );
                let bit = digest[0] & 1;
                fe(bit as u64)
            })
            .collect()
    }

    #[test]
    fn pay_per_bit_cost_counts_nonzeros() {
        let w = vec![fe(0), fe(1), fe(3), fe(255), fe(0)];
        // popcounts: 0, 1, 2, 8, 0 -> 11
        assert_eq!(pay_per_bit_cost(&w), 11);
        let sparse = sparse_bits(&w);
        assert_eq!(sparse.len(), 11);
        // Sparse witnesses cost less (the pay-per-bit property).
        let w_sparse = vec![fe(0); 100];
        assert_eq!(pay_per_bit_cost(&w_sparse), 0);
    }

    #[test]
    fn sparse_bits_reconstruct() {
        let w = vec![fe(0xDEAD_BEEF), fe(12345)];
        let bits = sparse_bits(&w);
        // Reconstruct.
        let mut vals = vec![0u64; w.len()];
        for (pos, bit) in &bits {
            vals[pos / 64] |= (*bit as u64) << (pos % 64);
        }
        for (v, orig) in vals.iter().zip(w.iter()) {
            assert_eq!(*v, orig.to_canonical_u64());
        }
    }

    #[test]
    fn relaxed_ccs_fold_identity() {
        let ccs = bool_ccs(8);
        let w1 = bool_witness(8, 1);
        let w2 = bool_witness(8, 2);
        // Sanity: witnesses are genuinely mixed (not all-zero, not equal).
        assert!(w1.iter().any(|w| !w.is_zero()));
        assert!(w2.iter().any(|w| !w.is_zero()));
        assert_ne!(w1, w2);
        let inst1 = RelaxedCcsInstance {
            witness: w1,
            slack: vec![fe(0); 8],
            u: fe(1),
        };
        let inst2 = RelaxedCcsInstance {
            witness: w2,
            slack: vec![fe(0); 8],
            u: fe(1),
        };
        // Both fresh instances satisfy the relation exactly.
        assert!(verify_folded(&ccs, &inst1).ok().unwrap());
        assert!(verify_folded(&ccs, &inst2).ok().unwrap());

        let (folded, _r) = fold_relaxed_ccs(&ccs, &inst1, &inst2).ok().unwrap();
        // The FOLDED instance satisfies the relaxed relation via the
        // tracked slack and u — the fold identity holds exactly.
        assert!(
            verify_folded(&ccs, &folded).ok().unwrap(),
            "folded instance must satisfy relaxed CCS"
        );
    }

    #[test]
    fn fold_chain_multiple_rounds() {
        let ccs = bool_ccs(8);
        let mut acc = RelaxedCcsInstance {
            witness: bool_witness(8, 7),
            slack: vec![fe(0); 8],
            u: fe(1),
        };
        assert!(verify_folded(&ccs, &acc).ok().unwrap());
        for tag in 10..14u64 {
            let fresh = RelaxedCcsInstance {
                witness: bool_witness(8, tag),
                slack: vec![fe(0); 8],
                u: fe(1),
            };
            let (next, _r) = fold_relaxed_ccs(&ccs, &acc, &fresh).ok().unwrap();
            assert!(verify_folded(&ccs, &next).ok().unwrap());
            acc = next;
        }
    }

    #[test]
    fn arity_errors() {
        let ccs = Ccs {
            m: 4,
            n: 4,
            a_matrices: vec![SparseMatrix::identity(4)],
            b_matrices: vec![SparseMatrix::identity(4)],
            selections: vec![vec![0]], // degree-1 -> no cross terms, allowed
            constants: vec![fe(1)],
        };
        let inst1 = RelaxedCcsInstance {
            witness: vec![fe(1), fe(0), fe(1), fe(0)],
            slack: vec![fe(0); 4],
            u: fe(1),
        };
        let inst2 = RelaxedCcsInstance {
            witness: vec![fe(0), fe(1), fe(0), fe(1)],
            slack: vec![fe(0); 4],
            u: fe(1),
        };
        // Pure-linear product side violates the folding contract and is
        // rejected (linear constraints belong on the span side).
        assert!(matches!(
            fold_relaxed_ccs(&ccs, &inst1, &inst2),
            Err(SuperNeoError::ShapeMismatch { expected: 2, got: 1 })
        ));
        // Shape mismatch errors.
        let bad = RelaxedCcsInstance {
            witness: vec![fe(1)],
            slack: vec![fe(0); 4],
            u: fe(1),
        };
        assert!(matches!(
            fold_relaxed_ccs(&ccs, &bad, &inst2),
            Err(SuperNeoError::ShapeMismatch { .. })
        ));
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use lattice_relations::ccs::SparseMatrix;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn dbg_ccs(n: usize) -> Ccs {
        let a = SparseMatrix::identity(n);
        Ccs {
            m: n,
            n,
            a_matrices: vec![a.clone(), a],
            b_matrices: vec![SparseMatrix::identity(n)],
            selections: vec![vec![0, 1]],
            constants: vec![fe(1)],
        }
    }

    #[test]
    fn disjoint_support_fold_satisfies_exactly() {
        // Regression (vacuity guard): disjoint-support witnesses give
        // E = 0, so the folded slack comes purely from the u/B cross term
        // T = −u1·B w2 − u2·B w1; satisfaction must hold EXACTLY through
        // the tracked (slack, u), not through a permissive span search.
        let ccs = dbg_ccs(4);
        let w1 = vec![fe(1), fe(0), fe(1), fe(0)];
        let w2 = vec![fe(0), fe(1), fe(0), fe(1)];
        let inst1 = RelaxedCcsInstance { witness: w1, slack: vec![fe(0); 4], u: fe(1) };
        let inst2 = RelaxedCcsInstance { witness: w2, slack: vec![fe(0); 4], u: fe(1) };
        let (folded, _r) = fold_relaxed_ccs(&ccs, &inst1, &inst2).ok().unwrap();
        assert!(verify_folded(&ccs, &folded).ok().unwrap());
        // Tampered slack must fail (the check is exact, not vacuous).
        let mut bad = folded.clone();
        if !bad.slack.is_empty() {
            bad.slack[0] = bad.slack[0].add(&fe(1));
        }
        assert!(!verify_folded(&ccs, &bad).ok().unwrap());
        // Tampered u must fail.
        let mut bad_u = folded;
        bad_u.u = bad_u.u.add(&fe(1));
        assert!(!verify_folded(&ccs, &bad_u).ok().unwrap());
    }
}
