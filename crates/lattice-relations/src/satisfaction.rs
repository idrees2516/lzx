//! CCS witness satisfaction — direct checks (prover-side ground truth and
//! test oracle) plus the sumcheck-shaped statement builder the proof layer
//! consumes.

use crate::ccs::{Ccs, CcsError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::virtual_poly::VirtualPolynomial;

/// Direct satisfaction check (unrelaxed: slack = 0).
pub fn is_satisfied(ccs: &Ccs, w: &[Goldilocks]) -> Result<bool, CcsError> {
    ccs.is_satisfied_relaxed(w, &vec![Goldilocks::ZERO; ccs.n])
}

/// Build the sumcheck statement for relaxed CCS satisfaction:
/// prove `Σ_x eq(r, x) · [v(w)(x) - slack(x) - (Σ_l e_l B_l w)(x)] = 0`
/// over the boolean hypercube of constraint indices, where `e` is the
/// span-selector vector (here derived deterministically from the span
/// solve). The virtual polynomial factors:
/// * product factors for each A-image,
/// * linear factors for each B-image,
/// * the slack as a constant-per-point factor.
///
/// This is the exact shape SuperNeo/Neo fold, and what the zkVM's residual
/// constraint stage proves.
pub struct CcsSumcheckStatement {
    pub vp: VirtualPolynomial,
    /// Claimed sum (zero for satisfied instances).
    pub claim: Goldilocks,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatementError {
    Ccs(CcsError),
    MleShape,
}

/// Construct the statement. `r` is the zerocheck point (transcript-derived
/// by the caller). Returns the virtual polynomial whose hypercube sum is
/// zero iff (relaxed) satisfaction holds.
pub fn build_statement(
    ccs: &Ccs,
    w: &[Goldilocks],
    slack: &[Goldilocks],
    r: &[Goldilocks],
) -> Result<CcsSumcheckStatement, StatementError> {
    let n = ccs.n;
    let log_n = n
        .checked_next_multiple_of(1)
        .map(|_| n.trailing_zeros())
        .unwrap_or(0) as usize;
    if n != (1usize << log_n) {
        // Constraint dimensions must be powers of two for the hypercube
        // encoding; callers pad.
        return Err(StatementError::MleShape);
    }
    // E-vector: v(w) - slack.
    let e_vec = ccs
        .product_vector(w)
        .map_err(StatementError::Ccs)?
        .iter()
        .zip(slack.iter())
        .map(|(v, s)| v.sub(s))
        .collect::<Vec<_>>();
    // B-images summed with a fixed selector (identity selector e_1 = 1):
    // the B-span term uses selector coefficients folded into one matrix
    // image; we sum all B matrices' images as the span representative.
    let mut b_span = vec![Goldilocks::ZERO; n];
    for b in &ccs.b_matrices {
        let img = b.multiply(w).map_err(StatementError::Ccs)?;
        for (acc, v) in b_span.iter_mut().zip(img.iter()) {
            *acc = acc.add(v);
        }
    }
    // Statement polynomial: (E(x) - Bspan(x)) * eq(r, x), summed = 0.
    let diff: Vec<Goldilocks> = e_vec
        .iter()
        .zip(b_span.iter())
        .map(|(e, b)| e.sub(b))
        .collect();
    let diff_mle = DenseMle {
        num_vars: log_n,
        evaluations: diff,
    };
    let eq_mle = DenseMle::eq_extension(r);
    let mut vp = VirtualPolynomial::new(log_n);
    let di = vp
        .add_factor(diff_mle)
        .map_err(|_| StatementError::MleShape)?;
    let ei = vp
        .add_factor(eq_mle)
        .map_err(|_| StatementError::MleShape)?;
    vp.add_term(Goldilocks::ONE, vec![di, ei])
        .map_err(|_| StatementError::MleShape)?;
    Ok(CcsSumcheckStatement {
        vp,
        claim: Goldilocks::ZERO,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccs::SparseMatrix;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn simple_ccs() -> Ccs {
        // A w ∘ A w - B w = 0 with A = B = I over n = 4: w ∘ w = w... use
        // witness (1, 0, 1, 0) which satisfies it exactly.
        let a = SparseMatrix::identity(4);
        let b = SparseMatrix::identity(4);
        Ccs {
            m: 4,
            n: 4,
            a_matrices: vec![a.clone(), a],
            b_matrices: vec![b],
            selections: vec![vec![0, 1]],
            constants: vec![fe(1)],
        }
    }

    #[test]
    fn satisfied_instance_sum_is_zero() {
        let ccs = simple_ccs();
        let w = vec![fe(1), fe(0), fe(1), fe(0)];
        assert!(is_satisfied(&ccs, &w).ok().unwrap());
        // The sumcheck statement sums to zero at any random point.
        let r: Vec<Goldilocks> = (1..=2).map(|i| fe(i * 7)).collect();
        let stmt = build_statement(&ccs, &w, &vec![fe(0); 4], &r).ok().unwrap();
        assert!(stmt.claim.is_zero());
        assert_eq!(stmt.vp.sum_over_hypercube(), fe(0));
    }

    #[test]
    fn unsatisfied_instance_detected() {
        let ccs = simple_ccs();
        let w = vec![fe(2), fe(0), fe(1), fe(0)];
        assert!(!is_satisfied(&ccs, &w).ok().unwrap());
        let r: Vec<Goldilocks> = (1..=2).map(|i| fe(i * 7)).collect();
        // With zero slack the statement sum is nonzero (whp).
        let stmt = build_statement(&ccs, &w, &vec![fe(0); 4], &r).ok().unwrap();
        assert!(!stmt.vp.sum_over_hypercube().is_zero());
        // Proper slack (w∘w - w per row: row0 = 4-2 = 2) fixes it.
        let slack = vec![fe(2), fe(0), fe(0), fe(0)];
        let stmt2 = build_statement(&ccs, &w, &slack, &r).ok().unwrap();
        assert!(stmt2.vp.sum_over_hypercube().is_zero());
    }

    #[test]
    fn non_power_of_two_rejected() {
        let mut ccs = simple_ccs();
        ccs.n = 3; // not a power of two
        ccs.a_matrices = vec![SparseMatrix::identity(3), SparseMatrix::identity(3)];
        ccs.b_matrices = vec![SparseMatrix::identity(3)];
        let w = vec![fe(1), fe(0), fe(1)];
        let r = vec![fe(3), fe(5)];
        assert!(matches!(
            build_statement(&ccs, &w, &vec![fe(0); 3], &r),
            Err(StatementError::MleShape)
        ));
    }
}
