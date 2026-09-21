//! Customizable Constraint Systems (CCS).
//!
//! A CCS instance over field F with witness `w ∈ F^m` (z = (w, 1)) is
//! defined by:
//! * `t` sparse matrices `A_1..A_t ∈ F^{N × m}` (N constraints),
//! * `d`-arity products selected by `S ⊆ [t]^d` (q selections),
//! * constants `c ∈ F^q` and a multiset-equality vector of weights
//!   `γ ∈ F^Σ` used by the folding layer.
//!
//! Satisfaction: `Σ_{j=1}^{q} c_j · Π_{k=1}^{d} ⟨A_{σ(j,k)} · w, γ⟩ = 0`
//! for every... — the canonical form used by folding (Nova/SuperNeo
//! lineage) is:
//!
//! ```text
//! Σ_{j=1}^{q} c_j · Π_{k∈S_j} (A_k · w)  ∈  span{B_1, ..., B_l} · w
//! ```
//!
//! We implement the equivalent "relaxed" satisfaction used by the folding
//! family: given matrices and selections, the vector
//! `v(w)_j = c_j · Π_{k∈S_j} (A_k · w)` must lie in the span of the
//! `B`-matrix images — the slack `u` absorbs folding error terms
//! (relaxed R1CS is the (t=3, d=2, l=1) special case).

use lattice_core::Goldilocks;

/// Sparse matrix in (row, col, value) triplets over F.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    /// Nonzero entries: (row, col, value).
    pub entries: Vec<(usize, usize, Goldilocks)>,
}

impl SparseMatrix {
    pub fn zero(rows: usize, cols: usize) -> Self {
        SparseMatrix {
            rows,
            cols,
            entries: Vec::new(),
        }
    }

    pub fn identity(n: usize) -> Self {
        SparseMatrix {
            rows: n,
            cols: n,
            entries: (0..n).map(|i| (i, i, Goldilocks::ONE)).collect(),
        }
    }

    /// Multiply: v = M · w (w has `cols` entries).
    pub fn multiply(&self, w: &[Goldilocks]) -> Result<Vec<Goldilocks>, CcsError> {
        if w.len() != self.cols {
            return Err(CcsError::DimensionMismatch {
                expected: self.cols,
                got: w.len(),
            });
        }
        let mut out = vec![Goldilocks::ZERO; self.rows];
        for (r, c, v) in &self.entries {
            if *r >= self.rows || *c >= self.cols {
                return Err(CcsError::IndexOutOfRange { row: *r, col: *c });
            }
            out[*r] = out[*r].add(&v.mul(&w[*c]));
        }
        Ok(out)
    }
}

/// A CCS structure: matrices, selection sets, constants.
#[derive(Clone, Debug)]
pub struct Ccs {
    /// Witness length m.
    pub m: usize,
    /// Constraint dimension N (rows of the A/B matrices).
    pub n: usize,
    /// "A" matrices (t of them): applied to the witness.
    pub a_matrices: Vec<SparseMatrix>,
    /// "B" matrices (l of them): span the allowed image space.
    pub b_matrices: Vec<SparseMatrix>,
    /// Selections: each is a tuple of indices into a_matrices (arity d).
    pub selections: Vec<Vec<usize>>,
    /// Constants c_j per selection.
    pub constants: Vec<Goldilocks>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CcsError {
    DimensionMismatch { expected: usize, got: usize },
    IndexOutOfRange { row: usize, col: usize },
    SelectionIndex { index: usize },
    MatrixIndex { index: usize },
    EmptySelection,
    ArityMismatch { expected: usize, got: usize },
}

impl Ccs {
    /// The product-structure vector v(w): entrywise (Hadamard) products of
    /// the selected matrix images, scaled by constants and summed across
    /// selections — the folding family's E-vector form.
    pub fn product_vector(&self, w: &[Goldilocks]) -> Result<Vec<Goldilocks>, CcsError> {
        if w.len() != self.m {
            return Err(CcsError::DimensionMismatch {
                expected: self.m,
                got: w.len(),
            });
        }
        // Compute all A_i · w once.
        let mut a_images = Vec::with_capacity(self.a_matrices.len());
        for a in &self.a_matrices {
            a_images.push(a.multiply(w)?);
        }
        let arity = self.selections.first().map(|s| s.len()).unwrap_or(0);
        let mut out = vec![Goldilocks::ZERO; self.n];
        for (j, sel) in self.selections.iter().enumerate() {
            if sel.is_empty() {
                return Err(CcsError::EmptySelection);
            }
            if sel.len() != arity {
                return Err(CcsError::ArityMismatch {
                    expected: arity,
                    got: sel.len(),
                });
            }
            // Hadamard product across the selection's images.
            let mut acc = vec![Goldilocks::ONE; self.n];
            for &ai in sel {
                let img = a_images
                    .get(ai)
                    .ok_or(CcsError::SelectionIndex { index: ai })?;
                if img.len() != self.n {
                    return Err(CcsError::DimensionMismatch {
                        expected: self.n,
                        got: img.len(),
                    });
                }
                for (acc_i, img_i) in acc.iter_mut().zip(img.iter()) {
                    *acc_i = acc_i.mul(img_i);
                }
            }
            let c = self.constants.get(j).copied().unwrap_or(Goldilocks::ZERO);
            for (o, a) in out.iter_mut().zip(acc.iter()) {
                *o = o.add(&c.mul(a));
            }
        }
        Ok(out)
    }

    /// Relaxed satisfaction: v(w) must equal Σ_l B_i · w · e_i for some
    /// selector — equivalently v(w) ∈ rowspace-span of B images. The
    /// folding layer proves this via sumcheck; here we check directly by
    /// solving: compute B_i·w and verify v(w) is in their span (small
    /// instances / tests).
    pub fn is_satisfied_relaxed(
        &self,
        w: &[Goldilocks],
        slack: &[Goldilocks],
    ) -> Result<bool, CcsError> {
        let v = self.product_vector(w)?;
        let mut target = v;
        // Subtract slack contributions (folding error absorbed here).
        if slack.len() != self.n {
            return Err(CcsError::DimensionMismatch {
                expected: self.n,
                got: slack.len(),
            });
        }
        for (t, s) in target.iter_mut().zip(slack.iter()) {
            *t = t.sub(s);
        }
        // Span check: target in span{B_i · w}.
        let images: Vec<Vec<Goldilocks>> = self
            .b_matrices
            .iter()
            .map(|b| b.multiply(w))
            .collect::<Result<_, _>>()?;
        // Solve coefficients greedily via elimination over the small
        // instance: for each nonzero position of target, try to cancel.
        // (Exact linear algebra; proof systems replace this entirely.)
        let mut residual = target.clone();
        for img in &images {
            // Find a pivot row where this image is nonzero.
            let mut coeff = None;
            for (i, val) in img.iter().enumerate() {
                if !val.is_zero() && !residual[i].is_zero() {
                    if let Some(inv) = val.inverse() {
                        coeff = Some(residual[i].mul(&inv));
                        break;
                    }
                }
            }
            if let Some(c) = coeff {
                for (i, val) in img.iter().enumerate() {
                    residual[i] = residual[i].sub(&c.mul(val));
                }
            }
        }
        Ok(residual.iter().all(|r| r.is_zero()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn sparse_matrix_multiply() {
        // M = [[0, 1], [2, 0], [0, 3]]
        let m = SparseMatrix {
            rows: 3,
            cols: 2,
            entries: vec![(0, 1, fe(1)), (1, 0, fe(2)), (2, 1, fe(3))],
        };
        let w = [fe(5), fe(7)];
        assert_eq!(m.multiply(&w).ok().unwrap(), vec![fe(7), fe(10), fe(21)]);
        assert!(m.multiply(&[fe(1)]).is_err());
    }

    #[test]
    fn identity_multiply() {
        let i = SparseMatrix::identity(3);
        let w = [fe(1), fe(2), fe(3)];
        assert_eq!(i.multiply(&w).ok().unwrap(), w.to_vec());
    }

    #[test]
    fn r1cs_as_ccs_satisfied() {
        // Relaxed-R1CS special case: A w ∘ B w = C w + slack.
        // A = I, B = I, C = I on m = n = 3; witness w with w ∘ w = w + slack.
        let a = SparseMatrix::identity(3);
        let c = SparseMatrix::identity(3);
        let ccs = Ccs {
            m: 3,
            n: 3,
            a_matrices: vec![a.clone(), a],
            b_matrices: vec![c],
            selections: vec![vec![0, 1]],
            constants: vec![fe(1)],
        };
        // w = (1, 0, 1): w∘w = w exactly -> slack zero.
        let w = [fe(1), fe(0), fe(1)];
        assert!(ccs.is_satisfied_relaxed(&w, &[fe(0); 3]).ok().unwrap());
        // w = (2, 0, 1): w∘w - w = (2, 0, 0) -> nonzero slack needed.
        let w2 = [fe(2), fe(0), fe(1)];
        assert!(!ccs.is_satisfied_relaxed(&w2, &[fe(0); 3]).ok().unwrap());
        // Correct slack fixes it.
        assert!(ccs
            .is_satisfied_relaxed(&w2, &[fe(2), fe(0), fe(0)])
            .ok()
            .unwrap());
    }

    #[test]
    fn shape_errors() {
        let ccs = Ccs {
            m: 2,
            n: 2,
            a_matrices: vec![SparseMatrix::identity(2)],
            b_matrices: vec![SparseMatrix::identity(2)],
            selections: vec![vec![0, 1]], // index 1 out of range
            constants: vec![fe(1)],
        };
        assert!(matches!(
            ccs.product_vector(&[fe(1), fe(2)]),
            Err(CcsError::SelectionIndex { index: 1 })
        ));
        assert!(matches!(
            ccs.product_vector(&[fe(1)]),
            Err(CcsError::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
    }
}
