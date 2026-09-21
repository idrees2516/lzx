//! Virtual polynomials: `P(x) = Σ_j c_j · Π_i f_{j,i}(x)` — the product
//! structure every staged protocol relation compiles to before sumcheck.

use lattice_core::{DenseMle, Goldilocks};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VirtualPolyError {
    VariableCountMismatch { expected: usize, got: usize },
    EmptyProduct,
}

impl From<lattice_core::mle::MleError> for VirtualPolyError {
    fn from(e: lattice_core::mle::MleError) -> Self {
        match e {
            lattice_core::mle::MleError::PointLengthMismatch { expected, got } => {
                VirtualPolyError::VariableCountMismatch { expected, got }
            }
            other => VirtualPolyError::VariableCountMismatch {
                expected: 0,
                got: match other {
                    lattice_core::mle::MleError::WrongEvaluationCount { got, .. } => got,
                    _ => 0,
                },
            },
        }
    }
}

impl core::fmt::Display for VirtualPolyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VirtualPolyError::VariableCountMismatch { expected, got } => {
                write!(f, "virtual polynomial variable count {got} != {expected}")
            }
            VirtualPolyError::EmptyProduct => {
                write!(f, "virtual polynomial product term with no factors")
            }
        }
    }
}

/// A virtual polynomial over a set of *shared* MLE factors.
///
/// All factors must have the same variable count; terms reference factor
/// indices. This layout lets the sumcheck prover bind each factor once per
/// round regardless of how many terms reuse it.
#[derive(Clone, Debug)]
pub struct VirtualPolynomial {
    pub num_vars: usize,
    /// Shared factor pool (each 2^num_vars evaluations).
    pub factors: Vec<DenseMle>,
    /// (coefficient, factor indices per term, multiplicity-free).
    pub terms: Vec<(Goldilocks, Vec<usize>)>,
}

impl VirtualPolynomial {
    /// Empty polynomial (identically zero) in `num_vars` variables.
    pub fn new(num_vars: usize) -> Self {
        VirtualPolynomial {
            num_vars,
            factors: Vec::new(),
            terms: Vec::new(),
        }
    }

    /// Add a factor and return its index.
    pub fn add_factor(&mut self, factor: DenseMle) -> Result<usize, VirtualPolyError> {
        if factor.num_vars != self.num_vars {
            return Err(VirtualPolyError::VariableCountMismatch {
                expected: self.num_vars,
                got: factor.num_vars,
            });
        }
        self.factors.push(factor);
        Ok(self.factors.len() - 1)
    }

    /// Add a product term `c · Π f_{indices}`.
    pub fn add_term(
        &mut self,
        coeff: Goldilocks,
        indices: Vec<usize>,
    ) -> Result<(), VirtualPolyError> {
        if indices.is_empty() {
            return Err(VirtualPolyError::EmptyProduct);
        }
        if indices.iter().any(|i| *i >= self.factors.len()) {
            return Err(VirtualPolyError::VariableCountMismatch {
                expected: self.factors.len(),
                got: indices.iter().max().copied().unwrap_or(0) + 1,
            });
        }
        self.terms.push((coeff, indices));
        Ok(())
    }

    /// Maximum degree in any single variable (max term length).
    pub fn max_degree(&self) -> usize {
        self.terms
            .iter()
            .map(|(_, ids)| ids.len())
            .max()
            .unwrap_or(0)
    }

    /// Sum of P over the full boolean hypercube.
    #[allow(clippy::needless_range_loop)]
    pub fn sum_over_hypercube(&self) -> Goldilocks {
        // Precompute per-point products term by term.
        let n = 1usize << self.num_vars;
        let mut acc = Goldilocks::ZERO;
        for (coeff, ids) in &self.terms {
            for pt in 0..n {
                let mut prod = Goldilocks::ONE;
                for fi in ids {
                    prod = prod.mul(&self.factors[*fi].evaluations[pt]);
                }
                acc = acc.add(&coeff.mul(&prod));
            }
        }
        acc
    }

    /// Evaluate P at an arbitrary point.
    pub fn evaluate(&self, point: &[Goldilocks]) -> Result<Goldilocks, VirtualPolyError> {
        if point.len() != self.num_vars {
            return Err(VirtualPolyError::VariableCountMismatch {
                expected: self.num_vars,
                got: point.len(),
            });
        }
        let mut acc = Goldilocks::ZERO;
        for (coeff, ids) in &self.terms {
            let mut prod = *coeff;
            for fi in ids {
                prod = prod.mul(&self.factors[*fi].evaluate(point)?);
            }
            acc = acc.add(&prod);
        }
        Ok(acc)
    }

    /// Materialize the dense MLE of P (exponential; tests/small cases).
    #[allow(clippy::needless_range_loop)]
    pub fn to_dense_mle(&self) -> Result<DenseMle, VirtualPolyError> {
        let n = 1usize << self.num_vars;
        let mut evals = vec![Goldilocks::ZERO; n];
        for (coeff, ids) in &self.terms {
            for pt in 0..n {
                let mut prod = Goldilocks::ONE;
                for fi in ids {
                    prod = prod.mul(&self.factors[*fi].evaluations[pt]);
                }
                evals[pt] = evals[pt].add(&coeff.mul(&prod));
            }
        }
        Ok(DenseMle {
            num_vars: self.num_vars,
            evaluations: evals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_core::Goldilocks;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn sum_matches_dense_mle() {
        let f = DenseMle::random(4, b"f");
        let g = DenseMle::random(4, b"g");
        let h = DenseMle::random(4, b"h");
        let mut vp = VirtualPolynomial::new(4);
        let fi = vp.add_factor(f.clone()).ok().unwrap();
        let gi = vp.add_factor(g.clone()).ok().unwrap();
        let hi = vp.add_factor(h.clone()).ok().unwrap();
        vp.add_term(fe(3), vec![fi, gi]).ok().unwrap();
        vp.add_term(fe(5), vec![fi, hi, gi]).ok().unwrap();
        vp.add_term(fe(7), vec![hi]).ok().unwrap();

        let dense = vp.to_dense_mle().ok().unwrap();
        assert_eq!(vp.sum_over_hypercube(), dense.sum_over_hypercube());
        // NOTE: point evaluation of `dense` is NOT expected to match
        // vp.evaluate — the product of MLEs has degree > 1 per variable,
        // so the dense table is not a multilinear extension of P. Only the
        // hypercube sums agree (they sum the same evaluation table).
        let pt: Vec<Goldilocks> = (1..=4).map(|i| fe(i * 3)).collect();
        let _ = dense.evaluate(&pt).ok().unwrap();
        assert!(vp.evaluate(&pt).is_ok());
        assert_eq!(vp.max_degree(), 3);
    }

    #[test]
    fn shape_errors() {
        let mut vp = VirtualPolynomial::new(3);
        assert!(matches!(
            vp.add_factor(DenseMle::random(4, b"x")),
            Err(VirtualPolyError::VariableCountMismatch {
                expected: 3,
                got: 4
            })
        ));
        let fi = vp.add_factor(DenseMle::random(3, b"y")).ok().unwrap();
        assert!(matches!(
            vp.add_term(fe(1), vec![]),
            Err(VirtualPolyError::EmptyProduct)
        ));
        assert!(vp.add_term(fe(1), vec![fi]).is_ok());
        assert!(vp.add_term(fe(1), vec![fi + 5]).is_err());
    }

    #[test]
    fn empty_poly_is_zero() {
        let vp = VirtualPolynomial::new(5);
        assert_eq!(vp.sum_over_hypercube(), fe(0));
        assert_eq!(vp.max_degree(), 0);
        let dense = vp.to_dense_mle().ok().unwrap();
        assert!(dense.evaluations.iter().all(|e| e.is_zero()));
    }
}
