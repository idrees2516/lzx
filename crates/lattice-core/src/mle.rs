//! Dense multilinear extensions over the boolean hypercube.
//!
//! A `DenseMle` in `m` variables stores 2^m evaluations in the canonical
//! big-endian variable order: the evaluation at the boolean point
//! `(b_0, ..., b_{m-1})` is stored at index `sum_i b_i * 2^(m-1-i)`
//! (variable 0 is the most significant index bit). `fix_variables` binds
//! variables in order 0, 1, ..., matching this convention.

use crate::field::Goldilocks;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMle {
    pub num_vars: usize,
    /// 2^num_vars evaluations in little-endian bit order.
    pub evaluations: Vec<Goldilocks>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MleError {
    WrongEvaluationCount { expected: usize, got: usize },
    PointLengthMismatch { expected: usize, got: usize },
}

impl DenseMle {
    /// Build from evaluations; length must be a power of two.
    pub fn new(evaluations: Vec<Goldilocks>) -> Result<Self, MleError> {
        let len = evaluations.len();
        if !len.is_power_of_two() {
            return Err(MleError::WrongEvaluationCount { expected: len, got: len });
        }
        Ok(DenseMle {
            num_vars: len.trailing_zeros() as usize,
            evaluations,
        })
    }

    /// Constant polynomial.
    pub fn constant(c: Goldilocks) -> Self {
        DenseMle { num_vars: 0, evaluations: vec![c] }
    }

    pub fn zero(num_vars: usize) -> Self {
        DenseMle { num_vars, evaluations: vec![Goldilocks::ZERO; 1 << num_vars] }
    }

    pub fn one(num_vars: usize) -> Self {
        DenseMle { num_vars, evaluations: vec![Goldilocks::ONE; 1 << num_vars] }
    }

    pub fn len(&self) -> usize {
        self.evaluations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.evaluations.is_empty()
    }

    /// Sample a random MLE from a seed (test/reference only, NOT for proofs).
    pub fn random(num_vars: usize, seed: &[u8]) -> Self {
        let bytes = crate::transcript::Transcript::xof(b"mle-random", seed, (1 << num_vars) * 8);
        let mut evals = Vec::with_capacity(1 << num_vars);
        for chunk in bytes.chunks(8) {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&chunk[..8.min(chunk.len())]);
            evals.push(Goldilocks::from_u64(u64::from_le_bytes(arr)));
        }
        DenseMle { num_vars, evaluations: evals }
    }

    /// The multilinear "eq" indicator relative to a boolean point `b`:
    /// the unique MLE that is 1 at `b` and 0 at every other hypercube vertex.
    /// (Identical to `eq_extension` restricted to boolean points.)
    pub fn lagrange_basis(num_vars: usize, point: &[Goldilocks]) -> Self {
        // eq(X, P) = prod_i ( X_i*P_i + (1-X_i)(1-P_i) )
        let mut evals = vec![Goldilocks::ONE; 1 << num_vars];
        for (var_idx, p) in point.iter().enumerate() {
            let bit_shift = num_vars - 1 - var_idx;
            for idx in 0..(1 << num_vars) {
                let bit = (idx >> bit_shift) & 1;
                let term = if bit == 1 { *p } else { Goldilocks::ONE.sub(p) };
                evals[idx] = evals[idx].mul(&term);
            }
        }
        DenseMle { num_vars, evaluations: evals }
    }

    /// Evaluate at a full point.
    pub fn evaluate(&self, point: &[Goldilocks]) -> Result<Goldilocks, MleError> {
        if point.len() != self.num_vars {
            return Err(MleError::PointLengthMismatch {
                expected: self.num_vars,
                got: point.len(),
            });
        }
        // Iterated fix_variables on a copy; O(2^m) total.
        let mut cur = self.evaluations.clone();
        let mut cur_len = cur.len();
        for p in point {
            let half = cur_len / 2;
            for i in 0..half {
                let a = cur[i];
                let b = cur[i + half];
                // a*(1-p) + b*p
                cur[i] = a.add(&b.sub(&a).mul(p));
            }
            cur_len = half;
        }
        Ok(cur[0])
    }

    /// Bind the first (lowest-index) `point.len()` variables in place,
    /// returning the shrunken MLE (the standard sumcheck binding step).
    pub fn fix_variables(&self, partial_point: &[Goldilocks]) -> Result<DenseMle, MleError> {
        if partial_point.len() > self.num_vars {
            return Err(MleError::PointLengthMismatch {
                expected: self.num_vars,
                got: partial_point.len(),
            });
        }
        let mut cur = self.evaluations.clone();
        let mut cur_len = cur.len();
        for p in partial_point {
            let half = cur_len / 2;
            for i in 0..half {
                let a = cur[i];
                let b = cur[i + half];
                cur[i] = a.add(&b.sub(&a).mul(p));
            }
            cur_len = half;
        }
        cur.truncate(cur_len);
        Ok(DenseMle {
            num_vars: self.num_vars - partial_point.len(),
            evaluations: cur,
        })
    }

    /// Sum of all evaluations over the hypercube.
    pub fn sum_over_hypercube(&self) -> Goldilocks {
        let mut acc = Goldilocks::ZERO;
        for e in &self.evaluations {
            acc = acc.add(e);
        }
        acc
    }

    /// Pointwise addition (same shape required).
    pub fn add(&self, other: &DenseMle) -> Result<DenseMle, MleError> {
        if self.num_vars != other.num_vars {
            return Err(MleError::PointLengthMismatch {
                expected: self.num_vars,
                got: other.num_vars,
            });
        }
        let evals = self
            .evaluations
            .iter()
            .zip(other.evaluations.iter())
            .map(|(a, b)| a.add(b))
            .collect();
        Ok(DenseMle { num_vars: self.num_vars, evaluations: evals })
    }

    /// Pointwise subtraction.
    pub fn sub(&self, other: &DenseMle) -> Result<DenseMle, MleError> {
        if self.num_vars != other.num_vars {
            return Err(MleError::PointLengthMismatch {
                expected: self.num_vars,
                got: other.num_vars,
            });
        }
        let evals = self
            .evaluations
            .iter()
            .zip(other.evaluations.iter())
            .map(|(a, b)| a.sub(b))
            .collect();
        Ok(DenseMle { num_vars: self.num_vars, evaluations: evals })
    }

    /// Scalar multiply.
    pub fn scale(&self, c: &Goldilocks) -> DenseMle {
        DenseMle {
            num_vars: self.num_vars,
            evaluations: self.evaluations.iter().map(|e| e.mul(c)).collect(),
        }
    }

    /// Outer-product tensor extension of two MLEs in disjoint variables:
    /// result has num_vars1 + num_vars2 variables, evaluation at (x, y)
    /// equals f(x) * g(y).
    pub fn tensor(&self, other: &DenseMle) -> DenseMle {
        let n = self.evaluations.len() * other.evaluations.len();
        let mut evals = Vec::with_capacity(n);
        for &a in &self.evaluations {
            for &b in &other.evaluations {
                evals.push(a.mul(&b));
            }
        }
        DenseMle {
            num_vars: self.num_vars + other.num_vars,
            evaluations: evals,
        }
    }

    /// The "eq" (equality) MLE: evaluations of
    /// `prod_i [ (1-x_i)(1-b_i) + x_i b_i ]` at every hypercube vertex `b`,
    /// given challenge point `x` (variable 0 = most significant index bit,
    /// matching `fix_variables`).
    pub fn eq_extension(point: &[Goldilocks]) -> DenseMle {
        let m = point.len();
        let mut evals = Vec::with_capacity(1 << m);
        // Start with the constant-1 array in 0 vars; iteratively double.
        // Processing variables in REVERSE order places variable 0 at the
        // most significant bit, matching the fix_variables convention.
        evals.push(Goldilocks::ONE);
        for p in point.iter().rev() {
            let one_minus_p = Goldilocks::ONE.sub(p);
            let mut next = Vec::with_capacity(evals.len() * 2);
            for e in &evals {
                next.push(e.mul(&one_minus_p));
            }
            for e in &evals {
                next.push(e.mul(p));
            }
            evals = next;
        }
        DenseMle { num_vars: m, evaluations: evals }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn evaluate_matches_definition() {
        // Evaluations [f(0,0), f(0,1), f(1,0), f(1,1)] = [1, 3, 2, 4]
        // (variable 0 = most significant index bit).
        // Multilinear interpolation: f(x0, x1) = 1 + x0 + 2*x1.
        // At (x0=2, x1=5): 1 + 2 + 10 = 13.
        let f = DenseMle::new(vec![fe(1), fe(3), fe(2), fe(4)]).ok().unwrap();
        let val = f.evaluate(&[fe(2), fe(5)]).ok().unwrap();
        assert_eq!(val, fe(13));
    }

    #[test]
    fn fix_variables_consistent_with_evaluate() {
        let f = DenseMle::random(6, b"seed-1");
        let point: Vec<Goldilocks> = (1..=6).map(|i| fe(i * 17 % 101)).collect();
        let direct = f.evaluate(&point).ok().unwrap();
        let g = f.fix_variables(&point[..3]).ok().unwrap();
        let staged = g.evaluate(&point[3..]).ok().unwrap();
        assert_eq!(direct, staged);
    }

    #[test]
    fn eq_extension_indicator() {
        // For a boolean point, eq is the point indicator on the hypercube.
        let point = vec![fe(1), fe(0), fe(1)];
        let eq = DenseMle::eq_extension(&point);
        assert_eq!(eq.evaluate(&point).unwrap(), fe(1));
        for idx in 0..8u64 {
            let p: Vec<Goldilocks> = (0..3).map(|i| fe((idx >> (2 - i)) & 1)).collect();
            let expected = if idx == 0b101 { fe(1) } else { fe(0) };
            assert_eq!(eq.evaluate(&p).unwrap(), expected);
        }
    }

    #[test]
    fn lagrange_basis_matches_eq() {
        let point: Vec<Goldilocks> = (1..=4).map(|i| fe(i)).collect();
        let lb = DenseMle::lagrange_basis(4, &point);
        let eq = DenseMle::eq_extension(&point);
        assert_eq!(lb, eq);
    }

    #[test]
    fn tensor_and_sum() {
        let f = DenseMle::random(3, b"a");
        let g = DenseMle::random(4, b"b");
        let t = f.tensor(&g);
        assert_eq!(t.num_vars, 7);
        let sf = f.sum_over_hypercube();
        let sg = g.sum_over_hypercube();
        let st = t.sum_over_hypercube();
        assert_eq!(st, sf.mul(&sg));
    }

    #[test]
    fn add_sub_scale() {
        let f = DenseMle::random(3, b"a");
        let g = DenseMle::random(3, b"b");
        let h = f.add(&g).ok().unwrap().sub(&g).ok().unwrap();
        assert_eq!(h, f);
        let s = f.scale(&fe(5));
        for (a, b) in s.evaluations.iter().zip(f.evaluations.iter()) {
            assert_eq!(*a, b.mul(&fe(5)));
        }
    }
}
