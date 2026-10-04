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

impl core::fmt::Display for MleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MleError::WrongEvaluationCount { expected, got } => {
                write!(
                    f,
                    "MLE evaluation count {got} != power-of-two expectation {expected}"
                )
            }
            MleError::PointLengthMismatch { expected, got } => {
                write!(f, "MLE point length {got} != {expected}")
            }
        }
    }
}

impl DenseMle {
    /// Build from evaluations; length must be a power of two.
    pub fn new(evaluations: Vec<Goldilocks>) -> Result<Self, MleError> {
        let len = evaluations.len();
        if !len.is_power_of_two() {
            return Err(MleError::WrongEvaluationCount {
                expected: len,
                got: len,
            });
        }
        Ok(DenseMle {
            num_vars: len.trailing_zeros() as usize,
            evaluations,
        })
    }

    /// Constant polynomial.
    pub fn constant(c: Goldilocks) -> Self {
        DenseMle {
            num_vars: 0,
            evaluations: vec![c],
        }
    }

    pub fn zero(num_vars: usize) -> Self {
        DenseMle {
            num_vars,
            evaluations: vec![Goldilocks::ZERO; 1 << num_vars],
        }
    }

    pub fn one(num_vars: usize) -> Self {
        DenseMle {
            num_vars,
            evaluations: vec![Goldilocks::ONE; 1 << num_vars],
        }
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
        DenseMle {
            num_vars,
            evaluations: evals,
        }
    }

    /// The multilinear "eq" indicator relative to a boolean point `b`:
    /// the unique MLE that is 1 at `b` and 0 at every other hypercube vertex.
    /// (Identical to `eq_extension` restricted to boolean points.)
    pub fn lagrange_basis(num_vars: usize, point: &[Goldilocks]) -> Self {
        if point.len() == num_vars {
            // SIMD: same canonical table as eq_extension (every entry is a
            // canonical product of the same per-variable factors), built by
            // the packed eq-table kernel.
            return DenseMle {
                num_vars,
                evaluations: crate::field_simd::eq_table(point),
            };
        }
        // Mismatched shapes keep the exact scalar semantics.
        let mut evals = vec![Goldilocks::ONE; 1 << num_vars];
        for (var_idx, p) in point.iter().enumerate() {
            let bit_shift = num_vars - 1 - var_idx;
            for (idx, e) in evals.iter_mut().enumerate() {
                let bit = (idx >> bit_shift) & 1;
                let term = if bit == 1 { *p } else { Goldilocks::ONE.sub(p) };
                *e = e.mul(&term);
            }
        }
        DenseMle {
            num_vars,
            evaluations: evals,
        }
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
            // SIMD: vectorized first-half binding, 8 field elements per chunk.
            crate::field_simd::bind_first_half_in_place(&mut cur[..cur_len], *p);
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
            // SIMD: vectorized first-half binding, 8 field elements per chunk.
            crate::field_simd::bind_first_half_in_place(&mut cur[..cur_len], *p);
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
        // SIMD: 8-lane lazy accumulation with exact carry accounting.
        crate::field_simd::sum_slice(&self.evaluations)
    }

    /// Pointwise addition (same shape required).
    pub fn add(&self, other: &DenseMle) -> Result<DenseMle, MleError> {
        if self.num_vars != other.num_vars {
            return Err(MleError::PointLengthMismatch {
                expected: self.num_vars,
                got: other.num_vars,
            });
        }
        let evals = {
            let mut out = vec![Goldilocks::ZERO; self.evaluations.len()];
            // SIMD: packed pointwise add.
            crate::field_simd::add_slices(&self.evaluations, &other.evaluations, &mut out);
            out
        };
        Ok(DenseMle {
            num_vars: self.num_vars,
            evaluations: evals,
        })
    }

    /// Pointwise subtraction.
    pub fn sub(&self, other: &DenseMle) -> Result<DenseMle, MleError> {
        if self.num_vars != other.num_vars {
            return Err(MleError::PointLengthMismatch {
                expected: self.num_vars,
                got: other.num_vars,
            });
        }
        let evals = {
            let mut out = vec![Goldilocks::ZERO; self.evaluations.len()];
            // SIMD: packed pointwise sub.
            crate::field_simd::sub_slices(&self.evaluations, &other.evaluations, &mut out);
            out
        };
        Ok(DenseMle {
            num_vars: self.num_vars,
            evaluations: evals,
        })
    }

    /// Scalar multiply.
    pub fn scale(&self, c: &Goldilocks) -> DenseMle {
        let mut evals = vec![Goldilocks::ZERO; self.evaluations.len()];
        // SIMD: broadcast-scalar packed multiply.
        crate::field_simd::mul_scalar_slice(&self.evaluations, *c, &mut evals);
        DenseMle {
            num_vars: self.num_vars,
            evaluations: evals,
        }
    }

    /// Outer-product tensor extension of two MLEs in disjoint variables:
    /// result has num_vars1 + num_vars2 variables, evaluation at (x, y)
    /// equals f(x) * g(y).
    pub fn tensor(&self, other: &DenseMle) -> DenseMle {
        let n = self.evaluations.len() * other.evaluations.len();
        let mut evals = Vec::with_capacity(n);
        // SIMD: each output row is a broadcast-scalar multiply of `other`.
        for &a in &self.evaluations {
            let start = evals.len();
            evals.resize(start + other.evaluations.len(), Goldilocks::ZERO);
            crate::field_simd::mul_scalar_slice(&other.evaluations, a, &mut evals[start..]);
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
        // SIMD: packed eq-table builder (doubling construction with two
        // broadcast-scalar multiplies per variable; see field_simd::eq_table).
        DenseMle {
            num_vars: point.len(),
            evaluations: crate::field_simd::eq_table(point),
        }
    }

    /// Bind the LEAST-significant variables (the suffix of a full point) to
    /// `tail`, returning the shrunken MLE over the remaining head variables.
    ///
    /// This is the transpose-side companion of [`Self::fix_variables`]:
    /// `f.fix_last_variables(&point[a..])` followed by evaluating the head
    /// at `point[..a]` equals `f.evaluate(point)`. Twist & Shout's
    /// cycle-indexed factors (e.g. the d = 1 core Shout PIOP, Fig 5 of
    /// ePrint 2025/105) need the cycle variables bound while the address
    /// variables stay free, which the head-first `fix_variables` cannot do.
    pub fn fix_last_variables(&self, tail: &[Goldilocks]) -> Result<DenseMle, MleError> {
        if tail.len() > self.num_vars {
            return Err(MleError::PointLengthMismatch {
                expected: self.num_vars,
                got: tail.len(),
            });
        }
        let mut cur = self.evaluations.clone();
        // Bind from the least-significant variable up: each binding folds
        // ADJACENT index pairs (variable `num_vars-1` is the index LSB).
        // SIMD: vectorized adjacent-pair binding with lane deinterleave.
        for p in tail.iter().rev() {
            crate::field_simd::bind_pairs_in_place(&mut cur, *p);
            cur.truncate(cur.len() / 2);
        }
        Ok(DenseMle {
            num_vars: self.num_vars - tail.len(),
            evaluations: cur,
        })
    }

    /// Evaluate the multilinear `eq` indicator at arbitrary field points:
    /// `prod_i [ a_i·b_i + (1-a_i)(1-b_i) ]` in O(len) field operations.
    ///
    /// This is the verifier-side companion of [`Self::eq_extension`] (the
    /// dense materialization); sumcheck verifiers need `eq(r, rho)` at the
    /// terminal randomness without building the full array.
    pub fn eq_eval(a: &[Goldilocks], b: &[Goldilocks]) -> Result<Goldilocks, MleError> {
        if a.len() != b.len() {
            return Err(MleError::PointLengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        let mut acc = Goldilocks::ONE;
        for (x, y) in a.iter().zip(b.iter()) {
            let same = x
                .mul(y)
                .add(&Goldilocks::ONE.sub(x).mul(&Goldilocks::ONE.sub(y)));
            acc = acc.mul(&same);
        }
        Ok(acc)
    }

    /// The **less-than multilinear extension** gadget (Twist & Shout
    /// ePrint 2025/105 §3.2; formula from [STW24, App. G]): `LT(x, y)` is
    /// the unique multilinear polynomial that is 1 exactly when
    /// `int(x) < int(y)` over boolean inputs (variable 0 = most significant
    /// bit, matching this crate's index convention):
    ///
    /// `LT(x, y) = sum_i ( prod_{j<i} eq(x_j, y_j) ) · (1 - x_i) · y_i`
    ///
    /// Evaluates in O(log T) field operations — the verifier-side gadget
    /// behind Twist's Val-evaluation sumcheck (Fig 9, Eq 11). Requires
    /// `x.len() == y.len()`.
    pub fn lt_extension(x: &[Goldilocks], y: &[Goldilocks]) -> Result<Goldilocks, MleError> {
        if x.len() != y.len() {
            return Err(MleError::PointLengthMismatch {
                expected: x.len(),
                got: y.len(),
            });
        }
        let mut acc = Goldilocks::ZERO;
        let mut prefix = Goldilocks::ONE;
        for (a, b) in x.iter().zip(y.iter()) {
            // The "x_i = 0, y_i = 1" indicator at the first differing bit.
            let term = Goldilocks::ONE.sub(a).mul(b);
            acc = acc.add(&prefix.mul(&term));
            prefix = prefix.mul(&Self::eq_eval(
                core::slice::from_ref(a),
                core::slice::from_ref(b),
            )?);
        }
        Ok(acc)
    }

    /// Dense `2t`-variate materialization of the less-than function over
    /// `(j', j)` (variable block `j'` first, block `j` last). O(4^t) — the
    /// kernel-scale PROVER-side table for Twist's read/write-checking
    /// sumchecks; production provers evaluate LT analytically (§8.2 of the
    /// paper never materializes it) while verifiers use [`Self::lt_extension`].
    pub fn lt_mle(t: usize) -> DenseMle {
        let n = 1usize << t;
        let mut evals = Vec::with_capacity(n * n);
        for xi in 0..n {
            for yi in 0..n {
                evals.push(if xi < yi {
                    Goldilocks::ONE
                } else {
                    Goldilocks::ZERO
                });
            }
        }
        DenseMle {
            num_vars: 2 * t,
            evaluations: evals,
        }
    }

    /// The affine extension of `k -> int(k)` (the digit-weight polynomial
    /// `w` of the raf-evaluation sumcheck, Figs 6/8 of ePrint 2025/105):
    /// `sum_i 2^(m-1-i) · point[i]`. Valid for `m <= 63` (field-reduced
    /// weights; the register-file setting uses m = 5).
    pub fn int_extension(point: &[Goldilocks]) -> Result<Goldilocks, MleError> {
        if point.len() >= 64 {
            return Err(MleError::PointLengthMismatch {
                expected: 63,
                got: point.len(),
            });
        }
        let mut acc = Goldilocks::ZERO;
        for (i, p) in point.iter().enumerate() {
            let shift = point.len() - 1 - i;
            acc = acc.add(&p.mul(&Goldilocks::from_u64(1u64 << shift)));
        }
        Ok(acc)
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
        let f = DenseMle::new(vec![fe(1), fe(3), fe(2), fe(4)])
            .ok()
            .unwrap();
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
        // Independent per-entry scalar reference (variable 0 = most
        // significant index bit) — NOT a wrapper of the packed eq kernel,
        // so this cross-check is non-circular for both `lagrange_basis` and
        // `eq_extension` (which both route through field_simd::eq_table).
        let point: Vec<Goldilocks> = (1..=4).map(fe).collect();
        let m = point.len();
        let mut expected = Vec::with_capacity(1 << m);
        for idx in 0..(1usize << m) {
            let mut v = Goldilocks::ONE;
            for (i, p) in point.iter().enumerate() {
                let bit = (idx >> (m - 1 - i)) & 1;
                let term = if bit == 1 { *p } else { Goldilocks::ONE.sub(p) };
                v = v.mul(&term);
            }
            expected.push(v);
        }
        let lb = DenseMle::lagrange_basis(4, &point);
        let eq = DenseMle::eq_extension(&point);
        assert_eq!(lb.evaluations, expected);
        assert_eq!(eq.evaluations, expected);
        assert_eq!(lb, eq);
        // The scalar (mismatched-shape) branch must agree per-factor too.
        let partial = DenseMle::lagrange_basis(4, &point[..2]);
        let mut expected_partial = Vec::with_capacity(1 << 4);
        for idx in 0..(1usize << 4) {
            let mut v = Goldilocks::ONE;
            for (i, p) in point[..2].iter().enumerate() {
                let bit = (idx >> (4 - 1 - i)) & 1;
                let term = if bit == 1 { *p } else { Goldilocks::ONE.sub(p) };
                v = v.mul(&term);
            }
            expected_partial.push(v);
        }
        assert_eq!(partial.evaluations, expected_partial);
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

    #[test]
    fn fix_last_variables_matches_direct_evaluate() {
        for (a, b) in [(3usize, 2usize), (5, 1), (0, 4), (4, 0), (2, 5)] {
            let f = DenseMle::random(a + b, b"fix-last");
            let tail: Vec<Goldilocks> = (1..=b).map(|i| fe((i * 29 % 101) as u64)).collect();
            let head: Vec<Goldilocks> = (1..=a).map(|i| fe((i * 37 % 97) as u64)).collect();
            let bound = f.fix_last_variables(&tail).ok().unwrap();
            assert_eq!(bound.num_vars, a);
            let staged = bound.evaluate(&head).ok().unwrap();
            let mut full = head.clone();
            full.extend(tail.iter().copied());
            let direct = f.evaluate(&full).ok().unwrap();
            assert_eq!(staged, direct, "a={a} b={b}");
        }
    }

    #[test]
    fn fix_last_variables_rejects_oversized_tail() {
        let f = DenseMle::random(3, b"fix-last-err");
        assert!(f.fix_last_variables(&[fe(1), fe(2), fe(3), fe(4)]).is_err());
        // Zero-length tail is the identity.
        assert_eq!(f.fix_last_variables(&[]).ok().unwrap(), f);
    }

    #[test]
    fn eq_eval_matches_eq_extension() {
        for m in [1usize, 2, 5] {
            let r: Vec<Goldilocks> = (1..=m).map(|i| fe((i * 13 % 101) as u64)).collect();
            let x: Vec<Goldilocks> = (1..=m).map(|i| fe((i * 7 % 103) as u64)).collect();
            let dense = DenseMle::eq_extension(&r).evaluate(&x).ok().unwrap();
            let closed = DenseMle::eq_eval(&r, &x).ok().unwrap();
            assert_eq!(dense, closed, "m={m}");
        }
    }

    #[test]
    fn lt_extension_agrees_with_dense_materialization() {
        for t in [1usize, 2, 3, 4] {
            let dense = DenseMle::lt_mle(t);
            for trial in 0..8 {
                let x: Vec<Goldilocks> = (0..t)
                    .map(|i| fe(((trial * 31 + i * 17) % 101) as u64))
                    .collect();
                let y: Vec<Goldilocks> = (0..t)
                    .map(|i| fe(((trial * 43 + i * 29) % 97) as u64))
                    .collect();
                let mut point = x.clone();
                point.extend(y.iter().copied());
                let dense_val = dense.evaluate(&point).ok().unwrap();
                let closed = DenseMle::lt_extension(&x, &y).ok().unwrap();
                assert_eq!(dense_val, closed, "t={t} trial={trial}");
            }
        }
    }

    #[test]
    fn lt_extension_boolean_semantics() {
        // At boolean points LT(x, y) = [int(x) < int(y)], MSB first.
        let t = 3;
        for xi in 0..8u64 {
            for yi in 0..8u64 {
                let x: Vec<Goldilocks> = (0..t).map(|i| fe((xi >> (t - 1 - i)) & 1)).collect();
                let y: Vec<Goldilocks> = (0..t).map(|i| fe((yi >> (t - 1 - i)) & 1)).collect();
                let v = DenseMle::lt_extension(&x, &y).ok().unwrap();
                let expected = if xi < yi { fe(1) } else { fe(0) };
                assert_eq!(v, expected, "x={xi} y={yi}");
            }
        }
    }

    #[test]
    fn int_extension_matches_hypercube_values() {
        for m in [1usize, 3, 5] {
            let mut evals = Vec::with_capacity(1 << m);
            for k in 0..(1u64 << m) {
                evals.push(fe(k));
            }
            let w = DenseMle::new(evals).ok().unwrap();
            let pt: Vec<Goldilocks> = (1..=m).map(|i| fe((i * 19 % 89) as u64)).collect();
            let dense = w.evaluate(&pt).ok().unwrap();
            let closed = DenseMle::int_extension(&pt).ok().unwrap();
            assert_eq!(dense, closed, "m={m}");
        }
        assert!(DenseMle::int_extension(&[fe(1); 64]).is_err());
    }
}
