//! Univariate extrapolation over the `U`-point domains
//! (ePrint 2026/587 §4 + Appendix D.1; ePrint 2025/1117 Lemma 2.2).
//!
//! The evaluation domain is `U_k = {∞, 0, 1, …, k−1}` — index 0 holds the
//! evaluation at infinity (the leading coefficient), index `i+1` the value
//! at the integer `i`. Extrapolating a degree-≤k polynomial from `U_k` to
//! `U_h` (h > k) uses the **shifted-evaluation recurrence**: the value at
//! the new point `k+c` is the *same small-integer stencil* applied to the
//! k most recent integer-point evaluations, plus a `k!`-weighted
//! contribution from the leading coefficient:
//!
//! ```text
//! p(k+c) = k!·p(∞) + Σ_{j=0..k−1} (−1)^{k−1−j}·C(k,j)·p(c+j)
//! ```
//!
//! (For `k = 4`: the stencil `[−1, 4, −6, 4]` with `∞`-coefficient `24`
//! — exactly Appendix D.1's `V₈V₄⁻¹` rows.) Every multiplication is a
//! small-by-big field multiplication: the stencil entries `C(k,j)` fit a
//! machine word for every practical degree, and `k!` fits `u64` for
//! `k ≤ 20` (computed mod p beyond that) — the delayed-reduction discipline
//! of §3 accumulates the stencil terms in one modular pass.

use lattice_core::Goldilocks;

/// Extrapolation stencil for degree k: `stencil[j] = (−1)^{k−1−j}·C(k,j)`
/// (signed, small) and the `∞`-coefficient `k!` reduced mod p.
#[derive(Clone, Debug)]
pub struct Stencil {
    pub k: usize,
    /// Signed binomial coefficients (small integers).
    pub weights: Vec<i64>,
    /// `k!` as a field element.
    pub inf_coeff: Goldilocks,
}

impl Stencil {
    pub fn new(k: usize) -> Self {
        let mut c: i128 = 1;
        let mut weights = Vec::with_capacity(k);
        for j in 0..k {
            // C(k, j) computed incrementally; sign (−1)^{k−1−j}.
            if j > 0 {
                c = c * (k as i128 - j as i128 + 1) / j as i128;
            }
            let sign: i64 = if (k - 1 - j) % 2 == 0 { 1 } else { -1 };
            weights.push(sign * (c as i64));
        }
        // k! accumulated as field elements (exact while it fits u128;
        // reduced mod p through from_u128 at the end — k <= 33 in every
        // practical setting keeps the product below 2^122).
        let mut fact: u128 = 1;
        for i in 2..=k {
            fact *= i as u128;
        }
        Stencil {
            k,
            weights,
            inf_coeff: Goldilocks::from_u128(fact),
        }
    }
}

/// Extend `evals` (values at `U_k`, ∞ first) in place to `U_h`
/// (`h ≥ k`): appends the values at `k, k+1, …, h−1`.
///
/// Every appended value costs k small-by-big multiplications — the
/// shifted-evaluation recurrence reusing the sliding window of the k most
/// recent integer-point evaluations.
pub fn extrapolate_in_place(evals: &mut Vec<Goldilocks>, k: usize, h: usize) {
    debug_assert_eq!(evals.len(), k + 1, "expected |U_k| = k+1 entries");
    debug_assert!(h >= k);
    if h == k {
        return;
    }
    let stencil = Stencil::new(k);
    for c in 0..(h - k) {
        // p(k+c) = k!·p(∞) + Σ_j stencil[j]·evals[1 + c + j]
        // Signed accumulation: 2^128 ≢ 0 (mod p) for Goldilocks, so u128
        // wrapping arithmetic would silently corrupt negative intermediate
        // sums — accumulate in i128 and reduce the signed value exactly.
        let mut acc: i128 = 0;
        for (j, &w) in stencil.weights.iter().enumerate() {
            acc += w as i128 * evals[1 + c + j].0 as i128;
        }
        // + k!·p(∞): one field multiplication (the k! constant is
        // machine-word small for k <= 20).
        let inf_term = stencil.inf_coeff.mul(&evals[0]);
        let acc_fe = if acc >= 0 {
            Goldilocks::from_u128(acc as u128)
        } else {
            // (-x) mod p computed exactly: |acc| < 2^127, p < 2^64.
            Goldilocks::from_u128(acc.rem_euclid(0xFFFF_FFFF_0000_0001i128) as u128)
        };
        let total = if k == 0 {
            inf_term
        } else {
            acc_fe.add(&inf_term)
        };
        evals.push(total);
    }
    debug_assert_eq!(evals.len(), h + 1);
}

/// Convenience: extrapolate a copy and return it.
pub fn extrapolate(evals: &[Goldilocks], k: usize, h: usize) -> Vec<Goldilocks> {
    let mut out = evals.to_vec();
    extrapolate_in_place(&mut out, k, h);
    out
}

/// Convert a boolean pair `(p(0), p(1))` to the `U_1 = {∞, 0}` form:
/// `(p(∞), p(0)) = (p(1) − p(0), p(0))` — the leading coefficient first.
#[inline]
pub fn bool_pair_to_u1(lo: Goldilocks, hi: Goldilocks) -> [Goldilocks; 2] {
    [hi.sub(&lo), lo]
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_core::Goldilocks;

    fn poly_eval(coeffs_desc: &[u64], x: u64) -> Goldilocks {
        // coeffs_desc: highest degree first.
        let mut acc = Goldilocks::ZERO;
        for &c in coeffs_desc {
            acc = acc
                .mul(&Goldilocks::from_u64(x))
                .add(&Goldilocks::from_u64(c));
        }
        acc
    }

    #[test]
    fn stencil_matches_binomials() {
        // k = 4: [−1, 4, −6, 4]; k! = 24.
        let s = Stencil::new(4);
        assert_eq!(s.weights, vec![-1, 4, -6, 4]);
        assert_eq!(s.inf_coeff, Goldilocks::from_u64(24));
        let s2 = Stencil::new(3);
        assert_eq!(s2.weights, vec![1, -3, 3]);
        assert_eq!(s2.inf_coeff, Goldilocks::from_u64(6));
    }

    #[test]
    fn extrapolation_matches_polynomial() {
        // p(X) = 3X³ + 2X² − 7X + 5 over U_3 then extended to U_9.
        let p = |x: u64| poly_eval(&[3, 2, 0xFFFF_FFF9, 5], x);
        let inf = Goldilocks::from_u64(3); // leading coefficient
        let mut evals = vec![inf, p(0), p(1), p(2)];
        extrapolate_in_place(&mut evals, 3, 9);
        for x in 0..9u64 {
            assert_eq!(evals[(x + 1) as usize], p(x), " mismatch at x={x}");
        }
    }

    #[test]
    fn extrapolation_high_degree() {
        // Degree-8 polynomial through random-ish coefficients.
        let coeffs: Vec<u64> = vec![7, 11, 13, 17, 19, 23, 29, 31, 37];
        let p = |x: u64| poly_eval(&coeffs, x);
        let inf = Goldilocks::from_u64(coeffs[0]);
        let mut evals = vec![inf];
        for x in 0..8u64 {
            evals.push(p(x));
        }
        extrapolate_in_place(&mut evals, 8, 20);
        for x in 0..20u64 {
            assert_eq!(evals[(x + 1) as usize], p(x), "mismatch at x={x}");
        }
    }

    #[test]
    fn extrapolation_degree_one() {
        // p(X) = 5X + 9: from the boolean pair to U_5.
        let [inf, zero] = bool_pair_to_u1(Goldilocks::from_u64(9), Goldilocks::from_u64(14));
        assert_eq!(inf, Goldilocks::from_u64(5));
        assert_eq!(zero, Goldilocks::from_u64(9));
        let mut evals = vec![inf, zero];
        extrapolate_in_place(&mut evals, 1, 5);
        for x in 0..5u64 {
            assert_eq!(evals[(x + 1) as usize], Goldilocks::from_u64(5 * x + 9));
        }
    }
}
