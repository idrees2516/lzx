//! Multilinear polynomials in the **monomial (coefficient) basis** — the
//! {0, ∞} interpolating set of *"The Sum-Check Protocol over the Monomial
//! Basis, and Other Optimizations"* (Dao, Biswas, Eagen, Milson, Papini,
//! Thaler; ePrint 2026/762).
//!
//! ## The projective viewpoint
//!
//! For a function `f : {0,1}^m → F` with truth table `a`, the multilinear
//! interpolant on the **infinity hypercube** `{0, ∞}^m` is
//!
//! ```text
//! p(X_0, ..., X_{m-1}) = Σ_S a_S · Π_{i∈S} X_i
//! ```
//!
//! whose monomial coefficients are *exactly the truth-table values*
//! (Proposition 3.1 of the paper): evaluating `p` at `b ∈ {0, ∞}^m`
//! extracts the coefficient indexed by `S(b) = { i : b_i = ∞ }`.
//! Consequently the monomial basis **is** the Lagrange basis on `{0, ∞}^m`,
//! and — under this crate's index convention (variable 0 = most significant
//! index bit, matching `DenseMle`) — the coefficient array occupies the
//! *same slots* as the Boolean-basis evaluation array.
//!
//! ## Why this representation
//!
//! * **Subtraction-free binding** (Corollary 3.2):
//!   `p(r, x') = p(0, x') + r · p(∞, x')` — one multiplication and one
//!   addition per coefficient, versus the Boolean
//!   `p(0,x') + r·(p(1,x') − p(0,x'))`. This saves `d·(2^m − 1)` field
//!   subtractions over a full sum-check (Proposition 4.1).
//! * **Structured tables get cheaper**: `eq` becomes `Π (1 + X_i·Y_i)`
//!   (one multiply + one add per factor, down from a quadratic with two
//!   subtractions), the `eq` full-domain table splits as `(e, e·r)` with a
//!   *free* left half, and the `LT` table recurrence loses its subtraction.
//! * **PCS alignment** (§4.3 of the paper): monomial-coefficient form is
//!   the native form of WHIR-style commitments — and of this codebase's
//!   compact Ajtai opening (`lattice-zkvm/src/compact.rs` packs coefficient
//!   columns), so the whole pipeline shares one representation with **no
//!   Möbius conversion** anywhere.
//! * **Streaming** (with `lattice-streaming`): coefficients are the raw
//!   witness data, so the truth table *is* the commitment input; a prover
//!   that streams the trace streams the coefficients directly.
//!
//! ## Conventions
//!
//! `n`-variable coefficient arrays of length `2^n` are indexed so that the
//! coefficient of `Π_{i∈S} X_i` sits at `Σ_{i∈S} 2^{n-1-i}` (variable 0 =
//! most significant bit) — the same layout `DenseMle` uses for evaluations,
//! so `MonomialMle::from_truth_table` is an array identity.

// Index-arithmetic loops (MSB-first bit extraction, limb walks,
// prefix/suffix products) read clearer with explicit indices.
#![allow(clippy::needless_range_loop)]
use lattice_core::field_simd;
use lattice_core::{DenseMle, Goldilocks};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjMleError {
    WrongCoefficientCount { expected: usize, got: usize },
    PointLengthMismatch { expected: usize, got: usize },
}

impl core::fmt::Display for ProjMleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProjMleError::WrongCoefficientCount { expected, got } => {
                write!(f, "monomial coefficient count {got} != power-of-two expectation {expected}")
            }
            ProjMleError::PointLengthMismatch { expected, got } => {
                write!(f, "monomial evaluation point length {got} != {expected}")
            }
        }
    }
}

/// A multilinear polynomial in monomial (coefficient) form — the
/// interpolant of a truth table on the infinity hypercube `{0, ∞}^n`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonomialMle {
    pub num_vars: usize,
    /// `2^num_vars` monomial coefficients; index bit `i` (MSB-first)
    /// selects the variable `X_i` into the monomial.
    pub coeffs: Vec<Goldilocks>,
}

impl MonomialMle {
    /// Build from a coefficient vector (length must be a power of two).
    pub fn new(coeffs: Vec<Goldilocks>) -> Result<Self, ProjMleError> {
        let len = coeffs.len();
        if !len.is_power_of_two() {
            return Err(ProjMleError::WrongCoefficientCount { expected: len, got: len });
        }
        Ok(MonomialMle {
            num_vars: len.trailing_zeros() as usize,
            coeffs,
        })
    }

    /// From the truth-table data of `f : {0,1}^n → F`.
    ///
    /// Proposition 3.1: the `{0, ∞}`-interpolant's monomial coefficients
    /// *are* the truth-table values, slot for slot (with this crate's
    /// shared index convention) — a plain array copy.
    pub fn from_truth_table(table: &[Goldilocks]) -> Result<Self, ProjMleError> {
        Self::new(table.to_vec())
    }

    /// From a `DenseMle` of evaluations: the evaluations *are* the truth
    /// table, so the coefficient form is the same array.
    pub fn from_dense(mle: &DenseMle) -> Self {
        MonomialMle {
            num_vars: mle.num_vars,
            coeffs: mle.evaluations.clone(),
        }
    }

    /// The truth table (coefficient array under the shared convention).
    pub fn truth_table(&self) -> Vec<Goldilocks> {
        self.coeffs.clone()
    }

    /// The total sum over the infinity hypercube `Σ_{b∈{0,∞}^n} p(b)`,
    /// which equals `Σ_{x∈{0,1}^n} f(x)` — the same claim value the
    /// Boolean sum-check would prove for the same data.
    pub fn total_sum(&self) -> Goldilocks {
        field_simd::sum_slice(&self.coeffs)
    }

    /// **Projective binding** of the first variable, in place
    /// (Corollary 3.2): `coeffs[i] ← coeffs[i] + r · coeffs[i + half]`.
    ///
    /// The first half holds the monomials independent of `X_0` (the value
    /// at `X_0 = 0` for every setting of the remaining variables), the
    /// second half holds the coefficients of `X_0` (the value at infinity).
    /// One multiply + one add per surviving coefficient; **no subtraction**.
    pub fn bind_first_var_in_place(&mut self, r: Goldilocks) {
        field_simd::bind_projective_first_half_in_place(&mut self.coeffs, r);
        let half = self.coeffs.len() / 2;
        self.coeffs.truncate(half);
        self.num_vars -= 1;
    }

    /// Non-allocating view form of [`Self::bind_first_var_in_place`]:
    /// binds the first variable of `coeffs` in place without truncating.
    pub fn bind_first_slice_in_place(coeffs: &mut [Goldilocks], r: Goldilocks) {
        field_simd::bind_projective_first_half_in_place(coeffs, r);
    }

    /// Split into `(p(0, x'), p(∞, x'))` — the value-at-zero half and the
    /// coefficient-of-`X_0` half.
    pub fn split_at_first_var(&self) -> (&[Goldilocks], &[Goldilocks]) {
        let half = self.coeffs.len() / 2;
        self.coeffs.split_at(half)
    }

    /// Evaluate at an affine point `(r_0, ..., r_{n-1})` by projective
    /// binding of every variable (the natural `O(n · 2^n)` Horner-style
    /// evaluation for coefficient form; each binding is subtraction-free).
    pub fn evaluate(&self, point: &[Goldilocks]) -> Result<Goldilocks, ProjMleError> {
        if point.len() != self.num_vars {
            return Err(ProjMleError::PointLengthMismatch {
                expected: self.num_vars,
                got: point.len(),
            });
        }
        let mut cur = self.coeffs.clone();
        let mut cur_len = cur.len();
        for r in point {
            let half = cur_len / 2;
            field_simd::bind_projective_first_half_in_place(&mut cur[..cur_len], *r);
            cur_len = half;
        }
        Ok(cur[0])
    }

    /// Evaluate the coefficient-form polynomial at an *infinity-pattern*
    /// point `b ∈ {0, ∞}^n` (given as bits, MSB-first): coefficient
    /// extraction (Proposition 3.1) — `p(b) = a_{S(b)}`.
    pub fn evaluate_projective_bits(&self, bits: &[bool]) -> Result<Goldilocks, ProjMleError> {
        if bits.len() != self.num_vars {
            return Err(ProjMleError::PointLengthMismatch {
                expected: self.num_vars,
                got: bits.len(),
            });
        }
        let mut idx = 0usize;
        for (i, &b) in bits.iter().enumerate() {
            if b {
                idx |= 1 << (self.num_vars - 1 - i);
            }
        }
        Ok(self.coeffs[idx])
    }

    /// The **Möbius transform** (multidimensional inclusion–exclusion,
    /// §4.3 of the paper): converts monomial coefficients of the `{0, ∞}`
    /// interpolant `p` into monomial coefficients of the Boolean multilinear
    /// extension `f̂` (the same polynomial written over the Boolean cube).
    ///
    /// Univariate identity: `f̂(X) = a_0·(1−X) + a_1·X = a_0 + (a_1 − a_0)·X`
    /// — per coordinate the butterfly `(a_0, a_1) → (a_0, a_1 − a_0)`,
    /// tensored over all variables. This is the basis mismatch WHIR-style
    /// commitments otherwise pay `O(n·2^n)` to bridge; the projective
    /// pipeline avoids it entirely, so this function exists for
    /// interoperation and cross-validation.
    pub fn mobius_to_boolean_coeffs(&self) -> Vec<Goldilocks> {
        let mut c = self.coeffs.clone();
        let mut len = c.len();
        while len > 1 {
            let half = len / 2;
            // Per-block butterfly over the current variable (the block's
            // MSB split): the bit-1 half absorbs the bit-0 half.
            for base in (0..c.len()).step_by(len) {
                for i in 0..half {
                    c[base + half + i] = c[base + half + i].sub(&c[base + i]);
                }
            }
            len = half;
        }
        c
    }

    /// Inverse Möbius transform: Boolean-basis monomial coefficients of
    /// `f̂` back to `{0, ∞}`-interpolant coefficients (the truth table):
    /// the per-coordinate butterfly `(c_0, c_1) → (c_0, c_0 + c_1)`.
    pub fn mobius_from_boolean_coeffs(num_vars: usize, boolean: &[Goldilocks]) -> Vec<Goldilocks> {
        let mut c = boolean.to_vec();
        // Undo the forward transform in reverse order: block sizes grow
        // from 2 (the last forward pass) up to 2^n (the first).
        let mut len = 2usize;
        for _ in 0..num_vars {
            let half = len / 2;
            for base in (0..c.len()).step_by(len) {
                for i in 0..half {
                    c[base + half + i] = c[base + half + i].add(&c[base + i]);
                }
            }
            len *= 2;
        }
        c
    }

    /// Random polynomial from a seed (tests and reference checks only).
    pub fn random(num_vars: usize, seed: &[u8]) -> Self {
        DenseMle::random(num_vars, seed) // evaluations == truth table
            .pipe_monomial()
    }

    /// `eq` on the infinity hypercube, as a coefficient-form polynomial in
    /// `Y`: `eq_b(r, Y) = Π_i (1 + r_i · Y_i)` — each factor drops from the
    /// Boolean quadratic `1 − X_i − Y_i + 2X_iY_i` (two subtractions) to a
    /// single multiplication and one addition (§4.2 of the paper).
    pub fn eq_projective(r: &[Goldilocks]) -> Self {
        let n = r.len();
        let mut coeffs = vec![Goldilocks::ZERO; 1usize << n];
        coeffs[0] = Goldilocks::ONE;
        let mut cur = 1usize;
        // Process variables in reverse so variable 0 lands on the MSB,
        // matching the shared index convention (same as eq_table).
        for ri in r.iter().rev() {
            // The paper's "(e, e·r)" recurrence: the left half (bit = 0,
            // monomials without the variable) *stays in place* — a free
            // copy — and only the right half (bit = 1) is written, as the
            // previous table scaled by r_i. The Boolean recurrence pays a
            // multiplication AND a subtraction per entry; this pays half
            // a multiplication per entry (measured 1.94× on BN254, §6.4
            // Table 4). `mul_scalar_slice` writes the first `cur` entries
            // of `second`, exactly the new right half.
            let (first, second) = coeffs.split_at_mut(cur);
            field_simd::mul_scalar_slice(first, *ri, second);
            cur *= 2;
        }
        MonomialMle { num_vars: n, coeffs }
    }

    /// `eq_b(r, Y)` evaluated at a single affine point `y`:
    /// `Π_i (1 + r_i · y_i)` — `n` multiplications and `n` additions.
    pub fn eq_projective_eval(r: &[Goldilocks], y: &[Goldilocks]) -> Goldilocks {
        let mut acc = Goldilocks::ONE;
        for (ri, yi) in r.iter().zip(y.iter()) {
            acc = acc.mul(&Goldilocks::ONE.add(&ri.mul(yi)));
        }
        acc
    }
}

impl From<&DenseMle> for MonomialMle {
    fn from(mle: &DenseMle) -> Self {
        MonomialMle::from_dense(mle)
    }
}

/// Private helper keeping `random` readable.
trait PipeMonomial {
    fn pipe_monomial(self) -> MonomialMle;
}

impl PipeMonomial for DenseMle {
    fn pipe_monomial(self) -> MonomialMle {
        MonomialMle {
            num_vars: self.num_vars,
            coeffs: self.evaluations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// Proposition 3.1 end to end: coefficient extraction at {0,∞} points
    /// reproduces the truth table exactly.
    #[test]
    fn coefficient_extraction_is_truth_table() {
        let data = [g(3), g(1), g(4), g(1), g(5), g(9), g(2), g(6)];
        let p = MonomialMle::from_truth_table(&data).unwrap();
        for idx in 0..8usize {
            let bits: Vec<bool> = (0..3).map(|i| (idx >> (2 - i)) & 1 == 1).collect();
            let val = p.evaluate_projective_bits(&bits).unwrap();
            assert_eq!(val, data[idx]);
        }
    }

    /// Binding identity (Corollary 3.2): after binding `X_0 ← r`, the
    /// shrunken coefficient array equals `p(0, x') + r·p(∞, x')` evaluated
    /// by the independent affine evaluation of the split halves.
    #[test]
    fn projective_binding_identity() {
        let data: Vec<Goldilocks> = (0..16u64).map(g).collect();
        let p = MonomialMle::from_truth_table(&data).unwrap();
        let r = Goldilocks::from_u64(0x1234_5678_9abc_def0);
        let mut bound = p.clone();
        bound.bind_first_var_in_place(r);
        let (lo, hi) = p.split_at_first_var();
        for m in 0..8usize {
            let expected = lo[m].add(&r.mul(&hi[m]));
            assert_eq!(bound.coeffs[m], expected);
        }
    }

    /// Affine evaluation of the coefficient form equals the Boolean MLE
    /// evaluation at the Möbius-mapped point times `Π (1 + r_i)`:
    /// `p(r) · Π_i (1 + r_i) = f̂(φ(r))` with `φ_i(r) = r_i / (1 + r_i)`.
    #[test]
    fn affine_evaluation_matches_boolean_mle() {
        let data: Vec<Goldilocks> = (0..32u64).map(|i| g(i * i + 7)).collect();
        let dense = DenseMle { num_vars: 5, evaluations: data.clone() };
        let p = MonomialMle::from_truth_table(&data).unwrap();
        let r: Vec<Goldilocks> = (1..=5u64)
            .map(|i| Goldilocks::from_u64(1_000_000_007 * i))
            .collect();
        let pr = p.evaluate(&r).unwrap();
        // φ_i = r_i / (1 + r_i);  f̂(φ) · Π(1 + r_i) = p(r).
        let mut phi = Vec::with_capacity(5);
        let mut denom = Goldilocks::ONE;
        for ri in &r {
            let one_plus = Goldilocks::ONE.add(ri);
            phi.push(ri.mul(&one_plus.inverse().unwrap()));
            denom = denom.mul(&one_plus);
        }
        let fhat_phi = dense.evaluate(&phi).unwrap();
        // p(r) = f̂(φ(r)) · Π (1 + r_i)  — the Möbius bridge.
        assert_eq!(pr, fhat_phi.mul(&denom));
    }

    /// Möbius roundtrip: monomial-of-{0,∞} → Boolean monomial → back.
    #[test]
    fn mobius_roundtrip() {
        let data: Vec<Goldilocks> = (0..16u64).map(|i| g(3 * i + 1)).collect();
        let p = MonomialMle::from_truth_table(&data).unwrap();
        let boolean = p.mobius_to_boolean_coeffs();
        let back = MonomialMle::mobius_from_boolean_coeffs(4, &boolean);
        assert_eq!(back, data);
        // The Boolean coefficients are those of f̂: evaluate f̂ via
        // DenseMle and via the boolean monomial form and compare.
        let dense = DenseMle { num_vars: 4, evaluations: data };
        let r: Vec<Goldilocks> =
            (1..=4u64).map(|i| Goldilocks::from_u64(997 * i + 13)).collect();
        let via_mle = dense.evaluate(&r).unwrap();
        // Horner over Boolean monomial coefficients (subtraction allowed).
        let mut acc = Goldilocks::ZERO;
        for (idx, &c) in boolean.iter().enumerate() {
            let mut term = c;
            for v in 0..4usize {
                if (idx >> (3 - v)) & 1 == 1 {
                    term = term.mul(&r[v]);
                }
            }
            acc = acc.add(&term);
        }
        assert_eq!(via_mle, acc);
    }

    /// The projective `eq` table: coefficients satisfy `a_S = Π_{i∈S} r_i`
    /// and agree with direct evaluation `Π (1 + r_i y_i)` on the cube.
    #[test]
    fn eq_projective_table() {
        let r: Vec<Goldilocks> = (1..=4u64).map(|i| Goldilocks::from_u64(31 * i)).collect();
        let t = MonomialMle::eq_projective(&r);
        assert_eq!(t.num_vars, 4);
        for idx in 0..16usize {
            let bits: Vec<bool> = (0..4).map(|i| (idx >> (3 - i)) & 1 == 1).collect();
            let got = t.evaluate_projective_bits(&bits).unwrap();
            // Coefficient extraction: a_S = Π_{i∈S} r_i.
            let mut want = Goldilocks::ONE;
            for (i, &b) in bits.iter().enumerate() {
                if b {
                    want = want.mul(&r[i]);
                }
            }
            assert_eq!(got, want, "eq coefficient mismatch at {idx}");
        }
        // Affine cross-check: Σ over cube of eq(r, y)·f(y) = f̂(r)·... and
        // the simple identity Σ_{y∈{0,∞}} eq_b(r,y) = Π (1 + r_i) + rest.
        let y: Vec<Goldilocks> = (1..=4u64).map(|i| Goldilocks::from_u64(7 * i)).collect();
        assert_eq!(
            t.evaluate(&y).unwrap(),
            MonomialMle::eq_projective_eval(&r, &y)
        );
    }
}
