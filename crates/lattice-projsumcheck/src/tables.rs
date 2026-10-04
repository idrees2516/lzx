//! Structured polynomials over the infinity hypercube — §4.2 and
//! Appendix A of ePrint 2026/762.
//!
//! Switching the interpolating set from `{0,1}` to `{0,∞}` simplifies the
//! closed forms of the polynomials zkVMs evaluate most:
//!
//! * **Equality**: `eq_b(X, Y) = Π_i (1 + X_i·Y_i)` — each factor drops
//!   from the quadratic `1 − X_i − Y_i + 2X_iY_i` (two subtractions) to a
//!   single multiplication and one addition.
//! * **Less-than**: the decisive factor `(1 − X_j)·Y_j` becomes `Y_j`, and
//!   each ignored lower bit contributes `(1 + X_i)(1 + Y_i)`.
//! * **Per-pair dictionary** (§4.2): every two-operand Boolean gate has a
//!   cheap `{0,∞}` factor — `AND → X·Y`, `XOR → X + Y` (multiplication
//!   free!), `OR → X + Y + X·Y`, equality → `1 + X·Y`, complement is the
//!   *constant 1*, and an ignored pair costs `(1+X)(1+Y)`.
//! * **Full-domain tables** (§6.4): the `eq` table splits as `(e, e·r)`
//!   with a *free* left half (measured 1.94× on BN254), and the `LT`
//!   recurrence drops its subtraction. Both doubling recurrences are
//!   implemented here and brute-force verified.
//!
//! ## Conventions
//!
//! * Variables are MSB-first: `X_0` is the most significant bit of an
//!   operand; interleaved two-operand tables use `(X_0, Y_0, …, X_{w−1},
//!   Y_{w−1})` as in Jolt's `evaluate_mle` routines.
//! * `eval_affine`-style functions evaluate the *affine* (homogenized)
//!   form — the correct notion at arbitrary field points, which is what
//!   sum-check opening points are.
//! * Full-domain tables are returned in **coefficient form** (truth-table
//!   slots), directly usable as `MonomialMle` factors.
//! * Correctness anchor: the monomial coefficients of each closed form —
//!   recovered by Möbius inversion over the Boolean cube in the tests —
//!   equal the discrete table entries (Proposition 3.1).

// Index-arithmetic loops (MSB-first bit extraction, limb walks,
// prefix/suffix products) read clearer with explicit indices.
#![allow(clippy::needless_range_loop)]
use crate::proj_mle::MonomialMle;
use lattice_core::Goldilocks;

/// `E_i = 1 + X_i·Y_i` — the equality pair factor.
#[inline]
pub fn pair_eq(x: Goldilocks, y: Goldilocks) -> Goldilocks {
    Goldilocks::ONE.add(&x.mul(&y))
}

/// `Ω_i = (1 + X_i)(1 + Y_i)` — the ignored-pair factor.
#[inline]
pub fn pair_ignore(x: Goldilocks, y: Goldilocks) -> Goldilocks {
    Goldilocks::ONE.add(&x).mul(&Goldilocks::ONE.add(&y))
}

/// The per-pair translation dictionary (Figure 1 / §4.2): the `{0,∞}`
/// factor for each single-bit role, evaluated affinely at `(x, y)`.
///
/// Every entry agrees with the Boolean dictionary factor on `{0,1}²`
/// (verified in the tests); on `{0,∞}²` they interpolate the same gates
/// by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairFactor {
    /// `x AND y` → `X·Y`.
    And,
    /// `x AND NOT y` → `X`.
    Andn,
    /// `x OR y` → `X + Y + X·Y`.
    Or,
    /// `x XOR y` → `X + Y` — multiplication-free.
    Xor,
    /// `x == y` → `1 + X·Y`.
    Eq,
    /// `NOT (x == y)` → `Ω − E`.
    Neq,
    /// `x < y` (the decisive differing pair) → `Y`.
    Lt,
    /// the left operand bit `x` → `X·(1 + Y)`.
    XTerm,
    /// the right operand bit `y` → `(1 + X)·Y`.
    YTerm,
    /// pair not involved → `(1 + X)(1 + Y)`.
    Ignore,
}

impl PairFactor {
    /// The affine `{0,∞}` factor value at `(x, y)`.
    #[inline]
    pub fn eval(self, x: Goldilocks, y: Goldilocks) -> Goldilocks {
        match self {
            PairFactor::And => x.mul(&y),
            PairFactor::Andn => x,
            PairFactor::Or => x.add(&y).add(&x.mul(&y)),
            PairFactor::Xor => x.add(&y),
            PairFactor::Eq => pair_eq(x, y),
            PairFactor::Neq => pair_ignore(x, y).sub(&pair_eq(x, y)),
            PairFactor::Lt => y,
            PairFactor::XTerm => x.mul(&Goldilocks::ONE.add(&y)),
            PairFactor::YTerm => Goldilocks::ONE.add(&x).mul(&y),
            PairFactor::Ignore => pair_ignore(x, y),
        }
    }

    /// The corresponding Boolean-hypercube factor, for cross-validation.
    #[inline]
    pub fn eval_boolean(self, x: Goldilocks, y: Goldilocks) -> Goldilocks {
        let one = Goldilocks::ONE;
        match self {
            PairFactor::And => x.mul(&y),
            PairFactor::Andn => x.mul(&one.sub(&y)),
            PairFactor::Or => one.sub(&one.sub(&x).mul(&one.sub(&y))),
            PairFactor::Xor => x.mul(&one.sub(&y)).add(&y.mul(&one.sub(&x))),
            PairFactor::Eq => x.mul(&y).add(&one.sub(&x).mul(&one.sub(&y))),
            PairFactor::Neq => one.sub(&x.mul(&y).add(&one.sub(&x).mul(&one.sub(&y)))),
            PairFactor::Lt => one.sub(&x).mul(&y),
            PairFactor::XTerm => x,
            PairFactor::YTerm => y,
            PairFactor::Ignore => one,
        }
    }
}

/// `LT_b(r, Y)` — the `{0,∞}` interpolant of unsigned less-than
/// `1[nat(x) < nat(y)]` in the second operand, `X = r` fixed (MSB-first):
///
/// ```text
/// LT_b(r, Y) = Σ_v Y_v · Π_{u<v} (1 + r_u·Y_u) · Π_{u>v} (1 + r_u)(1 + Y_u)
/// ```
///
/// The decisive factor `(1 − X_v)·Y_v` became `Y_v`; every less-significant
/// (ignored) bit contributes an `Ω` factor. `O(n)` per evaluation.
pub fn lt_projective_eval(r: &[Goldilocks], y: &[Goldilocks]) -> Goldilocks {
    let n = r.len();
    // r_suffix[v]  = Π_{u>v} (1 + r_u);  y_suffix[v] = Π_{u>v} (1 + y_u).
    let mut r_suffix = vec![Goldilocks::ONE; n + 1];
    let mut y_suffix = vec![Goldilocks::ONE; n + 1];
    for v in (0..n).rev() {
        r_suffix[v] = r_suffix[v + 1].mul(&Goldilocks::ONE.add(&r[v]));
        y_suffix[v] = y_suffix[v + 1].mul(&Goldilocks::ONE.add(&y[v]));
    }
    let mut acc = Goldilocks::ZERO;
    let mut prefix = Goldilocks::ONE; // Π_{u<v} (1 + r_u·y_u)
    for v in 0..n {
        let term = y[v]
            .mul(&prefix)
            .mul(&r_suffix[v + 1])
            .mul(&y_suffix[v + 1]);
        acc = acc.add(&term);
        prefix = prefix.mul(&Goldilocks::ONE.add(&r[v].mul(&y[v])));
    }
    acc
}

/// Full-domain **coefficient-form** `LT` table (over the `Y` hypercube,
/// MSB-first) of `LT_b(r, Y)`, built by the verified doubling recurrence —
/// the `{0,∞}` analogue of the paper's §6.4 LT recurrence:
///
/// ```text
/// T_{k+1}[2i]   = (1 + r_k) · T_k[i]
/// T_{k+1}[2i+1] = T_{k+1}[2i] + E_k[i]            (reuse the even entry)
/// E_{k+1}[2i]   = E_k[i]                           (free copy)
/// E_{k+1}[2i+1] = r_k · E_k[i]
/// ```
///
/// with `T_0 = [0]`, `E_0 = [1]` (the projective `eq` table), processing
/// variables MSB-first. One multiplication and one addition per output
/// pair — **no subtraction** (the Boolean recurrence pays one per entry).
/// Derived by splitting the least-significant variable:
/// `LT_n = (1 + r_lsb)(1 + Y_lsb)·LT_{n−1} + Y_lsb·eq_{n−1}`.
pub fn lt_projective_table(r: &[Goldilocks]) -> Vec<Goldilocks> {
    let _n = r.len();
    let mut t = vec![Goldilocks::ZERO; 1];
    let mut e = vec![Goldilocks::ONE; 1];
    for &rk in r.iter() {
        let one_plus = Goldilocks::ONE.add(&rk);
        let cur = t.len();
        let mut nt = vec![Goldilocks::ZERO; cur * 2];
        let mut ne = vec![Goldilocks::ZERO; cur * 2];
        for i in 0..cur {
            let even = one_plus.mul(&t[i]);
            nt[2 * i] = even;
            nt[2 * i + 1] = even.add(&e[i]);
            ne[2 * i] = e[i];
            ne[2 * i + 1] = rk.mul(&e[i]);
        }
        t = nt;
        e = ne;
    }
    t
}

/// Full-domain coefficient-form `eq` table via the `(e, e·r)` recurrence
/// (§6.4): the left half of every doubling step is a *free copy* — the
/// 1.94× construction win measured on BN254 (Table 4).
pub fn eq_projective_table(r: &[Goldilocks]) -> Vec<Goldilocks> {
    MonomialMle::eq_projective(r).coeffs
}

/// `shift_b(r, Y)` — the `{0,∞}` interpolant of the pcnext kernel
/// `1[val(y) + 1 = val(r)]` (no wraparound; `pcnext` never wraps because
/// the program counter is bounded below `2^n − 1`), MSB-first, `O(n²)`:
///
/// ```text
/// shift_b(r, Y) = Σ_v r_v · (Π_{u>v} Y_u) · (Π_{u<v} (1 + r_u·Y_u))
/// ```
///
/// Term `v` fires when `y`'s bits below `v` are all ones (the trailing
/// product `Π_{u>v} Y_u` — the monomial that extracts exactly those bits
/// being at infinity), `r_v = 1` supplies the carry, and the higher bits
/// agree (`E` factors). The Boolean `(1 − r_v)` decisive factor is absent
/// because the monomial already forces `y_v = 0`.
pub fn shift_projective_eval(r: &[Goldilocks], y: &[Goldilocks]) -> Goldilocks {
    let n = r.len();
    // trail[v] = Π_{u>v} y_u — the identity factors of the lower bits.
    let mut trail = vec![Goldilocks::ONE; n + 1];
    for v in (0..n).rev() {
        trail[v] = trail[v + 1].mul(&y[v]);
    }
    let mut acc = Goldilocks::ZERO;
    for v in (0..n).rev() {
        if r[v].is_zero() {
            continue; // r_v = 0 kills the carry — skip the O(v) prefix walk.
        }
        let mut prefix = Goldilocks::ONE;
        for u in 0..v {
            prefix = prefix.mul(&pair_eq(r[u], y[u]));
        }
        acc = acc.add(&r[v].mul(&trail[v + 1]).mul(&prefix));
    }
    acc
}

/// The two-word **bitwise tables** (Appendix A.3) in projective form.
///
/// `T_b_And  = Σ_i 2^{w−1−i}·X_i·Y_i·Π_{j≠i} Ω_j`
/// `T_b_Andn = Σ_i 2^{w−1−i}·X_i·Π_{j≠i} Ω_j`
/// `T_b_Or   = Σ_i 2^{w−1−i}·(X_i + Y_i + X_iY_i)·Π_{j≠i} Ω_j`
/// `T_b_Xor  = Σ_i 2^{w−1−i}·(X_i + Y_i)·Π_{j≠i} Ω_j`
///
/// XOR is the headline simplification: `X_i + Y_i − 2X_iY_i` collapses to
/// the pure addition `X_i + Y_i` — one multiplication saved per pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitwiseOp {
    And,
    Andn,
    Or,
    Xor,
}

impl BitwiseOp {
    /// Affine evaluation of the projective interpolant at the operand
    /// points `(x, y)` (MSB-first bit values).
    pub fn eval_projective(self, x: &[Goldilocks], y: &[Goldilocks]) -> Goldilocks {
        let w = x.len();
        // prefix[i] = Π_{j<i} Ω_j; suffix[i] = Π_{j≥i} Ω_j.
        let mut pre = vec![Goldilocks::ONE; w + 1];
        let mut suf = vec![Goldilocks::ONE; w + 1];
        for i in 0..w {
            pre[i + 1] = pre[i].mul(&pair_ignore(x[i], y[i]));
        }
        for i in (0..w).rev() {
            suf[i] = suf[i + 1].mul(&pair_ignore(x[i], y[i]));
        }
        let mut acc = Goldilocks::ZERO;
        for i in 0..w {
            let weight = Goldilocks::from_u64(1u64 << (w - 1 - i));
            let core = match self {
                BitwiseOp::And => x[i].mul(&y[i]),
                BitwiseOp::Andn => x[i],
                BitwiseOp::Or => x[i].add(&y[i]).add(&x[i].mul(&y[i])),
                BitwiseOp::Xor => x[i].add(&y[i]),
            };
            acc = acc.add(&weight.mul(&core).mul(&pre[i]).mul(&suf[i + 1]));
        }
        acc
    }

    /// The discrete table entry for Boolean inputs `(x, y)` (integers).
    pub fn discrete(self, x: u64, y: u64, w: usize) -> u64 {
        let m = mask(w);
        match self {
            BitwiseOp::And => x & y & m,
            BitwiseOp::Andn => (x & !y) & m,
            BitwiseOp::Or => (x | y) & m,
            BitwiseOp::Xor => (x ^ y) & m,
        }
    }
}

fn mask(w: usize) -> u64 {
    if w >= 64 {
        u64::MAX
    } else {
        (1u64 << w) - 1
    }
}

/// `1[x == y]` over words: `T_b_Equal = Π_i E_i`.
pub fn equal_projective_eval(x: &[Goldilocks], y: &[Goldilocks]) -> Goldilocks {
    let mut acc = Goldilocks::ONE;
    for i in 0..x.len() {
        acc = acc.mul(&pair_eq(x[i], y[i]));
    }
    acc
}

/// `1[x < y]` unsigned over words:
/// `T_b_LT = Σ_v Y_v·Π_{u<v} E_u·Π_{u>v} Ω_u` — the decisive `(1−X_v)Y_v`
/// became `Y_v` (one multiplication saved per pair, §4.2).
pub fn lt_word_projective_eval(x: &[Goldilocks], y: &[Goldilocks]) -> Goldilocks {
    let w = x.len();
    let mut suf = vec![Goldilocks::ONE; w + 1];
    for v in (0..w).rev() {
        suf[v] = suf[v + 1].mul(&pair_ignore(x[v], y[v]));
    }
    let mut acc = Goldilocks::ZERO;
    let mut pre = Goldilocks::ONE;
    for v in 0..w {
        acc = acc.add(&y[v].mul(&pre).mul(&suf[v + 1]));
        pre = pre.mul(&pair_eq(x[v], y[v]));
    }
    acc
}

/// `Movsign`: `(2^w − 1)·X_0·(1 + Y_0)·Π_{j≥1} Ω_j`.
pub fn movsign_projective_eval(x: &[Goldilocks], y: &[Goldilocks]) -> Goldilocks {
    let w = x.len();
    let mut acc = Goldilocks::from_u64(mask(w))
        .mul(&x[0])
        .mul(&Goldilocks::ONE.add(&y[0]));
    for j in 1..w {
        acc = acc.mul(&pair_ignore(x[j], y[j]));
    }
    acc
}

/// `VirtualXORROT(ρ)`: XOR then rotate right by `ρ` within the `w`-bit
/// word (the BLAKE2b rotation family):
/// `Σ_i 2^{w−1−((i+ρ) mod w)}·(X_i + Y_i)·Π_{j≠i} Ω_j`.
pub fn xorrot_projective_eval(x: &[Goldilocks], y: &[Goldilocks], rho: usize) -> Goldilocks {
    let w = x.len();
    let mut pre = vec![Goldilocks::ONE; w + 1];
    let mut suf = vec![Goldilocks::ONE; w + 1];
    for i in 0..w {
        pre[i + 1] = pre[i].mul(&pair_ignore(x[i], y[i]));
    }
    for i in (0..w).rev() {
        suf[i] = suf[i + 1].mul(&pair_ignore(x[i], y[i]));
    }
    let mut acc = Goldilocks::ZERO;
    for i in 0..w {
        let out_bit = (i + rho) % w;
        let weight = Goldilocks::from_u64(1u64 << (w - 1 - out_bit));
        acc = acc.add(&weight.mul(&x[i].add(&y[i])).mul(&pre[i]).mul(&suf[i + 1]));
    }
    acc
}

/// `VirtualXORROTW(ρ)`: the halfword variant acting on the low `h = w/2`
/// pairs (active indices `h..w`, MSB-first), with the ignored high half
/// contributing its `Ω` product.
pub fn xorrotw_projective_eval(x: &[Goldilocks], y: &[Goldilocks], rho: usize) -> Goldilocks {
    let w = x.len();
    let h = w / 2;
    let mut pre = vec![Goldilocks::ONE; w + 1];
    let mut suf = vec![Goldilocks::ONE; w + 1];
    for i in 0..w {
        pre[i + 1] = pre[i].mul(&pair_ignore(x[i], y[i]));
    }
    for i in (0..w).rev() {
        suf[i] = suf[i + 1].mul(&pair_ignore(x[i], y[i]));
    }
    let mut acc = Goldilocks::ZERO;
    for i in h..w {
        let j = i - h;
        let out_bit = (j + rho) % h;
        let weight = Goldilocks::from_u64(1u64 << (h - 1 - out_bit));
        // Π_{j≠i} Ω_j runs over ALL pairs (the ignored high half included),
        // which pre[i]·suf[i+1] already covers — no extra trailing factor.
        acc = acc.add(&weight.mul(&x[i].add(&y[i])).mul(&pre[i]).mul(&suf[i + 1]));
    }
    acc
}

/// `VirtualRev8W`: byte-reversal within each `w/2`-bit half of a `2w`-bit
/// raw index — `Σ_t 2^{π(t)}·Z_t·Π_{u≠t} (1 + Z_u)` over the MSB-first
/// raw index bits, where `π` maps each input bit to its output position.
pub fn rev8w_projective_eval(z: &[Goldilocks]) -> Goldilocks {
    let total = z.len();
    let mut pre = vec![Goldilocks::ONE; total + 1];
    let mut suf = vec![Goldilocks::ONE; total + 1];
    for t in 0..total {
        pre[t + 1] = pre[t].mul(&Goldilocks::ONE.add(&z[t]));
    }
    for t in (0..total).rev() {
        suf[t] = suf[t + 1].mul(&Goldilocks::ONE.add(&z[t]));
    }
    let mut acc = Goldilocks::ZERO;
    for t in 0..total {
        let weight = Goldilocks::from_u64(1u64 << rev8_out_bit(t, total));
        acc = acc.add(&weight.mul(&z[t]).mul(&pre[t]).mul(&suf[t + 1]));
    }
    acc
}

/// Output bit position (LSB-indexed) of raw-index bit `t` (MSB-first)
/// under byte reversal within each half of a `total = 2w`-bit index.
fn rev8_out_bit(t: usize, total: usize) -> usize {
    let w = total / 2;
    let half = if t < w { 0 } else { 1 };
    let within = t % w; // MSB-first position inside the half
    let bytes = w / 8;
    let b = within / 8;
    let p = within % 8;
    // Reversed byte keeps its within-byte bit position (MSB-first).
    let out_within_msb = (bytes - 1 - b) * 8 + p;
    let out_msb = half * w + out_within_msb;
    total - 1 - out_msb
}

/// `MulUNoOverflow`: the upper word of the raw `2w`-bit index is all zero
/// — `Ω_{[w, 2w−1]}(Z)`: the monomials live in the lower-word variables,
/// so the interpolant is the `Ω`-product over the **lower** word's bits
/// (MSB-first raw index: bits `w..2w−1`).
pub fn mul_no_overflow_projective_eval(z: &[Goldilocks]) -> Goldilocks {
    let total = z.len();
    let w = total / 2;
    let mut acc = Goldilocks::ONE;
    for i in w..total {
        acc = acc.mul(&Goldilocks::ONE.add(&z[i]));
    }
    acc
}

/// `Pow2`: `2^{val(shift bits)}` over the `k = log2 w` shift-amount bits
/// (MSB-first): `Π_t (1 + 2^{2^t}·U_t)` with `U_t` the `t`-th bit from
/// the LSB — the per-factor scalar is `2^{2^t}` because the shift amount
/// contributes `2^t` per set bit. The caller composes the `Ω` factors
/// over the remaining raw index bits (the ignored-variable products of
/// the full-table form).
pub fn pow2_projective_eval(shift_bits: &[Goldilocks], _w: usize) -> Goldilocks {
    let k = shift_bits.len();
    let mut acc = Goldilocks::ONE;
    for t in 0..k {
        // U_t = the t-th bit from the LSB = shift_bits[k−1−t] (MSB-first).
        let u = shift_bits[k - 1 - t];
        let scalar = Goldilocks::from_u64(1u64 << (1u64 << t).min(62));
        acc = acc.mul(&Goldilocks::ONE.add(&u.mul(&scalar)));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// Recover the monomial coefficients of a multilinear closed form `F`
    /// over `n` variables by Möbius inversion of its affine values on the
    /// Boolean cube — the rigorous correctness anchor: for the `{0,∞}`
    /// interpolant of a discrete table `T`, `coeff[S] == T(bits S)`.
    fn moebius_coefficients<F>(n: usize, f: F) -> Vec<Goldilocks>
    where
        F: Fn(&[Goldilocks]) -> Goldilocks,
    {
        let mut c = vec![Goldilocks::ZERO; 1usize << n];
        for idx in 0..(1usize << n) {
            let pt: Vec<Goldilocks> = (0..n)
                .map(|i| g(((idx >> (n - 1 - i)) & 1) as u64))
                .collect();
            c[idx] = f(&pt);
        }
        // In-place inclusion–exclusion butterfly, MSB-first variables.
        let mut len = c.len();
        while len > 1 {
            let half = len / 2;
            for block in (0..c.len()).step_by(len) {
                for i in 0..half {
                    c[block + half + i] = c[block + half + i].sub(&c[block + i]);
                }
            }
            len = half;
        }
        c
    }

    /// Split an interleaved combined index (MSB-first layout
    /// X_0, Y_0, X_1, Y_1, …) into the (x, y) integer operands.
    fn split_interleaved(idx: usize, w: usize) -> (u64, u64) {
        let mut x = 0u64;
        let mut y = 0u64;
        for j in 0..w {
            // MSB-first position 2j (X_j) = integer bit 2w−1−2j.
            let xb = (idx >> (2 * w - 1 - 2 * j)) & 1;
            // MSB-first position 2j+1 (Y_j) = integer bit 2w−2−2j.
            let yb = (idx >> (2 * w - 2 - 2 * j)) & 1;
            x |= (xb as u64) << (w - 1 - j);
            y |= (yb as u64) << (w - 1 - j);
        }
        (x, y)
    }

    /// The per-pair dictionary: the monomial coefficients of each affine
    /// factor (recovered by the 2-variable Möbius inversion over its four
    /// Boolean values) equal the discrete gate on `{0,1}^2` — the exact
    /// Proposition 3.1 statement that the `{0,∞}` factors interpolate the
    /// same gates the Boolean factors do.
    #[test]
    fn pair_dictionary_interpolates_gates() {
        type GateFn = fn(bool, bool) -> u64;
        let gates: [(PairFactor, GateFn); 10] = [
            (PairFactor::And, |a, b| u64::from(a && b)),
            (PairFactor::Andn, |a, b| u64::from(a && !b)),
            (PairFactor::Or, |a, b| u64::from(a || b)),
            (PairFactor::Xor, |a, b| u64::from(a != b)),
            (PairFactor::Eq, |a, b| u64::from(a == b)),
            (PairFactor::Neq, |a, b| u64::from(a != b)),
            (PairFactor::Lt, |a, b| u64::from(!a && b)),
            (PairFactor::XTerm, |a, _| u64::from(a)),
            (PairFactor::YTerm, |_, b| u64::from(b)),
            (PairFactor::Ignore, |_, _| 1),
        ];
        for (f, gate) in gates {
            // Affine values at the four Boolean points.
            let v = |b: bool| if b { Goldilocks::ONE } else { Goldilocks::ZERO };
            let f00 = f.eval(v(false), v(false));
            let f01 = f.eval(v(false), v(true));
            let f10 = f.eval(v(true), v(false));
            let f11 = f.eval(v(true), v(true));
            // Möbius inversion over the two coordinates: the monomial
            // coefficients of the affine factor.
            let c00 = f00;
            let c01 = f01.sub(&f00);
            let c10 = f10.sub(&f00);
            let c11 = f11.sub(&f01).sub(&f10).add(&f00);
            // Coefficients == the discrete gate at the matching bits —
            // Proposition 3.1 for the per-pair dictionary.
            assert_eq!(c00, g(gate(false, false)), "{f:?} at (0,0)");
            assert_eq!(c01, g(gate(false, true)), "{f:?} at (0,1)");
            assert_eq!(c10, g(gate(true, false)), "{f:?} at (1,0)");
            assert_eq!(c11, g(gate(true, true)), "{f:?} at (1,1)");
            // The Boolean factors agree with the gates on {0,1}^2
            // (Figure 1's bottom row, kept for cross-reference).
            assert_eq!(
                f.eval_boolean(v(false), v(false)),
                g(gate(false, false)),
                "{f:?} bool (0,0)"
            );
            assert_eq!(
                f.eval_boolean(v(false), v(true)),
                g(gate(false, true)),
                "{f:?} bool (0,1)"
            );
            assert_eq!(
                f.eval_boolean(v(true), v(false)),
                g(gate(true, false)),
                "{f:?} bool (1,0)"
            );
            assert_eq!(
                f.eval_boolean(v(true), v(true)),
                g(gate(true, true)),
                "{f:?} bool (1,1)"
            );
        }
    }

    /// Bitwise word tables: monomial coefficients (Möbius over all `2w`
    /// interleaved bits) equal the discrete tables — the full Appendix A.3
    /// correctness statement, exhaustive at `w = 4`.
    #[test]
    fn bitwise_tables_match_discrete() {
        let w = 4usize;
        for op in [
            BitwiseOp::And,
            BitwiseOp::Andn,
            BitwiseOp::Or,
            BitwiseOp::Xor,
        ] {
            let coeffs = moebius_coefficients(2 * w, |pt| {
                let x: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j]).collect();
                let y: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j + 1]).collect();
                op.eval_projective(&x, &y)
            });
            for idx in 0..(1usize << (2 * w)) {
                let (x, y) = split_interleaved(idx, w);
                let want = g(op.discrete(x, y, w));
                assert_eq!(coeffs[idx], want, "{op:?} at x={x} y={y}");
            }
        }
    }

    /// `eq` / `LT` word-level closed forms match the discrete tables via
    /// Möbius coefficients at `w = 4`, exhaustively.
    #[test]
    fn equal_lt_word_match_discrete() {
        let w = 4usize;
        let coeffs_eq = moebius_coefficients(2 * w, |pt| {
            let x: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j]).collect();
            let y: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j + 1]).collect();
            equal_projective_eval(&x, &y)
        });
        let coeffs_lt = moebius_coefficients(2 * w, |pt| {
            let x: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j]).collect();
            let y: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j + 1]).collect();
            lt_word_projective_eval(&x, &y)
        });
        for idx in 0..(1usize << (2 * w)) {
            let (x, y) = split_interleaved(idx, w);
            assert_eq!(coeffs_eq[idx], g(u64::from(x == y)), "eq at ({x},{y})");
            assert_eq!(coeffs_lt[idx], g(u64::from(x < y)), "lt at ({x},{y})");
        }
    }

    /// `LT_b(r, ·)` full-domain table: (a) the affine MLE of the
    /// coefficient table equals the closed form at random affine points;
    /// (b) the coefficient table *is* the monomial expansion of the affine
    /// `LT_b(r, Y)` (verified via the Y-only Möbius inversion).
    #[test]
    fn lt_table_closed_form_and_coefficients() {
        let n = 5usize;
        let r: Vec<Goldilocks> = (1..=n as u64).map(|i| g(31 * i + 5)).collect();
        let table = lt_projective_table(&r);
        assert_eq!(table.len(), 1 << n);
        // (a) affine MLE of the coefficient table == closed form.
        let y: Vec<Goldilocks> = (1..=n as u64).map(|i| g(97 * i)).collect();
        let mle = MonomialMle::from_truth_table(&table)
            .unwrap()
            .evaluate(&y)
            .unwrap();
        assert_eq!(mle, lt_projective_eval(&r, &y));
        // (b) Y-Möbius of the closed form == the recurrence's table.
        let coeffs = moebius_coefficients(n, |yy| lt_projective_eval(&r, yy));
        assert_eq!(coeffs, table);
    }

    /// `shift_b` (pcnext kernel): the bivariate monomial coefficients of
    /// the closed form (Möbius over all `2n` interleaved coordinates)
    /// equal the discrete no-wrap shift `1[val(y) + 1 = val(r)]` —
    /// Proposition 3.1 for the shift table.
    #[test]
    fn shift_matches_discrete() {
        let n = 4usize;
        let coeffs = moebius_coefficients(2 * n, |pt| {
            let r: Vec<Goldilocks> = (0..n).map(|j| pt[2 * j]).collect();
            let y: Vec<Goldilocks> = (0..n).map(|j| pt[2 * j + 1]).collect();
            shift_projective_eval(&r, &y)
        });
        for idx in 0..(1usize << (2 * n)) {
            let (r_val, y_val) = split_interleaved(idx, n);
            let want = g(u64::from(y_val + 1 == r_val));
            assert_eq!(coeffs[idx], want, "shift r={r_val} y={y_val}");
        }
    }

    /// `Movsign`, `XORROT(ρ)`, `XORROTW(ρ)`, `Rev8W`, `MulUNoOverflow`,
    /// and `Pow2` closed forms match their discrete semantics via Möbius
    /// coefficients at small widths.
    #[test]
    fn parameterized_families_match_discrete() {
        let w = 4usize;
        // Movsign: 2^w − 1 if MSB of x set, else 0.
        let coeffs = moebius_coefficients(2 * w, |pt| {
            let x: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j]).collect();
            let y: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j + 1]).collect();
            movsign_projective_eval(&x, &y)
        });
        for idx in 0..(1usize << (2 * w)) {
            let (x, _) = split_interleaved(idx, w);
            let want = g(if x >> (w - 1) == 1 { mask(w) } else { 0 });
            assert_eq!(coeffs[idx], want, "movsign at x={x}");
        }

        // XORROT(ρ) over w = 4.
        for rho in [0usize, 1, 3] {
            let coeffs = moebius_coefficients(2 * w, |pt| {
                let x: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j]).collect();
                let y: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j + 1]).collect();
                xorrot_projective_eval(&x, &y, rho)
            });
            let rot_r = |v: u64, r: usize, width: usize| -> u64 {
                let r = r % width;
                if r == 0 {
                    return v & mask(width);
                }
                ((v >> r) | (v << (width - r))) & mask(width)
            };
            for idx in 0..(1usize << (2 * w)) {
                let (x, y) = split_interleaved(idx, w);
                let want = g(rot_r(x ^ y, rho, w));
                assert_eq!(coeffs[idx], want, "xorrot({rho}) at ({x},{y})");
            }
        }

        // XORROTW(ρ) over w = 4, h = 2 (low halves).
        for rho in [0usize, 1] {
            let coeffs = moebius_coefficients(2 * w, |pt| {
                let x: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j]).collect();
                let y: Vec<Goldilocks> = (0..w).map(|j| pt[2 * j + 1]).collect();
                xorrotw_projective_eval(&x, &y, rho)
            });
            let h = w / 2;
            let rot_r = |v: u64, r: usize, width: usize| -> u64 {
                let r = r % width;
                if r == 0 {
                    return v & mask(width);
                }
                ((v >> r) | (v << (width - r))) & mask(width)
            };
            for idx in 0..(1usize << (2 * w)) {
                let (x, y) = split_interleaved(idx, w);
                let lo = rot_r((x ^ y) & mask(h), rho, h);
                assert_eq!(coeffs[idx], g(lo), "xorrotw({rho}) at ({x},{y})");
            }
        }

        // Rev8W over w = 16 (2 bytes per half), total 32 raw bits —
        // verified structurally on sampled supports S via the subset-sum
        // identity F(1_S) = 2^{|S|−1} · rev8(1_S) (the full Möbius walk
        // would need the whole 2^32 cube).
        let w16 = 16usize;
        let total = 2 * w16;
        let swap2 = |v: u64| ((v & 0xFF) << 8) | ((v >> 8) & 0xFF);
        let mut lcg: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lcg >> 33
        };
        for _ in 0..64 {
            let support: u64 = next() & 0x7fff_ffff;
            let z: Vec<Goldilocks> = (0..total)
                .map(|t| g((support >> (total - 1 - t)) & 1))
                .collect();
            let f = rev8w_projective_eval(&z);
            let upper = (support >> w16) & 0xFFFF;
            let lower = support & 0xFFFF;
            let direct = (swap2(upper) << w16) | swap2(lower);
            let ones = support.count_ones();
            if ones == 0 {
                assert_eq!(f, Goldilocks::ZERO, "rev8w empty support");
            } else {
                let scale = 1u64 << (ones - 1);
                assert_eq!(
                    f,
                    g(direct.wrapping_mul(scale)),
                    "rev8w structural at support {support:#x}"
                );
            }
        }

        // MulUNoOverflow over w = 4: upper word zero.
        let coeffs = moebius_coefficients(2 * w, mul_no_overflow_projective_eval);
        for idx in 0..(1usize << (2 * w)) {
            let upper = idx >> w;
            let want = g(u64::from(upper == 0));
            assert_eq!(coeffs[idx], want, "mul_no_overflow at {idx}");
        }

        // Pow2 over k = 3 shift bits, w = 8.
        for z in 0..(1u64 << 3) {
            let coeffs = moebius_coefficients(3, |b| pow2_projective_eval(b, 8));
            assert_eq!(coeffs[z as usize], g(1u64 << (z % 8)), "pow2 at {z}");
        }
    }

    /// `eq` table affine cross-check.
    #[test]
    fn eq_table_affine() {
        let r: Vec<Goldilocks> = (1..=5u64).map(|i| g(13 * i)).collect();
        let t = eq_projective_table(&r);
        let y: Vec<Goldilocks> = (1..=5u64).map(|i| g(7 * i + 3)).collect();
        let mle = MonomialMle::from_truth_table(&t)
            .unwrap()
            .evaluate(&y)
            .unwrap();
        assert_eq!(mle, MonomialMle::eq_projective_eval(&r, &y));
    }
}
