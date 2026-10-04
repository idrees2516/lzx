//! The **Fp256 window fast prover** — the port of ePrint 2025/1117 +
//! 2026/587's `EvalProduct SV,SC` (§5 + Appendix C.4.1) to the CIOS
//! 256-bit field of [`crate::fp256`], the regime where the window
//! actually pays.
//!
//! # Why Fp256 is the port target (the κ argument)
//!
//! On a 64-bit field a field multiplication costs one `u64` multiply —
//! the same order as a table read — so the big/small cost ratio is
//! `κ ≈ 1` and the window's extra bookkeeping can never win wall-clock
//! (the honest Goldilocks finding in `lattice-sumcheck::fastprover`).
//! On the 4-limb CIOS field the ratio is the papers'
//! `κ ≈ 2N² + N = 36` (N = 4 limbs):
//!
//! * a **bb** multiplication (two full-width Montgomery operands) is a
//!   36-native-multiplication CIOS;
//! * an **ss** multiplication (two small canonical integers, the digit /
//!   binary-table regime) is one native `i128` multiply — no CIOS at all,
//!   because `p > 2^254 > 2^127` keeps every small value exact;
//! * an **sb** multiplication (big Montgomery × small canonical) is a
//!   CIOS whose second operand's zero limbs skip whole phase-1 loops —
//!   the [`Fp256::mul_small`] kernel.
//!
//! At the optimal window `v* = log_{d+1}(d²κ)` (Lemma 5/12) the
//! big-equivalent runtime drops to `Θ(M·d^{2−2/δ}·κ^{−1/δ})` versus the
//! linear-time baseline's `Θ(d²·M)` — the papers' measured **2.5–4×**
//! on parameters like `d = 2..3, v* = 3..4`.
//!
//! # The two regimes
//!
//! * **SV (small values)** — every factor table holds small canonical
//!   integers (digit tables, binary columns — the zkVM's committed
//!   regime). The whole window phase — grid construction *and* the
//!   extrapolation stencils — runs on exact `i128` arithmetic (pure ss);
//!   only the challenge weighting and the post-window tail touch CIOS.
//! * **Generic** — arbitrary full-width factors: the multiproduct grid
//!   engine over Montgomery CIOS (the `(d log d)`-bb reduction of
//!   Procedure 1) plus the window's tail cut. Small tables mixed into a
//!   generic instance convert canonically for free (`canon_i128`, no
//!   CIOS).
//!
//! # Protocol shape
//!
//! Standard Boolean sum-check over the Fp256 field: round `j` transmits
//! `[g(0), …, g(d)]` (canonical, fixed-width 32-byte encodings), the
//! round identity is `g(0) + g(1) = C_{j−1}`, and challenges are drawn
//! from the **upper-limb set** (§5 of ePrint 2026/762 — limbs 0–1 zero:
//! always canonical-valid, no rejection, and every multiplication by
//! the challenge's Montgomery form takes the CIOS zero-limb skips).
//! The fast prover is **byte-identical** to [`prove_baseline`] — the
//! window restructures the prover's work, never the transcript.
//!
//! # Small-value soundness discipline
//!
//! The SV specialization is exact whenever the grid values stay within
//! the `i128` budget — enforced fail-closed by checked arithmetic
//! ([`FastFpError::SmallValueOverflow`]); the caller falls back to the
//! generic path. The bit-width precondition is the paper's own
//! ("extrapolated values must remain within the machine-word budget").

// Index-heavy loops (grid strides, window extraction) read clearer with
// explicit indices.
#![allow(clippy::needless_range_loop)]

use crate::fp256::Fp256;
use lattice_core::transcript::{Transcript, TranscriptError};

/// The modulus as a little-endian limb array (the inverse helper's view).
const MOD_P: [u64; 4] = [
    0x43e1_f593_f000_0001,
    0x2833_e848_79b9_7091,
    0xb850_45b6_8181_585d,
    0x3064_4e72_e131_a029,
];

/// Options for the Fp256 fast prover.
#[derive(Clone, Debug)]
pub struct FastFpOpts {
    /// Number of leading rounds to batch through the evaluation grid
    /// (0 = the pure optimized tail / baseline shape).
    pub window: usize,
    /// Collect multiplication statistics ([`take_last_stats`]).
    pub collect_stats: bool,
}

impl Default for FastFpOpts {
    fn default() -> Self {
        FastFpOpts {
            window: 3,
            collect_stats: true,
        }
    }
}

/// Native-multiplication instrumentation (bb = 36-mult CIOS,
/// sb = zero-limb-skipped CIOS ≈ 18–24, ss = native i128).
#[derive(Clone, Copy, Debug, Default)]
pub struct FastFpStats {
    pub bb_mults: u64,
    pub sb_mults: u64,
    pub ss_mults: u64,
}

thread_local! {
    static STATS: std::cell::RefCell<Option<FastFpStats>> =
        const { std::cell::RefCell::new(None) };
}

/// Take the stats collected by the last fast-prover run.
pub fn take_last_stats() -> Option<FastFpStats> {
    STATS.with(|s| s.borrow_mut().take())
}

/// The optimal window from the cost model (Lemma 5 / C.4.1):
/// `v* = log_{d+1}(d²·κ)`, clipped to `[0, ℓ]`.
pub fn optimal_window(d: usize, kappa: f64, ell: usize) -> usize {
    if d == 0 || ell == 0 {
        return 0;
    }
    let target = (d * d) as f64 * kappa;
    let v = target.log((d + 1) as f64).round();
    v.clamp(0.0, ell as f64) as usize
}

/// The paper's Montgomery-field cost ratio `κ ≈ 2N² + N` at N limbs.
pub fn kappa_limbs(n_limbs: usize) -> f64 {
    (2.0 * (n_limbs * n_limbs) as f64) + n_limbs as f64
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FastFpError {
    Transcript(TranscriptError),
    /// The claimed sum does not match the polynomial.
    ClaimMismatch,
    /// The terminal identity failed.
    FinalCheckFailed,
    /// Round shape wrong.
    BadRoundShape {
        round: usize,
        got: usize,
        expected: usize,
    },
    /// Round count wrong.
    BadRoundCount {
        got: usize,
        expected: usize,
    },
    EmptyInstance,
    /// An i128 grid value overflowed the exact-small budget — use the
    /// generic (Big-factor) path.
    SmallValueOverflow,
}

impl core::fmt::Display for FastFpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FastFpError::Transcript(e) => write!(f, "transcript error: {e:?}"),
            FastFpError::ClaimMismatch => write!(f, "claimed sum does not match polynomial"),
            FastFpError::FinalCheckFailed => write!(f, "terminal identity failed"),
            FastFpError::BadRoundShape {
                round,
                got,
                expected,
            } => {
                write!(f, "round {round} length {got} != expected {expected}")
            }
            FastFpError::BadRoundCount { got, expected } => {
                write!(f, "round count {got} != expected {expected}")
            }
            FastFpError::EmptyInstance => write!(f, "empty product term in instance"),
            FastFpError::SmallValueOverflow => {
                write!(f, "small-value grid overflowed the i128 budget")
            }
        }
    }
}

impl From<TranscriptError> for FastFpError {
    fn from(e: TranscriptError) -> Self {
        FastFpError::Transcript(e)
    }
}

// ---------------------------------------------------------------------------
// Instances
// ---------------------------------------------------------------------------

/// A factor table: small canonical integers (the SV regime) or
/// full-width canonical field elements.
///
/// `Big` values are stored **canonical** (limbs < p); the prover's hot
/// loops convert to Montgomery form exactly where the form-preserving
/// kernel needs it.
#[derive(Clone, Debug)]
pub enum FpFactor {
    /// Small canonical values, `|v| < 2^62` each (digit / binary tables).
    Small(Vec<i128>),
    /// Arbitrary canonical field elements.
    Big(Vec<Fp256>),
}

impl FpFactor {
    fn len(&self) -> usize {
        match self {
            FpFactor::Small(v) => v.len(),
            FpFactor::Big(v) => v.len(),
        }
    }

    /// The canonical value at `i`.
    fn canon_at(&self, i: usize) -> Fp256 {
        match self {
            FpFactor::Small(v) => Fp256::canon_i128(v[i]),
            FpFactor::Big(v) => v[i],
        }
    }
}

/// A virtual polynomial over Fp256:
/// `P(x) = Σ_j c_j · Π_{k ∈ term_j} f_k(x)` with **small** coefficients.
#[derive(Clone, Debug)]
pub struct FpVirtualPolynomial {
    pub num_vars: usize,
    pub factors: Vec<FpFactor>,
    /// (small signed coefficient, factor indices).
    pub terms: Vec<(i64, Vec<usize>)>,
}

impl FpVirtualPolynomial {
    pub fn new(num_vars: usize) -> Self {
        FpVirtualPolynomial {
            num_vars,
            factors: Vec::new(),
            terms: Vec::new(),
        }
    }

    pub fn add_factor(&mut self, factor: FpFactor) -> Result<usize, FastFpError> {
        if factor.len() != 1 << self.num_vars {
            return Err(FastFpError::BadRoundShape {
                round: usize::MAX,
                got: factor.len(),
                expected: 1 << self.num_vars,
            });
        }
        self.factors.push(factor);
        Ok(self.factors.len() - 1)
    }

    pub fn add_term(&mut self, coeff: i64, indices: Vec<usize>) -> Result<(), FastFpError> {
        if indices.is_empty() {
            return Err(FastFpError::EmptyInstance);
        }
        if indices.iter().any(|i| *i >= self.factors.len()) {
            return Err(FastFpError::BadRoundShape {
                round: usize::MAX,
                got: indices.iter().max().copied().unwrap_or(0) + 1,
                expected: self.factors.len(),
            });
        }
        self.terms.push((coeff, indices));
        Ok(())
    }

    /// Single degree-`ℓ` product of factors.
    pub fn product(factors: Vec<FpFactor>) -> Result<Self, FastFpError> {
        let num_vars = factors
            .first()
            .map(|f| f.len().trailing_zeros() as usize)
            .unwrap_or(0);
        let mut vp = FpVirtualPolynomial::new(num_vars);
        for f in factors {
            vp.add_factor(f)?;
        }
        let ids: Vec<usize> = (0..vp.factors.len()).collect();
        vp.add_term(1, ids)?;
        Ok(vp)
    }

    pub fn max_degree(&self) -> usize {
        self.terms
            .iter()
            .map(|(_, ids)| ids.len())
            .max()
            .unwrap_or(1)
    }

    fn all_small(&self) -> bool {
        self.factors.iter().all(|f| matches!(f, FpFactor::Small(_)))
    }

    /// The total sum over the Boolean hypercube (canonical).
    ///
    /// SV instances accumulate exactly in `i128` (the whole sum is a
    /// small integer — no CIOS at all); generic instances run the
    /// Montgomery product chain.
    pub fn total_sum(&self) -> Result<Fp256, FastFpError> {
        if self.terms.is_empty() || self.factors.is_empty() {
            return Ok(Fp256::ZERO);
        }
        let m = 1usize << self.num_vars;
        let small: Vec<Option<&Vec<i128>>> = self
            .factors
            .iter()
            .map(|f| match f {
                FpFactor::Small(v) => Some(v),
                FpFactor::Big(_) => None,
            })
            .collect();
        if small.iter().all(|f| f.is_some()) {
            // Exact small accumulation: |term products| < 2^{62·d}.
            let mut total: i128 = 0;
            for (c, ids) in &self.terms {
                let tables: Vec<&[i128]> = ids
                    .iter()
                    .map(|fi| small[*fi].map(|v| v.as_slice()))
                    .collect::<Option<Vec<&[i128]>>>()
                    .ok_or(FastFpError::EmptyInstance)?;
                let mut term_acc: i128 = 0;
                for e in 0..m {
                    let mut prod: i128 = 1;
                    for t in &tables {
                        prod = prod
                            .checked_mul(t[e])
                            .ok_or(FastFpError::SmallValueOverflow)?;
                    }
                    term_acc = term_acc
                        .checked_add(prod)
                        .ok_or(FastFpError::SmallValueOverflow)?;
                }
                total = total
                    .checked_add(
                        (*c as i128)
                            .checked_mul(term_acc)
                            .ok_or(FastFpError::SmallValueOverflow)?,
                    )
                    .ok_or(FastFpError::SmallValueOverflow)?;
            }
            Ok(Fp256::canon_i128(total))
        } else {
            // Generic: per-entry Montgomery chain (1̄ is the CIOS identity),
            // scaled by the small coefficient, canonical at the end.
            let mont_tables: Vec<Vec<Fp256>> = self
                .factors
                .iter()
                .map(|f| (0..m).map(|i| f.canon_at(i).to_mont()).collect())
                .collect();
            let one_bar = Fp256::ONE_CANON.to_mont();
            let mut acc = Fp256::ZERO; // Montgomery accumulator
            for e in 0..m {
                let mut term = Fp256::ZERO; // Montgomery
                for (c, ids) in &self.terms {
                    let mut prod = one_bar;
                    for &fi in ids {
                        prod = prod.mul(&mont_tables[fi][e]);
                    }
                    // CIOS(prod_mont, c_canonical) = TRUE c·Π (canonical).
                    let scaled = prod.mul_small(*c as i128);
                    term = term.add(&scaled.to_mont());
                }
                acc = acc.add(&term);
            }
            Ok(acc.from_mont())
        }
    }
}

// ---------------------------------------------------------------------------
// Proof types
// ---------------------------------------------------------------------------

/// A sum-check proof: round `j` stores `[g(0), …, g(d)]` (canonical).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FpSumcheckProof {
    pub rounds: Vec<Vec<Fp256>>,
}

/// Prover-side result: proof + terminal claims.
#[derive(Clone, Debug)]
pub struct FpSumcheckOutput {
    pub proof: FpSumcheckProof,
    /// The random point, canonical (variable 0 first).
    pub challenges: Vec<Fp256>,
    /// `P(r)` — the claimed evaluation of the whole virtual polynomial.
    pub final_claim: Fp256,
    /// Per-factor claimed canonical evaluations at `r`.
    pub factor_claims: Vec<Fp256>,
}

// ---------------------------------------------------------------------------
// Transcript helpers
// ---------------------------------------------------------------------------

fn absorb_round(ts: &mut Transcript, evals: &[Fp256]) -> Result<(), FastFpError> {
    let mut bytes = Vec::with_capacity(evals.len() * 32);
    for e in evals {
        bytes.extend_from_slice(&e.canon_bytes());
    }
    ts.append_message(b"fp256-round", &bytes)?;
    Ok(())
}

/// Draw the round challenge: 32 transcript bytes → the upper-limb
/// Montgomery form (always valid canonical limbs — no rejection).
fn challenge_mont(ts: &mut Transcript) -> Result<Fp256, FastFpError> {
    let bytes = ts.challenge_bytes(b"fp256-challenge", 32)?;
    let mut h = [0u8; 32];
    h.copy_from_slice(&bytes);
    Ok(Fp256::challenge_upper(&h))
}

/// Standard finite-node Lagrange over `0..=d` at the canonical point `r`
/// — the verifier-side and per-round-claim update. O(d²) canonical
/// products per call (`mul_canon`), negligible next to the hot loops.
fn interpolate_fin(evals: &[Fp256], r: &Fp256) -> Fp256 {
    let n = evals.len();
    let mut acc = Fp256::ZERO;
    for (i, &ei) in evals.iter().enumerate() {
        let mut weight = Fp256::ONE_CANON;
        for j in 0..n {
            if i == j {
                continue;
            }
            let num = r.sub(&Fp256::canon_i128(j as i128));
            let inv = small_inverse(i as i64 - j as i64);
            weight = weight.mul_canon(&num).mul_canon(&inv);
        }
        acc = acc.add(&ei.mul_canon(&weight));
    }
    acc
}

/// The inverse of a small nonzero integer mod p, **exactly and cheaply**:
/// `y = x⁻¹ mod p` satisfies `x·y = 1 + k·p` for some `0 ≤ k < |x|` with
/// `1 + k·p ≡ 0 (mod x)` — i.e. `k ≡ −p⁻¹ (mod x)`. So: fold `p mod x`
/// over its limbs, invert in `u64` (tiny Euclid), and divide `1 + k·p`
/// by `x` with one small-divisor long division. No exponentiation, no
/// 256-bit Euclid.
fn small_inverse(x: i64) -> Fp256 {
    let neg = x < 0;
    let a = x.unsigned_abs();
    debug_assert!((1..(1 << 32)).contains(&a));
    // p mod a (limb fold from the top).
    let mut rem: u128 = 0;
    for limb in MOD_P.iter().rev() {
        rem = ((rem << 64) | (*limb as u128)) % a as u128;
    }
    // k ≡ −(p mod a)⁻¹ (mod a): tiny extended Euclid for the inverse.
    let rinv = tiny_inverse(rem as u64, a);
    let k = (a - rinv) % a;
    // N = 1 + k·p as 5 limbs (k < 2^32, p < 2^254).
    let mut n = [0u64; 5];
    let mut c: u128 = 0;
    for i in 0..4 {
        let s = (k as u128) * (MOD_P[i] as u128) + c;
        n[i] = s as u64;
        c = s >> 64;
    }
    n[4] = c as u64;
    // N += 1.
    let (v0, o0) = n[0].overflowing_add(1);
    n[0] = v0;
    let mut ovf = o0;
    let mut i = 1;
    while ovf && i < 5 {
        let (v, o) = n[i].overflowing_add(1);
        n[i] = v;
        ovf = o;
        i += 1;
    }
    // Long division of the 5-limb N by the small a (top-down).
    let mut q = [0u64; 5];
    let mut r: u128 = 0;
    for i in (0..5).rev() {
        let cur = (r << 64) | (n[i] as u128);
        q[i] = (cur / a as u128) as u64;
        r = cur % a as u128;
    }
    debug_assert_eq!(r, 0, "1 + k·p must divide by a exactly");
    let out = Fp256 {
        limbs: [q[0], q[1], q[2], q[3]],
    };
    if neg {
        out.neg()
    } else {
        out
    }
}

/// Multiplicative inverse of `x` modulo `a` (small extended Euclid).
fn tiny_inverse(x: u64, a: u64) -> u64 {
    let (mut old_r, mut r) = (a, x);
    let (mut old_s, mut s): (i64, i64) = (0, 1);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q as i64 * s);
    }
    old_s.rem_euclid(a as i64) as u64
}

// ---------------------------------------------------------------------------
// The baseline (LinearTimeSC): the honest comparison point
// ---------------------------------------------------------------------------

/// The per-round tail kernel shared by the baseline and the fast prover.
///
/// `bound[k]` is factor `k`'s current table with form `mont[k]` (every
/// entry all-Montgomery or all-canonical — the CIOS form rule keeps the
/// chain's form deterministic). For each `t ∈ 0..=d`:
/// `g(t) = Σ_e c · Π_k (lo_k[e] + t·Δ_k[e])` — the §4 bind-once /
/// per-t-affine structure, one CIOS per (factor, entry, t).
///
/// **Form rule** (`CIOS(a, b) = a·b·R⁻¹`, operand forms `fa, fb`):
/// the result's form is `(fa + fb + 1) mod 2` and its VALUE is the TRUE
/// product — so multiplying by the **Montgomery** constants `t̄` and `c̄`
/// (form 1) preserves the operand's form while scaling it truly.
fn tail_round_messages(
    bound: &[&[Fp256]],
    terms: &[(i64, Vec<usize>)],
    d: usize,
    stats: &mut FastFpStats,
) -> Vec<Fp256> {
    let half = bound.first().map(|b| b.len() / 2).unwrap_or(0);
    // Δ̄_k = hī_k − lō_k (Montgomery subtraction).
    let deltas: Vec<Vec<Fp256>> = bound
        .iter()
        .map(|b| (0..half).map(|e| b[e + half].sub(&b[e])).collect())
        .collect();
    // t̄ (Montgomery) per t — CIOS(Δ̄, t̄) = mont(Δ·t), form (1,1) → 1.
    let t_bars: Vec<Fp256> = (0..=d as i128)
        .map(|t| Fp256::canon_i128(t).to_mont())
        .collect();
    // c̄ (Montgomery) per term — CIOS(prod, c̄) = mont(TRUE c·Π).
    let c_bars: Vec<Fp256> = terms
        .iter()
        .map(|(c, _)| Fp256::canon_i128(*c as i128).to_mont())
        .collect();

    let mut msgs = Vec::with_capacity(d + 1);
    for t_bar in t_bars.iter() {
        // Montgomery accumulator; converted to canonical once per t.
        let mut total_mont = Fp256::ZERO;
        for (tidx, (_c, ids)) in terms.iter().enumerate() {
            let mut term_sum = Fp256::ZERO; // Montgomery
            for e in 0..half {
                // v̄_k = lō + CIOS(Δ̄, t̄) — Montgomery, TRUE lo + t·Δ.
                // Chain: CIOS(prod, v̄_k) = mont(TRUE Π) — every operand
                // Montgomery, so the (1,1) case keeps the result valid.
                let mut prod = Fp256::ZERO;
                let mut first = true;
                for &k in ids {
                    let v = bound[k][e].add(&deltas[k][e].mul(t_bar));
                    stats.bb_mults += 36;
                    prod = if first { v } else { prod.mul(&v) };
                    if !first {
                        stats.bb_mults += 36;
                    }
                    first = false;
                }
                prod = prod.mul(&c_bars[tidx]);
                stats.bb_mults += 36;
                term_sum = term_sum.add(&prod);
            }
            total_mont = total_mont.add(&term_sum);
        }
        msgs.push(total_mont.from_mont());
    }
    msgs
}

// ---------------------------------------------------------------------------
// The small-value grid engine (pure i128 — the ss regime)
// ---------------------------------------------------------------------------

/// Per-axis boolean→U(1) grid conversion (axis digit 0 = ∞ = hi−lo,
/// digit 1 = lo), first variable outermost — the layout of the
/// Goldilocks engine, ported to exact i128.
fn bool_table_to_grid_small(table: &[i128], v: usize) -> Vec<i128> {
    let mut out = table.to_vec();
    let mut stride = 2usize;
    for _ in 0..v {
        let half = stride / 2;
        let mut next = vec![0i128; out.len()];
        for block in 0..(out.len() / stride) {
            for j in 0..half {
                let lo = out[block * stride + j];
                let hi = out[block * stride + half + j];
                next[block * stride + j] = hi - lo;
                next[block * stride + half + j] = lo;
            }
        }
        out = next;
        stride *= 2;
    }
    out
}

/// Extrapolation stencil: `weights[j] = (−1)^{k−1−j}·C(k,j)`, `inf = k!`.
fn stencil_weights(k: usize) -> (Vec<i64>, i64) {
    let mut c: i128 = 1;
    let mut weights = Vec::with_capacity(k);
    for j in 0..k {
        if j > 0 {
            c = c * (k as i128 - j as i128 + 1) / j as i128;
        }
        let sign: i64 = if (k - 1 - j) % 2 == 0 { 1 } else { -1 };
        weights.push(sign * (c as i64));
    }
    let mut fact: i128 = 1;
    for i in 2..=k as i128 {
        fact *= i;
    }
    (weights, fact as i64)
}

/// MultiExtrapolate over the i128 grid (axis-by-axis) — the axis line
/// lives on the **stack** (grid sides <= 24 for every practical degree)
/// and the stencil is computed once per call: no per-line allocation,
/// which the per-suffix grid churn would otherwise dominate.
#[allow(clippy::needless_range_loop)]
fn multi_extrapolate_small(
    grid: &mut Vec<i128>,
    v: usize,
    k: usize,
    h: usize,
    stats: &mut FastFpStats,
) -> Result<(), FastFpError> {
    if h <= k {
        return Ok(());
    }
    let (weights, inf) = stencil_weights(k);
    let mut sides: Vec<usize> = vec![k + 1; v];
    for a in 0..v {
        let old_side = sides[a];
        let new_side = h + 1;
        debug_assert!(new_side <= 24, "grid side exceeds the stack line buffer");
        let outer: usize = sides[..a].iter().product();
        let inner: usize = sides[a + 1..].iter().product();
        let mut next = vec![0i128; outer * new_side * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut line = [0i128; 24];
                for sx in 0..old_side {
                    line[sx] = grid[o * old_side * inner + sx * inner + i];
                }
                // The shifted-evaluation recurrence: p(k+c) = k!·p(∞) +
                // Σ_j stencil[j]·p(c+j) — exact checked i128.
                for c in 0..(h - k) {
                    let mut acc: i128 = 0;
                    for (j, &w) in weights.iter().enumerate() {
                        let term = (w as i128)
                            .checked_mul(line[1 + c + j])
                            .ok_or(FastFpError::SmallValueOverflow)?;
                        acc = acc
                            .checked_add(term)
                            .ok_or(FastFpError::SmallValueOverflow)?;
                    }
                    let inf_term = (inf as i128)
                        .checked_mul(line[0])
                        .ok_or(FastFpError::SmallValueOverflow)?;
                    line[k + 1 + c] = acc
                        .checked_add(inf_term)
                        .ok_or(FastFpError::SmallValueOverflow)?;
                }
                for sx in 0..new_side {
                    next[o * new_side * inner + sx * inner + i] = line[sx];
                }
            }
        }
        stats.ss_mults += (k * (h - k) * outer * inner) as u64;
        *grid = next;
        sides[a] = new_side;
    }
    Ok(())
}

/// MultiProductEval over exact i128 (Procedure 1 — all ss).
fn multi_product_eval_small(
    tables: &[Vec<i128>],
    v: usize,
    stats: &mut FastFpStats,
) -> Result<Vec<i128>, FastFpError> {
    let n = tables.len();
    let mut grid = product_recursive_small(tables, v, stats)?;
    multi_extrapolate_small(&mut grid, v, n, n + 1, stats)?;
    debug_assert_eq!(grid.len(), (n + 2).pow(v as u32));
    Ok(grid)
}

fn product_recursive_small(
    tables: &[Vec<i128>],
    v: usize,
    stats: &mut FastFpStats,
) -> Result<Vec<i128>, FastFpError> {
    let n = tables.len();
    if n == 1 {
        return Ok(bool_table_to_grid_small(&tables[0], v));
    }
    let m = n / 2;
    let (left, right) = tables.split_at(m);
    let mut lg = product_recursive_small(left, v, stats)?;
    let mut rg = product_recursive_small(right, v, stats)?;
    multi_extrapolate_small(&mut lg, v, m, n, stats)?;
    multi_extrapolate_small(&mut rg, v, n - m, n, stats)?;
    // Point-wise product — pure ss (i128).
    let mut out = Vec::with_capacity(lg.len());
    for (x, y) in lg.iter().zip(rg.iter()) {
        out.push(x.checked_mul(*y).ok_or(FastFpError::SmallValueOverflow)?);
        stats.ss_mults += 1;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The generic (Montgomery) grid engine
// ---------------------------------------------------------------------------

/// Montgomery-form stencil constants per degree (built on demand).
struct MontStencil {
    weights: Vec<Fp256>,
    inf: Fp256,
}

fn mont_stencil(k: usize) -> MontStencil {
    let (w, inf) = stencil_weights(k);
    MontStencil {
        weights: w
            .iter()
            .map(|&x| Fp256::canon_i128(x as i128).to_mont())
            .collect(),
        inf: Fp256::canon_i128(inf as i128).to_mont(),
    }
}

fn extrapolate_line_mont(line: &mut Vec<Fp256>, k: usize, h: usize, stats: &mut FastFpStats) {
    if h <= k {
        return;
    }
    let st = mont_stencil(k);
    for c in 0..(h - k) {
        // Montgomery accumulator: k!·p(∞) + Σ w·p — all mont×mont → mont.
        let mut acc = Fp256::ZERO;
        for (j, w) in st.weights.iter().enumerate() {
            acc = acc.add(&line[1 + c + j].mul(w));
            stats.bb_mults += 36;
        }
        let inf_term = line[0].mul(&st.inf);
        stats.bb_mults += 36;
        line.push(acc.add(&inf_term));
    }
}

fn multi_extrapolate_mont(
    grid: &mut Vec<Fp256>,
    v: usize,
    k: usize,
    h: usize,
    stats: &mut FastFpStats,
) {
    if h <= k {
        return;
    }
    let mut sides: Vec<usize> = vec![k + 1; v];
    for a in 0..v {
        let old_side = sides[a];
        let new_side = h + 1;
        let outer: usize = sides[..a].iter().product();
        let inner: usize = sides[a + 1..].iter().product();
        let mut next = vec![Fp256::ZERO; outer * new_side * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut line: Vec<Fp256> = Vec::with_capacity(old_side);
                for s in 0..old_side {
                    line.push(grid[o * old_side * inner + s * inner + i]);
                }
                extrapolate_line_mont(&mut line, k, h, stats);
                for (s, &val) in line.iter().enumerate() {
                    next[o * new_side * inner + s * inner + i] = val;
                }
            }
        }
        *grid = next;
        sides[a] = new_side;
    }
}

fn multi_product_eval_mont(tables: &[Vec<Fp256>], v: usize, stats: &mut FastFpStats) -> Vec<Fp256> {
    let n = tables.len();
    let mut grid = product_recursive_mont(tables, v, stats);
    multi_extrapolate_mont(&mut grid, v, n, n + 1, stats);
    grid
}

fn product_recursive_mont(tables: &[Vec<Fp256>], v: usize, stats: &mut FastFpStats) -> Vec<Fp256> {
    let n = tables.len();
    if n == 1 {
        // bool → U(1) grid on Montgomery values (pure subs).
        let mut out = tables[0].clone();
        let mut stride = 2usize;
        for _ in 0..v {
            let half = stride / 2;
            let mut next = vec![Fp256::ZERO; out.len()];
            for block in 0..(out.len() / stride) {
                for j in 0..half {
                    let lo = out[block * stride + j];
                    let hi = out[block * stride + half + j];
                    next[block * stride + j] = hi.sub(&lo);
                    next[block * stride + half + j] = lo;
                }
            }
            out = next;
            stride *= 2;
        }
        return out;
    }
    let m = n / 2;
    let (left, right) = tables.split_at(m);
    let mut lg = product_recursive_mont(left, v, stats);
    let mut rg = product_recursive_mont(right, v, stats);
    multi_extrapolate_mont(&mut lg, v, m, n, stats);
    multi_extrapolate_mont(&mut rg, v, n - m, n, stats);
    let mut out = Vec::with_capacity(lg.len());
    for (x, y) in lg.iter().zip(rg.iter()) {
        out.push(x.mul(y));
        stats.bb_mults += 36;
    }
    out
}

// ---------------------------------------------------------------------------
// The provers
// ---------------------------------------------------------------------------

/// The linear-time baseline (LinearTimeSC over Fp256): bind once per
/// round, per-t affine products, all CIOS. The honest comparison point
/// for the window prover's speedup.
pub fn prove_baseline(
    vp: &FpVirtualPolynomial,
    claim: Fp256,
    transcript: &mut Transcript,
) -> Result<FpSumcheckOutput, FastFpError> {
    prove_impl(vp, claim, transcript, 0, false)
}

/// The window fast prover (EvalProduct SV,SC): the first `opts.window`
/// rounds are answered from per-suffix evaluation grids.
pub fn prove_fast(
    vp: &FpVirtualPolynomial,
    claim: Fp256,
    transcript: &mut Transcript,
    opts: &FastFpOpts,
) -> Result<FpSumcheckOutput, FastFpError> {
    prove_impl(vp, claim, transcript, opts.window, opts.collect_stats)
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_range_loop)]
fn prove_impl(
    vp: &FpVirtualPolynomial,
    claim: Fp256,
    transcript: &mut Transcript,
    window: usize,
    collect: bool,
) -> Result<FpSumcheckOutput, FastFpError> {
    if vp.terms.is_empty() {
        return Err(FastFpError::EmptyInstance);
    }
    let m = vp.num_vars;
    let d = vp.max_degree();
    let v = window.min(m);
    let suffixes = 1usize << (m - v);
    let mut stats = FastFpStats::default();
    let sv = vp.all_small();

    // Factor views: canonical small tables (SV) or Montgomery tables
    // (generic — canonical storage converted once here, the one-time
    // setup pass).
    let mont_tables: Vec<Vec<Fp256>> = if sv {
        Vec::new()
    } else {
        vp.factors
            .iter()
            .map(|f| {
                (0..(1usize << m))
                    .map(|i| f.canon_at(i).to_mont())
                    .collect()
            })
            .collect()
    };

    let dbg_t0 = std::time::Instant::now();
    // ---- Phase W: per-suffix, per-term grids over U(d+1)^v ----
    let side = d + 2;
    // SV: grids[suffix][term] of exact i128.
    let mut grids_small: Vec<Vec<Vec<i128>>> = Vec::new();
    // Generic: grids_mont[suffix][term] of Montgomery values.
    let mut grids_mont: Vec<Vec<Vec<Fp256>>> = Vec::new();
    if v > 0 {
        if sv {
            let small_views: Vec<&[i128]> = vp
                .factors
                .iter()
                .map(|f| match f {
                    FpFactor::Small(t) => Some(t.as_slice()),
                    FpFactor::Big(_) => None,
                })
                .collect::<Option<Vec<&[i128]>>>()
                .ok_or(FastFpError::EmptyInstance)?;
            for x2 in 0..suffixes {
                let mut term_grids = Vec::with_capacity(vp.terms.len());
                for (_coeff, ids) in &vp.terms {
                    let mut tables: Vec<Vec<i128>> = ids
                        .iter()
                        .map(|&k| {
                            let t = small_views[k];
                            (0..(1usize << v)).map(|b| t[b * suffixes + x2]).collect()
                        })
                        .collect();
                    while tables.len() < d {
                        tables.push(vec![1i128; 1usize << v]);
                    }
                    term_grids.push(multi_product_eval_small(&tables, v, &mut stats)?);
                }
                grids_small.push(term_grids);
            }
        } else {
            for x2 in 0..suffixes {
                let mut term_grids = Vec::with_capacity(vp.terms.len());
                for (_coeff, ids) in &vp.terms {
                    let mut tables: Vec<Vec<Fp256>> = ids
                        .iter()
                        .map(|&k| {
                            (0..(1usize << v))
                                .map(|b| mont_tables[k][b * suffixes + x2])
                                .collect()
                        })
                        .collect();
                    while tables.len() < d {
                        tables.push(vec![Fp256::ONE_CANON.to_mont(); 1usize << v]);
                    }
                    term_grids.push(multi_product_eval_mont(&tables, v, &mut stats));
                }
                grids_mont.push(term_grids);
            }
        }
    }

    if std::env::var_os("LZX_FP_TRACE").is_some() {
        eprintln!(
            "[fp-fast] grids built in {:.2} ms",
            dbg_t0.elapsed().as_secs_f64() * 1e3
        );
    }
    let dbg_t1 = std::time::Instant::now();
    // ---- Window rounds 1..v ----
    let mut rounds: Vec<Vec<Fp256>> = Vec::with_capacity(m);
    let mut challenges: Vec<Fp256> = Vec::with_capacity(m);
    let mut current_claim = claim;
    // Composed Lagrange weights over the bound window axes (MONTGOMERY).
    let mut wj_mont: Vec<Fp256> = vec![Fp256::ONE_CANON.to_mont()];

    // Stride table (axis a's flat-index stride) — computed once.
    let strides: Vec<usize> = (0..v).map(|a| side.pow((v - a - 1) as u32)).collect();

    for j in 1..=v {
        let t_axis = j - 1;
        let tail_axes = v - j;
        let w_offsets: Vec<usize> = {
            let mut offs = Vec::with_capacity(1usize << tail_axes);
            for w in 0..1usize << tail_axes {
                let mut off = 0usize;
                for b in 0..tail_axes {
                    let bit = (w >> (tail_axes - 1 - b)) & 1;
                    off += (bit + 1) * strides[j + b];
                }
                offs.push(off);
            }
            offs
        };
        let stride_t = strides[t_axis];
        let u_count = side.pow((j - 1) as u32);
        // u-part offsets — precomputed once per round (the per-u digit
        // division walk would dominate the hot loop otherwise).
        let u_offsets: Vec<usize> = {
            let mut offs = Vec::with_capacity(u_count);
            for u in 0..u_count {
                let mut u_off = 0usize;
                let mut rem = u;
                for a in (0..(j - 1)).rev() {
                    u_off += (rem % side) * strides[a];
                    rem /= side;
                }
                offs.push(u_off);
            }
            offs
        };
        let mut evals_at = vec![Fp256::ZERO; d + 1];

        if sv {
            // msg[t] = Σ_{x'} Σ_term c · Σ_u W[u]·(Σ_w G[u,t,w]) — the
            // grid tail sums are exact i128, the weighting is the sb
            // kernel CIOS(W̄, small_canonical) = W·small (canonical).
            for term_grids in grids_small.iter() {
                for ((coeff, _ids), grid) in vp.terms.iter().zip(term_grids.iter()) {
                    let coeff_i = *coeff as i128;
                    for (u, &u_off) in u_offsets.iter().enumerate() {
                        let wgt = wj_mont[u];
                        if wgt.is_zero() {
                            continue;
                        }
                        for t in 0..=d {
                            let base = u_off + (t + 1) * stride_t;
                            let mut acc: i128 = 0;
                            for &off in &w_offsets {
                                acc = acc
                                    .checked_add(grid[base + off])
                                    .ok_or(FastFpError::SmallValueOverflow)?;
                            }
                            // c·acc first (ss), then ONE sb weighting mult.
                            let small = acc
                                .checked_mul(coeff_i)
                                .ok_or(FastFpError::SmallValueOverflow)?;
                            stats.ss_mults += 1;
                            let contrib = wgt.mul_small(small);
                            stats.sb_mults += 24;
                            evals_at[t] = evals_at[t].add(&contrib);
                        }
                    }
                }
            }
        } else {
            // Generic: weighting CIOS(W̄, ḡ) = W·G·R (Montgomery) —
            // accumulate per t in Montgomery, convert once per t.
            let mut mont_acc = vec![Fp256::ZERO; d + 1];
            for term_grids in grids_mont.iter() {
                for ((coeff, _ids), grid) in vp.terms.iter().zip(term_grids.iter()) {
                    let c_bar = Fp256::canon_i128(*coeff as i128).to_mont();
                    for (u, &u_off) in u_offsets.iter().enumerate() {
                        let wgt = wj_mont[u];
                        if wgt.is_zero() {
                            continue;
                        }
                        for t in 0..=d {
                            let base = u_off + (t + 1) * stride_t;
                            let mut acc = Fp256::ZERO;
                            for &off in &w_offsets {
                                acc = acc.add(&grid[base + off]);
                            }
                            let contrib = wgt.mul(&acc).mul(&c_bar);
                            stats.bb_mults += 72;
                            mont_acc[t] = mont_acc[t].add(&contrib);
                        }
                    }
                }
            }
            for (t, acc) in mont_acc.iter().enumerate() {
                evals_at[t] = acc.from_mont();
            }
        }

        // Honest guard + transcript (identical labels to the baseline).
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err(FastFpError::ClaimMismatch);
        }
        absorb_round(transcript, &evals_at)?;
        let c_bar = challenge_mont(transcript)?;
        let c_canon = c_bar.from_mont();
        challenges.push(c_canon);
        current_claim = interpolate_fin(&evals_at, &c_canon);
        rounds.push(evals_at);
        // Compose the weight vector for the next round (Montgomery).
        if j < v {
            let l = lagrange_weights_fin_mont(&c_canon, d);
            let mut next = Vec::with_capacity(wj_mont.len() * side);
            for wv in &wj_mont {
                for lv in &l {
                    next.push(wv.mul(lv));
                    stats.bb_mults += 36;
                }
            }
            wj_mont = next;
        }
    }

    if std::env::var_os("LZX_FP_TRACE").is_some() {
        eprintln!(
            "[fp-fast] window rounds in {:.2} ms",
            dbg_t1.elapsed().as_secs_f64() * 1e3
        );
    }
    let dbg_t2 = std::time::Instant::now();
    // ---- Bind the window variables (the prefix adaptation) ----
    // bound[k][x'] = Σ_b eq(r_{<v}, b)·table[b·S + x'].
    // SV: the sb kernel CIOS(ēq, small) = eq·table (canonical).
    // Generic: CIOS(ēq, ḡ) = eq·g·R (Montgomery).
    let eq_weights: Vec<Fp256> = {
        // eq(r_{<v}, b) over the window cube — canonical values.
        let mut eqs = vec![Fp256::ONE_CANON];
        for ci in challenges.iter().take(v) {
            let one_minus = Fp256::ONE_CANON.sub(ci);
            let mut next = Vec::with_capacity(eqs.len() * 2);
            for e in &eqs {
                next.push(e.mul_canon(&one_minus));
                next.push(e.mul_canon(ci));
            }
            eqs = next;
        }
        eqs
    };
    let eq_mont: Vec<Fp256> = eq_weights.iter().map(|e| e.to_mont()).collect();

    // Both paths produce MONTGOMERY bound tables (the tail's CIOS chain
    // requires a Montgomery operand at every multiply — the (0,0)
    // canonical×canonical case yields xy·R⁻¹, a value in neither form).
    // SV: the sb kernel CIOS(ēq, small) = eq·table canonical, then one
    // to_mont pass (the counted setup). Generic: CIOS(ēq, ḡ) = mont
    // directly.
    let mut bound: Vec<Vec<Fp256>> = Vec::with_capacity(vp.factors.len());
    if sv {
        let small_views: Vec<&[i128]> = vp
            .factors
            .iter()
            .map(|f| match f {
                FpFactor::Small(t) => Some(t.as_slice()),
                FpFactor::Big(_) => None,
            })
            .collect::<Option<Vec<&[i128]>>>()
            .ok_or(FastFpError::EmptyInstance)?;
        for t in small_views.iter() {
            let mut out = vec![Fp256::ZERO; suffixes];
            for b in 0..(1usize << v) {
                let w = eq_mont[b];
                if w.is_zero() {
                    continue;
                }
                let base = b * suffixes;
                for x2 in 0..suffixes {
                    out[x2] = out[x2].add(&w.mul_small(t[base + x2]));
                    stats.sb_mults += 24;
                }
            }
            // Canonical → Montgomery (one counted setup pass).
            for x2 in 0..suffixes {
                out[x2] = out[x2].to_mont();
                stats.bb_mults += 36;
            }
            bound.push(out);
        }
    } else {
        for k in 0..vp.factors.len() {
            let mut out = vec![Fp256::ZERO; suffixes];
            for b in 0..(1usize << v) {
                let w = eq_mont[b];
                if w.is_zero() {
                    continue;
                }
                let base = b * suffixes;
                for x2 in 0..suffixes {
                    out[x2] = out[x2].add(&w.mul(&mont_tables[k][base + x2]));
                    stats.bb_mults += 36;
                }
            }
            bound.push(out);
        }
    }

    if std::env::var_os("LZX_FP_TRACE").is_some() {
        eprintln!(
            "[fp-fast] prefix adaptation in {:.2} ms",
            dbg_t2.elapsed().as_secs_f64() * 1e3
        );
    }
    let dbg_t3 = std::time::Instant::now();
    // ---- Tail rounds v+1..m (shared with the baseline, all-Montgomery) ----
    let mut bound_refs: Vec<&[Fp256]> = bound.iter().map(|b| b.as_slice()).collect();
    for _round in v..m {
        let evals_at = tail_round_messages(&bound_refs, &vp.terms, d, &mut stats);
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err(FastFpError::ClaimMismatch);
        }
        absorb_round(transcript, &evals_at)?;
        let c_bar = challenge_mont(transcript)?;
        let c_canon = c_bar.from_mont();
        challenges.push(c_canon);
        current_claim = interpolate_fin(&evals_at, &c_canon);
        rounds.push(evals_at);
        // Bind: bound' = (1−r)·lo + r·hi = lo + r·(hi − lo) — the Δ
        // subtraction keeps the Montgomery form, and CIOS(Δ̄, r̄) =
        // mont(r·Δ) (the zero-limb CIOS skips make this ~20 mults).
        let half = bound_refs[0].len() / 2;
        for b in bound.iter_mut() {
            let lo: Vec<Fp256> = b[..half].to_vec();
            let hi: Vec<Fp256> = b[half..].to_vec();
            let mut next = Vec::with_capacity(half);
            for e in 0..half {
                let delta = hi[e].sub(&lo[e]);
                let rh = delta.mul(&c_bar);
                stats.sb_mults += 20;
                next.push(lo[e].add(&rh));
            }
            *b = next;
        }
        bound_refs = bound.iter().map(|b| b.as_slice()).collect();
    }

    // ---- Terminal claims ----
    let factor_claims: Vec<Fp256> = bound.iter().map(|b| b[0].from_mont()).collect();
    let mut final_claim = Fp256::ZERO;
    for (c, ids) in &vp.terms {
        let mut prod = Fp256::canon_i128(*c as i128);
        for fi in ids {
            prod = prod.mul_canon(&factor_claims[*fi]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current_claim {
        return Err(FastFpError::FinalCheckFailed);
    }

    if std::env::var_os("LZX_FP_TRACE").is_some() {
        eprintln!(
            "[fp-fast] tail in {:.2} ms",
            dbg_t3.elapsed().as_secs_f64() * 1e3
        );
    }
    if collect {
        STATS.with(|s| *s.borrow_mut() = Some(stats));
    }

    Ok(FpSumcheckOutput {
        proof: FpSumcheckProof { rounds },
        challenges,
        final_claim,
        factor_claims,
    })
}

/// Finite-node Lagrange weights over `0..=d` at `r`, in MONTGOMERY form,
/// padded to the grid's axis side `d+2` (index 0 = the ∞ slot at weight
/// ZERO — the integer-node interpolation never reads it). O(d²) CIOS —
/// negligible next to the hot loops.
fn lagrange_weights_fin_mont(r: &Fp256, d: usize) -> Vec<Fp256> {
    let mut out = vec![Fp256::ZERO];
    for j in 0..=d {
        let mut w = Fp256::ONE_CANON;
        for k in 0..=d {
            if k == j {
                continue;
            }
            let xk = Fp256::canon_i128(k as i128);
            let num = r.sub(&xk);
            let inv = small_inverse(j as i64 - k as i64);
            w = w.mul_canon(&num).mul_canon(&inv);
        }
        out.push(w.to_mont());
    }
    debug_assert_eq!(out.len(), d + 2);
    out
}

// ---------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------

/// Verifier-side reduction result.
#[derive(Clone, Debug)]
pub struct FpSumcheckVerifier {
    /// The random point (canonical, variable 0 first).
    pub point: Vec<Fp256>,
    /// The terminal claim the caller's PCS layer must authenticate.
    pub final_claim: Fp256,
}

impl FpSumcheckProof {
    /// Verify against a claimed sum — canonical replay of the round
    /// identities and Lagrange interpolation. The terminal factor claims
    /// are the caller's to authenticate (the ledger pattern).
    pub fn verify(
        &self,
        claim: Fp256,
        num_vars: usize,
        max_degree: usize,
        transcript: &mut Transcript,
    ) -> Result<FpSumcheckVerifier, FastFpError> {
        if self.rounds.len() != num_vars {
            return Err(FastFpError::BadRoundCount {
                got: self.rounds.len(),
                expected: num_vars,
            });
        }
        let mut current = claim;
        let mut point = Vec::with_capacity(num_vars);
        for (i, round) in self.rounds.iter().enumerate() {
            if round.len() != max_degree + 1 {
                return Err(FastFpError::BadRoundShape {
                    round: i,
                    got: round.len(),
                    expected: max_degree + 1,
                });
            }
            absorb_round(transcript, round)?;
            let c_bar = challenge_mont(transcript)?;
            let c_canon = c_bar.from_mont();
            let sum01 = round[0].add(&round[1]);
            if sum01 != current {
                return Err(FastFpError::ClaimMismatch);
            }
            current = interpolate_fin(round, &c_canon);
            point.push(c_canon);
        }
        Ok(FpSumcheckVerifier {
            point,
            final_claim: current,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_table(log_vars: usize, bound: i128, seed: u64) -> Vec<i128> {
        let n = 1usize << log_vars;
        (0..n)
            .map(|i| {
                let h = lattice_core::transcript::Transcript::hash_domain(
                    b"fp-fast",
                    &[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat(),
                );
                (u64::from_le_bytes(h[..8].try_into().unwrap()) % (bound as u64 * 2 + 1)) as i128
                    - bound
            })
            .collect()
    }

    fn big_table(log_vars: usize, seed: u64) -> Vec<Fp256> {
        let n = 1usize << log_vars;
        (0..n)
            .map(|i| {
                let h = lattice_core::transcript::Transcript::hash_domain(
                    b"fp-big",
                    &[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat(),
                );
                Fp256::from_limbs([
                    u64::from_le_bytes(h[0..8].try_into().unwrap()),
                    u64::from_le_bytes(h[8..16].try_into().unwrap()),
                    u64::from_le_bytes(h[16..24].try_into().unwrap()) & 0x0FFF_FFFF_FFFF_FFFF,
                    0,
                ])
            })
            .collect()
    }

    /// Byte-identity: the fast prover at every window size produces the
    /// same proof AND the same challenge path as the baseline (the
    /// papers' invariant — the window restructures work, never the
    /// protocol). Round messages plus per-round challenges (each a
    /// function of the full transcript state) pin the transcript.
    #[test]
    fn sv_byte_identity_all_windows() {
        for &(log_vars, d) in &[(8usize, 2usize), (10, 2), (8, 3)] {
            let factors: Vec<FpFactor> = (0..d)
                .map(|k| FpFactor::Small(small_table(log_vars, 255, 100 + k as u64)))
                .collect();
            let vp = FpVirtualPolynomial::product(factors).unwrap();
            let claim = vp.total_sum().unwrap();
            for w in 0..=4usize {
                let mut ts0 = Transcript::new_default(b"fp-ident");
                let base = prove_baseline(&vp, claim, &mut ts0).unwrap();
                let mut ts1 = Transcript::new_default(b"fp-ident");
                let fast = prove_fast(
                    &vp,
                    claim,
                    &mut ts1,
                    &FastFpOpts {
                        window: w,
                        collect_stats: false,
                    },
                )
                .unwrap();
                assert_eq!(
                    base.proof, fast.proof,
                    "proof mismatch: log={log_vars} d={d} w={w}"
                );
                assert_eq!(
                    base.challenges, fast.challenges,
                    "challenge path mismatch: log={log_vars} d={d} w={w}"
                );
                assert_eq!(base.final_claim, fast.final_claim);
                assert_eq!(base.factor_claims, fast.factor_claims);
            }
        }
    }

    /// Generic (full-width) factors: byte-identity across windows.
    #[test]
    fn generic_byte_identity() {
        for &(log_vars, d) in &[(8usize, 2usize), (9, 3)] {
            let factors: Vec<FpFactor> = (0..d)
                .map(|k| FpFactor::Big(big_table(log_vars, 200 + k as u64)))
                .collect();
            let vp = FpVirtualPolynomial::product(factors).unwrap();
            let claim = vp.total_sum().unwrap();
            for w in 0..=3usize {
                let mut ts0 = Transcript::new_default(b"fp-gen");
                let base = prove_baseline(&vp, claim, &mut ts0).unwrap();
                let mut ts1 = Transcript::new_default(b"fp-gen");
                let fast = prove_fast(
                    &vp,
                    claim,
                    &mut ts1,
                    &FastFpOpts {
                        window: w,
                        collect_stats: false,
                    },
                )
                .unwrap();
                assert_eq!(
                    base.proof, fast.proof,
                    "generic proof mismatch: log={log_vars} d={d} w={w}"
                );
                assert_eq!(
                    base.challenges, fast.challenges,
                    "generic challenge path mismatch: log={log_vars} d={d} w={w}"
                );
            }
        }
    }

    /// Mixed instance (small + big factors) takes the generic path and
    /// still matches the baseline byte-for-byte.
    #[test]
    fn mixed_byte_identity() {
        let factors = vec![
            FpFactor::Small(small_table(8, 255, 1)),
            FpFactor::Big(big_table(8, 2)),
        ];
        let vp = FpVirtualPolynomial::product(factors).unwrap();
        let claim = vp.total_sum().unwrap();
        for w in 0..=3usize {
            let mut ts0 = Transcript::new_default(b"fp-mix");
            let base = prove_baseline(&vp, claim, &mut ts0).unwrap();
            let mut ts1 = Transcript::new_default(b"fp-mix");
            let fast = prove_fast(
                &vp,
                claim,
                &mut ts1,
                &FastFpOpts {
                    window: w,
                    collect_stats: false,
                },
            )
            .unwrap();
            assert_eq!(base.proof, fast.proof, "w={w}");
            assert_eq!(base.challenges, fast.challenges, "w={w}");
        }
    }

    /// The verifier accepts the honest proof and binds the terminal
    /// claim; the factor claims satisfy the product identity.
    #[test]
    fn verify_roundtrip_sv() {
        let factors: Vec<FpFactor> = (0..3)
            .map(|k| FpFactor::Small(small_table(9, 255, 7 + k as u64)))
            .collect();
        let vp = FpVirtualPolynomial::product(factors).unwrap();
        let claim = vp.total_sum().unwrap();
        let mut ts = Transcript::new_default(b"fp-verify");
        let out = prove_fast(&vp, claim, &mut ts, &FastFpOpts::default()).unwrap();
        // Factor claims: product identity.
        let mut prod = Fp256::ONE_CANON;
        for fc in &out.factor_claims {
            prod = prod.mul_canon(fc);
        }
        assert_eq!(prod, out.final_claim);
        // Verifier replay.
        let mut ts2 = Transcript::new_default(b"fp-verify");
        let v = out.proof.verify(claim, 9, 3, &mut ts2).unwrap();
        assert_eq!(v.point, out.challenges);
        assert_eq!(v.final_claim, out.final_claim);
    }

    /// Tampered round messages break the round identity.
    #[test]
    fn tampered_round_rejected() {
        let factors: Vec<FpFactor> = (0..2)
            .map(|k| FpFactor::Small(small_table(8, 255, 11 + k as u64)))
            .collect();
        let vp = FpVirtualPolynomial::product(factors).unwrap();
        let claim = vp.total_sum().unwrap();
        let mut ts = Transcript::new_default(b"fp-tamper");
        let mut out = prove_fast(&vp, claim, &mut ts, &FastFpOpts::default()).unwrap();
        out.proof.rounds[2][1] = out.proof.rounds[2][1].add(&Fp256::ONE_CANON);
        let mut ts2 = Transcript::new_default(b"fp-tamper");
        assert!(matches!(
            out.proof.verify(claim, 8, 2, &mut ts2),
            Err(FastFpError::ClaimMismatch)
        ));
    }

    /// Wrong claim: the prover's own guard fires.
    #[test]
    fn wrong_claim_rejected() {
        let factors: Vec<FpFactor> = (0..2)
            .map(|k| FpFactor::Small(small_table(6, 255, 13 + k as u64)))
            .collect();
        let vp = FpVirtualPolynomial::product(factors).unwrap();
        let claim = vp.total_sum().unwrap().add(&Fp256::ONE_CANON);
        let mut ts = Transcript::new_default(b"fp-wrong");
        assert!(matches!(
            prove_fast(&vp, claim, &mut ts, &FastFpOpts::default()),
            Err(FastFpError::ClaimMismatch)
        ));
    }

    /// Field-layer differential: `mul` vs an independent schoolbook +
    /// reference-reduce path, and the form-bridge roundtrips.
    #[test]
    fn field_layer_differential() {
        use crate::fp256::reduce_wide_ref;
        const RINV: [u64; 4] = [
            0xdc5b_a005_6db1_194e,
            0x090e_f5a9_e111_ec87,
            0xc826_0de4_aeb8_5d5d,
            0x15eb_f951_82c5_551c,
        ];
        let mul_ref = |a: [u64; 4], b: [u64; 4]| -> [u64; 4] {
            let school = |x: [u64; 4], y: [u64; 4]| -> [u64; 8] {
                let mut wide = [0u64; 8];
                for i in 0..4 {
                    let mut carry: u128 = 0;
                    for j in 0..4 {
                        let s = (wide[i + j] as u128) + (x[i] as u128) * (y[j] as u128) + carry;
                        wide[i + j] = s as u64;
                        carry = s >> 64;
                    }
                    wide[i + 4] = (wide[i + 4] as u128 + carry) as u64;
                }
                wide
            };
            let ab = reduce_wide_ref(&school(a, b));
            reduce_wide_ref(&school(ab, RINV))
        };
        let cases: Vec<([u64; 4], [u64; 4])> = vec![
            ([1, 0, 0, 0], crate::fp256::R2_C),
            (crate::fp256::R2_C, [1, 0, 0, 0]),
            ([17, 0, 0, 0], [5, 0, 0, 0]),
            (crate::fp256::R_C, crate::fp256::R2_C),
            (
                [0x1234_5678_9abc_def0, 0x0fed_cba9_8765_4321, 0x1, 0x2],
                [0x999, 0x888, 0x777, 0x666],
            ),
            ([0xffff_ffff_ffff_ffff, 2, 0, 0], [3, 0, 0, 0]),
        ];
        for (a, b) in cases {
            let got = Fp256 { limbs: a }.mul(&Fp256 { limbs: b });
            let want = mul_ref(a, b);
            assert_eq!(got.limbs, want, "mul mismatch a={a:?} b={b:?}");
        }
        // Bridge roundtrips on random canonical values.
        for seed in 0..8u64 {
            let h = lattice_core::transcript::Transcript::hash_domain(b"diff", &seed.to_le_bytes());
            let v = Fp256 {
                limbs: [
                    u64::from_le_bytes(h[0..8].try_into().unwrap()),
                    u64::from_le_bytes(h[8..16].try_into().unwrap()),
                    u64::from_le_bytes(h[16..24].try_into().unwrap()) & 0x0fff_ffff_ffff_ffff,
                    0,
                ],
            };
            assert_eq!(
                v.to_mont().from_mont(),
                v,
                "to_mont/from_mont roundtrip {seed}"
            );
            assert_eq!(v.to_mont().from_mont(), v);
            // mul_small: mont(x)·s == canon(x·s).
            let x17 = v.to_mont().mul_small(17);
            assert_eq!(x17, v.mul_canon(&Fp256::canon_i128(17)), "mul_small {seed}");
        }
        // to_mont(1) == R.
        assert_eq!(Fp256::ONE_CANON.to_mont().limbs, crate::fp256::R_C);
    }

    /// The small-integer modular inverse: `x · x⁻¹ ≡ 1` for the Lagrange
    /// denominators (the tiny-Euclid + small-division construction).
    #[test]
    fn small_inverse_correct() {
        for x in [1i64, 2, 3, 4, 5, 7, 8, 16, 31, -1, -2, -3, -7, -31] {
            let inv = small_inverse(x);
            let a = Fp256::canon_i128(x as i128);
            let prod = a.mul_canon(&inv);
            assert_eq!(prod, Fp256::ONE_CANON, "inverse wrong for x={x}");
        }
    }

    /// Finite-node Lagrange interpolation against direct evaluation.
    #[test]
    fn interpolate_matches_direct() {
        // p(X) = 3X² + 2X + 7: values at 0,1,2 = 7, 12, 23; p(5) = 92.
        let evals = [
            Fp256::canon_i128(7),
            Fp256::canon_i128(12),
            Fp256::canon_i128(23),
        ];
        let r = Fp256::canon_i128(5);
        assert_eq!(interpolate_fin(&evals, &r), Fp256::canon_i128(92));
        // Degree-3: p(X) = X³ − X + 1 at 0..3 → p(7) = 343 − 7 + 1 = 337.
        let evals3 = [
            Fp256::canon_i128(1),
            Fp256::canon_i128(1),
            Fp256::canon_i128(7),
            Fp256::canon_i128(25),
        ];
        let r7 = Fp256::canon_i128(7);
        assert_eq!(interpolate_fin(&evals3, &r7), Fp256::canon_i128(337));
    }

    /// The optimal-window formula matches the paper's target values:
    /// `v* = log_{d+1}(d²·κ)` at the Montgomery `κ ≈ 2N²+N = 36`.
    #[test]
    fn optimal_window_matches_model() {
        let kappa = kappa_limbs(4); // 36
        assert_eq!(optimal_window(2, kappa, 20), 5); // log_3(144) ≈ 4.53 → 5
        assert_eq!(optimal_window(3, kappa, 20), 4); // log_4(324) ≈ 4.17 → 4
        assert_eq!(optimal_window(2, 1.0, 20), 1); // κ=1 (64-bit fields): log_3(4) ≈ 1.26 → 1
        assert_eq!(optimal_window(2, kappa, 2), 2); // clipped at ℓ
    }

    /// The i128 grid engine: products match the naive MLE evaluation.
    #[test]
    fn small_grid_matches_naive() {
        let v = 2usize;
        let n = 3usize;
        let tables: Vec<Vec<i128>> = (0..n)
            .map(|k| {
                vec![
                    (k * 3 + 1) as i128,
                    (k * 5 + 2) as i128,
                    (k * 7 + 3) as i128,
                    (k * 11 + 4) as i128,
                ]
            })
            .collect();
        let mut stats = FastFpStats::default();
        let grid = multi_product_eval_small(&tables, v, &mut stats).unwrap();
        let side = n + 2;
        assert_eq!(grid.len(), side * side);
        // MLE of a 2^v table at integer point (x, y) — variable 0 (the
        // outermost, table MSB) binds to x, variable 1 to y.
        let mle_at = |table: &[i128], x: i128, y: i128| -> i128 {
            let a = table[0] + x * (table[2] - table[0]);
            let b = table[1] + x * (table[3] - table[1]);
            a + y * (b - a)
        };
        for x in 0..=n as i128 {
            for y in 0..=n as i128 {
                let mut want = 1i128;
                for t in &tables {
                    want *= mle_at(t, x, y);
                }
                let got = grid[((x + 1) as usize) * side + (y + 1) as usize];
                assert_eq!(got, want, "grid mismatch at ({x},{y})");
            }
        }
    }

    /// Stats sanity: the SV window's grid construction is pure ss.
    #[test]
    fn sv_stats_are_ss_dominant() {
        let factors: Vec<FpFactor> = (0..2)
            .map(|k| FpFactor::Small(small_table(10, 255, 21 + k as u64)))
            .collect();
        let vp = FpVirtualPolynomial::product(factors).unwrap();
        let claim = vp.total_sum().unwrap();
        let mut ts = Transcript::new_default(b"fp-stats");
        let _ = prove_fast(
            &vp,
            claim,
            &mut ts,
            &FastFpOpts {
                window: 3,
                collect_stats: true,
            },
        );
        let stats = take_last_stats().unwrap();
        // The grid engine ran entirely on i128.
        assert!(stats.ss_mults > 0);
        // Window weighting + tail still paid bb/sb mults.
        assert!(stats.sb_mults > 0);
    }
}
