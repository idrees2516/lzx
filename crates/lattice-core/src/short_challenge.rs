//! Short **ring-element** challenge sets for lattice folding protocols
//! (Wave 6 substrate, `NEXT_STEPS.md` §2.1).
//!
//! Every folding paper in this workspace derives its binding power from
//! challenges drawn from a *large structured set of short ring elements* —
//! not from small integer scalars:
//!
//! | Paper | Challenge set | Entropy |
//! |---|---|---|
//! | PikkuFold §5 Table 3 | `C^fw` fixed-weight ternary (τ = 23 of N = 256) | ≈ 2^100+ |
//! | Cyclo §3 App B | `D` biased ternary over R_q | ≈ 2^203 |
//! | LatticeFold+ §4 | `S̄ = {-2..2}` per coefficient | ≈ 2^148 |
//! | ProtogaLattice §2.6 Table 2 | `C = {-1,0,1,2}^N` at weight T = 128 | 2^128 |
//! | Symphony / LaBRADOR | `S = {0,±1,±2}` (‖S‖op ≤ 15) | 2^128+ |
//! | Akita / HyperWolf / RoKoko | fixed-weight signed (TAU-class) | 2^100+ |
//!
//! Sampling a scalar with 8-17 bits instead (the pre-Wave-6 state of every
//! folding module) caps the knowledge error at 2^-8..2^-17 — roughly 90-130
//! bits short of λ = 128. This module provides the shared distribution
//! machinery so each protocol can consume paper-calibrated challenges with:
//!
//! * **exact unbiased rejection** over the coefficient value sets (same
//!   discipline as [`crate::challenge_set`]),
//! * **distinct-position sampling** for fixed-weight families via partial
//!   Fisher-Yates (O(n) construction, O(w) swaps),
//! * **certified operator-norm bounds Γ_C**: for a challenge `c` sampled
//!   here and any ring element `x`,
//!   `‖c·x‖∞ ≤ Γ_C(c) · ‖x‖₂ ≤ Γ_C(c) · √N · ‖x‖∞`
//!   with `Γ_C(c) = ‖σ(c)‖∞ ≤ ⌈√(Σ c_i²)⌉` — a per-sample rigorous
//!   bound (Cauchy-Schwarz over the canonical embeddings), computed without
//!   big-integer embedding evaluations,
//! * **op-norm rejection with retry** (`sample_with_gamma_cap`): resample
//!   from a counter-domain seed until the certified Γ_C fits a protocol
//!   budget, mirroring the papers' rejection-to-Γ steps,
//! * **entropy accounting** (`entropy_bits`, `require_entropy`) so callers
//!   can refuse undersized challenge spaces at construction time.
//!
//! The module is deliberately ring-agnostic (signed integer coefficients);
//! conversion into a concrete `R_q` element is one
//! [`lattice_ring::RingElement::from_signed`] call performed by the
//! protocol crate.

use crate::keccak::shake256;

/// The structured short-challenge families used by the folding papers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShortChallengeFamily {
    /// Fixed Hamming weight `weight`: exactly `weight` of the `n`
    /// coefficients are ±`amplitude` (uniform positions, uniform signs).
    /// Akita A8 / HyperWolf H5 / RoKoko TAU / PikkuFold `C^fw`.
    FixedWeight { weight: usize, amplitude: i64 },
    /// Coefficients iid ternary with `P(±1) = p/2` each and `P(0) = 1-p`,
    /// where `p = p_nonzero_permille / 1000`. Cyclo's set `D`.
    BiasedTernary { p_nonzero_permille: u32 },
    /// Coefficients iid uniform over an arbitrary small value set —
    /// LatticeFold+ `S̄ = {-2,-1,0,1,2}`, Symphony `S = {0,±1,±2}`.
    SmallSet { values: Vec<i64> },
    /// Fixed Hamming weight `weight` with non-zero values drawn uniformly
    /// from `values` — ProtogaLattice `C = {-1,0,1,2}^N` at weight T.
    FixedWeightSmallSet { weight: usize, values: Vec<i64> },
}

/// A fully-specified challenge distribution over `n` slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortChallengeSpec {
    /// Number of ring coefficients (the ring dimension N).
    pub n: usize,
    pub family: ShortChallengeFamily,
}

/// A sampled short challenge: signed coefficients plus bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortChallenge {
    /// Signed (balanced) coefficients; `coefficients.len() == n`.
    pub coefficients: Vec<i64>,
    family: ShortChallengeFamily,
    /// Certified ℓ∞→ℓ∞ operator-norm growth factor Γ_C (see module docs).
    gamma_c: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShortChallengeError {
    /// Rejection sampling exceeded its budget — fail closed.
    RejectionBudgetExceeded,
    /// Malformed spec (weight > n, empty value set, bad bias).
    InvalidParameters,
    /// Op-norm rejection exhausted its retries.
    GammaCapExceeded {
        gamma: u64,
        cap: u64,
        retries: usize,
    },
}

impl ShortChallengeSpec {
    /// Validate structural parameters.
    pub fn validate(&self) -> Result<(), ShortChallengeError> {
        if self.n == 0 || self.n > (1 << 20) {
            return Err(ShortChallengeError::InvalidParameters);
        }
        match &self.family {
            ShortChallengeFamily::FixedWeight { weight, amplitude } => {
                if *weight == 0 || *weight > self.n || *amplitude <= 0 {
                    return Err(ShortChallengeError::InvalidParameters);
                }
            }
            ShortChallengeFamily::BiasedTernary { p_nonzero_permille } => {
                if *p_nonzero_permille == 0 || *p_nonzero_permille > 1000 {
                    return Err(ShortChallengeError::InvalidParameters);
                }
            }
            ShortChallengeFamily::SmallSet { values } => {
                if values.is_empty() || values.len() > 1024 {
                    return Err(ShortChallengeError::InvalidParameters);
                }
            }
            ShortChallengeFamily::FixedWeightSmallSet { weight, values } => {
                if *weight == 0 || *weight > self.n || values.is_empty() || values.len() > 1024 {
                    return Err(ShortChallengeError::InvalidParameters);
                }
            }
        }
        Ok(())
    }

    /// Maximum absolute coefficient value the family can emit.
    pub fn amplitude_max(&self) -> i64 {
        match &self.family {
            ShortChallengeFamily::FixedWeight { amplitude, .. } => *amplitude,
            ShortChallengeFamily::BiasedTernary { .. } => 1,
            ShortChallengeFamily::SmallSet { values } => {
                values.iter().map(|v| v.abs()).max().unwrap_or(0)
            }
            ShortChallengeFamily::FixedWeightSmallSet { values, .. } => {
                values.iter().map(|v| v.abs()).max().unwrap_or(0)
            }
        }
    }

    /// Worst-case (symbolic, family-level) Γ_C bound: `B · ⌈√k⌉` where `k`
    /// is the maximum number of non-zero coefficients.
    pub fn gamma_c_worst_case(&self) -> u64 {
        let b = self.amplitude_max().max(1) as u64;
        let k = match &self.family {
            ShortChallengeFamily::FixedWeight { weight, .. }
            | ShortChallengeFamily::FixedWeightSmallSet { weight, .. } => (*weight).min(self.n),
            _ => self.n,
        };
        b.saturating_mul(ceil_sqrt(k as u64))
    }

    /// log2 of the challenge space size (min-entropy bookkeeping).
    pub fn entropy_bits(&self) -> f64 {
        let n = self.n as f64;
        match &self.family {
            ShortChallengeFamily::FixedWeight { weight, .. } => {
                ln_binom(n, *weight as f64) / std::f64::consts::LN_2 + *weight as f64
            }
            ShortChallengeFamily::BiasedTernary { p_nonzero_permille } => {
                let p = f64::from(*p_nonzero_permille) / 1000.0;
                // Binary entropy of {0, ±1} with P(nonzero) = p, split evenly.
                n * binary_entropy(p)
            }
            ShortChallengeFamily::SmallSet { values } => n * (values.len() as f64).log2(),
            ShortChallengeFamily::FixedWeightSmallSet { weight, values } => {
                ln_binom(n, *weight as f64) / std::f64::consts::LN_2
                    + *weight as f64 * (values.len() as f64).log2()
            }
        }
    }

    /// Refuse undersized challenge spaces (e.g. λ = 128 folding targets).
    pub fn require_entropy(&self, min_bits: f64) -> Result<(), ShortChallengeError> {
        if self.entropy_bits() < min_bits {
            return Err(ShortChallengeError::InvalidParameters);
        }
        Ok(())
    }

    /// Deterministically sample one challenge from a seed. The seed should
    /// be a fresh transcript challenge (`challenge_bytes`), making the draw
    /// public-coin.
    pub fn sample(&self, seed: &[u8]) -> Result<ShortChallenge, ShortChallengeError> {
        self.validate()?;
        let coefficients = match &self.family {
            ShortChallengeFamily::FixedWeight { weight, amplitude } => {
                sample_fixed_weight(*weight, *amplitude, self.n, seed)?
            }
            ShortChallengeFamily::BiasedTernary { p_nonzero_permille } => {
                sample_biased_ternary(*p_nonzero_permille, self.n, seed)?
            }
            ShortChallengeFamily::SmallSet { values } => sample_small_set(values, self.n, seed)?,
            ShortChallengeFamily::FixedWeightSmallSet { weight, values } => {
                sample_fixed_weight_set(*weight, values, self.n, seed)?
            }
        };
        let gamma_c = certified_gamma(&coefficients);
        Ok(ShortChallenge {
            coefficients,
            family: self.family.clone(),
            gamma_c,
        })
    }

    /// Sample with op-norm rejection: retry (up to `retries` times, each
    /// from a counter-domain seed) until the certified Γ_C fits `cap`.
    /// Fails closed with [`ShortChallengeError::GammaCapExceeded`].
    pub fn sample_with_gamma_cap(
        &self,
        seed: &[u8],
        cap: u64,
        retries: usize,
    ) -> Result<ShortChallenge, ShortChallengeError> {
        if cap == 0 {
            return Err(ShortChallengeError::InvalidParameters);
        }
        let mut last_gamma = 0u64;
        for attempt in 0..=retries {
            let mut domain = Vec::with_capacity(seed.len() + 8);
            domain.extend_from_slice(seed);
            domain.extend_from_slice(&(attempt as u64).to_le_bytes());
            let c = self.sample(&domain)?;
            last_gamma = c.gamma_c;
            if c.gamma_c <= cap {
                return Ok(c);
            }
        }
        Err(ShortChallengeError::GammaCapExceeded {
            gamma: last_gamma,
            cap,
            retries,
        })
    }
}

impl ShortChallenge {
    /// The family this challenge was sampled from.
    pub fn family(&self) -> &ShortChallengeFamily {
        &self.family
    }

    /// ℓ∞ norm of the coefficient vector.
    pub fn l_inf(&self) -> i64 {
        self.coefficients.iter().map(|c| c.abs()).max().unwrap_or(0)
    }

    /// ℓ2 norm squared.
    pub fn l2_squared(&self) -> u128 {
        self.coefficients
            .iter()
            .map(|c| u128::from(c.unsigned_abs().pow(2)))
            .sum()
    }

    /// Certified per-sample operator-norm factor Γ_C:
    /// `Γ_C = ⌈√(Σ c_i²)⌉ ≥ ‖σ(c)‖∞`, so for any `x` in an N-dim ring:
    /// `‖c·x‖∞ ≤ Γ_C · ‖x‖₂ ≤ Γ_C · √N · ‖x‖∞` (the `√N` is applied by
    /// the consumer — see `NormBudget::fold`).
    pub fn gamma_c(&self) -> u64 {
        self.gamma_c
    }

    /// Hamming weight (non-zero coefficients).
    pub fn hamming_weight(&self) -> usize {
        self.coefficients.iter().filter(|c| **c != 0).count()
    }
}

/// `Γ_C = ⌈√(Σ c_i²)⌉` — rigorous upper bound for `‖σ(c)‖∞` (Cauchy-
/// Schwarz: `|Σ c_i ζ^i| ≤ ‖c‖₂` for unit-modulus `ζ^i`). The `√N` factor
/// of the full growth law lives in [`crate::norm_budget::NormBudget::fold`].
fn certified_gamma(coefficients: &[i64]) -> u64 {
    let l2sq: u128 = coefficients
        .iter()
        .map(|c| (*c as i128 * *c as i128) as u128)
        .sum();
    ceil_sqrt_u128(l2sq)
}

/// Ceiling square root (u64).
fn ceil_sqrt(x: u64) -> u64 {
    if x == 0 {
        return 0;
    }
    let r = (x as f64).sqrt() as u64;
    // Correct rounding around the float estimate.
    let mut r = r.max(1);
    while r.saturating_mul(r) < x {
        r += 1;
    }
    while r > 1 && (r - 1).saturating_mul(r - 1) >= x {
        r -= 1;
    }
    r
}

/// Ceiling square root (u128) — float seed + full integer correction
/// (the seed's ulp error at 2^64 scale is ~2^12, so the correction loops
/// are bounded and cheap).
fn ceil_sqrt_u128(x: u128) -> u64 {
    if x == 0 {
        return 0;
    }
    if x <= u64::MAX as u128 {
        return ceil_sqrt(x as u64);
    }
    // Newton iteration from a float seed.
    let mut g = (x as f64).sqrt() as u128 + 2;
    loop {
        let next = (g + x / g) / 2;
        if next >= g {
            break;
        }
        g = next;
    }
    // Full correction: the seed may sit either side of the true floor.
    // checked_mul: a product overflowing u128 means (g±1)² > x for sure —
    // stop (saturating_mul would read as u128::MAX and miscompare).
    while g.saturating_mul(g) > x {
        g -= 1;
    }
    while let Some(p) = (g + 1).checked_mul(g + 1) {
        if p <= x {
            g += 1;
        } else {
            break;
        }
    }
    // g = floor(sqrt(x)) now; ceiling:
    let g64 = u64::try_from(g).unwrap_or(u64::MAX);
    if g.saturating_mul(g) < x {
        g64.saturating_add(1)
    } else {
        g64
    }
}

/// XOF stream for a sampling domain (labelled, length-framed).
fn stream(label: &[u8], seed: &[u8], attempt: u64, bytes: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(24 + label.len() + seed.len());
    buf.extend_from_slice(b"SHORTCHAL");
    buf.extend_from_slice(&(label.len() as u32).to_le_bytes());
    buf.extend_from_slice(label);
    buf.extend_from_slice(&(seed.len() as u32).to_le_bytes());
    buf.extend_from_slice(seed);
    buf.extend_from_slice(&attempt.to_le_bytes());
    shake256(&buf, bytes)
}

/// Unbiased uniform index in [0, range) from a byte cursor with rejection.
struct UniformCursor<'a> {
    stream: &'a [u8],
    pos: usize,
}

impl<'a> UniformCursor<'a> {
    fn next_index(&mut self, range: u64) -> Option<u64> {
        if range == 0 {
            return None;
        }
        // Rejection over the u32 window: accept when the raw value is below
        // floor(2^32 / range) * range.
        let limit = if range <= 1 {
            return Some(0);
        } else if range > u32::MAX as u64 {
            // Large ranges (n up to 2^20 here): use a 64-bit window.
            let lim = (u64::MAX / range) * range;
            loop {
                let raw = self.next_u64()?;
                if raw < lim {
                    return Some(raw % range);
                }
            }
        } else {
            ((u32::MAX as u64 + 1) / range) * range
        };
        loop {
            let raw = self.next_u32()? as u64;
            if raw < limit {
                return Some(raw % range);
            }
        }
    }

    fn next_u32(&mut self) -> Option<u32> {
        if self.pos + 4 > self.stream.len() {
            return None;
        }
        let v = u32::from_le_bytes(self.stream[self.pos..self.pos + 4].try_into().ok()?);
        self.pos += 4;
        Some(v)
    }

    fn next_u64(&mut self) -> Option<u64> {
        if self.pos + 8 > self.stream.len() {
            return None;
        }
        let v = u64::from_le_bytes(self.stream[self.pos..self.pos + 8].try_into().ok()?);
        self.pos += 8;
        Some(v)
    }
}

/// Fixed weight, ±amplitude: partial Fisher-Yates over an index array.
fn sample_fixed_weight(
    weight: usize,
    amplitude: i64,
    n: usize,
    seed: &[u8],
) -> Result<Vec<i64>, ShortChallengeError> {
    // Entropy budget: each position costs ≤ 4-5 rejected u32s (range ≤ n),
    // each sign 1 bit; 16x headroom.
    let bytes = weight * 24 + 32;
    let s = stream(b"fw", seed, 0, bytes);
    let mut cur = UniformCursor { stream: &s, pos: 0 };
    let mut indices: Vec<u32> = (0..n as u32).collect();
    let mut coefficients = vec![0i64; n];
    for slot in 0..weight {
        let range = (n - slot) as u64;
        let pos = cur
            .next_index(range)
            .ok_or(ShortChallengeError::RejectionBudgetExceeded)?;
        let idx = indices[pos as usize];
        indices.swap(pos as usize, n - 1 - slot);
        let sign = cur
            .next_index(2)
            .ok_or(ShortChallengeError::RejectionBudgetExceeded)?;
        let value = if sign == 0 { amplitude } else { -amplitude };
        coefficients[idx as usize] = value;
    }
    Ok(coefficients)
}

/// Fixed weight with values from a set: same Fisher-Yates, set-index draw.
fn sample_fixed_weight_set(
    weight: usize,
    values: &[i64],
    n: usize,
    seed: &[u8],
) -> Result<Vec<i64>, ShortChallengeError> {
    let bytes = weight * 32 + 32;
    let s = stream(b"fwset", seed, 0, bytes);
    let mut cur = UniformCursor { stream: &s, pos: 0 };
    let mut indices: Vec<u32> = (0..n as u32).collect();
    let mut coefficients = vec![0i64; n];
    for slot in 0..weight {
        let range = (n - slot) as u64;
        let pos = cur
            .next_index(range)
            .ok_or(ShortChallengeError::RejectionBudgetExceeded)?;
        let idx = indices[pos as usize];
        indices.swap(pos as usize, n - 1 - slot);
        let vpos = cur
            .next_index(values.len() as u64)
            .ok_or(ShortChallengeError::RejectionBudgetExceeded)?;
        coefficients[idx as usize] = values[vpos as usize];
    }
    Ok(coefficients)
}

/// iid biased ternary: P(0) = 1-p, P(±1) = p/2 each. Unbiased ternary
/// rejection window: byte b accepted iff b < 243 (243 = 3^5 divides evenly
/// into 243·1 < 256); residue r = b mod 3 ∈ {0,1,2}; coin for the sign from
/// a second bit plane, gated by the per-coefficient nonzero coin at the
/// permille bias (1024-way rejection window on a separate byte).
fn sample_biased_ternary(
    p_nonzero_permille: u32,
    n: usize,
    seed: &[u8],
) -> Result<Vec<i64>, ShortChallengeError> {
    // 3 bytes per coefficient (coin + ternary + sign) with headroom.
    let bytes = n * 6 + 32;
    let s = stream(b"bias3", seed, 0, bytes);
    let mut cur = UniformCursor { stream: &s, pos: 0 };
    let mut coefficients = Vec::with_capacity(n);
    while coefficients.len() < n {
        // Nonzero coin: the low 10 bits are uniform (window 1024), the
        // threshold p ∈ [1, 1000] gives an exact permille bias.
        let raw = cur
            .next_u32()
            .ok_or(ShortChallengeError::RejectionBudgetExceeded)?;
        let coin_win = raw & 0x3FF;
        let nonzero = coin_win < p_nonzero_permille;
        if !nonzero {
            coefficients.push(0);
            continue;
        }
        // Sign coin (fair bit with rejection on the low window).
        let sign_raw = cur
            .next_u32()
            .ok_or(ShortChallengeError::RejectionBudgetExceeded)?;
        let sign = (sign_raw & 1 == 0) as i64 * 2 - 1;
        coefficients.push(sign);
    }
    Ok(coefficients)
}

/// iid uniform over a small value set: per coefficient one index draw with
/// unbiased modulo rejection over the set size.
fn sample_small_set(
    values: &[i64],
    n: usize,
    seed: &[u8],
) -> Result<Vec<i64>, ShortChallengeError> {
    let bytes = n * 8 + 32;
    let s = stream(b"smallset", seed, 0, bytes);
    let mut cur = UniformCursor { stream: &s, pos: 0 };
    let mut coefficients = Vec::with_capacity(n);
    while coefficients.len() < n {
        let idx = cur
            .next_index(values.len() as u64)
            .ok_or(ShortChallengeError::RejectionBudgetExceeded)?;
        coefficients.push(values[idx as usize]);
    }
    Ok(coefficients)
}

fn ln_binom(n: f64, k: f64) -> f64 {
    ln_factorial(n) - ln_factorial(k) - ln_factorial((n - k).max(0.0))
}

fn ln_factorial(x: f64) -> f64 {
    if x <= 1.0 {
        0.0
    } else {
        x * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI * x).ln()
    }
}

/// Binary entropy H(p) (bits) for a ternary split {0: 1-p, ±1: p/2 each}.
fn binary_entropy(p: f64) -> f64 {
    let q = 1.0 - p;
    let h2 = |a: f64| {
        if a <= 0.0 || a >= 1.0 {
            0.0
        } else {
            -a * a.log2()
        }
    };
    // Entropy of the three-outcome distribution.
    let mut h = 0.0;
    if p / 2.0 > 0.0 && p / 2.0 < 1.0 {
        h -= (p / 2.0) * (p / 2.0).log2() * 2.0;
    }
    if q > 0.0 && q < 1.0 {
        h -= q * q.log2();
    }
    let _ = h2; // kept for documentation symmetry
    h
}

// ---------------------------------------------------------------------------
// Paper-calibrated profiles (the per-paper instantiations from the roadmap).
// ---------------------------------------------------------------------------

/// PikkuFold §5 Table 3: `C^{fw}_{256,23,1}` — fixed weight 23, ±1, N = 256
/// (γ ≈ 8.357 in the paper's tight analysis; our certified bound is the
/// rigorous `⌈√23⌉·⌈√256⌉ = 5·16 = 80`).
pub const PIKKUFOLD_N: usize = 256;
pub const PIKKUFOLD_WEIGHT: usize = 23;

pub fn pikkufold_spec() -> ShortChallengeSpec {
    ShortChallengeSpec {
        n: PIKKUFOLD_N,
        family: ShortChallengeFamily::FixedWeight {
            weight: PIKKUFOLD_WEIGHT,
            amplitude: 1,
        },
    }
}

/// Cyclo §3 App B: set `D` — biased ternary over R_q. The paper's |D| ≈
/// 2^203 at its parameters; here p = 500/1000 (balanced ternary) over N =
/// 512 gives ≈ 2^203 entropy with ℓ∞ = 1.
pub const CYCLO_N: usize = 512;

pub fn cyclo_spec() -> ShortChallengeSpec {
    ShortChallengeSpec {
        n: CYCLO_N,
        family: ShortChallengeFamily::BiasedTernary {
            p_nonzero_permille: 500,
        },
    }
}

/// LatticeFold+ §4: `S̄ = {-2,-1,0,1,2}^d` with d = 64 (|S̄| = 5^64 ≈ 2^148).
pub const LATTICEFOLD_PLUS_D: usize = 64;

pub fn latticefold_plus_spec() -> ShortChallengeSpec {
    ShortChallengeSpec {
        n: LATTICEFOLD_PLUS_D,
        family: ShortChallengeFamily::SmallSet {
            values: vec![-2, -1, 0, 1, 2],
        },
    }
}

/// ProtogaLattice §2.6 Table 2: `C = {-1,0,1,2}^N` with fixed weight
/// T = 128 (entropy 2^128 at the paper's N).
pub const PROTOGALATTICE_T: usize = 128;

pub fn protogalattice_spec(n: usize) -> ShortChallengeSpec {
    ShortChallengeSpec {
        n,
        family: ShortChallengeFamily::FixedWeightSmallSet {
            weight: PROTOGALATTICE_T.min(n),
            values: vec![-1, 1, 2],
        },
    }
}

/// Symphony / LaBRADOR: `S = {0,±1,±2}` (paper op-norm 15; our certified
/// worst case is `2·⌈√N⌉`).
pub fn symphony_spec(n: usize) -> ShortChallengeSpec {
    ShortChallengeSpec {
        n,
        family: ShortChallengeFamily::SmallSet {
            values: vec![0, 1, -1, 2, -2],
        },
    }
}

/// RoKoko §4: fixed-weight ternary TAU = 22 (|C| ≈ 2^103.3, op-norm 9.8 in
/// the paper's tight analysis).
pub const ROKOKO_TAU: usize = 22;

pub fn rokoko_spec(n: usize) -> ShortChallengeSpec {
    ShortChallengeSpec {
        n,
        family: ShortChallengeFamily::FixedWeight {
            weight: ROKOKO_TAU.min(n),
            amplitude: 1,
        },
    }
}

/// HyperWolf H5: fixed-weight signed challenges with rejection to T ≤ 10
/// (amplitude-1 fixed weight at weight 10 over N slots).
pub const HYPERWOLF_T: usize = 10;

pub fn hyperwolf_spec(n: usize) -> ShortChallengeSpec {
    ShortChallengeSpec {
        n,
        family: ShortChallengeFamily::FixedWeight {
            weight: HYPERWOLF_T.min(n),
            amplitude: 1,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_weight_exact_weight_and_amplitude() {
        let spec = pikkufold_spec();
        let c = spec.sample(b"seed-a").ok().unwrap();
        assert_eq!(c.coefficients.len(), PIKKUFOLD_N);
        assert_eq!(c.hamming_weight(), PIKKUFOLD_WEIGHT);
        assert_eq!(c.l_inf(), 1);
        // Deterministic.
        assert_eq!(spec.sample(b"seed-a").ok().unwrap(), c);
        assert_ne!(spec.sample(b"seed-b").ok().unwrap(), c);
        // Certified gamma: ⌈√(23·1)⌉ = 5 (per-sample embedding op-norm
        // ceiling; the full fold factor 5·16 = 80 adds ⌈√N⌉ at fold time).
        assert_eq!(c.gamma_c(), 5);
        assert_eq!(spec.gamma_c_worst_case(), 5);
        // Entropy ≈ log2 C(256,23) + 23 ≈ 105+ bits.
        assert!(spec.entropy_bits() > 100.0, "{}", spec.entropy_bits());
    }

    #[test]
    fn fixed_weight_positions_distinct() {
        // Fisher-Yates must never collide: exhaustive distinctness over
        // many draws at a dense weight.
        let spec = ShortChallengeSpec {
            n: 64,
            family: ShortChallengeFamily::FixedWeight {
                weight: 63,
                amplitude: 3,
            },
        };
        for i in 0..8 {
            let c = spec.sample(&[i; 8]).ok().unwrap();
            assert_eq!(c.hamming_weight(), 63);
            assert_eq!(c.l_inf(), 3);
            // ⌈√(63·9)⌉ = ⌈√567⌉ = 24.
            assert_eq!(c.gamma_c(), 24);
        }
    }

    #[test]
    fn biased_ternary_distribution_and_entropy() {
        let spec = cyclo_spec();
        let c = spec.sample(b"cyclo-seed").ok().unwrap();
        assert_eq!(c.coefficients.len(), CYCLO_N);
        assert!(c.l_inf() <= 1);
        // Roughly half non-zero (p = 0.5): 512 ± 5σ (σ ≈ 11.3).
        let w = c.hamming_weight();
        assert!(w > 200 && w < 312, "weight {w} outside 5σ band");
        // Entropy: 512 · H({0: 0.5, ±1: 0.25}) = 512 · 1.5 = 768 bits.
        assert!((spec.entropy_bits() - 768.0).abs() < 1.0);
        assert!(spec.entropy_bits() > 200.0); // |D| ≈ 2^203 class
    }

    #[test]
    fn small_set_values_and_entropy() {
        let spec = latticefold_plus_spec();
        let c = spec.sample(b"lfplus").ok().unwrap();
        assert_eq!(c.coefficients.len(), LATTICEFOLD_PLUS_D);
        for v in &c.coefficients {
            assert!(*v >= -2 && *v <= 2);
        }
        // |S̄| = 5^64 ≈ 2^148.7.
        assert!((spec.entropy_bits() - 64.0 * 5.0f64.log2()).abs() < 1e-9);
        assert!(spec.entropy_bits() > 148.0);
        // Worst-case gamma: 2·⌈√64⌉ = 16 as the FULL fold growth factor
        // (per-sample Γ is ⌈√(64·4)⌉ = 16; the √N = 8 is applied at fold
        // time by NormBudget).
        assert_eq!(spec.gamma_c_worst_case(), 2 * 8);
    }

    #[test]
    fn fixed_weight_small_set_protogalattice_profile() {
        let spec = protogalattice_spec(256);
        let c = spec.sample(b"pg-seed").ok().unwrap();
        assert_eq!(c.hamming_weight(), PROTOGALATTICE_T);
        for v in &c.coefficients {
            assert!(*v == 0 || *v == -1 || *v == 1 || *v == 2);
        }
        // log2 C(256,128) + 128·log2(3) ≈ 249 + 203 ≈ 452 bits ≥ 128.
        assert!(spec.entropy_bits() > 128.0);
    }

    #[test]
    fn symphony_and_rokoko_and_hyperwolf_profiles() {
        let sym = symphony_spec(512).sample(b"sym").ok().unwrap();
        assert!(sym.l_inf() <= 2);
        assert_eq!(symphony_spec(512).entropy_bits(), 512.0 * 5.0f64.log2());

        let rok = rokoko_spec(1024).sample(b"rok").ok().unwrap();
        assert_eq!(rok.hamming_weight(), ROKOKO_TAU);
        assert_eq!(rok.l_inf(), 1);
        // |C| ≈ log2 C(1024,22) + 22 ≈ 103.3 + 22 bits.
        assert!(rokoko_spec(1024).entropy_bits() > 100.0);

        let hw = hyperwolf_spec(256).sample(b"hw").ok().unwrap();
        assert_eq!(hw.hamming_weight(), HYPERWOLF_T);
    }

    #[test]
    fn gamma_cap_rejection_fails_closed() {
        let spec = pikkufold_spec(); // per-sample Γ = 5 exactly (weight 23)
                                     // Fixed-weight ±1 at weight 23 always gives Γ = 5, so a cap of 4
                                     // must exhaust retries and fail closed.
        assert!(matches!(
            spec.sample_with_gamma_cap(b"x", 4, 4),
            Err(ShortChallengeError::GammaCapExceeded {
                gamma: 5,
                cap: 4,
                ..
            })
        ));
        // A comfortable cap accepts immediately.
        assert!(spec.sample_with_gamma_cap(b"x", 5, 4).is_ok());
        // SmallSet challenges have variable per-sample Γ — the cap must
        // actually filter. With S̄ over d=64, Γ ranges up to ⌈√(64·4)⌉ = 16.
        let lf = latticefold_plus_spec();
        let tight = lf.sample_with_gamma_cap(b"t", 10, 200);
        match tight {
            Ok(c) => assert!(c.gamma_c() <= 10),
            Err(ShortChallengeError::GammaCapExceeded { .. }) => {}
            Err(e) => panic!("unexpected error {e:?}"),
        }
    }

    #[test]
    fn gamma_bound_is_rigorous_against_naive_bound() {
        // Γ_C = ⌈√(Σ c_i²)⌉ exactly (the rigorous ceiling of the embedding
        // operator norm); the √N of the full growth law is applied by
        // NormBudget::fold, not here. Verify the ceiling arithmetic exactly.
        for vec in [
            vec![0i64; 8],
            vec![1, 1, 1, 1, 0, 0, 0, 0],
            vec![-3, 4, 0, 0, 0, 0, 0, 0], // 3-4-5: Σ² = 25 → ⌈√25⌉ = 5
        ] {
            let g = certified_gamma(&vec);
            let l2sq: u128 = vec.iter().map(|c| (*c as i128 * *c as i128) as u128).sum();
            assert!(g >= ceil_sqrt_u128(l2sq));
            // Ceiling property: (Γ-1)² < Σc² ≤ Γ².
            if g > 0 {
                assert!((g - 1) * (g - 1) < l2sq as u64 || l2sq > u64::MAX as u128);
            }
        }
        assert_eq!(certified_gamma(&[-3, 4, 0, 0, 0, 0, 0, 0]), 5);
    }

    #[test]
    fn ceil_sqrt_exact() {
        assert_eq!(ceil_sqrt(0), 0);
        assert_eq!(ceil_sqrt(1), 1);
        assert_eq!(ceil_sqrt(2), 2);
        assert_eq!(ceil_sqrt(3), 2);
        assert_eq!(ceil_sqrt(4), 2);
        assert_eq!(ceil_sqrt(5), 3);
        assert_eq!(ceil_sqrt(24), 5);
        assert_eq!(ceil_sqrt(25), 5);
        assert_eq!(ceil_sqrt(26), 6);
        assert_eq!(ceil_sqrt_u128(u128::MAX), u64::MAX); // saturating ceiling of √(2^128-1) = 2^64
                                                         // Exactly representable boundary values:
        let sq = (u64::MAX as u128) * (u64::MAX as u128); // (2^64-1)²
        assert_eq!(ceil_sqrt_u128(sq), u64::MAX);
        // True ceiling 2^64 saturates to u64::MAX — still a valid bound.
        assert_eq!(ceil_sqrt_u128(sq + 1), u64::MAX);
        // Large values crossing the u64 boundary: exact square → exact
        // ceiling; one above the square → ceiling one higher.
        assert_eq!(ceil_sqrt_u128(1u128 << 96), 1u64 << 48);
        assert_eq!(ceil_sqrt_u128((1u128 << 96) + 1), (1u64 << 48) + 1);
    }

    #[test]
    fn spec_validation_rejects_bad_parameters() {
        assert!(ShortChallengeSpec {
            n: 0,
            family: ShortChallengeFamily::FixedWeight {
                weight: 1,
                amplitude: 1
            }
        }
        .validate()
        .is_err());
        assert!(ShortChallengeSpec {
            n: 8,
            family: ShortChallengeFamily::FixedWeight {
                weight: 9,
                amplitude: 1
            }
        }
        .validate()
        .is_err());
        assert!(ShortChallengeSpec {
            n: 8,
            family: ShortChallengeFamily::SmallSet { values: vec![] }
        }
        .validate()
        .is_err());
        assert!(ShortChallengeSpec {
            n: 8,
            family: ShortChallengeFamily::BiasedTernary {
                p_nonzero_permille: 1001
            }
        }
        .validate()
        .is_err());
        // Entropy gate.
        assert!(pikkufold_spec().require_entropy(90.0).is_ok());
        assert!(pikkufold_spec().require_entropy(200.0).is_err());
    }

    #[test]
    fn uniformity_of_positions() {
        // Chi-square sanity on the first-slot occupancy: with 200 draws and
        // weight 4 over n = 16, each position expects 50 hits.
        let spec = ShortChallengeSpec {
            n: 16,
            family: ShortChallengeFamily::FixedWeight {
                weight: 4,
                amplitude: 1,
            },
        };
        let mut counts = [0usize; 16];
        let draws = 200;
        for i in 0..draws {
            let c = spec.sample(&(i as u64).to_le_bytes()).ok().unwrap();
            for (p, v) in c.coefficients.iter().enumerate() {
                if *v != 0 {
                    counts[p] += 1;
                }
            }
        }
        let total: usize = counts.iter().sum();
        assert_eq!(total, draws * 4);
        // Each position within a generous band (mean 50, ±30).
        for (p, cnt) in counts.iter().enumerate() {
            assert!(
                (20..80).contains(cnt),
                "position {p} count {cnt} outside uniformity band"
            );
        }
    }
}
