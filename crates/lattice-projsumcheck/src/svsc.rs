//! **The small-value windowed projective sum-check** — the ss-class
//! weighting restructure over `Fp256` (Dao–DeStefano–Bagad–Domb–Thaler,
//! "Speeding Up Sum-Check Proving", ePrint 2025/1117 / 2026/587 §5 +
//! Appendix C.5, applied to the projective `{0,∞}` protocol of ePrint
//! 2026/762).
//!
//! ## The restructure (what changes and why it pays)
//!
//! The baseline prover binds every round with a full field challenge,
//! so from round 2 onward every message coefficient is a **bb**
//! (big×big) `Fp256` Montgomery multiplication. For a 4-limb Montgomery
//! field the measured bb:ss cost ratio is **κ ≈ 33** (§9's model:
//! `κ ≈ 2N² + N = 36` at `N = 4`) — one field multiplication buys ~33
//! small-small integer multiplications. The windowed prover spends that
//! budget where it counts:
//!
//! * **The window grid** — the first `v` rounds are answered from ONE
//!   integer-arithmetic pass: the window polynomial
//!   `q(X₁..X_v) = Σ_{x'∈{0,∞}^{ℓ−v}} P(X₁..X_v, x')` is materialized
//!   as its evaluations on the projective grid
//!   `U_d^v = {0, 1, …, d−1, ∞}^v` — every intermediate is a **u128
//!   integer** (ss multiplications and additions; the factors'
//!   coefficients are small machine words and the grid-point bindings
//!   `f(0) + u·f(∞)` multiply by small integers `u ≤ d−1`).
//! * **Round answering from the grid** (Appendix C.5.5): round `j`
//!   sums out the free window axes with `f(0) + f(∞)` (integer adds —
//!   the projective round identity), reads the message
//!   `[s_j(∞), s_j(1), …, s_j(d−1)]` straight off the `X_j` axis, and
//!   only then reduces mod `p`.
//! * **The challenge binding** — the sole bb work inside the window:
//!   collapsing the `X_j` axis to `r_j` costs `(d+1)^{v−j}`
//!   Lemma-2.2 interpolations per round (subdominant for the optimal
//!   `v`), and the challenges are sampled from the **upper-limb set**
//!   so each multiplication takes fp256's `mul_upper_limb` short-circuit
//!   (the 1.92× chained path) — the two optimizations stack.
//! * **Fail-closed bit-width precondition** — the paper's small-value
//!   validity condition: the integer grid and its axis collapses must
//!   fit `u128` before any reduction happens. The worst-case bound
//!   `2^{ℓ−v} · (#terms) · (d^v · 2^{κ_v})^d · c_max` is checked
//!   up-front and the instance is rejected otherwise.
//!
//! ## The protocol is untouched
//!
//! Byte-identical transcripts to the round-by-round reference prover
//! (`prove_reference`): same messages, same Fiat–Shamir challenges, same
//! terminal claims — the windowing is a prover-side algorithm change
//! only (the paper's "leave the protocol, verifier, and soundness
//! unchanged"). The equivalence is pinned by test.
//!
//! ## Scope
//!
//! Pure products (every term uses exactly `d` factors — the Shout /
//! read-RA instance shape). Mixed-degree instances stay on the
//! Goldilocks engine (`proj_sumcheck`); the mixed-degree message
//! requires the value at `d`, a `(d+2)`-point grid, and is left as the
//! documented extension. The streaming schedule (`Stream_k`, the
//! geometric window growth of Figure 2) is exposed through
//! [`WindowSchedule`] — the single-early-window `SV` setting is the
//! default here; the progressive schedule is the follow-up on the
//! `lattice-streaming` substrate.

use crate::fp256::Fp256;
use lattice_core::transcript::{Transcript, TranscriptError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SvscError {
    /// A coefficient exceeds the declared small-value bit bound.
    ValueTooLarge { value: u64, bits: u32 },
    /// The u128 bit-width precondition failed for this (d, v, ℓ, κ_v).
    BitWidthPrecondition { bits_needed: u32, window: usize },
    /// The claimed sum does not match the polynomial.
    ClaimMismatch,
    /// Round shape invalid.
    BadRoundShape { round: usize, got: usize, expected: usize },
    /// Terminal identity failed.
    FinalCheckFailed,
    Transcript(TranscriptError),
    /// The instance is empty or malformed.
    EmptyInstance,
    /// Mixed-degree instances are out of scope (see the module doc).
    MixedDegree { max: usize, min: usize },
}

impl From<TranscriptError> for SvscError {
    fn from(e: TranscriptError) -> Self {
        SvscError::Transcript(e)
    }
}

impl core::fmt::Display for SvscError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SvscError::ValueTooLarge { value, bits } => {
                write!(f, "coefficient {value} >= 2^{bits}")
            }
            SvscError::BitWidthPrecondition { bits_needed, window } => write!(
                f,
                "window {window} needs {bits_needed} integer bits > 128 — shrink the window, the degree, or the value bound"
            ),
            SvscError::ClaimMismatch => write!(f, "claimed sum does not match polynomial"),
            SvscError::BadRoundShape { round, got, expected } => {
                write!(f, "round {round} length {got} != {expected}")
            }
            SvscError::FinalCheckFailed => write!(f, "terminal identity failed"),
            SvscError::Transcript(e) => write!(f, "transcript: {e}"),
            SvscError::EmptyInstance => write!(f, "empty instance"),
            SvscError::MixedDegree { max, min } => {
                write!(f, "mixed-degree instance (max {max}, min {min}) — out of scope")
            }
        }
    }
}

/// A monomial-form multilinear factor with **small integer**
/// coefficients (`< 2^{value_bits}`) — the ss-class operand store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmallFactor {
    /// `2^num_vars` coefficients (the `{0,∞}` truth table — the monomial
    /// basis makes the two readings coincide).
    pub coeffs: Vec<u64>,
}

impl SmallFactor {
    pub fn num_vars(&self) -> usize {
        self.coeffs.len().trailing_zeros() as usize
    }

    /// Fail-closed small-value validation against the declared bound.
    pub fn validate(&self, bits: u32) -> Result<(), SvscError> {
        let limit = if bits >= 64 { u64::MAX } else { 1u64 << bits };
        for &c in &self.coeffs {
            if c >= limit {
                return Err(SvscError::ValueTooLarge { value: c, bits });
            }
        }
        Ok(())
    }

    /// Deterministic pseudorandom small-valued factor.
    pub fn random_small(num_vars: usize, bits: u32, seed: &[u8]) -> Self {
        let n = 1usize << num_vars;
        let mut coeffs = Vec::with_capacity(n);
        let mut ctr = 0u32;
        while coeffs.len() < n {
            let mut input = seed.to_vec();
            input.extend_from_slice(&ctr.to_le_bytes());
            let digest = Transcript::hash_domain(b"svsc-factor", &input);
            let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
            for chunk in digest.chunks(8) {
                if coeffs.len() >= n {
                    break;
                }
                let mut w = [0u8; 8];
                for (i, &b) in chunk.iter().enumerate() {
                    w[i] = b;
                }
                coeffs.push(u64::from_le_bytes(w) & mask);
            }
            ctr += 1;
        }
        SmallFactor { coeffs }
    }
}

/// A pure-product virtual polynomial with small coefficients:
/// `P(x) = Σ_j c_j · Π_{k ∈ term_j} f_k(x)` (every term the same arity).
#[derive(Clone, Debug)]
pub struct SvscInstance {
    pub num_vars: usize,
    pub factors: Vec<SmallFactor>,
    pub terms: Vec<(u64, Vec<usize>)>,
    /// The per-coefficient bit bound `κ_v`.
    pub value_bits: u32,
}

impl SvscInstance {
    /// A single `d`-way product (the canonical instance).
    pub fn product(factors: Vec<SmallFactor>, value_bits: u32) -> Result<Self, SvscError> {
        let num_vars = factors.first().map(|f| f.num_vars()).ok_or(SvscError::EmptyInstance)?;
        for f in &factors {
            if f.num_vars() != num_vars {
                return Err(SvscError::EmptyInstance);
            }
            f.validate(value_bits)?;
        }
        let ids: Vec<usize> = (0..factors.len()).collect();
        Ok(SvscInstance { num_vars, factors, terms: vec![(1, ids)], value_bits })
    }

    pub fn degree(&self) -> usize {
        self.terms.iter().map(|(_, ids)| ids.len()).max().unwrap_or(1)
    }

    /// The pure-product shape check.
    pub fn validate(&self) -> Result<usize, SvscError> {
        let d = self.degree();
        let min = self.terms.iter().map(|(_, ids)| ids.len()).min().unwrap_or(0);
        if min != d {
            return Err(SvscError::MixedDegree { max: d, min });
        }
        for f in &self.factors {
            if f.num_vars() != self.num_vars {
                return Err(SvscError::EmptyInstance);
            }
            f.validate(self.value_bits)?;
        }
        Ok(d)
    }

    /// The worst-case integer bit width of the windowed grid computation:
    /// every factor's window-bound value is at most `d^v · 2^{κ_v}`, the
    /// per-point term product at most `(d^v·2^{κ_v})^d · c_max`, and the
    /// suffix/axis accumulation adds `ℓ − v + (d+1)·v` bits.
    pub fn grid_bit_width(&self, window: usize) -> u32 {
        let d = self.degree() as u64;
        let v = window as u64;
        let c_max = self.terms.iter().map(|(c, _)| *c).max().unwrap_or(1);
        let dlog = u64::from(d.next_power_of_two().trailing_zeros()); // ceil(log2 d)
        let vb = u64::from(self.value_bits);
        // Each window binding scales a factor by ≤ d, so a factor's
        // window-bound value is ≤ d^v · 2^{κ_v}; a per-suffix term
        // product ≤ (d^v·2^{κ_v})^d · c_max; a grid entry sums
        // 2^{ℓ−v} of them; each axis collapse at most doubles the
        // stored value (v−1 collapses before the message reduction).
        let bits = (self.num_vars as u64 - v) // suffix count
            + (v - 1) // collapse doublings
            + d * (v * dlog + vb) // the product^d
            + u64::from(c_max.ilog2())
            + 4; // accumulator safety margin
        bits as u32
    }
}

// ---------------------------------------------------------------------------
// The proof + transcript plumbing.

/// The projective message: `[s_j(∞), s_j(1), …, s_j(d−1)]` per round —
/// the compressed form (the verifier derives `s_j(0) = C − s_j(∞)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SvscProof {
    pub rounds: Vec<Vec<Fp256>>,
}

/// Prover-side output: proof + terminal claims for the PCS layer.
#[derive(Clone, Debug)]
pub struct SvscOutput {
    pub proof: SvscProof,
    pub challenges: Vec<Fp256>,
    /// `P(r)` over the monomial basis.
    pub final_claim: Fp256,
    /// Per-factor claimed coefficient-form evaluations at `r`.
    pub factor_claims: Vec<Fp256>,
}

/// Verifier-side reduction result.
#[derive(Clone, Debug)]
pub struct SvscVerifier {
    pub point: Vec<Fp256>,
    pub final_claim: Fp256,
}

/// The window schedule (Figure 2): `SV` = one early window then the
/// linear-time tail (the small-value setting); the streaming variant
/// with geometric growth is the lattice-streaming follow-up.
#[derive(Clone, Debug)]
pub enum WindowSchedule {
    /// One window of `v` rounds, then round-by-round.
    Early { v: usize },
    /// No windowing — the reference path (also used past the window).
    None,
}

fn absorb_round(transcript: &mut Transcript, msg: &[Fp256]) -> Result<(), SvscError> {
    let mut bytes = Vec::with_capacity(msg.len() * 32);
    for m in msg {
        for limb in m.limbs {
            bytes.extend_from_slice(&limb.to_le_bytes());
        }
    }
    transcript.append_bytes(b"svsc-round", &bytes).map_err(SvscError::Transcript)
}

fn sample_challenge(transcript: &mut Transcript) -> Result<Fp256, SvscError> {
    let bytes = transcript
        .challenge_bytes(b"svsc-challenge", 32)
        .map_err(SvscError::Transcript)?;
    let mut h32 = [0u8; 32];
    h32.copy_from_slice(&bytes);
    Ok(Fp256::sample_upper_limb(&h32))
}

// ---------------------------------------------------------------------------
// Lemma 2.2 interpolation over Fp256.

/// `s(X) = s(∞)·Π_k (X−k) + Σ_k s(k)·L_k(X)` for the finite nodes
/// `0..d−1` (Lemma 2.2 of 2026/762) — the same evaluation the verifier
/// performs, over the 256-bit field.
fn interpolate_with_infinity_fp(
    finite: &[Fp256],
    leading: &Fp256,
    r: &Fp256,
    inv_small: &[Option<Fp256>],
) -> Fp256 {
    let d = finite.len();
    let mut lead_term = *leading;
    for k in 0..d {
        lead_term = lead_term.mul(&r.sub(&Fp256::from_canonical_u64(k as u64)));
    }
    let mut lag = Fp256::ZERO;
    for (k, &fk) in finite.iter().enumerate() {
        let xk = Fp256::from_canonical_u64(k as u64);
        let mut weight = Fp256::one_mont();
        for j in 0..d {
            if j == k {
                continue;
            }
            let xj = Fp256::from_canonical_u64(j as u64);
            let num = r.sub(&xj);
            // den = xk − xj = ±(k−j), a small integer.
            let diff = (k as i64 - j as i64).unsigned_abs() as usize;
            let den_inv = inv_small
                .get(diff)
                .and_then(|x| x.as_ref().copied());
            let den = match den_inv {
                Some(inv) => {
                    if k > j {
                        inv
                    } else {
                        inv.neg()
                    }
                }
                None => xk.sub(&xj).inverse().unwrap_or(Fp256::ZERO),
            };
            weight = weight.mul(&num.mul(&den));
        }
        lag = lag.add(&fk.mul(&weight));
    }
    lead_term.add(&lag)
}


/// `Σ_{suffix} Σ_j c_j · Π_k bound_k[bound-point values][suffix]` — the
/// round value at a given materialized bound-value slice per factor.
#[allow(clippy::needless_range_loop)] // indexes several slices in lockstep
fn sum_term_products(
    slices: &[&[Fp256]],
    terms: &[(u64, Vec<usize>)],
) -> Fp256 {
    let n = slices.first().map(|s| s.len()).unwrap_or(0);
    let mut acc = Fp256::ZERO;
    for idx in 0..n {
        let mut term_acc = Fp256::ZERO;
        for (c, ids) in terms {
            let mut prod = Fp256::from_canonical_u64(*c);
            for &fi in ids {
                prod = prod.mul(&slices[fi][idx]);
            }
            term_acc = term_acc.add(&prod);
        }
        acc = acc.add(&term_acc);
        let _ = idx;
    }
    acc
}

/// Materialize the bound values `lo + t·hi` per entry for one factor.
fn materialize_fp(lo: &[Fp256], hi: &[Fp256], t: u64, out: &mut Vec<Fp256>) {
    let tf = Fp256::from_canonical_u64(t);
    out.clear();
    out.extend(lo.iter().zip(hi.iter()).map(|(&l, &h)| l.add(&tf.mul(&h))));
}

/// The table of `1/i` for small `i ≤ d` (computed once per proof — the
/// interpolation denominators are tiny integers).
fn small_inverse_table(d: usize) -> Vec<Option<Fp256>> {
    let mut out = Vec::with_capacity(d + 2);
    out.push(None); // 1/0
    for i in 1..=d + 1 {
        out.push(Fp256::from_canonical_u64(i as u64).inverse());
    }
    out
}

// ---------------------------------------------------------------------------
// The reference (round-by-round) prover over Fp256.

/// The baseline: bind every round with a field challenge; every message
/// coefficient from round 2 on is a bb multiplication. Produces the
/// canonical transcript the windowed prover must match byte-for-byte.
pub fn prove_reference(
    inst: &SvscInstance,
    claim: &Fp256,
    transcript: &mut Transcript,
) -> Result<SvscOutput, SvscError> {
    let d = inst.validate()?;
    let inv_small = small_inverse_table(d);
    let mut bound: Vec<Vec<Fp256>> = inst
        .factors
        .iter()
        .map(|f| f.coeffs.iter().map(|&c| Fp256::from_canonical_u64(c)).collect())
        .collect();
    let mut current = *claim;
    let mut rounds = Vec::with_capacity(inst.num_vars);
    let mut challenges = Vec::with_capacity(inst.num_vars);

    for _round in 0..inst.num_vars {
        // The halves per factor (borrowed slices for the helper).
        let halves: Vec<(&[Fp256], &[Fp256])> = bound
            .iter()
            .map(|f| {
                let half = f.len() / 2;
                (&f[..half], &f[half..])
            })
            .collect();
        let lo_slices: Vec<&[Fp256]> = halves.iter().map(|(lo, _)| *lo).collect();
        let hi_slices: Vec<&[Fp256]> = halves.iter().map(|(_, hi)| *hi).collect();
        // s(∞): the second halves (the coefficients of X_round).
        let s_inf = sum_term_products(&hi_slices, &inst.terms);
        // s(0): the first halves.
        let s0 = sum_term_products(&lo_slices, &inst.terms);
        if s0.add(&s_inf) != current {
            return Err(SvscError::ClaimMismatch);
        }
        // The message: [s(∞), s(1), …, s(d−1)].
        let mut msg = Vec::with_capacity(d);
        msg.push(s_inf);
        for t in 1..d {
            let mut buffers: Vec<Vec<Fp256>> = bound
                .iter()
                .map(|f| vec![Fp256::ZERO; f.len() / 2])
                .collect();
            for (b, (lo, hi)) in buffers.iter_mut().zip(halves.iter()) {
                materialize_fp(lo, hi, t as u64, b);
            }
            let slices: Vec<&[Fp256]> = buffers.iter().map(|b| b.as_slice()).collect();
            msg.push(sum_term_products(&slices, &inst.terms));
        }
        absorb_round(transcript, &msg)?;
        let r = sample_challenge(transcript)?;
        challenges.push(r);

        // Lemma 2.2 evaluation at r: the finite nodes 0..d−1 (s(0)
        // derived) + the leading coefficient s(∞).
        let mut finite = vec![s0];
        finite.extend(msg.iter().skip(1).copied());
        current = interpolate_with_infinity_fp(&finite, &s_inf, &r, &inv_small);

        // Subtraction-free binding: f ← f(0) + r·f(∞).
        for f in bound.iter_mut() {
            let half = f.len() / 2;
            for j in 0..half {
                let lo = f[j];
                let hi = f[j + half];
                let prod = if r.is_upper_limb() {
                    hi.mul_upper_limb(&r)
                } else {
                    r.mul(&hi)
                };
                f[j] = lo.add(&prod);
            }
            f.truncate(half);
        }
        rounds.push(msg);
    }

    // Terminal: the factors are constants.
    let factor_claims: Vec<Fp256> = bound
        .iter()
        .map(|f| f.first().copied().unwrap_or(Fp256::ZERO))
        .collect();
    let mut final_claim = Fp256::ZERO;
    for (c, ids) in &inst.terms {
        let mut prod = Fp256::from_canonical_u64(*c);
        for &fi in ids {
            prod = prod.mul(&factor_claims[fi]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current {
        return Err(SvscError::FinalCheckFailed);
    }
    Ok(SvscOutput { proof: SvscProof { rounds }, challenges, final_claim, factor_claims })
}

// ---------------------------------------------------------------------------
// The windowed (SVSC) prover.

/// The projective grid `U_d = {0, 1, …, d−1, ∞}` — index `d` is `∞`.
fn grid_points(d: usize) -> Vec<Option<u64>> {
    let mut pts = (0..d as u64).map(Some).collect::<Vec<_>>();
    pts.push(None); // ∞
    pts
}

/// Bind one variable of an integer factor table to the grid point `u`:
/// `∞` selects the high half; `u` finite gives `lo + u·hi` (ss class).
fn bind_small(table: &[u64], u: Option<u64>) -> Vec<u64> {
    let half = table.len() / 2;
    match u {
        None => table[half..].to_vec(),
        Some(u) => table[..half]
            .iter()
            .zip(table[half..].iter())
            .map(|(&lo, &hi)| lo + u * hi)
            .collect(),
    }
}

/// The windowed prover: one integer-arithmetic grid pass answers the
/// first `v` rounds; the remaining rounds run the reference engine.
/// **The transcript is byte-identical to [`prove_reference`].**
pub fn prove_svsc(
    inst: &SvscInstance,
    claim: &Fp256,
    transcript: &mut Transcript,
    schedule: &WindowSchedule,
) -> Result<SvscOutput, SvscError> {
    let d = inst.validate()?;
    let v = match schedule {
        WindowSchedule::None => return prove_reference(inst, claim, transcript),
        WindowSchedule::Early { v } => *v.min(&inst.num_vars),
    };
    if v == 0 {
        return prove_reference(inst, claim, transcript);
    }
    // The fail-closed bit-width precondition.
    let needed = inst.grid_bit_width(v);
    if needed > 128 {
        return Err(SvscError::BitWidthPrecondition { bits_needed: needed, window: v });
    }
    let inv_small = small_inverse_table(d);
    let pts = grid_points(d);

    // ---- Phase 1: the integer grid of q over U_d^v ----------------------
    // DFS over the window variables, sharing prefixes: at depth i the
    // node holds a factor's 2^{ℓ−i}-entry suffix table. The leaf pass
    // accumulates the term products into the (d+1)^v grid (u128).
    let g = d + 1;
    let mut grid = vec![0u128; g.pow(v as u32)];

    // Per-factor bound tables at the current DFS depth.
    fn descend(
        inst: &SvscInstance,
        pts: &[Option<u64>],
        tables: &[Vec<u64>],
        grid: &mut [u128],
        digits: &mut Vec<usize>,
        g: usize,
        depth_target: usize,
    ) {
        if digits.len() == depth_target {
            // Leaf: the suffix tables are 2^{ℓ−v} entries each; accumulate
            // Σ_suffix Σ_term c·Π factor — all integer.
            // Forward fold: digits[0] (window variable 1, the first
            // round) is the MOST significant grid digit.
            let idx = digits.iter().fold(0usize, |acc, &dd| acc * g + dd);
            let suffix_len = tables.first().map(|t| t.len()).unwrap_or(0);
            #[allow(clippy::needless_range_loop)] // indexes several tables in lockstep
            for s in 0..suffix_len {
                let mut acc = 0u128;
                for (c, ids) in &inst.terms {
                    let mut prod = *c as u128;
                    for &fi in ids {
                        prod *= tables[fi][s] as u128;
                    }
                    acc += prod;
                }
                grid[idx] += acc;
            }
            return;
        }
        let depth = digits.len();
        for (pi, &u) in pts.iter().enumerate() {
            let next: Vec<Vec<u64>> = tables.iter().map(|t| bind_small(t, u)).collect();
            digits.push(pi);
            descend(inst, pts, &next, grid, digits, g, depth_target);
            digits.pop();
        }
        let _ = depth;
    }

    let initial: Vec<Vec<u64>> = inst.factors.iter().map(|f| f.coeffs.clone()).collect();
    let mut digits = Vec::with_capacity(v);
    descend(inst, &pts, &initial, &mut grid, &mut digits, g, v);

    // ---- The claimed sum: Σ over the {0,∞}^v sub-grid --------------------
    // (grid index d = ∞; index 0 = the point 0.) Enumerate the corners:
    // per window variable choose digit 0 or d.
    let mut corner_digits = vec![0usize; v];
    let mut sum_corners = 0u128;
    loop {
        let idx = corner_digits.iter().fold(0usize, |acc, &dd| acc * g + dd);
        sum_corners += grid[idx];
        // Increment with digits in {0, d}.
        let mut pos = v;
        loop {
            if pos == 0 {
                pos = usize::MAX;
                break;
            }
            pos -= 1;
            if corner_digits[pos] == 0 {
                corner_digits[pos] = d;
                break;
            }
            corner_digits[pos] = 0;
        }
        if pos == usize::MAX {
            break;
        }
    }
    let integer_claim = Fp256::from_canonical_u128(sum_corners);
    if integer_claim != *claim {
        return Err(SvscError::ClaimMismatch);
    }

    // ---- Phase 2: intra-window rounds from the grid ---------------------
    // The live grid holds q(r_1..r_{j-1}, X_j, ..., X_v) over
    // U_d^{v-j+1} with window variable j as the MOST significant axis
    // (C.5.5's two tracks: a scratch sum-out per round for the message,
    // the live grid bound at the challenge for the next round).
    let mut live: Vec<Fp256> = grid
        .iter()
        .map(|&x| Fp256::from_canonical_u128(x))
        .collect();
    let mut current = *claim;
    let mut challenges: Vec<Fp256> = Vec::with_capacity(inst.num_vars);
    let mut rounds: Vec<Vec<Fp256>> = Vec::with_capacity(inst.num_vars);

    for j in 0..v {
        // (a) The message: sum out axes j+1..v on a SCRATCH copy — each
        // collapse is f(0) + f(∞) over the last axis.
        let mut scratch = live.clone();
        let mut scratch_axes = v - j;
        while scratch_axes > 1 {
            let side = g; // (d+1) values on the last axis
            let mut next = Vec::with_capacity(scratch.len() / side);
            for o in 0..scratch.len() / side {
                // The last axis's digits live at o·g + {0..d}.
                next.push(scratch[o * side].add(&scratch[o * side + d]));
            }
            scratch = next;
            scratch_axes -= 1;
        }
        // scratch = (d+1) values of s_j over U_d: [0, 1, …, d−1, ∞].
        let s0 = scratch[0];
        let s_inf = scratch[d];
        if s0.add(&s_inf) != current {
            return Err(SvscError::ClaimMismatch);
        }
        let mut msg = Vec::with_capacity(d);
        msg.push(s_inf);
        msg.extend(scratch[1..d].iter().copied());
        absorb_round(transcript, &msg)?;
        let r = sample_challenge(transcript)?;
        challenges.push(r);

        // current = s_j(r_j) — Lemma 2.2 at the challenge (the same
        // evaluation the verifier performs).
        let mut finite = vec![s0];
        finite.extend(msg.iter().skip(1).copied());
        current = interpolate_with_infinity_fp(&finite, &s_inf, &r, &inv_small);

        // (b) Bind window variable j (the MOST significant axis): the
        // slice with the REMAINING axes fixed at `m` is
        // `[live[k·stride + m] for k in 0..g]` — a univariate over U_d —
        // interpolated at r_j.
        let stride = live.len() / g;
        let mut bound_grid = Vec::with_capacity(stride);
        for m in 0..stride {
            let slice: Vec<Fp256> = (0..g).map(|k| live[k * stride + m]).collect();
            let finite_slice: Vec<Fp256> = slice[..d].to_vec();
            bound_grid.push(interpolate_with_infinity_fp(
                &finite_slice,
                &slice[d],
                &r,
                &inv_small,
            ));
        }
        live = bound_grid;
        rounds.push(msg);
    }

    // ---- Phase 3: the post-window rounds (reference engine) -------------
    // Fast-forward the factors to (r_1..r_v): bind each window variable
    // with the field challenges (subtraction-free, upper-limb path).
    let mut bound: Vec<Vec<Fp256>> = inst
        .factors
        .iter()
        .map(|f| f.coeffs.iter().map(|&c| Fp256::from_canonical_u64(c)).collect())
        .collect();
    for &r in &challenges {
        for f in bound.iter_mut() {
            let half = f.len() / 2;
            for j in 0..half {
                let lo = f[j];
                let hi = f[j + half];
                let prod = if r.is_upper_limb() {
                    hi.mul_upper_limb(&r)
                } else {
                    r.mul(&hi)
                };
                f[j] = lo.add(&prod);
            }
            f.truncate(half);
        }
    }
    // The remaining rounds on the shrunk tables.
    for _round in v..inst.num_vars {
        let halves: Vec<(&[Fp256], &[Fp256])> = bound
            .iter()
            .map(|f| {
                let half = f.len() / 2;
                (&f[..half], &f[half..])
            })
            .collect();
        let lo_slices: Vec<&[Fp256]> = halves.iter().map(|(lo, _)| *lo).collect();
        let hi_slices: Vec<&[Fp256]> = halves.iter().map(|(_, hi)| *hi).collect();
        let s_inf = sum_term_products(&hi_slices, &inst.terms);
        let s0 = sum_term_products(&lo_slices, &inst.terms);
        if s0.add(&s_inf) != current {
            return Err(SvscError::ClaimMismatch);
        }
        let mut msg = Vec::with_capacity(d);
        msg.push(s_inf);
        for t in 1..d {
            let mut buffers: Vec<Vec<Fp256>> = bound
                .iter()
                .map(|f| vec![Fp256::ZERO; f.len() / 2])
                .collect();
            for (b, (lo, hi)) in buffers.iter_mut().zip(halves.iter()) {
                materialize_fp(lo, hi, t as u64, b);
            }
            let slices: Vec<&[Fp256]> = buffers.iter().map(|b| b.as_slice()).collect();
            msg.push(sum_term_products(&slices, &inst.terms));
        }
        absorb_round(transcript, &msg)?;
        let r = sample_challenge(transcript)?;
        challenges.push(r);
        let mut finite = vec![s0];
        finite.extend(msg.iter().skip(1).copied());
        current = interpolate_with_infinity_fp(&finite, &s_inf, &r, &inv_small);
        for f in bound.iter_mut() {
            let half = f.len() / 2;
            for j in 0..half {
                let lo = f[j];
                let hi = f[j + half];
                let prod = if r.is_upper_limb() {
                    hi.mul_upper_limb(&r)
                } else {
                    r.mul(&hi)
                };
                f[j] = lo.add(&prod);
            }
            f.truncate(half);
        }
        rounds.push(msg);
    }

    let factor_claims: Vec<Fp256> = bound
        .iter()
        .map(|f| f.first().copied().unwrap_or(Fp256::ZERO))
        .collect();
    let mut final_claim = Fp256::ZERO;
    for (c, ids) in &inst.terms {
        let mut prod = Fp256::from_canonical_u64(*c);
        for &fi in ids {
            prod = prod.mul(&factor_claims[fi]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current {
        return Err(SvscError::FinalCheckFailed);
    }
    Ok(SvscOutput { proof: SvscProof { rounds }, challenges, final_claim, factor_claims })
}

impl SvscProof {
    /// Verify against a claimed sum — the projective round identities
    /// with `s(0) = C − s(∞)` derived, Lemma 2.2 interpolation at each
    /// challenge. Returns the terminal binding for the PCS layer.
    pub fn verify(
        &self,
        claim: &Fp256,
        num_vars: usize,
        degree: usize,
        transcript: &mut Transcript,
    ) -> Result<SvscVerifier, SvscError> {
        if self.rounds.len() != num_vars {
            return Err(SvscError::BadRoundShape {
                round: usize::MAX,
                got: self.rounds.len(),
                expected: num_vars,
            });
        }
        let inv_small = small_inverse_table(degree);
        let mut current = *claim;
        let mut point = Vec::with_capacity(num_vars);
        for (i, round) in self.rounds.iter().enumerate() {
            if round.len() != degree || round.is_empty() {
                return Err(SvscError::BadRoundShape {
                    round: i,
                    got: round.len(),
                    expected: degree,
                });
            }
            absorb_round(transcript, round)?;
            let r = sample_challenge(transcript)?;
            let s_inf = round[0];
            let s0 = current.sub(&s_inf);
            let mut finite = vec![s0];
            finite.extend(round.iter().skip(1).copied());
            current = interpolate_with_infinity_fp(&finite, &s_inf, &r, &inv_small);
            point.push(r);
        }
        Ok(SvscVerifier { point, final_claim: current })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim_of(inst: &SvscInstance) -> Fp256 {
        // Σ_{x ∈ {0,∞}^ℓ} P(x) — integer arithmetic, reduced once.
        let mut accs = vec![0u128; 1usize << inst.num_vars];
        for (c, ids) in &inst.terms {
            for (idx, acc) in accs.iter_mut().enumerate() {
                let mut prod = *c as u128;
                for &fi in ids {
                    prod *= inst.factors[fi].coeffs[idx] as u128;
                }
                *acc += prod;
            }
        }
        let total: u128 = accs.iter().sum();
        Fp256::from_canonical_u128(total)
    }

    #[test]
    fn reference_prove_verify_roundtrip() {
        for &(d, ell, bits) in &[(2usize, 6usize, 33u32), (3, 5, 20), (2, 1, 8)] {
            let factors: Vec<SmallFactor> = (0..d)
                .map(|k| SmallFactor::random_small(ell, bits, format!("rf-{d}-{k}").as_bytes()))
                .collect();
            let inst = SvscInstance::product(factors, bits).ok().unwrap();
            let claim = claim_of(&inst);
            let mut t1 = Transcript::new_default(b"svsc-test");
            let out = prove_reference(&inst, &claim, &mut t1).map_err(|e| { println!("PROVE REF ERR {d} {ell}: {e:?}", ell = ell); e }).ok().unwrap();
            let mut t2 = Transcript::new_default(b"svsc-test");
            let v = out
                .proof
                .verify(&claim, inst.num_vars, d, &mut t2)
                .ok()
                .unwrap();
            assert_eq!(v.point, out.challenges);
            assert_eq!(v.final_claim, out.final_claim);
        }
    }

    #[test]
    fn windowed_matches_reference_bit_for_bit() {
        for &(d, ell, bits, v) in &[
            (2usize, 8usize, 33u32, 1usize),
            (2, 8, 33, 2),
            (2, 8, 33, 3),
            (3, 6, 16, 2),
            (2, 5, 33, 4),
        ] {
            let factors: Vec<SmallFactor> = (0..d)
                .map(|k| SmallFactor::random_small(ell, bits, format!("wf-{d}-{k}").as_bytes()))
                .collect();
            let inst = SvscInstance::product(factors, bits).ok().unwrap();
            let claim = claim_of(&inst);
            let mut t_ref = Transcript::new_default(b"svsc-test");
            let out_ref = prove_reference(&inst, &claim, &mut t_ref).ok().unwrap();
            let mut t_win = Transcript::new_default(b"svsc-test");
            let out_win = prove_svsc(
                &inst,
                &claim,
                &mut t_win,
                &WindowSchedule::Early { v },
            )
            .ok()
            .unwrap();
            assert_eq!(out_win.proof, out_ref.proof, "d={d} ell={ell} v={v}");
            assert_eq!(out_win.challenges, out_ref.challenges);
            assert_eq!(out_win.final_claim, out_ref.final_claim);
            assert_eq!(out_win.factor_claims, out_ref.factor_claims);
        }
    }

    #[test]
    fn bit_width_precondition_fails_closed() {
        // d=3, 33-bit values, a fat window: the integer grid exceeds u128
        // (3·(5·log2 3 + 33) + 7 + 4 > 128).
        let factors: Vec<SmallFactor> = (0..3)
            .map(|k| SmallFactor::random_small(8, 33, format!("bw-{k}").as_bytes()))
            .collect();
        let inst = SvscInstance::product(factors, 33).ok().unwrap();
        let mut t = Transcript::new_default(b"svsc-test");
        let claim = claim_of(&inst);
        let r = prove_svsc(&inst, &claim, &mut t, &WindowSchedule::Early { v: 5 });
        assert!(matches!(r, Err(SvscError::BitWidthPrecondition { .. })));
        // A window that fits passes the gate (and proves correctly).
        let mut t2 = Transcript::new_default(b"svsc-test");
        assert!(prove_svsc(&inst, &claim, &mut t2, &WindowSchedule::Early { v: 2 }).is_ok());
    }

    #[test]
    fn wrong_claim_rejected() {
        let factors: Vec<SmallFactor> = (0..2)
            .map(|k| SmallFactor::random_small(6, 33, format!("wc-{k}").as_bytes()))
            .collect();
        let inst = SvscInstance::product(factors, 33).ok().unwrap();
        let claim = Fp256::from_canonical_u64(12345);
        let mut t = Transcript::new_default(b"svsc-test");
        assert!(matches!(
            prove_reference(&inst, &claim, &mut t),
            Err(SvscError::ClaimMismatch)
        ));
    }

    #[test]
    fn tampered_round_rejected() {
        let factors: Vec<SmallFactor> = (0..2)
            .map(|k| SmallFactor::random_small(6, 33, format!("tr-{k}").as_bytes()))
            .collect();
        let inst = SvscInstance::product(factors, 33).ok().unwrap();
        let claim = claim_of(&inst);
        let mut t = Transcript::new_default(b"svsc-test");
        let mut out = prove_svsc(&inst, &claim, &mut t, &WindowSchedule::Early { v: 2 })
            .ok()
            .unwrap();
        out.proof.rounds[1][1] = out.proof.rounds[1][1].add(&Fp256::one_mont());
        let mut t2 = Transcript::new_default(b"svsc-test");
        // A tampered message derives a different terminal claim (the
        // caller's PCS check fires in practice).
        let v = out.proof.verify(&claim, inst.num_vars, 2, &mut t2).ok().unwrap();
        assert_ne!(v.final_claim, out.final_claim);
    }

    #[test]
    fn fp256_field_ops_are_consistent() {
        // sub/add/neg/inverse round trips on canonical values.
        let a = Fp256::from_canonical_u64(0xDEAD_BEEF_CAFE_F00D);
        let b = Fp256::from_canonical_u64(42);
        assert!(a.sub(&a).is_zero());
        assert_eq!(a.add(&a.neg()), Fp256::ZERO);
        let ainv = a.inverse().unwrap();
        assert_eq!(a.mul(&ainv), Fp256::one_mont());
        assert_eq!(
            a.sub(&b).add(&b),
            a
        );
        let big = Fp256::from_canonical_u128(u128::MAX - 7);
        assert_eq!(big.add(&big.neg()), Fp256::ZERO);
        let binv = big.inverse().unwrap();
        assert_eq!(big.mul(&binv), Fp256::one_mont());
    }
}

#[cfg(test)]
mod interp_check {
    use super::*;
    #[test]
    fn lemma22_known_values() {
        // s(X) = 5 + 3X + 7X^2: nodes s(0)=5, s(1)=15, leading 7.
        let inv = small_inverse_table(2);
        let finite = [Fp256::from_canonical_u64(5), Fp256::from_canonical_u64(15)];
        let lead = Fp256::from_canonical_u64(7);
        for (r, want) in [(2u64, 39u64), (3, 77), (5, 195)] {
            let got = interpolate_with_infinity_fp(&finite, &lead, &Fp256::from_canonical_u64(r), &inv);
            // Montgomery form of `want`.
            assert_eq!(got, Fp256::from_canonical_u64(want), "r={r}");
        }
        // degree 1: s(X) = 5 + 3X: node s(0)=5, leading 3.
        let inv1 = small_inverse_table(1);
        let f1 = [Fp256::from_canonical_u64(5)];
        let l1 = Fp256::from_canonical_u64(3);
        let got = interpolate_with_infinity_fp(&f1, &l1, &Fp256::from_canonical_u64(4), &inv1);
        assert_eq!(got, Fp256::from_canonical_u64(17));
    }
}

