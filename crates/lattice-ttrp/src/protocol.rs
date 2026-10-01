//! The `Π_TTRP` protocol: an approximate-range reduction of knowledge for
//! Ajtai-commitment polynomial relations (ePrint 2026/2146, Figure 1 +
//! Corollary 1).
//!
//! # Protocol (Figure 1, with the k′ claims γ-batched per §4.2)
//!
//! Public: TT parameters `(φ, ν, ℓ, µ₁, µ₂, c, k, k′)`, norm bound `B`,
//! and a statement digest binding the outer relation (the Ajtai
//! commitment context `T = A·W` in the full system).
//!
//! 1. **Cores.** Both parties derive `tM_i^{(j)}u ← χ_TT(k, m̄rφ, d, µ, c)`
//!    from the transcript (attempt counter first — the honest prover
//!    retries with fresh cores until the completeness check passes,
//!    which holds with probability ≥ 1/2 by Markov).
//! 2. **Projection.** The prover sends `y0 = ct(M·v̄) = M_Z·cf(v) mod q ∈
//!    Z_q^k`; the verifier checks `||y0||₂ ≤ B̂ = √(k(c/2)^{µ−1})·B`.
//! 3. **Aggregation.** The verifier samples `Γ ∈ Z_q^{k′×k}`; the prover
//!    responds `y1 = Γ·y ∈ R^{k′}` (with `y = M·v̄`); the verifier checks
//!    the constant-term identity `ct(y1_i) = Σ_j Γ_{ij}·y0_j`.
//! 4. **Linearisation.** The verifier samples `γ ∈ R^{k′}`; setting
//!    `γ̃ = γ·Γ ∈ R^k`, `y* = Σ_i γ_i·y1_i`, the single batched claim is
//!    `Σ_{z∈{0,1}^ν} mle(m*)(z)·mle(v̄)(z) = y*` with
//!    `m* = Σ_j γ̃_j·M[j] ∈ R^{m̄r}`.
//! 5. **Sumcheck.** ν rounds over `g(z) = mle(m*)(z)·mle(v̄)(z) −
//!    y*·2^{−ν}` (individual degree 2): round messages are the three
//!    evaluations at `X ∈ {0,1,2}`, challenges are uniform ring elements.
//! 6. **Terminal.** The prover sends `w_r = mle(v)(conj(r))` (it equals
//!    `conj` of its final bound array). The verifier evaluates
//!    `mle(m*)(r)` **locally by tensor contraction** — spatial cores
//!    MLE-evaluated at challenge chunks chained into a 1×c boundary
//!    vector, times the γ̃-scaled coefficient-core chain (Lemma 6) — and
//!    checks `C_ν = mle(m*)(r)·conj(w_r) − y*·2^{−ν}`.
//!
//! The output evaluation claim `(conj(r), w_r)` is appended to the outer
//! `Ξ_poly` relation by the caller (the claim-ledger pattern).
//!
//! # Documented deviations (the honest ledger)
//!
//! * **Challenge space.** The paper's Lemma 1 assumes a challenge set with
//!   pairwise-invertible differences and error `ℓ·deg/|C|`; over the
//!   power-of-two cyclotomic `R_q` (which splits completely for NTT
//!   primes) the honest per-round bound for uniform ring challenges is
//!   `deg·φ/q`-shaped via the CRT argument, and this crate follows the
//!   workspace's 32-bit-q interactive posture (the same documented
//!   deviation as `lattice-salsa/ring_sc`): soundness amplification,
//!   grinding, or a larger modulus belong to the outer composition.
//! * **k′-row collision term.** The knowledge error carries `q^{−k′}`
//!   from the Γ-aggregation (Corollary 1's `q^{−e}` term); k′ is a
//!   caller parameter for that reason.
//! * **Completeness error 1/2.** Retries re-derive cores from a public
//!   attempt counter (absorbed before the core seed), so the verifier
//!   replays exactly the accepted attempt.

use crate::cores::{sample_cores, CoreTensor, TtrpParams};
use crate::projection::{
    cf_vec, centered, coefficient_chain, conj, ct, spatial_matrix,
};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_ring::{RingConfig, RingElement, RingError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtrpError {
    Params(crate::cores::TtrpCoreError),
    Ring(RingError),
    Transcript(TranscriptError),
    /// Witness length != m̄r.
    WitnessLength { expected: usize, got: usize },
    /// Completeness retries exhausted (probability <= 2^-attempts).
    CompletenessRetries,
    /// Verifier-side norm check failed.
    NormCheckFailed,
    /// Verifier-side constant-term identity failed for row i.
    ConstantTermCheck { row: usize },
    /// Sumcheck round identity failed.
    RoundCheck { round: usize },
    /// Terminal identity failed.
    TerminalCheck,
}

impl core::fmt::Display for TtrpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TtrpError::Params(e) => write!(f, "ttrp params: {e}"),
            TtrpError::Ring(e) => write!(f, "ttrp ring: {e:?}"),
            TtrpError::Transcript(e) => write!(f, "ttrp transcript: {e:?}"),
            TtrpError::WitnessLength { expected, got } => {
                write!(f, "ttrp witness length {got} != {expected}")
            }
            TtrpError::CompletenessRetries => write!(f, "ttrp completeness retries exhausted"),
            TtrpError::NormCheckFailed => write!(f, "ttrp norm check failed"),
            TtrpError::ConstantTermCheck { row } => {
                write!(f, "ttrp constant-term identity failed at row {row}")
            }
            TtrpError::RoundCheck { round } => {
                write!(f, "ttrp sumcheck round {round} failed")
            }
            TtrpError::TerminalCheck => write!(f, "ttrp terminal check failed"),
        }
    }
}

impl From<crate::cores::TtrpCoreError> for TtrpError {
    fn from(e: crate::cores::TtrpCoreError) -> Self {
        TtrpError::Params(e)
    }
}
impl From<RingError> for TtrpError {
    fn from(e: RingError) -> Self {
        TtrpError::Ring(e)
    }
}
impl From<TranscriptError> for TtrpError {
    fn from(e: TranscriptError) -> Self {
        TtrpError::Transcript(e)
    }
}

/// The public statement: parameters + norm bound + outer-context digest.
pub struct TtrpStatement {
    pub params: TtrpParams,
    pub ring: RingConfig,
    /// Claimed Euclidean bound B on the witness coefficients
    /// (`||cf(v)||₂ ≤ B`).
    pub bound_b: u64,
    /// Binds the outer relation (Ajtai commitment context / claim ledger).
    pub statement_digest: [u8; 32],
}

impl TtrpStatement {
    /// `B̂² = k·c^{µ−1}·B²/2^µ` in u128 (the completeness-side bound the
    /// verifier checks `||y0||₂²` against — Theorem 4 with the corrected
    /// second moment of Lemma 3; see bounds::eta2).
    pub fn b_hat_squared(&self) -> u128 {
        let mu = self.params.mu() as u32;
        let numer = (self.params.k as u128)
            .saturating_mul(
                (self.params.c as u128)
                    .checked_pow(mu.saturating_sub(1))
                    .unwrap_or(u128::MAX),
            )
            .saturating_mul((self.bound_b as u128).pow(2));
        // ceil(numer / 2^mu) — conservative for the honest prover.
        numer
            .saturating_add((1u128 << mu.min(127)).saturating_sub(1))
            >> mu.min(127)
    }
}

/// The proof: y0, y1, the sumcheck rounds, and the terminal evaluation.
#[derive(Clone)]
pub struct TtrpProof {
    /// Completeness attempt counter (absorbed before core derivation).
    pub attempt: u32,
    /// The projection `y0 ∈ Z_q^k` (canonical residues).
    pub y0: Vec<u32>,
    /// `y1 = Γ·y ∈ R^{k′}`.
    pub y1: Vec<RingElement>,
    /// Round messages: `[g(0), g(1), g(2)] ∈ R³` per round.
    pub rounds: Vec<[RingElement; 3]>,
    /// `w_r = mle(v)(conj(r))`.
    pub w_r: RingElement,
    /// The sumcheck challenge point `r ∈ R^ν` (the eval-claim point for
    /// the caller's outer relation: `mle(v)(conj(r)) = w_r`).
    pub challenges: Vec<RingElement>,
}

/// Verifier output: the evaluation claim to append to the outer relation.
#[derive(Clone)]
pub struct TtrpVerified {
    /// The sumcheck challenge point `r ∈ R^ν`.
    pub challenges: Vec<RingElement>,
    /// The claimed `mle(v)(conj(r))` — the caller authenticates it against
    /// its witness-opening layer.
    pub w_r: RingElement,
}

// ---------------------------------------------------------------------------
// Transcript sampling helpers (unbiased, bounded-rejection discipline)
// ---------------------------------------------------------------------------

fn absorb_statement(transcript: &mut Transcript, stmt: &TtrpStatement) -> Result<(), TtrpError> {
    transcript.append_message(b"ttrp-stmt", &stmt.statement_digest)?;
    let p = &stmt.params;
    let meta: Vec<u64> = vec![
        stmt.ring.modulus.q as u64,
        stmt.ring.log_n as u64,
        stmt.bound_b,
        p.nu as u64,
        p.ell as u64,
        p.mu1 as u64,
        p.mu2 as u64,
        p.c as u64,
        p.k as u64,
        p.k1 as u64,
    ];
    let bytes: Vec<u8> = meta.iter().flat_map(|v| v.to_le_bytes()).collect();
    transcript.append_message(b"ttrp-params", &bytes)?;
    Ok(())
}

/// Unbiased `Z_q` draw from the transcript (u32 rejection — 2^32 mod q ≠ 0).
fn challenge_zq_u32(transcript: &mut Transcript, q: u32, label: &[u8]) -> Result<u32, TtrpError> {
    for _ in 0..16 {
        let bytes = transcript.challenge_bytes(label, 4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(&bytes[..4]);
        let v = u32::from_le_bytes(arr);
        if v < q {
            return Ok(v);
        }
    }
    Err(TtrpError::Transcript(TranscriptError::RejectionBudgetExceeded))
}

/// Uniform ring element (unbiased per coefficient).
fn challenge_ring(transcript: &mut Transcript, ring: &RingConfig) -> Result<RingElement, TtrpError> {
    let phi = ring.n();
    let q = ring.modulus.q;
    let mut coeffs = Vec::with_capacity(phi);
    let mut pending = Vec::new();
    for _ in 0..phi {
        loop {
            if pending.is_empty() {
                let need = 4 * 64;
                pending = transcript.challenge_bytes(b"ttrp-chal", need)?;
            }
            let (chunk, rest) = pending.split_at(4);
            let mut arr = [0u8; 4];
            arr.copy_from_slice(chunk);
            pending = rest.to_vec();
            let v = u32::from_le_bytes(arr);
            if v < q {
                coeffs.push(v);
                break;
            }
        }
    }
    Ok(RingElement::from_coeffs(ring, coeffs))
}

fn core_seed_from(transcript: &mut Transcript, attempt: u32) -> Result<[u8; 32], TtrpError> {
    transcript.append_message(b"ttrp-attempt", &attempt.to_le_bytes())?;
    let bytes = transcript.challenge_bytes(b"ttrp-core-seed", 32)?;
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&bytes[..32]);
    Ok(seed)
}

fn absorb_y0(transcript: &mut Transcript, y0: &[u32]) -> Result<(), TtrpError> {
    let bytes: Vec<u8> = y0.iter().flat_map(|v| v.to_le_bytes()).collect();
    transcript.append_message(b"ttrp-y0", &bytes)?;
    Ok(())
}

fn absorb_ring(transcript: &mut Transcript, elt: &RingElement) -> Result<(), TtrpError> {
    transcript.append_message(b"ttrp-ring", &elt.to_bytes())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Ring array helpers
// ---------------------------------------------------------------------------

/// Bind the first variable of a ring array to the scalar t ∈ {0, 1, 2}:
/// `out[z'] = lo[z'] + t·(hi[z'] − lo[z'])`.
fn bind_scalar(
    ring: &RingConfig,
    arr: &[RingElement],
    t: u32,
) -> Result<Vec<RingElement>, TtrpError> {
    let half = arr.len() / 2;
    let mut out = Vec::with_capacity(half);
    for z in 0..half {
        let lo = &arr[z];
        let hi = &arr[half + z];
        match t {
            0 => out.push(lo.clone()),
            1 => out.push(hi.clone()),
            _ => {
                let delta = hi.sub(lo)?;
                let scaled = delta.scale_i64(t as i64);
                out.push(lo.add(&scaled)?);
            }
        }
    }
    let _ = ring;
    Ok(out)
}

/// Bind the first variable to a ring challenge:
/// `out[z'] = lo[z'] + r·(hi[z'] − lo[z'])`.
fn bind_ring(
    arr: &[RingElement],
    r: &RingElement,
) -> Result<Vec<RingElement>, TtrpError> {
    let half = arr.len() / 2;
    let mut out = Vec::with_capacity(half);
    for z in 0..half {
        let delta = arr[half + z].sub(&arr[z])?;
        let scaled = r.mul(&delta)?;
        out.push(arr[z].add(&scaled)?);
    }
    Ok(out)
}

/// Ring dot product `Σ_i a_i·b_i`.
fn dot_ring(a: &[RingElement], b: &[RingElement]) -> Result<RingElement, TtrpError> {
    let ring = a[0].config();
    let mut acc = ring.zero();
    for (x, y) in a.iter().zip(b.iter()) {
        let p = x.mul(y)?;
        acc = acc.add(&p)?;
    }
    Ok(acc)
}

/// MLE of a ring array at a ring point r ∈ R^ν (successive binding).
pub fn mle_eval_ring(arr: &[RingElement], r: &[RingElement]) -> Result<RingElement, TtrpError> {
    debug_assert_eq!(arr.len(), 1usize << r.len());
    let mut cur = arr.to_vec();
    for challenge in r {
        cur = bind_ring(&cur, challenge)?;
    }
    debug_assert_eq!(cur.len(), 1);
    Ok(cur[0].clone())
}

// ---------------------------------------------------------------------------
// The prover
// ---------------------------------------------------------------------------

/// Prove the TTRP statement for witness `v ∈ R^{m̄r}` (with
/// `||cf(v)||₂ ≤ bound_b` for completeness).
///
/// The transcript must already carry any outer statement material; the
/// TTRP statement digest is absorbed first.
pub fn prove(
    stmt: &TtrpStatement,
    v: &[RingElement],
    transcript: &mut Transcript,
) -> Result<TtrpProof, TtrpError> {
    stmt.params.validate()?;
    let params = &stmt.params;
    let ring = &stmt.ring;
    if v.len() != params.m_bar() {
        return Err(TtrpError::WitnessLength {
            expected: params.m_bar(),
            got: v.len(),
        });
    }
    absorb_statement(transcript, stmt)?;

    // Precompute the conjugated witness (reused across attempts).
    let v_bar: Vec<RingElement> = v.iter().map(conj).collect();
    let x = cf_vec(v);

    let mut attempt: u32 = 0;
    loop {
        if attempt >= 64 {
            return Err(TtrpError::CompletenessRetries);
        }
        // Derive fresh cores for this attempt from a scratch transcript
        // state (the real transcript advances only on the accepted attempt).
        let mut scratch = transcript.clone();
        let seed = core_seed_from(&mut scratch, attempt)?;
        let rows = sample_cores(params, &seed);

        // Integer projection y0 = M_Z · x mod q.
        let y0_centered = crate::projection::project_integer(params, &rows, ring.modulus.q, &x);
        // Completeness gate: ||y0||² <= B̂² (Markov: passes w.p. >= 1/2).
        let norm_sq: u128 = y0_centered
            .iter()
            .map(|&c| (c as i128 * c as i128) as u128)
            .sum();
        if norm_sq > stmt.b_hat_squared() {
            attempt += 1;
            continue;
        }

        // Ring projection y = M · v̄.
        let y = crate::projection::project_ring(params, &rows, ring, &v_bar)?;

        // Accepted attempt: commit to it in the real transcript.
        let accepted_seed = core_seed_from(transcript, attempt)?;
        debug_assert_eq!(accepted_seed, seed);
        let y0: Vec<u32> = y0_centered
            .iter()
            .map(|&c| (c.rem_euclid(ring.modulus.q as i64)) as u32)
            .collect();
        absorb_y0(transcript, &y0)?;

        // Γ ∈ Z_q^{k'×k}; y1 = Γ·y ∈ R^{k'}.
        let mut gamma_mat = Vec::with_capacity(params.k1);
        for _ in 0..params.k1 {
            let mut row = Vec::with_capacity(params.k);
            for _ in 0..params.k {
                row.push(challenge_zq_u32(transcript, ring.modulus.q, b"ttrp-gamma")?);
            }
            gamma_mat.push(row);
        }
        let mut y1 = Vec::with_capacity(params.k1);
        for g_row in &gamma_mat {
            let mut acc = ring.zero();
            for (j, &g) in g_row.iter().enumerate() {
                let term = y[j].scale_i64(g as i64);
                acc = acc.add(&term)?;
            }
            y1.push(acc);
        }
        for elt in &y1 {
            absorb_ring(transcript, elt)?;
        }

        // γ ∈ R^{k'}; γ̃ = γ·Γ ∈ R^k; y* = Σ γ_i·y1_i.
        let mut gammas = Vec::with_capacity(params.k1);
        for _ in 0..params.k1 {
            gammas.push(challenge_ring(transcript, ring)?);
        }
        let mut gamma_tilde = vec![ring.zero(); params.k];
        for (i, g) in gammas.iter().enumerate() {
            for (j, &gij) in gamma_mat[i].iter().enumerate() {
                let term = g.scale_i64(gij as i64);
                gamma_tilde[j] = gamma_tilde[j].add(&term)?;
            }
        }
        let mut y_star = ring.zero();
        for (g, elt) in gammas.iter().zip(y1.iter()) {
            let term = g.mul(elt)?;
            y_star = y_star.add(&term)?;
        }

        // m* = Σ_j γ̃_j·M[j] via the S/W split with γ̃ folded into the
        // coefficient chains: m*[h] = Σ_j <S^{(j)}[h], γ̃_j·W^{(j)}>.
        let m_star = build_m_star(params, &rows, ring, &gamma_tilde, &v_bar)?;

        // The batched sumcheck over g(z) = mle(m*)(z)·mle(v̄)(z) − y*·2^{−ν}.
        let trace = sumcheck_prove(params, ring, &m_star, &v_bar, &y_star, transcript)?;
        let SumcheckTrace {
            rounds,
            challenges,
            b_final,
        } = trace;

        // w_r = conj(mle(v̄)(r)) = mle(v)(conj(r)).
        let w_r = conj(&b_final);

        return Ok(TtrpProof {
            attempt,
            y0,
            y1,
            rounds,
            w_r,
            challenges,
        });
    }
}

/// Build the aggregated row `m* ∈ R^{m̄r}`:
/// `m*[h] = Σ_j ⟨S^{(j)}[h], γ̃_j·W^{(j)}⟩` — integer coefficient
/// accumulation plus k·c ring multiplications (the O(k·c·m̄r·φ) path).
#[allow(clippy::needless_range_loop)]
pub fn build_m_star(
    params: &TtrpParams,
    rows: &[Vec<CoreTensor>],
    ring: &RingConfig,
    gamma_tilde: &[RingElement],
    v_bar: &[RingElement],
) -> Result<Vec<RingElement>, TtrpError> {
    let _ = v_bar; // (not needed — kept for signature symmetry)
    let phi = ring.n();
    let q = ring.modulus.q as i64;
    let c = params.c;
    let m_bar = params.m_bar();
    // Per row: scaled coefficient chain W̃^{(j)} = γ̃_j · W^{(j)} ∈ R^c.
    let mut scaled_chains = Vec::with_capacity(params.k);
    for (j, cores) in rows.iter().enumerate() {
        let w = coefficient_chain(params, cores, ring);
        let mut scaled = Vec::with_capacity(c);
        for wi in &w {
            scaled.push(gamma_tilde[j].mul(wi)?);
        }
        scaled_chains.push((spatial_matrix(params, cores), scaled));
    }
    // Coefficient-level accumulation of m* — deferred reduction: raw i128
    // accumulation over all rows, one modular pass at the end (the per-row
    // intermediate is bounded by k·c·max|S|·q, far below 2^127).
    let mut m_coeffs: Vec<Vec<i128>> = vec![vec![0i128; phi]; m_bar];
    for (s_mat, w_tilde) in &scaled_chains {
        for h in 0..m_bar {
            let base = h * c;
            let dst = &mut m_coeffs[h];
            for i in 0..c {
                let sv = s_mat[base + i];
                if sv == 0 {
                    continue;
                }
                let wc = w_tilde[i].coeffs();
                for l in 0..phi {
                    dst[l] += sv as i128 * wc[l] as i128;
                }
            }
        }
    }
    Ok(m_coeffs
        .into_iter()
        .map(|coeffs| {
            let red: Vec<u32> = coeffs
                .iter()
                .map(|&c| (c.rem_euclid(q as i128)) as u32)
                .collect();
            RingElement::from_coeffs(ring, red)
        })
        .collect())
}

/// The sumcheck trace: round messages, challenges, final bound b.
struct SumcheckTrace {
    rounds: Vec<[RingElement; 3]>,
    challenges: Vec<RingElement>,
    b_final: RingElement,
}

/// The ν-round degree-2 ring sumcheck on
/// `g(z) = mle(a)(z)·mle(b)(z) − y*·2^{−ν}`.
///
/// Returns (round messages, challenges, the final bound value of b).
#[allow(clippy::needless_range_loop)]
fn sumcheck_prove(
    params: &TtrpParams,
    ring: &RingConfig,
    a: &[RingElement],
    b: &[RingElement],
    y_star: &RingElement,
    transcript: &mut Transcript,
) -> Result<SumcheckTrace, TtrpError> {
    let nu = params.nu;
    let q = ring.modulus.q as u64;
    let inv2 = q.div_ceil(2) as i64; // 2^{-1} mod q (q odd)
    let mut a_arr = a.to_vec();
    let mut b_arr = b.to_vec();
    let mut rounds = Vec::with_capacity(nu);
    let mut challenges = Vec::with_capacity(nu);
    let mut inv2_pow = 1i64;
    for _ in 0..nu {
        inv2_pow = inv2_pow * inv2 % q as i64;
        let mut msg = [ring.zero(), ring.zero(), ring.zero()];
        for (t, slot) in [0u32, 1, 2].iter().zip(msg.iter_mut()) {
            let a_t = bind_scalar(ring, &a_arr, *t)?;
            let b_t = bind_scalar(ring, &b_arr, *t)?;
            let prod = dot_ring(&a_t, &b_t)?;
            let const_term = y_star.scale_i64(inv2_pow);
            *slot = prod.sub(&const_term)?;
        }
        for elt in &msg {
            absorb_ring(transcript, elt)?;
        }
        let r = challenge_ring(transcript, ring)?;
        a_arr = bind_ring(&a_arr, &r)?;
        b_arr = bind_ring(&b_arr, &r)?;
        rounds.push(msg);
        challenges.push(r);
    }
    debug_assert_eq!(a_arr.len(), 1);
    debug_assert_eq!(b_arr.len(), 1);
    Ok(SumcheckTrace {
        rounds,
        challenges,
        b_final: b_arr[0].clone(),
    })
}

// ---------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------

/// Verify a TTRP proof. On success returns the evaluation claim
/// `(r, w_r)` for the caller's outer relation.
pub fn verify(
    stmt: &TtrpStatement,
    proof: &TtrpProof,
    transcript: &mut Transcript,
) -> Result<TtrpVerified, TtrpError> {
    stmt.params.validate()?;
    let params = &stmt.params;
    let ring = &stmt.ring;
    if proof.y0.len() != params.k || proof.y1.len() != params.k1 {
        return Err(TtrpError::WitnessLength {
            expected: params.k,
            got: proof.y0.len(),
        });
    }
    absorb_statement(transcript, stmt)?;

    // Regenerate the cores from the attempt counter.
    let seed = core_seed_from(transcript, proof.attempt)?;
    let rows = sample_cores(params, &seed);

    // Norm check: ||y0||² <= B̂².
    let norm_sq: u128 = proof
        .y0
        .iter()
        .map(|&c| {
            let cc = centered(c, ring.modulus.q);
            (cc as i128 * cc as i128) as u128
        })
        .sum();
    if norm_sq > stmt.b_hat_squared() {
        return Err(TtrpError::NormCheckFailed);
    }
    absorb_y0(transcript, &proof.y0)?;

    // Γ and the constant-term identity ct(y1_i) = Σ_j Γ_ij·y0_j.
    let mut gamma_mat = Vec::with_capacity(params.k1);
    for _ in 0..params.k1 {
        let mut row = Vec::with_capacity(params.k);
        for _ in 0..params.k {
            row.push(challenge_zq_u32(transcript, ring.modulus.q, b"ttrp-gamma")?);
        }
        gamma_mat.push(row);
    }
    for (i, elt) in proof.y1.iter().enumerate() {
        let mut acc = 0u64;
        for (j, &g) in gamma_mat[i].iter().enumerate() {
            acc += (g as u64 * proof.y0[j] as u64) % ring.modulus.q as u64;
        }
        if ct(elt) as u64 != acc % ring.modulus.q as u64 {
            return Err(TtrpError::ConstantTermCheck { row: i });
        }
        absorb_ring(transcript, elt)?;
    }

    // γ, γ̃, y*.
    let mut gammas = Vec::with_capacity(params.k1);
    for _ in 0..params.k1 {
        gammas.push(challenge_ring(transcript, ring)?);
    }
    let mut gamma_tilde = vec![ring.zero(); params.k];
    for (i, g) in gammas.iter().enumerate() {
        for (j, &gij) in gamma_mat[i].iter().enumerate() {
            let term = g.scale_i64(gij as i64);
            gamma_tilde[j] = gamma_tilde[j].add(&term)?;
        }
    }
    let mut y_star = ring.zero();
    for (g, elt) in gammas.iter().zip(proof.y1.iter()) {
        let term = g.mul(elt)?;
        y_star = y_star.add(&term)?;
    }

    // Sumcheck verification.
    if proof.rounds.len() != params.nu {
        return Err(TtrpError::RoundCheck { round: usize::MAX });
    }
    let q = ring.modulus.q as u64;
    let inv2 = q.div_ceil(2) as i64;
    let mut inv2_pow = 1i64;
    let mut running: RingElement = ring.zero(); // C_0 = 0
    let mut challenges = Vec::with_capacity(params.nu);
    for (round, msg) in proof.rounds.iter().enumerate() {
        inv2_pow = inv2_pow * inv2 % q as i64;
        // g(0) + g(1) = C_{round}
        let sum01 = msg[0].add(&msg[1])?;
        if sum01 != running {
            return Err(TtrpError::RoundCheck { round });
        }
        for elt in msg {
            absorb_ring(transcript, elt)?;
        }
        let r = challenge_ring(transcript, ring)?;
        // C_{round+1} = g(r): Lagrange over {0,1,2}.
        running = lagrange3_eval(ring, msg, &r)?;
        challenges.push(r);
    }

    // Tensor-structured evaluation of mle(m*)(r).
    let mstar_at_r = tensor_eval_mstar(params, &rows, ring, &challenges, &gamma_tilde)?;

    // Terminal: C_ν = mle(m*)(r)·conj(w_r) − y*·2^{−ν}.
    let conj_w = conj(&proof.w_r);
    let rhs = mstar_at_r.mul(&conj_w)?;
    let const_term = y_star.scale_i64(inv2_pow);
    let expected = rhs.sub(&const_term)?;
    if expected != running {
        return Err(TtrpError::TerminalCheck);
    }
    absorb_ring(transcript, &proof.w_r)?;

    Ok(TtrpVerified {
        challenges,
        w_r: proof.w_r.clone(),
    })
}

/// Lagrange interpolation of a degree-2 polynomial given evaluations at
/// {0,1,2}, evaluated at `r`:
/// `g(r) = g(0)·(r−1)(r−2)/2 − g(1)·r(r−2) + g(2)·r(r−1)/2`.
fn lagrange3_eval(
    ring: &RingConfig,
    msg: &[RingElement; 3],
    r: &RingElement,
) -> Result<RingElement, TtrpError> {
    let one = ring.one();
    let two = ring.constant(2);
    let inv2 = {
        let q = ring.modulus.q;
        ring.constant(q.div_ceil(2) % q)
    };
    // (r-1), (r-2)
    let r1 = r.sub(&one)?;
    let r2 = r.sub(&two)?;
    // L0 = (r-1)(r-2)/2
    let l0 = r1.mul(&r2)?.mul(&inv2)?;
    // L1 = -r(r-2) = r(2-r)
    let neg_r2 = r2.neg();
    let l1 = r.mul(&neg_r2)?;
    // L2 = r(r-1)/2
    let l2 = r.mul(&r1)?.mul(&inv2)?;
    let t0 = msg[0].mul(&l0)?;
    let t1 = msg[1].mul(&l1)?;
    let t2 = msg[2].mul(&l2)?;
    Ok(t0.add(&t1)?.add(&t2)?)
}

/// The verifier's tensor-structured evaluation `mle(m*)(r)` (Lemma 6):
///
/// * chunk the ν challenges into µ₁ blocks of ℓ;
/// * per spatial core p, evaluate the MLE of its slices at the chunk via
///   the multilinear basis `E_p[n] = Π_{b: n_b=1} r^{(p)}_b`;
/// * chain-multiply into the 1×c boundary vector `V^{(j)}`;
/// * multiply by the γ̃-scaled coefficient chain `W̃^{(j)} = γ̃_j·W^{(j)}`;
/// * sum over rows.
///
/// Cost: `O(µ₁·2^ℓ)` shared basis mults + `O(k·µ₁·c²)` ring mults — never
/// materialising the `m̄r`-long rows.
#[allow(clippy::needless_range_loop)]
pub fn tensor_eval_mstar(
    params: &TtrpParams,
    rows: &[Vec<CoreTensor>],
    ring: &RingConfig,
    challenges: &[RingElement],
    gamma_tilde: &[RingElement],
) -> Result<RingElement, TtrpError> {
    let ell = params.ell;
    let d = params.d();
    let one = ring.one();
    // Shared per-chunk multilinear bases E[p][n], n ∈ {0,1}^ℓ (MSB-first
    // bit order matching the spatial digit decomposition).
    let mut bases: Vec<Vec<RingElement>> = Vec::with_capacity(params.mu1);
    for p in 0..params.mu1 {
        // Digit convention: the digit n_p's MOST-significant bit is h's
        // earliest bit of the chunk, which pairs with the chunk's FIRST
        // challenge. Building by doubling appends each new bit as the
        // current LSB, so the challenges are consumed in REVERSE order:
        // after processing b = ell-1 .. 0, index = sum_b bit_b * 2^{ell-1-b}
        // with bit_b <-> challenges[p*ell + b].
        let mut e = vec![one.clone()];
        for b in (0..ell).rev() {
            let r_b = &challenges[p * ell + b];
            let one_minus = one.sub(r_b)?;
            let mut next = Vec::with_capacity(e.len() * 2);
            for e_val in &e {
                next.push(e_val.mul(&one_minus)?);
            }
            for e_val in &e {
                next.push(e_val.mul(r_b)?);
            }
            e = next;
        }
        debug_assert_eq!(e.len(), d);
        bases.push(e);
    }

    let mut total = ring.zero();
    for (j, cores) in rows.iter().enumerate() {
        // Boundary chain: V = Ĉ_1 · Ĉ_2 · ... · Ĉ_{µ1}  (1×c over R).
        let mut v: Vec<RingElement> = {
            // Ĉ_1 ∈ R^{1×c}: Σ_n M_1(n)·E[0][n] on row 0.
            let core = &cores[0];
            let basis = &bases[0];
            let mut acc = vec![ring.zero(); core.r1];
            for n in 0..d {
                let e = &basis[n];
                for b in 0..core.r1 {
                    let w = core.at(n, 0, b) as i64;
                    if w != 0 {
                        let term = e.scale_i64(w);
                        acc[b] = acc[b].add(&term)?;
                    }
                }
            }
            acc
        };
        for p in 1..params.mu1 {
            // Ĉ_p ∈ R^{c×c}: Σ_n M_p(n)·E[p][n]; V := V·Ĉ_p.
            let core = &cores[p];
            let basis = &bases[p];
            let mut c_hat = vec![ring.zero(); core.r0 * core.r1];
            for n in 0..d {
                let e = &basis[n];
                for a in 0..core.r0 {
                    for b in 0..core.r1 {
                        let w = core.at(n, a, b) as i64;
                        if w != 0 {
                            let term = e.scale_i64(w);
                            let idx = a * core.r1 + b;
                            c_hat[idx] = c_hat[idx].add(&term)?;
                        }
                    }
                }
            }
            // (1×c) · (c×c): out[b] = Σ_a V[a]·Ĉ[a][b]
            let mut next = vec![ring.zero(); core.r1];
            for b in 0..core.r1 {
                let mut acc = ring.zero();
                for a in 0..core.r0 {
                    let prod = v[a].mul(&c_hat[a * core.r1 + b])?;
                    acc = acc.add(&prod)?;
                }
                next[b] = acc;
            }
            v = next;
        }
        // W̃^{(j)} = γ̃_j · W^{(j)} ∈ R^c; result V·W̃.
        let w = coefficient_chain(params, cores, ring);
        let mut acc = ring.zero();
        for i in 0..params.c {
            let w_tilde = gamma_tilde[j].mul(&w[i])?;
            let prod = v[i].mul(&w_tilde)?;
            acc = acc.add(&prod)?;
        }
        total = total.add(&acc)?;
    }
    Ok(total)
}
