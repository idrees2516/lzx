//! Symphony (Chen 2025, ePrint 2025/1905) §3.2 + §4 — Wave 7.9: the
//! tensor-ring substrate, the Hadamard RoK `Π_had` (Figure 1), and the
//! high-arity fold with **shared randomness** (Figure 4, Eqs 45/48–50).
//!
//! * **Tensor substrate** — `ts(r)` is the tensor of `(1 − r_k, r_k)` per
//!   variable (the eq table at `r`), the weight vector of the linear
//!   output relation `⟨M_i·f, ts(r)⟩ = v_i` (Eq 23). The paper's
//!   `K = F_{q²}` challenge field is realized over the base field `Z_q`
//!   of the RingSC engine (the `a = 1` subfield view — documented
//!   deviation; the eq/α algebra is identical).
//! * **Π_had (Fig 1)** — reduces the batched Hadamard relation
//!   `(M₁F) ∘ (M₂F) = M₃F` (position-wise ring products, Eq 22) to
//!   linear evaluation claims: the degree-3 sumcheck
//!   `Σ_b Σ_j α^{j−1}·eq(s,b)·(g₁,j(b)·g₂,j(b) − g₃,j(b)) = 0` with
//!   `g_{i,j} = M_i·F_{*,j}` (Eq 24), run through the R_q-native RingSC
//!   engine (`pikkufold_lrp::ring_sc_prove`). The terminal Eq-25 check:
//!   `Σ_j α^{j−1}·eq(s,r)·(U₁,j·U₂,j − U₃,j) = v_ev` with the prover's
//!   `U ∈ R_q^{3×d}` evaluation claims. The output instance is
//!   `(c, r, v ∈ E³)` with `v_i` packing `U_{i,*}` — checked by the
//!   decider against the commitment opening.
//! * **The O(μ) shared-randomness fold (Fig 4)** — `ℓ_np` instances run
//!   with ONE shared `(s, α)` and the `2ℓ_np` sumchecks merged into ONE
//!   via the α-power RLC of Eq 45:
//!   `Σ_b Σ_ℓ Σ_j α^{(ℓ−1)d+j−1}·f^{ℓ,j}(b) = 0` — the per-instance
//!   cross terms never enumerate subsets (the O(μ²) pairwise `E_ij`
//!   enumeration of the legacy `fold_many` is exactly the paper's §1.2
//!   strawman this replaces). The fold challenge `β ← S^{ℓ_np}` (the
//!   fixed-weight short-challenge set, Γ_C-certified) linearly combines
//!   commitments, evaluations, and witnesses (Eqs 48–49). The Eq-50
//!   feasibility gate
//!   `B_bnd ≥ √ℓ_np·∥S∥_op·max(B·n^{d/ℓ_h}, √n)` is enforced
//!   fail-closed on the folded witness norm.
//!
//! Kernel scale: `m = n = 8` slots, `d = 4` columns, ring dim 16 (the
//! column packing `f_b = Σ_j F_{b,j}·X^{j·φ/d}`), `ℓ_np = 4` instances —
//! ONE 3-round sumcheck regardless of `ℓ_np` (the O(μ) communication
//! story, test-pinned).

use crate::pikkufold_lrp::{
    ring_sc_prove, ring_sc_verify, RingScProof, RingVirtualPoly,
};
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiPublicKey};
use lattice_core::short_challenge::{ShortChallengeFamily, ShortChallengeSpec};
use lattice_core::transcript::Transcript;
use lattice_ring::{Modulus32, RingConfig, RingElement};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymphonyProtocolError {
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    Shape { expected: usize, got: usize },
    TranscriptFailure,
    Sumcheck(&'static str),
    /// The Eq-25 terminal identity failed.
    TerminalCheckFailed,
    /// The Eq-50 feasibility gate rejected the folded witness norm.
    Eq50GateExceeded { norm_sq: u128, bound_sq: u128 },
}

impl From<AjtaiError> for SymphonyProtocolError {
    fn from(e: AjtaiError) -> Self {
        SymphonyProtocolError::Ajtai(e)
    }
}
impl From<crate::pikkufold_lrp::LrpError> for SymphonyProtocolError {
    fn from(e: crate::pikkufold_lrp::LrpError) -> Self {
        match e {
            crate::pikkufold_lrp::LrpError::Sumcheck(m) => SymphonyProtocolError::Sumcheck(m),
            other => SymphonyProtocolError::Sumcheck(engine_msg(&other)),
        }
    }
}

fn engine_msg(_e: &crate::pikkufold_lrp::LrpError) -> &'static str {
    "engine"
}
impl From<lattice_ring::RingError> for SymphonyProtocolError {
    fn from(e: lattice_ring::RingError) -> Self {
        SymphonyProtocolError::Ring(e)
    }
}

/// `ts(r)_b = eq(b, r)` — the tensor weight vector (Eq 23's inner-product
/// weights; the MLE evaluation identity `MLE[g](r) = ⟨g, ts(r)⟩`).
pub fn ts(m: &Modulus32, r: &[u32]) -> Vec<u32> {
    let mut evals = vec![1u32; 1usize << r.len()];
    for (var, &e) in r.iter().enumerate() {
        let shift = r.len() - 1 - var;
        let one_minus = (m.q + 1 - e % m.q) % m.q;
        for (idx, val) in evals.iter_mut().enumerate() {
            let bit = (idx >> shift) & 1;
            let f = if bit == 1 { e } else { one_minus };
            *val = ((*val as u64 * f as u64) % m.q as u64) as u32;
        }
    }
    evals
}

/// `Π_had` parameters: ternary matrices `M₁, M₂, M₃ ∈ Z_q^{m×n}` over the
/// witness-column structure `F ∈ R_q^{n×d}`.
#[derive(Clone, Debug)]
pub struct HadParams {
    pub m: usize,
    pub n: usize,
    pub d: usize,
    /// Row-major ternary entries per matrix.
    pub mats: [Vec<i8>; 3],
    pub seed: Vec<u8>,
}

impl HadParams {
    pub fn from_seed(m: usize, n: usize, d: usize, seed: &[u8]) -> Self {
        let mut mats: [Vec<i8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (i, mat) in mats.iter_mut().enumerate() {
            let bytes = Transcript::xof(
                b"symphony-had-m",
                &[seed, &(i as u32).to_le_bytes()].concat(),
                m * n,
            );
            for &b in bytes.iter().take(m * n) {
                mat.push(match b & 0x3 {
                    0 | 1 => 0i8,
                    2 => 1,
                    _ => -1,
                });
            }
        }
        HadParams { m, n, d, mats, seed: seed.to_vec() }
    }

    /// `g_{i,j} = M_i·F_{*,j} ∈ R_q^m` (the column matvecs).
    fn g_tables(
        &self,
        ring: &RingConfig,
        f: &[RingElement],
    ) -> Result<Vec<Vec<Vec<RingElement>>>, SymphonyProtocolError> {
        let phi = ring.n();
        let block = phi / self.d;
        if f.len() != self.n {
            return Err(SymphonyProtocolError::Shape { expected: self.n, got: f.len() });
        }
        // Unpack F ∈ R_q^{n×d} from the packed witness f: column j's block
        // [j·block, (j+1)·block) becomes the low coefficients of the
        // standalone column entry (full-length — ring-element discipline).
        let unpack = |elem: &RingElement, j: usize| -> RingElement {
            let mut coeffs = vec![0u32; phi];
            for (k, c) in coeffs.iter_mut().enumerate().take(block) {
                *c = elem.coeff(j * block + k);
            }
            RingElement::from_coeffs(ring, coeffs)
        };
        let mut out = Vec::with_capacity(3);
        for mat in &self.mats {
            let mut per_col = Vec::with_capacity(self.d);
            for j in 0..self.d {
                let mut col = Vec::with_capacity(self.m);
                for a in 0..self.m {
                    let mut acc = ring.zero();
                    for b in 0..self.n {
                        let e = mat[a * self.n + b] as i64;
                        if e != 0 {
                            let fb = unpack(&f[b], j);
                            acc = acc.add(&fb.scale_i64(e))?;
                        }
                    }
                    col.push(acc);
                }
                per_col.push(col);
            }
            out.push(per_col);
        }
        Ok(out)
    }

    /// The linear output relation: `⟨M_i·f, ts(r)⟩ = v_i` (Eq 23) —
    /// constant-entry matrices commute with the column packing, so
    /// `M_i·f` packs the `g_{i,j}` columns.
    fn m_times_f(
        &self,
        ring: &RingConfig,
        f: &[RingElement],
    ) -> Result<Vec<Vec<RingElement>>, SymphonyProtocolError> {
        if f.len() != self.n {
            return Err(SymphonyProtocolError::Shape { expected: self.n, got: f.len() });
        }
        let mut out = Vec::with_capacity(3);
        for mat in &self.mats {
            let mut col = Vec::with_capacity(self.m);
            for a in 0..self.m {
                let mut acc = ring.zero();
                for b in 0..self.n {
                    let e = mat[a * self.n + b] as i64;
                    if e != 0 {
                        acc = acc.add(&f[b].scale_i64(e))?;
                    }
                }
                col.push(acc);
            }
            out.push(col);
        }
        Ok(out)
    }
}

/// A `Π_had` proof: the RingSC proof plus the `U ∈ R_q^{3×d}` evaluation
/// claims (Figure 1 step 3).
#[derive(Clone, Debug)]
pub struct HadProof {
    pub sumcheck: RingScProof,
    /// `U[i·d + j] = g_{i,j}(r)`.
    pub u: Vec<RingElement>,
}

/// The `Π_had` output instance `(c, r, v ∈ E³)` (Eq 23's statement).
#[derive(Clone, Debug)]
pub struct HadOutput {
    pub r: Vec<u32>,
    /// `v_i` packs `U_{i,*}` with the column packing.
    pub v: [RingElement; 3],
}

/// Build the Figure-1 virtual polynomial for one instance (or the shared
/// Eq-45 merge over `ℓ` instances when `fs` carries several witnesses).
#[allow(clippy::too_many_arguments)]
fn build_had_vp(
    ring: &RingConfig,
    params: &HadParams,
    s: &[u32],
    alpha: u32,
    fs: &[Vec<RingElement>],
) -> Result<RingVirtualPoly, SymphonyProtocolError> {
    let num_vars = log2(params.m);
    let mut vp = RingVirtualPoly::new(num_vars);
    let m = &ring.modulus;
    // Shared eq(s) factor (constants).
    let eq_tab: Vec<RingElement> = ts(m, s).iter().map(|&w| ring.constant(w)).collect();
    let eq_id = vp.add_factor(eq_tab)?;
    let ell = fs.len();
    let mut alpha_pow = 1u32;
    for (li, f) in fs.iter().enumerate() {
        let g = params.g_tables(ring, f)?;
        for (j, per_col) in g[0].iter().zip(g[1].iter().zip(g[2].iter())).enumerate() {
            let (g0j, (g1j, g2j)) = per_col;
            let g1 = vp.add_factor(g0j.clone())?;
            let g2 = vp.add_factor(g1j.clone())?;
            let g3 = vp.add_factor(g2j.clone())?;
            // Eq 45 weight: α^{(ℓ−1)d + j − 1}.
            let target_exp = li * params.d + j;
            let _ = target_exp;
            let c_pos = ring.constant(alpha_pow);
            let c_neg = ring.constant((m.q - alpha_pow % m.q) % m.q);
            vp.add_term(c_pos, vec![eq_id, g1, g2])?;
            vp.add_term(c_neg, vec![eq_id, g3])?;
            alpha_pow = (alpha_pow as u64 * alpha as u64 % m.q as u64) as u32;
        }
    }
    let _ = ell;
    Ok(vp)
}

/// Sample the shared challenges `s ← Z_q^{log m}` and `α ← Z_q`.
fn sample_shared_challenges(
    ring: &RingConfig,
    params: &HadParams,
    transcript: &mut Transcript,
) -> Result<(Vec<u32>, u32), SymphonyProtocolError> {
    let q = ring.modulus.q;
    let log_m = log2(params.m);
    let mut s = Vec::with_capacity(log_m);
    for _ in 0..log_m {
        s.push(challenge_zq(transcript, b"symphony-had-s", q)?);
    }
    let alpha = challenge_zq(transcript, b"symphony-had-alpha", q)?;
    Ok((s, alpha))
}

fn challenge_zq(
    transcript: &mut Transcript,
    label: &[u8],
    q: u32,
) -> Result<u32, SymphonyProtocolError> {
    let limit = u64::from(q);
    let bound = u64::MAX - (u64::MAX % limit) - 1;
    for _ in 0..16 {
        let bytes = transcript
            .challenge_bytes(label, 8)
            .map_err(|_| SymphonyProtocolError::TranscriptFailure)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes[..8]);
        let v = u64::from_le_bytes(arr);
        if v <= bound {
            return Ok((v % limit) as u32);
        }
    }
    Err(SymphonyProtocolError::TranscriptFailure)
}

fn log2(x: usize) -> usize {
    x.trailing_zeros() as usize
}

fn absorb_had_statement(
    params: &HadParams,
    commitments: &[AjtaiCommitment],
    transcript: &mut Transcript,
) -> Result<(), SymphonyProtocolError> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(params.m as u32).to_le_bytes());
    buf.extend_from_slice(&(params.n as u32).to_le_bytes());
    buf.extend_from_slice(&(params.d as u32).to_le_bytes());
    buf.extend_from_slice(&(commitments.len() as u32).to_le_bytes());
    buf.extend_from_slice(&params.seed);
    transcript
        .append_bytes(b"symphony-had-stmt", &buf)
        .map_err(|_| SymphonyProtocolError::TranscriptFailure)?;
    for c in commitments {
        transcript
            .append_bytes(b"symphony-had-c", &c.to_bytes())
            .map_err(|_| SymphonyProtocolError::TranscriptFailure)?;
    }
    Ok(())
}

/// Prove `Π_had` (Figure 1) for one committed instance — or, when
/// `witnesses` carries `ℓ_np` entries, the shared-randomness merged
/// protocol of Figure 4 Step 1 (one sumcheck for all instances, Eq 45).
/// Returns the proof with per-instance `U` claims.
pub fn prove_had(
    ring: &RingConfig,
    params: &HadParams,
    commitments: &[AjtaiCommitment],
    witnesses: &[Vec<RingElement>],
    transcript: &mut Transcript,
) -> Result<HadProof, SymphonyProtocolError> {
    if commitments.len() != witnesses.len() || witnesses.is_empty() {
        return Err(SymphonyProtocolError::Shape {
            expected: commitments.len(),
            got: witnesses.len(),
        });
    }
    absorb_had_statement(params, commitments, transcript)?;
    let (s, alpha) = sample_shared_challenges(ring, params, transcript)?;
    let vp = build_had_vp(ring, params, &s, alpha, witnesses)?;
    // The Hadamard relation holds ⇒ the batched claim is exactly zero.
    let claim = ring.zero();
    let out = ring_sc_prove(ring, &vp, &claim, transcript)?;
    // U claims: the g-factor claims (factor ids: 1 + 3·(ℓ·d) ordering).
    let u: Vec<RingElement> = out.factor_claims[1..].to_vec();
    Ok(HadProof { sumcheck: out.proof, u })
}

/// Verify `Π_had` / the merged Figure-4 Step-1 protocol. Returns the
/// per-instance outputs `(r, v^ℓ ∈ E³)`.
pub fn verify_had(
    ring: &RingConfig,
    params: &HadParams,
    commitments: &[AjtaiCommitment],
    proof: &HadProof,
    transcript: &mut Transcript,
) -> Result<Vec<HadOutput>, SymphonyProtocolError> {
    let m = &ring.modulus;
    let ell = commitments.len();
    absorb_had_statement(params, commitments, transcript)?;
    let (s, alpha) = sample_shared_challenges(ring, params, transcript)?;
    let num_vars = log2(params.m);
    let max_degree = 3;
    let claim = ring.zero();
    let verdict = ring_sc_verify(ring, num_vars, max_degree, &claim, &proof.sumcheck, transcript)?;
    let point = verdict.point;
    // Eq 25 terminal cross-check, merged over instances (Eq 45 weights):
    // Σ_ℓ Σ_j α^{(ℓ−1)d+j−1}·eq(s, r)·(U₁^ℓ,j·U₂^ℓ,j − U₃^ℓ,j) = terminal.
    // eq(s, r) = Π_k (s_k·r_k + (1−s_k)(1−r_k)) — the Figure-1 Eq-25 factor.
    let eq_sr = {
        let mut e = 1u32;
        for (k, &sv) in s.iter().enumerate() {
            let rv = point[k];
            let oms = (m.q + 1 - sv % m.q) % m.q;
            let omr = (m.q + 1 - rv % m.q) % m.q;
            let term = ((sv as u64 * rv as u64 + oms as u64 * omr as u64) % m.q as u64) as u32;
            e = ((e as u64 * term as u64) % m.q as u64) as u32;
        }
        e
    };
    let expected_u_len = 3 * params.d * ell;
    if proof.u.len() != expected_u_len {
        return Err(SymphonyProtocolError::Shape { expected: expected_u_len, got: proof.u.len() });
    }
    let mut alpha_pow = 1u32;
    let mut recomputed = ring.zero();
    for li in 0..ell {
        for j in 0..params.d {
            let u1 = &proof.u[(li * params.d + j) * 3];
            let u2 = &proof.u[(li * params.d + j) * 3 + 1];
            let u3 = &proof.u[(li * params.d + j) * 3 + 2];
            let prod = u1.mul(u2)?.sub(u3)?;
            let w = ((alpha_pow as u64 * eq_sr as u64) % m.q as u64) as u32;
            recomputed = recomputed.add(&prod.scale_i64(w as i64))?;
            alpha_pow = ((alpha_pow as u64 * alpha as u64) % m.q as u64) as u32;
        }
    }
    if recomputed != proof.sumcheck.terminal {
        return Err(SymphonyProtocolError::TerminalCheckFailed);
    }
    // Pack the per-instance outputs: v_i^ℓ = Σ_j U_{i,*}·X^{j·φ/d}.
    let phi = ring.n();
    let block = phi / params.d;
    let mut outputs = Vec::with_capacity(ell);
    for li in 0..ell {
        let mut v = [ring.zero(), ring.zero(), ring.zero()];
        for (i, vi) in v.iter_mut().enumerate() {
            let mut coeffs = vec![0u32; phi];
            for j in 0..params.d {
                let uij = &proof.u[(li * params.d + j) * 3 + i];
                for k in 0..block {
                    let c = uij.coeff(k);
                    if c != 0 {
                        coeffs[j * block + k] = c;
                    }
                }
            }
            *vi = RingElement::from_coeffs(ring, coeffs);
        }
        outputs.push(HadOutput { r: point.clone(), v });
    }
    Ok(outputs)
}

/// The decider for a `Π_had` output: `c` opens `f` and
/// `⟨M_i·f, ts(r)⟩ = v_i` for every `i ∈ [3]` (Eq 23).
pub fn verify_had_opening(
    pk: &AjtaiPublicKey,
    params: &HadParams,
    c: &AjtaiCommitment,
    f: &[RingElement],
    output: &HadOutput,
) -> Result<(), SymphonyProtocolError> {
    let ring = &pk.params.ring;
    pk.verify_opening(c, f)?;
    let ts_r = ts(&ring.modulus, &output.r);
    let mifs = params.m_times_f(ring, f)?;
    for (i, vi) in output.v.iter().enumerate() {
        let mut acc = ring.zero();
        for (b, &w) in ts_r.iter().enumerate() {
            if w != 0 {
                acc = acc.add(&mifs[i][b].scale_i64(w as i64))?;
            }
        }
        if acc != *vi {
            return Err(SymphonyProtocolError::TerminalCheckFailed);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The O(μ) shared-randomness fold (Figure 4, steps 4–6 + Eq 50)
// ---------------------------------------------------------------------------

/// The folded output statement of Figure 4: `(c*, v*)` combined under
/// `β ← S^{ℓ_np}` (Eqs 48–49) at the shared terminal point.
#[derive(Clone, Debug)]
pub struct SymphonyFoldOutput {
    /// `c* = Σ β_ℓ·c_ℓ` (Ajtai homomorphism).
    pub c_star: AjtaiCommitment,
    /// `v*_i = Σ β_ℓ·v^ℓ_i` per matrix (the E-element folds).
    pub v_star: [RingElement; 3],
    /// The shared terminal point.
    pub r: Vec<u32>,
    /// The fold challenges (public — transcript-derived).
    pub beta: Vec<RingElement>,
}

/// The Eq-50 feasibility bound:
/// `B_bnd ≥ √ℓ_np·∥S∥_op·max(B·n^{d/ℓ_h}, √n)` — returned as `B_bnd²`.
pub fn eq50_bound_squared(
    ell_np: usize,
    gamma_c: u64,
    b_f: u64,
    n: usize,
    d: usize,
    ell_h: usize,
) -> u128 {
    let nd_lh = (n as f64).powf(d as f64 / ell_h as f64);
    let term1 = b_f as f64 * nd_lh;
    let term2 = (n as f64).sqrt();
    let max_term = term1.max(term2);
    let bound = (ell_np as f64).sqrt() * gamma_c as f64 * max_term;
    (bound.ceil().max(0.0) as u128).saturating_pow(2)
}

/// Witness ℓ2 norm squared (over ring elements).
fn norm_sq(f: &[RingElement]) -> u128 {
    f.iter()
        .map(|e| u128::from(e.euclidean_norm_squared()))
        .sum()
}

/// The full Figure-4 fold proof: the merged `Π_had` execution (Steps 1–3)
/// plus everything the verifier needs for Steps 4–6.
pub struct SymphonyFoldProof {
    /// The merged Π_had proof (ONE sumcheck for all ℓ_np instances).
    pub had: HadProof,
}

/// Prove the Figure-4 fold: the merged shared-randomness `Π_had`
/// (Step 1–3, one sumcheck — Eq 45) followed by `β ← S^{ℓ_np}` (Step 4)
/// and the witness-side linear fold with the Eq-50 fail-closed gate
/// (Steps 5–6). The β challenges are absorbed for verifier replay.
pub fn prove_fold_symphony(
    pk: &AjtaiPublicKey,
    params: &HadParams,
    commitments: &[AjtaiCommitment],
    witnesses: &[Vec<RingElement>],
    b_f: u64,
    b_bnd_cap: u64,
    transcript: &mut Transcript,
) -> Result<SymphonyFoldProof, SymphonyProtocolError> {
    let ring = &pk.params.ring;
    let ell = commitments.len();
    // Steps 1–3: the merged protocol (ONE sumcheck — Eq 45).
    let had = prove_had(ring, params, commitments, witnesses, transcript)?;
    // Step 4: β ← S^{ℓ_np} (fixed-weight short challenges, Γ_C-certified).
    let spec = ShortChallengeSpec {
        n: ring.n(),
        family: ShortChallengeFamily::FixedWeight { weight: 3, amplitude: 1 },
    };
    let mut beta = Vec::with_capacity(ell);
    for _ in 0..ell {
        let seed = transcript
            .challenge_bytes(b"symphony-fold-beta", 32)
            .map_err(|_| SymphonyProtocolError::TranscriptFailure)?;
        let ch = spec.sample(&seed).map_err(|_| SymphonyProtocolError::TranscriptFailure)?;
        beta.push(RingElement::from_signed(ring, &ch.coefficients));
    }
    // Steps 5–6 (prover side): f* = Σ_ℓ β_ℓ·f_ℓ (ALL instances weighted —
    // Eq 49) with the Eq-50 gate.
    let gamma = beta
        .iter()
        .map(|b| (b.euclidean_norm_squared() as f64).sqrt().ceil() as u64)
        .max()
        .unwrap_or(1);
    let mut f_star: Vec<RingElement> = Vec::with_capacity(witnesses[0].len());
    for w in &witnesses[0] {
        f_star.push(w.mul(&beta[0])?);
    }
    for (li, bl) in beta.iter().enumerate().skip(1) {
        for (j, w) in witnesses[li].iter().enumerate() {
            f_star[j] = f_star[j].add(&w.mul(bl)?)?;
        }
    }
    let bound_sq = eq50_bound_squared(ell, gamma, b_f, params.n, params.d, params.d)
        .min(u128::from(b_bnd_cap).saturating_pow(2));
    let ns = norm_sq(&f_star);
    if ns > bound_sq {
        return Err(SymphonyProtocolError::Eq50GateExceeded { norm_sq: ns, bound_sq });
    }
    Ok(SymphonyFoldProof { had })
}

/// Verify the Figure-4 fold: verify the merged `Π_had` (replaying the
/// shared challenges and the single sumcheck), re-sample `β`, fold the
/// commitments and the E-element evaluations (Eqs 48–49), and enforce the
/// Eq-50 wraparound feasibility (the derived `B_bnd` must fit under
/// `q/2` — a folded witness past the opening bound would wrap mod q and
/// destroy the binding argument). Returns the folded output statement.
pub fn verify_fold_symphony(
    pk: &AjtaiPublicKey,
    params: &HadParams,
    commitments: &[AjtaiCommitment],
    proof: &SymphonyFoldProof,
    b_f: u64,
    transcript: &mut Transcript,
) -> Result<SymphonyFoldOutput, SymphonyProtocolError> {
    let ring = &pk.params.ring;
    let ell = commitments.len();
    // Steps 1–3 (verifier): the merged protocol.
    let outputs = verify_had(ring, params, commitments, &proof.had, transcript)?;
    let r = outputs[0].r.clone();
    // Step 4: β replay.
    let spec = ShortChallengeSpec {
        n: ring.n(),
        family: ShortChallengeFamily::FixedWeight { weight: 3, amplitude: 1 },
    };
    let mut beta = Vec::with_capacity(ell);
    for _ in 0..ell {
        let seed = transcript
            .challenge_bytes(b"symphony-fold-beta", 32)
            .map_err(|_| SymphonyProtocolError::TranscriptFailure)?;
        let ch = spec.sample(&seed).map_err(|_| SymphonyProtocolError::TranscriptFailure)?;
        beta.push(RingElement::from_signed(ring, &ch.coefficients));
    }
    // Steps 5–6: c* = Σ_ℓ β_ℓ·c_ℓ (Eq 48 — every instance weighted),
    // v*_i = Σ_ℓ β_ℓ·v^ℓ_i.
    let mut c_rows: Vec<RingElement> = Vec::with_capacity(commitments[0].rows.len());
    for r in &commitments[0].rows {
        c_rows.push(r.mul(&beta[0])?);
    }
    for (li, bl) in beta.iter().enumerate().skip(1) {
        for (j, row) in commitments[li].rows.iter().enumerate() {
            c_rows[j] = c_rows[j].add(&row.mul(bl)?)?;
        }
    }
    let c_star = AjtaiCommitment { rows: c_rows };
    let mut v_star = [ring.zero(), ring.zero(), ring.zero()];
    for (out, bl) in outputs.iter().zip(beta.iter()) {
        for (vi, vs) in out.v.iter().zip(v_star.iter_mut()) {
            *vs = vs.add(&vi.mul(bl)?)?;
        }
    }
    // Eq-50 feasibility: the derived B_bnd must respect the q/2 wraparound
    // margin (the relaxed-binding opening space must not wrap mod q).
    let gamma = beta
        .iter()
        .map(|b| (b.euclidean_norm_squared() as f64).sqrt().ceil() as u64)
        .max()
        .unwrap_or(1);
    let bound_sq = eq50_bound_squared(ell, gamma, b_f, params.n, params.d, params.d);
    let q_half_sq = ((ring.modulus.q as u64) / 2).pow(2) as u128;
    if bound_sq > q_half_sq {
        return Err(SymphonyProtocolError::Eq50GateExceeded { norm_sq: bound_sq, bound_sq: q_half_sq });
    }
    Ok(SymphonyFoldOutput { c_star, v_star, r, beta })
}

/// The decider for the folded output: `c*` opens `f*` (the folded witness,
/// caller-supplied on the proving side) and `⟨M_i·f*, ts(r)⟩ = v*_i`.
pub fn verify_fold_opening(
    pk: &AjtaiPublicKey,
    params: &HadParams,
    output: &SymphonyFoldOutput,
    f_star: &[RingElement],
) -> Result<(), SymphonyProtocolError> {
    let ring = &pk.params.ring;
    pk.verify_opening(&output.c_star, f_star)?;
    let ts_r = ts(&ring.modulus, &output.r);
    let mifs = params.m_times_f(ring, f_star)?;
    for (i, vs) in output.v_star.iter().enumerate() {
        let mut acc = ring.zero();
        for (b, &w) in ts_r.iter().enumerate() {
            if w != 0 {
                acc = acc.add(&mifs[i][b].scale_i64(w as i64))?;
            }
        }
        if acc != *vs {
            return Err(SymphonyProtocolError::TerminalCheckFailed);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};

    fn setup(log_n: u32, m_slots: usize) -> (AjtaiPublicKey, RingConfig) {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams { ring: ring.clone(), k: 2, m: m_slots, norm_bound: 1 << 20 };
        let pk = AjtaiPublicKey::from_seed(params, [77u8; 32]).ok().unwrap();
        (pk, ring)
    }

    fn small_w(ring: &RingConfig, m: usize, tag: &[u8]) -> Vec<RingElement> {
        lattice_commitment::ajtai::sample_small_secret(ring, m, 4, tag)
    }

    /// Build a witness F ∈ R_q^{n×d} satisfying (M1F) ∘ (M2F) = M3F per
    /// position: solve by choosing F freely and REPLACING column 3's
    /// positions — i.e. construct F then set M3·F_{*,3}... simplest honest
    /// construction: pick W1, W2 freely; F := columns such that M1F and
    /// M2F multiply cleanly is hard — instead pick F freely and define
    /// the THIRD matrix's action by the relation: we cheat the structure
    /// by using M3 = M1∘-adjusted? Honest approach at kernel scale: pick
    /// F freely, compute g1, g2, and DEFINE g3 := g1 ∘ g2 — the relation
    /// holds by construction; M3 is then derived as the matrix mapping
    /// F_{*,3} ↦ g3 — which requires M3 to exist. Since M_i are public
    /// fixed ternary, instead we SEARCH: with M1 = M2 = M3 = the identity
    /// (m = n), the relation is F ∘ F = F ⇒ F entries idempotent ring
    /// elements (0/1-coefficient Boolean-ish). We use 0/1 witnesses.
    fn hadamard_witness(ring: &RingConfig, n: usize, d: usize, tag: &[u8]) -> Vec<RingElement> {
        // f_b packs d Boolean columns; with M1 = M2 = M3 = I the Hadamard
        // relation needs F∘F = F only when all three matrices match — we
        // instead use the dedicated identity matrices below in the fixture.
        let mut f = Vec::with_capacity(n);
        for b in 0..n {
            let mut coeffs = vec![0u32; ring.n()];
            for j in 0..d {
                let bit = ((b + j + tag[0] as usize) % 3 == 0) as u32;
                coeffs[j * (ring.n() / d)] = bit;
            }
            f.push(RingElement::from_coeffs(ring, coeffs));
        }
        f
    }

    /// Identity-matrix parameters (m = n): the Hadamard relation becomes
    /// F ∘ F = F, satisfied by 0/1-valued witnesses.
    fn identity_params(m: usize, n: usize, d: usize) -> HadParams {
        let mut mats: [Vec<i8>; 3] = core::array::from_fn(|_| vec![0i8; m * n]);
        for a in 0..m.min(n) {
            for mat in mats.iter_mut() {
                mat[a * n + a] = 1;
            }
        }
        HadParams { m, n, d, mats, seed: b"sym-identity".to_vec() }
    }

    #[test]
    fn ts_is_eq_and_mle_weights() {
        // ts(r) = eq table; MLE[g](r) = ⟨g, ts(r)⟩.
        let (_pk, ring) = setup(4, 8);
        let r = vec![3u32, 1, 2];
        let t = ts(&ring.modulus, &r);
        // brute MLE of a table g at r.
        let g: Vec<RingElement> = (0..8)
            .map(|i| RingElement::from_signed(&ring, &[(i as i64 % 5); 16]))
            .collect();
        let mut acc = ring.zero();
        for (b, &w) in t.iter().enumerate() {
            if w != 0 {
                acc = acc.add(&g[b].scale_i64(w as i64)).ok().unwrap();
            }
        }
        // MLE via the pikkufold_lrp eq_table + mle_at.
        let mle = crate::pikkufold_lrp::mle_at(&ring, &g, &r).ok().unwrap();
        assert_eq!(acc, mle);
    }

    #[test]
    fn pi_had_happy_path_and_decider() {
        let (pk, ring) = setup(4, 8);
        let params = identity_params(8, 8, 4);
        let f = hadamard_witness(&ring, 8, 4, b"h1");
        let c = pk.commit(&f).ok().unwrap();
        let mut t = Transcript::new_default(b"sym-had");
        let proof = prove_had(&ring, &params, &[c.clone()], &[f.clone()], &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"sym-had");
        let outputs = verify_had(&ring, &params, &[c.clone()], &proof, &mut vt).ok().unwrap();
        assert_eq!(outputs.len(), 1);
        // Decider: c opens f and ⟨M_i f, ts(r)⟩ = v_i.
        assert!(verify_had_opening(&pk, &params, &c, &f, &outputs[0]).is_ok());
        // The merged claim is zero ⇒ rounds sum to zero.
        assert_eq!(proof.sumcheck.rounds.len(), 3); // log m
        assert!(proof.sumcheck.rounds.iter().all(|r| r.len() == 4)); // degree 3
    }

    #[test]
    fn pi_had_rejects_non_hadamard_witness() {
        let (pk, ring) = setup(4, 8);
        let params = identity_params(8, 8, 4);
        // A witness with a non-Boolean column violates F ∘ F = F.
        let mut f = hadamard_witness(&ring, 8, 4, b"h2");
        let mut coeffs = vec![0u32; ring.n()];
        coeffs[0] = 5; // 5 ≠ 5² mod nothing — not idempotent
        f[0] = RingElement::from_coeffs(&ring, coeffs);
        let c = pk.commit(&f).ok().unwrap();
        let mut t = Transcript::new_default(b"sym-had-bad");
        // The honest prover fail-closes (round-sum guard) since the
        // batched claim is nonzero.
        assert!(prove_had(&ring, &params, &[c], &[f], &mut t).is_err());
    }

    #[test]
    fn pi_had_tampered_u_rejected() {
        let (pk, ring) = setup(4, 8);
        let params = identity_params(8, 8, 4);
        let f = hadamard_witness(&ring, 8, 4, b"h3");
        let c = pk.commit(&f).ok().unwrap();
        let mut t = Transcript::new_default(b"sym-had-t");
        let mut proof = prove_had(&ring, &params, &[c.clone()], &[f.clone()], &mut t).ok().unwrap();
        // Tampered U: the Eq-25 terminal cross-check fails.
        proof.u[0] = proof.u[0].add(&ring.one()).ok().unwrap();
        let mut vt = Transcript::new_default(b"sym-had-t");
        assert!(matches!(
            verify_had(&ring, &params, &[c], &proof, &mut vt),
            Err(SymphonyProtocolError::TerminalCheckFailed)
        ));
    }

    #[test]
    fn fig4_fold_one_sumcheck_regardless_of_arity() {
        // THE O(μ) story: ℓ_np instances share ONE sumcheck — the rounds
        // count stays log(m) and the pairwise E_ij enumeration is gone.
        let (pk, ring) = setup(4, 8);
        let params = identity_params(8, 8, 4);
        let ell = 4;
        let mut cs = Vec::new();
        let mut ws = Vec::new();
        for i in 0..ell {
            let f = hadamard_witness(&ring, 8, 4, format!("f{}", i).as_bytes());
            cs.push(pk.commit(&f).ok().unwrap());
            ws.push(f);
        }
        let b_f = norm_sq(&ws[0]) as u64 + 1;
        let mut t = Transcript::new_default(b"sym-fold");
        let proof =
            prove_fold_symphony(&pk, &params, &cs, &ws, b_f, u64::MAX, &mut t).ok().unwrap();
        // ONE sumcheck: rounds = log m = 3, independent of ell.
        assert_eq!(proof.had.sumcheck.rounds.len(), 3);
        // Per-instance U claims: 3·d·ℓ.
        assert_eq!(proof.had.u.len(), 3 * 4 * ell);
        let mut vt = Transcript::new_default(b"sym-fold");
        let out =
            verify_fold_symphony(&pk, &params, &cs, &proof, b_f, &mut vt).ok().unwrap();
        // Decider: f* = Σ_ℓ β_ℓ·f_ℓ (all weighted) opens c* and satisfies
        // the linear relation.
        let mut f_star: Vec<RingElement> =
            ws[0].iter().map(|w| w.mul(&out.beta[0]).ok().unwrap()).collect();
        for (li, bl) in out.beta.iter().enumerate().skip(1) {
            for (j, w) in ws[li].iter().enumerate() {
                f_star[j] = f_star[j].add(&w.mul(bl).ok().unwrap()).ok().unwrap();
            }
        }
        assert!(verify_fold_opening(&pk, &params, &out, &f_star).is_ok());
    }

    #[test]
    fn fig4_eq50_gate_fails_closed() {
        let (pk, ring) = setup(4, 8);
        let params = identity_params(8, 8, 4);
        let ell = 4;
        let mut cs = Vec::new();
        let mut ws = Vec::new();
        for i in 0..ell {
            let f = hadamard_witness(&ring, 8, 4, format!("g{}", i).as_bytes());
            cs.push(pk.commit(&f).ok().unwrap());
            ws.push(f);
        }
        // A hard opening-bound cap B_bnd below the folded norm makes the
        // Eq-50 gate reject fail-closed.
        let b_f = norm_sq(&ws[0]) as u64 + 1;
        let mut t = Transcript::new_default(b"sym-fold-bad");
        assert!(matches!(
            prove_fold_symphony(&pk, &params, &cs, &ws, b_f, 1, &mut t),
            Err(SymphonyProtocolError::Eq50GateExceeded { .. })
        ));
    }

    #[test]
    fn eq50_bound_formula() {
        // Monotone in ℓ, B, and Γ; the n^{d/ℓh} term dominates when ℓh < d.
        let a = eq50_bound_squared(4, 2, 100, 8, 4, 4);
        let b = eq50_bound_squared(8, 2, 100, 8, 4, 4);
        let c = eq50_bound_squared(4, 2, 200, 8, 4, 4);
        let d = eq50_bound_squared(4, 3, 100, 8, 4, 4);
        let e = eq50_bound_squared(4, 2, 100, 8, 4, 2);
        assert!(b > a);
        assert!(c > a);
        assert!(d > a);
        assert!(e > a); // n^{d/ℓh} = 8² > 8
    }
}
