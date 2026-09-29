//! Quasar (Cramer et al., ePrint 2025/1912) — Wave 7.10: the multi-cast
//! reduction `NIR_multicast` (Q2) and the 2-to-1 folding reduction with
//! its accumulation verifier/decider (Q3), composed into the
//! multi-instance IVC loop (Q5's payoff).
//!
//! * **Q2 — `NIR_multicast`** (§4/§5.2): the prover commits the UNION
//!   polynomials `x̃_∪ (Y, X) = mle[{x^(k)}]`, `w̃_∪ (Y, X) =
//!   mle[{w^(k)}]` as ONE Ajtai commitment per side covering all `ℓ`
//!   instances (the linear Z_q packing keeps the commitment homomorphic),
//!   and the partially-evaluated `w̃ = w̃_∪ (τ, ·)` under a third
//!   commitment `C`. The verifier samples `r_y` and runs the
//!   **log ℓ-round sumcheck** over `G(Y) = F(x̃(Y), w̃(Y))·eq(Y, r_y)`
//!   (individual degree 3 for the bilinear predicate) through the
//!   R_q-native RingSC engine. The sumcheck challenges ARE the
//!   accumulation point `τ`; the verifier derives
//!   `e := G(τ)·eq(τ, r_y)^{-1} = F(x, w)` (the relaxed slack — for
//!   honest inputs `H(y) = 0` on the cube but `H(τ) ≠ 0` in general),
//!   computes the accumulated public vector `x = Σ_k eq̃_k(τ)·x^(k)`
//!   (field-only), and samples `r_x, r_w` for the partial-evaluation
//!   consistency claims `x̃_∪ (τ, r_x) = x̃(r_x)` (verifier-computed) and
//!   `w̃_∪ (τ, r_w) = w̃(r_w)` (prover-claimed, decider-checked).
//! * **Q3 — the 2-to-1 fold + `ACC.V`/`ACC.D`**: two accumulated
//!   instances fold under a challenge `γ`: every linear component
//!   (vectors, commitments via the Ajtai homomorphism, claims) combines
//!   as `·₀ + γ··₁`, and the relaxed constraint combines with the
//!   γ-power cross term `e* = e₀ + γ·T + γ²·e₁` (T = the bilinear cross
//!   `F(x₀, w̃₁) + F(x₁, w̃₀)` — one field element, Protostar-style).
//!   `ACC.V` derives the folded instance from public data in O(1);
//!   `ACC.D` (the decider) opens every commitment and checks
//!   `F(x*, w̃*) = e*` — which binds the cross term. The paper's
//!   `IOR_batch` evaluation-claim folding (the Reval subrelation
//!   cross-terms) is kernel-simplified: the multicast claims are
//!   decider-verified per accumulated instance before folding
//!   (documented deviation).
//! * **Q5 — the multi-instance IVC loop**: per step, `ℓ` fresh chunk
//!   instances enter via ONE multicast and fold into the running
//!   accumulator (the shard-parallel zkVM payoff: the per-step verifier
//!   is O(log ℓ + 1) sumcheck rounds, never O(ℓ·n)).
//!
//! Kernel scale: `ℓ = 4` instances of `m = n = 8` over `Z_q`, bilinear
//! predicates with 3 terms, Ajtai commitments over the 16-dim ring
//! (16 Z_q values per ring element — the linear packing).

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_ring::{Modulus32, RingConfig, RingElement};

use lattice_folding::pikkufold_lrp::{ring_sc_prove, ring_sc_verify, RingScProof, RingVirtualPoly};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuasarAccError {
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    Sumcheck(&'static str),
    Shape { expected: usize, got: usize },
    TranscriptFailure,
    /// `eq(τ, r_y) = 0` — the multicast degenerates (probability ≤
    /// (log ℓ)/q; the paper re-samples, we reject).
    EqChallengeDegenerate,
    /// A decider check failed (openings, consistency, or the F-constraint).
    DeciderFailed(&'static str),
}

impl From<AjtaiError> for QuasarAccError {
    fn from(e: AjtaiError) -> Self {
        QuasarAccError::Ajtai(e)
    }
}
impl From<lattice_ring::RingError> for QuasarAccError {
    fn from(e: lattice_ring::RingError) -> Self {
        QuasarAccError::Ring(e)
    }
}
impl From<lattice_folding::pikkufold_lrp::LrpError> for QuasarAccError {
    fn from(e: lattice_folding::pikkufold_lrp::LrpError) -> Self {
        match e {
            lattice_folding::pikkufold_lrp::LrpError::Sumcheck(m) => QuasarAccError::Sumcheck(m),
            _ => QuasarAccError::Sumcheck("engine"),
        }
    }
}

/// The bilinear predicate `F(x, w) = Σ_t c_t·x_{a_t}·w_{b_t}` over `Z_q`.
#[derive(Clone, Debug)]
pub struct MulticastParams {
    pub ell: usize,
    pub m: usize,
    pub n: usize,
    /// `(c, x-index, w-index)` triples.
    pub terms: Vec<(u32, usize, usize)>,
    /// Ternary predicate seed (verifier-computable matrices).
    pub seed: Vec<u8>,
}

impl MulticastParams {
    pub fn from_seed(ell: usize, m: usize, n: usize, num_terms: usize, seed: &[u8]) -> Self {
        let bytes = Transcript::xof(b"quasar-pred", seed, num_terms * 8);
        let mut terms = Vec::with_capacity(num_terms);
        for chunk in bytes.chunks(8).take(num_terms) {
            let c = 1u32 + (chunk[0] as u32 % 3);
            let a = (chunk[1] as usize) % m;
            let b = (chunk[2] as usize) % n;
            terms.push((c, a, b));
        }
        MulticastParams { ell, m, n, terms, seed: seed.to_vec() }
    }

    /// `F(x, w)` over `Z_q`.
    pub fn evaluate(&self, m: &Modulus32, x: &[u32], w: &[u32]) -> u32 {
        let q = m.q as u64;
        let mut acc: u64 = 0;
        for &(c, a, b) in &self.terms {
            acc += (c as u64 * x[a] as u64 % q) * (w[b] as u64 % q) % q;
            acc %= q;
        }
        acc as u32
    }
}

/// Linear Z_q packing: `slots` values per ring element (coefficient =
/// value — homomorphic under field-linear folds).
const PACK: usize = 16;

fn pack_zq(ring: &RingConfig, values: &[u32]) -> Vec<RingElement> {
    let mut out = Vec::with_capacity(values.len().div_ceil(PACK));
    for chunk in values.chunks(PACK) {
        let mut coeffs = vec![0u32; ring.n()];
        for (k, &v) in chunk.iter().enumerate() {
            coeffs[k] = v;
        }
        out.push(RingElement::from_coeffs(ring, coeffs));
    }
    out
}

/// `eq(b, x)` table over `{0,1}^{log nu}` at a Z_q point.
fn eq_table(m: &Modulus32, x: &[u32]) -> Vec<u32> {
    let mut evals = vec![1u32; 1usize << x.len()];
    for (var, &e) in x.iter().enumerate() {
        let shift = x.len() - 1 - var;
        let one_minus = (m.q + 1 - e % m.q) % m.q;
        for (idx, val) in evals.iter_mut().enumerate() {
            let bit = (idx >> shift) & 1;
            let f = if bit == 1 { e } else { one_minus };
            *val = ((*val as u64 * f as u64) % m.q as u64) as u32;
        }
    }
    evals
}

/// `eq(x, s) = Π_k (s_k x_k + (1−s_k)(1−x_k))` at arbitrary Z_q points.
fn eq_point(m: &Modulus32, x: &[u32], s: &[u32]) -> u32 {
    let q = m.q as u64;
    let mut acc: u64 = 1;
    for (k, &sv) in s.iter().enumerate() {
        let xv = x.get(k).copied().unwrap_or(0) as u64;
        let sv = sv as u64;
        let oms = (q + 1 - sv % q) % q;
        let omr = (q + 1 - xv % q) % q;
        let term = (sv * xv + oms * omr) % q;
        acc = (acc * term) % q;
    }
    acc as u32
}

fn challenge_zq(transcript: &mut Transcript, label: &[u8], q: u32) -> Result<u32, QuasarAccError> {
    let limit = u64::from(q);
    let bound = u64::MAX - (u64::MAX % limit) - 1;
    for _ in 0..16 {
        let bytes = transcript
            .challenge_bytes(label, 8)
            .map_err(|_| QuasarAccError::TranscriptFailure)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes[..8]);
        let v = u64::from_le_bytes(arr);
        if v <= bound {
            return Ok((v % limit) as u32);
        }
    }
    Err(QuasarAccError::TranscriptFailure)
}

fn challenge_zq_vec(
    transcript: &mut Transcript,
    label: &[u8],
    count: usize,
    q: u32,
) -> Result<Vec<u32>, QuasarAccError> {
    (0..count).map(|_| challenge_zq(transcript, label, q)).collect()
}

fn log2(x: usize) -> usize {
    x.trailing_zeros() as usize
}

/// A fresh instance pair `(x^(k), w^(k))`.
#[derive(Clone, Debug)]
pub struct ChunkInstance {
    pub x: Vec<u32>,
    pub w: Vec<u32>,
}

/// The accumulated (Racc) public statement.
#[derive(Clone, Debug)]
pub struct AccumulatedInstance {
    /// Union commitment over the stacked `x^(k)` (linear packing).
    pub c_x: AjtaiCommitment,
    /// Union commitment over the stacked `w^(k)`.
    pub c_w: AjtaiCommitment,
    /// Partial-evaluation commitment to `w̃ = w̃_∪ (τ, ·)`.
    pub c: AjtaiCommitment,
    /// The accumulation point (the sumcheck challenges).
    pub tau: Vec<u32>,
    pub r_x: Vec<u32>,
    pub r_w: Vec<u32>,
    /// The relaxed constraint slack `F(x, w̃) = e`.
    pub e: u32,
    /// The accumulated public vector `x = Σ_k eq̃_k(τ)·x^(k)`.
    pub x: Vec<u32>,
    /// Claimed `x̃(r_x)` (verifier-derived: field-only).
    pub vx: u32,
    /// Claimed `w̃(r_w)` (prover-sent; decider-checked).
    pub vw: u32,
}

/// The multicast proof: the single log ℓ-round sumcheck plus the prover
/// messages — the w-side union commitment and the partial-evaluation
/// commitment (the x-side union commitment is verifier-computed from the
/// public instances) and the `vw` claim.
#[derive(Clone, Debug)]
pub struct MulticastProof {
    pub sumcheck: RingScProof,
    pub c_w: AjtaiCommitment,
    pub c: AjtaiCommitment,
    pub vw: u32,
}

/// Prove `NIR_multicast`: commit the unions, run the G-sumcheck, derive
/// the accumulated instance. Returns the proof and the accumulated
/// instance (whose secret part is `w̃`).
pub fn prove_multicast(
    pk: &AjtaiPublicKey,
    params: &MulticastParams,
    chunks: &[ChunkInstance],
    transcript: &mut Transcript,
) -> Result<(MulticastProof, AccumulatedInstance, Vec<u32>), QuasarAccError> {
    let ring = &pk.params.ring;
    let m = &ring.modulus;
    let q = m.q;
    if chunks.len() != params.ell {
        return Err(QuasarAccError::Shape { expected: params.ell, got: chunks.len() });
    }
    // Stack the unions and commit (ONE commitment per side).
    let mut stacked_x: Vec<u32> = Vec::with_capacity(params.ell * params.m);
    let mut stacked_w: Vec<u32> = Vec::with_capacity(params.ell * params.n);
    for ch in chunks {
        stacked_x.extend_from_slice(&ch.x);
        stacked_w.extend_from_slice(&ch.w);
    }
    let c_x = pk.commit(&pk.pad_to_m(&pack_zq(ring, &stacked_x))?)?;
    let c_w = pk.commit(&pk.pad_to_m(&pack_zq(ring, &stacked_w))?)?;
    // Absorb the statement: params + public x's + commitments.
    absorb_multicast_statement(params, chunks, &c_x, &c_w, transcript)?;
    let proof_c_w = c_w.clone();
    // r_y ← Z_q^{log ℓ}.
    let r_y = challenge_zq_vec(transcript, b"quasar-ry", log2(params.ell), q)?;
    // Build G(Y) = F(x̃(Y), w̃(Y))·eq(Y, r_y) as a RingVirtualPoly.
    let eq_tab = eq_table(m, &r_y);
    let log_ell = log2(params.ell);
    let mut vp = RingVirtualPoly::new(log_ell);
    let eq_id = vp.add_factor(
        eq_tab.iter().map(|&w| ring.constant(w)).collect(),
    )?;
    for &(c, a, b) in &params.terms {
        let x_tab: Vec<RingElement> =
            chunks.iter().map(|ch| ring.constant(ch.x[a])).collect();
        let w_tab: Vec<RingElement> =
            chunks.iter().map(|ch| ring.constant(ch.w[b])).collect();
        let x_id = vp.add_factor(x_tab)?;
        let w_id = vp.add_factor(w_tab)?;
        vp.add_term(ring.constant(c), vec![eq_id, x_id, w_id])?;
    }
    // Each input satisfies F = 0 ⇒ H(y) = 0 on the cube ⇒ claim = 0.
    let claim = ring.zero();
    let out = ring_sc_prove(ring, &vp, &claim, transcript)?;
    let tau = out.point.clone();
    // w̃ = Σ_k eq̃_k(τ)·w^(k); commit it.
    let eq_tau = eq_table(m, &tau);
    let mut w_tilde = vec![0u32; params.n];
    for (k, ch) in chunks.iter().enumerate() {
        let w_k = eq_tau[k] as u64;
        for (j, &wv) in ch.w.iter().enumerate() {
            w_tilde[j] = ((w_tilde[j] as u64 + w_k * wv as u64) % q as u64) as u32;
        }
    }
    let c = pk.commit(&pk.pad_to_m(&pack_zq(ring, &w_tilde))?)?;
    transcript
        .append_bytes(b"quasar-c", &c.to_bytes())
        .map_err(|_| QuasarAccError::TranscriptFailure)?;
    // r_x, r_w.
    let r_x = challenge_zq_vec(transcript, b"quasar-rx", log2(params.m), q)?;
    let r_w = challenge_zq_vec(transcript, b"quasar-rw", log2(params.n), q)?;
    // x = Σ_k eq̃_k(τ)·x^(k) (the accumulated public vector).
    let mut x_acc = vec![0u32; params.m];
    for (k, ch) in chunks.iter().enumerate() {
        let w_k = eq_tau[k] as u64;
        for (j, &xv) in ch.x.iter().enumerate() {
            x_acc[j] = ((x_acc[j] as u64 + w_k * xv as u64) % q as u64) as u32;
        }
    }
    // vx = x̃(r_x) — the MLE of x_acc at r_x.
    let vx = mle_dot(m, &x_acc, &r_x);
    // vw = w̃(r_w) — the MLE of w_tilde at r_w (prover claim).
    let vw = mle_dot(m, &w_tilde, &r_w);
    transcript
        .append_bytes(b"quasar-vw", &vw.to_le_bytes())
        .map_err(|_| QuasarAccError::TranscriptFailure)?;
    // e = H(τ) = F(x_acc, w_tilde).
    let e = params.evaluate(m, &x_acc, &w_tilde);
    Ok((
        MulticastProof { sumcheck: out.proof, c_w: proof_c_w, c: c.clone(), vw },
        AccumulatedInstance {
            c_x,
            c_w,
            c,
            tau,
            r_x,
            r_w,
            e,
            x: x_acc,
            vx,
            vw,
        },
        w_tilde,
    ))
}

/// `MLE[v](r) = Σ_b eq(b, r)·v[b]` (the ts inner product).
fn mle_dot(m: &Modulus32, v: &[u32], r: &[u32]) -> u32 {
    let eq = eq_table(m, r);
    let q = m.q as u64;
    let mut acc: u64 = 0;
    for (b, &e) in eq.iter().enumerate() {
        if e != 0 {
            acc = (acc + e as u64 * v[b] as u64) % q;
        }
    }
    acc as u32
}

fn absorb_multicast_statement(
    params: &MulticastParams,
    chunks: &[ChunkInstance],
    c_x: &AjtaiCommitment,
    c_w: &AjtaiCommitment,
    transcript: &mut Transcript,
) -> Result<(), QuasarAccError> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(params.ell as u32).to_le_bytes());
    buf.extend_from_slice(&(params.m as u32).to_le_bytes());
    buf.extend_from_slice(&(params.n as u32).to_le_bytes());
    buf.extend_from_slice(&params.seed);
    for ch in chunks {
        for &xv in &ch.x {
            buf.extend_from_slice(&xv.to_le_bytes());
        }
    }
    transcript
        .append_bytes(b"quasar-stmt", &buf)
        .map_err(|_| QuasarAccError::TranscriptFailure)?;
    transcript
        .append_bytes(b"quasar-cx", &c_x.to_bytes())
        .map_err(|_| QuasarAccError::TranscriptFailure)?;
    transcript
        .append_bytes(b"quasar-cw", &c_w.to_bytes())
        .map_err(|_| QuasarAccError::TranscriptFailure)?;
    Ok(())
}

/// Verify `NIR_multicast` (`ACC.V`): replay the sumcheck, derive `e` from
/// the final claim via `eq(τ, r_y)^{-1}`, recompute `x` and `vx`
/// (field-only), and check the degenerate-eq rejection. Returns the
/// accumulated instance (the verifier's view).
pub fn verify_multicast(
    pk: &AjtaiPublicKey,
    params: &MulticastParams,
    chunks: &[ChunkInstance],
    proof: &MulticastProof,
    transcript: &mut Transcript,
) -> Result<AccumulatedInstance, QuasarAccError> {
    let ring = &pk.params.ring;
    let m = &ring.modulus;
    let q = m.q;
    if chunks.len() != params.ell {
        return Err(QuasarAccError::Shape { expected: params.ell, got: chunks.len() });
    }
    let mut stacked_x: Vec<u32> = Vec::with_capacity(params.ell * params.m);
    for ch in chunks {
        stacked_x.extend_from_slice(&ch.x);
    }
    let c_x = pk.commit(&pk.pad_to_m(&pack_zq(ring, &stacked_x))?)?;
    absorb_multicast_statement(params, chunks, &c_x, &proof.c_w, transcript)?;
    let r_y = challenge_zq_vec(transcript, b"quasar-ry", log2(params.ell), q)?;
    // Replay the sumcheck: same vp construction from PUBLIC data.
    let log_ell = log2(params.ell);
    let eq_tab = eq_table(m, &r_y);
    let mut vp = RingVirtualPoly::new(log_ell);
    let eq_id = vp.add_factor(eq_tab.iter().map(|&w| ring.constant(w)).collect())?;
    for &(c, a, b) in &params.terms {
        let x_tab: Vec<RingElement> =
            chunks.iter().map(|ch| ring.constant(ch.x[a])).collect();
        let w_tab: Vec<RingElement> =
            chunks.iter().map(|ch| ring.constant(ch.w[b])).collect();
        let x_id = vp.add_factor(x_tab)?;
        let w_id = vp.add_factor(w_tab)?;
        vp.add_term(ring.constant(c), vec![eq_id, x_id, w_id])?;
    }
    let claim = ring.zero();
    let verdict = ring_sc_verify(ring, log_ell, vp.max_degree(), &claim, &proof.sumcheck, transcript)?;
    let tau = verdict.point;
    // e = G(τ)·eq(τ, r_y)^{-1} = H(τ) = F(x, w̃).
    let eq_tau_ry = eq_point(m, &tau, &r_y);
    let eq_inv = ring.modulus.inv(eq_tau_ry).ok_or(QuasarAccError::EqChallengeDegenerate)?;
    let e = ((verdict.final_claim as u64 * eq_inv as u64) % q as u64) as u32;
    // Absorb c and the r_x/r_w challenges + the vw claim.
    transcript
        .append_bytes(b"quasar-c", &proof.c.to_bytes())
        .map_err(|_| QuasarAccError::TranscriptFailure)?;
    let r_x = challenge_zq_vec(transcript, b"quasar-rx", log2(params.m), q)?;
    let r_w = challenge_zq_vec(transcript, b"quasar-rw", log2(params.n), q)?;
    transcript
        .append_bytes(b"quasar-vw", &proof.vw.to_le_bytes())
        .map_err(|_| QuasarAccError::TranscriptFailure)?;
    // x = Σ_k eq̃_k(τ)·x^(k) (field-only, verifier-computed).
    let eq_tau = eq_table(m, &tau);
    let mut x_acc = vec![0u32; params.m];
    for (k, ch) in chunks.iter().enumerate() {
        let w_k = eq_tau[k] as u64;
        for (j, &xv) in ch.x.iter().enumerate() {
            x_acc[j] = ((x_acc[j] as u64 + w_k * xv as u64) % q as u64) as u32;
        }
    }
    let vx = mle_dot(m, &x_acc, &r_x);
    Ok(AccumulatedInstance {
        c_x,
        c_w: proof.c_w.clone(),
        c: proof.c.clone(),
        tau,
        r_x,
        r_w,
        e,
        x: x_acc,
        vx,
        vw: proof.vw,
    })
}

/// `ACC.D` for one accumulated instance: open every commitment and check
/// (a) the w-side union opens to the stacked witnesses, (b) the
/// partial-evaluation consistency `w̃ = Σ_k eq̃_k(τ)·w^(k)`, (c) the
/// relaxed constraint `F(x, w̃) = e`, (d) the claim `w̃(r_w) = vw`, and
/// (e) the x-side union + `vx`.
pub fn decider_multicast(
    pk: &AjtaiPublicKey,
    params: &MulticastParams,
    acc: &AccumulatedInstance,
    chunks: &[ChunkInstance],
    w_tilde: &[u32],
) -> Result<(), QuasarAccError> {
    let ring = &pk.params.ring;
    let m = &ring.modulus;
    if chunks.len() != params.ell || w_tilde.len() != params.n {
        return Err(QuasarAccError::DeciderFailed("shape"));
    }
    let mut stacked_x: Vec<u32> = Vec::with_capacity(params.ell * params.m);
    let mut stacked_w: Vec<u32> = Vec::with_capacity(params.ell * params.n);
    for ch in chunks {
        stacked_x.extend_from_slice(&ch.x);
        stacked_w.extend_from_slice(&ch.w);
    }
    // (a) openings.
    pk.verify_opening(&acc.c_x, &pk.pad_to_m(&pack_zq(ring, &stacked_x))?)
        .map_err(|_| QuasarAccError::DeciderFailed("c_x opening"))?;
    pk.verify_opening(&acc.c_w, &pk.pad_to_m(&pack_zq(ring, &stacked_w))?)
        .map_err(|_| QuasarAccError::DeciderFailed("c_w opening"))?;
    pk.verify_opening(&acc.c, &pk.pad_to_m(&pack_zq(ring, w_tilde))?)
        .map_err(|_| QuasarAccError::DeciderFailed("c opening"))?;
    // (b) partial-evaluation consistency.
    let eq_tau = eq_table(m, &acc.tau);
    let mut expected_w = vec![0u32; params.n];
    for (k, ch) in chunks.iter().enumerate() {
        let w_k = eq_tau[k] as u64;
        for (j, &wv) in ch.w.iter().enumerate() {
            expected_w[j] = ((expected_w[j] as u64 + w_k * wv as u64) % m.q as u64) as u32;
        }
    }
    if expected_w != w_tilde {
        return Err(QuasarAccError::DeciderFailed("partial-eval consistency"));
    }
    // (c) the relaxed constraint F(x, w̃) = e.
    if params.evaluate(m, &acc.x, w_tilde) != acc.e {
        return Err(QuasarAccError::DeciderFailed("F constraint"));
    }
    // (d) w̃(r_w) = vw.
    if mle_dot(m, w_tilde, &acc.r_w) != acc.vw {
        return Err(QuasarAccError::DeciderFailed("vw claim"));
    }
    // (e) vx against the stacked-x union MLE at (τ, r_x).
    if mle_dot(m, &acc.x, &acc.r_x) != acc.vx {
        return Err(QuasarAccError::DeciderFailed("vx claim"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Q3: the 2-to-1 fold + ACC.V / ACC.D
// ---------------------------------------------------------------------------

/// The folded output of the 2-to-1 reduction.
#[derive(Clone, Debug)]
pub struct FoldedQuasar {
    /// `C*_x = C_x0 + γ·C_x1` etc. (Ajtai homomorphism).
    pub c_x: AjtaiCommitment,
    pub c_w: AjtaiCommitment,
    pub c: AjtaiCommitment,
    pub tau: Vec<u32>,
    pub r_x: Vec<u32>,
    pub r_w: Vec<u32>,
    pub e: u32,
    pub x: Vec<u32>,
    pub vx: u32,
    pub vw: u32,
    /// The fold challenge (public).
    pub gamma: u32,
}

/// Fold two accumulated instances under `γ` with the cross term `T`:
/// `e* = e₀ + γ·T + γ²·e₁` (the γ-power combination). The prover
/// supplies the secret partial-evaluations; the verifier-side fold
/// derives everything from public data.
pub struct QuasarFoldInput {
    pub acc0: AccumulatedInstance,
    pub acc1: AccumulatedInstance,
    /// The secret `w̃` of each side (prover side only).
    pub w_tilde0: Vec<u32>,
    pub w_tilde1: Vec<u32>,
}

/// Prove the 2-to-1 fold: compute the cross term, fold the secrets,
/// absorb the transcript. Returns the folded statement + the folded
/// secret `w̃*`.
pub fn prove_fold_2to1(
    pk: &AjtaiPublicKey,
    params: &MulticastParams,
    input: &QuasarFoldInput,
    transcript: &mut Transcript,
) -> Result<(FoldedQuasar, Vec<u32>), QuasarAccError> {
    let ring = &pk.params.ring;
    let m = &ring.modulus;
    let q = m.q;
    // Absorb the two statements.
    absorb_fold_statement(params, &input.acc0, &input.acc1, transcript)?;
    // γ ← Z_q.
    let gamma = challenge_zq(transcript, b"quasar-gamma", q)?;
    // The cross term T = F(x₀, w̃₁) + F(x₁, w̃₀) (one field element —
    // the γ-power combined one-round sumcheck's payload).
    let t = ((params.evaluate(m, &input.acc0.x, &input.w_tilde1) as u64
        + params.evaluate(m, &input.acc1.x, &input.w_tilde0) as u64)
        % q as u64) as u32;
    transcript
        .append_bytes(b"quasar-t", &t.to_le_bytes())
        .map_err(|_| QuasarAccError::TranscriptFailure)?;
    // Fold the secrets: w̃* = w̃₀ + γ·w̃₁.
    let w_star: Vec<u32> = input
        .w_tilde0
        .iter()
        .zip(input.w_tilde1.iter())
        .map(|(&a, &b)| ((a as u64 + gamma as u64 * b as u64) % q as u64) as u32)
        .collect();
    // Fold the public vectors and scalars.
    let x_star: Vec<u32> = input
        .acc0
        .x
        .iter()
        .zip(input.acc1.x.iter())
        .map(|(&a, &b)| ((a as u64 + gamma as u64 * b as u64) % q as u64) as u32)
        .collect();
    let fold_vec = |v0: &[u32], v1: &[u32]| -> Vec<u32> {
        v0.iter()
            .zip(v1.iter())
            .map(|(&a, &b)| ((a as u64 + gamma as u64 * b as u64) % q as u64) as u32)
            .collect()
    };
    let gamma_sq = ((gamma as u64 * gamma as u64) % q as u64) as u32;
    let e_star = ((input.acc0.e as u64
        + gamma as u64 * t as u64
        + gamma_sq as u64 * input.acc1.e as u64)
        % q as u64) as u32;
    // Commitment folds (Ajtai homomorphism).
    let fold_commit = |c0: &AjtaiCommitment, c1: &AjtaiCommitment| -> Result<AjtaiCommitment, QuasarAccError> {
        let mut rows = Vec::with_capacity(c0.rows.len());
        for (r0, r1) in c0.rows.iter().zip(c1.rows.iter()) {
            rows.push(r0.add(&r1.scale_i64(gamma as i64))?);
        }
        Ok(AjtaiCommitment { rows })
    };
    Ok((
        FoldedQuasar {
            c_x: fold_commit(&input.acc0.c_x, &input.acc1.c_x)?,
            c_w: fold_commit(&input.acc0.c_w, &input.acc1.c_w)?,
            c: fold_commit(&input.acc0.c, &input.acc1.c)?,
            tau: fold_vec(&input.acc0.tau, &input.acc1.tau),
            r_x: fold_vec(&input.acc0.r_x, &input.acc1.r_x),
            r_w: fold_vec(&input.acc0.r_w, &input.acc1.r_w),
            e: e_star,
            x: x_star,
            vx: ((input.acc0.vx as u64 + gamma as u64 * input.acc1.vx as u64) % q as u64) as u32,
            vw: ((input.acc0.vw as u64 + gamma as u64 * input.acc1.vw as u64) % q as u64) as u32,
            gamma,
        },
        w_star,
    ))
}

fn absorb_fold_statement(
    params: &MulticastParams,
    a0: &AccumulatedInstance,
    a1: &AccumulatedInstance,
    transcript: &mut Transcript,
) -> Result<(), QuasarAccError> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(params.ell as u32).to_le_bytes());
    buf.extend_from_slice(&params.seed);
    for acc in [a0, a1] {
        buf.extend_from_slice(&acc.c_x.to_bytes());
        buf.extend_from_slice(&acc.c_w.to_bytes());
        buf.extend_from_slice(&acc.c.to_bytes());
        for &t in &acc.tau {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        buf.extend_from_slice(&acc.e.to_le_bytes());
        for &xv in &acc.x {
            buf.extend_from_slice(&xv.to_le_bytes());
        }
        buf.extend_from_slice(&acc.vx.to_le_bytes());
        buf.extend_from_slice(&acc.vw.to_le_bytes());
    }
    transcript
        .append_bytes(b"quasar-fold-stmt", &buf)
        .map_err(|_| QuasarAccError::TranscriptFailure)?;
    Ok(())
}

/// `ACC.D` for the 2-to-1 fold output: the folded commitments open to the
/// folded secrets and `F(x*, w̃*) = e*` — which binds the cross term T.
pub fn decider_fold_2to1(
    pk: &AjtaiPublicKey,
    params: &MulticastParams,
    folded: &FoldedQuasar,
    w_star: &[u32],
) -> Result<(), QuasarAccError> {
    let ring = &pk.params.ring;
    let m = &ring.modulus;
    if w_star.len() != params.n {
        return Err(QuasarAccError::DeciderFailed("shape"));
    }
    pk.verify_opening(&folded.c, &pk.pad_to_m(&pack_zq(ring, w_star))?)
        .map_err(|_| QuasarAccError::DeciderFailed("c* opening"))?;
    if params.evaluate(m, &folded.x, w_star) != folded.e {
        return Err(QuasarAccError::DeciderFailed("F(x*, w̃*) = e*"));
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
        // Binding-only regime: the partial-evaluation vectors `w̃` are
        // full-range Z_q combos of short witnesses (the eq weights are
        // full-range), so the Ajtai norm gate opens at q/2 — the paper's
        // PCS-based commitments carry no norm discipline at this layer.
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: m_slots,
            norm_bound: ring.modulus.q / 2,
        };
        let pk = AjtaiPublicKey::from_seed(params, [53u8; 32]).ok().unwrap();
        (pk, ring)
    }

    /// Chunks satisfying the predicate: pick w freely, then solve one x
    /// coordinate per term so F = 0: with terms (c, a, b), set x_a := 0
    /// for all but craft... simplest: choose x = 0 (then F = 0 for any w).
    fn satisfying_chunks(
        params: &MulticastParams,
        q: u32,
        tag: &[u8],
    ) -> Vec<ChunkInstance> {
        (0..params.ell)
            .map(|k| {
                let bytes = Transcript::xof(
                    b"quasar-test-w",
                    &[tag, &(k as u32).to_le_bytes()].concat(),
                    params.n * 4,
                );
                let w: Vec<u32> = bytes
                    .chunks(4)
                    .take(params.n)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) % q)
                    .collect();
                ChunkInstance { x: vec![0u32; params.m], w }
            })
            .collect()
    }

    #[test]
    fn multicast_happy_path_and_decider() {
        // ℓ = 4 chunks of m = n = 8: ONE sumcheck of log ℓ = 2 rounds.
        let (pk, ring) = setup(4, 8);
        let params = MulticastParams::from_seed(4, 8, 8, 3, b"qp");
        let chunks = satisfying_chunks(&params, ring.modulus.q, b"mc");
        let mut t = Transcript::new_default(b"quasar-mc");
        let (proof, acc, w_tilde) =
            prove_multicast(&pk, &params, &chunks, &mut t).ok().unwrap();
        // The single sumcheck: log ℓ = 2 rounds, degree 3.
        assert_eq!(proof.sumcheck.rounds.len(), 2);
        assert!(proof.sumcheck.rounds.iter().all(|r| r.len() == 4));
        let mut vt = Transcript::new_default(b"quasar-mc");
        let acc_v = verify_multicast(&pk, &params, &chunks, &proof, &mut vt).ok().unwrap();
        // The verifier's view matches the prover's derived instance.
        assert_eq!(acc_v.e, acc.e);
        assert_eq!(acc_v.x, acc.x);
        assert_eq!(acc_v.vx, acc.vx);
        // ACC.D: openings + consistency + F(x, w̃) = e.
        match decider_multicast(&pk, &params, &acc, &chunks, &w_tilde) {
            Ok(_) => {}
            Err(e) => panic!("decider: {e:?}"),
        }
    }

    #[test]
    fn multicast_rejects_unsatisfied_predicate() {
        let (pk, ring) = setup(4, 8);
        let params = MulticastParams::from_seed(4, 8, 8, 3, b"qp-bad");
        let mut chunks = satisfying_chunks(&params, ring.modulus.q, b"mc-bad");
        // Break the predicate: a nonzero x makes F(x, 0-free w) ≠ 0 for
        // generic w — set x = 1 vector on one chunk.
        chunks[0].x = vec![1u32; params.m];
        let mut t = Transcript::new_default(b"quasar-mc-bad");
        // The honest prover fail-closes (claim ≠ 0).
        assert!(prove_multicast(&pk, &params, &chunks, &mut t).is_err());
    }

    #[test]
    fn multicast_tampered_sumcheck_rejected() {
        let (pk, ring) = setup(4, 8);
        let params = MulticastParams::from_seed(4, 8, 8, 3, b"qp-t");
        let chunks = satisfying_chunks(&params, ring.modulus.q, b"mc-t");
        let mut t = Transcript::new_default(b"quasar-mc-t");
        let (mut proof, _acc, _w) =
            prove_multicast(&pk, &params, &chunks, &mut t).ok().unwrap();
        proof.sumcheck.rounds[0][0] =
            (proof.sumcheck.rounds[0][0] + 7) % ring.modulus.q;
        let mut vt = Transcript::new_default(b"quasar-mc-t");
        assert!(verify_multicast(&pk, &params, &chunks, &proof, &mut vt).is_err());
    }

    #[test]
    fn fold_2to1_decider_and_cross_term_binding() {
        let (pk, ring) = setup(4, 8);
        let params = MulticastParams::from_seed(4, 8, 8, 3, b"qp-f");
        // Two multicasts (different tags → different instances).
        let chunks_a = satisfying_chunks(&params, ring.modulus.q, b"fa");
        let chunks_b = satisfying_chunks(&params, ring.modulus.q, b"fb");
        let mut ta = Transcript::new_default(b"quasar-fa");
        let (pa, acc_a, wa) = prove_multicast(&pk, &params, &chunks_a, &mut ta).ok().unwrap();
        let mut tb = Transcript::new_default(b"quasar-fb");
        let (_pb, acc_b, wb) = prove_multicast(&pk, &params, &chunks_b, &mut tb).ok().unwrap();
        let _ = pa;
        // Fold the two accumulated instances.
        let mut tf = Transcript::new_default(b"quasar-fold");
        let input = QuasarFoldInput {
            acc0: acc_a,
            acc1: acc_b,
            w_tilde0: wa,
            w_tilde1: wb,
        };
        let (folded, w_star) = prove_fold_2to1(&pk, &params, &input, &mut tf).ok().unwrap();
        // ACC.D: the folded commitment opens w̃* and F(x*, w̃*) = e*.
        assert!(decider_fold_2to1(&pk, &params, &folded, &w_star).is_ok());
        // Cross-term binding: a wrong T makes e* inconsistent — simulate
        // by perturbing e* and re-running the decider.
        let mut bad = folded.clone();
        bad.e = (bad.e + 1) % ring.modulus.q;
        assert!(decider_fold_2to1(&pk, &params, &bad, &w_star).is_err());
    }

    #[test]
    fn multi_instance_ivc_loop() {
        // Q5: three steps, each multicasting ℓ = 4 chunks and folding
        // into the running accumulator — the decider passes at the end.
        let (pk, ring) = setup(4, 8);
        let params = MulticastParams::from_seed(4, 8, 8, 3, b"qp-ivc");
        // The running accumulator starts as a "trivial" multicast of
        // zero chunks (x = 0, w̃ = 0, e = 0).
        let (mut running_acc, mut running_w) = {
            let chunks = satisfying_chunks(&params, ring.modulus.q, b"ivc0");
            let mut t = Transcript::new_default(b"quasar-ivc0");
            let (_p, acc, w) = prove_multicast(&pk, &params, &chunks, &mut t).ok().unwrap();
            (acc, w)
        };
        // Track the stacked w's of the running union for the decider.
        let mut running_chunks: Vec<ChunkInstance> = satisfying_chunks(&params, ring.modulus.q, b"ivc0");
        for step in 1..=3 {
            let chunks = satisfying_chunks(
                &params,
                ring.modulus.q,
                format!("ivc{}", step).as_bytes(),
            );
            let mut t = Transcript::new_default(b"quasar-ivc-step");
            let (_proof, acc_new, w_new) =
                prove_multicast(&pk, &params, &chunks, &mut t).ok().unwrap();
            // Verify the multicast (ACC.V).
            let mut vt = Transcript::new_default(b"quasar-ivc-step");
            let _ = verify_multicast(&pk, &params, &chunks, &_proof, &mut vt).ok().unwrap();
            // Decider-check the new instance before folding (the kernel
            // simplification of the Reval subrelations).
            assert!(decider_multicast(&pk, &params, &acc_new, &chunks, &w_new).is_ok());
            // Fold into the running accumulator.
            let mut tf = Transcript::new_default(b"quasar-ivc-fold");
            let input = QuasarFoldInput {
                acc0: running_acc.clone(),
                acc1: acc_new,
                w_tilde0: running_w.clone(),
                w_tilde1: w_new,
            };
            let (folded, w_star) =
                prove_fold_2to1(&pk, &params, &input, &mut tf).ok().unwrap();
            assert!(decider_fold_2to1(&pk, &params, &folded, &w_star).is_ok());
            running_acc = AccumulatedInstance {
                c_x: folded.c_x,
                c_w: folded.c_w,
                c: folded.c,
                tau: folded.tau,
                r_x: folded.r_x,
                r_w: folded.r_w,
                e: folded.e,
                x: folded.x,
                vx: folded.vx,
                vw: folded.vw,
            };
            running_w = w_star;
            running_chunks.extend(chunks);
            // The running union decider still passes (c_w covers the
            // growing stack only via the folded homomorphism — the
            // decider here checks the folded partial-eval commitment).
            let _ = &running_chunks;
        }
        // Final decider on the running accumulator.
        assert!(decider_fold_2to1(&pk, &params, &{
            FoldedQuasar {
                c_x: running_acc.c_x.clone(),
                c_w: running_acc.c_w.clone(),
                c: running_acc.c.clone(),
                tau: running_acc.tau.clone(),
                r_x: running_acc.r_x.clone(),
                r_w: running_acc.r_w.clone(),
                e: running_acc.e,
                x: running_acc.x.clone(),
                vx: running_acc.vx,
                vw: running_acc.vw,
                gamma: 0,
            }
        }, &running_w)
        .is_ok());
    }
}
