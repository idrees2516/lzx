//! **TTRP — the tensor-train random-projection shortness check** (Geng &
//! Plançon, ePrint 2026/2146) + **the digit-free opening via the
//! compact-mode linear-functional bridge** — the claim ledger's norm-check
//! module replacing the JL/digit routes.
//!
//! # What replaces what
//!
//! * The **JL projection route** (unstructured `M ← D_{ghl}^{256×m̃φ}` in
//!   LaBRADOR; the structured block-diagonal projections in RoKoko — and
//!   this workspace's per-round JL blocks in `hyperwolf_compact`): the
//!   verifier materializes O(λ²)–O(ρλ²) ring elements and the prover
//!   either transmits the projection or **commits to it** (RoKoko's
//!   projection commitments — the size driver TTRP eliminates).
//! * The **digit route** (this ledger's `compact_norm_proof`: base-256
//!   gadget digits per coefficient, transmitted in the clear —
//!   Θ(N) proof): the TTRP terminal is a single **evaluation claim**
//!   `w_r = mler(x)(r)` opened through the compact-mode linear
//!   functional — **no digits at all**.
//!
//! # The protocol (Π^TTRP = Π^sc ∘ Π^0, Figure 1 + §4.2)
//!
//! 1. **Cores** — the verifier derives `{M_i^{(j)}}` (k rows × μ layers,
//!    internal rank c, core dimension d = 2^ℓ) from the transcript —
//!    O(μ·k·c²·d) seed material (the paper's `D_ghl` is realized as the
//!    ternary {−1, 0, 1} stand-in, documented below).
//! 2. **Projection** — the prover sends `y_j = ⟨TT_j, x⟩` (the
//!    right-to-left tensor contraction over Z, §Lemma 6's prover route).
//!    The verifier checks `‖y‖₂ ≤ B̂` — the completeness bound; a
//!    violating witness (‖x‖∞ ≥ B) survives with the paper's
//!    second-moment probability (Theorems 2/3).
//! 3. **Linearization (Π^0 + Π^sc)** — the well-formedness of the
//!    projection becomes ONE sum-check over the coefficient cube:
//!    `Σ_z Σ_j γ^j · mler_{TT_j}(z)·mler_x(z) = Σ_j γ^j · y_j`
//!    (degree 2, ν = log₂|x| variables). The terminal reduces to the
//!    **eval-claim API**: the point `r` + `w_r = mler_x(r)` — appended
//!    to the claim ledger exactly like every other eval claim.
//! 4. **The digit-free opening** — `w_r` is authenticated against the
//!    Ajtai commitment through the compact-mode bridge: the byte-packed
//!    r-aligned columns (the existing `compact.rs` layout discipline),
//!    the column-uniform shadow functional `Ψ(m) = eq_head(h(m))·2^{8b}`,
//!    the per-column values `ũ_j` (transmitted BEFORE the challenges),
//!    the scalar-challenge integer fold `v = Σ_j d_j·w_j`, and the three
//!    checks — (a) `Φ(v) = Σ_j d_j·ũ_j` (the Goldilocks functional
//!    commutes through the fold), (b) `w_r = Σ_j μ_j·ũ_j` (the MLE
//!    interpolation closes the claim), (c) `A·v = Σ_j d_j·y_j` (the Ajtai
//!    binding — the verifier's own `commit(v)`) — plus the fail-closed
//!    norm gate `r·A·255 < q/2`. Extraction terminates in MSIS on
//!    `[A | −y]` at the relaxed bound (the compact-mode argument).
//!
//! # Honest deviations
//!
//! * `D_ghl` (the small discrete Gaussian) is realized as the ternary
//!   {−1, 0, 1} distribution with p(0) = 1/2 — the second-moment
//!   structure the proofs need; the Gaussian instantiation is a
//!   drop-in parameter change.
//! * The sum-check runs over **Goldilocks** (the two-characteristic
//!   discipline: the projection is computed over Z, its images are
//!   exact in both fields at the small-value regime) instead of the
//!   paper's ring sum-check over R_q.
//! * The verifier's row evaluation uses the direct TT contraction at
//!   the challenge point (Lemma 6's O(μ₁c²d) route is the optimization;
//!   our d = 2^ℓ chunking makes every core spatial, μ₂ = 0).
//! * The response `v` transmits raw ring coefficients (the rANS coder
//!   from `compact.rs` wires in at the pipeline level).
//! * Γ-batching (the paper's k' = Ω(λ/log q) rows) is realized as the
//!   γ-power combination over the k rows (Goldilocks challenges); the
//!   "Lift-and-Batch" O(λ) refinement is the documented follow-up.

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiPublicKey};
use lattice_core::mle::{DenseMle, MleError};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::Goldilocks;
use lattice_ring::RingElement;
use lattice_sumcheck::sumcheck::{self, SumcheckProof};
use lattice_sumcheck::virtual_poly::VirtualPolynomial;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtrpError {
    Shape { expected: usize, got: usize },
    /// The projected norm exceeded the completeness bound.
    NormExceeded { norm2: u64, bound: u64 },
    /// The sum-check layer rejected.
    Sumcheck(String),
    /// The final claim check failed.
    FinalCheckFailed,
    /// The functional-opening checks failed.
    BridgeFailed(&'static str),
    Transcript(TranscriptError),
}

impl From<TranscriptError> for TtrpError {
    fn from(e: TranscriptError) -> Self {
        TtrpError::Transcript(e)
    }
}

impl core::fmt::Display for TtrpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TtrpError::Shape { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            TtrpError::NormExceeded { norm2, bound } => {
                write!(f, "projected ℓ2 norm {norm2} > bound {bound}")
            }
            TtrpError::Sumcheck(e) => write!(f, "sum-check: {e}"),
            TtrpError::FinalCheckFailed => write!(f, "final claim check failed"),
            TtrpError::BridgeFailed(step) => write!(f, "functional bridge failed at: {step}"),
            TtrpError::Transcript(e) => write!(f, "transcript: {e}"),
        }
    }
}

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

// ---------------------------------------------------------------------------
// The tensor-train cores.

/// The TTRP parameters: `mu` layers, core dimension `d = 2^{log_d}` (the
/// spatial chunking of the coefficient cube), internal rank `c`, and `k`
/// projection rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TtrpParams {
    pub mu: usize,
    pub log_d: usize,
    pub rank: usize,
    pub k: usize,
}

impl TtrpParams {
    /// The parameters covering a coefficient vector of length `n`
    /// (padded to `d^mu ≥ n`).
    pub fn for_len(n: usize, rank: usize, k: usize, log_d: usize) -> Self {
        let d = 1usize << log_d;
        let mut mu = 1;
        while d.pow(mu as u32) < n {
            mu += 1;
        }
        TtrpParams { mu, log_d, rank, k }
    }

    pub fn d(&self) -> usize {
        1usize << self.log_d
    }

    /// The padded coefficient length `d^mu`.
    pub fn padded_len(&self) -> usize {
        self.d().pow(self.mu as u32)
    }

    /// The number of cube variables `ν = log2(d^mu)`.
    pub fn num_vars(&self) -> usize {
        self.log_d * self.mu
    }
}

/// The derived TT cores (the verifier's public material): for each row
/// `j ∈ [k]` and layer `i ∈ [mu]`, a `c_{i-1} × d × c_i` tensor of
/// ternary entries, flattened row-major.
#[derive(Clone, Debug)]
pub struct TtCores {
    pub params: TtrpParams,
    /// `cores[j][i]` — the flattened core tensor (row-major:
    /// `[a][z][b]` at `a·d·c + z·c + b`).
    pub cores: Vec<Vec<Vec<i64>>>,
}

impl TtCores {
    /// The layer rank bookkeeping: `r_0 = 1`, `r_mu = 1`, interior = c.
    fn layer_ranks(params: &TtrpParams, i: usize) -> (usize, usize) {
        let c = params.rank;
        let r_in = if i == 0 { 1 } else { c };
        let r_out = if i + 1 == params.mu { 1 } else { c };
        (r_in, r_out)
    }

    /// Sample the cores from the transcript (the ternary `D_ghl`
    /// stand-in: p(−1) = p(1) = 1/4, p(0) = 1/2), flattened per layer
    /// as `[a][z][b] → a·d·r_out + z·r_out + b`.
    pub fn sample(params: &TtrpParams, transcript: &mut Transcript) -> Result<Self, TtrpError> {
        let d = params.d();
        let mut cores = Vec::with_capacity(params.k);
        for _j in 0..params.k {
            let mut row = Vec::with_capacity(params.mu);
            for i in 0..params.mu {
                let (r_in, r_out) = Self::layer_ranks(params, i);
                let total = r_in * d * r_out;
                let bytes = transcript
                    .challenge_bytes(b"ttrp-core", total)
                    .map_err(TtrpError::Transcript)?;
                let core: Vec<i64> = bytes
                    .iter()
                    .map(|&b| match b & 3 {
                        0 | 1 => 0i64,
                        2 => -1i64,
                        _ => 1i64,
                    })
                    .collect();
                row.push(core);
            }
            cores.push(row);
        }
        Ok(TtCores { params: params.clone(), cores })
    }

    /// The right-to-left tensor contraction (the prover's projection):
    /// `y_j = ⟨TT_j, x⟩` over the integers (Lemma 6's prover route).
    /// State invariant before contracting layer `i`: positions `d^{i+1}`
    /// × rank `r_{i+1}`; the layer consumes rank `r_{i+1}` and produces
    /// `r_i`, dropping one position digit.
    pub fn project(&self, x: &[i64]) -> Vec<i64> {
        let p = &self.params;
        let d = p.d();
        let n = p.padded_len();
        let mut xp = vec![0i64; n];
        for (i, &xi) in x.iter().enumerate() {
            if i < n {
                xp[i] = xi;
            }
        }
        let mut out = Vec::with_capacity(p.k);
        for row in &self.cores {
            let mut state = xp.clone();
            let mut pos_count = n; // d^mu
            let mut state_rank = 1usize; // r_mu
            for i in (0..p.mu).rev() {
                let (r_in, r_out) = Self::layer_ranks(p, i);
                debug_assert_eq!(state_rank, r_out);
                let core = &row[i];
                let new_pos = pos_count / d;
                let mut next = vec![0i64; new_pos * r_in];
                for pp in 0..new_pos {
                    for z in 0..d {
                        for b in 0..r_out {
                            let sv = state[(pp * d + z) * r_out + b];
                            if sv == 0 {
                                continue;
                            }
                            for a in 0..r_in {
                                let cv = core[a * d * r_out + z * r_out + b];
                                if cv == 0 {
                                    continue;
                                }
                                next[pp * r_in + a] += sv * cv;
                            }
                        }
                    }
                }
                state = next;
                pos_count = new_pos;
                state_rank = r_in;
            }
            out.push(state[0]);
        }
        out
    }

    /// The full truth table of row `j`'s TT matrix (the sum-check
    /// factor; the left-to-right contraction, O(d^mu·c²)).
    pub fn row_table(&self, j: usize) -> Vec<Goldilocks> {
        let p = &self.params;
        let d = p.d();
        let row = &self.cores[j];
        // state: positions d^i × rank r_i, flattened pos·r_i + a.
        let mut state: Vec<i64> = vec![1]; // i = 0: one position, r_0 = 1
        let mut pos_count = 1usize;
        let mut state_rank = 1usize;
        for i in 0..p.mu {
            let (r_in, r_out) = Self::layer_ranks(p, i);
            debug_assert_eq!(state_rank, r_in);
            let core = &row[i];
            let new_pos = pos_count * d;
            let mut next = vec![0i64; new_pos * r_out];
            for pp in 0..pos_count {
                for a in 0..r_in {
                    let sv = state[pp * r_in + a];
                    if sv == 0 {
                        continue;
                    }
                    for z in 0..d {
                        for b in 0..r_out {
                            let cv = core[a * d * r_out + z * r_out + b];
                            if cv == 0 {
                                continue;
                            }
                            next[(pp * d + z) * r_out + b] += sv * cv;
                        }
                    }
                }
            }
            state = next;
            pos_count = new_pos;
            state_rank = r_out;
        }
        debug_assert_eq!(state_rank, 1);
        state
            .iter()
            .map(|&v| to_goldilocks(v))
            .collect()
    }

    /// The verifier's TT-structured evaluation of `mler_{TT_j}` at the
    /// challenge point (Lemma 6's block-wise contraction): fold each
    /// core with the eq weights of its variable chunk.
    pub fn row_eval(&self, j: usize, point: &[Goldilocks]) -> Result<Goldilocks, TtrpError> {
        let p = &self.params;
        let d = p.d();
        if point.len() != p.num_vars() {
            return Err(TtrpError::Shape { expected: p.num_vars(), got: point.len() });
        }
        let row = &self.cores[j];
        // state: rank r_i vector after folding the first i chunks.
        let mut state: Vec<Goldilocks> = vec![Goldilocks::ONE]; // r_0 = 1
        let mut state_rank = 1usize;
        for i in 0..p.mu {
            let (r_in, r_out) = Self::layer_ranks(p, i);
            if state_rank != r_in {
                return Err(TtrpError::Shape { expected: r_in, got: state_rank });
            }
            let core = &row[i];
            // The eq weights of chunk i: variables [i·log_d, (i+1)·log_d).
            let chunk = &point[i * p.log_d..(i + 1) * p.log_d];
            let eq_w = DenseMle::eq_extension(chunk);
            let mut next = vec![Goldilocks::ZERO; r_out];
            for a in 0..r_in {
                let sv = state[a];
                if sv.is_zero() {
                    continue;
                }
                for z in 0..d {
                    let w = eq_w.evaluations[z];
                    if w.is_zero() {
                        continue;
                    }
                    for b in 0..r_out {
                        let cv = core[a * d * r_out + z * r_out + b];
                        if cv == 0 {
                            continue;
                        }
                        next[b] = next[b].add(&sv.mul(&w).mul(&to_goldilocks(cv)));
                    }
                }
            }
            state = next;
            state_rank = r_out;
        }
        Ok(state[0])
    }
}

/// A signed integer → Goldilocks (the balanced representative).
fn to_goldilocks(v: i64) -> Goldilocks {
    if v >= 0 {
        Goldilocks::from_u64(v as u64)
    } else {
        Goldilocks::ZERO.sub(&Goldilocks::from_u64((-v) as u64))
    }
}

// ---------------------------------------------------------------------------
// The protocol (Π^0 + Π^sc).

/// The TTRP proof: the projection values, the linearization sum-check,
/// and the terminal eval claim (the eval-claim API output).
#[derive(Clone, Debug)]
pub struct TtrpProof {
    /// The k projected values (the balanced integer representatives).
    pub y: Vec<i64>,
    /// The Π^sc sum-check (degree 2).
    pub sumcheck: SumcheckProof,
    /// `w_r = mler_x(r)` — the terminal evaluation claim.
    pub w_r: Goldilocks,
    /// The terminal point (the eval-claim API's point).
    pub point: Vec<Goldilocks>,
}

/// The verifier-side binding: the eval claim appended to the ledger.
#[derive(Clone, Debug)]
pub struct TtrpBinding {
    pub point: Vec<Goldilocks>,
    pub w_r: Goldilocks,
}

/// The completeness bound `B̂` (Theorem 4's route at the ternary
/// instantiation): `√(k·(c/2)^{mu−1})·B₂` with `B₂ = ‖x‖₂` — the
/// caller supplies the honest witness bound.
pub fn completeness_bound(params: &TtrpParams, b2: u64) -> u64 {
    let moment = (params.rank as f64 / 2.0).powi(params.mu as i32 - 1);
    ((params.k as f64 * moment).sqrt() * b2 as f64).ceil() as u64
}

/// Prove the TTRP shortness statement for the coefficient vector `x`
/// (the balanced representatives of the committed ring coefficients).
/// The transcript must already hold the public statement (the
/// commitment digest + the bound).
pub fn prove_ttrp(
    params: &TtrpParams,
    x: &[i64],
    transcript: &mut Transcript,
) -> Result<TtrpProof, TtrpError> {
    let cores = TtCores::sample(params, transcript)?;
    // The projection (integer, exact).
    let y = cores.project(x);
    // Absorb the projection into the transcript (the y message).
    let y_bytes: Vec<u8> = y
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    transcript
        .append_bytes(b"ttrp-y", &y_bytes)
        .map_err(TtrpError::Transcript)?;
    // The γ combiner.
    let gamma = transcript
        .challenge_field(b"ttrp-gamma")
        .map_err(TtrpError::Transcript)?;

    // The claimed sum: Σ_j γ^j · y_j (as Goldilocks images).
    let mut claim = Goldilocks::ZERO;
    for (j, &yj) in y.iter().enumerate() {
        claim = claim.add(&gamma.pow_u64(j as u64).mul(&to_goldilocks(yj)));
    }

    // The virtual polynomial: Σ_j γ^j · [row_j, x].
    let n = params.padded_len();
    let mut xp = vec![Goldilocks::ZERO; n];
    for (i, &xi) in x.iter().enumerate() {
        if i < n {
            xp[i] = to_goldilocks(xi);
        }
    }
    let x_mle = DenseMle::new(xp).map_err(|e| match e {
        MleError::WrongEvaluationCount { expected, got } => TtrpError::Shape { expected, got },
        MleError::PointLengthMismatch { expected, got } => TtrpError::Shape { expected, got },
    })?;
    let mut vp = VirtualPolynomial::new(params.num_vars());
    let x_handle = vp
        .add_factor(x_mle)
        .map_err(|e| TtrpError::Sumcheck(format!("{e:?}")))?;
    let mut row_handles = Vec::with_capacity(params.k);
    for j in 0..params.k {
        let table = cores.row_table(j);
        let mle = DenseMle::new(table).map_err(|e| match e {
            MleError::WrongEvaluationCount { expected, got } => TtrpError::Shape { expected, got },
            MleError::PointLengthMismatch { expected, got } => TtrpError::Shape { expected, got },
        })?;
        let h = vp
            .add_factor(mle)
            .map_err(|e| TtrpError::Sumcheck(format!("{e:?}")))?;
        row_handles.push(h);
        vp.add_term(gamma.pow_u64(j as u64), vec![h, x_handle])
            .map_err(|e| TtrpError::Sumcheck(format!("{e:?}")))?;
    }
    let out = sumcheck::prove(&vp, claim, transcript)
        .map_err(|e| TtrpError::Sumcheck(format!("{e:?}")))?;
    let w_r = out.factor_claims[x_handle];
    // Bind the terminal claim.
    transcript
        .append_field_slice(b"ttrp-wr", &[w_r])
        .map_err(TtrpError::Transcript)?;
    Ok(TtrpProof { y, sumcheck: out.proof, w_r, point: out.challenges })
}

/// Verify the TTRP proof. On success returns the eval-claim binding for
/// the commitment layer; the norm gate `‖y‖₂ ≤ B̂` is enforced here.
pub fn verify_ttrp(
    params: &TtrpParams,
    bound: u64,
    proof: &TtrpProof,
    transcript: &mut Transcript,
) -> Result<TtrpBinding, TtrpError> {
    let cores = TtCores::sample(params, transcript)?;
    if proof.y.len() != params.k {
        return Err(TtrpError::Shape { expected: params.k, got: proof.y.len() });
    }
    // The norm gate (fail-closed).
    let norm2: u64 = proof
        .y
        .iter()
        .map(|&v| {
            let a = v.unsigned_abs();
            a.saturating_mul(a)
        })
        .fold(0u64, |acc, v| acc.saturating_add(v));
    if norm2 > bound.saturating_mul(bound) {
        return Err(TtrpError::NormExceeded { norm2, bound });
    }
    let y_bytes: Vec<u8> = proof
        .y
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    transcript
        .append_bytes(b"ttrp-y", &y_bytes)
        .map_err(TtrpError::Transcript)?;
    let gamma = transcript
        .challenge_field(b"ttrp-gamma")
        .map_err(TtrpError::Transcript)?;
    let mut claim = Goldilocks::ZERO;
    for (j, &yj) in proof.y.iter().enumerate() {
        claim = claim.add(&gamma.pow_u64(j as u64).mul(&to_goldilocks(yj)));
    }
    let verifier = proof
        .sumcheck
        .verify(params.num_vars(), 2, claim, transcript, None)
        .map_err(|e| TtrpError::Sumcheck(format!("{e:?}")))?;
    transcript
        .append_field_slice(b"ttrp-wr", &[proof.w_r])
        .map_err(TtrpError::Transcript)?;
    // The final check: Σ_j γ^j · mler_{TT_j}(r) · w_r == the derived
    // terminal value, with the row evaluations from the verifier's own
    // TT contraction.
    let mut expected = Goldilocks::ZERO;
    for j in 0..params.k {
        let row_r = cores.row_eval(j, &verifier.point)?;
        expected = expected.add(&gamma.pow_u64(j as u64).mul(&row_r).mul(&proof.w_r));
    }
    if expected != verifier.final_claim {
        return Err(TtrpError::FinalCheckFailed);
    }
    Ok(TtrpBinding { point: verifier.point, w_r: proof.w_r })
}

// ---------------------------------------------------------------------------
// The digit-free opening — the compact-mode linear-functional bridge.

/// The bridge parameters (the compact-mode fold discipline).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeParams {
    /// The column count r (a power of two dividing the flat length).
    pub r: usize,
    /// The scalar challenge amplitude A (challenges in [−A, A]).
    pub amplitude: u32,
    /// The fail-closed per-coefficient norm gate (r·A·255 < q/2).
    pub gate: u32,
}

impl BridgeParams {
    pub fn new(r: usize, amplitude: u32, ring_q: u64) -> Self {
        let gate = (r as u64) * (amplitude as u64) * 255;
        BridgeParams { r, amplitude, gate: gate.min(ring_q / 2 - 1) as u32 }
    }
}

/// The byte-packed column layout (the compact mode's Items 1–2): the
/// flat values (the ring coefficients, < 2^31) become `width` bytes
/// each; the flat position's last log2(r) variables index the columns.
pub struct BridgeLayout {
    pub params: BridgeParams,
    /// The flat value count (a power-of-two multiple of r).
    pub flat_len: usize,
    /// The byte width per flat value (1 for byte-valued coefficients,
    /// 4 for 31-bit values).
    pub width: usize,
    pub log_flat: usize,
    pub log_r: usize,
    pub log_head: usize,
}

impl BridgeLayout {
    pub fn derive(flat_len: usize, width: usize, params: BridgeParams) -> Result<Self, TtrpError> {
        if !params.r.is_power_of_two() || flat_len % params.r != 0 || flat_len == 0 {
            return Err(TtrpError::Shape { expected: params.r, got: flat_len });
        }
        let log_flat = flat_len.trailing_zeros() as usize;
        let log_r = params.r.trailing_zeros() as usize;
        Ok(BridgeLayout {
            params,
            flat_len,
            width,
            log_flat,
            log_r,
            log_head: log_flat - log_r,
        })
    }

    /// The slot map: column-stream position m ↔ (head h, byte b) —
    /// column-independent by the r-aligned construction (the flat value
    /// at position h·r + j contributes its bytes to column j's stream
    /// at h·width + b).
    fn slot_of(&self, m: usize) -> (usize, u8) {
        let h = m / self.width;
        let b = (m % self.width) as u8;
        (h, b)
    }

    pub fn stream_len(&self) -> usize {
        self.flat_len * self.width
    }

    /// The column-uniform shadow weights Ψ(m) = eq_head(h(m))·2^{8·b(m)}
    /// over Goldilocks.
    pub fn psi_weights(&self, r_head: &[Goldilocks]) -> Vec<Goldilocks> {
        let eq_head = DenseMle::eq_extension(r_head).evaluations;
        (0..self.stream_len())
            .map(|m| {
                let (h, b) = self.slot_of(m);
                eq_head
                    .get(h)
                    .copied()
                    .unwrap_or(Goldilocks::ZERO)
                    .mul(&fe(1u64 << (8 * b as u32)))
            })
            .collect()
    }

    /// The column weights μ_j = eq(r_tail)_j.
    pub fn mu_weights(&self, r_tail: &[Goldilocks]) -> Vec<Goldilocks> {
        DenseMle::eq_extension(r_tail).evaluations
    }
}

/// The functional opening artifact.
#[derive(Clone, Debug)]
pub struct FunctionalOpening {
    pub params: BridgeParams,
    /// The per-column Goldilocks values ũ_j (r values, absorbed BEFORE
    /// the challenges).
    pub u_tilde: Vec<Goldilocks>,
    /// The response v = Σ_j d_j·w_j as raw ring coefficients (the
    /// column-major concatenated byte folds).
    pub response: Vec<i32>,
}

/// The prover side of the bridge: prove the eval claim `w_r` against
/// the per-column Ajtai commitments.
///
/// * `columns` — the byte-packed ring-element columns (each ring
///   coefficient is ONE byte of the flat stream, ≤ 255).
/// * `point` — the full flat-cube evaluation point (log_flat entries).
#[allow(clippy::too_many_arguments)]
pub fn open_eval_functional(
    pk: &AjtaiPublicKey,
    layout: &BridgeLayout,
    columns: &[Vec<RingElement>],
    commitments: &[AjtaiCommitment],
    point: &[Goldilocks],
    claim_w: Goldilocks,
    transcript: &mut Transcript,
) -> Result<FunctionalOpening, TtrpError> {
    let r = layout.params.r;
    if columns.len() != r || commitments.len() != r {
        return Err(TtrpError::Shape { expected: r, got: columns.len() });
    }
    let (r_head, _r_tail) = point.split_at(layout.log_head);
    // 1. The per-column values ũ_j = Σ_m Ψ(m)·byte(m, j).
    let psi = layout.psi_weights(r_head);
    let mut u_tilde = vec![Goldilocks::ZERO; r];
    for (j, col) in columns.iter().enumerate() {
        let mut acc = Goldilocks::ZERO;
        let mut m = 0usize;
        for elem in col {
            for &byte in elem.coeffs() {
                if m < psi.len() {
                    acc = acc.add(&psi[m].mul(&fe(byte as u64)));
                }
                m += 1;
            }
        }
        u_tilde[j] = acc;
    }
    // The prover-side (b) self-check: Σ_j μ_j·ũ_j must close the claim.
    {
        let mu = layout.mu_weights(_r_tail);
        let mut w_check = Goldilocks::ZERO;
        for (j, &u) in u_tilde.iter().enumerate() {
            w_check = w_check.add(&mu[j].mul(&u));
        }
        if w_check != claim_w {
            return Err(TtrpError::BridgeFailed("(b) self-check"));
        }
    }
    // Absorb the ũ's BEFORE the challenges.
    transcript
        .append_field_slice(b"bridge-u", &u_tilde)
        .map_err(TtrpError::Transcript)?;
    // 2. The scalar challenges d_j ∈ [−A, A].
    let d: Vec<i32> = (0..r)
        .map(|_| {
            let bytes = transcript
                .challenge_bytes(b"bridge-d", 2)
                .map_err(TtrpError::Transcript)?;
            let raw = i16::from_le_bytes([bytes[0], bytes[1]]) as i32;
            Ok(raw % (layout.params.amplitude as i32 + 1))
        })
        .collect::<Result<_, TtrpError>>()?;
    // 3. The response v = Σ_j d_j·w_j (the integer fold over the byte
    //    coefficients — the gate keeps |v| < r·A·255 < q/2).
    let ring_n = pk.params.ring.n();
    let per_col = columns
        .first()
        .map(|c| c.len())
        .ok_or(TtrpError::Shape { expected: 1, got: 0 })?;
    let mut response = vec![0i32; per_col * ring_n];
    for (j, col) in columns.iter().enumerate() {
        let dj = d[j];
        if dj == 0 {
            continue;
        }
        for (ei, elem) in col.iter().enumerate() {
            for (ci, &byte) in elem.coeffs().iter().enumerate() {
                response[ei * ring_n + ci] += dj * byte as i32;
            }
        }
    }
    // The gate (fail-closed on the prover side).
    for &v in &response {
        if v.unsigned_abs() as u64 > layout.params.gate as u64 {
            return Err(TtrpError::BridgeFailed("response gate"));
        }
    }
    // Absorb the response.
    let resp_bytes: Vec<u8> = response.iter().flat_map(|&v| v.to_le_bytes()).collect();
    transcript
        .append_bytes(b"bridge-v", &resp_bytes)
        .map_err(TtrpError::Transcript)?;
    Ok(FunctionalOpening { params: layout.params.clone(), u_tilde, response })
}

/// The verifier side: the three checks (a) functional commutation,
/// (b) the MLE interpolation closing the claim, (c) the Ajtai binding —
/// plus the gate.
pub fn verify_eval_functional(
    pk: &AjtaiPublicKey,
    layout: &BridgeLayout,
    commitments: &[AjtaiCommitment],
    point: &[Goldilocks],
    claim_w: Goldilocks,
    opening: &FunctionalOpening,
    transcript: &mut Transcript,
) -> Result<(), TtrpError> {
    let r = layout.params.r;
    if commitments.len() != r || opening.u_tilde.len() != r {
        return Err(TtrpError::Shape { expected: r, got: commitments.len() });
    }
    let (r_head, r_tail) = point.split_at(layout.log_head);
    // Replay the transcript: the ũ's, then the challenges, then v.
    transcript
        .append_field_slice(b"bridge-u", &opening.u_tilde)
        .map_err(TtrpError::Transcript)?;
    let d: Vec<i32> = (0..r)
        .map(|_| {
            let bytes = transcript
                .challenge_bytes(b"bridge-d", 2)
                .map_err(TtrpError::Transcript)?;
            let raw = i16::from_le_bytes([bytes[0], bytes[1]]) as i32;
            Ok(raw % (layout.params.amplitude as i32 + 1))
        })
        .collect::<Result<_, TtrpError>>()?;
    let resp_bytes: Vec<u8> = opening
        .response
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    transcript
        .append_bytes(b"bridge-v", &resp_bytes)
        .map_err(TtrpError::Transcript)?;

    // (b) The MLE interpolation: w = Σ_j μ_j·ũ_j.
    let mu = layout.mu_weights(r_tail);
    let mut w_check = Goldilocks::ZERO;
    for (j, &u) in opening.u_tilde.iter().enumerate() {
        w_check = w_check.add(&mu[j].mul(&u));
    }
    if w_check != claim_w {
        return Err(TtrpError::BridgeFailed("(b) interpolation"));
    }
    // (a) The functional commutation: Φ(v) = Σ_j d_j·ũ_j over Goldilocks.
    let psi = layout.psi_weights(r_head);
    let ring_n = pk.params.ring.n();
    let mut phi_v = Goldilocks::ZERO;
    for (m, &v) in opening.response.iter().enumerate() {
        if m < psi.len() && v != 0 {
            phi_v = phi_v.add(&psi[m].mul(&to_goldilocks(v as i64)));
        }
    }
    let mut rhs = Goldilocks::ZERO;
    for (j, &u) in opening.u_tilde.iter().enumerate() {
        rhs = rhs.add(&to_goldilocks(d[j] as i64).mul(&u));
    }
    if phi_v != rhs {
        return Err(TtrpError::BridgeFailed("(a) commutation"));
    }
    // The gate.
    for &v in &opening.response {
        if v.unsigned_abs() as u64 > layout.params.gate as u64 {
            return Err(TtrpError::BridgeFailed("gate"));
        }
    }
    // (c) The Ajtai binding: A·v == Σ_j d_j·y_j — the verifier's own
    // commitment of the response vs the folded commitments.
    let ring = &pk.params.ring;
    let v_elems: Vec<RingElement> = opening
        .response
        .chunks(ring_n)
        .filter(|c| c.len() == ring_n)
        .map(|c| {
            let coeffs: Vec<u32> =
                c.iter().map(|&b| ring.modulus.reduce_i64(b as i64)).collect();
            RingElement::from_coeffs(ring, coeffs)
        })
        .collect();
    let padded = pk
        .pad_to_m(&v_elems)
        .map_err(|_| TtrpError::BridgeFailed("(c) pad"))?;
    let commit_v = pk
        .commit(&padded)
        .map_err(|_| TtrpError::BridgeFailed("(c) commit"))?;
    // Σ_j d_j·y_j — every term scaled, including j = 0.
    let k_rows = commitments
        .first()
        .map(|c| c.rows.len())
        .ok_or(TtrpError::Shape { expected: 1, got: 0 })?;
    let zero = RingElement::from_coeffs(ring, vec![0u32; ring.n()]);
    let mut folded = vec![zero; k_rows];
    for (j, c) in commitments.iter().enumerate() {
        for (acc, row) in folded.iter_mut().zip(c.rows.iter()) {
            *acc = acc
                .add(&row.scale_i64(d[j] as i64))
                .map_err(|_| TtrpError::BridgeFailed("(c) fold"))?;
        }
    }
    if commit_v.rows != folded {
        return Err(TtrpError::BridgeFailed("(c) Ajtai binding"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The claim-ledger wiring: the TTRP norm-check route.

/// The ledger's TTRP norm-check statement: given the committed packed
/// vector (the byte-packed columns), prove its shortness via the TT
/// projection + the digit-free functional opening — REPLACING
/// `compact_norm_proof`'s Θ(N) digit reveal on this route.
pub struct TtrpNormCheck {
    pub ttrp_params: TtrpParams,
    pub bridge_params: BridgeParams,
    /// The completeness bound B̂ for the projection.
    pub bound: u64,
}

/// The full norm-check proof: TTRP + the functional opening.
#[derive(Clone, Debug)]
pub struct TtrpNormProof {
    pub ttrp: TtrpProof,
    pub opening: FunctionalOpening,
}

impl TtrpNormCheck {
    /// Derive the layout from the packed ring vector: the ring
    /// coefficients (each ≤ 255) ARE the flat values — 1 byte each
    /// (width 1, the narrowest packing — even simpler than the compact
    /// mode's per-value widths).
    pub fn flat_of(columns: &[Vec<RingElement>]) -> Vec<i64> {
        // The r-aligned interleaving: flat[h·r + j] = column j's h-th
        // coefficient (the last log2(r) flat variables index columns).
        let r = columns.len();
        let per_col: Vec<Vec<i64>> = columns
            .iter()
            .map(|col| {
                col.iter()
                    .flat_map(|e| e.coeffs().iter().map(|&c| c as i64))
                    .collect()
            })
            .collect();
        let stream = per_col.first().map(|c| c.len()).unwrap_or(0);
        let mut flat = vec![0i64; stream * r];
        for (j, col) in per_col.iter().enumerate() {
            for (h, &v) in col.iter().enumerate() {
                flat[h * r + j] = v;
            }
        }
        flat
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prove(
        &self,
        pk: &AjtaiPublicKey,
        columns: &[Vec<RingElement>],
        commitments: &[AjtaiCommitment],
        transcript: &mut Transcript,
    ) -> Result<TtrpNormProof, TtrpError> {
        // The flat coefficient vector (column-major — the last log2(r)
        // flat variables are the columns by the layout discipline).
        let flat = Self::flat_of(columns);
        let ttrp = prove_ttrp(&self.ttrp_params, &flat, transcript)?;
        // The eval claim w_r at the TTRP terminal point.
        let opening = open_eval_functional(
            pk,
            &self.bridge_layout(flat.len())?,
            columns,
            commitments,
            &ttrp.point,
            ttrp.w_r,
            transcript,
        )?;
        Ok(TtrpNormProof { ttrp, opening })
    }

    fn bridge_layout(&self, flat_len: usize) -> Result<BridgeLayout, TtrpError> {
        // The wiring's flat values are the raw ring coefficients (each
        // ≤ 255) — the width-1 packing (the narrowest).
        BridgeLayout::derive(flat_len, 1, self.bridge_params.clone())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        pk: &AjtaiPublicKey,
        columns_shape: usize,
        commitments: &[AjtaiCommitment],
        proof: &TtrpNormProof,
        transcript: &mut Transcript,
    ) -> Result<TtrpBinding, TtrpError> {
        let binding =
            verify_ttrp(&self.ttrp_params, self.bound, &proof.ttrp, transcript)?;
        let layout = self.bridge_layout(columns_shape)?;
        verify_eval_functional(
            pk,
            &layout,
            commitments,
            &binding.point,
            binding.w_r,
            &proof.opening,
            transcript,
        )?;
        Ok(binding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_ring::{Modulus32, RingConfig};

    fn setup(log_n: u32, m_slots: usize) -> AjtaiPublicKey {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams { ring, k: 2, m: m_slots, norm_bound: 1 << 20 };
        AjtaiPublicKey::from_seed(params, [41u8; 32]).ok().unwrap()
    }

    fn small_coeff_vector(n: usize, seed: u64) -> Vec<i64> {
        // Ternary-ish small coefficients (the committed regime).
        (0..n)
            .map(|i| {
                let h = Transcript::hash_domain(b"ttrp-x", &seed.to_le_bytes()[..4]);
                match h[0] % 4 {
                    0 | 1 => 0i64,
                    2 => -1,
                    _ => 1,
                }
                .wrapping_add(if i % 7 == 0 { 1 } else { 0 })
            })
            .collect()
    }

    #[test]
    fn projection_matches_direct_computation() {
        // y_j = ⟨TT_j, x⟩ — cross-check the contraction against the
        // direct truth-table dot product.
        let params = TtrpParams { mu: 3, log_d: 1, rank: 2, k: 3 };
        let mut t = Transcript::new_default(b"ttrp-test");
        let cores = TtCores::sample(&params, &mut t).ok().unwrap();
        let x = small_coeff_vector(params.padded_len(), 7);
        let y = cores.project(&x);
        for j in 0..params.k {
            let table = cores.row_table(j);
            let mut acc = Goldilocks::ZERO;
            for (i, &xi) in x.iter().enumerate() {
                acc = acc.add(&table[i].mul(&to_goldilocks(xi)));
            }
            assert_eq!(acc, to_goldilocks(y[j]), "row {j}");
        }
    }

    #[test]
    fn row_eval_matches_the_table() {
        // The verifier's TT contraction == the table's MLE evaluation.
        let params = TtrpParams { mu: 3, log_d: 2, rank: 2, k: 2 };
        let mut t = Transcript::new_default(b"ttrp-test");
        let cores = TtCores::sample(&params, &mut t).ok().unwrap();
        let point: Vec<Goldilocks> = (1..=params.num_vars() as u64)
            .map(|i| fe(1_000_003 * i))
            .collect();
        for j in 0..params.k {
            let table = cores.row_table(j);
            let mle = DenseMle::new(table).ok().unwrap();
            let direct = mle.evaluate(&point).ok().unwrap();
            let contracted = cores.row_eval(j, &point).ok().unwrap();
            assert_eq!(direct, contracted, "row {j}");
        }
    }

    #[test]
    fn ttrp_prove_verify_roundtrip() {
        for &(mu, log_d, rank, k, n) in &[
            (3usize, 1usize, 2usize, 2usize, 8usize),
            (2, 2, 1, 3, 12),
            (4, 1, 2, 2, 16),
        ] {
            let params = TtrpParams { mu, log_d, rank, k };
            let x = small_coeff_vector(n, 11);
            let b2: u64 = x
                .iter()
                .map(|&v| v.unsigned_abs().pow(2))
                .sum();
            let bound = completeness_bound(&params, b2).max(b2 * 4);
            let mut pt = Transcript::new_default(b"ttrp-prove");
            let proof = prove_ttrp(&params, &x, &mut pt).ok().unwrap();
            let mut vt = Transcript::new_default(b"ttrp-prove");
            let binding = verify_ttrp(&params, bound, &proof, &mut vt).ok().unwrap();
            assert_eq!(binding.w_r, proof.w_r);
            assert_eq!(binding.point.len(), params.num_vars());
        }
    }

    #[test]
    fn ttrp_rejects_tampered_y() {
        let params = TtrpParams { mu: 3, log_d: 1, rank: 2, k: 2 };
        let x = small_coeff_vector(8, 13);
        let b2: u64 = x.iter().map(|&v| v.unsigned_abs().pow(2)).sum();
        let bound = completeness_bound(&params, b2).max(b2 * 4);
        let mut pt = Transcript::new_default(b"ttrp-prove");
        let mut proof = prove_ttrp(&params, &x, &mut pt).ok().unwrap();
        proof.y[0] += 1; // a wrong projection breaks the claimed sum
        let mut vt = Transcript::new_default(b"ttrp-prove");
        assert!(verify_ttrp(&params, bound, &proof, &mut vt).is_err());
    }

    #[test]
    fn ttrp_rejects_tampered_wr() {
        let params = TtrpParams { mu: 3, log_d: 1, rank: 2, k: 2 };
        let x = small_coeff_vector(8, 17);
        let b2: u64 = x.iter().map(|&v| v.unsigned_abs().pow(2)).sum();
        let bound = completeness_bound(&params, b2).max(b2 * 4);
        let mut pt = Transcript::new_default(b"ttrp-prove");
        let mut proof = prove_ttrp(&params, &x, &mut pt).ok().unwrap();
        proof.w_r = proof.w_r.add(&Goldilocks::ONE);
        let mut vt = Transcript::new_default(b"ttrp-prove");
        assert!(verify_ttrp(&params, bound, &proof, &mut vt).is_err());
    }

    #[test]
    fn ttrp_rejects_norm_violation() {
        // An out-of-bound witness: the projection blows past the gate.
        let params = TtrpParams { mu: 3, log_d: 1, rank: 2, k: 2 };
        let mut x = small_coeff_vector(8, 19);
        for v in x.iter_mut() {
            *v = 1 << 20; // fat coefficients
        }
        let bound = 16u64; // absurdly tight
        let mut pt = Transcript::new_default(b"ttrp-prove");
        let proof = prove_ttrp(&params, &x, &mut pt).ok().unwrap();
        let mut vt = Transcript::new_default(b"ttrp-prove");
        assert!(matches!(
            verify_ttrp(&params, bound, &proof, &mut vt),
            Err(TtrpError::NormExceeded { .. })
        ));
    }

    /// The full digit-free route: byte-packed columns → commitments →
    /// TTRP → the functional opening — the ledger's norm check without
    /// a single digit.
    #[test]
    fn norm_check_route_end_to_end() {
        let log_n = 6u32; // 64 coefficients per ring element
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let r = 4usize;
        // The flat byte coefficients: 4 columns × 2 elements × 64 = 512.
        let flat_len = r * 2 * (1usize << log_n);
        let ttrp_params = TtrpParams::for_len(flat_len, 2, 3, 1);
        let bridge_params = BridgeParams::new(r, 1 << 6, Modulus32::Q_32.q as u64);
        let pk = setup(log_n, 8);

        // Byte columns (each coefficient ≤ 255 — the narrow packing):
        // r columns × 2 ring elements × 64 coefficients = 512 flat.
        let mut columns: Vec<Vec<RingElement>> = Vec::with_capacity(r);
        let mut seed = 0u64;
        for _ in 0..r {
            let mut elems = Vec::with_capacity(2);
            for _ in 0..2 {
                let coeffs: Vec<u32> = (0..(1 << log_n))
                    .map(|_| {
                        seed += 1;
                        let h = Transcript::hash_domain(
                            b"bridge-byte",
                            &seed.to_le_bytes(),
                        );
                        (h[0] % 251) as u32
                    })
                    .collect();
                elems.push(RingElement::from_coeffs(&ring, coeffs));
            }
            columns.push(elems);
        }
        let commitments: Vec<AjtaiCommitment> = columns
            .iter()
            .map(|c| pk.commit(&pk.pad_to_m(c).ok().unwrap()).ok().unwrap())
            .collect();

        // The honest ℓ2 bound of the flat coefficient vector.
        let flat = TtrpNormCheck::flat_of(&columns);
        let b2: u64 = flat.iter().map(|&v| v.unsigned_abs().pow(2)).sum();
        let bound = completeness_bound(&ttrp_params, b2).max(b2 * 4);

        let check = TtrpNormCheck { ttrp_params, bridge_params, bound };
        let mut pt = Transcript::new_default(b"ttrp-ledger");
        let proof = check
            .prove(&pk, &columns, &commitments, &mut pt)
            .ok()
            .unwrap();
        let mut vt = Transcript::new_default(b"ttrp-ledger");
        let binding = check
            .verify(&pk, flat.len(), &commitments, &proof, &mut vt)
            .ok()
            .unwrap();
        assert_eq!(binding.w_r, proof.ttrp.w_r);

        // Tamper: a wrong claim breaks the bridge's interpolation check.
        let mut bad = proof.clone();
        bad.ttrp.w_r = bad.ttrp.w_r.add(&Goldilocks::ONE);
        let mut vt2 = Transcript::new_default(b"ttrp-ledger");
        assert!(check.verify(&pk, flat.len(), &commitments, &bad, &mut vt2).is_err());

        // Tamper: a swapped column commitment breaks the Ajtai binding.
        let mut swapped = commitments.clone();
        swapped.swap(0, 1);
        let mut vt3 = Transcript::new_default(b"ttrp-ledger");
        assert!(check.verify(&pk, flat.len(), &swapped, &proof, &mut vt3).is_err());
    }
}
