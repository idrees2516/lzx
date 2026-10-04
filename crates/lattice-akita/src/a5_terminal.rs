//! A5 — Akita recursive opening + terminal handling + the signed-Rice
//! response encoding (ePrint 2026/1983, §8): Wave 7 item 7.11, A5 —
//! the driver that chains the folds (A1) into the terminal's direct
//! checks.
//!
//! * **§8.1 Composing the folds** — after each nonfinal fold `j` the
//!   parties hold a claim about the successor witness under the
//!   commitment absorbed during that fold. The analyzed opening-mode
//!   policy (Eq 162):
//!   `method(j, kind) = SubringCoefficientPacking` for nonterminal
//!   `j ∈ {0, 1}`, `EvaluationTrace` otherwise — a final packing fold
//!   can lead directly into the evaluation-trace terminal.
//! * **§8.2 Terminal levels (tail handling)** — following LaBRADOR's
//!   §5.6 discipline: omit outer commitments whose openings would be
//!   immediately revealed; the final edge binds the canonical inner
//!   state `t_i = A_term·s_i` for every block of its one incoming
//!   witness group (the specialization that permits omitting the outer
//!   B relation soundly). The terminal:
//!   * runs its tensor-reduction prefix (A4) producing the packed claim
//!     and the transparent factor — rejected (no resampling) if zero,
//!     an event SEPARATE from the response grind;
//!   * absorbs the clear partial opening evaluations
//!     `e_i = ⟨a_term, f_i⟩`;
//!   * runs the **scheduled response grind**: each candidate nonce is
//!     absorbed BEFORE the transcript derives the fold-challenge tuple
//!     (`§8.2`'s ordering); a nonce is admissible only when every
//!     bounded fixed-filter sampler produces an accepted challenge AND
//!     the clear response satisfies its scheduled norm check; at most
//!     `N_try,term` distinct nonces are tried;
//!   * checks the three direct identities (Eq 163–165):
//!     `A_term·z = Σ_i c_i·t_i` in `R_term`,
//!     `Σ_i c_i·e_i = a_term^⊤·G·z` in `R_term` (kernel: the
//!     position-weight row), and
//!     `Σ_i χ_blk(i)·T_{ρ_pack}(e_i) = v_tensor` in `E` (the Eq-165
//!     trace row; the θ-scaling is the paper's no-inversion
//!     optimization — at kernel scale `θ_term` is certified nonzero
//!     inside the A4 prefix and the row is carried unscaled);
//!   * checks the response bound DIRECTLY (the response is revealed —
//!     no digit-range sumcheck) and encodes it with a **signed Rice
//!     code** whose byte budget derives from the squared-norm bound:
//!     `Σ_i |z_i| ≤ √(N_z·S)` bounds the unary portion.
//!
//! LZX realization notes (kernel scale, honestly stated): the driver
//! chains the A1 fold's kernel-scale discipline (revealed digit
//! segments — the §8.2 terminal route) with the A4 tensor prefix and
//! the A3 norm gates; the grind's bounded fixed-filter sampler is the
//! certified short-challenge family's op-norm rejection
//! (`sample_with_gamma_cap`, fail-closed); `N_try,term` is enforced on
//! both sides; the Rice code is a real signed Golomb-Rice
//! encoder/decoder with the budget rejection on BOTH sides.

use crate::a4_tensor::{
    challenge_fq2q, chi_pack, opening_row_trace, prove_tensor_reduce, trace_functional,
    verify_tensor_reduce, Fq2Q, TensorReductionProof,
};
use crate::fold::{
    self, decompose_block, recompose_block, FoldError, FoldKeys, FoldParams, FoldProof, FoldSource,
};
use lattice_core::extension::Fq2;
use lattice_core::short_challenge::{
    ShortChallenge, ShortChallengeError, ShortChallengeFamily, ShortChallengeSpec,
};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::{RingConfig, RingElement};

// ---------------------------------------------------------------- errors #

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalError {
    Transcript(TranscriptError),
    Fold(FoldError),
    ShortChallenge(ShortChallengeError),
    /// The tensor-reduction prefix failed (a false input claim —
    /// fail-closed at prove; tampered data at verify).
    TensorPrefixFailed,
    /// The grind exhausted `N_try,term` without an admissible nonce.
    GrindExhausted {
        tries: u32,
    },
    /// The nonce is outside the scheduled set (verifier rejection).
    NonceOutOfRange {
        nonce: u32,
    },
    /// The response violates the scheduled squared-norm bound.
    ResponseBoundViolated {
        norm_sq: u128,
        bound: u128,
    },
    /// The Rice encoding violates its byte budget (§8.2's
    /// `Σ|z_i| ≤ √(N_z·S)` unary bound).
    RiceBudgetViolated {
        bytes: usize,
        budget: usize,
    },
    /// Rice decode failure (malformed stream).
    RiceDecodeFailed,
    /// A direct check failed (Eq 163/164/165 — tampered).
    DirectCheckFailed(&'static str),
    /// Shape mismatch.
    Shape {
        expected: usize,
        got: usize,
    },
    /// Ring error.
    Ring(String),
}

impl From<TranscriptError> for TerminalError {
    fn from(e: TranscriptError) -> Self {
        TerminalError::Transcript(e)
    }
}

impl From<FoldError> for TerminalError {
    fn from(e: FoldError) -> Self {
        TerminalError::Fold(e)
    }
}

impl From<ShortChallengeError> for TerminalError {
    fn from(e: ShortChallengeError) -> Self {
        TerminalError::ShortChallenge(e)
    }
}

// ------------------------------------------------- the opening-mode policy #

/// The analyzed opening-mode policy (Eq 162).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpeningMethod {
    /// Subring coefficient packing (nonterminal levels 0 and 1).
    SubringCoefficientPacking,
    /// Evaluation trace (all other receivers, including the terminal).
    EvaluationTrace,
}

/// `method(j, kind)` (Eq 162): packing for nonterminal `j ∈ {0, 1}`,
/// evaluation trace otherwise.
pub fn opening_method(level: u32, nonterminal: bool) -> OpeningMethod {
    if nonterminal && level <= 1 {
        OpeningMethod::SubringCoefficientPacking
    } else {
        OpeningMethod::EvaluationTrace
    }
}

// ---------------------------------------------------- signed Rice encoding #

/// The Rice–Golomb parameter (the power-of-two divisor).
const RICE_K: u32 = 4;

/// Encode signed values with the signed Rice code (ZigZag then
/// unary quotient / binary remainder at parameter `k`).
pub fn rice_encode(values: &[i64]) -> Vec<u8> {
    let mut bits: Vec<bool> = Vec::new();
    for &v in values {
        let zz = ((v << 1) ^ (v >> 63)) as u64;
        let q = zz >> RICE_K;
        let r = zz & ((1 << RICE_K) - 1);
        bits.resize(bits.len() + q as usize, true);
        bits.push(false);
        for b in (0..RICE_K).rev() {
            bits.push((r >> b) & 1 == 1);
        }
    }
    let mut out = Vec::with_capacity(bits.len().div_ceil(8));
    let mut acc = 0u8;
    let mut n = 0u32;
    for &b in &bits {
        if b {
            acc |= 1 << (7 - n);
        }
        n += 1;
        if n == 8 {
            out.push(acc);
            acc = 0;
            n = 0;
        }
    }
    if n > 0 {
        out.push(acc);
    }
    out
}

/// Decode a signed Rice stream of exactly `count` values.
pub fn rice_decode(bytes: &[u8], count: usize) -> Result<Vec<i64>, TerminalError> {
    let bits: Vec<bool> = bytes
        .iter()
        .flat_map(|&b| (0..8).rev().map(move |i| (b >> i) & 1 == 1))
        .collect();
    let mut pos = 0usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut q = 0u64;
        while pos < bits.len() && bits[pos] {
            q += 1;
            pos += 1;
            if q > 1 << 24 {
                return Err(TerminalError::RiceDecodeFailed);
            }
        }
        if pos >= bits.len() {
            return Err(TerminalError::RiceDecodeFailed);
        }
        pos += 1;
        if pos + RICE_K as usize > bits.len() {
            return Err(TerminalError::RiceDecodeFailed);
        }
        let mut r = 0u64;
        for _ in 0..RICE_K {
            r = (r << 1) | u64::from(bits[pos]);
            pos += 1;
        }
        let zz = (q << RICE_K) | r;
        let v = ((zz >> 1) as i64) ^ (-((zz & 1) as i64));
        out.push(v);
    }
    Ok(out)
}

/// The byte budget from the squared-norm bound: `Σ_i |z_i| ≤ √(N_z·S)`
/// bounds the unary portion, hence the total Rice bytes.
pub fn rice_budget(n_z: usize, s_bound: u64) -> usize {
    // Integer sqrt without the 1.84 `isqrt` API (MSRV discipline).
    let x: u128 = (n_z as u128) * u128::from(s_bound);
    let mut lo: u128 = 0;
    let mut hi: u128 = x.min(u128::MAX >> 1);
    while lo < hi {
        let mid = (lo + hi + 1).div_ceil(2);
        if mid <= x / mid.max(1) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let l1 = lo.max(1);
    let bits = l1 as usize + n_z * (1 + RICE_K as usize) + 8;
    bits.div_ceil(8)
}

// ---------------------------------------------------------- the terminal #

/// The terminal's public schedule.
#[derive(Clone, Debug)]
pub struct TerminalState {
    /// The terminal ring `R_term`.
    pub ring: RingConfig,
    /// The canonical inner images `t_i = A_term·s_i` (public input —
    /// bound by the preceding fold; derived by the driver when empty).
    pub t_images: Vec<Vec<RingElement>>,
    /// The response squared-norm ceiling `S`.
    pub s_bound: u64,
    /// The grind's nonce budget `N_try,term`.
    pub n_try: u32,
    /// The certified challenge family for the fold challenges.
    pub challenge_family: ShortChallengeFamily,
    /// The fixed-filter's op-norm cap (the admissibility filter).
    pub gamma_cap: u64,
    /// The response digitization (base, depth).
    pub response_base: u64,
    pub response_digits: usize,
}

/// The terminal proof.
#[derive(Clone, Debug)]
pub struct TerminalProof {
    /// The tensor-reduction prefix (A4) over the incoming opening.
    pub tensor: TensorReductionProof,
    /// The revealed partial opening evaluations `e_i`.
    pub partials: Vec<RingElement>,
    /// The selected grind nonce.
    pub nonce: u32,
    /// The fold challenges (certified, recomputed by the verifier).
    pub challenges: Vec<ShortChallenge>,
    /// The clear response `z = Σ_i c_i·s_i` (revealed).
    pub response: Vec<RingElement>,
    /// The Rice-encoded response bytes.
    pub response_rice: Vec<u8>,
    /// The trace row's public data (over the same-modulus F_{Q32²}).
    pub rho_pack: Vec<Fq2Q>,
    pub chi_blk: Vec<Fq2Q>,
    pub v_tensor: Fq2Q,
}

fn embed(ring: &RingConfig, c: &ShortChallenge) -> RingElement {
    RingElement::from_signed(ring, &c.coefficients)
}

/// Centered lift of a ring coefficient to `i64`.
fn centered(q: u32, c: u32) -> i64 {
    let q64 = u64::from(q);
    let c64 = u64::from(c);
    if c64 > q64 / 2 {
        c64 as i64 - q64 as i64
    } else {
        c64 as i64
    }
}

/// The certified fixed-filter sampler: draws one challenge from the
/// transcript under the op-norm cap; `None` = rejection (the nonce is
/// inadmissible).
fn fixed_filter(
    state: &TerminalState,
    transcript: &mut Transcript,
    index: usize,
) -> Result<Option<ShortChallenge>, TerminalError> {
    let spec = ShortChallengeSpec {
        n: state.ring.n(),
        family: state.challenge_family.clone(),
    };
    let seed = transcript.challenge_bytes(b"a5-terminal-c", 32)?;
    let mut domain = seed;
    domain.extend_from_slice(&(index as u64).to_le_bytes());
    match spec.sample_with_gamma_cap(&domain, state.gamma_cap, 0) {
        Ok(c) => Ok(Some(c)),
        Err(ShortChallengeError::GammaCapExceeded { .. }) => Ok(None),
        Err(e) => Err(TerminalError::ShortChallenge(e)),
    }
}

/// Prove the terminal (§8.2) over the final group's source blocks.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn prove_terminal(
    state: &TerminalState,
    f_blocks: &[Vec<RingElement>],
    a_weights: &[RingElement],
    _a_matrix: &[Vec<RingElement>],
    eval_mle: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    transcript: &mut Transcript,
) -> Result<TerminalProof, TerminalError> {
    let ring = &state.ring;
    if f_blocks.len() != state.t_images.len() {
        return Err(TerminalError::Shape {
            expected: state.t_images.len(),
            got: f_blocks.len(),
        });
    }
    // ---- The tensor-reduction prefix (the incoming opening). ----
    let tensor = prove_tensor_reduce(eval_mle, r_head, r_tail, v, transcript)
        .map_err(|_| TerminalError::TensorPrefixFailed)?;
    // ---- The clear partials e_i = ⟨a_weights, f_i⟩. ----
    let mut partials = Vec::with_capacity(f_blocks.len());
    for block in f_blocks {
        if block.len() > a_weights.len() {
            return Err(TerminalError::Shape {
                expected: a_weights.len(),
                got: block.len(),
            });
        }
        let mut e = ring.zero();
        for (w, f) in a_weights.iter().zip(block.iter()) {
            e = e
                .add(
                    &w.mul(f)
                        .map_err(|er| TerminalError::Ring(format!("{er:?}")))?,
                )
                .map_err(|er| TerminalError::Ring(format!("{er:?}")))?;
        }
        partials.push(e);
    }
    for e in &partials {
        transcript
            .append_message(b"a5-terminal-e", &e.to_bytes())
            .map_err(TerminalError::Transcript)?;
    }
    // ---- The public draws: ρ_pack and χ_blk (over F_{Q32²}). ----
    let mut rho_pack = Vec::with_capacity(2);
    for _ in 0..2 {
        rho_pack.push(
            challenge_fq2q(transcript, b"a5-terminal-rhopack")
                .map_err(|e| TerminalError::Ring(format!("{e:?}")))?,
        );
    }
    let mut chi_blk = Vec::with_capacity(partials.len());
    for _ in 0..partials.len() {
        chi_blk.push(
            challenge_fq2q(transcript, b"a5-terminal-chiblk")
                .map_err(|e| TerminalError::Ring(format!("{e:?}")))?,
        );
    }
    // ---- The response grind. ----
    let n_blocks = f_blocks.len();
    let s_blocks: Vec<Vec<RingElement>> = f_blocks
        .iter()
        .map(|block| {
            decompose_block(ring, block, state.response_base, state.response_digits)
                .map_err(TerminalError::Fold)
        })
        .collect::<Result<_, _>>()?;
    let block_digits = s_blocks.first().map(|b| b.len()).unwrap_or(0);
    let mut selected: Option<(u32, Vec<ShortChallenge>, Vec<RingElement>)> = None;
    let mut tries = 0u32;
    for nonce in 0..state.n_try {
        let mut t = transcript.clone();
        t.append_message(b"a5-terminal-nonce", &nonce.to_be_bytes())?;
        let mut challenges = Vec::with_capacity(n_blocks);
        let mut admissible = true;
        for i in 0..n_blocks {
            match fixed_filter(state, &mut t, i)? {
                Some(c) => challenges.push(c),
                None => {
                    admissible = false;
                    break;
                }
            }
        }
        if !admissible {
            tries += 1;
            continue;
        }
        // z = Σ_i c_i·s_i (ring-element challenges).
        let mut z = vec![ring.zero(); block_digits];
        for (i, s) in s_blocks.iter().enumerate() {
            let c_ring = embed(ring, &challenges[i]);
            for (zj, sj) in z.iter_mut().zip(s.iter()) {
                *zj = zj
                    .add(
                        &c_ring
                            .mul(sj)
                            .map_err(|er| TerminalError::Ring(format!("{er:?}")))?,
                    )
                    .map_err(|er| TerminalError::Ring(format!("{er:?}")))?;
            }
        }
        // The direct norm gate: ‖z‖² ≤ S.
        let norm_sq: u128 = z
            .iter()
            .map(|ze| u128::from(ze.euclidean_norm_squared()))
            .sum();
        if norm_sq > u128::from(state.s_bound) {
            tries += 1;
            continue;
        }
        selected = Some((nonce, challenges, z));
        break;
    }
    let (nonce, challenges, response) = selected.ok_or(TerminalError::GrindExhausted { tries })?;
    // Replay the selected nonce's grind on the MAIN transcript (the
    // attempts ran on clones; the verifier replays exactly this path):
    // absorb the nonce, redraw the challenges, then absorb z.
    transcript.append_message(b"a5-terminal-nonce", &nonce.to_be_bytes())?;
    let mut replayed = Vec::with_capacity(challenges.len());
    for i in 0..challenges.len() {
        match fixed_filter(state, transcript, i)? {
            Some(c) => replayed.push(c),
            None => return Err(TerminalError::Ring("grind replay diverged".into())),
        }
    }
    if replayed != challenges {
        return Err(TerminalError::Ring("grind replay mismatch".into()));
    }
    for ze in &response {
        transcript
            .append_message(b"a5-terminal-z", &ze.to_bytes())
            .map_err(TerminalError::Transcript)?;
    }
    // ---- The Rice encoding + budget gate. ----
    let coeffs: Vec<i64> = response
        .iter()
        .flat_map(|ze| {
            ze.coeffs()
                .iter()
                .map(|&c| centered(ring.modulus.q, c))
                .collect::<Vec<i64>>()
        })
        .collect();
    let encoded = rice_encode(&coeffs);
    let budget = rice_budget(coeffs.len(), state.s_bound);
    if encoded.len() > budget {
        return Err(TerminalError::RiceBudgetViolated {
            bytes: encoded.len(),
            budget,
        });
    }
    // ---- v_tensor (the Eq-165 row value over the revealed partials). ----
    let chi = chi_pack(ring, &rho_pack).map_err(|e| TerminalError::Ring(format!("{e:?}")))?;
    let mut v_tensor = Fq2Q::ZERO;
    for (e, w) in partials.iter().zip(chi_blk.iter()) {
        let tv =
            trace_functional(ring, e, &chi).map_err(|e| TerminalError::Ring(format!("{e:?}")))?;
        v_tensor = v_tensor.add(&w.mul(&tv));
    }
    Ok(TerminalProof {
        tensor,
        partials,
        nonce,
        challenges,
        response,
        response_rice: encoded,
        rho_pack,
        chi_blk,
        v_tensor,
    })
}

/// Verify the terminal's direct checks (Eq 163–165) + the grind
/// discipline + the Rice budget.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn verify_terminal(
    state: &TerminalState,
    proof: &TerminalProof,
    a_weights: &[RingElement],
    a_matrix: &[Vec<RingElement>],
    _eval_mle: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    transcript: &mut Transcript,
) -> Result<(), TerminalError> {
    let ring = &state.ring;
    if proof.nonce >= state.n_try {
        return Err(TerminalError::NonceOutOfRange { nonce: proof.nonce });
    }
    if proof.partials.len() != state.t_images.len() {
        return Err(TerminalError::Shape {
            expected: state.t_images.len(),
            got: proof.partials.len(),
        });
    }
    // ---- The tensor prefix. ----
    let _packed = verify_tensor_reduce(&proof.tensor, r_head, r_tail, v, transcript)
        .map_err(|_| TerminalError::TensorPrefixFailed)?;
    // ---- The partials + public draws replay. ----
    for e in &proof.partials {
        transcript
            .append_message(b"a5-terminal-e", &e.to_bytes())
            .map_err(TerminalError::Transcript)?;
    }
    let mut rho_pack = Vec::with_capacity(2);
    for _ in 0..2 {
        rho_pack.push(
            challenge_fq2q(transcript, b"a5-terminal-rhopack")
                .map_err(|e| TerminalError::Ring(format!("{e:?}")))?,
        );
    }
    if rho_pack != proof.rho_pack {
        return Err(TerminalError::DirectCheckFailed("rho_pack replay"));
    }
    let mut chi_blk = Vec::with_capacity(proof.partials.len());
    for _ in 0..proof.partials.len() {
        chi_blk.push(
            challenge_fq2q(transcript, b"a5-terminal-chiblk")
                .map_err(|e| TerminalError::Ring(format!("{e:?}")))?,
        );
    }
    if chi_blk != proof.chi_blk {
        return Err(TerminalError::DirectCheckFailed("chi_blk replay"));
    }
    // ---- The grind replay: the nonce, then the challenges. ----
    transcript.append_message(b"a5-terminal-nonce", &proof.nonce.to_be_bytes())?;
    let mut challenges = Vec::with_capacity(proof.challenges.len());
    for i in 0..proof.challenges.len() {
        match fixed_filter(state, transcript, i)? {
            Some(c) => challenges.push(c),
            None => return Err(TerminalError::DirectCheckFailed("nonce inadmissible")),
        }
    }
    if challenges != proof.challenges {
        return Err(TerminalError::DirectCheckFailed("challenge replay"));
    }
    // ---- The response replay + the direct norm gate. ----
    for ze in &proof.response {
        transcript
            .append_message(b"a5-terminal-z", &ze.to_bytes())
            .map_err(TerminalError::Transcript)?;
    }
    let norm_sq: u128 = proof
        .response
        .iter()
        .map(|ze| u128::from(ze.euclidean_norm_squared()))
        .sum();
    if norm_sq > u128::from(state.s_bound) {
        return Err(TerminalError::ResponseBoundViolated {
            norm_sq,
            bound: u128::from(state.s_bound),
        });
    }
    // ---- Eq 163: A_term·z = Σ_i c_i·t_i in R_term. ----
    let a_rows = a_matrix.len();
    let mut rhs = vec![ring.zero(); a_rows];
    for (i, t_img) in state.t_images.iter().enumerate() {
        if t_img.len() != a_rows {
            return Err(TerminalError::Shape {
                expected: a_rows,
                got: t_img.len(),
            });
        }
        let c_ring = embed(ring, &proof.challenges[i]);
        for (r, ti) in t_img.iter().enumerate() {
            rhs[r] = rhs[r]
                .add(
                    &c_ring
                        .mul(ti)
                        .map_err(|er| TerminalError::Ring(format!("{er:?}")))?,
                )
                .map_err(|er| TerminalError::Ring(format!("{er:?}")))?;
        }
    }
    let mut lhs = vec![ring.zero(); a_rows];
    for (r, row) in a_matrix.iter().enumerate() {
        for (a_ent, ze) in row.iter().zip(proof.response.iter()) {
            lhs[r] = lhs[r]
                .add(
                    &a_ent
                        .mul(ze)
                        .map_err(|er| TerminalError::Ring(format!("{er:?}")))?,
                )
                .map_err(|er| TerminalError::Ring(format!("{er:?}")))?;
        }
    }
    if lhs != rhs {
        return Err(TerminalError::DirectCheckFailed(
            "Eq163: A_term*z != Sum c_i*t_i",
        ));
    }
    // ---- Eq 164: Σ_i c_i·e_i = a_term^⊤·(G·z) — the fold-evaluation
    // row on the RECOMPOSED response (G·z = Σ_i c_i·f_i, so
    // ⟨a, G·z⟩ = Σ_i c_i·⟨a, f_i⟩ = Σ_i c_i·e_i). ----
    let mut lhs164 = ring.zero();
    for (i, e) in proof.partials.iter().enumerate() {
        let c_ring = embed(ring, &proof.challenges[i]);
        lhs164 = lhs164
            .add(
                &c_ring
                    .mul(e)
                    .map_err(|er| TerminalError::Ring(format!("{er:?}")))?,
            )
            .map_err(|er| TerminalError::Ring(format!("{er:?}")))?;
    }
    let z_rec = recompose_block(
        ring,
        &proof.response,
        state.response_base,
        state.response_digits,
    )
    .map_err(TerminalError::Fold)?;
    let mut rhs164 = ring.zero();
    for (w, ze) in a_weights.iter().zip(z_rec.iter()) {
        rhs164 = rhs164
            .add(
                &w.mul(ze)
                    .map_err(|er| TerminalError::Ring(format!("{er:?}")))?,
            )
            .map_err(|er| TerminalError::Ring(format!("{er:?}")))?;
    }
    if lhs164 != rhs164 {
        return Err(TerminalError::DirectCheckFailed(
            "Eq164: fold-evaluation row",
        ));
    }
    // ---- Eq 165: the trace row over the revealed partials. ----
    let ok = opening_row_trace(
        ring,
        &proof.partials,
        &proof.rho_pack,
        &proof.chi_blk,
        &proof.v_tensor,
    )
    .map_err(|e| TerminalError::Ring(format!("{e:?}")))?;
    if !ok {
        return Err(TerminalError::DirectCheckFailed("Eq165: trace row"));
    }
    // ---- The Rice budget + round-trip. ----
    let coeffs: Vec<i64> = proof
        .response
        .iter()
        .flat_map(|ze| {
            ze.coeffs()
                .iter()
                .map(|&c| centered(ring.modulus.q, c))
                .collect::<Vec<i64>>()
        })
        .collect();
    let budget = rice_budget(coeffs.len(), state.s_bound);
    if proof.response_rice.len() > budget {
        return Err(TerminalError::RiceBudgetViolated {
            bytes: proof.response_rice.len(),
            budget,
        });
    }
    let decoded = rice_decode(&proof.response_rice, coeffs.len())?;
    if decoded != coeffs {
        return Err(TerminalError::RiceDecodeFailed);
    }
    Ok(())
}

// ------------------------------------------------------ the recursion driver #

/// One level of the recursion driver's log.
#[derive(Clone, Debug)]
pub struct DriverLevel {
    pub level: u32,
    pub method: OpeningMethod,
    pub proof: FoldProof,
}

/// The driver's end-to-end proof: the fold chain + the terminal.
#[derive(Clone, Debug)]
pub struct RecursiveOpeningProof {
    pub levels: Vec<DriverLevel>,
    pub terminal: TerminalProof,
}

fn unit_point(block_len: usize, num_blocks: usize, value: u64) -> fold::OpeningPoint {
    fold::OpeningPoint {
        pos: (1..=block_len.ilog2().max(1) as usize)
            .map(|i| Goldilocks::from_u64((i * 997) as u64))
            .collect(),
        blk: (1..=num_blocks.ilog2().max(1) as usize)
            .map(|i| Goldilocks::from_u64((i * 1231) as u64))
            .collect(),
        value: Goldilocks::from_u64(value),
    }
}

/// Derive the terminal state's `t_i = A_term·s_i` images from the final
/// witness (the driver's binding specialization of §8.2).
#[allow(clippy::needless_range_loop)]
pub fn derive_terminal_state(
    ring: &RingConfig,
    f_blocks: &[Vec<RingElement>],
    a_matrix: &[Vec<RingElement>],
    state: &TerminalState,
) -> Result<Vec<Vec<RingElement>>, TerminalError> {
    let a_rows = a_matrix.len();
    let mut t_images = Vec::with_capacity(f_blocks.len());
    for block in f_blocks {
        let s = decompose_block(ring, block, state.response_base, state.response_digits)
            .map_err(TerminalError::Fold)?;
        let mut t = vec![ring.zero(); a_rows];
        for (r, row) in a_matrix.iter().enumerate() {
            for (a_ent, si) in row.iter().zip(s.iter()) {
                t[r] = t[r]
                    .add(
                        &a_ent
                            .mul(si)
                            .map_err(|er| TerminalError::Ring(format!("{er:?}")))?,
                    )
                    .map_err(|er| TerminalError::Ring(format!("{er:?}")))?;
            }
        }
        t_images.push(t);
    }
    Ok(t_images)
}

/// Chain `num_levels` folds (the §8.1 composition, level-0 params from
/// the caller, successor levels re-derived) and finish with the §8.2
/// terminal over the final group.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn prove_recursive_opening(
    params: &FoldParams,
    keys: &FoldKeys,
    source: &FoldSource,
    state: &TerminalState,
    a_weights: &[RingElement],
    a_matrix: &[Vec<RingElement>],
    eval_mle: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    num_levels: u32,
    transcript: &mut Transcript,
) -> Result<RecursiveOpeningProof, TerminalError> {
    let ring = &keys.ring;
    let mut levels: Vec<DriverLevel> = Vec::with_capacity(num_levels as usize);
    let mut current_params = params.clone();
    let mut current_blocks = source.blocks.clone();
    let mut current_bound = source.budget.beta();
    for level in 0..num_levels {
        let method = opening_method(level, level + 1 < num_levels);
        let point = unit_point(current_params.block_len, current_params.num_blocks, 7);
        let current = FoldSource::new(current_blocks.clone(), current_bound);
        let proof = if level == 0 {
            fold::prove_fold(&current_params, keys, &current, &point, transcript)?
        } else {
            let level_keys = FoldKeys::from_seed(&current_params, [7u8; 32])?;
            fold::prove_fold(&current_params, &level_keys, &current, &point, transcript)?
        };
        let successor = fold::successor_witness(&current_params, &proof);
        levels.push(DriverLevel {
            level,
            method,
            proof,
        });
        if level + 1 < num_levels {
            // The successor becomes the next level's single block
            // (zero-padded to a power-of-two block length).
            let padded_len = successor.len().max(1).next_power_of_two();
            let mut block = successor;
            block.resize(padded_len, ring.zero());
            let next_params = FoldParams {
                num_blocks: 1,
                block_len: padded_len,
                ..current_params.clone()
            };
            current_blocks = vec![block];
            current_bound = next_params.source_base / 2;
            current_params = next_params;
        } else {
            // The terminal group: the final successor as one block.
            let padded_len = successor.len().max(1).next_power_of_two();
            let mut block = successor;
            block.resize(padded_len, ring.zero());
            current_blocks = vec![block];
            current_bound = current_params.source_base / 2;
        }
    }
    // The terminal over the final group, with the state's t-images
    // derived from the final witness (the §8.2 binding).
    let final_blocks = current_blocks.clone();
    let mut state = state.clone();
    state.t_images = derive_terminal_state(ring, &final_blocks, a_matrix, &state)?;
    let terminal = prove_terminal(
        &state,
        &final_blocks,
        a_weights,
        a_matrix,
        eval_mle,
        r_head,
        r_tail,
        v,
        transcript,
    )?;
    Ok(RecursiveOpeningProof { levels, terminal })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ring() -> RingConfig {
        RingConfig::new(lattice_ring::Modulus32::Q_32, 4).unwrap()
    }

    fn fq(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn terminal_state(ring: &RingConfig, n_blocks: usize) -> TerminalState {
        TerminalState {
            ring: ring.clone(),
            t_images: vec![],
            s_bound: 1 << 24,
            n_try: 16,
            challenge_family: ShortChallengeFamily::FixedWeight {
                weight: 4,
                amplitude: 1,
            },
            gamma_cap: 1 << 10,
            response_base: 256,
            response_digits: 4,
        }
        .with_blocks(n_blocks)
    }

    impl TerminalState {
        fn with_blocks(mut self, _n: usize) -> Self {
            self.t_images = vec![];
            self
        }
    }

    /// The true evaluation claim `f(r_head, r_tail)` over E (the column
    /// recombination of the head-fixed halves).
    fn true_claim(f: &DenseMle, r_head: &Fq2, r_tail: &[Fq2]) -> Fq2 {
        let m = r_tail.len();
        let half = 1usize << m;
        let eq_w = |y: usize| -> Fq2 {
            if y == 0 {
                Fq2::ONE.sub(r_head)
            } else {
                *r_head
            }
        };
        let mut v = Fq2::ZERO;
        for y in 0..2 {
            let half_mle = DenseMle {
                num_vars: m,
                evaluations: f.evaluations[y * half..(y + 1) * half].to_vec(),
            };
            let mut acc = Fq2::ZERO;
            for w in 0..half {
                let mut eqv = Fq2::ONE;
                for (j, p) in r_tail.iter().enumerate() {
                    // Variable j is the index's bit (m−1−j).
                    let bit = if (w >> (m - 1 - j)) & 1 == 1 {
                        Fq2::ONE
                    } else {
                        Fq2::ZERO
                    };
                    eqv = eqv.mul(&p.mul(&bit).add(&Fq2::ONE.sub(p).mul(&Fq2::ONE.sub(&bit))));
                }
                acc = acc.add(&eqv.mul(&Fq2::from_base(half_mle.evaluations[w])));
            }
            v = v.add(&eq_w(y).mul(&acc));
        }
        v
    }

    #[test]
    fn opening_method_policy() {
        assert_eq!(
            opening_method(0, true),
            OpeningMethod::SubringCoefficientPacking
        );
        assert_eq!(
            opening_method(1, true),
            OpeningMethod::SubringCoefficientPacking
        );
        assert_eq!(opening_method(2, true), OpeningMethod::EvaluationTrace);
        assert_eq!(opening_method(0, false), OpeningMethod::EvaluationTrace);
        assert_eq!(opening_method(5, false), OpeningMethod::EvaluationTrace);
    }

    #[test]
    fn rice_roundtrip_and_budget() {
        let vals: Vec<i64> = vec![
            0,
            -1,
            1,
            15,
            -16,
            255,
            -256,
            4095,
            -4096,
            1 << 20,
            -(1 << 20),
        ];
        let enc = rice_encode(&vals);
        let dec = rice_decode(&enc, vals.len()).unwrap();
        assert_eq!(dec, vals);
        assert!(rice_budget(16, 1 << 10) < rice_budget(16, 1 << 20));
        assert!(rice_budget(16, 1 << 10) < rice_budget(64, 1 << 10));
        // A pathological stream (huge unary) exceeds a tight budget.
        let big = vec![1i64 << 30; 8];
        let enc_big = rice_encode(&big);
        let tight = rice_budget(8, 1 << 4);
        assert!(enc_big.len() > tight);
    }

    #[test]
    fn rice_decode_malformed_rejected() {
        assert!(rice_decode(&[0xff], 4).is_err());
        assert!(rice_decode(&[], 1).is_err());
    }

    #[test]
    fn terminal_prove_and_verify() {
        let ring = test_ring();
        let f_blocks: Vec<Vec<RingElement>> = vec![
            vec![
                RingElement::from_signed(&ring, &[1, -1, 2, -2]),
                RingElement::from_signed(&ring, &[3, -3, 4, -4]),
            ],
            vec![
                RingElement::from_signed(&ring, &[-5, 5, -6, 6]),
                RingElement::from_signed(&ring, &[7, -7, 8, -8]),
            ],
        ];
        let a_weights: Vec<RingElement> = vec![
            RingElement::from_signed(&ring, &[1, 0, 0, 0]),
            RingElement::from_signed(&ring, &[0, 1, 0, 0]),
        ];
        let a_matrix: Vec<Vec<RingElement>> = vec![
            vec![
                RingElement::from_signed(&ring, &[1, 0, 0, 0]),
                RingElement::from_signed(&ring, &[0, 1, 0, 0]),
            ],
            vec![
                RingElement::from_signed(&ring, &[0, 0, 1, 0]),
                RingElement::from_signed(&ring, &[0, 0, 0, 1]),
            ],
        ];
        let eval_mle = DenseMle::random(4, b"a5-mle");
        let r_head = Fq2::new(fq(0x11), fq(0x22));
        let r_tail: Vec<Fq2> = vec![
            Fq2::new(fq(0x33), fq(0x44)),
            Fq2::new(fq(0x55), fq(0x66)),
            Fq2::new(fq(0x77), fq(0x88)),
        ];
        let v = true_claim(&eval_mle, &r_head, &r_tail);
        let mut state = terminal_state(&ring, 2);
        state.t_images = derive_terminal_state(&ring, &f_blocks, &a_matrix, &state).unwrap();
        let mut t = Transcript::new_default(b"a5-test");
        let proof = prove_terminal(
            &state, &f_blocks, &a_weights, &a_matrix, &eval_mle, &r_head, &r_tail, &v, &mut t,
        )
        .unwrap();
        let mut tv = Transcript::new_default(b"a5-test");
        assert!(verify_terminal(
            &state, &proof, &a_weights, &a_matrix, &eval_mle, &r_head, &r_tail, &v, &mut tv
        )
        .is_ok());
    }

    #[test]
    fn terminal_wrong_claim_fails_closed() {
        let ring = test_ring();
        let f_blocks: Vec<Vec<RingElement>> = vec![vec![
            RingElement::from_signed(&ring, &[1, -1, 2, -2]),
            RingElement::from_signed(&ring, &[3, -3, 4, -4]),
        ]];
        let a_weights: Vec<RingElement> = vec![
            RingElement::from_signed(&ring, &[1, 0, 0, 0]),
            RingElement::from_signed(&ring, &[0, 1, 0, 0]),
        ];
        let a_matrix: Vec<Vec<RingElement>> = vec![vec![
            RingElement::from_signed(&ring, &[1, 0, 0, 0]),
            RingElement::from_signed(&ring, &[0, 1, 0, 0]),
        ]];
        let eval_mle = DenseMle::random(4, b"a5-wrong");
        let r_head = Fq2::new(fq(0x21), fq(0x43));
        let r_tail: Vec<Fq2> = (0..3u64)
            .map(|i| Fq2::new(fq(0x50 + i), fq(0x60 + i)))
            .collect();
        let mut state = terminal_state(&ring, 1);
        state.t_images = derive_terminal_state(&ring, &f_blocks, &a_matrix, &state).unwrap();
        let mut t = Transcript::new_default(b"a5-wrong");
        let err = prove_terminal(
            &state,
            &f_blocks,
            &a_weights,
            &a_matrix,
            &eval_mle,
            &r_head,
            &r_tail,
            &Fq2::ZERO, // a false claim
            &mut t,
        );
        assert!(matches!(err, Err(TerminalError::TensorPrefixFailed)));
    }

    #[test]
    fn terminal_tampered_t_images_rejected() {
        // Eq 163 breaks when the public t-images are tampered.
        let ring = test_ring();
        let f_blocks: Vec<Vec<RingElement>> = vec![vec![
            RingElement::from_signed(&ring, &[1, -1, 2, -2]),
            RingElement::from_signed(&ring, &[3, -3, 4, -4]),
        ]];
        let a_weights: Vec<RingElement> = vec![
            RingElement::from_signed(&ring, &[1, 0, 0, 0]),
            RingElement::from_signed(&ring, &[0, 1, 0, 0]),
        ];
        let a_matrix: Vec<Vec<RingElement>> = vec![vec![
            RingElement::from_signed(&ring, &[1, 0, 0, 0]),
            RingElement::from_signed(&ring, &[0, 1, 0, 0]),
        ]];
        let eval_mle = DenseMle::random(4, b"a5-tt");
        let r_head = Fq2::new(fq(0x31), fq(0x53));
        let r_tail: Vec<Fq2> = (0..3u64)
            .map(|i| Fq2::new(fq(0x60 + i), fq(0x70 + i)))
            .collect();
        let v = true_claim(&eval_mle, &r_head, &r_tail);
        let mut state = terminal_state(&ring, 1);
        state.t_images = derive_terminal_state(&ring, &f_blocks, &a_matrix, &state).unwrap();
        let mut t = Transcript::new_default(b"a5-tt");
        let proof = prove_terminal(
            &state, &f_blocks, &a_weights, &a_matrix, &eval_mle, &r_head, &r_tail, &v, &mut t,
        )
        .unwrap();
        // Tamper the state's t-image (the public input).
        let mut bad_state = state.clone();
        bad_state.t_images[0][0] = bad_state.t_images[0][0].add(&ring.constant(1)).unwrap();
        let mut tv = Transcript::new_default(b"a5-tt");
        assert!(matches!(
            verify_terminal(
                &bad_state, &proof, &a_weights, &a_matrix, &eval_mle, &r_head, &r_tail, &v, &mut tv
            ),
            Err(TerminalError::DirectCheckFailed(
                "Eq163: A_term*z != Sum c_i*t_i"
            ))
        ));
    }

    #[test]
    fn terminal_rice_budget_violation_rejected() {
        // Overstuffed Rice bytes: the verifier's budget gate rejects.
        let ring = test_ring();
        let f_blocks: Vec<Vec<RingElement>> = vec![vec![
            RingElement::from_signed(&ring, &[1, -1, 2, -2]),
            RingElement::from_signed(&ring, &[3, -3, 4, -4]),
        ]];
        let a_weights: Vec<RingElement> = vec![
            RingElement::from_signed(&ring, &[1, 0, 0, 0]),
            RingElement::from_signed(&ring, &[0, 1, 0, 0]),
        ];
        let a_matrix: Vec<Vec<RingElement>> = vec![vec![
            RingElement::from_signed(&ring, &[1, 0, 0, 0]),
            RingElement::from_signed(&ring, &[0, 1, 0, 0]),
        ]];
        let eval_mle = DenseMle::random(4, b"a5-rice");
        let r_head = Fq2::new(fq(0x41), fq(0x63));
        let r_tail: Vec<Fq2> = (0..3u64)
            .map(|i| Fq2::new(fq(0x70 + i), fq(0x80 + i)))
            .collect();
        let v = true_claim(&eval_mle, &r_head, &r_tail);
        let mut state = terminal_state(&ring, 1);
        state.t_images = derive_terminal_state(&ring, &f_blocks, &a_matrix, &state).unwrap();
        let mut t = Transcript::new_default(b"a5-rice");
        let mut proof = prove_terminal(
            &state, &f_blocks, &a_weights, &a_matrix, &eval_mle, &r_head, &r_tail, &v, &mut t,
        )
        .unwrap();
        // Pad the Rice bytes past the budget (garbage tail).
        proof.response_rice.extend_from_slice(&[0u8; 8192]);
        let mut tv = Transcript::new_default(b"a5-rice");
        assert!(matches!(
            verify_terminal(
                &state, &proof, &a_weights, &a_matrix, &eval_mle, &r_head, &r_tail, &v, &mut tv
            ),
            Err(TerminalError::RiceBudgetViolated { .. })
        ));
    }

    #[test]
    fn grind_exhaustion_fails_closed() {
        let ring = test_ring();
        let f_blocks: Vec<Vec<RingElement>> = vec![vec![ring.one()]];
        let a_weights = vec![ring.one()];
        let a_matrix = vec![vec![ring.one()]];
        let eval_mle = DenseMle::random(4, b"a5-grind");
        let r_head = Fq2::new(fq(1), fq(2));
        let r_tail: Vec<Fq2> = (0..3u64).map(|i| Fq2::new(fq(3 + i), fq(4 + i))).collect();
        let v = true_claim(&eval_mle, &r_head, &r_tail);
        let mut state = terminal_state(&ring, 1);
        state.n_try = 0; // no nonce is ever admissible
        state.t_images = derive_terminal_state(&ring, &f_blocks, &a_matrix, &state).unwrap();
        let mut t = Transcript::new_default(b"a5-grind");
        let err = prove_terminal(
            &state, &f_blocks, &a_weights, &a_matrix, &eval_mle, &r_head, &r_tail, &v, &mut t,
        );
        assert!(matches!(
            err,
            Err(TerminalError::GrindExhausted { tries: 0 })
        ));
    }

    #[test]
    fn driver_chains_folds_into_terminal() {
        use crate::fold::{FoldKeys, FoldParams, FoldSource};
        let ring = test_ring();
        let params = FoldParams {
            log_n: 4,
            num_blocks: 2,
            block_len: 2,
            source_base: 256,
            source_digits: 4,
            inner_rows: 1,
            inner_base: 2048,
            inner_digits: 3,
            outer_rows: 1,
            opening_rows: 1,
            response_base: 256,
            response_digits: 4,
            challenge_weight: 4,
        };
        let keys = FoldKeys::from_seed(&params, [41u8; 32]).unwrap();
        let blocks: Vec<Vec<RingElement>> = (0..params.num_blocks)
            .map(|i| {
                let mut tag = b"a5-driver".to_vec();
                tag.push(i as u8);
                (0..params.block_len)
                    .map(|j| ring.random(&[tag.clone(), vec![j as u8]].concat()))
                    .collect()
            })
            .collect();
        let source = FoldSource::new(blocks, params.source_base / 2);
        // Generously sized terminal weights (the zips truncate to the
        // final block's true length).
        let a_weights: Vec<RingElement> = (0..4096)
            .map(|i| ring.constant((i as u32 % 7 + 1) * 3))
            .collect();
        let a_matrix: Vec<Vec<RingElement>> = (0..2)
            .map(|r| {
                (0..4096)
                    .map(|i| ring.constant((i as u32 % 5 + 1) + r as u32))
                    .collect()
            })
            .collect();
        let eval_mle = DenseMle::random(4, b"a5-driver-mle");
        let r_head = Fq2::new(fq(0x91), fq(0xa3));
        let r_tail: Vec<Fq2> = (0..3u64)
            .map(|i| Fq2::new(fq(0xb0 + i), fq(0xc0 + i)))
            .collect();
        let v = true_claim(&eval_mle, &r_head, &r_tail);
        let state = terminal_state(&ring, 1);
        let mut t = Transcript::new_default(b"a5-driver");
        let proof = prove_recursive_opening(
            &params, &keys, &source, &state, &a_weights, &a_matrix, &eval_mle, &r_head, &r_tail,
            &v, 2, &mut t,
        )
        .unwrap();
        assert_eq!(proof.levels.len(), 2);
        // The opening-method policy across the chain.
        assert_eq!(
            proof.levels[0].method,
            OpeningMethod::SubringCoefficientPacking
        );
        assert_eq!(proof.levels[1].method, OpeningMethod::EvaluationTrace);
    }
}
