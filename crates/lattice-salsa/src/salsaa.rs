//! SALSAA A2–A4 (ePrint 2025/2124): the Π_norm+ / Π_bin / staircase /
//! VDF protocol stack over R_q — Wave 7 item 7.11 (ported from the
//! lattice-zk-lab reference implementation to this workspace's
//! conventions).
//!
//! * **A2 — Π_norm / Π_norm+** (Fig. 4): the conjugate inner-product norm
//!   check `t = ⟨w, w̄⟩` proven directly with the O(m) trick (the prover
//!   computes t coefficient-wise instead of committing a degree-m
//!   polynomial), the balanced-trace integer gate `Tr(t) = n‖cf(w)‖² ≤
//!   n·β²`, the c-power row-batching ladder, and ONE combined degree-2
//!   ring sumcheck `Σ_z [α·MLE[h](z)·MLE[w](z) + MLE[w](z)·MLE[w̄](z)] =
//!   α·s + t`; the verifier appends the eq-row for the final evaluation
//!   claim (Π_mle, the paper's free step) producing the reduced instance.
//! * **A3 — Π_bin** (Fig. 5): binariness `t = ⟨w, 1° − w⟩` with
//!   `1° = Σ_k X^k`; the verifier's `Tr(t) = 0` gate in the balanced
//!   representative forces every coefficient into {0,1} (Lemma 4.11) —
//!   wraparound-free because 1°−w entries are ≥ −(q−1)/2 centred and the
//!   product with conj(w) reproduces the exact integer sum.
//! * **A3 — staircase RoK** (Fig. 6): Ξ^stair `A W_0 = Y0`,
//!   `B W_{j−1} + A W_j = 0`, `B W_{K−1} = Y1` folded into the single
//!   degree-3 claim `Σ_z p(z_step)·d(z_inner)·MLE[W](z) = s` with the
//!   c-power row batching `d = Σ_ρ c^ρ A_ρ + c^{m̄+ρ} B_ρ`, geometric
//!   `p_j = c^{m̄·j}` and `s = c0·Y0 + c^{K·m̄}·c0·Y1`.
//! * **A4 — the VDF application** (§6): the delay chain
//!   `y_{i+1} = A·G^{−1}(−y_i)` proved as the binary staircase
//!   `G W_0 = −y0; A W_{j−1} + G W_j = 0; A W_{K−1} = yT` (K = T steps,
//!   m̄ = 1) composed with Π_bin over the whole flat witness
//!   (`Π_as = Π_staircase ∘ Π_bin`).
//!
//! Documented deviations (kernel scale, matching the Python lab's gap
//! ledger): round challenges are uniform `Z_q` scalars (the paper's
//! CRT-slot ring challenges are a size optimisation — the lzx
//! `pikkufold_lrp` discipline); the Ajtai commitment layer uses this
//! workspace's `AjtaiPublicKey` with seed-derived transparent keys; the
//! power-of-two ring precondition (m, n̄, K powers of two) is enforced
//! fail-closed.

use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_ring::{RingConfig, RingElement};

use crate::ring_sc::{
    challenge_ring_elt, conj, eq_table_ring, mle_eval_ring, norm_conjugate_inner,
    ring_dot, ring_sc_prove, ring_sc_verify, ProductClaim, RingScError, RingScProof,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SalsaaError {
    RingSc(RingScError),
    Transcript(TranscriptError),
    Ajtai(lattice_commitment::ajtai::AjtaiError),
    /// The balanced-trace gate failed (wraparound or non-binary witness).
    TraceGateFailed { trace: i64 },
    /// A shape precondition (power of two / matching dimensions) failed.
    Shape { expected: usize, got: usize },
    /// The terminal identity failed.
    TerminalFailed,
    /// The claimed value does not match the honest evaluation.
    ClaimMismatch,
}

impl From<RingScError> for SalsaaError {
    fn from(e: RingScError) -> Self {
        SalsaaError::RingSc(e)
    }
}
impl From<lattice_ring::RingError> for SalsaaError {
    fn from(e: lattice_ring::RingError) -> Self {
        SalsaaError::RingSc(RingScError::Ring(e))
    }
}
impl From<TranscriptError> for SalsaaError {
    fn from(e: TranscriptError) -> Self {
        SalsaaError::Transcript(e)
    }
}

// ---------------------------------------------------------------------------
// The committed linear-relation instance
// ---------------------------------------------------------------------------

/// Committed linear relation: `⟨row_i, w⟩ = target_i` for all i,
/// `com = A_com·w` (Ajtai), `‖cf(w)‖₂ ≤ beta`.
#[derive(Clone, Debug)]
pub struct SalsaInstance {
    pub rows: Vec<Vec<RingElement>>,
    pub targets: Vec<RingElement>,
    /// Prover-side witness (None on the verifier's copy).
    pub w: Option<Vec<RingElement>>,
    pub com: Vec<RingElement>,
    pub beta: u64,
    pub ring: RingConfig,
}

impl SalsaInstance {
    /// Create an honest instance: commit w, derive the targets.
    pub fn create(
        ring: &RingConfig,
        w: &[RingElement],
        rows: &[Vec<RingElement>],
        beta: u64,
        seed: [u8; 32],
    ) -> Result<(Self, AjtaiPublicKey), SalsaaError> {
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: w.len(),
            norm_bound: beta as u32,
        };
        let pk = AjtaiPublicKey::from_seed(params, seed).map_err(SalsaaError::Ajtai)?;
        let com = pk.commit(w).map_err(SalsaaError::Ajtai)?.rows;
        let targets = rows
            .iter()
            .map(|row| ring_dot(row, w))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((
            SalsaInstance {
                rows: rows.to_vec(),
                targets,
                w: Some(w.to_vec()),
                com,
                beta,
                ring: ring.clone(),
            },
            pk,
        ))
    }

    /// The witness length (row width).
    pub fn m(&self) -> usize {
        self.rows
            .first()
            .map(|r| r.len())
            .or_else(|| self.w.as_ref().map(|w| w.len()))
            .unwrap_or(0)
    }

    /// log2 of the padded hypercube dimension.
    pub fn mu(&self) -> Result<usize, SalsaaError> {
        let m = self.m();
        if m == 0 || !m.is_power_of_two() {
            return Err(SalsaaError::Shape {
                expected: 0,
                got: m,
            });
        }
        Ok(m.trailing_zeros() as usize)
    }

    /// Π_mle (Fig. 2's free step): append the eq-row with target
    /// `MLE[w](r) = value`.
    pub fn append_mle_row(&mut self, point: &[u32], value: RingElement) {
        self.rows.push(eq_table_ring(&self.ring, point));
        self.targets.push(value);
    }

    /// Π_batch power ladder: fold ALL rows with `c^i` weights into (h, s).
    pub fn fold_rows(&self, c: &RingElement) -> Result<(Vec<RingElement>, RingElement), SalsaaError> {
        let ring = &self.ring;
        let m = self.m();
        let mut h = vec![ring.zero(); m];
        let mut s = ring.zero();
        let mut cp = ring.one();
        for (row, tgt) in self.rows.iter().zip(&self.targets) {
            for j in 0..m {
                h[j] = h[j].add(&row[j].mul(&cp)?)?;
            }
            s = s.add(&tgt.mul(&cp)?)?;
            cp = cp.mul(c)?;
        }
        Ok((h, s))
    }
}

/// `c^e` in R_q (square-and-multiply; kernel scale).
fn pow_ring(c: &RingElement, e: usize) -> Result<RingElement, SalsaaError> {
    let mut acc = c.config().one();
    let mut base = c.clone();
    let mut e = e;
    while e > 0 {
        if e & 1 == 1 {
            acc = acc.mul(&base)?;
        }
        base = base.mul(&base)?;
        e >>= 1;
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// A2 — Π_norm / Π_norm+  (Fig. 4)
// ---------------------------------------------------------------------------

/// The Π_norm+ proof: the conjugate inner product, the combined sumcheck,
/// and the two final openings `v0 = MLE[w](r)`, `v1 = MLE[w̄](r)`.
#[derive(Clone, Debug)]
pub struct NormPlusProof {
    pub t: RingElement,
    pub sumcheck: RingScProof,
    pub v0: RingElement,
    pub v1: RingElement,
    pub point: Vec<u32>,
}

/// Π_norm+ prover: (1) `t = ⟨w, w̄⟩` directly; (2) batch the linear rows
/// with the c-power ladder; (3) ONE combined degree-2 sumcheck
/// `Σ_z [α·MLE[h](z)·MLE[w](z) + MLE[w](z)·MLE[w̄](z)] = α·s + t`;
/// (4) final openings v0, v1.
pub fn norm_plus_prove(
    inst: &SalsaInstance,
    transcript: &mut Transcript,
) -> Result<NormPlusProof, SalsaaError> {
    let ring = &inst.ring;
    let w = inst.w.as_ref().ok_or(SalsaaError::Shape {
        expected: 1,
        got: 0,
    })?;
    let wbar: Vec<RingElement> = w.iter().map(conj).collect();
    let t = norm_conjugate_inner(w)?;
    let mut groups = vec![ProductClaim {
        tables: vec![wbar.clone(), w.clone()],
        value: t.clone(),
    }];
    let mut combiners = vec![ring.one()];
    if !inst.rows.is_empty() {
        let c = challenge_ring_elt(transcript, b"salsaa:batch-c", ring)?;
        let (h_row, s_batch) = inst.fold_rows(&c)?;
        let alpha = challenge_ring_elt(transcript, b"salsaa:alpha", ring)?;
        groups.push(ProductClaim {
            tables: vec![h_row, w.clone()],
            value: s_batch,
        });
        combiners.push(alpha);
    }
    let sumcheck = ring_sc_prove(ring, &groups, &combiners, transcript)?;
    let point = sumcheck.point.clone();
    let v0 = mle_eval_ring(w, &point)?;
    let v1 = mle_eval_ring(&wbar, &point)?;
    Ok(NormPlusProof {
        t,
        sumcheck,
        v0,
        v1,
        point,
    })
}

/// Π_norm+ verifier: trace gate, round checks, terminal check, and the
/// Π_mle row append producing the REDUCED instance (n+1 rows). Returns
/// the reduced instance on success.
pub fn norm_plus_verify(
    inst: &SalsaInstance,
    proof: &NormPlusProof,
    transcript: &mut Transcript,
) -> Result<SalsaInstance, SalsaaError> {
    let ring = &inst.ring;
    // ---- trace gate (wraparound-free regime): Tr(t) = n‖w‖² ∈ [0, nβ²]
    let tr = super::ring_sc::trace_balanced(&proof.t);
    if tr < 0 || tr > (ring.n() as i64) * (inst.beta as i64) * (inst.beta as i64) {
        return Err(SalsaaError::TraceGateFailed { trace: tr });
    }
    // ---- rebuild the combined claim (verifier-side replay — each
    // challenge is drawn EXACTLY once, mirroring the prover's order)
    let mut combiners = vec![ring.one()];
    let mut h_row: Option<Vec<RingElement>> = None;
    let mut alpha_and_s: Option<(RingElement, RingElement)> = None;
    if !inst.rows.is_empty() {
        let c = challenge_ring_elt(transcript, b"salsaa:batch-c", ring)?;
        let (h, s_batch) = inst.fold_rows(&c)?;
        let alpha = challenge_ring_elt(transcript, b"salsaa:alpha", ring)?;
        combiners.push(alpha.clone());
        h_row = Some(h);
        alpha_and_s = Some((alpha, s_batch));
    }
    // combined target: t + alpha * s_batch
    let mut target = proof.t.clone();
    if let Some((alpha, s_batch)) = &alpha_and_s {
        target = target.add(&s_batch.mul(alpha)?)?;
    }
    let mu = inst.mu()?;
    let last = ring_sc_verify(ring, 2, mu, &target, &proof.sumcheck, transcript)?;
    // ---- terminal: v1·v0 (norm group) + α·MLE[h](r)·v0 (batch group)
    let mut total = proof.v0.mul(&proof.v1)?;
    if let Some(h) = &h_row {
        let hv = mle_eval_ring(h, &proof.point)?;
        total = total.add(&hv.mul(&proof.v0)?.mul(&combiners[1])?)?;
    }
    if total != last {
        return Err(SalsaaError::TerminalFailed);
    }
    // ---- Π_mle: append the eq-row for the evaluation claim
    let mut reduced = inst.clone();
    reduced.w = None;
    reduced.append_mle_row(&proof.point, proof.v0.clone());
    Ok(reduced)
}

// ---------------------------------------------------------------------------
// A3 — Π_bin (Fig. 5)
// ---------------------------------------------------------------------------

/// `1° := Σ_k X^k` — the all-ones-coefficient ring element.
pub fn one_degree(ring: &RingConfig) -> RingElement {
    RingElement::from_coeffs(ring, vec![1; ring.n()])
}

/// `t = ⟨w, 1° − w⟩ = Σ_j conj(w_j)·(1° − w_j)`.
pub fn bin_conjugate_inner(w: &[RingElement]) -> Result<RingElement, SalsaaError> {
    let ring = w
        .first()
        .map(|e| e.config().clone())
        .ok_or(SalsaaError::Shape {
            expected: 1,
            got: 0,
        })?;
    let one = one_degree(&ring);
    let mut acc = ring.zero();
    for x in w {
        let minus = one.sub(x)?;
        acc = acc.add(&conj(x).mul(&minus)?)?;
    }
    Ok(acc)
}

/// The Π_bin proof.
#[derive(Clone, Debug)]
pub struct BinProof {
    pub t: RingElement,
    pub sumcheck: RingScProof,
    pub v0: RingElement,
    pub v1: RingElement,
    pub point: Vec<u32>,
}

/// Π_bin prover (Fig. 5): `t = ⟨w, 1° − w⟩`; the degree-2 sumcheck
/// `Σ_z MLE[w̄](z)·MLE[1°−w](z) = t`.
pub fn bin_prove(
    inst: &SalsaInstance,
    transcript: &mut Transcript,
) -> Result<BinProof, SalsaaError> {
    let ring = &inst.ring;
    let w = inst.w.as_ref().ok_or(SalsaaError::Shape {
        expected: 1,
        got: 0,
    })?;
    let one = one_degree(ring);
    let wbar: Vec<RingElement> = w.iter().map(conj).collect();
    let minus: Vec<RingElement> = w.iter().map(|x| one.sub(x)).collect::<Result<_, _>>()?;
    let t = bin_conjugate_inner(w)?;
    let claim = ProductClaim {
        tables: vec![wbar, minus],
        value: t.clone(),
    };
    let sumcheck = ring_sc_prove(ring, &[claim], &[ring.one()], transcript)?;
    let point = sumcheck.point.clone();
    let v0 = mle_eval_ring(
        &{
            let wbar2: Vec<RingElement> = w.iter().map(conj).collect();
            wbar2
        },
        &point,
    )?;
    let v1 = mle_eval_ring(
        &w.iter().map(|x| one.sub(x)).collect::<Result<Vec<_>, _>>()?,
        &point,
    )?;
    Ok(BinProof {
        t,
        sumcheck,
        v0,
        v1,
        point,
    })
}

/// Π_bin verifier: `Tr(t) = 0` in the balanced representative
/// (Lemma 4.11 — forces all coefficients into {0,1}), round checks,
/// terminal `v0·v1 == last`.
pub fn bin_verify(
    inst: &SalsaInstance,
    proof: &BinProof,
    transcript: &mut Transcript,
) -> Result<(), SalsaaError> {
    let ring = &inst.ring;
    let tr = super::ring_sc::trace_balanced(&proof.t);
    if tr != 0 {
        return Err(SalsaaError::TraceGateFailed { trace: tr });
    }
    let mu = inst.mu()?;
    let last = ring_sc_verify(ring, 2, mu, &proof.t, &proof.sumcheck, transcript)?;
    if proof.v0.mul(&proof.v1)? != last {
        return Err(SalsaaError::TerminalFailed);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// A3 — the staircase RoK (Fig. 6)
// ---------------------------------------------------------------------------

/// Ξ^stair: `A W_0 = Y0`; `B W_{j−1} + A W_j = 0` (j = 1..K−1);
/// `B W_{K−1} = Y1`. W parsed into K row-blocks of n̄ witness slots.
#[derive(Clone, Debug)]
pub struct StaircaseInstance {
    pub a: Vec<Vec<RingElement>>, // m̄ x n̄
    pub b: Vec<Vec<RingElement>>, // m̄ x n̄
    /// K blocks, each n̄ ring elements (prover side).
    pub w_blocks: Option<Vec<Vec<RingElement>>>,
    pub k_blocks: usize,
    pub n_bar: usize,
    pub y0: Vec<RingElement>, // m̄
    pub y1: Vec<RingElement>, // m̄
    pub ring: RingConfig,
}

impl StaircaseInstance {
    pub fn m_bar(&self) -> usize {
        self.a.len()
    }

    pub fn flat_w(&self) -> Result<Vec<RingElement>, SalsaaError> {
        let blocks = self.w_blocks.as_ref().ok_or(SalsaaError::Shape {
            expected: 1,
            got: 0,
        })?;
        Ok(blocks.iter().flatten().cloned().collect())
    }

    /// All staircase constraints for the embedded witness.
    pub fn honest(&self) -> Result<bool, SalsaaError> {
        let blocks = self.w_blocks.as_ref().ok_or(SalsaaError::Shape {
            expected: 1,
            got: 0,
        })?;
        let k = blocks.len();
        // A W_0 = Y0
        for (row, y) in self.a.iter().zip(&self.y0) {
            if ring_dot(row, &blocks[0])? != *y {
                return Ok(false);
            }
        }
        // B W_{j-1} + A W_j = 0
        for j in 1..k {
            for (r, row) in self.a.iter().enumerate() {
                let lhs = ring_dot(row, &blocks[j])?
                    .add(&ring_dot(&self.b[r], &blocks[j - 1])?)?;
                if !lhs.is_zero() {
                    return Ok(false);
                }
            }
        }
        // B W_{K-1} = Y1
        for (row, y) in self.b.iter().zip(&self.y1) {
            if ring_dot(row, &blocks[k - 1])? != *y {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// The staircase proof: the degree-3 sumcheck + the derived data.
#[derive(Clone, Debug)]
pub struct StaircaseProof {
    pub sumcheck: RingScProof,
    pub c: RingElement,
    pub d_row: Vec<RingElement>,
    pub s: RingElement,
    pub point: Vec<u32>,
}

/// Π_ (Fig. 6) prover: both parties derive (c0, p, d, s) from the
/// challenge c; the claim is
/// `Σ_z MLE[p](z_step)·MLE[d](z_inner)·MLE[W](z) = s`
/// — a degree-3 product claim over the full hypercube
/// (μ = log K + log n̄).
pub fn staircase_prove(
    inst: &StaircaseInstance,
    transcript: &mut Transcript,
) -> Result<StaircaseProof, SalsaaError> {
    let ring = &inst.ring;
    let c = challenge_ring_elt(transcript, b"salsaa:stair-c", ring)?;
    let (d_row, p, s, total) = staircase_derive(inst, &c)?;
    let w_flat = inst.flat_w()?;
    // sanity: the claim value
    let mut claim = ring.zero();
    for z in 0..total {
        claim = claim.add(&p[(z / inst.n_bar) % inst.k_blocks].mul(&d_row[z % inst.n_bar])?.mul(&w_flat[z])?)?;
    }
    if claim != s {
        return Err(SalsaaError::ClaimMismatch);
    }
    // full-hypercube tables
    let p_table: Vec<RingElement> = (0..total)
        .map(|z| p[(z / inst.n_bar) % inst.k_blocks].clone())
        .collect();
    let d_table: Vec<RingElement> = (0..total).map(|z| d_row[z % inst.n_bar].clone()).collect();
    let sc_claim = ProductClaim {
        tables: vec![p_table, d_table, w_flat],
        value: s.clone(),
    };
    let sumcheck = ring_sc_prove(ring, &[sc_claim], &[ring.one()], transcript)?;
    let point = sumcheck.point.clone();
    Ok(StaircaseProof {
        sumcheck,
        c,
        d_row,
        s,
        point,
    })
}

/// The verifier-side derivation of (d, p, s) from c — shared by prove and
/// verify so the two cannot drift.
fn staircase_derive(
    inst: &StaircaseInstance,
    c: &RingElement,
) -> Result<(Vec<RingElement>, Vec<RingElement>, RingElement, usize), SalsaaError> {
    let ring = &inst.ring;
    let (m_bar, n_bar, k) = (inst.m_bar(), inst.n_bar, inst.k_blocks);
    if !n_bar.is_power_of_two() || !k.is_power_of_two() || n_bar == 0 || k == 0 {
        return Err(SalsaaError::Shape {
            expected: 0,
            got: n_bar + k,
        });
    }
    // d = Σ_ρ c^ρ A_ρ + c^{m̄+ρ} B_ρ
    let mut d_row = vec![ring.zero(); n_bar];
    for rho in 0..m_bar {
        let cp = pow_ring(c, rho)?;
        let cp2 = pow_ring(c, m_bar + rho)?;
        for j in 0..n_bar {
            d_row[j] = d_row[j].add(&cp.mul(&inst.a[rho][j])?)?;
            d_row[j] = d_row[j].add(&cp2.mul(&inst.b[rho][j])?)?;
        }
    }
    // p_j = (c^{m̄})^j; s = c0·Y0 + c^{K·m̄}·c0·Y1
    let cm = pow_ring(c, m_bar)?;
    let p: Vec<RingElement> = (0..k).map(|j| pow_ring(&cm, j)).collect::<Result<_, _>>()?;
    let c_k = pow_ring(&cm, k)?;
    let mut s = ring.zero();
    for (i, y) in inst.y0.iter().enumerate() {
        s = s.add(&pow_ring(c, i)?.mul(y)?)?;
    }
    for (i, y) in inst.y1.iter().enumerate() {
        s = s.add(&c_k.mul(&pow_ring(c, i)?)?.mul(y)?)?;
    }
    Ok((d_row, p, s, k * n_bar))
}

/// Π_ verifier: re-derive (c, d, p, s), check the rounds and the
/// terminal `p_v·d_v·w_v == last`.
pub fn staircase_verify(
    inst: &StaircaseInstance,
    proof: &StaircaseProof,
    transcript: &mut Transcript,
) -> Result<(), SalsaaError> {
    let ring = &inst.ring;
    let c = challenge_ring_elt(transcript, b"salsaa:stair-c", ring)?;
    if c != proof.c {
        return Err(SalsaaError::ClaimMismatch);
    }
    let (d_row, _p, s, total) = staircase_derive(inst, &c)?;
    if d_row != proof.d_row || s != proof.s {
        return Err(SalsaaError::ClaimMismatch);
    }
    let mu = (total.trailing_zeros()) as usize;
    let last = ring_sc_verify(ring, 3, mu, &s, &proof.sumcheck, transcript)?;
    // terminal: prod of final values == last
    let fv = &proof.sumcheck.final_values[0];
    if fv.len() != 3 {
        return Err(SalsaaError::Shape {
            expected: 3,
            got: fv.len(),
        });
    }
    if fv[0].mul(&fv[1])?.mul(&fv[2])? != last {
        return Err(SalsaaError::TerminalFailed);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// A4 — the VDF application (§6): binary staircase chain
// ---------------------------------------------------------------------------

/// VDF parameters: the delay matrix (1 x n for the kernel) and the chain
/// length T (power of two).
pub struct VdfParams {
    pub a: Vec<RingElement>, // 1 x n delay row
    pub t_steps: usize,
}

/// The number of binary gadget layers needed to cover every coefficient of
/// R_q: `⌈log2 q⌉` (32 for Q_32) — the VDF staircase's n̄.
pub fn gadget_layers(ring: &RingConfig) -> usize {
    let q = ring.modulus.q;
    (32 - q.leading_zeros()) as usize
}

/// `G^{−1}(−y)`: one binary digit LAYER per bit position of every
/// coefficient of −y (`G = (2^0, …, 2^{L−1})` per coefficient, L =
/// `gadget_layers` — full coverage of [0, q) so `G·W ≡ −y mod q` exactly).
pub fn gadget_binary_neg(ring: &RingConfig, y: &RingElement) -> Vec<RingElement> {
    let neg = y.neg();
    let layers = gadget_layers(ring);
    let mut out = Vec::with_capacity(layers);
    for r in 0..layers {
        let mut layer = vec![0u32; ring.n()];
        for (k, &c) in neg.coeffs().iter().enumerate() {
            layer[k] = (c >> r) & 1;
        }
        out.push(RingElement::from_coeffs(ring, layer));
    }
    out
}

/// `G = (2^0, 2^1, …, 2^{L−1})` as a 1 x L row of scalar ring elements
/// (L = `gadget_layers`).
pub fn gadget_row(ring: &RingConfig) -> Vec<RingElement> {
    (0..gadget_layers(ring))
        .map(|r| {
            let v = (1u64 << r) % u64::from(ring.modulus.q);
            ring.constant(v as u32)
        })
        .collect()
}

/// Evaluate the delay chain `y_{i+1} = A·G^{−1}(−y_i)`; returns
/// (y_T, the binary layer chain).
pub fn vdf_eval(
    params: &VdfParams,
    y0: &RingElement,
) -> Result<(RingElement, Vec<Vec<RingElement>>), SalsaaError> {
    let ring = y0.config();
    let mut y = y0.clone();
    let mut chain: Vec<Vec<RingElement>> = Vec::with_capacity(params.t_steps);
    for _ in 0..params.t_steps {
        let w = gadget_binary_neg(ring, &y);
        chain.push(w.clone());
        let mut acc = ring.zero();
        for (a, x) in params.a.iter().zip(&w) {
            acc = acc.add(&a.mul(x)?)?;
        }
        y = acc;
    }
    Ok((y, chain))
}

/// Prove the VDF chain as a BINARY STAIRCASE (§6): the stacked system
/// `G W_0 = −y0; A W_{j−1} + G W_j = 0; A W_{K−1} = yT` with m̄ = 1,
/// K = T, n̄ = n layers, PLUS the binariness of the whole flat witness via
/// Π_bin. `Π_as = Π_staircase ∘ Π_bin`.
pub fn vdf_prove(
    params: &VdfParams,
    y0: &RingElement,
    y_t: &RingElement,
    transcript: &mut Transcript,
) -> Result<(StaircaseProof, BinProof), SalsaaError> {
    let ring = y0.config().clone();
    let (y_check, chain) = vdf_eval(params, y0)?;
    if y_check != *y_t {
        return Err(SalsaaError::ClaimMismatch);
    }
    let g = vec![gadget_row(&ring)]; // 1 x L
    let a = vec![params.a.clone()]; // 1 x L
    let stair = StaircaseInstance {
        a: g,
        b: a,
        w_blocks: Some(chain),
        k_blocks: params.t_steps,
        n_bar: gadget_layers(&ring),
        y0: vec![y0.neg()],
        y1: vec![y_t.clone()],
        ring: ring.clone(),
    };
    if !stair.honest()? {
        return Err(SalsaaError::ClaimMismatch);
    }
    let sp = staircase_prove(&stair, transcript)?;
    // binariness of the flat witness
    let flat = stair.flat_w()?;
    let (sinst, _pk) = SalsaInstance::create(
        &ring,
        &flat,
        &[],
        (ring.n() as f64).sqrt().ceil() as u64 + 1,
        [0x76; 32],
    )?;
    let bp = bin_prove(&sinst, transcript)?;
    Ok((sp, bp))
}

/// Verify the VDF chain proof.
pub fn vdf_verify(
    params: &VdfParams,
    y0: &RingElement,
    y_t: &RingElement,
    sp: &StaircaseProof,
    bp: &BinProof,
    transcript: &mut Transcript,
) -> Result<(), SalsaaError> {
    let ring = y0.config().clone();
    let g = vec![gadget_row(&ring)];
    let a = vec![params.a.clone()];
    let stair = StaircaseInstance {
        a: g,
        b: a,
        w_blocks: None,
        k_blocks: params.t_steps,
        n_bar: gadget_layers(&ring),
        y0: vec![y0.neg()],
        y1: vec![y_t.clone()],
        ring: ring.clone(),
    };
    staircase_verify(&stair, sp, transcript)?;
    // the bin proof's instance: flat length K·L, no rows
    let mu = (params.t_steps * gadget_layers(&ring)).trailing_zeros() as usize;
    let tr = super::ring_sc::trace_balanced(&bp.t);
    if tr != 0 {
        return Err(SalsaaError::TraceGateFailed { trace: tr });
    }
    let last = ring_sc_verify(&ring, 2, mu, &bp.t, &bp.sumcheck, transcript)?;
    if bp.v0.mul(&bp.v1)? != last {
        return Err(SalsaaError::TerminalFailed);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingConfig {
        lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, 4)
            .ok()
            .unwrap()
    }

    fn small_vec(ring: &RingConfig, m: usize, tag: &[u8], span: u32) -> Vec<RingElement> {
        (0..m)
            .map(|i| {
                let bytes = Transcript::xof(
                    b"salsaa-test",
                    &[tag, &(i as u32).to_le_bytes()].concat(),
                    4 * ring.n(),
                );
                let coeffs: Vec<u32> = bytes
                    .chunks(4)
                    .take(ring.n())
                    .map(|c| {
                        let mut a = [0u8; 4];
                        a.copy_from_slice(&c[..4]);
                        u32::from_le_bytes(a) % (2 * span + 1)
                    })
                    .collect();
                RingElement::from_coeffs(ring, coeffs)
            })
            .collect()
    }

    fn binary_vec(ring: &RingConfig, m: usize, tag: &[u8]) -> Vec<RingElement> {
        (0..m)
            .map(|i| {
                let bytes = Transcript::xof(
                    b"salsaa-bin-test",
                    &[tag, &(i as u32).to_le_bytes()].concat(),
                    (ring.n() + 7) / 8,
                );
                let coeffs: Vec<u32> = (0..ring.n())
                    .map(|k| ((bytes[k / 8] >> (k % 8)) & 1) as u32)
                    .collect();
                RingElement::from_coeffs(ring, coeffs)
            })
            .collect()
    }

    #[test]
    fn norm_plus_honest_and_tampered() {
        let ring = ring();
        let m = 4;
        let w = small_vec(&ring, m, b"nw", 8);
        let rows = vec![
            small_vec(&ring, m, b"nr0", 6),
            small_vec(&ring, m, b"nr1", 6),
        ];
        let (inst, _pk) = SalsaInstance::create(&ring, &w, &rows, 200, [7u8; 32])
            .map_err(|e| panic!("create: {:?}", e))
            .unwrap();
        let mut t = Transcript::new_default(b"lzx-salsaa");
        let proof = norm_plus_prove(&inst, &mut t)
            .map_err(|e| panic!("prove: {:?}", e))
            .unwrap();
        let mut vt = Transcript::new_default(b"lzx-salsaa");
        let reduced = norm_plus_verify(&inst, &proof, &mut vt)
            .map_err(|e| panic!("verify: {:?}", e))
            .unwrap();
        // the reduced instance has one more row (the eq-row)
        assert_eq!(reduced.rows.len(), inst.rows.len() + 1);
        // tampered t fails the trace gate
        let mut bad = proof.clone();
        bad.t = bad.t.add(&ring.one()).ok().unwrap();
        let mut vt2 = Transcript::new_default(b"lzx-salsaa");
        assert!(norm_plus_verify(&inst, &bad, &mut vt2).is_err());
        // tampered v0 fails the terminal
        let mut bad2 = proof.clone();
        bad2.v0 = bad2.v0.add(&ring.one()).ok().unwrap();
        let mut vt3 = Transcript::new_default(b"lzx-salsaa");
        assert!(norm_plus_verify(&inst, &bad2, &mut vt3).is_err());
    }

    #[test]
    fn bin_honest_and_nonbinary_rejected() {
        let ring = ring();
        let m = 4;
        let w = binary_vec(&ring, m, b"bw");
        let (inst, _pk) =
            SalsaInstance::create(&ring, &w, &[], 8, [9u8; 32]).ok().unwrap();
        let mut t = Transcript::new_default(b"lzx-salsaa-bin");
        let proof = bin_prove(&inst, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-salsaa-bin");
        assert!(bin_verify(&inst, &proof, &mut vt).is_ok());
        // a NON-binary witness fails the Tr(t) = 0 gate
        let w2 = small_vec(&ring, m, b"nbw", 8);
        let (inst2, _pk2) =
            SalsaInstance::create(&ring, &w2, &[], 64, [9u8; 32]).ok().unwrap();
        let mut t2 = Transcript::new_default(b"lzx-salsaa-bin");
        let proof2 = bin_prove(&inst2, &mut t2).ok().unwrap();
        let mut vt2 = Transcript::new_default(b"lzx-salsaa-bin");
        assert!(bin_verify(&inst2, &proof2, &mut vt2).is_err());
    }

    fn stair_instance(ring: &RingConfig, k: usize, n_bar: usize) -> StaircaseInstance {
        // build an honest staircase: pick W, derive A/B rows that satisfy
        // the system with random public Y0
        let w_blocks: Vec<Vec<RingElement>> = (0..k)
            .map(|j| binary_vec(ring, n_bar, format!("sw{}", j).as_bytes()))
            .collect();
        // Degenerate-but-structurally-valid system: zero relation rows with
        // Y0 = Y1 = 0 and arbitrary W — the derivation/sumcheck machinery
        // is what this test exercises (the honest() check with a nontrivial
        // gadget structure runs in the VDF test).
        let zero_row = vec![ring.zero(); n_bar];
        StaircaseInstance {
            a: vec![zero_row.clone()],
            b: vec![zero_row],
            w_blocks: Some(w_blocks),
            k_blocks: k,
            n_bar,
            y0: vec![ring.zero()],
            y1: vec![ring.zero()],
            ring: ring.clone(),
        }
    }

    #[test]
    fn staircase_honest_and_tampered() {
        let ring = ring();
        let inst = stair_instance(&ring, 4, 4);
        assert!(inst.honest().ok().unwrap());
        let mut t = Transcript::new_default(b"lzx-salsaa-stair");
        let proof = staircase_prove(&inst, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-salsaa-stair");
        assert!(staircase_verify(&inst, &proof, &mut vt).is_ok());
        // tampered sumcheck rounds rejected
        let mut bad = proof.clone();
        let r0 = bad.sumcheck.rounds[0][0].clone();
        bad.sumcheck.rounds[0][0] = r0.add(&ring.one()).ok().unwrap();
        let mut vt2 = Transcript::new_default(b"lzx-salsaa-stair");
        assert!(staircase_verify(&inst, &bad, &mut vt2).is_err());
        // tampered s rejected at derivation
        let mut bad2 = proof.clone();
        bad2.s = bad2.s.add(&ring.one()).ok().unwrap();
        let mut vt3 = Transcript::new_default(b"lzx-salsaa-stair");
        assert!(staircase_verify(&inst, &bad2, &mut vt3).is_err());
    }

    #[test]
    fn vdf_end_to_end_and_wrong_output() {
        let ring = ring();
        let a_row = small_vec(&ring, gadget_layers(&ring), b"va", 4);
        let params = VdfParams {
            a: a_row,
            t_steps: 4,
        };
        let y0 = small_vec(&ring, 1, b"vy0", 8)[0].clone();
        let (y_t, _chain) = vdf_eval(&params, &y0).ok().unwrap();
        let mut t = Transcript::new_default(b"lzx-salsaa-vdf");
        let (sp, bp) = vdf_prove(&params, &y0, &y_t, &mut t)
            .map_err(|e| panic!("vdf_prove: {:?}", e))
            .unwrap();
        let mut vt = Transcript::new_default(b"lzx-salsaa-vdf");
        assert!(vdf_verify(&params, &y0, &y_t, &sp, &bp, &mut vt).is_ok());
        // wrong claimed output: the prover refuses
        let y_bad = y_t.add(&ring.one()).ok().unwrap();
        let mut t2 = Transcript::new_default(b"lzx-salsaa-vdf");
        assert!(vdf_prove(&params, &y0, &y_bad, &mut t2).is_err());
        // tampered staircase proof rejected
        let mut sp_bad = sp.clone();
        let r0 = sp_bad.sumcheck.rounds[0][0].clone();
        sp_bad.sumcheck.rounds[0][0] = r0.add(&ring.one()).ok().unwrap();
        let mut vt2 = Transcript::new_default(b"lzx-salsaa-vdf");
        assert!(vdf_verify(&params, &y0, &y_t, &sp_bad, &bp, &mut vt2).is_err());
    }
}
