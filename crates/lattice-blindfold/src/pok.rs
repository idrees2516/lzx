//! The ABDLOP proofs-of-knowledge stack (§3.2, "Companion procedures from
//! the ABDLOP proofs of knowledge") — the commit-and-prove engine.
//!
//! * **Π_many^(1)** (Lemma 3.6, [LNP22, Fig 4/5]): proof of knowledge of
//!   openings (s̄₁, s̄₂, m̄) of the concatenated commitment instance
//!   satisfying the public linear rows `u = R₁s̄₁ + R_m·m̄`. Transcript
//!   shape `(w, v, c, z₁, z₂)`:
//!   - first message `w⁽ⁿ⁾ = A₁y₁ + A₂y₂` per block and
//!     `v = rows(−B·y₂)` (the mask arrangement);
//!   - challenge `c ← C̄`;
//!   - response `z_i = y_i + c·s_i` with **Rej1** (and the **Rej2**
//!     sign-condition `⟨s₂, z₂⟩ ≥ 0` per the hybrid chain of Protocols
//!     1–3);
//!   - verification `A₁z₁ + A₂z₂ = w + c·t_A` per block and
//!     `rows(c·t_B − B·z₂) = v + c·u` — the masked-message identity.
//! * **Π_many^(2)** (Lemma 3.7): the quadratic-relation proof. The
//!   quadratic garbage pair (g₀, g₁) — the value and the cross-term of
//!   `f_j(x̃₀ + c·x)` — is committed in a BDLOP garbage block BEFORE the
//!   challenge (the Protocol-4 pattern), the outer response (z₁, z₂)
//!   carries the main commitment, and an inner Π_many^(1) certifies the
//!   c-dependent linear rows `g₀_j + c·g₁_j = f_j(m̃(c))` with the public
//!   right-hand side computed from the transcript. Three rewinds sharing
//!   the first messages recover `f_j(x) = 0` — the extraction harness
//!   `extract_quadratic` implements exactly this.
//! * **Π_many^(ct)** (Protocol 4, Lemma 3.8): the vanishing-constant-
//!   coefficient wrapper — garbage `g ← {ct = 0}`, challenge `γ⃗`,
//!   in-the-clear `h = Σγⱼfⱼ + g` with `ct(h) = 0`, then the full-ring
//!   linear relation. Soundness error `1/|K|` beyond the inner PoK
//!   (a single execution suffices at |K| = q² ≈ 2¹²⁸ — Remark 3.9).
//! * **Π_anc** (Def 3.15, Lemma 3.16): the anchoring call on the
//!   concatenated instance — the wrapper with N = 1 on the pre-assembled
//!   weight-rows, with the Lemma 3.10 salt/message substitution
//!   (`u = Σwₙt_B^{(n)}_{ι(n)} − u⋆` public).
//! * **Simulators** (Protocols 1–3): S₀ → S₁ → S_ABDLOP — the honest
//!   hybrid, the uniform-commitment hybrid, and the secret-free simulator.
//! * **Extraction**: the two-transcript coordinate-wise extractor
//!   (`s̄ = (z − z′)/(c − c′)`, `c̄ = c − c′ ∈ C̄`) with the MSIS-kernel
//!   recovery harness.

use crate::abdlop::{AbdlopCommitment, AbdlopOpening, AbdlopPp, MsgLayout, SlotKind};
use crate::fp::Fq;
use crate::fq2::K;
use crate::gauss::{rej1_decide, Rng};
use crate::ring::Poly;
use crate::rk::PolyK;

/// The R_K element Y = 0 + 1·Y — the slot-1 companion weight: a message
/// weight w on an R_K message (slots a, b) expands to (w on slot a,
/// w·Y on slot b) so that the row reads w·(a + bY) exactly.
pub fn y_weight(d: usize) -> PolyK {
    PolyK {
        a: Poly::zero(d),
        b: Poly::one(d),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PokError {
    Shape(String),
    CommitmentCheck(usize),
    RowCheck(usize),
    NormCheck(usize),
    ChallengeNotInvertible,
    ExtractionFailed(String),
}

/// A linear row over the concatenated BDLOP slots with R_K weights
/// (slot granularity — the componentwise §3.3.2 form is implicit in the
/// R_K evaluation: the two R_F rows are the coordinates of the result).
#[derive(Clone, Debug)]
pub struct RelRow {
    /// (block index, slot index, R_K weight) — the wₙ·m^{(n)}_{ι(n)} terms.
    pub weights: Vec<(usize, usize, PolyK)>,
    /// `true` = full-ring row (target ∈ R_K); `false` = constant-
    /// coefficient row (target ∈ K, checked via the wrapper's γ-masking).
    pub full_ring: bool,
    /// Full-ring target (ignored for ct rows).
    pub target_rk: PolyK,
    /// ct-row target (ignored for full-ring rows).
    pub target_k: K,
}

impl RelRow {
    pub fn full_ring(weights: Vec<(usize, usize, PolyK)>, target: PolyK) -> RelRow {
        let _d = target.d();
        RelRow {
            weights,
            full_ring: true,
            target_rk: target,
            target_k: K::ZERO,
        }
    }

    /// A ct-row asserting ct(Σ w·slots) = target.
    pub fn const_coeff(weights: Vec<(usize, usize, PolyK)>, target: K, d: usize) -> RelRow {
        RelRow {
            weights,
            full_ring: false,
            target_rk: PolyK::zero(d),
            target_k: target,
        }
    }

    /// Evaluate the weighted slot sum (an R_K value).
    pub fn eval_weighted(&self, slot_arrays: &[Vec<Poly>]) -> PolyK {
        let d = self
            .weights
            .first()
            .map(|(_, _, w)| w.d())
            .or_else(|| {
                slot_arrays
                    .first()
                    .and_then(|b| b.first())
                    .map(|p| p.d())
            })
            .unwrap_or(0);
        let mut acc = PolyK::zero(d);
        for (blk, slot, w) in &self.weights {
            if let Some(arr) = slot_arrays.get(*blk) {
                if let Some(s) = arr.get(*slot) {
                    acc.add_assign(&w.mul(&PolyK::from_poly(s.clone())));
                }
            }
        }
        acc
    }

    /// Evaluate as a full-ring R_K identity value: Σw·slots − target.
    pub fn eval_full(&self, slot_arrays: &[Vec<Poly>]) -> PolyK {
        let v = self.eval_weighted(slot_arrays);
        if self.full_ring {
            v.sub(&self.target_rk)
        } else {
            v.sub(&PolyK::degree0(self.target_k, v.d()))
        }
    }

    pub fn target(&self) -> PolyK {
        if self.full_ring {
            self.target_rk.clone()
        } else {
            PolyK::degree0(self.target_k, self.target_rk.d().max(1))
        }
    }
}

/// A batch of linear rows (the R_m matrix in relational form).
#[derive(Clone, Debug)]
pub struct RelRows {
    pub rows: Vec<RelRow>,
}

impl RelRows {
    pub fn eval_all(&self, slot_arrays: &[Vec<Poly>]) -> Vec<PolyK> {
        self.rows.iter().map(|r| r.eval_weighted(slot_arrays)).collect()
    }

    /// The u-vector: rows applied to the real messages.
    pub fn eval_messages(&self, blocks: &[Block]) -> Vec<PolyK> {
        let arrays: Vec<Vec<Poly>> = blocks.iter().map(|b| b.slots()).collect();
        self.eval_all(&arrays)
    }
}

/// A prover-side block: commitment + opening + layout.
#[derive(Clone, Debug)]
pub struct Block {
    pub com: AbdlopCommitment,
    pub op: AbdlopOpening,
}

impl Block {
    pub fn slots(&self) -> Vec<Poly> {
        self.op.slots.clone()
    }

    /// The R_K messages (pairs of slots per the layout).
    pub fn rk_messages(&self) -> Vec<PolyK> {
        slots_to_rk(&self.com.layout, &self.op.slots)
    }
}

/// Pair slots into R_K messages per the layout (the φ embedding inverse).
pub fn slots_to_rk(layout: &MsgLayout, slots: &[Poly]) -> Vec<PolyK> {
    let mut msgs = vec![PolyK::zero(0); layout.n_rk];
    if layout.n_rk > 0 {
        let d = slots.first().map(|p| p.d()).unwrap_or(0);
        for m in msgs.iter_mut() {
            *m = PolyK::zero(d);
        }
    }
    for (i, kind) in layout.kinds.iter().enumerate() {
        if let Some(s) = slots.get(i) {
            if let SlotKind::RkPart { msg, part } = kind {
                if *part == 0 {
                    msgs[*msg].a = s.clone();
                } else {
                    msgs[*msg].b = s.clone();
                }
            }
        }
    }
    msgs
}

/// The m̃ arrangement: c·t_B − B·z₂ per block (the verifier's masked
/// messages — R_K-paired per layout).
pub fn masked_slots(pp: &AbdlopPp, com: &AbdlopCommitment, c: &Poly, z2: &[Poly]) -> Vec<Poly> {
    let bsz2 = pp.b_s2(z2);
    com.t_b
        .iter()
        .zip(bsz2.iter())
        .map(|(tb, bs)| c.mul(tb).sub(bs))
        .collect()
}

/// The y-mask arrangement: −B·y₂ (the prover's v-input).
pub fn mask_slots(pp: &AbdlopPp, y2: &[Poly]) -> Vec<Poly> {
    pp.b_s2(y2).iter().map(|p| p.neg()).collect()
}

/// A (block, message) coordinate into the concatenated R_K messages.
pub type MsgCoord = (usize, usize);

/// A quadratic product term: w·x_a·x_b over two message coordinates.
pub type QuadTerm = (MsgCoord, MsgCoord, PolyK);

/// A quadratic relation over the R_K messages of the concatenated blocks:
/// Σ w·x_p·x_q + Σ w·x_p + const = 0.
#[derive(Clone, Debug)]
pub struct QuadRelation {
    /// ((block, msg), (block, msg), K-weight)
    pub products: Vec<QuadTerm>,
    pub linear: Vec<((usize, usize), PolyK)>,
    pub constant: PolyK,
}

impl QuadRelation {
    pub fn eval(&self, msgs: &[Vec<PolyK>]) -> PolyK {
        let d = self.constant.d();
        let mut acc = self.constant.clone();
        for ((ba, ma), (bb, mb), w) in &self.products {
            if let (Some(xa), Some(xb)) = (msgs.get(*ba).and_then(|b| b.get(*ma)), msgs.get(*bb).and_then(|b| b.get(*mb))) {
                let prod = xa.mul(xb);
                acc.add_assign(&prod.scale_k(&w.ct()));
            }
        }
        for ((b, m), w) in &self.linear {
            if let Some(x) = msgs.get(*b).and_then(|bb| bb.get(*m)) {
                acc.add_assign(&x.scale_k(&w.ct()));
            }
        }
        let _ = d;
        acc
    }

    /// The PURE quadratic part: Σ w·x_p·x_q — the c²-coefficient of
    /// f(x̃₀ + c·x). This is the third garbage piece (g₂) of the
    /// quadratic PoK: f(x̃₀ + cx) = f(x̃₀) + c·cross + c²·quad, and the
    /// c² term does NOT vanish when the relation f(x) = 0 holds (only
    /// its sum with the linear/constant parts does).
    pub fn quad_part(&self, msgs: &[Vec<PolyK>]) -> PolyK {
        let mut acc = PolyK::zero(self.constant.d());
        for ((ba, ma), (bb, mb), w) in &self.products {
            if let (Some(xa), Some(xb)) = (msgs.get(*ba).and_then(|b| b.get(*ma)), msgs.get(*bb).and_then(|b| b.get(*mb))) {
                acc.add_assign(&xa.mul(xb).scale_k(&w.ct()));
            }
        }
        acc
    }

    /// The cross-term: ∂-direction of f at x̃₀ in direction x —
    /// cross(x̃₀, x) = Σ w·(x̃₀_p·x_q + x_p·x̃₀_q) + Σ w·x_p.
    pub fn cross(&self, msgs0: &[Vec<PolyK>], msgs1: &[Vec<PolyK>]) -> PolyK {
        let mut acc = PolyK::zero(self.constant.d());
        for ((ba, ma), (bb, mb), w) in &self.products {
            let xa0 = msgs0.get(*ba).and_then(|b| b.get(*ma));
            let xb0 = msgs0.get(*bb).and_then(|b| b.get(*mb));
            let xa1 = msgs1.get(*ba).and_then(|b| b.get(*ma));
            let xb1 = msgs1.get(*bb).and_then(|b| b.get(*mb));
            if let (Some(a0), Some(b1)) = (xa0, xb1) {
                acc.add_assign(&a0.mul(b1).scale_k(&w.ct()));
            }
            if let (Some(a1), Some(b0)) = (xa1, xb0) {
                acc.add_assign(&a1.mul(b0).scale_k(&w.ct()));
            }
        }
        for ((b, m), w) in &self.linear {
            if let Some(x) = msgs1.get(*b).and_then(|bb| bb.get(*m)) {
                acc.add_assign(&x.scale_k(&w.ct()));
            }
        }
        acc
    }
}

/// The linear PoK transcript (Π_many^(1)).
#[derive(Clone, Debug)]
pub struct LinearPokTranscript {
    pub w: Vec<Vec<Poly>>,     // per block: A₁y₁ + A₂y₂
    pub v: Vec<PolyK>,         // rows(−B·y₂)
    pub c: Poly,               // the challenge
    pub z1: Vec<Vec<Poly>>,    // per block
    pub z2: Vec<Vec<Poly>>,    // per block
    pub norm_bounds: (f64, f64), // (τ·s₁, τ·s₂) for the verifier's checks
}

/// Sample the PoK challenge c ← C̄: coefficients uniform in
/// [−β_ch, β_ch] (§3.2's S_{βch} selection set).
pub fn sample_challenge(d: usize, beta_ch: i64, rng: &mut Rng) -> Poly {
    let spread = (2 * beta_ch + 1) as u64;
    Poly(
        (0..d)
            .map(|_| {
                let m = rng.below(spread) as i64;
                Fq::from_i64(m - beta_ch)
            })
            .collect(),
    )
}

/// Run Π_many^(1) on the concatenated instance.
///
/// * `blocks` — the commitments + openings (prover side).
/// * `rows`/`u` — the public linear relation (u = rows(real messages)).
/// * `widths` — (s₁, s₂) calibrated per Eq (4.18).
/// * `tau` — the tail multiplier for the norm checks.
#[allow(clippy::too_many_arguments)]
pub fn pok_linear(
    pp: &AbdlopPp,
    blocks: &[Block],
    rows: &RelRows,
    widths: (f64, f64),
    tau: f64,
    beta_ch: i64,
    wmax: u32,
    rng: &mut Rng,
) -> Result<LinearPokTranscript, PokError> {
    // The honest prover restarts the WHOLE first message (fresh masks)
    // after a few failed challenge draws — a fixed unlucky mask set
    // (e.g. a deeply negative ⟨s₂, y₂⟩ for Rej2) would otherwise burn
    // the entire budget redrawing challenges against it. The total
    // attempt count stays ≤ Wmax.
    let mut spent = 0u32;
    while spent < wmax {
        let masks = pok_linear_sample_masks(pp, blocks.len(), widths, tau, rng);
        let sub_budget = 8u32.min(wmax - spent);
        if let Ok(tr) = pok_linear_with_masks(
            pp,
            blocks,
            rows,
            widths,
            tau,
            beta_ch,
            sub_budget,
            &masks,
            rng,
        ) {
            return Ok(tr);
        }
        spent += sub_budget;
    }
    Err(PokError::ExtractionFailed("Rej budget exhausted".into()))
}

/// Sample fresh Gaussian masks (and the first message w) per block.
pub struct LinearPokMasks {
    pub ys1: Vec<Vec<Poly>>,
    pub ys2: Vec<Vec<Poly>>,
}

pub fn pok_linear_sample_masks(
    pp: &AbdlopPp,
    n_blocks: usize,
    widths: (f64, f64),
    tau: f64,
    rng: &mut Rng,
) -> LinearPokMasks {
    let mut ys1: Vec<Vec<Poly>> = Vec::with_capacity(n_blocks);
    let mut ys2: Vec<Vec<Poly>> = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        ys1.push((0..pp.m1).map(|_| rng.gaussian_poly(pp.d, widths.0, tau)).collect());
        ys2.push((0..pp.m2).map(|_| rng.gaussian_poly(pp.d, widths.1, tau)).collect());
    }
    LinearPokMasks { ys1, ys2 }
}

/// Run Π_many^(1) with FIXED masks — the rewind-compatible entry: the
/// first message (w, v) is a deterministic function of the masks, so two
/// calls with the same masks and different challenge draws model exactly
/// the extractor's rewind (same first messages, different c).
#[allow(clippy::too_many_arguments)]
pub fn pok_linear_with_masks(
    pp: &AbdlopPp,
    blocks: &[Block],
    rows: &RelRows,
    widths: (f64, f64),
    tau: f64,
    beta_ch: i64,
    wmax: u32,
    masks: &LinearPokMasks,
    rng: &mut Rng,
) -> Result<LinearPokTranscript, PokError> {
    let (s1_w, s2_w) = widths;
    // u := the rows' PUBLIC targets — the proven relation is rows(m) = u
    // (for the honest prover these coincide with rows(real messages)).
    let _u: Vec<PolyK> = rows.rows.iter().map(|r| r.target()).collect();
    // The first message: w and v are fixed by the masks.
    let mut w = Vec::with_capacity(blocks.len());
    for b in 0..blocks.len() {
        w.push(pp.t_a(&masks.ys1[b], &masks.ys2[b]));
    }
    let mask_arr: Vec<Vec<Poly>> = masks.ys2.iter().map(|y2| mask_slots(pp, y2)).collect();
    let _v = rows.eval_all(&mask_arr);
    let mut attempt = 0u32;
    loop {
        if attempt >= wmax {
            return Err(PokError::ExtractionFailed("Rej budget exhausted".into()));
        }
        attempt += 1;
        let ys1 = &masks.ys1;
        let ys2 = &masks.ys2;
        // v := rows(−B·y₂ arrangement)
        let mask_arr: Vec<Vec<Poly>> = ys2.iter().map(|y2| mask_slots(pp, y2)).collect();
        let v = rows.eval_all(&mask_arr);
        // Challenge.
        let c = sample_challenge(pp.d, beta_ch, rng);
        // Responses z_i = y_i + c·s_i with Rej1 (v = c·s as ring
        // products) and Rej2 (⟨s₂, z₂⟩ ≥ 0).
        let mut zs1: Vec<Vec<Poly>> = Vec::with_capacity(blocks.len());
        let mut zs2: Vec<Vec<Poly>> = Vec::with_capacity(blocks.len());
        for (b, y1) in ys1.iter().enumerate() {
            let z1n: Vec<Poly> = y1
                .iter()
                .zip(blocks[b].op.s1.iter())
                .map(|(y, sv)| y.add(&c.mul(sv)))
                .collect();
            let z2n: Vec<Poly> = ys2[b]
                .iter()
                .zip(blocks[b].op.s2.iter())
                .map(|(y, sv)| y.add(&c.mul(sv)))
                .collect();
            zs1.push(z1n);
            zs2.push(z2n);
        }
        // Rej2 on the concatenated ⟨s₂, z₂⟩:
        let mut ip: i128 = 0;
        for (b, z2n) in zs2.iter().enumerate() {
            for (z, s) in z2n.iter().zip(blocks[b].op.s2.iter()) {
                for (za, sa) in z.0.iter().zip(s.0.iter()) {
                    ip += za.sym() as i128 * sa.sym() as i128;
                }
            }
        }
        if ip < 0 {
            continue; // Rej2 restart
        }
        // Rej1 with s = the joint width (use s₂ — the BDLOP side dominates;
        // the calibration prescribes one width per side; we run the joint
        // decision over both vectors with their own widths by two calls).
        // Rej1 with v = c·s computed as the true RING products (the
        // per-coefficient shift is the negacyclic convolution, not a
        // termwise product).
        let ok1 = {
            let mut zz: Vec<i64> = Vec::new();
            let mut vv: Vec<i64> = Vec::new();
            for (b, z1n) in zs1.iter().enumerate() {
                for (p, s) in z1n.iter().zip(blocks[b].op.s1.iter()) {
                    let cs = c.mul(s);
                    for (za, va) in p.0.iter().zip(cs.0.iter()) {
                        zz.push(za.sym());
                        vv.push(va.sym());
                    }
                }
            }
            rej1_decide(rng, &zz, &vv, s1_w, 3.0)
        };
        let ok2 = {
            let mut zz: Vec<i64> = Vec::new();
            let mut vv: Vec<i64> = Vec::new();
            for (b, z2n) in zs2.iter().enumerate() {
                for (p, s) in z2n.iter().zip(blocks[b].op.s2.iter()) {
                    let cs = c.mul(s);
                    for (za, va) in p.0.iter().zip(cs.0.iter()) {
                        zz.push(za.sym());
                        vv.push(va.sym());
                    }
                }
            }
            rej1_decide(rng, &zz, &vv, s2_w, 3.0)
        };
        if !(ok1 && ok2) {
            continue;
        }
        return Ok(LinearPokTranscript {
            w,
            v,
            c,
            z1: zs1,
            z2: zs2,
            norm_bounds: (tau * s1_w, tau * s2_w),
        });
    }
}

/// Verify Π_many^(1): the commitment equations, the row identity
/// `rows(c·t_B − B·z₂) = v + c·u`, and the norm bounds.
pub fn verify_linear(
    pp: &AbdlopPp,
    coms: &[AbdlopCommitment],
    rows: &RelRows,
    u: &[PolyK],
    tr: &LinearPokTranscript,
) -> Result<(), PokError> {
    if tr.w.len() != coms.len() || tr.z1.len() != coms.len() {
        return Err(PokError::Shape("transcript arity".into()));
    }
    for (n, com) in coms.iter().enumerate() {
        // A₁z₁ + A₂z₂ = w + c·t_A
        let lhs_a = pp.t_a(&tr.z1[n], &tr.z2[n]);
        let rhs_a: Vec<Poly> = tr.w[n]
            .iter()
            .zip(com.t_a.iter())
            .map(|(w, ta)| w.add(&tr.c.mul(ta)))
            .collect();
        if lhs_a != rhs_a {
            return Err(PokError::CommitmentCheck(n));
        }
        // Norms.
        for p in &tr.z1[n] {
            if p.norm_inf() as f64 > tr.norm_bounds.0 {
                return Err(PokError::NormCheck(n));
            }
        }
        for p in &tr.z2[n] {
            if p.norm_inf() as f64 > tr.norm_bounds.1 {
                return Err(PokError::NormCheck(n));
            }
        }
    }
    // Row identity: rows(m̃) = v + c·u with m̃ = c·t_B − B·z₂.
    let mtilde: Vec<Vec<Poly>> = coms
        .iter()
        .enumerate()
        .map(|(n, com)| masked_slots(pp, com, &tr.c, &tr.z2[n]))
        .collect();
    let got = rows.eval_all(&mtilde);
    // c·u with c a ring element embedded in R_K as (c, 0) — the FULL R_K
    // product, not a degree-0 scalar action.
    let c_rk = PolyK::from_poly(tr.c.clone());
    for (j, g) in got.iter().enumerate() {
        let cu = c_rk.mul(&u[j]);
        if *g != tr.v[j].add(&cu) {
            return Err(PokError::RowCheck(j));
        }
    }
    Ok(())
}

/// The quadratic PoK transcript (Π_many^(2)).
#[derive(Clone, Debug)]
pub struct QuadPokTranscript {
    /// The garbage tuple commitments ((g₀; g₁; g₂) split across blocks).
    pub garbage_com: Vec<AbdlopCommitment>,
    /// The outer main-commitment transcript pieces.
    pub w: Vec<Vec<Poly>>,
    pub c: Poly,
    pub z1: Vec<Vec<Poly>>,
    pub z2: Vec<Vec<Poly>>,
    /// The inner linear PoK certifying g₀ + c·g₁ = f(m̃(c)).
    pub inner: LinearPokTranscript,
    pub norm_bounds: (f64, f64),
}

/// Run Π_many^(2) for the quadratic relations over the R_K messages.
///
/// The (g₀, g₁) garbage pair is committed BEFORE the challenge (the
/// Protocol-4 pattern); after the response (z₁, z₂), an inner linear PoK
/// certifies the c-dependent rows `g₀_j + c·g₁_j = f_j(m̃(c))` whose
/// right-hand side the verifier computes from the transcript.
#[allow(clippy::too_many_arguments)]
pub fn pok_quadratic(
    pp: &AbdlopPp,
    blocks: &[Block],
    rels: &[QuadRelation],
    widths: (f64, f64),
    inner_widths: (f64, f64),
    tau: f64,
    beta_ch: i64,
    wmax: u32,
    rng: &mut Rng,
) -> Result<QuadPokTranscript, PokError> {
    let n_rels = rels.len();
    let d = pp.d;
    // x: the real messages.
    let x: Vec<Vec<PolyK>> = blocks.iter().map(|b| b.rk_messages()).collect();
    let mut attempt = 0u32;
    loop {
        if attempt >= wmax {
            return Err(PokError::ExtractionFailed("outer Rej budget".into()));
        }
        attempt += 1;
        // --- Fresh masks per attempt (the whole first message is
        // retransmitted on a retry: w and the garbage commitment — the
        // garbage values depend on the masks).
        let mut ys1: Vec<Vec<Poly>> = Vec::with_capacity(blocks.len());
        let mut ys2: Vec<Vec<Poly>> = Vec::with_capacity(blocks.len());
        let mut w = Vec::with_capacity(blocks.len());
        for _ in blocks {
            let y1: Vec<Poly> =
                (0..pp.m1).map(|_| rng.gaussian_poly(d, widths.0, tau)).collect();
            let y2: Vec<Poly> =
                (0..pp.m2).map(|_| rng.gaussian_poly(d, widths.1, tau)).collect();
            let wi = pp.t_a(&y1, &y2);
            ys1.push(y1);
            ys2.push(y2);
            w.push(wi);
        }
        // x̃₀: the mask arrangement as R_K messages per block.
        let x0: Vec<Vec<PolyK>> = ys2
            .iter()
            .map(|y2| {
                let ms = mask_slots(pp, y2);
                slots_to_rk(
                    &blocks
                        .first()
                        .map(|b| b.com.layout.clone())
                        .unwrap_or_else(|| crate::abdlop::MsgLayout::rk_only(0)),
                    &ms,
                )
            })
            .collect();
        // Garbage values (g₀, g₁, g₂) — the three coefficients of
        // f(x̃₀ + c·x) as a polynomial in c: value, cross, pure-quadratic.
        let mut g0: Vec<PolyK> = Vec::with_capacity(n_rels);
        let mut g1: Vec<PolyK> = Vec::with_capacity(n_rels);
        let mut g2: Vec<PolyK> = Vec::with_capacity(n_rels);
        for rel in rels {
            g0.push(rel.eval(&x0));
            g1.push(rel.cross(&x0, &x));
            g2.push(rel.quad_part(&x));
        }
        let mut gmsgs = g0.clone();
        gmsgs.extend(g1.clone());
        gmsgs.extend(g2.clone());
        let garbage_blocks = commit_rk_tuple(pp, &gmsgs, rng);
        // --- The outer challenge and response (Rej1/Rej2 on fresh c).
        let c = sample_challenge(d, beta_ch, rng);
        let mut zs1: Vec<Vec<Poly>> = Vec::with_capacity(blocks.len());
        let mut zs2: Vec<Vec<Poly>> = Vec::with_capacity(blocks.len());
        for (b, y1) in ys1.iter().enumerate() {
            let z1n: Vec<Poly> = y1
                .iter()
                .zip(blocks[b].op.s1.iter())
                .map(|(y, s)| y.add(&c.mul(s)))
                .collect();
            let z2n: Vec<Poly> = ys2[b]
                .iter()
                .zip(blocks[b].op.s2.iter())
                .map(|(y, s)| y.add(&c.mul(s)))
                .collect();
            zs1.push(z1n);
            zs2.push(z2n);
        }
        // Rej2: ⟨s₂, z₂⟩ ≥ 0.
        let mut ip: i128 = 0;
        for (b, z2n) in zs2.iter().enumerate() {
            for (z, s) in z2n.iter().zip(blocks[b].op.s2.iter()) {
                for (za, sa) in z.0.iter().zip(s.0.iter()) {
                    ip += za.sym() as i128 * sa.sym() as i128;
                }
            }
        }
        if ip < 0 {
            continue;
        }
        // Rej1 with v = c·s computed as the true RING products.
        let mut decide = |zsv: &[Vec<Poly>], ssv: &[Vec<Poly>], width: f64| -> bool {
            let mut zz = Vec::new();
            let mut vv = Vec::new();
            for (zn, sn) in zsv.iter().zip(ssv.iter()) {
                for (p, s) in zn.iter().zip(sn.iter()) {
                    let cs = c.mul(s);
                    for (za, va) in p.0.iter().zip(cs.0.iter()) {
                        zz.push(za.sym());
                        vv.push(va.sym());
                    }
                }
            }
            rej1_decide(rng, &zz, &vv, width, 3.0)
        };
        let s1s: Vec<Vec<Poly>> = blocks.iter().map(|b| b.op.s1.clone()).collect();
        let s2s: Vec<Vec<Poly>> = blocks.iter().map(|b| b.op.s2.clone()).collect();
        if !(decide(&zs1, &s1s, widths.0) && decide(&zs2, &s2s, widths.1)) {
            continue;
        }
        // --- The inner linear PoK: rows g₀_j + c·g₁_j = f_j(m̃(c)).
        let coms: Vec<AbdlopCommitment> = blocks.iter().map(|b| b.com.clone()).collect();
        let mtilde: Vec<Vec<PolyK>> = coms
            .iter()
            .enumerate()
            .map(|(n, com)| {
                let ms = masked_slots(pp, com, &c, &zs2[n]);
                slots_to_rk(&com.layout, &ms)
            })
            .collect();

        // The garbage tuple starts after the main blocks.
        let gb0 = blocks.len();
        // c² as a ring element (for the g₂ weights).
        let c2 = c.mul(&c);
        let mut rows: Vec<RelRow> = Vec::with_capacity(n_rels);
        for (j, rel) in rels.iter().enumerate() {
            let rhs = rel.eval(&mtilde); // f_j(m̃(c)) — public
            let c_k = PolyK::from_poly(c.clone());
            let c2_k = PolyK::from_poly(c2.clone());
            let one = PolyK::one(d);
            let yw = y_weight(d);
            let c_y = PolyK {
                a: Poly::zero(d),
                b: c.clone(),
            };
            let c2_y = PolyK {
                a: Poly::zero(d),
                b: c2.clone(),
            };
            let mut weights = Vec::new();
            // g₀_j at message j; c·g₁_j at message n+j; c²·g₂_j at
            // message 2n+j — each mapped through the tuple layout with
            // the (w, w·Y) slot-weight expansion.
            let (b0, s0) = tuple_slot(pp, j);
            weights.push((gb0 + b0, s0, one.clone()));
            weights.push((gb0 + b0, s0 + 1, yw.clone()));
            let (b1, s1) = tuple_slot(pp, n_rels + j);
            weights.push((gb0 + b1, s1, c_k.clone()));
            weights.push((gb0 + b1, s1 + 1, c_y.clone()));
            let (b2, s2) = tuple_slot(pp, 2 * n_rels + j);
            weights.push((gb0 + b2, s2, c2_k.clone()));
            weights.push((gb0 + b2, s2 + 1, c2_y.clone()));
            rows.push(RelRow::full_ring(weights, rhs));
        }
        let rel_rows = RelRows { rows };
        let mut all_blocks: Vec<Block> = blocks.to_vec();
        for (com, op) in &garbage_blocks {
            all_blocks.push(Block {
                com: com.clone(),
                op: op.clone(),
            });
        }
        let inner = pok_linear(pp, &all_blocks, &rel_rows, inner_widths, tau, beta_ch, wmax, rng)?;
        return Ok(QuadPokTranscript {
            garbage_com: garbage_blocks.iter().map(|(c, _)| c.clone()).collect(),
            w,
            c,
            z1: zs1,
            z2: zs2,
            inner,
            norm_bounds: (tau * widths.0, tau * widths.1),
        });
    }
}

/// Verify Π_many^(2).
pub fn verify_quadratic(
    pp: &AbdlopPp,
    coms: &[AbdlopCommitment],
    rels: &[QuadRelation],
    tr: &QuadPokTranscript,
    inner_widths: (f64, f64),
) -> Result<(), PokError> {
    // Outer commitment checks.
    for (n, com) in coms.iter().enumerate() {
        let lhs_a = pp.t_a(&tr.z1[n], &tr.z2[n]);
        let rhs_a: Vec<Poly> = tr.w[n]
            .iter()
            .zip(com.t_a.iter())
            .map(|(w, ta)| w.add(&tr.c.mul(ta)))
            .collect();
        if lhs_a != rhs_a {
            return Err(PokError::CommitmentCheck(n));
        }
        for p in &tr.z1[n] {
            if p.norm_inf() as f64 > tr.norm_bounds.0 {
                return Err(PokError::NormCheck(n));
            }
        }
        for p in &tr.z2[n] {
            if p.norm_inf() as f64 > tr.norm_bounds.1 {
                return Err(PokError::NormCheck(n));
            }
        }
    }
    // The inner linear PoK: rebuild its rows and targets from public data.
    let n_rels = rels.len();
    let d = pp.d;
    let mtilde: Vec<Vec<PolyK>> = coms
        .iter()
        .enumerate()
        .map(|(n, com)| {
            let ms = masked_slots(pp, com, &tr.c, &tr.z2[n]);
            slots_to_rk(&com.layout, &ms)
        })
        .collect();
    let gb0 = coms.len(); // the garbage tuple's first block
    let c2 = tr.c.mul(&tr.c);
    let mut rows: Vec<RelRow> = Vec::with_capacity(n_rels);
    let mut u: Vec<PolyK> = Vec::with_capacity(n_rels);
    for (j, rel) in rels.iter().enumerate() {
        let rhs = rel.eval(&mtilde);
        let c_k = PolyK::from_poly(tr.c.clone());
        let c2_k = PolyK::from_poly(c2.clone());
        let one = PolyK::one(d);
        let yw = y_weight(d);
        let c_y = PolyK {
            a: Poly::zero(d),
            b: tr.c.clone(),
        };
        let c2_y = PolyK {
            a: Poly::zero(d),
            b: c2.clone(),
        };
        let mut weights = Vec::new();
        let (b0, s0) = tuple_slot(pp, j);
        weights.push((gb0 + b0, s0, one.clone()));
        weights.push((gb0 + b0, s0 + 1, yw.clone()));
        let (b1, s1) = tuple_slot(pp, n_rels + j);
        weights.push((gb0 + b1, s1, c_k.clone()));
        weights.push((gb0 + b1, s1 + 1, c_y));
        let (b2, s2) = tuple_slot(pp, 2 * n_rels + j);
        weights.push((gb0 + b2, s2, c2_k.clone()));
        weights.push((gb0 + b2, s2 + 1, c2_y));
        rows.push(RelRow::full_ring(weights, rhs.clone()));
        u.push(rhs);
    }
    let rel_rows = RelRows { rows };
    let mut all_coms: Vec<AbdlopCommitment> = coms.to_vec();
    all_coms.extend(tr.garbage_com.iter().cloned());
    // The inner transcript's blocks: [main blocks; garbage block].
    verify_linear(pp, &all_coms, &rel_rows, &u, &tr.inner)?;
    let _ = inner_widths;
    Ok(())
}

/// The Protocol-4 wrapper transcript (Π_many^(ct) / Π_anc).
#[derive(Clone, Debug)]
pub struct CtwTranscript {
    /// The garbage commitment (g appended to the committed messages).
    pub garbage_com: AbdlopCommitment,
    /// The wrapper challenges γ⃗ ∈ K^N.
    pub gammas: Vec<K>,
    /// The in-the-clear masked evaluation h = Σγⱼfⱼ + g.
    pub h: PolyK,
    /// The inner full-ring PoK (linear if the fⱼ are linear, quadratic
    /// otherwise).
    pub inner_linear: Option<LinearPokTranscript>,
    pub inner_quad: Option<QuadPokTranscript>,
}

/// Run the Π_many^(ct) wrapper (Protocol 4) on a family of ct-relations.
///
/// `f_eval` evaluates Σγⱼfⱼ(messages) + g on demand: the relations are
/// supplied as either linear rows (`linear`) or quadratic relations
/// (`quad`); exactly one must be non-empty.
#[allow(clippy::too_many_arguments)]
pub fn pok_ct_wrapper(
    pp: &AbdlopPp,
    blocks: &[Block],
    linear: Option<(&RelRows, Vec<K>)>,
    quad: Option<(&[QuadRelation], Vec<K>)>,
    widths: (f64, f64),
    tau: f64,
    beta_ch: i64,
    wmax: u32,
    rng: &mut Rng,
) -> Result<CtwTranscript, PokError> {
    let d = pp.d;
    // Step 1: garbage g ← {ct = 0}, committed in a fresh block.
    let g = PolyK::garbage_ct_zero(d, b"garbage", &mut gctr());
    let (garbage_com, garbage_op) = AbdlopOpening::commit_rk(pp, std::slice::from_ref(&g), &[], rng);
    // Step 2: challenges γ⃗ — Protocol 4's verifier draw. In this
    // architecture the caller supplies them (drawn from the shared
    // public-coin rng); they are used CONSISTENTLY for h, the inner rows
    // and the transcript.
    let gammas: Vec<K> = match (&linear, &quad) {
        (Some((_, g)), _) => g.clone(),
        (_, Some((_, g))) => g.clone(),
        _ => Vec::new(),
    };
    let _ = rng; // the caller's rng drives the inner PoK below
    // Step 3: h := Σγⱼfⱼ(messages) + g — in the clear. The fⱼ are the
    // RELATION VALUES: the weighted slot sum minus the row's target
    // (full-ring target, or the degree-0 embedding for ct-rows) — they
    // vanish (resp. their ct vanishes) for the honest prover, so that
    // ct(h) = 0 (Protocol 4, Step 4's check).
    let real_arrays: Vec<Vec<Poly>> = blocks.iter().map(|b| b.slots()).collect();
    let real_rk: Vec<Vec<PolyK>> = blocks.iter().map(|b| b.rk_messages()).collect();
    let mut h = g.clone();
    if let Some((rows, gammas_ref)) = &linear {
        for (gj, row) in gammas_ref.iter().zip(rows.rows.iter()) {
            let val = row.eval_weighted(&real_arrays).sub(&row.target());
            h.add_assign(&val.scale_k(gj));
        }
    }
    if let Some((rels, gammas_ref)) = &quad {
        for (gj, rel) in gammas_ref.iter().zip(rels.iter()) {
            let v = rel.eval(&real_rk);
            h.add_assign(&v.scale_k(gj));
        }
    }
    // ct(h) must be 0 for the honest prover (each ct(fⱼ) = 0 and ct(g) = 0).
    if !h.ct().is_zero() {
        return Err(PokError::RowCheck(0));
    }
    // Step 5: the full-ring relation Σγⱼfⱼ + g = h on [blocks; garbage].
    let mut all_blocks: Vec<Block> = blocks.to_vec();
    all_blocks.push(Block {
        com: garbage_com.clone(),
        op: garbage_op,
    });
    let (inner_linear, inner_quad) = if let Some((rows, _gammas_ref)) = &linear {
        // Σγⱼ·rows + g = h — a linear row per original row, plus the
        // garbage slot with weight 1 and target h.
        // The full-ring relation Σγⱼfⱼ + g = h with fⱼ = row_valueⱼ
        // = (Σw·m − targetⱼ) expands to
        //   Σⱼ γⱼ·Σw⁽ʲ⁾·m + g = h + Σⱼ γⱼ·targetⱼ  —
        // ONE aggregated row: the γⱼ-scaled weights summed slot-wise
        // (several rows may act on the same slot), plus the garbage
        // slots with weight (1, Y), against the combined target.
        let one = PolyK::one(d);
        let yw = y_weight(d);
        let gb = all_blocks.len() - 1;
        let mut combined_target = h.clone();
        let mut agg: std::collections::BTreeMap<(usize, usize), PolyK> =
            std::collections::BTreeMap::new();
        for (gj, row) in gammas.iter().zip(rows.rows.iter()) {
            for (blk, slot, w) in &row.weights {
                let scaled = w.scale_k(gj);
                agg.entry((*blk, *slot))
                    .and_modify(|e| e.add_assign(&scaled))
                    .or_insert(scaled);
            }
            combined_target = combined_target.add(&row.target().scale_k(gj));
        }
        agg.entry((gb, 0))
            .and_modify(|e| e.add_assign(&one))
            .or_insert(one);
        agg.entry((gb, 1))
            .and_modify(|e| e.add_assign(&yw))
            .or_insert(yw);
        let weights: Vec<(usize, usize, PolyK)> =
            agg.into_iter().map(|((b, s), w)| (b, s, w)).collect();
        let nr = RelRows {
            rows: vec![RelRow::full_ring(weights, combined_target)],
        };
        let tr = pok_linear(pp, &all_blocks, &nr, widths, tau, beta_ch, wmax, rng)?;
        (Some(tr), None)
    } else if let Some((rels, gammas_ref)) = &quad {
        // The single quadratic relation Σγⱼfⱼ + g − h = 0.
        let mut combined: Vec<QuadRelation> = Vec::with_capacity(1);
        let mut products = Vec::new();
        let mut lin = Vec::new();
        let gb = all_blocks.len() - 1;
        for (gj, rel) in gammas_ref.iter().zip(rels.iter()) {
            let w = PolyK::degree0(*gj, d);
            for ((ba, ma), (bb, mb), rw) in &rel.products {
                products.push(((*ba, *ma), (*bb, *mb), rw.mul(&w)));
            }
            for ((b, m), rw) in &rel.linear {
                lin.push(((*b, *m), rw.mul(&w)));
            }
        }
        // + g − h: linear terms on the garbage block's R_K message with
        // weight 1 and constant −h.
        lin.push(((gb, 0), PolyK::one(d)));
        combined.push(QuadRelation {
            products,
            linear: lin,
            constant: h.clone().neg(),
        });
        let tr = pok_quadratic(
            pp,
            &all_blocks,
            &combined,
            widths,
            widths,
            tau,
            beta_ch,
            wmax,
            rng,
        )?;
        (None, Some(tr))
    } else {
        return Err(PokError::Shape("no relations".into()));
    };
    Ok(CtwTranscript {
        garbage_com,
        gammas,
        h,
        inner_linear,
        inner_quad,
    })
}

thread_local! {
    static GCTR: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn gctr() -> u64 {
    GCTR.with(|c| {
        let v = c.get() + 1;
        c.set(v);
        v
    })
}

/// Verify the Protocol-4 wrapper transcript.
#[allow(clippy::too_many_arguments)]
pub fn verify_ct_wrapper(
    pp: &AbdlopPp,
    coms: &[AbdlopCommitment],
    linear: Option<(&RelRows, Vec<K>)>,
    quad: Option<(&[QuadRelation], Vec<K>)>,
    tr: &CtwTranscript,
) -> Result<(), PokError> {
    // Step 4: ct(h) = 0.
    if !tr.h.ct().is_zero() {
        return Err(PokError::RowCheck(0));
    }
    let d = pp.d;
    let mut all_coms: Vec<AbdlopCommitment> = coms.to_vec();
    all_coms.push(tr.garbage_com.clone());
    if let Some((rows, _)) = linear {
        let inner = tr
            .inner_linear
            .as_ref()
            .ok_or_else(|| PokError::Shape("missing inner linear".into()))?;
        // Rebuild the SAME aggregated row: γ-scaled weights summed
        // slot-wise + the garbage slots, against the combined target.
        let one = PolyK::one(d);
        let yw = y_weight(d);
        let gb = all_coms.len() - 1;
        let mut combined_target = tr.h.clone();
        let mut agg: std::collections::BTreeMap<(usize, usize), PolyK> =
            std::collections::BTreeMap::new();
        for (gj, row) in tr.gammas.iter().zip(rows.rows.iter()) {
            for (blk, slot, w) in &row.weights {
                let scaled = w.scale_k(gj);
                agg.entry((*blk, *slot))
                    .and_modify(|e| e.add_assign(&scaled))
                    .or_insert(scaled);
            }
            combined_target = combined_target.add(&row.target().scale_k(gj));
        }
        agg.entry((gb, 0)).and_modify(|e| e.add_assign(&one)).or_insert(one);
        agg.entry((gb, 1)).and_modify(|e| e.add_assign(&yw)).or_insert(yw);
        let weights: Vec<(usize, usize, PolyK)> =
            agg.into_iter().map(|((b, s), w)| (b, s, w)).collect();
        let nr = RelRows {
            rows: vec![RelRow::full_ring(weights, combined_target)],
        };
        let u: Vec<PolyK> = nr.rows.iter().map(|r| r.target()).collect();
        verify_linear(pp, &all_coms, &nr, &u, inner)?;
    }
    if let Some((rels, _)) = quad {
        let inner = tr
            .inner_quad
            .as_ref()
            .ok_or_else(|| PokError::Shape("missing inner quad".into()))?;
        let gb = all_coms.len() - 1;
        let mut products = Vec::new();
        let mut lin = Vec::new();
        for (gj, rel) in tr.gammas.iter().zip(rels.iter()) {
            let w = PolyK::degree0(*gj, d);
            for ((ba, ma), (bb, mb), rw) in &rel.products {
                products.push(((*ba, *ma), (*bb, *mb), rw.mul(&w)));
            }
            for ((b, m), rw) in &rel.linear {
                lin.push(((*b, *m), rw.mul(&w)));
            }
        }
        lin.push(((gb, 0), PolyK::one(d)));
        let combined = vec![QuadRelation {
            products,
            linear: lin,
            constant: tr.h.clone().neg(),
        }];
        verify_quadratic(pp, &all_coms, &combined, inner, widths_default())?;
    }
    Ok(())
}

fn widths_default() -> (f64, f64) {
    (30.0, 30.0)
}

/// Π_anc (Def 3.15): the anchoring call — the wrapper with N = 1 on the
/// pre-assembled weight-rows with target u (the Lemma 3.10 substitution).
///
/// The rows assert ct(Σₙ wₙ·m^{(n)}_{ι(n)} − u) = 0.
#[allow(clippy::too_many_arguments)]
pub fn pok_anchor(
    pp: &AbdlopPp,
    blocks: &[Block],
    weights: &[(usize, usize, PolyK)],
    target: K,
    widths: (f64, f64),
    tau: f64,
    beta_ch: i64,
    wmax: u32,
    rng: &mut Rng,
) -> Result<CtwTranscript, PokError> {
    let row = RelRow::const_coeff(weights.to_vec(), target, pp.d);
    let rows = RelRows { rows: vec![row] };
    let gammas = vec![K::from_fp(Fq::ONE)];
    pok_ct_wrapper(pp, blocks, Some((&rows, gammas)), None, widths, tau, beta_ch, wmax, rng)
}

/// Verify the anchoring call.
#[allow(clippy::too_many_arguments)]
pub fn verify_anchor(
    pp: &AbdlopPp,
    coms: &[AbdlopCommitment],
    weights: &[(usize, usize, PolyK)],
    target: K,
    tr: &CtwTranscript,
) -> Result<(), PokError> {
    let row = RelRow::const_coeff(weights.to_vec(), target, pp.d);
    let rows = RelRows { rows: vec![row] };
    let gammas = vec![K::from_fp(Fq::ONE)];
    verify_ct_wrapper(pp, coms, Some((&rows, gammas)), None, tr)
}


/// Commit a list of R_K messages as a TUPLE of ABDLOP blocks, each
/// carrying up to ⌊ℓ/2⌋ messages (Remark 4.21's splitting: a message too
/// long for one commitment rides in several at the ambient parameters).
pub fn commit_rk_tuple(
    pp: &AbdlopPp,
    msgs: &[PolyK],
    rng: &mut Rng,
) -> Vec<(AbdlopCommitment, AbdlopOpening)> {
    let per_block = (pp.ell / 2).max(1);
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < msgs.len() {
        let end = (idx + per_block).min(msgs.len());
        let chunk = &msgs[idx..end];
        let (com, op) = AbdlopOpening::commit_rk(pp, chunk, &[], rng);
        out.push((com, op));
        idx = end;
    }
    if out.is_empty() {
        let (com, op) = AbdlopOpening::commit_rk(pp, &[], &[], rng);
        out.push((com, op));
    }
    out
}

/// Map a message index within a tuple to (block_index, a_slot_index).
pub fn tuple_slot(pp: &AbdlopPp, msg_index: usize) -> (usize, usize) {
    let per_block = (pp.ell / 2).max(1);
    (msg_index / per_block, 2 * (msg_index % per_block))
}

/// The S_ABDLOP simulator (Protocol 3): produces an accepting transcript
/// with NO access to the secrets — (t_A, t_B) sampled uniformly, z's
/// Gaussian, w/v derived to satisfy the checks.
#[allow(clippy::too_many_arguments)]
pub fn simulate_linear_pok(
    pp: &AbdlopPp,
    n_blocks: usize,
    rows: &RelRows,
    u: &[PolyK],
    widths: (f64, f64),
    tau: f64,
    beta_ch: i64,
    rng: &mut Rng,
) -> (Vec<AbdlopCommitment>, LinearPokTranscript) {
    let d = pp.d;
    // Uniform commitments.
    let mut coms = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        let t_a: Vec<Poly> = (0..pp.kappa).map(|_| Poly::uniform(d, b"sim", &mut sim_ctr())).collect();
        let t_b: Vec<Poly> = (0..pp.ell).map(|_| Poly::uniform(d, b"sim", &mut sim_ctr())).collect();
        coms.push(AbdlopCommitment {
            t_a,
            t_b,
            layout: crate::abdlop::MsgLayout::rk_only(pp.ell / 2),
        });
    }
    // Challenge, Gaussian z's.
    let c = sample_challenge(d, beta_ch, rng);
    let mut zs1: Vec<Vec<Poly>> = Vec::with_capacity(n_blocks);
    let mut zs2: Vec<Vec<Poly>> = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        zs1.push((0..pp.m1).map(|_| rng.gaussian_poly(d, widths.0, tau)).collect());
        zs2.push((0..pp.m2).map(|_| rng.gaussian_poly(d, widths.1, tau)).collect());
    }
    // w := A₁z₁ + A₂z₂ − c·t_A (Protocol 3, line 6).
    let mut w = Vec::with_capacity(n_blocks);
    for (n, com) in coms.iter().enumerate() {
        let base = pp.t_a(&zs1[n], &zs2[n]);
        w.push(
            base.iter()
                .zip(com.t_a.iter())
                .map(|(b, ta)| b.sub(&c.mul(ta)))
                .collect(),
        );
    }
    // v := rows(c·t_B − B·z₂) − c·u (Protocol 3, line 7).
    let mtilde: Vec<Vec<Poly>> = coms
        .iter()
        .enumerate()
        .map(|(n, com)| masked_slots(pp, com, &c, &zs2[n]))
        .collect();
    let got = rows.eval_all(&mtilde);
    let c_rk = PolyK::from_poly(c.clone());
    let v: Vec<PolyK> = got
        .iter()
        .zip(u.iter())
        .map(|(g, uu)| g.sub(&c_rk.mul(uu)))
        .collect();
    let tr = LinearPokTranscript {
        w,
        v,
        c,
        z1: zs1,
        z2: zs2,
        norm_bounds: (tau * widths.0, tau * widths.1),
    };
    (coms, tr)
}

thread_local! {
    static SIMCTR: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn sim_ctr() -> u64 {
    SIMCTR.with(|c| {
        let v = c.get() + 1;
        c.set(v);
        v
    })
}

/// The two-transcript extractor (Lemma 3.6's uniqueness): from two
/// accepting transcripts sharing the first messages (a rewind),
/// s̄ = (z − z′)/(c − c′) opens the commitment.
pub fn extract_two_transcripts(
    pp: &AbdlopPp,
    com: &AbdlopCommitment,
    t1: &LinearPokTranscript,
    t2: &LinearPokTranscript,
    block_index: usize,
) -> Result<(Vec<Poly>, Vec<Poly>), PokError> {
    let d = pp.d;
    let cdiff = t1.c.sub(&t2.c);
    let inv = match poly_inverse(&cdiff, d) {
        Some(p) => p,
        None => return Err(PokError::ChallengeNotInvertible),
    };
    // s̄ = (z − z′)/(c − c′) = (z − z′)·(c − c′)^{−1} — the ring inverse
    // acting on the whole element.
    let s1: Vec<Poly> = t1.z1[block_index]
        .iter()
        .zip(t2.z1[block_index].iter())
        .map(|(a, b)| inv.mul(&a.sub(b)))
        .collect();
    let s2: Vec<Poly> = t1.z2[block_index]
        .iter()
        .zip(t2.z2[block_index].iter())
        .map(|(a, b)| inv.mul(&a.sub(b)))
        .collect();
    // Check the extraction opens the commitment.
    let t_a = pp.t_a(&s1, &s2);
    if t_a != com.t_a {
        return Err(PokError::ExtractionFailed(
            "extracted opening does not match t_A".into(),
        ));
    }
    Ok((s1, s2))
}

/// Polynomial inverse in R_F = F_q[X]/(X^d+1) via extended GCD.
pub fn poly_inverse(a: &Poly, d: usize) -> Option<Poly> {
    if a.is_zero() {
        return None;
    }
    // Extended Euclid on (a, X^d+1) over F_q[X].
    let mut r0 = a.0.clone();
    let mut r1 = {
        let mut m = vec![Fq::ZERO; d + 1];
        m[d] = Fq::ONE;
        m[0] = Fq::ONE;
        m
    };
    let mut s0 = {
        let mut v = vec![Fq::ZERO; d + 1];
        v[0] = Fq::ONE;
        v
    }; // 1
    let mut s1 = vec![Fq::ZERO; d + 1]; // 0
    while r1.iter().any(|c| !c.is_zero()) {
        let (q, rem) = poly_divmod(&r0, &r1);
        // s_next = s0 − q·s1
        let s_next = poly_sub_scaled(&s0, &q, &s1);
        r0 = r1;
        r1 = rem;
        s0 = s1;
        s1 = s_next;
    }
    // r0 = gcd; must be a nonzero constant.
    let deg = r0.iter().rposition(|c| !c.is_zero())?;
    if deg != 0 {
        return None; // not invertible
    }
    let c_inv = r0[0].inverse()?;
    // Reduce the Bézout coefficient modulo X^d + 1 (negacyclic fold) and
    // scale by gcd⁻¹.
    let mut red = vec![Fq::ZERO; d];
    for (i, c) in s0.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        if i < d {
            red[i] = red[i].add(c);
        } else {
            // X^i = X^{i−d}·X^d ≡ −X^{i−d}
            let j = i - d;
            let v = c.neg();
            red[j] = red[j].add(&v);
        }
    }
    let out: Vec<Fq> = red.iter().map(|c| c_inv.mul(c)).collect();
    Some(Poly(out))
}

fn poly_divmod(a: &[Fq], b: &[Fq]) -> (Vec<Fq>, Vec<Fq>) {
    let mut r = a.to_vec();
    let bdeg = b.iter().rposition(|c| !c.is_zero()).unwrap_or(0);
    let binv = match b[bdeg].inverse() {
        Some(v) => v,
        None => return (vec![Fq::ZERO; b.len()], r),
    };
    let mut q = vec![Fq::ZERO; b.len().max(a.len())];
    if bdeg == 0 {
        // Dividing by a nonzero constant: quotient = a·b⁻¹, remainder = 0
        // (a nonzero remainder here would loop the Euclid forever).
        let scale = binv;
        let q2: Vec<Fq> = r.iter().map(|x| x.mul(&scale)).collect();
        let r2 = vec![Fq::ZERO; r.len()];
        return (q2, r2);
    }
    while let Some(rdeg) = r.iter().rposition(|c| !c.is_zero()) {
        if rdeg < bdeg || rdeg == 0 {
            if rdeg == 0 && bdeg > 0 {
                break;
            }
            break;
        }
        let shift = rdeg - bdeg;
        let coef = r[rdeg].mul(&binv);
        q[shift] = q[shift].add(&coef);
        for i in 0..=bdeg {
            if !b[i].is_zero() {
                r[i + shift] = r[i + shift].sub(&coef.mul(&b[i]));
            }
        }
    }
    // trim r to a's length semantics — caller handles
    (q, r)
}

/// Full polynomial product (plain convolution, no modulus reduction).
fn poly_mul_full(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let n = a.len() + b.len() - 1;
    let mut out = vec![Fq::ZERO; n];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            if bj.is_zero() {
                continue;
            }
            out[i + j] = out[i + j].add(&ai.mul(bj));
        }
    }
    out
}

/// s_next = s0 − q·s1 as POLYNOMIALS (the extended-Euclid recurrence
/// needs true products; a termwise product silently corrupts the Bézout
/// coefficients).
fn poly_sub_scaled(a: &[Fq], q: &[Fq], b: &[Fq]) -> Vec<Fq> {
    let prod = poly_mul_full(q, b);
    let n = a.len().max(prod.len());
    let mut out = vec![Fq::ZERO; n];
    for i in 0..n {
        let ai = a.get(i).copied().unwrap_or(Fq::ZERO);
        let pi = prod.get(i).copied().unwrap_or(Fq::ZERO);
        out[i] = ai.sub(&pi);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
// temporary debug test
#[test]
fn debug_wrapper_ct() {
    #[allow(unused_imports)]
    use crate::abdlop::*;
    use crate::fp::Fq;
    use crate::fq2::K;
    use crate::gauss::Rng;
    use crate::pok::*;
    use crate::rk::PolyK;
    let mut rng = Rng::new(b"dbg");
    let pp = AbdlopPp::setup(3, 6, 4, 6, 4, &mut rng);
    let mut ctr = 0u64;
    let msgs: Vec<PolyK> = (0..2).map(|_| PolyK::uniform(4, b"w", &mut ctr)).collect();
    let target = msgs[0].ct().add(&msgs[1].ct());
    let (com, op) = AbdlopOpening::commit_rk(&pp, &msgs, &[], &mut rng);
    let block = Block { com: com.clone(), op: op.clone() };
    let yw = y_weight(4);
    let row = RelRow::const_coeff(
        vec![
            (0, 0, PolyK::one(4)),
            (0, 1, yw.clone()),
            (0, 2, PolyK::one(4)),
            (0, 3, yw.clone()),
        ],
        target,
        4,
    );
    // Evaluate the row value manually:
    let arrays: Vec<Vec<Poly>> = vec![vec![
        msgs[0].a.clone(), msgs[0].b.clone(),
        msgs[1].a.clone(), msgs[1].b.clone(),
    ]];
    let val = row.eval_weighted(&arrays);
    println!("row value ct = {:?}", val.ct());
    println!("target       = {:?}", row.target().ct());
    println!("row value    = {:?}", val);
    println!("m0+m1        = {:?}", msgs[0].add(&msgs[1]));
    assert_eq!(val, msgs[0].add(&msgs[1]));
    assert!(val.ct().sub(&row.target().ct()).is_zero());
    // Run the actual wrapper end-to-end:
    let rows = RelRows { rows: vec![row] };
    let _gammas = [K::from_fp(Fq::ONE)];
    let gammas2 = vec![K::from_fp(Fq::ONE)];
    let r = pok_ct_wrapper(&pp, std::slice::from_ref(&block), Some((&rows, gammas2.clone())), None, (40.0, 40.0), 6.0, 1, 128, &mut rng);
    match r {
        Ok(tr) => {
            println!("wrapper OK: h.ct = {:?}", tr.h.ct());
            // Manual replication of the inner check:
            let inner = tr.inner_linear.as_ref().unwrap();
            let mut all_coms = vec![com.clone()];
            all_coms.push(tr.garbage_com.clone());
            let yw2 = y_weight(4);
            let mut weights: Vec<(usize, usize, PolyK)> = vec![];
            for (blk, slot, w) in &rows.rows[0].weights {
                weights.push((*blk, *slot, w.scale_k(&gammas2[0])));
            }
            weights.push((1, 0, PolyK::one(4)));
            weights.push((1, 1, yw2));
            let combined = tr.h.clone().add(&rows.rows[0].target().scale_k(&gammas2[0]));
            let row = RelRow::full_ring(weights, combined.clone());
            let rr = RelRows { rows: vec![row] };
            let mtilde: Vec<Vec<Poly>> = all_coms
                .iter()
                .enumerate()
                .map(|(n, com)| masked_slots(&pp, com, &inner.c, &inner.z2[n]))
                .collect();
            let got = rr.eval_all(&mtilde)[0].clone();
            let cu = PolyK::from_poly(inner.c.clone()).mul(&combined);
            let expect = inner.v[0].add(&cu);
            println!("got    = {:?}", got);
            println!("expect = {:?}", expect);
            println!("v      = {:?}", inner.v[0]);
            println!("cu     = {:?}", cu);
            println!("diff   = {:?}", got.sub(&expect));
            let vr = verify_ct_wrapper(&pp, std::slice::from_ref(&com), Some((&rows, gammas2)), None, &tr);
            println!("verify result: {:?}", vr);
        }
        Err(e) => println!("wrapper ERR: {:?}", e),
    }
}

    use crate::abdlop::AbdlopPp;

    fn test_pp() -> (AbdlopPp, Rng) {
        let mut rng = Rng::new(b"pok-test");
        (AbdlopPp::setup(3, 6, 4, 6, 4, &mut rng), rng)
    }

    #[test]
    fn linear_pok_roundtrip_and_tamper() {
        let (pp, mut rng) = test_pp();
        let mut ctr = 0u64;
        let msgs: Vec<PolyK> = (0..2).map(|_| PolyK::uniform(4, b"m", &mut ctr)).collect();
        let (com, op) = AbdlopOpening::commit_rk(&pp, &msgs, &[], &mut rng);
        let block = Block {
            com: com.clone(),
            op,
        };
        // Row: the a-part of message 0 plus twice the b-part of message 1
        // equals a public target.
        let target = msgs[0].a.add(&msgs[1].b);
        let row = RelRow::full_ring(
            vec![
                (0, 0, PolyK::one(4)),
                (0, 3, PolyK::one(4)),
            ],
            PolyK::from_poly(target.clone()),
        );
        let rows = RelRows { rows: vec![row] };
        let widths = (300.0, 300.0);
        let tr = pok_linear(&pp, std::slice::from_ref(&block), &rows, widths, 6.0, 1, 64, &mut rng).unwrap();
        let u = rows.eval_messages(std::slice::from_ref(&block));
        verify_linear(&pp, std::slice::from_ref(&com), &rows, &u, &tr).unwrap();
        // Tamper the response: verification must fail.
        let mut bad = tr.clone();
        bad.z1[0][0] = bad.z1[0][0].add(&Poly::one(4));
        assert!(verify_linear(&pp, std::slice::from_ref(&com), &rows, &u, &bad).is_err());
        // Tamper v.
        let mut bad2 = tr.clone();
        bad2.v[0] = bad2.v[0].add(&PolyK::one(4));
        assert!(verify_linear(&pp, std::slice::from_ref(&com), &rows, &u, &bad2).is_err());
    }

    #[test]
    fn linear_pok_extraction_two_transcripts() {
        let (pp, mut rng) = test_pp();
        let mut ctr = 0u64;
        let msgs: Vec<PolyK> = (0..2).map(|_| PolyK::uniform(4, b"m", &mut ctr)).collect();
        let (com, op) = AbdlopOpening::commit_rk(&pp, &msgs, &[], &mut rng);
        let block = Block { com: com.clone(), op };
        let rows = RelRows { rows: vec![] };
        let widths = (300.0, 300.0);
        // Two transcripts from a REWIND: the same masks (same first
        // message (w, v)) with different challenge draws — exactly the
        // extractor's model (Lemma 3.6's uniqueness clause). When a mask
        // draw's Rej2 sign is unfavorable the run yields no transcripts,
        // and the extractor moves on to another first message — model
        // that by resampling the masks until two completions exist.
        let mut trs = Vec::new();
        let mut mask_rounds = 0;
        while trs.len() < 2 && mask_rounds < 32 {
            mask_rounds += 1;
            let masks = pok_linear_sample_masks(&pp, 1, widths, 6.0, &mut rng);
            let mut round_trs = Vec::new();
            for _ in 0..8 {
                if let Ok(tr) = pok_linear_with_masks(
                    &pp,
                    std::slice::from_ref(&block),
                    &rows,
                    widths,
                    6.0,
                    1,
                    64,
                    &masks,
                    &mut rng,
                ) {
                    round_trs.push(tr);
                }
            }
            if round_trs.len() >= 2 {
                trs = round_trs;
            }
        }
        let mut extracted = false;
        for i in 0..trs.len() {
            for j in (i + 1)..trs.len() {
                if trs[i].c != trs[j].c {
                    if let Ok((s1, s2)) =
                        extract_two_transcripts(&pp, &com, &trs[i], &trs[j], 0)
                    {
                        // The extraction must reproduce the salts exactly.
                        assert_eq!(s1, block.op.s1);
                        assert_eq!(s2, block.op.s2);
                        extracted = true;
                    }
                }
            }
        }
        assert!(extracted, "two-transcript extraction must succeed");
    }

    #[test]
    fn quadratic_pok_roundtrip() {
        let (pp, mut rng) = test_pp();
        let mut ctr = 0u64;
        // Two messages; relation: m0·m1 − t = 0 for a public t.
        let m0 = PolyK::uniform(4, b"q0", &mut ctr);
        let m1 = PolyK::uniform(4, b"q1", &mut ctr);
        let t = m0.mul(&m1);
        let (com, op) = AbdlopOpening::commit_rk(&pp, &[m0.clone(), m1.clone()], &[], &mut rng);
        let block = Block { com: com.clone(), op };
        let rel = QuadRelation {
            products: vec![((0, 0), (0, 1), PolyK::one(4))],
            linear: vec![],
            constant: t.clone().neg(),
        };
        let widths = (300.0, 300.0);
        let tr =
            pok_quadratic(&pp, &[block], std::slice::from_ref(&rel), widths, widths, 6.0, 1, 128, &mut rng)
                .unwrap();
        verify_quadratic(&pp, std::slice::from_ref(&com), std::slice::from_ref(&rel), &tr, widths).unwrap();
        // Tamper: change z1 — must fail.
        let mut bad = tr.clone();
        bad.z1[0][0] = bad.z1[0][0].add(&Poly::one(4));
        assert!(verify_quadratic(&pp, std::slice::from_ref(&com), std::slice::from_ref(&rel), &bad, widths).is_err());
    }

    #[test]
    fn ct_wrapper_roundtrip_and_tamper() {
        let (pp, mut rng) = test_pp();
        let mut ctr = 0u64;
        let msgs: Vec<PolyK> = (0..2).map(|_| PolyK::uniform(4, b"w", &mut ctr)).collect();
        // ct-relation: ct(m0 + m1) = target.
        let target = msgs[0].ct().add(&msgs[1].ct());
        let (com, op) = AbdlopOpening::commit_rk(&pp, &msgs, &[], &mut rng);
        let block = Block { com: com.clone(), op };
        // Message-weight expansion: w on the a-slot, w·Y on the b-slot
        // (a row reading w·(a + bY) needs both slots).
        let yw = y_weight(4);
        let rows = RelRows {
            rows: vec![RelRow::const_coeff(
                vec![
                    (0, 0, PolyK::one(4)),
                    (0, 1, yw.clone()),
                    (0, 2, PolyK::one(4)),
                    (0, 3, yw.clone()),
                ],
                target,
                4,
            )],
        };
        let gammas = vec![K::from_fp(Fq::ONE)];
        let widths = (300.0, 300.0);
        let tr = pok_ct_wrapper(
            &pp,
            &[block],
            Some((&rows, gammas.clone())),
            None,
            widths,
            6.0,
            1,
            128,
            &mut rng,
        )
        .unwrap();
        verify_ct_wrapper(&pp, std::slice::from_ref(&com), Some((&rows, gammas.clone())), None, &tr).unwrap();
        // Tamper h: ct(h) ≠ 0 must fail.
        let mut bad = tr.clone();
        bad.h = bad.h.add(&PolyK::degree0(K::from_fp(Fq::ONE), 4));
        assert!(verify_ct_wrapper(&pp, std::slice::from_ref(&com), Some((&rows, gammas.clone())), None, &bad).is_err());
    }

    #[test]
    fn anchoring_roundtrip() {
        let (pp, mut rng) = test_pp();
        let mut ctr = 0u64;
        let msgs: Vec<PolyK> = (0..2).map(|_| PolyK::uniform(4, b"a", &mut ctr)).collect();
        let (com, op) = AbdlopOpening::commit_rk(&pp, &msgs, &[], &mut rng);
        let block = Block { com: com.clone(), op };
        // Anchored: ct(3·m0 − target) = 0 — the message weight 3 expands
        // to (3 on the a-slot, 3·Y on the b-slot).
        let target = msgs[0].ct().scale_fp(&Fq::new(3));
        let w3 = PolyK::degree0(K::from_fp(Fq::new(3)), 4);
        let w3y = PolyK {
            a: Poly::zero(4),
            b: w3.a.clone(),
        };
        let weights = vec![
            (0usize, 0usize, w3),
            (0usize, 1usize, w3y),
        ];
        let widths = (300.0, 300.0);
        let tr = pok_anchor(&pp, &[block], &weights, target, widths, 6.0, 1, 128, &mut rng)
            .unwrap();
        verify_anchor(&pp, std::slice::from_ref(&com), &weights, target, &tr).unwrap();
    }

    #[test]
    fn simulator_produces_accepting_transcripts() {
        let (pp, mut rng) = test_pp();
        let rows = RelRows {
            rows: vec![RelRow::const_coeff(
                vec![(0, 0, PolyK::one(4))],
                K::ZERO,
                4,
            )],
        };
        let u = vec![PolyK::zero(4)];
        let widths = (300.0, 300.0);
        let (coms, tr) = simulate_linear_pok(&pp, 1, &rows, &u, widths, 6.0, 1, &mut rng);
        verify_linear(&pp, &coms, &rows, &u, &tr).unwrap();
    }

    #[test]
    fn poly_inverse_works() {
        let mut ctr = 0u64;
        for seed in 0..8 {
            let a = Poly::small_b(4, 2, format!("inv{seed}").as_bytes(), &mut ctr);
            if a.is_zero() {
                continue;
            }
            if let Some(inv) = poly_inverse(&a, 4) {
                assert_eq!(a.mul(&inv), Poly::one(4));
            }
        }
    }
}

#[cfg(test)]
mod quadratic_primitives {
    use super::*;
    use crate::abdlop::AbdlopPp;
    use crate::gauss::Rng;

    #[test]
    fn quadratic_expansion_identity() {
        // f(x̃0 + c·x) = f(x̃0) + c·cross(x̃0,x) + c²·f(x)
        let _rng = Rng::new(b"qexp");
        let d = 4;
        let mut ctr = 0u64;
        let x0 = vec![PolyK::uniform(d, b"x0", &mut ctr), PolyK::uniform(d, b"x0b", &mut ctr)];
        let x1 = vec![PolyK::uniform(d, b"x1", &mut ctr), PolyK::uniform(d, b"x1b", &mut ctr)];
        let c = Poly::small_b(d, 2, b"qc", &mut ctr);
        let t = x1[0].mul(&x1[1]);
        let rel = QuadRelation {
            products: vec![((0, 0), (0, 1), PolyK::one(d))],
            linear: vec![],
            constant: t.clone().neg(),
        };
        // x̃0 + c·x:
        let c_k = K(c.ct(), Fq::ZERO);
        let sum = vec![
            x0[0].add(&x1[0].scale_k(&c_k)),
            x0[1].add(&x1[1].scale_k(&c_k)),
        ];
        let lhs = rel.eval(&[sum]);
        let rhs = rel
            .eval(std::slice::from_ref(&x0))
            .add(&rel.cross(std::slice::from_ref(&x0), std::slice::from_ref(&x1)).scale_k(&c_k))
            .add(&rel.quad_part(std::slice::from_ref(&x1)).scale_k(&c_k).scale_k(&c_k));
        assert_eq!(lhs, rhs, "the quadratic expansion identity must hold");
    }

    #[test]
    fn masked_slots_identity() {
        // c·tB − B·z2 == c·m − B·y2 for z2 = y2 + c·s2, tB = Bs2 + m.
        let mut rng = Rng::new(b"msk");
        let pp = AbdlopPp::setup(3, 6, 4, 6, 4, &mut rng);
        let mut ctr = 0u64;
        let msgs: Vec<PolyK> = (0..2).map(|_| PolyK::uniform(4, b"mm", &mut ctr)).collect();
        let (com, op) = AbdlopOpening::commit_rk(&pp, &msgs, &[], &mut rng);
        let c = Poly::small_b(4, 2, b"mc", &mut ctr);
        let y2: Vec<Poly> = (0..pp.m2).map(|_| Poly::small_b(4, 2, b"my", &mut ctr)).collect();
        let z2: Vec<Poly> = y2.iter().zip(op.s2.iter()).map(|(y, s)| y.add(&c.mul(s))).collect();
        let mtilde = masked_slots(&pp, &com, &c, &z2);
        // Expected: c·m − B·y2 per slot.
        for slot in 0..pp.ell {
            let expect = c.mul(&op.slots[slot]).sub(&pp.b_s2(&y2)[slot]);
            assert_eq!(mtilde[slot], expect, "slot {slot}");
        }
    }
}
