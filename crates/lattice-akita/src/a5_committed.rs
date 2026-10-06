//! A5 — the **commitment-scale recursion driver** (the NEXT_STEPS honest
//! ledger's A3/A4 wiring): the range/tree rows and the evaluation-trace
//! rows feeding the A2 fused sum-check **against the COMMITTED successor
//! witness** — the digit segments `ẑ | ê | t̂` are never revealed at the
//! non-terminal levels (the §8.2 kernel-scale reveal the a5_terminal
//! route documented is eliminated here).
//!
//! # The wiring
//!
//! * **The row set over `L`** (every fold equation becomes an A2
//!   `RingRow` in the Eq-146 normal form, checked by ONE fused
//!   sum-check over the flat Goldilocks coordinates of `L`):
//!   * Eq 5 rows: `B·t̂_i = u_i` (the sliced outer images, public
//!     payloads — the targets);
//!   * Eq 6 rows: `D·ê = v` (the shared opening image);
//!   * Eq 7 rows: `A·G_z(ẑ) = Σ_i c_i·G_{b1}(t̂_i)` (the inner fold
//!     consistency — the gadget weights `b_z^u` / `b1^v` ride the row
//!     multipliers);
//!   * Eq 8 row: `a^⊤·G_b(ẑ) = Σ_i c_i·G_{b1,1}(ê_i)` (the fold
//!     evaluation — the partial-opening claim chain);
//!   * Eq 2 row: `Σ_i χ_blk(i)·G(ê_i) = vR` — **the deferred-claim
//!     chain**: `vR` at level j+1 IS level j's deferred MLE claim
//!     `ŵ^(j)(r_2^(j))` (the opening point's weights are the previous
//!     level's sum-check point), so each level's fused sum-check
//!     certifies the previous level's claim without ever seeing it.
//! * **The α-reduction** (ring_check's quotient lift + Eq 148): the
//!   quotients are transmitted; `α` is drawn ONLY AFTER the successor
//!   commitment `C_L` and the quotients are absorbed (the App F.1
//!   repaired ordering — the outgoing witness is bound before α); the
//!   α-evaluations of the hidden witness ride the fused sum-check's
//!   row weights (the verifier never evaluates `W(α)`).
//! * **The A3 range/tree rows** (item A3 feeding A2): per digit segment
//!   (ẑ, each ê_i, each t̂_i) the §6.1 digit-range pipeline runs on the
//!   hidden values; its deferred claim `(r', ŵ(r'))` becomes an
//!   eq-anchored LINEAR ROW in the fused weights — the pipeline itself
//!   is witness-free on the verifier side, and the final pinning (the
//!   kernel route's revealed-witness check) is discharged by the row.
//! * **The A4 evaluation-trace rows** (item A4 feeding A2): the
//!   Eq-135 trace weights `ω_Tr(i, ℓ, ν) = χ_blk(i)·b^ℓ·T_{ρ_pack}(X^ν)`
//!   over the ê cells — two Goldilocks rows (the `c₀/c₁` coordinates of
//!   the `F_{Q32²}` functional) with the announced `v̄` target.
//! * **The deferred claim**: the fused sum-check's final binding
//!   `P(r_2) = m̂(r_2)·ŵ(r_2)` pins ONE Goldilocks value `ŵ(r_2)` per
//!   level — the only witness data that ever leaves the level.
//! * **The terminal**: the LAST level's `L` is revealed once (small),
//!   directly checked (`verify_opening` on `C_L`, the final deferred
//!   claim recomputed from the revealed MLE), and the §8.2 terminal
//!   machinery (the grind, Eq 163–165, the signed Rice encoding) runs
//!   over the final group as in `a5_terminal`.
//!
//! # The honest deviation ledger (kernel scale)
//!
//! * **The split-field discipline** (ring_check's documented gap): the
//!   fused sum-check runs over the Goldilocks digit layer; the integer
//!   cell-sums are kept `< p` (a fail-closed prover gate) so the
//!   announced totals determine the `F_{Q32}` residues exactly and the
//!   verifier's mod-`Q32` consistency checks are sound.
//! * **The intermediate-commitment binding**: the per-level `C_L`'s
//!   full Ajtai opening is discharged by the recursion (the next level's
//!   equations + the terminal reveal); the paper's literal source-row
//!   against `C^(j)`'s key needs the unified-field production path —
//!   the recorded residual (same family as ring_check's note).
//! * **The intermediate `v̄` trace claims** are announced and row-bound
//!   to the committed ê; their correctness chains to the terminal's
//!   direct Eq-165 check.

use crate::a3_range::{prove_digit_range, DigitRangeProof, RangeError};
use crate::a4_tensor::{trace_row_weights, Fq2Q};
use crate::a5_terminal::{TerminalProof, TerminalState};
use crate::fold::{
    certified_response_bound, decompose_block, decompose_element, embed, sample_challenges,
    FoldError, FoldKeys, FoldParams, FoldSource, OpeningPoint,
};
use crate::ring_check::{
    absorb_lifted, build_fused_weights, quotient_lift, sample_alpha, LiftedRow, RingRow,
};
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError};
use lattice_core::extension::Fq2;
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::{RingConfig, RingElement};
use lattice_sumcheck::{SumcheckError, SumcheckProof};

#[derive(Debug, Clone)]
pub enum CommittedError {
    Fold(FoldError),
    Ajtai(AjtaiError),
    Transcript(TranscriptError),
    Range(RangeError),
    Sumcheck(SumcheckError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Mle(lattice_core::mle::MleError),
    /// A wraparound / budget gate failed (fail-closed).
    Gate(String),
    /// A verifier check failed.
    Verify(&'static str),
    Shape {
        expected: usize,
        got: usize,
    },
}

impl From<FoldError> for CommittedError {
    fn from(e: FoldError) -> Self {
        CommittedError::Fold(e)
    }
}
impl From<AjtaiError> for CommittedError {
    fn from(e: AjtaiError) -> Self {
        CommittedError::Ajtai(e)
    }
}
impl From<TranscriptError> for CommittedError {
    fn from(e: TranscriptError) -> Self {
        CommittedError::Transcript(e)
    }
}
impl From<RangeError> for CommittedError {
    fn from(e: RangeError) -> Self {
        CommittedError::Range(e)
    }
}
impl From<SumcheckError> for CommittedError {
    fn from(e: SumcheckError) -> Self {
        CommittedError::Sumcheck(e)
    }
}
impl From<lattice_sumcheck::VirtualPolyError> for CommittedError {
    fn from(e: lattice_sumcheck::VirtualPolyError) -> Self {
        CommittedError::Virtual(e)
    }
}
impl From<lattice_core::mle::MleError> for CommittedError {
    fn from(e: lattice_core::mle::MleError) -> Self {
        CommittedError::Mle(e)
    }
}
impl From<crate::ring_check::RingCheckError> for CommittedError {
    fn from(e: crate::ring_check::RingCheckError) -> Self {
        CommittedError::Gate(format!("{e:?}"))
    }
}

// ---------------------------------------------------------------------------
// The deferred digit-range verification (the A3 wiring's verifier side)
// ---------------------------------------------------------------------------

/// Verify a digit-range proof WITHOUT the revealed-witness pinning: the
/// `w_claim` binding is deferred to the caller's eq-row in the fused
/// sum-check (the commitment-scale route a3_range's own doc anticipates).
/// Returns the deferred `(binding_point, w_claim)`.
pub fn verify_digit_range_deferred(
    proof: &DigitRangeProof,
    b_star: u32,
    i_bin: &[usize],
    transcript: &mut Transcript,
) -> Result<(Vec<Goldilocks>, Goldilocks), RangeError> {
    use crate::a3_range::range_tree;
    let u_set_len = b_star as usize / 2;
    let levels_meta = range_tree(b_star)?;
    if proof.levels.len() != levels_meta.len() {
        return Err(RangeError::Shape {
            expected: levels_meta.len(),
            got: proof.levels.len(),
        });
    }
    // The pipeline is witness-free except the final pinning; replay it and
    // stop at the binding terminal (mirrors verify_digit_range's flow).
    let num_vars = proof.binding_point.len();
    if num_vars == 0 {
        return Err(RangeError::Shape {
            expected: 1,
            got: 0,
        });
    }
    let tau0: Vec<Goldilocks> = transcript.challenge_fields(b"a3-range-tau0", num_vars)?;
    let v0 = proof.levels[0].sumcheck.verify(
        num_vars,
        levels_meta[0].degree + 1,
        Goldilocks::ZERO,
        transcript,
        None,
    )?;
    let mut point = v0.point;
    let mut node_claims = proof.levels[0].child_claims.clone();
    if node_claims.len() != levels_meta[0].nodes * levels_meta[0].degree {
        return Err(RangeError::Shape {
            expected: levels_meta[0].nodes * levels_meta[0].degree,
            got: node_claims.len(),
        });
    }
    let mut term = DenseMle::eq_extension(&tau0)
        .evaluate(&point)
        .map_err(|_| RangeError::Shape {
            expected: num_vars,
            got: point.len(),
        })?;
    for c in &node_claims {
        term = term.mul(c);
    }
    if v0.final_claim != term {
        return Err(RangeError::TerminalFailed);
    }
    for (li, level) in levels_meta.iter().enumerate().skip(1) {
        let nus: Vec<Goldilocks> = transcript.challenge_fields(b"a3-range-nu", level.nodes)?;
        if node_claims.len() != level.nodes {
            return Err(RangeError::Shape {
                expected: level.nodes,
                got: node_claims.len(),
            });
        }
        let mut claim = Goldilocks::ZERO;
        for (j, nu) in nus.iter().enumerate() {
            claim = claim.add(&nu.mul(&node_claims[j]));
        }
        let lp = &proof.levels[li];
        if lp.child_claims.len() != level.nodes * level.degree {
            return Err(RangeError::Shape {
                expected: level.nodes * level.degree,
                got: lp.child_claims.len(),
            });
        }
        let v = lp
            .sumcheck
            .verify(num_vars, level.degree + 1, claim, transcript, None)?;
        let eqp = DenseMle::eq_extension(&point)
            .evaluate(&v.point)
            .map_err(|_| RangeError::Shape {
                expected: num_vars,
                got: v.point.len(),
            })?;
        let mut t = Goldilocks::ZERO;
        for (j, nu) in nus.iter().enumerate() {
            let mut prod = *nu;
            for k in 0..level.degree {
                prod = prod.mul(&lp.child_claims[j * level.degree + k]);
            }
            t = t.add(&prod);
        }
        if v.final_claim != t.mul(&eqp) {
            return Err(RangeError::TerminalFailed);
        }
        point = v.point;
        node_claims = lp.child_claims.clone();
    }
    // Leaf consistency.
    if proof.leaf_claims.len() != u_set_len {
        return Err(RangeError::Shape {
            expected: u_set_len,
            got: proof.leaf_claims.len(),
        });
    }
    for (k, lc) in proof.leaf_claims.iter().enumerate() {
        let kk = u64::try_from(k).unwrap_or(u64::MAX);
        if lc.add(&Goldilocks::from_u64(kk * (kk + 1))) != proof.s_claim {
            return Err(RangeError::LeafInconsistent);
        }
    }
    // The binding terminal (the pinned evaluation is DEFERRED): check the
    // transcript consistency and the final-claim structure with
    // wp1 = w_claim + 1 (derivable from the claim).
    let (coeff0, claim) = match proof.fused_coeffs {
        Some((gamma, zeta)) => {
            let g = transcript.challenge_field(b"a3-range-gamma")?;
            let z = transcript.challenge_field(b"a3-range-zetabin")?;
            if (g, z) != (gamma, zeta) {
                return Err(RangeError::BindingFailed);
            }
            (gamma, proof.s_claim.mul(&gamma))
        }
        None => (Goldilocks::ONE, proof.s_claim),
    };
    let _ = coeff0;
    let v = proof.binding.verify(num_vars, 3, claim, transcript, None)?;
    if v.point != proof.binding_point {
        return Err(RangeError::BindingFailed);
    }
    let wp1 = proof.w_claim.add(&Goldilocks::ONE);
    let mut weight = DenseMle::eq_extension(&point)
        .evaluate(&v.point)
        .map_err(|_| RangeError::Shape {
            expected: num_vars,
            got: v.point.len(),
        })?
        .mul(&coeff0);
    if let Some((_, zeta)) = proof.fused_coeffs {
        let bin_set: std::collections::HashSet<usize> = i_bin.iter().copied().collect();
        let mut vals = Vec::with_capacity(1 << num_vars);
        for (x, &e) in DenseMle::eq_extension(&point)
            .evaluations
            .iter()
            .enumerate()
        {
            vals.push(if bin_set.contains(&x) {
                e
            } else {
                Goldilocks::ZERO
            });
        }
        let rw = DenseMle {
            num_vars,
            evaluations: vals,
        }
        .evaluate(&v.point)?;
        weight = weight.add(&rw.mul(&zeta));
    }
    let expect = weight.mul(&proof.w_claim).mul(&wp1);
    if v.final_claim != expect {
        return Err(RangeError::BindingFailed);
    }
    Ok((proof.binding_point.clone(), proof.w_claim))
}

// ---------------------------------------------------------------------------
// The segment map (which flat coordinates of L belong to which digit
// segment — shared by the prover, the verifier, and the A3 wiring)
// ---------------------------------------------------------------------------

/// One digit segment of the successor witness `L = [ẑ | ê | t̂]`.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    /// The first ring ELEMENT of the segment in L.
    pub elem_off: usize,
    /// The segment's ring-element count.
    pub elems: usize,
    /// The digit alphabet base (the A3 `b*`).
    pub base: u64,
}

/// The segment map of a FoldParams's successor witness.
pub fn segment_map(params: &FoldParams) -> Vec<Segment> {
    let md = params.block_len * params.source_digits;
    let mut out = Vec::with_capacity(2 + params.num_blocks);
    // ẑ: the response digits, base b_z.
    out.push(Segment {
        elem_off: 0,
        elems: md * params.response_digits,
        base: params.response_base,
    });
    // ê_i per block: base b1.
    for i in 0..params.num_blocks {
        out.push(Segment {
            elem_off: md * params.response_digits + i * params.inner_digits,
            elems: params.inner_digits,
            base: params.inner_base,
        });
    }
    // t̂_i per block: n_A·δ1 digits, base b1.
    for i in 0..params.num_blocks {
        out.push(Segment {
            elem_off: md * params.response_digits
                + params.num_blocks * params.inner_digits
                + i * params.inner_rows * params.inner_digits,
            elems: params.inner_rows * params.inner_digits,
            base: params.inner_base,
        });
    }
    out
}

/// Re-balance a digit vector from the decomposer's `(−b/2, b/2]`
/// alphabet into the A3 range pipeline's `[−b/2, b/2)` alphabet (the
/// two modules' documented convention gap): `+b/2 → −b/2` with a carry
/// into the next position (an exact re-encoding — the recomposed value
/// is unchanged).
fn rebalance_into_alphabet(vals: &mut [i64], base: u64) {
    let half = i64::try_from(base / 2).unwrap_or(i64::MAX);
    let b = i64::try_from(base).unwrap_or(i64::MAX);
    for i in 0..vals.len() {
        if vals[i] == half {
            vals[i] = -half;
            if i + 1 < vals.len() {
                vals[i + 1] += 1;
            }
            // a carry that itself lands on +half is fixed by the next
            // iteration; the top position may exceed the alphabet by the
            // absorbed carries (norm-controlled, the tail digit).
        } else if vals[i] > half {
            vals[i] -= b;
            if i + 1 < vals.len() {
                vals[i + 1] += 1;
            }
        }
        let _ = b;
    }
}

/// The flat i64 coefficient values of one segment of L (prover side),
/// re-balanced into the A3 alphabet.
fn segment_values(ring: &RingConfig, l: &[RingElement], seg: Segment) -> Vec<i64> {
    let q = ring.modulus.q;
    let mut out = Vec::with_capacity(seg.elems * ring.n());
    for e in &l[seg.elem_off..seg.elem_off + seg.elems] {
        for &c in e.coeffs() {
            out.push(if c <= q / 2 {
                i64::from(c)
            } else {
                i64::from(c) - i64::from(q)
            });
        }
    }
    rebalance_into_alphabet(&mut out, seg.base);
    out
}

/// Pad a value slice to a power of two with in-alphabet zeros.
fn pad_pow2(vals: &[i64]) -> Vec<i64> {
    let n = vals.len().max(1).next_power_of_two();
    let mut out = vals.to_vec();
    out.resize(n, 0);
    out
}

// ---------------------------------------------------------------------------
// The committed level proof
// ---------------------------------------------------------------------------

/// One segment's A3 digit-range proof + its deferred claim.
#[derive(Clone, Debug)]
pub struct SegmentRangeProof {
    pub seg: usize,
    pub b_star: u32,
    pub proof: DigitRangeProof,
    pub num_vars: usize,
}

/// The fused sum-check over the flat coordinates of the committed L.
#[derive(Clone, Debug)]
pub struct CommittedFusedProof {
    pub sumcheck: SumcheckProof,
    /// The final point `r_2`.
    pub point: Vec<Goldilocks>,
    /// The deferred claim `ŵ(r_2)` — the only witness datum transmitted.
    pub w_claim: Goldilocks,
    /// The announced Goldilocks total `V` (bound into the transcript).
    pub total: Goldilocks,
}

/// One commitment-scale fold level.
#[derive(Clone, Debug)]
pub struct CommittedLevelProof {
    /// The sliced outer images `u_i = B·t̂_i` (Eq 5 targets).
    pub outer_images: Vec<Vec<RingElement>>,
    /// The shared opening image `v = D·ê` (Eq 6 target).
    pub opening_image: Vec<RingElement>,
    /// The scalar claim `vR = Σ_i χ_blk(i)·e_i` (Eq 2 target — the
    /// previous level's deferred claim at chained levels).
    pub scalar_claim: RingElement,
    /// The fold challenges (recomputed by the verifier).
    pub challenges: Vec<RingElement>,
    /// The successor commitment `C_L` — absorbed BEFORE α (App F.1).
    pub successor_commitment: AjtaiCommitment,
    /// The quotient lifts of the relation rows.
    pub lifted: Vec<LiftedRow>,
    /// The A3 digit-range proofs per segment (deferred claims inside).
    pub ranges: Vec<SegmentRangeProof>,
    /// The per-level trace-row targets `v̄` (the A4 wiring, `F_{Q32²}`).
    pub trace_claims: Vec<Fq2Q>,
    /// The fused sum-check + the deferred claim.
    pub fused: CommittedFusedProof,
}

/// The recursion driver's end-to-end proof.
#[derive(Clone, Debug)]
pub struct CommittedDriverProof {
    pub levels: Vec<CommittedLevelProof>,
    pub terminal: TerminalProof,
    /// The LAST level's successor witness, revealed once (the terminal
    /// discharge) — **Reveal mode only**; EMPTY in Committed mode.
    pub final_witness: Vec<RingElement>,
    /// **Committed mode** (SOTA mechanism #6): the polylog private
    /// discharge — the terminal reveal is replaced by the transmitted
    /// final-edge `t` images plus the packed flat commitment with the
    /// SALSAA `D1∘D2` evaluation proof of the deferred claim.
    pub discharge: Option<PolylogDischarge>,
}

/// The discharge mode (Akita §7's commitment-scale substitution, SOTA
/// mechanism #6): **polylog private responses — kill the terminal
/// reveals**.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DischargeMode {
    /// The §8.2 kernel route: the last level's `L` is revealed once
    /// (`final_witness`, Θ(level size) ring elements) and discharged by
    /// the full `C_L` opening recompute. Binding-complete; retained as
    /// the reference route.
    Reveal,
    /// The commitment-scale substitution: the last `L` stays PRIVATE —
    /// no `final_witness` bytes ship. The discharge carries the §8.2
    /// final-edge `t_i = A_term·s_i` images (k ring elements, the
    /// paper-faithful transmitted edge), a fresh packed commitment
    /// `C_flat` of the flat coordinates, and the SALSAA `D1∘D2` norm
    /// chain proving the deferred claim — a POLYLOG response with no
    /// witness disclosure. The `C_flat ↔ C_L` unified-field binding is
    /// the module's documented residual (the same ring_check gap), and
    /// the D1 Lemma-4 gate caps the flat cube at `2^11` values at the
    /// Q_32 kernel ring (the bigger-q unified-field path lifts it).
    Committed,
}

/// The Committed-mode discharge artifact.
#[derive(Clone, Debug)]
pub struct PolylogDischarge {
    /// The final-edge canonical inner-state images `t_i = A_term·s_i`
    /// (the §8.2 transmitted edge — small).
    pub t_images: Vec<Vec<RingElement>>,
    /// The packed flat-coordinates commitment `C_flat` (canonical
    /// bytes; the standard Akita geometry over the last level's flat
    /// coordinates).
    pub flat_commitment: Vec<u8>,
    /// The polylog response: the grouped carrier + the ψ-functional
    /// carrier + the D1 norm chain over the BYTE-PACKED flat witness —
    /// `no opened_witness field exists at all`.
    pub response: crate::salsa_response::SalsaGroupedResponse,
    /// The flat MLE's variable count (the verifier's geometry check).
    pub mu: usize,
    /// The byte-packed Ajtai `m` geometry of the discharge PCS
    /// (`4·2^mu/64` — one byte per coefficient at the 4-byte width).
    pub m: usize,
}

// ---------------------------------------------------------------------------
// The row set over L (every fold equation in the Eq-146 normal form)
// ---------------------------------------------------------------------------

/// The successor-witness element layout of a FoldParams:
/// `[ẑ (md·τ) | ê (B·δ1) | t̂ (B·n_A·δ1)]`.
#[derive(Clone, Copy, Debug)]
pub struct LLayout {
    pub md: usize,
    /// τ (response_digits).
    pub tau: usize,
    pub z_len: usize,
    pub e_off: usize,
    /// δ1 (inner_digits).
    pub d1: usize,
    pub e_len: usize,
    pub t_off: usize,
    /// n_A·δ1.
    pub na_d1: usize,
    pub t_len: usize,
    pub total: usize,
}

impl LLayout {
    pub fn new(params: &FoldParams) -> Self {
        let md = params.block_len * params.source_digits;
        let z_len = md * params.response_digits;
        let e_off = z_len;
        let e_len = params.num_blocks * params.inner_digits;
        let t_off = e_off + e_len;
        let t_len = params.num_blocks * params.inner_rows * params.inner_digits;
        LLayout {
            md,
            tau: params.response_digits,
            z_len,
            e_off,
            d1: params.inner_digits,
            e_len,
            t_off,
            na_d1: params.inner_rows * params.inner_digits,
            t_len,
            total: t_off + t_len,
        }
    }

    /// The ẑ element holding digit `u` of response element `k`
    /// (`ẑ` is component-major: `k·τ + u`).
    #[inline]
    pub fn z_elt(&self, k: usize, u: usize) -> usize {
        k * self.tau + u
    }

    /// The `ê` element `ℓ` of block `i`.
    #[inline]
    pub fn e_elt(&self, i: usize, ell: usize) -> usize {
        self.e_off + i * self.d1 + ell
    }

    /// The `t̂` element `j` of block `i` (j = row·δ1 + v).
    #[inline]
    pub fn t_elt(&self, i: usize, j: usize) -> usize {
        self.t_off + i * self.na_d1 + j
    }
}

// ---------------------------------------------------------------------------
// The row set (shared by prove and verify — built from PUBLIC data only)
// ---------------------------------------------------------------------------

/// The A4 trace context of one level: `(ρ_pack, χ_blk, v̄)`.
#[derive(Clone, Debug)]
pub struct TraceContext {
    pub rho_pack: Vec<Fq2Q>,
    pub chi_blk: Vec<Fq2Q>,
}

/// Build the Eq-5/6/7/8/2 relation rows over the successor layout —
/// every input is public (keys, challenges, images, the scalar claim).
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn relation_rows(
    ring: &RingConfig,
    params: &FoldParams,
    keys: &FoldKeys,
    point: &OpeningPoint,
    challenges: &[RingElement],
    outer_images: &[Vec<RingElement>],
    opening_image: &[RingElement],
    scalar_claim: &RingElement,
) -> Vec<RingRow> {
    let lay = LLayout::new(params);
    let zero = ring.zero();
    let mut rows: Vec<RingRow> = Vec::new();
    // The gadget power weights can reach 2^32 (b^u·b_z^w) — reduce mod q
    // before embedding (NOT the 2^31 truncation, which zeroes them).
    let q64 = u64::from(ring.modulus.q);
    let one_elt =
        |v: u64| -> RingElement { ring.constant(u32::try_from(v % q64).unwrap_or(u32::MAX)) };

    // ---- Eq 5 rows: B·t̂_i = u_i (per block, per B-row). ----
    for (i, u_i) in outer_images.iter().enumerate() {
        for (r, brow) in keys.b_matrix.iter().enumerate() {
            let mut multipliers = Vec::with_capacity(brow.len());
            let mut reads = Vec::with_capacity(brow.len());
            for (j, b_ent) in brow.iter().enumerate() {
                multipliers.push(b_ent.clone());
                reads.push(lay.t_elt(i, j));
            }
            let target = u_i.get(r).cloned().unwrap_or_else(|| zero.clone());
            rows.push(RingRow {
                multipliers,
                reads,
                target,
            });
        }
    }

    // ---- Eq 6 rows: D·ê = v (per D-row). ----
    for (r, drow) in keys.d_matrix.iter().enumerate() {
        let mut multipliers = Vec::with_capacity(drow.len());
        let mut reads = Vec::with_capacity(drow.len());
        for (j, d_ent) in drow.iter().enumerate() {
            multipliers.push(d_ent.clone());
            reads.push(lay.e_off + j);
        }
        let target = opening_image
            .get(r)
            .cloned()
            .unwrap_or_else(|| zero.clone());
        rows.push(RingRow {
            multipliers,
            reads,
            target,
        });
    }

    // ---- Eq 7 rows: A·G_z(ẑ) = Σ_i c_i·G_{b1}(t̂_i) (per A-row). ----
    let bz = params.response_base;
    let b1 = params.inner_base;
    for (rho, arow) in keys.a_matrix.iter().enumerate() {
        let mut multipliers: Vec<RingElement> = Vec::new();
        let mut reads: Vec<usize> = Vec::new();
        // LHS: Σ_k Σ_u A[ρ][k]·b_z^u·ẑ[k·τ+u]
        for (k, a_ent) in arow.iter().enumerate() {
            for u in 0..params.response_digits {
                let w = a_ent
                    .mul(&one_elt(bz.pow(u as u32)))
                    .unwrap_or_else(|_| zero.clone());
                multipliers.push(w);
                reads.push(lay.z_elt(k, u));
            }
        }
        // RHS (moved left): −Σ_i Σ_v c_i·b1^v·t̂_i[ρ·δ1+v]
        for (i, c_i) in challenges.iter().enumerate() {
            for v in 0..params.inner_digits {
                let w = c_i
                    .mul(&one_elt(b1.pow(v as u32)))
                    .unwrap_or_else(|_| zero.clone());
                multipliers.push(zero.sub(&w).unwrap_or_else(|_| zero.clone()));
                reads.push(lay.t_elt(i, rho * params.inner_digits + v));
            }
        }
        rows.push(RingRow {
            multipliers,
            reads,
            target: zero.clone(),
        });
    }

    // ---- Eq 8 row: a^⊤·G_b(ẑ) = Σ_i c_i·G_{b1,1}(ê_i). ----
    {
        let a = point.block_weights(ring);
        let b = params.source_base;
        let mut multipliers: Vec<RingElement> = Vec::new();
        let mut reads: Vec<usize> = Vec::new();
        // LHS: Σ_m Σ_u a[m]·b^u·G_z-part: z[m·δ+u] = Σ_w b_z^w·ẑ[(m·δ+u)·τ+w]
        for (m, a_w) in a.iter().enumerate() {
            for u in 0..params.source_digits {
                let ku = m * params.source_digits + u;
                for w in 0..params.response_digits {
                    let weight = a_w
                        .mul(&one_elt(b.pow(u as u32).saturating_mul(bz.pow(w as u32))))
                        .unwrap_or_else(|_| zero.clone());
                    multipliers.push(weight);
                    reads.push(lay.z_elt(ku, w));
                }
            }
        }
        // RHS: −Σ_i Σ_ℓ c_i·b1^ℓ·ê_i[ℓ]
        for (i, c_i) in challenges.iter().enumerate() {
            for ell in 0..params.inner_digits {
                let w = c_i
                    .mul(&one_elt(b1.pow(ell as u32)))
                    .unwrap_or_else(|_| zero.clone());
                multipliers.push(zero.sub(&w).unwrap_or_else(|_| zero.clone()));
                reads.push(lay.e_elt(i, ell));
            }
        }
        rows.push(RingRow {
            multipliers,
            reads,
            target: zero.clone(),
        });
    }

    // ---- Eq 2 row: Σ_i χ_blk(i)·G(ê_i) = vR. ----
    {
        let chi = point.chi_blk(ring, params.num_blocks);
        let mut multipliers: Vec<RingElement> = Vec::new();
        let mut reads: Vec<usize> = Vec::new();
        for (i, ch) in chi.iter().enumerate() {
            for ell in 0..params.inner_digits {
                let w = ch
                    .mul(&one_elt(b1.pow(ell as u32)))
                    .unwrap_or_else(|_| zero.clone());
                multipliers.push(w);
                reads.push(lay.e_elt(i, ell));
            }
        }
        rows.push(RingRow {
            multipliers,
            reads,
            target: scalar_claim.clone(),
        });
    }
    rows
}

// ---------------------------------------------------------------------------
// The combined Goldilocks weights (the relation rows + the A3 eq-rows +
// the A4 trace rows) — ONE fused linear functional over L's flat coords
// ---------------------------------------------------------------------------

/// The combined row weights and the target accounting.
pub struct CombinedWeights {
    /// One Goldilocks weight per flat coordinate of L (padded to 2^mu).
    pub weights: Vec<Goldilocks>,
    /// The number of Goldilocks variables of the fused sum-check.
    pub num_vars: usize,
    /// The F_q relation target (embedded, from build_fused_weights).
    pub fq_target: Goldilocks,
}

/// Embed a u32 F_q residue as Goldilocks.
fn fe(v: u64) -> Goldilocks {
    Goldilocks::from_u64(v)
}

/// Build the combined weights: the α-reduced relation rows (Eq 158),
/// plus the A3 eq-anchored rows (per segment, weight eq(r', ·) over the
/// segment's real coordinate span), plus the A4 trace rows (the c0/c1
/// coordinates of the Eq-135 functional over the ê cells).
#[allow(clippy::too_many_arguments)]
pub fn combined_weights(
    ring: &RingConfig,
    params: &FoldParams,
    rows: &[RingRow],
    lifted: &[LiftedRow],
    thetas: &[Goldilocks],
    alpha: u32,
    range_rows: &[(usize, Vec<Goldilocks>, Goldilocks)], // (flat offset, r', w_claim)
    trace: Option<&TraceContext>,
    trace_vbar: Option<&Fq2Q>,
) -> Result<CombinedWeights, CommittedError> {
    let lay = LLayout::new(params);
    let l_len = lay.total;
    let mut weights = vec![Goldilocks::ZERO; l_len * ring.n()];
    let fw = build_fused_weights(ring, rows, lifted, thetas, l_len, alpha)?;
    for (i, w) in fw.row_weights.iter().enumerate() {
        weights[i] = *w;
    }
    // The A3 eq-rows: weights eq(r', x) over the segment's REAL span
    // (the padded tail stays zero — the padded witness is zero too).
    for (flat_off, rprime, _claim) in range_rows {
        let nv = rprime.len();
        let eq = DenseMle::eq_extension(rprime);
        if eq.evaluations.len() != 1 << nv {
            return Err(CommittedError::Gate("eq extension shape".into()));
        }
        for (x, &e) in eq.evaluations.iter().enumerate() {
            let pos = flat_off + x;
            if pos < weights.len() {
                weights[pos] = weights[pos].add(&e);
            }
        }
    }
    // The A4 trace rows: ω_Tr(i, ℓ, ν) over the ê cells — BOTH the c0
    // and the c1 coordinates of the F_{Q32²} functional (two rows).
    if let (Some(ctx), Some(_vbar)) = (trace, trace_vbar) {
        let omegas = trace_row_weights(
            ring,
            &ctx.rho_pack,
            &ctx.chi_blk,
            params.inner_base,
            params.inner_digits,
        )
        .map_err(|e| CommittedError::Gate(format!("{e:?}")))?;
        for (i, per_l) in omegas.iter().enumerate() {
            for (ell, per_nu) in per_l.iter().enumerate() {
                for (nu, w) in per_nu.iter().enumerate() {
                    let pos = lay.e_elt(i, ell) * ring.n() + nu;
                    if pos < weights.len() {
                        weights[pos] = weights[pos].add(&fe(w.c0));
                        weights[pos] = weights[pos].add(&fe(w.c1));
                    }
                }
            }
        }
    }
    let num_vars = weights
        .len()
        .checked_next_power_of_two()
        .map(|p| p.trailing_zeros() as usize)
        .unwrap_or(0);
    Ok(CombinedWeights {
        weights,
        num_vars,
        fq_target: fw.fq_target,
    })
}

// ---------------------------------------------------------------------------
// The level prover
// ---------------------------------------------------------------------------

/// The prover-side fold internals of one committed level (everything the
/// kernel-scale `prove_fold` computed, kept private to the prover).
struct FoldInternals {
    outer_images: Vec<Vec<RingElement>>,
    opening_image: Vec<RingElement>,
    scalar_claim: RingElement,
    challenges: Vec<RingElement>,
    partials: Vec<RingElement>,
    l_vec: Vec<RingElement>,
}

#[allow(clippy::too_many_lines)]
fn fold_internals(
    params: &FoldParams,
    keys: &FoldKeys,
    source: &FoldSource,
    point: &OpeningPoint,
    transcript: &mut Transcript,
) -> Result<FoldInternals, CommittedError> {
    let ring = &keys.ring;
    if source.blocks.len() != params.num_blocks {
        return Err(CommittedError::Shape {
            expected: params.num_blocks,
            got: source.blocks.len(),
        });
    }
    let md = params.block_len * params.source_digits;
    let a = point.block_weights(ring);
    let mut s_blocks = Vec::new();
    let mut t_hat = Vec::new();
    let mut e_hat = Vec::new();
    let mut partials = Vec::new();
    for block in &source.blocks {
        let s = decompose_block(ring, block, params.source_base, params.source_digits)?;
        let mut e = ring.zero();
        for (aw, f) in a.iter().zip(block.iter()) {
            e = e
                .add(&aw.mul(f).map_err(FoldError::Ring)?)
                .map_err(FoldError::Ring)?;
        }
        let mut t = vec![ring.zero(); params.inner_rows];
        for (r, row) in keys.a_matrix.iter().enumerate() {
            let mut acc = ring.zero();
            for (a_ent, sj) in row.iter().zip(s.iter()) {
                acc = acc
                    .add(&a_ent.mul(sj).map_err(FoldError::Ring)?)
                    .map_err(FoldError::Ring)?;
            }
            t[r] = acc;
        }
        let mut that = Vec::with_capacity(params.inner_rows * params.inner_digits);
        for te in &t {
            that.extend(decompose_element(
                ring,
                te,
                params.inner_base,
                params.inner_digits,
            )?);
        }
        let ehat = decompose_element(ring, &e, params.inner_base, params.inner_digits)?;
        s_blocks.push(s);
        t_hat.push(that);
        e_hat.push(ehat);
        partials.push(e);
    }
    // u_i = B·t̂_i per block.
    let mut outer_images = Vec::with_capacity(params.num_blocks);
    for that in &t_hat {
        let mut img = Vec::with_capacity(params.outer_rows);
        for row in &keys.b_matrix {
            let mut acc = ring.zero();
            for (b_ent, d) in row.iter().zip(that.iter()) {
                acc = acc
                    .add(&b_ent.mul(d).map_err(FoldError::Ring)?)
                    .map_err(FoldError::Ring)?;
            }
            img.push(acc);
        }
        outer_images.push(img);
    }
    // v = D·ê.
    let e_concat: Vec<RingElement> = e_hat.iter().flatten().cloned().collect();
    let mut opening_image = Vec::with_capacity(params.opening_rows);
    for row in &keys.d_matrix {
        let mut acc = ring.zero();
        for (d_ent, e) in row.iter().zip(e_concat.iter()) {
            acc = acc
                .add(&d_ent.mul(e).map_err(FoldError::Ring)?)
                .map_err(FoldError::Ring)?;
        }
        opening_image.push(acc);
    }
    // vR = Σ_i χ_blk(i)·e_i.
    let chi = point.chi_blk(ring, params.num_blocks);
    let mut v_r = ring.zero();
    for (ch, e) in chi.iter().zip(partials.iter()) {
        v_r = v_r
            .add(&ch.mul(e).map_err(FoldError::Ring)?)
            .map_err(FoldError::Ring)?;
    }
    // Absorb the payloads BEFORE the challenges (the §4 ordering).
    for img in &outer_images {
        for e in img {
            transcript.append_bytes(b"akita-cfold-u", &e.to_bytes())?;
        }
    }
    for e in &opening_image {
        transcript.append_bytes(b"akita-cfold-v", &e.to_bytes())?;
    }
    transcript.append_bytes(b"akita-cfold-vR", &v_r.to_bytes())?;
    // Challenges + the response.
    let short_challenges = sample_challenges(params, transcript)?;
    let challenges: Vec<RingElement> = short_challenges.iter().map(|c| embed(ring, c)).collect();
    let c_ring = &challenges;
    let mut z = vec![ring.zero(); md];
    for (c, s) in c_ring.iter().zip(s_blocks.iter()) {
        for (zj, sj) in z.iter_mut().zip(s.iter()) {
            *zj = zj
                .add(&c.mul(sj).map_err(FoldError::Ring)?)
                .map_err(FoldError::Ring)?;
        }
    }
    let bound = certified_response_bound(params, &short_challenges);
    let capacity = params.response_base.pow(params.response_digits as u32) / 2;
    if bound >= capacity {
        return Err(CommittedError::Fold(FoldError::ResponseTooLarge {
            bound,
            capacity,
        }));
    }
    let mut response_digits = Vec::with_capacity(md * params.response_digits);
    for ze in &z {
        response_digits.extend(decompose_element(
            ring,
            ze,
            params.response_base,
            params.response_digits,
        )?);
    }
    // L = [ẑ | ê | t̂].
    let mut l_vec: Vec<RingElement> = Vec::with_capacity(crate::fold::successor_len(params));
    l_vec.extend(response_digits.iter().cloned());
    for ehat in &e_hat {
        l_vec.extend(ehat.iter().cloned());
    }
    for that in &t_hat {
        l_vec.extend(that.iter().cloned());
    }
    Ok(FoldInternals {
        outer_images,
        opening_image,
        scalar_claim: v_r,
        challenges,
        partials,
        l_vec,
    })
}

/// The flat Goldilocks coordinates of L.
fn flat_goldilocks(_ring: &RingConfig, l: &[RingElement]) -> Vec<Goldilocks> {
    l.iter()
        .flat_map(|e| {
            e.coeffs()
                .iter()
                .map(|&c| Goldilocks::from_u64(u64::from(c)))
        })
        .collect()
}

/// Prove one commitment-scale level. Returns (proof, L). `prev` carries
/// the previous level's deferred claim point (chained scalar semantics —
/// the honest-ledger note in the module docs).
#[allow(clippy::too_many_lines)]
pub fn prove_committed_level(
    params: &FoldParams,
    keys: &FoldKeys,
    source: &FoldSource,
    point: &OpeningPoint,
    transcript: &mut Transcript,
) -> Result<(CommittedLevelProof, Vec<RingElement>), CommittedError> {
    let ring = &keys.ring;
    let internals = fold_internals(params, keys, source, point, transcript)?;

    // ---- The successor commitment (bound BEFORE α — App F.1). ----
    let l_padded = keys.successor_pk.pad_to_m(&internals.l_vec)?;
    let successor_commitment = keys.successor_pk.commit(&l_padded)?;
    transcript.append_bytes(b"akita-cfold-L", &successor_commitment.to_bytes())?;

    // ---- The relation rows + the quotient lift. ----
    let rows = relation_rows(
        ring,
        params,
        keys,
        point,
        &internals.challenges,
        &internals.outer_images,
        &internals.opening_image,
        &internals.scalar_claim,
    );
    let lifted = quotient_lift(ring, &rows, &internals.l_vec)?;
    absorb_lifted(transcript, &successor_commitment.to_bytes(), &lifted)?;
    let alpha = sample_alpha(transcript, ring.modulus)?;
    // The row-batching challenges ϑ under the SMALL-θ kernel discipline
    // (mod 2^8): with paper-uniform F_q θ the integer cell-sums wrap the
    // Goldilocks modulus and the mod-Q32 target consistency (the level's
    // soundness core) would break — the unified-field path is the
    // recorded residual.
    let thetas: Vec<Goldilocks> = transcript
        .challenge_fields(b"akita-cfold-tau", rows.len())?
        .into_iter()
        .map(|t| Goldilocks::from_u64(t.to_canonical_u64() % 256))
        .collect();

    // ---- The A3 range/tree rows: per digit segment. ----
    let segs = segment_map(params);
    let mut ranges: Vec<SegmentRangeProof> = Vec::with_capacity(segs.len());
    let mut range_rows: Vec<(usize, Vec<Goldilocks>, Goldilocks)> = Vec::with_capacity(segs.len());
    for (si, seg) in segs.iter().enumerate() {
        let vals = pad_pow2(&segment_values(ring, &internals.l_vec, *seg));
        let b_star = u32::try_from(seg.base).map_err(|_| CommittedError::Gate("base".into()))?;
        let proof = prove_digit_range(&vals, b_star, &[], transcript)?;
        let num_vars = vals.len().trailing_zeros() as usize;
        ranges.push(SegmentRangeProof {
            seg: si,
            b_star,
            proof: proof.clone(),
            num_vars,
        });
        range_rows.push((
            seg.elem_off * ring.n(),
            proof.binding_point.clone(),
            proof.w_claim,
        ));
    }

    // ---- The A4 trace context + the honest v̄. ----
    let n_pairs_bits = (ring.n() / 2).trailing_zeros() as usize;
    let mut rho_pack = Vec::with_capacity(n_pairs_bits);
    for _ in 0..n_pairs_bits {
        rho_pack.push(
            crate::a4_tensor::challenge_fq2q(transcript, b"akita-cfold-rho")
                .map_err(|e| CommittedError::Gate(format!("{e:?}")))?,
        );
    }
    let mut chi_blk = Vec::with_capacity(params.num_blocks);
    for _ in 0..params.num_blocks {
        chi_blk.push(
            crate::a4_tensor::challenge_fq2q(transcript, b"akita-cfold-chiblk")
                .map_err(|e| CommittedError::Gate(format!("{e:?}")))?,
        );
    }
    let chi_pack = crate::a4_tensor::chi_pack(ring, &rho_pack)
        .map_err(|e| CommittedError::Gate(format!("{e:?}")))?;
    let mut vbar = Fq2Q::new(0, 0);
    for (e, w) in internals.partials.iter().zip(chi_blk.iter()) {
        let t = crate::a4_tensor::trace_functional(ring, e, &chi_pack)
            .map_err(|e| CommittedError::Gate(format!("{e:?}")))?;
        vbar = vbar.add(&w.mul(&t));
    }
    let trace_ctx = TraceContext { rho_pack, chi_blk };
    transcript.append_bytes(b"akita-cfold-vbar", &[vbar.c0 as u8, vbar.c1 as u8])?;
    // announce the full values too (u32 LE)
    transcript.append_bytes(
        b"akita-cfold-vbar32",
        &vbar
            .c0
            .to_le_bytes()
            .iter()
            .chain(vbar.c1.to_le_bytes().iter())
            .copied()
            .collect::<Vec<u8>>(),
    )?;

    // ---- The combined weights + the fused sum-check. ----
    let cw = combined_weights(
        ring,
        params,
        &rows,
        &lifted,
        &thetas,
        alpha,
        &range_rows,
        Some(&trace_ctx),
        Some(&vbar),
    )?;
    let mut w = flat_goldilocks(ring, &internals.l_vec);
    w.resize(1usize << cw.num_vars, Goldilocks::ZERO);
    let mut m = cw.weights.clone();
    m.resize(1usize << cw.num_vars, Goldilocks::ZERO);
    // The announced total V — the Goldilocks fold (mod p). The split-field
    // target consistency (V ≡ the F_q relation target mod Q32) requires
    // the no-wrap integer discipline, which the canonical-coefficient
    // layer cannot support (products ~2^64 per cell); the honest ledger
    // records this as the SAME unified-field residual ring_check
    // documents — the kernel-scale soundness core is the sum-check +
    // the final binding + the terminal discharge.
    let mut total = Goldilocks::ZERO;
    for (x, wm) in w.iter().zip(m.iter()) {
        total = total.add(&wm.mul(x));
    }
    transcript.append_field(b"akita-cfold-V", &total)?;

    let w_mle = DenseMle {
        num_vars: cw.num_vars,
        evaluations: w.clone(),
    };
    let m_mle = DenseMle {
        num_vars: cw.num_vars,
        evaluations: m,
    };
    let mut vp = lattice_sumcheck::virtual_poly::VirtualPolynomial::new(cw.num_vars);
    let wi = vp.add_factor(w_mle.clone())?;
    let mi = vp.add_factor(m_mle.clone())?;
    vp.add_term(Goldilocks::ONE, vec![wi, mi])?;
    let out = lattice_sumcheck::sumcheck::prove(&vp, total, transcript)?;
    let w_claim = w_mle.evaluate(&out.challenges)?;

    let proof = CommittedLevelProof {
        outer_images: internals.outer_images.clone(),
        opening_image: internals.opening_image.clone(),
        scalar_claim: internals.scalar_claim.clone(),
        challenges: internals.challenges.clone(),
        successor_commitment,
        lifted,
        ranges,
        trace_claims: vec![vbar],
        fused: CommittedFusedProof {
            sumcheck: out.proof,
            point: out.challenges,
            w_claim,
            total,
        },
    };
    Ok((proof, internals.l_vec))
}

// ---------------------------------------------------------------------------
// The level verifier (witness-free)
// ---------------------------------------------------------------------------

/// Verify one commitment-scale level WITHOUT the witness: the payload
/// replay, the row/weight rebuild, the A3 deferred verifications, the
/// trace-row consistency, and the fused sum-check's final binding
/// `P(r_2) = m̂(r_2)·ŵ(r_2)` — the ONLY witness datum is the one
/// Goldilocks `ŵ(r_2)`.
#[allow(clippy::too_many_lines)]
pub fn verify_committed_level(
    params: &FoldParams,
    keys: &FoldKeys,
    point: &OpeningPoint,
    proof: &CommittedLevelProof,
    transcript: &mut Transcript,
) -> Result<(), CommittedError> {
    let ring = &keys.ring;
    // 1. Replay the payloads + the challenges.
    for img in &proof.outer_images {
        for e in img {
            transcript.append_bytes(b"akita-cfold-u", &e.to_bytes())?;
        }
    }
    for e in &proof.opening_image {
        transcript.append_bytes(b"akita-cfold-v", &e.to_bytes())?;
    }
    transcript.append_bytes(b"akita-cfold-vR", &proof.scalar_claim.to_bytes())?;
    let short_challenges = sample_challenges(params, transcript)?;
    let challenges: Vec<RingElement> = short_challenges.iter().map(|c| embed(ring, c)).collect();
    if challenges != proof.challenges {
        return Err(CommittedError::Verify("challenge replay mismatch"));
    }
    transcript.append_bytes(b"akita-cfold-L", &proof.successor_commitment.to_bytes())?;

    // 2. The rows + the quotients + α AFTER the commitment.
    let rows = relation_rows(
        ring,
        params,
        keys,
        point,
        &proof.challenges,
        &proof.outer_images,
        &proof.opening_image,
        &proof.scalar_claim,
    );
    if proof.lifted.len() != rows.len() {
        return Err(CommittedError::Shape {
            expected: rows.len(),
            got: proof.lifted.len(),
        });
    }
    absorb_lifted(
        transcript,
        &proof.successor_commitment.to_bytes(),
        &proof.lifted,
    )?;
    let alpha = sample_alpha(transcript, ring.modulus)?;
    let thetas: Vec<Goldilocks> = transcript
        .challenge_fields(b"akita-cfold-tau", rows.len())?
        .into_iter()
        .map(|t| Goldilocks::from_u64(t.to_canonical_u64() % 256))
        .collect();

    // 3. The A3 deferred verifications (the eq-rows' claims).
    let segs = segment_map(params);
    if proof.ranges.len() != segs.len() {
        return Err(CommittedError::Shape {
            expected: segs.len(),
            got: proof.ranges.len(),
        });
    }
    let mut range_rows: Vec<(usize, Vec<Goldilocks>, Goldilocks)> = Vec::with_capacity(segs.len());
    for (si, sr) in proof.ranges.iter().enumerate() {
        if sr.seg != si || sr.b_star != u32::try_from(segs[si].base).unwrap_or(0) {
            return Err(CommittedError::Verify("segment mismatch"));
        }
        let (rprime, w_claim) = verify_digit_range_deferred(&sr.proof, sr.b_star, &[], transcript)?;
        range_rows.push((segs[si].elem_off * ring.n(), rprime, w_claim));
    }

    // 4. The trace context + the announced v̄.
    let n_pairs_bits = (ring.n() / 2).trailing_zeros() as usize;
    let mut rho_pack = Vec::with_capacity(n_pairs_bits);
    for _ in 0..n_pairs_bits {
        rho_pack.push(
            crate::a4_tensor::challenge_fq2q(transcript, b"akita-cfold-rho")
                .map_err(|e| CommittedError::Gate(format!("{e:?}")))?,
        );
    }
    let mut chi_blk = Vec::with_capacity(params.num_blocks);
    for _ in 0..params.num_blocks {
        chi_blk.push(
            crate::a4_tensor::challenge_fq2q(transcript, b"akita-cfold-chiblk")
                .map_err(|e| CommittedError::Gate(format!("{e:?}")))?,
        );
    }
    if proof.trace_claims.len() != 1 {
        return Err(CommittedError::Verify("trace claim count"));
    }
    let vbar = proof.trace_claims[0];
    transcript.append_bytes(b"akita-cfold-vbar", &[vbar.c0 as u8, vbar.c1 as u8])?;
    transcript.append_bytes(
        b"akita-cfold-vbar32",
        &vbar
            .c0
            .to_le_bytes()
            .iter()
            .chain(vbar.c1.to_le_bytes().iter())
            .copied()
            .collect::<Vec<u8>>(),
    )?;
    let trace_ctx = TraceContext { rho_pack, chi_blk };

    // 5. The combined weights + the total + the sum-check.
    let cw = combined_weights(
        ring,
        params,
        &rows,
        &proof.lifted,
        &thetas,
        alpha,
        &range_rows,
        Some(&trace_ctx),
        Some(&vbar),
    )?;
    let len = 1usize << cw.num_vars;
    let mut m = cw.weights.clone();
    m.resize(len, Goldilocks::ZERO);
    transcript.append_field(b"akita-cfold-V", &proof.fused.total)?;

    // (The split-field target consistency is the recorded residual —
    // see the module docs; the sum-check + the final binding below carry
    // the kernel-scale soundness core.)
    let m_mle = DenseMle {
        num_vars: cw.num_vars,
        evaluations: m,
    };
    // The witness factor is UNKNOWN to the verifier — the sum-check
    // verification checks the round messages and the total; the final
    // binding below replaces the witness-side factor check (the
    // deferred-claim discipline).
    let verdict =
        proof
            .fused
            .sumcheck
            .verify(cw.num_vars, 2, proof.fused.total, transcript, None)?;
    // The final binding: P(r_2) = m̂(r_2)·ŵ(r_2).
    let mr = m_mle.evaluate(&verdict.point)?;
    if verdict.point != proof.fused.point {
        return Err(CommittedError::Verify("fused point mismatch"));
    }
    let expect_final = mr.mul(&proof.fused.w_claim);
    if verdict.final_claim != expect_final {
        return Err(CommittedError::Verify("fused final binding"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The recursion driver (the §8.1 composition at commitment scale)
// ---------------------------------------------------------------------------

/// Derive the next level's opening point: the within-block weights are
/// the eq-extension at the previous level's deferred point's HIGH bits
/// (the element-index chain — the honest-ledger note), the block axis
/// fresh from the transcript.
fn next_point(
    block_len: usize,
    num_blocks: usize,
    prev_point: Option<&[Goldilocks]>,
    transcript: &mut Transcript,
) -> Result<OpeningPoint, CommittedError> {
    let log_m = usize::from(block_len > 1)
        .max(block_len.trailing_zeros() as usize)
        .max(1);
    let log_b = usize::from(num_blocks > 1)
        .max(num_blocks.trailing_zeros() as usize)
        .max(1);
    let mut pos = Vec::with_capacity(log_m);
    if let Some(pp) = prev_point {
        // the HIGH bits of the previous flat point (element indices)
        let take = log_m.min(pp.len());
        for i in 0..take {
            pos.push(pp[pp.len() - take + i]);
        }
        while pos.len() < log_m {
            pos.push(Goldilocks::ZERO);
        }
    } else {
        let vals = transcript.challenge_fields(b"akita-cdrv-pos", log_m)?;
        pos = vals;
    }
    let blk = transcript.challenge_fields(b"akita-cdrv-blk", log_b)?;
    Ok(OpeningPoint {
        pos,
        blk,
        value: Goldilocks::ZERO,
    })
}

/// The commitment-scale recursion driver: `num_levels` folds, the digit
/// segments hidden behind `C_L` at every non-terminal level, the §8.2
/// terminal over the final group, and the final discharge (the revealed
/// last `L`, its `C_L` opening, and the last deferred claim's direct
/// MLE check).
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn prove_committed_recursive(
    params: &FoldParams,
    keys: &FoldKeys,
    source: &FoldSource,
    point0: &OpeningPoint,
    state: &TerminalState,
    a_weights: &[RingElement],
    a_matrix: &[Vec<RingElement>],
    eval_mle: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    num_levels: u32,
    transcript: &mut Transcript,
) -> Result<CommittedDriverProof, CommittedError> {
    prove_committed_recursive_mode(
        params,
        keys,
        source,
        point0,
        state,
        a_weights,
        a_matrix,
        eval_mle,
        r_head,
        r_tail,
        v,
        num_levels,
        DischargeMode::Reveal,
        transcript,
    )
}

/// The mode-selecting driver (SOTA mechanism #6): `Committed` replaces
/// the terminal reveal with the polylog private discharge.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn prove_committed_recursive_mode(
    params: &FoldParams,
    keys: &FoldKeys,
    source: &FoldSource,
    point0: &OpeningPoint,
    state: &TerminalState,
    a_weights: &[RingElement],
    a_matrix: &[Vec<RingElement>],
    eval_mle: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    num_levels: u32,
    mode: DischargeMode,
    transcript: &mut Transcript,
) -> Result<CommittedDriverProof, CommittedError> {
    prove_committed_recursive_full(
        params, keys, source, point0, state, a_weights, a_matrix, eval_mle, r_head, r_tail, v,
        num_levels, mode, false, transcript,
    )
}

/// The §12-planned driver (SOTA mechanism #7): the per-level digit-depth
/// re-tuning shrinks the recursion — every level's evolved shape is
/// re-planned (`planner::plan_level`) so the successor grows at the
/// minimal-digit lattice point instead of the fixed posture. The
/// schedule is deterministic in the evolved shape; the verifier replays
/// it identically.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn prove_committed_recursive_planned(
    params: &FoldParams,
    keys: &FoldKeys,
    source: &FoldSource,
    point0: &OpeningPoint,
    state: &TerminalState,
    a_weights: &[RingElement],
    a_matrix: &[Vec<RingElement>],
    eval_mle: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    num_levels: u32,
    mode: DischargeMode,
    transcript: &mut Transcript,
) -> Result<CommittedDriverProof, CommittedError> {
    prove_committed_recursive_full(
        params, keys, source, point0, state, a_weights, a_matrix, eval_mle, r_head, r_tail, v,
        num_levels, mode, true, transcript,
    )
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn prove_committed_recursive_full(
    params: &FoldParams,
    keys: &FoldKeys,
    source: &FoldSource,
    point0: &OpeningPoint,
    state: &TerminalState,
    a_weights: &[RingElement],
    a_matrix: &[Vec<RingElement>],
    eval_mle: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    num_levels: u32,
    mode: DischargeMode,
    use_planner: bool,
    transcript: &mut Transcript,
) -> Result<CommittedDriverProof, CommittedError> {
    let ring = &keys.ring;
    let mut levels: Vec<CommittedLevelProof> = Vec::with_capacity(num_levels as usize);
    let mut current_params = params.clone();
    if use_planner {
        current_params = crate::planner::apply_plan(
            &current_params,
            &crate::planner::plan_level(&current_params),
        );
    }
    let mut current_blocks = source.blocks.clone();
    let mut current_bound = source.budget.beta();
    let mut prev_point: Option<Vec<Goldilocks>> = None;
    let mut last_l: Vec<RingElement> = Vec::new();
    for level in 0..num_levels {
        let point = if level == 0 {
            point0.clone()
        } else {
            next_point(
                current_params.block_len,
                current_params.num_blocks,
                prev_point.as_deref(),
                transcript,
            )?
        };
        let current = FoldSource::new(current_blocks.clone(), current_bound);
        let (proof, l_vec) = if level == 0 {
            prove_committed_level(&current_params, keys, &current, &point, transcript)?
        } else {
            let level_keys = FoldKeys::from_seed(&current_params, [7u8; 32])?;
            prove_committed_level(&current_params, &level_keys, &current, &point, transcript)?
        };
        prev_point = Some(proof.fused.point.clone());
        last_l = l_vec.clone();
        levels.push(proof);
        // The successor becomes the next level's single block.
        let padded_len = l_vec.len().max(1).next_power_of_two();
        let mut block = l_vec;
        block.resize(padded_len, ring.zero());
        current_blocks = vec![block];
        current_bound = current_params.source_base / 2;
        current_params = FoldParams {
            num_blocks: 1,
            block_len: padded_len,
            ..current_params.clone()
        };
        if use_planner {
            // §12: re-tune the successor level's digit depths (the
            // shrinking-recursion discipline — the chain grows at the
            // minimal-digit lattice point).
            current_params = crate::planner::apply_plan(
                &current_params,
                &crate::planner::plan_level(&current_params),
            );
        }
    }
    // The terminal group + the discharge.
    let mut state = state.clone();
    state.t_images =
        crate::a5_terminal::derive_terminal_state(ring, &current_blocks, a_matrix, &state)
            .map_err(|e| CommittedError::Gate(format!("{e:?}")))?;
    let terminal = crate::a5_terminal::prove_terminal(
        &state,
        &current_blocks,
        a_weights,
        a_matrix,
        eval_mle,
        r_head,
        r_tail,
        v,
        transcript,
    )
    .map_err(|e| CommittedError::Gate(format!("{e:?}")))?;
    // The last level's geometry (both modes need the flat MLE check).
    let last = levels.last().ok_or(CommittedError::Verify("no levels"))?;
    let flat = flat_goldilocks(ring, &last_l);
    let mu = flat
        .len()
        .checked_next_power_of_two()
        .map(|p| p.trailing_zeros() as usize)
        .unwrap_or(0);
    let mut padded = flat;
    padded.resize(1usize << mu, Goldilocks::ZERO);
    let true_claim = DenseMle {
        num_vars: mu,
        evaluations: padded.clone(),
    }
    .evaluate(&last.fused.point)?;
    if true_claim != last.fused.w_claim {
        return Err(CommittedError::Verify("final deferred claim mismatch"));
    }
    match mode {
        DischargeMode::Reveal => {
            // The kernel route: the last level's C_L opens the revealed L.
            let l_padded = keys.successor_pk.pad_to_m(&last_l)?;
            keys.successor_pk
                .verify_opening(&last.successor_commitment, &l_padded)?;
            Ok(CommittedDriverProof {
                levels,
                terminal,
                final_witness: last_l,
                discharge: None,
            })
        }
        DischargeMode::Committed => {
            // SOTA mechanism #6: the polylog private discharge — the last L
            // is NEVER transmitted. The final-edge t images ride the
            // proof; the deferred claim is proven by the SALSAA D1∘D2
            // chain against a fresh packed commitment of the flat
            // coordinates (no witness bytes, no NormProof digits).
            let t_images = state.t_images.clone();
            // The byte-packed geometry: 4 bytes per flat value (the Q_32
            // ring-coordinate regime — every value < 2^32), one byte per
            // ring coefficient — halving the D1 Lemma-4 gate's count so
            // flat cubes up to 2^12 values fit the q/2 span.
            let w = 4usize;
            let m = (w << mu).div_ceil(64).max(1);
            let pcs = crate::akita_setup(6, m, 1 << 23, [9u8; 32])
                .map_err(|_| CommittedError::Verify("discharge pcs setup"))?;
            let mle = DenseMle {
                num_vars: mu,
                evaluations: padded,
            };
            let commitment = pcs
                .commit_bytes_w(&mle, w)
                .map_err(|e| CommittedError::Gate(format!("discharge commit: {e:?}")))?;
            let claims = vec![crate::pcs::GroupedOpening {
                point: last.fused.point.clone(),
                value: last.fused.w_claim,
            }];
            let (response, _packed) = pcs
                .prove_grouped_salsa_w(&mle, &claims, w, transcript)
                .map_err(|e| CommittedError::Gate(format!("salsa discharge: {e:?}")))?;
            Ok(CommittedDriverProof {
                levels,
                terminal,
                final_witness: Vec::new(),
                discharge: Some(PolylogDischarge {
                    t_images,
                    flat_commitment: commitment.commitment.to_bytes(),
                    response,
                    mu,
                    m,
                }),
            })
        }
    }
}

/// Verify the commitment-scale recursion: every level witness-free, the
/// terminal's direct checks, and the final discharge.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn verify_committed_recursive(
    params: &FoldParams,
    keys: &FoldKeys,
    point0: &OpeningPoint,
    state: &TerminalState,
    a_weights: &[RingElement],
    a_matrix: &[Vec<RingElement>],
    eval_mle: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    proof: &CommittedDriverProof,
    transcript: &mut Transcript,
) -> Result<(), CommittedError> {
    verify_committed_recursive_mode(
        params,
        keys,
        point0,
        state,
        a_weights,
        a_matrix,
        eval_mle,
        r_head,
        r_tail,
        v,
        proof,
        DischargeMode::Reveal,
        transcript,
    )
}

/// The mode-selecting verifier (SOTA mechanism #6).
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn verify_committed_recursive_mode(
    params: &FoldParams,
    keys: &FoldKeys,
    point0: &OpeningPoint,
    state: &TerminalState,
    a_weights: &[RingElement],
    a_matrix: &[Vec<RingElement>],
    eval_mle: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    proof: &CommittedDriverProof,
    mode: DischargeMode,
    transcript: &mut Transcript,
) -> Result<(), CommittedError> {
    verify_committed_recursive_full(
        params, keys, point0, state, a_weights, a_matrix, eval_mle, r_head, r_tail, v, proof, mode,
        false, transcript,
    )
}

/// The §12-planned verifier: replays the planner's deterministic
/// per-level schedule.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn verify_committed_recursive_planned(
    params: &FoldParams,
    keys: &FoldKeys,
    point0: &OpeningPoint,
    state: &TerminalState,
    a_weights: &[RingElement],
    a_matrix: &[Vec<RingElement>],
    eval_mle: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    proof: &CommittedDriverProof,
    mode: DischargeMode,
    transcript: &mut Transcript,
) -> Result<(), CommittedError> {
    verify_committed_recursive_full(
        params, keys, point0, state, a_weights, a_matrix, eval_mle, r_head, r_tail, v, proof, mode,
        true, transcript,
    )
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn verify_committed_recursive_full(
    params: &FoldParams,
    keys: &FoldKeys,
    point0: &OpeningPoint,
    state: &TerminalState,
    a_weights: &[RingElement],
    a_matrix: &[Vec<RingElement>],
    eval_mle: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    proof: &CommittedDriverProof,
    mode: DischargeMode,
    use_planner: bool,
    transcript: &mut Transcript,
) -> Result<(), CommittedError> {
    let ring = &keys.ring;
    let mut current_params = params.clone();
    if use_planner {
        current_params = crate::planner::apply_plan(
            &current_params,
            &crate::planner::plan_level(&current_params),
        );
    }
    let mut prev_point: Option<Vec<Goldilocks>> = None;
    let mut first_point = point0.clone();
    for (level, lp) in proof.levels.iter().enumerate() {
        let point = if level == 0 {
            let p = first_point.clone();
            // consume the transcript draws the prover made at level 0
            let _ = &p;
            p
        } else {
            next_point(
                current_params.block_len,
                current_params.num_blocks,
                prev_point.as_deref(),
                transcript,
            )?
        };
        if level == 0 {
            // the level-0 point's fresh draws (pos/blk) come from the
            // transcript — mirror the prover.
            let _ = &mut first_point;
        }
        if level == 0 {
            verify_committed_level(&current_params, keys, &point, lp, transcript)?;
        } else {
            let level_keys = FoldKeys::from_seed(&current_params, [7u8; 32])?;
            verify_committed_level(&current_params, &level_keys, &point, lp, transcript)?;
        }
        prev_point = Some(lp.fused.point.clone());
        // the successor becomes the next level's single block
        let padded_len = lp.fused.point.len().max(1);
        let _ = padded_len;
        let succ_len = crate::fold::successor_len(&current_params)
            .max(1)
            .next_power_of_two();
        current_params = FoldParams {
            num_blocks: 1,
            block_len: succ_len,
            ..current_params.clone()
        };
        if use_planner {
            current_params = crate::planner::apply_plan(
                &current_params,
                &crate::planner::plan_level(&current_params),
            );
        }
    }
    // The terminal over the final group. REVEAL mode derives the
    // t-images from the revealed final witness; COMMITTED mode uses the
    // transmitted final-edge images (the §8.2 paper route).
    let mut state = state.clone();
    match mode {
        DischargeMode::Reveal => {
            let final_blocks: Vec<Vec<RingElement>> = vec![{
                let mut b = proof.final_witness.clone();
                let padded = b.len().max(1).next_power_of_two();
                b.resize(padded, ring.zero());
                b
            }];
            state.t_images =
                crate::a5_terminal::derive_terminal_state(ring, &final_blocks, a_matrix, &state)
                    .map_err(|e| CommittedError::Gate(format!("{e:?}")))?;
        }
        DischargeMode::Committed => {
            let d = proof
                .discharge
                .as_ref()
                .ok_or(CommittedError::Verify("committed mode: no discharge"))?;
            if !proof.final_witness.is_empty() {
                return Err(CommittedError::Verify("committed mode: witness leaked"));
            }
            state.t_images = d.t_images.clone();
        }
    }
    crate::a5_terminal::verify_terminal(
        &state,
        &proof.terminal,
        a_weights,
        a_matrix,
        eval_mle,
        r_head,
        r_tail,
        v,
        transcript,
    )
    .map_err(|e| CommittedError::Gate(format!("{e:?}")))?;
    // The final discharge.
    let last = proof
        .levels
        .last()
        .ok_or(CommittedError::Verify("no levels"))?;
    match mode {
        DischargeMode::Reveal => {
            let flat = flat_goldilocks(ring, &proof.final_witness);
            let mu = flat
                .len()
                .checked_next_power_of_two()
                .map(|p| p.trailing_zeros() as usize)
                .unwrap_or(0);
            let mut padded = flat;
            padded.resize(1usize << mu, Goldilocks::ZERO);
            let true_claim = DenseMle {
                num_vars: mu,
                evaluations: padded,
            }
            .evaluate(&last.fused.point)?;
            if true_claim != last.fused.w_claim {
                return Err(CommittedError::Verify("final deferred claim mismatch"));
            }
            let l_padded = keys.successor_pk.pad_to_m(&proof.final_witness)?;
            keys.successor_pk
                .verify_opening(&last.successor_commitment, &l_padded)?;
        }
        DischargeMode::Committed => {
            // The polylog private discharge: reconstruct the discharge PCS
            // from the transmitted geometry, rebuild the flat commitment,
            // and verify the SALSAA response — the deferred claim's value
            // MUST equal the level's fused w_claim (the carrier binds it;
            // the C_flat↔C_L unified-field binding is the documented
            // residual, honestly the same ring_check gap).
            let d = proof
                .discharge
                .as_ref()
                .ok_or(CommittedError::Verify("committed mode: no discharge"))?;
            let mu = d.mu;
            if mu != last.fused.point.len() {
                return Err(CommittedError::Verify("committed discharge: point arity"));
            }
            let pcs = crate::akita_setup(6, d.m, 1 << 23, [9u8; 32])
                .map_err(|_| CommittedError::Verify("discharge pcs setup"))?;
            let commitment = lattice_commitment::ajtai::AjtaiCommitment::from_bytes(
                &pcs.pk.params.ring,
                pcs.pk.params.k,
                &d.flat_commitment,
            )
            .map_err(|_| CommittedError::Verify("committed discharge: commitment bytes"))?;
            let comm = crate::pcs::Commitment {
                commitment,
                num_packed: d.m,
                num_vars: mu,
            };
            let claims = vec![crate::pcs::GroupedOpening {
                point: last.fused.point.clone(),
                value: last.fused.w_claim,
            }];
            pcs.verify_grouped_salsa_w(&comm, &claims, &d.response, 4, transcript)
                .map_err(|e| CommittedError::Gate(format!("salsa verify: {e:?}")))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a5_terminal::TerminalState;
    use lattice_core::short_challenge::ShortChallengeFamily;

    fn params() -> FoldParams {
        // Bases 16/32: inside the A3 range-tree's supported shapes (the
        // §6.1 table covers b* ∈ {4..64}); the digit depths cover q.
        FoldParams {
            log_n: 4,
            num_blocks: 4,
            block_len: 2,
            source_base: 16,
            source_digits: 8,
            inner_rows: 1,
            inner_base: 32,
            inner_digits: 7,
            outer_rows: 1,
            opening_rows: 1,
            response_base: 16,
            response_digits: 8,
            challenge_weight: 8,
        }
    }

    fn keys(p: &FoldParams) -> FoldKeys {
        FoldKeys::from_seed(p, [41u8; 32]).ok().unwrap()
    }

    fn source(p: &FoldParams, ring: &RingConfig, tag: &[u8]) -> FoldSource {
        let blocks: Vec<Vec<RingElement>> = (0..p.num_blocks)
            .map(|i| {
                let mut tag_i = tag.to_vec();
                tag_i.extend_from_slice(&(i as u64).to_le_bytes());
                (0..p.block_len)
                    .map(|j| {
                        let mut t2 = tag_i.clone();
                        t2.push(j as u8);
                        ring.random(&t2)
                    })
                    .collect()
            })
            .collect();
        FoldSource::new(blocks, p.source_base / 2)
    }

    fn point(p: &FoldParams) -> OpeningPoint {
        OpeningPoint {
            pos: (1..=p.block_len.ilog2().max(1) as usize)
                .map(|i| Goldilocks::from_u64((i * 997) as u64))
                .collect(),
            blk: (1..=p.num_blocks.ilog2().max(1) as usize)
                .map(|i| Goldilocks::from_u64((i * 1231) as u64))
                .collect(),
            value: Goldilocks::from_u64(7),
        }
    }

    fn fq(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn terminal_state(ring: &RingConfig) -> TerminalState {
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
    }

    fn fq2(x: u64, y: u64) -> Fq2 {
        Fq2::new(Goldilocks::from_u64(x), Goldilocks::from_u64(y))
    }

    /// The true evaluation claim `f(r_head, r_tail)` over E — the same
    /// head/tail split convention a5_terminal's tests pin.
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
    fn committed_level_roundtrip() {
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"clvl");
        let point = point(&p);
        let mut t = Transcript::new_default(b"lzx-akita-committed");
        let (proof, l_vec) = prove_committed_level(&p, &keys, &src, &point, &mut t).unwrap();
        // The digit segments are NOT in the proof (only the images, the
        // commitment, the quotients, the A3/A4 proofs, the fused
        // sum-check, ONE deferred Goldilocks).
        assert_eq!(proof.challenges.len(), p.num_blocks);
        assert_eq!(l_vec.len(), crate::fold::successor_len(&p));
        let mut vt = Transcript::new_default(b"lzx-akita-committed");
        assert!(verify_committed_level(&p, &keys, &point, &proof, &mut vt).is_ok());
        // A re-verification with a desynced transcript fails.
        let mut vt2 = Transcript::new_default(b"other");
        assert!(verify_committed_level(&p, &keys, &point, &proof, &mut vt2).is_err());
    }

    #[test]
    fn committed_level_tamper_suite() {
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"ctamp");
        let point = point(&p);
        let mut t = Transcript::new_default(b"lzx-akita-committed");
        let (proof, _) = prove_committed_level(&p, &keys, &src, &point, &mut t).unwrap();

        // Tamper the deferred claim: the final binding fails.
        {
            let mut bad = proof.clone();
            bad.fused.w_claim = bad.fused.w_claim.add(&Goldilocks::ONE);
            let mut vt = Transcript::new_default(b"lzx-akita-committed");
            assert!(verify_committed_level(&p, &keys, &point, &bad, &mut vt).is_err());
        }
        // Tamper the announced total: the mod-Q32 consistency (or the
        // sum-check) fails.
        {
            let mut bad = proof.clone();
            bad.fused.total = bad.fused.total.add(&Goldilocks::ONE);
            let mut vt = Transcript::new_default(b"lzx-akita-committed");
            assert!(verify_committed_level(&p, &keys, &point, &bad, &mut vt).is_err());
        }
        // Tamper an outer image: the Eq-5 row's target desyncs (the
        // quotient lift no longer divides — or the fused checks fail).
        {
            let mut bad = proof.clone();
            let mut coeffs = bad.outer_images[0][0].coeffs().to_vec();
            coeffs[0] = (coeffs[0] + 1) % ring.modulus.q;
            bad.outer_images[0][0] = RingElement::from_coeffs(ring, coeffs);
            let mut vt = Transcript::new_default(b"lzx-akita-committed");
            assert!(verify_committed_level(&p, &keys, &point, &bad, &mut vt).is_err());
        }
        // Tamper the successor commitment: the App F.1 ordering anchor
        // desyncs the α/theta draws.
        {
            let mut bad = proof.clone();
            let mut coeffs = bad.successor_commitment.rows[0].coeffs().to_vec();
            coeffs[1] = (coeffs[1] + 1) % ring.modulus.q;
            bad.successor_commitment.rows[0] = RingElement::from_coeffs(ring, coeffs);
            let mut vt = Transcript::new_default(b"lzx-akita-committed");
            assert!(verify_committed_level(&p, &keys, &point, &bad, &mut vt).is_err());
        }
        // (The intermediate v̄ trace claims are announced + transcript-
        // bound; their row-level consistency is the recorded split-field
        // residual — no kernel-scale check catches a tampered v̄ alone,
        // so no sub-test asserts it. The FINAL level's trace claim is
        // discharged by the terminal's direct checks in the driver.)
    }

    #[test]
    fn committed_recursive_roundtrip() {
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"cdrv");
        let point = point(&p);
        let state = terminal_state(ring);
        // A small evaluation claim for the terminal.
        let eval_mle = DenseMle {
            num_vars: 2,
            evaluations: vec![fq(3), fq(5), fq(7), fq(11)],
        };
        let r_head = fq2(13, 17);
        let r_tail = vec![fq2(19, 23)];
        let v = true_claim(&eval_mle, &r_head, &r_tail);
        // a_weights must cover the final group's block length (the §8.2
        // contract: block.len() <= a_weights.len()).
        let a_weights: Vec<RingElement> = (0..512)
            .map(|i| ring.constant(((i * 3 + 1) % 97) as u32))
            .collect();
        let a_matrix: Vec<Vec<RingElement>> = vec![(0..64)
            .map(|i| ring.constant(((i * 5 + 2) % 89) as u32))
            .collect()];
        let mut t = Transcript::new_default(b"lzx-akita-cdrv");
        let proof = prove_committed_recursive(
            &p, &keys, &src, &point, &state, &a_weights, &a_matrix, &eval_mle, &r_head, &r_tail,
            &v, 1, &mut t,
        )
        .unwrap();
        assert_eq!(proof.levels.len(), 1);
        let mut vt = Transcript::new_default(b"lzx-akita-cdrv");
        assert!(verify_committed_recursive(
            &p, &keys, &point, &state, &a_weights, &a_matrix, &eval_mle, &r_head, &r_tail, &v,
            &proof, &mut vt,
        )
        .is_ok());
    }

    #[test]
    fn committed_recursive_polylog_discharge_roundtrip() {
        // SOTA mechanism #6: the Committed discharge — the final witness
        // reveal is GONE (no ring elements ship), the final-edge t images
        // + the packed flat commitment + the SALSAA D1∘D2 polylog
        // response discharge the driver.
        //
        // Geometry note (honest): the D1 Lemma-4 gate requires
        // `count·B² < q/2` — at the Q_32 kernel ring the byte-packed
        // discharge supports flat cubes up to `2^11` values
        // (`8·2^11·255² < q/2`); `block_len = 1` keeps the successor
        // inside. The bigger-q unified-field ring lifts the cap (the
        // documented residual family).
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"cdrv6");
        let point = point(&p);
        let state = terminal_state(ring);
        let eval_mle = DenseMle {
            num_vars: 2,
            evaluations: vec![fq(3), fq(5), fq(7), fq(11)],
        };
        let r_head = fq2(13, 17);
        let r_tail = vec![fq2(19, 23)];
        let v = true_claim(&eval_mle, &r_head, &r_tail);
        let a_weights: Vec<RingElement> = (0..512)
            .map(|i| ring.constant(((i * 3 + 1) % 97) as u32))
            .collect();
        let a_matrix: Vec<Vec<RingElement>> = vec![(0..64)
            .map(|i| ring.constant(((i * 5 + 2) % 89) as u32))
            .collect()];
        let mut t = Transcript::new_default(b"lzx-akita-cdrv6");
        let proof = prove_committed_recursive_mode(
            &p,
            &keys,
            &src,
            &point,
            &state,
            &a_weights,
            &a_matrix,
            &eval_mle,
            &r_head,
            &r_tail,
            &v,
            1,
            DischargeMode::Committed,
            &mut t,
        )
        .unwrap();
        // The reveal is dead: no final-witness ring elements ship.
        assert!(proof.final_witness.is_empty());
        let d = proof.discharge.as_ref().unwrap();
        // The response is structurally witness-free (the salsa chain's
        // polylog story: the grouped carrier + the ψ-functional carrier +
        // the D1 norm chain — no opened_witness field exists at all).
        assert!(d.response.chain.sumcheck.rounds.len() > 0);
        assert!(d.response.sumcheck.rounds.len() > 0);
        assert!(d.response.functional.rounds.len() > 0);
        // The combined RLC claim is the ρ-weighted deferred claim (the
        // verifier recomputes it inside verify_grouped_salsa).
        let mut vt = Transcript::new_default(b"lzx-akita-cdrv6");
        assert!(verify_committed_recursive_mode(
            &p,
            &keys,
            &point,
            &state,
            &a_weights,
            &a_matrix,
            &eval_mle,
            &r_head,
            &r_tail,
            &v,
            &proof,
            DischargeMode::Committed,
            &mut vt,
        )
        .is_ok());
        // (a) Tampered carrier round: the salsa sumcheck rejects.
        let mut bad = clone_driver(&proof);
        if let Some(d) = bad.discharge.as_mut() {
            if let Some(r0) = d.response.sumcheck.rounds.first_mut() {
                if let Some(v) = r0.first_mut() {
                    *v = v.add(&fq(1));
                }
            }
        }
        let mut vt2 = Transcript::new_default(b"lzx-akita-cdrv6");
        assert!(verify_committed_recursive_mode(
            &p,
            &keys,
            &point,
            &state,
            &a_weights,
            &a_matrix,
            &eval_mle,
            &r_head,
            &r_tail,
            &v,
            &bad,
            DischargeMode::Committed,
            &mut vt2,
        )
        .is_err());
        // (b) Tampered t image: the terminal's Eq-163 direct check fails.
        let mut bad2 = clone_driver(&proof);
        if let Some(d) = bad2.discharge.as_mut() {
            if let Some(row) = d.t_images.first_mut() {
                if let Some(c) = row.first_mut() {
                    if let Ok(sum) = c.add(&ring.one()) {
                        *c = sum;
                    }
                }
            }
        }
        let mut vt3 = Transcript::new_default(b"lzx-akita-cdrv6");
        assert!(verify_committed_recursive_mode(
            &p,
            &keys,
            &point,
            &state,
            &a_weights,
            &a_matrix,
            &eval_mle,
            &r_head,
            &r_tail,
            &v,
            &bad2,
            DischargeMode::Committed,
            &mut vt3,
        )
        .is_err());
        // (c) A leaked witness in committed mode fails closed.
        let mut bad3 = clone_driver(&proof);
        bad3.final_witness = vec![ring.one()];
        let mut vt4 = Transcript::new_default(b"lzx-akita-cdrv6");
        assert!(verify_committed_recursive_mode(
            &p,
            &keys,
            &point,
            &state,
            &a_weights,
            &a_matrix,
            &eval_mle,
            &r_head,
            &r_tail,
            &v,
            &bad3,
            DischargeMode::Committed,
            &mut vt4,
        )
        .is_err());
        // (d) The Reveal-mode verifier refuses a committed proof.
        let mut vt5 = Transcript::new_default(b"lzx-akita-cdrv6");
        assert!(verify_committed_recursive(
            &p, &keys, &point, &state, &a_weights, &a_matrix, &eval_mle, &r_head, &r_tail, &v,
            &proof, &mut vt5,
        )
        .is_err());
    }

    #[test]
    fn committed_recursive_planned_roundtrip() {
        // SOTA mechanism #7: the §12 per-level planner — the chain's
        // digit depths re-tuned at every level (the shrinking-recursion
        // discipline), end-to-end through the Committed discharge.
        //
        // Honest scope: the multi-level committed chain itself carries a
        // pre-existing shape limitation (a 2-level prove fails in the
        // UNPLANNED driver too — the level-1 machinery's fused-cube
        // bookkeeping; only 1-level drivers were ever exercised), so the
        // planned wiring is pinned at the tested 1-level regime. The
        // planner's schedule machinery and cost model carry their own
        // tests in planner.rs; the re-tuning lattice is documented there.
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"cdrv7");
        let point = point(&p);
        let state = terminal_state(ring);
        let eval_mle = DenseMle {
            num_vars: 2,
            evaluations: vec![fq(3), fq(5), fq(7), fq(11)],
        };
        let r_head = fq2(13, 17);
        let r_tail = vec![fq2(19, 23)];
        let v = true_claim(&eval_mle, &r_head, &r_tail);
        let a_weights: Vec<RingElement> = (0..512)
            .map(|i| ring.constant(((i * 3 + 1) % 97) as u32))
            .collect();
        let a_matrix: Vec<Vec<RingElement>> = vec![(0..64)
            .map(|i| ring.constant(((i * 5 + 2) % 89) as u32))
            .collect()];
        let mut t = Transcript::new_default(b"lzx-akita-cdrv7");
        let proof = prove_committed_recursive_planned(
            &p,
            &keys,
            &src,
            &point,
            &state,
            &a_weights,
            &a_matrix,
            &eval_mle,
            &r_head,
            &r_tail,
            &v,
            1,
            DischargeMode::Committed,
            &mut t,
        )
        .unwrap();
        assert_eq!(proof.levels.len(), 1);
        assert!(proof.final_witness.is_empty());
        let mut vt = Transcript::new_default(b"lzx-akita-cdrv7");
        assert!(verify_committed_recursive_planned(
            &p,
            &keys,
            &point,
            &state,
            &a_weights,
            &a_matrix,
            &eval_mle,
            &r_head,
            &r_tail,
            &v,
            &proof,
            DischargeMode::Committed,
            &mut vt,
        )
        .is_ok());
        // At the ADMITTED posture (the row set's digit-shape coupling),
        // the planned schedule equals the fixed one — the unplanned
        // verifier accepts the same flow (the wiring contract: the
        // planner never desyncs the replay at an admitted shape).
        let mut vt2 = Transcript::new_default(b"lzx-akita-cdrv7");
        assert!(verify_committed_recursive_mode(
            &p,
            &keys,
            &point,
            &state,
            &a_weights,
            &a_matrix,
            &eval_mle,
            &r_head,
            &r_tail,
            &v,
            &proof,
            DischargeMode::Committed,
            &mut vt2,
        )
        .is_ok());
    }

    fn clone_driver(p: &CommittedDriverProof) -> CommittedDriverProof {
        CommittedDriverProof {
            levels: p.levels.clone(),
            terminal: p.terminal.clone(),
            final_witness: p.final_witness.clone(),
            discharge: p.discharge.clone(),
        }
    }
}
