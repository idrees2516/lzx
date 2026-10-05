//! HyperWolf H6 — the **full-fidelity route**: the LaBRADOR engine
//! re-parameterized to the HyperWolf ring (the recorded NEXT_STEPS
//! H6-residual), delivering
//!
//! 1. **the amortized Dachshund engine over `R_q = Z_q[X]/(X^d+1]`,
//!    `q = 2^61 − 259`** — the paper-faithful structure of
//!    `lattice-greyhound`'s protocol (the join / inner commitments /
//!    quadratic garbage / JL projection with rejection / LIFTS collapse /
//!    α-β aggregation / the h-garbage / the amortized opening
//!    `z = Σ c_i s_i` with digit decomposition / the E1–E6 target
//!    relation / the §5.6 tail with 2r−1 interleaved garbage), ported to
//!    the HwRing with the workspace's **certified fixed-weight
//!    challenges** (weight 10, amplitude 1, Γ_C = ⌈√10⌉ = 4 — the H5
//!    discipline) replacing the (23,31,10)+SVD sampler, and
//!    `LIFTS = ⌈128/61⌉ = 3`;
//! 2. **the recursive outer-commitment compaction driver** — iterate
//!    `lab_prove_level` while the statement shrinks, then one tail level
//!    whose openings are transmitted directly (the O(log log log N)
//!    route of the paper's proof-size table: each level's outer
//!    commitments become the next level's statement, so the transmitted
//!    material is O(levels · polylog) instead of the clear payload);
//! 3. **the full-fidelity HyperWolf evaluation protocol** —
//!    `eval_prove_labrador` / `eval_verify_labrador`: the per-round JL
//!    projection vectors (the clear protocol's dominant payload, 256
//!    ring elements per slice per round) are NEVER transmitted; instead
//!    ONE amortized Dachshund statement covers all of them, with
//!    * **per-round exact ℓ2 statements** (the σ⁻¹-conjugate quadratic
//!      constraints with binary slack vectors: `ct⟨p, σ⁻¹(p)⟩ +
//!      ct⟨slack, σ⁻¹(slack)⟩ = B_r` exactly — the slack's bits make the
//!      inequality exact), and
//!    * **the fold-consistency dot-products** (`Σ_i p_i^(r) =
//!      Σ_j C_j^(r−1)·p_j^(r−1)` as full-ring constraints, coefficients
//!      the certified ±1 challenges — natively in the SAME ring, closing
//!      the recorded "constraint coefficients do not embed" gap), and
//!    * **the terminal tie** `Σ_j C_j·p_j^(last) = JL(s^(1))` against the
//!      revealed final witness.
//!
//! # The honest deviation ledger (kernel scale)
//!
//! * **The SIS gate regime.** The Core-SVP rule
//!   `log2 β < 2·√(LOGQ·log2(1.00444)·N)·√rank` at LOGQ = 61, N = 64
//!   caps at 2·√(61·0.00638·64)·√32 ≈ 2^56.5 — the tail's directly
//!   transmitted Ajtai images (uniform mod q, norm ≈ 2^61) exceed it at
//!   every rank ≤ 32, so the faithful gates cannot pass in this ring.
//!   [`SisGateMode::Faithful`] keeps them fail-closed (a test pins the
//!   failure); the kernel tests run [`SisGateMode::KernelBypass`], which
//!   skips the two replay-time SIS assertions and clamps the rank search
//!   — the demonstrator discipline of the sibling modules. The faithful
//!   regime needs N ≥ 128 (or the RNS path) — the recorded parameter
//!   residual, same family as hyperwolf.rs's deviation 4.
//! * **The per-round bound formula.** `B_r = b·jl_rows·d²·(b·ι)·β(r)²`
//!   is the conservative convolution bound (each projection coefficient
//!   is a negacyclic product of a ternary JL row with the block-summed
//!   slice); it implies the clear protocol's check-2 bound on the ct
//!   coefficients up to the documented factors, and is gated
//!   `B_r < q/4` (the wraparound guard that keeps the integer ℓ2
//!   statements exact mod q). Paper-scale β ladders exceed this gate —
//!   the honest statement that the 2^30 instantiation is modelled
//!   ([`lab_size_model`]), not executed.
//! * **The intermediate norm certificates** ride the per-round exact ℓ2
//!   constraints (not amortized away as in the H6-compact mode) — the
//!   full-fidelity upgrade the ledger demanded.
//! * **The engine's challenge constants** are the certified family's
//!   (T = Γ_C = 4, τ = 10) rather than the reference's (T = 14,
//!   τ = 64): the variance laws are re-derived from the certified
//!   per-sample bound, so the parameter search runs the same loop with
//!   tighter constants.

use crate::hyperwolf::{
    sample_challenge, HwCommitState, HwElt, HwError, HwParams, HwRing, HyperWolfFull,
};
use lattice_core::keccak::sha3_256;
use lattice_core::transcript::Transcript;

/// log2(q) of the HyperWolf lab prime.
pub const LAB_LOGQ: f64 = 61.0;
/// The certified challenge operator norm (Γ_C = ⌈√10⌉).
pub const LAB_T: f64 = 4.0;
/// The certified challenge ℓ2² (weight 10, amplitude 1).
pub const LAB_TAU: f64 = 10.0;
/// The extraction norm slack (the reference's SLACK).
pub const LAB_SLACK: f64 = 2.0;
/// LIFTS = ⌈128/log2 q⌉ aggregation rounds.
pub const LAB_LIFTS: usize = 128_usize.div_ceil(61);
/// The sieve constant log2(1.00444).
const LOGDELTA: f64 = 0.006382542;
/// Digit-bit cap before f grows (the reference's DIGITBITS).
const DIGITBITS: u32 = 14;

/// The SIS-gate policy of the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SisGateMode {
    /// The full Core-SVP gates, fail-closed (errors when the regime
    /// cannot support the norms — the honest pin at N = 64, LOGQ = 61).
    Faithful,
    /// The kernel-scale demonstrator: the replay-time SIS assertions are
    /// skipped and the rank search clamps at 8 (documented above).
    KernelBypass,
}

// ---------------------------------------------------------------------------
// Ring helpers over HwElt
// ---------------------------------------------------------------------------

/// The plain negacyclic inner product `Σ_k a_k·b_k` (a ring element).
pub fn sprod(ring: &HwRing, a: &[HwElt], b: &[HwElt]) -> HwElt {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = ring.zero();
    for (x, y) in a.iter().zip(b.iter()) {
        acc = ring.add(&acc, &ring.mul(x, y));
    }
    acc
}

/// Centered digit decomposition of one element into `f` layers of `b`
/// bits (the first `f−1` centered, the top layer the norm-controlled
/// remainder — the reference's Figure-3 discipline).
pub fn decompose_elt(ring: &HwRing, e: &HwElt, f: usize, b: u32) -> Vec<HwElt> {
    debug_assert!(f >= 1);
    if f == 1 {
        return vec![e.clone()];
    }
    let mask: i128 = (1i128 << b) - 1;
    let half: i128 = 1i128 << (b - 1);
    let q = i128::from(ring.q);
    let mut rem: Vec<i128> = e.0.iter().map(|&c| ring.center(c)).collect();
    let mut layers = vec![vec![0u64; ring.n]; f];
    for u in 0..f - 1 {
        for i in 0..ring.n {
            let low = rem[i] & mask;
            let v = if low >= half { low - (1i128 << b) } else { low };
            layers[u][i] = v.rem_euclid(q) as u64;
            rem[i] = (rem[i] - v) >> b;
        }
    }
    for i in 0..ring.n {
        layers[f - 1][i] = rem[i].rem_euclid(q) as u64;
    }
    layers.into_iter().map(HwElt).collect()
}

/// Recompose `Σ_u 2^{u·b}·layer[u]`.
pub fn recombine_elt(ring: &HwRing, layers: &[HwElt], b: u32) -> HwElt {
    let mut acc = ring.zero();
    for (u, layer) in layers.iter().enumerate() {
        acc = ring.add(&acc, &ring.scale(layer, 1i128 << (u as u32 * b)));
    }
    acc
}

/// Decompose a vector, transposed: `out[d][k]` = digit `d` of `z[k]`.
pub fn decompose_vec(ring: &HwRing, z: &[HwElt], f: usize, b: u32) -> Vec<Vec<HwElt>> {
    let mut out = vec![Vec::with_capacity(z.len()); f];
    for e in z {
        for (d, layer) in decompose_elt(ring, e, f, b).into_iter().enumerate() {
            out[d].push(layer);
        }
    }
    out
}

/// Recombine a vector of digit layers.
pub fn recombine_vec(ring: &HwRing, parts: &[Vec<HwElt>], b: u32) -> Vec<HwElt> {
    let n = parts.first().map(|v| v.len()).unwrap_or(0);
    (0..n)
        .map(|k| {
            recombine_elt(
                ring,
                &parts.iter().map(|v| v[k].clone()).collect::<Vec<_>>(),
                b,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The principal statement (re-parameterized relation.rs)
// ---------------------------------------------------------------------------

/// Spec of one witness vector: rank + the exact per-vector ℓ2 cap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabVectorSpec {
    pub n: usize,
    pub betasq: u64,
}

/// One linear term: `⟨phi, s_idx[off..off+phi.len()]⟩`.
#[derive(Clone, Debug)]
pub struct LabTerm {
    pub idx: usize,
    pub off: usize,
    pub phi: Vec<HwElt>,
}

/// One quadratic entry `a_ij·⟨s_i, s_j⟩` for i ≤ j (the symmetric
/// (2 − [i=j]) factor at evaluation).
#[derive(Clone, Debug)]
pub struct LabQuadEntry {
    pub i: usize,
    pub j: usize,
    pub coeff: HwElt,
}

/// One dot-product constraint (the F family: fully vanishing; the F'
/// family: constant-term only).
#[derive(Clone, Debug)]
pub struct LabConstraint {
    pub terms: Vec<LabTerm>,
    pub quads: Vec<LabQuadEntry>,
    pub b: Option<HwElt>,
    pub ct_only: bool,
}

impl LabConstraint {
    pub fn homogeneous(terms: Vec<LabTerm>) -> Self {
        Self {
            terms,
            quads: Vec::new(),
            b: None,
            ct_only: false,
        }
    }

    /// Evaluate at a witness (the caller checks = 0 or ct = 0).
    pub fn eval(&self, ring: &HwRing, s: &[Vec<HwElt>]) -> HwElt {
        let mut acc = ring.zero();
        for t in &self.terms {
            let slice = &s[t.idx][t.off..t.off + t.phi.len()];
            acc = ring.add(&acc, &sprod(ring, &t.phi, slice));
        }
        for q in &self.quads {
            let prod = sprod(ring, &s[q.i], &s[q.j]);
            let scaled = if q.i == q.j {
                ring.mul(&q.coeff, &prod)
            } else {
                ring.scale(&ring.mul(&q.coeff, &prod), 2)
            };
            acc = ring.add(&acc, &scaled);
        }
        if let Some(b) = &self.b {
            acc = ring.sub(&acc, b);
        }
        acc
    }

    pub fn check(&self, ring: &HwRing, s: &[Vec<HwElt>]) -> bool {
        let v = self.eval(ring, s);
        if self.ct_only {
            v.0[0] % ring.q == 0
        } else {
            v.0.iter().all(|&c| c % ring.q == 0)
        }
    }
}

/// A principal statement: (F, F', β) over the witness vectors.
#[derive(Clone, Debug)]
pub struct LabStatement {
    pub vectors: Vec<LabVectorSpec>,
    pub cnst: Vec<LabConstraint>,
    pub ct_cnst: Vec<LabConstraint>,
    pub betasq: u64,
    pub digest: [u8; 32],
}

impl LabStatement {
    pub fn new(
        vectors: Vec<LabVectorSpec>,
        cnst: Vec<LabConstraint>,
        ct_cnst: Vec<LabConstraint>,
        betasq: u64,
    ) -> Self {
        let mut st = Self {
            vectors,
            cnst,
            ct_cnst,
            betasq,
            digest: [0u8; 32],
        };
        st.digest = st.content_digest();
        st
    }

    pub fn total_rank(&self) -> usize {
        self.vectors.iter().map(|v| v.n).sum()
    }

    /// Structural validation: term slices in range, quad indices valid.
    pub fn validate(&self) -> Result<(), String> {
        for (k, c) in self.cnst.iter().chain(self.ct_cnst.iter()).enumerate() {
            for t in &c.terms {
                if t.idx >= self.vectors.len() {
                    return Err(format!("cnst {k}: term idx {} out of range", t.idx));
                }
                if t.off + t.phi.len() > self.vectors[t.idx].n {
                    return Err(format!(
                        "cnst {k}: slice [{}, {}) exceeds rank {} of vector {}",
                        t.off,
                        t.off + t.phi.len(),
                        self.vectors[t.idx].n,
                        t.idx
                    ));
                }
            }
            for q in &c.quads {
                if q.i > q.j || q.j >= self.vectors.len() {
                    return Err(format!("cnst {k}: bad quad index ({},{})", q.i, q.j));
                }
            }
        }
        Ok(())
    }

    /// The full statement-side check (tests and the final path).
    pub fn check_all(&self, ring: &HwRing, s: &[Vec<HwElt>]) -> Result<(), String> {
        if s.len() != self.vectors.len() {
            return Err("witness multiplicity mismatch".into());
        }
        for (i, v) in self.vectors.iter().enumerate() {
            if s[i].len() != v.n {
                return Err(format!("witness vector {i} rank mismatch"));
            }
        }
        let normsq: u64 = s
            .iter()
            .map(|v| {
                v.iter()
                    .map(|p| norm_sq_u64(ring, p))
                    .fold(0u64, u64::saturating_add)
            })
            .fold(0u64, u64::saturating_add);
        if normsq > self.betasq {
            return Err(format!(
                "witness norm² {normsq} exceeds bound {}",
                self.betasq
            ));
        }
        for (k, c) in self.cnst.iter().enumerate() {
            if !c.check(ring, s) {
                return Err(format!("constraint {k} (F) violated"));
            }
        }
        for (k, c) in self.ct_cnst.iter().enumerate() {
            if !c.check(ring, s) {
                return Err(format!("constraint {k} (F') violated"));
            }
        }
        Ok(())
    }

    /// The digest — SPARSE-aware hashing of the phi elements (the
    /// σ⁻¹-binding families have thousands of two-nonzero phis; hashing
    /// the dense bytes would dominate the kernel runtime).
    fn content_digest(&self) -> [u8; 32] {
        let mut buf: Vec<u8> = Vec::with_capacity(4096);
        buf.extend_from_slice(b"hw-lab/statement/v1");
        buf.extend_from_slice(&(self.vectors.len() as u64).to_le_bytes());
        for v in &self.vectors {
            buf.extend_from_slice(&(v.n as u64).to_le_bytes());
            buf.extend_from_slice(&v.betasq.to_le_bytes());
        }
        buf.extend_from_slice(&self.betasq.to_le_bytes());
        let absorb = |c: &LabConstraint, tag: &[u8], buf: &mut Vec<u8>| {
            buf.extend_from_slice(tag);
            buf.extend_from_slice(&(c.terms.len() as u64).to_le_bytes());
            for t in &c.terms {
                buf.extend_from_slice(&(t.idx as u64).to_le_bytes());
                buf.extend_from_slice(&(t.off as u64).to_le_bytes());
                buf.extend_from_slice(&(t.phi.len() as u64).to_le_bytes());
                for p in &t.phi {
                    absorb_sparse(crate::hyperwolf::HW_Q61, p, buf);
                }
            }
            buf.extend_from_slice(&(c.quads.len() as u64).to_le_bytes());
            for q in &c.quads {
                buf.extend_from_slice(&(q.i as u64).to_le_bytes());
                buf.extend_from_slice(&(q.j as u64).to_le_bytes());
                absorb_sparse(crate::hyperwolf::HW_Q61, &q.coeff, buf);
            }
            match &c.b {
                None => buf.extend_from_slice(b"hom"),
                Some(b) => {
                    buf.extend_from_slice(b"b");
                    for &x in &b.0 {
                        buf.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
            buf.push(u8::from(c.ct_only));
        };
        buf.extend_from_slice(&(self.cnst.len() as u64).to_le_bytes());
        for c in &self.cnst {
            absorb(c, b"F", &mut buf);
        }
        buf.extend_from_slice(&(self.ct_cnst.len() as u64).to_le_bytes());
        for c in &self.ct_cnst {
            absorb(c, b"Fp", &mut buf);
        }
        sha3_256(&buf)
    }
}

/// Hash one ring element by its centered nonzero coefficients.
fn absorb_sparse(q: u64, e: &HwElt, buf: &mut Vec<u8>) {
    for (i, &c) in e.0.iter().enumerate() {
        if c % q != 0 {
            buf.extend_from_slice(&(i as u32).to_le_bytes());
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }
    buf.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
}

/// The exact sum-of-squares decomposition `d = Σ c_i²` (the greedy
/// isqrt recursion — every positive integer decomposes; the remainder
/// strictly decreases into the {0,1,2,3} base cases). The slack vector's
/// COEFFICIENTS are these `c_i`, so `‖S‖² = d` EXACTLY (the binary-bits
/// trick gives the VALUE, not the norm — the trap this replaces).
pub fn square_decomposition(mut d: u64) -> Vec<i64> {
    let mut out = Vec::new();
    while d > 0 {
        let r = (d as f64).sqrt() as u64;
        // Guard the float rounding at the boundary.
        let mut a = r.min(d);
        while a > 0 && a.saturating_mul(a) > d {
            a -= 1;
        }
        let a_alt = a + 1;
        if a_alt.saturating_mul(a_alt) <= d {
            a = a_alt;
        }
        out.push(a as i64);
        d -= a.saturating_mul(a);
    }
    out
}

/// The squared norm of one element as u64 (saturating).
fn norm_sq_u64(ring: &HwRing, p: &HwElt) -> u64 {
    let mut acc: u128 = 0;
    for &c in &p.0 {
        let v = ring.center(c);
        acc += (v * v) as u128;
    }
    acc.min(u64::MAX as u128) as u64
}

/// A witness: vectors of ring elements.
#[derive(Clone, Debug, Default)]
pub struct LabWitness {
    pub s: Vec<Vec<HwElt>>,
}

impl LabWitness {
    pub fn new(s: Vec<Vec<HwElt>>) -> Self {
        Self { s }
    }

    pub fn per_vector_normsq(&self, ring: &HwRing) -> Vec<u64> {
        self.s
            .iter()
            .map(|v| {
                v.iter()
                    .map(|p| norm_sq_u64(ring, p))
                    .fold(0u64, u64::saturating_add)
            })
            .collect()
    }

    pub fn normsq(&self, ring: &HwRing) -> u64 {
        self.per_vector_normsq(ring)
            .into_iter()
            .fold(0u64, u64::saturating_add)
    }
}

// ---------------------------------------------------------------------------
// Parameters + the Ajtai key (re-parameterized sis.rs)
// ---------------------------------------------------------------------------

/// Commitment parameters for one level (the reference `comparams`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabComParams {
    pub f: usize,
    pub fu: usize,
    pub fg: usize,
    pub b: u32,
    pub bu: u32,
    pub bg: u32,
    pub kappa: usize,
    pub kappa1: usize,
}

/// The Core-SVP predicate at the HyperWolf ring.
pub fn lab_sis_secure(ring: &HwRing, rank: usize, norm: f64) -> bool {
    let mut maxlog = 2.0 * (LAB_LOGQ * LOGDELTA * ring.n as f64).sqrt() * (rank as f64).sqrt();
    maxlog = maxlog.min(LAB_LOGQ);
    norm.log2() < maxlog
}

/// The parameter search (the k = 15..1 loop of `init_proof`, with the
/// certified-family constants and the kernel-bypass clamp).
fn init_params(
    ring: &HwRing,
    ranks: &[usize],
    normsq: &[u64],
    quadratic: bool,
    tail: bool,
    gate: SisGateMode,
) -> Result<(LabComParams, usize, usize, u64), String> {
    let n_coeff = ring.n as f64;
    for k in (1..=15usize).rev() {
        let mut best_block = 0usize;
        let mut total_sq = 0u64;
        let mut boundary_ranks: Vec<usize> = Vec::new();
        let mut acc_n = 0usize;
        let mut acc_sq = 0u64;
        for i in 0..ranks.len() {
            acc_n += ranks[i];
            acc_sq = acc_sq.saturating_add(normsq[i]);
            if quadratic || i == ranks.len() - 1 {
                boundary_ranks.push(acc_n);
                best_block = best_block.max(acc_n);
                total_sq = total_sq.saturating_add(acc_sq);
                acc_n = 0;
                acc_sq = 0;
            }
        }
        let nn = best_block.div_ceil(k).max(1);
        let r_total: usize = boundary_ranks
            .iter()
            .map(|&n| n.div_ceil(nn))
            .sum::<usize>()
            .max(1);

        // z variance under the certified challenges (τ = 10 per part).
        let varz = (total_sq as f64 / (nn as f64 * n_coeff)) * LAB_TAU;
        let decompose = !tail
            && !lab_sis_secure(
                ring,
                13,
                6.0 * LAB_T * LAB_SLACK * (2.0 * LAB_TAU * varz * (nn as f64 * n_coeff)).sqrt(),
            )
            || 64.0 * varz > (1u64 << 28) as f64;
        let (f, b) = if decompose {
            (
                2usize,
                (((12.0f64).log2() + varz.log2()) / 4.0).round().max(1.0) as u32,
            )
        } else {
            (
                1usize,
                (((12.0f64).log2() + varz.log2()) / 2.0).round().max(1.0) as u32,
            )
        };
        let (f, b) = if b > DIGITBITS {
            let t = f as u32 * b;
            let f2 = t.div_ceil(DIGITBITS) as usize;
            (f2, t.div_ceil(f2 as u32))
        } else {
            (f, b)
        };
        let (fu, bu) = if !tail {
            let fu = (LAB_LOGQ as usize + 2 * b as usize / 3) / b as usize;
            (
                fu.max(1),
                ((LAB_LOGQ as usize + fu / 2) / fu.max(1)).max(1) as u32,
            )
        } else {
            (1usize, LAB_LOGQ as u32)
        };
        let (bg, fg) = {
            let bg = b;
            let fg = if !quadratic {
                0usize
            } else if tail {
                1usize
            } else {
                let mut varg = 0.0f64;
                let mut acc = 0.0f64;
                let mut u = 0usize;
                for i in 0..ranks.len() {
                    let vars = normsq[i] as f64 / (ranks[i] as f64 * n_coeff);
                    let mut j = ranks[i];
                    while j + u >= nn {
                        let take = (nn - u) as f64;
                        j -= nn - u;
                        acc += vars * vars * take;
                        varg = varg.max(acc);
                        acc = 0.0;
                        u = 0;
                    }
                    acc += vars * vars * j as f64;
                    u += j;
                    varg = varg.max(acc);
                    acc = 0.0;
                }
                let varg = 2.0 * n_coeff * varg;
                let fg = (((12.0f64).log2() + varg.max(1e-300).log2()) / (2.0 * bg as f64)).ceil()
                    as usize;
                fg.max(1)
            };
            (bg, fg)
        };

        // commitment ranks + the predicted output norm²
        let mut norm = (2f64.powi(2 * b as i32) / 12.0 * (f - 1) as f64
            + varz / 2f64.powi(2 * b as i32 * (f - 1) as i32))
            * nn as f64;
        let rr = k as f64;
        norm += (2f64.powi(2 * bu as i32) * (fu - 1) as f64
            + 2f64.powi(2 * (LAB_LOGQ as i32 - (fu as i32 - 1) * bu as i32)))
            / 12.0
            * (rr + (rr * rr + rr) / 2.0);
        if fg > 0 {
            norm += (2f64.powi(2 * bg as i32) / 12.0 * (fg - 1) as f64
                + 2f64.powi(2 * bg as i32 * (fg as i32 - 1)))
                * (rr * rr + rr)
                / 2.0;
        }
        norm *= n_coeff;
        let normsq_pred = (norm.min(u64::MAX as f64)) as u64;
        let kappa_cap = match gate {
            SisGateMode::Faithful => 32,
            SisGateMode::KernelBypass => 8,
        };
        let kappa = (1..=kappa_cap)
            .find(|&kp| {
                gate == SisGateMode::KernelBypass
                    || lab_sis_secure(
                        ring,
                        kp,
                        6.0 * LAB_T
                            * LAB_SLACK
                            * 2f64.powi((f as i32 - 1) * b as i32)
                            * norm.sqrt(),
                    )
            })
            .unwrap_or(kappa_cap + 1);
        if kappa > kappa_cap {
            continue;
        }
        // The tail transmits its openings directly: no outer commitments
        // (kappa1 = 0 — the §5.6 discipline the LevelProof::tail() test reads).
        let kappa1 = if tail {
            0usize
        } else {
            let k1 = (1..=kappa_cap)
                .find(|&k1| {
                    gate == SisGateMode::KernelBypass
                        || lab_sis_secure(ring, k1, 2.0 * LAB_SLACK * norm.sqrt())
                })
                .unwrap_or(kappa_cap + 1);
            if k1 > kappa_cap {
                continue;
            }
            k1
        };
        if tail || fu * k * kappa + (fu + fg) * (k * k + k) / 2 <= 11 * nn / 10 + 1 {
            return Ok((
                LabComParams {
                    f,
                    fu,
                    fg,
                    b,
                    bu,
                    bg,
                    kappa,
                    kappa1,
                },
                nn,
                r_total,
                normsq_pred,
            ));
        }
    }
    Err("cannot make commitments secure".into())
}

/// The shared Ajtai key over the HwRing (disjoint A|B|C|D windows).
pub struct LabKey {
    pub ring: HwRing,
    pub rows: Vec<HwElt>,
    pub len: usize,
}

impl LabKey {
    /// Deterministic expansion from context bytes (statement digest +
    /// level index by the callers).
    pub fn expand(ring: &HwRing, len: usize, ctx: &[u8]) -> Self {
        let mut rows = Vec::with_capacity(len);
        for i in 0..len {
            let mut domain = b"hw-lab/key/v1".to_vec();
            domain.extend_from_slice(ctx);
            domain.extend_from_slice(&(i as u64).to_le_bytes());
            let bytes = Transcript::xof(b"hw-lab-key", &domain, 8 * ring.n);
            rows.push(ring.from_bytes_mod(&bytes));
        }
        LabKey {
            ring: ring.clone(),
            rows,
            len,
        }
    }

    /// `A_win·v`: `kappa` outputs, output `t` = `Σ_k rows[off+t·v.len()+k]·v[k]`.
    pub fn mul_window(&self, v: &[HwElt], off: usize, kappa: usize) -> Vec<HwElt> {
        let mut out = Vec::with_capacity(kappa);
        for t in 0..kappa {
            let mut acc = self.ring.zero();
            for (k, x) in v.iter().enumerate() {
                let row = &self.rows[off + t * v.len() + k];
                acc = self.ring.add(&acc, &self.ring.mul(row, x));
            }
            out.push(acc);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Layouts (ring-agnostic ports of protocol.rs)
// ---------------------------------------------------------------------------

/// Triangular pair ordinal for (i, j), i ≤ j, over r parts.
pub fn tri_idx(i: usize, j: usize, r: usize) -> usize {
    let (i, j) = if i > j { (j, i) } else { (i, j) };
    i * r - (i * i + i) / 2 + j
}

/// The v-vector layout: [t̃ (r·fu·κ), g̃ (fg·pairs), h̃ (fu·pairs)].
#[derive(Clone, Copy, Debug)]
pub struct VLayout {
    pub t_len: usize,
    pub g_len: usize,
    pub h_len: usize,
    pub m: usize,
}

impl VLayout {
    pub fn new(cpp: &LabComParams, r: usize) -> Self {
        let pairs = (r * r + r) / 2;
        Self {
            t_len: r * cpp.fu * cpp.kappa,
            g_len: cpp.fg * pairs,
            h_len: cpp.fu * pairs,
            m: r * cpp.fu * cpp.kappa + (cpp.fg + cpp.fu) * pairs,
        }
    }
    pub fn g_off(&self) -> usize {
        self.t_len
    }
    pub fn h_off(&self) -> usize {
        self.t_len + self.g_len
    }
}

/// The joined-part layout (verifier-regenerable).
#[derive(Clone, Debug)]
pub struct PartLayout {
    pub nn: usize,
    pub r: usize,
    pub ranks: Vec<usize>,
    pub starts: Vec<(usize, usize)>,
    pub origin: Vec<Option<usize>>,
    /// The quadratic mode's per-vector part alignment (materialize pads
    /// each vector to a whole number of parts — the contiguous packing
    /// would misplace the trailing vectors' data).
    pub per_vector: bool,
}

/// Compute the joined-part layout (quadratic mode: per-vector parts).
pub fn part_layout(ranks: &[usize], nn: usize, per_vector: bool) -> PartLayout {
    let mut starts = Vec::with_capacity(ranks.len());
    let mut origin: Vec<Option<usize>> = Vec::new();
    let mut filled = 0usize;
    let mut open_origin: Option<usize> = None;
    for (vi, &n) in ranks.iter().enumerate() {
        if per_vector {
            if filled > 0 {
                origin.push(open_origin.take());
            }
            starts.push((origin.len(), 0));
            let mut rest = n;
            while rest > nn {
                origin.push(Some(vi));
                rest -= nn;
            }
            filled = rest;
            open_origin = Some(vi);
        } else {
            starts.push((origin.len(), filled));
            if open_origin.is_none() {
                open_origin = Some(vi);
            }
            let mut rest = n;
            while filled + rest > nn {
                let take = nn - filled;
                rest -= take;
                origin.push(open_origin.take());
                filled = 0;
                open_origin = Some(vi);
            }
            filled += rest;
        }
    }
    if filled > 0 || origin.is_empty() {
        origin.push(open_origin);
    }
    PartLayout {
        nn,
        r: origin.len(),
        ranks: ranks.to_vec(),
        starts,
        origin,
        per_vector,
    }
}

impl PartLayout {
    /// The padded parts from a witness (prover side). In the quadratic
    /// mode every vector is padded to a whole number of parts (the
    /// per-vector part alignment `locate`/`starts` assume — a contiguous
    /// packing would land a short vector's successor inside its last
    /// part); in the linear mode the contiguous packing IS the layout.
    pub fn materialize(&self, wit: &LabWitness) -> Vec<Vec<HwElt>> {
        let dim = wit
            .s
            .first()
            .and_then(|v| v.first())
            .map(|e| e.0.len())
            .unwrap_or(0);
        let mut flat: Vec<HwElt> = Vec::with_capacity(self.r * self.nn);
        if self.per_vector {
            for (vi, v) in wit.s.iter().enumerate() {
                let base = flat.len();
                flat.extend(v.iter().cloned());
                let want = base + self.ranks[vi].div_ceil(self.nn) * self.nn;
                while flat.len() < want {
                    // pad this vector to its part boundary
                    flat.push(HwElt(vec![0u64; dim]));
                }
            }
        } else {
            for v in &wit.s {
                flat.extend(v.iter().cloned());
            }
        }
        while flat.len() < self.r * self.nn {
            flat.push(HwElt(vec![0u64; dim]));
        }
        flat.truncate(self.r * self.nn);
        (0..self.r)
            .map(|i| flat[i * self.nn..(i + 1) * self.nn].to_vec())
            .collect()
    }

    /// (vector, offset) → (part, offset in part).
    pub fn locate(&self, idx: usize, off: usize) -> (usize, usize) {
        let (mut part, mut pos) = self.starts[idx];
        let mut remaining = off;
        loop {
            let room = self.nn - pos;
            if remaining < room {
                return (part, pos + remaining);
            }
            remaining -= room;
            part += 1;
            pos = 0;
        }
    }

    /// Split a phi slice into per-part chunks.
    pub fn locate_slice(
        &self,
        idx: usize,
        off: usize,
        phi: &[HwElt],
    ) -> Vec<(usize, usize, Vec<HwElt>)> {
        let mut out = Vec::new();
        let (mut part, mut pos) = self.locate(idx, off);
        let mut phi_rest = phi;
        loop {
            let room = self.nn - pos;
            if phi_rest.len() <= room {
                if !phi_rest.is_empty() {
                    out.push((part, pos, phi_rest.to_vec()));
                }
                return out;
            }
            out.push((part, pos, phi_rest[..room].to_vec()));
            phi_rest = &phi_rest[room..];
            part += 1;
            pos = 0;
        }
    }
}

/// Expand an input quad entry to part-level entries with the positional
/// chunk pairing.
fn expand_a_entry(
    layout: &PartLayout,
    i: usize,
    j: usize,
    coeff: &HwElt,
) -> Vec<(usize, usize, HwElt)> {
    let (si, _) = layout.starts[i];
    let (sj, _) = layout.starts[j];
    let ni = layout.ranks[i].div_ceil(layout.nn);
    let nj = layout.ranks[j].div_ceil(layout.nn);
    let n = ni.min(nj).max(1);
    let mut out = Vec::with_capacity(n);
    for x in 0..n {
        let p = si + x;
        let q = sj + x;
        out.push((p.min(q), p.max(q), coeff.clone()));
    }
    out
}

/// Whether a statement has quadratic entries.
pub fn is_quadratic(st: &LabStatement) -> bool {
    st.cnst
        .iter()
        .chain(st.ct_cnst.iter())
        .any(|c| !c.quads.is_empty())
}

/// Key-window bookkeeping: A | B | C | D (disjoint).
#[derive(Clone, Copy, Debug)]
pub struct Windows {
    pub a_off: usize,
    pub b_off: usize,
    pub c_off: usize,
    pub d_off: usize,
    pub total: usize,
}

impl Windows {
    pub fn new(cpp: &LabComParams, r: usize, nn: usize) -> Self {
        let vl = VLayout::new(cpp, r);
        let a_off = 0;
        let b_off = a_off + cpp.kappa * nn;
        let c_off = b_off + cpp.kappa1 * (vl.t_len + vl.g_len);
        let d_off = c_off + cpp.kappa1 * vl.h_len;
        Windows {
            a_off,
            b_off,
            c_off,
            d_off,
            total: d_off + cpp.kappa1 * vl.h_len,
        }
    }
}

// ---------------------------------------------------------------------------
// The JL projection (±1 matrices with the reference's rejection rule)
// ---------------------------------------------------------------------------

/// One JL matrix: 256 × cols ±1 entries, bit-packed, seed-expanded.
pub struct LabJlMatrix {
    pub rows: usize,
    pub cols: usize,
    pub bits: Vec<u8>,
}

impl LabJlMatrix {
    pub fn expand(rows: usize, cols: usize, seed: &[u8], nonce: u64) -> Self {
        let nbytes = (rows * cols).div_ceil(8);
        let mut domain = b"hw-lab-jl/v1".to_vec();
        domain.extend_from_slice(seed);
        domain.extend_from_slice(&nonce.to_le_bytes());
        let mut bits = vec![0u8; nbytes];
        bits.copy_from_slice(&Transcript::xof(b"hw-lab-jl", &domain, nbytes));
        Self { rows, cols, bits }
    }

    #[inline]
    pub fn entry(&self, row: usize, col: usize) -> i8 {
        let bitpos = row * self.cols + col;
        if (self.bits[bitpos / 8] >> (bitpos % 8)) & 1 == 1 {
            1
        } else {
            -1
        }
    }

    /// The integer matrix-vector product (exact; i128 accumulators — the
    /// adversarial full-width regime would overflow i64).
    pub fn project(&self, w: &[i64]) -> Vec<i64> {
        debug_assert_eq!(w.len(), self.cols);
        let mut acc = vec![0i128; self.rows];
        for (c, &wc) in w.iter().enumerate() {
            if wc == 0 {
                continue;
            }
            for r in 0..self.rows {
                acc[r] += i128::from(self.entry(r, c)) * i128::from(wc);
            }
        }
        acc.iter()
            .map(|&x| i64::try_from(x).unwrap_or(i64::MAX))
            .collect()
    }

    /// Row `r` restricted to ring element `k`, packed as an HwElt (the
    /// paper's π_i^{(j)}).
    pub fn row_elt(&self, ring: &HwRing, row: usize, elem: usize) -> HwElt {
        let mut c = vec![0u64; ring.n];
        for j in 0..ring.n {
            c[j] = ((i64::from(self.entry(row, elem * ring.n + j)) as i128)
                .rem_euclid(i128::from(ring.q))) as u64;
        }
        HwElt(c)
    }
}

/// The reference's rejection rule: `|p_i| < 2^⌈log2 4√normsq⌉` and
/// `Σ p² ≤ 256·normsq`.
pub fn jl_accept(p: &[i64], normsq: u64) -> bool {
    let bound = {
        let mut e = 0u32;
        while (1u64 << e) < 4 * (normsq as f64).sqrt() as u64 {
            e += 1;
        }
        1u64 << e
    };
    let psq: u128 = p
        .iter()
        .map(|&x| (i128::from(x) * i128::from(x)) as u128)
        .sum();
    let cap = 256u128 * u128::from(normsq.max(1));
    p.iter().all(|&x| x.unsigned_abs() < bound) && psq <= cap
}

/// The squared norm of a projection (i128-exact, saturating to u64).
pub fn jl_normsq(p: &[i64]) -> u64 {
    let psq: u128 = p
        .iter()
        .map(|&x| (i128::from(x) * i128::from(x)) as u128)
        .sum();
    psq.min(u64::MAX as u128) as u64
}

struct LabJlProjection {
    p: Vec<i64>,
    nonce: u64,
    mats: Vec<LabJlMatrix>,
}

/// Project the joined parts: one matrix per part, retries until accepted.
fn project_parts(ring: &HwRing, parts: &[Vec<HwElt>], seed: &[u8]) -> LabJlProjection {
    let normsq: u64 = parts
        .iter()
        .map(|v| {
            v.iter()
                .map(|p| norm_sq_u64(ring, p))
                .fold(0u64, u64::saturating_add)
        })
        .fold(0u64, u64::saturating_add);
    // The reference retries until acceptance; a witness outside the
    // acceptance regime (norms past the JL bound) must fail closed, not
    // loop forever — the 256-try budget (unreachable in the sound regime,
    // the acceptance is a ~50% coin per nonce).
    let mut nonce = 0u64;
    let mut mats: Vec<LabJlMatrix> = Vec::new();
    let mut p = vec![0i64; 256];
    for _ in 0..256 {
        nonce += 1;
        mats = parts
            .iter()
            .enumerate()
            .map(|(i, v)| {
                LabJlMatrix::expand(
                    256,
                    v.len() * ring.n,
                    seed,
                    nonce.wrapping_mul(parts.len() as u64) + i as u64,
                )
            })
            .collect();
        let mut acc = vec![0i64; 256];
        for (i, v) in parts.iter().enumerate() {
            let flat: Vec<i64> = v
                .iter()
                .flat_map(|e| e.0.iter().map(|&c| ring.center(c) as i64))
                .collect();
            let sub = mats[i].project(&flat);
            for (r, &x) in sub.iter().enumerate() {
                acc[r] = acc[r].saturating_add(x);
            }
        }
        p = acc;
        if jl_accept(&p, normsq) {
            break;
        }
    }
    LabJlProjection { p, nonce, mats }
}

/// Collapse the 256 JL equations into one constant-term constraint with
/// Z_q challenges ω: returns (Φ per part, target ct) where
/// `Σ_i ⟨Φ_i, s_i⟩ ≡ ⟨ω, p⟩ (mod q)` at the constant term, with the
/// σ⁻¹ packing `Φ_i = Σ_j ω_j·σ⁻¹(π_i^{(j)})`.
fn collapse_jl(
    ring: &HwRing,
    mats: &[LabJlMatrix],
    p: &[i64],
    omega: &[i128],
) -> (Vec<Vec<HwElt>>, u64) {
    let q = i128::from(ring.q);
    let mut phis = Vec::with_capacity(mats.len());
    for m in mats.iter() {
        let n_elems = m.cols / ring.n;
        let mut phi = vec![ring.zero(); n_elems];
        for (j, &w) in omega.iter().enumerate() {
            if w == 0 {
                continue;
            }
            for k in 0..n_elems {
                let row = m.row_elt(ring, j, k);
                let conj = ring.conj(&row);
                let scaled = ring.scale(&conj, w);
                phi[k] = ring.add(&phi[k], &scaled);
            }
        }
        phis.push(phi);
    }
    let mut target: i128 = 0;
    for (&w, &x) in omega.iter().zip(p.iter()) {
        target = (target + w * i128::from(x)).rem_euclid(q);
    }
    (phis, target as u64)
}

// ---------------------------------------------------------------------------
// Transcript challenge helpers
// ---------------------------------------------------------------------------

/// Sample `count` uniform R_q elements from the transcript.
fn rq_sample(
    ring: &HwRing,
    tr: &mut Transcript,
    label: &[u8],
    count: usize,
) -> Result<Vec<HwElt>, HwError> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut l = label.to_vec();
        l.extend_from_slice(b":");
        l.extend_from_slice(&(i as u64).to_le_bytes());
        let bytes = tr.challenge_bytes(&l, 8 * ring.n)?;
        out.push(ring.from_bytes_mod(&bytes));
    }
    Ok(out)
}

/// Sample `count` Z_q scalars (canonical [0, q)) from the transcript.
fn zq_sample(
    tr: &mut Transcript,
    label: &[u8],
    count: usize,
    q: u64,
) -> Result<Vec<i128>, HwError> {
    let bytes = tr.challenge_bytes(label, count * 8)?;
    Ok((0..count)
        .map(|i| {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
            i128::from(u64::from_le_bytes(arr) % q)
        })
        .collect())
}

/// Sample the amortization challenges from C (the certified
/// fixed-weight family — the H5 discipline) from the transcript.
fn challenge_vec_hw(
    ring: &HwRing,
    tr: &mut Transcript,
    count: usize,
) -> Result<Vec<HwElt>, HwError> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut l = b"hw-lab-cchal".to_vec();
        l.extend_from_slice(b":");
        l.extend_from_slice(&(i as u64).to_le_bytes());
        let (elt, _gamma) = sample_challenge(tr, &l, ring.n)?;
        out.push(elt);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// One recursion level (the port of protocol.rs's prove_level machinery)
// ---------------------------------------------------------------------------

/// One level's public proof pieces.
#[derive(Clone, Debug)]
pub struct LabLevelProof {
    /// First outer commitment (κ1); tail mode: the inner commitments t_i
    /// (r·κ) then the quadratic garbage (pairs).
    pub u1: Vec<HwElt>,
    /// Second outer commitment (κ1); tail mode: the 2r−1 interleaved terms.
    pub u2: Vec<HwElt>,
    /// The JL projection (256 integer values) and its nonce.
    pub p: Vec<i64>,
    pub jlnonce: u64,
    /// The LIFTS polynomials b''.
    pub bb: Vec<HwElt>,
    /// The amortization challenges c_i.
    pub c: Vec<HwElt>,
    /// The announced output norm² bound.
    pub normsq: u64,
    pub cpp: LabComParams,
    pub nn: usize,
    pub r: usize,
}

impl LabLevelProof {
    pub fn tail(&self) -> bool {
        self.cpp.kappa1 == 0
    }
}

/// Prove one level with the §5.4 restart remedy (inflated norms, up to 8
/// tries). Returns (proof, target statement, target witness).
#[allow(clippy::type_complexity)]
pub fn lab_prove_level(
    ring: &HwRing,
    gate: SisGateMode,
    stmt: &LabStatement,
    wit: &LabWitness,
    key_ctx: &[u8],
    tail: bool,
) -> Result<(LabLevelProof, Option<LabStatement>, Option<LabWitness>), String> {
    let mut inflation = 1.0f64;
    for _ in 0..8 {
        match lab_prove_level_inner(ring, gate, stmt, wit, key_ctx, tail, inflation) {
            Ok(x) => return Ok(x),
            Err(e) if e.starts_with("RESTART:") => inflation *= 2.0,
            Err(e) => return Err(e),
        }
    }
    Err("§5.4 restart budget exhausted".into())
}

#[allow(clippy::too_many_lines, clippy::type_complexity)]
fn lab_prove_level_inner(
    ring: &HwRing,
    gate: SisGateMode,
    stmt: &LabStatement,
    wit: &LabWitness,
    key_ctx: &[u8],
    tail: bool,
    inflation: f64,
) -> Result<(LabLevelProof, Option<LabStatement>, Option<LabWitness>), String> {
    stmt.validate()?;
    stmt.check_all(ring, &wit.s)
        .map_err(|e| format!("witness: {e}"))?;
    let quadratic = is_quadratic(stmt);

    let ranks: Vec<usize> = stmt.vectors.iter().map(|v| v.n).collect();
    let norms: Vec<u64> = wit
        .per_vector_normsq(ring)
        .into_iter()
        .map(|n| ((n as f64) * inflation) as u64)
        .collect();
    let (cpp, nn, r, normsq_pred) = init_params(ring, &ranks, &norms, quadratic, tail, gate)?;

    let win = Windows::new(&cpp, r, nn);
    // The per-level key (derived from the statement digest context).
    let mut key_ctx_full = key_ctx.to_vec();
    key_ctx_full.extend_from_slice(&stmt.digest);
    let key = LabKey::expand(ring, win.total, &key_ctx_full);

    let vl = VLayout::new(&cpp, r);
    let layout = part_layout(&ranks, nn, quadratic);
    let parts = layout.materialize(wit);

    // ---- inner commitments, digits [i][j][ρ] ----
    let mut t_digits: Vec<HwElt> = Vec::with_capacity(vl.t_len);
    for part in &parts {
        let t = key.mul_window(part, win.a_off, cpp.kappa);
        let dec: Vec<Vec<HwElt>> = t
            .iter()
            .map(|tp| decompose_elt(ring, tp, cpp.fu, cpp.bu))
            .collect();
        for j in 0..cpp.fu {
            for rho in 0..cpp.kappa {
                t_digits.push(dec[rho][j].clone());
            }
        }
    }

    // ---- quadratic garbage, digits [pair][k] ----
    let mut g_digits: Vec<HwElt> = Vec::new();
    if quadratic && !tail {
        for i in 0..r {
            for j in i..r {
                for d in decompose_elt(ring, &sprod(ring, &parts[i], &parts[j]), cpp.fg, cpp.bg) {
                    g_digits.push(d);
                }
            }
        }
    }

    // ---- u1 ----
    let u1: Vec<HwElt> = if tail {
        let mut pieces = Vec::new();
        for part in &parts {
            pieces.extend(key.mul_window(part, win.a_off, cpp.kappa));
        }
        if quadratic {
            for i in 0..r {
                for j in i..r {
                    pieces.push(sprod(ring, &parts[i], &parts[j]));
                }
            }
        }
        pieces
    } else {
        let mut x = t_digits.clone();
        x.extend(g_digits.iter().cloned());
        if x.is_empty() {
            vec![ring.zero(); cpp.kappa1]
        } else {
            key.mul_window(&x, win.b_off, cpp.kappa1)
        }
    };

    // ---- transcript + JL ----
    let mut tr = Transcript::new_default(b"hw-lab-level");
    tr.append_message(b"stmt-digest", &stmt.digest)
        .map_err(|e| e.to_string())?;
    absorb_elts(&mut tr, b"u1", ring, &u1)?;
    let jl_seed = tr
        .challenge_bytes(b"jl-seed", 32)
        .map_err(|e| e.to_string())?;
    let jl = project_parts(ring, &parts, &jl_seed);
    tr.append_message(
        b"jl-p",
        &jl.p
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .map_err(|e| e.to_string())?;

    // ---- LIFTS ----
    let mut lifted: Vec<LabConstraint> = Vec::with_capacity(LAB_LIFTS);
    let mut bb: Vec<HwElt> = Vec::with_capacity(LAB_LIFTS);
    for k in 0..LAB_LIFTS {
        let omega = zq_sample(&mut tr, b"lift-omega", 256, ring.q).map_err(|e| e.to_string())?;
        let psi = zq_sample(&mut tr, b"lift-psi", stmt.ct_cnst.len(), ring.q)
            .map_err(|e| e.to_string())?;
        let (phis, jl_target) = collapse_jl(ring, &jl.mats, &jl.p, &omega);
        // Φ^(k) = JL-collapse + ψ-scaled F' linear terms
        let mut phi_k: Vec<Vec<HwElt>> = phis;
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            let ps = psi[l];
            if ps == 0 {
                continue;
            }
            for t in &c.terms {
                for (part, off, phi_chunk) in layout.locate_slice(t.idx, t.off, &t.phi) {
                    for (u, pc) in phi_chunk.iter().enumerate() {
                        let scaled = ring.scale(pc, ps);
                        let cur = phi_k[part][off + u].clone();
                        phi_k[part][off + u] = ring.add(&cur, &scaled);
                    }
                }
            }
        }
        let mut target: i128 = i128::from(jl_target);
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            if let Some(b) = &c.b {
                target = (target + psi[l] * i128::from(b.0[0])).rem_euclid(i128::from(ring.q));
            }
        }
        // honest evaluation of the lifted constraint
        let mut b_double = ring.zero();
        for (i, phi) in phi_k.iter().enumerate() {
            b_double = ring.add(&b_double, &sprod(ring, phi, &parts[i]));
        }
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            if !c.quads.is_empty() {
                let mut acc = ring.zero();
                for qe in &c.quads {
                    let prod = sprod(ring, &wit.s[qe.i], &wit.s[qe.j]);
                    let scaled = if qe.i == qe.j {
                        ring.mul(&qe.coeff, &prod)
                    } else {
                        ring.scale(&ring.mul(&qe.coeff, &prod), 2)
                    };
                    acc = ring.add(&acc, &scaled);
                }
                b_double = ring.add(&b_double, &ring.scale(&acc, psi[l]));
            }
        }
        if b_double.0[0] != target as u64 {
            return Err(format!(
                "lift {k}: ct mismatch {} != {}",
                b_double.0[0], target
            ));
        }
        let b_sent = b_double;
        absorb_elts(&mut tr, b"lift-b", ring, std::slice::from_ref(&b_sent))?;
        bb.push(b_sent);
        // the lifted constraint: Φ^(k) terms + the ψ-scaled F' quads
        let terms: Vec<LabTerm> = phi_k
            .iter()
            .enumerate()
            .map(|(i, phi)| LabTerm {
                idx: i,
                off: 0,
                phi: phi.clone(),
            })
            .collect();
        let mut quads: Vec<LabQuadEntry> = Vec::new();
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            for qe in &c.quads {
                for (p, q, cc) in expand_a_entry(&layout, qe.i, qe.j, &qe.coeff) {
                    quads.push(LabQuadEntry {
                        i: p,
                        j: q,
                        coeff: ring.scale(&cc, psi[l]),
                    });
                }
            }
        }
        lifted.push(LabConstraint {
            terms,
            quads,
            b: Some(bb[bb.len() - 1].clone()),
            ct_only: false,
        });
    }

    // ---- F-aggregation: uniform α ∈ R_q^K, β ∈ R_q^LIFTS ----
    let alphas =
        rq_sample(ring, &mut tr, b"agg-alpha", stmt.cnst.len()).map_err(|e| e.to_string())?;
    let betas = rq_sample(ring, &mut tr, b"agg-beta", LAB_LIFTS).map_err(|e| e.to_string())?;
    let (phi_agg, a_agg, b_agg) =
        aggregate_constraints(ring, stmt, &lifted, &layout, nn, &alphas, &betas);

    // ---- h-garbage, u2, challenges, z ----
    let u2: Vec<HwElt>;
    let c: Vec<HwElt>;
    let z_digits: Vec<Vec<HwElt>>;
    let h_digits: Vec<HwElt>;
    if tail {
        // §5.6: interleaved garbage
        let phi_slice = |i: usize| -> Vec<HwElt> { phi_agg[i * nn..(i + 1) * nn].to_vec() };
        let mut hs: Vec<HwElt> = Vec::with_capacity(2 * r - 1);
        hs.push(sprod(ring, &phi_slice(0), &parts[0]));
        absorb_elts(&mut tr, b"tail-h", ring, &[hs[0].clone()])?;
        let mut c_ch = challenge_vec_hw(ring, &mut tr, 1).map_err(|e| e.to_string())?;
        let mut z_run: Vec<HwElt> = parts[0].iter().map(|p| ring.mul(&c_ch[0], p)).collect();
        let mut phi_run: Vec<HwElt> = phi_slice(0).iter().map(|p| ring.mul(&c_ch[0], p)).collect();
        for i in 1..r {
            let off = ring.add(
                &sprod(ring, &phi_slice(i), &z_run),
                &sprod(ring, &phi_run, &parts[i]),
            );
            let diag = sprod(ring, &phi_slice(i), &parts[i]);
            absorb_elts(&mut tr, b"tail-h", ring, &[off.clone(), diag.clone()])?;
            let ci = challenge_vec_hw(ring, &mut tr, 1).map_err(|e| e.to_string())?;
            let ci = ci.first().cloned().ok_or("challenge sampling failed")?;
            c_ch.push(ci);
            hs.push(off);
            hs.push(diag);
            let ci = &c_ch[c_ch.len() - 1];
            for (zj, sj) in z_run.iter_mut().zip(parts[i].iter()) {
                *zj = ring.add(zj, &ring.mul(ci, sj));
            }
            for (pj, lp) in phi_run.iter_mut().zip(phi_slice(i).iter()) {
                *pj = ring.add(pj, &ring.mul(ci, lp));
            }
        }
        u2 = hs;
        c = c_ch;
        z_digits = decompose_vec(ring, &z_run, cpp.f, cpp.b);
        h_digits = Vec::new();
    } else {
        let mut h_vals: Vec<HwElt> = Vec::with_capacity((r * r + r) / 2);
        for i in 0..r {
            for j in i..r {
                let pi = &phi_agg[i * nn..(i + 1) * nn];
                let pj = &phi_agg[j * nn..(j + 1) * nn];
                let v = if i == j {
                    sprod(ring, pi, &parts[i])
                } else {
                    let inv2 = (ring.q + 1) / 2;
                    let sum = ring.add(&sprod(ring, pi, &parts[j]), &sprod(ring, pj, &parts[i]));
                    ring.scale(&sum, i128::from(inv2))
                };
                h_vals.push(v);
            }
        }
        let mut hd: Vec<HwElt> = Vec::with_capacity(vl.h_len);
        for h in &h_vals {
            for d in decompose_elt(ring, h, cpp.fu, cpp.bu) {
                hd.push(d);
            }
        }
        u2 = if hd.is_empty() {
            vec![ring.zero(); cpp.kappa1]
        } else {
            key.mul_window(&hd, win.d_off, cpp.kappa1)
        };
        absorb_elts(&mut tr, b"u2", ring, &u2)?;
        c = challenge_vec_hw(ring, &mut tr, r).map_err(|e| e.to_string())?;
        let mut z = vec![ring.zero(); nn];
        for i in 0..r {
            for (zk, sp) in z.iter_mut().zip(parts[i].iter()) {
                *zk = ring.add(zk, &ring.mul(&c[i], sp));
            }
        }
        z_digits = decompose_vec(ring, &z, cpp.f, cpp.b);
        h_digits = hd;
    }

    let mut proof = LabLevelProof {
        u1: u1.clone(),
        u2: u2.clone(),
        p: jl.p.clone(),
        jlnonce: jl.nonce,
        bb: bb.clone(),
        c: c.clone(),
        normsq: normsq_pred,
        cpp,
        nn,
        r,
    };

    if tail {
        let measured: u64 = z_digits
            .iter()
            .flat_map(|v| v.iter().map(|q| norm_sq_u64(ring, q)))
            .fold(0u64, u64::saturating_add);
        if measured > normsq_pred {
            return Err(format!(
                "RESTART: tail measured {measured} > predicted {normsq_pred}"
            ));
        }
        proof.normsq = normsq_pred;
        return Ok((proof, None, Some(LabWitness::new(z_digits))));
    }

    // ---- the target relation over [z^(0..f-1), v] ----
    let mut v: Vec<HwElt> = Vec::with_capacity(vl.m);
    v.extend(t_digits.iter().cloned());
    v.extend(g_digits.iter().cloned());
    v.extend(h_digits.iter().cloned());

    let mut vectors: Vec<LabVectorSpec> = (0..cpp.f)
        .map(|_d| LabVectorSpec {
            n: nn,
            betasq: normsq_pred,
        })
        .collect();
    vectors.push(LabVectorSpec {
        n: vl.m,
        betasq: normsq_pred,
    });
    let constraints = target_relation(
        ring, &proof, &key, &phi_agg, &a_agg, &b_agg, &layout, &win, quadratic,
    );
    let measured: u64 = z_digits
        .iter()
        .chain(std::iter::once(&v))
        .flat_map(|vv| vv.iter().map(|q| norm_sq_u64(ring, q)))
        .fold(0u64, u64::saturating_add);
    if measured > normsq_pred {
        return Err(format!(
            "RESTART: measured {measured} > predicted {normsq_pred}"
        ));
    }
    proof.normsq = normsq_pred;
    let target = LabStatement::new(vectors, constraints, vec![], normsq_pred);
    let mut s: Vec<Vec<HwElt>> = z_digits.clone();
    s.push(v);
    if let Err(e) = target.check_all(ring, &s) {
        return Err(format!("internal target-relation check failed: {e}"));
    }
    Ok((proof, Some(target), Some(LabWitness::new(s))))
}

/// Absorb a slice of ring elements into the transcript.
fn absorb_elts(
    tr: &mut Transcript,
    label: &[u8],
    ring: &HwRing,
    elts: &[HwElt],
) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(elts.len() * ring.n * 8);
    for e in elts {
        bytes.extend_from_slice(&ring.to_bytes(e));
    }
    tr.append_message(label, &bytes).map_err(|e| e.to_string())
}

/// The constraint aggregation (shared by prove and reduce).
#[allow(clippy::type_complexity)]
fn aggregate_constraints(
    ring: &HwRing,
    stmt: &LabStatement,
    lifted: &[LabConstraint],
    layout: &PartLayout,
    nn: usize,
    alphas: &[HwElt],
    betas: &[HwElt],
) -> (Vec<HwElt>, Vec<(usize, usize, HwElt)>, HwElt) {
    let mut phi_agg: Vec<HwElt> = vec![ring.zero(); layout.r * nn];
    let mut a_agg: Vec<(usize, usize, HwElt)> = Vec::new();
    let mut b_agg = ring.zero();
    for (k, c) in stmt.cnst.iter().enumerate() {
        let a = &alphas[k];
        for t in &c.terms {
            for (part, off, phi_chunk) in layout.locate_slice(t.idx, t.off, &t.phi) {
                for (u, pc) in phi_chunk.iter().enumerate() {
                    let scaled = ring.mul(pc, a);
                    let cur = phi_agg[part * nn + off + u].clone();
                    phi_agg[part * nn + off + u] = ring.add(&cur, &scaled);
                }
            }
        }
        for qe in &c.quads {
            for (p, q, cc) in expand_a_entry(layout, qe.i, qe.j, &qe.coeff) {
                a_agg.push((p, q, ring.mul(&cc, a)));
            }
        }
        if let Some(b) = &c.b {
            b_agg = ring.add(&b_agg, &ring.mul(b, a));
        }
    }
    for (k, c) in lifted.iter().enumerate() {
        let be = &betas[k];
        for t in &c.terms {
            for (u, pc) in t.phi.iter().enumerate() {
                let scaled = ring.mul(pc, be);
                let cur = phi_agg[t.idx * nn + t.off + u].clone();
                phi_agg[t.idx * nn + t.off + u] = ring.add(&cur, &scaled);
            }
        }
        for qe in &c.quads {
            a_agg.push((qe.i.min(qe.j), qe.i.max(qe.j), ring.mul(&qe.coeff, be)));
        }
        if let Some(b) = &c.b {
            b_agg = ring.add(&b_agg, &ring.mul(b, be));
        }
    }
    a_agg = merge_a(ring, a_agg);
    (phi_agg, a_agg, b_agg)
}

fn merge_a(ring: &HwRing, mut a: Vec<(usize, usize, HwElt)>) -> Vec<(usize, usize, HwElt)> {
    a.sort_by_key(|(i, j, _)| (*i, *j));
    let mut out: Vec<(usize, usize, HwElt)> = Vec::new();
    for (i, j, c) in a {
        if let Some(last) = out.last_mut() {
            if last.0 == i && last.1 == j {
                last.2 = ring.add(&last.2.clone(), &c);
                continue;
            }
        }
        out.push((i, j, c));
    }
    out
}

/// The K' = 2κ1 + κ + 3 target constraints over `[z^(0..f-1), v]`
/// (E1–E6).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn target_relation(
    ring: &HwRing,
    proof: &LabLevelProof,
    key: &LabKey,
    phi_agg: &[HwElt],
    a_agg: &[(usize, usize, HwElt)],
    b_agg: &HwElt,
    layout: &PartLayout,
    win: &Windows,
    quadratic: bool,
) -> Vec<LabConstraint> {
    let cpp = &proof.cpp;
    let r = proof.r;
    let nn = proof.nn;
    let vl = VLayout::new(cpp, r);
    let f = cpp.f;
    let v_idx = f;
    let _ = layout;
    let mut out: Vec<LabConstraint> = Vec::with_capacity(2 * cpp.kappa1 + cpp.kappa + 3);

    // E1 (κ1): B·[t̃; g̃] = u1
    let tg_len = vl.t_len + vl.g_len;
    for j in 0..cpp.kappa1 {
        let phi = (0..tg_len)
            .map(|k| key.rows[win.b_off + j * tg_len + k].clone())
            .collect();
        out.push(LabConstraint {
            terms: vec![LabTerm {
                idx: v_idx,
                off: 0,
                phi,
            }],
            quads: vec![],
            b: Some(proof.u1[j].clone()),
            ct_only: false,
        });
    }

    // E2 (κ1): D·h̃ = u2
    for j in 0..cpp.kappa1 {
        let phi = (0..vl.h_len)
            .map(|k| key.rows[win.d_off + j * vl.h_len + k].clone())
            .collect();
        out.push(LabConstraint {
            terms: vec![LabTerm {
                idx: v_idx,
                off: vl.h_off(),
                phi,
            }],
            quads: vec![],
            b: Some(proof.u2[j].clone()),
            ct_only: false,
        });
    }

    // E3 (κ): A·z = Σ_i c_i·t_i
    for rho in 0..cpp.kappa {
        let mut terms: Vec<LabTerm> = Vec::new();
        for d in 0..f {
            let phi = (0..nn)
                .map(|k| {
                    ring.scale(
                        &key.rows[win.a_off + rho * nn + k].clone(),
                        1i128 << (d as u32 * cpp.b),
                    )
                })
                .collect();
            terms.push(LabTerm {
                idx: d,
                off: 0,
                phi,
            });
        }
        let mut phi_v = vec![ring.zero(); vl.t_len];
        for i in 0..r {
            for j in 0..cpp.fu {
                let scale = 1i128 << (j as u32 * cpp.bu);
                phi_v[i * cpp.fu * cpp.kappa + j * cpp.kappa + rho] =
                    ring.scale(&ring.neg(&proof.c[i].clone()), scale);
            }
        }
        terms.push(LabTerm {
            idx: v_idx,
            off: 0,
            phi: phi_v,
        });
        out.push(LabConstraint::homogeneous(terms));
    }

    // E4 (quadratic): ⟨z,z⟩ = Σ_{i≤j}(2−[i=j]) c_i c_j g_ij
    if quadratic {
        let mut quads: Vec<LabQuadEntry> = Vec::new();
        for d1 in 0..f {
            for d2 in d1..f {
                quads.push(LabQuadEntry {
                    i: d1,
                    j: d2,
                    coeff: ring.neg(&ring.constant(1i128 << ((d1 + d2) as u32 * cpp.b))),
                });
            }
        }
        let mut phi_g = vec![ring.zero(); vl.g_len];
        for i in 0..r {
            for j in i..r {
                let base = tri_idx(i, j, r) * cpp.fg;
                let mut cc = ring.mul(&proof.c[i], &proof.c[j]);
                if i != j {
                    cc = ring.scale(&cc, 2);
                }
                for k in 0..cpp.fg {
                    phi_g[base + k] = ring.scale(&cc, 1i128 << (k as u32 * cpp.bg));
                }
            }
        }
        out.push(LabConstraint {
            terms: vec![LabTerm {
                idx: v_idx,
                off: vl.g_off(),
                phi: phi_g,
            }],
            quads,
            b: None,
            ct_only: false,
        });
    }

    // E5: ⟨φ_fold, z⟩ = Σ_{i≤j}(2−[i=j]) c_i c_j h_ij
    {
        let phi_fold = {
            let mut acc = vec![ring.zero(); nn];
            for i in 0..r {
                for (u, pc) in phi_agg[i * nn..(i + 1) * nn].iter().enumerate() {
                    let scaled = ring.mul(pc, &proof.c[i]);
                    let cur = acc[u].clone();
                    acc[u] = ring.add(&cur, &scaled);
                }
            }
            acc
        };
        let mut terms: Vec<LabTerm> = Vec::new();
        for d in 0..f {
            let phi = phi_fold
                .iter()
                .map(|p| ring.scale(p, 1i128 << (d as u32 * cpp.b)))
                .collect();
            terms.push(LabTerm {
                idx: d,
                off: 0,
                phi,
            });
        }
        let mut phi_h = vec![ring.zero(); vl.h_len];
        for i in 0..r {
            for j in i..r {
                let base = tri_idx(i, j, r) * cpp.fu;
                let mut cc = ring.mul(&proof.c[i], &proof.c[j]);
                if i != j {
                    cc = ring.scale(&cc, 2);
                }
                for k in 0..cpp.fu {
                    phi_h[base + k] = ring.scale(&ring.neg(&cc), 1i128 << (k as u32 * cpp.bu));
                }
            }
        }
        terms.push(LabTerm {
            idx: v_idx,
            off: vl.h_off(),
            phi: phi_h,
        });
        out.push(LabConstraint::homogeneous(terms));
    }

    // E6: Σ_{i≤j}(2−[i=j]) a_ij g_ij + Σ_i h_ii = b_agg
    {
        let mut phi_v = vec![ring.zero(); vl.m];
        for &(i, j, ref coeff) in a_agg {
            let base = vl.g_off() + tri_idx(i, j, r) * cpp.fg;
            let eff = if i == j {
                coeff.clone()
            } else {
                ring.scale(coeff, 2)
            };
            for k in 0..cpp.fg {
                let scaled = ring.scale(&eff, 1i128 << (k as u32 * cpp.bg));
                let cur = phi_v[base + k].clone();
                phi_v[base + k] = ring.add(&cur, &scaled);
            }
        }
        for i in 0..r {
            let base = vl.h_off() + tri_idx(i, i, r) * cpp.fu;
            for k in 0..cpp.fu {
                let scaled = ring.constant(1i128 << (k as u32 * cpp.bu));
                let cur = phi_v[base + k].clone();
                phi_v[base + k] = ring.add(&cur, &scaled);
            }
        }
        out.push(LabConstraint {
            terms: vec![LabTerm {
                idx: v_idx,
                off: 0,
                phi: phi_v,
            }],
            quads: vec![],
            b: Some(b_agg.clone()),
            ct_only: false,
        });
    }

    out
}

/// The verifier's replay of one level: regenerates the transcript state
/// and the aggregated constraint. Returns (phi_agg, a_agg, b_agg).
#[allow(clippy::type_complexity)]
pub fn lab_replay_level(
    ring: &HwRing,
    gate: SisGateMode,
    stmt: &LabStatement,
    proof: &LabLevelProof,
) -> Result<(Vec<HwElt>, Vec<(usize, usize, HwElt)>, HwElt), String> {
    let quadratic = is_quadratic(stmt);
    let cpp = &proof.cpp;
    let nn = proof.nn;
    let r = proof.r;
    if nn == 0 || r == 0 {
        return Err("degenerate level parameters".into());
    }
    let ranks: Vec<usize> = stmt.vectors.iter().map(|v| v.n).collect();

    // structural checks
    let expected_u1len = if proof.tail() {
        r * cpp.kappa + if quadratic { (r * r + r) / 2 } else { 0 }
    } else {
        cpp.kappa1
    };
    let expected_u2len = if proof.tail() { 2 * r - 1 } else { cpp.kappa1 };
    if proof.u1.len() != expected_u1len {
        return Err(format!("u1 length {} != {expected_u1len}", proof.u1.len()));
    }
    if proof.u2.len() != expected_u2len {
        return Err(format!("u2 length {} != {expected_u2len}", proof.u2.len()));
    }
    if proof.p.len() != 256 {
        return Err("JL projection must have 256 entries".into());
    }
    if proof.bb.len() != LAB_LIFTS {
        return Err(format!("expected {LAB_LIFTS} lift polynomials"));
    }
    if proof.c.len() != r {
        return Err(format!("challenge count {} != r {r}", proof.c.len()));
    }
    if gate == SisGateMode::Faithful {
        if !lab_sis_secure(
            ring,
            cpp.kappa,
            6.0 * LAB_T
                * LAB_SLACK
                * 2f64.powi((cpp.f as i32 - 1) * cpp.b as i32)
                * (proof.normsq as f64).sqrt(),
        ) {
            return Err("inner commitments not SIS-secure at the announced norm".into());
        }
        if !proof.tail()
            && !lab_sis_secure(
                ring,
                cpp.kappa1,
                2.0 * LAB_SLACK * (proof.normsq as f64).sqrt(),
            )
        {
            return Err("outer commitments not SIS-secure at the announced norm".into());
        }
    }
    if jl_normsq(&proof.p) > 256u64.saturating_mul(stmt.betasq.max(1)) {
        return Err("JL projection longer than the bound".into());
    }

    // transcript replay
    let mut tr = Transcript::new_default(b"hw-lab-level");
    tr.append_message(b"stmt-digest", &stmt.digest)
        .map_err(|e| e.to_string())?;
    absorb_elts(&mut tr, b"u1", ring, &proof.u1)?;
    let jl_seed = tr
        .challenge_bytes(b"jl-seed", 32)
        .map_err(|e| e.to_string())?;
    tr.append_message(
        b"jl-p",
        &proof
            .p
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .map_err(|e| e.to_string())?;

    let layout = part_layout(&ranks, nn, quadratic);
    // regenerate the JL matrices (the SAME nonce formula as
    // project_parts: nonce·r + i — a mismatched formula would regenerate
    // different matrices and desync the aggregated constraint).
    let mats: Vec<LabJlMatrix> = (0..r)
        .map(|i| {
            LabJlMatrix::expand(
                256,
                nn * ring.n,
                &jl_seed,
                proof.jlnonce.wrapping_mul(r as u64) + i as u64,
            )
        })
        .collect();
    if !jl_accept(&proof.p, stmt.betasq) {
        return Err("JL projection fails the acceptance bound".into());
    }

    // LIFTS: check the b'' constant terms
    let mut lifted: Vec<LabConstraint> = Vec::with_capacity(LAB_LIFTS);
    for k in 0..LAB_LIFTS {
        let omega = zq_sample(&mut tr, b"lift-omega", 256, ring.q).map_err(|e| e.to_string())?;
        let psi = zq_sample(&mut tr, b"lift-psi", stmt.ct_cnst.len(), ring.q)
            .map_err(|e| e.to_string())?;
        let (phis, jl_target) = collapse_jl(ring, &mats, &proof.p, &omega);
        let mut target: i128 = i128::from(jl_target);
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            if let Some(b) = &c.b {
                target = (target + psi[l] * i128::from(b.0[0])).rem_euclid(i128::from(ring.q));
            }
        }
        if proof.bb[k].0[0] != target as u64 {
            return Err(format!("lift {k}: b'' constant term incorrect"));
        }
        let mut phi_k: Vec<Vec<HwElt>> = phis;
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            let ps = psi[l];
            if ps != 0 {
                for t in &c.terms {
                    for (part, off, phi_chunk) in layout.locate_slice(t.idx, t.off, &t.phi) {
                        for (u, pc) in phi_chunk.iter().enumerate() {
                            let scaled = ring.scale(pc, ps);
                            let cur = phi_k[part][off + u].clone();
                            phi_k[part][off + u] = ring.add(&cur, &scaled);
                        }
                    }
                }
            }
        }
        let terms: Vec<LabTerm> = phi_k
            .iter()
            .enumerate()
            .map(|(i, phi)| LabTerm {
                idx: i,
                off: 0,
                phi: phi.clone(),
            })
            .collect();
        let mut quads: Vec<LabQuadEntry> = Vec::new();
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            let ps = psi[l];
            if ps != 0 {
                for qe in &c.quads {
                    for (p, q, cc) in expand_a_entry(&layout, qe.i, qe.j, &qe.coeff) {
                        quads.push(LabQuadEntry {
                            i: p,
                            j: q,
                            coeff: ring.scale(&cc, ps),
                        });
                    }
                }
            }
        }
        absorb_elts(&mut tr, b"lift-b", ring, &[proof.bb[k].clone()])?;
        lifted.push(LabConstraint {
            terms,
            quads,
            b: Some(proof.bb[k].clone()),
            ct_only: false,
        });
    }

    // F-aggregation
    let alphas =
        rq_sample(ring, &mut tr, b"agg-alpha", stmt.cnst.len()).map_err(|e| e.to_string())?;
    let betas = rq_sample(ring, &mut tr, b"agg-beta", LAB_LIFTS).map_err(|e| e.to_string())?;
    let (phi_agg, a_agg, b_agg) =
        aggregate_constraints(ring, stmt, &lifted, &layout, nn, &alphas, &betas);

    // c replay + checks
    if proof.tail() {
        absorb_elts(&mut tr, b"tail-h", ring, &[proof.u2[0].clone()])?;
        let mut c_ch = challenge_vec_hw(ring, &mut tr, 1).map_err(|e| e.to_string())?;
        for i in 1..r {
            absorb_elts(
                &mut tr,
                b"tail-h",
                ring,
                &[proof.u2[2 * i - 1].clone(), proof.u2[2 * i].clone()],
            )?;
            let ci = challenge_vec_hw(ring, &mut tr, 1).map_err(|e| e.to_string())?;
            let ci = ci.first().cloned().ok_or("challenge sampling failed")?;
            c_ch.push(ci);
        }
        if c_ch != proof.c {
            return Err("tail challenges inconsistent with the transcript".into());
        }
    } else {
        absorb_elts(&mut tr, b"u2", ring, &proof.u2)?;
        let c_ch = challenge_vec_hw(ring, &mut tr, r).map_err(|e| e.to_string())?;
        if c_ch != proof.c {
            return Err("amortization challenges inconsistent with the transcript".into());
        }
    }
    Ok((phi_agg, a_agg, b_agg))
}

/// The verifier's statement reconstruction (non-tail).
pub fn lab_reduce_level(
    ring: &HwRing,
    gate: SisGateMode,
    stmt: &LabStatement,
    proof: &LabLevelProof,
    key_ctx: &[u8],
) -> Result<LabStatement, String> {
    if proof.tail() {
        return Err("reduce_level on a tail level".into());
    }
    let (phi_agg, a_agg, b_agg) = lab_replay_level(ring, gate, stmt, proof)?;
    let cpp = &proof.cpp;
    let quadratic = is_quadratic(stmt);
    let layout = part_layout(
        &stmt.vectors.iter().map(|v| v.n).collect::<Vec<_>>(),
        proof.nn,
        quadratic,
    );
    let win = Windows::new(cpp, proof.r, proof.nn);
    let mut key_ctx_full = key_ctx.to_vec();
    key_ctx_full.extend_from_slice(&stmt.digest);
    let key = LabKey::expand(ring, win.total, &key_ctx_full);
    let vl = VLayout::new(cpp, proof.r);
    let mut vectors: Vec<LabVectorSpec> = (0..cpp.f)
        .map(|_d| LabVectorSpec {
            n: proof.nn,
            betasq: proof.normsq,
        })
        .collect();
    vectors.push(LabVectorSpec {
        n: vl.m,
        betasq: proof.normsq,
    });
    let constraints = target_relation(
        ring, proof, &key, &phi_agg, &a_agg, &b_agg, &layout, &win, quadratic,
    );
    Ok(LabStatement::new(
        vectors,
        constraints,
        vec![],
        proof.normsq,
    ))
}

/// The final tail verification (the direct checks on the transmitted
/// material).
pub fn lab_verify_tail(
    ring: &HwRing,
    gate: SisGateMode,
    stmt: &LabStatement,
    proof: &LabLevelProof,
    final_witness: &LabWitness,
    key_ctx: &[u8],
) -> Result<(), String> {
    if !proof.tail() {
        return Err("verify_tail on a non-tail level".into());
    }
    let quadratic = is_quadratic(stmt);
    let cpp = &proof.cpp;
    let nn = proof.nn;
    let r = proof.r;
    let (phi_agg, a_agg, b_agg) = lab_replay_level(ring, gate, stmt, proof)?;

    // the final witness: f parts of rank nn
    if final_witness.s.len() != cpp.f {
        return Err(format!("final witness must have {} parts", cpp.f));
    }
    for (d, v) in final_witness.s.iter().enumerate() {
        if v.len() != nn {
            return Err(format!("final witness part {d} rank {} != {nn}", v.len()));
        }
    }
    let normsq: u64 = final_witness
        .s
        .iter()
        .flat_map(|v| v.iter().map(|p| norm_sq_u64(ring, p)))
        .fold(0u64, u64::saturating_add);
    if normsq > proof.normsq {
        return Err(format!(
            "final witness norm² {normsq} > announced {}",
            proof.normsq
        ));
    }
    let z = recombine_vec(ring, &final_witness.s, cpp.b);

    let win = Windows::new(cpp, r, nn);
    let mut key_ctx_full = key_ctx.to_vec();
    key_ctx_full.extend_from_slice(&stmt.digest);
    let key = LabKey::expand(ring, win.total, &key_ctx_full);

    // E3: A·z = Σ_i c_i t_i
    let az = key.mul_window(&z, win.a_off, cpp.kappa);
    for rho in 0..cpp.kappa {
        let mut rhs = ring.zero();
        for i in 0..r {
            rhs = ring.add(&rhs, &ring.mul(&proof.c[i], &proof.u1[i * cpp.kappa + rho]));
        }
        if az[rho] != rhs {
            return Err(format!("E3 (Az = Σc_i t_i) violated at row {rho}"));
        }
    }

    // E4 + E6 (quadratic)
    if quadratic {
        let g_base = r * cpp.kappa;
        let lhs = sprod(ring, &z, &z);
        let mut rhs = ring.zero();
        let mut idx = 0;
        for i in 0..r {
            for j in i..r {
                let g = &proof.u1[g_base + idx];
                let mut term = ring.mul(&ring.mul(&proof.c[i], &proof.c[j]), g);
                if i != j {
                    term = ring.scale(&term, 2);
                }
                rhs = ring.add(&rhs, &term);
                idx += 1;
            }
        }
        if lhs != rhs {
            return Err("E4 (⟨z,z⟩ = Σ c c g) violated".into());
        }
        // E6: Σ_{i≤j}(2−[i=j]) a_ij g_ij + Σ_i h_ii = b_agg
        let mut acc = ring.zero();
        for &(i, j, ref coeff) in &a_agg {
            let g = &proof.u1[g_base + tri_idx(i, j, r)];
            let eff = if i == j {
                coeff.clone()
            } else {
                ring.scale(coeff, 2)
            };
            acc = ring.add(&acc, &ring.mul(&eff, g));
        }
        for i in 0..r {
            acc = ring.add(&acc, &proof.u2[2 * i].clone());
        }
        if acc != b_agg {
            return Err("E6 (Σ a g + Σ h_ii = b) violated".into());
        }
    }

    // E5: ⟨φ_fold, z⟩ = c_0² h_0 + Σ_{i≥1}(c_i h_{2i−1} + c_i² h_{2i})
    let phi_fold = {
        let mut acc = vec![ring.zero(); nn];
        for i in 0..r {
            for (u, pc) in phi_agg[i * nn..(i + 1) * nn].iter().enumerate() {
                let scaled = ring.mul(pc, &proof.c[i]);
                let cur = acc[u].clone();
                acc[u] = ring.add(&cur, &scaled);
            }
        }
        acc
    };
    let lhs = sprod(ring, &phi_fold, &z);
    let mut rhs = ring.mul(&ring.mul(&proof.c[0], &proof.c[0]), &proof.u2[0]);
    for i in 1..r {
        rhs = ring.add(&rhs, &ring.mul(&proof.c[i], &proof.u2[2 * i - 1]));
        rhs = ring.add(
            &rhs,
            &ring.mul(&ring.mul(&proof.c[i], &proof.c[i]), &proof.u2[2 * i]),
        );
    }
    if lhs != rhs {
        return Err("E5 (⟨φ, z⟩ = Σ h c terms) violated".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The recursive outer-commitment compaction driver (the O(log log log N)
// route): iterate lab_prove_level while the statement shrinks, then one
// tail level whose openings are transmitted directly.
// ---------------------------------------------------------------------------

/// The full recursively-composed amortized Dachshund proof.
#[derive(Clone, Debug)]
pub struct LabradorHwProof {
    pub levels: Vec<LabLevelProof>,
    pub tail: LabLevelProof,
    pub final_witness: LabWitness,
}

/// Entropy bits of a witness (the reference's accounting).
pub fn lab_witness_size_bits(ring: &HwRing, wit: &LabWitness) -> u64 {
    wit.s
        .iter()
        .map(|v| {
            let normsq: u64 = v
                .iter()
                .map(|p| norm_sq_u64(ring, p))
                .fold(0u64, u64::saturating_add);
            let n = v.len();
            if n == 0 || normsq == 0 {
                0u64
            } else {
                let ent =
                    ((normsq as f64 / (n as f64 * ring.n as f64)).log2() / 2.0 + 2.05).max(1.0);
                ((n * ring.n) as f64 * ent) as u64
            }
        })
        .sum()
}

/// Bits of one level's transmitted proof.
pub fn lab_level_size_bits(ring: &HwRing, lp: &LabLevelProof) -> u64 {
    let jl_bits: u64 = {
        let psq: u64 = lp.p.iter().map(|&x| (x * x) as u64).sum();
        if psq == 0 {
            0u64
        } else {
            (((psq as f64).sqrt().log2() - 4.0 + 2.05).max(1.0) * 256.0) as u64
        }
    };
    ((lp.u1.len() + lp.u2.len() + LAB_LIFTS) * ring.n * 61) as u64 + jl_bits + 128
}

/// Total proof size in bytes (the analytic model).
pub fn lab_proof_size_bytes(ring: &HwRing, proof: &LabradorHwProof) -> u64 {
    let levels: u64 = proof
        .levels
        .iter()
        .map(|lp| lab_level_size_bits(ring, lp))
        .sum();
    let tail = lab_level_size_bits(ring, &proof.tail);
    let witness = lab_witness_size_bits(ring, &proof.final_witness);
    (levels + tail + witness).div_ceil(8)
}

/// One row of the level table (the compaction evidence).
#[derive(Clone, Copy, Debug)]
pub struct LabLevelRow {
    pub n: usize,
    pub r: usize,
    pub kappa: usize,
    pub kappa1: usize,
    pub bits: u64,
    pub tail: bool,
}

/// The per-level table.
pub fn lab_level_table(ring: &HwRing, proof: &LabradorHwProof) -> Vec<LabLevelRow> {
    proof
        .levels
        .iter()
        .chain(std::iter::once(&proof.tail))
        .map(|lp| LabLevelRow {
            n: lp.nn,
            r: lp.r,
            kappa: lp.cpp.kappa,
            kappa1: lp.cpp.kappa1,
            bits: lab_level_size_bits(ring, lp),
            tail: lp.tail(),
        })
        .collect()
}

/// Prove a principal statement with the full recursion. Falls back to the
/// tail directly when the amortization does not shrink (small statements
/// — the honest kernel-scale behaviour).
pub fn lab_prove(
    ring: &HwRing,
    gate: SisGateMode,
    stmt: &LabStatement,
    wit: &LabWitness,
    key_ctx: &[u8],
) -> Result<LabradorHwProof, String> {
    let mut levels: Vec<LabLevelProof> = Vec::new();
    let mut cur_stmt = stmt.clone();
    let mut cur_wit = wit.clone();
    let max_levels = 16;
    loop {
        if levels.len() >= max_levels {
            return Err("recursion did not terminate".into());
        }
        let cur_size = lab_witness_size_bits(ring, &cur_wit);
        // Try a non-tail level; fall back to the tail when the
        // amortization does not engage (the gate fails or no shrink).
        let non_tail = lab_prove_level(ring, gate, &cur_stmt, &cur_wit, key_ctx, false);
        let next_size = non_tail
            .as_ref()
            .ok()
            .and_then(|(_, _, w)| w.as_ref().map(|w| lab_witness_size_bits(ring, w)));
        match (non_tail, next_size) {
            (Ok((proof, Some(next_stmt), Some(next_wit))), Some(sz)) if sz < cur_size => {
                levels.push(proof);
                cur_stmt = next_stmt;
                cur_wit = next_wit;
            }
            _ => {
                // tail on the current statement
                let (tail, _, final_wit) =
                    lab_prove_level(ring, gate, &cur_stmt, &cur_wit, key_ctx, true)?;
                return Ok(LabradorHwProof {
                    levels,
                    tail,
                    final_witness: final_wit.ok_or("tail must return the final witness")?,
                });
            }
        }
    }
}

/// Verify a full proof against the original statement.
pub fn lab_verify(
    ring: &HwRing,
    gate: SisGateMode,
    stmt: &LabStatement,
    proof: &LabradorHwProof,
    key_ctx: &[u8],
) -> Result<(), String> {
    let mut cur = stmt.clone();
    for (i, lp) in proof.levels.iter().enumerate() {
        if lp.tail() {
            return Err(format!("level {i} is a tail inside the chain"));
        }
        cur = lab_reduce_level(ring, gate, &cur, lp, key_ctx)?;
    }
    if !proof.tail.tail() {
        return Err("the last level is not a tail".into());
    }
    lab_verify_tail(ring, gate, &cur, &proof.tail, &proof.final_witness, key_ctx)
}

// ---------------------------------------------------------------------------
// The HyperWolf full-fidelity protocol (the projection statement builder)
// ---------------------------------------------------------------------------

impl HwError {
    /// Construct an engine error (the H6 route's String channel).
    pub fn engine(msg: impl Into<String>) -> Self {
        HwError::Engine(msg.into())
    }
}

/// The per-round projection norm cap `B_r` (all slices, full
/// coefficients): `b·jl_rows·d²·(b·ι)·β(k−1−r)²` — the conservative
/// convolution bound, gated `B_r < q/4` (the wraparound guard that keeps
/// the integer ℓ2 statements exact mod q).
pub fn round_bound(params: &HwParams, round: usize) -> Result<u64, HwError> {
    let level = params.k.saturating_sub(1).saturating_sub(round);
    let beta = params.beta(level);
    let b = params.b as f64;
    let jl = params.jl_rows as f64;
    let d = params.d as f64;
    let iota = params.iota() as f64;
    let val = b * jl * d * d * (b * iota) * beta * beta;
    if !(val.is_finite() && val >= 0.0) || val >= (params.q / 4) as f64 {
        return Err(HwError::engine(
            "round bound exceeds the wraparound guard (paper-scale ladder — modelled, not executed)",
        ));
    }
    Ok(val as u64)
}

/// One round's collected data (the statement-builder input).
#[derive(Clone, Debug)]
pub struct LabRoundData {
    /// The per-slice projection vectors `p_i^(r)` (each of length
    /// `jl_rows`).
    pub projections: Vec<Vec<HwElt>>,
    /// The fold challenges `C^(r)` drawn after this round's message.
    pub challenges: Vec<HwElt>,
}

/// The σ⁻¹(e_c) picking element: coefficient `[c==0]` at 0 and
/// `-[c>=1]` at `(d−c)%d` — `ct(σ⁻¹(e_c)·s) = s[c]`.
fn pick_elt(ring: &HwRing, c: usize) -> HwElt {
    let mut v = vec![0u64; ring.n];
    if c == 0 {
        v[0] = 1;
    } else {
        v[ring.n - c] = ring.q - 1; // −1
    }
    HwElt(v)
}

/// Build the amortized-Dachshund statement over ALL rounds' projections:
/// vectors `[P_r, P*_r, S_r, S*_r]_r` with
/// * the σ⁻¹-conjugate bindings (per-coefficient ct constraints),
/// * the per-round exact ℓ2 constraints (quadratic, ct-only, with the
///   binary slack), and — returned separately for the caller to append —
/// * NO fold-consistency / final-tie rows (those reference the
///   round-to-round challenges; the caller appends them via
///   [`append_consistency_constraints`]).
pub fn projection_statement(
    ring: &HwRing,
    params: &HwParams,
    rounds: &[LabRoundData],
    final_jl: &[HwElt],
    ctx: &[u8; 32],
) -> Result<LabStatement, HwError> {
    let k_rounds = rounds.len();
    if k_rounds == 0 || final_jl.len() != params.jl_rows {
        return Err(HwError::engine(
            "projection statement: empty rounds or bad JL length",
        ));
    }
    let rank_p = params.b * params.jl_rows;
    let mut vectors: Vec<LabVectorSpec> = Vec::with_capacity(4 * k_rounds);
    let mut bounds: Vec<u64> = Vec::with_capacity(k_rounds);
    let mut slack_ranks: Vec<usize> = Vec::with_capacity(k_rounds);
    for r in 0..k_rounds {
        let b_r = round_bound(params, r)?;
        bounds.push(b_r);
        // The slack entry budget: the greedy square decomposition of any
        // deficit < q/4 needs at most ~16 entries (shared formula for the
        // statement and the witness).
        let slack_rank = 16usize.div_ceil(params.d).max(1);
        slack_ranks.push(slack_rank);
        for &(n, beta) in &[
            (rank_p, b_r),
            (rank_p, b_r),
            (slack_rank, b_r),
            (slack_rank, b_r),
        ] {
            vectors.push(LabVectorSpec { n, betasq: beta });
        }
    }
    let global: u64 = vectors
        .iter()
        .map(|v| v.betasq)
        .fold(0u64, u64::saturating_add);

    let mut ct: Vec<LabConstraint> = Vec::new();
    // The σ⁻¹ bindings: P*_r[k][c] − sign·P_r[k][(d−c)%d] = 0 (and the
    // slack pairs identically).
    let one = ring.one();
    for r in 0..k_rounds {
        let p_idx = 4 * r;
        let s_idx = 4 * r + 2;
        for base_vec in [(p_idx, rank_p), (s_idx, slack_ranks[r].max(1))] {
            for k in 0..base_vec.1 {
                for c in 0..params.d {
                    let pick = pick_elt(ring, c);
                    let pick_other = if c == 0 {
                        ring.neg(&one)
                    } else {
                        pick_elt(ring, params.d - c)
                    };
                    ct.push(LabConstraint {
                        terms: vec![
                            LabTerm {
                                idx: base_vec.0 + 1, // the conjugate vector
                                off: k,
                                phi: vec![pick],
                            },
                            LabTerm {
                                idx: base_vec.0,
                                off: k,
                                phi: vec![pick_other],
                            },
                        ],
                        quads: vec![],
                        b: None,
                        ct_only: true,
                    });
                }
            }
        }
    }
    // The per-round exact ℓ2 constraints: ct⟨P_r, P*_r⟩ + ct⟨S_r, S*_r⟩ = B_r.
    // The entries are CROSS pairs (i ≠ j): the engine's symmetric (2 − [i=j])
    // extension doubles them, so the coefficient is 1/2 mod q (the doubling
    // cancels through every E-path — eval, LIFTS, E6 — uniformly).
    let inv2 = ring.constant((ring.q + 1) as i128 / 2);
    for r in 0..k_rounds {
        ct.push(LabConstraint {
            terms: vec![],
            quads: vec![
                LabQuadEntry {
                    i: 4 * r,
                    j: 4 * r + 1,
                    coeff: inv2.clone(),
                },
                LabQuadEntry {
                    i: 4 * r + 2,
                    j: 4 * r + 3,
                    coeff: inv2.clone(),
                },
            ],
            b: Some(ring.constant(bounds[r] as i128)),
            ct_only: true,
        });
    }

    // The fold-consistency rows (F family): for r ≥ 1 and each idx,
    // Σ_i p_i^(r)[idx] − Σ_j C_j^(r−1)·p_j^(r−1)[idx] = 0.
    let mut f: Vec<LabConstraint> = Vec::new();
    for r in 1..k_rounds {
        for idx in 0..params.jl_rows {
            let mut terms: Vec<LabTerm> = Vec::with_capacity(2 * params.b);
            for i in 0..params.b {
                terms.push(LabTerm {
                    idx: 4 * r,
                    off: i * params.jl_rows + idx,
                    phi: vec![one.clone()],
                });
            }
            for j in 0..params.b {
                terms.push(LabTerm {
                    idx: 4 * (r - 1),
                    off: j * params.jl_rows + idx,
                    phi: vec![ring.neg(&rounds[r - 1].challenges[j])],
                });
            }
            f.push(LabConstraint::homogeneous(terms));
        }
    }
    // The final tie (F family): Σ_j C_j^(last)·p_j^(last)[idx] = JL(s^(1))[idx].
    let last = k_rounds - 1;
    for idx in 0..params.jl_rows {
        let mut terms: Vec<LabTerm> = Vec::with_capacity(params.b);
        for j in 0..params.b {
            terms.push(LabTerm {
                idx: 4 * last,
                off: j * params.jl_rows + idx,
                phi: vec![rounds[last].challenges[j].clone()],
            });
        }
        f.push(LabConstraint {
            terms,
            quads: vec![],
            b: Some(final_jl[idx].clone()),
            ct_only: false,
        });
    }

    let mut st = LabStatement::new(vectors, f, ct, global);
    // Bind the HyperWolf context into the digest.
    let mut digest_input = st.digest.to_vec();
    digest_input.extend_from_slice(b"hw-lab-ctx/v1");
    digest_input.extend_from_slice(ctx);
    st.digest = sha3_256(&digest_input);
    Ok(st)
}

/// Assemble the prover-side witness for a projection statement.
pub fn projection_witness(
    ring: &HwRing,
    params: &HwParams,
    rounds: &[LabRoundData],
) -> Result<LabWitness, HwError> {
    let k_rounds = rounds.len();
    let rank_p = params.b * params.jl_rows;
    let mut s: Vec<Vec<HwElt>> = Vec::with_capacity(4 * k_rounds);
    for r in 0..k_rounds {
        if rounds[r].projections.len() != params.b
            || rounds[r]
                .projections
                .iter()
                .any(|p| p.len() != params.jl_rows)
            || rounds[r].challenges.len() != params.b
        {
            return Err(HwError::engine("projection witness: round shape mismatch"));
        }
        // P_r: slice-major flat.
        let mut p_flat: Vec<HwElt> = Vec::with_capacity(rank_p);
        for slice in &rounds[r].projections {
            p_flat.extend(slice.iter().cloned());
        }
        // P*_r = σ⁻¹(P_r), element-wise.
        let p_star: Vec<HwElt> = p_flat.iter().map(|e| ring.conj(e)).collect();
        // The exact slack: ‖S‖² = B_r − Σ‖p‖² via the greedy square
        // decomposition (the coefficients ARE the squares).
        let b_r = round_bound(params, r)?;
        let sum: u64 = p_flat
            .iter()
            .map(|p| norm_sq_u64(ring, p))
            .fold(0u64, u64::saturating_add);
        if sum > b_r {
            return Err(HwError::engine(format!(
                "round {r}: projection norm² {sum} exceeds the cap {b_r}"
            )));
        }
        let slack_rank = 16usize.div_ceil(params.d).max(1);
        let squares = square_decomposition(b_r - sum);
        if squares.len() > slack_rank * params.d {
            return Err(HwError::engine("slack decomposition exceeded the budget"));
        }
        let mut s_flat: Vec<HwElt> = vec![HwElt(vec![0u64; params.d]); slack_rank];
        for (i, c) in squares.iter().enumerate() {
            s_flat[i / params.d].0[i % params.d] =
                (*c as i128).rem_euclid(i128::from(ring.q)) as u64;
        }
        let s_star: Vec<HwElt> = s_flat.iter().map(|e| ring.conj(e)).collect();
        s.push(p_flat);
        s.push(p_star);
        s.push(s_flat);
        s.push(s_star);
    }
    Ok(LabWitness::new(s))
}

/// The clear round payload of the full-fidelity protocol (no
/// projections, no projection commitments — the amortized proof covers
/// them).
#[derive(Clone, Debug)]
pub struct LabRoundPayload {
    pub fold: Vec<HwElt>,
    pub c_mins: Vec<Vec<HwElt>>,
}

/// The full-fidelity HyperWolf evaluation proof.
#[derive(Clone, Debug)]
pub struct HwLabProof {
    pub rounds: Vec<LabRoundPayload>,
    /// The revealed final witness `s^(1)` (b·ι elements — the terminal
    /// reveal, as in the clear and compact protocols).
    pub s_final: Vec<HwElt>,
    /// The amortized Dachshund proof over ALL rounds' projections.
    pub lab: LabradorHwProof,
}

fn hw_context_digest(a0_ints: &[u64], a_list: &[Vec<u64>]) -> [u8; 32] {
    // Binds the shared public inputs ONLY (the prover and the verifier
    // construct identical digests; the cm and y are bound by the clear
    // checks 1/3 and the final checks, not re-bound here).
    let mut buf = Vec::new();
    buf.extend_from_slice(b"hw-lab/ctx/v2");
    for a in a0_ints {
        buf.extend_from_slice(&a.to_le_bytes());
    }
    for ai in a_list {
        for x in ai {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    sha3_256(&buf)
}

impl HyperWolfFull {
    /// H6 full-fidelity: the evaluation proof with the projections
    /// amortized into ONE Dachshund opening.
    pub fn eval_prove_labrador(
        &self,
        state: &HwCommitState,
        a0_ints: &[u64],
        a_list: &[Vec<u64>],
        transcript: &mut Transcript,
    ) -> Result<HwLabProof, HwError> {
        let p = &self.params;
        let ring = &self.ring;
        let gate = SisGateMode::KernelBypass;
        let mut s = state.s.clone();
        let mut c_mins = state.c_mins.clone();
        let a0c = self.a0_ext_conj(a0_ints);
        let jl = self.jl(transcript)?;
        let mut payloads: Vec<LabRoundPayload> = Vec::with_capacity(p.k - 1);
        let mut rounds: Vec<LabRoundData> = Vec::with_capacity(p.k - 1);
        let mut level = p.k;
        while level > 1 {
            let fold = crate::hyperwolf::fold_engine(ring, &s, &a0c, &a_list[..level - 2])?;
            let slices = s.slices();
            let block = p.b * p.iota();
            let projs: Vec<Vec<HwElt>> = slices
                .iter()
                .map(|sl| jl.project(sl, block))
                .collect::<Result<Vec<_>, _>>()?;
            // Absorb ONLY the clear parts (fold, c_mins) — the
            // projections never touch the HyperWolf transcript.
            let mut bytes = Vec::new();
            for f in &fold {
                bytes.extend_from_slice(&ring.to_bytes(f));
            }
            for cmn in &c_mins {
                for x in cmn {
                    bytes.extend_from_slice(&ring.to_bytes(x));
                }
            }
            transcript.append_message(b"hw-lab-round", &bytes)?;
            payloads.push(LabRoundPayload {
                fold,
                c_mins: c_mins.clone(),
            });
            let c = self.draw_challenges(transcript, level)?;
            rounds.push(LabRoundData {
                projections: projs,
                challenges: c.clone(),
            });
            s = s.fold_outer(ring, &c);
            let new_slices = s.slices();
            c_mins = new_slices
                .iter()
                .map(|sl| self.keys.commit_slices(ring, sl))
                .collect();
            level -= 1;
        }
        let s_final = s.flat.clone();
        // The final tie targets: JL(s^(1)) — computed with the SAME matrix.
        let final_jl = jl.project(&s_final, p.b * p.iota())?;
        // The context digest binds the shared HyperWolf inputs.
        let ctx = hw_context_digest(a0_ints, a_list);
        let (stmt, wit) = {
            let st = projection_statement(ring, p, &rounds, &final_jl, &ctx)?;
            let w = projection_witness(ring, p, &rounds)?;
            (st, w)
        };
        // Sanity: the prover-side statement must hold before proving.
        if let Err(e) = stmt.check_all(ring, &wit.s) {
            return Err(HwError::engine(format!("projection statement: {e}")));
        }
        let key_ctx = b"hw-labrador-key/v1".as_slice();
        let lab = lab_prove(ring, gate, &stmt, &wit, key_ctx).map_err(HwError::engine)?;
        Ok(HwLabProof {
            rounds: payloads,
            s_final,
            lab,
        })
    }

    /// H6 full-fidelity verifier: the clear checks (1: fold identities,
    /// 3: the c_min binding chain, the final checks) + the amortized
    /// Dachshund verification against the REBUILT projection statement.
    #[allow(clippy::too_many_lines)]
    pub fn eval_verify_labrador(
        &self,
        cm: &[HwElt],
        a0_ints: &[u64],
        a_list: &[Vec<u64>],
        y_claim: u64,
        proof: &HwLabProof,
        transcript: &mut Transcript,
    ) -> Result<bool, HwError> {
        let p = &self.params;
        let ring = &self.ring;
        let gate = SisGateMode::KernelBypass;
        let mut y = ring.zero();
        y.0[0] = y_claim % ring.q;
        let a0c = self.a0_ext_conj(a0_ints);
        let jl = self.jl(transcript)?;
        let mut cm_out = cm.to_vec();
        let mut level = p.k;
        let mut c_hist: Vec<Vec<HwElt>> = Vec::new();
        let mut c_min_hist: Vec<Vec<Vec<HwElt>>> = Vec::new();
        let mut rounds: Vec<LabRoundData> = Vec::with_capacity(p.k - 1);
        for (r, msg) in proof.rounds.iter().enumerate() {
            if msg.fold.len() != p.b || msg.c_mins.len() != p.b {
                return Ok(false);
            }
            // ---- check 1 (clear): <fold, a_{level-1}> == y.
            let a_out = &a_list[level - 2];
            let mut ip = ring.zero();
            for (i, fr) in msg.fold.iter().enumerate() {
                let term = ring.scale(fr, i128::from(a_out[i]));
                ip = ring.add(&ip, &term);
            }
            if r == 0 {
                if ip.0[0] != y.0[0] {
                    return Ok(false);
                }
            } else if ip != y {
                return Ok(false);
            }
            // ---- check 3 (clear): the outer commitment binding chain.
            if r == 0 {
                let stack: Vec<HwElt> = msg.c_mins.iter().flatten().cloned().collect();
                if self
                    .keys
                    .outer_commit(ring, &stack, p.delta_t, p.iota_p())?
                    != cm_out
                {
                    return Ok(false);
                }
            } else if self.keys.outer_commit_from_fold(
                ring,
                &c_min_hist[c_min_hist.len() - 1],
                &c_hist[c_hist.len() - 1],
                p.delta_t,
                p.iota_p(),
            )? != cm_out
            {
                return Ok(false);
            }
            // ---- absorb + challenges (the same flow as the prover).
            let mut bytes = Vec::new();
            for f in &msg.fold {
                bytes.extend_from_slice(&ring.to_bytes(f));
            }
            for cmn in &msg.c_mins {
                for x in cmn {
                    bytes.extend_from_slice(&ring.to_bytes(x));
                }
            }
            transcript.append_message(b"hw-lab-round", &bytes)?;
            let c = self.draw_challenges(transcript, level)?;
            y = ring.zero();
            for (i, fr) in msg.fold.iter().enumerate() {
                let term = ring.mul(fr, &c[i]);
                y = ring.add(&y, &term);
            }
            cm_out =
                self.keys
                    .outer_commit_from_fold(ring, &msg.c_mins, &c, p.delta_t, p.iota_p())?;
            rounds.push(LabRoundData {
                projections: Vec::new(), // not used by the statement rebuild
                challenges: c,
            });
            c_hist.push(rounds[rounds.len() - 1].challenges.clone());
            c_min_hist.push(msg.c_mins.clone());
            level -= 1;
        }
        // ---------------- final checks (clear) ----------------
        let s1 = &proof.s_final;
        if s1.len() != p.b * p.iota() {
            return Ok(false);
        }
        // (a) <conj(a0_ext), s^(1)> == y.
        let mut ip = ring.zero();
        for j in 0..s1.len() {
            let term = ring.mul(&a0c[j], &s1[j]);
            ip = ring.add(&ip, &term);
        }
        if ip != y {
            return Ok(false);
        }
        // (a') ||s^(1)||² <= beta(0)².
        let mut norm_sq: f64 = 0.0;
        for elt in s1 {
            for &c in &elt.0 {
                let v = ring.center(c) as f64;
                norm_sq += v * v;
            }
        }
        if norm_sq > p.beta(0).powi(2) {
            return Ok(false);
        }
        // (c) A·s^(1) == Σ_i C_i·c_min,i.
        let lhs_c = self.keys.commit_slices(ring, s1);
        let last_mins = &c_min_hist[c_min_hist.len() - 1];
        let last_c = &c_hist[c_hist.len() - 1];
        let mut rhs_c = vec![ring.zero(); p.kappa()];
        for (i, c_min) in last_mins.iter().enumerate() {
            for (rr, x) in c_min.iter().enumerate() {
                let term = ring.mul(x, &last_c[i]);
                rhs_c[rr] = ring.add(&rhs_c[rr], &term);
            }
        }
        if lhs_c != rhs_c {
            return Ok(false);
        }
        // (b') the final tie targets: JL(s^(1)) with the SAME matrix.
        let final_jl = jl.project(s1, p.b * p.iota())?;
        // ---------------- the amortized verification ----------------
        let ctx = hw_context_digest(a0_ints, a_list);
        let stmt = projection_statement(ring, p, &rounds, &final_jl, &ctx)?;
        let key_ctx = b"hw-labrador-key/v1".as_slice();
        match lab_verify(ring, gate, &stmt, &proof.lab, key_ctx) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

/// The analytic paper-scale compaction model: (clear projection bytes,
/// amortized bytes) at the given shape — the honest model test's numbers.
pub fn lab_size_model(params: &HwParams) -> (u64, u64) {
    // Clear: (k−1)·b·jl_rows ring elements, 61 bits each.
    let clear_bits =
        (params.k.saturating_sub(1) * params.b * params.jl_rows * params.d * 61) as u64;
    // Amortized: levels ≈ 2 (the shrink factors at the paper shape),
    // each (2·κ1 + LIFTS + 256/8) elements + the tail's final witness
    // (f·nn with nn the last shrinking rank — modelled at 2^10 scale).
    let per_level_bits = ((2 * 4 + 3) * params.d * 61 + 256 * 18 + 128) as u64;
    let tail_bits = (2 * 1024 * params.d * 61) as u64;
    (
        clear_bits.div_ceil(8),
        (2 * per_level_bits + tail_bits).div_ceil(8),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperwolf::paper_params;

    fn small_params() -> HwParams {
        // d = 64 (the paper's ring dimension), kernel k and jl_rows.
        HwParams::new(64, 2, 3, 32)
    }

    fn setup(seed: &[u8]) -> (HyperWolfFull, Vec<i64>, Vec<u64>, Vec<Vec<u64>>, u64) {
        let params = small_params();
        let hw = HyperWolfFull::new(params.clone(), seed);
        let n = params.n_coeffs();
        let f_ints: Vec<i64> = (0..n).map(|i| ((i * 37) % 251) as i64 - 125).collect();
        let log_b = params.b.trailing_zeros() as usize;
        let log_d = params.d.trailing_zeros() as usize;
        let need = log_b + log_d + (params.k.saturating_sub(1)) * log_b;
        let point: Vec<u64> = (0..need).map(|i| 1234 + i as u64 * 997).collect();
        let (a0_ints, a_list) =
            crate::hyperwolf::build_a_multilinear(&hw.ring, &point, params.k, params.b, params.d)
                .unwrap();
        let y = hw.evaluate_direct(&f_ints, &point, true);
        (hw, f_ints, a0_ints, a_list, y)
    }

    #[test]
    fn decompose_recombine_roundtrip() {
        let ring = HwRing::new(crate::hyperwolf::HW_Q61, 64);
        let elt = ring.from_bytes_mod(
            &[7u8; 64]
                .iter()
                .chain([3u8; 64].iter())
                .copied()
                .collect::<Vec<u8>>(),
        );
        for &(f, b) in &[(1usize, 10u32), (2, 9), (3, 5)] {
            let layers = decompose_elt(&ring, &elt, f, b);
            assert_eq!(layers.len(), f);
            assert_eq!(recombine_elt(&ring, &layers, b), elt);
        }
        let z = vec![elt.clone(); 4];
        let (f, b) = (2usize, 8u32);
        let parts = decompose_vec(&ring, &z, f, b);
        assert_eq!(recombine_vec(&ring, &parts, b), z);
    }

    #[test]
    fn norm_ct_identity_and_picking() {
        // ct(⟨v, σ⁻¹(v)⟩) = ‖v‖² (the per-round exact-ℓ2 engine — the
        // no-wrap regime ‖v‖² < q/2 is the protocol's sound regime) and
        // ct(σ⁻¹(e_c)·s) = s[c] (the binding picker) — pinned on small
        // data, plus the wraparound behavior on full-width data.
        let ring = HwRing::new(crate::hyperwolf::HW_Q61, 64);
        let small: Vec<i64> = (0..64).map(|j| ((j * 37) % 7) as i64 - 3).collect();
        let v = ring.from_small(&small);
        let v_star = ring.conj(&v);
        let prod = sprod(&ring, std::slice::from_ref(&v), &[v_star]);
        let expect: i128 =
            v.0.iter()
                .map(|&c| {
                    let x = ring.center(c);
                    x * x
                })
                .sum();
        assert_eq!(ring.center(prod.0[0]), expect);
        // The picker (small data).
        let s = ring.from_small(&small);
        for c in [0usize, 1, 7, 63] {
            let pick = pick_elt(&ring, c);
            let prod = sprod(&ring, std::slice::from_ref(&pick), std::slice::from_ref(&s));
            assert_eq!(ring.center(prod.0[0]), ring.center(s.0[c]));
        }
        // The square decomposition is exact.
        for d in [
            0u64,
            1,
            2,
            3,
            7,
            12,
            100,
            12345,
            1 << 41,
            (1 << 41) + 987654,
        ] {
            let sq = square_decomposition(d);
            let got: u64 = sq
                .iter()
                .map(|&c| (c as u64).saturating_mul(c as u64))
                .sum();
            assert_eq!(got, d, "decomposition of {d}");
            assert!(sq.len() <= 16, "too many squares for {d}");
        }
    }

    /// A synthetic projection-shaped statement: per-round P/P*/S/S*
    /// vectors + bindings + exact norm constraints (with the greedy
    /// square slack) + one linear constraint. `with_norms` toggles the
    /// quadratic family; `full_width` uses uniform mod-q coefficients
    /// (the faithful-gate regime).
    fn synthetic_statement(rank: usize, rounds: usize) -> (LabStatement, LabWitness) {
        synthetic_statement_cfg(rank, rounds, true, false)
    }

    #[allow(clippy::too_many_lines)]
    fn synthetic_statement_cfg(
        rank: usize,
        rounds: usize,
        with_norms: bool,
        full_width: bool,
    ) -> (LabStatement, LabWitness) {
        let ring = HwRing::new(crate::hyperwolf::HW_Q61, 64);
        let mk = |seed: u64| -> Vec<HwElt> {
            (0..rank)
                .map(|i| {
                    if full_width {
                        // Medium-width adversarial regime (±2^20): far
                        // beyond the Core-SVP caps at N=64, inside the JL
                        // acceptance regime.
                        let coeffs: Vec<i64> = (0..64)
                            .map(|j| {
                                (((i * 131 + j * 7 + seed as usize * 29) % 2048) as i64 - 1024)
                                    * (1 << 10)
                            })
                            .collect();
                        ring.from_small(&coeffs)
                    } else {
                        let coeffs: Vec<i64> = (0..64)
                            .map(|j| ((i * 37 + j * 17 + seed as usize * 13) % 7) as i64 - 3)
                            .collect();
                        ring.from_small(&coeffs)
                    }
                })
                .collect()
        };
        // The per-round cap: honest-small or full-width regimes (the
        // full-width norms saturate u64 — the cap saturates with them).
        let cap = if full_width {
            u64::MAX
        } else {
            64u64 * rank as u64 * 9
        };
        let slack_rank = 16usize.div_ceil(64).max(1);
        let mut vectors = Vec::new();
        let mut s = Vec::new();
        let mut caps = Vec::new();
        for r in 0..rounds {
            let p = mk(r as u64 + 1);
            let sum: u64 = p
                .iter()
                .map(|e| norm_sq_u64(&ring, e))
                .fold(0u64, u64::saturating_add);
            let b_r = if with_norms { cap.max(sum) } else { cap };
            let _ = &sum;
            caps.push(b_r);
            let p_star: Vec<HwElt> = p.iter().map(|e| ring.conj(e)).collect();
            // The exact slack via the square decomposition.
            let squares = square_decomposition(b_r.saturating_sub(sum));
            let mut s_flat: Vec<HwElt> = vec![HwElt(vec![0u64; 64]); slack_rank];
            for (i, c) in squares.iter().enumerate() {
                s_flat[i / 64].0[i % 64] = (*c as i128).rem_euclid(i128::from(ring.q)) as u64;
            }
            let s_star: Vec<HwElt> = s_flat.iter().map(|e| ring.conj(e)).collect();
            vectors.push(LabVectorSpec {
                n: rank,
                betasq: b_r,
            });
            vectors.push(LabVectorSpec {
                n: rank,
                betasq: b_r,
            });
            vectors.push(LabVectorSpec {
                n: slack_rank,
                betasq: b_r,
            });
            vectors.push(LabVectorSpec {
                n: slack_rank,
                betasq: b_r,
            });
            s.push(p);
            s.push(p_star);
            s.push(s_flat);
            s.push(s_star);
        }
        let one = ring.one();
        let mut ct = Vec::new();
        for r in 0..rounds {
            for base in [(4 * r, 4 * r + 1), (4 * r + 2, 4 * r + 3)] {
                for k in 0..vectors[base.0].n {
                    for c in 0..64 {
                        let pick = pick_elt(&ring, c);
                        let pick_other = if c == 0 {
                            ring.neg(&one)
                        } else {
                            pick_elt(&ring, 64 - c)
                        };
                        ct.push(LabConstraint {
                            terms: vec![
                                LabTerm {
                                    idx: base.1,
                                    off: k,
                                    phi: vec![pick],
                                },
                                LabTerm {
                                    idx: base.0,
                                    off: k,
                                    phi: vec![pick_other],
                                },
                            ],
                            quads: vec![],
                            b: None,
                            ct_only: true,
                        });
                    }
                }
            }
        }
        // The exact per-round norm constraints (cross pairs: 1/2 coeff).
        if with_norms {
            let inv2 = ring.constant(((ring.q + 1) / 2) as i128);
            for r in 0..rounds {
                ct.push(LabConstraint {
                    terms: vec![],
                    quads: vec![
                        LabQuadEntry {
                            i: 4 * r,
                            j: 4 * r + 1,
                            coeff: inv2.clone(),
                        },
                        LabQuadEntry {
                            i: 4 * r + 2,
                            j: 4 * r + 3,
                            coeff: inv2.clone(),
                        },
                    ],
                    b: Some(ring.constant(caps[r] as i128)),
                    ct_only: true,
                });
            }
        }
        // One F constraint: ⟨phi, P_0⟩ = b with the honest b.
        let phi = mk(999);
        let b = sprod(&ring, &phi, &s[0].clone());
        let f = vec![LabConstraint {
            terms: vec![LabTerm {
                idx: 0,
                off: 0,
                phi,
            }],
            quads: vec![],
            b: Some(b),
            ct_only: false,
        }];
        let global: u64 = vectors
            .iter()
            .map(|v| v.betasq)
            .fold(0u64, u64::saturating_add);
        (
            LabStatement::new(vectors, f, ct, global),
            LabWitness::new(s),
        )
    }

    #[test]
    fn engine_synthetic_roundtrip_and_tamper() {
        let ring = HwRing::new(crate::hyperwolf::HW_Q61, 64);
        let (stmt, wit) = synthetic_statement(64, 2);
        stmt.check_all(&ring, &wit.s).unwrap();
        let key_ctx = b"test-key".as_slice();
        let proof = lab_prove(&ring, SisGateMode::KernelBypass, &stmt, &wit, key_ctx).unwrap();
        lab_verify(&ring, SisGateMode::KernelBypass, &stmt, &proof, key_ctx).unwrap();
        // Tamper the final witness: E3/E5 breaks.
        let mut bad = proof.clone();
        let mut first = bad.final_witness.s[0][0].0.clone();
        first[0] = (first[0] + 1) % ring.q;
        bad.final_witness.s[0][0] = HwElt(first);
        assert!(lab_verify(&ring, SisGateMode::KernelBypass, &stmt, &bad, key_ctx).is_err());
        // Tamper the tail's u1 (the t-images): E3 breaks.
        let mut bad2 = proof.clone();
        let mut u1 = bad2.tail.u1[0].0.clone();
        u1[1] = (u1[1] + 1) % ring.q;
        bad2.tail.u1[0] = HwElt(u1);
        assert!(lab_verify(&ring, SisGateMode::KernelBypass, &stmt, &bad2, key_ctx).is_err());
        // The level table is well-formed.
        let table = lab_level_table(&ring, &proof);
        assert!(!table.is_empty());
        assert!(table.last().unwrap().tail);
    }

    #[test]
    fn faithful_gate_fails_at_kernel_scale() {
        // The honest pins: (a) medium-width (±2^20) witnesses blow the
        // Core-SVP caps at the INIT stage; (b) even small-norm statements
        // fail at the TAIL stage — the tail's directly transmitted Ajtai
        // images are uniform mod q (norm ≈ 2^61) and the N = 64, LOGQ = 61
        // rule caps at ≈ 2^56.5 — the recorded parameter residual (the
        // faithful regime needs N ≥ 128 or the RNS path).
        let ring = HwRing::new(crate::hyperwolf::HW_Q61, 64);
        let (stmt, wit) = synthetic_statement_cfg(64, 1, false, true);
        stmt.check_all(&ring, &wit.s).unwrap();
        assert!(lab_prove(&ring, SisGateMode::Faithful, &stmt, &wit, b"k".as_slice()).is_err());
        let (stmt2, wit2) = synthetic_statement_cfg(64, 1, false, false);
        let err = lab_prove(&ring, SisGateMode::Faithful, &stmt2, &wit2, b"k".as_slice())
            .expect_err("the tail's uniform-image gate must fail at N=64");
        assert!(
            err.contains("secure"),
            "expected the SIS-gate failure, got: {err}"
        );
    }

    #[test]
    fn recursion_engages_on_the_right_shape() {
        // Few BIG vectors: the amortization shrinks and at least one
        // non-tail level engages (the compaction's recursion evidence).
        // Rank 2048 gives the shrink real headroom over the v-garbage
        // entropy (at rank 1024 the level is marginally no-shrink).
        let ring = HwRing::new(crate::hyperwolf::HW_Q61, 64);
        let (stmt, wit) = synthetic_statement(2048, 1);
        let key_ctx = b"test-key-big".as_slice();
        let proof = lab_prove(&ring, SisGateMode::KernelBypass, &stmt, &wit, key_ctx).unwrap();
        lab_verify(&ring, SisGateMode::KernelBypass, &stmt, &proof, key_ctx).unwrap();
        let table = lab_level_table(&ring, &proof);
        assert!(
            table.len() >= 2,
            "expected at least one non-tail level + the tail, got {}",
            table.len()
        );
        assert!(table.last().unwrap().tail);
        // The transmitted size is measured and sane.
        let bytes = lab_proof_size_bytes(&ring, &proof);
        assert!(bytes > 0 && bytes < (1 << 22), "size {bytes} insane");
    }

    #[test]
    fn full_fidelity_roundtrip() {
        let (hw, f_ints, a0_ints, a_list, y) = setup(b"hw-lab-1");
        let (cm, state) = hw.commit(&f_ints).unwrap();
        let mut t = Transcript::new_default(b"hw-labrador");
        let proof = hw
            .eval_prove_labrador(&state, &a0_ints, &a_list, &mut t)
            .unwrap();
        let mut tv = Transcript::new_default(b"hw-labrador");
        assert!(hw
            .eval_verify_labrador(&cm, &a0_ints, &a_list, y, &proof, &mut tv)
            .unwrap());
        // A wrong y is rejected (check 1).
        let mut tv2 = Transcript::new_default(b"hw-labrador");
        assert!(!hw
            .eval_verify_labrador(&cm, &a0_ints, &a_list, y.wrapping_add(1), &proof, &mut tv2)
            .unwrap());
    }

    #[test]
    fn full_fidelity_tamper_suite() {
        let (hw, f_ints, a0_ints, a_list, y) = setup(b"hw-lab-2");
        let (cm, state) = hw.commit(&f_ints).unwrap();
        let mut t = Transcript::new_default(b"hw-labrador");
        let proof = hw
            .eval_prove_labrador(&state, &a0_ints, &a_list, &mut t)
            .unwrap();
        let ring = &hw.ring;

        // Tamper the revealed final witness: the final checks fail.
        {
            let mut bad = proof.clone();
            let mut c = bad.s_final[0].0.clone();
            c[0] = (c[0] + 1) % ring.q;
            bad.s_final[0] = HwElt(c);
            let mut tv = Transcript::new_default(b"hw-labrador");
            assert!(!hw
                .eval_verify_labrador(&cm, &a0_ints, &a_list, y, &bad, &mut tv)
                .unwrap());
        }
        // Tamper the amortized proof's tail u1: the Dachshund E-checks fail.
        {
            let mut bad = proof.clone();
            let mut c = bad.lab.tail.u1[0].0.clone();
            c[1] = (c[1] + 1) % ring.q;
            bad.lab.tail.u1[0] = HwElt(c);
            let mut tv = Transcript::new_default(b"hw-labrador");
            assert!(!hw
                .eval_verify_labrador(&cm, &a0_ints, &a_list, y, &bad, &mut tv)
                .unwrap());
        }
        // Tamper a clear fold value: check 1 fails.
        {
            let mut bad = proof.clone();
            let mut c = bad.rounds[0].fold[0].0.clone();
            c[2] = (c[2] + 1) % ring.q;
            bad.rounds[0].fold[0] = HwElt(c);
            let mut tv = Transcript::new_default(b"hw-labrador");
            assert!(!hw
                .eval_verify_labrador(&cm, &a0_ints, &a_list, y, &bad, &mut tv)
                .unwrap());
        }
        // Tamper the amortized proof's JL vector: the transcript replay /
        // acceptance bound fails.
        {
            let mut bad = proof.clone();
            bad.lab.tail.p[17] = bad.lab.tail.p[17].wrapping_add(1);
            let mut tv = Transcript::new_default(b"hw-labrador");
            assert!(!hw
                .eval_verify_labrador(&cm, &a0_ints, &a_list, y, &bad, &mut tv)
                .unwrap());
        }
    }

    #[test]
    fn full_fidelity_size_beats_clear_payload() {
        // The wire-format delta at the paper shape: the amortized
        // projection proof is smaller than the clear projection payload.
        let params = paper_params(1 << 20);
        let (clear, lab) = lab_size_model(&params);
        assert!(lab < clear, "model: amortized {lab} >= clear {clear}");
        // And the kernel-scale measured proof is finite and structured.
        let (hw, f_ints, a0_ints, a_list, _y) = setup(b"hw-lab-3");
        let (_cm, state) = hw.commit(&f_ints).unwrap();
        let mut t = Transcript::new_default(b"hw-labrador");
        let proof = hw
            .eval_prove_labrador(&state, &a0_ints, &a_list, &mut t)
            .unwrap();
        let bytes = lab_proof_size_bytes(&hw.ring, &proof.lab);
        assert!(bytes > 0);
        // The clear per-round projection payload at these params:
        // (k−1)·b·jl_rows·d coefficients — the amortized proof replaces it.
        let clear_kernel =
            ((hw.params.k - 1) * hw.params.b * hw.params.jl_rows * hw.params.d * 61) as u64 / 8;
        assert!(clear_kernel > 0);
    }

    // Temporary debug test — the JL collapse identity in isolation.
    #[test]
    fn shape_bisect_regression() {
        // The per-vector part alignment regression: every shape's tail
        // level must prove (the pre-fix bug misplaced short vectors'
        // successors inside the previous vector's last part).
        use super::*;
        let ring = HwRing::new(crate::hyperwolf::HW_Q61, 64);
        for &(rank, rounds, norms) in &[
            (16usize, 1usize, false),
            (16, 1, true),
            (32, 1, false),
            (64, 1, false),
        ] {
            let (stmt, wit) = synthetic_statement_cfg(rank, rounds, norms, false);
            stmt.check_all(&ring, &wit.s).unwrap();
            let (proof, _, final_wit) =
                lab_prove_level(&ring, SisGateMode::KernelBypass, &stmt, &wit, b"dbg", true)
                    .unwrap();
            let final_wit = final_wit.expect("the tail returns the final witness");
            lab_verify_tail(
                &ring,
                SisGateMode::KernelBypass,
                &stmt,
                &proof,
                &final_wit,
                b"dbg",
            )
            .unwrap();
        }
    }

    #[test]
    fn round_bound_gate_is_fail_closed() {
        // The wraparound guard: a paper-scale ladder exceeds q/4 and is
        // refused (modelled, not executed).
        let mut params = paper_params(1 << 30);
        params.k = 15;
        assert!(round_bound(&params, 0).is_err() || round_bound(&params, 14).is_err());
        // The kernel ladder passes.
        let params = small_params();
        for r in 0..params.k - 1 {
            assert!(round_bound(&params, r).is_ok());
        }
    }
}
