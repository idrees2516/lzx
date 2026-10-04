//! The 50 KB compact-opening pipeline (docs/DESIGN_50KB.md).
//!
//! Replaces the Θ(N) digit-revealing `BundleOpening` with a folded
//! amortized opening in the LaBinius Π_fold / LaBRADOR tradition, adapted
//! to LZX's two-characteristic stack (Goldilocks sumchecks + R_q Ajtai
//! commitments).
//!
//! # Architecture
//!
//! **Item 1:Narrow byte packing** (Stage 0): each flat value is stored as its
//! natural byte width, one byte per ring coefficient. Coefficients are
//! ≤ 255, which fixes the bits-bundle vacuous-gate bug (the old
//! 31-bits/coeff packing exceeded q/2, making the balanced norm claim
//! ambiguous — a security defect, not just a size problem).
//! **Item 2:The r-aligned column layout** (the load-bearing design): the flat
//! domain's *last* log₂r variables index the columns. Because factor
//! regions are r-aligned (every factor length is a power of two ≥ r),
//! each column's byte-stream has an identical (head, byte) structure:
//! the m-th byte of column j is byte b(m) of the flat position
//! h(m)·r + j. The shadow functional therefore factorizes:
//! `λ(slot m) = eq_head(h(m)) · 2^{8·b(m)} = Ψ(m)` — **column-uniform**.
//! **Item 3:The Goldilocks carrier** (Stage 1): the existing grouped carrier
//! binds every base claim to one terminal `w = f(r_sc)` over the flat
//! MLE (identical protocol to the Clear mode's carrier).
//! **Item 4:The column fold** (Stage 2): columns of n̄ ring elements each,
//!    committed under the column-uniform key F̄ ∈ R_q^{k×n̄} (LaBinius's
//!    structure — MSIS on [F̄|F̄|..] reduces to MSIS on F̄). The prover
//!    sends the per-column shadow values u_j = Λ̃(w_j) ∈ R (absorbed
//!    *before* the challenges), then the response v = Σ_j c_j·w_j
//!    (rANS-coded). The verifier checks:
//!    (a) `Λ̃(v) = Σ_j c_j·u_j` — R-linearity of the column-uniform
//!        functional: the fold commutes exactly (no cross terms);
//!    (b) `Σ_j μ_j·ct(u_j) = w` with μ_j = eq(r_tail)_j — the MLE
//!        interpolation over the column variables closes the carrier;
//!    (c) `F̄·v = Σ_j c_j·y_j` — Ajtai binding to the commitments;
//!    (d) the per-coefficient norm gate (fail-closed, worst-case bound
//!        r·w·A·255 < q/2).
//!
//! # Soundness
//!
//! Legs (Goldilocks) → claims → F_q carrier → w → (b) pins the u's to w,
//! (a) pins the u's to v, (c) pins v to the y's, extraction terminates in
//! MSIS on `[F̄ | −y₁..−y_r]` at the relaxed bound (2·gate). Extraction
//! never divides by challenge differences (the ring Q_32 ≡ 1 mod 128
//! splits completely, so division is unsound there); the relaxed relation
//! composed with the honest preimage reduces to plain MSIS on F̄.
//! Challenges: fixed-weight ring elements, weight 32, amplitude 2
//! (|C| ≈ 2^125 ⇒ 125-bit knowledge error per forking level).

use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::ring::{RingConfig, RingElement};
use lattice_ring::Modulus32;

use crate::ledger::LedgerError;

// ---------------------------------------------------------------------------
// F_q arithmetic (q = Q_32 = 3·2^30+1, the bundle ring's modulus)
// ---------------------------------------------------------------------------

pub const Q: u64 = 3_221_225_473;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fq(pub u64);

#[allow(clippy::should_implement_trait)]
impl Fq {
    pub const ZERO: Fq = Fq(0);
    pub const ONE: Fq = Fq(1);

    #[inline]
    pub fn from_u64(x: u64) -> Fq {
        Fq(x % Q)
    }

    #[inline]
    pub fn add(self, other: Fq) -> Fq {
        let s = self.0 + other.0;
        Fq(if s >= Q { s - Q } else { s })
    }

    #[inline]
    pub fn sub(self, other: Fq) -> Fq {
        Fq((self.0 + Q - other.0) % Q)
    }

    #[inline]
    pub fn mul(self, other: Fq) -> Fq {
        Fq(((self.0 as u128 * other.0 as u128) % Q as u128) as u64)
    }

    #[inline]
    pub fn neg(self) -> Fq {
        Fq(if self.0 == 0 { 0 } else { Q - self.0 })
    }

    /// Multiplicative inverse (q prime); None on zero.
    pub fn inv(self) -> Option<Fq> {
        if self.0 == 0 {
            return None;
        }
        let mut result = 1u64;
        let mut base = self.0;
        let mut exp = Q - 2;
        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % Q as u128) as u64;
            }
            base = ((base as u128 * base as u128) % Q as u128) as u64;
            exp >>= 1;
        }
        Some(Fq(result))
    }

    /// Lift a Goldilocks element canonically (verifier-side determinism:
    /// canonical u64 representative reduced mod q).
    pub fn lift_goldilocks(g: &Goldilocks) -> Fq {
        Fq::from_u64(g.to_canonical_u64())
    }

    /// The eq table over all 2^m boolean points, MSB-first variable order
    /// (matches DenseMle: variable 0 = most significant index bit).
    pub fn eq_table(point: &[Fq]) -> Vec<Fq> {
        // Iterate variables in reverse so variable 0 lands in the MOST
        // significant index bit (DenseMle's convention: variable 0 = MSB).
        let mut table = vec![Fq::ONE];
        for r in point.iter().rev() {
            let mut next = Vec::with_capacity(table.len() * 2);
            for t in &table {
                next.push(t.mul(Fq::ONE.sub(*r)));
            }
            for t in &table {
                next.push(t.mul(*r));
            }
            table = next;
        }
        table
    }

    /// eq(a, b) for equal-length points (product of per-variable
    /// agreements), matching DenseMle::eq_eval's semantics.
    pub fn eq_eval(a: &[Fq], b: &[Fq]) -> Fq {
        let mut acc = Fq::ONE;
        for (x, y) in a.iter().zip(b.iter()) {
            let same = x.mul(*y).add(Fq::ONE.sub(*x).mul(Fq::ONE.sub(*y)));
            acc = acc.mul(same);
        }
        acc
    }
}

/// Bind the first (most significant) variable of an F_q MLE to `r`,
/// halving the evaluation array in place.
pub fn bind_mle(cur: &mut Vec<Fq>, r: Fq) {
    let half = cur.len() / 2;
    for j in 0..half {
        let lo = cur[j];
        let hi = cur[j + half];
        cur[j] = lo.add(hi.sub(lo).mul(r));
    }
    cur.truncate(half);
}

/// Evaluate an F_q MLE at `point` (MSB-first), consuming a copy.
pub fn evaluate_mle(evals: &[Fq], point: &[Fq]) -> Fq {
    let mut cur = evals.to_vec();
    for r in point {
        bind_mle(&mut cur, *r);
    }
    cur[0]
}

/// The bundle ring: R_{Q_32} with n = 64 (X^64+1 negacyclic).
pub fn column_ring() -> Result<RingConfig, LedgerError> {
    lattice_widthfold::codec::q32_ring()
        .map_err(|e| LedgerError::Layout(format!("ring: {e:?}")))
}

// ---------------------------------------------------------------------------
// The r-aligned narrow packing (the load-bearing layout)
// ---------------------------------------------------------------------------

/// The packing shape: per-factor (length, byte width), r, flat_log.
/// Reconstructible by the verifier from public data + transmitted widths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackShape {
    /// Factor lengths in bundle order (powers of two, each ≥ r).
    pub factor_lens: Vec<usize>,
    /// Per-factor byte widths (0..=8; 0 treated as 1 for all-zero factors).
    pub widths: Vec<u8>,
    /// The column count r (a power of two ≤ every factor length).
    pub r: usize,
    /// log2 of the padded flat length.
    pub flat_log: usize,
}

impl PackShape {
    /// The padded flat length.
    pub fn flat_pad(&self) -> usize {
        1usize << self.flat_log
    }

    /// The factor owning flat position p (None in the padding region).
    fn factor_of(&self, p: usize) -> Option<usize> {
        let mut base = 0usize;
        for (fi, &len) in self.factor_lens.iter().enumerate() {
            if p < base + len {
                return Some(fi);
            }
            base += len;
        }
        None
    }

    /// The byte width at flat position p (0 in padding).
    fn width_at(&self, p: usize) -> usize {
        match self.factor_of(p) {
            Some(fi) => (self.widths[fi] as usize).max(1),
            None => 0,
        }
    }

    /// The per-column byte-stream length (identical for every column:
    /// factor regions are r-aligned so width_at(h·r + j) is
    /// j-independent).
    pub fn stream_len(&self) -> usize {
        let flat_pad = self.flat_pad();
        let h_count = flat_pad / self.r;
        let mut total = 0usize;
        for h in 0..h_count {
            total += self.width_at(h * self.r);
        }
        total
    }

    /// The slot map: for stream position m, the (head, byte) pair —
    /// column-independent by construction. Returns None if m is padding.
    fn slot_of(&self, m: usize) -> Option<(usize, u8)> {
        let flat_pad = self.flat_pad();
        let h_count = flat_pad / self.r;
        let mut acc = 0usize;
        for h in 0..h_count {
            let w = self.width_at(h * self.r);
            if m < acc + w {
                return Some((h, (m - acc) as u8));
            }
            acc += w;
        }
        None
    }

    /// The column-uniform shadow weights Ψ(m) = eq_head(h(m))·2^{8·b(m)}
    /// for every stream slot m (length = stream_len).
    pub fn psi_weights(&self, r_head: &[Fq]) -> Vec<Fq> {
        let eq_head = Fq::eq_table(r_head);
        let stream = self.stream_len();
        let mut out = Vec::with_capacity(stream);
        for m in 0..stream {
            if let Some((h, b)) = self.slot_of(m) {
                let w = eq_head
                    .get(h)
                    .copied()
                    .unwrap_or(Fq::ZERO)
                    .mul(Fq::from_u64(1u64 << (8 * b as u32)));
                out.push(w);
            } else {
                out.push(Fq::ZERO);
            }
        }
        out
    }

    /// The column weights μ_j = eq(r_tail)_j for j ∈ [0, r).
    pub fn mu_weights(&self, r_tail: &[Fq]) -> Vec<Fq> {
        Fq::eq_table(r_tail)
    }
}

/// The packed witness (prover side).
pub struct PackedWitness {
    pub shape: PackShape,
    /// Column-major byte streams: `columns[j][m]` = byte b(m) of the flat
    /// position h(m)·r + j.
    pub columns: Vec<Vec<u8>>,
    /// The flat MLE over Goldilocks (the carrier's polynomial).
    pub flat: DenseMle,
}

/// Pack the bundle factors into the r-aligned column layout.
///
/// Requires: r ≤ every factor length (powers of two ⇒ r-aligned regions).
pub fn pack_columns(
    entries: &[(u32, &DenseMle)],
    r: usize,
) -> Result<PackedWitness, LedgerError> {
    let mut factor_lens = Vec::with_capacity(entries.len());
    let mut widths = Vec::with_capacity(entries.len());
    let mut flat_len = 0usize;
    for (_, mle) in entries {
        let len = mle.evaluations.len();
        if len.count_ones() != 1 {
            return Err(LedgerError::Layout("factor length not a power of two".into()));
        }
        if len < r {
            return Err(LedgerError::Layout(format!(
                "factor length {len} < r {r}"
            )));
        }
        let mut max: u64 = 0;
        for v in &mle.evaluations {
            let c = v.to_canonical_u64();
            if c > max {
                max = c;
            }
        }
        let mut width = 0usize;
        while (max >> (8 * width)) > 0 {
            width += 1;
        }
        widths.push(width.clamp(1, 8) as u8);
        factor_lens.push(len);
        flat_len += len;
    }
    let flat_log = flat_len.next_power_of_two().max(1).trailing_zeros() as usize;
    let shape = PackShape {
        factor_lens,
        widths,
        r,
        flat_log,
    };
    // The flat values array (zero-padded).
    let flat_pad = shape.flat_pad();
    let mut flat: Vec<u64> = vec![0u64; flat_pad];
    let mut base = 0usize;
    for (_, mle) in entries {
        for (i, v) in mle.evaluations.iter().enumerate() {
            flat[base + i] = v.to_canonical_u64();
        }
        base += mle.evaluations.len();
    }
    // Column streams.
    let stream = shape.stream_len();
    let mut columns: Vec<Vec<u8>> = vec![vec![0u8; stream]; r];
    for j in 0..r {
        let mut m = 0usize;
        for h in 0..(flat_pad / r) {
            let w = shape.width_at(h * r);
            let v = flat[h * r + j];
            for b in 0..w {
                columns[j][m] = ((v >> (8 * b)) & 0xFF) as u8;
                m += 1;
            }
        }
        debug_assert_eq!(m, stream);
    }
    let flat = flat_mle(entries);
    Ok(PackedWitness {
        shape,
        columns,
        flat,
    })
}

/// The flat MLE over Goldilocks (values domain, zero-padded) — the
/// carrier's polynomial.
pub fn flat_mle(entries: &[(u32, &DenseMle)]) -> DenseMle {
    let mut evals: Vec<Goldilocks> = Vec::new();
    for (_, mle) in entries {
        evals.extend_from_slice(&mle.evaluations);
    }
    let padded = evals.len().next_power_of_two().max(1);
    evals.resize(padded, Goldilocks::ZERO);
    DenseMle {
        num_vars: padded.trailing_zeros() as usize,
        evaluations: evals,
    }
}

// ---------------------------------------------------------------------------
// The response codec: magnitude-class rANS + raw low bits
// ---------------------------------------------------------------------------

// The response codec + ring serialization moved to `lattice-widthfold`
// (the shared fold core); re-exported here for the zkvm-internal call
// sites and the public API.
pub use lattice_widthfold::codec::{
    decode_response, deserialize_elements, encode_response, serialize_elements, ResponseWire,
};

// ---------------------------------------------------------------------------
// The compact bundle: commit + prove + verify
// ---------------------------------------------------------------------------
//
// Design (final, see docs/DESIGN_50KB.md §3): the existing GOLDILOCKS
// grouped carrier (ledger.rs) binds every base claim to one terminal
// `w = f(r_sc)` over the flat MLE. The compact opening replaces only the
// Θ(N) digit reveal:
//
// 1. The packed coefficients split into r columns (r-aligned layout): the
//    flat domain's last log₂r variables index the columns, so the
//    Goldilocks evaluation functional factorizes per column:
//    `ũ_j = Σ_slots Ψ(m)·byte(m, j)` with the SAME column-uniform weights
//    `Ψ(m) = eq(r_head)_{h(m)} · 2^{8·b(m)}`.
// 2. The prover sends the per-column values `ũ_j` (Goldilocks, 8 B each)
//    BEFORE the challenges; the verifier checks the MLE interpolation
//    `w = Σ_j eq(r_tail)_j · ũ_j`.
// 3. Scalar challenges `d_j ∈ [−A, A]`; the response `v = Σ_j d_j·w_j`
//    is an INTEGER fold (the gate keeps every coefficient < q/2, so the
//    mod-q representation is exact) and the Goldilocks functional
//    commutes exactly: `Φ(v) = Σ_j d_j·ũ_j`.
// 4. The verifier checks (a) `Φ(v) = Σ_j d_j·ũ_j` over Goldilocks,
//    (b) the Ajtai fold `F̄·v = Σ_j d_j·y_j` over R_q, (c) the tight
//    norm gate. Soundness terminates in MSIS on `[F̄ | −y]` at the
//    relaxed bound with the mixed-moduli constraint lattice — the
//    honest-gap regime (the gate sits at ~6σ of the honest fold
//    distribution, far below the lattice covering radius at the chosen
//    (k, n̄)).

/// Fold parameters for one bundle (public shape, transmitted).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldParams {
    /// Number of columns r (a power of two, ≤ every factor length).
    pub r: usize,
    /// Commitment rows k per column.
    pub k: usize,
    /// Scalar challenge amplitude A (challenges in [−A, A] \ {0}-ish).
    pub amplitude: u32,
    /// The response norm gate (per coefficient; must be < q/2).
    pub gate: u32,
}

impl FoldParams {
    /// The estimator-tuned default (SECURITY.md's MSIS table): amplitude
    /// `2^6` tightens the relaxed bound `64×` at zero exactness cost
    /// (the integer fold's Goldilocks functional commutes at any
    /// amplitude; `r·A·255 ≪ q/2` holds with margin).
    pub fn new(r: usize, k: usize) -> Self {
        let amplitude = 1u32 << 6;
        let gate = (r as u64) * (amplitude as u64) * 255;
        FoldParams {
            r,
            k,
            amplitude,
            gate: gate.min((Modulus32::Q_32.q / 2 - 1) as u64) as u32,
        }
    }

    /// The worst-case fold bound r·A·β₀ (the completeness guarantee).
    pub fn worst_bound(&self, beta0: u64) -> u64 {
        (self.r as u64) * (self.amplitude as u64) * beta0
    }
}

/// The compact bundle opening artifact (one per bundle).
#[derive(Clone, Debug)]
pub struct CompactOpening {
    /// Fold parameters (public shape).
    pub params: FoldParams,
    /// Per-factor byte widths (the packing shape).
    pub widths: Vec<u8>,
    /// Per-column Goldilocks values ũ_j (r field elements).
    pub u_tilde: Vec<Goldilocks>,
    /// The rANS-coded response v = Σ_j d_j·w_j.
    pub response: ResponseWire,
}

/// The prover-side bundle state.
pub struct CompactBundleProver {
    pub ring: RingConfig,
    pub packed: PackedWitness,
    pub key: AjtaiPublicKey,
    pub params: FoldParams,
    /// r columns × n̄ ring elements.
    pub columns: Vec<Vec<RingElement>>,
    /// Per-column commitments y_j = F̄·w_j ∈ R^k.
    pub y: Vec<Vec<RingElement>>,
    pub n_bar: usize,
    pub seed: [u8; 32],
}

/// Commit a bundle in compact form (the columns + per-column Ajtai
/// commitments under the column-uniform key F̄) — the Clear profile's
/// default parameters.
pub fn compact_bundle_commit(
    entries: &[(u32, &DenseMle)],
    seed: [u8; 32],
    r: usize,
    k: usize,
) -> Result<CompactBundleProver, LedgerError> {
    compact_bundle_commit_with_params(entries, seed, r, k, FoldParams::new(r, k))
}

/// Commit a bundle with an explicit fold-parameter profile (the Sound
/// mode's `FoldParams::sound` — the amplitude the width fold's
/// estimator ceiling allows).
pub fn compact_bundle_commit_with_params(
    entries: &[(u32, &DenseMle)],
    seed: [u8; 32],
    r: usize,
    k: usize,
    params: FoldParams,
) -> Result<CompactBundleProver, LedgerError> {
    let ring = column_ring()?;
    let packed = pack_columns(entries, r)?;
    let stream = packed.shape.stream_len();
    let n = ring.n();
    // Each column carries the full per-column byte stream (the r-aligned
    // layout: column j holds the bytes of flat positions p ≡ j mod r, and
    // every column's (head, byte) structure is identical). The response
    // covers stream_len = total/r coefficients — the fold's shrink.
    let n_bar = stream.div_ceil(n).max(1);
    let ajtai = AjtaiParams {
        ring: ring.clone(),
        k,
        m: n_bar,
        norm_bound: params.gate,
    };
    let key = AjtaiPublicKey::from_seed(ajtai, seed).map_err(LedgerError::Ajtai)?;
    let mut columns: Vec<Vec<RingElement>> = Vec::with_capacity(r);
    for j in 0..r {
        let mut col: Vec<RingElement> = Vec::with_capacity(n_bar);
        for i in 0..n_bar {
            let mut coeffs = vec![0u32; n];
            for (kk, c) in coeffs.iter_mut().enumerate() {
                if let Some(&b) = packed.columns[j].get(i * n + kk) {
                    *c = b as u32;
                }
            }
            col.push(RingElement::from_coeffs(&ring, coeffs));
        }
        columns.push(col);
    }
    let mut y: Vec<Vec<RingElement>> = Vec::with_capacity(r);
    for col in &columns {
        let c = key.commit(col).map_err(LedgerError::Ajtai)?;
        y.push(c.rows);
    }
    Ok(CompactBundleProver {
        ring,
        packed,
        key,
        params,
        columns,
        y,
        n_bar,
        seed,
    })
}

/// The column-uniform Goldilocks weights Ψ over the stream slots:
/// Ψ(m) = eq(r_head)_{h(m)} · 2^{8·b(m)} mod p_G.
fn psi_weights_goldilocks(shape: &PackShape, r_head: &[Goldilocks]) -> Vec<Goldilocks> {
    // eq over the head variables, MSB-first (DenseMle convention).
    let mut table = vec![Goldilocks::ONE];
    for rr in r_head.iter().rev() {
        let one_minus = Goldilocks::ONE.sub(rr);
        let mut next = Vec::with_capacity(table.len() * 2);
        for t in &table {
            next.push(t.mul(&one_minus));
        }
        for t in &table {
            next.push(t.mul(rr));
        }
        table = next;
    }
    let stream = shape.stream_len();
    let mut out = Vec::with_capacity(stream);
    for m in 0..stream {
        if let Some((h, b)) = shape.slot_of(m) {
            let w = table
                .get(h)
                .copied()
                .unwrap_or(Goldilocks::ZERO)
                .mul(&Goldilocks::from_u64(1u64 << (8 * b as u32)));
            out.push(w);
        } else {
            out.push(Goldilocks::ZERO);
        }
    }
    out
}


/// The balanced-representative Goldilocks term (moved to
/// `lattice-widthfold::helpers`; re-exported).
pub use lattice_widthfold::helpers::phi_term;
impl CompactBundleProver {
    /// The serialized commitments (r × k ring elements) — the bundle's
    /// public commitment.
    pub fn commitment_bytes(&self) -> Vec<u8> {
        let flat: Vec<RingElement> = self.y.iter().flatten().cloned().collect();
        serialize_elements(&self.ring, &flat)
    }

    /// The flat MLE over Goldilocks (the carrier's polynomial — the SAME
    /// one the existing ledger carrier uses).
    pub fn flat_mle(&self) -> &DenseMle {
        &self.packed.flat
    }

    /// Prove the compact opening, given the carrier's terminal point
    /// `r_sc` (from the existing Goldilocks carrier) and its claim `w`.
    pub fn prove_compact_opening(
        &self,
        r_sc: &[Goldilocks],
        w: &Goldilocks,
        transcript: &mut Transcript,
    ) -> Result<CompactOpening, LedgerError> {
        let flat_log = self.packed.shape.flat_log;
        let log_r = self.params.r.trailing_zeros() as usize;
        if r_sc.len() != flat_log {
            return Err(LedgerError::Layout("r_sc arity".into()));
        }
        let r_head = &r_sc[..flat_log - log_r];
        let r_tail = &r_sc[flat_log - log_r..];

        // 1. The per-column values ũ_j (Goldilocks).
        let psi = psi_weights_goldilocks(&self.packed.shape, r_head);
        let n = self.ring.n();
        let n_bar = self.n_bar;
        let mut u_tilde: Vec<Goldilocks> = Vec::with_capacity(self.params.r);
        for j in 0..self.params.r {
            let mut acc = Goldilocks::ZERO;
            for m in 0..psi.len() {
                if let Some(&b) = self.packed.columns[j].get(m) {
                    acc = acc.add(&psi[m].mul(&Goldilocks::from_u64(b as u64)));
                }
            }
            u_tilde.push(acc);
        }
        // 2. The interpolation self-check: w = Σ_j eq(r_tail)_j·ũ_j.
        {
            let mut table = vec![Goldilocks::ONE];
            for rr in r_tail.iter().rev() {
                let one_minus = Goldilocks::ONE.sub(rr);
                let mut next = Vec::with_capacity(table.len() * 2);
                for t in &table {
                    next.push(t.mul(&one_minus));
                }
                for t in &table {
                    next.push(t.mul(rr));
                }
                table = next;
            }
            let mut check = Goldilocks::ZERO;
            for (j, uj) in u_tilde.iter().enumerate() {
                check = check.add(&table[j].mul(uj));
            }
            if check != *w {
                return Err(LedgerError::Layout(format!(
                    "interpolation self-check failed: {check:?} vs {w:?}"
                )));
            }
        }
        // Absorb the ũ's BEFORE the challenges.
        let mut ubuf: Vec<u8> = Vec::with_capacity(8 * u_tilde.len());
        for u in &u_tilde {
            ubuf.extend_from_slice(&u.to_canonical_u64().to_le_bytes());
        }
        transcript
            .append_bytes(b"fold-utilde", &ubuf)
            .map_err(LedgerError::Transcript)?;

        // 3. The scalar challenges d_j ∈ [−A, A].
        let d: Vec<i64> = (0..self.params.r)
            .map(|_| {
                let b = transcript
                    .challenge_bytes(b"fold-dchal", 2)
                    .map_err(LedgerError::Transcript)?;
                let raw = u16::from_le_bytes([b[0], b[1]]) as u64;
                // Map to [−A, A]: raw mod (2A+1) − A.
                let m = 2 * self.params.amplitude as u64 + 1;
                Ok((raw % m) as i64 - self.params.amplitude as i64)
            })
            .collect::<Result<_, _>>()?;

        // 4. The response v = Σ_j d_j·w_j (integer fold; the gate keeps
        //    every coefficient < q/2 so the mod-q rep is exact).
        let gate = self.params.gate as i64;
        if gate >= (self.ring.modulus.q / 2) as i64 {
            return Err(LedgerError::Layout("gate exceeds q/2".into()));
        }
        let mut v: Vec<RingElement> = vec![self.ring.zero(); n_bar];
        let mut v_coeffs: Vec<i32> = Vec::with_capacity(n_bar * n);
        for (j, &dj) in d.iter().enumerate() {
            if dj == 0 {
                continue;
            }
            for i in 0..n_bar {
                let prod = self.columns[j][i].scale_i64(dj);
                v[i] = v[i].add(&prod).map_err(|e| {
                    LedgerError::Layout(format!("ring add: {e:?}"))
                })?;
            }
        }
        for elem in &v {
            for &c in elem.coeffs() {
                let balanced = if c > self.ring.modulus.q / 2 {
                    c as i64 - self.ring.modulus.q as i64
                } else {
                    c as i64
                };
                if balanced.abs() > gate {
                    return Err(LedgerError::Layout(format!(
                        "norm gate: |{balanced}| > {gate}"
                    )));
                }
                v_coeffs.push(balanced as i32);
            }
        }
        // 5. The functional self-check: Φ(v) = Σ_j d_j·ũ_j over Goldilocks.
        {
            let q = self.ring.modulus.q;
            let mut phi_v = Goldilocks::ZERO;
            for (i, elem) in v.iter().enumerate() {
                let mut elem_sum = Goldilocks::ZERO;
                for (kk, &c) in elem.coeffs().iter().enumerate() {
                    let weight = psi.get(i * n + kk).copied().unwrap_or(Goldilocks::ZERO);
                    if weight != Goldilocks::ZERO {
                        let t = phi_term(&weight, c, q);
                        elem_sum = elem_sum.add(&t);
                    }
                }
                phi_v = phi_v.add(&elem_sum);
            }
            let mut rhs = Goldilocks::ZERO;
            for (j, &dj) in d.iter().enumerate() {
                if dj != 0 {
                    let term = u_tilde[j].mul(&Goldilocks::from_u64(dj.unsigned_abs()));
                    rhs = if dj < 0 { rhs.sub(&term) } else { rhs.add(&term) };
                }
            }
            if phi_v != rhs {
                return Err(LedgerError::Layout("functional self-check failed".into()));
            }
        }
        // Absorb the canonical response.
        transcript
            .append_bytes(b"fold-v", &serialize_elements(&self.ring, &v))
            .map_err(LedgerError::Transcript)?;
        let response = encode_response(&v_coeffs)
            .map_err(|e| LedgerError::Layout(format!("response encode: {e}")))?;

        Ok(CompactOpening {
            params: self.params.clone(),
            widths: self.packed.shape.widths.clone(),
            u_tilde,
            response,
        })
    }
}

/// Verify a compact bundle opening given the carrier's terminal (r_sc, w).
#[allow(clippy::too_many_arguments)]
pub fn verify_compact_opening(
    seed: [u8; 32],
    commitment_bytes: &[u8],
    factor_lens: &[usize],
    flat_log: usize,
    r_sc: &[Goldilocks],
    w: &Goldilocks,
    opening: &CompactOpening,
    transcript: &mut Transcript,
) -> Result<(), LedgerError> {
    let ring = column_ring()?;
    let params = &opening.params;
    let shape = PackShape {
        factor_lens: factor_lens.to_vec(),
        widths: opening.widths.clone(),
        r: params.r,
        flat_log,
    };
    let log_r = params.r.trailing_zeros() as usize;
    if flat_log < log_r || r_sc.len() != flat_log {
        return Err(LedgerError::Layout("r_sc arity".into()));
    }
    let r_head = &r_sc[..flat_log - log_r];
    let r_tail = &r_sc[flat_log - log_r..];

    // 1. The ũ's + interpolation check.
    if opening.u_tilde.len() != params.r {
        return Err(LedgerError::Layout("u count".into()));
    }
    let mut ubuf: Vec<u8> = Vec::with_capacity(8 * params.r);
    for u in &opening.u_tilde {
        ubuf.extend_from_slice(&u.to_canonical_u64().to_le_bytes());
    }
    transcript
        .append_bytes(b"fold-utilde", &ubuf)
        .map_err(LedgerError::Transcript)?;
    {
        let mut table = vec![Goldilocks::ONE];
        for rr in r_tail.iter().rev() {
            let one_minus = Goldilocks::ONE.sub(rr);
            let mut next = Vec::with_capacity(table.len() * 2);
            for t in &table {
                next.push(t.mul(&one_minus));
            }
            for t in &table {
                next.push(t.mul(rr));
            }
            table = next;
        }
        let mut check = Goldilocks::ZERO;
        for (j, uj) in opening.u_tilde.iter().enumerate() {
            check = check.add(&table[j].mul(uj));
        }
        if check != *w {
            return Err(LedgerError::DerivedMismatch);
        }
    }

    // 2. The challenge replay.
    let d: Vec<i64> = (0..params.r)
        .map(|_| {
            let b = transcript
                .challenge_bytes(b"fold-dchal", 2)
                .map_err(LedgerError::Transcript)?;
            let raw = u16::from_le_bytes([b[0], b[1]]) as u64;
            let m = 2 * params.amplitude as u64 + 1;
            Ok((raw % m) as i64 - params.amplitude as i64)
        })
        .collect::<Result<_, _>>()?;

    // 3. Decode + gate the response.
    let psi = psi_weights_goldilocks(&shape, r_head);
    let n = ring.n();
    let stream = shape.stream_len();
    let n_bar = stream.div_ceil(n).max(1);
    let v_coeffs = decode_response(&opening.response)
        .map_err(|e| LedgerError::Layout(format!("response: {e}")))?;
    if v_coeffs.len() != n_bar * n {
        return Err(LedgerError::Layout("response length".into()));
    }
    let gate = params.gate as i64;
    if gate >= (ring.modulus.q / 2) as i64 {
        return Err(LedgerError::Layout("gate exceeds q/2".into()));
    }
    let mut v: Vec<RingElement> = Vec::with_capacity(n_bar);
    for chunk in v_coeffs.chunks(n) {
        let mut coeffs = vec![0u32; n];
        for (i, &c) in chunk.iter().enumerate() {
            if (c as i64).abs() > gate {
                return Err(LedgerError::Layout("norm gate".into()));
            }
            coeffs[i] = (c as i64).rem_euclid(ring.modulus.q as i64) as u32;
        }
        v.push(RingElement::from_coeffs(&ring, coeffs));
    }
    transcript
        .append_bytes(b"fold-v", &serialize_elements(&ring, &v))
        .map_err(LedgerError::Transcript)?;

    // 4. The functional check: Φ(v) = Σ_j d_j·ũ_j over Goldilocks.
    let q = ring.modulus.q;
    let mut phi_v = Goldilocks::ZERO;
    for (i, elem) in v.iter().enumerate() {
        let mut elem_sum = Goldilocks::ZERO;
        for (kk, &c) in elem.coeffs().iter().enumerate() {
            let weight = psi.get(i * n + kk).copied().unwrap_or(Goldilocks::ZERO);
            if weight != Goldilocks::ZERO {
                let t = phi_term(&weight, c, q);
                elem_sum = elem_sum.add(&t);
            }
        }
        phi_v = phi_v.add(&elem_sum);
    }
    let mut rhs = Goldilocks::ZERO;
    for (j, &dj) in d.iter().enumerate() {
        if dj != 0 {
            let term = opening.u_tilde[j].mul(&Goldilocks::from_u64(dj.unsigned_abs()));
            rhs = if dj < 0 { rhs.sub(&term) } else { rhs.add(&term) };
        }
    }
    if phi_v != rhs {
        return Err(LedgerError::DerivedMismatch);
    }

    // 5. The Ajtai fold check: F̄·v = Σ_j d_j·y_j over R_q.
    let y_flat = deserialize_elements(&ring, commitment_bytes)
        .map_err(|e| LedgerError::Layout(format!("y: {e}")))?;
    if y_flat.len() != params.r * params.k {
        return Err(LedgerError::Layout("y count".into()));
    }
    let ajtai = AjtaiParams {
        ring: ring.clone(),
        k: params.k,
        m: n_bar,
        norm_bound: params.gate,
    };
    let key = AjtaiPublicKey::from_seed(ajtai, seed).map_err(LedgerError::Ajtai)?;
    let commitment = key.commit(&v).map_err(LedgerError::Ajtai)?;
    let mut y_comb = vec![ring.zero(); params.k];
    for (j, &dj) in d.iter().enumerate() {
        if dj == 0 {
            continue;
        }
        for kk in 0..params.k {
            let prod = y_flat[j * params.k + kk].scale_i64(dj);
            y_comb[kk] = y_comb[kk]
                .add(&prod)
                .map_err(|e| LedgerError::Layout(format!("ring add: {e:?}")))?;
        }
    }
    for kk in 0..params.k {
        if commitment.rows[kk].coeffs() != y_comb[kk].coeffs() {
            return Err(LedgerError::Ajtai(
                lattice_commitment::ajtai::AjtaiError::CommitmentMismatch,
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The Sound profile (DESIGN_50KB Stage 5.2 — the LaBRADOR decider wiring)
// ---------------------------------------------------------------------------

/// The compact mode's binding profile.
///
/// * `Clear` — the shipped single-level fold: the response `v` is
///   transmitted and bound by the level-1 instance `[F̄ | −y]` — the
///   Stage 5.1 estimator verdict's BROKEN regime at benchmark response
///   lengths (~2^12 bits; the interim `k = 4` hardening ships at 33 KB
///   with the honest caveat).
/// * `Sound` — the LaBRADOR decider: the level-1 response is never
///   transmitted; it is width-folded (the quadratic-garbage tail,
///   [`crate::width_fold`]) into the narrow response `z` bound by the
///   fresh `[A₂ | −T]` instance at the estimator's sound row. The
///   binding of the WHOLE opening is the width fold's instance — the
///   level-1 `[F̄ | −y]` never arises (the `y_j` enter only through
///   the public target `t = Σ_j d_j·y_j`, which `(W0)` pins).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactProfile {
    Clear,
    Sound,
}

impl FoldParams {
    /// The Sound profile's level-1 parameters: amplitude `A₁ = 2^4`
    /// (the estimator ceiling's regime — `β₁ = r·A₁·255 ≤ 2^20` at
    /// `r ≤ 128` keeps the width fold's `[A₂ | −T]` at the sound row
    /// `(w, r₂, κ, A₂) = (8, 2, 8, 2^2)`; the Clear mode's `2^6`
    /// doubles `β₁` per step and exits the sound regime by `r = 64`).
    pub fn sound(r: usize, k: usize) -> Self {
        let amplitude = 1u32 << 4;
        let gate = (r as u64) * (amplitude as u64) * 255;
        FoldParams {
            r,
            k,
            amplitude,
            gate: gate.min((Modulus32::Q_32.q / 2 - 1) as u64) as u32,
        }
    }
}

/// The Sound-profile fold parameters from the packed size: the level-1
/// column count `r₁` chosen so the response width lands in the width
/// fold's cheap sound row (`n̄ ≤ 16`, i.e. `r₁ ≈ stream/1024`), with
/// the amplitude the ceiling allows.
///
/// **The recursive staging takeover**: when re-packing cannot land
/// `n̄ ≤ 16` (the packing cap `r₁ ≤ min(128, factor length)` binds, or
/// `r₁` saturates at 128), the column count parks at the cap and the
/// RECURSIVE width-collapse chain stages the (now wider) response —
/// the coverage extension from the single-stage `n̄ ≤ 16` to the
/// benchmark streams (`lattice-widthfold::chain` publishes the
/// measured boundary table).
///
/// Fail-closed when the stream exceeds the staged coverage at the
/// resulting `β₁` (the honest Q_32 ceiling: the Modulus-50 class is
/// the documented follow-up).
pub fn sound_fold_params_for(
    total_values: usize,
    max_value_bytes: usize,
    min_factor_len: usize,
) -> Result<(usize, usize), String> {
    let stream = total_values.saturating_mul(max_value_bytes.max(1));
    // The width fold's cheap sound row covers n̄ ≤ 16: r₁ ≥ stream/1024.
    let mut r = 4usize;
    while r < 128 && stream.div_ceil(r * 64) > 16 {
        r *= 2;
    }
    // The packing constraint (r ≤ every factor length). When the
    // re-packing route cannot land n̄ ≤ 16, park at the cap: the
    // recursive width-collapse chain takes over from there.
    let cap = min_factor_len.next_power_of_two().min(128).max(1);
    if r > cap {
        r = cap;
    }
    let n_bar = stream.div_ceil(r * 64).max(1);
    // β₁ must sit under the staged sound ceiling (the chain's budget
    // runs from β₁ to the final row's gate — the estimator's verdict
    // at prove time fails closed beyond the coverage).
    let beta1 = (r as u64) * 16 * 255;
    if beta1 > (1 << 20) {
        return Err(format!(
            "sound profile: beta1 {beta1} exceeds the staged ceiling 2^20 at r={r}"
        ));
    }
    let _ = n_bar;
    Ok((r, 4usize))
}

/// The Sound-profile opening artifact: the level-1 fold's PUBLIC layer
/// (the ũ values the carrier's interpolation consumes) plus the
/// RECURSIVE width-collapse chain that replaces the transmitted
/// response (the log-staged sound rows — benchmark-stream coverage).
#[derive(Clone, Debug)]
pub struct SoundOpening {
    /// The level-1 fold parameters (public shape).
    pub params: FoldParams,
    /// Per-factor byte widths (the packing shape).
    pub widths: Vec<u8>,
    /// Per-column Goldilocks values ũ_j (r₁ field elements — the
    /// carrier's public layer, absorbed before the challenges).
    pub u_tilde: Vec<Goldilocks>,
    /// The recursive width-fold chain over the (never-transmitted)
    /// response (the staged sound rows).
    pub width_proof: crate::width_fold::WidthChainProof,
}

impl CompactBundleProver {
    /// The level-1 key's column blocks (the prover's view): block c =
    /// (F̄[0][c], …, F̄[k−1][c]) — the width fold's key-group source.
    fn key_blocks(&self) -> Vec<Vec<RingElement>> {
        (0..self.n_bar)
            .map(|c| {
                (0..self.params.k)
                    .map(|rr| self.key.entry(rr, c).cloned().unwrap_or_else(|| self.ring.zero()))
                    .collect()
            })
            .collect()
    }

    /// Prove the Sound opening: the level-1 fold's public layer (ũ's,
    /// interpolation self-check, the d challenges, the never-transmitted
    /// response `v` with its target/functional self-checks), then the
    /// LaBRADOR width fold over `v`.
    pub fn prove_sound_opening(
        &self,
        r_sc: &[Goldilocks],
        w: &Goldilocks,
        transcript: &mut Transcript,
    ) -> Result<SoundOpening, LedgerError> {
        use crate::second_fold::{apply_key, functional_of};

        let flat_log = self.packed.shape.flat_log;
        let log_r = self.params.r.trailing_zeros() as usize;
        if r_sc.len() != flat_log {
            return Err(LedgerError::Layout("r_sc arity".into()));
        }
        let r_head = &r_sc[..flat_log - log_r];
        let r_tail = &r_sc[flat_log - log_r..];

        // 1. The per-column values ũ_j + the interpolation self-check
        //    (identical to the Clear mode — the carrier's public layer).
        let psi = psi_weights_goldilocks(&self.packed.shape, r_head);
        let n = self.ring.n();
        let n_bar = self.n_bar;
        let mut u_tilde: Vec<Goldilocks> = Vec::with_capacity(self.params.r);
        for j in 0..self.params.r {
            let mut acc = Goldilocks::ZERO;
            for m in 0..psi.len() {
                if let Some(&b) = self.packed.columns[j].get(m) {
                    acc = acc.add(&psi[m].mul(&Goldilocks::from_u64(b as u64)));
                }
            }
            u_tilde.push(acc);
        }
        {
            let mut table = vec![Goldilocks::ONE];
            for rr in r_tail.iter().rev() {
                let one_minus = Goldilocks::ONE.sub(rr);
                let mut next = Vec::with_capacity(table.len() * 2);
                for t in &table {
                    next.push(t.mul(&one_minus));
                }
                for t in &table {
                    next.push(t.mul(rr));
                }
                table = next;
            }
            let mut check = Goldilocks::ZERO;
            for (j, uj) in u_tilde.iter().enumerate() {
                check = check.add(&table[j].mul(uj));
            }
            if check != *w {
                return Err(LedgerError::Layout(format!(
                    "interpolation self-check failed: {check:?} vs {w:?}"
                )));
            }
        }
        // Absorb the ũ's BEFORE the challenges (the Clear mode's order).
        let mut ubuf: Vec<u8> = Vec::with_capacity(8 * u_tilde.len());
        for u in &u_tilde {
            ubuf.extend_from_slice(&u.to_canonical_u64().to_le_bytes());
        }
        transcript
            .append_bytes(b"fold-utilde", &ubuf)
            .map_err(LedgerError::Transcript)?;

        // 2. The scalar challenges d_j ∈ [−A₁, A₁].
        let d: Vec<i64> = (0..self.params.r)
            .map(|_| {
                let b = transcript
                    .challenge_bytes(b"fold-dchal", 2)
                    .map_err(LedgerError::Transcript)?;
                let raw = u16::from_le_bytes([b[0], b[1]]) as u64;
                let m = 2 * self.params.amplitude as u64 + 1;
                Ok((raw % m) as i64 - self.params.amplitude as i64)
            })
            .collect::<Result<_, _>>()?;

        // 3. The never-transmitted response v = Σ_j d_j·w_j (the gate
        //    keeps every coefficient < q/2 so the mod-q rep is exact).
        let gate = self.params.gate as i64;
        if gate >= (self.ring.modulus.q / 2) as i64 {
            return Err(LedgerError::Layout("gate exceeds q/2".into()));
        }
        let mut v: Vec<RingElement> = vec![self.ring.zero(); n_bar];
        for (j, &dj) in d.iter().enumerate() {
            if dj == 0 {
                continue;
            }
            for i in 0..n_bar {
                let prod = self.columns[j][i].scale_i64(dj);
                v[i] = v[i].add(&prod).map_err(|e| {
                    LedgerError::Layout(format!("ring add: {e:?}"))
                })?;
            }
        }
        for elem in &v {
            for &c in elem.coeffs() {
                let balanced = if c > self.ring.modulus.q / 2 {
                    c as i64 - self.ring.modulus.q as i64
                } else {
                    c as i64
                };
                if balanced.abs() > gate {
                    return Err(LedgerError::Layout(format!(
                        "norm gate: |{balanced}| > {gate}"
                    )));
                }
            }
        }
        // 4. The public target/functional self-checks: t = F̄·v ≟ Σ d_j·y_j
        //    and Φ(v) ≟ Σ d_j·ũ_j (the verifier computes both sides from
        //    public data; the width fold's (W0)/(W0') pin them to the
        //    folded response).
        let q = u64::from(self.ring.modulus.q);
        let blocks = self.key_blocks();
        let t_target = apply_key(&self.ring, &blocks, &v, self.params.k);
        {
            let mut y_comb = vec![self.ring.zero(); self.params.k];
            for (j, &dj) in d.iter().enumerate() {
                if dj == 0 {
                    continue;
                }
                for kk in 0..self.params.k {
                    let prod = self.y[j][kk].scale_i64(dj);
                    y_comb[kk] = y_comb[kk]
                        .add(&prod)
                        .map_err(|e| LedgerError::Layout(format!("ring add: {e:?}")))?;
                }
            }
            for kk in 0..self.params.k {
                if t_target[kk].coeffs() != y_comb[kk].coeffs() {
                    return Err(LedgerError::Layout(
                        "self-check: F̄·v ≠ Σ d_j·y_j (the level-1 fold)".into(),
                    ));
                }
            }
        }
        let u_target = functional_of(&self.ring, &v, &psi, q);
        {
            let mut rhs = Goldilocks::ZERO;
            for (j, &dj) in d.iter().enumerate() {
                if dj != 0 {
                    let term = u_tilde[j].mul(&Goldilocks::from_u64(dj.unsigned_abs()));
                    rhs = if dj < 0 { rhs.sub(&term) } else { rhs.add(&term) };
                }
            }
            if u_target != rhs {
                return Err(LedgerError::Layout(
                    "self-check: Φ(v) ≠ Σ d_j·ũ_j (the level-1 functional)".into(),
                ));
            }
        }

        // 5. The RECURSIVE width-collapse chain over v (the staged
        //    estimator-sound rows; fail-closed below the floor + the
        //    grinding allowance). The single-stage cheap row covers
        //    n̄ ≤ 16; the staging takes the coverage to the benchmark
        //    streams.
        let beta1 = self.params.gate as u64;
        let chain_params = crate::width_fold::WidthChainParams::sound_chain_for(
            n_bar,
            beta1,
            q,
            n as u64,
        )
        .map_err(|e| LedgerError::Layout(format!("width-chain schedule: {e}")))?;
        let width_proof = crate::width_fold::prove_width_fold_chain(
            &self.ring,
            &v,
            &t_target,
            &u_target,
            &blocks,
            self.params.k,
            &psi,
            chain_params,
            beta1,
            self.seed,
            transcript,
        )
        .map_err(|e| LedgerError::Layout(format!("width chain: {e}")))?;

        Ok(SoundOpening {
            params: self.params.clone(),
            widths: self.packed.shape.widths.clone(),
            u_tilde,
            width_proof,
        })
    }
}

/// Verify a Sound-profile opening given the carrier's terminal
/// `(r_sc, w)`. Replays the level-1 public layer (the ũ interpolation,
/// the d challenges), computes the PUBLIC target
/// `t = Σ_j d_j·y_j` and functional `u = Σ_j d_j·ũ_j` from the
/// bundle's commitments, regenerates the level-1 key blocks, and hands
/// everything to the width fold's verifier (the `(W0)`–`(W4)` checks
/// plus the estimator posture gate).
#[allow(clippy::too_many_arguments)]
pub fn verify_sound_opening(
    seed: [u8; 32],
    commitment_bytes: &[u8],
    factor_lens: &[usize],
    flat_log: usize,
    r_sc: &[Goldilocks],
    w: &Goldilocks,
    opening: &SoundOpening,
    transcript: &mut Transcript,
) -> Result<(), LedgerError> {
    use crate::second_fold::{apply_key, functional_of};

    let ring = column_ring()?;
    let params = &opening.params;
    let shape = PackShape {
        factor_lens: factor_lens.to_vec(),
        widths: opening.widths.clone(),
        r: params.r,
        flat_log,
    };
    let log_r = params.r.trailing_zeros() as usize;
    if flat_log < log_r || r_sc.len() != flat_log {
        return Err(LedgerError::Layout("r_sc arity".into()));
    }
    let r_head = &r_sc[..flat_log - log_r];
    let r_tail = &r_sc[flat_log - log_r..];

    // 1. The ũ's + the interpolation check (the Clear mode's public
    //    layer — identical).
    if opening.u_tilde.len() != params.r {
        return Err(LedgerError::Layout("u count".into()));
    }
    let mut ubuf: Vec<u8> = Vec::with_capacity(8 * params.r);
    for u in &opening.u_tilde {
        ubuf.extend_from_slice(&u.to_canonical_u64().to_le_bytes());
    }
    transcript
        .append_bytes(b"fold-utilde", &ubuf)
        .map_err(LedgerError::Transcript)?;
    {
        let mut table = vec![Goldilocks::ONE];
        for rr in r_tail.iter().rev() {
            let one_minus = Goldilocks::ONE.sub(rr);
            let mut next = Vec::with_capacity(table.len() * 2);
            for t in &table {
                next.push(t.mul(&one_minus));
            }
            for t in &table {
                next.push(t.mul(rr));
            }
            table = next;
        }
        let mut check = Goldilocks::ZERO;
        for (j, uj) in opening.u_tilde.iter().enumerate() {
            check = check.add(&table[j].mul(uj));
        }
        if check != *w {
            return Err(LedgerError::DerivedMismatch);
        }
    }

    // 2. The challenge replay.
    let d: Vec<i64> = (0..params.r)
        .map(|_| {
            let b = transcript
                .challenge_bytes(b"fold-dchal", 2)
                .map_err(LedgerError::Transcript)?;
            let raw = u16::from_le_bytes([b[0], b[1]]) as u64;
            let m = 2 * params.amplitude as u64 + 1;
            Ok((raw % m) as i64 - params.amplitude as i64)
        })
        .collect::<Result<_, _>>()?;

    // 3. The PUBLIC target and functional (the verifier's own
    //    computation — never prover-supplied).
    let psi = psi_weights_goldilocks(&shape, r_head);
    let n = ring.n();
    let stream = shape.stream_len();
    let n_bar = stream.div_ceil(n).max(1);
    let y_flat = deserialize_elements(&ring, commitment_bytes)
        .map_err(|e| LedgerError::Layout(format!("y: {e}")))?;
    if y_flat.len() != params.r * params.k {
        return Err(LedgerError::Layout("y count".into()));
    }
    let mut t_target = vec![ring.zero(); params.k];
    for (j, &dj) in d.iter().enumerate() {
        if dj == 0 {
            continue;
        }
        for kk in 0..params.k {
            let prod = y_flat[j * params.k + kk].scale_i64(dj);
            t_target[kk] = t_target[kk]
                .add(&prod)
                .map_err(|e| LedgerError::Layout(format!("ring add: {e:?}")))?;
        }
    }
    let mut u_target = Goldilocks::ZERO;
    for (j, &dj) in d.iter().enumerate() {
        if dj != 0 {
            let term = opening.u_tilde[j].mul(&Goldilocks::from_u64(dj.unsigned_abs()));
            u_target = if dj < 0 { u_target.sub(&term) } else { u_target.add(&term) };
        }
    }
    let _ = functional_of; // (the functional enters through (W0') only)

    // 4. The level-1 key regeneration (the verifier's blocks).
    let ajtai = AjtaiParams {
        ring: ring.clone(),
        k: params.k,
        m: n_bar,
        norm_bound: params.gate,
    };
    let key = AjtaiPublicKey::from_seed(ajtai, seed).map_err(LedgerError::Ajtai)?;
    let blocks: Vec<Vec<RingElement>> = (0..n_bar)
        .map(|c| {
            (0..params.k)
                .map(|rr| key.entry(rr, c).cloned().unwrap_or_else(|| ring.zero()))
                .collect()
        })
        .collect();
    let _ = apply_key;

    // 5. The recursive width-collapse chain's verifier: the staged
    //    (W0)–(W4) checks + the per-stage posture gates + the derived
    //    public-claim threading.
    let beta1 = params.gate as u64;
    crate::width_fold::verify_width_fold_chain(
        &ring,
        &t_target,
        &u_target,
        &blocks,
        params.k,
        &psi,
        beta1,
        seed,
        &opening.width_proof,
        transcript,
    )
    .map_err(|e| LedgerError::Layout(format!("width chain: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// Synthetic bundle: descending sizes (128, 64, 64) so offsets are
    /// length-aligned (the invariant the real ledger gets from uniform
    /// tensor shapes).
    fn synthetic_entries() -> Vec<(u32, DenseMle)> {
        let mut out = Vec::new();
        let vals2: Vec<u64> = (0..128)
            .map(|i| (i as u64 * 48271 + 11) % 4_000_000_000)
            .collect();
        out.push((
            0,
            DenseMle {
                num_vars: 7,
                evaluations: vals2.iter().map(|&v| fe(v)).collect(),
            },
        ));
        let vals: Vec<u64> = (0..64).map(|i| (i * 2654435761) % 100_000).collect();
        out.push((
            1,
            DenseMle {
                num_vars: 6,
                evaluations: vals.iter().map(|&v| fe(v)).collect(),
            },
        ));
        let bits: Vec<u64> = (0..64).map(|i| (i % 3 == 0) as u64).collect();
        out.push((
            2,
            DenseMle {
                num_vars: 6,
                evaluations: bits.iter().map(|&v| fe(v)).collect(),
            },
        ));
        out
    }

    fn random_point(len: usize, seed: u64) -> Vec<Goldilocks> {
        (0..len)
            .map(|i| fe((seed.wrapping_mul(2_654_435_761).wrapping_add(i as u64)) % 1_000_000_007))
            .collect()
    }

    #[test]
    fn compact_opening_e2e_honest_and_tamper() {
        let entries = synthetic_entries();
        let refs: Vec<(u32, &DenseMle)> = entries.iter().map(|(d, m)| (*d, m)).collect();
        let seed = [42u8; 32];
        let (r, k) = (4usize, 2usize);
        let prover = compact_bundle_commit(&refs, seed, r, k).unwrap();
        let commitment = prover.commitment_bytes();
        let factor_lens: Vec<usize> = entries.iter().map(|(_, m)| m.evaluations.len()).collect();
        let flat_len: usize = factor_lens.iter().sum();
        let flat_log = flat_len.next_power_of_two().trailing_zeros() as usize;

        // The carrier's terminal (the test stands in for the carrier: a
        // random point + the flat MLE's true evaluation).
        let r_sc = random_point(flat_log, 12345);
        let w = prover.flat_mle().evaluate(&r_sc).unwrap();

        let mut transcript = Transcript::new_default(b"compact-e2e");
        transcript
            .append_bytes(b"bundle-commitment", &commitment)
            .unwrap();
        let opening = prover
            .prove_compact_opening(&r_sc, &w, &mut transcript)
            .unwrap();

        // Honest verify.
        let mut vt = Transcript::new_default(b"compact-e2e");
        vt.append_bytes(b"bundle-commitment", &commitment).unwrap();
        verify_compact_opening(
            seed,
            &commitment,
            &factor_lens,
            flat_log,
            &r_sc,
            &w,
            &opening,
            &mut vt,
        )
        .unwrap();

        // Size accounting (commitments + u's + response).
        let bytes = commitment.len()
            + opening.u_tilde.len() * 8
            + opening.response.hist.len()
            + opening.response.payload.len()
            + opening.response.raw.len()
            + 64;
        println!("compact opening size (synthetic): {bytes} B");
        assert!(bytes < 20_000);

        // ---- Tamper suite ----
        let verify_with = |commitment: &[u8],
                           r_sc: &[Goldilocks],
                           w: &Goldilocks,
                           opening: &CompactOpening|
         -> Result<(), LedgerError> {
            let mut vt = Transcript::new_default(b"compact-e2e");
            vt.append_bytes(b"bundle-commitment", commitment).unwrap();
            verify_compact_opening(
                seed,
                commitment,
                &factor_lens,
                flat_log,
                r_sc,
                w,
                opening,
                &mut vt,
            )
        };

        // Wrong claim: the interpolation check rejects.
        let w_bad = w.add(&fe(1));
        assert!(verify_with(&commitment, &r_sc, &w_bad, &opening).is_err());

        // Wrong point: the functional check rejects.
        let mut r_bad = r_sc.clone();
        r_bad[0] = r_bad[0].add(&fe(1));
        assert!(verify_with(&commitment, &r_bad, &w, &opening).is_err());

        // Tampered u: the interpolation or functional check rejects.
        let mut bad_u = opening.clone();
        bad_u.u_tilde[0] = bad_u.u_tilde[0].add(&fe(1));
        assert!(verify_with(&commitment, &r_sc, &w, &bad_u).is_err());

        // Tampered response: the functional or Ajtai check rejects.
        let mut bad_resp = opening.clone();
        if let Some(x) = bad_resp.response.raw.first_mut() {
            *x ^= 0x40;
        }
        assert!(verify_with(&commitment, &r_sc, &w, &bad_resp).is_err());

        // Tampered commitment: the Ajtai check rejects.
        let mut bad_commit = commitment.clone();
        bad_commit[12] ^= 0xFF;
        assert!(verify_with(&bad_commit, &r_sc, &w, &opening).is_err());

        // Wrong seed: the key mismatch rejects.
        let mut vt = Transcript::new_default(b"compact-e2e");
        vt.append_bytes(b"bundle-commitment", &commitment).unwrap();
        assert!(verify_compact_opening(
            [44u8; 32],
            &commitment,
            &factor_lens,
            flat_log,
            &r_sc,
            &w,
            &opening,
            &mut vt
        )
        .is_err());

        // Tampered widths: the layout mismatch rejects (different stream).
        let mut bad_w = opening.clone();
        if let Some(x) = bad_w.widths.first_mut() {
            *x += 1;
        }
        assert!(verify_with(&commitment, &r_sc, &w, &bad_w).is_err());
    }

    /// The Sound-profile opening end-to-end: the same bundle through
    /// `FoldParams::sound` (the amplitude the width fold's estimator
    /// ceiling allows) + the RECURSIVE width-collapse chain, with the
    /// tamper suite. The binding is the chain's per-stage `[A₂ | −T]`
    /// instances — the level-1 response is never transmitted.
    #[test]
    fn test_sound_opening_e2e() {
        let entries = synthetic_entries();
        let refs: Vec<(u32, &DenseMle)> = entries.iter().map(|(d, m)| (*d, m)).collect();
        let seed = [9u8; 32];
        let r = 4usize;
        let k = 4usize;
        let prover = compact_bundle_commit_with_params(
            &refs,
            seed,
            r,
            k,
            FoldParams::sound(r, k),
        )
        .unwrap();
        let commitment = prover.commitment_bytes();
        let factor_lens: Vec<usize> = entries.iter().map(|(_, m)| m.evaluations.len()).collect();
        let flat_len: usize = factor_lens.iter().sum();
        let flat_log = flat_len.next_power_of_two().trailing_zeros() as usize;

        let r_sc = random_point(flat_log, 54321);
        let w = prover.flat_mle().evaluate(&r_sc).unwrap();

        let mut transcript = Transcript::new_default(b"sound-e2e");
        transcript
            .append_bytes(b"bundle-commitment", &commitment)
            .unwrap();
        let opening = prover
            .prove_sound_opening(&r_sc, &w, &mut transcript)
            .unwrap();

        // Honest verify.
        let mut vt = Transcript::new_default(b"sound-e2e");
        vt.append_bytes(b"bundle-commitment", &commitment).unwrap();
        verify_sound_opening(
            seed,
            &commitment,
            &factor_lens,
            flat_log,
            &r_sc,
            &w,
            &opening,
            &mut vt,
        )
        .unwrap();

        // Size accounting: the commitment + ũ's + the staged chain's
        // artifacts (per stage: garbage + images + inner + z + the
        // functional terms).
        let wc = &opening.width_proof;
        let mut wf_bytes = 0usize;
        for st in &wc.stages {
            wf_bytes += st.p_images.len()
                + st.garbage.len()
                + st.t_inner.len()
                + st.u_parts.len()
                + st.g_func.len()
                + st.response.hist.len()
                + st.response.payload.len()
                + st.response.raw.len()
                + 16;
        }
        let bytes = commitment.len() + opening.u_tilde.len() * 8 + wf_bytes + 64;
        println!(
            "sound opening size (synthetic, r={r}, stages={}): {bytes} B",
            wc.stages.len()
        );
        // The honest multiple vs the Clear opening (~20 KB ceiling at
        // this scale): the quadratic garbage's price. The BENCHMARKS
        // §2j table records the measured ratio at the real bundles.
        assert!(bytes < 150_000);

        // ---- Tamper suite ----
        let verify_with = |commitment: &[u8],
                           r_sc: &[Goldilocks],
                           w: &Goldilocks,
                           opening: &SoundOpening|
         -> Result<(), LedgerError> {
            let mut vt = Transcript::new_default(b"sound-e2e");
            vt.append_bytes(b"bundle-commitment", commitment).unwrap();
            verify_sound_opening(
                seed,
                commitment,
                &factor_lens,
                flat_log,
                r_sc,
                w,
                opening,
                &mut vt,
            )
        };

        // Wrong claim: the interpolation check rejects.
        let w_bad = w.add(&fe(1));
        assert!(verify_with(&commitment, &r_sc, &w_bad, &opening).is_err());

        // Wrong point: the ψ weights change, the functional layer
        // (W0'/W3) rejects.
        let mut r_bad = r_sc.clone();
        r_bad[0] = r_bad[0].add(&fe(1));
        assert!(verify_with(&commitment, &r_bad, &w, &opening).is_err());

        // Tampered ũ: the interpolation or the W0' chain rejects.
        let mut bad_u = opening.clone();
        bad_u.u_tilde[0] = bad_u.u_tilde[0].add(&fe(1));
        assert!(verify_with(&commitment, &r_sc, &w, &bad_u).is_err());

        // Tampered final-stage response (z): W2 (the MSIS binding) or
        // W1 rejects.
        let mut bad_z = opening.clone();
        {
            let last = bad_z.width_proof.stages.len() - 1;
            let mut coeffs =
                decode_response(&bad_z.width_proof.stages[last].response).unwrap();
            assert!(!coeffs.is_empty());
            coeffs[0] = coeffs[0].wrapping_add(1);
            bad_z.width_proof.stages[last].response = encode_response(&coeffs).unwrap();
        }
        assert!(verify_with(&commitment, &r_sc, &w, &bad_z).is_err());

        // Tampered quadratic garbage: W1 (the exact fold identity)
        // rejects.
        let mut bad_g = opening.clone();
        assert!(!bad_g.width_proof.stages[0].garbage.is_empty());
        bad_g.width_proof.stages[0].garbage[5] ^= 0x10;
        assert!(verify_with(&commitment, &r_sc, &w, &bad_g).is_err());

        // Tampered part images: W0/W1 rejects.
        let mut bad_p = opening.clone();
        assert!(!bad_p.width_proof.stages[0].p_images.is_empty());
        bad_p.width_proof.stages[0].p_images[9] ^= 0x20;
        assert!(verify_with(&commitment, &r_sc, &w, &bad_p).is_err());

        // Tampered commitment: the W0 target chain rejects.
        let mut bad_commit = commitment.clone();
        bad_commit[12] ^= 0xFF;
        assert!(verify_with(&bad_commit, &r_sc, &w, &opening).is_err());

        // Wrong seed: the regenerated key blocks differ, W1 rejects.
        let mut vt = Transcript::new_default(b"sound-e2e");
        vt.append_bytes(b"bundle-commitment", &commitment).unwrap();
        assert!(verify_sound_opening(
            [44u8; 32],
            &commitment,
            &factor_lens,
            flat_log,
            &r_sc,
            &w,
            &opening,
            &mut vt
        )
        .is_err());
    }

    /// The Sound fold-parameter chooser: the re-packing route while it
    /// fits, the recursive-staging takeover beyond it, and the honest
    /// staged-coverage ceiling (the Q_32 boundary).
    #[test]
    fn test_sound_fold_params_chooser() {
        // Small stream: the cheap sound row (n̄ ≤ 16) — re-packing.
        let (r, k) = sound_fold_params_for(4096, 1, 256).unwrap();
        assert_eq!(k, 4);
        assert!(r >= 4);
        // The β₁ ceiling: at r = 128 the gate 128·16·255 = 522,240 ≤ 2^20.
        assert!((r as u64) * 16 * 255 <= (1 << 20));
        // The recursive-staging takeover: a 1 MB stream at the packing
        // cap 64 parks at r = 64 (n̄ = 256) — the chain stages it (the
        // coverage extension this wave lands; previously fail-closed).
        let (r2, _) = sound_fold_params_for(1 << 20, 1, 64).unwrap();
        assert_eq!(r2, 64);
        // The staged coverage ceiling: a 64 MB stream parks at r = 64
        // (n̄ = 16,384 at β₁ ≈ 2^18) — the CHOOSER returns the legal
        // shape, and the CHAIN's schedule search fails closed beyond the
        // staged budget (the honest Q_32 boundary; the Modulus-50 class
        // is the documented follow-up).
        let (r3, _) = sound_fold_params_for(1 << 26, 1, 64).unwrap();
        assert_eq!(r3, 64);
        let n_bar3 = (1u64 << 26).div_ceil((r3 * 64) as u64) as usize;
        let ring = column_ring().unwrap();
        assert!(
            crate::width_fold::WidthChainParams::sound_chain_for(
                n_bar3,
                (r3 as u64) * 16 * 255,
                u64::from(ring.modulus.q),
                ring.n() as u64,
            )
            .is_err(),
            "the staged coverage must fail closed beyond the Q_32 budget"
        );
    }

    /// The pack layout roundtrip: unpacking the columns recovers the flat
    /// values (byte recomposition).
    #[test]
    fn test_pack_roundtrip() {
        let entries = synthetic_entries();
        let refs: Vec<(u32, &DenseMle)> = entries.iter().map(|(d, m)| (*d, m)).collect();
        let r = 4usize;
        let packed = pack_columns(&refs, r).unwrap();
        let flat = &packed.flat;
        // Rebuild each flat position's value from the column streams.
        let shape = &packed.shape;
        let _flat_pad = shape.flat_pad();
        for p in 0..flat_len_of(&refs) {
            let j = p % r;
            let h = p / r;
            let w = shape.width_at(h * r); // = width_at(p) (r-aligned)
            let mut m = 0usize;
            // Find the stream offset of head h: count bytes of heads < h.
            for hh in 0..h {
                m += shape.width_at(hh * r);
            }
            let mut val = 0u64;
            for b in 0..w {
                val |= (packed.columns[j][m + b] as u64) << (8 * b);
            }
            assert_eq!(
                Goldilocks::from_u64(val),
                flat.evaluations[p],
                "position {p} mismatch"
            );
        }
    }

    fn flat_len_of(entries: &[(u32, &DenseMle)]) -> usize {
        let l: usize = entries.iter().map(|(_, m)| m.evaluations.len()).sum();
        let _ = flat_pad_dummy(entries);
        l
    }

    fn flat_pad_dummy(_entries: &[(u32, &DenseMle)]) -> usize {
        0
    }
}
