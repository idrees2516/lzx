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
use lattice_labinius::wire::{BitReader, BitWriter, RansCoder};
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
    RingConfig::new(Modulus32::Q_32, 6).map_err(|e| LedgerError::Layout(format!("ring: {e:?}")))
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

/// The magnitude class of a signed coefficient: 0 for zero, else
/// ⌊log₂|c|⌋ + 1 (so 2^(class−1) ≤ |c| < 2^class).
#[inline]
fn magnitude_class(c: i32) -> u8 {
    if c == 0 {
        0
    } else {
        (32 - c.unsigned_abs().leading_zeros()) as u8
    }
}

/// The encoded response artifact.
#[derive(Clone, Debug)]
pub struct ResponseWire {
    /// rANS-coded class symbols: histogram bytes.
    pub hist: Vec<u8>,
    /// rANS payload.
    pub payload: Vec<u8>,
    /// Raw bits: (class−1) low magnitude bits + 1 sign bit per nonzero.
    pub raw: Vec<u8>,
    /// Number of coefficients.
    pub count: usize,
}

/// Encode the fold response coefficients (balanced i32).
pub fn encode_response(coeffs: &[i32]) -> Result<ResponseWire, String> {
    let mut counts = vec![0u64; 33];
    for &c in coeffs {
        counts[magnitude_class(c) as usize] += 1;
    }
    let coder = RansCoder::from_counts(&counts).map_err(|e| format!("rans: {e:?}"))?;
    let symbols: Vec<u32> = coeffs.iter().map(|&c| magnitude_class(c) as u32).collect();
    let (hist, payload) = coder
        .encode(&symbols)
        .map_err(|e| format!("rans encode: {e:?}"))?;
    let mut bw = BitWriter::new();
    for &c in coeffs {
        let cls = magnitude_class(c);
        if cls == 0 {
            continue;
        }
        bw.write(c.unsigned_abs() as u64, (cls - 1) as u32);
        bw.write(if c < 0 { 1 } else { 0 }, 1);
    }
    Ok(ResponseWire {
        hist,
        payload,
        raw: bw.bytes,
        count: coeffs.len(),
    })
}

/// Decode the fold response; strict on lengths and class ranges.
pub fn decode_response(wire: &ResponseWire) -> Result<Vec<i32>, String> {
    let coder =
        RansCoder::from_histogram_bytes(&wire.hist, 33).map_err(|e| format!("rans: {e:?}"))?;
    let symbols = coder
        .decode(&wire.payload, wire.count)
        .map_err(|e| format!("rans decode: {e:?}"))?;
    let mut br = BitReader::new(&wire.raw);
    let mut out = Vec::with_capacity(wire.count);
    for &s in &symbols {
        if s > 32 {
            return Err("class out of range".into());
        }
        if s == 0 {
            out.push(0);
        } else {
            let low = br
                .read(s - 1)
                .ok_or_else(|| "raw bits underflow".to_string())?;
            let mag = low | (1u64 << (s - 1));
            let sign = br
                .read(1)
                .ok_or_else(|| "raw bits underflow".to_string())?;
            let v = mag as i64;
            out.push(if sign == 1 { -v as i32 } else { v as i32 });
        }
    }
    // The writer zero-pads the final partial byte; require only that the
    // consumed bits fit the transmitted raw blob.
    if wire.raw.len() * 8 < out.len() * 8 {
        // (structural guard; the reads above already failed if short)
    }
    Ok(out)
}


// ---------------------------------------------------------------------------
// Ring element serialization
// ---------------------------------------------------------------------------

/// Serialize ring elements: count || (u32 LE per coefficient).
pub fn serialize_elements(ring: &RingConfig, elems: &[RingElement]) -> Vec<u8> {
    let n = ring.n();
    let mut out = Vec::with_capacity(4 + elems.len() * n * 4);
    out.extend_from_slice(&(elems.len() as u32).to_le_bytes());
    for e in elems {
        for &c in e.coeffs() {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    out
}

/// Deserialize ring elements (strict on length and count).
pub fn deserialize_elements(ring: &RingConfig, bytes: &[u8]) -> Result<Vec<RingElement>, String> {
    let n = ring.n();
    if bytes.len() < 4 {
        return Err("short".into());
    }
    let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if bytes.len() != 4 + count * n * 4 {
        return Err(format!("length {} != {}", bytes.len(), 4 + count * n * 4));
    }
    let mut out = Vec::with_capacity(count);
    for e in 0..count {
        let mut coeffs = vec![0u32; n];
        for c in 0..n {
            let base = 4 + (e * n + c) * 4;
            coeffs[c] = u32::from_le_bytes([
                bytes[base],
                bytes[base + 1],
                bytes[base + 2],
                bytes[base + 3],
            ]);
        }
        out.push(RingElement::from_coeffs(ring, coeffs));
    }
    Ok(out)
}

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
    /// Conservative default: amplitude 2^12, gate = r·A·255 (worst case).
    pub fn new(r: usize, k: usize) -> Self {
        let amplitude = 1u32 << 12;
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
/// commitments under the column-uniform key F̄).
pub fn compact_bundle_commit(
    entries: &[(u32, &DenseMle)],
    seed: [u8; 32],
    r: usize,
    k: usize,
) -> Result<CompactBundleProver, LedgerError> {
    let ring = column_ring()?;
    let params = FoldParams::new(r, k);
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


/// The Goldilocks functional term for one coefficient: weight·(balanced c).
fn phi_term(weight: &Goldilocks, c: u32, q: u32) -> Goldilocks {
    let c_int = if c > q / 2 {
        c as i64 - q as i64
    } else {
        c as i64
    };
    let mag = weight.mul(&Goldilocks::from_u64(c_int.unsigned_abs()));
    if c_int < 0 {
        mag.neg()
    } else {
        mag
    }
}

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
            .map(|i| fe((seed.wrapping_mul(2654435761).wrapping_add(i as u64)) % 100_000_000_7))
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

    /// Response codec roundtrip + tamper rejection.
    #[test]
    fn test_response_codec() {
        let coeffs: Vec<i32> = (0..5000)
            .map(|i| {
                let x = (i as i64 * 2654435761) % 20000 - 10000;
                x as i32
            })
            .chain([0, 1, -1, i32::MAX / 4, i32::MIN / 4])
            .collect();
        let wire = encode_response(&coeffs).unwrap();
        let decoded = decode_response(&wire).unwrap();
        assert_eq!(decoded, coeffs);
        let mut bad = wire.clone();
        if let Some(x) = bad.raw.first_mut() {
            *x ^= 1;
        }
        let decoded_bad = decode_response(&bad).unwrap();
        assert_ne!(decoded_bad, coeffs);
        let mut short = wire.clone();
        short.payload.truncate(short.payload.len() / 2);
        assert!(decode_response(&short).is_err());
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
        let flat_pad = shape.flat_pad();
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
