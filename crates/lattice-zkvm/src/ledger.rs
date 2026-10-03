//! The claim ledger and the committed bundles (Wave 7.4's resolver↔PCS
//! glue — the "single highest-leverage missing component" of the pre-wave
//! zkVM).
//!
//! Every staged sumcheck in the zkVM protocol terminates in *factor
//! evaluation claims*. The ledger is the shared bookkeeping that (a) lets
//! the prover **derive** each claim from the committed witness — expanding
//! derived factors (limbs, combos, bit columns) into base claims on the
//! committed tensors — and (b) lets the verifier replay the exact same
//! derivation against the proof's claim list, checking internal
//! consistency. At the end, every base claim is authenticated by exactly
//! one grouped opening per bundle.
//!
//! Two bundles are committed:
//!
//! * **The bits bundle** — every bit-tensor (the six 64-bit value tensors,
//!   the 32-bit instruction tensor, and each memory instance's digit-bit
//!   tensor) flattened into one big MLE and committed **bit-packed**
//!   (31 bits per ring coefficient, the Wave-6.8 sparse-packing layer).
//! * **The values bundle** — the per-instance increment columns
//!   (offset-encoded into `[0, 2^18)`, so the 3×22-bit packing limbs stay
//!   small).
//!
//! Binding chain of a bundle opening (the honest-response discipline):
//! 1. the grouped carrier sumcheck `Σ_i ρ^i·eq(q_i, x)·f(x) = Σ ρ^i·v_i`
//!    over the flat MLE (one sumcheck for ALL claims of the bundle);
//! 2. the verifier reconstructs the response `s` from the opening's
//!    digits, recomputes `A·s = t`, and recomputes `f(r_sc)` from `s` —
//!    tying every claim to the revealed response;
//! 3. the compact norm proof (base-256, 3-4 digits, serialized as `i16`)
//!    certifies every coefficient of the response is small, which is what
//!    makes `verify_opening` binding rather than a free kernel solution.

use std::collections::{HashMap, VecDeque};

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiParams, AjtaiPublicKey};
use lattice_commitment::norm_proof::NormProof;
use lattice_commitment::sparse::{pack_bits, unpack_bits};
use lattice_core::decomposition::GadgetDecomposition;
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::packing::pack_field_elements;
use lattice_ring::{Modulus32, RingConfig, RingElement};
use lattice_sumcheck::sumcheck;
use lattice_sumcheck::SumcheckProof;
use lattice_sumcheck::VirtualPolynomial;

use crate::columns::VALUE_TENSORS;

/// Committed factors (the base-claim address space).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Factor {
    /// The 64-bit tensor of value slot `slot` (6 + log T vars).
    ValueBits { slot: usize },
    /// The 32-bit instruction tensor (5 + log T vars).
    InstrBits,
    /// Memory instance `inst`'s digit-bit tensor (log K + log T_s vars).
    DigitBits { inst: usize },
    /// Auxiliary boolean column over the cycle axis (`BitCol`s live in the
    /// bits bundle; `id` indexes the global auxiliary-column table).
    BitCol { id: usize },
    /// Auxiliary value column over the cycle axis (values bundle).
    ValCol { id: usize },
    /// Memory instance `inst`'s read-value stream column (log T_s vars,
    /// values bundle).
    RvCol { inst: usize },
    /// Memory instance `inst`'s write-value stream column.
    WvCol { inst: usize },
    /// Memory instance `inst`'s offset-encoded increment stream column.
    IncCol { inst: usize },
    /// Memory instance `inst`'s address stream column.
    AddrCol { inst: usize },
    /// Memory instance `inst`'s activity stream column; `id`'s low bit
    /// selects write (1) vs read (0) activity.
    ActiveCol { id: usize },
}

impl Factor {
    pub fn discriminant(self) -> u8 {
        match self {
            Factor::ValueBits { .. } => 0,
            Factor::InstrBits => 1,
            Factor::DigitBits { .. } => 2,
            Factor::BitCol { .. } => 3,
            Factor::ValCol { .. } => 4,
            Factor::RvCol { .. } => 5,
            Factor::WvCol { .. } => 6,
            Factor::IncCol { .. } => 7,
            Factor::AddrCol { .. } => 8,
            Factor::ActiveCol { .. } => 9,
        }
    }

    pub fn payload(self) -> usize {
        match self {
            Factor::ValueBits { slot } => slot,
            Factor::InstrBits => 0,
            Factor::DigitBits { inst }
            | Factor::RvCol { inst }
            | Factor::WvCol { inst }
            | Factor::IncCol { inst }
            | Factor::AddrCol { inst } => inst,
            Factor::BitCol { id } | Factor::ValCol { id } | Factor::ActiveCol { id } => id,
        }
    }

    pub fn from_parts(disc: u8, payload: usize) -> Option<Self> {
        match disc {
            0 if payload < VALUE_TENSORS => Some(Factor::ValueBits { slot: payload }),
            1 => Some(Factor::InstrBits),
            2 => Some(Factor::DigitBits { inst: payload }),
            3 => Some(Factor::BitCol { id: payload }),
            4 => Some(Factor::ValCol { id: payload }),
            5 => Some(Factor::RvCol { inst: payload }),
            6 => Some(Factor::WvCol { inst: payload }),
            7 => Some(Factor::IncCol { inst: payload }),
            8 => Some(Factor::AddrCol { inst: payload }),
            9 => Some(Factor::ActiveCol { id: payload }),
            _ => None,
        }
    }

    pub fn in_bits_bundle(self) -> bool {
        matches!(
            self,
            Factor::ValueBits { .. }
                | Factor::InstrBits
                | Factor::DigitBits { .. }
                | Factor::BitCol { .. }
                | Factor::ActiveCol { .. }
        )
    }
}

/// A base claim: (factor, full evaluation point, claimed value).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseClaim {
    pub factor: Factor,
    pub point: Vec<Goldilocks>,
    pub value: Goldilocks,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    QueueEmpty,
    KeyMismatch { expected: (u8, usize), got: (u8, usize) },
    PointArity { expected: usize, got: usize },
    InconsistentDuplicate,
    DerivedMismatch,
    Layout(String),
    Ajtai(lattice_commitment::ajtai::AjtaiError),
    Sumcheck(lattice_sumcheck::SumcheckError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Mle(lattice_core::mle::MleError),
    Transcript(lattice_core::transcript::TranscriptError),
}

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

/// Unit vector over `nbits` vars selecting index `idx` (MSB-first).
pub fn idx_point(nbits: usize, idx: usize) -> Vec<Goldilocks> {
    (0..nbits)
        .map(|i| fe(((idx >> (nbits - 1 - i)) & 1) as u64))
        .collect()
}

fn point_bytes(point: &[Goldilocks]) -> Vec<u8> {
    point.iter().flat_map(|f| f.to_bytes()).collect()
}

/// The ledger: prover mode derives from the witness, verifier mode pops
/// from the proof's claim queue. Both sides walk the identical protocol.
pub struct Ledger<'a> {
    /// Prover-side factor table (Factor -> its committed MLE).
    table: Vec<(Factor, &'a DenseMle)>,
    claims: Vec<BaseClaim>,
    seen: HashMap<(u8, usize, Vec<u8>), Goldilocks>,
    #[allow(dead_code)]
    queue: VecDeque<BaseClaim>,
    /// **`fix_last_variables` cache** (the prover-throughput design): the
    /// per-factor tensor bound at the most recent claim tail. The hot
    /// claim pattern — the digit-row claims `idx_point(log_rows, b) ∥
    /// terminal` for `b = 0..log_k` — shares the tail across `b`, so ONE
    /// binding pass (`O(N·|tail|)`, the cost of a single evaluate) serves
    /// the whole `b`-loop at `O(2^HEAD)` per claim instead of a fresh
    /// `O(N·vars)` evaluate each — the `(log_k + 1)×` resolution win the
    /// prover profile identified.
    tail_cache: HashMap<(u8, usize), (Vec<Goldilocks>, DenseMle)>,
    /// Verifier-mode key-indexed claim store: (disc, payload, point) ->
    /// FIFO of values. Consumption order ACROSS keys is free (the
    /// carrier's rho derivation uses the transmitted list order, which
    /// is unchanged); repeated keys (the values-only mode and the
    /// duplicated recordings) consume in list order.
    claim_map: HashMap<(u8, usize, Vec<u8>), VecDeque<Goldilocks>>,
    /// The unconsumed claim count (verifier mode).
    pending: usize,
}

/// The head size for the tail cache: claims whose points agree beyond the
/// first `HEAD_VARS` coordinates share one bound tensor.
const HEAD_VARS: usize = 6;

/// A values-only claim record: (factor, value) — the point is
/// verifier-derived from the leg replay (the compact mode's compressed
/// claim list; saves the ~104 B/claim point transmission).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueClaim {
    pub factor: Factor,
    pub value: Goldilocks,
}

impl<'a> Ledger<'a> {
    /// Prover-mode ledger over the full committed-factor table.
    pub fn prover(table: Vec<(Factor, &'a DenseMle)>) -> Self {
        Ledger {
            table,
            claims: Vec::new(),
            seen: HashMap::new(),
            queue: VecDeque::new(),
            tail_cache: HashMap::new(),
            claim_map: HashMap::new(),
            pending: 0,
        }
    }

    /// Verifier-mode ledger over the proof's claim list (key-indexed:
    /// the families consume the claims in their own order).
    pub fn verifier(claims: Vec<BaseClaim>) -> Self {
        let pending = claims.len();
        let mut claim_map: HashMap<(u8, usize, Vec<u8>), VecDeque<Goldilocks>> = HashMap::new();
        for c in claims.iter() {
            claim_map
                .entry((c.factor.discriminant(), c.factor.payload(), point_bytes(&c.point)))
                .or_default()
                .push_back(c.value);
        }
        Ledger {
            table: Vec::new(),
            claims: Vec::new(),
            seen: HashMap::new(),
            queue: VecDeque::new(),
            tail_cache: HashMap::new(),
            claim_map,
            pending,
        }
    }

    /// The recorded base claims.
    pub fn claims(&self) -> &[BaseClaim] {
        &self.claims
    }

    /// The number of unconsumed verifier claims (0 when the replay
    /// consumed exactly the transmitted list).
    pub fn queue_len(&self) -> usize {
        self.pending
    }

    fn record(&mut self, factor: Factor, point: &[Goldilocks], value: Goldilocks) {
        let key = (factor.discriminant(), factor.payload(), point_bytes(point));
        if self.seen.insert(key, value).is_some() {
            // Duplicate: keep the first recording (the verifier's
            // multi-valued key store consumes repeats in FIFO order).
        }
        self.claims.push(BaseClaim {
            factor,
            point: point.to_vec(),
            value,
        });
    }

    fn pop(&mut self, factor: Factor, point: &[Goldilocks]) -> Result<Goldilocks, LedgerError> {
        let key = (factor.discriminant(), factor.payload(), point_bytes(point));
        if let Some(prev) = self.seen.get(&key) {
            return Ok(*prev);
        }
        // The key-indexed store: consumption order across keys is free;
        // repeated keys consume in list order (FIFO per key).
        let value = if let Some(q) = self.claim_map.get_mut(&key) {
            match q.pop_front() {
                Some(v) => {
                    self.pending = self.pending.saturating_sub(1);
                    v
                }
                None => return Err(LedgerError::QueueEmpty),
            }
        } else {
            // Values-only fallback: the empty-point convention.
            let vkey = (factor.discriminant(), factor.payload(), Vec::new());
            match self.claim_map.get_mut(&vkey) {
                Some(q) => match q.pop_front() {
                    Some(v) => {
                        self.pending = self.pending.saturating_sub(1);
                        v
                    }
                    None => return Err(LedgerError::QueueEmpty),
                },
                None => return Err(LedgerError::QueueEmpty),
            }
        };
        self.seen.insert(key, value);
        self.claims.push(BaseClaim {
            factor,
            point: point.to_vec(),
            value,
        });
        Ok(value)
    }

    fn eval_tensor(&mut self, factor: Factor, point: &[Goldilocks]) -> Result<Goldilocks, LedgerError> {
        let tensor = self
            .table
            .iter()
            .find(|(f, _)| *f == factor)
            .map(|(_, m)| *m)
            .ok_or_else(|| LedgerError::Layout(format!("factor {factor:?} not in table")))?;
        // The tail cache: claims sharing the coordinates beyond the head
        // reuse one bound tensor.
        if point.len() > HEAD_VARS {
            let key = (factor.discriminant(), factor.payload());
            let head = &point[..HEAD_VARS];
            let tail = &point[HEAD_VARS..];
            let hit = match self.tail_cache.get(&key) {
                Some((cached_tail, bound)) if cached_tail.len() == tail.len() => {
                    cached_tail.iter().zip(tail.iter()).all(|(a, b)| a == b)
                        && bound.num_vars == HEAD_VARS
                }
                _ => false,
            };
            if !hit {
                let bound = tensor.fix_last_variables(tail).map_err(LedgerError::Mle)?;
                self.tail_cache.insert(key, (tail.to_vec(), bound));
            }
            if let Some((_, bound)) = self.tail_cache.get(&key) {
                return bound.evaluate(head).map_err(LedgerError::Mle);
            }
            // Unreachable: the entry was just inserted.
            return Err(LedgerError::Layout("tail cache".into()));
        }
        tensor.evaluate(point).map_err(LedgerError::Mle)
    }

    /// The values-only verifier ledger: pops match by factor only; the
    /// claim's point is the verifier's own derivation (keyed on the
    /// empty-point convention).
    pub fn verifier_values(pairs: Vec<ValueClaim>) -> Self {
        let claims: Vec<BaseClaim> = pairs
            .into_iter()
            .map(|vc| BaseClaim {
                factor: vc.factor,
                point: Vec::new(),
                value: vc.value,
            })
            .collect();
        let pending = claims.len();
        let mut claim_map: HashMap<(u8, usize, Vec<u8>), VecDeque<Goldilocks>> = HashMap::new();
        for c in claims.iter() {
            claim_map
                .entry((c.factor.discriminant(), c.factor.payload(), Vec::new()))
                .or_default()
                .push_back(c.value);
        }
        Ledger {
            table: Vec::new(),
            claims: Vec::new(),
            seen: HashMap::new(),
            queue: VecDeque::new(),
            tail_cache: HashMap::new(),
            claim_map,
            pending,
        }
    }

    /// A direct base claim on a tensor at an arbitrary point.
    pub fn tensor_claim(
        &mut self,
        factor: Factor,
        point: &[Goldilocks],
    ) -> Result<Goldilocks, LedgerError> {
        if self.table.is_empty() {
            self.pop(factor, point)
        } else {
            let v = self.eval_tensor(factor, point)?;
            self.record(factor, point, v);
            Ok(v)
        }
    }

    /// Claim = tensor(idx_point(log2(nbits), bit) ++ tail): bit-row
    /// `bit`. The tensor's bit block is `log2(nbits)` variables (the
    /// row-major grid's row axis), so the row point is the row index in
    /// binary, MSB first.
    fn bit_row_claim(
        &mut self,
        factor: Factor,
        nbits: usize,
        bit: usize,
        tail: &[Goldilocks],
    ) -> Result<Goldilocks, LedgerError> {
        let mut point = idx_point(nbits.trailing_zeros() as usize, bit);
        point.extend_from_slice(tail);
        self.tensor_claim(factor, &point)
    }

    /// The tensor row holding value-bit `bit` (LSB numbering) of an
    /// `nbits`-wide bit tensor: the tensors are packed MSB-first, so
    /// value-bit `b` lives at row `nbits - 1 - b`.
    fn value_bit_row(nbits: usize, bit: usize) -> usize {
        nbits - 1 - bit
    }

    /// The limb-`limb` (16-bit) MLE evaluation of value slot `slot` at a
    /// cycle point — 16 tensor row claims (MSB-first row packing).
    pub fn limb(
        &mut self,
        slot: usize,
        limb: usize,
        cycle_pt: &[Goldilocks],
    ) -> Result<Goldilocks, LedgerError> {
        let factor = Factor::ValueBits { slot };
        let mut acc = Goldilocks::ZERO;
        for i in 0..16usize {
            let bit = limb * 16 + i;
            let row = Self::value_bit_row(64, bit);
            let v = self.bit_row_claim(factor, 64, row, cycle_pt)?;
            acc = acc.add(&fe(1u64 << i).mul(&v));
        }
        Ok(acc)
    }

    /// The 64-bit combo of value slot `slot` at a cycle point.
    pub fn value_combo(
        &mut self,
        slot: usize,
        cycle_pt: &[Goldilocks],
    ) -> Result<Goldilocks, LedgerError> {
        let factor = Factor::ValueBits { slot };
        let mut acc = Goldilocks::ZERO;
        for bit in 0..64usize {
            let row = Self::value_bit_row(64, bit);
            let v = self.bit_row_claim(factor, 64, row, cycle_pt)?;
            acc = acc.add(&fe(1u64 << bit).mul(&v));
        }
        Ok(acc)
    }

    /// The instruction word MLE evaluation at a cycle point.
    pub fn instr_word(&mut self, cycle_pt: &[Goldilocks]) -> Result<Goldilocks, LedgerError> {
        let mut acc = Goldilocks::ZERO;
        for bit in 0..32usize {
            let row = Self::value_bit_row(32, bit);
            let v = self.bit_row_claim(Factor::InstrBits, 32, row, cycle_pt)?;
            acc = acc.add(&fe(1u64 << bit).mul(&v));
        }
        Ok(acc)
    }

    /// The `bit`-th digit-bit column of memory instance `inst` at a stream
    /// point.
    pub fn digit_bit(
        &mut self,
        inst: usize,
        log_k: usize,
        bit: usize,
        stream_pt: &[Goldilocks],
    ) -> Result<Goldilocks, LedgerError> {
        self.bit_row_claim(Factor::DigitBits { inst }, log_k, bit, stream_pt)
    }

    /// The increment column of instance `inst` at a stream point.
    pub fn inc_col(
        &mut self,
        inst: usize,
        stream_pt: &[Goldilocks],
    ) -> Result<Goldilocks, LedgerError> {
        self.tensor_claim(Factor::IncCol { inst }, stream_pt)
    }
}

/// Serialized proof of one bundle's grouped opening.
#[derive(Clone, Debug)]
pub struct BundleOpening {
    pub carrier: SumcheckProof,
    pub gadget_base: u64,
    pub gadget_digits: usize,
    /// Digits, element-major / coefficient-minor / digit-last.
    pub digits: Vec<i16>,
    /// The response length m (for shape checks).
    pub m: usize,
}

/// Layout entry: (factor, num_vars, flat offset).
#[derive(Clone, Debug)]
pub struct BundleLayoutEntry {
    pub factor: Factor,
    pub num_vars: usize,
    pub offset: usize,
}

/// Flatten a set of tensors into the big MLE + layout. The big MLE's
/// variable order is (offset bits, factor's own variables).
pub fn build_flat_mle(
    entries: &[(Factor, DenseMle)],
) -> Result<(DenseMle, Vec<BundleLayoutEntry>), LedgerError> {
    let mut flat: Vec<Goldilocks> = Vec::new();
    let mut layout = Vec::with_capacity(entries.len());
    for (factor, mle) in entries {
        let offset = flat.len();
        flat.extend_from_slice(&mle.evaluations);
        layout.push(BundleLayoutEntry {
            factor: *factor,
            num_vars: mle.num_vars,
            offset,
        });
    }
    let flat_len = flat.len();
    let padded = flat_len.next_power_of_two().max(1);
    flat.resize(padded, Goldilocks::ZERO);
    let log_flat = padded.trailing_zeros() as usize;
    Ok((DenseMle { num_vars: log_flat, evaluations: flat }, layout))
}

/// Map a claim to its flat point on the big MLE.
pub fn flat_point(entry: &BundleLayoutEntry, point: &[Goldilocks], log_flat: usize) -> Vec<Goldilocks> {
    let head_bits = log_flat - entry.num_vars;
    // The slice's head index: offset is always a multiple of 2^num_vars
    // (each tensor's length), so the head bits are offset >> num_vars.
    let slice = entry.offset >> entry.num_vars;
    let mut out = idx_point(head_bits, slice);
    out.extend_from_slice(point);
    out
}

/// The flat padded length implied by a layout.
pub fn flat_log_len(layout: &[BundleLayoutEntry]) -> usize {
    let mut end = 0usize;
    for e in layout {
        end = end.max(e.offset + (1usize << e.num_vars));
    }
    end.next_power_of_two().max(1).trailing_zeros() as usize
}

/// The ring used for zkvm bundles (n = 64 coefficients).
pub fn bundle_ring() -> Result<RingConfig, LedgerError> {
    RingConfig::new(Modulus32::Q_32, 6)
        .map_err(|e| LedgerError::Layout(format!("ring: {e:?}")))
}

/// Norm bound for bit-packed coefficients (< 2^31).
pub const BITS_NORM_BOUND: u32 = (1 << 31) - 1;
/// Norm bound for the inc columns' packing limbs (< 2^22).
pub const VALUES_NORM_BOUND: u32 = (1 << 22) - 1;

/// One committed bundle (prover side).
pub struct BundleProver {
    pub ring: RingConfig,
    pub pk: AjtaiPublicKey,
    pub flat: DenseMle,
    pub layout: Vec<BundleLayoutEntry>,
    pub s: Vec<RingElement>,
    pub commitment: AjtaiCommitment,
    pub is_bits: bool,
}

/// Commit the bits bundle (bit-packing at 31 bits/coefficient).
pub fn bits_bundle_commit(
    entries: &[(Factor, DenseMle)],
    seed: [u8; 32],
) -> Result<BundleProver, LedgerError> {
    let (flat, layout) = build_flat_mle(entries)?;
    let bits: Vec<u8> = flat
        .evaluations
        .iter()
        .map(|v| (v.to_canonical_u64() & 1) as u8)
        .collect();
    let ring = bundle_ring()?;
    let packed = pack_bits(&ring, &bits);
    let m = packed.len().max(1);
    let params = AjtaiParams {
        ring: ring.clone(),
        k: 2,
        m,
        norm_bound: BITS_NORM_BOUND,
    };
    let pk = AjtaiPublicKey::from_seed(params, seed).map_err(LedgerError::Ajtai)?;
    let s = pk.pad_to_m(&packed).map_err(LedgerError::Ajtai)?;
    let commitment = pk.commit(&s).map_err(LedgerError::Ajtai)?;
    Ok(BundleProver {
        ring,
        pk,
        flat,
        layout,
        s,
        commitment,
        is_bits: true,
    })
}

/// Commit the values bundle (3×22-bit limb packing).
pub fn values_bundle_commit(
    entries: &[(Factor, DenseMle)],
    seed: [u8; 32],
) -> Result<BundleProver, LedgerError> {
    let (flat, layout) = build_flat_mle(entries)?;
    let ring = bundle_ring()?;
    let packed = pack_field_elements(&ring, &flat.evaluations);
    let m = packed.len().max(1);
    let params = AjtaiParams {
        ring: ring.clone(),
        k: 2,
        m,
        norm_bound: VALUES_NORM_BOUND,
    };
    let pk = AjtaiPublicKey::from_seed(params, seed).map_err(LedgerError::Ajtai)?;
    let s = pk.pad_to_m(&packed).map_err(LedgerError::Ajtai)?;
    let commitment = pk.commit(&s).map_err(LedgerError::Ajtai)?;
    Ok(BundleProver {
        ring,
        pk,
        flat,
        layout,
        s,
        commitment,
        is_bits: false,
    })
}

impl BundleProver {
    /// Grouped opening over claims on this bundle's factors.
    pub fn prove_opening(
        &self,
        claims: &[BaseClaim],
        transcript: &mut Transcript,
    ) -> Result<BundleOpening, LedgerError> {
        let log_flat = self.flat.num_vars;
        let points: Vec<Vec<Goldilocks>> = claims
            .iter()
            .map(|c| self.claim_point(c, log_flat))
            .collect::<Result<_, _>>()?;
        let digits = if self.is_bits { 4 } else { 3 };
        let bound = if self.is_bits { BITS_NORM_BOUND } else { VALUES_NORM_BOUND };
        prove_grouped_carrier(
            &self.flat,
            &self.s,
            &points,
            &claims.iter().map(|c| c.value).collect::<Vec<_>>(),
            bound,
            digits,
            self.s.len(),
            transcript,
        )
    }

    fn claim_point(
        &self,
        claim: &BaseClaim,
        log_flat: usize,
    ) -> Result<Vec<Goldilocks>, LedgerError> {
        let entry = self
            .layout
            .iter()
            .find(|e| e.factor == claim.factor)
            .ok_or_else(|| LedgerError::Layout(format!("factor {:?} not in bundle", claim.factor)))?;
        if claim.point.len() != entry.num_vars {
            return Err(LedgerError::PointArity {
                expected: entry.num_vars,
                got: claim.point.len(),
            });
        }
        Ok(flat_point(entry, &claim.point, log_flat))
    }
}

/// The shared grouped-carrier prover (both bundles).
#[allow(clippy::too_many_arguments)]
fn prove_grouped_carrier(
    flat: &DenseMle,
    s: &[RingElement],
    points: &[Vec<Goldilocks>],
    values: &[Goldilocks],
    norm_bound: u32,
    norm_digits: usize,
    m: usize,
    transcript: &mut Transcript,
) -> Result<BundleOpening, LedgerError> {
    if points.is_empty() {
        return Err(LedgerError::Layout("no claims".into()));
    }
    let rhos = transcript
        .challenge_fields(b"bundle-rho", points.len())
        .map_err(LedgerError::Transcript)?;
    let mut combined = Goldilocks::ZERO;
    for (i, v) in values.iter().enumerate() {
        combined = combined.add(&rhos[i].mul(v));
    }
    // The factored carrier: sum_i rho_i·flat·eq_i = flat·(sum_i rho_i·eq_i)
    // — ONE product term whose round values are IDENTICAL to the
    // per-claim-term form (the products distribute), so the transcript
    // and the round messages are byte-identical while the sumcheck work
    // drops from (claims x flat x rounds) to (claims x flat + rounds x
    // flat).
    //
    // The eq-table BUILD is itself prefix-factored (the second-level
    // factoring): the flat points' leading coordinates are BOOLEAN —
    // the layout's head/slice bits, and the bit-row heads of tensor
    // claims — so claims ROUTE exactly through those coordinates
    // (eq(b, x) for boolean b vanishes off the routed half), and only
    // the remaining FIELD coordinates need a dense per-group eq table.
    // Work drops from claims x 2^log_flat to
    // sum_groups claims_g x 2^{field_vars_g} + 2^log_flat — at the
    // semantics scale (30.8k claims, log_flat 17..21) a ~10^2-10^3x
    // win — with the produced array bit-identical (field addition is
    // commutative), so transcripts stay byte-identical.
    let flat_len = 1usize << flat.num_vars;
    let mut combined_eq = vec![Goldilocks::ZERO; flat_len];
    let order: Vec<usize> = (0..points.len()).collect();
    rec_eq_acc(
        &mut combined_eq,
        &order,
        points,
        &rhos,
        0,
        0,
        flat.num_vars,
    );
    let mut vp = VirtualPolynomial::new(flat.num_vars);
    let fi = vp.add_factor(flat.clone()).map_err(LedgerError::Virtual)?;
    let ei = vp
        .add_factor(DenseMle {
            num_vars: flat.num_vars,
            evaluations: combined_eq,
        })
        .map_err(LedgerError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![fi, ei])
        .map_err(LedgerError::Virtual)?;
    let out = sumcheck::prove(&vp, combined, transcript).map_err(LedgerError::Sumcheck)?;
    let norm = compact_norm_proof(s, norm_bound, norm_digits)
        .map_err(|e| LedgerError::Layout(format!("norm: {e:?}")))?;
    let digits: Vec<i16> = norm
        .digits
        .iter()
        .flat_map(|elem| elem.iter().copied())
        .map(|d| d as i16)
        .collect();
    Ok(BundleOpening {
        carrier: out.proof,
        gadget_base: norm.gadget.base,
        gadget_digits: norm.gadget.num_digits,
        digits,
        m,
    })
}

/// Prefix-routed eq accumulation (the prover-side carrier factoring):
/// accumulate `sum_i rho_i · eq(q_i, x)` into `out` over the subcube
/// rooted at `offset` with `log_flat - level` variables remaining.
///
/// At each coordinate, if EVERY live claim's coordinate is boolean, the
/// claims split exactly (eq(b, ·) vanishes off the b-half for boolean b)
/// and the recursion routes without any field work; otherwise the group
/// accumulates densely over its remaining variables. The result is
/// bit-identical to the naive full-cube per-claim accumulation (field
/// addition is commutative and associative), so the transcript is
/// byte-identical — only the work changes.
fn rec_eq_acc(
    out: &mut [Goldilocks],
    live: &[usize],
    points: &[Vec<Goldilocks>],
    rhos: &[Goldilocks],
    level: usize,
    offset: usize,
    log_flat: usize,
) {
    if live.is_empty() {
        return;
    }
    if level == log_flat {
        // All live claims share this exact flat point (they matched
        // every routed coordinate); duplicates sum their weights.
        let mut acc = Goldilocks::ZERO;
        for &i in live {
            acc = acc.add(&rhos[i]);
        }
        out[offset] = out[offset].add(&acc);
        return;
    }
    let all_bool = live.iter().all(|&i| {
        let c = points[i][level];
        c == Goldilocks::ZERO || c == Goldilocks::ONE
    });
    let rem = log_flat - level;
    if all_bool {
        let mut zeros: Vec<usize> = Vec::with_capacity(live.len());
        let mut ones: Vec<usize> = Vec::with_capacity(live.len());
        for &i in live {
            if points[i][level] == Goldilocks::ONE {
                ones.push(i);
            } else {
                zeros.push(i);
            }
        }
        rec_eq_acc(out, &zeros, points, rhos, level + 1, offset, log_flat);
        rec_eq_acc(
            out,
            &ones,
            points,
            rhos,
            level + 1,
            offset + (1usize << (rem - 1)),
            log_flat,
        );
    } else {
        // Dense accumulation over this group's subcube only.
        for &i in live {
            let suffix = &points[i][level..];
            let tab = DenseMle::eq_extension(suffix);
            let rho = rhos[i];
            for (t, v) in tab.evaluations.iter().enumerate() {
                out[offset + t] = out[offset + t].add(&rho.mul(v));
            }
        }
    }
}

/// Verify a bundle opening. `pk` must be the same seeded key the prover
/// used; the layout describes the flat MLE.
#[allow(clippy::too_many_arguments)]
pub fn verify_bundle_opening(
    pk: &AjtaiPublicKey,
    commitment: &AjtaiCommitment,
    layout: &[BundleLayoutEntry],
    claims: &[BaseClaim],
    opening: &BundleOpening,
    is_bits: bool,
    transcript: &mut Transcript,
) -> Result<(), LedgerError> {
    let ring = pk.params.ring.clone();
    let log_flat = flat_log_len(layout);
    let m = pk.params.m;
    if opening.m != m {
        return Err(LedgerError::Layout("m mismatch".into()));
    }
    // Map claims to flat points.
    let points: Vec<Vec<Goldilocks>> = claims
        .iter()
        .map(|c| {
            let entry = layout
                .iter()
                .find(|e| e.factor == c.factor)
                .ok_or_else(|| {
                    LedgerError::Layout(format!("factor {:?} not in bundle", c.factor))
                })?;
            if c.point.len() != entry.num_vars {
                return Err(LedgerError::PointArity {
                    expected: entry.num_vars,
                    got: c.point.len(),
                });
            }
            Ok(flat_point(entry, &c.point, log_flat))
        })
        .collect::<Result<_, _>>()?;

    // Carrier replay.
    let rhos = transcript
        .challenge_fields(b"bundle-rho", points.len())
        .map_err(LedgerError::Transcript)?;
    let mut combined = Goldilocks::ZERO;
    for (rho, c) in rhos.iter().zip(claims.iter()) {
        combined = combined.add(&rho.mul(&c.value));
    }
    let verdict = opening
        .carrier
        .verify(log_flat, 2, combined, transcript, None)
        .map_err(LedgerError::Sumcheck)?;
    let r_sc = verdict.point;

    // Reconstruct the response s from the digits.
    let gadget = GadgetDecomposition {
        base: opening.gadget_base,
        num_digits: opening.gadget_digits,
    };
    let digit_bound = (gadget.base / 2) as i64;
    let per_elem = ring.n() * gadget.num_digits;
    if opening.digits.len() != m * per_elem {
        return Err(LedgerError::Layout("digit count mismatch".into()));
    }
    let mut s: Vec<RingElement> = Vec::with_capacity(m);
    for e in 0..m {
        let mut coeffs = vec![0u32; ring.n()];
        for (c, coeff) in coeffs.iter_mut().enumerate() {
            let mut acc: i128 = 0;
            let mut power: i128 = 1;
            for d in 0..gadget.num_digits {
                let digit = opening.digits[e * per_elem + c * gadget.num_digits + d] as i64;
                if digit.abs() > digit_bound {
                    return Err(LedgerError::Layout("digit out of range".into()));
                }
                acc += (digit as i128) * power;
                power *= gadget.base as i128;
            }
            *coeff = acc.rem_euclid(ring.modulus.q as i128) as u32;
        }
        let elem = RingElement::from_coeffs(&ring, coeffs);
        s.push(elem);
    }

    // 1. A·s = t exactly (binding against the committed t).
    pk.verify_opening(commitment, &s)
        .map_err(LedgerError::Ajtai)?;

    // 2. The final carrier identity with f(r_sc) recomputed from s.
    let flat_len = 1usize << log_flat;
    let flat = if is_bits {
        let bits = unpack_bits(&ring, &s, flat_len);
        DenseMle {
            num_vars: log_flat,
            evaluations: bits.into_iter().map(|b| fe(b as u64)).collect(),
        }
    } else {
        let mut limb_iter: Vec<u64> = Vec::with_capacity(m * ring.n());
        for e in &s {
            for c in e.coeffs() {
                limb_iter.push(*c as u64);
            }
        }
        let mut evals: Vec<Goldilocks> = Vec::with_capacity(flat_len);
        let mut li = 0usize;
        while evals.len() < flat_len {
            let l0 = limb_iter.get(li).copied().unwrap_or(0) & ((1 << 22) - 1);
            let l1 = limb_iter.get(li + 1).copied().unwrap_or(0) & ((1 << 22) - 1);
            let l2 = limb_iter.get(li + 2).copied().unwrap_or(0);
            li += 3;
            evals.push(Goldilocks::from_u64(l0 | (l1 << 22) | (l2 << 44)));
        }
        DenseMle {
            num_vars: log_flat,
            evaluations: evals,
        }
    };
    let f_sc = flat.evaluate(&r_sc).map_err(LedgerError::Mle)?;
    let mut expect = Goldilocks::ZERO;
    for (i, pt) in points.iter().enumerate() {
        let eq_v = DenseMle::eq_eval(pt, &r_sc).map_err(LedgerError::Mle)?;
        expect = expect.add(&rhos[i].mul(&eq_v).mul(&f_sc));
    }
    if expect != verdict.final_claim {
        return Err(LedgerError::DerivedMismatch);
    }
    Ok(())
}

/// Compact norm proof: base-256, `num_digits` digits per coefficient.
fn compact_norm_proof(
    s: &[RingElement],
    bound: u32,
    num_digits: usize,
) -> Result<NormProof, String> {
    for e in s {
        if e.infinity_norm() > bound {
            return Err("norm exceeded".into());
        }
    }
    let gadget = GadgetDecomposition::power_of_two(8 * num_digits as u32, 3);
    let mut digits = Vec::with_capacity(s.len());
    for e in s {
        let q = e.config().modulus.q;
        let half = q / 2;
        let mut elem_digits = Vec::with_capacity(e.coeffs().len() * gadget.num_digits);
        for &c in e.coeffs() {
            let balanced = if c <= half {
                c as i64
            } else {
                c as i64 - q as i64
            };
            let (magnitude, sign) = if balanced >= 0 {
                (balanced as u64, 1i64)
            } else {
                ((-balanced) as u64, -1i64)
            };
            let mut d = gadget.decompose(magnitude).map_err(|e| format!("{e:?}"))?;
            if sign < 0 {
                for digit in d.iter_mut() {
                    *digit = -*digit;
                }
            }
            elem_digits.extend(d);
        }
        digits.push(elem_digits);
    }
    Ok(NormProof { digits, gadget })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(log_vars: usize, seed: u64) -> DenseMle {
        DenseMle::random(log_vars, &seed.to_le_bytes())
    }

    #[test]
    fn prefix_routed_eq_matches_naive_build() {
        // The factored carrier's eq build must be bit-identical to the
        // naive full-cube per-claim accumulation, across mixed
        // boolean-head / field-tail claim structures (the semantics
        // layer's shape) and fully field-valued points (the adversarial
        // shape where the factoring degenerates to the dense build).
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng
        };
        for log_flat in [5usize, 7, 9] {
            for n_claims in [1usize, 5, 40, 200] {
                for shape in ["rowhead", "field"] {
                    let points: Vec<Vec<Goldilocks>> = (0..n_claims)
                        .map(|i| {
                            (0..log_flat)
                                .map(|j| {
                                    if shape == "rowhead" && j < 3 {
                                        // boolean head (row/slice bits)
                                        fe(((i >> j) & 1) as u64)
                                    } else if shape == "rowhead" && j == 3 && i % 3 == 0 {
                                        fe(0) // some constant-boolean coords too
                                    } else {
                                        Goldilocks::from_u64(next() % 0xffff)
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    let rhos: Vec<Goldilocks> = (0..n_claims)
                        .map(|_| Goldilocks::from_u64(next() % 0xffff))
                        .collect();
                    // Naive reference.
                    let mut naive = vec![Goldilocks::ZERO; 1usize << log_flat];
                    for (i, pt) in points.iter().enumerate() {
                        let eq = DenseMle::eq_extension(pt);
                        for (e, v) in naive.iter_mut().zip(eq.evaluations.iter()) {
                            *e = e.add(&rhos[i].mul(v));
                        }
                    }
                    // Factored build.
                    let mut factored = vec![Goldilocks::ZERO; 1usize << log_flat];
                    let order: Vec<usize> = (0..points.len()).collect();
                    rec_eq_acc(&mut factored, &order, &points, &rhos, 0, 0, log_flat);
                    assert_eq!(
                        naive, factored,
                        "log_flat={log_flat} n={n_claims} shape={shape}"
                    );
                }
            }
        }
    }

    #[test]
    fn flat_layout_maps_points() {
        let t0 = tensor(4, 1);
        let t1 = tensor(3, 2);
        let (flat, layout) = build_flat_mle(&[
            (Factor::InstrBits, t0.clone()),
            (Factor::DigitBits { inst: 0 }, t1.clone()),
        ])
        .ok()
        .unwrap();
        assert_eq!(flat.num_vars, 5); // 24 -> 32
        let entry = layout
            .iter()
            .find(|e| e.factor == Factor::DigitBits { inst: 0 })
            .unwrap();
        let pt = flat_point(entry, &[fe(1), fe(0), fe(1)], 5);
        let direct = t1.evaluate(&[fe(1), fe(0), fe(1)]).ok().unwrap();
        let via_flat = flat.evaluate(&pt).ok().unwrap();
        assert_eq!(direct, via_flat);
    }

    fn bit_tensor(log_vars: usize, seed: u64) -> DenseMle {
        let n = 1usize << log_vars;
        let evals: Vec<Goldilocks> = (0..n)
            .map(|i| fe(((i as u64).wrapping_mul(2654435761) ^ seed.wrapping_mul(i as u64)) & 1))
            .collect();
        DenseMle { num_vars: log_vars, evaluations: evals }
    }

    #[test]
    fn bits_bundle_roundtrip_binds_claims() {
        let t0 = bit_tensor(5, 7); // 32 entries
        let t1 = bit_tensor(4, 11); // 16 entries
        let prover =
            bits_bundle_commit(&[(Factor::InstrBits, t0.clone()), (Factor::DigitBits { inst: 0 }, t1)], [9u8; 32])
                .ok()
                .unwrap();
        // Claims at random points on both tensors.
        let c0 = BaseClaim {
            factor: Factor::InstrBits,
            point: tensor_point(5, 3),
            value: t0.evaluate(&tensor_point(5, 3)).ok().unwrap(),
        };
        let c1 = BaseClaim {
            factor: Factor::DigitBits { inst: 0 },
            point: tensor_point(4, 8),
            value: prover
                .flat
                .evaluate(&{
                    let entry = prover.layout.iter().find(|e| e.factor == Factor::DigitBits { inst: 0 }).unwrap();
                    flat_point(entry, &tensor_point(4, 8), prover.flat.num_vars)
                })
                .ok()
                .unwrap(),
        };
        let mut t = Transcript::new_default(b"bundle-test");
        let opening = prover
            .prove_opening(&[c0.clone(), c1.clone()], &mut t)
            .ok()
            .unwrap();
        let mut t2 = Transcript::new_default(b"bundle-test");
        assert!(verify_bundle_opening(
            &prover.pk,
            &prover.commitment,
            &prover.layout,
            &[c0.clone(), c1.clone()],
            &opening,
            true,
            &mut t2,
        )
        .is_ok());
        // Tampered claim value rejected.
        let mut bad = c0.clone();
        bad.value = bad.value.add(&fe(1));
        let mut t3 = Transcript::new_default(b"bundle-test");
        assert!(verify_bundle_opening(
            &prover.pk,
            &prover.commitment,
            &prover.layout,
            &[bad, c1.clone()],
            &opening,
            true,
            &mut t3,
        )
        .is_err());
        // Tampered digits rejected (A·s or the final identity breaks).
        let mut o2 = opening.clone();
        if let Some(d) = o2.digits.first_mut() {
            *d = d.wrapping_add(1);
        }
        let mut t4 = Transcript::new_default(b"bundle-test");
        assert!(verify_bundle_opening(
            &prover.pk,
            &prover.commitment,
            &prover.layout,
            &[c0, c1],
            &o2,
            true,
            &mut t4,
        )
        .is_err());
    }

    #[test]
    fn values_bundle_roundtrip_binds_claims() {
        // Inc-column-like values: offset-encoded, < 2^18.
        let vals: Vec<Goldilocks> = (0..16u64).map(|i| fe((i * 7919) % (1 << 18))).collect();
        let col = DenseMle { num_vars: 4, evaluations: vals };
        let prover = values_bundle_commit(&[(Factor::IncCol { inst: 0 }, col.clone())], [5u8; 32])
            .ok()
            .unwrap();
        let pt = tensor_point(4, 13);
        let claim = BaseClaim {
            factor: Factor::IncCol { inst: 0 },
            point: pt.clone(),
            value: col.evaluate(&pt).ok().unwrap(),
        };
        let mut t = Transcript::new_default(b"vals-test");
        let opening = prover.prove_opening(std::slice::from_ref(&claim), &mut t).ok().unwrap();
        let mut t2 = Transcript::new_default(b"vals-test");
        assert!(verify_bundle_opening(
            &prover.pk,
            &prover.commitment,
            &prover.layout,
            &[claim],
            &opening,
            false,
            &mut t2,
        )
        .is_ok());
    }

    fn tensor_point(nvars: usize, seed: u64) -> Vec<Goldilocks> {
        (0..nvars)
            .map(|i| fe(((seed >> (i % 8)) ^ ((i * 37) as u64)) % 97))
            .collect()
    }
}
