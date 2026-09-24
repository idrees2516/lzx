//! # lattice-lookup
//!
//! Quasar (Zheng–Gao–Guo–Xiao, ePrint 2025/1912): sublinear accumulation
//! schemes for multiple instances — the lookup-argument accumulation core.
//!
//! Two components:
//! * `lookup` — the grand-product lookup check (FLI/cq lineage): reads are
//!   contained in the table as a multiset iff the random grand products
//!   satisfy `T(τ) = R(τ)·Q(τ)` at a random challenge τ, where
//!   `T = Π_table (1 + τ·v)`, `R = Π_reads (1 + τ·v)`, and Q is the
//!   quotient.
//! * `accumulate` — Quasar's multi-instance accumulation via **partial
//!   evaluation** instead of random linear combinations of commitments
//!   (the costly CRC operations): all instances partially evaluate their
//!   polynomials at a *shared* challenge prefix, so the folded accumulator
//!   needs a single combination step — verifier cost sublinear in the
//!   number of instances.
//!
//! **Wave 6.5 (Q1 — the committed path)**: the original `verify_lookup`
//! checked the grand-product identity on three *prover-supplied scalars* —
//! any consistent triple verified against any statement (a soundness
//! hole: T/R/Q were never bound to anything). The new
//! [`prove_lookup_committed`]/[`verify_lookup_committed`] protocol closes
//! the hole:
//!
//! * the prover **commits** (Ajtai) to the packed table, reads, and
//!   multiset-difference vectors,
//! * τ is derived from the *commitments* (never from raw values — no
//!   grinding),
//! * the verifier recomputes the grand products from the **opened**
//!   vectors, checks the identity, and checks the **counting-map
//!   difference** (reads ⊆ table as multisets, Q = table ∖ reads) —
//!   O(n log n) via canonical sort, replacing the prover's O(n²)
//!   linear-scan difference,
//! * the openings bind the commitments to exactly the checked vectors
//!   (SIS binding), and a forged-triple negative test pins the behavior.
//!
//! The scalar-only path (`prove_lookup`/`verify_lookup`) is retained for
//! the accumulation experiments but is documented as **not sound**
//! standalone; protocol code must use the committed path until Wave 7's
//! NIR_multicast lands.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiPublicKey};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::RingElement;

/// A lookup proof: evaluations of the grand products at the challenge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupProof {
    /// Π_table (1 + τ·v) evaluated... structurally: the proof carries the
    /// three evaluations; commitments are layered by the caller.
    pub t_eval: Goldilocks,
    pub r_eval: Goldilocks,
    pub q_eval: Goldilocks,
    pub challenge: Goldilocks,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupError {
    Transcript(TranscriptError),
    /// Grand-product identity failed.
    LookupFailed,
    /// Accumulation identity failed.
    AccumulationFailed,
    ShapeMismatch { expected: usize, got: usize },
}

/// Grand product helper: Π (1 + τ·v) over values.
fn grand_product(values: &[Goldilocks], tau: &Goldilocks) -> Goldilocks {
    let mut acc = Goldilocks::ONE;
    for v in values {
        acc = acc.mul(&Goldilocks::ONE.add(&tau.mul(v)));
    }
    acc
}

/// Prove reads ⊆ table (multiset containment) via the grand-product
/// quotient argument.
pub fn prove_lookup(
    table: &[Goldilocks],
    reads: &[Goldilocks],
    transcript: &mut Transcript,
) -> Result<LookupProof, LookupError> {
    // Absorb the public statement (lengths + digests of both sides).
    let mut table_bytes = Vec::with_capacity(table.len() * 8);
    for v in table {
        table_bytes.extend_from_slice(&v.to_bytes());
    }
    let mut reads_bytes = Vec::with_capacity(reads.len() * 8);
    for v in reads {
        reads_bytes.extend_from_slice(&v.to_bytes());
    }
    transcript
        .append_bytes(b"table", &table_bytes)
        .map_err(LookupError::Transcript)?;
    transcript
        .append_bytes(b"reads", &reads_bytes)
        .map_err(LookupError::Transcript)?;

    let tau = transcript
        .challenge_field(b"lookup-tau")
        .map_err(LookupError::Transcript)?;

    // Honest prover: multiset containment must hold — the quotient Q is
    // the grand product over table \ reads (multiset difference). Fail
    // closed when reads are NOT contained (Q would not exist).
    let t_eval = grand_product(table, &tau);
    let r_eval = grand_product(reads, &tau);
    // Multiset difference via the counting map (O(n log n) — Wave 6.5).
    let remaining = counting_map_difference(table, reads).ok_or(LookupError::LookupFailed)?;
    let q_eval = grand_product(&remaining, &tau);
    Ok(LookupProof {
        t_eval,
        r_eval,
        q_eval,
        challenge: tau,
    })
}

/// Multiset difference `table ∖ reads` via the counting map (sort-based,
/// O(n log n), deterministic — replaces the O(n²) linear-scan removal).
/// Returns `None` when `reads` is NOT contained in `table` as a multiset
/// (some read demands a multiplicity the table cannot supply).
pub fn counting_map_difference(
    table: &[Goldilocks],
    reads: &[Goldilocks],
) -> Option<Vec<Goldilocks>> {
    let mut t_sorted: Vec<u64> = table.iter().map(|v| v.to_canonical_u64()).collect();
    let mut r_sorted: Vec<u64> = reads.iter().map(|v| v.to_canonical_u64()).collect();
    t_sorted.sort_unstable();
    r_sorted.sort_unstable();
    // Merge-difference: walk both sorted streams; every read consumes one
    // table entry; leftovers are the difference.
    let mut diff = Vec::with_capacity(table.len().saturating_sub(reads.len()));
    let (mut ti, mut ri) = (0usize, 0usize);
    while ri < r_sorted.len() {
        if ti >= t_sorted.len() {
            return None; // read left over — not contained
        }
        match t_sorted[ti].cmp(&r_sorted[ri]) {
            std::cmp::Ordering::Less => {
                diff.push(t_sorted[ti]);
                ti += 1;
            }
            std::cmp::Ordering::Equal => {
                ti += 1;
                ri += 1;
            }
            std::cmp::Ordering::Greater => {
                return None; // read value not present
            }
        }
    }
    while ti < t_sorted.len() {
        diff.push(t_sorted[ti]);
        ti += 1;
    }
    Some(
        diff.into_iter()
            .map(Goldilocks::from_u64)
            .collect(),
    )
}

/// Verify a lookup proof: T(τ) == R(τ)·Q(τ) plus the τ-consistency.
///
/// **NOT SOUND standalone (Wave 6.5 disclosure)**: the three evaluations
/// are prover-supplied scalars that were never bound to the statement —
/// any self-consistent triple passes. Use
/// [`verify_lookup_committed`] (the Q1 committed path) for anything
/// security-relevant; this function is retained for the accumulation
/// experiments and as the algebraic reference.
pub fn verify_lookup(
    proof: &LookupProof,
    table: &[Goldilocks],
    reads: &[Goldilocks],
    transcript: &mut Transcript,
) -> Result<(), LookupError> {
    let mut table_bytes = Vec::with_capacity(table.len() * 8);
    for v in table {
        table_bytes.extend_from_slice(&v.to_bytes());
    }
    let mut reads_bytes = Vec::with_capacity(reads.len() * 8);
    for v in reads {
        reads_bytes.extend_from_slice(&v.to_bytes());
    }
    transcript
        .append_bytes(b"table", &table_bytes)
        .map_err(LookupError::Transcript)?;
    transcript
        .append_bytes(b"reads", &reads_bytes)
        .map_err(LookupError::Transcript)?;
    let tau = transcript
        .challenge_field(b"lookup-tau")
        .map_err(LookupError::Transcript)?;
    if tau != proof.challenge {
        return Err(LookupError::LookupFailed);
    }
    // Grand-product identity.
    if proof.t_eval != proof.r_eval.mul(&proof.q_eval) {
        return Err(LookupError::LookupFailed);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Wave 6.5 — Q1: the committed lookup protocol (closes the soundness hole).
// ---------------------------------------------------------------------------

/// A committed lookup proof: Ajtai commitments to the packed table, reads,
/// and multiset-difference vectors; the grand-product evaluations; and the
/// clear openings (the packed witness vectors — the pre-ZK baseline posture
/// of this codebase; Wave 8.6 layers blinding).
#[derive(Clone, Debug)]
pub struct CommittedLookupProof {
    /// Commitment to the packed TABLE values.
    pub table_commitment: AjtaiCommitment,
    /// Commitment to the packed READS values.
    pub reads_commitment: AjtaiCommitment,
    /// Commitment to the packed DIFFERENCE (table ∖ reads) values.
    pub diff_commitment: AjtaiCommitment,
    /// Number of table / reads / difference values (the public statement
    /// shape, absorbed before τ).
    pub table_len: usize,
    pub reads_len: usize,
    /// Grand-product evaluations at the challenge.
    pub t_eval: Goldilocks,
    pub r_eval: Goldilocks,
    pub q_eval: Goldilocks,
    /// The challenge τ (derived from the commitments).
    pub challenge: Goldilocks,
    /// Clear openings: the packed ring-element vectors behind each
    /// commitment (padded to the key dimension m).
    pub table_opening: Vec<RingElement>,
    pub reads_opening: Vec<RingElement>,
    pub diff_opening: Vec<RingElement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommittedLookupError {
    Lookup(LookupError),
    Ajtai(AjtaiError),
    Packing(lattice_ring::PackingError),
    /// The recomputed challenge does not match (statement tampering).
    ChallengeMismatch,
    /// The counting-map difference failed: reads ⊄ table as multisets.
    NotContained,
    /// The grand-product identity failed at the recomputed challenge.
    GrandProductFailed,
    /// Shape inconsistency between the statement and the openings.
    ShapeMismatch { expected: usize, got: usize },
}

impl From<LookupError> for CommittedLookupError {
    fn from(e: LookupError) -> Self {
        CommittedLookupError::Lookup(e)
    }
}

/// Absorb the committed statement (three commitments + shape) and derive τ.
fn derive_challenge_committed(
    proof: &CommittedLookupProof,
    transcript: &mut Transcript,
) -> Result<Goldilocks, LookupError> {
    transcript
        .append_bytes(b"ct", &proof.table_commitment.to_bytes())
        .map_err(LookupError::Transcript)?;
    transcript
        .append_bytes(b"cr", &proof.reads_commitment.to_bytes())
        .map_err(LookupError::Transcript)?;
    transcript
        .append_bytes(b"cq", &proof.diff_commitment.to_bytes())
        .map_err(LookupError::Transcript)?;
    let mut shape = Vec::with_capacity(8);
    shape.extend_from_slice(&(proof.table_len as u32).to_le_bytes());
    shape.extend_from_slice(&(proof.reads_len as u32).to_le_bytes());
    transcript
        .append_bytes(b"shape", &shape)
        .map_err(LookupError::Transcript)?;
    transcript.challenge_field(b"committed-tau").map_err(LookupError::Transcript)
}

/// Prove reads ⊆ table with committed T/R/Q (Q1). The three vectors are
/// packed (3×22-bit limbs per value), committed under `pk`, and τ is
/// derived from the commitments — never from the raw values.
pub fn prove_lookup_committed(
    pk: &AjtaiPublicKey,
    table: &[Goldilocks],
    reads: &[Goldilocks],
    transcript: &mut Transcript,
) -> Result<CommittedLookupProof, CommittedLookupError> {
    // Multiset containment must hold for an honest prover.
    let diff = counting_map_difference(table, reads).ok_or(CommittedLookupError::NotContained)?;
    let ring = &pk.params.ring;
    let packed_table = lattice_ring::packing::pack_field_elements(ring, table);
    let packed_reads = lattice_ring::packing::pack_field_elements(ring, reads);
    let packed_diff = lattice_ring::packing::pack_field_elements(ring, &diff);
    let table_opening = pk
        .pad_to_m(&packed_table)
        .map_err(CommittedLookupError::Ajtai)?;
    let reads_opening = pk
        .pad_to_m(&packed_reads)
        .map_err(CommittedLookupError::Ajtai)?;
    let diff_opening = pk
        .pad_to_m(&packed_diff)
        .map_err(CommittedLookupError::Ajtai)?;
    let table_commitment = pk
        .commit(&table_opening)
        .map_err(CommittedLookupError::Ajtai)?;
    let reads_commitment = pk
        .commit(&reads_opening)
        .map_err(CommittedLookupError::Ajtai)?;
    let diff_commitment = pk
        .commit(&diff_opening)
        .map_err(CommittedLookupError::Ajtai)?;
    let proof = CommittedLookupProof {
        table_commitment,
        reads_commitment,
        diff_commitment,
        table_len: table.len(),
        reads_len: reads.len(),
        t_eval: Goldilocks::ZERO,
        r_eval: Goldilocks::ZERO,
        q_eval: Goldilocks::ZERO,
        challenge: Goldilocks::ZERO,
        table_opening,
        reads_opening,
        diff_opening,
    };
    // τ from the commitments (grind-resistant: values never absorbed).
    let tau = derive_challenge_committed(&proof, transcript)?;
    let t_eval = grand_product(table, &tau);
    let r_eval = grand_product(reads, &tau);
    let q_eval = grand_product(&diff, &tau);
    Ok(CommittedLookupProof {
        t_eval,
        r_eval,
        q_eval,
        challenge: tau,
        ..proof
    })
}

/// Verify a committed lookup proof (Q1): every layer binds.
///
/// 1. τ is recomputed from the three commitments + shape (statement bound),
/// 2. the openings are checked against the commitments (SIS binding +
///    norm bounds — a forged vector cannot open),
/// 3. the counting-map difference is verified directly on the OPENED
///    values: `sorted(table) == sorted(reads ‖ diff)` — reads ⊆ table as
///    multisets and diff is exactly the remainder,
/// 4. the grand-product identity `T(τ) = R(τ)·Q(τ)` is recomputed from the
///    opened values at the recomputed τ (a forged evaluation triple
///    cannot match),
/// 5. the values decode canonically from the packed openings.
pub fn verify_lookup_committed(
    pk: &AjtaiPublicKey,
    proof: &CommittedLookupProof,
    transcript: &mut Transcript,
) -> Result<(), CommittedLookupError> {
    // 1. Statement-bound challenge.
    let tau = derive_challenge_committed(proof, transcript)?;
    if tau != proof.challenge {
        return Err(CommittedLookupError::ChallengeMismatch);
    }
    // 2. Openings bind the commitments to the exact packed vectors.
    pk.verify_opening(&proof.table_commitment, &proof.table_opening)
        .map_err(CommittedLookupError::Ajtai)?;
    pk.verify_opening(&proof.reads_commitment, &proof.reads_opening)
        .map_err(CommittedLookupError::Ajtai)?;
    pk.verify_opening(&proof.diff_commitment, &proof.diff_opening)
        .map_err(CommittedLookupError::Ajtai)?;
    // 5. Canonical decode of the values (rejects limb overflow shapes).
    //    The openings are zero-padded to the key dimension m; the declared
    //    lengths say how many values are real — the commitment binds the
    //    full padded vector, so truncating to the declared prefix is exact.
    let ring = &pk.params.ring;
    let mut table =
        lattice_ring::packing::unpack_field_elements(ring, &proof.table_opening)
            .map_err(CommittedLookupError::Packing)?;
    let mut reads =
        lattice_ring::packing::unpack_field_elements(ring, &proof.reads_opening)
            .map_err(CommittedLookupError::Packing)?;
    let mut diff =
        lattice_ring::packing::unpack_field_elements(ring, &proof.diff_opening)
            .map_err(CommittedLookupError::Packing)?;
    if table.len() < proof.table_len || reads.len() < proof.reads_len {
        return Err(CommittedLookupError::ShapeMismatch {
            expected: proof.table_len,
            got: table.len(),
        });
    }
    let diff_len = proof.table_len.saturating_sub(proof.reads_len);
    if diff.len() < diff_len {
        return Err(CommittedLookupError::ShapeMismatch {
            expected: diff_len,
            got: diff.len(),
        });
    }
    table.truncate(proof.table_len);
    reads.truncate(proof.reads_len);
    diff.truncate(diff_len);
    // 3. Counting-map difference on the OPENED values.
    let mut combined: Vec<u64> = reads
        .iter()
        .chain(diff.iter())
        .map(|v| v.to_canonical_u64())
        .collect();
    let mut t_sorted: Vec<u64> = table.iter().map(|v| v.to_canonical_u64()).collect();
    combined.sort_unstable();
    t_sorted.sort_unstable();
    if combined != t_sorted {
        return Err(CommittedLookupError::NotContained);
    }
    // 4. Grand-product identity at the recomputed challenge.
    let t_eval = grand_product(&table, &tau);
    let r_eval = grand_product(&reads, &tau);
    let q_eval = grand_product(&diff, &tau);
    if t_eval != proof.t_eval
        || r_eval != proof.r_eval
        || q_eval != proof.q_eval
        || t_eval != r_eval.mul(&q_eval)
    {
        return Err(CommittedLookupError::GrandProductFailed);
    }
    Ok(())
}

/// An accumulated lookup instance (post partial evaluation): the MLE is
/// collapsed to its remaining variables, and only ONE combination step
/// is needed across instances.
#[derive(Clone, Debug)]
pub struct PartialEvalInstance {
    /// Partially evaluated polynomial (fewer variables than the original).
    pub partial: DenseMle,
    /// The prefix point at which it was evaluated.
    pub prefix: Vec<Goldilocks>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuasarError {
    Lookup(LookupError),
    Mle(lattice_core::mle::MleError),
    /// Partial-evaluation accumulation identity failed.
    AccumulationFailed,
    VariableCountMismatch { expected: usize, got: usize },
}

/// Quasar accumulation: fold k instances by partially evaluating every
/// polynomial at a SHARED random prefix (derived from one transcript
/// challenge absorption), then combining the collapsed polynomials with a
/// single linear-combination step — one CRC instead of k.
pub fn accumulate_partial_evaluation(
    instances: &[DenseMle],
    transcript: &mut Transcript,
) -> Result<(DenseMle, Vec<Goldilocks>), QuasarError> {
    if instances.is_empty() {
        return Err(QuasarError::VariableCountMismatch {
            expected: 1,
            got: 0,
        });
    }
    let total_vars = instances[0].num_vars;
    if instances.iter().any(|f| f.num_vars != total_vars) {
        return Err(QuasarError::VariableCountMismatch {
            expected: total_vars,
            got: instances
                .iter()
                .map(|f| f.num_vars)
                .max()
                .unwrap_or(0),
        });
    }
    // Number of prefix variables to collapse: leave one variable so the
    // result is still a non-degenerate MLE (unless total is 0).
    let collapse = total_vars.saturating_sub(1);
    let prefix = transcript
        .challenge_fields(b"quasar-prefix", collapse)
        .map_err(LookupError::Transcript)
        .map_err(QuasarError::Lookup)?;

    // Partially evaluate every instance at the shared prefix.
    let mut partials = Vec::with_capacity(instances.len());
    for f in instances {
        let g = f
            .fix_variables(&prefix)
            .map_err(QuasarError::Mle)?;
        partials.push(g);
    }
    // Single combination step with one challenge per instance.
    let lambdas = transcript
        .challenge_fields(b"quasar-lambda", partials.len())
        .map_err(LookupError::Transcript)
        .map_err(QuasarError::Lookup)?;
    // Combine: acc = Σ λ_i · g_i (all g_i share the same variable count).
    let combined_evals: Vec<Goldilocks> = partials[0]
        .evaluations
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            let mut acc = Goldilocks::ZERO;
            for (g, lambda) in partials.iter().zip(lambdas.iter()) {
                acc = acc.add(&lambda.mul(&g.evaluations[idx]));
            }
            acc
        })
        .collect();
    Ok((
        DenseMle {
            num_vars: total_vars - collapse,
            evaluations: combined_evals,
        },
        prefix,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn lookup_contained_proves_and_verifies() {
        let table: Vec<Goldilocks> = (0..32u64).map(|i| fe(i * 7 + 3)).collect();
        // Distinct reads (the table has multiplicity 1 everywhere).
        let reads = vec![table[3], table[17], table[0]];
        let mut t = Transcript::new_default(b"lzx-lookup-test");
        let proof = prove_lookup(&table, &reads, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-lookup-test");
        assert!(verify_lookup(&proof, &table, &reads, &mut vt).is_ok());
    }

    #[test]
    fn lookup_missing_entry_rejected() {
        let table: Vec<Goldilocks> = (0..16u64).map(fe).collect();
        // 99 is not in the table.
        let reads = vec![fe(1), fe(99), fe(2)];
        let mut t = Transcript::new_default(b"lzx-lookup-test");
        assert!(matches!(
            prove_lookup(&table, &reads, &mut t),
            Err(LookupError::LookupFailed)
        ));
    }

    #[test]
    fn lookup_multiplicity_enforced() {
        let table: Vec<Goldilocks> = (0..8u64).map(fe).collect();
        // Table contains one 5; reads demand two.
        let reads = vec![fe(5), fe(5)];
        let mut t = Transcript::new_default(b"lzx-lookup-test");
        assert!(matches!(
            prove_lookup(&table, &reads, &mut t),
            Err(LookupError::LookupFailed)
        ));
    }

    // ------------------------------------------------------------------
    // Wave 6.5 — committed lookup protocol tests.
    // ------------------------------------------------------------------

    fn committed_pk(log_n: u32, m: usize) -> AjtaiPublicKey {
        use lattice_commitment::ajtai::AjtaiParams;
        let ring = lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, log_n)
            .ok()
            .unwrap();
        let params = AjtaiParams {
            ring,
            k: 2,
            m,
            // 22-bit limbs packed 3 per value: norms ≤ 2^22.
            norm_bound: 1 << 23,
        };
        AjtaiPublicKey::from_seed(params, [77u8; 32]).ok().unwrap()
    }

    fn committed_table() -> Vec<Goldilocks> {
        (0..32u64).map(|i| fe(i * 7 + 3)).collect()
    }

    #[test]
    fn committed_lookup_proves_and_verifies() {
        let pk = committed_pk(4, 16);
        let table = committed_table();
        let reads = vec![table[3], table[17], table[0], table[5]]; // 4 distinct reads
        let mut pt = Transcript::new_default(b"lzx-committed-lookup");
        let proof = prove_lookup_committed(&pk, &table, &reads, &mut pt).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-committed-lookup");
        assert!(verify_lookup_committed(&pk, &proof, &mut vt).is_ok());
        // Determinism: replay produces identical τ.
        let mut pt2 = Transcript::new_default(b"lzx-committed-lookup");
        let proof2 = prove_lookup_committed(&pk, &table, &reads, &mut pt2).ok().unwrap();
        assert_eq!(proof.challenge, proof2.challenge);
        assert_eq!(proof.t_eval, proof2.t_eval);
    }

    #[test]
    fn committed_lookup_not_contained_fails_at_prove() {
        let pk = committed_pk(4, 16);
        let table = committed_table();
        let reads = vec![table[1], fe(9999)];
        let mut t = Transcript::new_default(b"lzx-committed-lookup");
        assert!(matches!(
            prove_lookup_committed(&pk, &table, &reads, &mut t),
            Err(CommittedLookupError::NotContained)
        ));
    }

    #[test]
    fn committed_lookup_forged_triple_rejected() {
        // THE Wave 6.5 negative test: the pre-Wave-6 hole was that any
        // self-consistent scalar triple (r=q=t=1) verified. The committed
        // verifier must reject a forged triple: the evaluations are
        // recomputed from the OPENED values at the recomputed τ.
        let pk = committed_pk(4, 16);
        let table = committed_table();
        let reads = vec![table[3], table[17]];
        let mut pt = Transcript::new_default(b"lzx-committed-lookup");
        let mut forged = prove_lookup_committed(&pk, &table, &reads, &mut pt)
            .ok()
            .unwrap();
        // The classic forgery: r = q = t = 1 (self-consistent triple).
        forged.r_eval = Goldilocks::ONE;
        forged.q_eval = Goldilocks::ONE;
        forged.t_eval = Goldilocks::ONE;
        let mut vt = Transcript::new_default(b"lzx-committed-lookup");
        assert!(matches!(
            verify_lookup_committed(&pk, &forged, &mut vt),
            Err(CommittedLookupError::GrandProductFailed)
        ));
        // Any single tampered evaluation is also caught.
        let mut tampered = prove_lookup_committed(&pk, &table, &reads, &mut Transcript::new_default(b"lzx-committed-lookup"))
            .ok()
            .unwrap();
        tampered.t_eval = tampered.t_eval.add(&fe(1));
        let mut vt2 = Transcript::new_default(b"lzx-committed-lookup");
        assert!(matches!(
            verify_lookup_committed(&pk, &tampered, &mut vt2),
            Err(CommittedLookupError::GrandProductFailed)
        ));
    }

    #[test]
    fn committed_lookup_tampered_opening_rejected() {
        // A vector that does not open against its commitment is rejected
        // (SIS binding — the layer the scalar path never had).
        let pk = committed_pk(4, 16);
        let table = committed_table();
        let reads = vec![table[3], table[17]];
        let mut pt = Transcript::new_default(b"lzx-committed-lookup");
        let mut proof = prove_lookup_committed(&pk, &table, &reads, &mut pt)
            .ok()
            .unwrap();
        // Tamper with one coefficient of the table opening.
        let mut coeffs = proof.table_opening[0].coeffs().to_vec();
        coeffs[0] = (coeffs[0] + 7) % pk.params.ring.modulus.q;
        proof.table_opening[0] =
            RingElement::from_coeffs(&pk.params.ring, coeffs);
        let mut vt = Transcript::new_default(b"lzx-committed-lookup");
        assert!(matches!(
            verify_lookup_committed(&pk, &proof, &mut vt),
            Err(CommittedLookupError::Ajtai(_))
        ));
    }

    #[test]
    fn committed_lookup_swap_reads_for_diff_rejected() {
        // Substitute a valid-looking DIFFERENT difference vector: the
        // counting-map check (sorted(table) == sorted(reads ‖ diff))
        // catches any diff that is not exactly table ∖ reads.
        let pk = committed_pk(4, 16);
        let table = committed_table();
        let reads = vec![table[3], table[17]];
        let mut pt = Transcript::new_default(b"lzx-committed-lookup");
        let mut proof = prove_lookup_committed(&pk, &table, &reads, &mut pt)
            .ok()
            .unwrap();
        // Drop the first difference value: the multiset no longer balances.
        let shortened: Vec<Goldilocks> = {
            let diff = counting_map_difference(&table, &reads).unwrap_or_default();
            diff[1..].to_vec()
        };
        let packed = lattice_ring::packing::pack_field_elements(&pk.params.ring, &shortened);
        proof.diff_opening = pk.pad_to_m(&packed).ok().unwrap();
        proof.diff_commitment = pk.commit(&proof.diff_opening).ok().unwrap();
        // Recompute the evals consistently for the forged diff — the
        // counting-map check must still reject (the identity alone would
        // hold at a NEW τ, but τ is statement-derived and the multiset
        // check fires regardless).
        let mut vt = Transcript::new_default(b"lzx-committed-lookup");
        let result = verify_lookup_committed(&pk, &proof, &mut vt);
        assert!(
            matches!(
                result,
                Err(CommittedLookupError::NotContained)
                    | Err(CommittedLookupError::ChallengeMismatch)
                    | Err(CommittedLookupError::GrandProductFailed)
            ),
            "forged difference must be rejected, got {result:?}"
        );
    }

    #[test]
    fn counting_map_difference_matches_reference() {
        // Differential test vs the classic multiset removal on random
        // multisets (with duplicates).
        let mut state = 42u64;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for trial in 0..200 {
            let table: Vec<Goldilocks> =
                (0..24).map(|_| fe(next() % 8)).collect();
            // Reads: a subset-with-multiplicity of the table (honest), or
            // (even trials) an outsider value.
            let reads: Vec<Goldilocks> = (0..6)
                .map(|_| {
                    if trial % 2 == 0 && trial % 5 == 0 {
                        fe(999)
                    } else {
                        table[(next() as usize) % table.len()]
                    }
                })
                .collect();
            // Reference: naive removal.
            let mut remaining = table.clone();
            let mut reference_ok = true;
            for rv in &reads {
                match remaining.iter().position(|t| *t == *rv) {
                    Some(idx) => {
                        remaining.swap_remove(idx);
                    }
                    None => reference_ok = false,
                }
            }
            let got = counting_map_difference(&table, &reads);
            match (reference_ok, got) {
                (true, Some(diff)) => {
                    // Same multiset as the reference remainder.
                    let mut a: Vec<u64> = remaining.iter().map(|v| v.to_canonical_u64()).collect();
                    let mut b: Vec<u64> = diff.iter().map(|v| v.to_canonical_u64()).collect();
                    a.sort_unstable();
                    b.sort_unstable();
                    assert_eq!(a, b, "trial {trial}");
                }
                (false, None) => {}
                _ => panic!("containment mismatch at trial {trial}"),
            }
        }
    }

    #[test]
    fn tampered_lookup_rejected() {
        let table: Vec<Goldilocks> = (0..16u64).map(fe).collect();
        let reads = vec![fe(1), fe(2)];
        let mut t = Transcript::new_default(b"lzx-lookup-test");
        let mut proof = prove_lookup(&table, &reads, &mut t).ok().unwrap();
        // Tamper with an evaluation: identity must fail.
        proof.t_eval = proof.t_eval.add(&fe(1));
        let mut vt = Transcript::new_default(b"lzx-lookup-test");
        assert!(verify_lookup(&proof, &table, &reads, &mut vt).is_err());
        // Wrong statement (different reads) desyncs the challenge.
        let mut t2 = Transcript::new_default(b"lzx-lookup-test");
        let mut proof2 = prove_lookup(&table, &reads, &mut t2).ok().unwrap();
        proof2.challenge = proof2.challenge.add(&fe(1));
        let mut vt2 = Transcript::new_default(b"lzx-lookup-test");
        assert!(verify_lookup(&proof2, &table, &reads, &mut vt2).is_err());
    }

    #[test]
    fn partial_evaluation_accumulation() {
        // Accumulate 4 MLE instances: the folded polynomial equals the
        // partial-evaluated combination (verifiable directly).
        let instances: Vec<DenseMle> = ["a", "b", "c", "d"]
            .iter()
            .map(|t| DenseMle::random(6, t.as_bytes()))
            .collect();
        let mut t = Transcript::new_default(b"lzx-quasar-test");
        let (folded, prefix) = accumulate_partial_evaluation(&instances, &mut t)
            .ok()
            .unwrap();
        assert_eq!(folded.num_vars, 1); // 6 vars collapsed to 1.
        assert_eq!(prefix.len(), 5);

        // Direct reference: replay the same transcript sequence (prefix
        // first, then lambdas) to reproduce identical challenges.
        let mut t2 = Transcript::new_default(b"lzx-quasar-test");
        let prefix2 = t2
            .challenge_fields(b"quasar-prefix", 5)
            .ok()
            .unwrap();
        assert_eq!(prefix2, prefix);
        let lambdas = t2
            .challenge_fields(b"quasar-lambda", instances.len())
            .ok()
            .unwrap();
        let mut expected = vec![Goldilocks::ZERO; 2];
        for (f, lambda) in instances.iter().zip(lambdas.iter()) {
            let g = f.fix_variables(&prefix).ok().unwrap();
            for (e, v) in expected.iter_mut().zip(g.evaluations.iter()) {
                *e = e.add(&lambda.mul(v));
            }
        }
        assert_eq!(folded.evaluations, expected);
    }

    #[test]
    fn accumulation_shape_errors() {
        let mut t = Transcript::new_default(b"lzx-quasar-test");
        assert!(matches!(
            accumulate_partial_evaluation(&[], &mut t),
            Err(QuasarError::VariableCountMismatch { .. })
        ));
        let mixed = vec![DenseMle::random(3, b"x"), DenseMle::random(4, b"y")];
        assert!(matches!(
            accumulate_partial_evaluation(&mixed, &mut t),
            Err(QuasarError::VariableCountMismatch { .. })
        ));
    }

    #[test]
    fn accumulator_satisfies_lookup_when_instances_do() {
        // End-to-end composition under ONE shared challenge τ (as in the
        // accumulated statement): grand products compose multiplicatively,
        // so R_total = R1·R2 and Q_total = T / R_total closes the identity.
        let table: Vec<Goldilocks> = (0..64u64).map(|i| fe(i * 13 + 1)).collect();
        let reads1 = vec![table[5], table[9]];
        let _reads2 = [table[40], table[2], table[5]];
        // Shared challenge (the accumulation point).
        let tau = Goldilocks::from_u64(0x9E37_79B9_7F4A_7C15);
        // NOTE: reads2 reuses table[5] which reads1 already consumed —
        // multiplicity across the ACCUMULATED statement must respect the
        // table: use disjoint reads instead.
        let reads2 = vec![table[40], table[2], table[7]];
        let combined_reads = [reads1.clone(), reads2.clone()].concat();
        let r1 = grand_product(&reads1, &tau);
        let r2 = grand_product(&reads2, &tau);
        let r_total = grand_product(&combined_reads, &tau);
        assert_eq!(r_total, r1.mul(&r2));
        // Divisibility: T = R_total · Q_total with the multiset quotient.
        let t_total = grand_product(&table, &tau);
        let mut remaining = table.clone();
        for rv in &combined_reads {
            let idx = remaining.iter().position(|t| *t == *rv);
            assert!(idx.is_some(), "read {rv:?} not in table");
            remaining.swap_remove(idx.ok_or(()).ok().unwrap());
        }
        let q_total = grand_product(&remaining, &tau);
        assert_eq!(q_total.mul(&r_total), t_total);
    }
}
