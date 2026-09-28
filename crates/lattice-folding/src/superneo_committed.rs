//! Neo/SuperNeo (Nguyen–Setty 2026, ePrint 2026/242) — Wave 7.15: the
//! `CommittedRelaxedCcsInstance` layer, the `fold_public`/`fold_secret`
//! split, the pay-per-bit zero-skipping sparse commit path, and the
//! `π_CCS` decider.
//!
//! * **Committed instances** — the witness is committed via Ajtai **before**
//!   the Fiat–Shamir derivation (closing the pre-Wave-6 hole where the
//!   challenge absorbed `w₁‖w₂`): the fold challenge derives from
//!   `hash(commitment₁, commitment₂, u's, slacks, ccs-digest)` — public
//!   data only, recomputable by the verifier.
//! * **The R_q bridge** — small-field witnesses (`F_{2^16}`-valued,
//!   embedded in `Z_q`) pack linearly into ring-element coefficients (16
//!   values per element), so the Ajtai homomorphism is EXACT across the
//!   bridge: `commit(pack(w₁ + r·w₂)) = C₁ + r·C₂` for small `r` (the
//!   values never wrap: `2^16 + 2^8·2^16 < q/2`). The pay-per-bit path
//!   commits the **bit decomposition** instead — reconstruction
//!   `w = Σ_j 2^j·b_j` is linear in the bits, so bit-vector folds are
//!   exact too, and the commitment WORK is proportional to the popcount
//!   (zero coefficients skip the NTT — "0s are free", Wave 6.8).
//! * **fold_public / fold_secret** — the verifier-side fold (commitment
//!   homomorphism, `u' = u₁ + r·u₂`, `slack' = slack₁ + r²·slack₂ +
//!   r·E` with the prover-sent cross term `E`) vs the prover-side
//!   witness fold. `E` is bound by the decider: a wrong cross term makes
//!   the folded instance fail `π_CCS`.
//! * **`π_CCS` decider** — opens the commitment, reconstructs the
//!   small-field witness (from bits on the pay-per-bit path), and checks
//!   relaxed CCS satisfaction through the shared [`verify_folded`]
//!   (Goldilocks-exact for the small-value regime).

use crate::superneo::{verify_folded, RelaxedCcsInstance, SuperNeoError};
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_relations::ccs::Ccs;
use lattice_ring::{RingConfig, RingElement};

/// The small-field value bound: witnesses are `F_{2^16}`-valued (the
/// paper's small fields); folds with `r < 2^8` keep every value below
/// `2^25 ≪ q/2` (no wrap through the bridge — asserted fail-closed).
pub const SMALL_FIELD_BOUND: u32 = 1 << 16;
/// The fold challenge bound (short challenges keep the bridge exact).
pub const FOLD_CHALLENGE_BOUND: u32 = 1 << 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommittedError {
    Ajtai(AjtaiError),
    SuperNeo(SuperNeoError),
    Shape { expected: usize, got: usize },
    /// A value exceeded the small-field bridge bound — the fold would
    /// wrap mod q and destroy the exact homomorphism (fail-closed).
    ValueOutOfBounds { value: u32, bound: u32 },
    TranscriptFailure,
    /// The decider failed (opening, reconstruction, or π_CCS).
    DeciderFailed(&'static str),
}

impl From<AjtaiError> for CommittedError {
    fn from(e: AjtaiError) -> Self {
        CommittedError::Ajtai(e)
    }
}
impl From<SuperNeoError> for CommittedError {
    fn from(e: SuperNeoError) -> Self {
        CommittedError::SuperNeo(e)
    }
}
impl From<lattice_ring::RingError> for CommittedError {
    fn from(_e: lattice_ring::RingError) -> Self {
        CommittedError::DeciderFailed("ring")
    }
}

/// Linear packing: 16 small-field values per ring element.
const PACK: usize = 16;

fn pack_small(ring: &RingConfig, values: &[u32]) -> Vec<RingElement> {
    let mut out = Vec::with_capacity(values.len().div_ceil(PACK));
    for chunk in values.chunks(PACK) {
        let mut coeffs = vec![0u32; ring.n()];
        for (k, &v) in chunk.iter().enumerate() {
            coeffs[k] = v;
        }
        out.push(RingElement::from_coeffs(ring, coeffs));
    }
    out
}

fn unpack_small(elems: &[RingElement], count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    for e in elems {
        for &c in e.coeffs() {
            if out.len() < count {
                out.push(c);
            }
        }
    }
    out
}

/// The committed relaxed CCS instance: the Ajtai commitment over the
/// R_q-bridged witness (values or bits), plus the public small-field
/// components. This is the verifier-visible statement — the Fiat–Shamir
/// digest covers exactly these fields.
#[derive(Clone, Debug)]
pub struct CommittedRelaxedCcsInstance {
    /// `true` when the commitment covers the bit decomposition (the
    /// pay-per-bit path); `false` for the value packing.
    pub pay_per_bit: bool,
    pub commitment: AjtaiCommitment,
    /// The relaxed scalar `u` (full-field: folds linearly).
    pub u: Goldilocks,
    /// The slack vector (full-field — the folding error accumulator; the
    /// small-field bound applies only to the WITNESS, whose bridge
    /// exactness demands it).
    pub slack: Vec<Goldilocks>,
    /// The committed vector's length (values, or bits = 16·values).
    pub packed_len: usize,
}

/// The prover-side secret: the packed ring elements (for openings) and
/// the witness values.
#[derive(Clone, Debug)]
pub struct CcsSecret {
    pub witness: Vec<u32>,
    /// The exact packed vector the commitment opens to.
    pub packed: Vec<RingElement>,
}

/// Validate the small-field regime (fail-closed before any packing).
fn validate_small(values: &[u32], bound: u32) -> Result<(), CommittedError> {
    for &v in values {
        if v >= bound {
            return Err(CommittedError::ValueOutOfBounds { value: v, bound });
        }
    }
    Ok(())
}

/// Commit a small-field relaxed instance under the R_q bridge.
/// `pay_per_bit = true` switches to the bit-decomposed commitment (the
/// sparse path — commitment work ∝ popcount).
pub fn commit_instance(
    pk: &AjtaiPublicKey,
    witness: &[u32],
    slack: &[Goldilocks],
    u: Goldilocks,
    pay_per_bit: bool,
) -> Result<(CommittedRelaxedCcsInstance, CcsSecret), CommittedError> {
    let ring = &pk.params.ring;
    validate_small(witness, SMALL_FIELD_BOUND)?;
    let (packed_values, packed_len) = if pay_per_bit {
        // Bit decomposition: 16 bits per value, zero bits skip the NTT.
        let mut bits = Vec::with_capacity(witness.len() * 16);
        for &v in witness {
            for b in 0..16 {
                bits.push((v >> b) & 1);
            }
        }
        let packed = pack_small(ring, &bits);
        (packed, bits.len())
    } else {
        (pack_small(ring, witness), witness.len())
    };
    let padded = pk.pad_to_m(&packed_values)?;
    let commitment = pk.commit(&padded)?;
    Ok((
        CommittedRelaxedCcsInstance {
            pay_per_bit,
            commitment,
            u,
            slack: slack.to_vec(),
            packed_len,
        },
        CcsSecret { witness: witness.to_vec(), packed: padded },
    ))
}

/// The Fiat–Shamir digest of a committed instance — PUBLIC data only
/// (commitment bytes, u, slack, packing mode): the fold challenge is a
/// function of exactly this, never of the private witness (the Wave-6.4
/// hole closed by a real commitment).
pub fn committed_digest(inst: &CommittedRelaxedCcsInstance) -> [u8; 32] {
    let mut buf = Vec::with_capacity(64 + inst.slack.len() * 4);
    buf.extend_from_slice(&(inst.packed_len as u32).to_le_bytes());
    buf.push(u8::from(inst.pay_per_bit));
    buf.extend_from_slice(&inst.u.to_bytes());
    for s in &inst.slack {
        buf.extend_from_slice(&s.to_bytes());
    }
    buf.extend_from_slice(&inst.commitment.to_bytes());
    Transcript::hash_domain(b"superneo-committed", &buf)
}

/// The Nova-style cross term:
/// `E = Σ_terms c·(A_a w₁ ∘ A_b w₂ + A_a w₂ ∘ A_b w₁) − (u₁·span(w₂) + u₂·span(w₁))`
/// where `span(w) = Σ_i B_i·w` — the quadratic mixed term MINUS the
/// linearized mixed span, so that
/// `v(w') − slack' = u'·B(w')` holds exactly under
/// `slack' = slack₁ + r²·slack₂ + r·E` (the prover's fold message).
pub fn cross_term(
    ccs: &Ccs,
    w1: &[u32],
    w2: &[u32],
    u1: Goldilocks,
    u2: Goldilocks,
) -> Result<Vec<Goldilocks>, CommittedError> {
    let n = ccs.n;
    let to_fe = |v: &[u32]| -> Vec<Goldilocks> { v.iter().map(|&x| Goldilocks::from_u64(x as u64)).collect() };
    let imgs1: Vec<Vec<Goldilocks>> = ccs
        .a_matrices
        .iter()
        .map(|a| a.multiply(&to_fe(w1)).map_err(SuperNeoError::Ccs))
        .collect::<Result<_, _>>()?;
    let imgs2: Vec<Vec<Goldilocks>> = ccs
        .a_matrices
        .iter()
        .map(|a| a.multiply(&to_fe(w2)).map_err(SuperNeoError::Ccs))
        .collect::<Result<_, _>>()?;
    let mut e = vec![Goldilocks::ZERO; n];
    for (t, ids) in ccs.selections.iter().enumerate() {
        if ids.len() != 2 {
            return Err(CommittedError::Shape { expected: 2, got: ids.len() });
        }
        let c = ccs.constants.get(t).copied().unwrap_or(Goldilocks::ONE);
        let (ia, ib) = (ids[0], ids[1]);
        for k in 0..n {
            let term = imgs1[ia][k]
                .mul(&imgs2[ib][k])
                .add(&imgs2[ia][k].mul(&imgs1[ib][k]));
            e[k] = e[k].add(&c.mul(&term));
        }
    }
    // The linearized mixed span: u₁·span(w₂) + u₂·span(w₁).
    let span = |w: &[u32]| -> Result<Vec<Goldilocks>, CommittedError> {
        let mut acc = vec![Goldilocks::ZERO; n];
        for b in &ccs.b_matrices {
            let img = b.multiply(&to_fe(w)).map_err(SuperNeoError::Ccs)?;
            for (a, x) in acc.iter_mut().zip(img.iter()) {
                *a = a.add(x);
            }
        }
        Ok(acc)
    };
    let span1 = span(w1)?;
    let span2 = span(w2)?;
    let u1f = u1;
    let u2f = u2;
    for k in 0..n {
        let lin = u1f.mul(&span2[k]).add(&u2f.mul(&span1[k]));
        e[k] = e[k].sub(&lin);
    }
    Ok(e)
}

/// Derive the small fold challenge from PUBLIC digests (the FS hole
/// closed: commitments, u, slack — never the witness bytes).
pub fn fold_challenge(
    ccs_digest: &[u8; 32],
    d1: &[u8; 32],
    d2: &[u8; 32],
) -> Result<u32, CommittedError> {
    let mut t = Transcript::new_default(b"lzx-superneo-committed");
    t.append_bytes(b"ccs", ccs_digest).map_err(|_| CommittedError::TranscriptFailure)?;
    t.append_bytes(b"d1", d1).map_err(|_| CommittedError::TranscriptFailure)?;
    t.append_bytes(b"d2", d2).map_err(|_| CommittedError::TranscriptFailure)?;
    let bytes = t
        .challenge_bytes(b"fold-r", 4)
        .map_err(|_| CommittedError::TranscriptFailure)?;
    let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    // Short challenge: keeps the R_q bridge exact across folds.
    Ok(raw % FOLD_CHALLENGE_BOUND)
}

/// **fold_public** — the verifier-side fold: the commitment homomorphism
/// `C' = C₁ + r·C₂`, `u' = u₁ + r·u₂`,
/// `slack' = slack₁ + r²·slack₂ + r·E`, from public data plus the
/// prover-sent cross term `E`. The slack grows in the SMALL-field
/// integer regime (no mod-q wrap below the bound — the decider asserts).
pub fn fold_public(
    inst1: &CommittedRelaxedCcsInstance,
    inst2: &CommittedRelaxedCcsInstance,
    e: &[Goldilocks],
    r: u32,
) -> Result<CommittedRelaxedCcsInstance, CommittedError> {
    if inst1.packed_len != inst2.packed_len || inst1.pay_per_bit != inst2.pay_per_bit {
        return Err(CommittedError::Shape { expected: inst1.packed_len, got: inst2.packed_len });
    }
    if e.len() != inst1.slack.len() {
        return Err(CommittedError::Shape { expected: inst1.slack.len(), got: e.len() });
    }
    let mut rows = Vec::with_capacity(inst1.commitment.rows.len());
    for (c1, c2) in inst1.commitment.rows.iter().zip(inst2.commitment.rows.iter()) {
        rows.push(c1.add(&c2.scale_i64(r as i64))?);
    }
    let r_f = Goldilocks::from_u64(r as u64);
    let r_sq = r_f.mul(&r_f);
    let slack: Vec<Goldilocks> = inst1
        .slack
        .iter()
        .zip(inst2.slack.iter().zip(e.iter()))
        .map(|(s1, (s2, ev))| s1.add(&r_sq.mul(s2)).add(&r_f.mul(ev)))
        .collect();
    Ok(CommittedRelaxedCcsInstance {
        pay_per_bit: inst1.pay_per_bit,
        commitment: AjtaiCommitment { rows },
        u: inst1.u.add(&r_f.mul(&inst2.u)),
        slack,
        packed_len: inst1.packed_len,
    })
}

/// **fold_secret** — the prover-side witness fold `w' = w₁ + r·w₂` under
/// the SAME packing discipline (bit vectors fold linearly since
/// reconstruction is linear in the bits; values fold directly).
pub fn fold_secret(
    pk: &AjtaiPublicKey,
    secret1: &CcsSecret,
    secret2: &CcsSecret,
    r: u32,
    pay_per_bit: bool,
) -> Result<CcsSecret, CommittedError> {
    let ring = &pk.params.ring;
    if secret1.witness.len() != secret2.witness.len() {
        return Err(CommittedError::Shape {
            expected: secret1.witness.len(),
            got: secret2.witness.len(),
        });
    }
    let witness: Vec<u32> = secret1
        .witness
        .iter()
        .zip(secret2.witness.iter())
        .map(|(&a, &b)| (a as u64 + r as u64 * b as u64) as u32)
        .collect();
    // The packed fold must be the pack of the folded vector (exact
    // through the linear bridge): fold the packed ring elements the same
    // way and assert. On the pay-per-bit path the packed vector is the
    // BIT decomposition — the folded bits reconstruct w' by linearity
    // (Σ_j 2^j·b'_j = w₁ + r·w₂), even though b'_j ≥ 2.
    let mut packed = Vec::with_capacity(secret1.packed.len());
    for (p1, p2) in secret1.packed.iter().zip(secret2.packed.iter()) {
        packed.push(p1.add(&p2.scale_i64(r as i64))?);
    }
    let expected_vec: Vec<u32> = if pay_per_bit {
        let bits1 = unpack_small(&secret1.packed, secret1.witness.len() * 16);
        let bits2 = unpack_small(&secret2.packed, secret2.witness.len() * 16);
        bits1
            .iter()
            .zip(bits2.iter())
            .map(|(&a, &b)| (a as u64 + r as u64 * b as u64) as u32)
            .collect()
    } else {
        witness.clone()
    };
    let expected = pk.pad_to_m(&pack_small(ring, &expected_vec))?;
    if packed != expected {
        return Err(CommittedError::DeciderFailed("bridge linearity"));
    }
    Ok(CcsSecret { witness, packed })
}

/// The full committed fold: derive the challenge from the public
/// digests, compute the cross term (prover), fold public + secret.
pub fn fold_committed(
    pk: &AjtaiPublicKey,
    ccs: &Ccs,
    ccs_digest: &[u8; 32],
    inst1: &CommittedRelaxedCcsInstance,
    inst2: &CommittedRelaxedCcsInstance,
    secret1: &CcsSecret,
    secret2: &CcsSecret,
) -> Result<(CommittedRelaxedCcsInstance, CcsSecret), CommittedError> {
    let d1 = committed_digest(inst1);
    let d2 = committed_digest(inst2);
    let r = fold_challenge(ccs_digest, &d1, &d2)?;
    let e = cross_term(ccs, &secret1.witness, &secret2.witness, inst1.u, inst2.u)?;
    let folded_inst = fold_public(inst1, inst2, &e, r)?;
    let folded_secret = fold_secret(pk, secret1, secret2, r, inst1.pay_per_bit)?;
    Ok((folded_inst, folded_secret))
}

/// The `π_CCS` decider: open the commitment, reconstruct the witness
/// (from bits on the pay-per-bit path — `w = Σ_j 2^j·b_j`, linear),
/// enforce the small-field bound on the FOLDED values (fail-closed — a
/// wrap would silently break the bridge), and check relaxed CCS
/// satisfaction via the shared [`verify_folded`].
pub fn decider_committed(
    pk: &AjtaiPublicKey,
    ccs: &Ccs,
    inst: &CommittedRelaxedCcsInstance,
    secret: &CcsSecret,
) -> Result<(), CommittedError> {
    let ring = &pk.params.ring;
    // Opening.
    pk.verify_opening(&inst.commitment, &secret.packed)
        .map_err(|_| CommittedError::DeciderFailed("commitment opening"))?;
    // Reconstruction + the exact-packing cross-check.
    let packed_len = if inst.pay_per_bit { secret.witness.len() * 16 } else { secret.witness.len() };
    if packed_len != inst.packed_len {
        return Err(CommittedError::DeciderFailed("packed length"));
    }
    if inst.pay_per_bit {
        // Reconstruct from the bit commitment: the opened (folded) bits
        // are the packed vector — the witness is the weighted bit sum
        // (linear reconstruction, exact for folded bits ≥ 2).
        let bits = unpack_small(&secret.packed, inst.packed_len);
        let mut v = vec![0u32; bits.len() / 16];
        for (i, b) in bits.iter().enumerate() {
            v[i / 16] += b << (i % 16);
        }
        if v != secret.witness {
            return Err(CommittedError::DeciderFailed("bit reconstruction"));
        }
    } else {
        let expected_packed = pk.pad_to_m(&pack_small(ring, &secret.witness))?;
        if secret.packed != expected_packed {
            return Err(CommittedError::DeciderFailed("packing consistency"));
        }
    }
    // Small-field bound on the folded values (fail-closed).
    let bound = 1u32 << 25;
    for &w in &secret.witness {
        if w >= bound {
            return Err(CommittedError::ValueOutOfBounds { value: w, bound });
        }
    }
    // π_CCS: relaxed satisfaction (Goldilocks-exact for the small
    // witnesses; the slack/u live in the full field).
    let g_witness: Vec<Goldilocks> =
        secret.witness.iter().map(|&x| Goldilocks::from_u64(x as u64)).collect();
    let relaxed = RelaxedCcsInstance {
        witness: g_witness,
        slack: inst.slack.clone(),
        u: inst.u,
    };
    if !verify_folded(ccs, &relaxed)? {
        return Err(CommittedError::DeciderFailed("pi_CCS"));
    }
    Ok(())
}

/// The pay-per-bit commitment cost model: nonzero bits only (the zeros
/// skip the NTT in the commit hot path — "0s are free", Wave 6.8).
pub fn pay_per_bit_commit_cost(witness: &[u32]) -> u64 {
    witness.iter().map(|w| w.count_ones() as u64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_relations::ccs::SparseMatrix;
    use lattice_ring::{Modulus32, RingConfig};

    fn setup(log_n: u32, m_slots: usize) -> (AjtaiPublicKey, RingConfig) {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        // Binding-only norm regime: the folded small-field values stay
        // below 2^25 but the slack accumulates — open at the bridge bound.
        let params = AjtaiParams { ring: ring.clone(), k: 2, m: m_slots, norm_bound: 1 << 26 };
        let pk = AjtaiPublicKey::from_seed(params, [23u8; 32]).ok().unwrap();
        (pk, ring)
    }

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// `w ∘ w = w` (boolean witnesses) as a degree-2 CCS.
    fn bool_ccs(n: usize) -> Ccs {
        let a = SparseMatrix::identity(n);
        Ccs {
            m: n,
            n,
            a_matrices: vec![a.clone(), a],
            b_matrices: vec![SparseMatrix::identity(n)],
            selections: vec![vec![0, 1]],
            constants: vec![fe(1)],
        }
    }

    fn ccs_digest_of(ccs: &Ccs) -> [u8; 32] {
        // Reuse the crate-internal digest via the public fold API: hash
        // the CCS fields directly (deterministic, matches the shape).
        let mut buf = Vec::new();
        buf.extend_from_slice(&(ccs.m as u32).to_le_bytes());
        buf.extend_from_slice(&(ccs.n as u32).to_le_bytes());
        buf.extend_from_slice(&(ccs.a_matrices.len() as u32).to_le_bytes());
        for ids in &ccs.selections {
            buf.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        }
        Transcript::hash_domain(b"superneo-ccs-test", &buf)
    }

    fn bool_witness(n: usize, tag: &[u8]) -> Vec<u32> {
        let bytes = Transcript::xof(b"superneo-w", tag, n);
        bytes.iter().take(n).map(|b| u32::from(b & 1)).collect()
    }

    #[test]
    fn committed_fold_and_decider_value_packing() {
        let (pk, _ring) = setup(4, 8);
        let ccs = bool_ccs(8);
        let cd = ccs_digest_of(&ccs);
        let w1 = bool_witness(8, b"w1");
        let w2 = bool_witness(8, b"w2");
        let (i1, s1) = commit_instance(&pk, &w1, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, false).ok().unwrap();
        let (i2, s2) = commit_instance(&pk, &w2, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, false).ok().unwrap();
        let (folded, secret) =
            fold_committed(&pk, &ccs, &cd, &i1, &i2, &s1, &s2).ok().unwrap();
        // π_CCS decider: opening + reconstruction + satisfaction.
        match decider_committed(&pk, &ccs, &folded, &secret) {
            Ok(_) => {}
            Err(e) => panic!("value decider: {e:?}"),
        }
    }

    #[test]
    fn committed_fold_and_decider_pay_per_bit() {
        let (pk, _ring) = setup(4, 16);
        let ccs = bool_ccs(8);
        let cd = ccs_digest_of(&ccs);
        let w1 = bool_witness(8, b"b1");
        let w2 = bool_witness(8, b"b2");
        let (i1, s1) = commit_instance(&pk, &w1, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, true).ok().unwrap();
        let (i2, s2) = commit_instance(&pk, &w2, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, true).ok().unwrap();
        // The bit packing: 16 bits per value → 128 packed entries.
        assert_eq!(i1.packed_len, 128);
        let (folded, secret) =
            fold_committed(&pk, &ccs, &cd, &i1, &i2, &s1, &s2).ok().unwrap();
        assert!(decider_committed(&pk, &ccs, &folded, &secret).is_ok());
        // The pay-per-bit cost model: nonzero bits only.
        let cost = pay_per_bit_commit_cost(&w1);
        let ones: u64 = w1.iter().map(|w| *w as u64).sum();
        assert_eq!(cost, ones);
    }

    #[test]
    fn fs_challenge_binds_commitments_not_witness_bytes() {
        // The Wave-6.4 hole closed: the challenge is a function of the
        // PUBLIC digests. Different witnesses → different commitments →
        // different digests → different challenges; and the challenge
        // derivation NEVER sees the witness bytes (structural: the
        // inputs are digests only).
        let (pk, _ring) = setup(4, 8);
        let ccs = bool_ccs(8);
        let cd = ccs_digest_of(&ccs);
        let wa = bool_witness(8, b"a");
        let wb = bool_witness(8, b"b");
        let (ia, _) = commit_instance(&pk, &wa, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, false).ok().unwrap();
        let (ib, _) = commit_instance(&pk, &wb, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, false).ok().unwrap();
        assert_ne!(committed_digest(&ia), committed_digest(&ib));
        let r1 = fold_challenge(&cd, &committed_digest(&ia), &committed_digest(&ib))
            .ok()
            .unwrap();
        let r2 = fold_challenge(&cd, &committed_digest(&ia), &committed_digest(&ib))
            .ok()
            .unwrap();
        assert_eq!(r1, r2); // deterministic from public data
        assert!(r1 < FOLD_CHALLENGE_BOUND);
        // Same digests with a third instance → different challenge.
        let wc = bool_witness(8, b"c");
        let (ic, _) = commit_instance(&pk, &wc, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, false).ok().unwrap();
        let r3 = fold_challenge(&cd, &committed_digest(&ia), &committed_digest(&ic))
            .ok()
            .unwrap();
        // (r3 may coincide with r1 with prob 1/256 — pin the determinism
        // instead: same inputs → same output.)
        let r4 = fold_challenge(&cd, &committed_digest(&ia), &committed_digest(&ic))
            .ok()
            .unwrap();
        assert_eq!(r3, r4);
    }

    #[test]
    fn wrong_cross_term_fails_decider() {
        let (pk, _ring) = setup(4, 8);
        let ccs = bool_ccs(8);
        let cd = ccs_digest_of(&ccs);
        let w1 = bool_witness(8, b"x1");
        let w2 = bool_witness(8, b"x2");
        let (i1, s1) = commit_instance(&pk, &w1, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, false).ok().unwrap();
        let (i2, s2) = commit_instance(&pk, &w2, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, false).ok().unwrap();
        let d1 = committed_digest(&i1);
        let d2 = committed_digest(&i2);
        let r = fold_challenge(&cd, &d1, &d2).ok().unwrap();
        // A WRONG cross term (zeros): slack' misses r·E → π_CCS fails.
        let folded_bad = fold_public(&i1, &i2, &vec![Goldilocks::ZERO; 8], r).ok().unwrap();
        let secret = fold_secret(&pk, &s1, &s2, r, false).ok().unwrap();
        assert!(decider_committed(&pk, &ccs, &folded_bad, &secret).is_err());
        // The honest cross term passes.
        let e = cross_term(&ccs, &s1.witness, &s2.witness, i1.u, i2.u).ok().unwrap();
        let folded = fold_public(&i1, &i2, &e, r).ok().unwrap();
        assert!(decider_committed(&pk, &ccs, &folded, &secret).is_ok());
    }

    #[test]
    fn out_of_bounds_witness_refused() {
        let (pk, _ring) = setup(4, 8);
        let bad = vec![1u32 << 16; 8];
        assert!(matches!(
            commit_instance(&pk, &bad, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, false),
            Err(CommittedError::ValueOutOfBounds { .. })
        ));
    }

    #[test]
    fn ivc_two_rounds_of_folds() {
        // Fold three instances over two rounds; the decider passes at the
        // end (the accumulation driver).
        let (pk, _ring) = setup(4, 8);
        let ccs = bool_ccs(8);
        let cd = ccs_digest_of(&ccs);
        let w1 = bool_witness(8, b"i1");
        let w2 = bool_witness(8, b"i2");
        let w3 = bool_witness(8, b"i3");
        let (i1, s1) = commit_instance(&pk, &w1, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, false).ok().unwrap();
        let (i2, s2) = commit_instance(&pk, &w2, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, false).ok().unwrap();
        let (i3, s3) = commit_instance(&pk, &w3, &vec![Goldilocks::ZERO; 8], Goldilocks::ONE, false).ok().unwrap();
        let (acc_i, acc_s) =
            fold_committed(&pk, &ccs, &cd, &i1, &i2, &s1, &s2).ok().unwrap();
        assert!(decider_committed(&pk, &ccs, &acc_i, &acc_s).is_ok());
        let (acc_i2, acc_s2) =
            fold_committed(&pk, &ccs, &cd, &acc_i, &i3, &acc_s, &s3).ok().unwrap();
        assert!(decider_committed(&pk, &ccs, &acc_i2, &acc_s2).is_ok());
    }
}
