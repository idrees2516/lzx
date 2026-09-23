//! The zero-knowledge sumcheck over a secret committed value vector.
//!
//! **Statement**: a secret vector `F` of `2^m` field values is Ajtai
//! committed; prove `\Sigma_{x\in{0,1}^m} F(x) = claim` and reveal the
//! evaluation `F(r) = v` at the Fiat-Shamir point — without leaking
//! anything else about `F`.
//!
//! **Construction** (Libra-style masking, made lattice-native with
//! CRT-carrier relations):
//!
//! 1.  The prover samples a uniform masking MLE `\rho` with
//!     `\Sigma \rho = 0` from **secret entropy** and forms the blinded
//!     vector `F\~ = F + \rho`.
//! 2.  **Pre-round commitment** `T1` covers `[F-limbs | rho-limbs |
//!     F\~-limbs | per-value decomposition carries | zero-sum carry
//!     limbs]` and is absorbed into the transcript **before any
//!     challenge** — the Fiat-Shamir ordering that makes the sumcheck
//!     extractable (F\~ is fixed before the round challenges).
//! 3.  A standard multilinear sumcheck runs over `F\~` (degree-1
//!     rounds). Since `\rho` is uniform on the `\Sigma=0` hyperplane,
//!     `F\~ = F + \rho` is uniform given the claim — the round messages
//!     are exactly simulatable (perfect ZK at the round layer).
//! 4.  **Post-round commitment** `T2` covers the evaluation-carry limbs
//!     (functions of the challenge point), absorbed before the linear
//!     proof's challenge.
//! 5.  Four carrier linear relations over the concatenated slot vector,
//!     all proven by one multi-instance [`ZkLinearProof`]:
//!     - **decomposition**: `F\~ = F + \rho − p\cdot\kappa` (per-value
//!       carries, T1),
//!     - **zero-sum**: `\Sigma \rho = 0 (mod p)` (carry limbs in T1),
//!     - **F\~-evaluation**: `F\~(r) = \~v` (carry limbs in T2),
//!     - **\rho-evaluation**: `\rho(r) = \rho_r` (carry limbs in T2).
//!
//!     The carry limbs are **committed and norm-bounded** (16-bit
//!     limbs): a cheater trying to satisfy a relation with a wrong
//!     carry must find a short lattice vector — Module-SIS. (A *public*
//!     free-range carry would make the relations vacuous; committing
//!     the limbs is what makes the mod-q slack a SIS problem.)
//! 6.  The caller's evaluation claim `v = \~v − \rho_r` is checked
//!     against the caller-authenticated anchor.
//!
//! **Simulator** ([`zk_simulate`]): given only `(claim, num_vars)`,
//! sample `\rho_sim` uniform on the `\Sigma=0` hyperplane and `u`
//! uniform with `\Sigma u = claim`, set `F_sim := u − \rho_sim`, and
//! run the honest prover. Every public artifact — rounds, commitments,
//! finals, carries — matches the real law. Statistical KATs check this
//! empirically.

use crate::carrier::{compute_carry, field_limbs, MASK_LIMBS};
use crate::entropy::ShakeStream;
use crate::zk_linear::{ZkLinearProof, ZkLinearProofError};
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiPublicKey};
use lattice_commitment::linear_proof::LinearRelation;
use lattice_core::field::GOLDILOCKS_MODULUS;
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::{RingConfig, RingElement};

/// Carry limb width (committed, norm-bounded).
pub const CARRY_LIMB_BITS: u32 = 16;
/// Carry limbs per carry value.
pub const CARRY_LIMBS: usize = 2;

/// Public statement of the ZK sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkSumcheckStatement {
    /// Number of hypercube variables (vector length = 2^num_vars).
    pub num_vars: usize,
    /// The claimed sum `Σ F(x)`.
    pub claim: Goldilocks,
}

/// The proof.
#[derive(Clone)]
pub struct ZkSumcheckProof {
    /// Multilinear round messages of the blinded sumcheck: (t=0, t=1)
    /// values per round.
    pub rounds: Vec<[Goldilocks; 2]>,
    /// Pre-round commitment over `[F ‖ ρ ‖ F̃ ‖ carries]`.
    pub commitment: AjtaiCommitment,
    /// Post-round commitment over the evaluation-carry limbs.
    pub carry_commitment: AjtaiCommitment,
    /// Blinded final claim `F̃(r)`.
    pub blinded_final: Goldilocks,
    /// Mask evaluation `ρ(r)`.
    pub mask_final: Goldilocks,
    /// The ABDLOP proof of the four carrier relations.
    pub linear_proof: ZkLinearProof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZkSumcheckError {
    Linear(ZkLinearProofError),
    Ajtai(AjtaiError),
    Transcript(TranscriptError),
    Mle(lattice_core::mle::MleError),
    /// Round shapes or counts inconsistent with the statement.
    BadShape { expected: usize, got: usize },
    /// A round identity failed.
    RoundCheckFailed { round: usize },
    /// The anchor does not match the revealed claims.
    AnchorMismatch,
    /// The statement's slot requirements exceed the commitment key.
    TooManySlots { needed: usize, available: usize },
}

impl core::fmt::Display for ZkSumcheckError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ZkSumcheckError::Linear(e) => write!(f, "zk-linear: {e}"),
            ZkSumcheckError::Ajtai(e) => write!(f, "ajtai: {e:?}"),
            ZkSumcheckError::Transcript(e) => write!(f, "transcript: {e:?}"),
            ZkSumcheckError::Mle(e) => write!(f, "mle: {e:?}"),
            ZkSumcheckError::BadShape { expected, got } => write!(f, "shape {got} != {expected}"),
            ZkSumcheckError::RoundCheckFailed { round } => write!(f, "round {round} failed"),
            ZkSumcheckError::AnchorMismatch => write!(f, "anchor mismatch"),
            ZkSumcheckError::TooManySlots { needed, available } => {
                write!(f, "needs {needed} slots, key has {available}")
            }
        }
    }
}

/// T₁ slots for `n` values: F ‖ ρ ‖ F̃ limbs (9n), per-value carries
/// (n), zero-sum carry limbs (2).
pub fn required_slots(n: usize) -> usize {
    10 * n + 2
}

/// Split a mod-q carry residue into committed 16-bit limbs.
fn carry_split(kappa_mod_q: u32) -> [u32; CARRY_LIMBS] {
    let mask = (1u32 << CARRY_LIMB_BITS) - 1;
    [kappa_mod_q & mask, (kappa_mod_q >> CARRY_LIMB_BITS) & mask]
}

/// Sample a uniform masking MLE with zero sum (uniform conditioned on
/// the zero-sum constraint: sample uniform, subtract the mean).
fn sample_zero_sum_mask(num_vars: usize, stream: &mut ShakeStream) -> Vec<Goldilocks> {
    let n = 1usize << num_vars;
    let mut values = stream.next_fields(n);
    let mut sum = Goldilocks::ZERO;
    for v in &values {
        sum = sum.add(v);
    }
    let n_inv = Goldilocks::from_u64(n as u64)
        .inverse()
        .unwrap_or(Goldilocks::ZERO);
    let mean = sum.mul(&n_inv);
    for v in values.iter_mut() {
        *v = v.sub(&mean);
    }
    values
}

/// Per-value decomposition carries: `F̃_k = F_k + ρ_k − p·κ_k`.
fn decomposition_carries(f: &[Goldilocks], rho: &[Goldilocks]) -> Vec<u32> {
    f.iter()
        .zip(rho.iter())
        .map(|(a, b)| {
            // u128: the integer sum can exceed 2^64 before the mod-p
            // wrap (two near-p values); u64 wrapping would hide the
            // p-wrap for ~25% of uniform pairs.
            let sum = a.to_canonical_u64() as u128 + b.to_canonical_u64() as u128;
            if sum >= GOLDILOCKS_MODULUS as u128 {
                1
            } else {
                0
            }
        })
        .collect()
}

/// A constant-coefficient ring element.
fn const_elem(ring: &RingConfig, value: u32) -> RingElement {
    let mut coeffs = vec![0u32; ring.n()];
    coeffs[0] = value % ring.modulus.q;
    RingElement::from_coeffs(ring, coeffs)
}

/// The pre-round secret slot vector (T₁): `[F ‖ ρ ‖ F̃ ‖ κ_dec ‖
/// κ_Σ-limbs]`, one limb (or carry) per constant-coefficient slot.
fn pre_round_slots(
    ring: &RingConfig,
    f: &[Goldilocks],
    rho: &[Goldilocks],
    ftilde: &[Goldilocks],
) -> Vec<RingElement> {
    let n = f.len();
    let mut values: Vec<u32> = Vec::with_capacity(required_slots(n));
    for v in f {
        values.extend_from_slice(&field_limbs(v));
    }
    for v in rho {
        values.extend_from_slice(&field_limbs(v));
    }
    for v in ftilde {
        values.extend_from_slice(&field_limbs(v));
    }
    values.extend_from_slice(&decomposition_carries(f, rho));
    // Zero-sum carry (depends only on ρ — pre-round).
    let ones = vec![Goldilocks::ONE; n];
    let k_sum = compute_carry(&ones, rho, &Goldilocks::ZERO).unwrap_or(0);
    values.extend_from_slice(&carry_split(k_sum));
    values.into_iter().map(|v| const_elem(ring, v)).collect()
}

/// The post-round carry slot vector (T₂): `[κ_eq-limbs ‖ κ_r-limbs]`.
fn post_round_slots(
    ring: &RingConfig,
    eq: &[Goldilocks],
    ftilde: &[Goldilocks],
    rho: &[Goldilocks],
    blinded_final: &Goldilocks,
    mask_final: &Goldilocks,
) -> Vec<RingElement> {
    let k_eq = compute_carry(eq, ftilde, blinded_final).unwrap_or(0);
    let k_r = compute_carry(eq, rho, mask_final).unwrap_or(0);
    let mut values: Vec<u32> = Vec::with_capacity(2 * CARRY_LIMBS);
    values.extend_from_slice(&carry_split(k_eq));
    values.extend_from_slice(&carry_split(k_r));
    values.into_iter().map(|v| const_elem(ring, v)).collect()
}

/// Build the four carrier relations over the concatenated (padded)
/// slot vector: T₁ slots at `[0, m)`, T₂ slots at `[m, 2m)`.
///
/// T₁ layout: `[F (3n) ‖ ρ (3n) ‖ F̃ (3n) ‖ κ_dec (n) ‖ κ_Σ (2)]`.
/// T₂ layout: `[κ_eq (2) ‖ κ_r (2)]` (then zero padding to m).
fn build_relations(
    ring: &RingConfig,
    m: usize,
    n: usize,
    eq: &[Goldilocks],
    blinded_final: &Goldilocks,
    mask_final: &Goldilocks,
) -> Vec<LinearRelation> {
    let q = ring.modulus.q;
    let q64 = q as u64;
    let p_q = (GOLDILOCKS_MODULUS % q64) as u32;
    let total = 2 * m;
    let f_off = 0;
    let rho_off = 3 * n;
    let ftilde_off = 6 * n;
    let dec_off = 9 * n;
    let ksum_off = 10 * n;
    // T₂ offsets (post-m).
    let keq_off = m;
    let kr_off = m + CARRY_LIMBS;

    let rel = |coeff_spec: Vec<(usize, u32)>, target: u32| -> LinearRelation {
        let mut coeffs = vec![ring.zero(); total];
        for (pos, value) in coeff_spec {
            coeffs[pos] = const_elem(ring, value);
        }
        LinearRelation {
            coefficients: coeffs,
            target: const_elem(ring, target),
        }
    };

    // R1: Σ_k Σ_l 2^{22l}(F̃ − F − ρ)_{k,l} + Σ_k p·κ_k ≡ 0.
    let mut spec1: Vec<(usize, u32)> = Vec::with_capacity(10 * n);
    for k in 0..n {
        for l in 0..MASK_LIMBS {
            let shift = ((1u64 << (22 * l as u32)) % q64) as u32;
            spec1.push((ftilde_off + k * MASK_LIMBS + l, shift));
            spec1.push((f_off + k * MASK_LIMBS + l, (q - shift) % q));
            spec1.push((rho_off + k * MASK_LIMBS + l, (q - shift) % q));
        }
        spec1.push((dec_off + k, p_q));
    }
    let r1 = rel(spec1, 0);

    // R2: Σ 2^{22l} ρ-limbs − Σ_j p·2^{16j} κ_Σ-limbs ≡ 0.
    let mut spec2: Vec<(usize, u32)> = Vec::with_capacity(3 * n + 2);
    for k in 0..n {
        for l in 0..MASK_LIMBS {
            let shift = ((1u64 << (22 * l as u32)) % q64) as u32;
            spec2.push((rho_off + k * MASK_LIMBS + l, shift));
        }
    }
    for j in 0..CARRY_LIMBS {
        let coeff = ((p_q as u64 * ((1u64 << (CARRY_LIMB_BITS * j as u32)) % q64)) % q64) as u32;
        spec2.push((ksum_off + j, (q - coeff) % q));
    }
    let r2 = rel(spec2, 0);

    // R3: Σ eq·2^{22l} F̃-limbs − Σ_j p·2^{16j} κ_eq-limbs ≡ ṽ.
    let mut spec3: Vec<(usize, u32)> = Vec::with_capacity(3 * n + 2);
    for k in 0..n {
        let lam = eq[k].to_canonical_u64() % q64;
        for l in 0..MASK_LIMBS {
            let shift = ((1u64 << (22 * l as u32)) % q64) as u32;
            let coeff = ((lam * shift as u64) % q64) as u32;
            spec3.push((ftilde_off + k * MASK_LIMBS + l, coeff));
        }
    }
    for j in 0..CARRY_LIMBS {
        let coeff = ((p_q as u64 * ((1u64 << (CARRY_LIMB_BITS * j as u32)) % q64)) % q64) as u32;
        spec3.push((keq_off + j, (q - coeff) % q));
    }
    let target3 = (blinded_final.to_canonical_u64() % q64) as u32;
    let r3 = rel(spec3, target3);

    // R4: Σ eq·2^{22l} ρ-limbs − Σ_j p·2^{16j} κ_r-limbs ≡ ρ_r.
    let mut spec4: Vec<(usize, u32)> = Vec::with_capacity(3 * n + 2);
    for k in 0..n {
        let lam = eq[k].to_canonical_u64() % q64;
        for l in 0..MASK_LIMBS {
            let shift = ((1u64 << (22 * l as u32)) % q64) as u32;
            let coeff = ((lam * shift as u64) % q64) as u32;
            spec4.push((rho_off + k * MASK_LIMBS + l, coeff));
        }
    }
    for j in 0..CARRY_LIMBS {
        let coeff = ((p_q as u64 * ((1u64 << (CARRY_LIMB_BITS * j as u32)) % q64)) % q64) as u32;
        spec4.push((kr_off + j, (q - coeff) % q));
    }
    let target4 = (mask_final.to_canonical_u64() % q64) as u32;
    let r4 = rel(spec4, target4);

    vec![r1, r2, r3, r4]
}

/// Prove `Σ F(x) = claim` in zero knowledge.
///
/// Returns `(proof, r, v)` where `v = F(r)` is the caller's evaluation
/// anchor (authenticate it through the outer statement layer).
#[allow(clippy::too_many_lines)]
pub fn zk_prove(
    pk: &AjtaiPublicKey,
    f: &[Goldilocks],
    claim: Goldilocks,
    stream: &mut ShakeStream,
    transcript: &mut Transcript,
) -> Result<(ZkSumcheckProof, Vec<Goldilocks>, Goldilocks), ZkSumcheckError> {
    let num_vars = f.len().trailing_zeros() as usize;
    if (1usize << num_vars) != f.len() || num_vars == 0 {
        return Err(ZkSumcheckError::BadShape {
            expected: 0,
            got: f.len(),
        });
    }
    let n = f.len();
    let m = pk.params.m;
    let needed = required_slots(n);
    if needed > m {
        return Err(ZkSumcheckError::TooManySlots {
            needed,
            available: m,
        });
    }
    let ring = &pk.params.ring;

    // 1. Secret masking MLE and blinded vector.
    let rho = sample_zero_sum_mask(num_vars, stream);
    let ftilde: Vec<Goldilocks> = f.iter().zip(rho.iter()).map(|(a, b)| a.add(b)).collect();

    // 2. Pre-round commitment (Fiat-Shamir ordering: F̃ fixed before
    //    any challenge).
    let secret1 = pre_round_slots(ring, f, &rho, &ftilde);
    let padded1 = pk.pad_to_m(&secret1).map_err(ZkSumcheckError::Ajtai)?;
    let commitment = pk.commit(&padded1).map_err(ZkSumcheckError::Ajtai)?;

    // 3. Statement + T₁ binding, then the multilinear rounds over F̃.
    transcript
        .append_field(b"zk-sc-claim", &claim)
        .map_err(ZkSumcheckError::Transcript)?;
    transcript
        .append_message(b"zk-sc-vars", &(num_vars as u32).to_le_bytes())
        .map_err(ZkSumcheckError::Transcript)?;
    transcript
        .append_bytes(b"zk-sc-commit", &commitment.to_bytes())
        .map_err(ZkSumcheckError::Transcript)?;

    let mut ftilde_mle = DenseMle::new(ftilde.clone()).map_err(ZkSumcheckError::Mle)?;
    let mut rounds: Vec<[Goldilocks; 2]> = Vec::with_capacity(num_vars);
    let mut point: Vec<Goldilocks> = Vec::with_capacity(num_vars);
    let mut current = claim;
    for _ in 0..num_vars {
        let half = ftilde_mle.evaluations.len() / 2;
        let mut a = Goldilocks::ZERO;
        let mut b = Goldilocks::ZERO;
        for v in &ftilde_mle.evaluations[..half] {
            a = a.add(v);
        }
        for v in &ftilde_mle.evaluations[half..] {
            b = b.add(v);
        }
        transcript
            .append_field_slice(b"zk-sc-round", &[a, b])
            .map_err(ZkSumcheckError::Transcript)?;
        let r = transcript
            .challenge_field(b"zk-sc-challenge")
            .map_err(ZkSumcheckError::Transcript)?;
        rounds.push([a, b]);
        point.push(r);
        current = a.add(&b.sub(&a).mul(&r));
        ftilde_mle = ftilde_mle.fix_variables(&[r]).map_err(ZkSumcheckError::Mle)?;
    }
    let blinded_final = current; // = F̃(r)

    // 4. The caller's evaluation anchor and the mask final.
    let f_mle = DenseMle::new(f.to_vec()).map_err(ZkSumcheckError::Mle)?;
    let true_eval = f_mle.evaluate(&point).map_err(ZkSumcheckError::Mle)?;
    let mask_final = blinded_final.sub(&true_eval);
    transcript
        .append_field(b"zk-sc-blind-final", &blinded_final)
        .map_err(ZkSumcheckError::Transcript)?;
    transcript
        .append_field(b"zk-sc-mask-final", &mask_final)
        .map_err(ZkSumcheckError::Transcript)?;

    // 5. Post-round carry commitment (T₂), absorbed before the linear
    //    proof's challenge.
    let eq = DenseMle::eq_extension(&point).evaluations;
    let secret2 = post_round_slots(ring, &eq, &ftilde, &rho, &blinded_final, &mask_final);
    let padded2 = pk.pad_to_m(&secret2).map_err(ZkSumcheckError::Ajtai)?;
    let carry_commitment = pk.commit(&padded2).map_err(ZkSumcheckError::Ajtai)?;
    transcript
        .append_bytes(b"zk-sc-carry-commit", &carry_commitment.to_bytes())
        .map_err(ZkSumcheckError::Transcript)?;

    // 6. The four carrier relations, one multi-instance ABDLOP proof.
    let relations = build_relations(ring, m, n, &eq, &blinded_final, &mask_final);
    let linear_proof = ZkLinearProof::prove(
        pk,
        &relations,
        &[padded1, padded2],
        &[commitment.clone(), carry_commitment.clone()],
        stream,
    )
    .map_err(ZkSumcheckError::Linear)?;

    Ok((
        ZkSumcheckProof {
            rounds,
            commitment,
            carry_commitment,
            blinded_final,
            mask_final,
            linear_proof,
        },
        point,
        true_eval,
    ))
}

/// Verify a ZK sumcheck proof against the caller-authenticated anchor.
pub fn zk_verify(
    pk: &AjtaiPublicKey,
    statement: &ZkSumcheckStatement,
    anchor: &Goldilocks,
    proof: &ZkSumcheckProof,
    transcript: &mut Transcript,
) -> Result<Vec<Goldilocks>, ZkSumcheckError> {
    let num_vars = statement.num_vars;
    let n = 1usize << num_vars;
    let m = pk.params.m;
    if proof.rounds.len() != num_vars {
        return Err(ZkSumcheckError::BadShape {
            expected: num_vars,
            got: proof.rounds.len(),
        });
    }
    transcript
        .append_field(b"zk-sc-claim", &statement.claim)
        .map_err(ZkSumcheckError::Transcript)?;
    transcript
        .append_message(b"zk-sc-vars", &(num_vars as u32).to_le_bytes())
        .map_err(ZkSumcheckError::Transcript)?;
    transcript
        .append_bytes(b"zk-sc-commit", &proof.commitment.to_bytes())
        .map_err(ZkSumcheckError::Transcript)?;
    let mut current = statement.claim;
    let mut point = Vec::with_capacity(num_vars);
    for (i, [a, b]) in proof.rounds.iter().enumerate() {
        if a.add(b) != current {
            return Err(ZkSumcheckError::RoundCheckFailed { round: i });
        }
        transcript
            .append_field_slice(b"zk-sc-round", &[*a, *b])
            .map_err(ZkSumcheckError::Transcript)?;
        let r = transcript
            .challenge_field(b"zk-sc-challenge")
            .map_err(ZkSumcheckError::Transcript)?;
        current = a.add(&b.sub(a).mul(&r));
        point.push(r);
    }
    if current != proof.blinded_final {
        return Err(ZkSumcheckError::RoundCheckFailed { round: num_vars });
    }
    transcript
        .append_field(b"zk-sc-blind-final", &proof.blinded_final)
        .map_err(ZkSumcheckError::Transcript)?;
    transcript
        .append_field(b"zk-sc-mask-final", &proof.mask_final)
        .map_err(ZkSumcheckError::Transcript)?;
    transcript
        .append_bytes(b"zk-sc-carry-commit", &proof.carry_commitment.to_bytes())
        .map_err(ZkSumcheckError::Transcript)?;
    // Anchor: v = blinded − mask.
    if proof.blinded_final.sub(&proof.mask_final) != *anchor {
        return Err(ZkSumcheckError::AnchorMismatch);
    }
    // Rebuild the relations from public data and verify the ABDLOP
    // proof (which recomputes its Fiat-Shamir challenge over the full
    // statement, mask commitments, and relations).
    let eq = DenseMle::eq_extension(&point).evaluations;
    let relations = build_relations(
        &pk.params.ring,
        m,
        n,
        &eq,
        &proof.blinded_final,
        &proof.mask_final,
    );
    proof
        .linear_proof
        .verify(
            pk,
            &relations,
            &[proof.commitment.clone(), proof.carry_commitment.clone()],
        )
        .map_err(ZkSumcheckError::Linear)?;
    Ok(point)
}

/// The distributional simulator: produce a full proof of the statement
/// **given only `(claim, num_vars)`** — no witness.
///
/// In the real protocol the evaluation anchor `v = F(r)` is
/// transcript-emergent (the caller authenticates it afterwards through
/// its own statement layer), so the simulation needs no conditioning on
/// it. The simulator samples `ρ_sim` uniform on the `Σ=0` hyperplane
/// and `u` uniform with `Σu = claim`, sets `F_sim := u − ρ_sim`, and
/// runs the honest prover on `F_sim`. Every public artifact — rounds,
/// commitments, finals, carries, linear proof — matches the real law:
/// * rounds are partial sums of `F̃ = u` (uniform given the claim),
/// * `blinded`, `mask_final` are (uniform, uniform) pairs,
/// * `κ_Σ` is the carry of a genuinely uniform `Σ=0` mask (not zero),
/// * the commitments hide their blocks behind the ρ-block's entropy.
pub fn zk_simulate(
    pk: &AjtaiPublicKey,
    statement: &ZkSumcheckStatement,
    stream: &mut ShakeStream,
) -> Result<(ZkSumcheckProof, Vec<Goldilocks>, Goldilocks), ZkSumcheckError> {
    let n = 1usize << statement.num_vars;
    // ρ_sim: uniform on the Σ=0 hyperplane.
    let rho = sample_zero_sum_mask(statement.num_vars, stream);
    // u: uniform with the exact claimed sum.
    let mut u = stream.next_fields(n);
    let mut sum = Goldilocks::ZERO;
    for v in &u {
        sum = sum.add(v);
    }
    let delta = sum.sub(&statement.claim);
    if let Some(first) = u.first_mut() {
        *first = first.sub(&delta);
    }
    // F_sim := u − ρ_sim (so F̃ = F_sim + ρ_sim = u).
    let f_sim: Vec<Goldilocks> = u.iter().zip(rho.iter()).map(|(a, b)| a.sub(b)).collect();
    let mut transcript = Transcript::new_default(b"lzx-zk-sumcheck");
    zk_prove(pk, &f_sim, statement.claim, stream, &mut transcript)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::SecretSeed;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn setup_pk(num_vars: usize) -> AjtaiPublicKey {
        let n = 1usize << num_vars;
        let m = required_slots(n).max(16);
        let params = lattice_commitment::ajtai::AjtaiParams {
            ring: RingConfig::new(lattice_ring::Modulus32::Q_32, 4).ok().unwrap(),
            k: 2,
            m,
            norm_bound: 1 << 24,
        };
        AjtaiPublicKey::from_seed(params, [77u8; 32]).ok().unwrap()
    }

    fn random_values(n: usize, seed: &[u8]) -> Vec<Goldilocks> {
        let bytes = Transcript::xof(b"zk-sc-test", seed, n * 8);
        (0..n)
            .map(|i| {
                let mut a = [0u8; 16];
                a[..8].copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
                a[8..].copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
                Goldilocks::from_uniform_bytes(&a)
            })
            .collect()
    }

    fn sum_of(values: &[Goldilocks]) -> Goldilocks {
        values.iter().fold(Goldilocks::ZERO, |acc, v| acc.add(v))
    }

    #[test]
    fn prove_verify_roundtrip_and_exact_outputs() {
        for num_vars in [2usize, 3, 4] {
            let pk = setup_pk(num_vars);
            let f = random_values(1 << num_vars, format!("f-{num_vars}").as_bytes());
            let claim = sum_of(&f);
            let mut stream = ShakeStream::new(SecretSeed::from_kat_label(b"zk-sc-1"), b"mask");
            let mut t = Transcript::new_default(b"lzx-zk-sumcheck");
            let (proof, point, v) = match zk_prove(&pk, &f, claim, &mut stream, &mut t) {
                Ok(r) => r,
                Err(e) => panic!("prove err nv={num_vars}: {e:?}"),
            };
            // The revealed anchor is the exact MLE evaluation.
            let f_mle = DenseMle::new(f.clone()).ok().unwrap();
            assert_eq!(v, f_mle.evaluate(&point).ok().unwrap());

            let statement = ZkSumcheckStatement { num_vars, claim };
            let mut vt = Transcript::new_default(b"lzx-zk-sumcheck");
            let got_point = zk_verify(&pk, &statement, &v, &proof, &mut vt).ok().unwrap();
            assert_eq!(got_point, point);
        }
    }

    #[test]
    fn tampered_proofs_rejected() {
        let pk = setup_pk(3);
        let f = random_values(8, b"f-tamper");
        let claim = sum_of(&f);
        let mut stream = ShakeStream::new(SecretSeed::from_kat_label(b"zk-sc-2"), b"mask");
        let mut t = Transcript::new_default(b"lzx-zk-sumcheck");
        let (proof, _point, v) = zk_prove(&pk, &f, claim, &mut stream, &mut t).ok().unwrap();
        let statement = ZkSumcheckStatement {
            num_vars: 3,
            claim,
        };

        // Wrong anchor.
        let mut vt = Transcript::new_default(b"lzx-zk-sumcheck");
        assert!(zk_verify(&pk, &statement, &v.add(&fe(1)), &proof, &mut vt).is_err());

        // Wrong claim.
        let bad_statement = ZkSumcheckStatement {
            num_vars: 3,
            claim: claim.add(&fe(1)),
        };
        let mut vt = Transcript::new_default(b"lzx-zk-sumcheck");
        assert!(zk_verify(&pk, &bad_statement, &v, &proof, &mut vt).is_err());

        // Tampered round value.
        let mut bad = proof.clone();
        if let Some(r0) = bad.rounds.first_mut() {
            r0[0] = r0[0].add(&fe(1));
        }
        let mut vt = Transcript::new_default(b"lzx-zk-sumcheck");
        assert!(zk_verify(&pk, &statement, &v, &bad, &mut vt).is_err());

        // Tampered blinded final.
        let mut bad3 = proof.clone();
        bad3.blinded_final = bad3.blinded_final.add(&fe(1));
        let mut vt = Transcript::new_default(b"lzx-zk-sumcheck");
        assert!(zk_verify(&pk, &statement, &v, &bad3, &mut vt).is_err());

        // Tampered linear proof response.
        let mut bad4 = proof.clone();
        if let Some(z) = bad4.linear_proof.responses.first_mut() {
            if !z.is_empty() {
                let ring = &pk.params.ring;
                let mut coeffs = z[0].coeffs().to_vec();
                coeffs[0] = (coeffs[0] + 1) % ring.modulus.q;
                z[0] = RingElement::from_coeffs(ring, coeffs);
            }
        }
        let mut vt = Transcript::new_default(b"lzx-zk-sumcheck");
        assert!(zk_verify(&pk, &statement, &v, &bad4, &mut vt).is_err());
    }

    #[test]
    fn simulator_proves_valid_statements_without_witness() {
        let pk = setup_pk(3);
        // The simulator gets only the statement (claim, num_vars).
        let f = random_values(8, b"f-sim");
        let claim = sum_of(&f);
        let statement = ZkSumcheckStatement {
            num_vars: 3,
            claim,
        };
        let mut sim_stream = ShakeStream::new(SecretSeed::from_kat_label(b"zk-sc-sim"), b"sim");
        let (sim_proof, sim_point, sim_v) =
            zk_simulate(&pk, &statement, &mut sim_stream).ok().unwrap();
        // The simulated proof verifies against its own emergent anchor.
        let mut vt = Transcript::new_default(b"lzx-zk-sumcheck");
        assert!(zk_verify(&pk, &statement, &sim_v, &sim_proof, &mut vt).is_ok());
        assert_eq!(sim_point.len(), 3);
        // The simulator never saw the witness f.
    }

    /// Statistical KAT: real vs simulated round-message distributions.
    #[test]
    fn round_distributions_real_vs_simulated() {
        let pk = setup_pk(3);
        let f = random_values(8, b"f-kat");
        let claim = sum_of(&f);
        let statement = ZkSumcheckStatement {
            num_vars: 3,
            claim,
        };

        let bucket_bits = 6u32;
        let buckets = 1usize << bucket_bits;
        let mut real_hist = vec![0u64; buckets];
        let mut sim_hist = vec![0u64; buckets];
        let trials = 12u64;
        for i in 0..trials {
            let mut stream = ShakeStream::new(
                SecretSeed::from_kat_label(format!("real-kat-{i}").as_bytes()),
                b"mask",
            );
            let mut t = Transcript::new_default(b"lzx-zk-sumcheck");
            let (proof, _, _) = zk_prove(&pk, &f, claim, &mut stream, &mut t).ok().unwrap();
            for [a, b] in &proof.rounds {
                for v in [a, b] {
                    let idx = ((v.to_canonical_u64() >> (64 - bucket_bits)) as usize)
                        .min(buckets - 1);
                    real_hist[idx] += 1;
                }
            }
            let mut sstream = ShakeStream::new(
                SecretSeed::from_kat_label(format!("sim-kat-{i}").as_bytes()),
                b"sim",
            );
            let (sim, _, _) = zk_simulate(&pk, &statement, &mut sstream).ok().unwrap();
            for [a, b] in &sim.rounds {
                for v in [a, b] {
                    let idx = ((v.to_canonical_u64() >> (64 - bucket_bits)) as usize)
                        .min(buckets - 1);
                    sim_hist[idx] += 1;
                }
            }
        }
        // Two-sample chi-square (df = 63, critical at alpha=0.001
        // ~ 106.4).
        let mut chi2 = 0.0f64;
        for b in 0..buckets {
            let o1 = real_hist[b] as f64;
            let o2 = sim_hist[b] as f64;
            if o1 + o2 > 0.0 {
                chi2 += (o1 - o2).powi(2) / (o1 + o2);
            }
        }
        assert!(
            chi2 < 106.4,
            "round distributions differ: chi2={chi2:.2} real={real_hist:?} sim={sim_hist:?}"
        );
    }
}
