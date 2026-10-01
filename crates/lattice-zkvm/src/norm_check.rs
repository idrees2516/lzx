//! The zkVM claim ledger's **norm-check module** — the TTRP shortness
//! proof (ePrint 2026/2146, `Π_TTRP`) wired in as the bundle responses'
//! norm layer, replacing the JL-projection lineage.
//!
//! # What it replaces
//!
//! The codebase's prior norm-check layers: the LaBRADOR-style **JL
//! projection** (256 ±1 signed sums of all witness coefficients —
//! `O(λ·m̄r)` verifier row processing, approximate with a `β/√30`
//! slack) and the ledger's **digit-gadget revelation** (`Θ(m·φ)` digits
//! whose range checks imply the bound). The TTRP module proves the
//! same statement — `‖cf(v)‖₂ ≤ B` for the Ajtai-committed response —
//! with a **few-kilobyte** proof and the tensor-structured verifier
//! (`O(k·µ₁·c²·d)` ring operations, never materializing the `m̄r`-long
//! rows).
//!
//! # The wiring (the claim-ledger pattern)
//!
//! 1. **Statement digest** — binds the outer relation: the bundle's
//!    Ajtai commitment, the ring, the response length, the coefficient
//!    bound, and the carrier sum-check's terminal claim (so the norm
//!    proof is about *this* opening).
//! 2. **`prove`** — runs `Π_TTRP` over the response vector with
//!    parameters sized for the bundle ring (`φ = 64`, `Q_32`).
//! 3. **`verify`** — returns the **eval claim**
//!    `mle(v)(conj(r)) = w_r` — the terminal claim the caller's outer
//!    relation must absorb (the paper §5.2's LaBRADOR-substitution
//!    pattern: the claim is a linear relation on the committed
//!    response).
//! 4. Two documented bridges consume the claim:
//!    * [`eval_claim_against_response`] — the clear-mode cross-check
//!      (the reconstructed response *is* the TTRP witness: a cheating
//!      prover that ran TTRP on a different, shorter vector fails it);
//!    * [`eval_claim_relation`] — the ABDLOP-layer
//!      [`LinearRelation`] adapter for constraint absorption
//!      (`⟨tensor(conj(r)), v⟩ = w_r`, exactly the paper's
//!      inner-product decomposition).
//!
//! # The honest scope note
//!
//! In the clear mode the response digits are still transmitted for the
//! `A·s = t` Ajtai binding and the flat-MLE reconstruction; the TTRP
//! module replaces the *norm semantics* (the digit range checks become
//! redundant) and adds the eval-claim binding. The fully digit-free
//! opening additionally needs the compact-mode linear-functional
//! bridge for `f(r_sc)` — the documented follow-up, not claimed here.

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiPublicKey};
use lattice_commitment::linear_proof::LinearRelation;
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_ring::{RingConfig, RingElement};
use lattice_ttrp::cores::TtrpParams;
use lattice_ttrp::protocol::{self, TtrpProof, TtrpVerified};
use lattice_ttrp::projection::conj;

use crate::ledger::{BaseClaim, BundleOpening, BundleProver, LedgerError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormCheckError {
    Ttrp(String),
    Ledger(LedgerError),
    /// The eval claim does not match the response (the TTRP witness
    /// differs from the opened response — binding failure).
    EvalClaimMismatch,
    /// Response length not a power of two / empty.
    BadLength { got: usize },
}

impl core::fmt::Display for NormCheckError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NormCheckError::Ttrp(e) => write!(f, "ttrp: {e}"),
            NormCheckError::Ledger(e) => write!(f, "ledger: {e:?}"),
            NormCheckError::EvalClaimMismatch => {
                write!(f, "TTRP eval claim does not match the response")
            }
            NormCheckError::BadLength { got } => {
                write!(f, "response length {got} not a power of two")
            }
        }
    }
}

impl From<LedgerError> for NormCheckError {
    fn from(e: LedgerError) -> Self {
        NormCheckError::Ledger(e)
    }
}

/// The TTRP-backed norm proof for one bundle opening.
#[derive(Clone)]
pub struct TtrpNormProof {
    pub params: TtrpParams,
    /// The Euclidean coefficient bound `B` (`‖cf(s)‖₂ ≤ B`).
    pub bound_b: u64,
    pub proof: TtrpProof,
}

/// The TTRP witness: the response padded with zero ring elements to a
/// power-of-two length ≥ 2 (the MLE needs ≥ 1 variable). Zero padding
/// preserves the norm statement and is deterministic on both sides.
fn ttrp_witness(ring: &RingConfig, response: &[RingElement]) -> Vec<RingElement> {
    let padded_len = response.len().next_power_of_two().max(2);
    let mut out = response.to_vec();
    while out.len() < padded_len {
        out.push(ring.zero());
    }
    out
}

/// Size the TTRP parameters for a bundle response: the ring is fixed
/// (`Q_32`, `φ = 64`), `ν = log₂ m`, and the core tiling satisfies
/// `ℓ·µ₁ = ν`, `ℓ·µ₂ = φ_log`.
fn bundle_params(m: usize) -> Result<TtrpParams, NormCheckError> {
    if m == 0 || !m.is_power_of_two() {
        return Err(NormCheckError::BadLength { got: m });
    }
    let nu = m.trailing_zeros() as usize;
    const PHI_LOG: usize = 6; // Q_32 bundle ring, n = 64.
    // ℓ divides both ν and φ_log when possible; ℓ = 1 always works.
    let ell = if nu % 2 == 0 { 2 } else { 1 };
    debug_assert!(nu >= 1);
    let mu1 = nu / ell;
    let mu2 = PHI_LOG / ell;
    Ok(TtrpParams {
        phi_log: PHI_LOG as u32,
        nu,
        ell,
        mu1,
        mu2,
        c: 4,
        k: 16,
        k1: 2,
    })
}

/// The worst-case Euclidean bound on the response coefficients:
/// `B = ⌈√(m·φ)⌉ · bound` (every coefficient ≤ the infinity bound).
fn worst_case_bound(m: usize, phi: usize, coeff_bound: u32) -> u64 {
    let total = (m * phi) as f64;
    let b = total.sqrt().ceil() as u64;
    b.saturating_mul(coeff_bound as u64).max(1)
}

/// The statement digest binding the norm proof to its ledger context:
/// the Ajtai commitment, the ring identity, the response shape, the
/// bound, and the carrier's terminal claim.
pub fn norm_context_digest(
    commitment: &AjtaiCommitment,
    m: usize,
    coeff_bound: u32,
    carrier_final_claim: &Goldilocks,
) -> [u8; 32] {
    let mut msg = Vec::with_capacity(64 + 8 + 8 + 8);
    msg.extend_from_slice(&commitment.to_bytes());
    msg.extend_from_slice(&m.to_le_bytes());
    msg.extend_from_slice(&coeff_bound.to_le_bytes());
    msg.extend_from_slice(&carrier_final_claim.to_bytes());
    Transcript::hash_domain(b"zkvm-ttrp-norm", &msg)
}

/// Prove `‖cf(response)‖₂ ≤ B` with `Π_TTRP`, bound to the ledger
/// context digest.
pub fn prove_norm_check_ttrp(
    ring: &RingConfig,
    response: &[RingElement],
    coeff_bound: u32,
    context: &[u8; 32],
    transcript: &mut Transcript,
) -> Result<TtrpNormProof, NormCheckError> {
    let witness = ttrp_witness(ring, response);
    let m = witness.len();
    let params = bundle_params(m)?;
    let bound_b = worst_case_bound(m, ring.n(), coeff_bound);
    let stmt = protocol::TtrpStatement {
        params: params.clone(),
        ring: ring.clone(),
        bound_b,
        statement_digest: *context,
    };
    let proof = protocol::prove(&stmt, &witness, transcript)
        .map_err(|e| NormCheckError::Ttrp(e.to_string()))?;
    Ok(TtrpNormProof {
        params,
        bound_b,
        proof,
    })
}

/// Verify the TTRP norm proof; returns the **eval claim**
/// `mle(v)(conj(r)) = w_r` for the caller's outer relation.
pub fn verify_norm_check_ttrp(
    ring: &RingConfig,
    response_len: usize,
    coeff_bound: u32,
    context: &[u8; 32],
    proof: &TtrpNormProof,
    transcript: &mut Transcript,
) -> Result<TtrpVerified, NormCheckError> {
    let m = response_len.next_power_of_two().max(2);
    let params = bundle_params(m)?;
    let bound_b = worst_case_bound(m, ring.n(), coeff_bound);
    if proof.params != params || proof.bound_b != bound_b {
        return Err(NormCheckError::Ttrp("parameter mismatch".into()));
    }
    let stmt = protocol::TtrpStatement {
        params,
        ring: ring.clone(),
        bound_b,
        statement_digest: *context,
    };
    protocol::verify(&stmt, &proof.proof, transcript)
        .map_err(|e| NormCheckError::Ttrp(e.to_string()))
}

/// The clear-mode eval-claim cross-check: the reconstructed response
/// *is* the TTRP witness — `mle(response)(conj(r)) = w_r`. Without
/// this, a cheating prover could run the shortness proof on a
/// different (short) vector while the digits reveal a long one.
pub fn eval_claim_against_response(
    ring: &RingConfig,
    verified: &TtrpVerified,
    response: &[RingElement],
) -> Result<(), NormCheckError> {
    let witness = ttrp_witness(ring, response);
    let conj_r: Vec<RingElement> = verified.challenges.iter().map(conj).collect();
    let got = protocol::mle_eval_ring(&witness, &conj_r)
        .map_err(|e| NormCheckError::Ttrp(e.to_string()))?;
    if got != verified.w_r {
        return Err(NormCheckError::EvalClaimMismatch);
    }
    Ok(())
}

/// The ABDLOP-layer adapter: the eval claim as a **linear relation** on
/// the committed response — `⟨tensor(conj(r)), v⟩ = w_r` with
/// `tensor(x)` the structured expansion tensor `(1−x₁,x₁) ⊗ ⋯` (the
/// paper §5.2's inner-product decomposition, one row per response
/// slot). Feed it to `lattice_commitment::linear_proof` for
/// constraint-absorption without response revelation.
pub fn eval_claim_relation(
    ring: &RingConfig,
    verified: &TtrpVerified,
    response_len: usize,
) -> Result<LinearRelation, NormCheckError> {
    let m = response_len.next_power_of_two().max(2);
    let nu = m.trailing_zeros() as usize;
    if verified.challenges.len() != nu {
        return Err(NormCheckError::Ttrp(format!(
            "challenge arity {} != log2(m) = {nu}",
            verified.challenges.len()
        )));
    }
    // tensor(conj(r)) ∈ R^m: eq-weights over the conj challenge bits.
    let conj_r: Vec<RingElement> = verified.challenges.iter().map(conj).collect();
    let one = ring.one();
    let mut weights: Vec<RingElement> = vec![one.clone()];
    for rj in conj_r.iter().take(nu) {
        let one_minus = one.sub(rj).map_err(|e| NormCheckError::Ttrp(format!("{e:?}")))?;
        let mut next = Vec::with_capacity(weights.len() * 2);
        for w in &weights {
            next.push(w.mul(&one_minus).map_err(NormCheckError::ring)?);
            next.push(w.mul(rj).map_err(NormCheckError::ring)?);
        }
        weights = next;
    }
    Ok(LinearRelation {
        coefficients: weights,
        target: verified.w_r.clone(),
    })
}

impl NormCheckError {
    fn ring(e: lattice_ring::RingError) -> Self {
        NormCheckError::Ttrp(format!("ring: {e:?}"))
    }
}

// ---------------------------------------------------------------------------
// The ledger integration: the TTRP bundle-opening mode
// ---------------------------------------------------------------------------

/// A bundle opening with the TTRP norm proof (the carrier plus the
/// digit transmission for the `A·s = t` binding and flat
/// reconstruction — the norm semantics itself is the KB-scale `Π_TTRP`,
/// and the eval claim cross-binds the TTRP witness to the revealed
/// response).
#[derive(Clone)]
pub struct TtrpBundleOpening {
    pub carrier: BundleOpening,
    pub norm: TtrpNormProof,
}

impl BundleProver {
    /// The TTRP-mode opening: carrier + digits (binding/reconstruction)
    /// + the `Π_TTRP` shortness proof + the eval claim.
    pub fn prove_opening_ttrp(
        &self,
        claims: &[BaseClaim],
        transcript: &mut Transcript,
    ) -> Result<TtrpBundleOpening, NormCheckError> {
        let carrier = self.prove_opening(claims, transcript)?;
        // The norm context binds the commitment, shape, bound, and the
        // carrier's terminal claim.
        let final_claim = carrier.carrier_terminal_claim();
        let context = norm_context_digest(
            &self.commitment,
            self.s.len(),
            self.coeff_bound(),
            &final_claim,
        );
        let norm =
            prove_norm_check_ttrp(&self.ring, &self.s, self.coeff_bound(), &context, transcript)?;
        Ok(TtrpBundleOpening { carrier, norm })
    }

    /// The coefficient bound of this bundle's packing.
    fn coeff_bound(&self) -> u32 {
        if self.is_bits {
            crate::ledger::BITS_NORM_BOUND
        } else {
            crate::ledger::VALUES_NORM_BOUND
        }
    }
}

/// Verify a TTRP-mode opening: the carrier replay (which reconstructs
/// the response from the digits and checks `A·s = t` and the final
/// identity), the `Π_TTRP` verification, and the eval-claim
/// cross-check against the reconstructed response.
#[allow(clippy::too_many_arguments)]
pub fn verify_bundle_opening_ttrp(
    pk: &AjtaiPublicKey,
    commitment: &AjtaiCommitment,
    layout: &[crate::ledger::BundleLayoutEntry],
    claims: &[BaseClaim],
    opening: &TtrpBundleOpening,
    is_bits: bool,
    transcript: &mut Transcript,
) -> Result<(), NormCheckError> {
    // 1. Carrier + digit reconstruction + A·s = t + flat identity.
    crate::ledger::verify_bundle_opening(
        pk,
        commitment,
        layout,
        claims,
        &opening.carrier,
        is_bits,
        transcript,
    )?;
    // 2. Re-derive the statement digest (public data only).
    let final_claim = opening.carrier.carrier_terminal_claim();
    let m = opening.carrier.m;
    let coeff_bound = if is_bits {
        crate::ledger::BITS_NORM_BOUND
    } else {
        crate::ledger::VALUES_NORM_BOUND
    };
    let context = norm_context_digest(commitment, m, coeff_bound, &final_claim);
    // 3. Π_TTRP verification -> the eval claim.
    let verified = verify_norm_check_ttrp(&pk.params.ring, m, coeff_bound, &context, &opening.norm, transcript)?;
    // 4. The eval-claim cross-check: reconstruct the response from the
    //    digits (the same path verify_bundle_opening used) and check
    //    mle(response)(conj(r)) = w_r.
    let response = reconstruct_response(&pk.params.ring, &opening.carrier)?;
    eval_claim_against_response(&pk.params.ring, &verified, &response)?;
    Ok(())
}

/// Reconstruct the response vector from the transmitted digits (the
/// shared clear-mode path).
fn reconstruct_response(
    ring: &RingConfig,
    opening: &BundleOpening,
) -> Result<Vec<RingElement>, NormCheckError> {
    let gadget = lattice_core::decomposition::GadgetDecomposition {
        base: opening.gadget_base,
        num_digits: opening.gadget_digits,
    };
    let per_elem = ring.n() * gadget.num_digits;
    if opening.digits.len() % per_elem != 0 {
        return Err(NormCheckError::Ttrp("digit count mismatch".into()));
    }
    let m = opening.digits.len() / per_elem;
    let mut s: Vec<RingElement> = Vec::with_capacity(m);
    for e in 0..m {
        let mut coeffs = vec![0u32; ring.n()];
        for (c, coeff) in coeffs.iter_mut().enumerate() {
            let mut acc: i128 = 0;
            let mut power: i128 = 1;
            for d in 0..gadget.num_digits {
                let digit = opening.digits[e * per_elem + c * gadget.num_digits + d] as i64;
                acc += digit as i128 * power;
                power *= gadget.base as i128;
            }
            *coeff = acc.rem_euclid(ring.modulus.q as i128) as u32;
        }
        s.push(RingElement::from_coeffs(ring, coeffs));
    }
    Ok(s)
}

/// The carrier's terminal claim (the PCS binding value) — extracted
/// from the opening for the statement digest. Implemented as an
/// extension on the ledger's `BundleOpening` (kept local to avoid
/// changing the shared type's API).
trait CarrierTerminalClaim {
    fn carrier_terminal_claim(&self) -> Goldilocks;
}

impl CarrierTerminalClaim for BundleOpening {
    fn carrier_terminal_claim(&self) -> Goldilocks {
        // The carrier's final round message's last entry is the degree-2
        // univariate's value at 2; the terminal claim the caller's PCS
        // must bind is the interpolated value at the (replayed)
        // challenge — for the digest we bind the LAST round message
        // (deterministic public data that pins the carrier).
        self.carrier
            .rounds
            .last()
            .and_then(|r| r.last().copied())
            .unwrap_or(Goldilocks::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{bits_bundle_commit, Factor, BITS_NORM_BOUND};
    use lattice_core::{DenseMle, Goldilocks};
    use lattice_ring::Modulus32;

    fn bit_tensor(log_vars: usize, seed: u64) -> DenseMle {
        // Hash-based bits (the multiply-xor form collapses to all-zero
        // for odd seeds — a degeneracy the TTRP context binding would
        // silently paper over).
        let n = 1usize << log_vars;
        let evals: Vec<Goldilocks> = (0..n)
            .map(|i| {
                let h = Transcript::hash_domain(
                    b"norm-test",
                    &[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat(),
                );
                Goldilocks::from_u64((h[0] & 1) as u64)
            })
            .collect();
        DenseMle { num_vars: log_vars, evaluations: evals }
    }

    fn tensor_point(nvars: usize, seed: u64) -> Vec<Goldilocks> {
        (0..nvars)
            .map(|i| Goldilocks::from_u64(((seed >> (i % 8)) ^ ((i * 37) as u64)) % 97))
            .collect()
    }

    /// Happy path: TTRP-mode opening verifies; the eval claim matches.
    #[test]
    fn ttrp_opening_roundtrip() {
        let t0 = bit_tensor(5, 7);
        let t1 = bit_tensor(4, 11);
        let prover = bits_bundle_commit(
            &[(Factor::InstrBits, t0.clone()), (Factor::DigitBits { inst: 0 }, t1)],
            [9u8; 32],
        )
        .ok()
        .unwrap();
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
                    let entry = prover
                        .layout
                        .iter()
                        .find(|e| e.factor == Factor::DigitBits { inst: 0 })
                        .unwrap();
                    crate::ledger::flat_point(entry, &tensor_point(4, 8), prover.flat.num_vars)
                })
                .ok()
                .unwrap(),
        };
        let mut t = Transcript::new_default(b"ttrp-bundle");
        let opening = prover
            .prove_opening_ttrp(&[c0.clone(), c1.clone()], &mut t)
            .ok()
            .unwrap();
        let mut t2 = Transcript::new_default(b"ttrp-bundle");
        assert!(verify_bundle_opening_ttrp(
            &prover.pk,
            &prover.commitment,
            &prover.layout,
            &[c0.clone(), c1.clone()],
            &opening,
            true,
            &mut t2,
        )
        .is_ok());
    }

    /// A tampered digit breaks the eval-claim cross-check (or the
    /// carrier): the TTRP witness no longer matches the response.
    #[test]
    fn tampered_digits_rejected() {
        let t0 = bit_tensor(4, 21);
        let prover =
            bits_bundle_commit(&[(Factor::InstrBits, t0.clone())], [5u8; 32]).ok().unwrap();
        let pt = tensor_point(4, 13);
        let claim = BaseClaim {
            factor: Factor::InstrBits,
            point: pt.clone(),
            value: t0.evaluate(&pt).ok().unwrap(),
        };
        let mut t = Transcript::new_default(b"ttrp-tamper");
        let mut opening = prover
            .prove_opening_ttrp(std::slice::from_ref(&claim), &mut t)
            .ok()
            .unwrap();
        // Corrupt one digit: the reconstructed response changes; the
        // eval claim (about the honest response) no longer matches —
        // unless A·s = t fails first.
        if let Some(d) = opening.carrier.digits.first_mut() {
            *d = d.wrapping_add(1);
        }
        let mut t2 = Transcript::new_default(b"ttrp-tamper");
        assert!(verify_bundle_opening_ttrp(
            &prover.pk,
            &prover.commitment,
            &prover.layout,
            std::slice::from_ref(&claim),
            &opening,
            true,
            &mut t2,
        )
        .is_err());
    }

    /// A tampered TTRP proof (the y0 projection) is rejected.
    #[test]
    fn tampered_ttrp_proof_rejected() {
        let t0 = bit_tensor(4, 31);
        let prover =
            bits_bundle_commit(&[(Factor::InstrBits, t0.clone())], [7u8; 32]).ok().unwrap();
        let pt = tensor_point(4, 17);
        let claim = BaseClaim {
            factor: Factor::InstrBits,
            point: pt.clone(),
            value: t0.evaluate(&pt).ok().unwrap(),
        };
        let mut t = Transcript::new_default(b"ttrp-proof");
        let mut opening = prover
            .prove_opening_ttrp(std::slice::from_ref(&claim), &mut t)
            .ok()
            .unwrap();
        // Corrupt the projection: the norm check or the sumcheck
        // terminal breaks.
        if let Some(y) = opening.norm.proof.y0.first_mut() {
            *y = (*y + 1) % prover.ring.modulus.q;
        }
        let mut t2 = Transcript::new_default(b"ttrp-proof");
        assert!(verify_bundle_opening_ttrp(
            &prover.pk,
            &prover.commitment,
            &prover.layout,
            std::slice::from_ref(&claim),
            &opening,
            true,
            &mut t2,
        )
        .is_err());
    }

    /// A wrong context digest (a different commitment) is rejected.
    #[test]
    fn wrong_context_rejected() {
        let t0 = bit_tensor(4, 41);
        let prover =
            bits_bundle_commit(&[(Factor::InstrBits, t0.clone())], [9u8; 32]).ok().unwrap();
        // A different bundle (different tensor AND seed) => a different
        // commitment => a different context digest.
        let other_tensor = bit_tensor(4, 99);
        let other = bits_bundle_commit(&[(Factor::InstrBits, other_tensor)], [10u8; 32])
            .ok()
            .unwrap();
        let pt = tensor_point(4, 23);
        let claim = BaseClaim {
            factor: Factor::InstrBits,
            point: pt.clone(),
            value: t0.evaluate(&pt).ok().unwrap(),
        };
        let mut t = Transcript::new_default(b"ttrp-ctx");
        let opening = prover
            .prove_opening_ttrp(std::slice::from_ref(&claim), &mut t)
            .ok()
            .unwrap();
        // Verify against the OTHER commitment's context: the statement
        // digest mismatch breaks the TTRP verification (the cores are
        // derived from the digest).
        let final_claim = opening.carrier.carrier_terminal_claim();
        let ctx = norm_context_digest(&other.commitment, opening.carrier.m, BITS_NORM_BOUND, &final_claim);

        let mut t3 = Transcript::new_default(b"ttrp-ctx");
        let verified = verify_norm_check_ttrp(
            &prover.ring,
            opening.carrier.m,
            BITS_NORM_BOUND,
            &ctx,
            &opening.norm,
            &mut t3,
        );
        assert!(verified.is_err());
    }

    /// The standalone module: the eval-claim relation adapter is
    /// consistent with the response (⟨tensor(conj(r)), v⟩ = w_r).
    #[test]
    fn eval_claim_relation_consistent() {
        let ring = RingConfig::new(Modulus32::Q_32, 6).ok().unwrap();
        // A small deterministic response (m = 8 ring elements).
        let m = 8usize;
        let mut st = 0x1234_5678_u64 | 1;
        let mut response = Vec::with_capacity(m);
        for _ in 0..m {
            let mut coeffs = Vec::with_capacity(ring.n());
            for _ in 0..ring.n() {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                coeffs.push((st % 9) as u32);
            }
            response.push(RingElement::from_coeffs(&ring, coeffs));
        }
        // Prove + verify standalone.
        let ctx = Transcript::hash_domain(b"rel-test", b"ctx");
        let bound = worst_case_bound(m, ring.n(), 8);
        let params = bundle_params(m).ok().unwrap();
        let stmt = protocol::TtrpStatement {
            params: params.clone(),
            ring: ring.clone(),
            bound_b: bound,
            statement_digest: ctx,
        };
        let mut tp = Transcript::new_default(b"rel");
        let proof = protocol::prove(&stmt, &response, &mut tp).ok().unwrap();
        let mut tv = Transcript::new_default(b"rel");
        let verified = protocol::verify(&stmt, &proof, &mut tv).ok().unwrap();
        // (a) the direct cross-check passes on the honest response.
        assert!(eval_claim_against_response(&ring, &verified, &response).is_ok());
        // (b) a wrong response fails.
        let mut bad = response.clone();
        bad[0] = bad[0].add(&ring.one()).ok().unwrap();
        assert!(matches!(
            eval_claim_against_response(&ring, &verified, &bad),
            Err(NormCheckError::EvalClaimMismatch)
        ));
        // (c) the relation adapter evaluates to w_r on the response.
        let rel = eval_claim_relation(&ring, &verified, m).ok().unwrap();
        let got = rel.evaluate(&response).ok().unwrap();
        assert_eq!(got, verified.w_r);
    }
}

