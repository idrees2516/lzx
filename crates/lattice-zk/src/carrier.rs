//! The CRT-carrier linear relation: bridging field-level claims to exact
//! mod-q ring relations.
//!
//! Problem: the ZK sumcheck's accumulated mask contribution is a
//! *field* identity `Δ = Σ_k λ_k · μ_k (mod p)` over Goldilocks (p ≈ 2^64)
//! with public coefficients `λ` (interpolation values at the Fiat-Shamir
//! challenges) and secret uniform masks `μ`. The binding proof lives
//! over `R_q` (q ≈ 2^31.6) — a different modulus. A naive mod-q
//! reduction of the identity is wrong whenever the integer sum wraps
//! past p.
//!
//! Solution (the "carrier" embedding, in the Akita/Hachi lineage):
//! decompose every secret mask into 22-bit limbs and track the number
//! of p-wraps as an explicit committed carry. The exact statement is
//! (all congruences mod q):
//!
//! ```text
//! Σ_k Σ_l (λ_k · 2^{22l} mod q) · μ_{k,l}  −  Σ_j (p · 2^{16j} mod q) · κ_j
//!     ≡ Δ (mod q)
//! ```
//!
//! where `μ_{k,l}` are the mask limbs (`μ_k = Σ_l 2^{22l} μ_{k,l}` as an
//! exact integer), and `κ = (T_int − Δ)/p mod q` with
//! `T_int = Σ_k λ_k μ_k` over the integers. Because
//! `Σ_l (λ_k 2^{22l} mod q)(μ_{k,l}) ≡ λ_k μ_k (mod q)` term-by-term and
//! `T_int = Δ + p·κ_true`, the relation holds exactly for honest provers
//! and binds the field-level claim.
//!
//! Every secret (mask limb, carry limb) occupies the **constant
//! coefficient** of a dedicated ring slot, so the ring products in the
//! linear relation are exact scalar products with zero "garbage"
//! coefficients — publishing the relation target reveals only `Δ mod q`,
//! which is already public.

use lattice_commitment::linear_proof::LinearRelation;
use lattice_core::field::GOLDILOCKS_MODULUS;
use lattice_core::Goldilocks;
use lattice_ring::{RingConfig, RingElement};

/// Limb width for mask decomposition.
pub const MASK_LIMB_BITS: u32 = 22;
/// Carry limbs: the mod-q carry residue splits into 16-bit limbs.
pub const CARRY_LIMB_BITS: u32 = 16;
/// Number of mask limbs per field value.
pub const MASK_LIMBS: usize = 3;
/// Number of carry limbs.
pub const CARRY_LIMBS: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CarrierError {
    /// The secret vector does not fit the relation geometry.
    SlotOverflow { slots: usize },
    /// Ring-level relation evaluation mismatch (honest-construction bug).
    IdentityFailed,
    /// Carry derivation produced a non-integer quotient (internal bug).
    CarryNonIntegral,
}

/// p·q as u128 — the modulus for exact integer accumulation.
fn modulus_pq() -> u128 {
    GOLDILOCKS_MODULUS as u128 * (lattice_ring::Modulus32::Q_32.q as u128)
}

/// Decompose a canonical field value into three 22-bit limbs.
pub fn field_limbs(value: &Goldilocks) -> [u32; MASK_LIMBS] {
    let raw = value.to_canonical_u64();
    let mask = (1u64 << MASK_LIMB_BITS) - 1;
    [
        (raw & mask) as u32,
        ((raw >> MASK_LIMB_BITS) & mask) as u32,
        (raw >> (2 * MASK_LIMB_BITS)) as u32,
    ]
}

/// Split the mod-q carry residue into 16-bit limbs.
pub fn carry_limbs(kappa_mod_q: u32) -> [u32; CARRY_LIMBS] {
    let mask = (1u32 << CARRY_LIMB_BITS) - 1;
    [kappa_mod_q & mask, (kappa_mod_q >> CARRY_LIMB_BITS) & mask]
}

/// The exact integer carry: `κ_true mod q` where
/// `T_int = Δ + p·κ_true`, computed via accumulation mod (p·q).
///
/// Returns `Err` on non-integrality (impossible for honest inputs — the
/// check exists so a malformed construction fails loudly).
pub fn compute_carry(
    lambdas: &[Goldilocks],
    masks: &[Goldilocks],
    delta_field: &Goldilocks,
) -> Result<u32, CarrierError> {
    let m = modulus_pq();
    let mut acc: u128 = 0u128;
    for (lam, mu) in lambdas.iter().zip(masks.iter()) {
        // λ·μ as an exact integer reduced mod p·q (u128 product fits).
        let term = (lam.to_canonical_u64() as u128) * (mu.to_canonical_u64() as u128);
        acc = (acc + term % m) % m;
    }
    let delta = delta_field.to_canonical_u64() as u128;
    // X = (acc − Δ) mod (p·q) = p · (κ_true mod q).
    let x = (acc + m - delta % m) % m;
    if x % (GOLDILOCKS_MODULUS as u128) != 0 {
        return Err(CarrierError::CarryNonIntegral);
    }
    Ok((x / GOLDILOCKS_MODULUS as u128) as u32)
}

/// The public relation coefficients for a mask limb at limb-index `l`.
pub fn mask_coefficient(q: u32, lambda: &Goldilocks, limb: usize) -> u32 {
    let q64 = q as u64;
    let lam = lambda.to_canonical_u64() % q64;
    let shift = match limb {
        0 => 1u64,
        1 => 1u64 << MASK_LIMB_BITS,
        _ => 1u64 << (2 * MASK_LIMB_BITS),
    };
    ((lam * (shift % q64)) % q64) as u32
}

/// The public relation coefficient for carry limb `j` (negated: the
/// relation subtracts p·κ).
pub fn carry_coefficient(q: u32, limb: usize) -> u32 {
    let q64 = q as u64;
    let p_mod_q = GOLDILOCKS_MODULUS % q64;
    let shift = if limb == 0 { 1u64 } else { 1u64 << CARRY_LIMB_BITS };
    q.wrapping_sub(((p_mod_q * (shift % q64)) % q64) as u32) % q
}

/// A constant-coefficient ring element.
pub fn const_elem(ring: &RingConfig, value: u32) -> RingElement {
    let mut coeffs = vec![0u32; ring.n()];
    coeffs[0] = value % ring.modulus.q;
    RingElement::from_coeffs(ring, coeffs)
}

/// The full carrier relation for one linear combination.
///
/// `lambdas[k]` multiplies mask `masks[k]`; `delta_field` is the
/// field-level target. Returns the relation (public) and the secret
/// slot vector (constant-coefficient limbs: 3 per mask + 2 carry).
///
/// The secret vector is what gets Ajtai-committed; the relation is what
/// the ZK linear proof binds.
pub struct CarrierRelation {
    /// Public relation for the ZK linear proof.
    pub relation: LinearRelation,
    /// Secret constant-coefficient slots (mask limbs then carry limbs).
    pub secret_slots: Vec<RingElement>,
}

#[allow(clippy::too_many_arguments)]
pub fn build_carrier(
    ring: &RingConfig,
    lambdas: &[Goldilocks],
    masks: &[Goldilocks],
    delta_field: &Goldilocks,
) -> Result<CarrierRelation, CarrierError> {
    if lambdas.len() != masks.len() {
        return Err(CarrierError::IdentityFailed);
    }
    let q = ring.modulus.q;
    let kappa = compute_carry(lambdas, masks, delta_field)?;
    let k_limb = carry_limbs(kappa);

    // Relation coefficients: mask limbs then carry limbs.
    let mut coefficients = Vec::with_capacity(lambdas.len() * MASK_LIMBS + CARRY_LIMBS);
    for lam in lambdas {
        for l in 0..MASK_LIMBS {
            coefficients.push(const_elem(ring, mask_coefficient(q, lam, l)));
        }
    }
    for j in 0..CARRY_LIMBS {
        coefficients.push(const_elem(ring, carry_coefficient(q, j)));
    }

    // Secret slots.
    let mut secret_slots = Vec::with_capacity(coefficients.len());
    for mu in masks {
        for limb in field_limbs(mu) {
            secret_slots.push(const_elem(ring, limb));
        }
    }
    for limb in k_limb {
        secret_slots.push(const_elem(ring, limb));
    }

    // Target: Δ mod q at the constant coefficient.
    let target = const_elem(ring, (delta_field.to_canonical_u64() % q as u64) as u32);

    let relation = LinearRelation {
        coefficients,
        target,
    };

    // Self-check: the exact identity must hold for the constructed
    // secret (fail loudly on any derivation bug).
    if relation.evaluate(&secret_slots).map_err(|_| CarrierError::IdentityFailed)? != relation.target
    {
        return Err(CarrierError::IdentityFailed);
    }
    Ok(CarrierRelation {
        relation,
        secret_slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_core::transcript::Transcript;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn random_fields(n: usize, seed: &[u8]) -> Vec<Goldilocks> {
        let bytes = Transcript::xof(b"carrier-rand", seed, n * 8);
        (0..n)
            .map(|i| {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
                Goldilocks::from_uniform_bytes(&{
                    let mut a = [0u8; 16];
                    a[..8].copy_from_slice(&arr);
                    a[8..].copy_from_slice(&arr);
                    a
                })
            })
            .collect()
    }

    #[test]
    fn limb_decomposition_exact() {
        // 3×22-bit limbs recompose every canonical value exactly.
        let values = random_fields(500, b"limb");
        for v in &values {
            let limbs = field_limbs(v);
            let raw = v.to_canonical_u64();
            let recomposed = (limbs[0] as u64)
                + ((limbs[1] as u64) << MASK_LIMB_BITS)
                + ((limbs[2] as u64) << (2 * MASK_LIMB_BITS));
            assert_eq!(recomposed, raw);
        }
    }

    #[test]
    fn carrier_identity_exact_for_random_instances() {
        let ring = RingConfig::new(lattice_ring::Modulus32::Q_32, 4).ok().unwrap();
        for trial in 0..8u64 {
            let lambdas = random_fields(6, format!("lam-{trial}").as_bytes());
            let masks = random_fields(6, format!("mu-{trial}").as_bytes());
            // Field-level target: Δ = Σ λ·μ mod p.
            let mut delta = Goldilocks::ZERO;
            for (l, m) in lambdas.iter().zip(masks.iter()) {
                delta = delta.add(&l.mul(m));
            }
            let carrier = build_carrier(&ring, &lambdas, &masks, &delta).ok().unwrap();
            // The relation evaluates exactly to the target on the secret.
            let got = carrier
                .relation
                .evaluate(&carrier.secret_slots)
                .ok()
                .unwrap();
            assert_eq!(got, carrier.relation.target);
            // And the target encodes Δ mod q.
            let expect = const_elem(
                &ring,
                (delta.to_canonical_u64() % ring.modulus.q as u64) as u32,
            );
            assert_eq!(carrier.relation.target, expect);
        }
    }

    #[test]
    fn carrier_detects_wrong_delta() {
        let ring = RingConfig::new(lattice_ring::Modulus32::Q_32, 4).ok().unwrap();
        let lambdas = random_fields(4, b"lam-w");
        let masks = random_fields(4, b"mu-w");
        let mut delta = Goldilocks::ZERO;
        for (l, m) in lambdas.iter().zip(masks.iter()) {
            delta = delta.add(&l.mul(m));
        }
        let wrong = delta.add(&fe(1));
        // Building against the wrong target fails the internal identity
        // check (the carry cannot absorb an off-by-p error).
        assert!(build_carrier(&ring, &lambdas, &masks, &wrong).is_err());
    }

    #[test]
    fn secret_slots_are_norm_bounded() {
        // Mask limbs < 2^22, carry limbs < 2^16: the SIS norm budget of
        // the ZK linear proof accommodates them.
        let ring = RingConfig::new(lattice_ring::Modulus32::Q_32, 4).ok().unwrap();
        let lambdas = random_fields(8, b"lam-n");
        let masks = random_fields(8, b"mu-n");
        let mut delta = Goldilocks::ZERO;
        for (l, m) in lambdas.iter().zip(masks.iter()) {
            delta = delta.add(&l.mul(m));
        }
        let carrier = build_carrier(&ring, &lambdas, &masks, &delta).ok().unwrap();
        for (i, slot) in carrier.secret_slots.iter().enumerate() {
            let bound = if i < lambdas.len() * MASK_LIMBS {
                1u32 << MASK_LIMB_BITS
            } else {
                1u32 << CARRY_LIMB_BITS
            };
            assert!(slot.infinity_norm() < bound);
        }
    }
}
