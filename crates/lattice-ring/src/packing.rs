//! Coefficient packing: embedding Goldilocks field elements and small
//! integers into ring coefficients and back (Akita `tensor.rs` /
//! Hachi carrier-embedding lineage).
//!
//! Two mechanisms:
//! * **Split packing** — a 64-bit Goldilocks residue becomes three
//!   balanced 22-bit limbs in consecutive coefficients. Limbs must fit the
//!   balanced coefficient range (|c| <= q/2 ≈ 2^30.6), so 22-bit limbs give
//!   comfortable margin; three of them cover 64 bits. This is the witness
//!   embedding used when committing MLE evaluation columns into R_q.
//! * **CRT packing** — several small values bounded by pairwise coprime
//!   moduli share a single coefficient slot via the Chinese Remainder
//!   Theorem (the "carrier embedding" that lets one ring coefficient
//!   carry multiple logical one-hot columns).

use crate::ring::{RingConfig, RingElement};

/// Limb width in bits for split packing (3 limbs cover 64 bits).
pub const LIMB_BITS: u32 = 22;
/// Number of limbs per Goldilocks value.
pub const LIMBS_PER_VALUE: usize = 3;

fn limb_mask() -> u64 {
    (1u64 << LIMB_BITS) - 1
}

/// Pack field elements into ring elements: each Goldilocks value becomes
/// three balanced 22-bit limbs. Returns ceil(3*len/n) ring elements
/// (at least one), zero-padded.
pub fn pack_field_elements(
    config: &RingConfig,
    values: &[lattice_core::Goldilocks],
) -> Vec<RingElement> {
    let n = config.n();
    let mut limbs: Vec<i64> = Vec::with_capacity(values.len() * LIMBS_PER_VALUE);
    for v in values {
        let raw = v.to_canonical_u64();
        limbs.push((raw & limb_mask()) as i64);
        limbs.push(((raw >> LIMB_BITS) & limb_mask()) as i64);
        limbs.push((raw >> (2 * LIMB_BITS)) as i64); // top limb < 2^20
    }
    let num_elems = (limbs.len().div_ceil(n)).max(1);
    let mut out = Vec::with_capacity(num_elems);
    for e in 0..num_elems {
        let start = e * n;
        let end = (start + n).min(limbs.len());
        let mut chunk: Vec<i64> = limbs[start..end].to_vec();
        chunk.resize(n, 0);
        out.push(RingElement::from_signed(config, &chunk));
    }
    out
}

/// Inverse of `pack_field_elements`: recovers packed values from ring
/// elements. Fails closed if any balanced limb exceeds the 22-bit packing
/// range — a larger limb would make the embedding ambiguous (soundness).
pub fn unpack_field_elements(
    config: &RingConfig,
    packed: &[RingElement],
) -> Result<Vec<lattice_core::Goldilocks>, PackingError> {
    let n = config.n();
    let q = config.modulus.q;
    let half = q / 2;
    let bound = 1i64 << LIMB_BITS;
    let mut limbs: Vec<i64> = Vec::with_capacity(packed.len() * n);
    for elem in packed {
        for &c in elem.coeffs() {
            let balanced = if c <= half {
                c as i64
            } else {
                c as i64 - q as i64
            };
            if balanced <= -bound || balanced >= bound {
                return Err(PackingError::LimbOutOfRange { value: balanced });
            }
            limbs.push(balanced);
        }
    }
    let mut out = Vec::with_capacity(limbs.len() / LIMBS_PER_VALUE);
    for group in limbs.chunks(LIMBS_PER_VALUE) {
        if group.len() < LIMBS_PER_VALUE {
            break; // trailing padding
        }
        let mut raw = 0u64;
        raw |= (group[0] as u64) & limb_mask();
        raw |= ((group[1] as u64) & limb_mask()) << LIMB_BITS;
        raw |= (group[2] as u64) << (2 * LIMB_BITS);
        out.push(lattice_core::Goldilocks::from_u64(raw));
    }
    Ok(out)
}

/// CRT-pack small values into one coefficient per group: values `v_i`
/// bounded by pairwise coprime `moduli[i]` (product < q) combine into a
/// single residue `V` with `V ≡ v_i (mod moduli[i])`.
pub fn crt_pack(
    config: &RingConfig,
    values: &[u64],
    moduli: &[u64],
) -> Result<Vec<RingElement>, PackingError> {
    if moduli.is_empty() {
        return Err(PackingError::LengthMismatch { expected: 1, got: 0 });
    }
    let mut product = 1u64;
    for &m in moduli {
        if m == 0 || m >= config.modulus.q as u64 {
            return Err(PackingError::ModulusTooLarge { modulus: m });
        }
        product = product.checked_mul(m).ok_or(PackingError::ModulusTooLarge { modulus: m })?;
        if product >= config.modulus.q as u64 {
            return Err(PackingError::ModulusTooLarge { modulus: product });
        }
    }
    let k = moduli.len();
    let n = config.n();
    let num_elems = values.len().div_ceil(k).max(1);
    let mut out = Vec::with_capacity(num_elems);
    for e in 0..num_elems {
        // Incremental CRT: fold the value group into a single residue.
        let start = e * k;
        let mut combined = 0u64;
        let mut prod_so_far = 1u64;
        for (slot, &m) in moduli.iter().enumerate().take(k.min(n)) {
            let idx = start + slot;
            if idx >= values.len() {
                break;
            }
            if values[idx] >= m {
                return Err(PackingError::ValueOutOfRange {
                    value: values[idx],
                    bound: m,
                });
            }
            // Solve x ≡ v (mod m), x ≡ combined (mod prod_so_far):
            // x = combined + prod_so_far * t,
            // t ≡ (v - combined) * prod_so_far^{-1} (mod m).
            let v = values[idx];
            let diff = (v + m - (combined % m)) % m;
            let inv = mod_inverse(prod_so_far % m, m)
                .ok_or(PackingError::NotCoprime { modulus: m })?;
            let t = (diff * inv) % m;
            combined += prod_so_far * t;
            prod_so_far *= m;
        }
        let mut group_coeffs = vec![0u32; n];
        group_coeffs[0] = config.modulus.reduce_u64(combined);
        out.push(RingElement::from_coeffs(config, group_coeffs));
    }
    Ok(out)
}

/// Unpack CRT-packed values: each packed coefficient is reduced modulo
/// every modulus to recover its group.
pub fn crt_unpack(
    config: &RingConfig,
    packed: &[RingElement],
    moduli: &[u64],
    expected_values: usize,
) -> Result<Vec<u64>, PackingError> {
    let mut out = Vec::with_capacity(expected_values);
    'outer: for elem in packed {
        let combined = elem.coeff(0) as u64;
        for &m in moduli {
            if out.len() >= expected_values {
                break 'outer;
            }
            out.push(combined % m);
        }
    }
    let _ = config;
    if out.len() != expected_values {
        return Err(PackingError::LengthMismatch {
            expected: expected_values,
            got: out.len(),
        });
    }
    Ok(out)
}

/// Extended-Euclid inverse for small coprime moduli.
fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    if m <= 1 {
        return None;
    }
    let (mut old_r, mut r) = (a % m, m);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        let tmp = old_r - q * r;
        old_r = r;
        r = tmp;
        let tmp = old_s - q as i128 * s;
        old_s = s;
        s = tmp;
    }
    if old_r != 1 {
        return None;
    }
    let inv = ((old_s % m as i128) + m as i128) % m as i128;
    Some(inv as u64)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackingError {
    LengthMismatch { expected: usize, got: usize },
    LimbOutOfRange { value: i64 },
    ModulusTooLarge { modulus: u64 },
    ValueOutOfRange { value: u64, bound: u64 },
    NotCoprime { modulus: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulus::Modulus32;

    fn cfg(log_n: u32) -> RingConfig {
        RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap()
    }

    #[test]
    fn split_pack_roundtrip() {
        let c = cfg(4); // n = 16 -> 5 values per element (16/3)
        let values: Vec<lattice_core::Goldilocks> = (0..40u64)
            .map(|i| {
                lattice_core::Goldilocks::from_u64(
                    i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i),
                )
            })
            .collect();
        let packed = pack_field_elements(&c, &values);
        assert_eq!(packed.len(), 8); // 120 limbs / 16 = 7.5 -> 8
        let unpacked = unpack_field_elements(&c, &packed).ok().unwrap();
        assert_eq!(unpacked[..values.len()], values[..]);
    }

    #[test]
    fn split_pack_all_field_values() {
        // Values at modulus edges must survive the 3x22-bit split.
        let c = cfg(3);
        let p = lattice_core::field::GOLDILOCKS_MODULUS;
        let values = vec![
            lattice_core::Goldilocks::ZERO,
            lattice_core::Goldilocks::ONE,
            lattice_core::Goldilocks::from_u64(p - 1),
            lattice_core::Goldilocks::from_u64(p / 2),
            lattice_core::Goldilocks::from_u64(0x7FFF_FFFF_0000_0000),
        ];
        let packed = pack_field_elements(&c, &values);
        let unpacked = unpack_field_elements(&c, &packed).ok().unwrap();
        assert_eq!(unpacked[..values.len()], values[..]);
    }

    #[test]
    fn split_pack_rejects_oversized_limb() {
        let c = cfg(3);
        // A coefficient with balanced value 2^22 or more is not a valid
        // packed limb (proof must reject, not wrap).
        let bad = RingElement::from_signed(&c, &[1 << 22, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            unpack_field_elements(&c, &[bad]).err(),
            Some(PackingError::LimbOutOfRange { value: 1 << 22 })
        );
    }

    #[test]
    fn crt_pack_roundtrip() {
        let c = cfg(4);
        let moduli = [7u64, 11, 13, 17]; // product 17017 < q
        let values = [3u64, 5, 9, 15, 0, 6, 10, 2, 1, 1];
        let packed = crt_pack(&c, &values, &moduli).ok().unwrap();
        let unpacked = crt_unpack(&c, &packed, &moduli, values.len()).ok().unwrap();
        assert_eq!(unpacked, values.to_vec());
    }

    #[test]
    fn crt_pack_rejects_out_of_range() {
        let c = cfg(4);
        let moduli = [7u64, 11];
        assert_eq!(
            crt_pack(&c, &[7, 0], &moduli).err(),
            Some(PackingError::ValueOutOfRange { value: 7, bound: 7 })
        );
        let big = [1_000_000u64, 1_000_003, 1_000_019, 999_999];
        assert!(crt_pack(&c, &[1, 1, 1, 1], &big).is_err());
    }

    #[test]
    fn mod_inverse_small() {
        assert_eq!(mod_inverse(3, 7), Some(5)); // 3*5 = 15 = 1 mod 7
        assert_eq!(mod_inverse(2, 4), None);
        assert_eq!(mod_inverse(1, 11), Some(1));
    }
}
