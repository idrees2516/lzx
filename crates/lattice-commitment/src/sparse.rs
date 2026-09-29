//! Zero-skipping commitment inputs and bit-packed one-hot columns
//! (Wave 6.8, `NEXT_STEPS.md` §2.8 — the papers' "0s are free" doctrine).
//!
//! Twist & Shout §2.9.2 and Neo/SuperNeo's pay-per-bit commitment make
//! sparse witnesses cost proportionally to their *support*, not their
//! *length*: zero ring elements contribute nothing to `A·s`, so the MAC
//! loop can skip them entirely (exact — `A·0 = 0` — and homomorphism
//! preserving). [`AjtaiPublicKey::commit`] already skips zero elements;
//! this module provides the packing layer that makes the doctrine pay:
//!
//! * [`pack_bits`] — a bit vector packed **31 bits per ring coefficient**
//!   (vs. one 22-bit limb per bit in the value-oriented packing — a ~31x
//!   density gain for one-hot indicator columns; each packed coefficient
//!   is < 2^31 < q/2 so norms stay exactly analyzable),
//! * [`pack_onehot_column`] — the one-hot indicator of a length-`L` column
//!   with value `v` at position `p`, packed at 31 bits/coefficient and
//!   padded to `m` ring elements (the commit skips the zero padding),
//! * [`hamming_weight`] — the pay-per-bit cost model: the number of ring
//!   elements the MAC loop actually touches.

use crate::ajtai::{AjtaiCommitment, AjtaiError, AjtaiPublicKey};
use lattice_ring::{RingConfig, RingElement};

/// Bits per packed ring coefficient: 31 (< log2(q/2) for the Q32 class,
/// so every packed coefficient is a small positive integer).
pub const BITS_PER_COEFF: usize = 31;

/// Pack a bit vector into ring elements at 31 bits per coefficient.
/// `bits.len()` ≤ 31 · (number of elements returned); the caller pads.
pub fn pack_bits(ring: &RingConfig, bits: &[u8]) -> Vec<RingElement> {
    let n = ring.n();
    let elems = bits.len().div_ceil(BITS_PER_COEFF * n);
    let mut out = Vec::with_capacity(elems);
    for e in 0..elems {
        let mut coeffs = vec![0u32; n];
        for (c, coeff) in coeffs.iter_mut().enumerate() {
            let base = (e * n + c) * BITS_PER_COEFF;
            let mut v: u64 = 0;
            for b in 0..BITS_PER_COEFF {
                if let Some(bit) = bits.get(base + b) {
                    if *bit != 0 {
                        v |= 1 << b;
                    }
                }
            }
            *coeff = v as u32; // < 2^31 < q/2
        }
        out.push(RingElement::from_coeffs(ring, coeffs));
    }
    if out.is_empty() {
        out.push(ring.zero());
    }
    out
}

/// Unpack ring elements produced by [`pack_bits`] back into bits.
pub fn unpack_bits(ring: &RingConfig, elems: &[RingElement], num_bits: usize) -> Vec<u8> {
    let mut bits = vec![0u8; num_bits];
    for (idx, bit) in bits.iter_mut().enumerate() {
        let e = idx / (BITS_PER_COEFF * ring.n());
        let within = idx % (BITS_PER_COEFF * ring.n());
        let c = within / BITS_PER_COEFF;
        let b = within % BITS_PER_COEFF;
        if let Some(elem) = elems.get(e) {
            *bit = ((elem.coeff(c) >> b) & 1) as u8;
        }
    }
    bits
}

/// A one-hot column: length `len`, a single value `value` at `position`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnehotColumn {
    pub len: usize,
    pub position: usize,
    pub value: u32,
}

impl OnehotColumn {
    /// The indicator bits (1 at `position`, 0 elsewhere).
    pub fn indicator(&self) -> Vec<u8> {
        let mut bits = vec![0u8; self.len];
        if self.position < self.len {
            bits[self.position] = 1;
        }
        bits
    }

    /// Pack into ring elements: the bit-packed indicator column padded to
    /// exactly `m` ring elements (zeros at the tail — skipped by the MAC
    /// loop). The Hamming-weight-one structure is preserved coefficient-wise:
    /// exactly one packed coefficient is a power of two, the rest are zero.
    pub fn pack(&self, ring: &RingConfig, m: usize) -> Result<Vec<RingElement>, AjtaiError> {
        let bits = self.indicator();
        let mut elems = pack_bits(ring, &bits);
        if elems.len() > m {
            return Err(AjtaiError::DimensionMismatch {
                expected: m,
                got: elems.len(),
            });
        }
        while elems.len() < m {
            elems.push(ring.zero());
        }
        Ok(elems)
    }
}

/// The pay-per-bit cost model: how many ring elements the commitment MAC
/// loop actually touches (zero elements are skipped for free).
pub fn mac_work_units(s: &[RingElement]) -> usize {
    s.iter().filter(|e| !e.is_zero()).count()
}

/// Commit a set of one-hot columns (each padded to m ring elements) and
/// return the packed elements alongside the commitment — the pay-per-bit
/// entry point used by Twist-style one-hot constraint systems.
pub fn commit_onehot_columns(
    pk: &AjtaiPublicKey,
    columns: &[OnehotColumn],
) -> Result<(AjtaiCommitment, Vec<RingElement>), AjtaiError> {
    let ring = &pk.params.ring;
    let m = pk.params.m;
    // Pack column-major: element j aggregates the j-th packed element of
    // every column (a column-count-weighted one-hot blend — the sum of
    // one-hot indicators is a small-norm "counts" vector, exactly what the
    // Hamming-weight sumchecks consume).
    let mut s: Vec<Vec<u32>> = vec![vec![0u32; ring.n()]; m];
    for col in columns {
        let packed = col.pack(ring, m)?;
        let q = ring.modulus;
        for (j, elem) in packed.iter().enumerate() {
            for (c, acc) in s[j].iter_mut().enumerate() {
                *acc = q.add(*acc, elem.coeff(c));
            }
        }
    }
    let s: Vec<RingElement> = s.into_iter().map(|coeffs| RingElement::from_coeffs(ring, coeffs)).collect();
    let commitment = pk.commit(&s)?;
    Ok((commitment, s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_ring::Modulus32;

    fn test_pk(log_n: u32, k: usize, m: usize) -> AjtaiPublicKey {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams {
            ring,
            k,
            m,
            norm_bound: 1 << 23,
        };
        AjtaiPublicKey::from_seed(params, [77u8; 32]).ok().unwrap()
    }

    #[test]
    fn pack_bits_roundtrip() {
        let ring = RingConfig::new(Modulus32::Q_32, 4).ok().unwrap();
        for len in [1usize, 31, 32, 100, 31 * 16 /* exactly 1 elem */, 31 * 16 + 7] {
            let bits: Vec<u8> = (0..len).map(|i| ((i * 7 + 3) % 5 == 0) as u8).collect();
            let packed = pack_bits(&ring, &bits);
            let unpacked = unpack_bits(&ring, &packed, len);
            assert_eq!(unpacked, bits, "len = {len}");
            // Every packed coefficient is < 2^31 (small-norm invariant).
            for e in &packed {
                assert!(e.infinity_norm() < (1 << 31));
            }
        }
        // Empty bit vector → single zero element.
        let empty = pack_bits(&ring, &[]);
        assert_eq!(empty.len(), 1);
        assert!(empty[0].is_zero());
    }

    #[test]
    fn packing_density_is_31_bits_per_coefficient() {
        // One ring element (n = 16) holds 31·16 = 496 bits — vs. one bit
        // per element in the value-oriented packing (a 496x density gain
        // at this ring size).
        let ring = RingConfig::new(Modulus32::Q_32, 4).ok().unwrap();
        let bits = vec![1u8; 31 * 16];
        let packed = pack_bits(&ring, &bits);
        assert_eq!(packed.len(), 1);
        assert_eq!(BITS_PER_COEFF, 31);
        // 497 bits spill into a second element.
        let spill = pack_bits(&ring, &vec![1u8; 31 * 16 + 1]);
        assert_eq!(spill.len(), 2);
    }

    #[test]
    fn onehot_column_packs_with_single_nonzero_coefficient() {
        let ring = RingConfig::new(Modulus32::Q_32, 4).ok().unwrap();
        let col = OnehotColumn {
            len: 500,
            position: 137,
            value: 9,
        };
        let packed = col.pack(&ring, 16).ok().unwrap();
        assert_eq!(packed.len(), 16);
        // Exactly one packed coefficient is a power of two (the indicator
        // bit); all others are zero.
        let nonzeros: Vec<u32> = packed
            .iter()
            .flat_map(|e| e.coeffs().iter().copied())
            .filter(|c| *c != 0)
            .collect();
        assert_eq!(nonzeros.len(), 1, "one-hot must have exactly one nonzero");
        assert!(nonzeros[0].is_power_of_two());
        // The tail padding is all zeros (skipped by the MAC loop).
        assert!(packed[15].is_zero());
        // MAC work units: exactly 1 of 16.
        assert_eq!(mac_work_units(&packed), 1);
    }

    #[test]
    fn commit_onehot_columns_pays_per_bit() {
        // Sparse columns cost proportionally to their packed support.
        let pk = test_pk(5, 2, 8);
        let ring = &pk.params.ring;
        // 3 columns at positions inside the first packed element.
        let cols: Vec<OnehotColumn> = [3usize, 100, 200]
            .iter()
            .map(|p| OnehotColumn {
                len: 256,
                position: *p,
                value: 5,
            })
            .collect();
        let (t, s) = commit_onehot_columns(&pk, &cols).ok().unwrap();
        // All three land in element 0 (positions < 31·32): 1 work unit.
        assert_eq!(mac_work_units(&s), 1);
        assert!(pk.verify_opening(&t, &s).is_ok());
        // The packed sum has exactly 3 one-bits (Hamming weight 3).
        let total_bits: u32 = s[0]
            .coeffs()
            .iter()
            .map(|c| c.count_ones())
            .sum();
        assert_eq!(total_bits, 3);

        // Spread across ELEMENTS: with n = 32, one element holds
        // 31·32 = 992 bits; positions one bank apart land in different
        // elements (8 non-zero elements → 8 MAC work units).
        let spread: Vec<OnehotColumn> = (0..8usize)
            .map(|i| OnehotColumn {
                len: 992 * 8,
                position: i * 992,
                value: 1,
            })
            .collect();
        let (t2, s2) = commit_onehot_columns(&pk, &spread).ok().unwrap();
        assert_eq!(mac_work_units(&s2), 8);
        assert!(pk.verify_opening(&t2, &s2).is_ok());
        // Homomorphism: committing columns separately and adding equals
        // the combined commitment.
        let mut acc = pk.commit(&vec![ring.zero(); pk.params.m]).ok().unwrap();
        for c in &spread {
            let single = [c.clone()];
            let (tc, sc) = commit_onehot_columns(&pk, &single).ok().unwrap();
            assert!(pk.verify_opening(&tc, &sc).is_ok());
            for (r, r2) in acc.rows.iter_mut().zip(tc.rows.iter()) {
                *r = r.add(r2).ok().unwrap();
            }
        }
        assert_eq!(acc, t2);
    }

    #[test]
    fn onehot_pack_rejects_overflow() {
        let ring = RingConfig::new(Modulus32::Q_32, 2).ok().unwrap();
        // n = 4 coefficients · 31 bits = 124 bits per element; a column of
        // length 400 needs 4 elements; m = 2 is too small.
        let col = OnehotColumn {
            len: 400,
            position: 3,
            value: 1,
        };
        assert!(matches!(
            col.pack(&ring, 2),
            Err(AjtaiError::DimensionMismatch { .. })
        ));
    }
}
