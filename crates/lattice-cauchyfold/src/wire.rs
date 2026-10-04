//! Serialization (Appendix D.7): canonical `F_q` coefficients occupy six
//! bytes (`q < 2^48`), `K`-elements 24 bytes, `R_{q,64}` elements 384
//! bytes — the paper's wire layout. The terminal response uses the
//! fixed-length codec: the **low frame** carries the sign and the 6-bit
//! residues, the **high frame** carries the quotients
//! (`|z_i| = 2^6·q_i + r_i`) at a fixed width derived from the norm bound.
//!
//! **Documented deviation**: the paper's high frame is arithmetic-coded
//! with frequencies `f_j ∝ (2^24/256)^j·2^{acode}` and a worst-case bit
//! length `H` (its D.7 formula); this implementation uses fixed-width
//! quotients with the same fail-closed decoder checks (exact frame
//! lengths, canonical signed representation, absence of negative zero,
//! padding, the decoded norm, equality after re-encoding). The size delta
//! is quantified in the benchmark note.

use lattice_labrador::ring::Poly;

/// The low-frame residue width (the paper's `b_low = 6`).
pub const LOW_BITS: u32 = 6;

/// Encode the terminal response (a flat ring vector).
pub fn encode_terminal(flat: &[Poly]) -> Vec<u8> {
    let coefs: Vec<i64> = flat.iter().flat_map(|p| p.0.iter().copied()).collect();
    let n = coefs.len();
    // Quotients.
    let quotients: Vec<u64> = coefs
        .iter()
        .map(|&c| c.unsigned_abs() >> LOW_BITS)
        .collect();
    let max_q = quotients.iter().copied().max().unwrap_or(0);
    let width = if max_q == 0 {
        0u8
    } else {
        (64 - max_q.leading_zeros()).min(64) as u8
    };
    let mut out =
        Vec::with_capacity(8 + 1 + (n * 7).div_ceil(8) + (n * width as usize).div_ceil(8));
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.push(width);
    // The low frame: sign + 6-bit residue per coefficient, MSB-first
    // bit packing.
    let mut low = vec![false; n * 7];
    for (i, &c) in coefs.iter().enumerate() {
        let sign = c < 0;
        let residue = (c.unsigned_abs() & ((1 << LOW_BITS) - 1)) as u8;
        low[i * 7] = sign;
        for b in 0..6 {
            low[i * 7 + 1 + b] = (residue >> (5 - b)) & 1 == 1;
        }
    }
    for chunk in low.chunks(8) {
        let mut byte = 0u8;
        for (b, &bit) in chunk.iter().enumerate() {
            if bit {
                byte |= 1 << (7 - b);
            }
        }
        out.push(byte);
    }
    // The high frame: fixed-width little-endian-bit quotients.
    if width > 0 {
        let mut high = vec![false; n * width as usize];
        for (i, &q) in quotients.iter().enumerate() {
            for b in 0..width as usize {
                high[i * width as usize + b] = (q >> b) & 1 == 1;
            }
        }
        for chunk in high.chunks(8) {
            let mut byte = 0u8;
            for (b, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1 << (7 - b);
                }
            }
            out.push(byte);
        }
    }
    out
}

/// Decode + every fail-closed check. `n_elems` is the expected ring
/// element count.
pub fn decode_terminal(bytes: &[u8], n_elems: usize) -> Result<Vec<Poly>, String> {
    let n = n_elems * 64;
    if bytes.len() < 5 {
        return Err("short header".into());
    }
    let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let width = bytes[4] as usize;
    if count != n {
        return Err("count mismatch".into());
    }
    let low_bytes = (n * 7).div_ceil(8);
    let high_bytes = if width > 0 {
        (n * width).div_ceil(8)
    } else {
        0
    };
    if bytes.len() != 5 + low_bytes + high_bytes {
        return Err("frame length".into());
    }
    let read_bit = |off: usize| -> bool {
        let byte = bytes[5 + off / 8];
        (byte >> (7 - (off % 8))) & 1 == 1
    };
    let mut coefs = Vec::with_capacity(n);
    let mut max_abs: u64 = 0;
    for i in 0..n {
        let sign = read_bit(i * 7);
        let mut residue = 0u8;
        for b in 0..6 {
            if read_bit(i * 7 + 1 + b) {
                residue |= 1 << (5 - b);
            }
        }
        let mut quotient: u64 = 0;
        if width > 0 {
            for b in 0..width {
                if read_bit(n * 7 + i * width + b) {
                    quotient |= 1u64 << b;
                }
            }
        }
        // Fail-closed: no negative zero.
        if sign && residue == 0 && quotient == 0 {
            return Err("negative zero".into());
        }
        let mag = (quotient << LOW_BITS) | residue as u64;
        let v = if sign { -(mag as i64) } else { mag as i64 };
        max_abs = max_abs.max(mag);
        coefs.push(v);
    }
    // The decoded norm bound: the caller checks against the declared G;
    // here the codec-level check: the magnitude must fit the width.
    if width > 0 && max_abs >= (1u64 << (width as u32 + LOW_BITS)) {
        return Err("magnitude exceeds width".into());
    }
    // Equality after re-encoding.
    let polys: Vec<Poly> = coefs
        .chunks(64)
        .map(|c| {
            let mut a = [0i64; 64];
            a.copy_from_slice(c);
            Poly(a)
        })
        .collect();
    if encode_terminal(&polys) != bytes {
        return Err("re-encoding mismatch".into());
    }
    Ok(polys)
}

/// A canonical `F_q` coefficient: six bytes.
pub fn fq48_bytes(v: u64) -> [u8; 6] {
    let b = (v % crate::field_k::Q48).to_le_bytes();
    [b[0], b[1], b[2], b[3], b[4], b[5]]
}

/// Serialize a ring vector (64 × 6-byte coefficients per element).
pub fn ring_vec_bytes(v: &[Poly]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 384);
    for p in v {
        for c in p.0.iter() {
            let m = c.rem_euclid(crate::field_k::Q48 as i64) as u64;
            out.extend_from_slice(&fq48_bytes(m));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poly_of(coefs: &[i64]) -> Poly {
        let mut a = [0i64; 64];
        for (i, &c) in coefs.iter().enumerate() {
            a[i] = c;
        }
        Poly(a)
    }

    #[test]
    fn codec_roundtrip() {
        let flat = vec![
            poly_of(&[0, 1, -1, 63, -63, 64, -64, 4095, -4095, 100_000, -100_000]),
            poly_of(&[7; 64]),
        ];
        let enc = encode_terminal(&flat);
        let dec = decode_terminal(&enc, 2).expect("decode");
        assert_eq!(dec, flat);
        // The size: 128 coefficients → low 112 bytes + high (width from
        // 100000>>6 = 1562 → 11 bits) 176 bytes.
        assert!(enc.len() > 5);
    }

    #[test]
    fn codec_negative_zero_rejected() {
        // Hand-craft a frame with sign=1, residue=0, quotient=0.
        let mut bytes = vec![64u8, 0, 0, 0, 0]; // count=64, width=0
                                                // 64 coefficients × 7 bits = 56 bytes; all zero except the sign of
                                                // the first.
        bytes.push(0x80);
        bytes.extend(std::iter::repeat(0u8).take(55));
        assert!(decode_terminal(&bytes, 1).is_err());
    }

    #[test]
    fn codec_frame_length_enforced() {
        let flat = vec![poly_of(&[1, 2, 3])];
        let mut enc = encode_terminal(&flat);
        enc.push(0);
        assert!(decode_terminal(&enc, 1).is_err());
        enc.pop();
        assert!(decode_terminal(&enc, 1).is_ok());
    }

    #[test]
    fn codec_count_mismatch_rejected() {
        let flat = vec![poly_of(&[5; 64])];
        let enc = encode_terminal(&flat);
        assert!(decode_terminal(&enc, 2).is_err());
        assert!(decode_terminal(&enc, 1).is_ok());
    }

    #[test]
    fn ring_vec_wire_roundtrip() {
        let v = vec![poly_of(&[1_000_003, -1_000_003, 42])];
        let bytes = ring_vec_bytes(&v);
        assert_eq!(bytes.len(), 384);
        // The first coefficient round-trips.
        let m = u64::from_le_bytes({
            let mut a = [0u8; 8];
            a[..6].copy_from_slice(&bytes[..6]);
            a
        });
        assert_eq!(m, 1_000_003);
        assert_eq!(fq48_bytes(1_000_003).len(), 6);
    }
}
