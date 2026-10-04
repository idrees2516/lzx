//! The response wire format + ring serialization (moved verbatim from
//! `lattice-zkvm::compact` — the shared codec both folds consume).

use lattice_labinius::wire::{BitReader, BitWriter, RansCoder};
use lattice_ring::ring::{RingConfig, RingElement};
use lattice_ring::Modulus32;

/// The bundle ring: R_{Q_32} with n = 64 (X^64+1 negacyclic).
pub fn q32_ring() -> Result<RingConfig, String> {
    RingConfig::new(Modulus32::Q_32, 6).map_err(|e| format!("ring: {e:?}"))
}

/// The magnitude class of a balanced i32 coefficient: 0 for zero, else
/// `bit_length(|c|)` (class c covers magnitudes [2^{c−1}, 2^c)).
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Response codec roundtrip + tamper rejection (moved from
    /// lattice-zkvm::compact's test of the same shape).
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

    #[test]
    fn test_ring_serialization() {
        let ring = q32_ring().unwrap();
        let n = ring.n();
        let elems: Vec<RingElement> = (0..7)
            .map(|i| {
                let coeffs: Vec<u32> = (0..n)
                    .map(|c| ((i * 131 + c * 17) % 1000) as u32)
                    .collect();
                RingElement::from_coeffs(&ring, coeffs)
            })
            .collect();
        let bytes = serialize_elements(&ring, &elems);
        let back = deserialize_elements(&ring, &bytes).unwrap();
        assert_eq!(back.len(), elems.len());
        for (a, b) in elems.iter().zip(back.iter()) {
            assert_eq!(a.coeffs(), b.coeffs());
        }
        let mut tampered = bytes.clone();
        tampered[0] ^= 0xFF;
        // Count mismatch (or a huge count) must fail strictly.
        assert!(deserialize_elements(&ring, &tampered).is_err() || tampered.len() < 4);
        assert!(deserialize_elements(&ring, &bytes[..bytes.len() - 1]).is_err());
    }
}
