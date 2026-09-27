//! labinius `wire/` — serialized, decodable proof artifacts with
//! bit-packing and rANS entropy coding (Wave 7 item 7.16).
//!
//! Upstream labinius ships a `wire/` module that turns the in-memory
//! proof structures into a decodable byte artifact: small coefficient
//! fields are bit-packed at their true widths (not padded to u32 lanes),
//! and the coefficient stream is entropy-coded with a static-histogram
//! rANS coder (upstream: 664 KB → ~370 KB at ~8.8 bits/coeff, LANES=64
//! interleaved streams). This module ports that layer:
//!
//! * [`BitWriter`]/[`BitReader`] — LSB-first bit packing with exact
//!   width control and `encoded_len` accounting (the "0s are cheap"
//!   doctrine at the serialization layer).
//! * [`RansCoder`] — a 64-bit-state rANS with a transmitted static
//!   frequency table: exact roundtrip, deterministic, no data-dependent
//!   branches on secrets beyond the (public) histogram.
//! * [`WireArtifact`] — the framed container: magic + version + params
//!   digest header, a section table, packed small fields, and the rANS
//!   coefficient stream. `decode` is STRICT: malformed, truncated, or
//!   digest-mismatched inputs are rejected (the envelope discipline the
//!   zkVM crate applies to its own envelopes — replicated here without
//!   a cross-crate dependency).
//!
//! LZX realization notes: single rANS stream (upstream interleaves 64 —
//! a constant-factor throughput concern, not a correctness one); the
//! histogram is built by the encoder and transmitted verbatim; the
//! codec is exact-arithmetic (no lossy quantization — frequencies are
//! exact and sum to 2^M by construction).

// ---------------------------------------------------------------------------
// Bit packing
// ---------------------------------------------------------------------------

/// LSB-first bit writer with exact width control.
pub struct BitWriter {
    pub bytes: Vec<u8>,
    /// Bit position within the current partial byte (0..8).
    bit: u32,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter {
            bytes: Vec::new(),
            bit: 0,
        }
    }

    /// Write `width` low bits of `value` (width ≤ 64).
    pub fn write(&mut self, value: u64, width: u32) {
        for i in 0..width {
            let bit = (value >> i) & 1;
            if self.bit == 0 {
                self.bytes.push(0);
            }
            if bit == 1 {
                let last = self.bytes.len() - 1;
                self.bytes[last] |= 1 << self.bit;
            }
            self.bit = (self.bit + 1) % 8;
        }
    }

    /// Write a byte slice at byte alignment (pads to the next byte).
    pub fn write_bytes(&mut self, data: &[u8]) {
        if self.bit != 0 {
            self.bit = 0;
        }
        self.bytes.extend_from_slice(data);
    }

    /// Bytes written so far.
    pub fn encoded_len(&self) -> usize {
        self.bytes.len()
    }
}

/// LSB-first bit reader over a byte slice with a consumed-bit cursor.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            bit: 0,
        }
    }

    /// Read `width` bits; `None` on exhaustion (strict).
    pub fn read(&mut self, width: u32) -> Option<u64> {
        let mut out = 0u64;
        for i in 0..width {
            let byte = *self.data.get(self.pos)?;
            let bit = (byte >> self.bit) & 1;
            out |= (bit as u64) << i;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }
        Some(out)
    }

    /// Read `len` bytes at byte alignment (strict).
    pub fn read_bytes(&mut self, len: usize) -> Option<&'a [u8]> {
        if self.bit != 0 {
            self.bit = 0;
            self.pos += 1;
        }
        if self.pos + len > self.data.len() {
            return None;
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Some(slice)
    }

    /// True when all input has been consumed (exact-framing check).
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.data.len()
    }
}

// ---------------------------------------------------------------------------
// rANS entropy coder (static transmitted histogram)
// ---------------------------------------------------------------------------

const RANS_M: u32 = 12;
const RANS_MASK: u64 = (1 << RANS_M) - 1;
/// State floor L: the decoder pulls bytes until x ≥ L.
const RANS_MIN_STATE: u64 = 1 << 32;
/// Per-symbol renormalization threshold (the ryg-rANS form):
/// `x_max(s) = ((L >> M) << 8) · f_s` — the encoder pushes bytes while
/// `x ≥ x_max(s)`, which is exactly the condition that the transition
/// output would exceed 256·L. The threshold is frequency-dependent: for
/// small-frequency symbols the post-renorm state legitimately drops
/// below L, which is precisely when the decoder pulls.
fn rans_x_max(freq: u32) -> u64 {
    ((RANS_MIN_STATE >> RANS_M) << 8) * freq as u64
}

/// A static-histogram rANS coder over a bounded symbol alphabet.
#[derive(Clone, Debug)]
pub struct RansCoder {
    /// Per-symbol frequencies (exact, summing to 2^M).
    pub freqs: Vec<u32>,
    /// Cumulative starts.
    cum: Vec<u32>,
    /// Slot → symbol inverse table (length 2^M).
    slot_sym: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    /// Malformed magic/version.
    BadHeader,
    /// Parameters digest mismatch.
    DigestMismatch,
    /// Truncated or over-long input.
    Framing,
    /// Histogram does not sum to 2^M or has a zero-frequency used symbol.
    BadHistogram,
    /// Symbol out of the declared alphabet.
    SymbolOutOfRange { got: u32, alphabet: u32 },
    /// rANS stream decode failed (state underflow / stream exhaustion).
    RansStream,
}

impl RansCoder {
    /// Build from exact symbol counts over the data to encode. Symbols
    /// with zero count get frequency ≥ 1 so the decoder can never land
    /// in an empty slot (frequencies sum exactly to 2^M).
    pub fn from_counts(counts: &[u64]) -> Result<Self, WireError> {
        let total: u64 = counts.iter().sum();
        if total == 0 || counts.len() > 1024 || counts.len() < 2 {
            return Err(WireError::BadHistogram);
        }
        let scale = 1u64 << RANS_M;
        let mut freqs = vec![0u32; counts.len()];
        let mut assigned: u64 = 0;
        for (i, &c) in counts.iter().enumerate() {
            // Largest-remainder rounding; every symbol keeps freq ≥ 1.
            let f = ((c * scale).div_ceil(total)).max(1);
            freqs[i] = f.min(scale) as u32;
            assigned += freqs[i] as u64;
        }
        // Trim the largest buckets down to hit exactly 2^M.
        let mut i = 0;
        while assigned > scale {
            let idx = i % counts.len();
            if freqs[idx] > 1 {
                freqs[idx] -= 1;
                assigned -= 1;
            }
            i += 1;
        }
        // Grow (rare) if rounding under-assigned.
        let mut i = 0;
        while assigned < scale {
            let idx = i % counts.len();
            freqs[idx] += 1;
            assigned += 1;
            i += 1;
        }
        let mut cum = vec![0u32; counts.len()];
        let mut acc = 0u32;
        for i in 0..counts.len() {
            cum[i] = acc;
            acc += freqs[i];
        }
        let mut slot_sym = vec![0u8; 1 << RANS_M];
        for (s, &f) in freqs.iter().enumerate() {
            for slot in cum[s]..cum[s] + f {
                slot_sym[slot as usize] = s as u8;
        }
        }
        Ok(RansCoder {
            freqs,
            cum,
            slot_sym,
        })
    }

    /// The transmitted histogram bytes (little-endian u32 frequencies).
    pub fn histogram_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.freqs.len() * 4);
        for f in &self.freqs {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out
    }

    /// Reconstruct from transmitted histogram bytes.
    pub fn from_histogram_bytes(bytes: &[u8], alphabet: usize) -> Result<Self, WireError> {
        if bytes.len() != alphabet * 4 || !(2..=1024).contains(&alphabet) {
            return Err(WireError::BadHistogram);
        }
        let mut freqs = Vec::with_capacity(alphabet);
        let (chunks, rest) = bytes.as_chunks::<4>();
        if !rest.is_empty() {
            return Err(WireError::BadHistogram);
        }
        for chunk in chunks {
            freqs.push(u32::from_le_bytes(*chunk));
        }
        let sum: u64 = freqs.iter().map(|&f| f as u64).sum();
        if sum != 1 << RANS_M {
            return Err(WireError::BadHistogram);
        }
        let mut cum = vec![0u32; alphabet];
        let mut acc = 0u32;
        for i in 0..alphabet {
            cum[i] = acc;
            acc += freqs[i];
        }
        let mut slot_sym = vec![0u8; 1 << RANS_M];
        for (s, &f) in freqs.iter().enumerate() {
            for slot in cum[s]..cum[s] + f {
                slot_sym[slot as usize] = s as u8;
            }
        }
        Ok(RansCoder {
            freqs,
            cum,
            slot_sym,
        })
    }

    /// Encode a symbol stream. Returns (histogram, payload) where the
    /// payload is: final-state u64 (LE) || renormalization words (LE u32,
    /// in emission order).
    pub fn encode(&self, symbols: &[u32]) -> Result<(Vec<u8>, Vec<u8>), WireError> {
        for &s in symbols {
            if s as usize >= self.freqs.len() {
                return Err(WireError::SymbolOutOfRange {
                    got: s,
                    alphabet: self.freqs.len() as u32,
                });
            }
        }
        let mut x: u64 = RANS_MIN_STATE;
        let mut emitted: Vec<u8> = Vec::new();
        // rANS encodes in reverse order so the decoder reads forward.
        for &s in symbols.iter().rev() {
            let f = self.freqs[s as usize] as u64;
            // Byte-wise renormalization before the state transition, with
            // the per-symbol threshold (see rans_x_max).
            let x_max = rans_x_max(self.freqs[s as usize]);
            while x >= x_max {
                emitted.push(x as u8);
                x >>= 8;
            }
            x = (x / f) * (RANS_MASK + 1) + self.cum[s as usize] as u64 + (x % f);
        }
        // Payload: final state (8 bytes, LE) followed by the renorm bytes
        // in reverse emission order (LIFO — the decoder pulls the most
        // recently pushed byte first).
        let mut payload = Vec::with_capacity(emitted.len() + 8);
        payload.extend_from_slice(&x.to_le_bytes());
        for b in emitted.iter().rev() {
            payload.push(*b);
        }
        Ok((self.histogram_bytes(), payload))
    }

    /// Decode `count` symbols from a payload produced by [`Self::encode`].
    pub fn decode(&self, payload: &[u8], count: usize) -> Result<Vec<u32>, WireError> {
        if payload.len() < 8 {
            return Err(WireError::RansStream);
        }
        let mut x = u64::from_le_bytes([
            payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6],
            payload[7],
        ]);
        let bytes = &payload[8..];
        let mut pos = 0usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let slot = (x & RANS_MASK) as usize;
            let s = self.slot_sym[slot];
            let f = self.freqs[s as usize] as u64;
            // Inverse state transition.
            x = f * (x >> RANS_M) + (x & RANS_MASK) - self.cum[s as usize] as u64;
            // Byte-wise renormalization: pull until the state is back
            // above the floor — exactly inverting the encoder's pushes.
            while x < RANS_MIN_STATE {
                match bytes.get(pos) {
                    Some(&b) => {
                        x = (x << 8) | b as u64;
                        pos += 1;
                    }
                    None => return Err(WireError::RansStream),
                }
            }
            out.push(s as u32);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// The framed wire artifact
// ---------------------------------------------------------------------------

/// A framed, strictly-decodable wire artifact for a small-coefficient
/// proof layer: bit-packed small fields + an rANS-coded coefficient
/// stream + raw commitment blobs, under a params digest.
#[derive(Clone, Debug)]
pub struct WireArtifact {
    /// Digest binding the artifact to its cryptographic parameters
    /// (caller-derived; e.g. the commitment key digest).
    pub params_digest: [u8; 32],
    /// Bit-packed small fields: (value, width) pairs.
    pub small_fields: Vec<(u64, u32)>,
    /// rANS-coded coefficient symbols (bounded alphabet, caller-mapped).
    pub coefficients: Vec<u32>,
    /// Raw commitment/other opaque blobs (byte-aligned sections).
    pub blobs: Vec<Vec<u8>>,
}

impl WireArtifact {
    pub const MAGIC: [u8; 4] = *b"LZXW";
    pub const VERSION: u16 = 1;

    /// Serialize: header || section table || packed fields || histogram ||
    /// rANS payload || blobs. Returns the bytes and the achieved
    /// encoded sizes for size accounting.
    pub fn encode(&self) -> Result<(Vec<u8>, WireSizes), WireError> {
        // Count symbol occurrences for the static histogram.
        let alphabet = self
            .coefficients
            .iter()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(2) as usize;
        let mut counts = vec![0u64; alphabet];
        for &s in &self.coefficients {
            counts[s as usize] += 1;
        }
        let coder = RansCoder::from_counts(&counts)?;
        let (hist, payload) = coder.encode(&self.coefficients)?;
        // Packed small fields: count || (value, width) pairs.
        let mut packed = BitWriter::new();
        packed.write(self.small_fields.len() as u64, 32);
        for (v, w) in &self.small_fields {
            packed.write(*w as u64, 6);
            packed.write(*v, *w);
        }
        // Header: magic || version || digest || alphabet || n_coeffs ||
        // n_blobs || packed_len || payload_len.
        let mut out = Vec::new();
        out.extend_from_slice(&Self::MAGIC);
        out.extend_from_slice(&Self::VERSION.to_le_bytes());
        out.extend_from_slice(&self.params_digest);
        out.extend_from_slice(&(alphabet as u32).to_le_bytes());
        out.extend_from_slice(&(self.coefficients.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.blobs.len() as u32).to_le_bytes());
        out.extend_from_slice(&(packed.encoded_len() as u32).to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&(hist.len() as u32).to_le_bytes());
        for blob in &self.blobs {
            out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        }
        out.extend_from_slice(&packed.bytes);
        out.extend_from_slice(&hist);
        out.extend_from_slice(&payload);
        for blob in &self.blobs {
            out.extend_from_slice(blob);
        }
        let sizes = WireSizes {
            total: out.len(),
            packed_fields: packed.encoded_len(),
            histogram: hist.len(),
            rans_payload: payload.len(),
            blobs: self.blobs.iter().map(|b| b.len()).sum(),
            raw_coefficients: self.coefficients.len() * 4,
        };
        Ok((out, sizes))
    }

    /// Strict decode: framing, magic/version, digest match, exact length.
    pub fn decode(bytes: &[u8], params_digest: &[u8; 32]) -> Result<Self, WireError> {
        if bytes.len() < 40 || bytes[0..4] != Self::MAGIC {
            return Err(WireError::BadHeader);
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != Self::VERSION {
            return Err(WireError::BadHeader);
        }
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&bytes[6..38]);
        if &digest != params_digest {
            return Err(WireError::DigestMismatch);
        }
        let rd_u32 = |off: usize| -> Result<u32, WireError> {
            let b = bytes
                .get(off..off + 4)
                .ok_or(WireError::Framing)?;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let alphabet = rd_u32(38)? as usize;
        let n_coeffs = rd_u32(42)? as usize;
        let n_blobs = rd_u32(46)? as usize;
        let packed_len = rd_u32(50)? as usize;
        let payload_len = rd_u32(54)? as usize;
        let hist_len = rd_u32(58)? as usize;
        let mut off = 62;
        // Blob length table first (the encoder writes it directly after
        // the fixed header).
        let mut blob_lens = Vec::with_capacity(n_blobs);
        for _ in 0..n_blobs {
            blob_lens.push(rd_u32(off)? as usize);
            off += 4;
        }
        // Packed fields section.
        let mut reader = BitReader::new(
            bytes
                .get(off..off + packed_len)
                .ok_or(WireError::Framing)?,
        );
        let n_fields = reader.read(32).ok_or(WireError::Framing)? as usize;
        let mut small_fields = Vec::with_capacity(n_fields);
        for _ in 0..n_fields {
            let w = reader.read(6).ok_or(WireError::Framing)? as u32;
            let v = reader.read(w).ok_or(WireError::Framing)?;
            small_fields.push((v, w));
        }
        off += packed_len;
        let hist = bytes
            .get(off..off + hist_len)
            .ok_or(WireError::Framing)?;
        let coder = RansCoder::from_histogram_bytes(hist, alphabet)?;
        off += hist_len;
        let payload = bytes
            .get(off..off + payload_len)
            .ok_or(WireError::Framing)?;
        let coefficients = coder.decode(payload, n_coeffs)?;
        off += payload_len;
        let mut blobs = Vec::with_capacity(n_blobs);
        for len in blob_lens {
            let blob = bytes.get(off..off + len).ok_or(WireError::Framing)?;
            blobs.push(blob.to_vec());
            off += len;
        }
        if off != bytes.len() {
            return Err(WireError::Framing);
        }
        Ok(WireArtifact {
            params_digest: digest,
            small_fields,
            coefficients,
            blobs,
        })
    }
}

/// Size accounting for a serialized artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WireSizes {
    pub total: usize,
    pub packed_fields: usize,
    pub histogram: usize,
    pub rans_payload: usize,
    pub blobs: usize,
    /// The uncompressed comparison basis (u32 lanes per coefficient).
    pub raw_coefficients: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(tag: u8) -> [u8; 32] {
        let v = lattice_core::keccak::shake256(&[tag], 32);
        let mut out = [0u8; 32];
        out.copy_from_slice(&v);
        out
    }

    #[test]
    fn bit_writer_reader_roundtrip() {
        let mut w = BitWriter::new();
        w.write(0b101, 3);
        w.write(0xABCD, 16);
        w.write(7, 5);
        w.write_bytes(b"tail");
        let bytes = w.bytes.clone();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(3), Some(0b101));
        assert_eq!(r.read(16), Some(0xABCD));
        assert_eq!(r.read(5), Some(7));
        assert_eq!(r.read_bytes(4), Some(&b"tail"[..]));
        assert!(r.is_exhausted());
    }

    #[test]
    fn bit_reader_strict_on_truncation() {
        let mut w = BitWriter::new();
        w.write(0xFFFF_FFFF_FFFF_FFFF, 64);
        let mut r = BitReader::new(&w.bytes[..w.bytes.len() - 1]);
        assert!(r.read(64).is_none());
    }

    #[test]
    fn rans_roundtrip_exact() {
        // Skewed distribution: entropy coding must roundtrip exactly.
        let symbols: Vec<u32> = (0..1000)
            .map(|i| match i % 8 {
                0 | 1 | 2 | 3 => 0,
                4 | 5 => 1,
                6 => 2,
                _ => 3,
            })
            .collect();
        let counts = vec![500, 250, 125, 125];
        let coder = RansCoder::from_counts(&counts).ok().unwrap();
        let (hist, payload) = coder.encode(&symbols).ok().unwrap();
        let decoder = RansCoder::from_histogram_bytes(&hist, 4).ok().unwrap();
        let decoded = decoder.decode(&payload, symbols.len()).ok().unwrap();
        assert_eq!(decoded, symbols);
        // The coded stream beats raw u32 lanes on skewed data.
        assert!(8 + payload.len() < symbols.len() * 4);
    }

    #[test]
    fn rans_uniform_roundtrip() {
        let symbols: Vec<u32> = (0u32..2000).map(|i| i.wrapping_mul(2654435761u32) % 97).collect();
        let mut counts = vec![0u64; 97];
        for &s in &symbols {
            counts[s as usize] += 1;
        }
        let coder = RansCoder::from_counts(&counts).ok().unwrap();
        let (hist, payload) = coder.encode(&symbols).ok().unwrap();
        let decoder = RansCoder::from_histogram_bytes(&hist, 97).ok().unwrap();
        assert_eq!(decoder.decode(&payload, symbols.len()).ok().unwrap(), symbols);
    }

    #[test]
    fn rans_bad_histogram_rejected() {
        // Frequencies not summing to 2^M.
        let bad = vec![1u8; 8];
        assert!(matches!(
            RansCoder::from_histogram_bytes(&bad, 2),
            Err(WireError::BadHistogram)
        ));
    }

    #[test]
    fn rans_symbol_out_of_range_rejected() {
        let coder = RansCoder::from_counts(&[4, 4]).ok().unwrap();
        assert!(matches!(
            coder.encode(&[5]),
            Err(WireError::SymbolOutOfRange { got: 5, alphabet: 2 })
        ));
    }

    #[test]
    fn wire_artifact_roundtrip() {
        let art = WireArtifact {
            params_digest: digest(1),
            small_fields: vec![(0b1010_1, 5), (12345, 17), (1, 1)],
            coefficients: vec![3, 1, 0, 0, 2, 0, 1, 3, 0, 0, 0, 1],
            blobs: vec![vec![9u8; 64], vec![0xAB; 32]],
        };
        let (bytes, sizes) = art.encode().ok().unwrap();
        let decoded = WireArtifact::decode(&bytes, &digest(1)).ok().unwrap();
        assert_eq!(decoded.small_fields, art.small_fields);
        assert_eq!(decoded.coefficients, art.coefficients);
        // Size accounting: total covers every section.
        let header = 62 + 4 * art.blobs.len();
        let sections = sizes.packed_fields + sizes.histogram + sizes.rans_payload + sizes.blobs + header;
        assert_eq!(sizes.total, sections);
        // Entropy: the rANS payload beats raw u32 lanes here (skewed).
        assert!(sizes.rans_payload < sizes.raw_coefficients);
    }

    #[test]
    fn wire_artifact_strict_decode() {
        let art = WireArtifact {
            params_digest: digest(2),
            small_fields: vec![(7, 3)],
            coefficients: vec![0, 1, 1, 0],
            blobs: vec![vec![1u8; 16]],
        };
        let (bytes, _) = art.encode().ok().unwrap();
        // Wrong digest.
        assert!(matches!(
            WireArtifact::decode(&bytes, &digest(9)),
            Err(WireError::DigestMismatch)
        ));
        // Truncated.
        assert!(WireArtifact::decode(&bytes[..bytes.len() - 3], &digest(2)).is_err());
        // Bad magic.
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(matches!(
            WireArtifact::decode(&bad, &digest(2)),
            Err(WireError::BadHeader)
        ));
        // Over-long input.
        let mut long = bytes.clone();
        long.push(0);
        assert!(matches!(
            WireArtifact::decode(&long, &digest(2)),
            Err(WireError::Framing)
        ));
    }

    #[test]
    fn bit_packing_beats_padded_lanes() {
        // 4-bit fields: packed uses ~half the padded-u32 bytes.
        let mut w = BitWriter::new();
        for i in 0..100u64 {
            w.write(i % 16, 4);
        }
        assert!(w.encoded_len() * 8 < 100 * 32);
    }
}
