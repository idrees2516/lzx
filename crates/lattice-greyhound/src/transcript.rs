//! The Fiat-Shamir transcript: 16-byte hash chaining exactly as in the reference
//! (`h = shake(h || data)[0..16]`), with SHAKE-256 replacing SHAKE128/AES-CTR
//! (documented port deviation — same absorb discipline, different primitive).
//!
//! Every verifier-visible value is absorbed before the challenges that depend on
//! it, so the non-interactive form is a faithful Fiat-Shamir of the papers'
//! public-coin protocols.

use crate::ring::Poly;
use lattice_core::keccak::KeccakSponge;

#[derive(Clone)]
pub struct Transcript {
    pub h: [u8; 16],
}

impl Transcript {
    pub fn new(domain: &[u8], seed: &[u8]) -> Self {
        let mut h = KeccakSponge::new_shake256();
        h.update(b"greyhound/transcript/v1");
        h.update(domain);
        h.update(seed);
        let out = h.finalize(32);
        Self { h: out[..16].try_into().unwrap() }
    }

    pub fn from_state(h: [u8; 16]) -> Self {
        Self { h }
    }

    fn shake(h: &[u8; 16], data: &[u8], out_len: usize) -> Vec<u8> {
        let mut s = KeccakSponge::new_shake256();
        s.update(h);
        s.update(&(data.len() as u64).to_le_bytes());
        s.update(data);
        s.finalize_in_place();
        let mut out = vec![0u8; out_len.max(32)];
        s.squeeze(&mut out);
        out
    }

    /// Absorb arbitrary bytes; advance the chaining state.
    pub fn absorb(&mut self, data: &[u8]) {
        let out = Self::shake(&self.h, data, 32);
        self.h.copy_from_slice(&out[..16]);
    }

    /// Absorb ring elements (bit-packed, the reference's `polzvec_bitpack`).
    pub fn absorb_polys(&mut self, polys: &[Poly]) {
        let mut buf = Vec::with_capacity(polys.len() * 256);
        for p in polys {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        self.absorb(&buf);
    }

    /// Absorb the JL projection vector (256 i32's, the reference's raw memcpy).
    pub fn absorb_i32(&mut self, vals: &[i32]) {
        let mut buf = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        self.absorb(&buf);
    }

    /// Squeeze 32 bytes (challenge seed), absorbing to prevent reuse.
    pub fn squeeze32(&mut self) -> [u8; 32] {
        let out = Self::shake(&self.h, b"", 32);
        self.h.copy_from_slice(&out[16..32]);
        out[..32].try_into().unwrap()
    }

    /// Squeeze a 16-byte sub-seed for a challenge sampler (does not advance the
    /// chaining state — mirrors the reference's `&hashbuf[16]` pattern).
    pub fn challenge_seed(&self) -> [u8; 16] {
        let out = Self::shake(&self.h, b"chal", 32);
        out[..16].try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_is_deterministic_and_sensitive() {
        let t1 = Transcript::new(b"a", b"seed");
        let t2 = Transcript::new(b"a", b"seed");
        let t3 = Transcript::new(b"b", b"seed");
        let mut t1 = t1;
        let mut t2 = t2;
        t1.absorb(b"hello");
        t2.absorb(b"hello");
        assert_eq!(t1.h, t2.h);
        t1.absorb(b"world");
        assert_ne!(t1.h, t2.h);
        assert_ne!(t3.h, t1.h);
    }

    #[test]
    fn absorb_polys_changes_state() {
        let mut t = Transcript::new(b"t", b"s");
        let h0 = t.h;
        t.absorb_polys(&[Poly::constant(5)]);
        assert_ne!(h0, t.h);
        let mut t2 = Transcript::new(b"t", b"s");
        t2.absorb_polys(&[Poly::constant(6)]);
        assert_ne!(t.h, t2.h);
    }
}
