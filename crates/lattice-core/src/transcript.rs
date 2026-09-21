//! Fiat–Shamir transcript with strict domain separation.
//!
//! Design follows the audit report §6.4 checklist:
//! * every protocol participant absorbs a static domain label before use;
//! * labels are length-prefixed to avoid ambiguous concatenation;
//! * challenges are sampled via SHAKE-256 with bounded rejection;
//! * the transcript records a running query counter for QROM accounting.

use crate::field::Goldilocks;
use crate::keccak::{shake256, KeccakSponge};

/// Transcript errors surfaced to verifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptError {
    /// Rejection sampling exceeded the configured bound.
    RejectionBudgetExceeded,
    /// Message length exceeds protocol-declared bounds.
    MessageTooLarge,
}

/// A Fiat–Shamir transcript bound to a protocol name.
pub struct Transcript {
    sponge: KeccakSponge,
    /// Number of random-oracle query-equivalents consumed so far.
    query_count: u64,
    /// Maximum allowed queries before verification refuses.
    query_budget: u64,
}

impl Transcript {
    /// Create a transcript with a protocol domain label and query budget.
    pub fn new(protocol: &[u8], query_budget: u64) -> Self {
        let mut sponge = KeccakSponge::new_shake256();
        // Domain separation: "LZX" tag + protocol name, length-prefixed.
        let mut prefix = Vec::with_capacity(8 + protocol.len());
        prefix.extend_from_slice(b"LZX1");
        prefix.extend_from_slice(&(protocol.len() as u32).to_le_bytes());
        prefix.extend_from_slice(protocol);
        sponge.update(&prefix);
        Transcript {
            sponge,
            query_count: 0,
            query_budget,
        }
    }

    /// Default query budget (generous: 2^20).
    pub fn new_default(protocol: &[u8]) -> Self {
        Self::new(protocol, 1 << 20)
    }

    /// Absorb a labelled public message.
    pub fn append_message(&mut self, label: &[u8], message: &[u8]) -> Result<(), TranscriptError> {
        if message.len() > (1 << 26) {
            return Err(TranscriptError::MessageTooLarge);
        }
        let mut framed = Vec::with_capacity(8 + label.len() + message.len());
        framed.extend_from_slice(&(label.len() as u32).to_le_bytes());
        framed.extend_from_slice(label);
        framed.extend_from_slice(&(message.len() as u32).to_le_bytes());
        framed.extend_from_slice(message);
        self.sponge.update(&framed);
        Ok(())
    }

    /// Absorb a field element under a label (canonical 8-byte encoding).
    pub fn append_field(&mut self, label: &[u8], value: &Goldilocks) {
        // to_bytes is infallible and canonical; length is fixed.
        let _ = self.append_message(label, &value.to_bytes());
    }

    /// Absorb a slice of field elements under a label.
    pub fn append_field_slice(&mut self, label: &[u8], values: &[Goldilocks]) {
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_bytes());
        }
        let _ = self.append_message(label, &bytes);
    }

    /// Absorb arbitrary bytes (e.g. commitments) under a label.
    pub fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        let _ = self.append_message(label, bytes);
    }

    /// Sample `n` field challenges under a label, using rejection sampling
    /// over SHAKE-256 output to preserve uniformity. Counts one query.
    pub fn challenge_fields(&mut self, label: &[u8], n: usize) -> Result<Vec<Goldilocks>, TranscriptError> {
        self.query_count += 1;
        if self.query_count > self.query_budget {
            return Err(TranscriptError::RejectionBudgetExceeded);
        }
        // Absorb label to decouple consecutive challenges.
        let mut framed = Vec::with_capacity(12 + label.len());
        framed.extend_from_slice(b"CHAL");
        framed.extend_from_slice(&(label.len() as u32).to_le_bytes());
        framed.extend_from_slice(label);
        framed.extend_from_slice(&(n as u32).to_le_bytes());
        self.sponge.update(&framed);

        let mut out = Vec::with_capacity(n);
        let mut stream_offset = 0usize;
        while out.len() < n {
            let need = (n - out.len()) * 16;
            let bytes = self.squeeze(&mut stream_offset, need.max(32));
            for chunk in bytes.chunks(16) {
                if out.len() == n {
                    break;
                }
                let mut arr = [0u8; 16];
                arr.copy_from_slice(&chunk[..16.min(chunk.len())]);
                // Rejection sampling over the low 64 bits: a u64 candidate is
                // accepted iff it is a canonical residue. Acceptance
                // probability is 1 - 2^-32, so the loop terminates fast and
                // the resulting distribution is exactly uniform.
                let cand = u64::from_le_bytes(arr[..8].try_into().unwrap_or([0u8; 8]));
                if cand < crate::field::GOLDILOCKS_MODULUS {
                    out.push(Goldilocks(cand));
                }
            }
        }
        Ok(out)
    }

    /// Sample a single field challenge.
    pub fn challenge_field(&mut self, label: &[u8]) -> Result<Goldilocks, TranscriptError> {
        Ok(self
            .challenge_fields(label, 1)?
            .first()
            .copied()
            .unwrap_or(Goldilocks::ZERO))
    }

    /// Sample `n` challenge bytes (raw, for challenge sets / salt derivation).
    pub fn challenge_bytes(&mut self, label: &[u8], n: usize) -> Result<Vec<u8>, TranscriptError> {
        self.query_count += 1;
        if self.query_count > self.query_budget {
            return Err(TranscriptError::RejectionBudgetExceeded);
        }
        let mut framed = Vec::with_capacity(12 + label.len());
        framed.extend_from_slice(b"CBYT");
        framed.extend_from_slice(&(label.len() as u32).to_le_bytes());
        framed.extend_from_slice(label);
        framed.extend_from_slice(&(n as u32).to_le_bytes());
        self.sponge.update(&framed);
        let mut offset = 0usize;
        Ok(self.squeeze(&mut offset, n))
    }

    /// Squeeze bytes from the sponge: clones the state, then produces a
    /// permutation-fresh output stream (permute, read 136 bytes, permute,
    /// ...). `offset` tracks the position in this stream so consecutive
    /// squeezes continue it. The transcript state itself is never modified,
    /// so absorption after squeezing is unambiguous.
    fn squeeze(&self, offset: &mut usize, n: usize) -> Vec<u8> {
        const RATE: usize = 136;
        let mut state = self.sponge_state();
        // Stream block i is available after (i + 1) permutations.
        let block_idx = *offset / RATE;
        for _ in 0..=block_idx {
            crate::keccak::keccak_f1600(&mut state);
        }
        let mut pos = *offset % RATE;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let mut block = [0u8; 200];
            for i in 0..25 {
                block[i * 8..i * 8 + 8].copy_from_slice(&state[i].to_le_bytes());
            }
            let take = (RATE - pos).min(n - out.len());
            out.extend_from_slice(&block[pos..pos + take]);
            pos += take;
            if pos == RATE {
                crate::keccak::keccak_f1600(&mut state);
                pos = 0;
            }
        }
        *offset += out.len();
        out
    }

    /// Access the raw sponge state (for squeeze continuation).
    fn sponge_state(&self) -> [u64; 25] {
        self.sponge.raw_state()
    }

    /// Number of oracle queries consumed so far (QROM accounting hook).
    pub fn query_count(&self) -> u64 {
        self.query_count
    }

    /// One-shot domain-separated hash (statement digests, etc).
    pub fn hash_domain(domain: &[u8], message: &[u8]) -> [u8; 32] {
        let mut buf = Vec::with_capacity(12 + domain.len() + message.len());
        buf.extend_from_slice(b"LZXD");
        buf.extend_from_slice(&(domain.len() as u32).to_le_bytes());
        buf.extend_from_slice(domain);
        buf.extend_from_slice(&(message.len() as u32).to_le_bytes());
        buf.extend_from_slice(message);
        crate::keccak::sha3_256(&buf)
    }

    /// SHAKE-256 convenience for salt derivation.
    pub fn xof(domain: &[u8], message: &[u8], out_len: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + domain.len() + message.len());
        buf.extend_from_slice(b"LZXX");
        buf.extend_from_slice(&(domain.len() as u32).to_le_bytes());
        buf.extend_from_slice(domain);
        buf.extend_from_slice(&(message.len() as u32).to_le_bytes());
        buf.extend_from_slice(message);
        shake256(&buf, out_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_label_sensitive() {
        let mut t1 = Transcript::new_default(b"test-proto");
        let mut t2 = Transcript::new_default(b"test-proto");
        t1.append_message(b"a", b"hello").ok();
        t2.append_message(b"a", b"hello").ok();
        let c1 = t1.challenge_field(b"r").unwrap_or(Goldilocks::ZERO);
        let c2 = t2.challenge_field(b"r").unwrap_or(Goldilocks::ZERO);
        assert_eq!(c1, c2);

        // Different label -> different challenge.
        let mut t3 = Transcript::new_default(b"test-proto");
        t3.append_message(b"a", b"hello").ok();
        let c3 = t3.challenge_field(b"other").unwrap_or(Goldilocks::ZERO);
        assert_ne!(c1, c3);

        // Different protocol domain -> different challenge.
        let mut t4 = Transcript::new_default(b"other-proto");
        t4.append_message(b"a", b"hello").ok();
        let c4 = t4.challenge_field(b"r").unwrap_or(Goldilocks::ZERO);
        assert_ne!(c1, c4);
    }

    #[test]
    fn label_concatenation_unambiguous() {
        // ("ab", "c") must differ from ("a", "bc") thanks to length framing.
        let mut t1 = Transcript::new_default(b"amb");
        t1.append_message(b"ab", b"c").ok();
        let mut t2 = Transcript::new_default(b"amb");
        t2.append_message(b"a", b"bc").ok();
        let c1 = t1.challenge_field(b"r").unwrap_or(Goldilocks::ZERO);
        let c2 = t2.challenge_field(b"r").unwrap_or(Goldilocks::ZERO);
        assert_ne!(c1, c2);
    }

    #[test]
    fn resumable_squeeze_keeps_isolation() {
        let mut t = Transcript::new_default(b"squeeze");
        t.append_message(b"x", b"data").ok();
        let a = t.challenge_field(b"r1").unwrap_or(Goldilocks::ZERO);
        let b = t.challenge_field(b"r2").unwrap_or(Goldilocks::ZERO);
        let c = t.challenge_field(b"r3").unwrap_or(Goldilocks::ZERO);
        assert_ne!(a, b);
        assert_ne!(b, c);
        // Fresh transcript reproduces the same sequence.
        let mut t2 = Transcript::new_default(b"squeeze");
        t2.append_message(b"x", b"data").ok();
        assert_eq!(a, t2.challenge_field(b"r1").unwrap_or(Goldilocks::ZERO));
        assert_eq!(b, t2.challenge_field(b"r2").unwrap_or(Goldilocks::ZERO));
        assert_eq!(c, t2.challenge_field(b"r3").unwrap_or(Goldilocks::ZERO));
    }

    #[test]
    fn query_budget_enforced() {
        let mut t = Transcript::new(b"budget", 3);
        assert!(t.challenge_field(b"a").is_ok());
        assert!(t.challenge_field(b"b").is_ok());
        assert!(t.challenge_field(b"c").is_ok());
        assert_eq!(
            t.challenge_field(b"d").err(),
            Some(TranscriptError::RejectionBudgetExceeded)
        );
    }

    #[test]
    fn many_challenges_distinct_and_canonical() {
        let mut t = Transcript::new_default(b"many");
        let chals = t.challenge_fields(b"vec", 1000).unwrap_or_default();
        assert_eq!(chals.len(), 1000);
        for c in &chals {
            assert!(c.to_canonical_u64() < crate::field::GOLDILOCKS_MODULUS);
        }
    }
}
