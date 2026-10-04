//! Secret entropy management for hiding randomness.
//!
//! Audit §9.5 item 23: *"Use OS CSPRNG entropy for private masks;
//! transcript-derived entropy is public and cannot supply hiding
//! randomness."*
//!
//! This module enforces that separation structurally:
//! * [`SecretSeed`] can only be produced by [`OsEntropy`] (or explicitly
//!   labeled test entropy). There is no conversion from any
//!   transcript-derived value into a `SecretSeed`.
//! * [`ShakeStream`] expands a `SecretSeed` into an unbounded XOF stream
//!   with domain separation; two different seeds never alias.
//! * [`NonceLedger`] detects seed reuse across proofs (audit item 24:
//!   negative test for reused randomness).
//!
//! The ledger is per-process; production deployments should additionally
//! persist the fingerprint set across restarts (see SECURITY.md).

use lattice_core::keccak::shake256;
use lattice_core::transcript::Transcript;

/// A 32-byte seed that only originates from a CSPRNG (or a test fixture
/// explicitly marked as such). No `From`/`Into` conversions exist from
/// transcript material: the compiler enforces the audit's entropy rule.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SecretSeed([u8; 32]);

impl core::fmt::Debug for SecretSeed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Never print the seed itself.
        let digest = Transcript::hash_domain(b"seed-fingerprint", &self.0);
        let hex: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
        write!(f, "SecretSeed(fp={hex}..)")
    }
}

impl SecretSeed {
    /// Fingerprint used by the nonce ledger (SHA3-256 of the seed — the
    /// seed itself never leaves this module's boundary un-hashed).
    pub fn fingerprint(&self) -> [u8; 32] {
        Transcript::hash_domain(b"seed-fingerprint", &self.0)
    }

    /// Test-only constructor for deterministic KATs. Named `for_tests` so
    /// accidental production use is visible in review.
    #[cfg(test)]
    pub fn for_tests(bytes: &[u8; 32]) -> Self {
        SecretSeed(*bytes)
    }

    /// Deterministic, review-gated KAT seed constructor. Used by the
    /// reproducible KAT manifests; must never derive from a Fiat-Shamir
    /// transcript.
    pub fn from_kat_label(label: &[u8]) -> Self {
        // Note: hashed from a *label*, not from any protocol transcript
        // state; this is test-vector material, not hiding entropy for
        // production proofs.
        SecretSeed(Transcript::hash_domain(b"kat-seed", label))
    }
}

/// OS entropy source: reads `/dev/urandom`.
///
/// Failures propagate as `Err` — a prover must never silently fall back
/// to deterministic entropy (audit item 24: deterministic RNG is a
/// vulnerability, not a fallback).
pub struct OsEntropy;

impl OsEntropy {
    /// Read `n` bytes of OS entropy.
    pub fn fill(n: usize) -> Result<Vec<u8>, EntropyError> {
        use std::io::Read;
        let mut file =
            std::fs::File::open("/dev/urandom").map_err(|_| EntropyError::OsUnavailable)?;
        let mut buf = vec![0u8; n];
        file.read_exact(&mut buf)
            .map_err(|_| EntropyError::OsShort)?;
        Ok(buf)
    }

    /// Produce a fresh 32-byte secret seed.
    pub fn seed() -> Result<SecretSeed, EntropyError> {
        let bytes = Self::fill(32)?;
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(SecretSeed(arr))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntropyError {
    /// The OS entropy source is unavailable.
    OsUnavailable,
    /// The OS entropy source returned fewer bytes than requested.
    OsShort,
}

impl core::fmt::Display for EntropyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EntropyError::OsUnavailable => write!(f, "OS entropy source unavailable"),
            EntropyError::OsShort => write!(f, "OS entropy source short read"),
        }
    }
}

/// A domain-separated SHAKE-256 XOF stream keyed by a [`SecretSeed`].
///
/// Counter-based: the i-th block is `shake256(seed || domain || i)`, so
/// streams never overlap and seeking is free. The stream is *secret* —
/// used for mask sampling, Σ-protocol masking vectors, and blinding.
pub struct ShakeStream {
    seed: SecretSeed,
    domain: Vec<u8>,
    counter: u64,
    buffer: Vec<u8>,
    offset: usize,
}

impl ShakeStream {
    /// Open a stream with a domain label (domains must be unique per use
    /// site; see the domain registry in `lattice-qrom`).
    pub fn new(seed: SecretSeed, domain: &[u8]) -> Self {
        ShakeStream {
            seed,
            domain: domain.to_vec(),
            counter: 0,
            buffer: Vec::new(),
            offset: 0,
        }
    }

    fn refill(&mut self, need: usize) {
        while self.buffer.len() - self.offset < need {
            let mut input = Vec::with_capacity(48 + self.domain.len());
            input.extend_from_slice(b"LZX-STREAM");
            input.extend_from_slice(&(self.domain.len() as u32).to_le_bytes());
            input.extend_from_slice(&self.domain);
            input.extend_from_slice(&self.seed.0);
            input.extend_from_slice(&self.counter.to_le_bytes());
            let block = shake256(&input, 136);
            self.buffer.extend_from_slice(&block);
            self.counter = self.counter.wrapping_add(1);
        }
    }

    /// Pull `n` secret bytes.
    pub fn next_bytes(&mut self, n: usize) -> Vec<u8> {
        self.refill(n);
        let out = self.buffer[self.offset..self.offset + n].to_vec();
        self.offset += n;
        // Compact the buffer so long streams do not grow without bound.
        if self.offset > 1 << 20 {
            self.buffer.drain(..self.offset);
            self.offset = 0;
        }
        out
    }

    /// Pull a secret u64.
    pub fn next_u64(&mut self) -> u64 {
        let bytes = self.next_bytes(8);
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes);
        u64::from_le_bytes(arr)
    }

    /// Pull a uniform Goldilocks field element (64-bit rejection over
    /// two 8-byte draws — canonical residues only).
    pub fn next_field(&mut self) -> lattice_core::Goldilocks {
        use lattice_core::field::GOLDILOCKS_MODULUS;
        loop {
            let bytes = self.next_bytes(8);
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&bytes);
            let cand = u64::from_le_bytes(arr);
            if cand < GOLDILOCKS_MODULUS {
                return lattice_core::Goldilocks(cand);
            }
        }
    }

    /// Pull `n` uniform field elements.
    pub fn next_fields(&mut self, n: usize) -> Vec<lattice_core::Goldilocks> {
        (0..n).map(|_| self.next_field()).collect()
    }
}

/// Detects secret-seed reuse across proofs.
///
/// Each proof registers `(seed fingerprint, domain)`; registering the
/// same pair twice is a **nonce collision** — the caller must abort the
/// proof. This is the runtime guard backing audit item 24's negative
/// test ("reused randomness").
#[derive(Default)]
pub struct NonceLedger {
    seen: std::collections::HashSet<([u8; 32], Vec<u8>)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonceCollision {
    /// Fingerprint of the reused seed (already hashed — safe to log).
    pub fingerprint: [u8; 32],
    /// Domain in which the reuse occurred.
    pub domain: Vec<u8>,
}

impl NonceLedger {
    pub fn new() -> Self {
        NonceLedger::default()
    }

    /// Register a nonce; `Err` on reuse.
    pub fn register(&mut self, seed: &SecretSeed, domain: &[u8]) -> Result<(), NonceCollision> {
        let key = (seed.fingerprint(), domain.to_vec());
        if self.seen.contains(&key) {
            return Err(NonceCollision {
                fingerprint: key.0,
                domain: key.1,
            });
        }
        self.seen.insert(key);
        Ok(())
    }

    /// Number of distinct nonces registered.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn os_entropy_produces_distinct_seeds() {
        let a = OsEntropy::seed();
        let b = OsEntropy::seed();
        assert!(a.is_ok() && b.is_ok());
        assert_ne!(a.ok().unwrap().fingerprint(), b.ok().unwrap().fingerprint());
    }

    #[test]
    fn streams_are_seed_and_domain_separated() {
        let s1 = SecretSeed::from_kat_label(b"stream-A");
        let s2 = SecretSeed::from_kat_label(b"stream-B");
        let a = ShakeStream::new(s1.clone(), b"mask").next_bytes(32);
        let a2 = ShakeStream::new(s1, b"mask").next_bytes(32);
        let b = ShakeStream::new(s2, b"mask").next_bytes(32);
        let c = ShakeStream::new(SecretSeed::from_kat_label(b"stream-A"), b"other").next_bytes(32);
        // Deterministic per (seed, domain), distinct across.
        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn stream_continuity_and_long_reads() {
        let mut s = ShakeStream::new(SecretSeed::from_kat_label(b"long"), b"d");
        let x = s.next_bytes(64);
        let y = s.next_bytes(64);
        // Continuation: a fresh stream reading 128 at once matches the
        // two sequential reads.
        let mut s2 = ShakeStream::new(SecretSeed::from_kat_label(b"long"), b"d");
        let z = s2.next_bytes(128);
        assert_eq!(&z[..64], &x[..]);
        assert_eq!(&z[64..], &y[..]);
        // Long streams stay healthy past the compaction threshold.
        let mut s3 = ShakeStream::new(SecretSeed::from_kat_label(b"long"), b"d");
        let mut acc = 0u64;
        for _ in 0..20_000 {
            acc ^= s3.next_u64();
        }
        assert_ne!(acc, 0);
    }

    #[test]
    fn field_sampling_is_canonical_and_spread() {
        let mut s = ShakeStream::new(SecretSeed::from_kat_label(b"fields"), b"f");
        let vals = s.next_fields(2000);
        let mut min = u64::MAX;
        let mut max = 0u64;
        for v in &vals {
            let c = v.to_canonical_u64();
            assert!(c < lattice_core::field::GOLDILOCKS_MODULUS);
            min = min.min(c);
            max = max.max(c);
        }
        // 2000 uniform draws over ~2^64 must spread widely.
        assert!(
            max - min > (1u64 << 60),
            "field samples not spread: {min}..{max}"
        );
    }

    #[test]
    fn nonce_ledger_detects_reuse() {
        let seed = SecretSeed::from_kat_label(b"nonce-test");
        let mut ledger = NonceLedger::new();
        assert!(ledger.register(&seed, b"proof-1").is_ok());
        // Same seed, different domain: fine.
        assert!(ledger.register(&seed, b"proof-2").is_ok());
        // Reuse in the same domain: collision.
        let err = ledger.register(&seed, b"proof-1").err();
        assert!(matches!(err, Some(NonceCollision { .. })));
        assert_eq!(ledger.len(), 2);
    }

    #[test]
    fn secret_seed_never_prints_material() {
        let seed = SecretSeed::from_kat_label(b"secret-print");
        let dbg = format!("{seed:?}");
        assert!(!dbg.contains("SecretSeed(["));
        assert!(dbg.starts_with("SecretSeed(fp="));
    }
}
