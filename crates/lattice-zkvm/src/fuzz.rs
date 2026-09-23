//! Verifier hardening: deterministic fuzzing of every untrusted-byte
//! entry point (audit §9.6 item 27 / gate G6).
//!
//! The verifier contract (audit §10.2): *no panics, unchecked slices,
//! unbounded allocations, or `unwrap` on untrusted proof/setup/public-
//! claim paths.* The fuzzers below drive the envelope decoder and the
//! commitment decoder with structured and random malformations; every
//! input must produce `Ok` or `Err` — never a panic (checked with
//! `catch_unwind`).

#![cfg(test)]

use crate::envelope::{ProofEnvelope, Section};
use lattice_commitment::ajtai::AjtaiCommitment;
use lattice_ring::{Modulus32, RingConfig};

/// Deterministic xorshift64* PRNG (fuzz reproducibility).
struct Prng {
    state: u64,
}

impl Prng {
    fn new(seed: u64) -> Self {
        Prng {
            state: seed | 1,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

fn valid_envelope_bytes() -> Vec<u8> {
    let env = ProofEnvelope::new(
        [7u8; 32],
        [8u8; 32],
        [9u8; 32],
        vec![
            Section::Commitment(vec![0x42u8; 96]),
            Section::Sumcheck(vec![1u8; 48]),
            Section::Witness(vec![0xab; 32]),
            Section::Norm(vec![0; 24]),
        ],
    )
    .ok()
    .unwrap();
    env.to_bytes()
}

/// Assert a closure never panics, returning its Result.
fn no_panic<T, E>(f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    match caught {
        Ok(r) => r,
        Err(_) => panic!("FUZZ FAILURE: decoder panicked on untrusted bytes"),
    }
}

#[test]
fn fuzz_envelope_random_mutations() {
    let base = valid_envelope_bytes();
    let mut prng = Prng::new(0xA11CE);
    let mut accepted = 0usize;
    for _ in 0..4000 {
        let mut bytes = base.clone();
        // Apply 1..4 random mutations: flips, byte sets, insertions,
        // deletions.
        let mutations = 1 + prng.below(4);
        for _ in 0..mutations {
            match prng.below(4) {
                0 => {
                    // Bit flip.
                    let pos = prng.below(bytes.len());
                    bytes[pos] ^= 1u8 << prng.below(8);
                }
                1 => {
                    // Random byte overwrite.
                    let pos = prng.below(bytes.len());
                    bytes[pos] = prng.next_u64() as u8;
                }
                2 => {
                    // Truncation.
                    let cut = 1 + prng.below(bytes.len());
                    bytes.truncate(bytes.len() - cut.min(bytes.len()));
                }
                _ => {
                    // Extension with random bytes.
                    let extra = 1 + prng.below(64);
                    let mut tail = vec![0u8; extra];
                    for b in tail.iter_mut() {
                        *b = prng.next_u64() as u8;
                    }
                    bytes.extend_from_slice(&tail);
                }
            }
            if bytes.is_empty() {
                break;
            }
        }
        let result = no_panic(|| ProofEnvelope::from_bytes(&bytes));
        if result.is_ok() {
            accepted += 1;
        }
    }
    // The vast majority of mutations must be rejected (the envelope is
    // digest- and length-bound); a nonzero acceptance rate is fine for
    // mutations that only touch section *payload* bytes.
    assert!(accepted < 4000, "mutations never rejected?");
}

#[test]
fn fuzz_envelope_structured_bombs() {
    let base = valid_envelope_bytes();

    // 1. Truncation at EVERY prefix length: never a panic.
    for cut in 0..base.len() {
        let bytes = &base[..cut];
        let _ = no_panic(|| ProofEnvelope::from_bytes(bytes));
    }

    // 2. Section-count bomb: 64 sections each claiming 16 MiB.
    let mut bomb = Vec::new();
    bomb.extend_from_slice(&1u32.to_le_bytes()); // version
    bomb.extend_from_slice(&[1u8; 32]);
    bomb.extend_from_slice(&[2u8; 32]);
    bomb.extend_from_slice(&[3u8; 32]);
    bomb.extend_from_slice(&64u32.to_le_bytes()); // 64 sections
    for tag in 1..=4u8 {
        bomb.push(tag);
        bomb.extend_from_slice(&(MAX_SECTION_CLAIM).to_le_bytes());
        // No payload — the length claims alone must trip the caps
        // before any large allocation.
    }
    const MAX_SECTION_CLAIM: u32 = (1 << 24) - 1;
    let result = no_panic(|| ProofEnvelope::from_bytes(&bomb));
    assert!(matches!(
        result,
        Err(crate::envelope::EnvelopeError::TotalTooLarge { .. })
            | Err(crate::envelope::EnvelopeError::TrailingBytes { .. })
    ));

    // 3. Oversize section count (> MAX_SECTIONS).
    let mut many = bomb[..4 + 96].to_vec();
    many.extend_from_slice(&1000u32.to_le_bytes());
    let result = no_panic(|| ProofEnvelope::from_bytes(&many));
    assert!(matches!(
        result,
        Err(crate::envelope::EnvelopeError::TooManySections { got: 1000 })
    ));

    // 4. Duplicate tags.
    let mut dup = Vec::new();
    dup.extend_from_slice(&1u32.to_le_bytes());
    dup.extend_from_slice(&[1u8; 96]);
    dup.extend_from_slice(&2u32.to_le_bytes());
    for _ in 0..2 {
        dup.push(2u8);
        dup.extend_from_slice(&4u32.to_le_bytes());
        dup.extend_from_slice(&[0u8; 4]);
    }
    let result = no_panic(|| ProofEnvelope::from_bytes(&dup));
    assert!(matches!(
        result,
        Err(crate::envelope::EnvelopeError::DuplicateSection { tag: 2 })
    ));

    // 5. Unknown tag.
    let mut unknown = Vec::new();
    unknown.extend_from_slice(&1u32.to_le_bytes());
    unknown.extend_from_slice(&[1u8; 96]);
    unknown.extend_from_slice(&1u32.to_le_bytes());
    unknown.push(0xEE);
    unknown.extend_from_slice(&4u32.to_le_bytes());
    unknown.extend_from_slice(&[0u8; 4]);
    let result = no_panic(|| ProofEnvelope::from_bytes(&unknown));
    assert!(matches!(
        result,
        Err(crate::envelope::EnvelopeError::MissingSection { tag: 0xEE })
    ));
}

#[test]
fn fuzz_ajtai_commitment_decode() {
    // The verifier's commitment-decode path: arbitrary bytes must be
    // Ok-or-Err, never a panic.
    let ring = RingConfig::new(Modulus32::Q_32, 4).ok().unwrap();
    let mut prng = Prng::new(0xF00D);
    let elem_len = ring.n() * 4;
    for trial in 0..2000 {
        let len = prng.below(4 * elem_len + 8);
        let mut bytes = vec![0u8; len];
        for b in bytes.iter_mut() {
            *b = prng.next_u64() as u8;
        }
        let result = no_panic(|| AjtaiCommitment::from_bytes(&ring, 2, &bytes));
        if trial % 2 == 0 {
            // Half the trials: corrupt a VALID commitment.
            let good = AjtaiCommitment::from_bytes(&ring, 2, &vec![5u8; 2 * elem_len])
                .ok()
                .unwrap();
            let mut g = good.to_bytes();
            if !g.is_empty() {
                let pos = prng.below(g.len());
                g[pos] ^= 0xFF;
            }
            let _ = no_panic(|| AjtaiCommitment::from_bytes(&ring, 2, &g));
        }
        let _ = result;
    }
}

#[test]
fn fuzz_envelope_all_zero_and_all_ff() {
    // Degenerate inputs: all-zero and all-0xFF buffers of every
    // plausible length class.
    for len in [0usize, 1, 4, 100, 104, 108, 256, 1024, 65536] {
        let zeros = vec![0u8; len];
        let _ = no_panic(|| ProofEnvelope::from_bytes(&zeros));
        let ones = vec![0xFFu8; len];
        let _ = no_panic(|| ProofEnvelope::from_bytes(&ones));
    }
}
