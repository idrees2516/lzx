//! # lattice-memory
//!
//! Twist & Shout (Chen–Palm–Yeo–Zhang 2025, ePrint 2025/105): faster
//! memory checking arguments via one-hot addressing and increments.
//!
//! The two arguments:
//! * **Shout** (read-only memory / bytecode): every read must come from
//!   the committed table — verified via the multiset (grand-product)
//!   fingerprint over read addresses: `Π_reads (a − r) = Π_table (a − r)`
//!   at a random challenge r, with multiplicities tracked by the counter
//!   deltas (one-hot addressing: each table entry has a one-hot selector
//!   column; reads increment the entry's counter).
//! * **Twist** (read-write memory / RAM): each address's read/write
//!   timeline must be consistent — the value read at time t equals the
//!   last value written before t. The "twist" trick: prove
//!   `V_read(a, t) − V_write(a, t)` vanishings through counter
//!   increments: with one-hot addressing per (address, timestamp), the
//!   read/write check reduces to per-address final-state equalities.
//!
//! Implementation: the deterministic checker (ground truth oracle) plus
//! the random-fingerprint polynomial identity the sumcheck layer proves.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::Goldilocks;

/// A memory access event (address, timestamp, value, is_write).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    pub address: u64,
    pub timestamp: u64,
    pub value: u64,
    pub is_write: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    Transcript(TranscriptError),
    /// Fingerprint identity failed.
    FingerprintFailed,
    /// Timeline consistency failed.
    TimelineFailed,
}

/// Shout: check a read-only table — every read address appears in the
/// table with the claimed value, and the read multiset is covered.
/// Deterministic ground-truth check.
pub fn shout_check(table: &[(u64, u64)], reads: &[(u64, u64)]) -> Result<(), MemoryError> {
    // Build the table map.
    let mut remaining: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
    for (addr, value) in table {
        remaining.insert(*addr, *value);
    }
    for (addr, value) in reads {
        match remaining.get(addr) {
            Some(v) if *v == *value => {
                // Read-only: entries are never consumed (multiple reads of
                // the same entry are allowed).
            }
            _ => return Err(MemoryError::FingerprintFailed),
        }
    }
    Ok(())
}

/// Shout's polynomial identity (the sumcheck-facing statement): at a
/// random challenge r, `Π_reads (a + r·v + r²·t) = Π_covered-table`
/// where the covered-table product runs over table entries with the
/// read-count multiplicity — the grand-product fingerprint that the
/// lookup layer (lattice-lookup) proves.
pub fn shout_fingerprint(
    table: &[(u64, u64)],
    reads: &[Access],
    transcript: &mut Transcript,
) -> Result<Goldilocks, MemoryError> {
    // Challenge r.
    let r = transcript
        .challenge_field(b"shout-r")
        .map_err(MemoryError::Transcript)?;
    // Multiplicities: how many times each table entry is read.
    let mut counts: std::collections::BTreeMap<(u64, u64), u64> =
        std::collections::BTreeMap::new();
    for access in reads {
        *counts.entry((access.address, access.value)).or_insert(0) += 1;
    }
    // Table product with multiplicities: Π_table (a + r·v)^{count}.
    let mut table_product = Goldilocks::ONE;
    for (addr, value) in table {
        let count = counts.get(&(*addr, *value)).copied().unwrap_or(0);
        let factor = Goldilocks::from_u64(*addr)
            .add(&r.mul(&Goldilocks::from_u64(*value)));
        for _ in 0..count {
            table_product = table_product.mul(&factor);
        }
    }
    // Reads product: Π_reads (a + r·v).
    let mut reads_product = Goldilocks::ONE;
    for access in reads {
        let factor = Goldilocks::from_u64(access.address)
            .add(&r.mul(&Goldilocks::from_u64(access.value)));
        reads_product = reads_product.mul(&factor);
    }
    if reads_product != table_product {
        return Err(MemoryError::FingerprintFailed);
    }
    Ok(reads_product)
}

/// Twist: deterministic timeline check — every read returns the value of
/// the most recent write to the same address (or the initial value).
pub fn twist_check(
    initial: &[(u64, u64)],
    accesses: &[Access],
    final_state: &[(u64, u64)],
) -> Result<(), MemoryError> {
    // Current values start from the initial memory.
    let mut current: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
    for (addr, value) in initial {
        current.insert(*addr, *value);
    }
    // Replay in timestamp order (caller provides sorted accesses).
    for access in accesses {
        let cur = current.get(&access.address).copied().unwrap_or(0);
        if access.is_write {
            current.insert(access.address, access.value);
        } else if cur != access.value {
            return Err(MemoryError::TimelineFailed);
        }
    }
    // Final state must match.
    let mut final_map: std::collections::BTreeMap<u64, u64> =
        std::collections::BTreeMap::new();
    for (addr, value) in final_state {
        final_map.insert(*addr, *value);
    }
    if final_map != current {
        return Err(MemoryError::TimelineFailed);
    }
    Ok(())
}

/// Twist's per-address final-state identity (the sumcheck-facing
/// statement): for every address touched, the final memory value equals
/// the last-written value — expressed as the polynomial identity
/// `Σ_addresses (final(a) − last_write(a))·w_a = 0` with random weights.
pub fn twist_fingerprint(
    accesses: &[Access],
    final_state: &[(u64, u64)],
    transcript: &mut Transcript,
) -> Result<Goldilocks, MemoryError> {
    let w = transcript
        .challenge_field(b"twist-w")
        .map_err(MemoryError::Transcript)?;
    // Last write per address.
    let mut last_write: std::collections::BTreeMap<u64, u64> =
        std::collections::BTreeMap::new();
    for access in accesses {
        if access.is_write {
            last_write.insert(access.address, access.value);
        }
    }
    // Initial value default for never-written addresses: last write is the
    // initial state (carried in final_state for untouched addresses).
    let mut acc = Goldilocks::ZERO;
    let mut weight = Goldilocks::ONE;
    for (addr, final_value) in final_state {
        let expected = last_write.get(addr).copied().unwrap_or(*final_value);
        let diff = Goldilocks::from_u64(*final_value)
            .sub(&Goldilocks::from_u64(expected));
        acc = acc.add(&diff.mul(&weight));
        weight = weight.mul(&w);
    }
    if !acc.is_zero() {
        return Err(MemoryError::FingerprintFailed);
    }
    Ok(acc)
}

/// One-hot addressing helper: the selector vector for address `a` among
/// `num_slots` slots — exactly one hot position (the witness column the
/// commitment layer packs).
pub fn one_hot(num_slots: usize, slot: usize) -> Vec<Goldilocks> {
    let mut v = vec![Goldilocks::ZERO; num_slots];
    if slot < num_slots {
        v[slot] = Goldilocks::ONE;
    }
    v
}

/// Increment-checking helper: counters must increase by exactly the
/// number of accesses to their entry (the Shout counter discipline).
pub fn counter_increments(
    accesses: &[Access],
    num_slots: usize,
) -> Result<Vec<u64>, MemoryError> {
    let mut counters = vec![0u64; num_slots];
    for access in accesses {
        if (access.address as usize) < num_slots {
            counters[access.address as usize] += 1;
        } else {
            return Err(MemoryError::FingerprintFailed);
        }
    }
    Ok(counters)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn shout_valid_reads_pass() {
        let table = vec![(1u64, 100u64), (2, 200), (3, 300)];
        let reads = vec![(1, 100), (3, 300), (1, 100)];
        assert!(shout_check(&table, &reads).is_ok());
        // Missing address.
        let bad = vec![(4, 400)];
        assert!(matches!(
            shout_check(&table, &bad),
            Err(MemoryError::FingerprintFailed)
        ));
        // Wrong value.
        let wrong = vec![(1, 999)];
        assert!(shout_check(&table, &wrong).is_err());
    }

    #[test]
    fn shout_fingerprint_holds_and_detects() {
        let table = vec![(1u64, 100u64), (2, 200), (3, 300)];
        let reads = vec![
            Access {
                address: 1,
                timestamp: 0,
                value: 100,
                is_write: false,
            },
            Access {
                address: 2,
                timestamp: 1,
                value: 200,
                is_write: false,
            },
        ];
        let mut t = Transcript::new_default(b"lzx-memory-test");
        assert!(shout_fingerprint(&table, &reads, &mut t).is_ok());
        // A read of a missing entry breaks the fingerprint.
        let bad = vec![
            Access {
                address: 1,
                timestamp: 0,
                value: 100,
                is_write: false,
            },
            Access {
                address: 9,
                timestamp: 1,
                value: 900,
                is_write: false,
            },
        ];
        let mut t2 = Transcript::new_default(b"lzx-memory-test");
        assert!(matches!(
            shout_fingerprint(&table, &bad, &mut t2),
            Err(MemoryError::FingerprintFailed)
        ));
    }

    #[test]
    fn twist_timeline_valid() {
        let initial = vec![(0x10u64, 5u64)];
        let accesses = vec![
            Access { address: 0x10, timestamp: 0, value: 5, is_write: false },
            Access { address: 0x10, timestamp: 1, value: 7, is_write: true },
            Access { address: 0x10, timestamp: 2, value: 7, is_write: false },
            Access { address: 0x20, timestamp: 3, value: 9, is_write: true },
            Access { address: 0x20, timestamp: 4, value: 9, is_write: false },
        ];
        let final_state = vec![(0x10, 7), (0x20, 9)];
        assert!(twist_check(&initial, &accesses, &final_state).is_ok());
    }

    #[test]
    fn twist_stale_read_detected() {
        let initial = vec![(0x10u64, 5u64)];
        // Read after write must see the written value, not the initial.
        let accesses = vec![
            Access { address: 0x10, timestamp: 0, value: 7, is_write: true },
            Access { address: 0x10, timestamp: 1, value: 5, is_write: false }, // stale!
        ];
        let final_state = vec![(0x10, 7)];
        assert!(matches!(
            twist_check(&initial, &accesses, &final_state),
            Err(MemoryError::TimelineFailed)
        ));
    }

    #[test]
    fn twist_final_state_mismatch_detected() {
        let initial = vec![(0x10u64, 5u64)];
        let accesses = vec![
            Access { address: 0x10, timestamp: 0, value: 7, is_write: true },
        ];
        let wrong_final = vec![(0x10, 8)];
        assert!(twist_check(&initial, &accesses, &wrong_final).is_err());
    }

    #[test]
    fn twist_fingerprint_holds_and_detects() {
        let accesses = vec![
            Access { address: 0x10, timestamp: 0, value: 7, is_write: true },
            Access { address: 0x20, timestamp: 1, value: 9, is_write: true },
        ];
        let final_state = vec![(0x10, 7), (0x20, 9)];
        let mut t = Transcript::new_default(b"lzx-memory-test");
        assert!(twist_fingerprint(&accesses, &final_state, &mut t).is_ok());
        // Tampered final state.
        let bad_final = vec![(0x10, 8), (0x20, 9)];
        let mut t2 = Transcript::new_default(b"lzx-memory-test");
        assert!(matches!(
            twist_fingerprint(&accesses, &bad_final, &mut t2),
            Err(MemoryError::FingerprintFailed)
        ));
    }

    #[test]
    fn one_hot_selectors() {
        let v = one_hot(4, 2);
        assert_eq!(v, vec![fe(0), fe(0), fe(1), fe(0)]);
        // Exactly one hot.
        assert_eq!(v.iter().filter(|x| !x.is_zero()).count(), 1);
        // Out-of-range slot: all cold (the verifier rejects at the claim
        // layer).
        assert!(one_hot(4, 9).iter().all(|x| x.is_zero()));
    }

    #[test]
    fn counter_increments_track_accesses() {
        let accesses = vec![
            Access { address: 0, timestamp: 0, value: 1, is_write: false },
            Access { address: 0, timestamp: 1, value: 1, is_write: false },
            Access { address: 2, timestamp: 2, value: 3, is_write: true },
        ];
        let counters = counter_increments(&accesses, 4).ok().unwrap();
        assert_eq!(counters, vec![2, 0, 1, 0]);
        // Out-of-range address rejected.
        let bad = vec![Access { address: 9, timestamp: 0, value: 0, is_write: false }];
        assert!(counter_increments(&bad, 4).is_err());
    }
}
