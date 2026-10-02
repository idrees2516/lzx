//! The ring-lookup memory argument — follow-up (c) of
//! `ring-lookups.md`: the zkVM's memory arguments re-proven over
//! `lattice-lookup-ring` (ePrint 2026/471's RAM batch verification)
//! instead of the Twist & Shout lookup layer.
//!
//! The statement is the v2 memory statement: a read/write access
//! stream over a memory window, starting from the public initial image
//! and ending at the public final state, in which every read observes
//! the most recent write. The proof changes only the lookup layer:
//!
//! * **ROM instances** (fetch, input): a single Ring-LogUp of the
//!   read values into the public table (the paper's remark that
//!   "lookup protocols already suffice for ROM").
//! * **RAM instances** (memory, registers): the Section-6 composition —
//!   the sub-RAM isolation lookups (the touched addresses' initial and
//!   final values in the public images, tagged by `g(addr)`), the
//!   offline-memory-checking record layer with the permutation logup
//!   (footnote 5), the read/write-value Hadamard, the timestamp
//!   positivity lookup, and the almost-identical sum-check.
//!
//! u64 values are packed into ring elements as `c₀ + c₁·q` (two
//! coefficients, bijective for `v < q²`).
//!
//! The verifier never re-executes: it consumes the public images, the
//! trace-derived streams, and the proof.

use lattice_core::transcript::Transcript;
use lattice_lookup_ring::ram::{prove_ram_batch, verify_ram_batch, RamOp, RamOracles, RamProof};
use lattice_lookup_ring::ring_d::{Elem, RingD};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupMemoryError {
    Shape(String),
    Ring(String),
    Verify(String),
}

impl From<lattice_lookup_ring::subprotocols::SubError> for LookupMemoryError {
    fn from(e: lattice_lookup_ring::subprotocols::SubError) -> Self {
        LookupMemoryError::Ring(format!("{e:?}"))
    }
}

/// Pack a u64 value into a ring element (`c₀ + c₁·q`).
pub fn pack_u64(ring: &RingD, v: u64) -> Elem {
    let c0 = v % ring.q;
    let c1 = v / ring.q;
    let mut e = ring.zero();
    e.c[0] = c0;
    e.c[1] = c1;
    e
}

/// Unpack a packed value (the inverse of [`pack_u64`]).
pub fn unpack_u64(ring: &RingD, e: &Elem) -> u64 {
    e.coeffs()[0] + e.coeffs()[1] * ring.q
}

/// The bridge instance: the same data `memory.rs`'s `MemoryInstance`
/// carries, at the bridge's granularity.
#[derive(Clone, Debug)]
pub struct LookupMemoryInstance {
    /// The RAM window (the public images' length — a power of two).
    pub window: Vec<u64>,
    /// The final image (public).
    pub final_window: Vec<u64>,
    /// The op stream: `(is_write, address, value)`.
    pub ops: Vec<(bool, u64, u64)>,
    /// Read-only instances: the public table (then `ops` are reads).
    pub table: Option<Vec<u64>>,
}

/// The proof of one bridge instance.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum LookupMemoryProof {
    /// A RAM instance: the Section-6 batch-verification proof, plus
    /// the touched-address list (the sub-window layout — witness-side
    /// data the compiled layer commits).
    Ram {
        proof: RamProof,
        oracles: RamOracles,
        touched: Vec<u64>,
    },
    /// A ROM instance: the Ring-LogUp of the reads into the table.
    Rom {
        proof: lattice_lookup_ring::ring_logup::RingLogupProof,
        oracles: lattice_lookup_ring::ring_logup::LogupOracles,
    },
}

/// Prove one bridge instance.
pub fn prove_lookup_memory(
    ring: &RingD,
    inst: &LookupMemoryInstance,
    transcript: &mut Transcript,
) -> Result<LookupMemoryProof, LookupMemoryError> {
    match &inst.table {
        None => {
            // ---- the RAM path: the Section-6 composition ----
            let m = inst.window.len();
            if !m.is_power_of_two() {
                return Err(LookupMemoryError::Shape("window not a power of two".into()));
            }
            // The sub-RAM: the touched addresses only (the paper's
            // isolation step keeps the logup layer O(|touched| +
            // poly(d)·k)); the untouched rest is covered by the
            // almost-identical layer inside prove_ram_batch.
            let mut touched: Vec<u64> = Vec::new();
            for (_, a, _) in &inst.ops {
                if !touched.contains(a) {
                    touched.push(*a);
                }
            }
            // sub-window: the touched addresses' initial/final values,
            // indexed by the touched order, padded to a power of two
            // with an untouched sentinel address.
            let mut sub_init: Vec<Elem> = Vec::new();
            let mut sub_final: Vec<Elem> = Vec::new();
            for &a in &touched {
                sub_init.push(pack_u64(ring, inst.window[a as usize]));
                sub_final.push(pack_u64(ring, inst.final_window[a as usize]));
            }
            let n_sub = sub_init.len().next_power_of_two().max(2);
            while sub_init.len() < n_sub {
                sub_init.push(pack_u64(ring, 0));
                sub_final.push(pack_u64(ring, 0));
            }
            // the ops: remap addresses to sub-window indices; pad the
            // count to k = 2·n_sub with sentinel reads (the record
            // shape requires 2M + k to be a power of two).
            let k = 2 * n_sub;
            let mut ops: Vec<RamOp> = Vec::with_capacity(k);
            for &(w, a, v) in &inst.ops {
                let idx = touched
                    .iter()
                    .position(|&x| x == a)
                    .ok_or(LookupMemoryError::Shape("op address outside the touched set".into()))?
                    as u64;
                ops.push(RamOp { write: w, addr: idx, value: pack_u64(ring, v) });
            }
            while ops.len() < k {
                // sentinel reads appended at the END of the stream: the
                // live value at sub-address 0 is then the FINAL value
                let v0 = sub_final[0].clone();
                ops.push(RamOp { write: false, addr: 0, value: v0 });
            }
            let (proof, oracles) = prove_ram_batch(ring, &sub_init, &ops, &sub_final, transcript)?;
            Ok(LookupMemoryProof::Ram { proof, oracles, touched })
        }
        Some(table) => {
            // ---- the ROM path: one Ring-LogUp ----
            let n_tab = table.len();
            if !n_tab.is_power_of_two() {
                return Err(LookupMemoryError::Shape("table not a power of two".into()));
            }
            let table_elems: Vec<Elem> = table.iter().map(|&v| pack_u64(ring, v)).collect();
            let mut q_vals: Vec<Elem> = Vec::new();
            let mut q_tags: Vec<Elem> = Vec::new();
            for &(_, a, v) in &inst.ops {
                q_vals.push(pack_u64(ring, v));
                q_tags.push(ring.g_map(a));
            }
            let n_q = q_vals.len().next_power_of_two().max(2);
            while q_vals.len() < n_q {
                q_vals.push(pack_u64(ring, table[0]));
                q_tags.push(ring.g_map(0));
            }
            let (proof, oracles) = lattice_lookup_ring::ring_logup::prove_ring_logup(
                ring,
                &q_vals,
                &table_elems,
                &q_tags,
                transcript,
            )?;
            Ok(LookupMemoryProof::Rom { proof, oracles })
        }
    }
}

/// Verify one bridge instance against the public statement.
pub fn verify_lookup_memory(
    ring: &RingD,
    inst: &LookupMemoryInstance,
    proof: &LookupMemoryProof,
    transcript: &mut Transcript,
) -> Result<(), LookupMemoryError> {
    match proof {
        LookupMemoryProof::Ram { proof, oracles, touched } => {
            let m = inst.window.len();
            if !m.is_power_of_two() {
                return Err(LookupMemoryError::Shape("window not a power of two".into()));
            }
            // the touched set is witness-side (carried in the proof);
            // every address must lie in the window
            for &a in touched {
                if a >= m as u64 {
                    return Err(LookupMemoryError::Verify("touched address out of window".into()));
                }
            }
            let n_sub = touched.len().next_power_of_two().max(2);
            let k = 2 * n_sub;
            if inst.ops.len() > k {
                return Err(LookupMemoryError::Shape("op count exceeds the record shape".into()));
            }
            // the op stream: the real ops form a prefix of the padded
            // stream; the padding is sentinel reads of sub-address 0
            if oracles.ops.len() < inst.ops.len() {
                return Err(LookupMemoryError::Verify("op stream shorter than the statement".into()));
            }
            for (i, &(w, a, v)) in inst.ops.iter().enumerate() {
                let op = &oracles.ops[i];
                let idx = touched.iter().position(|&x| x == a).unwrap_or(u64::MAX as usize);
                if op.write != w || op.addr != idx as u64 || unpack_u64(ring, &op.value) != v {
                    return Err(LookupMemoryError::Verify("op stream mismatch".into()));
                }
            }
            // the RAM proof's sub-image consistency with the full window
            for (si, &a) in touched.iter().enumerate() {
                if si >= oracles.initial.len() {
                    break;
                }
                if unpack_u64(ring, &oracles.initial[si]) != inst.window[a as usize] {
                    return Err(LookupMemoryError::Verify("sub-image initial mismatch".into()));
                }
                if unpack_u64(ring, &oracles.final_[si]) != inst.final_window[a as usize] {
                    return Err(LookupMemoryError::Verify("sub-image final mismatch".into()));
                }
            }
            // the untouched rest: V == V' outside the touched set —
            // the Section-6.2 almost-identical relation, checked
            // directly at the bridge layer (the plain-PIOP model; the
            // compiled layer replaces it with the committed sum-check)
            for (a, &v) in inst.window.iter().enumerate() {
                if !touched.contains(&(a as u64)) && inst.final_window[a] != v {
                    return Err(LookupMemoryError::Verify("untouched word changed".into()));
                }
            }
            verify_ram_batch(ring, n_sub, k, &oracles.initial, &oracles.final_, proof, oracles, transcript)
                .map_err(|e| LookupMemoryError::Ring(format!("{e:?}")))?;
            Ok(())
        }
        LookupMemoryProof::Rom { proof, oracles } => {
            let table = inst.table.as_ref().ok_or_else(|| {
                LookupMemoryError::Shape("ROM proof against a RAM statement".into())
            })?;
            let n_tab = table.len();
            if oracles.b.len() != n_tab {
                return Err(LookupMemoryError::Verify("table shape mismatch".into()));
            }
            for (j, &v) in table.iter().enumerate() {
                if unpack_u64(ring, &oracles.b[j]) != v {
                    return Err(LookupMemoryError::Verify("oracle table mismatch".into()));
                }
            }
            let n_q = oracles.a.len();
            lattice_lookup_ring::ring_logup::verify_ring_logup(
                ring,
                n_q,
                n_tab,
                proof,
                oracles,
                transcript,
            )
            .map_err(|e| LookupMemoryError::Ring(format!("{e:?}")))?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingD {
        RingD::new(4).ok().unwrap()
    }

    #[test]
    fn pack_unpack_bijective() {
        let r = ring();
        for v in [0u64, 1, 42, 1 << 31, u64::MAX / 3, 1 << 63, u64::MAX] {
            let e = pack_u64(&r, v);
            assert_eq!(unpack_u64(&r, &e), v);
        }
    }

    #[test]
    fn ram_instance_roundtrip() {
        let r = ring();
        // a 4-word RAM, 8 ops (k = 2·M_sub): reads and writes
        let window = vec![10u64, 20, 30, 40];
        let mut final_window = window.clone();
        // ops: read 0 (10), write 1 <- 99, read 1 (99), read 2 (30), ...
        let ops = vec![
            (false, 0u64, 10u64),
            (true, 1u64, 99u64),
            (false, 1u64, 99u64),
            (false, 2u64, 30u64),
        ];
        final_window[1] = 99;
        let inst = LookupMemoryInstance { window: window.clone(), final_window, ops, table: None };
        let mut tr = Transcript::new_default(b"lm");
        let proof = prove_lookup_memory(&r, &inst, &mut tr).unwrap_or_else(|e| panic!("prove: {e:?}"));
        let mut tr2 = Transcript::new_default(b"lm");
        verify_lookup_memory(&r, &inst, &proof, &mut tr2).unwrap_or_else(|e| panic!("verify: {e:?}"));
    }

    #[test]
    fn rom_instance_roundtrip() {
        let r = ring();
        let table = vec![100u64, 200, 300, 400];
        let ops = vec![(false, 1u64, 200u64), (false, 3u64, 400u64)];
        let inst = LookupMemoryInstance {
            window: table.clone(),
            final_window: table.clone(),
            ops,
            table: Some(table),
        };
        let mut tr = Transcript::new_default(b"rom");
        let proof = prove_lookup_memory(&r, &inst, &mut tr).unwrap_or_else(|e| panic!("prove: {e:?}"));
        let mut tr2 = Transcript::new_default(b"rom");
        verify_lookup_memory(&r, &inst, &proof, &mut tr2).unwrap_or_else(|e| panic!("verify: {e:?}"));
    }

    #[test]
    fn tampered_final_rejected() {
        let r = ring();
        let window = vec![10u64, 20, 30, 40];
        let mut final_window = window.clone();
        let ops = vec![(true, 1u64, 99u64)];
        final_window[1] = 99;
        let inst = LookupMemoryInstance { window, final_window, ops, table: None };
        let mut tr = Transcript::new_default(b"lm2");
        let proof = prove_lookup_memory(&r, &inst, &mut tr).unwrap_or_else(|e| panic!("prove: {e:?}"));
        // tamper the public final image
        let mut bad = inst.clone();
        bad.final_window[2] = 777;
        let mut tr2 = Transcript::new_default(b"lm2");
        assert!(verify_lookup_memory(&r, &bad, &proof, &mut tr2).is_err());
        // tamper the public initial image
        let mut bad2 = inst.clone();
        bad2.window[0] = 5;
        let mut tr3 = Transcript::new_default(b"lm2");
        assert!(verify_lookup_memory(&r, &bad2, &proof, &mut tr3).is_err());
    }

    #[test]
    fn rom_tampered_table_rejected() {
        let r = ring();
        let table = vec![100u64, 200, 300, 400];
        let ops = vec![(false, 1u64, 200u64)];
        let inst = LookupMemoryInstance {
            window: table.clone(),
            final_window: table.clone(),
            ops,
            table: Some(table),
        };
        let mut tr = Transcript::new_default(b"rom2");
        let proof = prove_lookup_memory(&r, &inst, &mut tr).unwrap_or_else(|e| panic!("prove: {e:?}"));
        let mut bad = inst.clone();
        if let Some(t) = bad.table.as_mut() {
            t[1] = 999;
        }
        let mut tr2 = Transcript::new_default(b"rom2");
        assert!(verify_lookup_memory(&r, &bad, &proof, &mut tr2).is_err());
    }
}
