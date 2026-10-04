//! Section 6 of ePrint 2026/471: batch verification of RAM updates
//! over the split ring — the substrate the zkVM's memory arguments
//! compile onto (follow-up (c) of `ring-lookups.md`).
//!
//! * **Construction 6.3** (`[`memcheck`]`): the offline memory checker
//!   — the augmented operation list (initial writes in address order,
//!   the ops, final reads), producing the read/write record lists
//!   `(V_R, A_R, T_R)` and `(V_W, A_W, T_W)` of Lemma 6.4.
//! * **The composition** (Theorem 6.13's blueprint):
//!   1. *Indexed lookup isolation* — the read values land in the
//!      initial image and the written values in the final image, via
//!      Ring-LogUp with address tags `g(addr)`;
//!   2. *Memory consistency* — the record multiset equality
//!      (`R[M:2M+k]` vs `W[0:M+k]`, the footnote-5 logup with
//!      all-ones multiplicities over the α-combined records), the
//!      read/write value Hadamard
//!      `(1^M‖o₁‖0^M) ∘ (V−V_R ‖ o₃−V_R ‖ V′−V_R) = V_W − V_R`
//!      (the paper's printed segment order has a sign slip — see the
//!      method docs), the op/addr/timestamp range lookups, and the
//!      timestamp-positivity lookup;
//!   3. *Almost-identical states* — the sum-check
//!      `Σ EQ(x,r)·(1−û)(V̂−V̂′) = 0` with the touched-indicator `u`
//!      bound by binarity + an indexed lookup of the touched set into
//!      the ops' addresses (a sound tightening of Construction 6.9's
//!      `v̂/û̃` dance; documented as the deviation).

// (Kernel loops use explicit indices by convention.)
#![allow(clippy::needless_range_loop)]
use crate::ring_d::{Elem, RingD};
use crate::ring_sumcheck::{
    prove_sumcheck, verify_sumcheck, RingFactor, RingSumcheckProof, RingTerm, RingVirtualPoly,
};
use crate::subprotocols::{
    prove_binary_check, prove_hadamard, verify_binary_check, verify_hadamard, SubError,
};
use lattice_core::transcript::Transcript;

/// One RAM operation: `(op, addr, value)` with `op ∈ {read, write}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamOp {
    pub write: bool,
    pub addr: u64,
    /// For writes: the new value. For reads: the observed value (the
    /// paper's `⊥` placeholder realized as the read value).
    pub value: Elem,
}

/// The read/write record lists of Construction 6.3.
#[derive(Clone, Debug)]
pub struct Records {
    pub vr: Vec<Elem>,
    pub ar: Vec<u64>,
    pub tr: Vec<u64>,
    pub vw: Vec<Elem>,
    pub aw: Vec<u64>,
    pub tw: Vec<u64>,
}

impl Records {
    /// The α-combination `(α₁·V, α₂·A, α₃·T)` of a record segment.
    fn combine(
        ring: &RingD,
        alpha1: &Elem,
        alpha2: &Elem,
        alpha3: &Elem,
        seg: &[Elem],
        addrs: &[u64],
        times: &[u64],
    ) -> Vec<Elem> {
        seg.iter()
            .zip(addrs.iter().zip(times.iter()))
            .map(|(v, (&a, &t))| {
                let t1 = ring.mul(alpha1, v);
                let t2 = ring.mul(alpha2, &ring.constant(a));
                let t3 = ring.mul(alpha3, &ring.constant(t));
                ring.add(&ring.add(&t1, &t2), &t3)
            })
            .collect()
    }
}

/// Construction 6.3: run the offline memory checker over the augmented
/// operation list. Returns the record lists (length `2M + k` each).
pub fn memcheck(
    ring: &RingD,
    m: usize,
    initial: &[Elem],
    ops: &[RamOp],
    final_: &[Elem],
) -> Result<Records, SubError> {
    if initial.len() != m || final_.len() != m {
        return Err(SubError::Shape("RAM image arity mismatch".into()));
    }
    let k = ops.len();
    // the live state
    let mut cur: Vec<Elem> = initial.to_vec();
    let mut last_time: Vec<u64> = vec![0; m];
    let mut vr: Vec<Elem> = Vec::with_capacity(2 * m + k);
    let mut ar: Vec<u64> = Vec::with_capacity(2 * m + k);
    let mut tr: Vec<u64> = Vec::with_capacity(2 * m + k);
    let mut vw: Vec<Elem> = Vec::with_capacity(2 * m + k);
    let mut aw: Vec<u64> = Vec::with_capacity(2 * m + k);
    let mut tw: Vec<u64> = Vec::with_capacity(2 * m + k);
    // Phase 1: write the initial image in address order (times 1..=M).
    for (addr, v) in initial.iter().enumerate() {
        let t = addr as u64 + 1;
        vr.push(ring.zero());
        ar.push(addr as u64);
        tr.push(0);
        vw.push(v.clone());
        aw.push(addr as u64);
        tw.push(t);
        cur[addr] = v.clone();
        last_time[addr] = t;
    }
    // Phase 2: the ops (times M+1..=M+k).
    for (i, op) in ops.iter().enumerate() {
        let t = m as u64 + i as u64 + 1;
        let addr = op.addr as usize;
        if addr >= m {
            return Err(SubError::Shape("op address out of range".into()));
        }
        let old_t = last_time[addr];
        match op.write {
            true => {
                let old = cur[addr].clone();
                vr.push(old);
                ar.push(op.addr);
                tr.push(old_t);
                vw.push(op.value.clone());
                aw.push(op.addr);
                tw.push(t);
                cur[addr] = op.value.clone();
            }
            false => {
                if cur[addr] != op.value {
                    return Err(SubError::Verify(format!(
                        "read at {} inconsistent",
                        op.addr
                    )));
                }
                vr.push(op.value.clone());
                ar.push(op.addr);
                tr.push(old_t);
                vw.push(op.value.clone());
                aw.push(op.addr);
                tw.push(t);
            }
        }
        last_time[addr] = t;
    }
    // Phase 3: read the final image in address order (times M+k+1..).
    for (addr, v) in final_.iter().enumerate() {
        let t = (m + k) as u64 + addr as u64 + 1;
        let old_t = last_time[addr];
        vr.push(v.clone());
        ar.push(addr as u64);
        tr.push(old_t);
        vw.push(v.clone());
        aw.push(addr as u64);
        tw.push(t);
    }
    Ok(Records {
        vr,
        ar,
        tr,
        vw,
        aw,
        tw,
    })
}

/// The plain-PIOP oracle view of the RAM proof.
#[derive(Clone, Debug)]
pub struct RamOracles {
    pub initial: Vec<Elem>,
    pub final_: Vec<Elem>,
    pub ops: Vec<RamOp>,
    pub records: Records,
    /// The Hadamard vectors of the consistency layer.
    pub had_a: Vec<Elem>,
    pub had_b: Vec<Elem>,
    pub had_c: Vec<Elem>,
    /// The touched indicator u over the RAM addresses.
    pub u: Vec<Elem>,
}

#[derive(Clone, Debug)]
pub struct RamProof {
    /// Transcript-sync canaries (bisecting the challenge-stream
    /// alignment between the prover and the verifier).
    pub canary_1: Vec<u8>,
    pub canary_2: Vec<u8>,
    pub canary_3: Vec<u8>,
    pub canary_5: Vec<u8>,
    /// Sub-protocol 1: reads ∈ initial image (values, tags on addr).
    pub lookup_reads: crate::ring_logup::RingLogupProof,
    pub reads_oracles: crate::ring_logup::LogupOracles,
    /// Sub-protocol 1: writes ∈ final image.
    pub lookup_writes: crate::ring_logup::RingLogupProof,
    pub writes_oracles: crate::ring_logup::LogupOracles,
    /// Sub-protocol 2: the permutation logup over the α-combined
    /// records (all-ones multiplicities — footnote 5).
    pub perm_logup: crate::ring_logup::RingLogupProof,
    pub perm_oracles: crate::ring_logup::LogupOracles,
    /// Sub-protocol 2: the read/write-value Hadamard.
    pub hadamard: crate::subprotocols::HadamardProof,
    /// Sub-protocol 2: timestamp positivity + op well-formedness.
    pub ts_lookup: crate::ring_logup::RingLogupProof,
    pub ts_oracles: crate::ring_logup::LogupOracles,
    /// Sub-protocol 3: the almost-identical sum-check.
    pub aid_sc: RingSumcheckProof,
    /// Sub-protocol 3: the touched-indicator binary check.
    pub u_binary: crate::subprotocols::BinaryCheckProof,
    /// The touched-addresses lookup into the ops' addresses.
    pub touched_lookup: crate::ring_logup::RingLogupProof,
    pub touched_oracles: crate::ring_logup::LogupOracles,
}

/// The Hadamard vectors for Lemma 6.4's items 6–7:
/// `a = 1^M ‖ o₁ ‖ 0^M`,
/// `b = (V − V_R[0:M]) ‖ (o₃ − V_R[M:M+k]) ‖ (V′ − V_R[M+k:])`,
/// `c = V_W − V_R` (full length).
/// (The paper's printed `b = (V_R − V ‖ o₃ ‖ V′)` has a sign/segment
/// slip: with `V_R[0:M] = 0` and `V_W[0:M] = V` the printed form gives
/// `a∘b = −V ≠ V = c`; the form above is the identity its proof
/// intends.)
fn consistency_vectors(
    ring: &RingD,
    m: usize,
    k: usize,
    initial: &[Elem],
    final_: &[Elem],
    ops: &[RamOp],
    records: &Records,
) -> (Vec<Elem>, Vec<Elem>, Vec<Elem>) {
    let total = 2 * m + k;
    let mut a = Vec::with_capacity(total);
    let mut b = Vec::with_capacity(total);
    // initial segment
    for i in 0..m {
        a.push(ring.one());
        b.push(ring.sub(&initial[i], &records.vr[i]));
    }
    // ops segment
    for (i, op) in ops.iter().enumerate() {
        a.push(if op.write { ring.one() } else { ring.zero() });
        let vr = &records.vr[m + i];
        b.push(ring.sub(&op.value, vr));
    }
    // final segment
    for i in 0..m {
        a.push(ring.zero());
        b.push(ring.sub(&final_[i], &records.vr[m + k + i]));
    }
    let c: Vec<Elem> = (0..total)
        .map(|i| ring.sub(&records.vw[i], &records.vr[i]))
        .collect();
    (a, b, c)
}

/// The touched-indicator `u` over RAM addresses.
fn touched_indicator(ring: &RingD, m: usize, ops: &[RamOp]) -> Vec<Elem> {
    let mut u = vec![ring.zero(); m];
    for op in ops {
        u[op.addr as usize] = ring.one();
    }
    u
}

/// Run the batch-verification prover (the Theorem 6.13 composition).
#[allow(clippy::too_many_lines)]
pub fn prove_ram_batch(
    ring: &RingD,
    initial: &[Elem],
    ops: &[RamOp],
    final_: &[Elem],
    transcript: &mut Transcript,
) -> Result<(RamProof, RamOracles), SubError> {
    let m = initial.len();
    let k = ops.len();
    if final_.len() != m {
        return Err(SubError::Shape("image arity mismatch".into()));
    }
    if !m.is_power_of_two() || !k.is_power_of_two() || (2 * m + k).count_ones() != 1 {
        return Err(SubError::Shape("M, k, 2M+k must be powers of two".into()));
    }
    // fail-closed: the ops actually transform initial into final_
    {
        let mut cur = initial.to_vec();
        for op in ops {
            match op.write {
                true => cur[op.addr as usize] = op.value.clone(),
                false => {
                    if cur[op.addr as usize] != op.value {
                        return Err(SubError::Verify("inconsistent read".into()));
                    }
                }
            }
        }
        if cur != final_ {
            return Err(SubError::Verify(
                "ops do not produce the final image".into(),
            ));
        }
    }
    let records = memcheck(ring, m, initial, ops, final_)?;
    // ---- Sub-protocol 1: the two isolation lookups (DGPPS sub-RAMs) ----
    // W = {(addr, V[addr]) : touched} and W' = {(addr, V'[addr]) :
    // touched} — the touched addresses' initial and final values,
    // looked up in the public images with address tags g(addr). The
    // per-op values are bound through the memcheck record layer below.
    let mut touched_addrs: Vec<u64> = Vec::new();
    for op in ops {
        if !touched_addrs.contains(&op.addr) {
            touched_addrs.push(op.addr);
        }
    }
    let n_touched = touched_addrs.len().next_power_of_two().max(2);
    // V-isolation
    let mut qv_vals: Vec<Elem> = touched_addrs
        .iter()
        .map(|&a| initial[a as usize].clone())
        .collect();
    let mut qv_tags: Vec<Elem> = touched_addrs.iter().map(|&a| ring.g_map(a)).collect();
    while qv_vals.len() < n_touched {
        qv_vals.push(initial[0].clone());
        qv_tags.push(ring.g_map(0));
    }
    let (lookup_reads, reads_oracles) =
        crate::ring_logup::prove_ring_logup(ring, &qv_vals, initial, &qv_tags, transcript)?;
    // V'-isolation
    let mut qw_vals: Vec<Elem> = touched_addrs
        .iter()
        .map(|&a| final_[a as usize].clone())
        .collect();
    let mut qw_tags: Vec<Elem> = touched_addrs.iter().map(|&a| ring.g_map(a)).collect();
    while qw_vals.len() < n_touched {
        qw_vals.push(final_[0].clone());
        qw_tags.push(ring.g_map(0));
    }
    let (lookup_writes, writes_oracles) =
        crate::ring_logup::prove_ring_logup(ring, &qw_vals, final_, &qw_tags, transcript)?;
    let canary_1 = transcript
        .challenge_bytes(b"canary1", 8)
        .unwrap_or_default();
    // ---- Sub-protocol 2: the permutation logup (footnote 5) ----
    let alpha1 = ring.sample_challenge(transcript, b"ram-al1");
    let alpha2 = ring.sample_challenge(transcript, b"ram-al2");
    let alpha3 = ring.sample_challenge(transcript, b"ram-al3");
    let seg_r = &records.vr[m..];
    let seg_w = &records.vw[..m + k];
    let comb_r = Records::combine(
        ring,
        &alpha1,
        &alpha2,
        &alpha3,
        seg_r,
        &records.ar[m..],
        &records.tr[m..],
    );
    let comb_w = Records::combine(
        ring,
        &alpha1,
        &alpha2,
        &alpha3,
        seg_w,
        &records.aw[..m + k],
        &records.tw[..m + k],
    );
    // The permutation argument (footnote 5): comb_r (the R records
    // `R[M:2M+k]`) and comb_w (the W records `W[0:M+k]`) are equal as
    // multisets — the logup with all-ones multiplicities over the
    // length-(M+k) lists, padded to a power of two with self-consistent
    // sentinel pairs on both sides. The table rows are tagged
    // positionally `g(j)`; each query row is tagged by the position of
    // its match in the table (the permutation's action, recovered by
    // multiset matching).
    let n_perm = comb_w.len().next_power_of_two().max(2);
    let mut perm_t: Vec<Elem> = comb_r.clone();
    let perm_pad_base = {
        // distinct sentinel values beyond the record range
        let mut b = ring.constant((2 * m + k + 1) as u64);
        // ensure distinctness from real records (random whp)
        b = ring.add(&b, &ring.pow(&ring.x_gen(), 61));
        b
    };
    let first_pad_pos = perm_t.len();
    for i in 0..(n_perm - comb_r.len()) {
        perm_t.push(ring.add(&perm_pad_base, &ring.constant(i as u64)));
    }
    let perm_tags: Vec<Elem> = (0..n_perm).map(|i| ring.g_map(i as u64)).collect();
    let mut perm_q_vals = Vec::with_capacity(n_perm);
    let mut perm_q_tags = Vec::with_capacity(n_perm);
    {
        let mut used = vec![false; n_perm];
        for wv in comb_w.iter() {
            let mut found = None;
            for (j, rv) in perm_t.iter().enumerate() {
                if !used[j] && rv == wv {
                    found = Some(j);
                    used[j] = true;
                    break;
                }
            }
            let j = found
                .ok_or_else(|| SubError::Verify("record multisets differ (fail-closed)".into()))?;
            perm_q_vals.push(wv.clone());
            perm_q_tags.push(ring.g_map(j as u64));
        }
        // the padded query rows reference the sentinel positions
        for i in 0..(n_perm - comb_w.len()) {
            let pos = first_pad_pos + i;
            perm_q_vals.push(perm_t[pos].clone());
            perm_q_tags.push(perm_tags[pos].clone());
        }
    }
    if std::env::var("LZX_DBG").is_ok() {
        eprintln!(
            "perm q0={:?} tag0={:?}; t0={:?} t1={:?}; value-positions of q0: {:?}",
            perm_q_vals[0].coeffs().to_vec(),
            perm_q_tags[0].coeffs().to_vec(),
            perm_t[0].coeffs().to_vec(),
            perm_t[1].coeffs().to_vec(),
            perm_t
                .iter()
                .enumerate()
                .filter(|(_, v)| **v == perm_q_vals[0])
                .map(|(j, _)| j)
                .collect::<Vec<_>>()
        );
    }
    let (perm_logup, perm_oracles) =
        crate::ring_logup::prove_ring_logup(ring, &perm_q_vals, &perm_t, &perm_q_tags, transcript)?;
    let canary_2 = transcript
        .challenge_bytes(b"canary2", 8)
        .unwrap_or_default();
    // ---- Sub-protocol 2: the read/write-value Hadamard ----
    let (had_a, had_b, had_c) = consistency_vectors(ring, m, k, initial, final_, ops, &records);
    let (hadamard, _hq) = prove_hadamard(ring, &had_a, &had_b, &had_c, transcript)?;
    // ---- Sub-protocol 2: timestamp positivity + op well-formedness ----
    // [2M+k] − T_R ∈ [1, 2M+k] with T_R[0:M] = 0 (condition 8): the
    // table {1..2M+k} tagged positionally.
    let total = (2 * m + k) as u64;
    // the table rows are tagged positionally: value v sits at row v−1
    let ts_vals: Vec<Elem> = (0..(2 * m + k))
        .map(|i| ring.constant(total - records.tr[i]))
        .collect();
    let ts_tags: Vec<Elem> = (0..(2 * m + k))
        .map(|i| ring.g_map(total - records.tr[i] - 1))
        .collect();
    let ts_table: Vec<Elem> = (1..=total).map(|v| ring.constant(v)).collect();
    let (ts_lookup, ts_oracles) =
        crate::ring_logup::prove_ring_logup(ring, &ts_vals, &ts_table, &ts_tags, transcript)?;
    let canary_3 = transcript
        .challenge_bytes(b"canary3", 8)
        .unwrap_or_default();
    // ---- Sub-protocol 3: almost identical ----
    let u = touched_indicator(ring, m, ops);
    // the sum-check: Σ_x EQ(x,r)·(1−û(x))(V̂(x)−V̂′(x)) = 0
    let log_m = m.trailing_zeros() as usize;
    let r_pt: Vec<Elem> = (0..log_m)
        .map(|i| ring.sample_challenge(transcript, format!("aid-r{i}").as_bytes()))
        .collect();
    let eq = ring.eq_row(&r_pt);
    let one_minus_u: Vec<Elem> = u.iter().map(|x| ring.sub(&ring.one(), x)).collect();
    let diff: Vec<Elem> = (0..m).map(|i| ring.sub(&initial[i], &final_[i])).collect();
    let poly = RingVirtualPoly {
        num_vars: log_m,
        claimed_sum: ring.zero(),
        terms: vec![RingTerm {
            coeff: ring.one(),
            factors: vec![
                RingFactor::Eq(eq),
                RingFactor::Mle(one_minus_u),
                RingFactor::Mle(diff),
            ],
        }],
    };
    let aid_sc = prove_sumcheck(ring, &poly, transcript)?;
    // u binary + the touched set ⊆ ops' addresses (indexed lookup)
    let (u_binary, _ubq) = prove_binary_check(ring, &u, transcript)?;
    let canary_5 = transcript
        .challenge_bytes(b"canary5", 8)
        .unwrap_or_default();
    // The touched-address lookup: every address with u_i = 1 was
    // accessed by some op. Table = the UNIQUE op addresses (values as
    // ring constants, tags g(addr)); query = the touched addresses
    // with tags g(addr). Padded query rows reference the first table
    // row's pair.
    let mut uniq_addrs: Vec<u64> = Vec::new();
    for op in ops {
        if !uniq_addrs.contains(&op.addr) {
            uniq_addrs.push(op.addr);
        }
    }
    let mut tab_vals: Vec<Elem> = uniq_addrs.iter().map(|&a| ring.constant(a)).collect();
    let n_tab_touched = tab_vals.len().next_power_of_two().max(2);
    // sentinel rows: fresh addresses (never queried); the table is
    // tagged positionally g(j) by the logup.
    let sentinel_a = m as u64 + k as u64 + 1;
    while tab_vals.len() < n_tab_touched {
        let a = sentinel_a + tab_vals.len() as u64;
        tab_vals.push(ring.constant(a));
    }
    let tab_tags: Vec<Elem> = (0..n_tab_touched).map(|j| ring.g_map(j as u64)).collect();
    // query rows: the touched addresses, tagged by their row's position
    let mut touched_vals: Vec<Elem> = Vec::new();
    let mut touched_tags: Vec<Elem> = Vec::new();
    for i in 0..m {
        if u[i].ct() == 1 {
            let pos = uniq_addrs
                .iter()
                .position(|&a| a == i as u64)
                .ok_or_else(|| {
                    SubError::Verify("touched address not in the op set (fail-closed)".into())
                })?;
            touched_vals.push(ring.constant(i as u64));
            touched_tags.push(ring.g_map(pos as u64));
        }
    }
    while touched_vals.len() < n_tab_touched {
        touched_vals.push(tab_vals[0].clone());
        touched_tags.push(tab_tags[0].clone());
    }
    let (touched_lookup, touched_oracles) = crate::ring_logup::prove_ring_logup(
        ring,
        &touched_vals,
        &tab_vals,
        &touched_tags,
        transcript,
    )?;

    let oracles = RamOracles {
        initial: initial.to_vec(),
        final_: final_.to_vec(),
        ops: ops.to_vec(),
        records,
        had_a,
        had_b,
        had_c,
        u,
    };
    Ok((
        RamProof {
            canary_1,
            canary_2,
            canary_3,
            canary_5,
            lookup_reads,
            reads_oracles,
            lookup_writes,
            writes_oracles,
            perm_logup,
            perm_oracles,
            hadamard,
            ts_lookup,
            ts_oracles,
            aid_sc,
            u_binary,
            touched_lookup,
            touched_oracles,
        },
        oracles,
    ))
}

/// Verify the batch-verification proof.
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
pub fn verify_ram_batch(
    ring: &RingD,
    m: usize,
    k: usize,
    initial: &[Elem],
    final_: &[Elem],
    proof: &RamProof,
    oracles: &RamOracles,
    transcript: &mut Transcript,
) -> Result<(), SubError> {
    if initial.len() != m || final_.len() != m {
        return Err(SubError::Shape("image arity mismatch".into()));
    }
    // Statement binding: the oracle views must equal the PUBLIC
    // images (the plain-PIOP proximity discipline — the compiled layer
    // replaces these with commitments + openings).
    if oracles.initial != initial {
        return Err(SubError::Verify("oracle V ≠ public initial image".into()));
    }
    if oracles.final_ != final_ {
        return Err(SubError::Verify("oracle V' ≠ public final image".into()));
    }
    if proof.reads_oracles.b != initial {
        return Err(SubError::Verify(
            "V-isolation table ≠ public initial image".into(),
        ));
    }
    if proof.writes_oracles.b != final_ {
        return Err(SubError::Verify(
            "V'-isolation table ≠ public final image".into(),
        ));
    }
    // Lemma 6.4 record conditions (the transmitted segments):
    // 1: V_R[0:M] = 0, T_R[0:M] = 0;  2: V_W[0:M] = V;
    // 3: V_W[M+k:] = V';  8: T_W = [2M+k];  5: the ops' addresses.
    let recs = &oracles.records;
    for i in 0..m {
        if !recs.vr[i].is_zero() || recs.tr[i] != 0 {
            return Err(SubError::Verify("condition 1 violated".into()));
        }
        if recs.vw[i] != initial[i] {
            return Err(SubError::Verify("condition 2 violated".into()));
        }
        if recs.vw[m + k + i] != final_[i] {
            return Err(SubError::Verify("condition 3 violated".into()));
        }
        if recs.ar[i] != i as u64 || recs.aw[i] != i as u64 || recs.ar[m + k + i] != i as u64 {
            return Err(SubError::Verify(
                "condition 2/3 address order violated".into(),
            ));
        }
    }
    for i in 0..(2 * m + k) {
        if recs.tw[i] != (i + 1) as u64 {
            return Err(SubError::Verify("condition 8 violated".into()));
        }
    }
    for (i, op) in oracles.ops.iter().enumerate() {
        if recs.ar[m + i] != op.addr || recs.aw[m + i] != op.addr {
            return Err(SubError::Verify(
                "condition 5 (op addresses) violated".into(),
            ));
        }
    }
    // Sub-protocol 1 (both isolation lookups): the query lengths ride
    // the recorded oracle shapes.
    let n_q = proof.reads_oracles.a.len();
    crate::ring_logup::verify_ring_logup(
        ring,
        n_q,
        m,
        &proof.lookup_reads,
        &proof.reads_oracles,
        transcript,
    )?;
    let n_qw = proof.writes_oracles.a.len();
    crate::ring_logup::verify_ring_logup(
        ring,
        n_qw,
        m,
        &proof.lookup_writes,
        &proof.writes_oracles,
        transcript,
    )?;
    let canary_1 = transcript
        .challenge_bytes(b"canary1", 8)
        .unwrap_or_default();
    if canary_1 != proof.canary_1 {
        return Err(SubError::Verify(
            "canary1 mismatch (after isolation lookups)".into(),
        ));
    }
    // Sub-protocol 2: replay the α-combination challenges, then the
    // permutation logup.
    let _alpha1 = ring.sample_challenge(transcript, b"ram-al1");
    let _alpha2 = ring.sample_challenge(transcript, b"ram-al2");
    let _alpha3 = ring.sample_challenge(transcript, b"ram-al3");
    crate::ring_logup::verify_ring_logup(
        ring,
        2 * m + k,
        2 * m + k,
        &proof.perm_logup,
        &proof.perm_oracles,
        transcript,
    )?;
    let canary_2 = transcript
        .challenge_bytes(b"canary2", 8)
        .unwrap_or_default();
    if canary_2 != proof.canary_2 {
        return Err(SubError::Verify(
            "canary2 mismatch (after perm logup)".into(),
        ));
    }
    // the read/write-value Hadamard.
    verify_hadamard(
        ring,
        2 * m + k,
        &proof.hadamard,
        transcript,
        &|l, pt| match l {
            "a" => ring
                .mle_eval(&oracles.had_a, pt)
                .map_err(|e| format!("{e:?}")),
            "b" => ring
                .mle_eval(&oracles.had_b, pt)
                .map_err(|e| format!("{e:?}")),
            "c" => ring
                .mle_eval(&oracles.had_c, pt)
                .map_err(|e| format!("{e:?}")),
            _ => Err("bad label".into()),
        },
    )?;
    // timestamp positivity.
    crate::ring_logup::verify_ring_logup(
        ring,
        2 * m + k,
        2 * m + k,
        &proof.ts_lookup,
        &proof.ts_oracles,
        transcript,
    )?;
    let canary_3 = transcript
        .challenge_bytes(b"canary3", 8)
        .unwrap_or_default();
    if canary_3 != proof.canary_3 {
        return Err(SubError::Verify("canary3 mismatch (after ts logup)".into()));
    }
    // Sub-protocol 3: the almost-identical sum-check.
    let log_m = m.trailing_zeros() as usize;
    // The EQ point is sampled from the (synced) transcript; the
    // sum-check's own recorded point is internal and rides its own
    // challenge stream.
    let r_pt: Vec<Elem> = (0..log_m)
        .map(|i| ring.sample_challenge(transcript, format!("aid-r{i}").as_bytes()))
        .collect();
    let eq = ring.eq_row(&r_pt);
    let shape = crate::ring_sumcheck::RingSumcheckShape {
        num_vars: log_m,
        terms: vec![crate::ring_sumcheck::RingTermShape {
            coeff: ring.one(),
            num_factors: 3,
        }],
    };
    verify_sumcheck(
        ring,
        &shape,
        &ring.zero(),
        &proof.aid_sc,
        transcript,
        &mut |ti, fi, pt| {
            let _ = ti;
            match fi {
                0 => ring.mle_eval(&eq, pt).map_err(|e| format!("{e:?}")),
                1 => {
                    // 1 − u at the point: the verifier queries u and negates
                    let uv = ring
                        .mle_eval(&oracles.u, pt)
                        .map_err(|e| format!("{e:?}"))?;
                    Ok(ring.sub(&ring.one(), &uv))
                }
                2 => {
                    let iv = ring
                        .mle_eval(&oracles.initial, pt)
                        .map_err(|e| format!("{e:?}"))?;
                    let fv = ring
                        .mle_eval(&oracles.final_, pt)
                        .map_err(|e| format!("{e:?}"))?;
                    Ok(ring.sub(&iv, &fv))
                }
                _ => Err("bad factor".into()),
            }
        },
    )
    .map_err(SubError::from)?;
    // u binary.
    verify_binary_check(ring, m, &proof.u_binary, transcript, &|l, pt| match l {
        "bc-f" => ring.mle_eval(&oracles.u, pt).map_err(|e| format!("{e:?}")),
        other => {
            if let Some(j) = other.strip_prefix("bc-cf") {
                let j: usize = j.parse().map_err(|_| "bad index")?;
                let row: Vec<Elem> = oracles
                    .u
                    .iter()
                    .map(|e| ring.constant(e.coeffs()[j]))
                    .collect();
                return ring.mle_eval(&row, pt).map_err(|e| format!("{e:?}"));
            }
            Err(format!("unknown label {other}"))
        }
    })?;
    let canary_5 = transcript
        .challenge_bytes(b"canary5", 8)
        .unwrap_or_default();
    if canary_5 != proof.canary_5 {
        return Err(SubError::Verify(
            "canary5 mismatch (after the u binary check)".into(),
        ));
    }
    // the touched-addr lookup.
    let n_tab = proof.touched_oracles.a.len();
    crate::ring_logup::verify_ring_logup(
        ring,
        n_tab,
        n_tab,
        &proof.touched_lookup,
        &proof.touched_oracles,
        transcript,
    )?;
    let _ = initial;
    let _ = final_;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingD {
        RingD::new(4).ok().unwrap()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn build_ops(r: &RingD, m: usize, k: usize, seed: &str) -> (Vec<Elem>, Vec<RamOp>, Vec<Elem>) {
        let initial: Vec<Elem> = (0..m)
            .map(|i| r.random(format!("{seed}-i{i}").as_bytes()))
            .collect();
        let mut ops = Vec::new();
        let mut cur = initial.clone();
        for i in 0..k {
            let addr = ((i * 5 + 1) % m) as u64;
            let val = r.random(format!("{seed}-v{i}").as_bytes());
            let write = i % 2 == 0;
            ops.push(RamOp {
                write,
                addr,
                value: if write {
                    val.clone()
                } else {
                    cur[addr as usize].clone()
                },
            });
            if write {
                cur[addr as usize] = val;
            }
        }
        (initial, ops, cur)
    }

    #[test]
    fn memcheck_records_shape() {
        let r = ring();
        let (initial, ops, final_) = build_ops(&r, 4, 8, "mc");
        let recs = memcheck(&r, 4, &initial, &ops, &final_).ok().unwrap();
        assert_eq!(recs.vr.len(), 16);
        // condition 1: V_R[0:M] = 0, T_R[0:M] = 0
        for i in 0..4 {
            assert!(recs.vr[i].is_zero());
            assert_eq!(recs.tr[i], 0);
            assert_eq!(recs.ar[i], i as u64);
        }
        // condition 2: V_W[0:M] = V, T_W[0:M] = [M]
        for i in 0..4 {
            assert_eq!(recs.vw[i], initial[i]);
            assert_eq!(recs.tw[i], (i + 1) as u64);
        }
        // condition 3: V_W[M+k:] = V'
        for i in 0..4 {
            assert_eq!(recs.vw[12 + i], final_[i]);
            assert_eq!(recs.ar[12 + i], i as u64);
        }
        // condition 8: T_W = [2M+k]
        for i in 0..16 {
            assert_eq!(recs.tw[i], (i + 1) as u64);
        }
        // condition 9: T_W − T_R > 0
        for i in 0..16 {
            assert!(recs.tw[i] > recs.tr[i]);
        }
    }

    #[test]
    fn memcheck_rejects_bad_read() {
        let r = ring();
        let (initial, mut ops, final_) = build_ops(&r, 4, 8, "br");
        // corrupt a read value
        for op in ops.iter_mut() {
            if !op.write {
                op.value = r.add(&op.value, &r.one());
                break;
            }
        }
        assert!(memcheck(&r, 4, &initial, &ops, &final_).is_err());
    }

    #[test]
    fn ram_batch_end_to_end() {
        let r = ring();
        let (initial, ops, final_) = build_ops(&r, 4, 8, "rb");
        let mut tr = Transcript::new_default(b"ram");
        let (proof, oracles) = prove_ram_batch(&r, &initial, &ops, &final_, &mut tr)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut tr2 = Transcript::new_default(b"ram");
        verify_ram_batch(&r, 4, 8, &initial, &final_, &proof, &oracles, &mut tr2)
            .unwrap_or_else(|e| panic!("verify: {e:?}"));
    }

    #[test]
    fn ram_batch_prover_rejects_wrong_final() {
        let r = ring();
        let (initial, ops, mut final_) = build_ops(&r, 4, 8, "wf");
        final_[0] = r.add(&final_[0], &r.one());
        let mut tr = Transcript::new_default(b"ram2");
        assert!(prove_ram_batch(&r, &initial, &ops, &final_, &mut tr).is_err());
    }

    #[test]
    fn ram_batch_verify_rejects_tampered() {
        let r = ring();
        let (initial, ops, final_) = build_ops(&r, 4, 8, "tv");
        let (proof, mut oracles) = {
            let mut tr = Transcript::new_default(b"ram3");
            let (p, o) = prove_ram_batch(&r, &initial, &ops, &final_, &mut tr)
                .ok()
                .unwrap();
            (p, o)
        };
        // tamper the initial image: the aid sum-check catches it
        oracles.initial[1] = r.add(&oracles.initial[1], &r.one());
        let mut tr = Transcript::new_default(b"ram3");
        assert!(verify_ram_batch(&r, 4, 8, &initial, &final_, &proof, &oracles, &mut tr).is_err());
    }
}
