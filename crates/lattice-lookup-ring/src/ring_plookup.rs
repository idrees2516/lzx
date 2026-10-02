//! Ring-Plookup (Construction 5.5 of ePrint 2026/471): the indexed
//! lookup PIOP over the split ring, generalizing Plookup [GW20] via
//! the [BLMG25] tag technique.
//!
//! The relation (Lemma 5.4): with the challenge space `C`, the
//! injective map `g: [N] → C`, the merge witness `w ∈ merge_b(a)` and
//! its tag vector `σ`, the trivariate identity
//!
//! ```text
//! ∏_{k}(w_k + shift(w)_k·x₁ + σ_k·x₂ − x₃)
//!     = ∏_{i}(a_i + a_i·x₁ + c_i·x₂ − x₃) · ∏_{j}(b_j + shift(b)_j·x₁ + g(j)·x₂ − x₃)
//! ```
//!
//! holds iff `a` is a valid *indexed* lookup into `b` with tags `c`
//! (the tags pin the per-CRT-component index order — closing the
//! Section-4 CRT-swap attack).
//!
//! The PIOP evaluates the identity at `(α, γ, β) ← C³` through the
//! combined vectors `a* = (1+α)a + γc − β`,
//! `b* = b + α·b⟳ + γ·g([N]) − β`, `w* = w + α·w⟳ + γ·σ − β` with
//! entry products and the top-level check `χ_w* = χ_a*·χ_b*`.
//! (Appendix D's "(1+β)^M" factor is a typo carried over from the
//! original Plookup normalization — the `(1+α)` is already absorbed
//! into `a*`; the main construction text's check is the correct one
//! and the one implemented.)
//!
//! Round flow: oracles `(b⟳, w, w⟳)` → challenges `(α, γ, β)` →
//! `(a*, b*, w*)` + `(χ_a*, χ_b*, χ_w*)` → three entry-product
//! protocols → the `χ` check → random points `(η₁, η₂, η₃)` with the
//! MLE linearity checks → two cyclic-shift tests → the binary check
//! on `c`.
//!
//! The plain-PIOP verifier consumes a [`PlookupOracles`] — the full
//! IOP-of-proximity witness vectors — mirroring the prover's
//! transcript absorptions; the compiled layer (`compile`) replaces
//! this with Ajtai commitments + windowed openings.


// (Kernel loops use explicit indices by convention.)
#![allow(clippy::needless_range_loop)]
use crate::ring_d::{Elem, RingD};
use crate::subprotocols::{
    prove_binary_check, prove_cyclic_shift, prove_entry_product, verify_binary_check,
    verify_cyclic_shift, verify_entry_product, SubError,
};
use lattice_core::transcript::Transcript;

/// Prefix-product vectors of an entry-product instance.
#[derive(Debug, Clone)]
pub struct EntryProductVectors {
    pub c: Vec<Elem>,
    pub d: Vec<Elem>,
    pub e: Vec<Elem>,
}

/// All oracle vectors of a Ring-Plookup instance (the plain-PIOP
/// verifier's witness-side view).
#[derive(Debug, Clone)]
pub struct PlookupOracles {
    pub a: Vec<Elem>,
    pub b: Vec<Elem>,
    pub c: Vec<Elem>,
    pub w: Vec<Elem>,
    pub sigma: Vec<Elem>,
    pub b_cyc: Vec<Elem>,
    pub w_cyc: Vec<Elem>,
    pub g_vec: Vec<Elem>,
    pub a_star: Vec<Elem>,
    pub b_star: Vec<Elem>,
    pub w_star: Vec<Elem>,
    pub ep_a: EntryProductVectors,
    pub ep_b: EntryProductVectors,
    pub ep_w: EntryProductVectors,
    /// CF rows of the binary check on `c`.
    pub cf_rows: Vec<Vec<Elem>>,
}

impl PlookupOracles {
    /// MLE evaluation by label (`pl-*` names).
    pub fn eval(&self, ring: &RingD, label: &str, point: &[Elem]) -> Result<Elem, String> {
        let v: &[Elem] = match label {
            "pl-a" => &self.a,
            "pl-b" => &self.b,
            "pl-c" => &self.c,
            "pl-w" => &self.w,
            "pl-sigma" => &self.sigma,
            "pl-b-cyc" => &self.b_cyc,
            "pl-w-cyc" => &self.w_cyc,
            "pl-gN" => &self.g_vec,
            "pl-a-star" => &self.a_star,
            "pl-b-star" => &self.b_star,
            "pl-w-star" => &self.w_star,
            "pl-ep-a" => &self.a_star,
            "pl-ep-b" => &self.ep_a.c,
            "pl-ep-c" | "pl-ep-d" => &self.ep_a.d,
            "pl-ep-e" => &self.ep_a.e,
            "pl-ep2-a" => &self.b_star,
            "pl-ep2-b" => &self.ep_b.c,
            "pl-ep2-c" | "pl-ep2-d" => &self.ep_b.d,
            "pl-ep2-e" => &self.ep_b.e,
            "pl-ep3-a" => &self.w_star,
            "pl-ep3-b" => &self.ep_w.c,
            "pl-ep3-c" | "pl-ep3-d" => &self.ep_w.d,
            "pl-ep3-e" => &self.ep_w.e,
            _ => {
                if let Some(j) = label.strip_prefix("pl-bc-cf") {
                    let j: usize = j.parse().map_err(|_| "bad cf index")?;
                    return ring
                        .mle_eval(&self.cf_rows[j], point)
                        .map_err(|e| format!("{e:?}"));
                }
                return Err(format!("unknown oracle label {label}"));
            }
        };
        ring.mle_eval(v, point).map_err(|e| format!("{e:?}"))
    }
}

#[derive(Debug, Clone)]
pub struct RingPlookupProof {
    pub alpha: Elem,
    pub gamma: Elem,
    pub beta: Elem,
    pub chi_a: Elem,
    pub chi_b: Elem,
    pub chi_w: Elem,
    pub ep_a: crate::subprotocols::EntryProductProof,
    pub ep_b: crate::subprotocols::EntryProductProof,
    pub ep_w: crate::subprotocols::EntryProductProof,
    pub eta1: Vec<Elem>,
    pub eta2: Vec<Elem>,
    pub eta3: Vec<Elem>,
    pub shift_b: crate::subprotocols::CyclicShiftProof,
    pub shift_w: crate::subprotocols::CyclicShiftProof,
    pub binary: crate::subprotocols::BinaryCheckProof,
}

/// Absorb a vector oracle into the transcript.
fn absorb_vec(ring: &RingD, label: &[u8], v: &[Elem], tr: &mut Transcript) {
    let mut buf = Vec::with_capacity(v.len() * ring.d * 8);
    for e in v {
        for &c in e.coeffs() {
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }
    let _ = tr.append_bytes(label, &buf);
}

/// The merge construction's output: `(w, σ, freq)`.
pub type MergeWitness = (Vec<Elem>, Vec<Elem>, Vec<usize>);

/// Value equality helper (indirection to keep the merge matcher
/// readable against the pair check).
fn b_value_matches(_ring: &RingD, a: &Elem, b: &Elem) -> bool {
    a == b
}

/// The merge construction: `w ∈ merge_b(a)` with left-insertion and
/// the accompanying tag vector `σ`. Returns `(w, σ, freq)`.
pub fn build_merge(
    ring: &RingD,
    a: &[Elem],
    b: &[Elem],
    c: &[Elem],
) -> Result<MergeWitness, SubError> {
    let m = a.len();
    let n = b.len();
    let mut freq = vec![1usize; n];
    for (i, ci) in c.iter().enumerate() {
        let mut matched = false;
        for (j, bj) in b.iter().enumerate() {
            let gj = ring.g_map(j as u64);
            if *ci == gj && b_value_matches(ring, &a[i], bj) {
                freq[j] += 1;
                matched = true;
                break;
            }
        }
        if !matched {
            return Err(SubError::Verify("tag c_i outside g([N]) or value mismatch".into()));
        }
    }
    let mut w = Vec::with_capacity(m + n);
    let mut sigma = Vec::with_capacity(m + n);
    for (j, bj) in b.iter().enumerate() {
        let gj = ring.g_map(j as u64);
        for (i, ai) in a.iter().enumerate() {
            if c[i] == gj {
                w.push(ai.clone());
                sigma.push(gj.clone());
            }
        }
        w.push(bj.clone());
        sigma.push(gj.clone());
    }
    Ok((w, sigma, freq))
}

fn prefix_vectors(ring: &RingD, v: &[Elem]) -> EntryProductVectors {
    let n = v.len();
    let mut c = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    let mut run = ring.one();
    for e in v {
        c.push(run.clone());
        run = ring.mul(&run, e);
        d.push(run.clone());
    }
    let mut e = c[1..].to_vec();
    e.push(ring.one());
    EntryProductVectors { c, d, e }
}

/// Prove the indexed lookup `(a, b, c) ∈ RILU`. Returns the proof and
/// the full oracle view (plain-PIOP semantics).
#[allow(clippy::too_many_lines)]
pub fn prove_ring_plookup(
    ring: &RingD,
    a: &[Elem],
    b: &[Elem],
    c: &[Elem],
    transcript: &mut Transcript,
) -> Result<(RingPlookupProof, PlookupOracles), SubError> {
    let m = a.len();
    let n = b.len();
    if c.len() != m {
        return Err(SubError::Shape("c length mismatch".into()));
    }
    if !n.is_power_of_two() || !m.is_power_of_two() || (m + n).count_ones() != 1 {
        return Err(SubError::Shape("M, N, M+N must be powers of two".into()));
    }
    for (i, ai) in a.iter().enumerate() {
        let ci = &c[i];
        if !ci.is_binary() {
            return Err(SubError::Verify("c_i not in C".into()));
        }
        let ok = b.iter().enumerate().any(|(j, bj)| ai == bj && *ci == ring.g_map(j as u64));
        if !ok {
            return Err(SubError::Verify(format!("a_{i} has no matching (b_j, g(j)) pair")));
        }
    }
    for j in 0..n {
        for k in (j + 1)..n {
            if b[j] == b[k] {
                return Err(SubError::Verify("table entries must be unique".into()));
            }
        }
    }
    let (w, sigma, _freq) = build_merge(ring, a, b, c)?;
    let b_cyc: Vec<Elem> = (0..n).map(|i| b[(i + 1) % n].clone()).collect();
    let w_cyc: Vec<Elem> = (0..(m + n)).map(|i| w[(i + 1) % (m + n)].clone()).collect();
    let g_vec: Vec<Elem> = (0..n).map(|j| ring.g_map(j as u64)).collect();

    // Round 1: oracles (b⟳, w, w⟳) absorbed.
    absorb_vec(ring, b"pl-b-cyc", &b_cyc, transcript);
    absorb_vec(ring, b"pl-w", &w, transcript);
    absorb_vec(ring, b"pl-w-cyc", &w_cyc, transcript);
    // Round 2: challenges α, γ, β ∈ C.
    let alpha = ring.sample_challenge(transcript, b"pl-alpha");
    let gamma = ring.sample_challenge(transcript, b"pl-gamma");
    let beta = ring.sample_challenge(transcript, b"pl-beta");
    // Round 3: a*, b*, w* and their entry products.
    let one_plus_alpha = ring.add(&ring.one(), &alpha);
    let a_star: Vec<Elem> = (0..m)
        .map(|i| {
            let t1 = ring.mul(&one_plus_alpha, &a[i]);
            let t2 = ring.mul(&gamma, &c[i]);
            ring.sub(&ring.add(&t1, &t2), &beta)
        })
        .collect();
    let b_star: Vec<Elem> = (0..n)
        .map(|j| {
            let t1 = ring.mul(&alpha, &b_cyc[j]);
            let t2 = ring.mul(&gamma, &g_vec[j]);
            let s = ring.add(&b[j], &ring.add(&t1, &t2));
            ring.sub(&s, &beta)
        })
        .collect();
    let w_star: Vec<Elem> = (0..(m + n))
        .map(|k| {
            let t1 = ring.mul(&alpha, &w_cyc[k]);
            let t2 = ring.mul(&gamma, &sigma[k]);
            let s = ring.add(&w[k], &ring.add(&t1, &t2));
            ring.sub(&s, &beta)
        })
        .collect();
    let mut chi_a = ring.one();
    for e in &a_star {
        chi_a = ring.mul(&chi_a, e);
    }
    let mut chi_b = ring.one();
    for e in &b_star {
        chi_b = ring.mul(&chi_b, e);
    }
    let mut chi_w = ring.one();
    for e in &w_star {
        chi_w = ring.mul(&chi_w, e);
    }
    if chi_w != ring.mul(&chi_a, &chi_b) {
        return Err(SubError::Verify("completeness: χ_w ≠ χ_a·χ_b".into()));
    }
    absorb_vec(ring, b"pl-a-star", &a_star, transcript);
    absorb_vec(ring, b"pl-b-star", &b_star, transcript);
    absorb_vec(ring, b"pl-w-star", &w_star, transcript);
    // Round 4: the three entry-product protocols.
    let (ep_a, _qs_a) = prove_entry_product(ring, &a_star, &chi_a, transcript)?;
    let (ep_b, _qs_b) = prove_entry_product(ring, &b_star, &chi_b, transcript)?;
    let (ep_w, _qs_w) = prove_entry_product(ring, &w_star, &chi_w, transcript)?;
    // Round 5: η points.
    let log_m = m.trailing_zeros() as usize;
    let log_n = n.trailing_zeros() as usize;
    let log_mn = (m + n).trailing_zeros() as usize;
    let eta1: Vec<Elem> = (0..log_m)
        .map(|i| ring.sample_challenge(transcript, format!("pl-eta1-{i}").as_bytes()))
        .collect();
    let eta2: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("pl-eta2-{i}").as_bytes()))
        .collect();
    let eta3: Vec<Elem> = (0..log_mn)
        .map(|i| ring.sample_challenge(transcript, format!("pl-eta3-{i}").as_bytes()))
        .collect();
    // Round 6: cyclic-shift tests.
    let (shift_b, _qs_sb) = prove_cyclic_shift(ring, b, &b_cyc, transcript)?;
    let (shift_w, _qs_sw) = prove_cyclic_shift(ring, &w, &w_cyc, transcript)?;
    // Round 7: binary check on c.
    let (binary, _qs_bin) = prove_binary_check(ring, c, transcript)?;

    let cf_rows: Vec<Vec<Elem>> = (0..ring.d)
        .map(|j| c.iter().map(|e| ring.constant(e.coeffs()[j])).collect())
        .collect();
    let epv_a = prefix_vectors(ring, &a_star);
    let epv_b = prefix_vectors(ring, &b_star);
    let epv_w = prefix_vectors(ring, &w_star);

    let oracles = PlookupOracles {
        a: a.to_vec(),
        b: b.to_vec(),
        c: c.to_vec(),
        w,
        sigma,
        b_cyc,
        w_cyc,
        g_vec,
        a_star,
        b_star,
        w_star,
        ep_a: epv_a,
        ep_b: epv_b,
        ep_w: epv_w,
        cf_rows,
    };
    Ok((
        RingPlookupProof {
            alpha,
            gamma,
            beta,
            chi_a,
            chi_b,
            chi_w,
            ep_a,
            ep_b,
            ep_w,
            eta1,
            eta2,
            eta3,
            shift_b,
            shift_w,
            binary,
        },
        oracles,
    ))
}

/// Verify the Ring-Plookup proof against the oracle view (the plain
/// PIOP verifier). The transcript is replayed in the prover's order:
/// oracle absorptions, `(α, γ, β)`, entry products, η's, shifts,
/// binary check.
#[allow(clippy::too_many_lines)]
pub fn verify_ring_plookup(
    ring: &RingD,
    m: usize,
    n: usize,
    proof: &RingPlookupProof,
    oracles: &PlookupOracles,
    transcript: &mut Transcript,
) -> Result<(), SubError> {
    if !n.is_power_of_two() || !m.is_power_of_two() || (m + n).count_ones() != 1 {
        return Err(SubError::Shape("M, N, M+N must be powers of two".into()));
    }
    // Replay rounds 1-3 absorptions.
    absorb_vec(ring, b"pl-b-cyc", &oracles.b_cyc, transcript);
    absorb_vec(ring, b"pl-w", &oracles.w, transcript);
    absorb_vec(ring, b"pl-w-cyc", &oracles.w_cyc, transcript);
    let alpha = ring.sample_challenge(transcript, b"pl-alpha");
    let gamma = ring.sample_challenge(transcript, b"pl-gamma");
    let beta = ring.sample_challenge(transcript, b"pl-beta");
    if alpha != proof.alpha || gamma != proof.gamma || beta != proof.beta {
        return Err(SubError::Verify("challenge replay mismatch".into()));
    }
    absorb_vec(ring, b"pl-a-star", &oracles.a_star, transcript);
    absorb_vec(ring, b"pl-b-star", &oracles.b_star, transcript);
    absorb_vec(ring, b"pl-w-star", &oracles.w_star, transcript);
    // Round 4: entry products.
    verify_entry_product(ring, m, &proof.chi_a, &proof.ep_a, transcript, &|l, pt| {
        // l arrives already mapped by verify_entry_product: "ep-a",
        // "ep-b", "ep-c"(=d), "ep-d", "ep-e", "ep-shift-a/b".
        oracles.eval(ring, &format!("pl-{l}"), pt)
    })?;
    verify_entry_product(ring, n, &proof.chi_b, &proof.ep_b, transcript, &|l, pt| {
        let mapped = l.replace("ep-", "ep2-");
        oracles.eval(ring, &format!("pl-{mapped}"), pt)
    })?;
    verify_entry_product(ring, m + n, &proof.chi_w, &proof.ep_w, transcript, &|l, pt| {
        let mapped = l.replace("ep-", "ep3-");
        oracles.eval(ring, &format!("pl-{mapped}"), pt)
    })?;
    // The χ check.
    if ring.mul(&proof.chi_a, &proof.chi_b) != proof.chi_w {
        return Err(SubError::Verify("χ_w ≠ χ_a·χ_b".into()));
    }
    // Round 5: η points + consistency checks.
    let log_m = m.trailing_zeros() as usize;
    let log_n = n.trailing_zeros() as usize;
    let log_mn = (m + n).trailing_zeros() as usize;
    let eta1: Vec<Elem> = (0..log_m)
        .map(|i| ring.sample_challenge(transcript, format!("pl-eta1-{i}").as_bytes()))
        .collect();
    let eta2: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("pl-eta2-{i}").as_bytes()))
        .collect();
    let eta3: Vec<Elem> = (0..log_mn)
        .map(|i| ring.sample_challenge(transcript, format!("pl-eta3-{i}").as_bytes()))
        .collect();
    if eta1 != proof.eta1 || eta2 != proof.eta2 || eta3 != proof.eta3 {
        return Err(SubError::Verify("η replay mismatch".into()));
    }
    let ev = |label: &str, pt: &[Elem]| oracles.eval(ring, label, pt).map_err(SubError::Verify);
    {
        let av = ev("pl-a", &eta1)?;
        let cv = ev("pl-c", &eta1)?;
        let asv = ev("pl-a-star", &eta1)?;
        let t1 = ring.mul(&ring.add(&ring.one(), &alpha), &av);
        let t2 = ring.mul(&gamma, &cv);
        let expect = ring.sub(&ring.add(&t1, &t2), &beta);
        if asv != expect {
            return Err(SubError::Verify("a* consistency failed".into()));
        }
    }
    {
        let bv = ev("pl-b", &eta2)?;
        let bcv = ev("pl-b-cyc", &eta2)?;
        let gv = ev("pl-gN", &eta2)?;
        let bsv = ev("pl-b-star", &eta2)?;
        let t1 = ring.mul(&alpha, &bcv);
        let t2 = ring.mul(&gamma, &gv);
        let expect = ring.sub(&ring.add(&bv, &ring.add(&t1, &t2)), &beta);
        if bsv != expect {
            return Err(SubError::Verify("b* consistency failed".into()));
        }
    }
    {
        let wv = ev("pl-w", &eta3)?;
        let wcv = ev("pl-w-cyc", &eta3)?;
        let sv = ev("pl-sigma", &eta3)?;
        let wsv = ev("pl-w-star", &eta3)?;
        let t1 = ring.mul(&alpha, &wcv);
        let t2 = ring.mul(&gamma, &sv);
        let expect = ring.sub(&ring.add(&wv, &ring.add(&t1, &t2)), &beta);
        if wsv != expect {
            return Err(SubError::Verify("w* consistency failed".into()));
        }
    }
    // Round 6: cyclic shifts.
    verify_cyclic_shift(ring, n, &proof.shift_b, transcript, &|l, pt| match l {
        "shift-a" => oracles.eval(ring, "pl-b", pt),
        "shift-b" => oracles.eval(ring, "pl-b-cyc", pt),
        _ => Err("bad label".into()),
    })?;
    verify_cyclic_shift(ring, m + n, &proof.shift_w, transcript, &|l, pt| match l {
        "shift-a" => oracles.eval(ring, "pl-w", pt),
        "shift-b" => oracles.eval(ring, "pl-w-cyc", pt),
        _ => Err("bad label".into()),
    })?;
    // Round 7: binary check on c.
    verify_binary_check(ring, m, &proof.binary, transcript, &|l, pt| match l {
        "bc-f" => oracles.eval(ring, "pl-c", pt),
        other => oracles.eval(ring, &format!("pl-{other}"), pt),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingD {
        RingD::new(4).ok().unwrap()
    }

    fn build_case(r: &RingD, m: usize, n: usize, seed: &str) -> (Vec<Elem>, Vec<Elem>, Vec<Elem>) {
        let b: Vec<Elem> = (0..n).map(|j| r.random(format!("{seed}-b{j}").as_bytes())).collect();
        let mut a = Vec::with_capacity(m);
        let mut c = Vec::with_capacity(m);
        for i in 0..m {
            let j = (i * 7 + 1) % n;
            a.push(b[j].clone());
            c.push(r.g_map(j as u64));
        }
        (a, b, c)
    }

    #[test]
    fn plookup_end_to_end() {
        let r = ring();
        let (a, b, c) = build_case(&r, 4, 4, "rt");
        let mut tr = Transcript::new_default(b"pl");
        let (proof, oracles) =
            prove_ring_plookup(&r, &a, &b, &c, &mut tr).unwrap_or_else(|e| panic!("{e:?}"));
        let mut tr2 = Transcript::new_default(b"pl");
        verify_ring_plookup(&r, 4, 4, &proof, &oracles, &mut tr2)
            .unwrap_or_else(|e| panic!("verify: {e:?}"));
    }

    #[test]
    fn plookup_larger_instance() {
        let r = ring();
        let (a, b, c) = build_case(&r, 8, 8, "big");
        let mut tr = Transcript::new_default(b"pl5");
        let (proof, oracles) = prove_ring_plookup(&r, &a, &b, &c, &mut tr).ok().unwrap();
        let mut tr2 = Transcript::new_default(b"pl5");
        verify_ring_plookup(&r, 8, 8, &proof, &oracles, &mut tr2).ok().unwrap();
    }

    #[test]
    fn plookup_tampered_oracles_rejected() {
        let r = ring();
        let (a, b, c) = build_case(&r, 4, 4, "tam");
        let mut tr = Transcript::new_default(b"pl");
        let (proof, mut oracles) = prove_ring_plookup(&r, &a, &b, &c, &mut tr).ok().unwrap();
        // Tamper the w vector: the shift test / entry product fails.
        oracles.w[3] = r.add(&oracles.w[3], &r.one());
        let mut tr2 = Transcript::new_default(b"pl");
        assert!(verify_ring_plookup(&r, 4, 4, &proof, &oracles, &mut tr2).is_err());
        // Tamper the a* vector: the entry product for a* fails.
        let (proof2, mut oracles2) = {
            let mut tr = Transcript::new_default(b"pl");
            let (p, o) = prove_ring_plookup(&r, &a, &b, &c, &mut tr).ok().unwrap();
            (p, o)
        };
        oracles2.a_star[1] = r.add(&oracles2.a_star[1], &r.one());
        let mut tr3 = Transcript::new_default(b"pl");
        assert!(verify_ring_plookup(&r, 4, 4, &proof2, &oracles2, &mut tr3).is_err());
        // Tamper the c tags to non-binary: binary check fails (also the
        // χ products shift).
        let (proof3, mut oracles3) = {
            let mut tr = Transcript::new_default(b"pl");
            let (p, o) = prove_ring_plookup(&r, &a, &b, &c, &mut tr).ok().unwrap();
            (p, o)
        };
        oracles3.c[0] = r.constant(2);
        let mut tr4 = Transcript::new_default(b"pl");
        assert!(verify_ring_plookup(&r, 4, 4, &proof3, &oracles3, &mut tr4).is_err());
    }

    #[test]
    fn plookup_prover_rejects_invalid_lookups() {
        let r = ring();
        let (a, b, c) = build_case(&r, 4, 4, "bad");
        let mut a_bad = a.clone();
        a_bad[1] = r.add(&a_bad[1], &r.one());
        let mut tr = Transcript::new_default(b"pl2");
        assert!(prove_ring_plookup(&r, &a_bad, &b, &c, &mut tr).is_err());
        let mut c_bad = c.clone();
        c_bad[0] = r.g_map(999);
        let mut tr2 = Transcript::new_default(b"pl3");
        assert!(prove_ring_plookup(&r, &a, &b, &c_bad, &mut tr2).is_err());
        let mut c_nb = c.clone();
        c_nb[2] = r.constant(2);
        let mut tr3 = Transcript::new_default(b"pl4");
        assert!(prove_ring_plookup(&r, &a, &b, &c_nb, &mut tr3).is_err());
    }

    #[test]
    fn merge_is_sorted_by_b() {
        let r = ring();
        let (a, b, c) = build_case(&r, 4, 4, "mg");
        let (w, sigma, freq) = build_merge(&r, &a, &b, &c).ok().unwrap();
        assert_eq!(w.len(), 8);
        assert_eq!(sigma.len(), 8);
        assert_eq!(freq.iter().sum::<usize>(), 8);
        let tag_int = |e: &Elem| {
            e.coeffs().iter().enumerate().fold(0u64, |acc, (i, &c)| acc | (c << i))
        };
        for k in 1..w.len() {
            assert!(tag_int(&sigma[k - 1]) <= tag_int(&sigma[k]));
        }
        for ai in &a {
            assert!(w.iter().any(|x| x == ai));
        }
    }
}
