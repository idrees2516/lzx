//! The LaBRADOR main protocol (Figure 2 prover / Figure 3 verifier, §5.2–§5.6):
//! one recursion level of the proof system for the principal relation.
//!
//! Level flow (paper-faithful; the reference's structure where the paper is
//! silent):
//!
//! 1. **Join** the input witness vectors into `r` amortized parts of rank ≤ nn
//!    (linear statements: one contiguous block; quadratic: per-vector blocks).
//! 2. **Commit**: inner commitments `t_i = A·s'_i` (rank κ), digit-decomposed
//!    (fu × bu, flat layout `[i][j][ρ]`); quadratic garbage `g_ij = ⟨s'_i,s'_j⟩`
//!    (fg × bg, layout `[pair][k]`); first outer commitment `u1 = B·[t̃;g̃]`.
//! 3. **Project** (JL, §4): ±1 matrices with rejection, `p` (256 values).
//! 4. **Lift** (LIFTS = ⌈128/log q⌉ = 4 rounds): collapse the 256 JL rows and
//!    the F' family with ψ ∈ Z_q^L, ω ∈ Z_q^256; the prover sends `b''(k)`
//!    (constant terms checked) and the lifted constraints become vanishing.
//! 5. **Aggregate** (§5.2): uniform α ∈ R_q^K, β ∈ R_q^4 fold F ⊎ F'' into one
//!    constraint (φ_agg, a_agg, b_agg) over the parts.
//! 6. **Amortize**: linear garbage `h_ij = ½(⟨φ_i,s'_j⟩+⟨φ_j,s'_i⟩)` (fu × bu,
//!    layout `[pair][k]`), second outer commitment `u2 = D·h̃`, challenges
//!    `c_i ← C`, opening `z = Σ c_i s'_i` (rank nn) decomposed into f × b
//!    digits.
//! 7. **Target relation** (§5.3): the output statement over `[z^(0..f-1), v]`
//!    with the K' = 2κ1+κ+3 constraints E1–E6. The tail (§5.6) transmits the
//!    openings directly with 2r−1 interleaved garbage terms.
//!
//! Honest deviations (full ledger in `docs/papers/implemented/labrador.md`):
//! uniform R_q α/β per Theorem 5.1 (the reference pre-folds with quarternary
//! challenges); E4's full digit-cross quadratic structure (chunking-invariant);
//! power-of-two bases; disjoint key windows.

use crate::challenge::{challenge_vec, is_challenge, uniform_rq_vec, zq_scalars, INV2};
use crate::jl::{collapse_jl, jl_accept, jl_normsq, JlMatrix, JlMode, JlProjection};
use crate::relation::{DotCnst, PrincipalStatement, PrincipalWitness, Term, VectorSpec};
use crate::ring::{sprod, Poly, N};
use crate::sis::{init_proof, jl_max_normsq, ComKey, ComParams, LIFTS};
use crate::transcript::Transcript;

/// Triangular pair ordinal for (i, j), i ≤ j, over r parts.
pub fn tri_idx(i: usize, j: usize, r: usize) -> usize {
    let (i, j) = if i > j { (j, i) } else { (i, j) };
    i * r - (i * i + i) / 2 + j
}

/// The v-vector layout: [t̃ (r·fu·κ), g̃ (fg·(r²+r)/2), h̃ (fu·(r²+r)/2)].
#[derive(Clone, Copy, Debug)]
pub struct VLayout {
    pub t_len: usize,
    pub g_len: usize,
    pub h_len: usize,
    pub m: usize,
}

impl VLayout {
    pub fn new(cpp: &ComParams, r: usize) -> Self {
        let pairs = (r * r + r) / 2;
        Self {
            t_len: r * cpp.fu * cpp.kappa,
            g_len: cpp.fg * pairs,
            h_len: cpp.fu * pairs,
            m: r * cpp.fu * cpp.kappa + (cpp.fg + cpp.fu) * pairs,
        }
    }
    pub fn g_off(&self) -> usize {
        self.t_len
    }
    pub fn h_off(&self) -> usize {
        self.t_len + self.g_len
    }
}

/// The joined-part layout (verifier-regenerable from ranks + params).
#[derive(Clone, Debug)]
pub struct PartLayout {
    pub nn: usize,
    pub r: usize,
    pub ranks: Vec<usize>,
    /// Per input vector: (first part, offset within it).
    pub starts: Vec<(usize, usize)>,
    /// Per part: originating input vector (None for padding-only parts).
    pub origin: Vec<Option<usize>>,
}

/// Compute the joined-part layout.
pub fn part_layout(ranks: &[usize], nn: usize, per_vector: bool) -> PartLayout {
    let mut starts = Vec::with_capacity(ranks.len());
    let mut origin: Vec<Option<usize>> = Vec::new();
    let mut filled = 0usize;
    let mut open_origin: Option<usize> = None;
    for (vi, &n) in ranks.iter().enumerate() {
        if per_vector {
            if filled > 0 {
                origin.push(open_origin.take());
            }
            starts.push((origin.len(), 0));
            let mut rest = n;
            while rest > nn {
                origin.push(Some(vi));
                rest -= nn;
            }
            filled = rest;
            open_origin = Some(vi);
        } else {
            starts.push((origin.len(), filled));
            if open_origin.is_none() {
                open_origin = Some(vi);
            }
            let mut rest = n;
            while filled + rest > nn {
                let take = nn - filled;
                rest -= take;
                origin.push(open_origin.take());
                filled = 0;
                open_origin = Some(vi);
            }
            filled += rest;
            let _ = &filled;
        }
    }
    if filled > 0 || origin.is_empty() {
        origin.push(open_origin);
    }
    PartLayout {
        nn,
        r: origin.len(),
        ranks: ranks.to_vec(),
        starts,
        origin,
    }
}

impl PartLayout {
    /// The padded parts from a witness (prover side).
    pub fn materialize(&self, wit: &PrincipalWitness) -> Vec<Vec<Poly>> {
        let mut flat: Vec<Poly> = Vec::with_capacity(self.r * self.nn);
        for v in &wit.s {
            flat.extend(v.iter().copied());
        }
        while flat.len() < self.r * self.nn {
            flat.push(Poly::zero());
        }
        (0..self.r)
            .map(|i| flat[i * self.nn..(i + 1) * self.nn].to_vec())
            .collect()
    }

    /// (vector, offset) → (part, offset in part).
    pub fn locate(&self, idx: usize, off: usize) -> (usize, usize) {
        let (mut part, mut pos) = self.starts[idx];
        let mut remaining = off;
        loop {
            let room = self.nn - pos;
            if remaining < room {
                return (part, pos + remaining);
            }
            remaining -= room;
            part += 1;
            pos = 0;
        }
    }

    /// Split a phi slice into per-part chunks.
    pub fn locate_slice(
        &self,
        idx: usize,
        off: usize,
        phi: &[Poly],
    ) -> Vec<(usize, usize, Vec<Poly>)> {
        let mut out = Vec::new();
        let (mut part, mut pos) = self.locate(idx, off);
        let mut phi_rest = phi;
        loop {
            let room = self.nn - pos;
            if phi_rest.len() <= room {
                if !phi_rest.is_empty() {
                    out.push((part, pos, phi_rest.to_vec()));
                }
                return out;
            }
            out.push((part, pos, phi_rest[..room].to_vec()));
            phi_rest = &phi_rest[room..];
            part += 1;
            pos = 0;
        }
    }

    /// The part-span of input vector vi: (start, end).
    pub fn span(&self, vi: usize) -> (usize, usize) {
        let (start, _) = self.starts[vi];
        let rank = self.ranks[vi];
        if rank == 0 {
            return (start, start.max(1).min(self.r));
        }
        let (end_part, _) = self.locate(vi, rank - 1);
        (start, end_part + 1)
    }
}

/// Expand an input a-entry (i, j) to part-level entries with the POSITIONAL
/// chunk pairing: ⟨s_i, s_j⟩ = Σ_x ⟨chunk_x^i, chunk_x^j⟩ (the ring inner
/// product pairs coefficients positionally — the concatenation blocks pair at
/// the same chunk offset, NOT as a double sum; this is the reference's
/// same-index structure and the correct expansion of the paper's equations).
/// Only called in the quadratic mode (per-vector boundaries), where vector
/// i's chunks are the parts [starts[i].0, starts[i].0 + ceil(rank/nn)).
fn expand_a_entry(
    layout: &PartLayout,
    i: usize,
    j: usize,
    coeff: &Poly,
) -> Vec<(usize, usize, Poly)> {
    let (si, _) = layout.starts[i];
    let (sj, _) = layout.starts[j];
    let ni = layout.ranks[i].div_ceil(layout.nn);
    let nj = layout.ranks[j].div_ceil(layout.nn);
    let n = ni.min(nj).max(1);
    let mut out = Vec::with_capacity(n);
    for x in 0..n {
        let p = si + x;
        let q = sj + x;
        out.push((p.min(q), p.max(q), *coeff));
    }
    out
}

/// The quadratic-ness of a statement.
pub fn is_quadratic(st: &PrincipalStatement) -> bool {
    st.cnst
        .iter()
        .chain(st.ct_cnst.iter())
        .any(|c| !c.a.is_empty())
}

/// Decompose a vector into f digit-vectors (transposed).
pub fn decompose_vec(z: &[Poly], f: usize, b: u32) -> Vec<Vec<Poly>> {
    let mut out = vec![Vec::with_capacity(z.len()); f];
    for p in z {
        for (d, digit) in p.decompose(f, b).into_iter().enumerate() {
            out[d].push(digit);
        }
    }
    out
}

/// Recombine: z[k] = Σ_d 2^{d·b}·parts[d][k].
pub fn recombine_vec(parts: &[Vec<Poly>], b: u32) -> Vec<Poly> {
    let n = parts.first().map(|v| v.len()).unwrap_or(0);
    (0..n)
        .map(|k| Poly::recombine(&parts.iter().map(|v| v[k]).collect::<Vec<_>>(), b))
        .collect()
}

/// Key-window bookkeeping: A | B | C | D (disjoint).
#[derive(Clone, Copy, Debug)]
pub struct Windows {
    pub a_off: usize,
    pub b_off: usize,
    pub c_off: usize,
    pub d_off: usize,
    pub total: usize,
}

impl Windows {
    pub fn new(cpp: &ComParams, r: usize, nn: usize) -> Self {
        let vl = VLayout::new(cpp, r);
        let a_off = 0;
        let b_off = a_off + cpp.kappa * nn;
        let c_off = b_off + cpp.kappa1 * (vl.t_len + vl.g_len);
        let d_off = c_off + cpp.kappa1 * vl.h_len;
        Windows {
            a_off,
            b_off,
            c_off,
            d_off,
            total: d_off + cpp.kappa1 * vl.h_len,
        }
    }
}

/// One level's public proof pieces.
#[derive(Clone, Debug)]
pub struct LevelProof {
    /// First outer commitment (κ1); tail mode: inner commitments t_i (r·κ)
    /// then quadratic garbage ((r²+r)/2) if quadratic.
    pub u1: Vec<Poly>,
    /// Second outer commitment (κ1); tail mode: the 2r−1 interleaved terms.
    pub u2: Vec<Poly>,
    pub p: Vec<i64>,
    pub jlnonce: u64,
    pub bb: Vec<Poly>,
    pub c: Vec<Poly>,
    pub normsq: u64,
    pub cpp: ComParams,
    pub nn: usize,
    pub r: usize,
}

impl LevelProof {
    pub fn tail(&self) -> bool {
        self.cpp.kappa1 == 0
    }
}

/// Prove one level: (stmt, wit) → (proof, target statement, target witness).
/// §5.4's restart remedy: when the measured output norm exceeds the heuristic
/// prediction, the level restarts with inflated input norms (up to 3 tries).
pub fn prove_level(
    stmt: &PrincipalStatement,
    wit: &PrincipalWitness,
    key: &ComKey,
    tail: bool,
) -> Result<
    (
        LevelProof,
        Option<PrincipalStatement>,
        Option<PrincipalWitness>,
    ),
    String,
> {
    let mut inflation = 1.0f64;
    for _ in 0..8 {
        match prove_level_inner(stmt, wit, key, tail, inflation) {
            Ok(x) => return Ok(x),
            Err(e) if e.starts_with("RESTART:") => inflation *= 2.0,
            Err(e) => return Err(e),
        }
    }
    Err("§5.4 restart budget exhausted".into())
}

fn prove_level_inner(
    stmt: &PrincipalStatement,
    wit: &PrincipalWitness,
    key: &ComKey,
    tail: bool,
    inflation: f64,
) -> Result<
    (
        LevelProof,
        Option<PrincipalStatement>,
        Option<PrincipalWitness>,
    ),
    String,
> {
    stmt.validate()?;
    stmt.check_all(&wit.s)
        .map_err(|e| format!("witness: {e}"))?;
    let quadratic = is_quadratic(stmt);

    let ranks: Vec<usize> = stmt.vectors.iter().map(|v| v.n).collect();
    let norms: Vec<u64> = wit
        .per_vector_normsq()
        .into_iter()
        .map(|n| ((n as f64) * inflation) as u64)
        .collect();
    let (cpp, nn, r, normsq_pred) = init_proof(&ranks, &norms, quadratic, tail)?;

    let win = Windows::new(&cpp, r, nn);
    if win.total > key.len {
        return Err(format!(
            "key too short: need {}, have {}",
            win.total, key.len
        ));
    }
    let vl = VLayout::new(&cpp, r);
    let layout = part_layout(&ranks, nn, quadratic);
    let parts = layout.materialize(wit);

    // ---- inner commitments, digits [i][j][ρ] ----
    let mut t_digits: Vec<Poly> = Vec::with_capacity(vl.t_len);
    for part in &parts {
        let t = key.mul_window(part, win.a_off, cpp.kappa);
        let dec: Vec<Vec<Poly>> = t.iter().map(|tp| tp.decompose(cpp.fu, cpp.bu)).collect();
        for j in 0..cpp.fu {
            for rho in 0..cpp.kappa {
                t_digits.push(dec[rho][j]);
            }
        }
    }

    // ---- quadratic garbage, digits [pair][k] ----
    let mut g_digits: Vec<Poly> = Vec::new();
    if quadratic && !tail {
        for i in 0..r {
            for j in i..r {
                for d in sprod(&parts[i], &parts[j]).decompose(cpp.fg, cpp.bg) {
                    g_digits.push(d);
                }
            }
        }
    }

    // ---- u1 ----
    let u1: Vec<Poly> = if tail {
        let mut pieces = Vec::new();
        for part in &parts {
            pieces.extend(key.mul_window(part, win.a_off, cpp.kappa));
        }
        if quadratic {
            for i in 0..r {
                for j in i..r {
                    pieces.push(sprod(&parts[i], &parts[j]));
                }
            }
        }
        pieces
    } else {
        let mut x = t_digits.clone();
        x.extend(g_digits.iter().copied());
        if x.is_empty() {
            vec![Poly::zero(); cpp.kappa1]
        } else {
            key.mul_window(&x, win.b_off, cpp.kappa1)
        }
    };

    // ---- transcript + JL ----
    let mut tr = Transcript::from_state(stmt.digest[..16].try_into().unwrap());
    tr.absorb_polys(&u1);
    let jl_seed = tr.squeeze32();
    let JlProjection { p, nonce, mats } = project_parts_exact(&parts, &jl_seed);
    tr.absorb_i32(&p.iter().map(|&x| x as i32).collect::<Vec<i32>>());

    // ---- LIFTS ----
    let mut lifted: Vec<DotCnst> = Vec::with_capacity(LIFTS);
    let mut bb: Vec<Poly> = Vec::with_capacity(LIFTS);
    for k in 0..LIFTS {
        let chal_seed = tr.challenge_seed();
        let omega = zq_scalars(256, &chal_seed, 0);
        let psi = zq_scalars(stmt.ct_cnst.len(), &chal_seed, 1);
        let (phis, jl_target) = collapse_jl(&mats, &p, &omega);
        // Φ^(k) = JL-collapse + ψ-scaled F' terms
        let mut phi_k: Vec<Vec<Poly>> = phis;
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            let ps = psi[l];
            if ps == 0 {
                continue;
            }
            for t in &c.terms {
                for (part, off, phi_chunk) in layout.locate_slice(t.idx, t.off, &t.phi) {
                    for (u, pc) in phi_chunk.iter().enumerate() {
                        phi_k[part][off + u].add_assign(&pc.scale(ps));
                    }
                }
            }
        }
        let mut target = jl_target;
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            if let Some(b) = &c.b {
                target =
                    crate::ring::cmod(target as i128 + psi[l] as i128 * b.constant_term() as i128);
            }
        }
        // honest evaluation
        let mut b_double = Poly::zero();
        for (i, phi) in phi_k.iter().enumerate() {
            b_double.add_assign(&sprod(phi, &parts[i]));
        }
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            if !c.a.is_empty() {
                let mut acc = Poly::zero();
                for &(i, j, ref coeff) in &c.a {
                    let prod = sprod(&wit.s[i], &wit.s[j]);
                    acc.add_assign(&if i == j {
                        coeff.mul(&prod)
                    } else {
                        coeff.mul(&prod).scale(2)
                    });
                }
                b_double.add_assign(&acc.scale(psi[l]));
            }
        }
        if b_double.constant_term() != target {
            return Err(format!(
                "lift {k}: ct mismatch {} != {}",
                b_double.constant_term(),
                target
            ));
        }
        let b_sent = b_double;
        tr.absorb_polys(&[b_sent]);
        bb.push(b_sent);
        // the lifted constraint: the per-part Φ^(k) (already containing the
        // ψ-scaled F' linear terms — folded above, NOT duplicated here) plus
        // the ψ-scaled F' quadratic entries, with b = b_sent
        let terms: Vec<Term> = phi_k
            .iter()
            .enumerate()
            .map(|(i, phi)| Term {
                idx: i,
                off: 0,
                phi: phi.clone(),
            })
            .collect();
        let mut a_entries: Vec<(usize, usize, Poly)> = Vec::new();
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            for &(i, j, ref coeff) in &c.a {
                for (p, q, cc) in expand_a_entry(&layout, i, j, coeff) {
                    a_entries.push((p, q, cc.scale(psi[l])));
                }
            }
        }
        lifted.push(DotCnst {
            terms,
            a: a_entries,
            b: Some(b_sent),
            ct_only: false,
        });
    }

    // ---- F-aggregation: uniform α ∈ R_q^K, β ∈ R_q^4 ----
    let chal_seed = tr.challenge_seed();
    let alphas = uniform_rq_vec(stmt.cnst.len(), &chal_seed, 0);
    let betas = uniform_rq_vec(LIFTS, &chal_seed, 1);
    let (mut phi_agg, a_agg, b_agg) =
        aggregate_constraints(stmt, &lifted, &layout, nn, &alphas, &betas);
    let _ = &mut phi_agg;

    // ---- h-garbage, u2, challenges, z ----
    let u2: Vec<Poly>;
    let c: Vec<Poly>;
    let z_digits: Vec<Vec<Poly>>;
    let h_digits: Vec<Poly>;
    if tail {
        // §5.6: interleaved garbage
        let phi_slice = |i: usize| -> Vec<Poly> { phi_agg[i * nn..(i + 1) * nn].to_vec() };
        let mut hs: Vec<Poly> = Vec::with_capacity(2 * r - 1);
        hs.push(sprod(&phi_slice(0), &parts[0]));
        tr.absorb_polys(&[hs[0]]);
        let mut c_ch = challenge_vec(1, &tr.challenge_seed(), 0);
        let mut z_run: Vec<Poly> = mul_vec(&c_ch[0], &parts[0]);
        let mut phi_run: Vec<Poly> = mul_vec(&c_ch[0], &phi_slice(0));
        for i in 1..r {
            let off = sprod(&phi_slice(i), &z_run).add(&sprod(&phi_run, &parts[i]));
            let diag = sprod(&phi_slice(i), &parts[i]);
            tr.absorb_polys(&[off, diag]);
            let ci = challenge_vec(1, &tr.challenge_seed(), 0).pop().unwrap();
            c_ch.push(ci);
            hs.push(off);
            hs.push(diag);
            z_run = add_vec(&z_run, &mul_vec(&c_ch[i], &parts[i]));
            phi_run = add_vec(&phi_run, &mul_vec(&c_ch[i], &phi_slice(i)));
        }
        u2 = hs;
        c = c_ch;
        z_digits = decompose_vec(&z_run, cpp.f, cpp.b);
        h_digits = Vec::new();
    } else {
        let mut h_vals: Vec<Poly> = Vec::with_capacity((r * r + r) / 2);
        for i in 0..r {
            for j in i..r {
                let pi = &phi_agg[i * nn..(i + 1) * nn];
                let pj = &phi_agg[j * nn..(j + 1) * nn];
                let v = if i == j {
                    sprod(pi, &parts[i])
                } else {
                    sprod(pi, &parts[j]).add(&sprod(pj, &parts[i])).scale(INV2)
                };
                h_vals.push(v);
            }
        }
        let mut hd: Vec<Poly> = Vec::with_capacity(vl.h_len);
        for h in &h_vals {
            for d in h.decompose(cpp.fu, cpp.bu) {
                hd.push(d);
            }
        }
        u2 = if hd.is_empty() {
            vec![Poly::zero(); cpp.kappa1]
        } else {
            key.mul_window(&hd, win.d_off, cpp.kappa1)
        };
        tr.absorb_polys(&u2);
        c = challenge_vec(r, &tr.challenge_seed(), 0);
        let mut z = vec![Poly::zero(); nn];
        for i in 0..r {
            for (k, sp) in parts[i].iter().enumerate() {
                z[k].add_assign(&c[i].mul(sp));
            }
        }
        z_digits = decompose_vec(&z, cpp.f, cpp.b);
        h_digits = hd;
    }

    let mut proof = LevelProof {
        u1: u1.clone(),
        u2: u2.clone(),
        p: p.clone(),
        jlnonce: nonce,
        bb,
        c: c.clone(),
        normsq: normsq_pred,
        cpp,
        nn,
        r,
    };

    if tail {
        // §5.4: restart with inflated parameters when the measured norm
        // exceeds the prediction (the announced bound must stay SIS-consistent)
        let measured: u64 = z_digits
            .iter()
            .flat_map(|v| v.iter().map(|q| q.normsq()))
            .sum();
        if measured > normsq_pred {
            return Err(format!(
                "RESTART: tail measured {measured} > predicted {normsq_pred}"
            ));
        }
        proof.normsq = normsq_pred;
        return Ok((proof, None, Some(PrincipalWitness::new(z_digits))));
    }

    // ---- the target relation over [z^(0..f-1), v] ----
    let mut v: Vec<Poly> = Vec::with_capacity(vl.m);
    v.extend(t_digits.iter().copied());
    v.extend(g_digits.iter().copied());
    v.extend(h_digits.iter().copied());

    let mut vectors: Vec<VectorSpec> = (0..cpp.f).map(|d| VectorSpec::z_part(nn, d)).collect();
    vectors.push(VectorSpec::plain(vl.m));
    let constraints = target_relation(
        &proof, key, &phi_agg, &a_agg, &b_agg, &layout, &win, quadratic,
    );
    // §5.4: restart when the measured output norm exceeds the prediction
    let measured: u64 = z_digits
        .iter()
        .chain(std::iter::once(&v))
        .flat_map(|vv| vv.iter().map(|q| q.normsq()))
        .sum();
    if measured > normsq_pred {
        return Err(format!(
            "RESTART: measured {measured} > predicted {normsq_pred}"
        ));
    }
    proof.normsq = normsq_pred;
    let target = PrincipalStatement::new(vectors, constraints, vec![], normsq_pred);
    let mut s: Vec<Vec<Poly>> = z_digits.clone();
    s.push(v);
    // self-check: every target constraint must hold of the target witness
    if let Err(e) = target.check_all(&s) {
        return Err(format!("internal target-relation check failed: {e}"));
    }
    Ok((proof, Some(target), Some(PrincipalWitness::new(s))))
}

/// The K' = 2κ1 + κ + 3 target constraints over `[z^(0..f-1), v]` (E1–E6).
#[allow(clippy::too_many_arguments)]
fn target_relation(
    proof: &LevelProof,
    key: &ComKey,
    phi_agg: &[Poly],
    a_agg: &[(usize, usize, Poly)],
    b_agg: &Poly,
    layout: &PartLayout,
    win: &Windows,
    quadratic: bool,
) -> Vec<DotCnst> {
    let cpp = &proof.cpp;
    let r = proof.r;
    let nn = proof.nn;
    let vl = VLayout::new(cpp, r);
    let f = cpp.f;
    let v_idx = f;
    let _pairs = (r * r + r) / 2;
    let _ = layout;
    let mut out: Vec<DotCnst> = Vec::with_capacity(2 * cpp.kappa1 + cpp.kappa + 3);

    // E1 (κ1): B·[t̃; g̃] = u1
    let tg_len = vl.t_len + vl.g_len;
    for j in 0..cpp.kappa1 {
        let phi = (0..tg_len)
            .map(|k| key.rows[win.b_off + j * tg_len + k])
            .collect();
        out.push(DotCnst {
            terms: vec![Term {
                idx: v_idx,
                off: 0,
                phi,
            }],
            a: vec![],
            b: Some(proof.u1[j]),
            ct_only: false,
        });
    }

    // E2 (κ1): D·h̃ = u2
    for j in 0..cpp.kappa1 {
        let phi = (0..vl.h_len)
            .map(|k| key.rows[win.d_off + j * vl.h_len + k])
            .collect();
        out.push(DotCnst {
            terms: vec![Term {
                idx: v_idx,
                off: vl.h_off(),
                phi,
            }],
            a: vec![],
            b: Some(proof.u2[j]),
            ct_only: false,
        });
    }

    // E3 (κ): A·z = Σ_i c_i t_i
    for rho in 0..cpp.kappa {
        let mut terms: Vec<Term> = Vec::new();
        for d in 0..f {
            let phi = (0..nn)
                .map(|k| key.rows[win.a_off + rho * nn + k].scale(1i64 << (d as u32 * cpp.b)))
                .collect();
            terms.push(Term {
                idx: d,
                off: 0,
                phi,
            });
        }
        let mut phi_v = vec![Poly::zero(); vl.t_len];
        for i in 0..r {
            for j in 0..cpp.fu {
                let scale = 1i64 << (j as u32 * cpp.bu);
                phi_v[i * cpp.fu * cpp.kappa + j * cpp.kappa + rho] = proof.c[i].neg().scale(scale);
            }
        }
        terms.push(Term {
            idx: v_idx,
            off: 0,
            phi: phi_v,
        });
        out.push(DotCnst::homogeneous(terms));
    }

    // E4 (quadratic): ⟨z,z⟩ = Σ_{i≤j}(2−[i=j]) c_i c_j g_ij
    if quadratic {
        let mut a_entries: Vec<(usize, usize, Poly)> = Vec::new();
        for d1 in 0..f {
            for d2 in d1..f {
                a_entries.push((
                    d1,
                    d2,
                    Poly::constant(1i64 << ((d1 + d2) as u32 * cpp.b)).neg(),
                ));
            }
        }
        let mut phi_g = vec![Poly::zero(); vl.g_len];
        for i in 0..r {
            for j in i..r {
                let base = tri_idx(i, j, r) * cpp.fg;
                let mut cc = proof.c[i].mul(&proof.c[j]);
                if i != j {
                    cc = cc.scale(2);
                }
                for k in 0..cpp.fg {
                    phi_g[base + k] = cc.scale(1i64 << (k as u32 * cpp.bg));
                }
            }
        }
        out.push(DotCnst {
            terms: vec![Term {
                idx: v_idx,
                off: vl.g_off(),
                phi: phi_g,
            }],
            a: a_entries,
            b: None,
            ct_only: false,
        });
    }

    // E5: ⟨φ_fold, z⟩ = Σ_{i≤j}(2−[i=j]) c_i c_j h_ij
    {
        let phi_fold = {
            let mut acc = vec![Poly::zero(); nn];
            for i in 0..r {
                for (u, pc) in phi_agg[i * nn..(i + 1) * nn].iter().enumerate() {
                    acc[u].add_assign(&pc.mul(&proof.c[i]));
                }
            }
            acc
        };
        let mut terms: Vec<Term> = Vec::new();
        for d in 0..f {
            let phi = phi_fold
                .iter()
                .map(|p| p.scale(1i64 << (d as u32 * cpp.b)))
                .collect();
            terms.push(Term {
                idx: d,
                off: 0,
                phi,
            });
        }
        let mut phi_h = vec![Poly::zero(); vl.h_len];
        for i in 0..r {
            for j in i..r {
                let base = tri_idx(i, j, r) * cpp.fu;
                let mut cc = proof.c[i].mul(&proof.c[j]);
                if i != j {
                    cc = cc.scale(2);
                }
                for k in 0..cpp.fu {
                    // the RHS of line 17 moves left: −Σ c_i c_j h_ij
                    phi_h[base + k] = cc.scale(1i64 << (k as u32 * cpp.bu)).neg();
                }
            }
        }
        terms.push(Term {
            idx: v_idx,
            off: vl.h_off(),
            phi: phi_h,
        });
        out.push(DotCnst::homogeneous(terms));
    }

    // E6: Σ_{i≤j}(2−[i=j]) a_ij g_ij + Σ_i h_ii = b_agg
    {
        let mut phi_v = vec![Poly::zero(); vl.m];
        for &(i, j, ref coeff) in a_agg {
            let base = vl.g_off() + tri_idx(i, j, r) * cpp.fg;
            let eff = if i == j { *coeff } else { coeff.scale(2) };
            for k in 0..cpp.fg {
                phi_v[base + k].add_assign(&eff.scale(1i64 << (k as u32 * cpp.bg)));
            }
        }
        for i in 0..r {
            let base = vl.h_off() + tri_idx(i, i, r) * cpp.fu;
            for k in 0..cpp.fu {
                phi_v[base + k].add_assign(&Poly::constant(1i64 << (k as u32 * cpp.bu)));
            }
        }
        out.push(DotCnst {
            terms: vec![Term {
                idx: v_idx,
                off: 0,
                phi: phi_v,
            }],
            a: vec![],
            b: Some(*b_agg),
            ct_only: false,
        });
    }

    out
}

/// The constraint aggregation (shared by prove and reduce).
fn aggregate_constraints(
    stmt: &PrincipalStatement,
    lifted: &[DotCnst],
    layout: &PartLayout,
    nn: usize,
    alphas: &[Poly],
    betas: &[Poly],
) -> (Vec<Poly>, Vec<(usize, usize, Poly)>, Poly) {
    let mut phi_agg: Vec<Poly> = vec![Poly::zero(); layout.r * nn];
    let mut a_agg: Vec<(usize, usize, Poly)> = Vec::new();
    let mut b_agg = Poly::zero();
    for (k, c) in stmt.cnst.iter().enumerate() {
        let a = &alphas[k];
        for t in &c.terms {
            for (part, off, phi_chunk) in layout.locate_slice(t.idx, t.off, &t.phi) {
                for (u, pc) in phi_chunk.iter().enumerate() {
                    phi_agg[part * nn + off + u].add_assign(&pc.mul(a));
                }
            }
        }
        for &(i, j, ref coeff) in &c.a {
            for (p, q, cc) in expand_a_entry(layout, i, j, coeff) {
                a_agg.push((p, q, cc.mul(a)));
            }
        }
        if let Some(b) = &c.b {
            b_agg.add_assign(&b.mul(a));
        }
    }
    for (k, c) in lifted.iter().enumerate() {
        let be = &betas[k];
        for t in &c.terms {
            for (u, pc) in t.phi.iter().enumerate() {
                phi_agg[t.idx * nn + t.off + u].add_assign(&pc.mul(be));
            }
        }
        for &(i, j, ref coeff) in &c.a {
            a_agg.push((i.min(j), i.max(j), coeff.mul(be)));
        }
        if let Some(b) = &c.b {
            b_agg.add_assign(&b.mul(be));
        }
    }
    a_agg = merge_a(a_agg);
    (phi_agg, a_agg, b_agg)
}

/// The verifier's replay of one level: regenerates the transcript state, the
/// aggregated constraint, and (tail) the interleaved challenges. Returns
/// (phi_agg, a_agg, b_agg) on success.
pub fn replay_level(
    stmt: &PrincipalStatement,
    proof: &LevelProof,
) -> Result<(Vec<Poly>, Vec<(usize, usize, Poly)>, Poly), String> {
    let quadratic = is_quadratic(stmt);
    let cpp = &proof.cpp;
    let nn = proof.nn;
    let r = proof.r;
    if nn == 0 || r == 0 {
        return Err("degenerate level parameters".into());
    }
    let ranks: Vec<usize> = stmt.vectors.iter().map(|v| v.n).collect();

    // structural checks
    let expected_u1len = if proof.tail() {
        r * cpp.kappa + if quadratic { (r * r + r) / 2 } else { 0 }
    } else {
        cpp.kappa1
    };
    let expected_u2len = if proof.tail() { 2 * r - 1 } else { cpp.kappa1 };
    if proof.u1.len() != expected_u1len {
        return Err(format!("u1 length {} != {expected_u1len}", proof.u1.len()));
    }
    if proof.u2.len() != expected_u2len {
        return Err(format!("u2 length {} != {expected_u2len}", proof.u2.len()));
    }
    if proof.p.len() != 256 {
        return Err("JL projection must have 256 entries".into());
    }
    if proof.bb.len() != LIFTS {
        return Err(format!("expected {LIFTS} lift polynomials"));
    }
    if proof.c.len() != r {
        return Err(format!("challenge count {} != r {r}", proof.c.len()));
    }
    for (i, c) in proof.c.iter().enumerate() {
        if !is_challenge(c) {
            return Err(format!("challenge {i} outside C"));
        }
    }
    if !crate::sis::sis_secure(
        cpp.kappa,
        6.0 * crate::challenge::T
            * crate::sis::SLACK
            * 2f64.powi((cpp.f as i32 - 1) * cpp.b as i32)
            * (proof.normsq as f64).sqrt(),
    ) {
        return Err("inner commitments not SIS-secure at the announced norm".into());
    }
    if !proof.tail()
        && !crate::sis::sis_secure(
            cpp.kappa1,
            2.0 * crate::sis::SLACK * (proof.normsq as f64).sqrt(),
        )
    {
        return Err("outer commitments not SIS-secure at the announced norm".into());
    }
    if jl_normsq(&proof.p) > 256 * stmt.betasq.min(jl_max_normsq()) {
        return Err("JL projection longer than the bound".into());
    }

    // transcript replay
    let mut tr = Transcript::from_state(stmt.digest[..16].try_into().unwrap());
    tr.absorb_polys(&proof.u1);
    let jl_seed = tr.squeeze32();
    tr.absorb_i32(&proof.p.iter().map(|&x| x as i32).collect::<Vec<i32>>());

    let layout = part_layout(&ranks, nn, quadratic);
    // regenerate the JL matrices
    let mats: Vec<JlMatrix> = (0..r)
        .map(|i| {
            JlMatrix::expand(
                256,
                nn * N,
                JlMode::PlusMinus1,
                &jl_seed,
                (proof.jlnonce << 8) | (i as u64 & 0xff),
            )
        })
        .collect();
    if !jl_accept(
        &proof.p,
        stmt.betasq.min(jl_max_normsq()),
        JlMode::PlusMinus1,
    ) {
        return Err("JL projection fails the acceptance bound".into());
    }

    // LIFTS: check the b'' constant terms
    let mut lifted: Vec<DotCnst> = Vec::with_capacity(LIFTS);
    for k in 0..LIFTS {
        let chal_seed = tr.challenge_seed();
        let omega = zq_scalars(256, &chal_seed, 0);
        let psi = zq_scalars(stmt.ct_cnst.len(), &chal_seed, 1);
        let (phis, jl_target) = collapse_jl(&mats, &proof.p, &omega);
        let mut target = jl_target;
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            if let Some(b) = &c.b {
                target =
                    crate::ring::cmod(target as i128 + psi[l] as i128 * b.constant_term() as i128);
            }
        }
        if proof.bb[k].constant_term() != target {
            return Err(format!("lift {k}: b'' constant term incorrect"));
        }
        // the lifted constraint (mirrors the prover: Φ^(k) with the ψ-scaled
        // F' linear terms folded in — no duplication)
        let mut phi_k: Vec<Vec<Poly>> = phis;
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            let ps = psi[l];
            if ps != 0 {
                for t in &c.terms {
                    for (part, off, phi_chunk) in layout.locate_slice(t.idx, t.off, &t.phi) {
                        for (u, pc) in phi_chunk.iter().enumerate() {
                            phi_k[part][off + u].add_assign(&pc.scale(ps));
                        }
                    }
                }
            }
        }
        let terms: Vec<Term> = phi_k
            .iter()
            .enumerate()
            .map(|(i, phi)| Term {
                idx: i,
                off: 0,
                phi: phi.clone(),
            })
            .collect();
        let mut a_entries: Vec<(usize, usize, Poly)> = Vec::new();
        for (l, c) in stmt.ct_cnst.iter().enumerate() {
            let ps = psi[l];
            if ps != 0 {
                for &(i, j, ref coeff) in &c.a {
                    for (p, q, cc) in expand_a_entry(&layout, i, j, coeff) {
                        a_entries.push((p, q, cc.scale(ps)));
                    }
                }
            }
        }
        tr.absorb_polys(&[proof.bb[k]]);
        lifted.push(DotCnst {
            terms,
            a: a_entries,
            b: Some(proof.bb[k]),
            ct_only: false,
        });
    }

    // F-aggregation
    let chal_seed = tr.challenge_seed();
    let alphas = uniform_rq_vec(stmt.cnst.len(), &chal_seed, 0);
    let betas = uniform_rq_vec(LIFTS, &chal_seed, 1);
    let (phi_agg, a_agg, b_agg) =
        aggregate_constraints(stmt, &lifted, &layout, nn, &alphas, &betas);

    // c replay + checks
    if proof.tail() {
        tr.absorb_polys(&[proof.u2[0]]);
        let mut c_ch = challenge_vec(1, &tr.challenge_seed(), 0);
        for i in 1..r {
            tr.absorb_polys(&[proof.u2[2 * i - 1], proof.u2[2 * i]]);
            let ci = challenge_vec(1, &tr.challenge_seed(), 0).pop().unwrap();
            c_ch.push(ci);
        }
        if c_ch != proof.c {
            return Err("tail challenges inconsistent with the transcript".into());
        }
    } else {
        tr.absorb_polys(&proof.u2);
        let c_ch = challenge_vec(r, &tr.challenge_seed(), 0);
        if c_ch != proof.c {
            return Err("amortization challenges inconsistent with the transcript".into());
        }
    }
    Ok((phi_agg, a_agg, b_agg))
}

/// The verifier's statement reconstruction (non-tail): rebuild the target.
pub fn reduce_level(
    stmt: &PrincipalStatement,
    proof: &LevelProof,
    key: &ComKey,
) -> Result<PrincipalStatement, String> {
    if proof.tail() {
        return Err("reduce_level on a tail level".into());
    }
    let (phi_agg, a_agg, b_agg) = replay_level(stmt, proof)?;
    let cpp = &proof.cpp;
    let quadratic = is_quadratic(stmt);
    let layout = part_layout(
        &stmt.vectors.iter().map(|v| v.n).collect::<Vec<_>>(),
        proof.nn,
        quadratic,
    );
    let win = Windows::new(cpp, proof.r, proof.nn);
    if win.total > key.len {
        return Err("key too short for the level windows".into());
    }
    let vl = VLayout::new(cpp, proof.r);
    let mut vectors: Vec<VectorSpec> = (0..cpp.f)
        .map(|d| VectorSpec::z_part(proof.nn, d))
        .collect();
    vectors.push(VectorSpec::plain(vl.m));
    let constraints = target_relation(
        proof, key, &phi_agg, &a_agg, &b_agg, &layout, &win, quadratic,
    );
    Ok(PrincipalStatement::new(
        vectors,
        constraints,
        vec![],
        proof.normsq,
    ))
}

/// The final tail verification (Figure 3 on the transmitted material).
pub fn verify_tail(
    stmt: &PrincipalStatement,
    proof: &LevelProof,
    final_witness: &PrincipalWitness,
    key: &ComKey,
) -> Result<(), String> {
    if !proof.tail() {
        return Err("verify_tail on a non-tail level".into());
    }
    let quadratic = is_quadratic(stmt);
    let cpp = &proof.cpp;
    let nn = proof.nn;
    let r = proof.r;
    let (phi_agg, a_agg, b_agg) = replay_level(stmt, proof)?;

    // the final witness: f parts of rank nn
    if final_witness.s.len() != cpp.f {
        return Err(format!("final witness must have {} parts", cpp.f));
    }
    for (d, v) in final_witness.s.iter().enumerate() {
        if v.len() != nn {
            return Err(format!("final witness part {d} rank {} != {nn}", v.len()));
        }
    }
    let normsq: u64 = final_witness
        .s
        .iter()
        .flat_map(|v| v.iter().map(|p| p.normsq()))
        .sum();
    if normsq > proof.normsq {
        return Err(format!(
            "final witness norm² {normsq} > announced {}",
            proof.normsq
        ));
    }
    let z = recombine_vec(&final_witness.s, cpp.b);

    let win = Windows::new(cpp, r, nn);
    // E3: A·z = Σ_i c_i t_i
    let az = key.mul_window(&z, win.a_off, cpp.kappa);
    for rho in 0..cpp.kappa {
        let mut rhs = Poly::zero();
        for i in 0..r {
            rhs.add_assign(&proof.c[i].mul(&proof.u1[i * cpp.kappa + rho]));
        }
        if az[rho] != rhs {
            return Err(format!("E3 (Az = Σc_i t_i) violated at row {rho}"));
        }
    }

    // E4 + E6 (quadratic)
    if quadratic {
        let g_base = r * cpp.kappa;
        let lhs = sprod(&z, &z);
        let mut rhs = Poly::zero();
        let mut idx = 0;
        for i in 0..r {
            for j in i..r {
                let g = &proof.u1[g_base + idx];
                let mut term = proof.c[i].mul(&proof.c[j]).mul(g);
                if i != j {
                    term = term.scale(2);
                }
                rhs.add_assign(&term);
                idx += 1;
            }
        }
        if lhs != rhs {
            return Err("E4 (⟨z,z⟩ = Σ c c g) violated".into());
        }
        // E6: Σ_{i≤j}(2−[i=j]) a_ij g_ij + Σ_i h_ii = b_agg
        let mut acc = Poly::zero();
        for &(i, j, ref coeff) in &a_agg {
            let g = &proof.u1[g_base + tri_idx(i, j, r)];
            let eff = if i == j { *coeff } else { coeff.scale(2) };
            acc.add_assign(&eff.mul(g));
        }
        for i in 0..r {
            acc.add_assign(&proof.u2[2 * i]);
        }
        if acc != b_agg {
            return Err("E6 (Σ a g + Σ h_ii = b) violated".into());
        }
    }

    // E5: ⟨φ_fold, z⟩ = c_0² h_0 + Σ_{i≥1}(c_i h_{2i-1} + c_i² h_{2i})
    let phi_fold = {
        let mut acc = vec![Poly::zero(); nn];
        for i in 0..r {
            for (u, pc) in phi_agg[i * nn..(i + 1) * nn].iter().enumerate() {
                acc[u].add_assign(&pc.mul(&proof.c[i]));
            }
        }
        acc
    };
    let lhs = sprod(&phi_fold, &z);
    let mut rhs = proof.c[0].mul(&proof.c[0]).mul(&proof.u2[0]);
    for i in 1..r {
        rhs.add_assign(&proof.c[i].mul(&proof.u2[2 * i - 1]));
        rhs.add_assign(&proof.c[i].mul(&proof.c[i]).mul(&proof.u2[2 * i]));
    }
    if lhs != rhs {
        return Err("E5 (⟨φ, z⟩ = Σ h c terms) violated".into());
    }
    Ok(())
}

fn add_vec(a: &[Poly], b: &[Poly]) -> Vec<Poly> {
    a.iter().zip(b.iter()).map(|(x, y)| x.add(y)).collect()
}

fn mul_vec(c: &Poly, v: &[Poly]) -> Vec<Poly> {
    v.iter().map(|p| c.mul(p)).collect()
}

fn merge_a(mut a: Vec<(usize, usize, Poly)>) -> Vec<(usize, usize, Poly)> {
    a.sort_by_key(|(i, j, _)| (*i, *j));
    let mut out: Vec<(usize, usize, Poly)> = Vec::new();
    for (i, j, c) in a {
        if let Some(last) = out.last_mut() {
            if last.0 == i && last.1 == j {
                last.2.add_assign(&c);
                continue;
            }
        }
        out.push((i, j, c));
    }
    out
}

/// Deterministic JL projection with the reference's rejection rule.
fn project_parts_exact(parts: &[Vec<Poly>], seed: &[u8]) -> JlProjection {
    let normsq: u64 = parts
        .iter()
        .map(|v| v.iter().map(|p| p.normsq()).sum::<u64>())
        .sum();
    let mut nonce = 0u64;
    loop {
        nonce += 1;
        let mats: Vec<JlMatrix> = parts
            .iter()
            .enumerate()
            .map(|(i, v)| {
                JlMatrix::expand(
                    256,
                    v.len() * N,
                    JlMode::PlusMinus1,
                    seed,
                    (nonce << 8) | (i as u64 & 0xff),
                )
            })
            .collect();
        let mut p = vec![0i64; 256];
        for (i, v) in parts.iter().enumerate() {
            let flat: Vec<i64> = v.iter().flat_map(|poly| poly.0.iter().copied()).collect();
            let sub = mats[i].project(&flat);
            for (rr, &x) in sub.iter().enumerate() {
                p[rr] += x;
            }
        }
        if jl_accept(&p, normsq, JlMode::PlusMinus1) {
            return JlProjection { p, nonce, mats };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_layout_linear_and_quadratic() {
        let lay = part_layout(&[3, 4, 2], 4, false);
        assert_eq!(lay.r, 3);
        assert_eq!(lay.starts[0], (0, 0));
        assert_eq!(lay.starts[1], (0, 3));
        assert_eq!(lay.starts[2], (1, 3));
        let layq = part_layout(&[3, 4, 2], 4, true);
        assert_eq!(layq.r, 3);
        assert_eq!(layq.starts[1], (1, 0));
        let layb = part_layout(&[10], 4, true);
        assert_eq!(layb.r, 3);
    }

    #[test]
    fn locate_roundtrip() {
        let lay = part_layout(&[3, 4, 2], 4, false);
        assert_eq!(lay.locate(1, 0), (0, 3));
        assert_eq!(lay.locate(1, 1), (1, 0));
        assert_eq!(lay.locate(2, 2), (2, 1));
        assert_eq!(lay.span(1), (0, 2));
    }

    #[test]
    fn decompose_recombine_vec() {
        let z: Vec<Poly> = (0..6)
            .map(|i| Poly::almost_uniform(&[9], i as u64 + 3))
            .collect();
        for &(f, b) in &[(2usize, 7u32), (3, 5)] {
            let parts = decompose_vec(&z, f, b);
            assert_eq!(parts.len(), f);
            assert_eq!(recombine_vec(&parts, b), z);
        }
    }

    #[test]
    fn expand_a_entry_positional() {
        // vector 0 (rank 512) chunks into parts 0,1; vector 1 into 2,3 —
        // the positional pairing gives (0,2), (1,3) ONLY
        let layout = part_layout(&[512, 512], 256, true);
        let out = expand_a_entry(&layout, 0, 1, &Poly::constant(5));
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], (0, 2, Poly::constant(5)));
        assert_eq!(out[1], (1, 3, Poly::constant(5)));
        // the same-vector diagonal: (x, x) pairs
        let out2 = expand_a_entry(&layout, 0, 0, &Poly::constant(1));
        assert_eq!(out2.len(), 2);
        assert_eq!(out2[0], (0, 0, Poly::constant(1)));
        assert_eq!(out2[1], (1, 1, Poly::constant(1)));
    }
}

#[cfg(test)]
mod quadratic_iso_tests {
    use super::*;

    fn mk(seed: u64, rank: usize) -> Vec<Poly> {
        (0..rank)
            .map(|i| {
                let mut p = [0i64; N];
                for (j, c) in p.iter_mut().enumerate() {
                    *c = ((i * 37 + j * 17 + seed as usize * 13) % 7) as i64 - 3;
                }
                Poly(p)
            })
            .collect()
    }

    #[test]
    fn quadratic_tail_no_fp() {
        // ONE quadratic constraint, NO F' — the minimal E4/E6 repro
        let s0 = mk(1, 512);
        let s1 = mk(2, 512);
        let phi = mk(3, 512);
        let quad = sprod(&s0, &s1).scale(10);
        let lin = sprod(&phi, &s0);
        let b = quad.add(&lin);
        let stmt = PrincipalStatement::new(
            vec![VectorSpec::plain(512), VectorSpec::plain(512)],
            vec![DotCnst {
                terms: vec![Term {
                    idx: 0,
                    off: 0,
                    phi,
                }],
                a: vec![(0, 1, Poly::constant(5))],
                b: Some(b),
                ct_only: false,
            }],
            vec![],
            u32::MAX as u64,
        );
        let wit = PrincipalWitness::new(vec![s0, s1]);
        let key = ComKey::expand(1 << 16, &[1u8; 32]);
        let (proof, _, final_wit) = prove_level(&stmt, &wit, &key, true).unwrap();
        verify_tail(&stmt, &proof, final_wit.as_ref().unwrap(), &key).unwrap();
    }
}
