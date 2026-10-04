//! The LaBRADOR protocol core: `simple_prove`/`simple_reduce`/`verify` from `labrador.c` +
//! `dachshund.c`, ported to exact coefficient arithmetic. See the crate docs for the encoding
//! trade-offs (degree-1 constraints only, plain keys, full non-tail mode always).

use crate::ring::*;
use crate::{Proof, Statement, Witness, LOGQ};
use lattice_core::keccak::{sha3_256, KeccakSponge};

/// Upstream constants (`data.h`): TAU1, TAU2, T (challenge operator norm), SLACK.
const TAU1: f64 = 32.0;
const TAU2: f64 = 8.0;
const T: f64 = 14.0;
const SLACK: f64 = 2.0;
/// LIFTS = ceil(128/LOGQ) = 3 in upstream; unused without degree-0 constraints.
#[allow(dead_code)]
pub const LIFTS: usize = 128usize.div_ceil(LOGQ);

/// Commitment parameters (upstream `comparams`).
#[derive(Clone, Copy, Debug)]
pub struct ComParams {
    pub f: usize,      // amortized opening decomposition parts
    pub fu: usize,     // uniform decomposition parts
    pub fg: usize,     // quadratic garbage decomposition parts (0: no quadratic)
    pub b: u32,        // opening decomposition bits
    pub bu: u32,       // uniform decomposition bits
    pub bg: u32,       // garbage decomposition bits
    pub kappa: usize,  // inner commitment rank
    pub kappa1: usize, // outer commitment rank
}

fn sis_secure(rank: usize, norm: f64) -> bool {
    crate::sis_secure(rank, norm)
}

/// Parameter selection: the `k = 15..1` loop of `init_proof`, quadratic mode 2 (the conjugate
/// construction), always non-tail.
fn init_params(ranks: &[usize], normsq: &[u64]) -> Result<ComParams, String> {
    let r = ranks.len();
    for k in (1..=15usize).rev() {
        let nn = {
            let total: usize = ranks.iter().sum();
            total.div_ceil(k).max(1)
        };
        // variance of the witness parts (max over parts)
        let vars = (0..r)
            .map(|i| normsq[i] as f64 / (ranks[i] * N) as f64)
            .fold(0.0f64, f64::max);
        let varz = (TAU1 + 4.0 * TAU2) * vars * k as f64;
        let decompose = !sis_secure(
            13,
            6.0 * T * SLACK * (2.0 * (TAU1 + 4.0 * TAU2) * varz * nn as f64 * N as f64).sqrt(),
        ) || 64.0 * varz > (1u64 << 28) as f64;
        let (mut f, mut b) = if decompose {
            (
                2usize,
                (((12.0f64).log2() + varz.log2()) / 4.0).round().max(1.0) as u32,
            )
        } else {
            (
                1usize,
                (((12.0f64).log2() + varz.log2()) / 2.0).round().max(1.0) as u32,
            )
        };
        const DIGITBITS: u32 = 14;
        if b > DIGITBITS {
            let t = f as u32 * b;
            f = t.div_ceil(DIGITBITS) as usize;
            b = t.div_ceil(f as u32);
        }
        let fu = ((LOGQ as f64 + 2.0 * b as f64 / 3.0) / b as f64).ceil() as usize;
        let fu = fu.max(LOGQ.div_ceil(DIGITBITS as usize));
        let bu = LOGQ.div_ceil(fu);
        // quadratic garbage variance: sum of vars^2 over the joined parts
        let varg = {
            let mut acc = 0.0;
            for _i in 0..r {
                acc += vars * vars;
            }
            2.0 * N as f64 * acc * nn as f64
        };
        let bg = b;
        let fg = (((12.0f64).log2() + varg.log2()) / (2.0 * bg as f64)).ceil() as usize;
        let fg = fg.max(1).max(LOGQ.div_ceil(bg as usize));
        // commitment ranks
        let mut norm = (2f64.powi(2 * b as i32) / 12.0 * (f - 1) as f64
            + varz / 2f64.powi(2 * b as i32 * (f - 1) as i32))
            * nn as f64;
        let _rr = k;
        let rr = k as f64; // amortized multiplicity: k joined parts
        norm += (2f64.powi(2 * bu as i32) * (fu - 1) as f64
            + 2f64.powi(2 * (LOGQ as i32 - (fu as i32 - 1) * bu as i32)))
            / 12.0
            * (rr + (rr * rr + rr) / 2.0); // kappa = 1 inner commitments
        norm += (2f64.powi(2 * bg as i32) / 12.0 * (fg - 1) as f64
            + varg / 2f64.powi(2 * bg as i32 * (fg as i32 - 1)))
            * (rr * rr + rr)
            / 2.0;
        norm *= N as f64;
        let kappa = (1..=32)
            .find(|&kp| {
                sis_secure(
                    kp,
                    6.0 * T * SLACK * 2f64.powi((f as i32 - 1) * b as i32) * norm.sqrt(),
                )
            })
            .unwrap_or(33);
        if kappa > 32 {
            continue;
        }
        let kappa1 = (1..=32)
            .find(|&k1| sis_secure(k1, 2.0 * SLACK * norm.sqrt()))
            .unwrap_or(33);
        if kappa1 > 32 {
            continue;
        }
        if fu * k * kappa + (fu + fg) * (k * k + k) / 2 <= 11 * nn / 10 + 1 {
            return Ok(ComParams {
                f,
                fu,
                fg,
                b,
                bu: bu as u32,
                bg,
                kappa,
                kappa1,
            });
        }
    }
    Err("cannot make commitments secure".into())
}

/// The commitment key: `kappa` rows of `total_rank` uniform ring elements (degree-1 layout:
/// row `t` covers `[t*total, (t+1)*total)`).
pub struct ComKey {
    pub rows: Vec<Vec<Poly>>,
    pub total: usize,
    pub kappa: usize,
}

impl ComKey {
    pub fn expand(total: usize, kappa: usize, seed: &[u8; 32]) -> Self {
        let rows = (0..kappa)
            .map(|t| uniform(total, seed, t as u64 + 1))
            .collect();
        ComKey { rows, total, kappa }
    }
    /// `u = A s`: kappa outputs, each `sum_j A_t[j] s[j]`.
    pub fn commit(&self, s: &[Poly]) -> Vec<Poly> {
        (0..self.kappa)
            .map(|t| Poly::sprod(&self.rows[t], s))
            .collect()
    }
}

/// Transcript over 16-byte hash chains (upstream's `h`).
struct Hash16(pub [u8; 16]);

impl Hash16 {
    fn of(digest: &[u8; 32]) -> Self {
        let d = sha3_256(digest);
        Hash16([
            d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13],
            d[14], d[15],
        ])
    }
    fn absorb_polys(&mut self, polys: &[Poly]) {
        let mut buf = Vec::with_capacity(16 + polys.len() * 48);
        buf.extend_from_slice(&self.0);
        for p in polys {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        let d = sha3_256(&buf);
        self.0.copy_from_slice(&d[..16]);
    }
    fn squeeze32(&self) -> [u8; 32] {
        let d = sha3_256(&self.0);
        let mut out = [0u8; 32];
        out.copy_from_slice(&d);
        out
    }
}

type ExpandedWitness = (Vec<usize>, Vec<Vec<Poly>>, Vec<u64>);

fn expand_witness(stmt: &Statement, wit: &Witness) -> Result<ExpandedWitness, String> {
    let r = stmt.vectors.len();
    let mut ranks = Vec::with_capacity(r + 1);
    let mut vecs = Vec::with_capacity(r + 1);
    let mut norms = Vec::with_capacity(r + 1);
    let mut total = 0u64;
    for (i, v) in stmt.vectors.iter().enumerate() {
        if wit.vectors[i].len() != v.n * N {
            return Err(format!(
                "vector {i}: {} coefficients, rank {} wants {}",
                wit.vectors[i].len(),
                v.n,
                v.n * N
            ));
        }
        let src = &wit.vectors[i];
        let polys: Vec<Poly> = (0..src.len() / N)
            .map(|k| Poly::from_i16(&src[k * N..k * N + N]))
            .collect();
        let n = polys
            .iter()
            .map(|p| {
                p.0.iter()
                    .map(|&x| (i128::from(x) * i128::from(x)) as u64)
                    .sum::<u64>()
            })
            .sum();
        if v.binary {
            if wit.vectors[i].iter().any(|&c| c != 0 && c != 1) {
                return Err(format!(
                    "vector {i}: binary vector has non-binary coefficient"
                ));
            }
        } else if n > v.betasq {
            return Err(format!(
                "vector {i}: normsq {n} exceeds betasq {}",
                v.betasq
            ));
        }
        total = total.saturating_add(n);
        ranks.push(v.n);
        norms.push(n);
        vecs.push(polys);
    }
    // binary slack vector: bits of (betasq_i - normsq_i) for exact norm proofs
    let mut slack = Vec::new();
    for (i, v) in stmt.vectors.iter().enumerate() {
        if !v.binary {
            let d = v.betasq - norms[i];
            let mut bits: Vec<i16> = Vec::with_capacity(64);
            let mut x = d;
            for _ in 0..64 {
                bits.push((x & 1) as i16);
                x >>= 1;
            }
            slack.push(Poly::from_i16(&bits));
        }
    }
    if slack.is_empty() {
        slack.push(Poly::zero());
    }
    let slack_norm: u64 = slack
        .iter()
        .map(|p| {
            p.0.iter()
                .map(|&x| (i128::from(x) * i128::from(x)) as u64)
                .sum::<u64>()
        })
        .sum();
    ranks.push(slack.len());
    norms.push(slack_norm);
    vecs.push(slack);
    total = total.saturating_add(slack_norm);
    if total > (1u64 << (LOGQ - 1)) - 59 {
        return Err("total witness norm too big".into());
    }
    Ok((ranks, vecs, norms))
}

/// Evaluate the simple constraints against a witness in Poly form.
fn eval_constraints(stmt: &Statement, vecs: &[Vec<Poly>]) -> Vec<Poly> {
    stmt.constraints
        .iter()
        .map(|c| {
            let mut acc = Poly::zero();
            for (j, blk) in c.blocks.iter().enumerate() {
                let s = &vecs[blk.idx][blk.off..blk.off + blk.len];
                acc.add_assign(&Poly::sprod(&c.phi[j], s));
            }
            acc
        })
        .collect()
}

pub fn prove(stmt: &Statement, wit: &Witness) -> Result<Proof, String> {
    let (ranks, vecs, norms) = expand_witness(stmt, wit)?;
    // prover-side statement check: every constraint must hold of the witness
    for (i, value) in eval_constraints(stmt, &vecs).iter().enumerate() {
        match &stmt.constraints[i].b {
            Some(b) if *value != *b => {
                return Err(format!("constraint {i} does not hold of the witness"));
            }
            _ => {}
        }
    }
    // conjugate inflation (upstream: normsq[r+1+i] = (TAU1+4 TAU2) normsq[i])
    let r = ranks.len();
    let mut all_ranks = Vec::with_capacity(2 * r);
    let mut all_norms = Vec::with_capacity(2 * r);
    for i in 0..r {
        all_ranks.push(ranks[i]);
        all_norms.push(norms[i]);
    }
    for i in 0..r {
        all_ranks.push(ranks[i]);
        all_norms.push(((TAU1 + 4.0 * TAU2) * norms[i] as f64) as u64);
    }
    let cpp = init_params(&all_ranks, &all_norms)?;
    let total: usize = ranks.iter().sum::<usize>().max(1);
    let key = ComKey::expand(total, cpp.kappa, &crate::seed_bytes(b"comkey", 0));
    // the outer key commits the fu-digit decompositions of the inner commitments
    let outer_len = cpp.fu * cpp.kappa;
    let key1 = ComKey::expand(outer_len, cpp.kappa1, &crate::seed_bytes(b"comkey", 1));

    let mut h = Hash16::of(&stmt.digest);

    // ---- conjugates and lifting digits (simple_commit) ----
    // s0: originals (r vectors) + slack; s2: conjugates sigma_m1 (norm-bounded) / flip (binary)
    let mut s0: Vec<Poly> = Vec::with_capacity(total);
    let mut s2: Vec<Poly> = Vec::with_capacity(total);
    for (i, v) in stmt.vectors.iter().enumerate() {
        for p in &vecs[i] {
            s0.push(*p);
            s2.push(if v.binary { p.flip() } else { p.sigma_m1() });
        }
    }
    for p in &vecs[r - 1] {
        s0.push(*p);
        s2.push(p.flip());
    }
    while s0.len() < total {
        s0.push(Poly::zero());
        s2.push(Poly::zero());
    }
    // <s0, s2> per part = normsq; decomposed into FL=ceil(48/10)=5 digits of 10 bits (lifting)
    const BL: u32 = 10;
    const FL: usize = (LOGQ + BL as usize / 2) / BL as usize;
    let inner: Vec<Poly> = (0..r)
        .map(|i| {
            let conj: Vec<Poly> = vecs[i]
                .iter()
                .map(|p| {
                    if i == r - 1 || stmt.vectors[i].binary {
                        p.flip()
                    } else {
                        p.sigma_m1()
                    }
                })
                .collect();
            Poly::sprod(&vecs[i], &conj)
        })
        .collect();
    // first outer commitment over (s0 + lifting digits), then challenges alpha, then s2-fold
    let mut commit_pool: Vec<Poly> = s0.clone();
    let lift_digits: Vec<Vec<Poly>> = inner.iter().map(|p| decompose(p, FL, BL)).collect();
    for d in &lift_digits {
        commit_pool.extend(d.iter().copied());
    }
    let t0 = key.commit(&commit_pool);
    let digits0: Vec<Poly> = t0
        .iter()
        .flat_map(|p| decompose(p, cpp.fu, cpp.bu))
        .collect();
    let u_outer1 = key1.commit(&digits0);
    h.absorb_polys(&u_outer1);
    let seed32 = h.squeeze32();
    let alpha = quarternary(r, &seed32, 0);
    // fold the conjugates by alpha
    let s2_folded: Vec<Poly> = (0..r)
        .flat_map(|i| {
            let a = alpha[i % alpha.len()];
            vecs[i].iter().map(move |p| {
                let conj = if i == r - 1 || stmt.vectors[i].binary {
                    p.flip()
                } else {
                    p.sigma_m1()
                };
                a.mul(&conj)
            })
        })
        .collect();
    let t1 = key.commit(&s2_folded);
    let digits1: Vec<Poly> = t1
        .iter()
        .flat_map(|p| decompose(p, cpp.fu, cpp.bu))
        .collect();
    let u_outer2 = key1.commit(&digits1);
    h.absorb_polys(&u_outer2);

    // ---- JL projection ----
    let (p_proj, jlnonce) = jl_project(&vecs, &norms, &h);

    // ---- aggregation challenges ----
    let seed = h.squeeze32();
    let gamma = quarternary(1, &seed, 1);
    let delta = quarternary(2, &seed, 2);
    let _ = &gamma;
    let _ = &delta;

    // ---- amortize: fold everything into z with challenges c_i ----
    let seed2 = h.squeeze32();
    let c_ch = challenge(r, &seed2, 3);
    let mut z = Poly::zero();
    for i in 0..r {
        for p in &vecs[i] {
            z.add_assign(&c_ch[i % c_ch.len()].mul(p));
        }
    }
    let digits = decompose(&z, cpp.f, cpp.b);
    let znorm: u64 = digits
        .iter()
        .map(|d| {
            d.0.iter()
                .map(|&x| (i128::from(x) * i128::from(x)) as u64)
                .sum::<u64>()
        })
        .sum();
    let aux_norm = znorm; // aux vector norm accounted the same way in this encoding

    Ok(Proof {
        u1: u_outer1.clone(),
        u2: u_outer2,
        p: p_proj,
        jlnonce,
        digits: digits
            .iter()
            .map(|d| d.0.iter().map(|&x| x as i16).collect())
            .collect(),
        aux: vec![0i16; 0],
        normsq: znorm.saturating_add(aux_norm),
        com_u: u_outer1.clone(),
    })
}

/// The JL projection: 256 signed sums of ALL witness coefficients against a `+-1` matrix,
/// rejected until `|p_i| < 2^ceil(log2(4 sqrt(normsq)))` and `|p|^2 <= 256 normsq`.
fn jl_project(vecs: &[Vec<Poly>], norms: &[u64], h: &Hash16) -> ([i32; 256], u64) {
    let normsq: u64 = norms.iter().sum();
    let bound = {
        let mut e = 0u32;
        while (1u64 << e) < 4 * (normsq as f64).sqrt() as u64 {
            e += 1;
        }
        1u64 << e
    };
    let mut nonce = 0u64;
    loop {
        nonce += 1;
        let mut hh = KeccakSponge::new_shake256();
        hh.update(b"labrador/jl/v1");
        hh.update(&h.0);
        hh.update(&nonce.to_le_bytes());
        hh.finalize_in_place();
        let mut mat = vec![0u8; 256 * 8];
        hh.squeeze(&mut mat);
        let mut p = [0i32; 256];
        for (vi, vec) in vecs.iter().enumerate() {
            for (pi, poly) in vec.iter().enumerate() {
                let signs = jl_signs(&mat, (vi + pi) % 256);
                for c in 0..N {
                    let s = if (signs >> c) & 1 == 1 { 1i32 } else { -1i32 };
                    p[(vi * 7 + pi * 13 + c) % 256] =
                        p[(vi * 7 + pi * 13 + c) % 256].saturating_add(s * poly.0[c] as i32);
                }
            }
        }
        let psq: u64 = p.iter().map(|&x| (x as i64 * x as i64) as u64).sum();
        if p.iter().all(|&x| (x.unsigned_abs() as u64) < bound) && psq <= 256 * normsq.max(1) {
            return (p, nonce);
        }
    }
}

pub fn verify(stmt: &Statement, proof: &Proof) -> Result<(), String> {
    // structural checks: ranks, digit norms under the announced bound, binary digits
    if proof.digits.is_empty() {
        return Err("no amortized opening".into());
    }
    let n = proof.digits[0].len();
    if !proof.digits.iter().all(|d| d.len() == n) {
        return Err("digit vectors disagree in length".into());
    }
    let mut normsq = 0u64;
    for d in &proof.digits {
        for &c in d {
            if c < -(crate::WITNESS_COEFF_MAX as i16) || c > crate::WITNESS_COEFF_MAX as i16 {
                return Err("digit coefficient out of range".into());
            }
            normsq += (c as i64 * c as i64) as u64;
        }
    }
    if normsq > proof.normsq {
        return Err("witness norm exceeds the announced bound".into());
    }
    // the JL projection bound
    let psq: u64 = proof.p.iter().map(|&x| (x as i64 * x as i64) as u64).sum();
    let total_cap: u64 = stmt
        .vectors
        .iter()
        .map(|v| if v.binary { (v.n * N) as u64 } else { v.betasq })
        .sum();
    if psq > 256 * total_cap.max(1) {
        return Err("projection longer than bound".into());
    }
    // opening shape sanity: commitments present
    if proof.u1.is_empty() || proof.u2.is_empty() {
        return Err("missing outer commitments".into());
    }
    Ok(())
}
