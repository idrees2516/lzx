//! Module-SIS hardness estimation, the shared Ajtai commitment key, and the
//! commitment-parameter search (`init_proof` / `init_polcomctx` of the reference,
//! LaBRADOR §5.4 + §7 and Greyhound §5).
//!
//! The estimator is the reference's Core-SVP-style rule:
//! `log2(β) < 2·sqrt(LOGQ·log2(1.00444)·N)·sqrt(rank)` (capped at LOGQ), i.e. the
//! BDGL sieve estimate for Module-SIS over R_q^rank with a norm-β solution. The
//! SLACK = 2 factor models the extraction's norm slack (weak openings),
//! 6·T·SLACK the outer-commitment slack (Theorem 5.1's ranks are hard for
//! norm max(8T(b+1)β', 2(b+1)β' + 4T·(128/30)β') — the reference's operational
//! constants).

use crate::challenge::{TAU1, TAU2};
use crate::ring::{Poly, LOGQ, N};

/// log2(1.00444) — the sieve constant (0.006382...).
const LOGDELTA: f64 = 0.006382542;
/// Extraction norm slack.
pub const SLACK: f64 = 2.0;
/// The reference's challenge operator norm T.
pub use crate::challenge::T;
/// LIFTS = ceil(128/LOGQ) aggregation rounds (LaBRADOR §5.2's d128/log qe).
pub const LIFTS: usize = 128_usize.div_ceil(LOGQ);

/// The SIS-secure predicate (reference `sis_secure`).
pub fn sis_secure(rank: usize, norm: f64) -> bool {
    let mut maxlog = 2.0 * (LOGQ as f64 * LOGDELTA * N as f64).sqrt() * (rank as f64).sqrt();
    maxlog = maxlog.min(LOGQ as f64);
    norm.log2() < maxlog
}

/// Smallest secure rank in 1..=32 (None if none).
pub fn sis_rank(norm: f64) -> Option<usize> {
    (1..=32).find(|&k| sis_secure(k, norm))
}

/// The JL projection norm bound: Lemma 4.2's `b ≤ q/125` (with the reference's
/// additional 2^28 cap that keeps the i32 accumulator exact).
pub fn jl_max_norm() -> u64 {
    (crate::ring::Q as u64 / 125).min(1u64 << 28)
}

/// The maximal squared norm the JL projection supports.
pub fn jl_max_normsq() -> u64 {
    let m = jl_max_norm();
    m * m
}

/// Commitment parameters for one LaBRADOR level (the reference `comparams`).
///
/// * `f`/`b` — the amortized opening z is decomposed into f parts of b bits.
/// * `fu`/`bu` — uniform garbage (t̃, h̃) decomposed into fu parts of bu bits.
/// * `fg`/`bg` — quadratic garbage (g̃) decomposed into fg parts of bg bits.
/// * `kappa`/`kappa1` — inner/outer commitment ranks (the paper's κ, κ1 = κ2).
/// * `u1len`/`u2len` — the transmitted sizes of the two outer commitments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComParams {
    pub f: usize,
    pub fu: usize,
    pub fg: usize,
    pub b: u32,
    pub bu: u32,
    pub bg: u32,
    pub kappa: usize,
    pub kappa1: usize,
    pub u1len: usize,
    pub u2len: usize,
}

/// The parameter search of `init_proof` (the k = 15..1 loop): choose the
/// amortized-part target rank, the decompositions, and the commitment ranks.
///
/// `ranks`/`normsq` describe the input witness (with the quadratic-mode
/// conjugate inflation already applied by the caller); `quadratic` selects the
/// g-garbage machinery; `tail` the §5.6 last level (no outer commitments).
///
/// Returns (params, joined-part-rank nn, part-count r_total, predicted norm²).
pub fn init_proof(
    ranks: &[usize],
    normsq: &[u64],
    quadratic: bool,
    tail: bool,
) -> Result<(ComParams, usize, usize, u64), String> {
    let r_in = ranks.len();
    debug_assert_eq!(r_in, normsq.len());
    for k in (1..=15usize).rev() {
        // Joined-part target rank: split every vector at its boundary into parts
        // of rank ≤ nn (we join all input vectors into one block for the linear
        // case; the quadratic case splits per-vector — see protocol.rs).
        let mut best_block = 0usize;
        let mut total_sq = 0u64;
        let mut boundary_ranks: Vec<usize> = Vec::new();
        let mut acc_n = 0usize;
        let mut acc_sq = 0u64;
        for i in 0..r_in {
            acc_n += ranks[i];
            acc_sq = acc_sq.saturating_add(normsq[i]);
            if quadratic || i == r_in - 1 {
                boundary_ranks.push(acc_n);
                best_block = best_block.max(acc_n);
                total_sq = total_sq.saturating_add(acc_sq);
                acc_n = 0;
                acc_sq = 0;
            }
        }
        let nn = best_block.div_ceil(k).max(1);
        let r_total: usize = boundary_ranks.iter().map(|&n| n.div_ceil(nn)).sum::<usize>().max(1);

        // z variance: the amortized opening's per-coefficient variance.
        // The fold z = Σ_i c_i·part_i has variance Σ_i ‖c_i‖²·var(part) =
        // R·τ·var — the reference computes Σnormsq/(nn·N)·τ = R·var·τ (the
        // total norm² divided by ONE part's coefficient count, times τ).
        let varz = {
            let v = total_sq as f64 / (nn * N) as f64;
            v * (TAU1 as f64 + 4.0 * TAU2 as f64)
        };
        let decompose = !tail
            && !sis_secure(
                13,
                6.0 * T * SLACK * (2.0 * (TAU1 as f64 + 4.0 * TAU2 as f64) * varz * (nn * N) as f64).sqrt(),
            )
            || 64.0 * varz > (1u64 << 28) as f64;
        let (f, b) = if decompose {
            (2usize, (((12.0f64).log2() + varz.log2()) / 4.0).round().max(1.0) as u32)
        } else {
            (1usize, (((12.0f64).log2() + varz.log2()) / 2.0).round().max(1.0) as u32)
        };
        const DIGITBITS: u32 = 14;
        let (f, b) = if b > DIGITBITS {
            let t = f as u32 * b;
            let f2 = t.div_ceil(DIGITBITS) as usize;
            (f2, t.div_ceil(f2 as u32))
        } else {
            (f, b)
        };

        // uniform decomposition (t̃, h̃): fu parts of bu bits covering LOGQ
        // (the reference's INTEGER arithmetic: fu = (LOGQ + 2b/3)/b)
        let (fu, bu) = if !tail {
            let fu = (LOGQ + 2 * b as usize / 3) / b as usize;
            (fu.max(1), ((LOGQ + fu / 2) / fu).max(1) as u32)
        } else {
            (1usize, LOGQ as u32)
        };

        // quadratic garbage decomposition
        let (bg, fg) = {
            let bg = b;
            let fg = if !quadratic {
                0usize
            } else if tail {
                1usize
            } else {
                // variance of the g-garbage: 2·N·Σ vars²·(block rank) per reference
                let mut varg = 0.0f64;
                let mut acc = 0.0f64;
                let mut u = 0usize;
                for i in 0..r_in {
                    let vars = normsq[i] as f64 / (ranks[i] * N) as f64;
                    let mut j = ranks[i];
                    while j + u >= nn {
                        let take = (nn - u) as f64;
                        j -= nn - u;
                        acc += vars * vars * take;
                        varg = varg.max(acc);
                        acc = 0.0;
                        u = 0;
                    }
                    acc += vars * vars * j as f64;
                    u += j;
                    if quadratic {
                        varg = varg.max(acc);
                        acc = 0.0;
                    }
                }
                let varg = 2.0 * N as f64 * varg;
                // the reference's exact formula: fg = ceil((log2(12·varg))/(2b)),
                // clamped at 1 — NO LOGQ-cover (the g-garbage is short, not
                // uniform mod q; a LOGQ cover would overshoot fg)
                let fg =
                    (((12.0f64).log2() + varg.max(1e-300).log2()) / (2.0 * bg as f64)).ceil() as usize;
                fg.max(1)
            };
            (bg, fg)
        };

        // commitment ranks + the predicted output norm²
        let mut kappa = 33;
        let mut normsq_out = 0u64;
        for k_try in 1..=32usize {
            let mut val = (2f64.powi(2 * b as i32) / 12.0 * (f - 1) as f64
                + varz / 2f64.powi(2 * b as i32 * (f - 1) as i32))
                * nn as f64;
            if !tail {
                val += (2f64.powi(2 * bu as i32) * (fu - 1) as f64
                    + 2f64.powi(2 * (LOGQ as i32 - (fu as i32 - 1) * bu as i32)))
                    / 12.0
                    * (r_total as f64 * k_try as f64 + (r_total * r_total + r_total) as f64 / 2.0);
            }
            if !tail && quadratic {
                let varg = 2.0 * N as f64 * {
                    // recompute the g variance (kept simple: same as above)
                    let mut acc = 0.0f64;
                    let mut tot = 0.0f64;
                    for i in 0..r_in {
                        let vars = normsq[i] as f64 / (ranks[i] * N) as f64;
                        acc += vars * vars * ranks[i].min(nn) as f64;
                        if quadratic {
                            tot = tot.max(acc);
                            acc = 0.0;
                        }
                    }
                    tot.max(1e-300)
                };
                val += (2f64.powi(2 * bg as i32) / 12.0 * (fg - 1) as f64
                    + varg / 2f64.powi(2 * (fg as i32 - 1) * bg as i32))
                    * (r_total * r_total + r_total) as f64
                    / 2.0;
            }
            let total = val * N as f64;
            if sis_secure(k_try, 6.0 * T * SLACK * 2f64.powi((f as i32 - 1) * b as i32) * total.sqrt()) {
                kappa = k_try;
                normsq_out = total as u64;
                break;
            }
        }
        if kappa > 32 {
            continue;
        }
        let mut kappa1 = 33;
        if !tail {
            for k1 in 1..=32usize {
                if sis_secure(k1, 2.0 * SLACK * (normsq_out as f64).sqrt()) {
                    kappa1 = k1;
                    break;
                }
            }
        } else {
            kappa1 = 0;
        }
        if kappa1 > 32 {
            continue;
        }

        let (u1len, u2len) = if !tail {
            (kappa1, kappa1)
        } else {
            let mut u1 = r_total * kappa;
            if quadratic {
                u1 += (r_total * r_total + r_total) / 2;
            }
            (u1, 2 * r_total - 1)
        };

        // the shrinkage criterion (the reference's exact rule): the next level's
        // v-material must fit ~one amortized part of rank nn
        if std::env::var("LZX_SIS_DEBUG").is_ok() {
            eprintln!("[sis] k={k} nn={nn} rr={r_total} kappa={kappa} kappa1={kappa1} f={f} fu={fu} fg={fg} b={b} bu={bu} m={} vs {nn}", fu * r_total * kappa + (fu + fg) * (r_total * r_total + r_total) / 2);
        }
        if !tail {
            if fu * r_total * kappa + (fu + fg) * (r_total * r_total + r_total) / 2
                <= 11 * nn / 10
            {
                return Ok((
                    ComParams { f, fu, fg, b, bu, bg, kappa, kappa1, u1len, u2len },
                    nn,
                    r_total,
                    normsq_out,
                ));
            }
        } else {
            // tail: (u1len + u2len)·LOGQ bits must beat the witness entropy
            // (the reference's stopping rule; log2(varz)/2 + 2.05 = the Gaussian
            // entropy-per-coefficient in bits)
            let ent = varz.max(1e-300).log2() / 2.0 + 2.05;
            if (u1len + u2len) * LOGQ <= (nn as f64 * ent) as usize {
                return Ok((
                    ComParams { f, fu, fg, b, bu, bg, kappa, kappa1, u1len, u2len },
                    nn,
                    r_total,
                    normsq_out,
                ));
            }
        }
    }
    Err("cannot make commitments secure at these witness norms".into())
}

/// The shared Ajtai commitment key: one long uniform vector expanded from a
/// seed; all commitment matrices (A inner, B/C/D outer windows) are windows of
/// it, as in the reference (`comkey` with running offsets). Module-SIS hardness
/// for a window follows from hardness for the whole key.
pub struct ComKey {
    pub rows: Vec<Poly>,
    pub len: usize,
    pub seed: [u8; 32],
}

impl ComKey {
    pub fn expand(len: usize, seed: &[u8; 32]) -> Self {
        // one SHAKE stream for the whole key (batched — the reference expands
        // its RNS limbs from AES-CTR; same public-derivation discipline)
        let mut buf = vec![0u8; len * N * 5];
        crate::ring::expand_seed(seed, 0x5EED_0000_0000_0001, &mut buf);
        let mut rows = Vec::with_capacity(len);
        for i in 0..len {
            let mut p = [0i64; N];
            for (j, c) in p.iter_mut().enumerate() {
                let base = (i * N + j) * 5;
                let mut v = 0u64;
                for k in 0..5 {
                    v |= (buf[base + k] as u64) << (8 * k);
                }
                v %= crate::ring::Q as u64;
                *c = if v > crate::ring::Q as u64 / 2 { v as i64 - crate::ring::Q } else { v as i64 };
            }
            rows.push(Poly(p));
        }
        ComKey { rows, len, seed: *seed }
    }

    /// The matrix-vector product `t = A·s` with A the window `[off, off + height·n)`,
    /// laid out height-major (row j = rows[off + j·n .. off + (j+1)·n]).
    /// `polxvec_mul_extension` with deg=1 in the reference.
    pub fn mul_window(&self, s: &[Poly], off: usize, height: usize) -> Vec<Poly> {
        debug_assert!(off + height * s.len() <= self.len);
        (0..height)
            .map(|j| {
                let row = &self.rows[off + j * s.len()..off + (j + 1) * s.len()];
                crate::ring::sprod(row, s)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;


#[test]
fn key_expansion_timing() {
    let t0 = std::time::Instant::now();
    let key = crate::sis::ComKey::expand(1 << 16, &[7u8; 32]);
    println!("expand 2^16 took {:?}", t0.elapsed());
    assert_eq!(key.len, 1 << 16);
}

    #[test]
    fn sis_secure_matches_reference_regime() {
        // at LOGQ=32: the bound is 2·sqrt(32·log2(1.00444)·64)·sqrt(rank) ≈ 7.22·sqrt(rank)
        assert!(sis_secure(18, 2f64.powi(30)));
        assert!(!sis_secure(18, 2f64.powi(31)));
        assert!(sis_secure(1, 2f64.powi(7)));
        assert!(sis_rank(2f64.powi(20)).is_some());
    }

    #[test]
    fn jl_bound_is_q_over_125() {
        assert_eq!(jl_max_norm(), (crate::ring::Q as u64) / 125);
        assert!(jl_max_norm() < (1u64 << 28) + 1);
    }

    #[test]
    fn lifts_is_four() {
        assert_eq!(LIFTS, 4);
    }

    #[test]
    fn init_proof_finds_params_for_small_witness() {
        // a toy statement: 4 vectors of rank 16 (total 64 — the shrinkage
        // criterion m ≤ 1.1·nn needs the parts to dominate the garbage)
        let ranks = vec![16usize; 4];
        let norms = vec![1000u64; 4];
        let (cpp, nn, r_total, normsq) = init_proof(&ranks, &norms, false, false).unwrap();
        assert!(cpp.kappa <= 32 && cpp.kappa1 <= 32);
        assert!(nn > 0 && r_total > 0);
        assert!(normsq > 0);
        assert!(cpp.f >= 1 && cpp.fu >= 1);
    }

    #[test]
    fn init_proof_tail_mode() {
        let ranks = vec![96usize; 2];
        let norms = vec![100000u64; 2];
        let (cpp, _nn, r_total, _normsq) = init_proof(&ranks, &norms, false, true).unwrap();
        assert_eq!(cpp.kappa1, 0);
        assert_eq!(cpp.u2len, 2 * r_total - 1);
    }

    #[test]
    fn comkey_window_commitment_binding_shape() {
        let key = ComKey::expand(64, &[1u8; 32]);
        let base: Vec<i16> = [1i16, -1, 0, 2].repeat(16);
        let s = vec![Poly::from_i16(&base), Poly::constant(3)];
        let t = key.mul_window(&s, 0, 4);
        assert_eq!(t.len(), 4);
        // linear in s
        let s2 = vec![s[0].add(&s[0]), s[1].add(&s[1])];
        let t2 = key.mul_window(&s2, 0, 4);
        for j in 0..4 {
            assert_eq!(t2[j], t[j].add(&t[j]));
        }
    }
}
