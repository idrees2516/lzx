//! The modular Johnson-Lindenstrauss projection (LaBRADOR §4, the JL part of
//! Figure 2; the reference's `jlproj.c` + `project`/`collaps_jlproj`).
//!
//! The verifier's random matrices Π_i ∈ {±1}^{256×(n·64)} (reference mode —
//! the paper's Lemma 4.1 uses ternary {−1,0,1} with P(0) = 1/2, and §6 of the
//! Greyhound paper documents the ±1 deviation as the implemented variant; both
//! modes are provided here, ±1 by default since the 53KB parameter tables are
//! calibrated for it). The prover sends `p = Π s` (256 values), the verifier
//! checks `‖p‖₂ ≤ √256·β` (mode-dependent constant), and correctness of `p mod
//! q` is proven by 256 constant-term dot-product constraints that the level's
//! aggregation folds in (the σ^{-1} packing of the paper's §5.2 "Projecting").
//!
//! Soundness side-conditions (Lemma 4.2): the witness norm bound must satisfy
//! `b ≤ q/125` — enforced by [`crate::sis::jl_max_norm`]. The rejection loop
//! re-derives the matrix from `(h, jlnonce)` so the verifier regenerates it.

use crate::ring::{cmod, Poly, N};

/// The matrix entries' distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JlMode {
    /// ±1 entries (reference; E‖p‖² = 256‖w‖²).
    PlusMinus1,
    /// Ternary {−1,0,1} with P(0)=1/2 (paper Lemma 4.1; E‖p‖² = 128‖w‖²).
    Ternary,
}

/// One JL matrix: `rows` × `cols` sign entries, one bit (±1 mode) or two bits
/// (ternary) per entry, expanded from a seed. Stored row-major packed.
pub struct JlMatrix {
    pub rows: usize,
    pub cols: usize,
    pub mode: JlMode,
    /// Packed entries; ±1 mode: 1 bit per entry; ternary: 2 bits.
    pub bits: Vec<u8>,
}

impl JlMatrix {
    /// Deterministic expansion from (seed, nonce) — the prover and verifier
    /// derive the same matrix.
    pub fn expand(rows: usize, cols: usize, mode: JlMode, seed: &[u8], nonce: u64) -> Self {
        let bits_per = if mode == JlMode::Ternary { 2 } else { 1 };
        let nbytes = (rows * cols * bits_per).div_ceil(8);
        let mut bits = vec![0u8; nbytes];
        crate::ring::expand_seed(seed, nonce, &mut bits);
        Self { rows, cols, mode, bits }
    }

    /// The (row, col) entry ∈ {−1, 0, 1} (ternary) or {−1, 1} (±1 mode).
    #[inline]
    pub fn entry(&self, row: usize, col: usize) -> i8 {
        match self.mode {
            JlMode::PlusMinus1 => {
                let bitpos = row * self.cols + col;
                if (self.bits[bitpos / 8] >> (bitpos % 8)) & 1 == 1 {
                    1
                } else {
                    -1
                }
            }
            JlMode::Ternary => {
                let bitpos = (row * self.cols + col) * 2;
                let v = (self.bits[bitpos / 8] >> (bitpos % 8)) & 3;
                match v {
                    0 | 1 => 0,
                    2 => 1,
                    _ => -1,
                }
            }
        }
    }

    /// The matrix-vector product over the integers (exact; the accumulator is
    /// i64 — for the soundness regime Σ|w|·‖w‖∞ stays below 2^31 by the
    /// JLMAXNORM bound, so i32 would also fit, as in the reference).
    pub fn project(&self, w: &[i64]) -> Vec<i64> {
        debug_assert_eq!(w.len(), self.cols);
        let mut out = vec![0i64; self.rows];
        for (c, &wc) in w.iter().enumerate() {
            if wc == 0 {
                continue;
            }
            for r in 0..self.rows {
                out[r] += self.entry(r, c) as i64 * wc;
            }
        }
        out
    }

    /// Row `r` restricted to the ring element `k` of a witness vector, packed
    /// back as a ring element π^{(r)}_k with the matrix's sign pattern in its
    /// coefficients (the paper's π_i^{(j)}).
    pub fn row_poly(&self, row: usize, elem: usize) -> Poly {
        let mut p = [0i64; N];
        for j in 0..N {
            p[j] = self.entry(row, elem * N + j) as i64;
        }
        Poly(p)
    }

    /// Row `r` as a vector of ring elements (for a witness vector of `n_elems`
    /// ring elements).
    pub fn row_polys(&self, row: usize, n_elems: usize) -> Vec<Poly> {
        (0..n_elems).map(|k| self.row_poly(row, k)).collect()
    }
}

/// The per-vector JL matrices Π_i and the projected value p (the level's state).
pub struct JlProjection {
    pub p: Vec<i64>,
    pub nonce: u64,
    /// One matrix per joined part (cols = part_rank·N).
    pub mats: Vec<JlMatrix>,
}

/// The rejection bound of the reference: |p_i| < next_power_of_2(4·√normsq),
/// and Σ p² ≤ 256·normsq (±1 mode; 128·normsq for ternary — the paper's
/// √128·β check).
pub fn jl_accept(p: &[i64], normsq: u64, mode: JlMode) -> bool {
    let factor = if mode == JlMode::Ternary { 128u64 } else { 256u64 };
    let bound = {
        let mut e = 0u32;
        while (1u64 << e) < 4 * (normsq as f64).sqrt() as u64 {
            e += 1;
        }
        1u64 << e
    };
    let psq: u64 = p.iter().map(|&x| (x * x) as u64).sum();
    let cap = factor.saturating_mul(normsq.max(1));
    p.iter().all(|&x| x.unsigned_abs() < bound) && psq <= cap
}

/// The squared norm of the projection (the size-model input).
pub fn jl_normsq(p: &[i64]) -> u64 {
    p.iter().map(|&x| (x * x) as u64).sum()
}

/// Project the joined parts: one matrix per part, p = Σ_i Π_i·coeffs(s_i).
/// Retries with fresh nonces until accepted (the reference's `project`).
pub fn project_parts(
    parts: &[Vec<Poly>],
    mode: JlMode,
    seed: &[u8],
) -> JlProjection {
    let normsq: u64 = parts
        .iter()
        .map(|v| v.iter().map(|p| p.normsq()).sum::<u64>())
        .sum();
    let mut nonce = 0u64;
    loop {
        nonce += 1;
        let mats: Vec<JlMatrix> = parts
            .iter()
            .map(|v| JlMatrix::expand(256, v.len() * N, mode, seed, nonce.wrapping_mul(parts.len() as u64) + v.len() as u64))
            .collect();
        let mut p = vec![0i64; 256];
        for (i, v) in parts.iter().enumerate() {
            let flat: Vec<i64> = v.iter().flat_map(|poly| poly.0.iter().copied()).collect();
            let sub = mats[i].project(&flat);
            for (r, &x) in sub.iter().enumerate() {
                p[r] += x;
            }
        }
        if jl_accept(&p, normsq, mode) {
            return JlProjection { p, nonce, mats };
        }
    }
}

/// Collapse the 256 JL equations into one constant-term constraint with Z_q
/// challenges ω (the paper's aggregation of the JL rows with ω^{(k)} ∈ Z_q^256):
/// returns (Φ per part, target ct) where the constraint is
/// Σ_i ⟨Φ_i, s_i⟩ ≡ ⟨ω, p⟩ (mod q) at the constant term.
///
/// Φ_i = Σ_j ω_j·σ^{-1}(π_i^{(j)}) — note the σ^{-1} packing: the paper's
/// constraint ⟨σ^{-1}(π_i^{(j)}), s_i⟩ uses the conjugated row so that the
/// constant term of the ring product equals the coefficient dot product.
pub fn collapse_jl(
    mats: &[JlMatrix],
    p: &[i64],
    omega: &[i64],
) -> (Vec<Vec<Poly>>, i64) {
    debug_assert_eq!(omega.len(), 256);
    let mut phis = Vec::with_capacity(mats.len());
    for (i, m) in mats.iter().enumerate() {
        let n_elems = m.cols / N;
        let mut phi = vec![Poly::zero(); n_elems];
        for (j, &w) in omega.iter().enumerate() {
            if w == 0 {
                continue;
            }
            let row = m.row_polys(j, n_elems);
            for (k, rp) in row.iter().enumerate() {
                // σ^{-1}(π) has ±1 coefficients in the same support; the
                // conjugation of a ±1 pattern: σ^{-1}(Σ e_i X^i) = e_0 - Σ_{i≥1} e_i X^{64-i}
                let conj = rp.sigma_m1();
                phi[k].add_assign(&conj.scale(w));
            }
        }
        let _ = i;
        phis.push(phi);
    }
    let target: i64 = omega
        .iter()
        .zip(p.iter())
        .map(|(&w, &x)| cmod(w as i128 * x as i128))
        .fold(0i64, |acc, x| cmod(acc as i128 + x as i128));
    (phis, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_parts() -> Vec<Vec<Poly>> {
        let mut s0 = vec![Poly::zero(); 2];
        for (i, p) in s0.iter_mut().enumerate() {
            let mut coeffs = [0i64; N];
            for (j, c) in coeffs.iter_mut().enumerate() {
                *c = (((i * 71 + j * 13) % 7) as i64) - 3;
            }
            *p = Poly(coeffs);
        }
        vec![s0]
    }

    #[test]
    fn projection_norm_scaling() {
        let parts = small_parts();
        let normsq: u64 = parts.iter().map(|v| v.iter().map(|p| p.normsq()).sum::<u64>()).sum();
        let proj = project_parts(&parts, JlMode::PlusMinus1, b"jl");
        // the acceptance guarantees the bound
        assert!(jl_normsq(&proj.p) <= 256 * normsq.max(1));
        let proj_t = project_parts(&parts, JlMode::Ternary, b"jl");
        assert!(jl_normsq(&proj_t.p) <= 128 * normsq.max(1));
        // E[p²] ≈ 256·normsq for ±1: statistical sanity (loose)
        let m = JlMatrix::expand(256, 128, JlMode::PlusMinus1, b"m", 1);
        let w: Vec<i64> = (0..128).map(|i| ((i % 5) - 2) as i64).collect();
        let p = m.project(&w);
        let wsq: u64 = w.iter().map(|&x| (x * x) as u64).sum();
        let psq = jl_normsq(&p);
        assert!(psq > 100 * wsq && psq < 1000 * wsq, "psq={psq} wsq={wsq}");
    }

    #[test]
    fn collapse_matches_projection() {
        // the collapsed constraint evaluated at the witness must have ct = ⟨ω, p⟩
        let parts = small_parts();
        let seed = b"jl-collapse";
        let normsq: u64 = parts.iter().map(|v| v.iter().map(|p| p.normsq()).sum::<u64>()).sum();
        // build a fixed (non-rejected) projection
        let mats: Vec<JlMatrix> = parts
            .iter()
            .map(|v| JlMatrix::expand(256, v.len() * N, JlMode::PlusMinus1, seed, 7))
            .collect();
        let mut p = vec![0i64; 256];
        for (i, v) in parts.iter().enumerate() {
            let flat: Vec<i64> = v.iter().flat_map(|poly| poly.0.iter().copied()).collect();
            let sub = mats[i].project(&flat);
            for (r, &x) in sub.iter().enumerate() {
                p[r] += x;
            }
        }
        let _ = normsq;
        let omega: Vec<i64> = (0..256).map(|i| (i * 2654435761u64 % (crate::ring::Q as u64 - 2) + 1) as i64).collect();
        let (phis, target) = collapse_jl(&mats, &p, &omega);
        // evaluate Σ_i ⟨Φ_i, s_i⟩ — constant term must equal target
        let mut acc = Poly::zero();
        for (i, phi) in phis.iter().enumerate() {
            acc.add_assign(&crate::ring::sprod(phi, &parts[i]));
        }
        assert_eq!(acc.constant_term(), target);
    }

    #[test]
    fn entry_distributions() {
        let m = JlMatrix::expand(256, 640, JlMode::PlusMinus1, b"e", 3);
        let mut plus = 0;
        let mut minus = 0;
        for r in 0..m.rows {
            for c in 0..m.cols {
                match m.entry(r, c) {
                    1 => plus += 1,
                    -1 => minus += 1,
                    _ => panic!("bad ±1 entry"),
                }
            }
        }
        assert_eq!(plus + minus, 256 * 640);
        let t = JlMatrix::expand(256, 640, JlMode::Ternary, b"t", 3);
        let mut zeros = 0;
        for r in 0..t.rows {
            for c in 0..t.cols {
                if t.entry(r, c) == 0 {
                    zeros += 1;
                }
            }
        }
        // P(0) = 1/2: half of the 2-bit patterns map to 0
        assert!(zeros > 256 * 640 * 45 / 100 && zeros < 256 * 640 * 55 / 100);
    }
}
