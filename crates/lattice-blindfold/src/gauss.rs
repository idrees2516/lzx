//! Discrete Gaussians and rejection sampling (§3.6, "A brief recall on
//! rejection sampling", following [LNP22, §2.6]).
//!
//! * `D_s^ℓ` over Z (per coefficient), with the tail bound of Lemma 3.19
//!   ([Ban93, Mic16]): Pr[∥z∥ > τ·s] < 2·ℓd·e^{−πτ²};
//! * **Rej1** (Lemma 3.20 / [LNP22, Lemma 2.4]): sample v ← ρ, y ← D_s,
//!   z = y + v; accept with probability min{1, D_s(z)/(M·D_{s,v}(z))};
//!   conditioned on acceptance (v, z) is within statistical distance
//!   2^{−λ} of ρ × D_s — the blinding engine of Π'_RLC;
//! * **Rej2** (the [LNP22, Lemma 2.14] sign-flip branch): sample z2
//!   conditioned on ⟨s2, z2⟩ ≥ 0;
//! * the **width calibration** of Eq (4.18):
//!   α^(l)(Nc, nfold) = √(nfold·(Bfold−1)² + (Nc−nfold)(B̃−1)²·m_l·d),
//!   s_l = γ_l·η·α^(l);
//! * M = exp(14/ξ + 1/(2ξ²)) (Lemma 3.20's constant at λ = 128, used
//!   unchanged at λ = 120 "which is conservative") and the attempt
//!   budgets Wmax = ⌈λ/log2(M/(M−1))⌉, Wmax^PoK.

use crate::fp::Fq;
use crate::ring::Poly;

/// A deterministic seeded RNG (xorshift128+ style) used everywhere in the
/// crate so protocol transcripts are replayable.
#[derive(Clone)]
pub struct Rng {
    s: [u64; 2],
}

impl Rng {
    pub fn new(seed: &[u8]) -> Rng {
        // FNV-1a expand to two lanes.
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in seed {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let mut h2: u64 = 0x9e3779b97f4a7c15;
        for &b in seed.iter().rev() {
            h2 ^= (b as u64).wrapping_mul(0x2545f4914f6cdd1d);
            h2 = h2.rotate_left(17);
        }
        if h == 0 {
            h = 0x853c49e6748fea9b;
        }
        if h2 == 0 {
            h2 = 0xda3e39cb94b95bdb;
        }
        Rng { s: [h, h2] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut s1 = self.s[0];
        let s0 = self.s[1];
        let result = s0.wrapping_add(s1);
        self.s[0] = s0;
        s1 ^= s1 << 23;
        self.s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        result
    }

    /// Uniform f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in [0, n) (n ≥ 1).
    pub fn below(&mut self, n: u64) -> u64 {
        if n <= 1 {
            return 0;
        }
        // rejection-free modulo-bias elimination for small n
        let zone = u64::MAX - u64::MAX % n;
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % n;
            }
        }
    }

    /// Uniform {0, 1, 2}-adic ternary for salts (B̃ = 2).
    pub fn ternary(&mut self) -> i64 {
        match self.below(3) {
            0 => 0,
            1 => 1,
            _ => -1,
        }
    }

    /// Uniform in [−(b−1), b−1] (χ_b, small witnesses).
    pub fn small_b(&mut self, b: i64) -> i64 {
        let v = self.below((2 * b - 1) as u64) as i64;
        v - (b - 1)
    }

    /// A discrete Gaussian sample over Z with standard deviation s,
    /// sampled by rejection inside [−B, B] with B = ⌈τ·s⌉ (the tails
    /// beyond τ·s carry < 2^{−λ} mass — Lemma 3.19).
    pub fn gaussian(&mut self, s: f64, tau: f64) -> i64 {
        let b = (tau * s).ceil() as i64;
        loop {
            let x = self.below((2 * b + 1) as u64) as i64 - b;
            // density ∝ exp(−x²/(2s²)); accept with that probability.
            let p = (-(x as f64 * x as f64) / (2.0 * s * s)).exp();
            if self.next_f64() < p {
                return x;
            }
        }
    }

    /// A ring element of independent Gaussians.
    pub fn gaussian_poly(&mut self, d: usize, s: f64, tau: f64) -> Poly {
        Poly(
            (0..d)
                .map(|_| Fq::from_i64(self.gaussian(s, tau)))
                .collect(),
        )
    }
}

/// The repetition rate M = exp(14/ξ + 1/(2ξ²)) (Lemma 3.20, the constant
/// 14 derived in [LNP22, Lemma 2.14] at λ = 128).
pub fn repetition_rate(xi: f64) -> f64 {
    (14.0 / xi + 1.0 / (2.0 * xi * xi)).exp()
}

/// The attempt budget Wmax = ⌈λ / log2(M/(M−1))⌉ (§2.1, Remark 4.3).
pub fn attempt_budget(lambda: f64, m: f64) -> u32 {
    let ratio = m / (m - 1.0);
    (lambda / ratio.log2()).ceil() as u32
}

/// τ_{λ,ξ} ≥ sqrt((λ + log2(2 nF)) ln 2 / π) (Lemma 3.19).
pub fn tau_lambda(lambda: f64, nf: f64) -> f64 {
    ((lambda + (2.0 * nf).log2()) * std::f64::consts::LN_2 / std::f64::consts::PI).sqrt()
}

/// The Rej1 rejection-sampling decision (Lemma 3.20):
/// accept (return true) with probability
/// min{1, D_s(z) / (M · D_{s,v}(z))} where D_s(z) ∝ exp(−∥z∥²/2s²) and
/// D_{s,v}(z) = D_s(z − v). Equivalently:
/// accept iff u < exp(−(2⟨z,v⟩ + ∥v∥²)/(2s²)) / M  for u ← [0,1).
pub fn rej1_decide(rng: &mut Rng, z: &[i64], v: &[i64], s: f64, m_rate: f64) -> bool {
    // Compute ⟨z, v⟩ and ∥v∥² over the integer representatives.
    let mut zv: i128 = 0;
    let mut v2: i128 = 0;
    for (a, b) in z.iter().zip(v.iter()) {
        zv += *a as i128 * *b as i128;
        v2 += *b as i128 * *b as i128;
    }
    // Ratio = exp((∥z−v∥² − ∥z∥²)/2s²) = exp((−2⟨z,v⟩ + ∥v∥²)/2s²).
    let expo = (-2.0 * zv as f64 + v2 as f64) / (2.0 * s * s);
    let accept_prob = (if expo < -700.0 { 0.0 } else { expo.exp() }) / m_rate;
    let u = rng.next_f64();
    u < accept_prob.min(1.0)
}

/// Extract the integer representatives of a poly vector (for Rej1 bookkeeping).
pub fn poly_syms(polys: &[Poly]) -> Vec<i64> {
    let mut out = Vec::with_capacity(polys.len() * polys.first().map(|p| p.d()).unwrap_or(0));
    for p in polys {
        for c in &p.0 {
            out.push(c.sym());
        }
    }
    out
}

/// Rej2 (the sign-flip branch of [LNP22, Lemma 2.14]): the acceptance
/// condition ⟨s2, z2⟩ ≥ 0. Implemented as rejection (resample on failure),
/// which realizes the conditional distribution directly.
pub fn rej2_accept(s2: &[Poly], z2: &[Poly]) -> bool {
    let mut ip: i128 = 0;
    for (s, z) in s2.iter().zip(z2.iter()) {
        for (a, b) in s.0.iter().zip(z.0.iter()) {
            ip += a.sym() as i128 * b.sym() as i128;
        }
    }
    ip >= 0
}

/// The width calibration of Eq (4.18):
/// α^(l)(Nc, nfold) = sqrt(nfold·(Bfold−1)² + (Nc−nfold)·(B̃−1)²·m_l·d)
/// and s_l = γ_l·η·α^(l).
#[allow(clippy::too_many_arguments)]
pub fn calibrate_widths(
    nc: usize,
    nfold: usize,
    bfold: i64,
    b_tilde: i64,
    m1: usize,
    m2: usize,
    d: usize,
    gamma1: f64,
    gamma2: f64,
    eta: f64,
) -> (f64, f64) {
    let nf_f = nfold as f64;
    let fresh = (nc - nfold) as f64;
    let bt = (b_tilde - 1) as f64;
    let bf = (bfold - 1) as f64;
    let alpha1 = (nf_f * bf * bf + fresh * bt * bt * m1 as f64 * d as f64).sqrt();
    let alpha2 = (nf_f * bf * bf + fresh * bt * bt * m2 as f64 * d as f64).sqrt();
    let s1 = gamma1 * eta * alpha1;
    let s2 = gamma2 * eta * alpha2;
    (s1, s2)
}

/// The masking width for Π'_RLC's witness loop (§3.6):
/// s = ξ·(K + k)·T·(b − 1)·√nF — hmm, the paper's formula:
/// "s := ξ · (K + k) · T · (b − 1) · √nF" — note this is the *deviation*
/// before the τ·s < B check with L := (K+k)T(b−1) (Lemma 3.20's L·√nF
/// shape: s = ξ·L·√nF with L = (K+k)·T·(b−1)).
pub fn rlc_mask_width(xi: f64, k_plus_capital_k: usize, t_exp: i64, b: i64, nf: usize) -> f64 {
    let l_bound = k_plus_capital_k as f64 * t_exp as f64 * (b - 1) as f64;
    xi * l_bound * (nf as f64).sqrt()
}

/// The k-digits requirement (Remark 4.18, Eq (4.17)):
/// τ_{λ,ξ}·ξ·(K+k)·T·(b−1)·√nF < b^k — returns the smallest k.
pub fn required_k(
    lambda: f64,
    xi: f64,
    k_plus_capital_k: usize,
    t_exp: i64,
    b: i64,
    nf: usize,
) -> usize {
    let tau = tau_lambda(lambda, nf as f64);
    let lhs =
        tau * xi * k_plus_capital_k as f64 * t_exp as f64 * (b - 1) as f64 * (nf as f64).sqrt();
    let mut k = 1u32;
    while (b as f64).powi(k as i32) <= lhs && k < 64 {
        k += 1;
    }
    k as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_moments() {
        let mut rng = Rng::new(b"gauss");
        let s = 10.0;
        let tau = 6.0;
        let n = 4000;
        let mut mean = 0.0f64;
        let mut var = 0.0f64;
        let mut count = 0usize;
        for _ in 0..n {
            let x = rng.gaussian(s, tau) as f64;
            mean += x;
            var += x * x;
            count += 1;
        }
        mean /= count as f64;
        var /= count as f64;
        assert!(mean.abs() < 0.6, "mean ~0, got {mean}");
        assert!(
            (var - s * s).abs() / (s * s) < 0.12,
            "variance ~s², got {var} vs {}",
            s * s
        );
    }

    #[test]
    fn tau_lambda_values_match_paper() {
        // §3.6: τ ≈ 5.50 for nF = 2^16 and ≈ 5.58 for nF = 2^20 at λ=120.
        let t16 = tau_lambda(120.0, 65536.0);
        let t20 = tau_lambda(120.0, 1048576.0);
        assert!((t16 - 5.50).abs() < 0.05, "t16 = {t16}");
        assert!((t20 - 5.58).abs() < 0.05, "t20 = {t20}");
        let t128 = tau_lambda(128.0, 65536.0);
        assert!((t128 - 5.657).abs() < 0.02, "t128 = {t128}");
    }

    #[test]
    fn repetition_rate_and_budgets() {
        // §4.3.4: ξ ≈ 13 gives M ≈ 3 repetitions; ξ ≈ 20 gives ≈ 2.
        let m13 = repetition_rate(13.0);
        assert!((m13 - 3.0).abs() < 0.7, "M(13) = {m13}");
        let m20 = repetition_rate(20.0);
        assert!((m20 - 2.0).abs() < 0.3, "M(20) = {m20}");
        // Remark 4.3: Wmax = 206 for M ≈ 3, 120 for M ≈ 2 at λ = 120.
        let w3 = attempt_budget(120.0, m13);
        let w2 = attempt_budget(120.0, m20);
        assert!((w3 as i64 - 206).abs() < 8, "Wmax(M≈3) = {w3}");
        assert!((w2 as i64 - 120).abs() < 6, "Wmax(M≈2) = {w2}");
    }

    #[test]
    fn rej1_flattens_the_distribution() {
        // Statistical heart of blinding: conditioned on acceptance,
        // (v, z) ≈ ρ × D_s — i.e. z's distribution is independent of v.
        // The width must satisfy the LNP22 calibration (s ≥ ξ·∥v∥₂ so
        // that M bounds the likelihood ratio without capping — a capped
        // ratio silently biases the accepted distribution).
        let mut rng = Rng::new(b"rej1");
        let s = 60.0;
        let m_rate = 3.0;
        let dim = 256;
        let tau = 6.0;
        // Secrets ±1: ∥v∥₂ = 16, and max ratio ≈ exp(3·16/60 + …) < M.
        let v1 = vec![1i64; dim];
        let v2 = vec![-1i64; dim];
        let mut means = [0.0f64; 2];
        let mut counts = [0usize; 2];
        let mut attempts = 0usize;
        while (counts[0] < 300 || counts[1] < 300) && attempts < 200_000 {
            attempts += 1;
            let which = if counts[0] < 300 { 0 } else { 1 };
            let v = if which == 0 { &v1 } else { &v2 };
            let y: Vec<i64> = (0..dim).map(|_| rng.gaussian(s, tau)).collect();
            let z: Vec<i64> = y.iter().zip(v.iter()).map(|(a, b)| a + b).collect();
            if rej1_decide(&mut rng, &z, v, s, m_rate) {
                means[which] += z.iter().map(|&x| x as f64).sum::<f64>();
                counts[which] += 1;
            }
        }
        assert!(counts[0] >= 300 && counts[1] >= 300, "both should accept");
        for i in 0..2 {
            means[i] /= counts[i] as f64;
        }
        // Under D_s the per-coordinate mean is 0: the mean of the
        // per-sample SUMS is ~ N(0, s·√dim/√count). The unflattened
        // shift would be ±dim·1 = ±256 — the accepted means must sit far
        // below that.
        let mean_std = s * (dim as f64).sqrt() / (300.0f64).sqrt();
        assert!(
            means[0].abs() < 5.0 * mean_std,
            "mean0 ~ N(0, s√dim/√n): {means:?} (std {mean_std})"
        );
        assert!(
            means[1].abs() < 5.0 * mean_std,
            "mean1 ~ N(0, s√dim/√n): {means:?} (std {mean_std})"
        );
        // And critically: the two means do not separate along the sign
        // of v (an unflattened transcript would show ±dim).
        assert!(
            (means[0] - means[1]).abs() < 8.0 * mean_std,
            "means must not separate: {means:?}"
        );
        assert!(
            means.iter().all(|m| m.abs() < 0.4 * dim as f64),
            "means far below the unflattened shift ±{}: {means:?}",
            dim
        );
    }

    #[test]
    fn required_k_matches_paper_rows() {
        // §3.6's rows: k = 29 for T=128, d=64, nF=2^20, ξ≈13
        // (K = b−1 = 1 so K+k uses k itself... the paper's rows assume the
        // self-consistent k; check the ballpark).
        let k = required_k(120.0, 13.0, 31, 128, 2, 1_048_576);
        assert!((24..=40).contains(&k), "k = {k}");
    }

    #[test]
    fn width_calibration_grows_with_nfold() {
        let (s1a, _) = calibrate_widths(32, 0, 3970, 2, 15, 83, 64, 1.0, 1.0, 1.0);
        let (s1b, _) = calibrate_widths(32, 1, 3970, 2, 15, 83, 64, 1.0, 1.0, 1.0);
        assert!(s1b > s1a, "a folded block inflates the width");
    }
}
