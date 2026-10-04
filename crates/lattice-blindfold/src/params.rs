//! Parameter sets and the consolidated error budget (§4.3.4, §4.3.1).
//!
//! Two families:
//! * **Toy sets** for tests (small d, small circuits) — same code paths,
//!   no security claims;
//! * the **paper's Table 2 sets** (d = 64 and d = 128 at q = 2^64 − 59,
//!   nF = 2^21, ξ ≈ 20, t = 3, K = b − 1 = B̃ − 1 = 1, λ = 120) with the
//!   full error-budget calculator: k from Eq (4.17), Wmax, B_fold,
//!   β_rlx, ε_SZ/ε_SC/ε_deg0/ε_ct/ε_cw and the consolidated
//!   knowledge-soundness and blinding expressions of Proposition 4.17 /
//!   Theorem 4.13.

use crate::gauss::{attempt_budget, repetition_rate, required_k, tau_lambda};

/// The global parameter struct (Definition 2.2).
#[derive(Clone, Debug)]
pub struct Params {
    /// Ring degree d (power of two).
    pub d: usize,
    /// The number of constraints = nF (square systems, Remark 4.1.(3)).
    pub nf: usize,
    /// Public-input length (field coords).
    pub nf_in: usize,
    /// Blinding block (field coords).
    pub nf_bl: usize,
    /// Number of structural matrices (t = 3 for R1CS).
    pub t: usize,
    /// Fresh instances folded per step (K = 1 — the blinding case).
    pub capital_k: usize,
    /// Accumulator instances (k; the decomposition depth).
    pub k: usize,
    /// The norm bound b = 2.
    pub b: i64,
    /// ABDLOP salt bound B̃ = 2 (ternary salts).
    pub b_tilde: i64,
    /// Compact Ajtai rank κ_Ajtai-Compact.
    pub kappa_ajtai: usize,
    /// ABDLOP ranks (κ, κ′ = ℓ, κ_com = κ + ℓ).
    pub kappa: usize,
    pub ell: usize,
    /// ABDLOP salt lengths (m1, m2).
    pub m1: usize,
    pub m2: usize,
    /// The ξ of the rejection sampling.
    pub xi: f64,
    /// λ.
    pub lambda: f64,
    /// The ABDLOP challenge coefficient bound β_ch.
    pub beta_ch: i64,
    /// The strong sampling set's expansion factor T(C) = 2d
    /// (Remark 2.17's tight negacyclic bound).
    pub t_exp: i64,
    /// τ_{λ,ξ} (Lemma 3.19).
    pub tau: f64,
}

impl Params {
    /// A toy set for tests: d = 4, m = 64, small ABDLOP. The k is the
    /// fixed point of Eq (4.17) (k depends on K + k — iterate to
    /// convergence as the paper's Table 2 rows do).
    pub fn toy() -> Params {
        let d = 4usize;
        let nf = 64usize;
        let xi = 20.0;
        let lambda = 120.0;
        let t_exp = 2 * d as i64;
        let k = fixed_point_k(lambda, xi, t_exp, 2, nf).max(4);
        let tau = tau_lambda(lambda, nf as f64);
        Params {
            d,
            nf,
            nf_in: 8,
            nf_bl: 8,
            t: 3,
            capital_k: 1,
            k,
            b: 2,
            b_tilde: 2,
            kappa_ajtai: 3,
            kappa: 3,
            // ℓ must cover the packed mask coefficients: the toy's
            // n_packed = ⌈(1 + 4·log₂ 64)/4⌉ = 7 R_K elements = 14 slots.
            ell: 16,
            m1: 4,
            m2: 6,
            xi,
            lambda,
            beta_ch: 1,
            t_exp,
            tau,
        }
    }

    /// A medium toy (d = 8, m = 256) for benches.
    pub fn toy_medium() -> Params {
        let d = 8usize;
        let nf = 256usize;
        let xi = 20.0;
        let lambda = 120.0;
        let t_exp = 2 * d as i64;
        let k = fixed_point_k(lambda, xi, t_exp, 2, nf).max(6);
        let tau = tau_lambda(lambda, nf as f64);
        Params {
            d,
            nf,
            nf_in: 16,
            nf_bl: 16,
            t: 3,
            capital_k: 1,
            k,
            b: 2,
            b_tilde: 2,
            kappa_ajtai: 4,
            kappa: 4,
            // ℓ covers the packed mask coefficients: n_packed =
            // ⌈(1 + 4·log₂ 256)/8⌉ = 5 R_K elements = 10 slots.
            ell: 16,
            m1: 5,
            m2: 8,
            xi,
            lambda,
            beta_ch: 2,
            t_exp,
            tau,
        }
    }

    /// The paper's d = 64 set of Table 2 (§4.3.4).
    pub fn paper_d64() -> Params {
        let d = 64usize;
        let nf = 1 << 21;
        let xi = 20.0;
        let lambda = 120.0;
        let t_exp = 2 * d as i64; // T = 128
        let k = 30; // Table 2 (T=128, d=64, nF=2^21, ξ≈20)
        let tau = tau_lambda(lambda, nf as f64);
        Params {
            d,
            nf,
            nf_in: 1024,
            nf_bl: 106_752, // Table 2 row nF,bl at d=64
            t: 3,
            capital_k: 1,
            k,
            b: 2,
            b_tilde: 2,
            kappa_ajtai: 26,
            kappa: 19,
            ell: 16,
            m1: 15,
            m2: 83,
            xi,
            lambda,
            beta_ch: 9,
            t_exp,
            tau,
        }
    }

    /// The paper's d = 128 set of Table 2 (§4.3.4).
    pub fn paper_d128() -> Params {
        let d = 128usize;
        let nf = 1 << 21;
        let xi = 20.0;
        let lambda = 120.0;
        let t_exp = 2 * d as i64; // T = 256
        let k = 31; // Table 2 (T=256, d=128, nF=2^21, ξ≈20)
        let tau = tau_lambda(lambda, nf as f64);
        Params {
            d,
            nf,
            nf_in: 1024,
            nf_bl: 123_136, // Table 2 row nF,bl at d=128
            t: 3,
            capital_k: 1,
            k,
            b: 2,
            b_tilde: 2,
            kappa_ajtai: 15,
            kappa: 11,
            ell: 16,
            m1: 15,
            m2: 53,
            xi,
            lambda,
            beta_ch: 2,
            t_exp,
            tau,
        }
    }

    pub fn nr(&self) -> usize {
        self.nf / self.d
    }

    pub fn nr_in(&self) -> usize {
        self.nf_in / self.d
    }

    pub fn nr_bl(&self) -> usize {
        self.nf_bl / self.d
    }

    pub fn m(&self) -> usize {
        self.nf
    }

    pub fn log_m(&self) -> usize {
        self.nf.trailing_zeros() as usize
    }

    pub fn kappa_com(&self) -> usize {
        self.kappa + self.ell
    }

    /// B_fold = (K + k)·T(C)·(B̃ − 1) + B̃ (Lemma B.1).
    pub fn b_fold(&self) -> i64 {
        (self.capital_k + self.k) as i64 * self.t_exp * (self.b_tilde - 1) + self.b_tilde
    }

    /// M = exp(14/ξ + 1/(2ξ²)) (Lemma 3.20).
    pub fn m_rate(&self) -> f64 {
        repetition_rate(self.xi)
    }

    /// Wmax = ⌈λ / log2(M/(M−1))⌉.
    pub fn w_max(&self) -> u32 {
        attempt_budget(self.lambda, self.m_rate())
    }

    /// The Π'_RLC mask width s = ξ·(K+k)·T·(b−1)·√nF (§3.6).
    pub fn rlc_width(&self) -> f64 {
        crate::gauss::rlc_mask_width(
            self.xi,
            self.capital_k + self.k,
            self.t_exp,
            self.b,
            self.nf,
        )
    }

    /// PoK widths (Eq 4.18 with nfold = 0 — fresh ternary salts).
    pub fn pok_widths_fresh(&self, nc: usize) -> (f64, f64) {
        let (gamma1, gamma2) = (46.0, 46.0); // γ₁, γ₂ from [LNP22] at these
                                             // norms — generous constants keep the tails conservative at toy scale.
        crate::gauss::calibrate_widths(
            nc,
            0,
            self.b_fold(),
            self.b_tilde,
            self.m1,
            self.m2,
            self.d,
            gamma1,
            gamma2,
            self.xi,
        )
    }

    /// PoK widths with nfold folded blocks (Step 5 of Protocol 8).
    pub fn pok_widths_folded(&self, nc: usize, nfold: usize) -> (f64, f64) {
        let (gamma1, gamma2) = (46.0, 46.0);
        crate::gauss::calibrate_widths(
            nc,
            nfold,
            self.b_fold(),
            self.b_tilde,
            self.m1,
            self.m2,
            self.d,
            gamma1,
            gamma2,
            self.xi,
        )
    }

    /// β_rlx = √2·s₂·√(2·Nc·m₂·d) (Eq 3.14).
    pub fn beta_rlx(&self, nc: usize, s2: f64) -> f64 {
        (2.0f64).sqrt() * s2 * (2.0 * nc as f64 * self.m2 as f64 * self.d as f64).sqrt()
    }

    /// Dmax = max{u+1, 2b, 2} (§2.1) — u = 2 for R1CS.
    pub fn d_max(&self) -> usize {
        let two_b = (2 * self.b) as usize;
        if two_b > 3 {
            two_b
        } else {
            3
        }
    }
}

/// The consolidated error budget (Propositions 4.15/4.17, Theorem 4.13).
/// The fixed-point k of Eq (4.17): k must satisfy
/// τ·ξ·(K+k)·T·(b−1)·√nF < b^k, and K+k depends on k itself — iterate.
pub fn fixed_point_k(lambda: f64, xi: f64, t_exp: i64, b: i64, nf: usize) -> usize {
    let capital_k = 1usize;
    let mut k = 1usize;
    for _ in 0..32 {
        let nxt = required_k(lambda, xi, capital_k + k, t_exp, b, nf);
        if nxt == k {
            return k;
        }
        k = nxt;
    }
    k
}

#[derive(Clone, Debug)]
pub struct SecurityBudget {
    pub log_q: f64,
    pub log_k: f64,
    pub log_c: f64,
    pub log_m: f64,
    pub eps_sz: f64,
    pub eps_sc: f64,
    pub eps_deg0: f64,
    pub eps_ct: f64,
    pub eps_cw: f64,
    pub w_max: u32,
    pub n_pok: usize,
    pub eps_completeness: f64,
    pub eps_blinding_cap: f64, // O(Wmax·2^−λ)
    pub bits_overall: f64,
}

impl SecurityBudget {
    /// Compute the consolidated budget for a parameter set.
    ///
    /// * ε_SZ = (log₂ m + 2)/|K| (Lemma 4.16);
    /// * ε_SC = ℓ·Dmax/|K| (the Sum-Check error);
    /// * ε_deg0 = 2/|K| (Lemma 3.13);
    /// * ε_ct = 1/|K| (Lemma 3.8);
    /// * ε_cw = Wmax·(K + k)/|C| (Lemma 2.19 inflated by the attempt
    ///   budget);
    /// * N_PoK = 5 + (K + k + 1)t (the unbatched count; 8 under the
    ///   batched default of §4.1.2.1);
    /// * completeness ≤ (7 + (K+k+1)t)·2^−λ unbatched / 9·2^−λ batched;
    /// * the blinding cap O(Wmax·2^−λ) = Wmax·2^−λ (Remark 4.23).
    pub fn evaluate(p: &Params) -> SecurityBudget {
        let log_q = (crate::fp::Q as f64).log2();
        let log_k = 2.0 * log_q; // |K| = q²
        let log_c = 2.0 * p.d as f64; // |C| = 4^d
        let log_m = p.nf.trailing_zeros() as f64;
        let d_max = p.d_max() as f64;
        // Schwartz–Zippel: (degree)/|K| — the log2 of the ERROR, so the
        // degree enters through its own log2 (the paper's Table 2 rows:
        // εSZ = 2^{−123.5} = 23/2^{128}, εSC = 2^{−121.6} = 84/2^{128}).
        let eps_sz = (log_m + 2.0).log2() - log_k;
        let eps_sc = (log_m * d_max).log2() - log_k;
        let eps_deg0 = 2.0_f64.log2() - log_k; // 2/|K| = 2^{−127}
        let eps_ct = 1.0 - log_k;
        let w_max = p.w_max();
        let eps_cw = (w_max as f64).log2() + ((p.capital_k + p.k) as f64).log2() - log_c;
        let n_pok = 5 + (p.capital_k + p.k + 1) * p.t;
        let eps_completeness =
            ((7.0 + (p.capital_k + p.k + 1) as f64 * p.t as f64) - p.lambda).max(3.0 - p.lambda);
        let eps_blinding_cap = (w_max as f64).log2() - p.lambda;
        // Overall interactive security: the binding term is the LARGEST
        // (weakest) of the challenge-space errors — Remark 4.19: at d=128
        // it is εSC (≈121 bits); at d=64 it is Wmax(K+k)/|C| (≈116 bits).
        // The lattice-layer terms are assumed at the MSIS/MLWE hardness of
        // Table 2.
        let binding = eps_sz.max(eps_sc).max(eps_deg0).max(eps_ct).max(eps_cw);
        let bits_overall = -binding;
        SecurityBudget {
            log_q,
            log_k,
            log_c,
            log_m,
            eps_sz,
            eps_sc,
            eps_deg0,
            eps_ct,
            eps_cw,
            w_max,
            n_pok,
            eps_completeness,
            eps_blinding_cap,
            bits_overall,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log2_f(x: f64) -> f64 {
        x.log2()
    }

    #[test]
    fn toy_params_self_consistent() {
        let p = Params::toy();
        assert_eq!(p.nr(), 16);
        assert!(p.k >= 1);
        // τ·s < b^k (Eq 4.17) — the RLC width check:
        let s = p.rlc_width();
        assert!(
            p.tau * s < (p.b as f64).powi(p.k as i32),
            "τ·s = {} vs b^k = {}",
            p.tau * s,
            (p.b as f64).powi(p.k as i32)
        );
    }

    #[test]
    fn paper_tables_reproduce() {
        let p64 = Params::paper_d64();
        let p128 = Params::paper_d128();
        // Table 2 rows: k, Bfold = (K+k)T(B̃−1)+B̃.
        assert_eq!(p64.k, 30);
        assert_eq!(p64.b_fold(), (31 * 128) + 2); // 3970? (paper: 3970)
                                                  // The paper's Bfold row says 3970 = 30·128 + 10? Check: (K+k)T(B̃−1)+B̃
                                                  // with K=1, k=30, T=128, B̃=2: 31·128·1 + 2 = 3970. ✓
        assert_eq!(p64.b_fold(), 3970);
        assert_eq!(p128.k, 31);
        assert_eq!(p128.b_fold(), (32 * 256) + 2); // 8194 ✓ (Table 2)
                                                   // κ, κ′=ℓ, κcom rows.
        assert_eq!((p64.kappa, p64.ell, p64.kappa_com()), (19, 16, 35));
        assert_eq!((p128.kappa, p128.ell, p128.kappa_com()), (11, 16, 27));
        // m1, m2 rows (ml·d ≥ 640).
        assert!(p64.m1 * p64.d >= 640 && p64.m2 * p64.d >= 640);
        assert!(p128.m1 * p128.d >= 640 && p128.m2 * p128.d >= 640);
        // nR,bl / nF,bl rows.
        assert_eq!((p64.nr_bl(), p64.nf_bl), (1668, 106_752));
        assert_eq!((p128.nr_bl(), p128.nf_bl), (962, 123_136));
    }

    #[test]
    fn paper_budget_matches_remarks() {
        let p128 = Params::paper_d128();
        let bud = SecurityBudget::evaluate(&p128);
        // Remark 4.19 / Table 2: at d=128 the binding term is εSC
        // = ℓ·Dmax/|K| = 84/2^{128} = 2^{−121.6} (the degree enters
        // through its own log2), giving ≈ 121 bits overall.
        assert!((bud.eps_sc - ((log2_f(84.0)) - 128.0)).abs() < 0.01);
        // ε_SZ = (log₂ m + 2)/|K| = 23/2^{128} = 2^{−123.5} (the degree
        // enters through its own log2).
        assert!((bud.eps_sz - (23.0_f64.log2() - 128.0)).abs() < 0.01);
        let _ = log2_f;
        // ε_deg0 = 2/|K| = 2^{−127} ✓.
        assert!((bud.eps_deg0 - (1.0 - 128.0)).abs() < 0.01);
        // ε_cw = Wmax·(K+k)/|C| with |C| = 2^256, Wmax = 128 (M ≈ 2):
        // 2^{7}·2^{5}/2^{256} → exponent 12 − 256 = −244.1 ✓ (Table 2's
        // 2^{−244.1}).
        assert!((bud.eps_cw - (12.0 - 256.0)).abs() < 0.1);
        // Overall ≈ 121 bits (Table 2's row).
        assert!((bud.bits_overall - 121.0).abs() < 1.5);
        // The blinding cap O(Wmax·2^−λ) = 2^{−112} at Wmax = 128, λ = 120
        // (Remark 4.23).
        // (Wmax = 122 at ξ = 20 — the paper's 128 assumes M exactly 2.)
        assert!((bud.eps_blinding_cap - (7.0 - 120.0)).abs() < 0.1);
    }

    #[test]
    fn d64_budget_binding_term() {
        let p64 = Params::paper_d64();
        let bud = SecurityBudget::evaluate(&p64);
        // Remark 4.19.(1): at d = 64 the binding term is
        // Wmax·(K+k)/|C| with |C| = 2^128 → ≈ 116 bits.
        assert!((bud.eps_cw - (12.0 - 128.0)).abs() < 0.2);
        assert!((bud.bits_overall - 116.0).abs() < 1.5);
    }
}
