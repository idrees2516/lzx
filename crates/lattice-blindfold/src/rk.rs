//! The extension ring R_K = K[X]/(X^d + 1) and the **rank-doubling
//! embedding** of §3.3.
//!
//! Every R_K element is stored in the {1, Y}-coordinates: `PolyK { a, b }`
//! with a, b ∈ R_F. Multiplication follows Eq (3.5):
//! `(Ra + RbY)(a + bY) = (Ra·a + ν·Rb·b) + (Ra·b + Rb·a)·Y`.
//!
//! * The **regular representation** ψ (Eq 3.2): a + bY ↦ [[a, νb],[b, a]] —
//!   R_K-linear maps become 2×2 block matrices over R_F, so every R_K
//!   relation translates to a pair of R_F relations once, at parameter
//!   generation (§3.3.2).
//! * The **quadratic rank-doubling** matrices R̂^(a), R̂^(b) of Eq (3.9)
//!   for the componentwise expansion of x^T R2 x.
//! * The **norm convention** of §3.3.4: ∥x∥² = ∥a∥² + ∥b∥² (φ is an exact
//!   isometry), proved as two independent R_F statements — never through
//!   the K-automorphism identity (which would prove ∥a∥² + ν∥b∥²).
//! * τ_ℓ rotations and packaged rotations with **K-valued weights**
//!   (Lemma 3.12 over R_K): Σ_ℓ w_ℓ·cf(x)_ℓ = ct(ρ·x) with ρ = Σw_ℓτ_ℓ.
//! * Degree-0 elements (Lemma 2.20): products and differences of
//!   degree-0 elements stay degree-0 and ct multiplies coordinate-wise.

use crate::fp::{Fq, NU};
use crate::fq2::K;
use crate::ring::Poly;

/// An element of R_K = K[X]/(X^d+1) in the {1, Y} basis.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PolyK {
    pub a: Poly,
    pub b: Poly,
}

impl PolyK {
    pub fn zero(d: usize) -> PolyK {
        PolyK {
            a: Poly::zero(d),
            b: Poly::zero(d),
        }
    }

    pub fn one(d: usize) -> PolyK {
        PolyK {
            a: Poly::one(d),
            b: Poly::zero(d),
        }
    }

    pub fn from_poly(a: Poly) -> PolyK {
        PolyK {
            b: Poly::zero(a.d()),
            a,
        }
    }

    /// From K-coefficients: the R_K element whose coefficient vector is
    /// `coeffs` (each a K-element).
    pub fn from_coeffs(coeffs: &[K]) -> PolyK {
        let d = coeffs.len();
        let mut a = vec![Fq::ZERO; d];
        let mut b = vec![Fq::ZERO; d];
        for (i, c) in coeffs.iter().enumerate() {
            a[i] = c.0;
            b[i] = c.1;
        }
        PolyK {
            a: Poly(a),
            b: Poly(b),
        }
    }

    /// The K-coefficient vector (the inverse of from_coeffs).
    pub fn coeffs(&self) -> Vec<K> {
        self.a
            .0
            .iter()
            .zip(self.b.0.iter())
            .map(|(x, y)| K(*x, *y))
            .collect()
    }

    pub fn d(&self) -> usize {
        self.a.d()
    }

    pub fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }

    pub fn add(&self, o: &PolyK) -> PolyK {
        PolyK {
            a: self.a.add(&o.a),
            b: self.b.add(&o.b),
        }
    }

    pub fn sub(&self, o: &PolyK) -> PolyK {
        PolyK {
            a: self.a.sub(&o.a),
            b: self.b.sub(&o.b),
        }
    }

    pub fn neg(&self) -> PolyK {
        PolyK {
            a: self.a.neg(),
            b: self.b.neg(),
        }
    }

    pub fn add_assign(&mut self, o: &PolyK) {
        self.a.add_assign(&o.a);
        self.b.add_assign(&o.b);
    }

    pub fn sub_assign(&mut self, o: &PolyK) {
        self.a.sub_assign(&o.a);
        self.b.sub_assign(&o.b);
    }

    /// Scaling by a base-field scalar.
    pub fn scale_fp(&self, s: &Fq) -> PolyK {
        PolyK {
            a: self.a.scale(s),
            b: self.b.scale(s),
        }
    }

    /// Scaling by a K-scalar λ = λa + λbY:
    /// λ·x = (λa·a + ν·λb·b) + (λa·b + λb·a)·Y.
    pub fn scale_k(&self, lam: &K) -> PolyK {
        let nu = Fq::new(NU);
        PolyK {
            a: self
                .a
                .scale(&lam.0)
                .add(&self.b.scale(&nu.mul(&lam.1))),
            b: self.b.scale(&lam.0).add(&self.a.scale(&lam.1)),
        }
    }

    /// Ring multiplication in R_K (Eq 3.5's structure with full operands):
    /// (a1 + b1Y)(a2 + b2Y) = (a1a2 + ν b1b2) + (a1b2 + b1a2) Y —
    /// computed here as full ring products of the R_F parts.
    pub fn mul(&self, o: &PolyK) -> PolyK {
        let nu = Fq::new(NU);
        let a = self
            .a
            .mul(&o.a)
            .add(&self.b.mul(&o.b).scale(&nu));
        let b = self.a.mul(&o.b).add(&self.b.mul(&o.a));
        PolyK { a, b }
    }

    pub fn square(&self) -> PolyK {
        self.mul(self)
    }

    /// The constant coefficient ct(x) ∈ K (§2.1, over R_K).
    pub fn ct(&self) -> K {
        K(self.a.ct(), self.b.ct())
    }

    /// The ℓ-th coefficient cf(x)_ℓ ∈ K (1-indexed).
    pub fn cf(&self, ell: usize) -> K {
        K(self.a.cf(ell), self.b.cf(ell))
    }

    /// Degree-0 test: cf(x)_ℓ = 0 for all ℓ ≥ 2 (only the constant
    /// K-coefficient survives).
    pub fn is_degree0(&self) -> bool {
        self.a.0.iter().skip(1).all(|c| c.is_zero())
            && self.b.0.iter().skip(1).all(|c| c.is_zero())
    }

    /// Make the degree-0 element with constant coefficient v.
    pub fn degree0(v: K, d: usize) -> PolyK {
        let mut a = vec![Fq::ZERO; d];
        let mut b = vec![Fq::ZERO; d];
        a[0] = v.0;
        b[0] = v.1;
        PolyK {
            a: Poly(a),
            b: Poly(b),
        }
    }

    /// τ_ℓ ∈ R_K (Lemma 3.12 over R_K — same monomials as over R_F).
    pub fn tau(d: usize, ell: usize) -> PolyK {
        PolyK::from_poly(Poly::tau(d, ell))
    }

    /// The packaged rotation ρ = Σ_ℓ w_ℓ τ_ℓ with K-valued weights
    /// (Lemma 3.12 + §3.2.1.2).
    pub fn packaged_rotation(weights: &[K]) -> PolyK {
        let d = weights.len();
        let mut acc = PolyK::zero(d);
        for (idx, w) in weights.iter().enumerate() {
            if w.is_zero() {
                continue;
            }
            let t = PolyK::tau(d, idx + 1);
            acc.add_assign(&t.scale_k(w));
        }
        acc
    }

    /// ct(ρ · x) — the weighted coefficient sum Σ w_ℓ cf(x)_ℓ.
    pub fn rotated_ct(rho: &PolyK, x: &PolyK) -> K {
        rho.mul(x).ct()
    }

    /// ∥x∥₂² = ∥a∥₂² + ∥b∥₂² (§3.3.4 — the isometric norm convention).
    pub fn norm_2_sq(&self) -> u128 {
        self.a.norm_2_sq() + self.b.norm_2_sq()
    }

    /// The φ embedding: x ↦ (a; b) ∈ R_F^{2} — slot extraction.
    pub fn phi(&self) -> (Poly, Poly) {
        (self.a.clone(), self.b.clone())
    }

    /// Uniform R_K element.
    pub fn uniform(d: usize, seed: &[u8], counter: &mut u64) -> PolyK {
        PolyK {
            a: Poly::uniform(d, seed, counter),
            b: Poly::uniform(d, seed, counter),
        }
    }

    /// Uniform garbage on {x ∈ R_K : ct(x) = 0} (Protocol 4, Step 1):
    /// uniform K-coefficients at slots ≥ 2, zero constant.
    pub fn garbage_ct_zero(d: usize, seed: &[u8], counter: &mut u64) -> PolyK {
        let mut a = vec![Fq::ZERO; d];
        let mut b = vec![Fq::ZERO; d];
        for i in 1..d {
            a[i] = Fq::uniform(seed, counter);
            b[i] = Fq::uniform(seed, counter);
        }
        PolyK {
            a: Poly(a),
            b: Poly(b),
        }
    }

    /// The automorphism σ acting on X only, fixing Y (§3.3.3.4):
    /// σ(a + bY) = σ(a) + σ(b)Y — component-wise.
    pub fn sigma_inv(&self) -> PolyK {
        PolyK {
            a: self.a.sigma_inv(),
            b: self.b.sigma_inv(),
        }
    }
}

/// ψ(a + bY) = [[a, νb], [b, a]] — the regular representation block
/// (Eq 3.2). `psi_mul` computes ψ(m)·(u; v) for the module action.
pub fn psi_mul(m: &PolyK, u: &Poly, v: &Poly) -> (Poly, Poly) {
    let nu = Fq::new(NU);
    // [[a, νb],[b, a]] · (u; v) = (a·u + νb·v; b·u + a·v)
    let top = m.a.mul(u).add(&m.b.mul(v).scale(&nu));
    let bot = m.b.mul(u).add(&m.a.mul(v));
    (top, bot)
}

/// The componentwise expansion of a quadratic form (§3.3.3):
/// for x = a + bY stacked as s = (a; b), the two R_F-quadratic matrices
/// R̂^(a), R̂^(b) of Eq (3.9) so that
/// s^T R̂^(a) s = (x^T R2 x)_a and s^T R̂^(b) s = (x^T R2 x)_b.
///
/// Here R2 is given componentwise as (Ra, Rb) with R2 = Ra + Rb·Y.
pub fn quad_expand_matrices(ra: &Poly, rb: &Poly) -> (QuadRF, QuadRF) {
    let nu = Fq::new(NU);
    // R̂^(a) = [[Ra, νRb], [νRb, νRa]]  (from Eq (3.7):
    // a^T Ra a + ν b^T Ra b + ν a^T Rb b + ν b^T Rb a — we represent the
    // quadratic form by its four blocks (R11, R12, R21, R22) acting on
    // s = (a; b): s^T R̂ s = aR11a + aR12b + bR21a + bR22b).
    //
    // Matching Eq (3.7): a^T Ra a + ν(a^T Rb b + b^T Rb a) + ν b^T Ra b:
    let hat_a = QuadRF {
        r11: ra.clone(),
        r12: rb.scale(&nu),
        r21: rb.scale(&nu),
        r22: ra.scale(&nu),
    };
    // R̂^(b) from Eq (3.8): b^T Ra a + a^T Rb a + a^T Ra b + ν b^T Rb b.
    let hat_b = QuadRF {
        r11: rb.clone(),
        r12: ra.clone(),
        r21: ra.clone(),
        r22: rb.scale(&nu),
    };
    (hat_a, hat_b)
}

/// A quadratic form over R_F^n in block form: s^T R̂ s with s = (u; v)
/// and R̂ = [[R11, R12], [R21, R22]] (each block a ring element acting as
/// a diagonal/quadratic coefficient — the concrete relations of Protocol 6
/// are single-slot products, so blocks are single ring elements and the
/// "matrix" acts on pairs of slots).
#[derive(Clone, Debug)]
pub struct QuadRF {
    pub r11: Poly,
    pub r12: Poly,
    pub r21: Poly,
    pub r22: Poly,
}

impl QuadRF {
    /// Evaluate s^T R̂ s for s = (u; v) ∈ R_F² (single-slot form):
    /// u·R11·u + u·R12·v + v·R21·u + v·R22·v.
    pub fn eval(&self, u: &Poly, v: &Poly) -> Poly {
        let mut acc = self.r11.mul(u).mul(u);
        acc.add_assign(&self.r12.mul(u).mul(v));
        acc.add_assign(&self.r21.mul(v).mul(u));
        acc.add_assign(&self.r22.mul(v).mul(v));
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rk_multiplication_matches_k_coefficient_wise() {
        // Multiply two degree-1 R_K elements by hand and via the ring.
        let d = 4;
        let x = PolyK::from_coeffs(&[
            K::from_fp2(Fq::new(3), Fq::new(5)),
            K::ZERO,
            K::from_fp2(Fq::new(2), Fq::new(7)),
            K::ONE,
        ]);
        let y = PolyK::from_coeffs(&[
            K::from_fp2(Fq::new(11), Fq::new(13)),
            K::from_fp2(Fq::new(1), Fq::new(1)),
            K::ZERO,
            K::from_fp2(Fq::new(4), Fq::new(9)),
        ]);
        let z = x.mul(&y);
        // Cross-check coefficient 0 by hand: c0 = x0·y0 + (negacyclic wraps)
        // X^d ≡ −1: contributions to slot 0: (i,j) with i+j=0 → x0y0;
        // i+j=d → −x_d-1... (i=1,j=d-1), (i=2,j=d-2), (i=3,j=d-3), (i=d-1,j=1)
        let xc = x.coeffs();
        let yc = y.coeffs();
        let mut expect = xc[0].mul(&yc[0]);
        for i in 1..d {
            expect = expect.sub(&xc[i].mul(&yc[d - i]));
        }
        assert_eq!(z.cf(1), expect);
    }

    #[test]
    fn tau_rotations_over_rk() {
        let d = 8;
        let mut ctr = 0u64;
        let x = PolyK::uniform(d, b"tauk", &mut ctr);
        for ell in 1..=d {
            let t = PolyK::tau(d, ell);
            assert_eq!(
                PolyK::rotated_ct(&t, &x),
                x.cf(ell),
                "ct(tau_{ell}·x) = cf(x)_{ell} over R_K"
            );
        }
    }

    #[test]
    fn packaged_rotation_k_weights() {
        let d = 8;
        let mut ctr = 0u64;
        let x = PolyK::uniform(d, b"prk", &mut ctr);
        let weights: Vec<K> = (0..d).map(|_| K::uniform(b"prk-w", &mut ctr)).collect();
        let rho = PolyK::packaged_rotation(&weights);
        let rotated = PolyK::rotated_ct(&rho, &x);
        let manual: K = weights
            .iter()
            .enumerate()
            .map(|(i, w)| w.mul(&x.cf(i + 1)))
            .fold(K::ZERO, |acc, v| acc.add(&v));
        assert_eq!(rotated, manual);
    }

    #[test]
    fn degree0_products_scalar() {
        // Lemma 2.20: degree-0 elements multiply as scalars.
        let d = 8;
        let u = K::from_fp2(Fq::new(7), Fq::new(3));
        let v = K::from_fp2(Fq::new(5), Fq::new(11));
        let x = PolyK::degree0(u, d);
        let y = PolyK::degree0(v, d);
        let z = x.mul(&y);
        assert!(z.is_degree0());
        assert_eq!(z.ct(), u.mul(&v));
    }

    #[test]
    fn psi_regular_representation() {
        // ψ(A)·φ(z) = φ(Az) (§3.3.1's compatibility).
        let d = 4;
        let mut ctr = 0u64;
        let m = PolyK::uniform(d, b"psim", &mut ctr);
        let z = PolyK::uniform(d, b"psiz", &mut ctr);
        let (zu, zv) = z.phi();
        let (top, bot) = psi_mul(&m, &zu, &zv);
        let prod = m.mul(&z);
        assert_eq!((top, bot), prod.phi());
    }

    #[test]
    fn garbage_has_zero_constant() {
        let d = 16;
        let mut ctr = 0u64;
        for _ in 0..8 {
            let g = PolyK::garbage_ct_zero(d, b"gb", &mut ctr);
            assert!(g.ct().is_zero());
            assert!(!g.is_zero());
        }
    }

    #[test]
    fn norm_convention_isometric() {
        let d = 8;
        let mut ctr = 0u64;
        let x = PolyK {
            a: Poly::small_b(d, 2, b"nc-a", &mut ctr),
            b: Poly::small_b(d, 2, b"nc-b", &mut ctr),
        };
        assert_eq!(x.norm_2_sq(), x.a.norm_2_sq() + x.b.norm_2_sq());
    }

    #[test]
    fn quadratic_rank_doubling() {
        // Verify Eq (3.7)/(3.8): the componentwise quadratic expansion.
        let d = 4;
        let mut ctr = 0u64;
        let ra = Poly::small_b(d, 2, b"qra", &mut ctr);
        let rb = Poly::small_b(d, 2, b"qrb", &mut ctr);
        let a = Poly::small_b(d, 2, b"qa", &mut ctr);
        let b = Poly::small_b(d, 2, b"qb", &mut ctr);
        let (hat_a, hat_b) = quad_expand_matrices(&ra, &rb);
        let sa = hat_a.eval(&a, &b);
        let sb = hat_b.eval(&a, &b);
        // Direct: x = a + bY; x²  = ... take x^T R2 x with R2 = Ra + RbY,
        // x = s (single vector): x·R2·x = x·(Ra + RbY)·x.
        let x = PolyK {
            a: a.clone(),
            b: b.clone(),
        };
        let r2 = PolyK {
            a: ra.clone(),
            b: rb.clone(),
        };
        // x^T R2 x := x · R2 · x (ring products, R2 central in Y):
        let prod = x.mul(&r2).mul(&x);
        assert_eq!((sa, sb), prod.phi(), "componentwise quadratic expansion");
    }
}
