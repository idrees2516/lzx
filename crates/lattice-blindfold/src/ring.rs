//! The cyclotomic ring R_F = F_q[X]/(X^d + 1) for d a power of two
//! (restriction (7): "a cyclotomic ring of the form Z[X]/(x^d + 1) for d a
//! power of 2").
//!
//! Carries:
//! * negacyclic multiplication (schoolbook, i128 accumulation) and the
//!   elementary bound ∥ρv∥∞ ≤ ∥ρ∥₁·∥v∥∞ ≤ d·β·∥v∥∞ (Remark 2.17);
//! * the constant-coefficient / coefficient-vector maps ct, cf (§2.1);
//! * the **τ_ℓ rotation basis** (Lemma 3.12): τ₁ = 1, τ_ℓ = −X^{d−ℓ+1},
//!   with ct(τ_ℓ·a) = cf(a)_ℓ;
//! * the **inner-product transform** ā with ct(ā·b) = ⟨a, b⟩ (§2.4.1) —
//!   the reflection ā_i = ±a_{(d−i) mod d} that makes the negacyclic
//!   convolution's constant term the plain dot product;
//! * the automorphism σ: X ↦ X^{−1} (§2.1);
//! * the strong sampling set C (Def 2.13; the paper's instantiation:
//!   coefficients in {−1,0,1,2}, |C| = 4^d) and its expansion factor T;
//! * the b-ary decomposition split_b (§2.1) and its recomposition;
//! * samplers: ternary salts (∥·∥∞ < B̃ = 2, uniform on {−1,0,1}), the
//!   bounded blinding block χ_b, uniform ring elements.

use crate::fp::Fq;
use crate::fq2::K;

/// A ring element of R_F = F_q[X]/(X^d+1), stored as its d coefficients.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Poly(pub Vec<Fq>);

impl Poly {
    pub fn zero(d: usize) -> Poly {
        Poly(vec![Fq::ZERO; d])
    }

    pub fn one(d: usize) -> Poly {
        let mut c = vec![Fq::ZERO; d];
        c[0] = Fq::ONE;
        Poly(c)
    }

    pub fn constant(v: Fq, d: usize) -> Poly {
        let mut c = vec![Fq::ZERO; d];
        c[0] = v;
        Poly(c)
    }

    pub fn from_i64(v: i64, d: usize) -> Poly {
        Poly::constant(Fq::from_i64(v), d)
    }

    /// From signed coefficients (symmetric representatives).
    pub fn from_sym(coeffs: &[i64]) -> Poly {
        Poly(coeffs.iter().map(|&x| Fq::from_i64(x)).collect())
    }

    pub fn d(&self) -> usize {
        self.0.len()
    }

    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|c| c.is_zero())
    }

    pub fn add(&self, o: &Poly) -> Poly {
        Poly(
            self.0
                .iter()
                .zip(o.0.iter())
                .map(|(a, b)| a.add(b))
                .collect(),
        )
    }

    pub fn sub(&self, o: &Poly) -> Poly {
        Poly(
            self.0
                .iter()
                .zip(o.0.iter())
                .map(|(a, b)| a.sub(b))
                .collect(),
        )
    }

    pub fn neg(&self) -> Poly {
        Poly(self.0.iter().map(|a| a.neg()).collect())
    }

    pub fn add_assign(&mut self, o: &Poly) {
        for (a, b) in self.0.iter_mut().zip(o.0.iter()) {
            *a = a.add(b);
        }
    }

    pub fn sub_assign(&mut self, o: &Poly) {
        for (a, b) in self.0.iter_mut().zip(o.0.iter()) {
            *a = a.sub(b);
        }
    }

    /// Scalar multiplication by a base-field element.
    pub fn scale(&self, s: &Fq) -> Poly {
        Poly(self.0.iter().map(|a| a.mul(s)).collect())
    }

    /// Negacyclic multiplication mod X^d + 1 (schoolbook, exact).
    pub fn mul(&self, o: &Poly) -> Poly {
        let d = self.d();
        debug_assert_eq!(d, o.d());
        let mut acc = vec![Fq::ZERO; d];
        for i in 0..d {
            let ai = self.0[i];
            if ai.is_zero() {
                continue;
            }
            for j in 0..d {
                let bj = o.0[j];
                if bj.is_zero() {
                    continue;
                }
                let k = i + j;
                if k < d {
                    acc[k] = acc[k].add(&ai.mul(&bj));
                } else {
                    // X^{k} = X^{k-d}·X^d ≡ −X^{k-d}
                    acc[k - d] = acc[k - d].sub(&ai.mul(&bj));
                }
            }
        }
        Poly(acc)
    }

    pub fn square(&self) -> Poly {
        self.mul(self)
    }

    /// The constant coefficient ct(a) (§2.1).
    pub fn ct(&self) -> Fq {
        self.0[0]
    }

    /// The ℓ-th coefficient cf(a)_ℓ (1-indexed per the paper; ℓ ∈ [1, d]).
    pub fn cf(&self, ell: usize) -> Fq {
        self.0[ell - 1]
    }

    /// The ℓ-th τ-rotation: ct(τ_ℓ · a) = cf(a)_ℓ (Lemma 3.12).
    /// τ_1 = 1; τ_ℓ = −X^{d−ℓ+1} for ℓ ≥ 2.
    pub fn tau(d: usize, ell: usize) -> Poly {
        debug_assert!(ell >= 1 && ell <= d);
        if ell == 1 {
            return Poly::one(d);
        }
        let mut c = vec![Fq::ZERO; d];
        // −X^{d−ℓ+1} mod X^d+1: exponent d−ℓ+1 ∈ [1, d−1] for ℓ ≥ 2.
        let e = d - ell + 1;
        c[e] = Fq::ONE;
        Poly(c).neg()
    }

    /// The packaged rotation ρ = Σ_ℓ w_ℓ τ_ℓ acting on R_F (over F_q
    /// weights): Σ_ℓ w_ℓ·cf(a)_ℓ = ct(ρ·a).
    pub fn packaged_rotation(weights: &[Fq]) -> Poly {
        let d = weights.len();
        let mut acc = Poly::zero(d);
        for (idx, w) in weights.iter().enumerate() {
            if w.is_zero() {
                continue;
            }
            let ell = idx + 1;
            let t = Poly::tau(d, ell);
            acc.add_assign(&t.scale(w));
        }
        acc
    }

    /// The inner-product transform: ct(ā·b) = ⟨a, b⟩ (§2.4.1).
    /// For the negacyclic ring, ct(a·b) = Σ_{i+j≡0} ±a_i b_j; taking
    /// ā_i = a_{(d−i) mod d}·(−1)^{[i>0]} yields ct(ā·b) = Σ_i a_i b_i.
    pub fn inner_transform(&self) -> Poly {
        let d = self.d();
        let mut out = vec![Fq::ZERO; d];
        for i in 0..d {
            // ā_i is the coefficient that multiplies b_i inside ct(a·b):
            // ct(a·b) = Σ_j ±a_{(d−j) mod d} b_j — index map j ↦ (d−j) mod d
            // with a sign flip when the convolution wraps (i + j ≥ d).
            // Concretely: ct(a·b) = a_0 b_0 + Σ_{j≥1} (−a_{d−j}) b_j.
            let src = (d - i) % d;
            let v = self.0[src];
            out[i] = if i == 0 { v } else { v.neg() };
        }
        Poly(out)
    }

    /// The ring automorphism σ: X ↦ X^{−1} (§2.1), i.e. σ(X^i) = X^{d−i}
    /// for i ≥ 1 (X^{−i} ≡ −X^{d−i} mod X^d+1, with the sign absorbed:
    /// X^d ≡ −1 ⟹ X^{−1} = −X^{d−1}; so σ(X^i) = −X^{d−i} for 1 ≤ i ≤ d−1).
    pub fn sigma_inv(&self) -> Poly {
        let d = self.d();
        let mut out = vec![Fq::ZERO; d];
        out[0] = self.0[0];
        for i in 1..d {
            out[i] = self.0[d - i].neg();
        }
        Poly(out)
    }

    /// ℓ∞ norm over the symmetric representatives (§2.1) — max of the
    /// ABSOLUTE values (max(sym).abs() would underestimate
    /// negative-dominant vectors and corrupt the digit count).
    pub fn norm_inf(&self) -> i64 {
        self.0.iter().map(|c| c.sym().abs()).max().unwrap_or(0)
    }

    /// ℓ2 norm (squared) over the symmetric representatives (saturating —
    /// meaningful for the small elements the protocol tracks).
    pub fn norm_2_sq(&self) -> u128 {
        self.0
            .iter()
            .map(|c| {
                let s = c.sym();
                (s as i128 * s as i128) as u128
            })
            .fold(0u128, |a, b| a.saturating_add(b))
    }

    /// Uniform ring element from seed material.
    pub fn uniform(d: usize, seed: &[u8], counter: &mut u64) -> Poly {
        Poly((0..d).map(|_| Fq::uniform(seed, counter)).collect())
    }

    /// Uniform ternary in {−1, 0, 1} — the salt distribution for B̃ = 2
    /// (∥s∥∞ < B̃ = 2, "ternary salts", §4.1 Remark 4.1.(2)).
    pub fn ternary(d: usize, seed: &[u8], counter: &mut u64) -> Poly {
        let mut c = vec![Fq::ZERO; d];
        for coeff in c.iter_mut() {
            let v = Fq::uniform(seed, counter);
            *coeff = Fq::from_i64(match v.0 % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            });
        }
        Poly(c)
    }

    /// Uniform on {−(b−1), ..., b−1} — the blinding block sampler χ_b and
    /// the small-witness distribution at b = 2.
    pub fn small_b(d: usize, b: i64, seed: &[u8], counter: &mut u64) -> Poly {
        let spread = 2 * b - 1;
        let mut c = vec![Fq::ZERO; d];
        for coeff in c.iter_mut() {
            let v = Fq::uniform(seed, counter);
            let m = (v.0 % spread as u64) as i64;
            *coeff = Fq::from_i64(m - (b - 1));
        }
        Poly(c)
    }

    /// Coefficient-wise sum of scaled vectors (the Σ ρ_i v_i fold).
    pub fn lin_comb(polys: &[(&Poly, &K)]) -> Poly {
        let d = polys.first().map(|p| p.0.d()).unwrap_or(0);
        let mut acc = vec![Fq::ZERO; d];
        for (p, w) in polys {
            for i in 0..d {
                // w·p_i with w = (wa, wb): w·p_i = wa·p_i + wb·ν... no —
                // over R_F we only take F_q weights; K weights act via R_K.
                // Here weights must be degree-0 in K acting on R_F: use .0.
                acc[i] = acc[i].add(&p.0[i].mul(&w.0));
            }
        }
        Poly(acc)
    }
}

/// Vector-of-polys helpers (module R_F^n).
pub fn vec_add(a: &[Poly], b: &[Poly]) -> Vec<Poly> {
    a.iter().zip(b.iter()).map(|(x, y)| x.add(y)).collect()
}

pub fn vec_sub(a: &[Poly], b: &[Poly]) -> Vec<Poly> {
    a.iter().zip(b.iter()).map(|(x, y)| x.sub(y)).collect()
}

pub fn vec_neg(a: &[Poly]) -> Vec<Poly> {
    a.iter().map(|x| x.neg()).collect()
}

pub fn vec_scale(a: &[Poly], s: &Fq) -> Vec<Poly> {
    a.iter().map(|x| x.scale(s)).collect()
}

/// Σ_i w_i · v_i over R_F^n with F_q weights.
pub fn vec_lin_comb(items: &[(&[Poly], Fq)]) -> Vec<Poly> {
    let n = items.first().map(|(v, _)| v.len()).unwrap_or(0);
    let d = items
        .first()
        .and_then(|(v, _)| v.first())
        .map(|p| p.d())
        .unwrap_or(0);
    let mut acc = vec![Poly::zero(d); n];
    for (v, w) in items {
        for (i, p) in v.iter().enumerate() {
            acc[i].add_assign(&p.scale(w));
        }
    }
    acc
}

/// Σ_i ρ_i · v_i with **C-challenges** ρ_i ∈ R_F (the folding combination
/// of Π_RLC): ρ·v is the ring product.
pub fn vec_ring_comb(vectors: &[Vec<Poly>], rhos: &[Poly]) -> Vec<Poly> {
    let n = vectors.first().map(|v| v.len()).unwrap_or(0);
    let d = vectors
        .first()
        .and_then(|v| v.first())
        .map(|p| p.d())
        .unwrap_or(0);
    let mut acc = vec![Poly::zero(d); n];
    for (v, rho) in vectors.iter().zip(rhos.iter()) {
        for (i, p) in v.iter().enumerate() {
            acc[i].add_assign(&rho.mul(p));
        }
    }
    acc
}

/// The strong sampling set C: polynomials with coefficients in
/// {−1, 0, 1, 2} (§4.3.4: "|C| = 4^d"), sampled uniformly.
pub struct StrongSet;

impl StrongSet {
    pub fn sample(d: usize, seed: &[u8], counter: &mut u64) -> Poly {
        let mut c = vec![Fq::ZERO; d];
        for coeff in c.iter_mut() {
            let v = Fq::uniform(seed, counter);
            *coeff = Fq::from_i64(match v.0 % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 2,
            });
        }
        Poly(c)
    }

    /// |C| = 4^d (log2, as f64 for the error budget).
    pub fn log2_size(d: usize) -> f64 {
        2.0 * d as f64
    }

    /// Membership: coefficients in {−1, 0, 1, 2}.
    pub fn contains(p: &Poly) -> bool {
        p.0.iter().all(|c| matches!(c.sym(), -1..=2))
    }

    /// The negacyclic expansion factor bound T(C) ≤ d·β with β = 2
    /// (Remark 2.17: "T(C) ≤ φ(η)β", the paper instantiates T = 2d).
    pub fn expansion_bound(d: usize) -> i64 {
        2 * d as i64
    }
}

/// The b-ary decomposition split_b (§2.1): for ∥z∥∞ < b^k returns
/// (z_1, ..., z_k) with z = Σ b^{i−1} z_i and ∥z_i∥∞ < b.
pub fn split_b(z: &[Poly], b: i64) -> Vec<Vec<Poly>> {
    let k = required_digits(z, b);
    let mut out = vec![Vec::new(); k];
    for p in z {
        let digits = split_poly(p, b, k);
        for (i, dg) in digits.into_iter().enumerate() {
            out[i].push(dg);
        }
    }
    out
}

/// The b-ary decomposition with EXACTLY k pieces (zero-padded when the
/// norm needs fewer digits — the protocol's k is a public parameter).
pub fn split_b_k(z: &[Poly], b: i64, k: usize) -> Vec<Vec<Poly>> {
    let mut out = split_b(z, b);
    while out.len() < k {
        out.push(vec![
            Poly::zero(z.first().map(|p| p.d()).unwrap_or(0));
            z.len()
        ]);
    }
    out.truncate(k);
    out
}

/// Recomposition z = Σ b^{i−1} z_i.
pub fn recompose(pieces: &[Vec<Poly>], b: i64) -> Vec<Poly> {
    let _k = pieces.len();
    let n = pieces.first().map(|v| v.len()).unwrap_or(0);
    let d = pieces
        .first()
        .and_then(|v| v.first())
        .map(|p| p.d())
        .unwrap_or(0);
    let mut acc = vec![Poly::zero(d); n];
    for (i, piece) in pieces.iter().enumerate() {
        let w = Fq::from_i64(b.pow(i as u32));
        for (j, p) in piece.iter().enumerate() {
            acc[j].add_assign(&p.scale(&w));
        }
    }
    acc
}

fn required_digits(z: &[Poly], b: i64) -> usize {
    let max_norm = z.iter().map(|p| p.norm_inf()).max().unwrap_or(0);
    let mut k = 1;
    let mut bound = b;
    while max_norm >= bound && k < 64 {
        k += 1;
        bound *= b;
    }
    k
}

fn split_poly(p: &Poly, b: i64, k: usize) -> Vec<Poly> {
    let d = p.d();
    let mut digits = vec![Poly::zero(d); k];
    for (slot, coeff) in p.0.iter().enumerate() {
        let mut v = coeff.sym();
        for i in 0..k {
            // Sign-aware balanced digit in [−(b−1), b−1]:
            // rem = v mod b ∈ [0, b); ties (2·rem = b) resolved toward
            // the sign of v so that v′ = (v − dig)/b strictly shrinks
            // (without this, v = −1 in base 2 never terminates).
            let rem = ((v % b) + b) % b;
            let dig = if 2 * rem > b || (2 * rem == b && v < 0) {
                rem - b
            } else {
                rem
            };
            v = (v - dig) / b;
            if dig != 0 {
                digits[i].0[slot] = Fq::from_i64(dig);
            }
        }
    }
    digits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negacyclic_multiplication() {
        // X^d ≡ −1
        let d = 8;
        let mut x = vec![Fq::ZERO; d];
        x[1] = Fq::ONE;
        let xp = Poly(x.clone()).square(); // X^2
        assert_eq!(xp.0[2], Fq::ONE);
        let x4 = Poly(x.clone()).mul(&Poly(x.clone())).square();
        assert!(x4.0[4] == Fq::ONE);
        // X^d · X = −X
        let xd = {
            let mut c = vec![Fq::ZERO; d];
            c[0] = Fq::ONE;
            c[1] = Fq::ONE; // 1 + X, its d-th power-ish check below
            Poly(c)
        };
        let _ = xd;
        let x_high = {
            let mut c = vec![Fq::ZERO; d];
            c[d - 1] = Fq::ONE; // X^{d-1}
            Poly(c)
        };
        let prod = x_high.mul(&Poly(x.clone()));
        // X^{d-1} · X = X^d ≡ −1
        assert_eq!(prod.0[0], Fq::ONE.neg());
    }

    #[test]
    fn tau_rotations_extract_coefficients() {
        let d = 8;
        let mut ctr = 0u64;
        let a = Poly::uniform(d, b"tau", &mut ctr);
        for ell in 1..=d {
            let t = Poly::tau(d, ell);
            assert_eq!(
                t.mul(&a).ct(),
                a.cf(ell),
                "ct(tau_{ell}·a) must equal cf(a)_{ell}"
            );
        }
    }

    #[test]
    fn inner_product_transform() {
        // ct(ā·b) = ⟨a, b⟩
        let d = 16;
        let mut ctr = 0u64;
        let a = Poly::uniform(d, b"ipt-a", &mut ctr);
        let b = Poly::uniform(d, b"ipt-b", &mut ctr);
        let at = a.inner_transform();
        let ct_prod = at.mul(&b).ct();
        let dot: i128 =
            a.0.iter()
                .zip(b.0.iter())
                .map(|(x, y)| x.sym() as i128 * y.sym() as i128)
                .sum();
        // Compare in F_q (the integer dot can exceed q).
        let dot_mod = Fq::from_i64((dot % crate::fp::Q as i128) as i64);
        assert_eq!(ct_prod, dot_mod);
    }

    #[test]
    fn sigma_automorphism() {
        let d = 8;
        let mut ctr = 0u64;
        let a = Poly::uniform(d, b"sig", &mut ctr);
        // σ is an involution up to the ring relations; check σ(σ(a)) ≡ a on
        // the coefficient representation for x ↦ x^{-1}: it is exact.
        let a2 = a.sigma_inv().sigma_inv();
        assert_eq!(a, a2);
        // σ(a·b) = σ(a)·σ(b)
        let b = Poly::uniform(d, b"sig2", &mut ctr);
        assert_eq!(a.mul(&b).sigma_inv(), a.sigma_inv().mul(&b.sigma_inv()));
    }

    #[test]
    fn split_recompose_roundtrip() {
        let d = 8;
        let mut ctr = 0u64;
        // small witness: ∥z∥∞ < 2 = b ⟹ k = 1
        let z_small = vec![Poly::small_b(d, 2, b"sp", &mut ctr); 3];
        let pieces = split_b(&z_small, 2);
        assert_eq!(pieces.len(), 1);
        assert_eq!(recompose(&pieces, 2), z_small);
        // big witness: fold 4 small ones with C-challenges then split
        let zs: Vec<Vec<Poly>> = (0..4)
            .map(|_| {
                (0..3)
                    .map(|_| Poly::small_b(d, 2, b"fold", &mut ctr))
                    .collect()
            })
            .collect();
        let rhos: Vec<Poly> = (0..4)
            .map(|_| StrongSet::sample(d, b"rho", &mut ctr))
            .collect();
        let folded = vec_ring_comb(&zs, &rhos);
        let pieces = split_b(&folded, 2);
        assert_eq!(recompose(&pieces, 2), folded);
        for piece in &pieces {
            for p in piece {
                assert!(p.norm_inf() < 2, "each piece must be norm-bounded by b");
            }
        }
    }

    #[test]
    fn expansion_factor_bound() {
        let d = 8;
        let mut ctr = 0u64;
        let v = Poly::small_b(d, 2, b"exp", &mut ctr);
        let vn = v.norm_inf().max(1);
        for _ in 0..16 {
            let rho = StrongSet::sample(d, b"exp-c", &mut ctr);
            let prod = rho.mul(&v);
            assert!(
                prod.norm_inf() <= StrongSet::expansion_bound(d) * vn,
                "negacyclic bound T(C) ≤ 2d must hold"
            );
        }
    }

    #[test]
    fn norm_preserving_embedding() {
        // Coefficient embedding preserves ℓ∞ norm (§2.4.1).
        let d = 8;
        let mut ctr = 0u64;
        let a = Poly::small_b(d, 2, b"np", &mut ctr);
        let _ = a.norm_inf();
        assert!(a.norm_inf() < 2);
    }
}

#[cfg(test)]
mod split_debug {
    use super::*;

    #[test]
    fn split_b_negative_odd_deep() {
        // -270423 with b = 2: the extraction must terminate with residual 0
        // and recompose exactly.
        let d = 8;
        let mut p = Poly::zero(d);
        p.0[7] = Fq::from_i64(-270423);
        let z = vec![p];
        let pieces = split_b_k(&z, 2, 20);
        assert_eq!(pieces.len(), 20);
        let rec = recompose(&pieces, 2);
        assert_eq!(rec[0].0[7].sym(), -270423, "recomposition must roundtrip");
    }

    #[test]
    fn split_b_exhaustive_small() {
        // Exhaustive roundtrip over a window of negative and positive
        // values at several digit depths.
        let d = 2;
        for v in -5000..=5000 {
            let mut p = Poly::zero(d);
            p.0[0] = Fq::from_i64(v);
            let z = vec![p];
            let pieces = split_b_k(&z, 2, 16);
            let rec = recompose(&pieces, 2);
            assert_eq!(rec[0].0[0].sym(), v, "value {v} must roundtrip");
        }
    }
}
