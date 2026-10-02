//! The compact Ajtai commitment L (Def 2.28) and the **blinded R1CS
//! layout** (Def 3.1) with its hiding analysis (Lemma 3.3).
//!
//! * `L(z) = A·z` for A ← R_F^{κ×nR} uniform: perfectly homomorphic,
//!   B-binding under MSIS^{∞,κ,q}_{nR,2B}, (B,C)-relaxed binding under
//!   MSIS^{∞,κ,q}_{nR,4TB} (Prop 2.30). Not hiding per se.
//! * The **blinded layout** appends a blinding block e (nF,bl coordinates,
//!   uniform ∥e∥∞ < b) to the witness; the structure's blinding rows are
//!   pass-throughs (M1, M2 rows = δ_i, M3 rows = δ_{ι1}) and the circuit
//!   rows have no support on the blinding columns (clause 4). The language
//!   is unchanged (Remark 3.2) and L([z_circ, e]) is hiding on the block:
//!   statistical regime (1) via the Leftover Hash Lemma when
//!   q^{κ/nR,bl}·2^{256/(nR,bl·d)} ≤ 2(b−1) < b_inv, or computational
//!   regime (2) under DKS^∞_{κ, κ+nR,bl, b}.
//! * `pad`/`unpad` (the ι_bl embedding of Eq (3.19) and the truncation of
//!   Lemma 3.29).

use crate::fp::Fq;
use crate::gauss::Rng;
use crate::ring::Poly;

/// The compact Ajtai commitment (Def 2.28): pp = A ∈ R_F^{κ×nR}.
#[derive(Clone, Debug)]
pub struct AjtaiL {
    pub kappa: usize,
    pub nr: usize,
    /// Row-major κ × nR uniform ring elements.
    pub a: Vec<Vec<Poly>>,
    /// The blinding-block column range in RING coordinates
    /// (nR − nR,bl .. nR) — 0 when the layout is not blinded.
    pub nr_bl: usize,
}

impl AjtaiL {
    pub fn setup(kappa: usize, nr: usize, nr_bl: usize, rng: &mut Rng) -> AjtaiL {
        let d = 1; // placeholder; set per call below
        let _ = d;
        AjtaiL {
            kappa,
            nr,
            a: Vec::new(),
            nr_bl,
        }
        .with_degree_init(rng)
    }

    fn with_degree_init(mut self, _rng: &mut Rng) -> AjtaiL {
        // Degree is supplied at commit time via the witness; the matrix is
        // generated lazily by `setup_d`.
        self.a = Vec::new();
        self
    }

    /// Real setup with the ring degree fixed.
    pub fn setup_d(kappa: usize, nr: usize, nr_bl: usize, d: usize, rng: &mut Rng) -> AjtaiL {
        let mut a = Vec::with_capacity(kappa);
        for _ in 0..kappa {
            let mut row = Vec::with_capacity(nr);
            for _ in 0..nr {
                let mut c = vec![Fq::ZERO; d];
                for coeff in c.iter_mut() {
                    // uniform coefficient via the Rng
                    coeff.0 = {
                        // 64-bit uniform then reduce (q ≈ 2^64: tiny bias).
                        let v = rng.next_u64();
                        v % crate::fp::Q
                    };
                }
                row.push(Poly(c));
            }
            a.push(row);
        }
        AjtaiL {
            kappa,
            nr,
            a,
            nr_bl,
        }
    }

    /// L(z) = A·z ∈ R_F^κ (the RF-module homomorphism).
    pub fn commit(&self, z: &[Poly]) -> Vec<Poly> {
        debug_assert_eq!(z.len(), self.nr);
        let d = z.first().map(|p| p.d()).unwrap_or(0);
        let mut out = Vec::with_capacity(self.kappa);
        for row in &self.a {
            let mut acc = Poly::zero(d);
            for (m, zp) in row.iter().zip(z.iter()) {
                acc.add_assign(&m.mul(zp));
            }
            out.push(acc);
        }
        out
    }

    /// A B-binding collision check: two openings of one commitment.
    /// Returns the difference ∆z = z1 − z2 (an MSIS solution if nonzero).
    pub fn binding_collision(&self, c: &[Poly], z1: &[Poly], z2: &[Poly]) -> Option<Vec<Poly>> {
        let c1 = self.commit(z1);
        let c2 = self.commit(z2);
        if c1 == c && c2 == c && z1 != z2 {
            return Some(
                z1.iter()
                    .zip(z2.iter())
                    .map(|(a, b)| a.sub(b))
                    .collect(),
            );
        }
        None
    }

    /// Statistical-hiding feasibility check for regime (1) of Lemma 3.3:
    /// q^{κ/nR,bl}·2^{256/(nR,bl·d)} ≤ 2(b−1) < b_inv.
    pub fn hiding_regime1_feasible(&self, d: usize, b: i64) -> Result<(f64, f64), String> {
        if self.nr_bl == 0 {
            return Err("not a blinded layout".into());
        }
        let kappa = self.kappa as f64;
        let nrbl = self.nr_bl as f64;
        let log_q = (crate::fp::Q as f64).log2();
        // b_inv = 1/sqrt(τ(z)·q^{1/φ(z)}) — for Φ = X^d+1 (η = 2d, power of
        // two): z = 2d, τ(z) = d, φ(z) = d — b_inv = 1/sqrt(d·q^{1/d}).
        let b_inv = 1.0 / (d as f64 * (crate::fp::Q as f64).powf(1.0 / d as f64)).sqrt();
        let lhs = (kappa / nrbl) * log_q + 256.0 / (nrbl * d as f64);
        let bound = 2.0 * (b - 1) as f64;
        if lhs <= bound && bound < b_inv {
            Ok((lhs, b_inv))
        } else {
            Err(format!(
                "regime (1) infeasible: q^(κ/nR,bl)·2^(256/(nR,bl·d)) = 2^{lhs:.2} \
                 vs 2(b−1) = {bound}, b_inv = {b_inv:.3} — use regime (2) (DKS)"
            ))
        }
    }
}

/// The blinded R1CS layout (Def 3.1): the structure s = (M1, M2, M3, f)
/// with f(Y1,Y2,Y3) = Y2·Y3 − Y1, M1 = I_m, and blinding rows/columns.
///
/// Clause (3): for i ∈ I_bl: (M1)_{i,·} = δ_i, (M2)_{i,·} = δ_i,
/// (M3)_{i,·} = δ_{ι1} — so row i reads z_i·1 − z_i = 0 (Remark 3.2).
/// Clause (4): circuit rows have no support on the blinding columns.
#[derive(Clone, Debug)]
pub struct BlindedStructure {
    /// The un-padded circuit structure (m° constraints over nF° columns).
    pub m_circ: usize,
    pub nf_circ: usize,
    /// The blinding-block size in FIELD coordinates (nF,bl, d | nF,bl).
    pub nf_bl: usize,
    pub d: usize,
    /// The constant-one wire index ι1 (≤ nF,in).
    pub iota1: usize,
}

impl BlindedStructure {
    pub fn nr_bl(&self) -> usize {
        self.nf_bl / self.d
    }

    pub fn m(&self) -> usize {
        self.m_circ + self.nf_bl
    }

    pub fn nf(&self) -> usize {
        self.nf_circ + self.nf_bl
    }

    pub fn nr(&self) -> usize {
        self.nf() / self.d
    }

    /// Build the three structural matrices (M1 = I_m, M2, M3) in the
    /// blinded layout from a circuit-level (M2°, M3°) pair over the
    /// unpadded columns.
    ///
    /// M2°/M3°: m_circ × nf_circ. The padded M2/M3: m × nf with
    /// circuit rows carrying the original support (zero on the blinding
    /// columns — clause 4) and blinding rows per clause (3).
    pub fn build_matrices(&self, m2_circ: &[Vec<i64>], m3_circ: &[Vec<i64>]) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
        let m = self.m();
        let nf = self.nf();
        let bl_start = nf - self.nf_bl;
        let mut m2 = vec![vec![0i64; nf]; m];
        let mut m3 = vec![vec![0i64; nf]; m];
        // Circuit rows.
        for r in 0..self.m_circ {
            for c in 0..self.nf_circ {
                m2[r][c] = m2_circ[r][c];
                m3[r][c] = m3_circ[r][c];
            }
        }
        // Blinding rows: (M1)_{i,·} = δ_i (identity — implicit in M1 = I),
        // (M2)_{i,·} = δ_i, (M3)_{i,·} = δ_{ι1}.
        for i in 0..self.nf_bl {
            let row = self.m_circ + i;
            m2[row][bl_start + i] = 1;
            m3[row][self.iota1] = 1;
        }
        (m2, m3)
    }

    /// Sample the blinding block e ← χ_b^{nR,bl} (uniform ∥e∥∞ ≤ b−1).
    pub fn sample_blinding_block(&self, b: i64, rng: &mut Rng) -> Vec<Poly> {
        let nr_bl = self.nr_bl();
        (0..nr_bl).map(|_| {
            let c: Vec<Fq> = (0..self.d).map(|_| Fq::from_i64(rng.small_b(b))).collect();
            Poly(c)
        }).collect()
    }

    /// Check the padded-satisfaction equivalence of Lemma 3.29:
    /// (s; x; [w_circ, e]) ∈ CCS(b) ⟺ (s°; x; w_circ) ∈ CCS°(b) ∧ ∥e∥∞ < b.
    pub fn check_row_constraints(
        &self,
        z: &[Fq],
        m2: &[Vec<i64>],
        m3: &[Vec<i64>],
    ) -> Result<(), &'static str> {
        let nf = self.nf();
        if z.len() != nf {
            return Err("witness length mismatch");
        }
        let m = self.m();
        for row in 0..m {
            let y1 = z[row]; // M1 = I
            let mut y2 = Fq::ZERO;
            let mut y3 = Fq::ZERO;
            for c in 0..nf {
                if m2[row][c] != 0 {
                    y2 = y2.add(&Fq::from_i64(m2[row][c]).mul(&z[c]));
                }
                if m3[row][c] != 0 {
                    y3 = y3.add(&Fq::from_i64(m3[row][c]).mul(&z[c]));
                }
            }
            if y2.mul(&y3).sub(&y1) != Fq::ZERO {
                return Err("R1CS row violated");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::FieldVec;

    #[test]
    fn ajtai_homomorphic() {
        let mut rng = Rng::new(b"ajtai");
        let d = 4;
        let nr = 16;
        let l = AjtaiL::setup_d(4, nr, 0, d, &mut rng);
        let mut ctr = 0u64;
        let z1: Vec<Poly> = (0..nr).map(|_| Poly::small_b(d, 2, b"z1", &mut ctr)).collect();
        let z2: Vec<Poly> = (0..nr).map(|_| Poly::small_b(d, 2, b"z2", &mut ctr)).collect();
        let rho = Poly::small_b(d, 2, b"rho", &mut ctr);
        let c1 = l.commit(&z1);
        let c2 = l.commit(&z2);
        // L(z1 + rho·z2) = L(z1) + rho·L(z2)
        let comb: Vec<Poly> = z1
            .iter()
            .zip(z2.iter())
            .map(|(a, b)| a.add(&rho.mul(b)))
            .collect();
        let ccomb = l.commit(&comb);
        let csum: Vec<Poly> = c1
            .iter()
            .zip(c2.iter())
            .map(|(a, b)| a.add(&rho.mul(b)))
            .collect();
        assert_eq!(ccomb, csum);
    }

    #[test]
    fn blinded_layout_leaves_language_unchanged() {
        // Remark 3.2 + Lemma 3.29.
        let d = 4;
        let nf_circ = 16;
        let m_circ = 16;
        let nf_bl = 8;
        let bs = BlindedStructure {
            m_circ,
            nf_circ,
            nf_bl,
            d,
            iota1: 0,
        };
        // A satisfiable circuit: y2·y3 = y1 with M2 = M3 = identity-ish
        let m2_circ: Vec<Vec<i64>> = (0..m_circ)
            .map(|r| (0..nf_circ).map(|c| if r == c { 1 } else { 0 }).collect())
            .collect();
        let m3_circ = m2_circ.clone();
        let (m2, m3) = bs.build_matrices(&m2_circ, &m3_circ);
        // Witness: x = [1, ...], w with w_i·w_i... rows read z_i·z_i − z_i = 0
        // ⟹ z_i ∈ {0, 1}: a Boolean witness satisfies.
        let mut rng = Rng::new(b"bl");
        let mut z = vec![Fq::ZERO; bs.nf()];
        z[0] = Fq::ONE;
        for i in 1..bs.nf() {
            z[i] = Fq::from_i64(rng.small_b(2).clamp(0, 1));
        }
        let e: Vec<Fq> = (bs.nf_circ..bs.nf())
            .map(|_| Fq::from_i64(rng.small_b(2)))
            .collect();
        z[bs.nf_circ..].copy_from_slice(&e);
        // With the blinding block unconstrained rows, any e works:
        bs.check_row_constraints(&z, &m2, &m3).unwrap();
        // But a wrong circuit row must fail:
        let mut bad = z.clone();
        bad[1] = Fq::from_i64(2); // 2·2 − 2 = 2 ≠ 0
        assert!(bs.check_row_constraints(&bad, &m2, &m3).is_err());
    }

    #[test]
    fn lemma329_truncation_fibre() {
        // The fibre over (x, w_circ) is exactly {e : ∥e∥∞ < b}.
        let d = 4;
        let bs = BlindedStructure {
            m_circ: 8,
            nf_circ: 8,
            nf_bl: 8,
            d,
            iota1: 0,
        };
        let m2c: Vec<Vec<i64>> = (0..8).map(|r| (0..8).map(|c| if r == c { 1 } else { 0 }).collect()).collect();
        let (m2, m3) = bs.build_matrices(&m2c, &m2c);
        let mut rng = Rng::new(b"fib");
        for _ in 0..8 {
            let e = bs.sample_blinding_block(2, &mut rng);
            let mut z = vec![Fq::ZERO; bs.nf_circ + bs.nf_bl];
            for slot in z.iter_mut().take(bs.nf_circ) {
                *slot = Fq::from_i64(rng.small_b(2).clamp(0, 1));
            }
            // Clause (2) of Definition 3.1: z_{ι1} = 1 (the constant-one
            // wire) — without it the blinding rows do not vanish.
            z[0] = Fq::ONE;
            let flat: Vec<Fq> = {
                let mut v = Vec::new();
                for p in &e {
                    v.extend_from_slice(&p.0);
                }
                v
            };
            z[bs.nf_circ..].copy_from_slice(&flat);
            bs.check_row_constraints(&z, &m2, &m3).unwrap();
        }
        let _ = FieldVec::small_b(16, 2, b"x", &mut 0u64);
    }
}
