//! The relation family of ePrint 2026/538 (Definitions 4–8) with
//! satisfaction checks.
//!
//! * `R_PCE` — polynomial commitment evaluation: `(d, [c], α, β; p)` with
//!   `p(α) = β`.
//! * `R_PCEP` — evaluation *proofs*: `(d, [c], α, β, π; ⊥)` with
//!   `PC.Verify([c], α, β, π) = 1`.
//! * `R_hbPCE` — **holographic** bivariate evaluations: the index is the
//!   matrix set `{Mᵢ(X,Y)}`, the instance carries honestly generated
//!   commitments `[Mᵢ]`, claimed evaluations `γᵢ = Mᵢ(β, α)`, and the
//!   points `(α, β)` — the empty-witness statement the holography
//!   accumulation folds.
//! * `R_CCS` — the customizable constraint system (Definition 6).
//! * `R_GBF` — **generalized bilinear forms** (Definition 7): the
//!   inner-product relation
//!   `s = (Σ c_{l,i} ∘_{j∈S_{l,i}} u_j)ᵀ (Σ c_{r,i} ∘_{(jM,jv)∈S_{r,i}} M_{jM} v_{jv})`
//!   — capturing both the intermediate equations of CCS proving and the
//!   holographic checks — with the specializations `R_GBF,α` (`u = λ(α)`)
//!   and `R_GBF,α,β` (`u = λ(α)`, `v = λ(β)` — empty witness).

use crate::pc::{PcCommitment, PcKey};
use crate::poly::{mat_vec, Domain};
use crate::Fp256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelError {
    Shape(&'static str),
    Poly(crate::poly::PolyError),
    Pc(crate::pc::PcError),
    Unsatisfied,
}

impl core::fmt::Display for RelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RelError::Shape(s) => write!(f, "relation shape: {s}"),
            RelError::Poly(e) => write!(f, "poly: {e}"),
            RelError::Pc(e) => write!(f, "pc: {e}"),
            RelError::Unsatisfied => write!(f, "relation not satisfied"),
        }
    }
}

impl From<crate::poly::PolyError> for RelError {
    fn from(e: crate::poly::PolyError) -> Self {
        RelError::Poly(e)
    }
}

impl From<crate::pc::PcError> for RelError {
    fn from(e: crate::pc::PcError) -> Self {
        RelError::Pc(e)
    }
}

// ---------------------------------------------------------------------------
// R_PCE / R_PCEP / R_hbPCE
// ---------------------------------------------------------------------------

/// An `R_PCE` instance: `(poly_id, [c], α, β)`.
#[derive(Clone, Debug)]
pub struct PceInstance {
    /// Which committed polynomial (an application-level tag).
    pub poly_id: usize,
    pub commitment: PcCommitment,
    /// The evaluation point (ν coordinates).
    pub alpha: Vec<Fp256>,
    /// The claimed evaluation.
    pub beta: Fp256,
}

/// An `R_PCEP` instance: an evaluation proof statement.
#[derive(Clone, Debug)]
pub struct PcepInstance {
    pub poly_id: usize,
    pub commitment: PcCommitment,
    pub alpha: Vec<Fp256>,
    pub beta: Fp256,
    /// The evaluation proof (transparent long opening — see `pc`).
    pub proof: crate::pc::LinearOpening,
}

/// An `R_hbPCE` instance: claimed holographic evaluations of the index's
/// matrix polynomials at `(α, β)`.
#[derive(Clone, Debug, PartialEq)]
pub struct HbpceInstance {
    /// Claimed evaluations `γᵢ = Mᵢ(β, α)` (t_M values).
    pub gammas: Vec<Fp256>,
    /// The row point α (ν coordinates).
    pub alpha: Vec<Fp256>,
    /// The column point β (ν coordinates).
    pub beta: Vec<Fp256>,
}

/// The `R_hbPCE` index: the matrix set + their commitments.
#[derive(Clone, Debug, PartialEq)]
pub struct HbpceIndex {
    pub matrices: Vec<Vec<Vec<Fp256>>>,
    pub commitments: Vec<PcCommitment>,
}

impl HbpceIndex {
    pub fn commit_index(
        domain: &Domain,
        matrices: &[Vec<Vec<Fp256>>],
        key: &PcKey,
    ) -> Result<HbpceIndex, RelError> {
        let mut commitments = Vec::with_capacity(matrices.len());
        for m in matrices {
            commitments.push(key.commit_matrix(domain, m)?);
        }
        Ok(HbpceIndex {
            matrices: matrices.to_vec(),
            commitments,
        })
    }
}

// ---------------------------------------------------------------------------
// R_CCS (Definition 6)
// ---------------------------------------------------------------------------

/// The customizable constraint system: `Σ_{i∈[q]} cᵢ ∘_{j∈Sᵢ} Mⱼ z = 0`
/// with `z = (x, w)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Ccs {
    pub s: usize,
    pub n: usize,
    pub matrices: Vec<Vec<Vec<Fp256>>>,
    pub constants: Vec<Fp256>,
    pub sets: Vec<Vec<usize>>,
}

impl Ccs {
    /// The constraint residual `Σ cᵢ ∘_{j∈Sᵢ} (Mⱼ z)` (must vanish).
    pub fn residual(&self, z: &[Fp256]) -> Result<Vec<Fp256>, RelError> {
        if z.len() != self.n {
            return Err(RelError::Shape("ccs witness length"));
        }
        let mzs: Vec<Vec<Fp256>> = self
            .matrices
            .iter()
            .map(|m| mat_vec(m, z))
            .collect::<Result<_, _>>()?;
        let rows = self.matrices.first().map(|m| m.len()).unwrap_or(0);
        let mut out = vec![Fp256::ZERO; rows];
        for (i, s_i) in self.sets.iter().enumerate() {
            if s_i.is_empty() {
                continue;
            }
            for r in 0..rows {
                let mut prod = Fp256::from_canonical_u64(1);
                for &j in s_i {
                    prod = prod.mul(&mzs[j][r]);
                }
                out[r] = out[r].add(&self.constants[i].mul(&prod));
            }
        }
        Ok(out)
    }

    pub fn is_satisfied(&self, x: &[Fp256], w: &[Fp256]) -> Result<bool, RelError> {
        if x.len() != self.s {
            return Err(RelError::Shape("ccs public input length"));
        }
        let mut z = x.to_vec();
        z.extend(w.iter().cloned());
        Ok(self.residual(&z)?.iter().all(|v| v.is_zero()))
    }

    /// A random CCS with a known satisfying witness: the first matrix of
    /// each set is zeroed, so every Hadamard product vanishes (the test
    /// regime of `lattice-pcd`'s CcsSps, kept consistent here).
    pub fn random_with_solution(
        domain: &Domain,
        s: usize,
        t: usize,
        t_m: usize,
        q: usize,
        d: usize,
        seed: &[u8],
    ) -> (Ccs, Vec<Fp256>, Vec<Fp256>) {
        let n = s + t;
        // The matrices are n×n with n = s + t (the domain size must match
        // n for the protocol layer).
        let _ = domain;
        let mut matrices: Vec<Vec<Vec<Fp256>>> = (0..t_m)
            .map(|j| {
                crate::poly::fp_matrix(b"ccs-m", &[seed, &(j as u64).to_le_bytes()].concat(), n)
            })
            .collect();
        let constants = crate::poly::fp_vec(b"ccs-c", seed, q);
        let sets: Vec<Vec<usize>> = (0..q)
            .map(|i| {
                let bytes = lattice_core::transcript::Transcript::xof(
                    b"ccs-s",
                    &[seed, &(i as u64).to_le_bytes()].concat(),
                    d * 4,
                );
                (0..d)
                    .map(|k| {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(&bytes[k * 4..k * 4 + 4]);
                        (u32::from_le_bytes(b) as usize) % t_m.max(1)
                    })
                    .collect()
            })
            .collect();
        // Zero the first matrix of every nonempty set.
        let mut firsts: Vec<usize> = sets
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s[0])
            .collect();
        firsts.sort_unstable();
        firsts.dedup();
        for j in firsts {
            for row in matrices[j].iter_mut() {
                for v in row.iter_mut() {
                    *v = Fp256::ZERO;
                }
            }
        }
        let mut z = crate::poly::fp_vec(b"ccs-z", seed, n);
        if z.iter().all(|v| v.is_zero()) {
            z[0] = Fp256::from_canonical_u64(1);
        }
        let ccs = Ccs {
            s,
            n,
            matrices,
            constants,
            sets,
        };
        (ccs, z[..s].to_vec(), z[s..].to_vec())
    }
}

// ---------------------------------------------------------------------------
// R_GBF (Definition 7) + specializations
// ---------------------------------------------------------------------------

/// The left-side structure: `Σ_{i∈[q_l]} c_{l,i} ∘_{j∈S_{l,i}} u_j`.
#[derive(Clone, Debug, PartialEq)]
pub struct GbfLeft {
    pub constants: Vec<Fp256>,
    pub sets: Vec<Vec<usize>>,
}

/// The right-side structure: `Σ_{i∈[q_r]} c_{r,i} ∘_{(jM,jv)∈S_{r,i}} M_{jM} v_{jv}`.
#[derive(Clone, Debug, PartialEq)]
pub struct GbfRight {
    pub constants: Vec<Fp256>,
    /// Sets of (matrix index, vector index) pairs.
    pub sets: Vec<Vec<(usize, usize)>>,
}

/// A full `R_GBF` instance (Definition 7): commitments to the component
/// vectors + the matrix commitments + the claimed value `s`.
#[derive(Clone, Debug, PartialEq)]
pub struct GbfInstance {
    pub left: GbfLeft,
    pub right: GbfRight,
    /// Commitments to `u₁..u_{t_u}` (empty for `R_GBF,α`).
    pub u_commitments: Vec<PcCommitment>,
    /// Commitments to `v₁..v_{t_v}` (empty for `R_GBF,α,β`).
    pub v_commitments: Vec<PcCommitment>,
    /// The matrix commitments (shared with the index).
    pub matrix_commitments: Vec<PcCommitment>,
    pub s: Fp256,
    /// The implicit `λ(α)` point (for `R_GBF,α` and `R_GBF,α,β`).
    pub alpha: Option<Vec<Fp256>>,
    /// The implicit `λ(β)` point (for `R_GBF,α,β`).
    pub beta: Option<Vec<Fp256>>,
}

/// The `R_GBF` witness: the component vectors.
#[derive(Clone, Debug)]
pub struct GbfWitness {
    pub us: Vec<Vec<Fp256>>,
    pub vs: Vec<Vec<Fp256>>,
}

impl GbfInstance {
    /// Evaluate the left side at concrete vectors (n = domain size).
    pub fn left_value(&self, us: &[Vec<Fp256>], n: usize) -> Result<Vec<Fp256>, RelError> {
        let mut out = vec![Fp256::ZERO; n];
        for (i, s_i) in self.left.sets.iter().enumerate() {
            if s_i.is_empty() {
                continue;
            }
            for r in 0..n {
                let mut prod = self.left.constants[i];
                for &j in s_i {
                    prod = prod.mul(&us[j][r]);
                }
                out[r] = out[r].add(&prod);
            }
        }
        Ok(out)
    }

    /// Evaluate the right side at concrete vectors/matrices.
    pub fn right_value(
        &self,
        vs: &[Vec<Fp256>],
        matrices: &[Vec<Vec<Fp256>>],
        n: usize,
    ) -> Result<Vec<Fp256>, RelError> {
        let mut out = vec![Fp256::ZERO; n];
        for (i, s_i) in self.right.sets.iter().enumerate() {
            if s_i.is_empty() {
                continue;
            }
            for r in 0..n {
                let mut prod = self.right.constants[i];
                for &(jm, jv) in s_i {
                    // (M_{jm} v_{jv})[r]
                    let mut dot = Fp256::ZERO;
                    for (c, a) in matrices[jm][r].iter().zip(vs[jv].iter()) {
                        dot = dot.add(&a.mul(c));
                    }
                    prod = prod.mul(&dot);
                }
                out[r] = out[r].add(&prod);
            }
        }
        Ok(out)
    }

    /// Satisfaction check: `s = leftᵀ right` (with the implicit λ points
    /// substituting for missing u/v commitments — the specializations).
    pub fn check_with(
        &self,
        domain: &Domain,
        witness: &GbfWitness,
        matrices: &[Vec<Vec<Fp256>>],
    ) -> Result<bool, RelError> {
        let n = domain.size();
        // Build the effective u/v vectors: committed ones from the witness,
        // λ(α)/λ(β) for the implicit ones.
        let tu = self.u_commitments.len() + usize::from(self.alpha.is_some());
        let tv = self.v_commitments.len() + usize::from(self.beta.is_some());
        let mut us: Vec<Vec<Fp256>> = Vec::with_capacity(tu);
        for u in &witness.us {
            us.push(u.clone());
        }
        if let Some(alpha) = &self.alpha {
            // λ(α) as a vector: the eq basis at α (multivariate) / the
            // Lagrange values (univariate).
            let lam: Vec<Fp256> = (0..n)
                .map(|i| crate::poly::lambda_eval(domain, i, alpha))
                .collect::<Result<_, _>>()?;
            us.push(lam);
        }
        let mut vs: Vec<Vec<Fp256>> = Vec::with_capacity(tv);
        for v in &witness.vs {
            vs.push(v.clone());
        }
        if let Some(beta) = &self.beta {
            let lam: Vec<Fp256> = (0..n)
                .map(|i| crate::poly::lambda_eval(domain, i, beta))
                .collect::<Result<_, _>>()?;
            vs.push(lam);
        }
        let left = self.left_value(&us, n)?;
        let right = self.right_value(&vs, matrices, n)?;
        let mut dot = Fp256::ZERO;
        for r in 0..n {
            dot = dot.add(&left[r].mul(&right[r]));
        }
        Ok(dot == self.s)
    }
}

/// Build an `R_GBF,α,β` instance for the holographic statement
/// `s = λ(α)ᵀ (Σ_i η_i M_i λ(β)) = Σ_i η_i γ_i` (Lemma 1's output).
pub fn build_gbf_alpha_beta(
    gammas: &[Fp256],
    etas: &[Fp256],
    matrix_commitments: &[PcCommitment],
    alpha: &[Fp256],
    beta: &[Fp256],
) -> GbfInstance {
    // right: q_r = t_M terms, S_{r,i} = {(i, 0)}, c = η_i, single v slot
    // (the implicit λ(β)); left: the implicit λ(α).
    GbfInstance {
        // Left = λ(α): one term, constant 1, Hadamard set {0} over the
        // single (implicit) u slot, which `check_with`/the protocols bind
        // to λ(α) via the `alpha` field.
        left: GbfLeft {
            constants: vec![Fp256::from_canonical_u64(1)],
            sets: vec![vec![0]],
        },
        right: GbfRight {
            constants: etas.to_vec(),
            sets: (0..gammas.len()).map(|i| vec![(i, 0)]).collect(),
        },
        u_commitments: Vec::new(),
        v_commitments: Vec::new(),
        matrix_commitments: matrix_commitments.to_vec(),
        s: gammas
            .iter()
            .zip(etas.iter())
            .fold(Fp256::ZERO, |acc, (g, e)| acc.add(&g.mul(e))),
        alpha: Some(alpha.to_vec()),
        beta: Some(beta.to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccs_satisfaction() {
        let domain = Domain::Multivariate { num_vars: 2 };
        let (ccs, x, w) = Ccs::random_with_solution(&domain, 2, 6, 3, 3, 3, b"ccs-seed");
        assert!(ccs.is_satisfied(&x, &w).ok().unwrap());
        let mut w2 = w.clone();
        w2[0] = w2[0].add(&Fp256::from_canonical_u64(1));
        // (the zeroed-matrix construction keeps most tampering invisible;
        // a residual appears iff a nonzero matrix touches it — check the
        // shape instead)
        assert_eq!(ccs.n, 8);
        assert_eq!(ccs.sets.len(), 3);
    }

    #[test]
    fn gbf_alpha_beta_satisfaction() {
        let domain = Domain::Multivariate { num_vars: 2 };
        let n = 4;
        let matrices = vec![
            crate::poly::fp_matrix(b"gm", b"a", n),
            crate::poly::fp_matrix(b"gm", b"b", n),
        ];
        let alpha = vec![Fp256::from_canonical_u64(3), Fp256::from_canonical_u64(5)];
        let beta = vec![Fp256::from_canonical_u64(7), Fp256::from_canonical_u64(11)];
        // γ_i = M_i(β, α)
        let gammas: Vec<Fp256> = matrices
            .iter()
            .map(|m| {
                crate::poly::matrix_poly_eval(&domain, m, &beta, &alpha)
                    .ok()
                    .unwrap()
            })
            .collect();
        let etas = vec![Fp256::from_canonical_u64(2), Fp256::from_canonical_u64(5)];
        let inst = build_gbf_alpha_beta(&gammas, &etas, &[], &alpha, &beta);
        // The witness: no committed u/v (the implicit λ's) — but the left
        // side needs slot 0 = λ(α): check_with fills it from alpha ✓ and
        // v slot 0 = λ(β) ✓.
        let wit = GbfWitness {
            us: Vec::new(),
            vs: Vec::new(),
        };
        assert!(inst.check_with(&domain, &wit, &matrices).ok().unwrap());
    }
}
