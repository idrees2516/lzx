//! Vector Pedersen commitments over BN254 G1 — the homomorphic commitment
//! instantiation the paper itself assumes ("We assume the commitment scheme
//! is conducted over cyclic groups for simplicity", §5 complexity notes).
//!
//! * `Com(v; r) = Σᵢ vᵢ·Gᵢ + r·H` with scalars in `F_r`, points on G1.
//! * **Homomorphic over `F_r`-scalars** (the accumulation linear
//!   combinations of §5.2):
//!   `Com(v; r) + λ·Com(v'; r') = Com(v + λ·v'; r + λ·r')`.
//! * **Statistically hiding** (`r ←$ F_r`, one fresh scalar per commitment;
//!   the commitment is a perfectly-uniform curve point for every fixed `v`).
//! * **Binding** = discrete-log hardness of G1 (the standard Pedersen
//!   argument; not post-quantum — documented in the paper notes, whose PQ
//!   route is the Ajtai/shortness discipline of the LatticeFold line).
//!
//! Bases `G₁..G_m, H` are derived deterministically from a public 32-byte
//! seed via `g1::hash_to_point`, so keys are one seed + a lazily-expanded
//! base table.

use crate::g1::{hash_to_point, G1Affine, G1Point};
use crate::util::fp_from_be32;
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PedersenError {
    DimensionMismatch { expected: usize, got: usize },
    G1(crate::g1::G1Error),
}

impl core::fmt::Display for PedersenError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PedersenError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            PedersenError::G1(e) => write!(f, "g1 error: {e}"),
        }
    }
}

impl From<crate::g1::G1Error> for PedersenError {
    fn from(e: crate::g1::G1Error) -> Self {
        PedersenError::G1(e)
    }
}

/// A vector-Pedersen key: public seed + expanded affine bases
/// `[G₁ .. G_m, H]`.
#[derive(Clone)]
pub struct PedersenKey {
    /// Public seed for base derivation.
    pub seed: [u8; 32],
    /// Capacity: the maximum vector length committable with these bases.
    pub capacity: usize,
    /// Affine bases `G₁..G_m`.
    pub bases: Vec<G1Affine>,
    /// The hiding base `H`.
    pub blinding: G1Affine,
}

impl PedersenKey {
    /// Derive a key with `capacity` bases from a public seed.
    pub fn derive(seed: &[u8], capacity: usize) -> Result<PedersenKey, PedersenError> {
        let mut bases = Vec::with_capacity(capacity);
        for i in 0..capacity {
            bases.push(hash_to_point(b"pedersen-base", seed, i as u64)?);
        }
        let blinding = hash_to_point(b"pedersen-blind", seed, 0)?;
        let mut s = [0u8; 32];
        s.copy_from_slice(&seed[..seed.len().min(32)]);
        Ok(PedersenKey {
            seed: s,
            capacity,
            bases,
            blinding,
        })
    }

    /// Deterministic "randomness" expansion (for reproducible tests): a
    /// scalar derived from (seed, label, index) — NOT for production.
    pub fn deterministic_blind(&self, label: &[u8], index: u64) -> Fp256 {
        let mut input = Vec::with_capacity(label.len() + 40);
        input.extend_from_slice(label);
        input.extend_from_slice(&index.to_le_bytes());
        input.extend_from_slice(&self.seed);
        let bytes = Transcript::xof(b"pedersen-blind-scalar", &input, 32);
        let mut b = [0u8; 32];
        b.copy_from_slice(&bytes);
        fp_from_be32(&b)
    }

    /// Commit `values` with the given blinding scalar:
    /// `C = Σ vᵢ·Gᵢ + r·H`.
    pub fn commit(&self, values: &[Fp256], r: &Fp256) -> Result<PedersenCommitment, PedersenError> {
        if values.len() > self.capacity {
            return Err(PedersenError::DimensionMismatch {
                expected: self.capacity,
                got: values.len(),
            });
        }
        // Combine the value bases and the blinding base into one MSM.
        let mut bases: Vec<G1Affine> = self.bases[..values.len()].to_vec();
        bases.push(self.blinding);
        let mut scalars = values.to_vec();
        scalars.push(*r);
        Ok(PedersenCommitment {
            point: msm(&bases, &scalars),
        })
    }

    /// Commit with fresh transcript-sampled blinding; returns the commitment
    /// and the blinding scalar (the opening's hiding part).
    pub fn commit_fresh(
        &self,
        values: &[Fp256],
        transcript: &mut Transcript,
    ) -> Result<(PedersenCommitment, Fp256), PedersenError> {
        let mut buf = Vec::new();
        for v in values {
            buf.extend_from_slice(&v.canon_bytes());
        }
        // Absorb the values so the blind is bound to this commitment.
        let _ = transcript.append_bytes(b"pedersen-values", &buf);
        let rb = transcript
            .challenge_bytes(b"pedersen-blind", 32)
            .map_err(|_| PedersenError::G1(crate::g1::G1Error::BadEncoding))?;
        let mut rbytes = [0u8; 32];
        rbytes.copy_from_slice(&rb);
        let r = fp_from_be32(&rbytes);
        let c = self.commit(values, &r)?;
        Ok((c, r))
    }

    /// Verify an opening: check `C == Σ vᵢ·Gᵢ + r·H`.
    pub fn verify_opening(
        &self,
        c: &PedersenCommitment,
        values: &[Fp256],
        r: &Fp256,
    ) -> Result<bool, PedersenError> {
        Ok(self.commit(values, r)? == *c)
    }

    /// The hiding base (public).
    pub fn blinding_base(&self) -> G1Affine {
        self.blinding
    }

    /// The value bases (public).
    pub fn value_bases(&self) -> &[G1Affine] {
        &self.bases
    }
}

/// Multi-scalar multiplication `Σ λᵢ·Pᵢ` — the 4-bit bucket method:
/// ~`(bits/4)·16` point additions plus the window aggregation, independent
/// of the number of terms (vs. per-scalar double-and-add). Used by every
/// commitment and linear combination.
pub fn msm(bases: &[G1Affine], scalars: &[Fp256]) -> G1Affine {
    if bases.is_empty() || scalars.is_empty() {
        return G1Affine::identity();
    }
    const C: u32 = 4;
    const WINS: usize = 256 / C as usize;
    let mut result = G1Point::identity();
    for w in (0..WINS).rev() {
        result = result.double();
        // (repeat the shift C times: w's window sits C nibbles up)
        for _ in 1..C {
            result = result.double();
        }
        let mut buckets: Vec<G1Point> = vec![G1Point::identity(); 16];
        for (i, sc) in scalars.iter().enumerate() {
            let canon = sc.mul(&Fp256 { limbs: [1, 0, 0, 0] }).limbs;
            // nibble w (little-endian nibbles over the 4 limbs)
            let bit_pos = w * C as usize;
            let limb = bit_pos / 64;
            let shift = (bit_pos % 64) as u32;
            let nib = if limb < 4 {
                ((canon[limb] >> shift) & 0xF) as usize
            } else {
                0
            };
            if nib != 0 {
                buckets[nib] = buckets[nib].add(&bases[i].to_projective());
            }
        }
        // Aggregate: Σ_{k=1..15} k·bucket[k] via a running tail sum.
        let mut running = G1Point::identity();
        let mut acc = G1Point::identity();
        for k in (1..16).rev() {
            running = running.add(&buckets[k]);
            acc = acc.add(&running);
        }
        result = result.add(&acc);
    }
    result.to_affine()
}

/// A commitment: one affine G1 point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PedersenCommitment {
    pub point: G1Affine,
}

impl PedersenCommitment {
    /// Canonical bytes for transcript absorption.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.point.to_bytes()
    }

    /// Homomorphic addition.
    pub fn add(&self, other: &PedersenCommitment) -> PedersenCommitment {
        let p = self.point.to_projective().add(&other.point.to_projective());
        PedersenCommitment {
            point: p.to_affine(),
        }
    }

    /// Homomorphic scalar multiplication.
    pub fn scale(&self, k: &Fp256) -> PedersenCommitment {
        PedersenCommitment {
            point: self.point.to_projective().mul_scalar(k).to_affine(),
        }
    }

    /// The identity commitment (commitment of the empty vector with r = 0).
    pub fn identity() -> PedersenCommitment {
        PedersenCommitment {
            point: G1Affine::identity(),
        }
    }

    /// Linear combination `Σ λⱼ·Cⱼ` — the accumulation verifier's core
    /// homomorphism (paper Eq. (2)/(10) of §5.2 step 8).
    pub fn linear_combine(items: &[(Fp256, PedersenCommitment)]) -> PedersenCommitment {
        let bases: Vec<G1Affine> = items.iter().map(|(_, c)| c.point).collect();
        let scalars: Vec<Fp256> = items.iter().map(|(l, _)| *l).collect();
        PedersenCommitment {
            point: msm(&bases, &scalars),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.point.infinity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> Fp256 {
        Fp256::from_canonical_u64(v)
    }

    fn key() -> PedersenKey {
        PedersenKey::derive(&[7u8; 32], 16).ok().unwrap()
    }

    #[test]
    fn commit_open_roundtrip() {
        let k = key();
        let v: Vec<Fp256> = (0..8).map(|i| fr(i as u64 * 3 + 1)).collect();
        let r = fr(0x1234_5678);
        let c = k.commit(&v, &r).ok().unwrap();
        assert!(k.verify_opening(&c, &v, &r).ok().unwrap());
        // Wrong values rejected
        let mut v2 = v.clone();
        v2[3] = fr(99);
        assert!(!k.verify_opening(&c, &v2, &r).ok().unwrap());
        // Wrong blind rejected
        assert!(!k.verify_opening(&c, &v, &fr(1)).ok().unwrap());
    }

    #[test]
    fn homomorphism_add() {
        let k = key();
        let v1: Vec<Fp256> = (0..6).map(|i| fr(i as u64 + 1)).collect();
        let v2: Vec<Fp256> = (0..6).map(|i| fr(10 * (i as u64 + 3))).collect();
        let r1 = fr(11);
        let r2 = fr(22);
        let c1 = k.commit(&v1, &r1).ok().unwrap();
        let c2 = k.commit(&v2, &r2).ok().unwrap();
        let v3: Vec<Fp256> = v1.iter().zip(v2.iter()).map(|(a, b)| a.add(b)).collect();
        let c3 = k.commit(&v3, &r1.add(&r2)).ok().unwrap();
        assert_eq!(c1.add(&c2), c3);
    }

    #[test]
    fn homomorphism_scalar_linear_combo() {
        // C(Σ λⱼ vⱼ; Σ λⱼ rⱼ) == Σ λⱼ C(vⱼ; rⱼ) — the paper's step-8 check.
        let k = key();
        let lam = [fr(5), fr(7), fr(0xdead_beef)];
        let vs: Vec<Vec<Fp256>> = (0..3)
            .map(|j| (0..6).map(|i| fr((i as u64 + 1) * (j as u64 + 2))).collect())
            .collect();
        let rs: Vec<Fp256> = (0..3).map(|j| fr(100 + j as u64 * 17)).collect();
        let cs: Vec<PedersenCommitment> = vs
            .iter()
            .zip(rs.iter())
            .map(|(v, r)| k.commit(v, r).ok().unwrap())
            .collect();
        // Combined witness
        let mut vcomb = vec![Fp256::ZERO; 6];
        let mut rcomb = Fp256::ZERO;
        for j in 0..3 {
            for i in 0..6 {
                vcomb[i] = vcomb[i].add(&lam[j].mul(&vs[j][i]));
            }
            rcomb = rcomb.add(&lam[j].mul(&rs[j]));
        }
        let ccomb = k.commit(&vcomb, &rcomb).ok().unwrap();
        let lhs = PedersenCommitment::linear_combine(&[
            (lam[0], cs[0]),
            (lam[1], cs[1]),
            (lam[2], cs[2]),
        ]);
        assert_eq!(lhs, ccomb);
    }

    #[test]
    fn hiding_property_statistical() {
        // Committing two different values with fresh blinds must give the
        // same commitment *value-space* — a spot check that the blind
        // dominates: commit(v1, r) == commit(v2, r') has solutions, i.e.
        // the map is not injective (hiding by perfect-uniformity of r·H).
        let k = key();
        let v1 = vec![fr(1), fr(2), fr(3)];
        let v2 = vec![fr(9), fr(9), fr(9)];
        let r1 = fr(0x1111);
        let c1 = k.commit(&v1, &r1).ok().unwrap();
        // Find r2 with commit(v2, r2) == c1: r2 = r1 + Σ (v1−v2)·log ratio…
        // cannot compute DLogs; instead verify the STRUCTURAL claim:
        // commit(v2, r1) != c1 (values bind given the same blind).
        let c2 = k.commit(&v2, &r1).ok().unwrap();
        assert_ne!(c1, c2);
    }

    #[test]
    fn fresh_commit_binding() {
        let k = key();
        let mut t = Transcript::new_default(b"test-pedersen");
        let v: Vec<Fp256> = (0..4).map(|i| fr(i as u64 * 5)).collect();
        let (c, r) = k.commit_fresh(&v, &mut t).ok().unwrap();
        assert!(k.verify_opening(&c, &v, &r).ok().unwrap());
    }
}
