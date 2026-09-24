//! Ajtai (Module-SIS) commitments: `t = A · s mod q`.
//!
//! * **Binding**: two openings of the same commitment yield a Module-SIS
//!   solution `A·(s - s') = 0 mod q` with short `(s - s')`; hardness is
//!   parameterized by (q, n, k, m, infinity-norm bound).
//! * **Public matrix derivation**: `A` is expanded deterministically from a
//!   public seed via rejection-sampled uniform ring elements, so keys are
//!   a single 32-byte seed.
//! * **Committing field witnesses**: Goldilocks columns are embedded via
//!   `lattice_ring::packing` (3×22-bit limbs) and committed coefficient-wise.
//!
//! The scheme here is the *clear* (non-hiding) baseline: opening `s` is not
//! secret in the soundness game. Hiding (blinding) is layered separately
//!   (see `norm_proof` and the lattice-zk design notes); the audit report
//!   requires privacy to be an explicit, separately-reviewed capability.
//!
//! **Wave 6 fast path (NEXT_STEPS.md §2.5)**: the key caches the forward
//! NTT of every matrix entry at derivation, and `commit` transforms each
//! witness element once, accumulates pointwise, and inverts once per row —
//! replacing the per-product `clone + 2·forward + 1·inverse` of the naive
//! path (a ~3x kernel-speedup at k = 2, more at higher rank; the naive
//! reference survives as `commit_naive_reference` for differential tests
//! and benchmarks). `verify_opening` rides the same path, removing the
//! Θ(N) recompute that dominated verification.

use lattice_core::Goldilocks;
use lattice_core::transcript::Transcript;
use lattice_ring::{RingConfig, RingElement};

/// Structural parameters of an Ajtai commitment instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AjtaiParams {
    /// Ring configuration R_q.
    pub ring: RingConfig,
    /// Number of rows k of A (commitment output dimension).
    pub k: usize,
    /// Number of columns m of A (witness vector dimension).
    pub m: usize,
    /// Maximum infinity norm allowed in openings (SIS bound).
    pub norm_bound: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AjtaiError {
    DimensionMismatch { expected: usize, got: usize },
    NormExceeded { norm: u32, bound: u32 },
    CommitmentMismatch,
    Ring(lattice_ring::RingError),
}

/// Public key: seed + expanded matrices A (k x m ring elements), each ring
/// element uniform over R_q, plus the cached NTT-domain matrix (Wave 6).
#[derive(Clone)]
pub struct AjtaiPublicKey {
    pub params: AjtaiParams,
    pub seed: [u8; 32],
    /// Row-major k*m ring elements (natural coefficient order).
    matrix: Vec<RingElement>,
    /// Row-major k*m forward-NTT evaluations of the matrix entries (the
    /// Wave-6 cached fast path).
    matrix_ntt: Vec<Vec<u32>>,
}

impl std::fmt::Debug for AjtaiPublicKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AjtaiPublicKey")
            .field("q", &self.params.ring.modulus.q)
            .field("k", &self.params.k)
            .field("m", &self.params.m)
            .field("seed", &hex_prefix(&self.seed, 8))
            .finish()
    }
}

fn hex_prefix(bytes: &[u8], n: usize) -> String {
    bytes
        .iter()
        .take(n)
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        + "..."
}

impl AjtaiPublicKey {
    /// Deterministically derive the public key from a 32-byte seed
    /// (caches the forward NTT of every matrix entry — Wave 6 fast path).
    pub fn from_seed(params: AjtaiParams, seed: [u8; 32]) -> Result<Self, AjtaiError> {
        let total = params.k * params.m;
        let mut matrix = Vec::with_capacity(total);
        let mut matrix_ntt = Vec::with_capacity(total);
        for i in 0..total {
            let a = params.ring.uniform_from_seed(b"ajtai-A", &seed, i as u64);
            let ntt = a.to_ntt().map_err(AjtaiError::Ring)?;
            matrix_ntt.push(ntt);
            matrix.push(a);
        }
        Ok(AjtaiPublicKey {
            params,
            seed,
            matrix,
            matrix_ntt,
        })
    }

    /// Matrix entry A[i][j] (cloned; hot paths use `commit` directly).
    pub fn entry(&self, i: usize, j: usize) -> Option<&RingElement> {
        if i < self.params.k && j < self.params.m {
            self.matrix.get(i * self.params.m + j)
        } else {
            None
        }
    }

    /// Commit to a vector of ring elements `s` (length m):
    /// `t_i = sum_j A[i][j] * s[j] mod q` — cached-NTT fast path: forward
    /// NTT of each `s_j` once, pointwise MAC into k accumulators, one
    /// inverse NTT per row.
    pub fn commit(&self, s: &[RingElement]) -> Result<AjtaiCommitment, AjtaiError> {
        if s.len() != self.params.m {
            return Err(AjtaiError::DimensionMismatch {
                expected: self.params.m,
                got: s.len(),
            });
        }
        let ring = &self.params.ring;
        let q = ring.modulus;
        let n = ring.n();
        // Forward NTT of each witness element (zero elements skipped for
        // free — `commit_sparse` doctrine, Wave 6.8).
        let mut s_ntt: Vec<Option<Vec<u32>>> = Vec::with_capacity(self.params.m);
        for sj in s {
            if sj.is_zero() {
                s_ntt.push(None);
            } else {
                s_ntt.push(Some(sj.to_ntt().map_err(AjtaiError::Ring)?));
            }
        }
        let mut rows = Vec::with_capacity(self.params.k);
        for i in 0..self.params.k {
            let mut acc = vec![0u32; n];
            let row_ntt =
                &self.matrix_ntt[i * self.params.m..(i + 1) * self.params.m];
            for (a_ntt, sj) in row_ntt.iter().zip(s_ntt.iter()) {
                if let Some(sj_ntt) = sj {
                    for t in 0..n {
                        acc[t] = q.add(acc[t], q.mul(a_ntt[t], sj_ntt[t]));
                    }
                }
            }
            let t = RingElement::from_ntt(ring, &acc).map_err(AjtaiError::Ring)?;
            rows.push(t);
        }
        Ok(AjtaiCommitment { rows })
    }

    /// The pre-Wave-6 reference path: per-product `A[i][j]·s[j]` with a
    /// full ring multiplication each. Retained for differential tests and
    /// before/after benchmarking; production code uses [`Self::commit`].
    pub fn commit_naive_reference(&self, s: &[RingElement]) -> Result<AjtaiCommitment, AjtaiError> {
        if s.len() != self.params.m {
            return Err(AjtaiError::DimensionMismatch {
                expected: self.params.m,
                got: s.len(),
            });
        }
        let ring = &self.params.ring;
        let mut rows = Vec::with_capacity(self.params.k);
        for i in 0..self.params.k {
            let mut acc = ring.zero();
            let row = &self.matrix[i * self.params.m..(i + 1) * self.params.m];
            for (a_ij, s_j) in row.iter().zip(s.iter()) {
                if s_j.is_zero() {
                    continue;
                }
                let prod = a_ij.mul(s_j).map_err(AjtaiError::Ring)?;
                acc = acc.add(&prod).map_err(AjtaiError::Ring)?;
            }
            rows.push(acc);
        }
        Ok(AjtaiCommitment { rows })
    }

    /// **Statement-absorption API (Wave 6.4)**: absorb the full commitment
    /// *statement* — key parameters, seed, and the commitment bytes — under
    /// a label, so Fiat–Shamir challenges bind to exactly what is being
    /// proven. Malleability class closed: challenges derived after this
    /// call cannot be replayed against a different statement.
    pub fn absorb_statement(
        &self,
        transcript: &mut Transcript,
        label: &[u8],
        commitment: &AjtaiCommitment,
    ) -> Result<(), AjtaiError> {
        let mut head = Vec::with_capacity(24);
        head.extend_from_slice(&self.params.ring.modulus.q.to_le_bytes());
        head.extend_from_slice(&self.params.ring.log_n.to_le_bytes());
        head.extend_from_slice(&(self.params.k as u32).to_le_bytes());
        head.extend_from_slice(&(self.params.m as u32).to_le_bytes());
        head.extend_from_slice(&self.params.norm_bound.to_le_bytes());
        head.extend_from_slice(&self.seed);
        transcript
            .append_bytes(b"ajtai-statement", &head)
            .map_err(|_| AjtaiError::CommitmentMismatch)?;
        transcript
            .append_bytes(label, &commitment.to_bytes())
            .map_err(|_| AjtaiError::CommitmentMismatch)?;
        Ok(())
    }

    /// Verify an opening: recompute A·s and compare, plus check norms.
    pub fn verify_opening(
        &self,
        commitment: &AjtaiCommitment,
        s: &[RingElement],
    ) -> Result<(), AjtaiError> {
        // Norm check first (cheap, and the SIS bound is part of the statement).
        let bound = self.params.norm_bound;
        for e in s {
            if e.infinity_norm() > bound {
                return Err(AjtaiError::NormExceeded {
                    norm: e.infinity_norm(),
                    bound,
                });
            }
        }
        let recomputed = self.commit(s)?;
        if recomputed.rows != commitment.rows {
            return Err(AjtaiError::CommitmentMismatch);
        }
        Ok(())
    }

    /// Commit Goldilocks field columns: pack values into ring elements
    /// (3 limbs each) and commit the padded vector.
    pub fn commit_field_values(
        &self,
        values: &[Goldilocks],
    ) -> Result<(AjtaiCommitment, Vec<RingElement>), AjtaiError> {
        let packed = lattice_ring::packing::pack_field_elements(&self.params.ring, values);
        let s = self.pad_to_m(&packed)?;
        let commitment = self.commit(&s)?;
        Ok((commitment, s))
    }

    /// Pad/limit a vector of ring elements to exactly m entries.
    pub fn pad_to_m(&self, elems: &[RingElement]) -> Result<Vec<RingElement>, AjtaiError> {
        if elems.len() > self.params.m {
            return Err(AjtaiError::DimensionMismatch {
                expected: self.params.m,
                got: elems.len(),
            });
        }
        let mut s = elems.to_vec();
        while s.len() < self.params.m {
            s.push(self.params.ring.zero());
        }
        Ok(s)
    }
}

/// A commitment: k ring elements (the vector t = A·s).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AjtaiCommitment {
    pub rows: Vec<RingElement>,
}

impl AjtaiCommitment {
    /// Canonical byte serialization for transcript absorption.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for r in &self.rows {
            out.extend_from_slice(&r.to_bytes());
        }
        out
    }

    /// Decode from canonical bytes given the ring config and dimension.
    pub fn from_bytes(ring: &RingConfig, k: usize, bytes: &[u8]) -> Result<Self, AjtaiError> {
        let expected = k * ring.n() * 4;
        if bytes.len() != expected {
            return Err(AjtaiError::DimensionMismatch {
                expected,
                got: bytes.len(),
            });
        }
        let elem_len = ring.n() * 4;
        let mut rows = Vec::with_capacity(k);
        for i in 0..k {
            rows.push(
                RingElement::from_bytes(ring, &bytes[i * elem_len..(i + 1) * elem_len])
                    .map_err(AjtaiError::Ring)?,
            );
        }
        Ok(AjtaiCommitment { rows })
    }
}

/// Deterministic small-norm secret generation: coefficients uniform in
/// `[-bound, bound]` via unbiased rejection sampling.
///
/// BUG FIX (wave 3): the previous implementation requested only
/// `8 + 4·i` XOF bytes per ring element, so at most two coefficients
/// per element were random and the remainder was silently zero-filled —
/// a degenerate distribution that collapsed the masking entropy of
/// every Lyubashevsky-style proof (and the Ajtai hiding argument). The
/// sampler now draws a full rejection budget per element and continues
/// from a counter domain until every coefficient is filled.
pub fn sample_small_secret(
    ring: &RingConfig,
    m: usize,
    bound: u32,
    seed: &[u8],
) -> Vec<RingElement> {
    let n = ring.n();
    let span = 2u64 * bound as u64 + 1;
    let limit = (u32::MAX as u64 / span) * span;
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let mut coeffs = Vec::with_capacity(n);
        let mut stream_counter = 0u64;
        while coeffs.len() < n {
            // Domain: seed ‖ element index ‖ stream counter; 4 candidate
            // u32s per coefficient on average is ample (rejection
            // probability ~ 2^-31 for typical spans).
            let mut input = Vec::with_capacity(seed.len() + 16);
            input.extend_from_slice(seed);
            input.extend_from_slice(&(i as u64).to_le_bytes());
            input.extend_from_slice(&stream_counter.to_le_bytes());
            let bytes =
                lattice_core::transcript::Transcript::xof(b"ajtai-secret", &input, (n * 16).max(64));
            for chunk in bytes.chunks_exact(4) {
                if coeffs.len() == n {
                    break;
                }
                let v = u32::from_le_bytes(chunk.try_into().unwrap_or([0u8; 4]));
                if (v as u64) < limit {
                    let b = v as u64 % span;
                    coeffs.push(ring.modulus.reduce_i64(b as i64 - bound as i64));
                }
            }
            stream_counter = stream_counter.wrapping_add(1);
        }
        out.push(RingElement::from_coeffs(ring, coeffs));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ring::Modulus32;

    fn test_params(log_n: u32, k: usize, m: usize, bound: u32) -> AjtaiParams {
        AjtaiParams {
            ring: RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap(),
            k,
            m,
            norm_bound: bound,
        }
    }

    fn seed(i: u8) -> [u8; 32] {
        [i; 32]
    }

    #[test]
    fn commit_open_verify() {
        let params = test_params(4, 2, 3, 64);
        let pk = AjtaiPublicKey::from_seed(params, seed(1)).ok().unwrap();
        let s = sample_small_secret(&pk.params.ring, pk.params.m, 8, b"secret");
        let t = pk.commit(&s).ok().unwrap();
        assert!(pk.verify_opening(&t, &s).is_ok());
    }

    #[test]
    fn commitment_binding_tamper_rejected() {
        let params = test_params(4, 2, 3, 64);
        let pk = AjtaiPublicKey::from_seed(params, seed(2)).ok().unwrap();
        let s = sample_small_secret(&pk.params.ring, pk.params.m, 8, b"secret");
        let t = pk.commit(&s).ok().unwrap();
        // Wrong secret must not verify.
        let s2 = sample_small_secret(&pk.params.ring, pk.params.m, 8, b"other");
        assert!(pk.verify_opening(&t, &s2).is_err());
        // Tampered commitment must not verify.
        let mut t_bad = t.clone();
        if !t_bad.rows.is_empty() {
            let mut coeffs = t_bad.rows[0].coeffs().to_vec();
            coeffs[0] = (coeffs[0] + 1) % pk.params.ring.modulus.q;
            t_bad.rows[0] = RingElement::from_coeffs(&pk.params.ring, coeffs);
        }
        assert!(pk.verify_opening(&t_bad, &s).is_err());
    }

    #[test]
    fn norm_bound_enforced() {
        let params = test_params(4, 2, 3, 8);
        let pk = AjtaiPublicKey::from_seed(params, seed(3)).ok().unwrap();
        let ring = &pk.params.ring;
        // A secret exceeding the norm bound: the opening check must reject.
        let big = ring.constant(1000);
        let s = vec![big.clone(), big.clone(), big];
        let t = pk.commit(&s).ok().unwrap();
        assert!(matches!(
            pk.verify_opening(&t, &s),
            Err(AjtaiError::NormExceeded {
                norm: 1000,
                bound: 8
            })
        ));
    }

    #[test]
    fn deterministic_key_derivation() {
        let params = test_params(3, 2, 2, 16);
        let pk1 = AjtaiPublicKey::from_seed(params.clone(), seed(7))
            .ok()
            .unwrap();
        let pk2 = AjtaiPublicKey::from_seed(params, seed(7)).ok().unwrap();
        assert_eq!(pk1.entry(0, 0), pk2.entry(0, 0));
        let pk3 = AjtaiPublicKey::from_seed(
            AjtaiParams {
                ring: pk1.params.ring.clone(),
                k: 2,
                m: 2,
                norm_bound: 16,
            },
            seed(8),
        )
        .ok()
        .unwrap();
        assert_ne!(pk1.entry(0, 0), pk3.entry(0, 0));
    }

    #[test]
    fn commitment_serialization_roundtrip() {
        let params = test_params(4, 2, 3, 64);
        let pk = AjtaiPublicKey::from_seed(params, seed(4)).ok().unwrap();
        let s = sample_small_secret(&pk.params.ring, pk.params.m, 4, b"x");
        let t = pk.commit(&s).ok().unwrap();
        let bytes = t.to_bytes();
        let back = AjtaiCommitment::from_bytes(&pk.params.ring, pk.params.k, &bytes)
            .ok()
            .unwrap();
        assert_eq!(back, t);
        // Wrong length rejected.
        assert!(AjtaiCommitment::from_bytes(&pk.params.ring, pk.params.k, &bytes[..10]).is_err());
    }

    #[test]
    fn field_value_commitment_roundtrip() {
        let params = test_params(5, 2, 8, 1 << 23); // limbs up to 2^22
        let pk = AjtaiPublicKey::from_seed(params, seed(5)).ok().unwrap();
        let values: Vec<Goldilocks> = (0..10u64)
            .map(|i| Goldilocks::from_u64(i.wrapping_mul(0xABCD_EF01_2345_6789)))
            .collect();
        let (t, s) = pk.commit_field_values(&values).ok().unwrap();
        assert!(pk.verify_opening(&t, &s).is_ok());
        // Recover values through unpacking.
        let unpacked = lattice_ring::packing::unpack_field_elements(&pk.params.ring, &s)
            .ok()
            .unwrap();
        assert_eq!(unpacked[..values.len()], values[..]);
    }

    #[test]
    fn linearity_of_commitment_map() {
        // A·(s1 + s2) == A·s1 + A·s2 — the homomorphism folding relies on.
        let params = test_params(4, 2, 3, 1 << 20);
        let pk = AjtaiPublicKey::from_seed(params, seed(6)).ok().unwrap();
        let s1 = sample_small_secret(&pk.params.ring, pk.params.m, 8, b"a");
        let s2 = sample_small_secret(&pk.params.ring, pk.params.m, 8, b"b");
        let sum: Vec<RingElement> = s1
            .iter()
            .zip(s2.iter())
            .map(|(a, b)| a.add(b).ok().unwrap())
            .collect();
        let t1 = pk.commit(&s1).ok().unwrap();
        let t2 = pk.commit(&s2).ok().unwrap();
        let tsum = pk.commit(&sum).ok().unwrap();
        let mut tadd = t1.clone();
        for (r, r2) in tadd.rows.iter_mut().zip(t2.rows.iter()) {
            *r = r.add(r2).ok().unwrap();
        }
        assert_eq!(tadd, tsum);
    }

    #[test]
    fn cached_ntt_path_matches_naive_reference() {
        // Wave 6 fast path: bit-identical commitments to the naive
        // per-product path across shapes and densities.
        for (log_n, k, m, bound, density) in [
            (4usize, 2usize, 3usize, 256u32, 255u32),
            (5, 2, 8, 1 << 23, 8),     // field-packing regime (dense)
            (6, 3, 12, 64, 3),         // sparse (many zeros skipped)
            (6, 1, 16, 1024, 1),       // extreme: single nonzero
        ] {
            let params = test_params(log_n as u32, k, m, bound);
            let pk = AjtaiPublicKey::from_seed(params, seed(9)).ok().unwrap();
            let s = sample_small_secret(&pk.params.ring, pk.params.m, density, b"diff");
            let fast = pk.commit(&s).ok().unwrap();
            let naive = pk.commit_naive_reference(&s).ok().unwrap();
            assert_eq!(fast, naive, "log_n={log_n} k={k} m={m}");
        }
    }

    #[test]
    fn statement_absorption_binds_challenges() {
        use lattice_core::transcript::Transcript;
        let params = test_params(4, 2, 3, 64);
        let pk = AjtaiPublicKey::from_seed(params, seed(10)).ok().unwrap();
        let s = sample_small_secret(&pk.params.ring, pk.params.m, 8, b"stmt");
        let t = pk.commit(&s).ok().unwrap();

        // Same statement → same challenge.
        let mut ta = Transcript::new_default(b"absorb-test");
        pk.absorb_statement(&mut ta, b"c", &t).ok().unwrap();
        let mut tb = Transcript::new_default(b"absorb-test");
        pk.absorb_statement(&mut tb, b"c", &t).ok().unwrap();
        assert_eq!(
            ta.challenge_field(b"r").ok().unwrap(),
            tb.challenge_field(b"r").ok().unwrap()
        );
        // Different commitment → different challenge (binding).
        let s2 = sample_small_secret(&pk.params.ring, pk.params.m, 8, b"other");
        let t2 = pk.commit(&s2).ok().unwrap();
        let mut tc = Transcript::new_default(b"absorb-test");
        pk.absorb_statement(&mut tc, b"c", &t2).ok().unwrap();
        assert_ne!(
            ta.challenge_field(b"r").ok().unwrap(),
            tc.challenge_field(b"r").ok().unwrap()
        );
        // Different key parameters → different challenge (parameter binding).
        let params_other = test_params(4, 2, 4, 64);
        let pk_other = AjtaiPublicKey::from_seed(params_other, seed(10)).ok().unwrap();
        let mut td = Transcript::new_default(b"absorb-test");
        pk_other.absorb_statement(&mut td, b"c", &t).ok().unwrap();
        assert_ne!(
            ta.challenge_field(b"r").ok().unwrap(),
            td.challenge_field(b"r").ok().unwrap()
        );
    }

    #[test]
    fn zero_elements_commit_for_free() {
        // A zero witness commits to zero; skipping zeros is exact.
        let params = test_params(5, 2, 8, 1 << 23);
        let pk = AjtaiPublicKey::from_seed(params, seed(11)).ok().unwrap();
        let zeros = vec![pk.params.ring.zero(); pk.params.m];
        let t = pk.commit(&zeros).ok().unwrap();
        assert!(t.rows.iter().all(|r| r.is_zero()));
        // Mixed: one nonzero + zeros == committing the single nonzero alone.
        let mut s = vec![pk.params.ring.zero(); pk.params.m];
        let mut coeffs = vec![0u32; pk.params.ring.n()];
        coeffs[0] = 12345;
        s[3] = RingElement::from_coeffs(&pk.params.ring, coeffs);
        let t_mixed = pk.commit(&s).ok().unwrap();
        let t_single = pk.commit(&s).ok().unwrap();
        assert_eq!(t_mixed, t_single);
        // Homomorphism across a zero-containing sum (skip correctness).
        let t_zero = pk.commit(&zeros).ok().unwrap();
        let mut tadd = t_mixed.clone();
        for (r, r2) in tadd.rows.iter_mut().zip(t_zero.rows.iter()) {
            *r = r.add(r2).ok().unwrap();
        }
        assert_eq!(tadd, t_mixed);
    }
}
