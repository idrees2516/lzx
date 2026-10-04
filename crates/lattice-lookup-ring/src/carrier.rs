//! The Ajtai carrier over the split ring — the commitment layer the
//! lookup PIOPs compile onto (follow-up (a) of `ring-lookups.md`).
//!
//! This is the workspace's `lattice-commitment` Ajtai stack (same
//! Module-SIS relation `t = A·s mod q`, same statement-absorption and
//! norm-bound discipline) ported onto `RingD`: the NTT kernel there
//! requires `2d | q−1`, which forces the *full* CRT split
//! (`q ≡ 1 mod 8`), incompatible with Lemma 5.8's two-component ring
//! (`q ≡ 5 mod 8`) — so the kernel here is the schoolbook negacyclic
//! MAC, `O(d²)` per product at `d ≤ 64`, with the same cached-row
//! accumulation shape as `ajtai.rs`'s fast path.
//!
//! The carrier is the *digit-window* commitment discipline of
//! Greyhound §2.4: arbitrary ring elements (norm up to `q/2`) cannot
//! be Ajtai-committed directly — binding needs short openings — so
//! the windowed engine (`windowed.rs`) decomposes each oracle vector
//! into `2^w`-bounded digit layers and the carrier commits those.

// (Kernel loops use explicit indices by convention.)
#![allow(clippy::needless_range_loop)]
use crate::ring_d::{Elem, RingD};
use lattice_core::transcript::Transcript;

/// Carrier parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CarrierParams {
    pub ring: RingD,
    /// Commitment output dimension (rows of A).
    pub k: usize,
    /// Witness vector dimension (slots).
    pub m: usize,
    /// ℓ∞ bound for openings (the SIS bound).
    pub norm_bound: u64,
}

impl CarrierParams {
    pub fn verify_shape(&self, s: &[Elem]) -> Result<(), CarrierError> {
        if s.len() != self.m {
            return Err(CarrierError::Dimension {
                expected: self.m,
                got: s.len(),
            });
        }
        for e in s {
            if e.inf_norm(self.ring.q) > self.norm_bound {
                return Err(CarrierError::Norm {
                    norm: e.inf_norm(self.ring.q),
                    bound: self.norm_bound,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CarrierError {
    Dimension { expected: usize, got: usize },
    Norm { norm: u64, bound: u64 },
    Binding,
    Ring(String),
}

/// The public key: `A ∈ R^{k×m}` expanded from a 32-byte seed.
#[derive(Clone)]
pub struct CarrierKey {
    pub params: CarrierParams,
    pub seed: [u8; 32],
    /// Row-major k·m ring elements.
    matrix: Vec<Elem>,
}

impl std::fmt::Debug for CarrierKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CarrierKey")
            .field("q", &self.params.ring.q)
            .field("d", &self.params.ring.d)
            .field("k", &self.params.k)
            .field("m", &self.params.m)
            .field("bound", &self.params.norm_bound)
            .finish()
    }
}

impl CarrierKey {
    pub fn from_seed(params: CarrierParams, seed: [u8; 32]) -> Self {
        let total = params.k * params.m;
        let mut matrix = Vec::with_capacity(total);
        for i in 0..total {
            let mut salt = Vec::with_capacity(48);
            salt.extend_from_slice(&seed);
            salt.extend_from_slice(&(i as u64).to_le_bytes());
            matrix.push(params.ring.random(&salt));
        }
        CarrierKey {
            params,
            seed,
            matrix,
        }
    }

    /// `t = A·s mod q` — schoolbook MAC per row (the split ring has no
    /// NTT at these parameters; see the module docs).
    pub fn commit(&self, s: &[Elem]) -> Result<CarrierCommitment, CarrierError> {
        self.params.verify_shape(s)?;
        let ring = &self.params.ring;
        let mut rows = Vec::with_capacity(self.params.k);
        for i in 0..self.params.k {
            let mut acc = ring.zero();
            for (j, sj) in s.iter().enumerate() {
                if sj.is_zero() {
                    continue;
                }
                let aij = &self.matrix[i * self.params.m + j];
                let prod = ring.mul(aij, sj);
                acc = ring.add(&acc, &prod);
            }
            rows.push(acc);
        }
        Ok(CarrierCommitment { rows })
    }

    /// Absorb the full commitment statement (the wave-6.4 discipline
    /// from `lattice-commitment`): parameters, seed, commitment rows.
    pub fn absorb_statement(
        &self,
        tr: &mut Transcript,
        label: &[u8],
        commitment: &CarrierCommitment,
    ) -> Result<(), CarrierError> {
        let mut head = Vec::with_capacity(64);
        head.extend_from_slice(&self.params.ring.q.to_le_bytes());
        head.extend_from_slice(&(self.params.ring.d as u32).to_le_bytes());
        head.extend_from_slice(&(self.params.k as u32).to_le_bytes());
        head.extend_from_slice(&(self.params.m as u32).to_le_bytes());
        head.extend_from_slice(&self.params.norm_bound.to_le_bytes());
        head.extend_from_slice(&self.seed);
        tr.append_bytes(label, &head)
            .map_err(|_| CarrierError::Binding)?;
        tr.append_bytes(label, &commitment.to_bytes())
            .map_err(|_| CarrierError::Binding)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CarrierCommitment {
    pub rows: Vec<Elem>,
}

impl CarrierCommitment {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for r in &self.rows {
            for &c in r.coeffs() {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out
    }
}

/// Sample a short masking vector (balanced coefficients in
/// `[-bound, bound]`) from a seed stream.
pub fn sample_short(ring: &RingD, m: usize, bound: u64, seed: &[u8]) -> Vec<Elem> {
    let bytes = Transcript::xof(b"carrier-mask", seed, m * ring.d * 2);
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let mut e = ring.zero();
        for j in 0..ring.d {
            let off = (i * ring.d + j) * 2;
            let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            // balanced: v - 2^15 scaled into [-bound, bound]
            let v = (raw as i32) - 32768;
            let scaled = (v as i64) * (bound as i64) / 32768;
            let red = ((scaled % ring.q as i64) + ring.q as i64) % ring.q as i64;
            e.c[j] = red as u64;
        }
        out.push(e);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_binds_and_norm_gates() {
        let ring = RingD::new(8).ok().unwrap();
        let params = CarrierParams {
            ring: ring.clone(),
            k: 2,
            m: 6,
            norm_bound: 1 << 20,
        };
        let key = CarrierKey::from_seed(params, [7u8; 32]);
        let s = sample_short(&ring, 6, 1024, b"s");
        let t = key.commit(&s).ok().unwrap();
        // determinism
        assert_eq!(key.commit(&s).ok().unwrap(), t);
        // homomorphic: commit(s1) + commit(s2) = commit(s1+s2) when norms allow
        let s1 = sample_short(&ring, 6, 512, b"s1");
        let s2 = sample_short(&ring, 6, 512, b"s2");
        let t1 = key.commit(&s1).ok().unwrap();
        let t2 = key.commit(&s2).ok().unwrap();
        let s12: Vec<Elem> = (0..6).map(|i| ring.add(&s1[i], &s2[i])).collect();
        let t12 = key.commit(&s12).ok().unwrap();
        for i in 0..2 {
            assert_eq!(ring.add(&t1.rows[i], &t2.rows[i]), t12.rows[i]);
        }
        // an oversized-norm witness is rejected by the shape gate
        let mut bad = s.clone();
        bad[0] = ring.constant((1 << 30) + 5);
        assert!(matches!(key.commit(&bad), Err(CarrierError::Norm { .. })));
    }

    #[test]
    fn statement_absorption_deterministic() {
        let ring = RingD::new(4).ok().unwrap();
        let params = CarrierParams {
            ring,
            k: 1,
            m: 4,
            norm_bound: 1 << 16,
        };
        let key = CarrierKey::from_seed(params, [3u8; 32]);
        let s = sample_short(&key.params.ring, 4, 32, b"x");
        let t = key.commit(&s).ok().unwrap();
        let mut tr1 = Transcript::new_default(b"a");
        key.absorb_statement(&mut tr1, b"st", &t).ok().unwrap();
        let c1 = tr1.challenge_bytes(b"c", 8).ok().unwrap();
        let mut tr2 = Transcript::new_default(b"a");
        key.absorb_statement(&mut tr2, b"st", &t).ok().unwrap();
        let c2 = tr2.challenge_bytes(b"c", 8).ok().unwrap();
        assert_eq!(c1, c2);
        // different commitment -> different challenge
        let s2 = sample_short(&key.params.ring, 4, 32, b"y");
        let t2 = key.commit(&s2).ok().unwrap();
        let mut tr3 = Transcript::new_default(b"a");
        key.absorb_statement(&mut tr3, b"st", &t2).ok().unwrap();
        let c3 = tr3.challenge_bytes(b"c", 8).ok().unwrap();
        assert_ne!(c1, c3);
    }
}
