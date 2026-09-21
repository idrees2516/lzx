//! Ring elements of `R_q = Z_q[X]/(X^n + 1)` with NTT-based multiplication,
//! norm computation, and small-coefficient helpers used across every
//! lattice protocol in the workspace.

use crate::modulus::Modulus32;
use crate::ntt::{NttError, NttTables};
use std::sync::Arc;

/// Immutable ring configuration; cheap to clone (tables shared).
#[derive(Clone)]
pub struct RingConfig {
    pub modulus: Modulus32,
    pub log_n: u32,
    tables: Arc<NttTables>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RingError {
    Ntt(NttError),
    LengthMismatch { expected: usize, got: usize },
}

impl std::fmt::Debug for RingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingConfig")
            .field("q", &self.modulus.q)
            .field("log_n", &self.log_n)
            .finish()
    }
}

impl RingConfig {
    pub fn new(modulus: Modulus32, log_n: u32) -> Result<Self, RingError> {
        let tables = NttTables::new(modulus, log_n).map_err(RingError::Ntt)?;
        Ok(RingConfig {
            modulus,
            log_n,
            tables: Arc::new(tables),
        })
    }

    pub fn n(&self) -> usize {
        1usize << self.log_n
    }

    pub fn zero(&self) -> RingElement {
        RingElement {
            config: self.clone(),
            coeffs: vec![0u32; self.n()],
        }
    }

    pub fn one(&self) -> RingElement {
        let mut coeffs = vec![0u32; self.n()];
        coeffs[0] = 1;
        RingElement {
            config: self.clone(),
            coeffs,
        }
    }

    /// The ring generator X (i.e., the polynomial X).
    pub fn x_gen(&self) -> RingElement {
        let mut coeffs = vec![0u32; self.n()];
        coeffs[1] = 1;
        RingElement {
            config: self.clone(),
            coeffs,
        }
    }

    /// Constant ring element.
    pub fn constant(&self, c: u32) -> RingElement {
        let mut coeffs = vec![0u32; self.n()];
        coeffs[0] = self.modulus.reduce_u64(c as u64);
        RingElement {
            config: self.clone(),
            coeffs,
        }
    }

    /// Deterministic pseudorandom element (reference/testing only).
    pub fn random(&self, seed: &[u8]) -> RingElement {
        let bytes = lattice_core::transcript::Transcript::xof(b"ring-random", seed, self.n() * 4);
        let coeffs: Vec<u32> = bytes
            .chunks(4)
            .take(self.n())
            .map(|c| {
                let mut arr = [0u8; 4];
                arr.copy_from_slice(&c[..4.min(c.len())]);
                self.modulus.reduce_u64(u32::from_le_bytes(arr) as u64)
            })
            .collect();
        RingElement {
            config: self.clone(),
            coeffs,
        }
    }

    /// Uniform element sampled via rejection from a seed stream (production
    /// path for deriving public matrices from seeds).
    pub fn uniform_from_seed(&self, domain: &[u8], seed: &[u8], index: u64) -> RingElement {
        let mut salt = Vec::with_capacity(domain.len() + seed.len() + 8);
        salt.extend_from_slice(domain);
        salt.extend_from_slice(seed);
        salt.extend_from_slice(&index.to_le_bytes());
        // 4 bytes per coefficient -> 32 bits of entropy vs 31.58-bit modulus:
        // rejection keeps uniformity exact.
        let mut coeffs = Vec::with_capacity(self.n());
        let mut counter = 0u64;
        let q = self.modulus.q as u64;
        while coeffs.len() < self.n() {
            let bytes = lattice_core::transcript::Transcript::xof(
                b"uniform",
                &salt,
                8 + (counter as usize) * 4 + 4,
            );
            // take the tail 4 bytes
            let off = bytes.len() - 4;
            let mut arr = [0u8; 4];
            arr.copy_from_slice(&bytes[off..]);
            let cand = u32::from_le_bytes(arr) as u64;
            // reject >= 2^32 - (2^32 mod q): exact uniformity
            let limit = (u32::MAX as u64 + 1) - ((u32::MAX as u64 + 1) % q);
            if cand < limit {
                coeffs.push((cand % q) as u32);
            }
            counter += 1;
        }
        RingElement {
            config: self.clone(),
            coeffs,
        }
    }
}

/// An element of R_q (natural coefficient order).
#[derive(Clone)]
pub struct RingElement {
    config: RingConfig,
    pub(crate) coeffs: Vec<u32>,
}

impl std::fmt::Debug for RingElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingElement")
            .field("q", &self.config.modulus.q)
            .field("log_n", &self.config.log_n)
            .field(
                "coeffs[..4]",
                &self.coeffs.iter().take(4).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl PartialEq for RingElement {
    fn eq(&self, other: &Self) -> bool {
        self.config.modulus.q == other.config.modulus.q
            && self.config.log_n == other.config.log_n
            && self.coeffs == other.coeffs
    }
}

impl Eq for RingElement {}

impl PartialEq for RingConfig {
    fn eq(&self, other: &Self) -> bool {
        self.modulus.q == other.modulus.q && self.log_n == other.log_n
    }
}

impl Eq for RingConfig {}

impl RingElement {
    /// Construct from raw coefficients (length must be n).
    pub fn from_coeffs(config: &RingConfig, coeffs: Vec<u32>) -> Self {
        debug_assert_eq!(coeffs.len(), config.n());
        RingElement {
            config: config.clone(),
            coeffs,
        }
    }

    /// Construct from signed (balanced) coefficients.
    pub fn from_signed(config: &RingConfig, signed: &[i64]) -> Self {
        let coeffs: Vec<u32> = signed
            .iter()
            .map(|&c| config.modulus.reduce_i64(c))
            .collect();
        let mut padded = coeffs;
        padded.resize(config.n(), 0);
        RingElement {
            config: config.clone(),
            coeffs: padded,
        }
    }

    pub fn coeffs(&self) -> &[u32] {
        &self.coeffs
    }

    pub fn config(&self) -> &RingConfig {
        &self.config
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| *c == 0)
    }

    /// Pointwise addition.
    pub fn add(&self, other: &Self) -> Result<Self, RingError> {
        self.zip(other, |q, a, b| q.add(a, b))
    }

    /// Pointwise subtraction.
    pub fn sub(&self, other: &Self) -> Result<Self, RingError> {
        self.zip(other, |q, a, b| q.sub(a, b))
    }

    /// Additive inverse.
    pub fn neg(&self) -> Self {
        let q = self.config.modulus;
        RingElement {
            config: self.config.clone(),
            coeffs: self.coeffs.iter().map(|c| q.neg(*c)).collect(),
        }
    }

    /// Scalar multiplication by a small integer.
    pub fn scale_i64(&self, c: i64) -> Self {
        let q = self.config.modulus;
        let r = q.reduce_i64(c);
        RingElement {
            config: self.config.clone(),
            coeffs: self.coeffs.iter().map(|a| q.mul(*a, r)).collect(),
        }
    }

    fn zip(&self, other: &Self, f: impl Fn(Modulus32, u32, u32) -> u32) -> Result<Self, RingError> {
        if self.config.modulus.q != other.config.modulus.q
            || self.config.log_n != other.config.log_n
            || self.coeffs.len() != other.coeffs.len()
        {
            return Err(RingError::LengthMismatch {
                expected: self.coeffs.len(),
                got: other.coeffs.len(),
            });
        }
        let q = self.config.modulus;
        Ok(RingElement {
            config: self.config.clone(),
            coeffs: self
                .coeffs
                .iter()
                .zip(other.coeffs.iter())
                .map(|(a, b)| f(q, *a, *b))
                .collect(),
        })
    }

    /// Negacyclic multiplication via NTT.
    pub fn mul(&self, other: &Self) -> Result<Self, RingError> {
        if self.config.modulus.q != other.config.modulus.q
            || self.config.log_n != other.config.log_n
        {
            return Err(RingError::LengthMismatch {
                expected: self.coeffs.len(),
                got: other.coeffs.len(),
            });
        }
        let q = self.config.modulus;
        let mut a = self.coeffs.clone();
        let mut b = other.coeffs.clone();
        self.config.tables.forward(&mut a).map_err(RingError::Ntt)?;
        self.config.tables.forward(&mut b).map_err(RingError::Ntt)?;
        for i in 0..a.len() {
            a[i] = q.mul(a[i], b[i]);
        }
        self.config.tables.inverse(&mut a).map_err(RingError::Ntt)?;
        Ok(RingElement {
            config: self.config.clone(),
            coeffs: a,
        })
    }

    /// NTT-domain representation (evaluations at odd powers of psi,
    /// bit-reversed order).
    pub fn to_ntt(&self) -> Result<Vec<u32>, RingError> {
        let mut a = self.coeffs.clone();
        self.config.tables.forward(&mut a).map_err(RingError::Ntt)?;
        Ok(a)
    }

    /// Reconstruct from NTT-domain evaluations.
    pub fn from_ntt(config: &RingConfig, evals: &[u32]) -> Result<Self, RingError> {
        if evals.len() != config.n() {
            return Err(RingError::LengthMismatch {
                expected: config.n(),
                got: evals.len(),
            });
        }
        let mut a = evals.to_vec();
        config.tables.inverse(&mut a).map_err(RingError::Ntt)?;
        Ok(RingElement {
            config: config.clone(),
            coeffs: a,
        })
    }

    /// Canonical serialization: 4-byte little-endian per coefficient.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.coeffs.len() * 4);
        for c in &self.coeffs {
            out.extend_from_slice(&c.to_le_bytes());
        }
        out
    }

    /// Decode canonical bytes; rejects wrong length only (values are
    /// reduced canonically by the caller's modulus — full canonicality of
    /// each u32 < q is enforced here too).
    pub fn from_bytes(config: &RingConfig, bytes: &[u8]) -> Result<Self, RingError> {
        let n = config.n();
        if bytes.len() != n * 4 {
            return Err(RingError::LengthMismatch {
                expected: n * 4,
                got: bytes.len(),
            });
        }
        let mut coeffs = Vec::with_capacity(n);
        for chunk in bytes.chunks(4) {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(chunk);
            let v = u32::from_le_bytes(arr);
            if v >= config.modulus.q {
                // Non-canonical residue: reject.
                return Err(RingError::LengthMismatch {
                    expected: 0,
                    got: v as usize,
                });
            }
            coeffs.push(v);
        }
        Ok(RingElement {
            config: config.clone(),
            coeffs,
        })
    }

    /// Infinity norm of the *balanced* representative: for each coefficient
    /// c in [0, q), the balanced value is c if c <= q/2 else c - q.
    pub fn infinity_norm(&self) -> u32 {
        let q = self.config.modulus.q;
        let half = q / 2;
        self.coeffs
            .iter()
            .map(|c| if *c <= half { *c } else { q - *c })
            .max()
            .unwrap_or(0)
    }

    /// Euclidean (l2) norm squared, in u64 (balanced representatives).
    pub fn euclidean_norm_squared(&self) -> u64 {
        let q = self.config.modulus.q;
        let half = q / 2;
        let mut acc: u64 = 0;
        for &c in &self.coeffs {
            let b = if c <= half {
                c as i64
            } else {
                c as i64 - q as i64
            };
            acc = acc.saturating_add((b * b) as u64);
        }
        acc
    }

    /// Check all balanced coefficients lie in [-bound, bound].
    pub fn is_bounded_by(&self, bound: u32) -> bool {
        self.infinity_norm() <= bound
    }

    /// Evaluate the polynomial at X = 2^log2_x... not meaningful in the
    /// quotient ring; instead expose coefficient access.
    pub fn coeff(&self, i: usize) -> u32 {
        self.coeffs.get(i).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(log_n: u32) -> RingConfig {
        RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap()
    }

    #[test]
    fn ring_arithmetic() {
        let c = cfg(4);
        let a = c.random(b"a");
        let b = c.random(b"b");
        // a + b - b == a
        assert_eq!(a.add(&b).ok().unwrap().sub(&b).ok().unwrap(), a);
        // a - a == 0
        assert!(a.sub(&a).ok().unwrap().is_zero());
        // neg
        assert!(a.add(&a.neg()).ok().unwrap().is_zero());
        // scale
        let s = a.scale_i64(3);
        assert_eq!(
            s.sub(&a)
                .ok()
                .unwrap()
                .sub(&a)
                .ok()
                .unwrap()
                .sub(&a)
                .ok()
                .unwrap()
                .infinity_norm(),
            0
        );
    }

    #[test]
    fn multiplication_ring_identity() {
        let c = cfg(6);
        let a = c.random(b"m-a");
        let one = c.one();
        assert_eq!(a.mul(&one).ok().unwrap(), a);
        assert!(a.mul(&c.zero()).ok().unwrap().is_zero());
        // X * X^{n-1} = X^n = -1
        let x = c.x_gen();
        let mut x_nm1_coeffs = vec![0u32; c.n()];
        x_nm1_coeffs[c.n() - 1] = 1;
        let x_nm1 = RingElement::from_coeffs(&c, x_nm1_coeffs);
        let prod = x.mul(&x_nm1).ok().unwrap();
        let neg_one = c.constant(c.modulus.q - 1);
        assert_eq!(prod, neg_one);
    }

    #[test]
    fn multiplication_associative_and_distributive() {
        let c = cfg(5);
        let a = c.random(b"d-a");
        let b = c.random(b"d-b");
        let d = c.random(b"d-d");
        let ab_d = a.mul(&b).ok().unwrap().mul(&d).ok().unwrap();
        let a_bd = a.mul(&b.mul(&d).ok().unwrap()).ok().unwrap();
        assert_eq!(ab_d, a_bd);
        let lhs = a.mul(&b.add(&d).ok().unwrap()).ok().unwrap();
        let rhs = a
            .mul(&b)
            .ok()
            .unwrap()
            .add(&a.mul(&d).ok().unwrap())
            .ok()
            .unwrap();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn serialization_canonical_roundtrip() {
        let c = cfg(5);
        let a = c.random(b"s-a");
        let bytes = a.to_bytes();
        let back = RingElement::from_bytes(&c, &bytes).ok().unwrap();
        assert_eq!(back, a);
        // Non-canonical coefficient rejected.
        let mut bad = bytes.clone();
        let q = c.modulus.q;
        let inflated = q + 1;
        bad[..4].copy_from_slice(&inflated.to_le_bytes());
        assert!(RingElement::from_bytes(&c, &bad).is_err());
    }

    #[test]
    fn norms_of_signed_elements() {
        let c = cfg(4);
        let a = RingElement::from_signed(&c, &[1, -2, 3, -4, 0, 0, 0, 0]);
        assert_eq!(a.infinity_norm(), 4);
        assert_eq!(a.euclidean_norm_squared(), 1 + 4 + 9 + 16);
        assert!(a.is_bounded_by(4));
        assert!(!a.is_bounded_by(3));
    }

    #[test]
    fn uniform_sampling_deterministic_and_in_range() {
        let c = cfg(6);
        let u1 = c.uniform_from_seed(b"matrix", b"seed", 0);
        let u2 = c.uniform_from_seed(b"matrix", b"seed", 0);
        assert_eq!(u1, u2);
        for coeff in u1.coeffs() {
            assert!(*coeff < c.modulus.q);
        }
        // Different index -> different element (whp).
        let u3 = c.uniform_from_seed(b"matrix", b"seed", 1);
        assert_ne!(u1, u3);
    }

    #[test]
    fn ntt_view_roundtrip() {
        let c = cfg(5);
        let a = c.random(b"ntt-a");
        let evals = a.to_ntt().ok().unwrap();
        let back = RingElement::from_ntt(&c, &evals).ok().unwrap();
        assert_eq!(back, a);
    }
}
