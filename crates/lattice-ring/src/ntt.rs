//! Complete negacyclic number-theoretic transform over `Z_q[X]/(X^n + 1)`.
//!
//! Strategy (standard "psi-scaling" construction): a negacyclic transform of
//! length n equals a *cyclic* NTT of length n after pre-scaling coefficient
//! `a[i]` by `psi^i`, where psi is a primitive 2n-th root of unity. The
//! cyclic NTT is an iterative Cooley–Tukey (DIT: bit-reverse input) forward
//! and Gentleman–Sande (DIF: bit-reverse output) inverse.
//!
//! Also provides the RoKoko-style *incomplete* NTT: apply only the first
//! `levels` butterfly levels, leaving a mixed representation that supports
//! component-wise operations on partially-transformed limbs.

use crate::modulus::Modulus32;

/// Precomputed tables for a fixed (modulus, log2 n) pair.
pub struct NttTables {
    pub modulus: Modulus32,
    pub log_n: u32,
    /// pow_psi[i] = psi^i for i in [0, n): pre-scaling factors.
    pow_psi: Vec<u32>,
    /// pow_psi_inv[i] = psi^{-i}: post-scaling factors.
    pow_psi_inv: Vec<u32>,
    /// pow_omega[e] = omega^e for e in [0, n/2), omega = psi^2.
    pow_omega: Vec<u32>,
    /// pow_omega_inv[e] = omega^{-e}.
    pow_omega_inv: Vec<u32>,
    /// n^{-1} mod q.
    n_inv: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NttError {
    LengthMismatch,
    AdicityTooSmall,
}

impl NttTables {
    /// Build tables for transform length n = 2^log_n.
    pub fn new(modulus: Modulus32, log_n: u32) -> Result<Self, NttError> {
        if log_n + 1 > modulus.two_adicity {
            // psi must have order 2n = 2^(log_n+1) <= 2^adicity.
            return Err(NttError::AdicityTooSmall);
        }
        let n = 1usize << log_n;
        let psi = modulus
            .root_of_unity_order_2n(log_n)
            .ok_or(NttError::AdicityTooSmall)?;
        let psi_inv = modulus.inv(psi).ok_or(NttError::AdicityTooSmall)?;
        let omega = modulus.mul(psi, psi);
        let omega_inv = modulus.inv(omega).ok_or(NttError::AdicityTooSmall)?;
        let n_inv = modulus.pow(n as u32, modulus.q as u64 - 2);

        let mut pow_psi = Vec::with_capacity(n);
        let mut pow_psi_inv = Vec::with_capacity(n);
        let mut pow_omega = Vec::with_capacity(n / 2);
        let mut pow_omega_inv = Vec::with_capacity(n / 2);
        let mut p = 1u32;
        let mut pi = 1u32;
        for _ in 0..n {
            pow_psi.push(p);
            pow_psi_inv.push(pi);
            p = modulus.mul(p, psi);
            pi = modulus.mul(pi, psi_inv);
        }
        let mut w = 1u32;
        let mut wi = 1u32;
        for _ in 0..(n / 2).max(1) {
            pow_omega.push(w);
            pow_omega_inv.push(wi);
            w = modulus.mul(w, omega);
            wi = modulus.mul(wi, omega_inv);
        }
        Ok(NttTables {
            modulus,
            log_n,
            pow_psi,
            pow_psi_inv,
            pow_omega,
            pow_omega_inv,
            n_inv,
        })
    }

    pub fn n(&self) -> usize {
        1usize << self.log_n
    }

    /// Forward negacyclic NTT: coefficients (natural order) -> evaluations
    /// at psi^{2j+1} (bit-reversed order). In place.
    /// (Butterflies index paired slots a[i], a[i+half]; range-loop lint
    /// intentionally allowed.)
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, a: &mut [u32]) -> Result<(), NttError> {
        if a.len() != self.n() {
            return Err(NttError::LengthMismatch);
        }
        let q = self.modulus;
        // Pre-scale by psi^i.
        for i in 0..a.len() {
            a[i] = q.mul(a[i], self.pow_psi[i]);
        }
        // Cyclic CT: bit-reverse, then butterflies len = 2 .. n.
        bit_reverse(a);
        let n = a.len();
        let mut len = 2usize;
        while len <= n {
            let stride = n / len; // exponent step for twiddles
            let mut start = 0usize;
            while start < n {
                for j in 0..len / 2 {
                    let w = self.pow_omega[stride * j];
                    let u = a[start + j];
                    let v = q.mul(a[start + j + len / 2], w);
                    a[start + j] = q.add(u, v);
                    a[start + j + len / 2] = q.sub(u, v);
                }
                start += len;
            }
            len <<= 1;
        }
        Ok(())
    }

    /// Inverse negacyclic NTT (in place, exact inverse of `forward`).
    #[allow(clippy::needless_range_loop)]
    pub fn inverse(&self, a: &mut [u32]) -> Result<(), NttError> {
        if a.len() != self.n() {
            return Err(NttError::LengthMismatch);
        }
        let q = self.modulus;
        let n = a.len();
        // GS butterflies len = n .. 2, then bit-reverse, then rescale.
        let mut len = n;
        while len >= 2 {
            let stride = n / len;
            let mut start = 0usize;
            while start < n {
                for j in 0..len / 2 {
                    let w = self.pow_omega_inv[stride * j];
                    let u = a[start + j];
                    let v = a[start + j + len / 2];
                    a[start + j] = q.add(u, v);
                    a[start + j + len / 2] = q.mul(q.sub(u, v), w);
                }
                start += len;
            }
            len >>= 1;
        }
        bit_reverse(a);
        // Post-scale by psi^{-i} and n^{-1} (fused).
        for i in 0..n {
            a[i] = q.mul(q.mul(a[i], self.pow_psi_inv[i]), self.n_inv);
        }
        Ok(())
    }

    /// Incomplete NTT (RoKoko-style): pre-scale, bit-reverse, then apply
    /// only the first `levels` butterfly levels (len = 2 .. 2^levels).
    /// The result is a linear mixed representation; the same tables drive
    /// `complete_partial` to finish the transform.
    #[allow(clippy::needless_range_loop)]
    pub fn forward_partial(&self, a: &mut [u32], levels: u32) -> Result<(), NttError> {
        if a.len() != self.n() || levels > self.log_n {
            return Err(NttError::LengthMismatch);
        }
        let q = self.modulus;
        for i in 0..a.len() {
            a[i] = q.mul(a[i], self.pow_psi[i]);
        }
        bit_reverse(a);
        let n = a.len();
        let mut len = 2usize;
        let mut applied = 0u32;
        while len <= n && applied < levels {
            let stride = n / len;
            let mut start = 0usize;
            while start < n {
                for j in 0..len / 2 {
                    let w = self.pow_omega[stride * j];
                    let u = a[start + j];
                    let v = q.mul(a[start + j + len / 2], w);
                    a[start + j] = q.add(u, v);
                    a[start + j + len / 2] = q.sub(u, v);
                }
                start += len;
            }
            len <<= 1;
            applied += 1;
        }
        Ok(())
    }

    /// Complete a partial transform: apply the remaining levels
    /// (level `already + 1` onwards, where level j uses len = 2^j).
    #[allow(clippy::needless_range_loop)]
    pub fn complete_partial(&self, a: &mut [u32], already: u32) -> Result<(), NttError> {
        if a.len() != self.n() || already > self.log_n {
            return Err(NttError::LengthMismatch);
        }
        let q = self.modulus;
        let n = a.len();
        // Next un-applied level is `already + 1` (len = 2^(already+1));
        // for already == log_n nothing remains.
        let mut len = 1usize << (already + 1);
        while len <= n {
            let stride = n / len;
            let mut start = 0usize;
            while start < n {
                for j in 0..len / 2 {
                    let w = self.pow_omega[stride * j];
                    let u = a[start + j];
                    let v = q.mul(a[start + j + len / 2], w);
                    a[start + j] = q.add(u, v);
                    a[start + j + len / 2] = q.sub(u, v);
                }
                start += len;
            }
            len <<= 1;
        }
        Ok(())
    }
}

/// In-place bit reversal permutation.
fn bit_reverse(a: &mut [u32]) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = reverse_bits(i as u32, bits) as usize;
        if i < j {
            a.swap(i, j);
        }
    }
}

fn reverse_bits(x: u32, bits: u32) -> u32 {
    let mut r = 0u32;
    let mut v = x;
    for _ in 0..bits {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::{RingConfig, RingElement};

    fn rand_poly(cfg: &RingConfig, seed: &[u8]) -> RingElement {
        let n = cfg.n();
        let bytes = lattice_core::transcript::Transcript::xof(b"ring-rand", seed, n * 4);
        let coeffs: Vec<u32> = bytes
            .chunks(4)
            .take(n)
            .map(|c| {
                let mut arr = [0u8; 4];
                arr.copy_from_slice(&c[..4.min(c.len())]);
                cfg.modulus.reduce_u64(u32::from_le_bytes(arr) as u64)
            })
            .collect();
        RingElement::from_coeffs(cfg, coeffs)
    }

    #[test]
    fn ntt_roundtrip() {
        let m = Modulus32::Q_32;
        for &log_n in &[1u32, 2, 3, 5, 8] {
            let cfg = RingConfig::new(m, log_n).ok().unwrap();
            let a = rand_poly(&cfg, b"ntt-a");
            let mut coeffs = a.coeffs().to_vec();
            let tables = NttTables::new(m, log_n).ok().unwrap();
            tables.forward(&mut coeffs).ok().unwrap();
            tables.inverse(&mut coeffs).ok().unwrap();
            assert_eq!(coeffs, a.coeffs().to_vec(), "roundtrip failed at log_n={log_n}");
        }
    }

    #[test]
    fn ntt_mul_matches_schoolbook() {
        let m = Modulus32::Q_32;
        let log_n = 5u32;
        let cfg = RingConfig::new(m, log_n).ok().unwrap();
        let a = rand_poly(&cfg, b"sb-a");
        let b = rand_poly(&cfg, b"sb-b");

        // Schoolbook negacyclic reference: (X^n = -1).
        let n = cfg.n();
        let q = m;
        let mut school = vec![0u32; n];
        for i in 0..n {
            for j in 0..n {
                let v = q.mul(a.coeffs()[i], b.coeffs()[j]);
                if i + j < n {
                    school[i + j] = q.add(school[i + j], v);
                } else {
                    school[i + j - n] = q.sub(school[i + j - n], v);
                }
            }
        }

        let mut ca = a.coeffs().to_vec();
        let mut cb = b.coeffs().to_vec();
        let tables = NttTables::new(m, log_n).ok().unwrap();
        tables.forward(&mut ca).ok().unwrap();
        tables.forward(&mut cb).ok().unwrap();
        for i in 0..n {
            ca[i] = q.mul(ca[i], cb[i]);
        }
        tables.inverse(&mut ca).ok().unwrap();
        assert_eq!(ca, school);
    }

    #[test]
    fn partial_then_complete_equals_full() {
        let m = Modulus32::Q_32;
        let log_n = 5u32;
        let cfg = RingConfig::new(m, log_n).ok().unwrap();
        let a = rand_poly(&cfg, b"pc-a");
        let tables = NttTables::new(m, log_n).ok().unwrap();

        let mut full = a.coeffs().to_vec();
        tables.forward(&mut full).ok().unwrap();

        for levels in 0..=log_n {
            let mut partial = a.coeffs().to_vec();
            tables.forward_partial(&mut partial, levels).ok().unwrap();
            tables.complete_partial(&mut partial, levels).ok().unwrap();
            assert_eq!(
                partial, full,
                "partial({levels}) + complete != full forward"
            );
        }
    }

    #[test]
    fn rejects_bad_geometry() {
        let m = Modulus32::Q_12289; // adicity 12 -> max log_n = 11
        assert!(NttTables::new(m, 11).is_ok());
        assert!(NttTables::new(m, 12).is_err());
        let t = NttTables::new(m, 8).ok().unwrap();
        assert_eq!(
            t.forward(&mut vec![0u32; 128]).err(),
            Some(NttError::LengthMismatch)
        );
    }
}
