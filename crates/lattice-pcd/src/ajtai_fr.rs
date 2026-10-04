//! The PQ commitment route (the zk-pcd-accumulation deviation ledger's
//! item #6): an **Ajtai commitment over the BN254 scalar field** — the
//! d = 1 module `F_r^n` with a seeded uniform matrix
//! `A ∈ F_r^{rows × cols}` — plus the digit-layer regime that keeps
//! openings short, the hiding variant (the appended uniform-pad columns),
//! and the Module-SIS kernel utilities.
//!
//! The accumulation's algebra (the field-scalar E-fold, the eq-weighted
//! combinations) is preserved verbatim — `F_r`-scalars act on the module
//! coordinate-wise — while the binding root changes from Pedersen-DL to
//! **MSIS at the digit radius**: two short openings of one commitment
//! yield a short kernel `A·(w − w') = 0`, the lattice-soundness
//! counterpart of the Pedersen binding argument the paper's decider and
//! extractor rely on.
//!
//! ## The digit regime
//!
//! Arbitrary `F_r` vectors commit through 16-bit unsigned digit layers
//! (`f = Σ_j 2^{16j} f^{(j)}` with `‖f^{(j)}‖∞ ≤ 2^16 − 1`, 16 layers
//! covering 254 bits): the committed vector is SHORT at radius
//! `β = 2^16 − 1`. The MSIS dimension rule at `q ≈ 2^254`:
//! `λ₁ ≈ q^{(cols−rows)/cols} ≥ β` needs `rows/cols ≤ 1 − 16/254 ≈ 0.937`
//! — satisfied with wide margin at the demo shapes (e.g. rows = 8,
//! cols = 256 → `λ₁ ≈ 2^{226}`).

use crate::fp_base::FpBase;
use lattice_core::transcript::Transcript;

/// The 16-bit digit-layer count covering a canonical `F_r` value.
pub const FR_DIGIT_LAYERS: usize = 16;
/// The digit radius (the shortness bound of committed openings).
pub const FR_DIGIT_BOUND: u64 = (1u64 << 16) - 1;

/// A seeded Ajtai matrix over `F_r` (the d = 1 module).
#[derive(Clone)]
pub struct AjtaiFrKey {
    pub rows: usize,
    pub cols: usize,
    /// Row-major `rows × cols` field elements, uniform from the seed.
    pub matrix: Vec<FpBase>,
}

impl AjtaiFrKey {
    /// Derive the key from a seed.
    pub fn from_seed(rows: usize, cols: usize, seed: &[u8]) -> Self {
        let mut matrix = Vec::with_capacity(rows * cols);
        let mut counter = 0u32;
        while matrix.len() < rows * cols {
            let salt = {
                let mut s = seed.to_vec();
                s.extend_from_slice(&counter.to_le_bytes());
                s
            };
            let bytes = Transcript::xof(b"ajtai-fr", &salt, 64);
            // 32 bytes ≈ 2^256 — reject ≥ p (bias < 2^-126).
            for chunk in bytes.chunks(32) {
                if matrix.len() == rows * cols {
                    break;
                }
                let mut arr = [0u8; 32];
                arr.copy_from_slice(chunk);
                let cand = FpBase::from_be_bytes_wide(&arr);
                let canonical = cand.to_be_bytes();
                // Reject all-0xFF prefix forms above p: the widest byte
                // must not be 0xFF for a canonical value (< 2^254).
                if canonical[0] != 0xFF {
                    matrix.push(cand);
                }
            }
            counter += 1;
        }
        AjtaiFrKey { rows, cols, matrix }
    }

    pub fn entry(&self, r: usize, c: usize) -> &FpBase {
        &self.matrix[r * self.cols + c]
    }

    /// Commit a short scalar vector `w` (length = cols):
    /// `cm = A·w` — `rows` field elements.
    pub fn commit(&self, w: &[FpBase]) -> Result<Vec<FpBase>, String> {
        if w.len() != self.cols {
            return Err(format!("witness {} vs cols {}", w.len(), self.cols));
        }
        let mut cm = vec![FpBase::ZERO; self.rows];
        for (c, &wc) in w.iter().enumerate() {
            if wc.is_zero() {
                continue;
            }
            for r in 0..self.rows {
                cm[r] = cm[r].add(&self.entry(r, c).mul(&wc));
            }
        }
        Ok(cm)
    }

    /// Check `A·w = cm`.
    pub fn verify_opening(&self, w: &[FpBase], cm: &[FpBase]) -> bool {
        self.commit(w).map(|c| c == cm).unwrap_or(false)
    }

    /// Check a mod-`r` kernel relation with a nonzero coefficient vector.
    pub fn is_kernel(&self, k: &[FpBase]) -> bool {
        if k.len() != self.cols || k.iter().all(|x| x.is_zero()) {
            return false;
        }
        let mut acc = vec![FpBase::ZERO; self.rows];
        for (c, &kc) in k.iter().enumerate() {
            if kc.is_zero() {
                continue;
            }
            for r in 0..self.rows {
                acc[r] = acc[r].add(&self.entry(r, c).mul(&kc));
            }
        }
        acc.iter().all(|a| a.is_zero())
    }
}

/// The hiding variant: `Com(w) = A·w + A_pad·ρ` with `ρ` uniform over
/// `pad_cols` extra columns — the smoothing-style one-time pad (the
/// hiding argument is heuristic at this profile — the deviation ledger
/// records it; binding is inherited from MSIS on the extended matrix
/// `[A | A_pad]` for the *combined* coefficient vector).
#[derive(Clone)]
pub struct HidingAjtaiKey {
    pub base: AjtaiFrKey,
    pub pad: AjtaiFrKey,
}

impl HidingAjtaiKey {
    pub fn from_seed(rows: usize, cols: usize, pad_cols: usize, seed: &[u8]) -> Self {
        HidingAjtaiKey {
            base: AjtaiFrKey::from_seed(rows, cols, seed),
            pad: AjtaiFrKey::from_seed(rows, pad_cols, &[seed, b"-pad"].concat()),
        }
    }

    /// Commit with fresh uniform blinding drawn from the transcript.
    pub fn commit_hiding(
        &self,
        w: &[FpBase],
        transcript: &mut Transcript,
    ) -> Result<(Vec<FpBase>, Vec<FpBase>), String> {
        let rho = sample_uniform_vec(self.pad.cols, transcript)?;
        let cw = self.base.commit(w)?;
        let cp = self.pad.commit(&rho)?;
        let cm: Vec<FpBase> = cw.iter().zip(cp.iter()).map(|(a, b)| a.add(b)).collect();
        Ok((cm, rho))
    }

    /// Verify: `cm = A·w + A_pad·ρ`.
    pub fn verify_hiding(&self, w: &[FpBase], rho: &[FpBase], cm: &[FpBase]) -> bool {
        if rho.len() != self.pad.cols {
            return false;
        }
        match (self.base.commit(w), self.pad.commit(rho)) {
            (Ok(a), Ok(b)) => {
                cm == a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.add(y))
                    .collect::<Vec<_>>()
            }
            _ => false,
        }
    }
}

/// Decompose a field vector into 16-bit digit layers (the shortness
/// regime): `layers[j][i] = (f_i >> 16j) & 0xFFFF`, with the top layer
/// masked to the canonical range.
pub fn digit_layers(f: &[FpBase]) -> Vec<Vec<FpBase>> {
    let n = f.len();
    let mut layers = vec![vec![FpBase::ZERO; n]; FR_DIGIT_LAYERS];
    for (i, v) in f.iter().enumerate() {
        let bytes = v.to_be_bytes();
        // bytes[2..32] carry the canonical 254-bit value (big-endian).
        for j in 0..FR_DIGIT_LAYERS {
            // digit j = bits [16j, 16j+16) — layer 0 is the LOW 16 bits.
            let bit_lo = 16 * j;
            let mut digit = 0u64;
            for b in 0..16 {
                let bit = bit_lo + b;
                if bit >= 254 {
                    break;
                }
                // bit position from the LSB of the 254-bit value: byte
                // index 31 - bit/8, mask 1 << (bit%8).
                let byte_idx = 31 - bit / 8;
                if (bytes[byte_idx] >> (bit % 8)) & 1 == 1 {
                    digit |= 1 << b;
                }
            }
            layers[j][i] = FpBase::from_canonical_u64(digit);
        }
    }
    layers
}

/// Recompose: `f = Σ_j 2^{16j}·layers[j]` (mod r) — the powers
/// `2^{16j}` for `j ≥ 16` exceed u64, so the weight is built by
/// successive doublings in the field.
pub fn recompose(layers: &[Vec<FpBase>]) -> Vec<FpBase> {
    let n = layers.first().map(|l| l.len()).unwrap_or(0);
    let mut f = vec![FpBase::ZERO; n];
    let two = FpBase::from_canonical_u64(2);
    for (j, layer) in layers.iter().enumerate() {
        // weight = 2^{16j} mod r.
        let mut weight = FpBase::from_canonical_u64(1);
        for _ in 0..(16 * j) {
            weight = weight.mul(&two);
        }
        for (i, &d) in layer.iter().enumerate() {
            f[i] = f[i].add(&d.mul(&weight));
        }
    }
    f
}

/// The shortness verdict: every coordinate of every layer within the
/// digit radius (in canonical form).
pub fn layers_are_short(layers: &[Vec<FpBase>]) -> bool {
    layers
        .iter()
        .all(|l| l.iter().all(|d| digit_value(d) <= FR_DIGIT_BOUND))
}

/// The canonical u64 value of a small field element (0 when ≥ 2^63).
pub fn digit_value(d: &FpBase) -> u64 {
    let bytes = d.to_be_bytes();
    if bytes[..25].iter().any(|&b| b != 0) {
        return u64::MAX; // out of the small range — not a digit
    }
    let mut v = 0u64;
    for &b in bytes[25..32].iter() {
        v = (v << 8) | b as u64;
    }
    v
}

/// Draw a uniform field element from the transcript (32 bytes, reject
/// the ≥ 2^254 forms).
pub fn sample_uniform(transcript: &mut Transcript) -> Result<FpBase, String> {
    for _ in 0..64 {
        let bytes = transcript
            .challenge_bytes(b"ajtai-fr-chal", 32)
            .map_err(|e| e.to_string())?;
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        if arr[0] != 0xFF {
            return Ok(FpBase::from_be_bytes_wide(&arr));
        }
    }
    Err("uniform sampling exhausted".into())
}

/// Draw a uniform vector.
pub fn sample_uniform_vec(n: usize, transcript: &mut Transcript) -> Result<Vec<FpBase>, String> {
    (0..n).map(|_| sample_uniform(transcript)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_vec(n: usize, seed: u64) -> Vec<FpBase> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                FpBase::from_canonical_u64((x >> 33) & FR_DIGIT_BOUND)
            })
            .collect()
    }

    #[test]
    fn commit_linearity_and_binding_regime() {
        let key = AjtaiFrKey::from_seed(4, 32, b"k1");
        let w = small_vec(32, 1);
        let w2 = small_vec(32, 2);
        let cm = key.commit(&w).unwrap();
        let cm2 = key.commit(&w2).unwrap();
        // Linearity.
        let sum: Vec<FpBase> = w.iter().zip(w2.iter()).map(|(a, b)| a.add(b)).collect();
        let cm_sum: Vec<FpBase> = cm.iter().zip(cm2.iter()).map(|(a, b)| a.add(b)).collect();
        assert_eq!(key.commit(&sum).unwrap(), cm_sum);
        // Distinct short witnesses commit distinctly.
        assert_ne!(cm, cm2);
        // Random nonzero kernel coefficients do not vanish.
        let k = small_vec(32, 7);
        assert!(!key.is_kernel(&k));
        // Zero is not a kernel.
        assert!(!key.is_kernel(&vec![FpBase::ZERO; 32]));
    }

    #[test]
    fn digit_roundtrip_and_shortness() {
        let f: Vec<FpBase> = (0..8u64)
            .map(|i| FpBase::from_canonical_u64(i.wrapping_mul(0x9E3779B97F4A7C15) % (1u64 << 60)))
            .collect();
        let layers = digit_layers(&f);
        assert!(layers_are_short(&layers));
        assert_eq!(recompose(&layers), f);
    }

    #[test]
    fn hiding_commit_roundtrip() {
        let key = HidingAjtaiKey::from_seed(4, 16, 8, b"hide");
        let w = small_vec(16, 3);
        let mut t = Transcript::new_default(b"hide-t");
        let (cm, rho) = key.commit_hiding(&w, &mut t).unwrap();
        assert!(key.verify_hiding(&w, &rho, &cm));
        // A wrong blinding fails.
        let mut rho2 = rho.clone();
        rho2[0] = rho2[0].add(&FpBase::from_canonical_u64(1));
        assert!(!key.verify_hiding(&w, &rho2, &cm));
        // A wrong witness fails.
        let w2 = small_vec(16, 4);
        assert!(!key.verify_hiding(&w2, &rho, &cm));
    }
}
