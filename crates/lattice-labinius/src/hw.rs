//! Runtime-detected hardware acceleration, pure `std` (x86-64 intrinsics only, no external
//! dependencies, portable fallbacks everywhere).
//!
//! Two facilities:
//! * [`clmul64`] — 64x64 carry-less multiplication through `PCLMULQDQ` when present (one
//!   instruction, ~5 cycles pipelined) with the software bitwise product as the fallback. This
//!   is the primitive under every `F162`/`B128` field product, i.e. the whole binary-field
//!   front end (eq tables, row evaluations, challenge sampling, fold parities).
//! * [`avx512_pcs`] — the full AVX-512 feature set the PCS kernels in [`crate::simd`] need
//!   (F/BW/VL/VBMI/VBMI2/VNNI/GFNI), detected once and cached.
//!
//! Detection happens exactly once per process through a `OnceLock`; the kernels themselves are
//! `#[target_feature]` functions that are only called behind their gate. On non-x86-64 targets
//! every gate is `false` and the scalar reference paths run.

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::x86_64::{
        _mm_clmulepi64_si128, _mm_cvtsi128_si64, _mm_extract_epi64, _mm_set_epi64x,
    };
    use std::sync::OnceLock;

    static PCLMUL: OnceLock<bool> = OnceLock::new();
    static AVX512_PCS: OnceLock<bool> = OnceLock::new();

    /// Is `PCLMULQDQ` available?
    pub fn pclmulqdq() -> bool {
        *PCLMUL.get_or_init(|| is_x86_feature_detected!("pclmulqdq"))
    }

    /// Is the full AVX-512 PCS feature set available (F/BW/VL/VBMI/VBMI2/VNNI/GFNI)?
    pub fn avx512_pcs() -> bool {
        *AVX512_PCS.get_or_init(|| {
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512vbmi")
                && is_x86_feature_detected!("avx512vbmi2")
                && is_x86_feature_detected!("avx512vnni")
                && is_x86_feature_detected!("gfni")
        })
    }

    /// `PCLMULQDQ`-backed `a (*) b` (full 128-bit carry-less product).
    ///
    /// # Safety
    /// Caller must have checked [`pclmulqdq`].
    #[target_feature(enable = "pclmulqdq,sse4.1")]
    #[inline]
    pub unsafe fn clmul64_hw(a: u64, b: u64) -> u128 {
        let x = _mm_set_epi64x(0, a as i64);
        let y = _mm_set_epi64x(0, b as i64);
        let r = _mm_clmulepi64_si128(x, y, 0x00);
        let lo = _mm_cvtsi128_si64(r) as u64;
        let hi = _mm_extract_epi64(r, 1) as u64;
        (lo as u128) | ((hi as u128) << 64)
    }
}

#[cfg(target_arch = "x86_64")]
pub use imp::{avx512_pcs, pclmulqdq};

#[cfg(not(target_arch = "x86_64"))]
pub fn pclmulqdq() -> bool {
    false
}
#[cfg(not(target_arch = "x86_64"))]
pub fn avx512_pcs() -> bool {
    false
}

/// The software 64x64 carry-less product (bitwise schoolbook) — the portable reference.
#[inline]
pub fn clmul64_soft(a: u64, b: u64) -> u128 {
    let mut acc: u128 = 0;
    let mut b = b;
    let mut i = 0u32;
    while b != 0 {
        if b & 1 != 0 {
            acc ^= (a as u128) << i;
        }
        b >>= 1;
        i += 1;
    }
    acc
}

/// 64x64 carry-less multiplication: `PCLMULQDQ` when the CPU has it, software otherwise.
#[inline]
pub fn clmul64(a: u64, b: u64) -> u128 {
    #[cfg(target_arch = "x86_64")]
    {
        if pclmulqdq() {
            // Safety: the gate above was just checked.
            return unsafe { imp::clmul64_hw(a, b) };
        }
    }
    clmul64_soft(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clmul_agrees_with_software() {
        // A spread of operands incl. all-ones, single bits, and pseudo-random pairs.
        let cases = [
            (0u64, 0u64),
            (u64::MAX, u64::MAX),
            (1, 1),
            (1 << 63, 1),
            (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
            (0x9E37_79B9_7F4A_7C15, 0x2545_F491_4F6C_DD1D),
            (0xDEAD_BEEF_CAFE_F00D, 0x1337),
        ];
        for (a, b) in cases {
            assert_eq!(clmul64(a, b), clmul64_soft(a, b), "clmul64({a:#x},{b:#x})");
            assert_eq!(clmul64(b, a), clmul64(a, b), "clmul is commutative");
        }
        // pseudo-random sweep
        let mut x = 0x853c_49e6_748f_ea9bu64;
        let mut y = 0xda3e_39cb_94b9_5bdbu64;
        for _ in 0..64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            y = y.wrapping_mul(0x2545_F491_4F6C_DD1D);
            y = y.wrapping_add(0x9E37_79B9_7F4A_7C15);
            assert_eq!(clmul64(x, y), clmul64_soft(x, y));
        }
    }

    #[test]
    fn feature_report() {
        // Informational: the gates must be callable and stable.
        let a = pclmulqdq();
        let b = avx512_pcs();
        assert_eq!(a, pclmulqdq());
        assert_eq!(b, avx512_pcs());
    }
}
