//! The bit-dropped opening: the commitment's base-limb coefficients are sent with their low
//! `dropped_bits` bits dropped (rounded), the other limbs as Garner mixed-radix digits, and the
//! verifier recomputes the folded commitment approximately and bounds the residual norm.
//! Port of `labinius` `bd/`.

use crate::challenge::ShortChallenge;
use crate::key::CommitmentKey;
use crate::params::{inv_mod, N, QUAD_CLASS_SLOT, QUAD_POW3_CLASS};
use crate::ring::{PowerOfThreeRing, N162, SLOT_648};

pub const BD_CAP: f64 = 4.0;

/// The dropped commitment: `top[i]` is the rounded high part of the base-limb coefficient i,
/// `digits[k-1][i]` the k-th Garner mixed-radix digit.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Dropped {
    pub primes: Vec<u16>,
    pub columns: usize,
    pub dropped_bits: u32,
    pub top: Vec<u16>,
    pub digits: Vec<Vec<u16>>,
}

impl Dropped {
    pub fn wire_bytes(&self) -> usize {
        bytes(&self.primes, self.columns, self.dropped_bits)
    }
}

pub const fn top_bound(q: u16, dropped_bits: u32) -> u32 {
    ((q as u32 - 1) + (1 << (dropped_bits - 1))) >> dropped_bits
}

pub const fn top_bits(q: u16, dropped_bits: u32) -> u32 {
    u32::BITS - top_bound(q, dropped_bits).leading_zeros()
}

pub fn coefficient_bits(primes: &[u16], dropped_bits: u32) -> u32 {
    top_bits(primes[0], dropped_bits) + primes[1..].iter().map(|&q| residue_bits(q)).sum::<u32>()
}

/// Bits one residue modulo q occupies: `ceil(log2 q)`.
pub const fn residue_bits(q: u16) -> u32 {
    u32::BITS - (q as u32 - 1).leading_zeros()
}

pub fn bytes(primes: &[u16], columns: usize, dropped_bits: u32) -> usize {
    (columns * N * coefficient_bits(primes, dropped_bits) as usize).div_ceil(8)
}

/// The cap on the residual squared norm at this shape.
pub fn cap(columns: usize, dropped_bits: u32, weight: usize) -> u64 {
    let spread = ((1u64 << (2 * dropped_bits)) - 1) as f64 / 12.0;
    (BD_CAP * (N * columns * weight) as f64 * spread).ceil() as u64
}

/// Rebuild the 648-row coefficients of one commitment column from its four `R_162` components
/// (splitting limb): the inverse of the length-4 DFT with the twist, i.e. the recombination
/// `E_t = sum_k psi^{v_s k} i^{t k} Y_k(v_s)` at slot `SLOT_648[t][s]`.
fn columns_split<const Q: u16>(column: &[PowerOfThreeRing; 4]) -> [i16; N] {
    let q = Q as u64;
    let half = (Q as i64 - 1) / 2;
    let psi = crate::params::Params::<Q>::PSI as u64;
    let i_root = crate::params::pow_mod(psi, 486, q);
    let mut out = [0i16; N];
    for s in 0..N162 {
        let v = crate::ring::POW3_SLOT_EXP[s] as u64;
        let y: [i64; 4] = core::array::from_fn(|k| column[k].v[s] as i64);
        // forward length-4 DFT of the components: E_t = sum_k (psi^v)^k i^{t k} Y_k
        let pv: [i64; 4] =
            core::array::from_fn(|k| crate::params::pow_mod(psi, (v * k as u64) % 1944, q) as i64);
        let e: [i64; 4] = core::array::from_fn(|t| {
            let mut acc = 0i64;
            for k in 0..4 {
                let itk = crate::params::pow_mod(i_root, ((t * k) % 4) as u64, q) as i64;
                acc += y[k] * (itk * pv[k] % q as i64) % q as i64;
            }
            acc.rem_euclid(q as i64)
        });
        for t in 0..4 {
            let r = e[t];
            out[SLOT_648[t][s] as usize] = if r > half {
                (r - q as i64) as i16
            } else {
                r as i16
            };
        }
    }
    out
}

/// The quadratic-limb variant: `E^pm_k = Y_k +- psi'^v Y_{k+2}` at the two leaves of class v.
fn columns_quad<const Q: u16>(column: &[PowerOfThreeRing; 4]) -> [i16; N] {
    let q = Q as u64;
    let half = (Q as i64 - 1) / 2;
    let pv: [i64; N162] = core::array::from_fn(|s| {
        crate::params::ParamsQ::<Q>::psi_pow(QUAD_POW3_CLASS[s] as u32) as i64
    });
    let mut out = [0i16; N];
    for s in 0..N162 {
        let jp = QUAD_CLASS_SLOT[0][s] as usize;
        let jm = QUAD_CLASS_SLOT[1][s] as usize;
        for k in 0..2 {
            let y0 = (column[k].v[s] as i64).rem_euclid(q as i64);
            let y2 = (pv[s] * (column[k + 2].v[s] as i64).rem_euclid(q as i64)) % q as i64;
            let plus = (y0 + y2) % q as i64;
            let minus = (y0 + q as i64 - y2) % q as i64;
            out[2 * jp + k] = if plus > half {
                (plus - q as i64) as i16
            } else {
                plus as i16
            };
            out[2 * jm + k] = if minus > half {
                (minus - q as i64) as i16
            } else {
                minus as i16
            };
        }
    }
    out
}

/// The 648-row centered coefficient vector of one commitment column for one limb: the
/// components are recombined into their 648 slots (upstream fills an NTT batch), and the
/// limb's inverse transform maps them back to coefficients.
pub fn column_coefficients(q: u16, column: &[PowerOfThreeRing; 4]) -> [i16; N] {
    let slots = match q {
        3889 => columns_split::<3889>(column),
        9721 => columns_split::<9721>(column),
        17497 => columns_split::<17497>(column),
        19441 => columns_split::<19441>(column),
        2917 => columns_quad::<2917>(column),
        4861 => columns_quad::<4861>(column),
        12637 => columns_quad::<12637>(column),
        _ => unreachable!("unknown limb prime {q}"),
    };
    let mut reduced = [0u32; N];
    for i in 0..N {
        reduced[i] = (slots[i].rem_euclid(q as i16) as i64 as u32) % q as u32;
    }
    let coeffs = if crate::params::quadratic_slots(q) {
        crate::ring::intt_quad_of(q, &reduced)
    } else {
        crate::ring::intt_of(q, &reduced)
    };
    let half = (q as i32 - 1) / 2;
    let mut out = [0i16; N];
    for i in 0..N {
        out[i] = if coeffs[i] as i32 > half {
            coeffs[i] as i32 - q as i32
        } else {
            coeffs[i] as i32
        } as i16;
    }
    out
}

/// Drop the low bits of the commitment matrix: `top` from the base limb (with rounding), the
/// Garner digits of the others.
pub fn drop_bits(
    matrix: &crate::key::CommitmentMatrix,
    primes: &[u16],
    dropped_bits: u32,
) -> Dropped {
    let columns = matrix.cols();
    let count = columns * N;
    let limbs = primes.len();
    let inverses: Vec<Vec<u32>> = (0..limbs)
        .map(|k| {
            (0..k)
                .map(|m| inv_mod(primes[m] as u64, primes[k] as u64) as u32)
                .collect()
        })
        .collect();
    let mut top = vec![0u16; count];
    let mut digits: Vec<Vec<u16>> = (1..limbs).map(|_| vec![0u16; count]).collect();
    let round = 1u32 << (dropped_bits - 1);
    let mut mixed = vec![0u32; limbs];
    let mut column = [PowerOfThreeRing::zero(); 4];
    for c in 0..columns {
        for row in 0..4 {
            column[row] = *matrix.element(row, c, 0);
        }
        let base_coeffs = column_coefficients(primes[0], &column);
        for i in 0..N {
            let idx = c * N + i;
            let base = (base_coeffs[i] as i32).rem_euclid(primes[0] as i32) as u32;
            top[idx] = ((base + round) >> dropped_bits) as u16;
            mixed[0] = base;
            for k in 1..limbs {
                let q = primes[k] as u64;
                for row in 0..4 {
                    column[row] = *matrix.element(row, c, k);
                }
                let coeffs = column_coefficients(primes[k], &column);
                let mut x = (coeffs[i] as i32).rem_euclid(primes[k] as i32) as u64;
                for m in 0..k {
                    x = (x + q - mixed[m] as u64 % q) % q * inverses[k][m] as u64 % q;
                }
                mixed[k] = x as u32;
                digits[k - 1][idx] = x as u16;
            }
        }
    }
    Dropped {
        primes: primes.to_vec(),
        columns,
        dropped_bits,
        top,
        digits,
    }
}

/// Reconstruct the limb-`k` residue of coefficient i from the dropped form:
/// `X ~= (top << dropped_bits) + q0 * (digits[0] + q1 * digits[1] + ...)`, taken mod `q_k`,
/// centered. The only approximation error is the dropped low bits of the base limb.
fn combine_limb(
    primes: &[u16],
    top: &[u16],
    digits: &[Vec<u16>],
    dropped_bits: u32,
    k: usize,
    i: usize,
) -> i64 {
    let qk = primes[k] as i64;
    let last = digits.len();
    let mut inner: i128 = 0;
    if last > 0 {
        // inner = digits[last-1]; then fold m from last-2 down: inner = digits[m] + primes[m+1]*inner
        inner = digits[last - 1][i] as i128;
        for m in (0..last - 1).rev() {
            inner = digits[m][i] as i128 + primes[m + 1] as i128 * inner;
        }
    }
    let t = (top[i] as i128) << dropped_bits;
    let r = (t + primes[0] as i128 * inner).rem_euclid(qk as i128);
    let r = r as i64;
    if r > (qk - 1) / 2 {
        r - qk
    } else {
        r
    }
}

/// Garner over the limb primes with the full modulus.
pub struct Garner {
    primes: Vec<u64>,
    modulus: u64,
}

impl Garner {
    pub fn of(primes: &[u16]) -> Garner {
        Garner {
            primes: primes.iter().map(|&q| q as u64).collect(),
            modulus: primes.iter().map(|&q| q as u64).product(),
        }
    }

    /// Centered integer value of one coefficient across limbs from centered residues.
    pub fn value(&self, centred: &[i64]) -> i64 {
        let limbs = self.primes.len();
        let mut mixed = vec![0u64; limbs];
        for k in 0..limbs {
            let q = self.primes[k];
            let mut x = (centred[k].rem_euclid(q as i64) as u64) % q;
            for m in 0..k {
                // x = (x - mixed[m]) * q_m^{-1} mod q_k
                let inv = inv_mod(self.primes[m], q);
                x = (x + q - mixed[m] % q) % q * inv % q;
            }
            mixed[k] = x;
        }
        let mut value: u64 = 0;
        for k in (0..limbs).rev() {
            value = value.wrapping_mul(self.primes[k]).wrapping_add(mixed[k]);
        }
        let m = self.modulus;
        if value > m / 2 {
            value.wrapping_sub(m) as i64
        } else {
            value as i64
        }
    }
}

/// The residual check: reconstruct the dropped columns per limb, fold them against the real
/// challenges, recompute `A v` for the *actual* folded witness, and Garner-combine the per-limb
/// difference into an integer vector whose squared norm is bounded by `cap`.
pub fn residual(
    key: &CommitmentKey,
    dropped: &Dropped,
    challenges: &[ShortChallenge],
    folded: &[[i16; N]],
) -> Option<u128> {
    let limbs = key.limbs();
    let matched = dropped.primes.len() == limbs
        && (0..limbs).all(|k| dropped.primes[k] == key.prime(k))
        && dropped.top.len() == dropped.columns * N
        && dropped
            .digits
            .iter()
            .all(|d| d.len() == dropped.columns * N);
    if !matched || dropped.columns != challenges.len() {
        return None;
    }
    // fold the reconstructed columns against the challenges, per limb, in NTT domain
    let mut centred = vec![[0i64; N]; limbs];
    let q0 = key.prime(0);
    for k in 0..limbs {
        let q = key.prime(k);
        // folded commitment approximation: NTT-domain sum_j NTT(c_j) * NTT(C~_j)
        let mut acc = [0u64; N];
        for (j, c) in challenges.iter().enumerate() {
            let ch = crate::fold::challenge_slots(q, c);
            // reconstruct column j's coefficients mod q
            let mut col = [0u32; N];
            for i in 0..N {
                let v = combine_limb(
                    &dropped.primes,
                    &dropped.top,
                    &dropped.digits,
                    dropped.dropped_bits,
                    k,
                    j * N + i,
                );
                col[i] = ((v.rem_euclid(q as i64) as u64) % q as u64) as u32;
            }
            let t = crate::ring::ntt_of(q, &col);
            // the challenge is a subring element: its leaf image is a scalar written into both
            // rows, so the fold is row-wise pointwise for either tree (as in fold_witness)
            for u in 0..N {
                acc[u] += (ch[u].rem_euclid(q as i16) as i64 as u64) * t[u] as u64;
            }
        }
        // actual A v
        let av = crate::fold::a_times_v_of(q, &key.a[k], folded);
        // difference per slot, back to coefficients
        let mut diff = [0u32; N];
        for u in 0..N {
            diff[u] = ((av[u] as u64 + q as u64 - acc[u] % q as u64) % q as u64) as u32;
        }
        let coeffs = if crate::params::quadratic_slots(q) {
            crate::ring::intt_quad_of(q, &diff)
        } else {
            crate::ring::intt_of(q, &diff)
        };
        let half = (q as i64 - 1) / 2;
        for i in 0..N {
            let r = coeffs[i] as i64;
            centred[k][i] = if r > half { r - q as i64 } else { r };
        }
    }
    let garner = Garner::of(&(0..limbs).map(|k| key.prime(k)).collect::<Vec<u16>>());
    let mut normsq = 0u128;
    for i in 0..N {
        let v = garner.value(&centred.iter().map(|c| c[i]).collect::<Vec<i64>>());
        normsq += (v as i128 * v as i128) as u128;
    }
    let _ = q0;
    Some(normsq)
}

/// Debug: reconstruct limb k of coefficient i from a dropped commitment (tests).
pub fn debug_combine(dropped: &Dropped, k: usize, i: usize) -> i64 {
    combine_limb(
        &dropped.primes,
        &dropped.top,
        &dropped.digits,
        dropped.dropped_bits,
        k,
        i,
    )
}

/// Debug: the 648-row coefficient vector of one column (tests).
pub fn debug_column_coefficients(q: u16, column: &[PowerOfThreeRing; 4]) -> [i16; N] {
    column_coefficients(q, column)
}
