//! The two rings and the decomposition between them: `R_648 = Z_q[X]/(X^648 - X^324 + 1)` and
//! its height-4 view over `R_162 = Z_q[Z]/Phi_243(Z)`, `Phi_243(Z) = Z^162 + Z^81 + 1`.
//!
//! Port of `labinius` `ring/mod.rs` + `ring/element.rs`, scalar path only. `PowerOfThreeRing`
//! is one element of `R_162` in its 162-slot NTT domain (slot s holds the evaluation at
//! `theta^{v_s}`, `theta = psi^4` a primitive 486-th root); the public commitment is the
//! `4 x r` matrix of such elements, one per `(basis component, column)` per modulus.
//!
//! `decompose`: with `Y = X^4`, `y = y_0 + X y_1 + X^2 y_2 + X^3 y_3` and
//! `E_t = y(psi^{v + 486 t})`, `Y_k(v) = 4^-1 psi^{-vk} sum_t i^{-tk} E_t` with `i = psi^486`;
//! the quadratic-slot variant is a 2-point butterfly per class.

use crate::params::*;
use crate::scalar::Coeffs;

/// Degree of the small ring.
pub const N162: usize = 162;
/// Order of `theta = psi^4`.
pub const CONDUCTOR162: u32 = CONDUCTOR / 4;

/// The seven moduli, ascending. Suffixes: `_FS` fully splitting, `_Q` quadratic-slot,
/// `_S`/`_L` below/above `2^14`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum Modulus {
    Q2917_Q_S,
    Q3889_FS_S,
    Q4861_Q_S,
    Q9721_FS_S,
    Q12637_Q_S,
    Q17497_FS_L,
    Q19441_FS_L,
}

impl Modulus {
    pub const ALL: [Modulus; 7] = [
        Modulus::Q2917_Q_S,
        Modulus::Q3889_FS_S,
        Modulus::Q4861_Q_S,
        Modulus::Q9721_FS_S,
        Modulus::Q12637_Q_S,
        Modulus::Q17497_FS_L,
        Modulus::Q19441_FS_L,
    ];
    /// The default base limb.
    pub const BASE: Modulus = Modulus::Q3889_FS_S;

    pub const fn prime(self) -> u16 {
        match self {
            Modulus::Q2917_Q_S => QS_QUAD[0],
            Modulus::Q3889_FS_S => QS[0],
            Modulus::Q4861_Q_S => QS_QUAD[1],
            Modulus::Q9721_FS_S => QS[1],
            Modulus::Q12637_Q_S => QS_QUAD[2],
            Modulus::Q17497_FS_L => QS_LARGE[0],
            Modulus::Q19441_FS_L => QS_LARGE[1],
        }
    }

    pub const fn is_quadratic(self) -> bool {
        quadratic_slots(self.prime())
    }

    pub fn from_prime(q: u16) -> Option<Modulus> {
        Modulus::ALL.into_iter().find(|l| l.prime() == q)
    }
}

// =============================================================================================
// slot order of R_162
// =============================================================================================

const fn pow3_tables() -> ([u16; N162], [[u16; N162]; 4]) {
    let mut v_of = [0u16; N162];
    let mut idx = [[0u16; N162]; 4];
    let mut pos = [u16::MAX; CONDUCTOR162 as usize];
    let mut n = 0usize;
    let mut j = 0usize;
    while j < N {
        let u = SLOT_EXP[j] as usize;
        let v = u % CONDUCTOR162 as usize;
        let t = (u - v) / CONDUCTOR162 as usize;
        if pos[v] == u16::MAX {
            pos[v] = n as u16;
            v_of[n] = v as u16;
            n += 1;
        }
        idx[t][pos[v] as usize] = j as u16;
        j += 1;
    }
    assert!(n == N162);
    (v_of, idx)
}

const POW3: ([u16; N162], [[u16; N162]; 4]) = pow3_tables();

/// `POW3_SLOT_EXP[s] = v_s`: slot s of `R_162` holds the value at `theta^{v_s}`. The order is
/// the order in which the 162 classes `v = u mod 486` first appear in `SLOT_EXP`.
pub const POW3_SLOT_EXP: [u16; N162] = POW3.0;

/// `SLOT_648[t][s]` = the slot of `R_648` holding `y(psi^{v_s + 486 t})`.
pub const SLOT_648: [[u16; N162]; 4] = POW3.1;

const _: () = {
    // both trees name the 162 classes in the same order
    let mut s = 0;
    while s < N162 {
        assert!(QUAD_POW3_CLASS[s] == POW3_SLOT_EXP[s]);
        s += 1;
    }
};

// =============================================================================================
// R_162 slot elements
// =============================================================================================

/// One element of `R_162` in its 162-slot NTT domain, centered i16 residues.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PowerOfThreeRing {
    pub v: [i16; N162],
}

impl Default for PowerOfThreeRing {
    fn default() -> Self {
        Self { v: [0; N162] }
    }
}

impl PowerOfThreeRing {
    pub fn zero() -> Self {
        Self { v: [0; N162] }
    }

    /// The `R_162` NTT of a signed coefficient vector: slot s holds `c(theta^{v_s})`.
    pub fn from_coeffs<const Q: u16>(c: &[i64; N162]) -> Self {
        let q = Q as u64;
        let theta = pow_mod(Params::<Q>::PSI as u64, 4, q);
        let mut out = Self::zero();
        for s in 0..N162 {
            let v = POW3_SLOT_EXP[s] as u64;
            let x = pow_mod(theta, v, q);
            let mut acc = 0u64;
            for k in (0..N162).rev() {
                acc = (acc * x + (c[k].rem_euclid(q as i64) as u64)) % q;
            }
            out.v[s] = center(acc, q);
        }
        out
    }

    /// Slot-wise product of two slot-domain elements (their coefficient product in `R_162`).
    pub fn mul_slots(&self, o: &Self, q: u16) -> Self {
        let q = q as u64;
        let mut out = Self::zero();
        for s in 0..N162 {
            let a = (self.v[s] as i64).rem_euclid(q as i64) as u64;
            let b = (o.v[s] as i64).rem_euclid(q as i64) as u64;
            out.v[s] = center(a * b % q, q);
        }
        out
    }
}

// =============================================================================================
// the decomposition 648 -> 4 x 162
// =============================================================================================

const fn pow3_consts<const Q: u16>() -> ([u16; CONDUCTOR as usize], [[u16; N162]; 4]) {
    let mut psi_pow = [0u16; CONDUCTOR as usize];
    let mut e = 0usize;
    while e < CONDUCTOR as usize {
        psi_pow[e] = pow_mod(Params::<Q>::PSI as u64, e as u64, Q as u64) as u16;
        e += 1;
    }
    let inv4 = inv_mod(4, Q as u64);
    let mut tw = [[0u16; N162]; 4];
    let mut s = 0usize;
    while s < N162 {
        let v = POW3_SLOT_EXP[s] as usize;
        let mut k = 0usize;
        while k < 4 {
            let e = (CONDUCTOR as usize - v * k % CONDUCTOR as usize) % CONDUCTOR as usize;
            tw[k][s] = (inv4 * psi_pow[e] as u64 % Q as u64) as u16;
            k += 1;
        }
        s += 1;
    }
    (psi_pow, tw)
}

/// The four `R_162` components of a splitting-limb ring element, from its 648-slot transform:
/// `Y_k(v_s) = 4^-1 psi^{-v_s k} sum_t i^{-tk} y(psi^{v_s + 486 t})`, centered.
///
/// Every reduction runs through `barrett_mod_u64` (one u64 mulhi + conditional subtract) —
/// the `%`-per-slot i64 divisions this used to run (6 per slot) cost ~30 cycles each and this
/// decomposition runs once per (limb, chunk) of every commitment.
pub fn decompose_components<const Q: u16>(y: &Coeffs) -> [PowerOfThreeRing; 4] {
    let q = Q as u64;
    let (psi_pow, tw) = pow3_consts::<Q>();
    let i_root = psi_pow[CONDUCTOR162 as usize] as i64; // psi^486, primitive 4th root
    let mut out = [PowerOfThreeRing::zero(); 4];
    for s in 0..N162 {
        let e: [i64; 4] = core::array::from_fn(|t| {
            let j = SLOT_648[t][s] as usize;
            let v = y[j] as u64;
            // the raw commitment is in [0, q); the conditional subtract guards the invariant
            if v >= q {
                (v % q) as i64
            } else {
                v as i64
            }
        });
        // length-4 inverse DFT with i^2 = -1 (all intermediates non-negative, < 4 q)
        let a = e[0] + e[2];
        let d0 = e[0] + q as i64 - e[2];
        let c = e[1] + e[3];
        let d1 = e[1] + q as i64 - e[3];
        // d1 < 2q and i_root < q: reduce d1 first (Barrett's valid range is v < q^2 + q,
        // and 2q * q exceeds it), then the product of two sub-q values
        let id = crate::params::barrett_mod_u64(
            crate::params::barrett_mod_u64(d1 as u64, Q) * (i_root as u64),
            Q,
        ) as i64;
        let m = [a + c, d0 + q as i64 - id, a + 2 * q as i64 - c, d0 + id];
        for k in 0..4 {
            let t = tw[k][s] as u64;
            // m[k] < 4 q (non-negative), t < q: two Barrett passes replace the two `%`s
            let r = crate::params::barrett_mod_u64(
                crate::params::barrett_mod_u64(m[k] as u64, Q) * t,
                Q,
            ) as i64;
            out[k].v[s] = if r > (q as i64 - 1) / 2 {
                (r - q as i64) as i16
            } else {
                r as i16
            };
        }
    }
    out
}

const fn quad_consts<const Q: u16>() -> [u16; N162] {
    let inv2 = inv_mod(2, Q as u64);
    let mut t = [0u16; N162];
    let mut s = 0usize;
    while s < N162 {
        let v = QUAD_POW3_CLASS[s] as u32;
        let e = (CONDUCTOR_QUAD - v) % CONDUCTOR_QUAD;
        t[s] = (inv2 * ParamsQ::<Q>::psi_pow(e) as u64 % Q as u64) as u16;
        s += 1;
    }
    t
}

/// The four components for a quadratic-slot limb: one 2-point butterfly per class over the two
/// leaves that share it — `Y_0 = (E^+ + E^-)/2`, `Y_2 = (E^+ - E^-)/(2 psi'^v)` and likewise
/// for the X rows — in `POW3_SLOT_EXP` order, centered.
///
/// Barrett-reduced throughout (`%`-per-slot i64 divisions replaced; this runs once per
/// (limb, chunk) of every commitment).
pub fn decompose_components_quad<const Q: u16>(y: &Coeffs) -> [PowerOfThreeRing; 4] {
    let q = Q as u64;
    let half = (Q as i64 - 1) / 2;
    let tw = quad_consts::<Q>();
    let inv2 = inv_mod(2, q) as u64;
    // center a Barrett result already in [0, q)
    let ctr = |r: i64| -> i16 {
        if r > half {
            (r - q as i64) as i16
        } else {
            r as i16
        }
    };
    let mut out = [PowerOfThreeRing::zero(); 4];
    for s in 0..N162 {
        let jp = QUAD_CLASS_SLOT[0][s] as usize;
        let jm = QUAD_CLASS_SLOT[1][s] as usize;
        let tws = tw[s] as u64;
        for k in 0..2 {
            // the raw rows are in [0, q); the conditional keeps a defensive reduction without
            // paying a division per slot on the honest path
            let red = |x: u32| -> u64 {
                let v = x as u64;
                if v >= q {
                    v % q
                } else {
                    v
                }
            };
            let ep = red(y[2 * jp + k]);
            let em = red(y[2 * jm + k]);
            // reduce the 2q-range sums first (Barrett's valid range is v < q^2 + q), then the
            // sub-q products
            out[k].v[s] = ctr(
                crate::params::barrett_mod_u64(
                    crate::params::barrett_mod_u64(ep + em, Q) * inv2,
                    Q,
                ) as i64,
            );
            out[k + 2].v[s] = ctr(
                crate::params::barrett_mod_u64(
                    crate::params::barrett_mod_u64(ep + q - em, Q) * tws,
                    Q,
                ) as i64,
            );
        }
    }
    out
}

/// The four components of a commitment for any limb, split or quadratic.
pub fn components_of(q: u16, y: &Coeffs) -> [PowerOfThreeRing; 4] {
    match q {
        3889 => decompose_components::<3889>(y),
        9721 => decompose_components::<9721>(y),
        17497 => decompose_components::<17497>(y),
        19441 => decompose_components::<19441>(y),
        2917 => decompose_components_quad::<2917>(y),
        4861 => decompose_components_quad::<4861>(y),
        12637 => decompose_components_quad::<12637>(y),
        _ => unreachable!("unknown limb prime {q}"),
    }
}

/// Dispatch helper: forward NTT of fully-reduced coefficients for any supported prime.
pub fn ntt_of(q: u16, a: &Coeffs) -> Coeffs {
    match q {
        3889 => crate::scalar::ntt::<3889>(a),
        9721 => crate::scalar::ntt::<9721>(a),
        17497 => crate::scalar::ntt::<17497>(a),
        19441 => crate::scalar::ntt::<19441>(a),
        2917 => crate::scalar::ntt_quad::<2917>(a),
        4861 => crate::scalar::ntt_quad::<4861>(a),
        12637 => crate::scalar::ntt_quad::<12637>(a),
        _ => unreachable!("unknown limb prime {q}"),
    }
}

/// Dispatch helper: exact inverse transform for any supported prime (splitting tree only used
/// for splitting primes; quadratic primes use the quad inverse = recombination through
/// `decompose` is NOT an inverse — callers on the quad tree reconstruct via slot products).
pub fn intt_of(q: u16, a: &Coeffs) -> Coeffs {
    match q {
        3889 => crate::scalar::intt::<3889>(a),
        9721 => crate::scalar::intt::<9721>(a),
        17497 => crate::scalar::intt::<17497>(a),
        19441 => crate::scalar::intt::<19441>(a),
        _ => unreachable!("no inverse transform for quadratic prime {q} in this port"),
    }
}

/// Inverse of the quad transform: rebuilds coefficients from the 324 leaf pairs. The quad tree
/// butterflies are invertible (they never divide), so the inverse walks them backwards.
pub fn intt_quad_of(q: u16, a: &Coeffs) -> Coeffs {
    match q {
        2917 => intt_quad_impl::<2917>(a),
        4861 => intt_quad_impl::<4861>(a),
        12637 => intt_quad_impl::<12637>(a),
        _ => unreachable!("not a quadratic prime: {q}"),
    }
}

fn intt_quad_impl<const Q: u16>(v: &Coeffs) -> Coeffs {
    let q = Q as u64;
    let psi = ParamsQ::<Q>::PSI972 as u64;
    let w = ParamsQ::<Q>::OMEGA as u64;
    let w2 = w * w % q;
    let inv2 = inv_mod(2, q);
    let inv3 = inv_mod(3, q);
    let mut u = [0u64; N];
    for i in 0..N {
        u[i] = v[i] as u64 % q;
    }
    for level in (1..=5).rev() {
        let n = DEGREE_Q[level];
        let p = RADIX_Q[level];
        let m = n / p;
        for k in 0..SUBRINGS_Q[level] {
            let base = k * n;
            let zi = inv_mod(pow_mod(psi, twiddle_exp_quad(level, k) as u64, q), q);
            if p == 2 {
                for i in 0..m {
                    let (y0, y1) = (u[base + i], u[base + m + i]);
                    u[base + i] = (y0 + y1) % q * inv2 % q;
                    u[base + m + i] = (y0 + q - y1) % q * inv2 % q * zi % q;
                }
            } else {
                let zi2 = zi * zi % q;
                for i in 0..m {
                    let (y0, y1, y2) = (u[base + i], u[base + m + i], u[base + 2 * m + i]);
                    let a0 = (y0 + y1 + y2) % q * inv3 % q;
                    let t1 = (y0 + w2 * y1 + w * y2) % q * inv3 % q;
                    let t2 = (y0 + w * y1 + w2 * y2) % q * inv3 % q;
                    u[base + i] = a0;
                    u[base + m + i] = t1 * zi % q;
                    u[base + 2 * m + i] = t2 * zi2 % q;
                }
            }
        }
    }
    let z6 = ParamsQ::<Q>::ZETA6 as u64;
    let det = inv_mod((2 * z6 + q - 1) % q, q);
    for i in 0..324 {
        let (y0, y1) = (u[i], u[i + 324]);
        let a1 = (y0 + q - y1) % q * det % q;
        let a0 = (y0 + q - z6 * a1 % q) % q;
        u[i] = a0;
        u[i + 324] = a1;
    }
    let mut out = [0u32; N];
    for i in 0..N {
        out[i] = u[i] as u32;
    }
    out
}
