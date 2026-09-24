//! Ring parameters for `R_q = Z_q[X] / Phi_1944(X)`, `Phi_1944(X) = X^648 - X^324 + 1`.
//!
//! Port of `labinius/crates/pcs/src/params.rs` (upstream @ main). Everything is `const`-evaluated;
//! the NTT tree is fixed so every implementation produces the same slot order:
//!
//! level 0: the Phi_6 split   `X^648 - X^324 + 1 = (X^324 - psi^324)(X^324 - psi^1620)`
//! level 1..2: radix 2        `X^324 -> X^162 -> X^81`
//! level 3..6: radix 3        `X^81 -> X^27 -> X^9 -> X^3 -> X^1`
//!
//! A sub-ring is `Z_q[X]/(X^n - psi^e)` with `n | e`; radix-p split into children
//! `X^{n/p} - psi^{(e + 1944 s)/p}`, child `s` at block offset `s*n/p`. Leaf `j` is
//! `Z_q[X]/(X - psi^SLOT_EXP[j])`, so `NTT(a)[j] = a(psi^SLOT_EXP[j])`.
//!
//! The quadratic-slot tree (primes `1 mod 972`, not `1 mod 1944`) ends in 324 leaves
//! `Z_q[X]/(X^2 - psi'^u)`; see the second half of this module.

/// Degree of the ring.
pub const N: usize = 648;
/// Conductor.
pub const CONDUCTOR: u32 = 1944;
/// The supported primes below `2^14` (fully splitting).
pub const QS: [u16; 2] = [3889, 9721];
/// Fully splitting primes above `2^14`.
pub const QS_LARGE: [u16; 2] = [17497, 19441];
/// Radix of the split turning level `l` into level `l+1`.
pub const RADIX: [usize; 7] = [2, 2, 2, 3, 3, 3, 3];
/// Number of sub-rings at level `l` (level 7 = the 648 leaves).
pub const SUBRINGS: [usize; 8] = [1, 2, 4, 8, 24, 72, 216, 648];
/// Degree of one sub-ring at level `l`.
pub const DEGREE: [usize; 8] = [648, 324, 162, 81, 27, 9, 3, 1];

pub const fn pow_mod(mut b: u64, mut e: u64, q: u64) -> u64 {
    let mut r = 1u64;
    b %= q;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % q;
        }
        b = b * b % q;
        e >>= 1;
    }
    r
}

pub const fn inv_mod(a: u64, q: u64) -> u64 {
    pow_mod(a, q - 2, q)
}

/// Smallest x in [2, q) of multiplicative order exactly 1944.
pub const fn find_psi(q: u64) -> u64 {
    let mut x = 2u64;
    loop {
        if pow_mod(x, 1944, q) == 1 && pow_mod(x, 972, q) != 1 && pow_mod(x, 648, q) != 1 {
            return x;
        }
        x += 1;
    }
}

/// Exponent e of sub-ring k at level `level` (1..=7).
pub const fn subring_exp(level: usize, k: usize) -> u32 {
    if level == 1 {
        return if k == 0 { 324 } else { 1620 };
    }
    let p = RADIX[level - 1] as u32;
    (subring_exp(level - 1, k / p as usize) + CONDUCTOR * (k as u32 % p)) / p
}

/// psi-exponent of the twiddle splitting sub-ring k of level `level`.
pub const fn twiddle_exp(level: usize, k: usize) -> u32 {
    subring_exp(level, k) / RADIX[level] as u32
}

const fn slot_exp_table() -> [u16; N] {
    let mut t = [0u16; N];
    let mut j = 0;
    while j < N {
        t[j] = subring_exp(7, j) as u16;
        j += 1;
    }
    t
}

/// `SLOT_EXP[j]` = u such that slot j holds `a(psi^u)`.
pub const SLOT_EXP: [u16; N] = slot_exp_table();

/// Centered representative in `(-(q-1)/2, (q-1)/2]` as i16 (q < 2^15 here).
pub const fn center(x: u64, q: u64) -> i16 {
    let x = x % q;
    if x > q / 2 {
        (x as i64 - q as i64) as i16
    } else {
        x as i16
    }
}

/// `q^-1 mod 2^16` for odd `q` (Newton iteration, doubling precision each round).
pub const fn qinv16(q: u16) -> u16 {
    let mut x = 1u16;
    let mut i = 0;
    while i < 4 {
        x = x.wrapping_mul(2u16.wrapping_sub(q.wrapping_mul(x)));
        i += 1;
    }
    x
}

/// `round(2^15 / q)`, the `vpmulhrsw` constant of [`barrett_i16`].
pub const fn barrett_v(q: u16) -> i16 {
    (((1u32 << 15) + (q as u32) / 2) / q as u32) as i16
}

/// Cheap partial reduction with `vpmulhrsw` semantics (2 multiply uops):
/// `t = round(a * BARRETT_V / 2^15)`, `r = a - t*q`, guarantee only `|r| < q`.
#[inline]
pub fn barrett_i16(a: i16, q: u16) -> i16 {
    let v = barrett_v(q) as i32;
    let t = (((a as i32) * v * 2 + (1 << 15)) >> 16) as i16;
    a.wrapping_sub(t.wrapping_mul(q as i16))
}

/// Barrett constant `m = floor(2^32 / q)` for a 16-bit prime.
pub const fn barrett_m32(q: u16) -> u32 {
    ((1u64 << 32) / q as u64) as u32
}

/// `v mod q` through one u64 mulhi and one conditional subtract, for `v < q^2 + q` (which every
/// Horner step `acc * x + coeff` of values below `q` satisfies with room to spare: `q < 2^15`
/// gives `v < 2^30`). Against `%` this trades a ~20-cycle division for ~6 cycles of multiply
/// and shift; exhaustively checked per prime in `tests/` (see `tests/barrett.rs`).
#[inline]
pub fn barrett_mod_u64(v: u64, q: u16) -> u64 {
    let m = barrett_m32(q) as u64;
    let t = ((v * m) >> 32) * q as u64;
    let r = v - t;
    if r >= q as u64 {
        r - q as u64
    } else {
        r
    }
}

/// Signed Montgomery multiplication on 16-bit values, exactly what the SIMD kernels do lane-wise:
/// returns `a * x mod q` in `(-q, q)` where `w = to_mont(x)`, `w_pre = mont_pre(w)`; `a` any i16.
#[inline]
pub fn mont_mul_i16(a: i16, w: i16, w_pre: i16, q: u16) -> i16 {
    let m = a.wrapping_mul(w_pre);
    let hi = ((a as i32 * w as i32) >> 16) as i16;
    let t = ((m as i32 * q as i16 as i32) >> 16) as i16;
    hi.wrapping_sub(t)
}

/// Per-prime constants: `Params::<3889>::PSI` etc.
pub struct Params<const Q: u16>;

impl<const Q: u16> Params<Q> {
    pub const Q: u16 = Q;
    /// `q^-1 mod 2^16`.
    pub const QINV: u16 = qinv16(Q);
    /// Smallest primitive 1944-th root of unity.
    pub const PSI: u16 = find_psi(Q as u64) as u16;
    /// Primitive cube root of unity, `omega = psi^648`.
    pub const OMEGA: u16 = pow_mod(Self::PSI as u64, 648, Q as u64) as u16;
    /// Primitive sixth root, `zeta6 = psi^324`; `zeta6^-1 = 1 - zeta6`.
    pub const ZETA6: u16 = pow_mod(Self::PSI as u64, 324, Q as u64) as u16;
    /// `2^16 mod q`, the Montgomery constant the twiddle tables are built in.
    pub const R: u16 = (65536u64 % Q as u64) as u16;

    pub const fn psi_pow(e: u32) -> u16 {
        pow_mod(Self::PSI as u64, e as u64, Q as u64) as u16
    }
    /// Plain twiddle zeta for sub-ring k at level `level` (1..=6).
    pub const fn zeta(level: usize, k: usize) -> u16 {
        Self::psi_pow(twiddle_exp(level, k))
    }
    /// `x * 2^16 mod q`, centered: the Montgomery form used by the SIMD kernels.
    pub const fn to_mont(x: u16) -> i16 {
        center(x as u64 * 65536u64, Q as u64)
    }
    /// `x * R mod q` (plain, not centered): the scaling applied to a kernel's tables or twiddles
    /// to make its outputs come out in Montgomery form.
    pub const fn scale_r(x: u16) -> u16 {
        (x as u64 * Self::R as u64 % Q as u64) as u16
    }
    /// For a Montgomery-form constant `w`, the precomputed `w * q^-1 mod 2^16` (signed), so that
    /// `mont(a, w, w') = a * x mod q` needs only mullo/mulhi/mulhi.
    pub const fn mont_pre(w: i16) -> i16 {
        w.wrapping_mul(Self::QINV as i16)
    }
    /// Table of plain twiddles for a whole level (`K = SUBRINGS[level]`).
    pub const fn zetas<const K: usize>(level: usize) -> [u16; K] {
        let mut t = [0u16; K];
        let mut k = 0;
        while k < K {
            t[k] = Self::zeta(level, k);
            k += 1;
        }
        t
    }
    pub const ZETA_L1: [u16; 2] = Self::zetas::<2>(1);
    pub const ZETA_L2: [u16; 4] = Self::zetas::<4>(2);
    pub const ZETA_L3: [u16; 8] = Self::zetas::<8>(3);
    pub const ZETA_L4: [u16; 24] = Self::zetas::<24>(4);
    pub const ZETA_L5: [u16; 72] = Self::zetas::<72>(5);
    pub const ZETA_L6: [u16; 216] = Self::zetas::<216>(6);
}

const _: () = {
    assert!((Params::<3889>::QINV as u32 * 3889u32) % 65536 == 1);
    assert!((Params::<9721>::QINV as u32 * 9721u32) % 65536 == 1);
    assert!((Params::<17497>::QINV as u32 * 17497u32) % 65536 == 1);
    assert!((Params::<19441>::QINV as u32 * 19441u32) % 65536 == 1);
    assert!((Params::<2917>::QINV as u32 * 2917u32) % 65536 == 1);
    assert!((Params::<4861>::QINV as u32 * 4861u32) % 65536 == 1);
    assert!((Params::<12637>::QINV as u32 * 12637u32) % 65536 == 1);
};

/// Every splitting prime is `1 mod 1944`.
const _: () = {
    let mut i = 0;
    while i < 2 {
        assert!((QS[i] as u32 - 1).is_multiple_of(CONDUCTOR) && QS[i] < 1 << 14);
        assert!((QS_LARGE[i] as u32 - 1).is_multiple_of(CONDUCTOR) && QS_LARGE[i] > 1 << 14);
        i += 1;
    }
};

// =============================================================================================
// The quadratic-slot tree: q = 1 mod 972 but not mod 1944
// =============================================================================================

/// Conductor of the quadratic tree's root of unity (a primitive 972-nd root psi').
pub const CONDUCTOR_QUAD: u32 = 972;
/// Primes for which `R_648` ends in 324 quadratic leaves.
pub const QS_QUAD: [u16; 3] = [2917, 4861, 12637];
/// Number of quadratic leaves.
pub const QUAD_SLOTS: usize = 324;

/// Does `R_648` end in quadratic leaves modulo `q`?
pub const fn quadratic_slots(q: u16) -> bool {
    (q as u32 - 1).is_multiple_of(CONDUCTOR_QUAD) && !(q as u32 - 1).is_multiple_of(CONDUCTOR)
}

/// Radix of the split turning level `l` into `l+1` in the quadratic tree.
pub const RADIX_Q: [usize; 6] = [2, 2, 3, 3, 3, 3];
/// Sub-rings at level `l` (level 6 = the 324 quadratic leaves).
pub const SUBRINGS_Q: [usize; 7] = [1, 2, 4, 12, 36, 108, 324];
/// Degree of one sub-ring at level `l` in the quadratic tree.
pub const DEGREE_Q: [usize; 7] = [648, 324, 162, 54, 18, 6, 2];

/// Smallest x in [2, q) of multiplicative order exactly 972.
pub const fn find_psi972(q: u64) -> u64 {
    let mut x = 2u64;
    loop {
        if pow_mod(x, 972, q) == 1 && pow_mod(x, 486, q) != 1 && pow_mod(x, 324, q) != 1 {
            return x;
        }
        x += 1;
    }
}

/// Exponent e of sub-ring k at level `level` (1..=6) of the quadratic tree.
pub const fn subring_exp_quad(level: usize, k: usize) -> u32 {
    if level == 1 {
        return if k == 0 { 162 } else { 810 };
    }
    let p = RADIX_Q[level - 1] as u32;
    (subring_exp_quad(level - 1, k / p as usize) + CONDUCTOR_QUAD * (k as u32 % p)) / p
}

/// psi'-exponent of the twiddle splitting sub-ring k of level `level` (1..=5).
pub const fn twiddle_exp_quad(level: usize, k: usize) -> u32 {
    subring_exp_quad(level, k) / RADIX_Q[level] as u32
}

const fn quad_slot_exp_table() -> [u16; QUAD_SLOTS] {
    let mut t = [0u16; QUAD_SLOTS];
    let mut j = 0;
    while j < QUAD_SLOTS {
        t[j] = subring_exp_quad(6, j) as u16;
        j += 1;
    }
    t
}

/// `QUAD_SLOT_EXP[j]` = u such that leaf j is `Z_q[X]/(X^2 - psi'^u)`.
pub const QUAD_SLOT_EXP: [u16; QUAD_SLOTS] = quad_slot_exp_table();

const _: () = {
    // every leaf exponent is a unit mod 972, each occurring exactly once
    let mut seen = [false; 972];
    let mut j = 0;
    while j < QUAD_SLOTS {
        let u = QUAD_SLOT_EXP[j] as usize;
        assert!(u % 2 == 1 && !u.is_multiple_of(3));
        assert!(!seen[u]);
        seen[u] = true;
        j += 1;
    }
    let mut i = 0;
    while i < 3 {
        assert!(quadratic_slots(QS_QUAD[i]));
        i += 1;
    }
};

/// The `R_162` class of leaf j and which of the class's two leaves it is.
const fn quad_class_tables() -> ([u16; 162], [[u16; 162]; 2]) {
    let mut class = [0u16; QUAD_SLOTS];
    let mut slot = [[u16::MAX; 162]; 2];
    let mut j = 0;
    while j < QUAD_SLOTS {
        let u = QUAD_SLOT_EXP[j] as usize;
        class[j] = (u % 486) as u16;
        j += 1;
    }
    let mut s = 0;
    while s < 162 {
        let v = crate::ring::POW3_SLOT_EXP[s] as usize;
        let mut j = 0;
        while j < QUAD_SLOTS {
            let u = QUAD_SLOT_EXP[j] as usize;
            if u == v {
                slot[0][s] = j as u16;
            } else if u == v + 486 {
                slot[1][s] = j as u16;
            }
            j += 1;
        }
        assert!(slot[0][s] != u16::MAX && slot[1][s] != u16::MAX);
        s += 1;
    }
    let mut cl = [0u16; 162];
    let mut s = 0;
    while s < 162 {
        cl[s] = crate::ring::POW3_SLOT_EXP[s];
        s += 1;
    }
    (cl, slot)
}

const QUAD_CLASS: ([u16; 162], [[u16; 162]; 2]) = quad_class_tables();

/// The 162 `R_162` classes in `POW3_SLOT_EXP` order.
pub const QUAD_POW3_CLASS: [u16; 162] = QUAD_CLASS.0;
/// `QUAD_CLASS_SLOT[sign][s]`: the leaf of class s with constant `+psi'^v` / `-psi'^v`.
pub const QUAD_CLASS_SLOT: [[u16; 162]; 2] = QUAD_CLASS.1;

/// Per-prime constants of the quadratic-slot tree.
pub struct ParamsQ<const Q: u16>;

impl<const Q: u16> ParamsQ<Q> {
    /// Smallest primitive 972-nd root of unity mod q.
    pub const PSI972: u16 = find_psi972(Q as u64) as u16;
    /// Primitive cube root, `omega = psi'^324`.
    pub const OMEGA: u16 = pow_mod(Self::PSI972 as u64, 324, Q as u64) as u16;
    /// Primitive sixth root, `zeta6 = psi'^162`.
    pub const ZETA6: u16 = pow_mod(Self::PSI972 as u64, 162, Q as u64) as u16;

    pub const fn psi_pow(e: u32) -> u16 {
        pow_mod(Self::PSI972 as u64, e as u64, Q as u64) as u16
    }
    /// Plain twiddle zeta for sub-ring k at level `level` (1..=5).
    pub const fn zeta(level: usize, k: usize) -> u16 {
        Self::psi_pow(twiddle_exp_quad(level, k))
    }
    /// Table of plain twiddles for a whole level of the quadratic tree.
    pub const fn zetas<const K: usize>(level: usize) -> [u16; K] {
        let mut t = [0u16; K];
        let mut k = 0;
        while k < K {
            t[k] = Self::zeta(level, k);
            k += 1;
        }
        t
    }
    pub const ZETA_L1: [u16; 2] = Self::zetas::<2>(1);
    pub const ZETA_L2: [u16; 4] = Self::zetas::<4>(2);
    pub const ZETA_L3: [u16; 12] = Self::zetas::<12>(3);
    pub const ZETA_L4: [u16; 36] = Self::zetas::<36>(4);
    pub const ZETA_L5: [u16; 108] = Self::zetas::<108>(5);
    /// `LEAF_C[j] = psi'^QUAD_SLOT_EXP[j]`, the constant of leaf j.
    pub const LEAF_C: [u16; QUAD_SLOTS] = {
        let mut t = [0u16; QUAD_SLOTS];
        let mut j = 0;
        while j < QUAD_SLOTS {
            t[j] = Self::psi_pow(QUAD_SLOT_EXP[j] as u32);
            j += 1;
        }
        t
    };
}
