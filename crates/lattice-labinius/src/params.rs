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

/// Per-prime constants: `Params::<3889>::PSI` etc.
pub struct Params<const Q: u16>;

impl<const Q: u16> Params<Q> {
    pub const Q: u16 = Q;
    /// Smallest primitive 1944-th root of unity.
    pub const PSI: u16 = find_psi(Q as u64) as u16;
    /// Primitive cube root of unity, `omega = psi^648`.
    pub const OMEGA: u16 = pow_mod(Self::PSI as u64, 648, Q as u64) as u16;
    /// Primitive sixth root, `zeta6 = psi^324`; `zeta6^-1 = 1 - zeta6`.
    pub const ZETA6: u16 = pow_mod(Self::PSI as u64, 324, Q as u64) as u16;

    pub const fn psi_pow(e: u32) -> u16 {
        pow_mod(Self::PSI as u64, e as u64, Q as u64) as u16
    }
    /// Plain twiddle zeta for sub-ring k at level `level` (1..=6).
    pub const fn zeta(level: usize, k: usize) -> u16 {
        Self::psi_pow(twiddle_exp(level, k))
    }
}

/// Every splitting prime is `1 mod 1944`.
const _: () = {
    let mut i = 0;
    while i < 2 {
        assert!((QS[i] as u32 - 1) % CONDUCTOR == 0 && QS[i] < 1 << 14);
        assert!((QS_LARGE[i] as u32 - 1) % CONDUCTOR == 0 && QS_LARGE[i] > 1 << 14);
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
    (q as u32 - 1) % CONDUCTOR_QUAD == 0 && (q as u32 - 1) % CONDUCTOR != 0
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
        assert!(u % 2 == 1 && u % 3 != 0);
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
