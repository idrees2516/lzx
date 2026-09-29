//! Forward NTT for **binary** inputs on the *quadratic-slot* tree (q in `QS_QUAD`), in the
//! vertical batch-of-32 layout: primes `2917`, `4861`, `12637`.
//!
//! Port of upstream `simd/ntt/bin_quad.rs` (with `bin_asm`'s lookup-Barrett helpers inlined
//! here). For q = 1 mod 972 but not mod 1944 the ring does not split completely: `Phi_1944`
//! factors into 324 irreducible quadratics and the transform ends at `Z_q[X]/(X^2 - psi'^u)`
//! leaves. The tree is the splitting one with its *second* radix-2 level removed — Phi_6, one
//! radix-2 level, then four radix-3 levels `162 -> 54 -> 18 -> 6 -> 2`.
//!
//! | phase                | what it does                                                       |
//! |----------------------|--------------------------------------------------------------------|
//! | lookups + levels 2, 3| 9 or 27 `vpermb` per group of 9 vectors, then two radix-3 rounds  |
//! | levels 4 + 5         | two passes over one 18-block of the L1 scratch, to the sink       |
//!
//! Levels 0 and 1 are a linear function of the 4-bit nibble of each polynomial, so they are one
//! 16-entry byte-split `vpermb` lookup per output vector on exactly the [`BinaryIndex32`] rows
//! the splitting kernel consumes — the front end is shared verbatim.
//!
//! ## What the lookup tables carry
//!
//! Level 2 is radix 3 and because each level-1 value is consumed in exactly one role its
//! level-2 twiddle folds into the table: 3 tables of 16 entries per 162-block, 12 in all (the
//! **unfolded** phase 1; q = 12637 runs it).
//!
//! Level 3's twiddles cannot be folded the same way, but they fold if the butterfly is
//! *replaced by the sum it computes*: three lookups and two adds from tables indexed by
//! `(k, s, r, i/18)` — 108 of them, 6912 bytes. Level 2's omega multiplication disappears with
//! the butterfly and level 3 becomes the omega-only `r3_folded`, so a group of 9 rows costs
//! `27 vpermb + 18 add + 3 r3_folded` = 78 ALU uops against 99, and the batch costs **1512
//! Montgomery products instead of 2160**. What it spends is bound head-room: 12637 cannot pay
//! and keeps the unfolded phase 1.
//!
//! ## Output layout
//!
//! `out.v[2j][p]` and `out.v[2j+1][p]` are the two coefficients of
//! `a_p mod (X^2 - psi'^QUAD_SLOT_EXP[j])`. The kernel finishes the 648 rows as **36 blocks of
//! 18 consecutive rows** — block `blk` is rows `18 blk .. 18 blk + 18`, i.e. the 9 quadratic
//! leaves `9 blk .. 9 blk + 9` — each produced from 18 registers by one levels-4+5 tail.
//!
//! ## Bounds (|lane| as a multiple of q; `tests/simd.rs` checks every output against the
//! declared bound and against `scalar::ntt_quad` of the same lift)
//!
//! | after            | q = 2917 | q = 4861 | q = 12637 |
//! |------------------|---------:|---------:|----------:|
//! | phase 1          |   1.50 q |   1.50 q |    1.60 q |
//! | level 3          |   4.50 q |   4.50 q |    1.88 q |
//! | level 4          |   5.70 q |   2.02 q |    1.93 q |
//! | level 5 (output) |   6.96 q |   3.17 q |    1.94 q |
//!
//! `2^15/q` is 11.23 (2917), 6.74 (4861) and 2.59 (12637). **2917 needs no reduction anywhere**;
//! 4861 needs exactly one, and 12637, on the unfolded phase 1, needs three. All of them are the
//! shuffle-port **lookup Barrett** (`vpmultishiftqb` + `vpandd` + `vpord` + `vpermb` + `vpaddw`
//! — 2 port-5 and 3 flexible uops, not one multiply-port slot) applied to the untwiddled `a0`
//! input of a level.

use crate::params::*;
pub use crate::simd::transpose::BinaryIndex32;
use crate::simd::Batch32;
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------------------------
// the lookup Barrett (from upstream's bin_asm)
// ---------------------------------------------------------------------------------------------

const fn lut_k(s: usize, q: u16) -> i64 {
    let sgn = if s < 16 { s as i64 } else { s as i64 - 32 };
    let num = 4096 * sgn + 2047;
    let den = 2 * q as i64;
    if num >= 0 {
        (num + den / 2) / den
    } else {
        -((-num + den / 2) / den)
    }
}

/// `-k(s) * q`, the i16 the byte-split table adds back for window `s`.
pub const fn barrett_lut_corr(s: usize, q: u16) -> i16 {
    (-lut_k(s, q) * q as i64) as i16
}

/// The kernel's lane-wise lookup Barrett: `vpmultishiftqb` (the 5-bit window `(a >> 11) & 31`
/// into both bytes of the lane), `vpandd` + `vpord` (drop the junk bits, +32 on the high byte),
/// `vpermb` on the 64-byte byte-split table of `-k q`, `vpaddw`. Exhaustively over all i16,
/// `max |r| <= q/2 + 2^10`.
#[inline]
pub fn barrett_lut_i16(a: i16, q: u16) -> i16 {
    a.wrapping_add(barrett_lut_corr(((a >> 11) & 31) as usize, q))
}

// ---------------------------------------------------------------------------------------------
// bounds
// ---------------------------------------------------------------------------------------------

/// `|barrett_lut_i16(a, q)|` for the worst i16 `a`, per prime — an exhaustive sweep, evaluated
/// once (`q/2 + 2^10` is the theoretical bound; the sweep is a little tighter).
const BLM: [i32; 3] = [
    barrett_lut_sweep(QS_QUAD[0]),
    barrett_lut_sweep(QS_QUAD[1]),
    barrett_lut_sweep(QS_QUAD[2]),
];

/// `|barrett_lut_i16(a, q)|` for the worst i16 `a`.
pub const fn barrett_lut_max(q: u16) -> i32 {
    if q == QS_QUAD[0] {
        BLM[0]
    } else if q == QS_QUAD[1] {
        BLM[1]
    } else {
        BLM[2]
    }
}

const fn barrett_lut_sweep(q: u16) -> i32 {
    let mut m = 0i32;
    let mut a = -32768i32;
    while a < 32768 {
        let r = ((a + barrett_lut_corr(((a >> 11) & 31) as usize, q) as i32) as i16) as i32;
        let r = if r < 0 { -r } else { r };
        if r > m {
            m = r;
        }
        a += 1;
    }
    m
}

/// `|mont(a, w)| <= |a| q / 2^17 + q/2` (`params::mont_mul_i16`, |w| <= q/2).
const fn mont_bound(b: i32, q: u16) -> i32 {
    ((b as i64 * q as i64) >> 17) as i32 + (q as i32 + 1) / 2
}

/// The kernel's schedule replayed on bounds, position by position: the maximum |lane| after each
/// of levels 2 (the fused lookups), 3, 4 and 5, and the largest intermediate ever formed (which
/// is what has to stay inside i16). `bar[l]` reduces the untwiddled `a0` input of level `3 + l`.
pub const fn bin_model(q: u16, bar: [bool; 3]) -> ([i32; 4], i32) {
    if fold3(q) {
        bin_model_f3(q, bar)
    } else {
        bin_model_split(q, bar)
    }
}

/// The unfolded schedule: the fused lookups carry only the level-2 twiddle, level 2 is the
/// omega-only butterfly and levels 3, 4, 5 are full radix-3 with `bar[l]` on the `a0` of
/// level `3 + l`.
pub const fn bin_model_split(q: u16, bar: [bool; 3]) -> ([i32; 4], i32) {
    let r = barrett_lut_max(q);
    let h = (q as i32 + 1) / 2;
    let mut v = [0i32; N];
    let mut lm = [0i32; 4];
    let mut peak = 0i32;
    // fused lookups (|T| <= q/2 in all three roles) + level 2, omega-only radix-3
    let u = mont_bound(2 * h, q);
    let mut k = 0;
    while k < 4 {
        let mut i = 0;
        while i < 54 {
            let b = 162 * k + i;
            v[b] = 3 * h;
            v[b + 54] = 2 * h + u;
            v[b + 108] = 2 * h + u;
            if 3 * h > peak {
                peak = 3 * h;
            }
            if 2 * h + u > peak {
                peak = 2 * h + u;
            }
            i += 1;
        }
        k += 1;
    }
    let mut i = 0;
    while i < N {
        if v[i] > lm[0] {
            lm[0] = v[i];
        }
        i += 1;
    }
    // levels 3, 4, 5: radix-3 on 54-, 18- and 6-blocks
    let mut l = 0;
    while l < 3 {
        let blk = [54usize, 18, 6][l];
        let m = blk / 3;
        let mut base = 0;
        while base < N {
            let mut i = 0;
            while i < m {
                let (i0, i1, i2) = (base + i, base + i + m, base + i + 2 * m);
                let b0 = if bar[l] { r } else { v[i0] };
                let t1 = mont_bound(v[i1], q);
                let t2 = mont_bound(v[i2], q);
                let uu = mont_bound(t1 + t2, q);
                if t1 + t2 > peak {
                    peak = t1 + t2;
                }
                if b0 + t1 + t2 > peak {
                    peak = b0 + t1 + t2;
                }
                if b0 + t2 + uu > peak {
                    peak = b0 + t2 + uu;
                }
                if b0 + t1 + uu > peak {
                    peak = b0 + t1 + uu;
                }
                v[i0] = b0 + t1 + t2;
                v[i1] = b0 + t2 + uu;
                v[i2] = b0 + t1 + uu;
                i += 1;
            }
            base += blk;
        }
        let mut i = 0;
        while i < N {
            if v[i] > lm[l + 1] {
                lm[l + 1] = v[i];
            }
            i += 1;
        }
        l += 1;
    }
    (lm, peak)
}

/// The same replay for the **level-3-folded** phase 1: every row leaves the three lookups at
/// `3 h`, level 3 is the omega-only butterfly (no `a0` to reduce), and levels 4 and 5 are the
/// full radix-3 of [`bin_model`]. `bar[0]` is ignored; `bar[1]`, `bar[2]` reduce the untwiddled
/// `a0` of levels 4 and 5.
pub const fn bin_model_f3(q: u16, bar: [bool; 3]) -> ([i32; 4], i32) {
    let r = barrett_lut_max(q);
    let h = (q as i32 + 1) / 2;
    let mut v = [0i32; N];
    let mut lm = [0i32; 4];
    let mut peak = 3 * h;
    // three lookups summed: |T| <= h in every role
    let mut i = 0;
    while i < N {
        v[i] = 3 * h;
        i += 1;
    }
    lm[0] = 3 * h;
    // level 3: omega-only radix-3 on (i, i+18, i+36) of each 54-block
    let u = mont_bound(6 * h, q);
    if 6 * h > peak {
        peak = 6 * h;
    }
    if 9 * h > peak {
        peak = 9 * h;
    }
    if 6 * h + u > peak {
        peak = 6 * h + u;
    }
    let mut base = 0;
    while base < N {
        let mut i = 0;
        while i < 18 {
            v[base + i] = 9 * h;
            v[base + i + 18] = 6 * h + u;
            v[base + i + 36] = 6 * h + u;
            i += 1;
        }
        base += 54;
    }
    lm[1] = 9 * h;
    // levels 4 and 5
    let mut l = 1;
    while l < 3 {
        let blk = [0usize, 18, 6][l];
        let m = blk / 3;
        let mut base = 0;
        while base < N {
            let mut i = 0;
            while i < m {
                let (i0, i1, i2) = (base + i, base + i + m, base + i + 2 * m);
                let b0 = if bar[l] { r } else { v[i0] };
                let t1 = mont_bound(v[i1], q);
                let t2 = mont_bound(v[i2], q);
                let uu = mont_bound(t1 + t2, q);
                if t1 + t2 > peak {
                    peak = t1 + t2;
                }
                if b0 + t1 + t2 > peak {
                    peak = b0 + t1 + t2;
                }
                if b0 + t2 + uu > peak {
                    peak = b0 + t2 + uu;
                }
                if b0 + t1 + uu > peak {
                    peak = b0 + t1 + uu;
                }
                v[i0] = b0 + t1 + t2;
                v[i1] = b0 + t2 + uu;
                v[i2] = b0 + t1 + uu;
                i += 1;
            }
            base += blk;
        }
        let mut i = 0;
        while i < N {
            if v[i] > lm[l + 1] {
                lm[l + 1] = v[i];
            }
            i += 1;
        }
        l += 1;
    }
    (lm, peak)
}

/// The cheapest reduction schedule of the level-3-folded phase 1, and whether one exists at all.
const fn f3_sched(q: u16) -> ([bool; 3], bool) {
    let opts = [
        [false, false, false],
        [false, true, false],
        [false, false, true],
        [false, true, true],
    ];
    let mut i = 0;
    while i < 4 {
        if bin_model_f3(q, opts[i]).1 <= 32767 {
            return (opts[i], true);
        }
        i += 1;
    }
    ([false; 3], false)
}

/// Does this prime run the phase 1 that folds the level-3 twiddles into the lookup tables?
pub const fn fold3(q: u16) -> bool {
    FOLD3[qi(q)]
}

const FOLD3: [bool; 3] = [
    f3_sched(QS_QUAD[0]).1,
    f3_sched(QS_QUAD[1]).1,
    f3_sched(QS_QUAD[2]).1,
];

/// Does the un-reduced schedule leave i16? (2917 and 4861: no. 12637: yes.)
pub const fn needs_barrett(q: u16) -> bool {
    bin_model_split(q, [false; 3]).1 > 32767
}

/// `(reduction flags, output bound)` per prime, evaluated once.
const BIN_SCHED: [([bool; 3], i32); 3] = [
    bin_sched(QS_QUAD[0]),
    bin_sched(QS_QUAD[1]),
    bin_sched(QS_QUAD[2]),
];

const fn bin_sched(q: u16) -> ([bool; 3], i32) {
    if fold3(q) {
        let bar = f3_sched(q).0;
        return (bar, bin_model_f3(q, bar).0[3]);
    }
    let bar = if needs_barrett(q) {
        [true; 3]
    } else {
        [false; 3]
    };
    (bar, bin_model_split(q, bar).0[3])
}

const fn qi(q: u16) -> usize {
    if q == QS_QUAD[0] {
        0
    } else if q == QS_QUAD[1] {
        1
    } else {
        2
    }
}

/// Which of levels 3, 4 and 5 reduce their untwiddled `a0` input with the lookup Barrett.
pub const fn bar_levels(q: u16) -> [bool; 3] {
    BIN_SCHED[qi(q)].0
}

/// Declared output bound: max |lane| of [`ntt_quad_bin_batch32`], per prime.
pub const fn output_bound(q: u16) -> i32 {
    BIN_SCHED[qi(q)].1
}

const _: () = assert!(bin_model(2917, bar_levels(2917)).1 <= 32767);
const _: () = assert!(bin_model(4861, bar_levels(4861)).1 <= 32767);
const _: () = assert!(bin_model(12637, bar_levels(12637)).1 <= 32767);
// 2917 and 4861 fold level 3 into the tables, 2917 for free and 4861 for one lookup Barrett at
// level 4; 12637 cannot and keeps the unfolded phase 1, where it needs all three Barretts.
const _: () = assert!(fold3(2917) && fold3(4861) && !fold3(12637));
const _: () = assert!(!bar_levels(2917)[1] && !bar_levels(2917)[2]);
const _: () = assert!(bar_levels(4861)[1] && !bar_levels(4861)[2]);
const _: () = assert!(!needs_barrett(2917) && !needs_barrett(4861) && needs_barrett(12637));
const _: () = assert!(bin_model_split(12637, [false, true, true]).1 > 32767);
const _: () = assert!(bin_model_split(12637, [true, false, true]).1 > 32767);
const _: () = assert!(bin_model_split(12637, [true, true, false]).1 > 32767);

// ---------------------------------------------------------------------------------------------
// constant tables
// ---------------------------------------------------------------------------------------------

const fn dup(x: i16) -> u32 {
    (x as u16 as u32) | ((x as u16 as u32) << 16)
}

#[repr(C, align(64))]
pub struct Tables {
    /// `lut[3 k + r]`: the 16 centered i16 values of `base_k(n) * zeta2_k^r`, r = the level-2
    /// role `i/54` of the position, **byte-split** so that one `vpermb` (1 uop, port 5) does
    /// the lookup: byte n is the low half of entry n, byte 16+n the high half.
    lut: [[u8; 64]; 12],
    /// `lut3[((3 k + s) * 3 + r) * 3 + a]`: the same 16 values scaled by
    /// `zeta2_k^r omega^{r s} zeta3_{3k+s}^a`, byte-split the same way — the level-2 role `r`,
    /// the level-2 output `s` and the level-3 role `a` of the position all folded in, so an
    /// output row of level 2 is three `vpermb` and two `vpaddw` and level 3 has no twiddle left
    /// to apply. Used when [`fold3`]; 108 tables, 6912 bytes.
    lut3: [[u8; 64]; 108],
    /// 512-bit constants of the lookup Barrett and the omega product, as memory operands:
    /// `[ms, corr, and, or]` — the `vpmultishiftqb` control, the byte-split `-k q` table and
    /// the index fix-up masks.
    cv: [[i16; 32]; 4],
    /// `[w, w', w2, w2']` (Montgomery twiddle and companion for zeta and zeta^2) per sub-ring,
    /// each i16 duplicated into a u32 so `vpbroadcastd` is a pure load.
    tw3: [[u32; 4]; 12],
    tw4: [[u32; 4]; 36],
    tw5: [[u32; 4]; 108],
    /// omega and its companion.
    om: [u32; 2],
    /// q, duplicated.
    qd: u32,
}

const fn mont_pair<const Q: u16>(x: u16) -> (u32, u32) {
    let w = Params::<Q>::to_mont(x);
    (dup(w), dup(Params::<Q>::mont_pre(w)))
}

const fn r3_pair<const Q: u16>(z: u16) -> [u32; 4] {
    let z2 = (z as u64 * z as u64 % Q as u64) as u16;
    let (a, b) = mont_pair::<Q>(z);
    let (c, d) = mont_pair::<Q>(z2);
    [a, b, c, d]
}

macro_rules! r3_twiddles {
    ($n:literal, $pair:path, $zetas:expr) => {{
        let mut t = [[0u32; 4]; $n];
        let mut i = 0;
        while i < $n {
            t[i] = $pair($zetas[i]);
            i += 1;
        }
        t
    }};
}

const fn build_tables<const Q: u16>() -> Tables {
    let q = Q as u64;
    let z6 = ParamsQ::<Q>::ZETA6 as u64;
    let kappa = [z6, (1 + q - z6) % q];

    // levels 0 + 1 as a function of the nibble, with the level-2 twiddle folded in
    let mut lut = [[0u8; 64]; 12];
    let mut k = 0;
    while k < 4 {
        let s0 = k / 2;
        let s1 = k % 2;
        let ka = kappa[s0];
        let z1 = ParamsQ::<Q>::ZETA_L1[s0] as u64;
        let z2 = ParamsQ::<Q>::ZETA_L2[k] as u64;
        let mut r = 0;
        while r < 3 {
            let f = pow_mod(z2, r as u64, q);
            let mut n = 0;
            while n < 16 {
                let n0 = (n & 1) as u64;
                let n1 = ((n >> 1) & 1) as u64;
                let n2 = ((n >> 2) & 1) as u64;
                let n3 = ((n >> 3) & 1) as u64;
                let inner = z1 * ((n1 + ka * n3) % q) % q;
                let t = if s1 == 0 { inner } else { (q - inner) % q };
                let base = ((n0 + ka * n2) % q + t) % q;
                let e = center(base * f % q, q) as u16;
                lut[3 * k + r][n] = e as u8;
                lut[3 * k + r][16 + n] = (e >> 8) as u8;
                n += 1;
            }
            r += 1;
        }
        k += 1;
    }

    // the same base values, with the level-2 role and output and the level-3 role folded in
    let om = ParamsQ::<Q>::OMEGA as u64;
    let mut lut3 = [[0u8; 64]; 108];
    let mut k = 0;
    while k < 4 {
        let s0 = k / 2;
        let s1 = k % 2;
        let ka = kappa[s0];
        let z1 = ParamsQ::<Q>::ZETA_L1[s0] as u64;
        let z2 = ParamsQ::<Q>::ZETA_L2[k] as u64;
        let mut s = 0;
        while s < 3 {
            let z3 = ParamsQ::<Q>::ZETA_L3[3 * k + s] as u64;
            let mut r = 0;
            while r < 3 {
                let f = pow_mod(z2, r as u64, q) * pow_mod(om, (r * s) as u64, q) % q;
                let mut a = 0;
                while a < 3 {
                    let g = f * pow_mod(z3, a as u64, q) % q;
                    let mut n = 0;
                    while n < 16 {
                        let n0 = (n & 1) as u64;
                        let n1 = ((n >> 1) & 1) as u64;
                        let n2 = ((n >> 2) & 1) as u64;
                        let n3 = ((n >> 3) & 1) as u64;
                        let inner = z1 * ((n1 + ka * n3) % q) % q;
                        let t = if s1 == 0 { inner } else { (q - inner) % q };
                        let base = ((n0 + ka * n2) % q + t) % q;
                        let e = center(base * g % q, q) as u16;
                        let ix = ((3 * k + s) * 3 + r) * 3 + a;
                        lut3[ix][n] = e as u8;
                        lut3[ix][16 + n] = (e >> 8) as u8;
                        n += 1;
                    }
                    a += 1;
                }
                r += 1;
            }
            s += 1;
        }
        k += 1;
    }

    let tw3 = r3_twiddles!(12, r3_pair::<Q>, ParamsQ::<Q>::ZETA_L3);
    let tw4 = r3_twiddles!(36, r3_pair::<Q>, ParamsQ::<Q>::ZETA_L4);
    let tw5 = r3_twiddles!(108, r3_pair::<Q>, ParamsQ::<Q>::ZETA_L5);
    let (oa, ob) = mont_pair::<Q>(ParamsQ::<Q>::OMEGA);

    let mut cv = [[0i16; 32]; 4];
    let mut i = 0;
    while i < 32 {
        // vpmultishiftqb control: both bytes of word j of a qword take bits 11..18 of that word.
        cv[0][i] = ((16 * (i % 4) + 11) * 257) as i16;
        // byte-split correction table: byte u (u < 32) is the low half of -k(u) q, byte 32 + u
        // its high half.
        let (b0, b1) = (lut_byte(2 * i, Q), lut_byte(2 * i + 1, Q));
        cv[1][i] = (b0 as u16 | ((b1 as u16) << 8)) as i16;
        cv[2][i] = 0x1f1f;
        cv[3][i] = 0x2000;
        i += 1;
    }
    Tables {
        lut,
        lut3,
        cv,
        tw3,
        tw4,
        tw5,
        om: [oa, ob],
        qd: dup(Q as i16),
    }
}

/// Byte `u` of the 64-byte `vpermb` correction table of the lookup Barrett: the low halves of
/// `-k(s) q` at u = s < 32, the high halves at u = 32 + s.
pub const fn lut_byte(u: usize, q: u16) -> u8 {
    if u < 32 {
        barrett_lut_corr(u, q) as u16 as u8
    } else {
        (barrett_lut_corr(u - 32, q) as u16 >> 8) as u8
    }
}

static T2917: Tables = build_tables::<2917>();
static T4861: Tables = build_tables::<4861>();
static T12637: Tables = build_tables::<12637>();

#[inline(always)]
fn tables<const Q: u16>() -> &'static Tables {
    match Q {
        2917 => &T2917,
        4861 => &T4861,
        _ => &T12637,
    }
}

// ---------------------------------------------------------------------------------------------
// arithmetic helpers
// ---------------------------------------------------------------------------------------------

/// `vpbroadcastd zmm, m32` - one pure load uop. Written as `asm!` because LLVM otherwise
/// "recognises" the duplicated-u32 splat and rebuilds it with vpmovsxwd/vpmovdw/vinserti64x4.
#[target_feature(enable = "avx512f")]
unsafe fn bc(p: *const u32) -> __m512i {
    let r: __m512i;
    core::arch::asm!(
        "vpbroadcastd {0}, dword ptr [{1}]",
        out(zmm_reg) r,
        in(reg) p,
        options(pure, readonly, nostack, preserves_flags)
    );
    r
}

/// The four twiddle broadcasts of one sub-ring off a single base register.
#[target_feature(enable = "avx512f")]
unsafe fn bc4(p: *const u32) -> (__m512i, __m512i, __m512i, __m512i) {
    let (a, b, c, d): (__m512i, __m512i, __m512i, __m512i);
    core::arch::asm!(
        "vpbroadcastd {0}, dword ptr [{4}]",
        "vpbroadcastd {1}, dword ptr [{4} + 4]",
        "vpbroadcastd {2}, dword ptr [{4} + 8]",
        "vpbroadcastd {3}, dword ptr [{4} + 12]",
        out(zmm_reg) a,
        out(zmm_reg) b,
        out(zmm_reg) c,
        out(zmm_reg) d,
        in(reg) p,
        options(pure, readonly, nostack, preserves_flags)
    );
    (a, b, c, d)
}

/// `vpermb zmm, zmm, m512` — the table straight out of L1, one port-5 uop and one load. Written
/// as `asm!` because LLVM otherwise hoists all 27 tables of a phase-1 iteration into registers
/// and spills them.
#[target_feature(enable = "avx512f")]
unsafe fn permb_m(idx: __m512i, p: *const u8) -> __m512i {
    let r: __m512i;
    core::arch::asm!(
        "vpermb {0}, {1}, [{2}]",
        out(zmm_reg) r,
        in(zmm_reg) idx,
        in(reg) p,
        options(pure, readonly, nostack, preserves_flags)
    );
    r
}

/// `vpmulhw`. stdarch's `_mm512_mulhi_epi16` is written as sext -> mul -> shr -> trunc; LLVM
/// folds the multiply but leaves a `vpmovsxwd`/`vpmovdw`/`vinserti64x4` round trip on operands
/// it cannot see through (broadcast constants), and rematerialises it inside the hot loops.
#[target_feature(enable = "avx512f")]
unsafe fn mulhi(a: __m512i, b: __m512i) -> __m512i {
    let r: __m512i;
    core::arch::asm!(
        "vpmulhw {0}, {1}, {2}",
        out(zmm_reg) r,
        in(zmm_reg) a,
        in(zmm_reg) b,
        options(pure, nomem, nostack, preserves_flags)
    );
    r
}

/// 3-uop signed Montgomery twiddle multiply: `a * x mod q` in `(-q, q)`.
#[inline(always)]
unsafe fn mont(a: __m512i, w: __m512i, wp: __m512i, q: __m512i) -> __m512i {
    let m = _mm512_mullo_epi16(a, wp);
    let hi = mulhi(a, w);
    let t = mulhi(m, q);
    _mm512_sub_epi16(hi, t)
}

/// The shuffle-port lookup Barrett: 2 port-5 + 3 flexible uops, no multiply-port slot,
/// `|r| <= q/2 + 2^10`.
#[inline(always)]
unsafe fn barrett_lut(a: __m512i, c: &C) -> __m512i {
    let s = _mm512_multishift_epi64_epi8(c.ms, a);
    let s = _mm512_and_si512(s, c.andm);
    let s = _mm512_or_si512(s, c.orm);
    _mm512_add_epi16(a, _mm512_permutexvar_epi8(s, c.corr))
}

struct C {
    q: __m512i,
    om: __m512i,
    omp: __m512i,
    ms: __m512i,
    corr: __m512i,
    andm: __m512i,
    orm: __m512i,
}

/// Radix-3 butterfly with twiddles from `tw = [w, w', w2, w2']`, optionally reducing `a0`.
#[inline(always)]
unsafe fn r3<const BAR: bool>(
    c: &C,
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    tw: *const u32,
) -> (__m512i, __m512i, __m512i) {
    let (w1, w1p, w2, w2p) = bc4(tw);
    let t1 = mont(a1, w1, w1p, c.q);
    let t2 = mont(a2, w2, w2p, c.q);
    let u = mont(_mm512_sub_epi16(t1, t2), c.om, c.omp, c.q);
    let a0 = if BAR { barrett_lut(a0, c) } else { a0 };
    (
        _mm512_add_epi16(a0, _mm512_add_epi16(t1, t2)),
        _mm512_add_epi16(_mm512_sub_epi16(a0, t2), u),
        _mm512_sub_epi16(_mm512_sub_epi16(a0, t1), u),
    )
}

/// Radix-3 butterfly whose twiddles are already folded into the inputs (level 2).
#[inline(always)]
unsafe fn r3_folded(c: &C, a0: __m512i, t1: __m512i, t2: __m512i) -> (__m512i, __m512i, __m512i) {
    let u = mont(_mm512_sub_epi16(t1, t2), c.om, c.omp, c.q);
    (
        _mm512_add_epi16(a0, _mm512_add_epi16(t1, t2)),
        _mm512_add_epi16(_mm512_sub_epi16(a0, t2), u),
        _mm512_sub_epi16(_mm512_sub_epi16(a0, t1), u),
    )
}

#[inline(always)]
unsafe fn ldb(p: *const u8, j: usize) -> __m512i {
    _mm512_load_si512(p.add(64 * j) as *const __m512i)
}
#[inline(always)]
unsafe fn ld(p: *const i16, j: usize) -> __m512i {
    _mm512_load_si512(p.add(32 * j) as *const __m512i)
}
#[inline(always)]
unsafe fn st(p: *mut i16, j: usize, v: __m512i) {
    _mm512_store_si512(p.add(32 * j) as *mut __m512i, v);
}

#[repr(C, align(64))]
struct Blk([i16; 162 * 32]);

/// The per-prime schedule as compile-time constants of the kernel.
struct Sched<const Q: u16>;

impl<const Q: u16> Sched<Q> {
    const FOLD3: bool = fold3(Q);
}

// ---------------------------------------------------------------------------------------------
// where the finished blocks go
// ---------------------------------------------------------------------------------------------

/// Consumer of the transform's output, one **18-row block** at a time.
///
/// The kernel finishes the 648 rows as 36 independent blocks of 18: block `blk` holds rows
/// `18 blk .. 18 blk + 18`, which is the pair of coefficients of each of the 9 quadratic leaves
/// `9 blk .. 9 blk + 9` (leaf j in rows 2j, 2j+1), and is produced out of 18 registers by one
/// levels-4+5 tail. The sink says where those 18 vectors go ([`dst`](BlockSink::dst)) and is
/// handed them the instant they are stored ([`block`](BlockSink::block)), so a consumer can read
/// a block while it is still in L1.
///
/// # Safety
/// `dst` must return a 64-byte aligned pointer to 1152 writable bytes (18 vectors); the kernel
/// writes them and then calls `block` with the same pointer.
pub trait BlockSink {
    /// Where block `blk` is to be stored.
    ///
    /// # Safety
    /// The returned pointer must be 64-byte aligned and cover 18 writable vectors.
    unsafe fn dst(&mut self, blk: usize) -> *mut i16;
    /// Called once the 18 vectors of block `blk` are stored at `dst`.
    ///
    /// # Safety
    /// `dst` is the pointer this sink's `dst` returned, now holding written data.
    unsafe fn block(&mut self, blk: usize, dst: *const i16);
}

/// The sink of the plain entry points: block `blk` goes to its own place in the output batch and
/// nothing further happens, so the kernel is exactly the loop it would be without the hook.
pub struct OutSink(pub *mut i16);

impl BlockSink for OutSink {
    #[inline(always)]
    unsafe fn dst(&mut self, blk: usize) -> *mut i16 {
        self.0.add(32 * 18 * blk)
    }
    #[inline(always)]
    unsafe fn block(&mut self, _blk: usize, _dst: *const i16) {}
}

// ---------------------------------------------------------------------------------------------
// the kernel
// ---------------------------------------------------------------------------------------------

#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "avx512vbmi", enable = "avx512vbmi2", enable = "avx512vnni", enable = "gfni")]
unsafe fn ntt_core<const Q: u16, S: BlockSink>(input: &BinaryIndex32, sink: &mut S) {
    let t = tables::<Q>();
    let cvp = t.cv.as_ptr() as *const __m512i;
    let c = C {
        q: bc(&t.qd),
        om: bc(&t.om[0]),
        omp: bc(&t.om[1]),
        ms: _mm512_load_si512(cvp),
        corr: _mm512_load_si512(cvp.add(1)),
        andm: _mm512_load_si512(cvp.add(2)),
        orm: _mm512_load_si512(cvp.add(3)),
    };
    let bar = bar_levels(Q);
    // As an associated const so the choice is made at compile time and only one phase 1 is
    // emitted: `fold3(Q)` on its own is a `const fn` call LLVM does not fold, and both bodies
    // alive at once spills every table.
    let fold3 = Sched::<Q>::FOLD3;

    // The caller already holds the `vpermb` byte-index rows; the kernel reads them straight.
    let ip: *const u8 = input.rows.as_ptr() as *const u8;
    let mut blk: core::mem::MaybeUninit<Blk> = core::mem::MaybeUninit::uninit();
    let bp = blk.as_mut_ptr() as *mut i16;

    for k in 0..4 {
        let lut = t.lut.as_ptr().add(3 * k) as *const u8;
        let l = |r: usize| -> __m512i { _mm512_load_si512(lut.add(64 * r) as *const __m512i) };
        let (l0, l1, l2) = (l(0), l(1), l(2));
        let l3 = t.lut3.as_ptr() as *const u8;

        // Levels 0-3, 9 vectors at a time: the three level-2 triples (i, i+54, i+108) for
        // i = i0, i0+18, i0+36 are exactly the three inputs of one level-3 butterfly in each of
        // the three 54-blocks, so the nine level-2 outputs of one `i0` never leave registers.
        for i0 in 0..18 {
            let mut y = [_mm512_setzero_si512(); 9];
            for a in 0..3 {
                let i = i0 + 18 * a;
                let (n0, n1, n2) = (ldb(ip, i), ldb(ip, i + 54), ldb(ip, i + 108));
                if fold3 {
                    // level 2 straight out of the tables: output s is the sum of the three
                    // lookups whose tables carry omega^{r s} and the level-3 twiddle of role a.
                    for s in 0..3 {
                        let p = l3.add(64 * (9 * (3 * k + s) + a));
                        y[3 * s + a] = _mm512_add_epi16(
                            _mm512_add_epi16(permb_m(n0, p), permb_m(n1, p.add(192))),
                            permb_m(n2, p.add(384)),
                        );
                    }
                } else {
                    let x0 = _mm512_permutexvar_epi8(n0, l0);
                    let x1 = _mm512_permutexvar_epi8(n1, l1);
                    let x2 = _mm512_permutexvar_epi8(n2, l2);
                    let (u0, u1, u2) = r3_folded(&c, x0, x1, x2);
                    y[a] = u0;
                    y[3 + a] = u1;
                    y[6 + a] = u2;
                }
            }
            for s in 0..3 {
                let (v0, v1, v2) = if fold3 {
                    r3_folded(&c, y[3 * s], y[3 * s + 1], y[3 * s + 2])
                } else if bar[0] {
                    let tw = t.tw3[3 * k + s].as_ptr();
                    r3::<true>(&c, y[3 * s], y[3 * s + 1], y[3 * s + 2], tw)
                } else {
                    let tw = t.tw3[3 * k + s].as_ptr();
                    r3::<false>(&c, y[3 * s], y[3 * s + 1], y[3 * s + 2], tw)
                };
                let b = 54 * s + i0;
                st(bp, b, v0);
                st(bp, b + 18, v1);
                st(bp, b + 36, v2);
            }
        }

        // levels 4 and 5, one 18-block at a time, straight to the sink. Two passes over the
        // L1-resident block (the register-resident form measures the same when it is the only
        // tail in the build, and offering both behind a `const` parameter costs 60: LLVM then
        // schedules the register-resident copy badly).
        for j in 0..9 {
            let kk = 9 * k + j;
            let op = sink.dst(kk);
            let base = 18 * j;
            let t4 = t.tw4[kk].as_ptr();
            for i in 0..6 {
                let (b0, b1, b2) = (base + i, base + i + 6, base + i + 12);
                let (o0, o1, o2) = if bar[1] {
                    r3::<true>(&c, ld(bp, b0), ld(bp, b1), ld(bp, b2), t4)
                } else {
                    r3::<false>(&c, ld(bp, b0), ld(bp, b1), ld(bp, b2), t4)
                };
                st(bp, b0, o0);
                st(bp, b1, o1);
                st(bp, b2, o2);
            }
            for g in 0..3 {
                let t5 = t.tw5[3 * kk + g].as_ptr();
                let b = 6 * g;
                for i in 0..2 {
                    let (b0, b1, b2) = (base + b + i, base + b + i + 2, base + b + i + 4);
                    let (o0, o1, o2) = if bar[2] {
                        r3::<true>(&c, ld(bp, b0), ld(bp, b1), ld(bp, b2), t5)
                    } else {
                        r3::<false>(&c, ld(bp, b0), ld(bp, b1), ld(bp, b2), t5)
                    };
                    st(op, b + i, o0);
                    st(op, b + i + 2, o1);
                    st(op, b + i + 4, o2);
                }
            }
            sink.block(kk, op);
        }
    }
}

/// Forward quadratic-slot NTT of 32 binary polynomials: rows `2j`, `2j+1` of the output hold
/// `a_p mod (X^2 - psi'^QUAD_SLOT_EXP[j])`, lazily reduced (`|lane| <= output_bound(Q)`).
///
/// # Safety
/// The host must have AVX-512 F/BW/VL/VBMI/VBMI2/VNNI/GFNI (checked by
/// [`crate::simd::available`]); `out` is 64-byte aligned (`Batch32` is).
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "avx512vbmi", enable = "avx512vbmi2", enable = "avx512vnni", enable = "gfni")]
pub unsafe fn ntt_quad_bin_batch32<const Q: u16>(input: &BinaryIndex32, out: &mut Batch32) {
    ntt_core::<Q, _>(input, &mut OutSink(out.v.as_mut_ptr() as *mut i16));
}

/// The same transform with the output handed to `sink` 18 rows at a time instead of being
/// written to a [`Batch32`], for consumers that want each block while it is still in L1.
///
/// # Safety
/// See [`BlockSink`]: `sink.dst` must give 18 writable 64-byte aligned vectors per block.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "avx512vbmi", enable = "avx512vbmi2", enable = "avx512vnni", enable = "gfni")]
pub unsafe fn ntt_quad_bin_batch32_sink<const Q: u16, S: BlockSink>(
    input: &BinaryIndex32,
    sink: &mut S,
) {
    ntt_core::<Q, S>(input, sink);
}
