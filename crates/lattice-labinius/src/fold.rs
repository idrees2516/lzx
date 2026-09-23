//! The folding step `v = sum_j c_j W_j`, entirely in the 648-slot NTT domain of `R_648`, plus
//! the verifier's `A v` recomputation and the commitment fold `sum_j c_j C_j`.
//! Port of `labinius` `fold.rs`, scalar path.
//!
//! A challenge `c_j` (an element of the subring `R_162`) enters `R_648` as `c_j(-X^4)`
//! (coefficient of `X^{4m}` is `(-1)^m c_{j,m}`, everything else zero), so the fold is a single
//! length-`r` inner product per slot with no ring multiplication anywhere. For a quadratic-slot
//! base limb, `c(-X^4)` is the scalar `c(-theta^v)` in leaf `X^2 - psi'^v`, written into both
//! rows of the leaf.

use crate::challenge::ShortChallenge;
use crate::key::AuxData;
use crate::params::{quadratic_slots, N, QUAD_CLASS_SLOT};
use crate::ring::{Modulus, PowerOfThreeRing, N162};
use crate::scalar::Coeffs;

/// Embed `c` as `c(-X^4)` into `R_648` coefficients (degree-162 multiples of 4 only).
fn embed(c: &ShortChallenge) -> [i64; N] {
    let mut out = [0i64; N];
    for i in 0..c.weight {
        let m = c.positions[i] as usize;
        let co = 1 - 2 * ((c.signs >> i) & 1) as i64;
        out[4 * m] = if m % 2 == 0 { co } else { -co };
    }
    out
}

/// `NTT(c(-X^4))` for a splitting prime: the 648 tree-order slots, centered i16.
fn challenge_ntt<const Q: u16>(c: &ShortChallenge) -> [i16; N] {
    let e = embed(c);
    let mut coeffs = [0u32; N];
    for (i, &x) in e.iter().enumerate() {
        coeffs[i] = (x.rem_euclid(Q as i64) as u64 % Q as u64) as u32;
    }
    let t = crate::scalar::ntt::<Q>(&coeffs);
    let half = (Q as i32 - 1) / 2;
    let mut out = [0i16; N];
    for u in 0..N {
        out[u] = if t[u] as i32 > half {
            t[u] as i32 - Q as i32
        } else {
            t[u] as i32
        } as i16;
    }
    out
}

/// The quadratic-slot analogue: leaf image of `c(-X^4)` is the scalar `c(-theta^v)` in both rows.
fn challenge_ntt_quad<const Q: u16>(c: &ShortChallenge) -> [i16; N] {
    let e = embed(c);
    let mut coeffs = [0u32; N];
    for (i, &x) in e.iter().enumerate() {
        coeffs[i] = (x.rem_euclid(Q as i64) as u64 % Q as u64) as u32;
    }
    let t = crate::scalar::ntt_quad::<Q>(&coeffs);
    let half = (Q as i32 - 1) / 2;
    let mut out = [0i16; N];
    for u in 0..N {
        out[u] = if t[u] as i32 > half {
            t[u] as i32 - Q as i32
        } else {
            t[u] as i32
        } as i16;
    }
    // duplicate the leaf scalars into both rows: the transform put the scalar in row 2j and 0 in
    // row 2j+1 (X^4 = psi'^{2u} is a constant per leaf); verify and copy.
    let mut scalars = [0i16; crate::params::QUAD_SLOTS];
    for j in 0..crate::params::QUAD_SLOTS {
        debug_assert_eq!(out[2 * j + 1], 0, "a challenge leaf is not a scalar");
        scalars[j] = out[2 * j];
    }
    for j in 0..crate::params::QUAD_SLOTS {
        out[2 * j + 1] = scalars[j];
    }
    out
}

/// The transformed challenge for any limb prime.
pub fn challenge_slots(q: u16, c: &ShortChallenge) -> [i16; N] {
    match q {
        3889 => challenge_ntt::<3889>(c),
        9721 => challenge_ntt::<9721>(c),
        17497 => challenge_ntt::<17497>(c),
        19441 => challenge_ntt::<19441>(c),
        2917 => challenge_ntt_quad::<2917>(c),
        4861 => challenge_ntt_quad::<4861>(c),
        12637 => challenge_ntt_quad::<12637>(c),
        _ => unreachable!("unknown limb prime {q}"),
    }
}

/// `v = sum_j c_j W_j` in coefficient form modulo the base prime, centered — the amortised
/// witness. Slot-wise over the kept witness transform, then the base limb's inverse transform.
/// A coefficient is a sum of `r * w` signed 0/1 terms (standard deviation `sqrt(r w / 2)`), two
/// orders below `q/2`.
pub fn fold_witness(
    aux: &AuxData,
    challenges: &[ShortChallenge],
    q: u16,
) -> Vec<[i16; N]> {
    assert_eq!(challenges.len(), aux.chunks, "one challenge per chunk");
    let nr = aux.batches.len() / aux.chunks;
    let ch: Vec<[i16; N]> = challenges.iter().map(|c| challenge_slots(q, c)).collect();
    let q64 = q as u64;
    let mut out = Vec::with_capacity(nr);
    for i in 0..nr {
        let mut acc = [0u64; N];
        for c in 0..aux.chunks {
            let w = &aux.batches[c * nr + i];
            let chrow = &ch[c];
            for u in 0..N {
                acc[u] += (chrow[u].rem_euclid(q as i16) as i64 as u64) * w[u] as u64;
            }
        }
        let mut coeffs = [0u32; N];
        for u in 0..N {
            coeffs[u] = (acc[u] % q64) as u32;
        }
        let v = if quadratic_slots(q) {
            crate::ring::intt_quad_of(q, &coeffs)
        } else {
            crate::ring::intt_of(q, &coeffs)
        };
        let half = (q as i32 - 1) / 2;
        let mut e = [0i16; N];
        for u in 0..N {
            e[u] = if v[u] as i32 > half {
                v[u] as i32 - q as i32
            } else {
                v[u] as i32
            } as i16;
        }
        out.push(e);
    }
    out
}

/// `A v` for one limb from centered coefficient batches: forward-transform `v`, pointwise inner
/// product against the key rows. Returns the 648-row raw commitment form.
pub fn a_times_v(q: u16, a: &[[i16; N]], v: &[[i16; N]]) -> Coeffs {
    assert_eq!(a.len(), v.len());
    let q64 = q as u64;
    let mut acc = [0u64; N];
    for i in 0..v.len() {
        let mut coeffs = [0u32; N];
        for u in 0..N {
            coeffs[u] = ((v[i][u].rem_euclid(q as i16) as i64 as u32) % q as u32);
        }
        let t = crate::ring::ntt_of(q, &coeffs);
        for u in 0..N {
            acc[u] += (a[i][u].rem_euclid(q as i16) as i64 as u64) * t[u] as u64;
        }
    }
    let mut out = [0u32; N];
    for u in 0..N {
        out[u] = (acc[u] % q64) as u32;
    }
    out
}

/// `sum_j c_j C_j` per modulus: multiplication by a challenge acts on the four `R_162`
/// components of a commitment alike and slot-wise in the `R_162` transform, so this is four
/// length-`r` inner products per slot and modulus. Returns the four folded rows.
pub fn fold_commitment(
    q: u16,
    challenges: &[ShortChallenge],
    columns: &[Vec<PowerOfThreeRing>],
) -> [PowerOfThreeRing; 4] {
    // columns[j][row] — the 4 components of column j (one modulus)
    let ch: Vec<Vec<i16>> = challenges
        .iter()
        .map(|c| challenge_r162_slots(q, c).to_vec())
        .collect();
    let q64 = q as u64;
    let half = (q as i64 - 1) / 2;
    let mut out = [PowerOfThreeRing::zero(); 4];
    for row in 0..4 {
        for s in 0..N162 {
            let mut acc = 0u64;
            for (j, c) in columns.iter().enumerate() {
                let a = (c[row].v[s] as i64).rem_euclid(q64 as i64) as u64;
                acc += a * (ch[j][s].rem_euclid(q as i16) as i64 as u64);
            }
            let r = (acc % q64) as i64;
            out[row].v[s] = if r > half { (r - q as i64) as i16 } else { r as i16 };
        }
    }
    out
}

/// The `R_162` slot multiplier of a challenge for modulus q: slot s holds `c(-theta^{v_s})` —
/// the challenge enters `R_648` as `c(-X^4)`, i.e. as the `Z`-basis element read in the `Y`
/// basis (`Z = -Y`), so the component slots (Y-basis evaluations at `theta^{v_s}`) are scaled
/// by `c` evaluated at `-theta^{v_s}`: coefficient m picks up `(-1)^m`.
pub fn challenge_r162_slots(q: u16, c: &ShortChallenge) -> [i16; N162] {
    let mut coeffs = c.coeffs64();
    for m in 0..N162 {
        if m % 2 == 1 {
            coeffs[m] = -coeffs[m];
        }
    }
    let q64 = q as u64;
    let theta = if quadratic_slots(q) {
        let psi = match q {
            2917 => crate::params::ParamsQ::<2917>::PSI972 as u64,
            4861 => crate::params::ParamsQ::<4861>::PSI972 as u64,
            12637 => crate::params::ParamsQ::<12637>::PSI972 as u64,
            _ => unreachable!(),
        };
        psi * psi % q64
    } else {
        let psi = match q {
            3889 => crate::params::Params::<3889>::PSI as u64,
            9721 => crate::params::Params::<9721>::PSI as u64,
            17497 => crate::params::Params::<17497>::PSI as u64,
            19441 => crate::params::Params::<19441>::PSI as u64,
            _ => unreachable!(),
        };
        crate::params::pow_mod(psi, 4, q64)
    };
    let half = (q as i64 - 1) / 2;
    let mut out = [0i16; N162];
    for s in 0..N162 {
        let v = crate::ring::POW3_SLOT_EXP[s] as u64;
        let x = crate::params::pow_mod(theta, v, q64);
        let mut acc = 0u64;
        for k in (0..N162).rev() {
            acc = (acc * x + (coeffs[k].rem_euclid(q64 as i64) as u64)) % q64;
        }
        let r = acc as i64;
        out[s] = if r > half { (r - q as i64) as i16 } else { r as i16 };
    }
    out
}

/// Slot indices for the fold of a quad-limb commitment (tests and bd).
pub fn quad_component_slot(t: usize, s: usize) -> usize {
    let base = QUAD_CLASS_SLOT[t / 2][s] as usize;
    2 * base + t % 2
}

/// The `4 * v.len()` field elements of `v mod 2`, in the witness's own index order: element
/// `4m + k` is component `k` of packed ring element `m`, its bit `p` the parity of coefficient
/// `4p + k` of that element.
pub fn components_mod_2(v: &[[i16; N]]) -> Vec<crate::binfield::F162> {
    let mut out = vec![crate::binfield::F162::ZERO; 4 * v.len()];
    for (m, e) in v.iter().enumerate() {
        for p in 0..N162 {
            for k in 0..4 {
                if e[4 * p + k] & 1 == 1 {
                    out[4 * m + k].0[p >> 6] |= 1u64 << (p & 63);
                }
            }
        }
    }
    out
}

/// `A v` for one limb, dispatched (tests / verifier).
pub fn a_times_v_of(q: u16, a: &[[i16; N]], v: &[[i16; N]]) -> Coeffs {
    if quadratic_slots(q) {
        // quadratic limb: slot product is per-leaf bilinear; use mul_quad_slots per element
        let q64 = q as u64;
        let mut acc = [0u64; N];
        for i in 0..v.len() {
            let mut coeffs = [0u32; N];
            for u in 0..N {
                coeffs[u] = ((v[i][u].rem_euclid(q as i16) as i64 as u32) % q as u32);
            }
            let t = crate::ring::ntt_of(q, &coeffs);
            let mut acoef = [0u32; N];
            for u in 0..N {
                acoef[u] = (a[i][u].rem_euclid(q as i16) as i64 as u32) % q as u32;
            }
            let prod = match q {
                2917 => crate::scalar::mul_quad_slots::<2917>(&acoef, &t),
                4861 => crate::scalar::mul_quad_slots::<4861>(&acoef, &t),
                12637 => crate::scalar::mul_quad_slots::<12637>(&acoef, &t),
                _ => unreachable!(),
            };
            for u in 0..N {
                acc[u] += prod[u] as u64;
            }
        }
        let mut out = [0u32; N];
        for u in 0..N {
            out[u] = (acc[u] % q64) as u32;
        }
        out
    } else {
        a_times_v(q, a, v)
    }
}

/// Rebuild the raw 648-row commitment of one folded chunk for one limb (tests / verifier):
/// `A v` then decompose. Returns the four `R_162` rows.
pub fn a_times_v_components(q: u16, a: &[[i16; N]], v: &[[i16; N]]) -> [PowerOfThreeRing; 4] {
    components_of_q(q, &a_times_v_of(q, a, v))
}

fn components_of_q(q: u16, y: &Coeffs) -> [PowerOfThreeRing; 4] {
    crate::ring::components_of(q, y)
}
