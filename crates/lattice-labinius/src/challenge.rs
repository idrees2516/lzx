//! Short challenges over `R_162 = Z[Z]/Phi_243(Z)` and the Fiat-Shamir transcript that samples
//! them. Port of `labinius` `challenge.rs`.
//!
//! Upstream's transcript is blake3; this port uses the LZX stack's SHAKE-256 (KeccakSponge)
//! with the *same* structure — absorb length-prefixed, derive by cloning the state and appending
//! a sample counter and a label — so two derivations from one transcript stay independent and
//! every derivation is bound to everything absorbed before it. The per-candidate signs, which
//! upstream keys with `blake3 Hasher::new_derive_key`, are a keyed SHA3-256 here.
//!
//! A challenge is a weight-`w` *signed* element of `R_162`: `w` of the 162 coefficients are
//! `+-1`, the rest zero. Sampling is a partial Fisher-Yates over positions; each candidate is
//! signed and then rejected until it is short in the canonical embedding:
//! `max_u |c(zeta^u)|^2 <= bound^2` over the 162 primitive 243-rd roots of unity. The power
//! basis of a power-of-three conductor is not orthogonal there, so the coefficient expansion
//! factor is `sqrt(3) * bound` (`sqrt(3) * 12 = 20.78` at the default `weight = 28, bound = 12`).

use crate::binfield::{B128, F162};
use crate::ring::{PowerOfThreeRing, N162};
use lattice_core::keccak::{sha3_256, KeccakSponge};
use std::f64::consts::PI;

/// Conductor of the small ring.
pub const CONDUCTOR243: usize = 243;
/// Largest weight a [`ShortChallenge`] can hold.
pub const MAX_WEIGHT: usize = 32;
/// The default sampling weight.
pub const DEFAULT_WEIGHT: usize = 28;
/// The default canonical-embedding bound.
pub const DEFAULT_BOUND: f64 = 12.0;

// =============================================================================================
// transcript
// =============================================================================================

/// A SHAKE-256 Fiat-Shamir transcript with upstream's absorb/derive discipline.
#[derive(Clone)]
pub struct Transcript {
    state: KeccakSponge,
    counter: u64,
}

impl Transcript {
    pub fn new(domain: &[u8]) -> Self {
        let mut state = KeccakSponge::new_shake256();
        state.update(&(domain.len() as u64).to_le_bytes());
        state.update(domain);
        Transcript { state, counter: 0 }
    }

    /// Absorb raw bytes, length-prefixed.
    pub fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.state.update(&(bytes.len() as u64).to_le_bytes());
        self.state.update(bytes);
    }

    pub fn absorb_u64(&mut self, x: u64) {
        self.state.update(&x.to_le_bytes());
    }

    /// Absorb `F162` elements: 24 little-endian bytes each, length-prefixed.
    pub fn absorb_f162(&mut self, xs: &[F162]) {
        self.state.update(&(xs.len() as u64).to_le_bytes());
        for x in xs {
            self.state.update(&x.to_le24());
        }
    }

    pub fn absorb_b128(&mut self, xs: &[B128]) {
        self.state.update(&(xs.len() as u64).to_le_bytes());
        for x in xs {
            self.state.update(&x.0.to_le_bytes());
        }
    }

    /// Absorb `R_162` slot elements: raw little-endian i16 slots, limb after limb.
    pub fn absorb_elements(&mut self, elements: &[PowerOfThreeRing]) {
        self.state.update(&(elements.len() as u64).to_le_bytes());
        let mut buf = [0u8; 2 * N162];
        for e in elements {
            for &x in e.v.iter() {
                buf[..2].copy_from_slice(&x.to_le_bytes());
                self.state.update(&buf[..2]);
            }
        }
    }

    /// One XOF derivation under `label`, advancing the sample counter.
    pub fn fill(&mut self, label: &[u8], out: &mut [u8]) {
        let mut st = self.state.clone();
        st.update(&self.counter.to_le_bytes());
        st.update(&(label.len() as u64).to_le_bytes());
        st.update(label);
        self.counter += 1;
        let bytes = st.finalize(out.len());
        out.copy_from_slice(&bytes);
    }

    /// A streaming XOF derivation (for samplers that consume an unknown number of bytes).
    fn reader(&mut self, label: &[u8]) -> Xof {
        let mut st = self.state.clone();
        st.update(&self.counter.to_le_bytes());
        st.update(&(label.len() as u64).to_le_bytes());
        st.update(label);
        self.counter += 1;
        st.finalize_in_place();
        Xof {
            sponge: st,
            buf: [0u8; 136],
            pos: 136,
        }
    }

    /// `n` uniform `F162` (24 bytes each, top 30 bits of limb 2 cleared).
    pub fn sample_f162(&mut self, label: &[u8], n: usize) -> Vec<F162> {
        let mut bytes = vec![0u8; 24 * n];
        self.fill(label, &mut bytes);
        bytes.chunks(24).map(F162::from_le24).collect()
    }
}

/// A buffered view of one XOF derivation (SHAKE-256 raw squeeze).
pub struct Xof {
    sponge: KeccakSponge,
    buf: [u8; 136],
    pos: usize,
}

impl Xof {
    fn refill(&mut self) {
        self.sponge.squeeze(&mut self.buf);
        self.pos = 0;
    }

    pub fn byte(&mut self) -> u8 {
        if self.pos == self.buf.len() {
            self.refill();
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        b
    }

    pub fn u16le(&mut self) -> u16 {
        u16::from_le_bytes([self.byte(), self.byte()])
    }

    /// Unbiased uniform value below `n <= 162`.
    pub fn below(&mut self, n: u16) -> u16 {
        let limit = ((u16::MAX as u32 + 1) / n as u32) * n as u32;
        loop {
            let r = self.u16le() as u32;
            if r < limit {
                return (r % n as u32) as u16;
            }
        }
    }
}

// =============================================================================================
// the challenge
// =============================================================================================

/// A weight-`w` signed element of `R_162`: coefficient `positions[i]` is `-1` when bit `i` of
/// `signs` is set and `+1` otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShortChallenge {
    pub positions: [u8; MAX_WEIGHT],
    pub signs: u32,
    pub weight: usize,
}

impl ShortChallenge {
    pub fn zero() -> Self {
        ShortChallenge {
            positions: [0u8; MAX_WEIGHT],
            signs: 0,
            weight: 0,
        }
    }

    /// The dense signed coefficient vector.
    pub fn coeffs(&self) -> [i8; N162] {
        let mut c = [0i8; N162];
        for i in 0..self.weight {
            c[self.positions[i] as usize] = 1 - 2 * ((self.signs >> i) & 1) as i8;
        }
        c
    }

    /// The i64 coefficient vector (S-basis).
    pub fn coeffs64(&self) -> [i64; N162] {
        let mut c = [0i64; N162];
        for i in 0..self.weight {
            c[self.positions[i] as usize] = 1 - 2 * ((self.signs >> i) & 1) as i64;
        }
        c
    }

    /// Keyed signs of a candidate position set: a public map, independent of the transcript.
    pub fn signed(&self) -> Self {
        let mut h = Vec::with_capacity(8 + MAX_WEIGHT);
        h.extend_from_slice(b"labinius 2026 challenge signs v1");
        h.extend_from_slice(&(self.weight as u64).to_le_bytes());
        h.extend_from_slice(&self.positions[..self.weight]);
        let digest = sha3_256(&h);
        let bits = u32::from_le_bytes(digest[..4].try_into().unwrap());
        let mask = if self.weight >= 32 {
            u32::MAX
        } else {
            (1u32 << self.weight) - 1
        };
        ShortChallenge {
            signs: bits & mask,
            ..*self
        }
    }

    /// The challenge modulo 2, as an `F162`: a bit at each of its positions (`+-1` agree mod 2).
    pub fn to_f162(&self) -> F162 {
        let mut x = F162::ZERO;
        for i in 0..self.weight {
            let p = self.positions[i] as usize;
            x.0[p >> 6] |= 1u64 << (p & 63);
        }
        x
    }

    /// The sparse form of a signed coefficient vector (entries in `{-1,0,1}`).
    pub fn from_coeffs(c: &[i8; N162]) -> Self {
        let mut out = Self::zero();
        for (p, &x) in c.iter().enumerate() {
            if x != 0 {
                assert!(x == 1 || x == -1, "coefficient {p} is not signed binary");
                assert!(out.weight < MAX_WEIGHT, "weight exceeds MAX_WEIGHT");
                out.positions[out.weight] = p as u8;
                out.signs |= ((x < 0) as u32) << out.weight;
                out.weight += 1;
            }
        }
        out
    }

    /// `log2` of the number of weight-`w` challenges.
    pub fn log2_cardinality(weight: usize) -> f64 {
        let mut bits = 0.0f64;
        for i in 0..weight {
            bits += ((N162 - i) as f64 / (i + 1) as f64).log2();
        }
        bits
    }
}

// =============================================================================================
// the canonical embedding
// =============================================================================================

/// The 162 units `u mod 243` in increasing order.
pub fn units() -> [u16; N162] {
    let mut u = [0u16; N162];
    let mut n = 0;
    for x in 1..CONDUCTOR243 {
        if !x.is_multiple_of(3) {
            u[n] = x as u16;
            n += 1;
        }
    }
    u
}

/// `max_u |c(zeta^u)|^2` over the 162 primitive 243-rd roots — the squared sup norm of the
/// canonical embedding. Scalar Horner evaluation over all roots (upstream's blocked AVX-512
/// version computes the same quantity from the nonzero terms only).
pub fn canonical_inf_norm_sq(c: &ShortChallenge) -> f64 {
    let coeffs = c.coeffs();
    let mut best = 0.0f64;
    for &u in units().iter() {
        let angle = 2.0 * PI * (u as f64) / (CONDUCTOR243 as f64);
        let (zr, zi) = (angle.cos(), angle.sin());
        let (mut ar, mut ai) = (0.0f64, 0.0f64);
        for p in (0..N162).rev() {
            let (nr, ni) = (ar * zr - ai * zi, ar * zi + ai * zr);
            ar = nr + coeffs[p] as f64;
            ai = ni;
        }
        let m = ar * ar + ai * ai;
        if m > best {
            best = m;
        }
    }
    best
}

fn within(c: &ShortChallenge, bound_sq: f64) -> bool {
    canonical_inf_norm_sq(c) <= bound_sq
}

// =============================================================================================
// sampling
// =============================================================================================

/// One uniform weight-`w` position set, unsigned: a partial Fisher-Yates, then sorted.
fn attempt(x: &mut Xof, weight: usize, perm: &mut [u8; N162]) -> ShortChallenge {
    assert!(weight <= MAX_WEIGHT); // MAX_WEIGHT <= N162 by construction
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i as u8;
    }
    for i in 0..weight {
        let j = i + x.below((N162 - i) as u16) as usize;
        perm.swap(i, j);
    }
    let mut positions: [u8; MAX_WEIGHT] = [0; MAX_WEIGHT];
    let mut set = [false; N162];
    for &p in &perm[..weight] {
        set[p as usize] = true;
    }
    let mut n = 0;
    for (p, &s) in set.iter().enumerate() {
        if s {
            positions[n] = p as u8;
            n += 1;
        }
    }
    ShortChallenge {
        positions,
        signs: 0,
        weight,
    }
}

/// Rejection-sample a weight-`w` challenge with `canonical_inf_norm_sq <= bound^2`, returning
/// it with the number of attempts. All attempts read one XOF derivation of the transcript.
pub fn sample_short_challenge(
    t: &mut Transcript,
    weight: usize,
    bound: f64,
) -> (ShortChallenge, u64) {
    let bound_sq = bound * bound + 1e-12;
    let mut x = t.reader(b"short-challenge");
    let mut perm = [0u8; N162];
    let mut attempts = 0u64;
    loop {
        attempts += 1;
        let c = attempt(&mut x, weight, &mut perm).signed();
        if within(&c, bound_sq) {
            return (c, attempts);
        }
    }
}

/// One unsigned attempt (tests).
pub fn sample_attempt(t: &mut Transcript, weight: usize) -> ShortChallenge {
    let mut x = t.reader(b"short-challenge");
    let mut perm = [0u8; N162];
    attempt(&mut x, weight, &mut perm)
}
