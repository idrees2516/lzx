//! The CRT-split lattice ring `R = Z_q[X]/(X^d + 1)` of ePrint 2026/471
//! ("Lookup Arguments over Rings", Bootle–Guskind–Patranabis–Sotiraki).
//!
//! Lemma 3.8 (via [LS17]): for a prime `q ≡ 5 (mod 8)` and every
//! power-of-two `d ≥ 4`, `X^d + 1` factors mod `q` as
//!
//! ```text
//! X^d + 1 = (X^{d/2} − r)·(X^{d/2} + r),   r² ≡ −1 (mod q),
//! ```
//!
//! with **both factors irreducible**, so
//! `R ≅ F_{q^{d/2}} × F_{q^{d/2}}` — exactly the two-integral-domain
//! split that Ring-LogUp (Lemma 5.8 / Remark 5.10) requires. The
//! CRT projection and reconstruction maps are the pair
//!
//! ```text
//! slot⁺_j = y_j + r·y_{j+d/2},   slot⁻_j = y_j − r·y_{j+d/2},
//! y_j     = (slot⁺_j + slot⁻_j)/2,   y_{j+d/2} = (slot⁺_j − slot⁻_j)/(2r).
//! ```
//!
//! Inversion in `R` is CRT inversion: invert each slot in the field
//! `F_{q^{d/2}} = Z_q[X]/(X^{d/2} ∓ r)` (polynomial extended Euclid) and
//! reconstruct. An element is invertible iff **both** slots are nonzero;
//! a single zero slot makes it a zero-divisor — the paper's attacks
//! (Section 4) live exactly there.
//!
//! **Why not `lattice-ring`?** Its NTT kernel requires `2d | q−1`, i.e.
//! `q ≡ 1 (mod 2d)` — which forces `X^d+1` to split *completely* into
//! linear factors (`t = d` CRT components). The two-component split
//! needs `v₂(q−1) = 2` exactly, i.e. `q ≡ 5 (mod 8)`, incompatible with
//! the NTT at any `d ≥ 8`. This module therefore runs a schoolbook
//! negacyclic kernel: `O(d²)` with `i128` accumulation, cheap at the
//! `d ≤ 64` the paper's parameters consider.
//!
//! The **challenge space** `C` is the set of all binary-coefficient ring
//! elements (`|C| = 2^d`): for distinct `u, v ∈ C` the difference has
//! coefficients in `{−1, 0, 1}`, so `‖u−v‖∞ = 1 ≪ q^{1/2}/√2` and
//! Lemma 3.8 makes it invertible — a *sampling space* in the sense of
//! Definition 3.5, which is what the ring Schwartz–Zippel lemma
//! (Lemma 3.6) needs. The public injective map `g: [N] → C` is the
//! binary decomposition of the index (Remark 5.9).

// Index-arithmetic kernels (negacyclic accumulation, CRT projections,
// polynomial division) read clearer with explicit indices.
#![allow(clippy::needless_range_loop)]

use lattice_core::transcript::Transcript;

/// The default split prime: 4294967197 = 2³² − 99, prime, ≡ 5 (mod 8).
pub const Q_SPLIT: u64 = 4_294_967_197;
/// A square root of −1 mod `Q_SPLIT` (r² = q−1).
pub const SQRT_M1: u64 = 983_270_775;
/// Largest supported ring degree (power of two).
pub const MAX_D: usize = 64;

/// Errors of the split-ring layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RingDError {
    DegreeNotPowerOfTwo { d: usize },
    DegreeTooLarge { d: usize },
    DegreeTooSmall { d: usize },
    LengthMismatch { expected: usize, got: usize },
    NotInvertible,
    ModulusNotPrime,
    BadSqrtM1,
    Transcript(String),
}

/// The split-ring environment `R = Z_q[X]/(X^d+1) ≅ F_{q^{d/2}} × F_{q^{d/2}}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingD {
    pub q: u64,
    pub d: usize,
    /// `r² ≡ −1 (mod q)` — the split point of `X^d+1`.
    pub r: u64,
}

impl RingD {
    /// The default environment at degree `d` (prime `Q_SPLIT`).
    pub fn new(d: usize) -> Result<Self, RingDError> {
        Self::with_prime(Q_SPLIT, SQRT_M1, d)
    }

    /// A custom split environment (tests / alternate parameters).
    pub fn with_prime(q: u64, r: u64, d: usize) -> Result<Self, RingDError> {
        if d < 4 {
            return Err(RingDError::DegreeTooSmall { d });
        }
        if !d.is_power_of_two() {
            return Err(RingDError::DegreeNotPowerOfTwo { d });
        }
        if d > MAX_D {
            return Err(RingDError::DegreeTooLarge { d });
        }
        if q % 8 != 5 {
            // The two-component split needs v2(q-1) = 2 exactly.
            return Err(RingDError::ModulusNotPrime);
        }
        if (r * r) % q != (q - 1) % q {
            return Err(RingDError::BadSqrtM1);
        }
        Ok(RingD { q, d, r })
    }

    #[inline]
    pub fn hd(&self) -> usize {
        self.d / 2
    }

    #[inline]
    fn red64(&self, x: u64) -> u64 {
        x % self.q
    }

    pub fn zero(&self) -> Elem {
        Elem { c: vec![0; self.d] }
    }

    pub fn one(&self) -> Elem {
        let mut e = self.zero();
        e.c[0] = 1;
        e
    }

    /// The generator X.
    pub fn x_gen(&self) -> Elem {
        let mut e = self.zero();
        e.c[1] = 1;
        e
    }

    pub fn constant(&self, v: u64) -> Elem {
        let mut e = self.zero();
        e.c[0] = self.red64(v);
        e
    }

    pub fn from_coeffs(&self, c: Vec<u64>) -> Result<Elem, RingDError> {
        if c.len() != self.d {
            return Err(RingDError::LengthMismatch {
                expected: self.d,
                got: c.len(),
            });
        }
        Ok(Elem {
            c: c.into_iter().map(|v| v % self.q).collect(),
        })
    }

    /// Balanced (signed) coefficients → canonical representative.
    pub fn from_signed(&self, s: &[i64]) -> Elem {
        let mut e = self.zero();
        for (i, &v) in s.iter().enumerate().take(self.d) {
            let m = ((v % self.q as i64) + self.q as i64) % self.q as i64;
            e.c[i] = m as u64;
        }
        e
    }

    /// A uniformly random element from a transcript-derived seed stream.
    pub fn random(&self, seed: &[u8]) -> Elem {
        let bytes = Transcript::xof(b"ringd-random", seed, self.d * 8);
        let mut e = self.zero();
        for i in 0..self.d {
            let mut w = [0u8; 8];
            w.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
            e.c[i] = u64::from_le_bytes(w) % self.q;
        }
        e
    }

    /// Sample a fresh challenge from `C` (binary coefficients) off a
    /// transcript — one bit per coefficient, `|C| = 2^d`.
    pub fn sample_challenge(&self, tr: &mut Transcript, label: &[u8]) -> Elem {
        let nbytes = self.d.div_ceil(8);
        let bytes = tr.challenge_bytes(label, nbytes).unwrap_or_default();
        let mut e = self.zero();
        for i in 0..self.d {
            e.c[i] = u64::from((bytes[i / 8] >> (i % 8)) & 1);
        }
        e
    }

    /// The public injective map `g: [N] → C` — binary decomposition of
    /// the index into `d` coefficients (Remark 5.9). Injective while
    /// `N ≤ 2^d = |C|`.
    pub fn g_map(&self, j: u64) -> Elem {
        let mut e = self.zero();
        for i in 0..self.d {
            e.c[i] = (j >> i) & 1;
        }
        e
    }
}

/// An element of the split ring (canonical coefficients in `[0, q)`).
#[derive(Clone, Debug)]
pub struct Elem {
    pub c: Vec<u64>,
}

impl PartialEq for Elem {
    fn eq(&self, other: &Self) -> bool {
        self.c == other.c
    }
}
impl Eq for Elem {}

impl Elem {
    pub fn coeffs(&self) -> &[u64] {
        &self.c
    }

    pub fn is_zero(&self) -> bool {
        self.c.iter().all(|&v| v == 0)
    }

    /// Degree-0 (integer-valued) element? — the integer-check predicate.
    pub fn is_integer(&self) -> bool {
        self.c[1..].iter().all(|&v| v == 0)
    }

    /// All coefficients in `{0, 1}` — membership in the challenge space C.
    pub fn is_binary(&self) -> bool {
        self.c.iter().all(|&v| v <= 1)
    }

    /// Constant coefficient (the `ct` map of the paper).
    pub fn ct(&self) -> u64 {
        self.c[0]
    }

    /// ℓ∞ norm over the balanced representative.
    pub fn inf_norm(&self, q: u64) -> u64 {
        let half = q / 2;
        self.c
            .iter()
            .map(|&v| if v <= half { v } else { q - v })
            .max()
            .unwrap_or(0)
    }
}

// ----- arithmetic -----------------------------------------------------------

impl RingD {
    pub fn add(&self, a: &Elem, b: &Elem) -> Elem {
        let mut e = self.zero();
        for i in 0..self.d {
            e.c[i] = self.red64(a.c[i] + b.c[i]);
        }
        e
    }

    pub fn sub(&self, a: &Elem, b: &Elem) -> Elem {
        let mut e = self.zero();
        for i in 0..self.d {
            e.c[i] = self.red64(a.c[i] + self.q - b.c[i] % self.q);
        }
        e
    }

    pub fn neg(&self, a: &Elem) -> Elem {
        let mut e = self.zero();
        for i in 0..self.d {
            e.c[i] = self.red64(self.q - a.c[i]);
        }
        e
    }

    /// Scalar multiplication by a small integer.
    pub fn scale(&self, a: &Elem, k: u64) -> Elem {
        let mut e = self.zero();
        for i in 0..self.d {
            e.c[i] = self.red64((a.c[i] * (k % self.q)) % self.q);
        }
        e
    }

    /// Schoolbook negacyclic product `a·b mod (X^d+1)` with `i128`
    /// accumulation (d ≤ 64, q < 2^32 ⇒ sums < 2^70).
    pub fn mul(&self, a: &Elem, b: &Elem) -> Elem {
        let d = self.d;
        let mut acc = vec![0i128; d];
        for i in 0..d {
            if a.c[i] == 0 {
                continue;
            }
            let ai = a.c[i] as i128;
            for j in 0..d {
                if b.c[j] == 0 {
                    continue;
                }
                let k = i + j;
                let term = ai * b.c[j] as i128;
                if k < d {
                    acc[k] += term;
                } else {
                    // X^d = -1: wraps to k-d with a sign flip.
                    acc[k - d] -= term;
                }
            }
        }
        let mut e = self.zero();
        for k in 0..d {
            let mut v = acc[k] % self.q as i128;
            if v < 0 {
                v += self.q as i128;
            }
            e.c[k] = v as u64;
        }
        e
    }

    pub fn pow(&self, a: &Elem, mut k: u64) -> Elem {
        let mut result = self.one();
        let mut base = a.clone();
        while k > 0 {
            if k & 1 == 1 {
                result = self.mul(&result, &base);
            }
            base = self.mul(&base, &base);
            k >>= 1;
        }
        result
    }
}

// ----- CRT split ------------------------------------------------------------

/// One CRT slot: a degree `< d/2` polynomial over `Z_q`
/// (element of the field `Z_q[X]/(X^{d/2} − r)` or `…(X^{d/2} + r)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    pub c: Vec<u64>,
}

impl RingD {
    /// Project `y` onto the `(X^{d/2} − r)` slot (`X^{d/2} ≡ +r`).
    pub fn slot_plus(&self, y: &Elem) -> Slot {
        let h = self.hd();
        let mut c = vec![0u64; h];
        for j in 0..h {
            c[j] = self.red64(y.c[j] + (self.r * (y.c[j + h] % self.q)) % self.q);
        }
        Slot { c }
    }

    /// Project `y` onto the `(X^{d/2} + r)` slot (`X^{d/2} ≡ −r`).
    pub fn slot_minus(&self, y: &Elem) -> Slot {
        let h = self.hd();
        let mut c = vec![0u64; h];
        for j in 0..h {
            c[j] = self.red64(y.c[j] + self.q - (self.r * (y.c[j + h] % self.q)) % self.q);
        }
        Slot { c }
    }

    /// CRT reconstruct from the two slots.
    pub fn from_slots(&self, plus: &Slot, minus: &Slot) -> Result<Elem, RingDError> {
        let h = self.hd();
        if plus.c.len() != h || minus.c.len() != h {
            return Err(RingDError::LengthMismatch {
                expected: h,
                got: plus.c.len(),
            });
        }
        let inv2 = mod_pow(2, self.q - 2, self.q);
        let inv2r = mod_pow((2 * self.r) % self.q, self.q - 2, self.q);
        let mut e = self.zero();
        for j in 0..h {
            // y_j = (s+ + s-)/2 ;  y_{j+h} = (s+ - s-)/(2r)
            let sp = plus.c[j];
            let sm = minus.c[j];
            e.c[j] = self.red64(((sp + sm) % self.q) * inv2 % self.q);
            let diff = self.red64(sp + self.q - sm);
            e.c[j + h] = self.red64(diff * inv2r % self.q);
        }
        Ok(e)
    }

    /// Slot multiplication mod `X^{h} − r` (the `+` slot's modulus).
    fn slot_mul_mod(&self, a: &Slot, b: &Slot, root: u64) -> Slot {
        let h = self.hd();
        let mut acc = vec![0i128; h];
        for i in 0..h {
            if a.c[i] == 0 {
                continue;
            }
            let ai = a.c[i] as i128;
            for j in 0..h {
                if b.c[j] == 0 {
                    continue;
                }
                let k = i + j;
                let term = ai * b.c[j] as i128;
                if k < h {
                    acc[k] += term;
                } else {
                    // X^h = root (r for the + slot, -r for the - slot).
                    acc[k - h] += term * root as i128;
                }
            }
        }
        let mut c = vec![0u64; h];
        for k in 0..h {
            let mut v = acc[k] % self.q as i128;
            if v < 0 {
                v += self.q as i128;
            }
            c[k] = v as u64;
        }
        Slot { c }
    }

    /// Multiply in the `+` slot's field `Z_q[X]/(X^{d/2} − r)`.
    pub fn slot_mul_plus(&self, a: &Slot, b: &Slot) -> Slot {
        self.slot_mul_mod(a, b, self.r)
    }

    /// Multiply in the `−` slot's field `Z_q[X]/(X^{d/2} + r)` (`X^h ≡ −r`).
    pub fn slot_mul_minus(&self, a: &Slot, b: &Slot) -> Slot {
        self.slot_mul_mod(a, b, self.q - self.r % self.q)
    }

    /// Invert a slot in `Z_q[X]/(X^h − root)` via polynomial extended
    /// Euclid (`q` prime ⇒ coefficient field). Returns `None` for the
    /// zero slot (a zero-divisor of the full ring). The Bezout
    /// coefficient is maintained reduced modulo the slot modulus, so
    /// degrees stay `< h` and the loop terminates in at most `h` steps
    /// (remainder degrees strictly decrease over a field).
    fn slot_inv_mod(&self, a: &Slot, root: u64) -> Option<Slot> {
        let h = self.hd();
        if a.c.iter().all(|&v| v == 0) {
            return None;
        }
        // m(x) = x^h - root (monic, degree h)
        let m = {
            let mut v = vec![0u64; h + 1];
            v[h] = 1;
            v[0] = self.red64(self.q - root % self.q);
            v
        };
        let trim = |mut v: Vec<u64>| -> Vec<u64> {
            while v.len() > 1 && *v.last().unwrap() == 0 {
                v.pop();
            }
            v
        };
        let deg = |v: &[u64]| -> usize { v.len() - 1 };
        // Euclidean walk: (r_old, s_old), (r_cur, s_cur)
        let mut r_old = m.clone();
        let mut s_old: Vec<u64> = vec![0];
        let mut r_cur = trim(a.c.clone());
        let mut s_cur: Vec<u64> = vec![1];
        while !r_cur.iter().all(|&v| v == 0) {
            let (q, rem) = self.poly_divmod(&r_old, deg(&r_cur), &r_cur);
            let qs = self.poly_mul(&q, &s_cur);
            let qs_red = self.poly_rem(&qs, &m);
            let mut next_s = self.poly_sub(&s_old, &qs_red);
            next_s = trim(self.poly_rem(&next_s, &m));
            r_old = r_cur;
            s_old = s_cur;
            r_cur = trim(rem);
            s_cur = next_s;
        }
        // gcd = r_old; must be a nonzero constant (m irreducible)
        if r_old.len() != 1 || r_old[0] == 0 {
            return None;
        }
        let inv_const = mod_pow(r_old[0], self.q - 2, self.q);
        let mut s = s_old;
        for c in s.iter_mut() {
            *c = self.red64(*c * inv_const % self.q);
        }
        s = trim(s);
        s.resize(h, 0);
        Some(Slot { c: s })
    }

    /// Schoolbook polynomial product over Z_q.
    fn poly_mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        if a.iter().all(|&v| v == 0) || b.iter().all(|&v| v == 0) {
            return vec![0];
        }
        let mut acc = vec![0i128; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            if ai == 0 {
                continue;
            }
            for (j, &bj) in b.iter().enumerate() {
                acc[i + j] += ai as i128 * bj as i128;
            }
        }
        acc.into_iter()
            .map(|v| {
                let m = v % self.q as i128;
                (if m < 0 { m + self.q as i128 } else { m }) as u64
            })
            .collect()
    }

    /// Polynomial difference over Z_q (result padded to `a`'s length).
    fn poly_sub(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let n = a.len().max(b.len());
        let mut out = vec![0u64; n];
        for i in 0..n {
            let x = a.get(i).copied().unwrap_or(0) % self.q;
            let y = b.get(i).copied().unwrap_or(0) % self.q;
            out[i] = self.red64(x + self.q - y);
        }
        out
    }

    /// Polynomial reduction `a mod m` (m monic after normalization).
    fn poly_rem(&self, a: &[u64], m: &[u64]) -> Vec<u64> {
        if a.iter().all(|&v| v == 0) {
            return vec![0];
        }
        let md = m.len() - 1;
        if md == 0 {
            return vec![0];
        }
        let lc_inv = mod_pow(m[md], self.q - 2, self.q);
        let mut r = {
            let mut v = vec![0u64; a.len().max(m.len())];
            for (i, &x) in a.iter().enumerate() {
                v[i] = x % self.q;
            }
            v
        };
        let mut dr = r.len() - 1;
        loop {
            while dr > 0 && r[dr] == 0 {
                dr -= 1;
            }
            if dr < md || (dr == 0 && md > 0) {
                break;
            }
            if r[dr] == 0 {
                break;
            }
            let factor = self.red64(r[dr] * lc_inv % self.q);
            if factor == 0 {
                break;
            }
            for i in 0..=md {
                let idx = dr - md + i;
                let sub = self.red64(factor * m[i] % self.q);
                r[idx] = self.red64(r[idx] + self.q - sub);
            }
            r[dr] = 0;
        }
        r.truncate(md);
        let mut out = vec![0u64; md];
        for (i, &v) in r.iter().enumerate() {
            if i < md {
                out[i] = v;
            }
        }
        out
    }

    /// Polynomial division over Z_q. `den` must be nonzero; returns
    /// `(quotient, remainder)` with `num = q·den + rem`, `deg rem < deg den`.
    fn poly_divmod(&self, num: &[u64], den_deg: usize, den: &[u64]) -> (Vec<u64>, Vec<u64>) {
        if den.iter().all(|&v| v == 0) || (den.len() == 1 && den[0] == 0) {
            return (vec![0], num.to_vec());
        }
        if den_deg == 0 {
            // constant divisor: scale everything, zero remainder
            let c_inv = mod_pow(den[0], self.q - 2, self.q);
            let q: Vec<u64> = num
                .iter()
                .map(|&v| self.red64(v * c_inv % self.q))
                .collect();
            return (q, vec![0]);
        }
        let lc_inv = mod_pow(den[den_deg], self.q - 2, self.q);
        let mut r: Vec<u64> = num.iter().map(|&v| v % self.q).collect();
        let mut q = vec![0u64; num.len()];
        let mut dr = {
            let mut d = r.len() - 1;
            while d > 0 && r[d] == 0 {
                d -= 1;
            }
            d
        };
        while dr >= den_deg {
            if r[dr] == 0 {
                if dr == 0 {
                    break;
                }
                dr -= 1;
                continue;
            }
            let factor = self.red64(r[dr] * lc_inv % self.q);
            q[dr - den_deg] = factor;
            for i in 0..=den_deg {
                let idx = dr - den_deg + i;
                let sub = self.red64(factor * den[i] % self.q);
                r[idx] = self.red64(r[idx] + self.q - sub);
            }
            r[dr] = 0;
            if dr == 0 {
                break;
            }
            dr -= 1;
        }
        let rem: Vec<u64> = r.iter().take(den_deg).copied().collect();
        (q, rem)
    }

    pub fn inv(&self, a: &Elem) -> Option<Elem> {
        let sp = self.slot_plus(a);
        let sm = self.slot_minus(a);
        let ip = self.slot_inv_mod(&sp, self.r)?;
        let im = self.slot_inv_mod(&sm, self.q - self.r % self.q)?;
        self.from_slots(&ip, &im).ok()
    }

    /// Multiplicative unit test.
    pub fn is_unit(&self, a: &Elem) -> bool {
        self.inv(a).is_some()
    }

    /// Montgomery-style batch inversion over a vector (one inversion,
    /// two passes); entries that hit a zero-divisor abort as `None`.
    /// `out[i] = prefix[i-1] · inv(prefix[i])` walking down, where
    /// `prefix[i] = ∏_{j≤i} v_j` — the standard trick.
    pub fn batch_inv(&self, vals: &[Elem]) -> Option<Vec<Elem>> {
        let n = vals.len();
        if n == 0 {
            return Some(Vec::new());
        }
        let mut prefix = vec![self.one(); n];
        let mut acc = self.one();
        for i in 0..n {
            acc = self.mul(&acc, &vals[i]);
            prefix[i] = acc.clone();
        }
        let inv_all = self.inv(&acc)?;
        let mut out = vec![self.zero(); n];
        let mut running = inv_all;
        for i in (0..n).rev() {
            if i > 0 {
                out[i] = self.mul(&prefix[i - 1], &running);
            } else {
                out[i] = running.clone();
            }
            running = self.mul(&running, &vals[i]);
        }
        // fail-closed: every unit entry must verify
        let one = self.one();
        for i in 0..n {
            if vals[i].is_zero() {
                continue;
            }
            if self.mul(&vals[i], &out[i]) != one {
                return None;
            }
        }
        Some(out)
    }
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result as u128 * base as u128 % modulus as u128) as u64;
        }
        base = (base as u128 * base as u128 % modulus as u128) as u64;
        exp >>= 1;
    }
    result
}

// ----- MLE machinery --------------------------------------------------------

impl RingD {
    /// The EQ row over the ring: `EQ(k, x)` for all `k ∈ {0,1}^μ`, as a
    /// length-`2^μ` vector of ring elements. Computed by tensor doubling:
    /// row `k` = ∏_i (x_i if k_i else (1−x_i)).
    pub fn eq_row(&self, point: &[Elem]) -> Vec<Elem> {
        let mu = point.len();
        let mut row = vec![self.one()];
        for xi in point {
            let one_minus = self.sub(&self.one(), xi);
            let mut next = Vec::with_capacity(row.len() * 2);
            for e in &row {
                next.push(self.mul(e, &one_minus));
            }
            for e in &row {
                next.push(self.mul(e, xi));
            }
            row = next;
        }
        let _ = mu;
        row
    }

    /// The transposed EQ: `EQ(x, k)` evaluated as a function of the cube
    /// index — same table (EQ is symmetric in its two arguments).
    pub fn eq_col(&self, point: &[Elem]) -> Vec<Elem> {
        self.eq_row(point)
    }

    /// Evaluate the multilinear extension of `evals` (length `2^μ`,
    /// little-endian hypercube indexing) at `point` (length μ) by the
    /// standard fold: halve with `(1−x_j)·low + x_j·high`, consuming
    /// variables from the most significant bit down — i.e. iterating
    /// the point in reverse.
    pub fn mle_eval(&self, evals: &[Elem], point: &[Elem]) -> Result<Elem, RingDError> {
        let mu = point.len();
        if evals.len() != (1usize << mu) {
            return Err(RingDError::LengthMismatch {
                expected: 1 << mu,
                got: evals.len(),
            });
        }
        let mut cur = evals.to_vec();
        for xi in point.iter().rev() {
            let one_minus = self.sub(&self.one(), xi);
            let half = cur.len() / 2;
            let mut next = Vec::with_capacity(half);
            for i in 0..half {
                let lo = self.mul(&cur[i], &one_minus);
                let hi = self.mul(&cur[i + half], xi);
                next.push(self.add(&lo, &hi));
            }
            cur = next;
        }
        Ok(cur[0].clone())
    }

    /// Tensor product `⊗(1−r_j, r_j)` — the verifier-side coefficient
    /// vector of an MLE evaluation, in `O(μ)` ring work (the Greyhound
    /// `a`/`b` vectors; see `windowed.rs`).
    pub fn tensor_eq(&self, point: &[Elem]) -> Vec<Elem> {
        self.eq_row(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(d: usize) -> RingD {
        RingD::new(d).ok().unwrap()
    }

    #[test]
    fn ring_axioms() {
        for d in [4usize, 8, 16] {
            let r = ring(d);
            let a = r.random(b"a");
            let b = r.random(b"b");
            let c = r.random(b"c");
            // commutative ring axioms
            assert_eq!(r.add(&a, &b), r.add(&b, &a));
            assert_eq!(r.mul(&a, &b), r.mul(&b, &a));
            assert_eq!(r.mul(&r.mul(&a, &b), &c), r.mul(&a, &r.mul(&b, &c)));
            let lhs = r.mul(&a, &r.add(&b, &c));
            let rhs = r.add(&r.mul(&a, &b), &r.mul(&a, &c));
            assert_eq!(lhs, rhs);
            assert!(r.add(&a, &r.neg(&a)).is_zero());
            // X^d = -1
            let x = r.x_gen();
            let xd = r.pow(&x, d as u64);
            assert_eq!(xd, r.neg(&r.one()));
        }
    }

    #[test]
    fn crt_split_roundtrip() {
        for d in [4usize, 8, 16, 32] {
            let r = ring(d);
            let y = r.random(b"y");
            let sp = r.slot_plus(&y);
            let sm = r.slot_minus(&y);
            let back = r.from_slots(&sp, &sm).ok().unwrap();
            assert_eq!(back, y);
        }
    }

    #[test]
    fn slot_multiplication_matches_full_product() {
        // CRT is a ring homomorphism: (a*b) projected == slot products.
        let r = ring(8);
        let a = r.random(b"ca");
        let b = r.random(b"cb");
        let prod = r.mul(&a, &b);
        let sp = r.slot_mul_plus(&r.slot_plus(&a), &r.slot_plus(&b));
        let sm = r.slot_mul_minus(&r.slot_minus(&a), &r.slot_minus(&b));
        let back = r.from_slots(&sp, &sm).ok().unwrap();
        assert_eq!(back, prod);
    }

    #[test]
    fn inversion_via_slots() {
        let r = ring(8);
        for i in 0..8 {
            let a = r.random(format!("inv-{i}").as_bytes());
            if let Some(inv) = r.inv(&a) {
                let one = r.mul(&a, &inv);
                assert_eq!(one, r.one());
            }
        }
        // zero is not invertible; a pure-slot element is a zero divisor
        assert!(r.inv(&r.zero()).is_none());
        // an element that is zero only in the + slot: construct via slots
        let h = r.hd();
        let zero_plus = Slot { c: vec![0; h] };
        let nonzero_minus = Slot {
            c: (0..h).map(|j| (j as u64 * 37 + 5) % r.q).collect(),
        };
        let zd_elem = r.from_slots(&zero_plus, &nonzero_minus).ok().unwrap();
        assert!(!zd_elem.is_zero());
        assert!(r.inv(&zd_elem).is_none(), "zero-divisor must not invert");
    }

    #[test]
    fn batch_inversion_consistency() {
        let r = ring(8);
        let vals: Vec<Elem> = (0..7)
            .map(|i| r.random(format!("bv-{i}").as_bytes()))
            .collect();
        let invs = r.batch_inv(&vals).unwrap();
        for (v, inv) in vals.iter().zip(invs.iter()) {
            assert_eq!(r.mul(v, inv), r.one());
        }
    }

    #[test]
    fn challenge_space_is_sampling() {
        // Distinct binary-coefficient elements have invertible pairwise
        // differences (Lemma 3.8: ||u-v||_\infty = 1 < q^{1/2}/sqrt(2)).
        let r = ring(8);
        for i in 0..24u64 {
            for j in (i + 1)..24u64 {
                let u = r.g_map(i);
                let v = r.g_map(j);
                let diff = r.sub(&u, &v);
                assert!(!diff.is_zero());
                assert!(r.is_unit(&diff), "g({i}) - g({j}) must be a unit");
            }
        }
        // g is injective on the range tested
        assert_ne!(r.g_map(3), r.g_map(5));
        // challenges sampled off a transcript are binary
        let mut tr = Transcript::new_default(b"chal");
        let c = r.sample_challenge(&mut tr, b"c1");
        assert!(c.is_binary());
        let c2 = r.sample_challenge(&mut tr, b"c2");
        assert!(c2.is_binary());
    }

    #[test]
    fn mle_eval_matches_eq_expansion() {
        let r = ring(4);
        let mu = 3;
        let evals: Vec<Elem> = (0..(1 << mu))
            .map(|i| r.random(format!("m{i}").as_bytes()))
            .collect();
        let point: Vec<Elem> = (0..mu)
            .map(|i| {
                r.sample_challenge(
                    &mut Transcript::new_default(b"p"),
                    format!("x{i}").as_bytes(),
                )
            })
            .collect();
        // evaluation via fold
        let v1 = r.mle_eval(&evals, &point).ok().unwrap();
        // evaluation via explicit EQ row
        let row = r.eq_row(&point);
        let mut acc = r.zero();
        for (k, w) in row.iter().enumerate() {
            acc = r.add(&acc, &r.mul(w, &evals[k]));
        }
        assert_eq!(v1, acc);
        // EQ at cube points: EQ(k, point) with k = point's binary rep at
        // cube point (1,1,...,1) picks evals[7]
        let ones = vec![r.one(); mu];
        let row2 = r.eq_row(&ones);
        // EQ(1^mu, x) at x = ones is 1
        assert_eq!(row2[7], r.one());
    }

    #[test]
    fn integer_and_binary_predicates() {
        let r = ring(4);
        assert!(r.constant(17).is_integer());
        assert!(!r.x_gen().is_integer());
        assert!(r.g_map(5).is_binary());
        assert!(!r.constant(2).is_binary());
    }

    #[test]
    fn degree_validation() {
        assert!(RingD::new(3).is_err());
        assert!(RingD::new(6).is_err());
        assert!(RingD::new(128).is_err());
        // q must be 5 mod 8 for the two-way split
        assert!(RingD::with_prime(15, 2, 8).is_err());
    }
}
