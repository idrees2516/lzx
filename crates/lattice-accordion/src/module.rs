//! The lattice module layer: the field `F_q` with `q = 2^50 − 2687` (the
//! workspace's 50-bit modulus class), module points `M = (R_q)^{rows}` as
//! coefficient vectors, the seeded Ajtai SRS `[G | P]` whose columns are the
//! paper's "generators", the digit-layer regime that keeps committed vectors
//! short, and the Module-SIS kernel utilities.
//!
//! ## Why the same prime for field and module
//!
//! The paper stresses `|G| = |F| = p`: the group must admit scalar
//! multiplication by challenge-field elements, i.e. the module must be an
//! `F`-vector space. Over lattices this forces the challenge field to be the
//! commitment ring's base field, so both live over
//! `q = 2^50 − 2687` (`lattice_ring::modulus50::Q_50`) — prime, with
//! `−log₂(soundness-per-round) ≈ 48` for degree-2 module sumchecks, matching
//! the Goldilocks-class margins used across this workspace.
//!
//! The module `M` is only ever used with its `F_q`-vector-space structure
//! (addition + field-scalar multiplication) — the accordion protocols never
//! multiply two module elements — so a module point is represented directly
//! as its coefficient vector `[Fq; rows * d]`. The *derivation* of the SRS
//! columns treats each column as `rows` ring elements of the negacyclic ring
//! `R_q = F_q[X]/(X^d + 1)` (uniform coefficients from a seeded XOF), so the
//! underlying Module-SIS instance retains the negacyclic block structure
//! even though the protocol arithmetic never needs the ring product.

use lattice_core::transcript::Transcript;
use lattice_ring::modulus50::Modulus50;
pub use lattice_ring::modulus50::Q_50;

/// Digit-window width of the layered commitment regime: 16-bit unsigned
/// windows, so one `F_q` value (< 2^50) splits into at most 4 layers.
pub const ACCORDION_DIGIT_BITS: u32 = 16;
/// Number of digit layers in the layered cube (a power of two; the top
/// layers stay zero for values below 2^48 and partially used up to 2^50).
pub const ACCORDION_LAYERS: usize = 4;

/// The accordion field: `F_q` with `q = 2^50 − 2687`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fq(pub u64);

impl Fq {
    pub const ZERO: Fq = Fq(0);
    pub const ONE: Fq = Fq(1);

    pub fn from_u64(x: u64) -> Self {
        Fq(Modulus50::Q_50.reduce_u64(x))
    }

    pub fn add(&self, other: &Self) -> Self {
        Fq(Modulus50::Q_50.add(self.0, other.0))
    }

    pub fn sub(&self, other: &Self) -> Self {
        Fq(Modulus50::Q_50.sub(self.0, other.0))
    }

    pub fn mul(&self, other: &Self) -> Self {
        Fq(Modulus50::Q_50.mul(self.0, other.0))
    }

    pub fn neg(&self) -> Self {
        Fq(Modulus50::Q_50.neg(self.0))
    }

    pub fn pow(&self, exp: u64) -> Self {
        let mut acc = Fq::ONE;
        let mut base = *self;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base);
            }
            base = base.mul(&base);
            e >>= 1;
        }
        acc
    }

    /// Multiplicative inverse (Fermat; `q` prime, returns `None` at zero).
    pub fn inv(&self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }
        Some(self.pow(Q_50 - 2))
    }

    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Serialize (little-endian, 7 bytes carry ≥ 50 bits) for transcripts.
    pub fn to_bytes(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Draw a uniform field element from the transcript (7 bytes masked to
    /// 50 bits with rejection — bias below 2^-37).
    pub fn challenge(transcript: &mut Transcript, label: &[u8]) -> Result<Self, String> {
        let mut counter = 0u32;
        loop {
            let bytes = transcript
                .challenge_bytes(label, 8)
                .map_err(|e| e.to_string())?;
            let mut cand = [0u8; 8];
            cand[..7].copy_from_slice(&bytes[..7]);
            let v = u64::from_le_bytes(cand) & ((1u64 << 50) - 1);
            if v < Q_50 {
                return Ok(Fq(v));
            }
            counter += 1;
            if counter > 64 {
                return Err("challenge rejection loop exhausted".into());
            }
        }
    }
}

/// A point of the module `(R_q)^{rows}` — the concatenated coefficient
/// vector of its `rows` ring elements (length `rows * d`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModulePoint(pub Vec<Fq>);

impl ModulePoint {
    pub fn zero(dim: usize) -> Self {
        ModulePoint(vec![Fq::ZERO; dim])
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    pub fn add(&self, other: &Self) -> Self {
        ModulePoint(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| a.add(b))
                .collect(),
        )
    }

    pub fn sub(&self, other: &Self) -> Self {
        ModulePoint(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| a.sub(b))
                .collect(),
        )
    }

    /// `self + c * other` (the fused linear-accumulate kernel).
    pub fn axpy(&self, c: &Fq, other: &Self) -> Self {
        ModulePoint(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| a.add(&b.mul(c)))
                .collect(),
        )
    }

    pub fn scale(&self, c: &Fq) -> Self {
        ModulePoint(self.0.iter().map(|a| a.mul(c)).collect())
    }

    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|a| a.0 == 0)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.0.len() * 8);
        for g in &self.0 {
            out.extend_from_slice(&g.0.to_le_bytes());
        }
        out
    }
}

/// The structured public parameters / SRS of the lattice ml-PCS:
/// `generator_count` module columns `G₀ … G_{N−1}` plus the value column
/// `P`, each column being `rows` ring elements of
/// `R_q = F_q[X]/(X^d + 1)` with uniform coefficients derived from a seed.
///
/// Column `N` is `P` — the paper's extra generator carrying the claimed
/// evaluation value through `P' = α·P`.
#[derive(Clone)]
pub struct Srs {
    rows: usize,
    ring_degree: usize,
    /// Column-major module points; the last entry is the value column `P`.
    columns: Vec<ModulePoint>,
}

impl std::fmt::Debug for Srs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Srs")
            .field("rows", &self.rows)
            .field("ring_degree", &self.ring_degree)
            .field("generators", &self.generator_count())
            .finish()
    }
}

impl Srs {
    /// Derive the SRS from a seed. `generator_count = N` is the layered-cube
    /// size; one additional column `P` is appended.
    pub fn from_seed(rows: usize, ring_degree: usize, generator_count: usize, seed: &[u8]) -> Self {
        let mut columns = Vec::with_capacity(generator_count + 1);
        for c in 0..generator_count + 1 {
            columns.push(ModulePoint(Self::uniform_column(
                b"accordion-srs",
                seed,
                c,
                rows,
                ring_degree,
            )));
        }
        Srs {
            rows,
            ring_degree,
            columns,
        }
    }

    /// One uniform column: `rows * ring_degree` coefficients in `[0, q)`,
    /// derived by XOF expansion with exact rejection sampling.
    fn uniform_column(domain: &[u8], seed: &[u8], index: usize, rows: usize, d: usize) -> Vec<Fq> {
        let mut salt = Vec::with_capacity(domain.len() + seed.len() + 8);
        salt.extend_from_slice(domain);
        salt.extend_from_slice(seed);
        salt.extend_from_slice(&(index as u64).to_le_bytes());
        let need = rows * d;
        let mut coeffs = Vec::with_capacity(need);
        let mut counter = 0u32;
        while coeffs.len() < need {
            let bytes = Transcript::xof(b"uniform-col", &salt, 8 + counter as usize * 8 + 8);
            let mut off = bytes.len() - 8;
            while coeffs.len() < need && off >= 8 {
                off -= 8;
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes[off..off + 8]);
                let v = u64::from_le_bytes(arr) & ((1u64 << 50) - 1);
                if v < Q_50 {
                    coeffs.push(Fq(v));
                }
            }
            counter += 1;
        }
        coeffs
    }

    /// Construct an SRS from pre-built columns (test hook for crafted
    /// Module-SIS relations). The last column is the value column `P`.
    pub fn from_columns(rows: usize, ring_degree: usize, columns: Vec<ModulePoint>) -> Self {
        Srs {
            rows,
            ring_degree,
            columns,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn ring_degree(&self) -> usize {
        self.ring_degree
    }

    pub fn generator_count(&self) -> usize {
        self.columns.len() - 1
    }

    pub fn module_dim(&self) -> usize {
        self.rows * self.ring_degree
    }

    /// The `b`-th generator `G_b` (a module point).
    pub fn generator(&self, b: usize) -> &ModulePoint {
        &self.columns[b]
    }

    /// The value column `P`.
    pub fn value_column(&self) -> &ModulePoint {
        &self.columns[self.columns.len() - 1]
    }

    /// Commit a short scalar vector `w` (length = generator count):
    /// `cm = Σ_b w_b G_b`. Binding holds for short `w` under Module-SIS on
    /// the structured matrix.
    pub fn commit_scalars(&self, w: &[Fq]) -> ModulePoint {
        assert_eq!(w.len(), self.generator_count(), "witness length vs SRS");
        let mut acc = ModulePoint::zero(self.module_dim());
        for (b, &wb) in w.iter().enumerate() {
            if wb.0 == 0 {
                continue;
            }
            for (a, &cj) in acc.0.iter_mut().zip(self.columns[b].0.iter()) {
                *a = a.add(&cj.mul(&wb));
            }
        }
        acc
    }

    /// `cm + v·P` — the reduce protocol's sumcheck target with `P' = α·P`.
    pub fn commit_with_value(&self, w: &[Fq], v: &Fq) -> ModulePoint {
        let mut cm = self.commit_scalars(w);
        let p = self.value_column();
        for (a, &pj) in cm.0.iter_mut().zip(p.0.iter()) {
            *a = a.add(&pj.mul(v));
        }
        cm
    }

    /// Evaluate the generator multilinear extension `Ĝ` at a field point `r`
    /// over the layered cube: `Ĝ(r) = Σ_b eq(b, r)·G_b` — the lattice
    /// analogue of the paper's `O(n)` verifier MSM (paper note §"decide":
    /// ring-scalar operations, executed once per decided batch).
    pub fn eval_generator_mle(&self, r: &[Fq]) -> ModulePoint {
        let n = self.generator_count();
        assert_eq!(r.len(), n.next_power_of_two().trailing_zeros() as usize);
        let mut table: Vec<Vec<Fq>> = self.columns[..n].iter().map(|c| c.0.clone()).collect();
        for &ri in r.iter() {
            let half = table.len() / 2;
            let mut next = Vec::with_capacity(half);
            for t in 0..half {
                // First remaining variable is the MSB of the table index:
                // lo = t (bit 0), hi = t + half (bit 1); restrict via
                // (1 - r)*lo + r*hi = lo + r*(hi - lo).
                let lo = &table[t];
                let hi = &table[t + half];
                let mut restricted = lo.clone();
                for (a, b) in restricted.iter_mut().zip(hi.iter()) {
                    *a = a.add(&b.mul(&ri));
                }
                for (a, b) in restricted.iter_mut().zip(lo.iter()) {
                    *a = a.sub(&b.mul(&ri));
                }
                next.push(restricted);
            }
            table = next;
        }
        ModulePoint(table.into_iter().next().expect("non-empty"))
    }

    /// Check a mod-`q` kernel relation on the extended matrix `[G | P]`:
    /// `Σ_b k_b G_b + k_P P ≡ 0` with a nonzero coefficient vector.
    pub fn is_extended_kernel(&self, k_g: &[Fq], k_p: &Fq) -> bool {
        let mut acc = vec![Fq::ZERO; self.module_dim()];
        for (b, &kb) in k_g.iter().enumerate() {
            if kb.0 == 0 {
                continue;
            }
            for (a, &cj) in acc.iter_mut().zip(self.columns[b].0.iter()) {
                *a = a.add(&cj.mul(&kb));
            }
        }
        if k_p.0 != 0 {
            let p = self.value_column();
            for (a, &pj) in acc.iter_mut().zip(p.0.iter()) {
                *a = a.add(&pj.mul(k_p));
            }
        }
        let all_zero = k_g.iter().all(|k| k.0 == 0) && k_p.0 == 0;
        acc.iter().all(|x| x.0 == 0) && !all_zero
    }

    /// The Module-SIS radius of the digit-layer regime.
    pub fn digit_norm_bound(&self) -> u64 {
        (1u64 << ACCORDION_DIGIT_BITS) - 1
    }

    /// Shortness verdict for a recovered opening — centered magnitude
    /// (a coefficient of `q−1` is `−1`, which is short).
    pub fn is_short(&self, w: &[Fq]) -> bool {
        let bound = self.digit_norm_bound();
        w.iter().all(|x| {
            let v = x.0;
            v <= bound || (Q_50 - v) <= bound
        })
    }
}

/// The layered cube: data variables (the polynomial's `k` variables) over
/// digit-layer variables (`κ = log2 J`), MSB-first bit layout with the data
/// bits high and the layer bits low.
#[derive(Clone, Debug)]
pub struct LayeredCube {
    /// Number of data variables `k` (the original cube is `2^k` values).
    pub num_data_vars: usize,
    /// Number of digit layers `J` (a power of two; excess layers are zero).
    pub num_layers: usize,
}

impl LayeredCube {
    pub fn new(num_data_vars: usize, num_layers: usize) -> Self {
        assert!(num_layers.is_power_of_two());
        LayeredCube {
            num_data_vars,
            num_layers,
        }
    }

    /// Total variables of the layered cube `m = k + κ`.
    pub fn num_vars(&self) -> usize {
        self.num_data_vars + self.num_layers.trailing_zeros() as usize
    }

    /// Total cube size `N = n * J`.
    pub fn size(&self) -> usize {
        (1usize << self.num_data_vars) * self.num_layers
    }

    /// The layered-cube index of value `i`, layer `j`.
    pub fn index(&self, i: usize, j: usize) -> usize {
        i * self.num_layers + j
    }

    /// Decompose a value vector into the layered witness.
    pub fn digit_layers(&self, f: &[Fq]) -> Vec<Fq> {
        assert_eq!(f.len(), 1usize << self.num_data_vars);
        let mut w = vec![Fq::ZERO; self.size()];
        for (i, &fi) in f.iter().enumerate() {
            for j in 0..self.num_layers {
                w[self.index(i, j)] = Fq::from_u64((fi.0 >> (16 * j)) & 0xFFFF);
            }
        }
        w
    }

    /// Recompose the value vector from a layered witness (mod `q`).
    pub fn recompose(&self, w: &[Fq]) -> Vec<Fq> {
        assert_eq!(w.len(), self.size());
        let n = 1usize << self.num_data_vars;
        let mut f = vec![Fq::ZERO; n];
        for i in 0..n {
            let mut acc = Fq::ZERO;
            for j in 0..self.num_layers {
                let weight = Fq::from_u64(1u64 << (16 * j));
                acc = acc.add(&w[self.index(i, j)].mul(&weight));
            }
            f[i] = acc;
        }
        f
    }

    /// The layer-mixing multilinear `E(ℓ) = Σ_j 2^{16j}·eq(ℓ, e_j)` evaluated
    /// on the layer cube (length `J`).
    pub fn layer_weights(&self) -> Vec<Fq> {
        let mut e = vec![Fq::ZERO; self.num_layers];
        for j in 0..self.num_layers.min(ACCORDION_LAYERS) {
            e[j] = Fq::from_u64(1u64 << (16 * j));
        }
        e
    }

    /// The combined equality factor `T` on the full layered cube:
    /// `T(x, ℓ) = eq(x_D, u)·E(ℓ)` — the paper's `eq(X, z)` generalized to
    /// the layered setting.
    pub fn t_table(&self, u: &[Fq]) -> Vec<Fq> {
        assert_eq!(u.len(), self.num_data_vars);
        let n = 1usize << self.num_data_vars;
        let weights = self.layer_weights();
        let mut eq_u = vec![Fq::ZERO; n];
        for i in 0..n {
            eq_u[i] = eq_eval_index(i, u, self.num_data_vars);
        }
        let mut t = vec![Fq::ZERO; self.size()];
        for i in 0..n {
            for j in 0..self.num_layers {
                t[self.index(i, j)] = eq_u[i].mul(&weights[j]);
            }
        }
        t
    }

    /// Evaluate `T` at a full layered point `(r_D, r_L)` (verifier-side).
    pub fn eval_t(&self, r: &[Fq], u: &[Fq]) -> Fq {
        let kappa = self.num_layers.trailing_zeros() as usize;
        assert_eq!(r.len(), self.num_vars());
        let (r_d, r_l) = r.split_at(self.num_vars() - kappa);
        let eqv = eq_eval_point(r_d, u);
        let mut e = Fq::ZERO;
        for j in 0..self.num_layers.min(ACCORDION_LAYERS) {
            let eqj = eq_layer(r_l, j);
            e = e.add(&eqj.mul(&Fq::from_u64(1u64 << (16 * j))));
        }
        eqv.mul(&e)
    }

    /// Evaluate `e(X) = Σ_i γⁱ eq(X, rᵢ)` at a point (verifier-side).
    pub fn eval_eq_batch(&self, r: &[Fq], points: &[Vec<Fq>], gammas: &[Fq]) -> Fq {
        let mut acc = Fq::ZERO;
        for (ri, g) in points.iter().zip(gammas.iter()) {
            let mut eqv = Fq::ONE;
            for (a, b) in r.iter().zip(ri.iter()) {
                eqv = eqv.mul(&a.mul(b).add(&Fq::ONE.sub(a).mul(&Fq::ONE.sub(b))));
            }
            acc = acc.add(&eqv.mul(g));
        }
        acc
    }

    /// The `e` table on the full layered cube (prover-side folding input).
    pub fn eq_batch_table(&self, points: &[Vec<Fq>], gammas: &[Fq]) -> Vec<Fq> {
        let mut t = vec![Fq::ZERO; self.size()];
        for (ri, g) in points.iter().zip(gammas.iter()) {
            for b in 0..self.size() {
                let eqv = eq_eval_index(b, ri, self.num_vars());
                t[b] = t[b].add(&eqv.mul(g));
            }
        }
        t
    }
}

/// `eq(i, u)` with `i`'s bits MSB-first over `k` variables.
pub fn eq_eval_index(i: usize, u: &[Fq], k: usize) -> Fq {
    let mut val = Fq::ONE;
    for (bit, &ui) in u.iter().enumerate() {
        let b = (i >> (k - 1 - bit)) & 1;
        let bf = Fq::from_u64(b as u64);
        val = val.mul(&bf.mul(&ui).add(&Fq::ONE.sub(&bf).mul(&Fq::ONE.sub(&ui))));
    }
    val
}

/// `eq(r_D, u)` for two field points.
pub fn eq_eval_point(a: &[Fq], b: &[Fq]) -> Fq {
    let mut val = Fq::ONE;
    for (x, y) in a.iter().zip(b.iter()) {
        val = val.mul(&x.mul(y).add(&Fq::ONE.sub(x).mul(&Fq::ONE.sub(y))));
    }
    val
}

/// `eq(r_L, e_j)` — the layer-cube indicator of layer `j` (bits MSB-first).
fn eq_layer(r_l: &[Fq], j: usize) -> Fq {
    let kappa = r_l.len();
    let mut val = Fq::ONE;
    for (bit, &rl) in r_l.iter().enumerate() {
        let jb = ((j >> (kappa - 1 - bit)) & 1) as u64;
        let bf = Fq::from_u64(jb);
        val = val.mul(&rl.mul(&bf).add(&Fq::ONE.sub(&rl).mul(&Fq::ONE.sub(&bf))));
    }
    val
}

/// Free-function wrapper: split values into 16-bit digit layers over the
/// standard 4-layer cube.
pub fn digit_layers(f: &[Fq]) -> Vec<Fq> {
    LayeredCube::new(f.len().trailing_zeros() as usize, ACCORDION_LAYERS).digit_layers(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fq_arithmetic() {
        let a = Fq::from_u64(Q_50 - 3);
        let b = Fq::from_u64(5);
        assert_eq!(a.add(&b), Fq::from_u64(2));
        assert_eq!(a.mul(&Fq::ONE), a);
        let c = a.mul(&b);
        assert_eq!(c.mul(&b.inv().unwrap()), a);
        assert!(Fq::ZERO.inv().is_none());
        assert_eq!(Fq::from_u64(Q_50).sub(&Fq::ONE), Fq::from_u64(Q_50 - 1));
    }

    #[test]
    fn module_point_arithmetic() {
        let a = ModulePoint(vec![Fq::from_u64(5), Fq::from_u64(7)]);
        let b = ModulePoint(vec![Fq::from_u64(2), Fq::from_u64(11)]);
        assert_eq!(a.add(&b).0[0], Fq::from_u64(7));
        assert_eq!(a.sub(&b).0[1], Fq::from_u64(Q_50 - 4));
        assert_eq!(a.axpy(&Fq::from_u64(3), &b).0[1], Fq::from_u64(7 + 33));
    }

    #[test]
    fn digit_roundtrip() {
        let cube = LayeredCube::new(4, 4);
        let f: Vec<Fq> = (0..16u64)
            .map(|i| Fq::from_u64(i * 1_000_003 + 7))
            .collect();
        let w = cube.digit_layers(&f);
        assert!(w.iter().all(|x| x.0 <= 0xFFFF));
        assert_eq!(cube.recompose(&w), f);
    }

    #[test]
    fn t_table_matches_eval() {
        let cube = LayeredCube::new(3, 4);
        let u = vec![Fq::from_u64(12345), Fq::from_u64(6789), Fq::from_u64(999)];
        let t = cube.t_table(&u);
        // On-cube agreement: layered index (i, j) has r = (bits(i), bits(j)).
        for (i, j) in [(0b010usize, 0b01usize), (0b101, 0b10)] {
            let mut r = Vec::new();
            for bit in 0..3 {
                r.push(Fq::from_u64(((i >> (2 - bit)) & 1) as u64));
            }
            for bit in 0..2 {
                r.push(Fq::from_u64(((j >> (1 - bit)) & 1) as u64));
            }
            assert_eq!(cube.eval_t(&r, &u), t[cube.index(i, j)]);
        }
        // Off-cube point: T = eq(r_D, u) * E(r_L) with E = Σ_j 2^{16j}
        // eq(r_L, e_j).
        let r = vec![
            Fq::from_u64(7),
            Fq::from_u64(9),
            Fq::from_u64(11),
            Fq::from_u64(2),
            Fq::from_u64(3),
        ];
        let v = cube.eval_t(&r, &u);
        let eq = eq_eval_point(&r[..3], &u);
        let mut e = Fq::ZERO;
        for j in 0..4usize {
            e = e.add(&eq_layer(&r[3..], j).mul(&Fq::from_u64(1u64 << (16 * j))));
        }
        assert_eq!(v, eq.mul(&e));
    }

    #[test]
    fn srs_commit_linearity() {
        let srs = Srs::from_seed(2, 64, 64, b"seed-A");
        let w: Vec<Fq> = (0..64).map(|i| Fq::from_u64((i * 7919) % 0xFFFF)).collect();
        let w2: Vec<Fq> = (0..64)
            .map(|i| Fq::from_u64((i * 104729) % 0xFFFF))
            .collect();
        let cm = srs.commit_scalars(&w);
        let cm2 = srs.commit_scalars(&w2);
        let sum: Vec<Fq> = w.iter().zip(w2.iter()).map(|(a, b)| a.add(b)).collect();
        assert_eq!(cm.add(&cm2), srs.commit_scalars(&sum));
        assert!(srs.is_short(&w));
        let mut big = w.clone();
        big[0] = Fq::from_u64(0xFFFF_FFFF);
        assert!(!srs.is_short(&big));
    }

    #[test]
    fn srs_generator_mle_matches_naive() {
        let n_vars = 5;
        let srs = Srs::from_seed(1, 64, 32, b"seed-B");
        let r: Vec<Fq> = (0..n_vars)
            .map(|i| Fq::from_u64(1000 + i as u64 * 7717))
            .collect();
        let folded = srs.eval_generator_mle(&r);
        let mut naive = ModulePoint::zero(srs.module_dim());
        for b in 0..32usize {
            let eqv = eq_eval_index(b, &r, n_vars);
            for (a, &cj) in naive.0.iter_mut().zip(srs.columns[b].0.iter()) {
                *a = a.add(&cj.mul(&eqv));
            }
        }
        assert_eq!(folded, naive);
    }

    #[test]
    fn extended_kernel_detection() {
        let srs = Srs::from_seed(1, 64, 8, b"seed-C");
        assert!(!srs.is_extended_kernel(&[Fq::ZERO; 8], &Fq::ZERO));
        let k: Vec<Fq> = (0..8).map(|i| Fq::from_u64(i as u64 + 3)).collect();
        assert!(!srs.is_extended_kernel(&k, &Fq::ZERO));
    }

    #[test]
    fn duplicated_column_kernel() {
        // Crafted SRS whose first two generators coincide: (1, -1, 0, ...) is
        // a true short kernel — the Module-SIS solution shape that honest
        // sampling rules out.
        let base = Srs::from_seed(1, 64, 8, b"seed-D");
        let mut columns: Vec<ModulePoint> =
            (0..9).map(|c| base.columns[c.min(7)].clone()).collect();
        columns[1] = columns[0].clone();
        let srs = Srs::from_columns(1, 64, columns);
        let mut k = vec![Fq::ZERO; 8];
        k[0] = Fq::ONE;
        k[1] = Fq::from_u64(Q_50 - 1);
        assert!(srs.is_extended_kernel(&k, &Fq::ZERO));
    }
}
