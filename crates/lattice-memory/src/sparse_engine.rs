//! The sparse "0s are free" sumcheck engine (Twist & Shout §2.9.2, §6–§7).
//!
//! The dense engine route costs `O(K·T·log(K·T))` field operations and
//! `O(K·T)` memory per leg because it materializes the embedded one-hot
//! matrices and iterates every point of the full `(k, j)` cube. This
//! module replaces the *round-message computation* with an entry-list
//! pass: the nonzero structure of the one-hot / increment factors defines
//! a support of at most `T` entries, and every round polynomial is
//! computed by iterating those entries only — the zeros of the cube
//! contribute nothing ("0s are free").
//!
//! The engine emits **byte-identical sumcheck proofs** to the dense
//! `lattice_sumcheck` engine over the same virtual polynomial (same
//! round values, same transcript labels `sumcheck-round` /
//! `sumcheck-challenge`), so the existing PIOP verifiers
//! (`verify_shout`, `verify_onehot`) work unchanged and the two provers
//! are interchangeable — pinned by equivalence tests below.
//!
//! Factors are described in a *projected* form: each factor spans only a
//! subset of the sumcheck's variables (`var_map`), which is how the
//! cycle-scoped `eq` factors (`T` entries) and the address-scoped table
//! factors (`K` entries) avoid materializing the full `K·T` space.
//!
//! Cost profile per leg: `O(nnz · n · (d+2))` field operations for the
//! round messages plus `O(Σ dense-factor sizes)` for the per-round
//! `fix_variables` binding — never `O(K·T)` message work.

use crate::onehot::OneHotLayout;
use crate::twist::TwistProof;
use crate::{FactorId, FactorResolver, PiopError};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::SumcheckProof;

/// A dense MLE factor spanning a subset of the sumcheck variables.
///
/// `var_map` is ascending: own variable `i` (MSB-first in the factor's
/// own space) is the full-space variable `var_map[i]`.
#[derive(Clone)]
pub struct ProjectedDense {
    pub mle: DenseMle,
    pub var_map: Vec<usize>,
}

/// A dense factor source: a materialized table, or the VIRTUAL current-
/// value matrix `Val` (the virtual-Val route — `Val(k, j) = init(k) +
/// Σ_{writes to k before j} inc`, never materialized as a `K·T` table).
///
/// The virtual factor spans ALL `log_k + log_t` variables (address bits
/// first, slot bits last — the same layout the materialized Val table
/// uses), so its round messages are IDENTICAL to the dense engine's —
/// the proofs are byte-identical (pinned by the equivalence tests).
#[derive(Clone)]
pub enum DenseFactor {
    /// A materialized table + its variable map.
    Table(DenseMle, Vec<usize>),
    /// The virtual current-value matrix from (init, the write events).
    VirtualVal(VirtualValSpec),
}

impl DenseFactor {
    /// The variable map (the virtual Val spans all variables).
    pub fn var_map(&self) -> Vec<usize> {
        match self {
            DenseFactor::Table(_, vm) => vm.clone(),
            DenseFactor::VirtualVal(spec) => (0..spec.log_k + spec.log_t).collect(),
        }
    }
}

/// One write event: `(address, slot j, increment)` — the value of
/// `Val(addr, ·)` jumps by `inc` after slot `j`.
#[derive(Clone, Debug)]
pub struct ValWrite {
    pub addr: u64,
    pub j: u64,
    pub inc: Goldilocks,
}

/// The virtual-Val specification: the O(K + T) witness replacing the
/// O(K·T) materialized matrix (the container-scale cap lift).
#[derive(Clone, Debug)]
pub struct VirtualValSpec {
    /// The public initial state (K entries).
    pub init: Vec<Goldilocks>,
    pub log_k: usize,
    pub log_t: usize,
    /// The write events, in slot order.
    pub writes: Vec<ValWrite>,
}

impl VirtualValSpec {
    /// Build from the ports streams (the same inputs as
    /// `build_twist_ports`): the writes are the nonzero increments.
    pub fn from_ports(
        write_addr: &[u64],
        write_val: &[Goldilocks],
        init: &[Goldilocks],
        log_k: usize,
        log_t: usize,
    ) -> Result<Self, PiopError> {
        let k = 1usize << log_k;
        let t = 1usize << log_t;
        if write_addr.len() != t || write_val.len() != t || init.len() != k {
            return Err(PiopError::Shape { expected: t, got: write_addr.len() });
        }
        for &a in write_addr {
            if a >= k as u64 {
                return Err(PiopError::AddressOutOfRange { address: a, k });
            }
        }
        let mut running = init.to_vec();
        let mut writes = Vec::new();
        for j in 0..t {
            let wa = write_addr[j] as usize;
            let delta = write_val[j].sub(&running[wa]);
            if !delta.is_zero() {
                writes.push(ValWrite { addr: wa as u64, j: j as u64, inc: delta });
            }
            running[wa] = write_val[j];
        }
        Ok(VirtualValSpec { init: init.to_vec(), log_k, log_t, writes })
    }

    /// The number of `(read, prior write)` pairs the prover's j-rounds
    /// will touch — the dispatch heuristic: virtual iff this stays under
    /// the materialization cost `K·T/8`.
    pub fn pairwise_cost(read_addr: &[u64], write_addr: &[u64]) -> u64 {
        let t = read_addr.len();
        let mut writes_per: std::collections::HashMap<u64, u64> = Default::default();
        for &a in write_addr {
            *writes_per.entry(a).or_insert(0) += 1;
        }
        read_addr
            .iter()
            .map(|&a| writes_per.get(&a).copied().unwrap_or(0))
            .sum::<u64>()
            .min(t as u64 * 64)
    }
}

/// The cached Boolean-endpoint evaluation for one (round, suffix).
#[derive(Clone, Copy)]
struct ValCache {
    round: usize,
    suffix: u64,
    v0: Goldilocks,
    v1: Goldilocks,
}

/// The per-round state of the virtual Val factor.
struct VirtualValState {
    spec: VirtualValSpec,
    /// The init MLE bound on the k-vars bound so far (starts at the full
    /// K-table; one `fix_variables` per k-round).
    init_bound: DenseMle,
    /// Per-write eq weights over the BOUND k-vars.
    w_eq: Vec<Goldilocks>,
    /// The bound k-vars' challenges (in binding order).
    bound_k: usize,
    /// The bound j-vars' challenges (in binding order).
    j_rho: Vec<Goldilocks>,
    /// The k-round bucket cache: free-k-bit key → write indices.
    buckets: std::collections::HashMap<u64, Vec<usize>>,
    /// The round the bucket cache was built for.
    bucket_round: usize,
    /// The Boolean-endpoint cache (one entry — the engine queries each
    /// suffix group across the whole t-loop before moving on).
    cache: Option<ValCache>,
    /// Precomputed write bit-vectors (MSB-first) — allocated once.
    write_bits: Vec<Vec<Goldilocks>>,
}

impl VirtualValState {
    fn new(spec: VirtualValSpec) -> Self {
        let nw = spec.writes.len();
        let init_mle = DenseMle {
            num_vars: spec.log_k,
            evaluations: spec.init.clone(),
        };
        let log_t = spec.log_t;
        let write_bits: Vec<Vec<Goldilocks>> = spec
            .writes
            .iter()
            .map(|w| {
                (0..log_t)
                    .map(|b| Goldilocks::from_u64((w.j >> (log_t - 1 - b)) & 1))
                    .collect()
            })
            .collect();
        VirtualValState {
            w_eq: vec![Goldilocks::ONE; nw],
            init_bound: init_mle,
            bound_k: 0,
            j_rho: Vec::new(),
            buckets: Default::default(),
            bucket_round: usize::MAX,
            cache: None,
            write_bits,
            spec,
        }
    }

    /// The k-bucket for a round: writes grouped by their address's bits on
    /// the free k-vars (vars `ell+1 .. log_k-1` — the low
    /// `log_k-1-ell` address bits, matching `suffix >> log_t`).
    fn build_buckets(&mut self, ell: usize) {
        if self.bucket_round == ell {
            return;
        }
        self.buckets.clear();
        let free_k = self.spec.log_k.saturating_sub(1 + ell);
        let mask: u64 = if free_k >= 64 { u64::MAX } else { (1u64 << free_k) - 1 };
        for (wi, w) in self.spec.writes.iter().enumerate() {
            let key = w.addr & mask;
            self.buckets.entry(key).or_default().push(wi);
        }
        self.bucket_round = ell;
    }

    /// The partial value of the Val MLE at the point
    /// `(ρ_0..ρ_{ell−1}, t, suffix-bits on ell+1..n−1)`.
    ///
    /// The Val factor's contribution is LINEAR in the round variable `t`
    /// (the init lerp and the per-write eq-lerp / LT-extension are all
    /// degree ≤ 1 in `t`), so the value at ANY node `t` is the lerp of
    /// the two Boolean evaluations. The Boolean pair is cached per
    /// (round, suffix) — the engine's `deg+1`-node t-loop costs two
    /// passes per suffix group, not `2·(deg+1)`.
    fn partial_value(
        &mut self,
        ell: usize,
        t: Goldilocks,
        suffix: u64,
    ) -> Goldilocks {
        let (v0, v1) = match self.cache {
            Some(c) if c.round == ell && c.suffix == suffix => (c.v0, c.v1),
            _ => {
                let v0 = self.partial_value_bool(ell, false, suffix);
                let v1 = self.partial_value_bool(ell, true, suffix);
                self.cache = Some(ValCache { round: ell, suffix, v0, v1 });
                (v0, v1)
            }
        };
        let one = Goldilocks::ONE;
        one.sub(&t).mul(&v0).add(&t.mul(&v1))
    }

    /// The partial value at the round variable set to the Boolean `bit`.
    fn partial_value_bool(
        &mut self,
        ell: usize,
        bit: bool,
        suffix: u64,
    ) -> Goldilocks {
        let log_k = self.spec.log_k;
        let log_t = self.spec.log_t;
        let n = log_k + log_t;
        let one = Goldilocks::ONE;
        let t = if bit { one } else { Goldilocks::ZERO };
        if ell < log_k {
            // ---- A k-round: the free j-vars are all Boolean (the entry's
            // slot bits), so LT̃(j_w, j_e) = [j_w < j_e] as integers. ----
            // The init part: the bound-init table's partial value at
            // (bound_k, t, the suffix's k-bits) — the same arithmetic as
            // `dense_partial_value` with var_map (0..log_k).
            let init_part = {
                let var_map: Vec<usize> = (0..log_k).collect();
                let bound = self.bound_k;
                let len = log_k;
                if bound < len && var_map[bound] == ell {
                    let points = 1usize << (len - bound - 1);
                    let rem = suffix_bits_at(suffix, n, &var_map[bound + 1..]);
                    let a = self.init_bound.evaluations[rem];
                    let b = self.init_bound.evaluations[rem + points];
                    a.add(&b.sub(&a).mul(&t))
                } else {
                    let rem = suffix_bits_at(suffix, n, &var_map[bound..]);
                    self.init_bound.evaluations[rem]
                }
            };
            // The Inc part: the bucket for the suffix's free-k bits; each
            // write contributes w_eq·inc·eq_lerp(t, addr's var-ell bit)·[j_w < j_e].
            self.build_buckets(ell);
            let j_e = suffix & ((1u64 << log_t) - 1); // the low log_t bits = the j-vars
            let mut acc = Goldilocks::ZERO;
            if let Some(bucket) = self.buckets.get(&(suffix >> log_t)) {
                for &wi in bucket {
                    let w = &self.spec.writes[wi];
                    let bit_ell = (w.addr >> (log_k - 1 - ell)) & 1;
                    // At the Boolean node t = bit: eq_lerp(t, bit_ell) is
                    // 1 iff bit == bit_ell.
                    if bit_ell == (bit as u64) {
                        let _ = one;
                        if w.j < j_e {
                            acc = acc.add(&self.w_eq[wi].mul(&w.inc));
                        }
                    }
                }
            }
            init_part.add(&acc)
        } else {
            // ---- A j-round: all k-vars are bound (x_k = ρ_k); the Inc
            // part sums over ALL writes with the per-write k-eq weights
            // and the LT-extension at the mixed j-point (the paper's
            // pairwise cost profile — the dispatch heuristic keeps this
            // on the sparse-address regime). ----
            let init_part = self.init_bound.evaluations[0];
            // The mixed j-point: (j_rho on the bound j-vars, t on var
            // ell, the suffix's free j-bits).
            let bound_j = ell - log_k; // the number of j-vars bound so far (incl. the round var)
            let x_j: Vec<Goldilocks> = (0..log_t)
                .map(|v| {
                    if v < bound_j {
                        self.j_rho[v]
                    } else if v == bound_j {
                        t
                    } else {
                        // The suffix's bit for the full-space var log_k+v:
                        // the suffix covers vars ell+1..n-1, MSB-first.
                        let bit = (suffix >> (n - 1 - (log_k + v))) & 1;
                        Goldilocks::from_u64(bit)
                    }
                })
                .collect();
            let mut acc = Goldilocks::ZERO;
            for (wi, w) in self.spec.writes.iter().enumerate() {
                // LT̃(j_w, x_j) — allocation-free inline (the same
                // recurrence as DenseMle::lt_extension).
                let bits = &self.write_bits[wi];
                let mut lt = Goldilocks::ZERO;
                let mut prefix = Goldilocks::ONE;
                for (a, b) in bits.iter().zip(x_j.iter()) {
                    let term = Goldilocks::ONE.sub(a).mul(b);
                    lt = lt.add(&prefix.mul(&term));
                    let same = a.mul(b).add(
                        &Goldilocks::ONE.sub(a).mul(&Goldilocks::ONE.sub(b)),
                    );
                    prefix = prefix.mul(&same);
                }
                acc = acc.add(&self.w_eq[wi].mul(&w.inc).mul(&lt));
            }
            init_part.add(&acc)
        }
    }

    /// Bind the round variable (index `ell`) to `r`.
    fn bind(&mut self, ell: usize, r: Goldilocks) {
        let log_k = self.spec.log_k;
        self.cache = None; // invalidate the endpoint cache
        if ell < log_k {
            // The init table halves.
            if let Ok(bound) = self.init_bound.fix_variables(&[r]) {
                self.init_bound = bound;
            }
            // The write weights pick up eq(r, addr's var-ell bit).
            for (wi, w) in self.spec.writes.iter().enumerate() {
                let bit = (w.addr >> (log_k - 1 - ell)) & 1;
                self.w_eq[wi] = self.w_eq[wi].mul(&eq_point(&r, bit));
            }
            self.bound_k += 1;
            self.bucket_round = usize::MAX; // invalidate the cache
        } else {
            self.j_rho.push(r);
        }
    }

    /// The terminal claim: Val̃(ρ) via the Eq-11 identity —
    /// `init̃(ρ_k) + Σ_w eq(ρ_k, addr_w)·inc_w·LT̃(j_w, ρ_j)`.
    fn terminal(&self) -> Goldilocks {
        let init_part = self.init_bound.evaluations[0];
        let mut acc = Goldilocks::ZERO;
        for (wi, w) in self.spec.writes.iter().enumerate() {
            let bits = &self.write_bits[wi];
            let lt = match DenseMle::lt_extension(bits, &self.j_rho) {
                Ok(v) => v,
                Err(_) => Goldilocks::ZERO,
            };
            acc = acc.add(&self.w_eq[wi].mul(&w.inc).mul(&lt));
        }
        init_part.add(&acc)
    }
}

/// A sparse factor: nonzero entries over its own variable subset.
///
/// `entries[j] = (own_index, value)`; the own index bits pair with
/// `var_map` MSB-first.
#[derive(Clone)]
pub struct SparseFactor {
    pub entries: Vec<(u64, Goldilocks)>,
    pub var_map: Vec<usize>,
}

/// One product term: `coeff · Π(sparse) · Π(dense)` over the entries.
///
/// Invariant: every sparse factor referenced by the term has entries
/// aligned with `positions` (same length, same full-space positions).
/// Terms with NO sparse factors are legal: they are evaluated densely
/// over `positions`, which the caller sets to the full cube `0..2^n`
/// (the dense-engine cost for that term — used when a term's factors
/// are all dense, e.g. booleanity-style products).
#[derive(Clone)]
pub struct SparseTerm {
    pub coeff: Goldilocks,
    /// Full-space positions of the term's support (address bits first,
    /// cycle bits last — the combined `(k, j)` index).
    pub positions: Vec<u64>,
    pub sparse: Vec<usize>,
    pub dense: Vec<usize>,
}

/// A sumcheck instance in sparse form.
pub struct SparseInstance {
    pub num_vars: usize,
    pub sparse: Vec<SparseFactor>,
    pub dense: Vec<DenseFactor>,
    pub terms: Vec<SparseTerm>,
}

/// The working state of one dense factor (materialized or virtual).
enum DenseState {
    Table(DenseMle),
    Virtual(Box<VirtualValState>),
}

/// Prover output: the proof plus per-factor evaluation claims.
#[derive(Clone, Debug)]
pub struct SparseOutput {
    pub proof: SumcheckProof,
    pub challenges: Vec<Goldilocks>,
    pub final_claim: Goldilocks,
    /// Per sparse factor: its evaluation at the challenge point restricted
    /// to its own variables (`Σ_j w_j · v_j`).
    pub sparse_claims: Vec<Goldilocks>,
    /// Per dense factor: its final partial value.
    pub dense_claims: Vec<Goldilocks>,
}

fn eq_lerp(t: Goldilocks, bit: u64) -> Goldilocks {
    if bit == 1 {
        t
    } else {
        Goldilocks::ONE.sub(&t)
    }
}

// The scalar reference forms below (`eq_point`, `dense_partial_value`,
// `own_index`) are superseded by the affine Wave-10 round loop; kept for
// differential reading and future bisects.
#[allow(dead_code)]
fn eq_point(p: &Goldilocks, bit: u64) -> Goldilocks {
    eq_lerp(*p, bit)
}

/// Extract a factor's own index from a full-space position.
#[allow(dead_code)]
fn own_index(position: u64, num_vars: usize, var_map: &[usize]) -> u64 {
    let len = var_map.len();
    let mut own = 0u64;
    for (i, &v) in var_map.iter().enumerate() {
        let bit = (position >> (num_vars - 1 - v)) & 1;
        own |= bit << (len - 1 - i);
    }
    own
}

/// Reverse the low `n` bits of x.
fn reverse_bits(x: u64, n: usize) -> u64 {
    let mut r = 0u64;
    for i in 0..n {
        r |= ((x >> i) & 1) << (n - 1 - i);
    }
    r
}

/// A dense factor's partial value at (bound prefix, t, suffix) — the
/// scalar reference form of the affine (a, b) endpoints.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn dense_partial_value(
    state: &DenseMle,
    bound: usize,
    var_map: &[usize],
    ell: usize,
    n: usize,
    t: Goldilocks,
    suffix: u64,
) -> Goldilocks {
    let len = var_map.len();
    let arr = &state.evaluations;
    if bound < len && var_map[bound] == ell {
        // Binding this round's variable: half-bind lerp.
        let points = 1usize << (len - bound - 1);
        let rem = suffix_bits_at(suffix, n, &var_map[bound + 1..]);
        let a = arr[rem];
        let b = arr[rem + points];
        a.add(&b.sub(&a).mul(&t))
    } else {
        let rem = suffix_bits_at(suffix, n, &var_map[bound..]);
        arr[rem]
    }
}

/// The own-index bits a remaining suffix assigns to the given own vars.
fn suffix_bits_at(suffix: u64, n: usize, vars: &[usize]) -> usize {
    let len = vars.len();
    let mut rem = 0usize;
    for (i, &v) in vars.iter().enumerate() {
        let bit = ((suffix >> (n - 1 - v)) & 1) as usize;
        rem |= bit << (len - 1 - i);
    }
    rem
}

/// Lagrange-evaluate the round polynomial (values at nodes 0..d) at r.
fn interpolate_at(evals: &[Goldilocks], r: &Goldilocks) -> Goldilocks {
    let n = evals.len();
    let mut acc = Goldilocks::ZERO;
    for i in 0..n {
        let mut weight = Goldilocks::ONE;
        let xi = Goldilocks::from_u64(i as u64);
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = Goldilocks::from_u64(j as u64);
            let num = r.sub(&xj);
            let den = xi.sub(&xj);
            let inv = den.inverse().unwrap_or(Goldilocks::ZERO);
            weight = weight.mul(&num.mul(&inv));
        }
        acc = acc.add(&evals[i].mul(&weight));
    }
    acc
}

/// Prove `Σ_{x ∈ {0,1}^n} P(x) = claim` for the sparse product system.
///
/// Produces the identical proof object the dense engine would produce
/// for the equivalent `VirtualPolynomial` (equivalence is test-pinned).
///
/// # The Wave-10 round loop (affine partitioned sums + SIMD)
///
/// Every factor's per-group contribution to the round polynomial is
/// AFFINE in the round variable `t`: for a sparse factor spanning the
/// round variable, `Σ_k w_k·v_k·eq_lerp(t, b_k) = W0·(1−t) + W1·t`
/// with `W0/W1` the weight-value sums over the group's bit-0 / bit-1
/// entries; for a dense factor, `a + (b−a)·t` over the group's lerp
/// pair. The engine therefore touches each entry ONCE per round
/// (independent of the round length `deg+1`), evaluates the `t`-nodes
/// from the affine coefficients, and runs the entry sums / weight
/// binding through the `field_simd` AVX-512 slice kernels. Field
/// arithmetic is exact and commutative, so every round value is
/// bit-identical to the scalar reference (pinned by the equivalence
/// tests and `LZX_NO_SIMD=1` bisects).
///
/// Entries are canonically ordered by `reverse_bits(position)` once at
/// setup: the remaining-suffix groups are then CONTIGUOUS ranges, and
/// within each group the current variable's 0/1 entries form two
/// contiguous sub-ranges — the partitioned sums are plain slice sums.
pub fn prove_sparse_sumcheck(
    inst: &SparseInstance,
    claim: Goldilocks,
    transcript: &mut Transcript,
) -> Result<SparseOutput, PiopError> {
    // The borrowing form clones the instance once (the pre-Wave-10
    // engine cloned the entries and dense states internally — the same
    // order of memory); hot callers use the owned form.
    let owned = SparseInstance {
        num_vars: inst.num_vars,
        sparse: inst.sparse.clone(),
        dense: inst.dense.clone(),
        terms: inst.terms.clone(),
    };
    prove_sparse_sumcheck_owned(owned, claim, transcript)
}

/// The owned form: permutes the instance's own entries/positions into
/// the canonical round order IN PLACE (no second copy of the entry
/// lists stays live — roughly half the peak memory of the borrowing
/// form at large T) and moves the dense MLE states into the working
/// buffers.
pub fn prove_sparse_sumcheck_owned(
    mut inst: SparseInstance,
    claim: Goldilocks,
    transcript: &mut Transcript,
) -> Result<SparseOutput, PiopError> {
    let n = inst.num_vars;
    if n == 0 {
        return Err(PiopError::BadLayout { log_k: 0, d: 0 });
    }
    // Round length: max factor count over terms + 1 values at t = 0..=deg
    // (matches VirtualPolynomial::max_degree, which counts FACTORS).
    let deg = inst
        .terms
        .iter()
        .map(|t| t.sparse.len() + t.dense.len())
        .max()
        .unwrap_or(1)
        .max(1);

    // ---- Canonical ordering (setup, once, IN PLACE). ----
    // Per term: positions sorted by reverse_bits so the remaining-suffix
    // groups are CONTIGUOUS ranges and the current variable's 0/1 entries
    // form two contiguous sub-ranges inside each group. The sort
    // permutation is applied to every first-referenced factor's entries
    // (the SparseTerm alignment invariant makes it well-defined).
    {
        let mut permuted = vec![false; inst.sparse.len()];
        for ti in 0..inst.terms.len() {
            let orig = inst.terms[ti].positions.clone();
            let mut perm: Vec<usize> = (0..orig.len()).collect();
            perm.sort_by_key(|&j| reverse_bits(orig[j], n));
            let needs_reorder = perm.iter().enumerate().any(|(a, &b)| a != b);
            if needs_reorder {
                inst.terms[ti].positions = perm.iter().map(|&j| orig[j]).collect();
            }
            if !orig.is_empty() {
                for &fi in &inst.terms[ti].sparse {
                    if !permuted[fi] && orig.len() == inst.sparse[fi].entries.len() {
                        if needs_reorder {
                            let entries = std::mem::take(&mut inst.sparse[fi].entries);
                            inst.sparse[fi].entries = perm.iter().map(|&j| entries[j]).collect();
                        }
                        permuted[fi] = true;
                    }
                }
            }
        }
        // Fail-closed alignment validation: every referencing term's
        // support must enumerate the factor's entries.
        for term in &inst.terms {
            for &fi in &term.sparse {
                if term.positions.len() != inst.sparse[fi].entries.len() {
                    return Err(PiopError::BadLayout { log_k: n, d: 0 });
                }
            }
        }
    }

    // Sparse factor working state: `wv[j] = w_j · val_j` — the accumulated
    // eq weight times the entry value, maintained in one buffer (the
    // terminal claim and the round-message sums both consume exactly this
    // product; one-hot factors start as all-ONE for free).
    let mut wv: Vec<Vec<Goldilocks>> = inst
        .sparse
        .iter()
        .map(|f| f.entries.iter().map(|&(_, v)| v).collect())
        .collect();
    // var -> own position map per sparse factor (usize::MAX = not ours).
    let sparse_pos: Vec<Vec<usize>> = inst
        .sparse
        .iter()
        .map(|f| {
            let mut m = vec![usize::MAX; n];
            for (i, &v) in f.var_map.iter().enumerate() {
                if v < n {
                    m[v] = i;
                }
            }
            m
        })
        .collect();
    // Dense factor working state: the Table MLEs are MOVED in
    // (zero-copy; fix_variables is already the SIMD first-half binding)
    // and the virtual-Val factors start their O(K+T) write-event state.
    let mut dense_state: Vec<DenseState> = Vec::with_capacity(inst.dense.len());
    for f in &mut inst.dense {
        dense_state.push(match f {
            DenseFactor::Table(mle, _) => DenseState::Table(std::mem::replace(
                mle,
                DenseMle {
                    num_vars: 0,
                    evaluations: Vec::new(),
                },
            )),
            DenseFactor::VirtualVal(spec) => {
                DenseState::Virtual(Box::new(VirtualValState::new(spec.clone())))
            }
        });
    }
    let mut dense_bound: Vec<usize> = vec![0; inst.dense.len()];
    // The per-factor var maps, hoisted for the round loop (the virtual
    // Val spans ALL variables — the identity map).
    let dense_var_maps: Vec<Vec<usize>> = inst.dense.iter().map(|f| f.var_map()).collect();

    // Scratch buffers reused across rounds / groups / factors.
    let mut eqw_buf: Vec<Goldilocks> = Vec::new();
    let mut wv_swap: Vec<Goldilocks> = Vec::new();
    let mut dense_ab: Vec<(Goldilocks, Goldilocks, bool)> = Vec::new();
    let mut sparse_ab: Vec<(Goldilocks, Goldilocks, bool)> = Vec::new();
    let t_nodes: Vec<Goldilocks> = (0..=deg).map(|t| Goldilocks::from_u64(t as u64)).collect();

    let mut current_claim = claim;
    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(n);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(n);

    for ell in 0..n {
        // Remaining-suffix mask after binding variable ell.
        let suffix_len = n - 1 - ell;
        let suffix_mask: u64 = if suffix_len >= 64 {
            u64::MAX
        } else {
            (1u64 << suffix_len) - 1
        };
        // Round polynomial values g(0..=deg). Per suffix group, every
        // factor's contribution is AFFINE in t (the W0/W1 resp. (a, b)
        // forms), so each entry is touched once per round regardless of
        // the round length, and the sums run through the SIMD slice
        // kernels. The product-of-sums per group reproduces the true
        // round polynomial including the cross terms between entries of
        // different one-hot factors sharing a suffix.
        let mut evals_at: Vec<Goldilocks> = vec![Goldilocks::ZERO; deg + 1];
        for (ti, term) in inst.terms.iter().enumerate() {
            if term.positions.is_empty() {
                // A term whose support is empty contributes zero to every
                // round (all its sparse factors vanish identically).
                continue;
            }
            let pos = &inst.terms[ti].positions;
            let mut seg = 0usize;
            while seg < pos.len() {
                let suffix = pos[seg] & suffix_mask;
                let mut end = seg + 1;
                while end < pos.len() && (pos[end] & suffix_mask) == suffix {
                    end += 1;
                }
                // Dense endpoints: (a, b) once per group and factor. Table
                // factors read the two children of the bound array directly
                // (contiguous by the canonical order); the virtual Val
                // resolves its Boolean pair through the endpoint cache —
                // its partial value is AFFINE in the round variable, so the
                // pair pins every t-node value (byte-identical to the
                // per-node partial_value calls the materialized route pins).
                dense_ab.clear();
                for &di in &term.dense {
                    let var_map: &[usize] = &dense_var_maps[di];
                    let bound = dense_bound[di];
                    let len = var_map.len();
                    match &mut dense_state[di] {
                        DenseState::Table(mle) => {
                            let arr = &mle.evaluations;
                            if bound < len && var_map[bound] == ell {
                                let points = 1usize << (len - bound - 1);
                                let rem = suffix_bits_at(suffix, n, &var_map[bound + 1..]);
                                if rem + points >= arr.len() {
                                    return Err(PiopError::Shape {
                                        expected: arr.len(),
                                        got: rem + points,
                                    });
                                }
                                dense_ab.push((arr[rem], arr[rem + points], true));
                            } else {
                                let rem = suffix_bits_at(suffix, n, &var_map[bound..]);
                                if rem >= arr.len() {
                                    return Err(PiopError::Shape {
                                        expected: arr.len(),
                                        got: rem,
                                    });
                                }
                                dense_ab.push((arr[rem], Goldilocks::ZERO, false));
                            }
                        }
                        DenseState::Virtual(vs) => {
                            // The virtual Val spans ALL variables (identity
                            // map): every round variable is one of its own, so
                            // the endpoints are the two Boolean partials.
                            let a = vs.partial_value(ell, Goldilocks::ZERO, suffix);
                            let b = vs.partial_value(ell, Goldilocks::ONE, suffix);
                            dense_ab.push((a, b, true));
                        }
                    }
                }
                // Sparse partitioned sums: W0 over the group's bit-0 run,
                // W1 over the bit-1 run (both contiguous by the canonical
                // order). Non-spanning factors: the plain group sum.
                sparse_ab.clear();
                for &fi in &term.sparse {
                    let p = sparse_pos[fi][ell];
                    let f_wv = &wv[fi];
                    if p != usize::MAX {
                        let mut mid = seg;
                        while mid < end && ((pos[mid] >> (n - 1 - ell)) & 1) == 0 {
                            mid += 1;
                        }
                        let w0 = if mid > seg {
                            lattice_core::field_simd::sum_slice(&f_wv[seg..mid])
                        } else {
                            Goldilocks::ZERO
                        };
                        let w1 = if end > mid {
                            lattice_core::field_simd::sum_slice(&f_wv[mid..end])
                        } else {
                            Goldilocks::ZERO
                        };
                        sparse_ab.push((w0, w1, true));
                    } else {
                        let w = lattice_core::field_simd::sum_slice(&f_wv[seg..end]);
                        sparse_ab.push((w, Goldilocks::ZERO, false));
                    }
                }
                // Assemble the t-node values from the affine forms (every
                // factor's contribution — Table, virtual, and sparse alike —
                // is degree ≤ 1 in the round variable).
                for (t, ev) in evals_at.iter_mut().enumerate() {
                    let t_fe = t_nodes[t];
                    let mut prod = term.coeff;
                    for &(a, b, spanning) in &dense_ab {
                        let v = if spanning {
                            a.add(&b.sub(&a).mul(&t_fe))
                        } else {
                            a
                        };
                        prod = prod.mul(&v);
                    }
                    for &(w0, w1, spanning) in &sparse_ab {
                        let v = if spanning {
                            w0.add(&w1.sub(&w0).mul(&t_fe))
                        } else {
                            w0
                        };
                        prod = prod.mul(&v);
                    }
                    *ev = ev.add(&prod);
                }
                seg = end;
            }
        }
        // Prover-side guard: g(0) + g(1) == running claim.
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err(PiopError::Sumcheck(
                lattice_sumcheck::SumcheckError::ClaimMismatch,
            ));
        }
        transcript
            .append_field_slice(b"sumcheck-round", &evals_at)
            .map_err(PiopError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(PiopError::Transcript)?;
        challenges.push(r);
        current_claim = interpolate_at(&evals_at, &r);
        rounds.push(evals_at);

        // Bind factors to r. Sparse: wv[j] *= eq(r, own_bit_j) — the two
        // possible multipliers are selected per entry (branch-free load)
        // and applied through the SIMD product kernel.
        for (fi, f) in inst.sparse.iter().enumerate() {
            let p = sparse_pos[fi][ell];
            if p != usize::MAX {
                let shift = f.var_map.len() - 1 - p;
                let e0 = Goldilocks::ONE.sub(&r);
                eqw_buf.clear();
                eqw_buf.extend(inst.sparse[fi].entries.iter().map(|&(own, _)| {
                    if (own >> shift) & 1 == 1 {
                        r
                    } else {
                        e0
                    }
                }));
                let len = wv[fi].len();
                wv_swap.clear();
                wv_swap.resize(len, Goldilocks::ZERO);
                lattice_core::field_simd::mul_slices(&wv[fi], &eqw_buf, &mut wv_swap);
                std::mem::swap(&mut wv[fi], &mut wv_swap);
            }
        }
        for (di, df) in inst.dense.iter().enumerate() {
            let len = df.var_map().len();
            if dense_bound[di] < len && dense_var_maps[di][dense_bound[di]] == ell {
                match &mut dense_state[di] {
                    DenseState::Table(mle) => {
                        *mle = mle
                            .fix_variables(&[r])
                            .map_err(PiopError::Mle)?;
                    }
                    DenseState::Virtual(vs) => vs.bind(ell, r),
                }
                dense_bound[di] += 1;
            }
        }
    }

    // Terminal: per-factor claims and the final combined claim. The
    // sparse claim is Σ_j w_j·val_j — exactly the maintained buffer.
    let mut sparse_claims = Vec::with_capacity(inst.sparse.len());
    for fi in 0..inst.sparse.len() {
        sparse_claims.push(lattice_core::field_simd::sum_slice(&wv[fi]));
    }
    let mut dense_claims = Vec::with_capacity(inst.dense.len());
    for dstate in &dense_state {
        let claim = match dstate {
            DenseState::Table(mle) => mle.evaluations[0],
            DenseState::Virtual(vs) => vs.terminal(),
        };
        dense_claims.push(claim);
    }
    let mut final_claim = Goldilocks::ZERO;
    for term in &inst.terms {
        let mut prod = term.coeff;
        for &fi in &term.sparse {
            prod = prod.mul(&sparse_claims[fi]);
        }
        for &di in &term.dense {
            prod = prod.mul(&dense_claims[di]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current_claim {
        return Err(PiopError::Sumcheck(
            lattice_sumcheck::SumcheckError::FinalCheckFailed,
        ));
    }

    Ok(SparseOutput {
        proof: SumcheckProof { rounds },
        challenges,
        final_claim,
        sparse_claims,
        dense_claims,
    })
}

// ---------------------------------------------------------------------------
// Sparse Shout (Fig 7): read-checking for read-only tables.
// ---------------------------------------------------------------------------

/// Sparse Fig-7 Shout prover over per-cycle read addresses.
///
/// `read_addresses[j]` is the table index read at cycle j (`T` of them).
/// The transcript flow is identical to [`crate::shout::prove_shout`].
#[allow(clippy::too_many_arguments)]
pub fn prove_shout_sparse(
    table: &[Goldilocks],
    read_addresses: &[u64],
    log_k: usize,
    log_t: usize,
    d: usize,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<(crate::shout::ShoutProof, Vec<FactorClaim>), PiopError> {
    let layout = OneHotLayout::new(log_k, log_t, d, usize::MAX)?;
    let log_n = layout.log_n();
    if read_addresses.len() != layout.t() || table.len() != layout.k() {
        return Err(PiopError::Shape {
            expected: layout.t(),
            got: read_addresses.len(),
        });
    }
    // Meta + table absorb (identical to the dense prover).
    transcript
        .append_field_slice(
            b"shout-meta",
            &[
                Goldilocks::from_u64(log_k as u64),
                Goldilocks::from_u64(log_t as u64),
                Goldilocks::from_u64(d as u64),
            ],
        )
        .map_err(PiopError::Transcript)?;
    transcript
        .append_field_slice(b"shout-table", table)
        .map_err(PiopError::Transcript)?;
    let rcycle = transcript.challenge_fields(b"shout-rcycle", log_t)?;
    let claim = resolver.eval(FactorId::ReadValues, &rcycle)?;
    transcript.append_field(b"shout-claim", &claim)?;

    // Instance: eq(rcycle, j) · Val(k) · Π_i ra_i(k_i, j).
    let t = layout.t();
    let mut ra_dims = Vec::with_capacity(d);
    for i in 0..d {
        let mut var_map: Vec<usize> = (i * log_n..(i + 1) * log_n).collect();
        var_map.extend(log_k..log_k + log_t);
        let mut entries = Vec::with_capacity(t);
        for (j, &a) in read_addresses.iter().enumerate() {
            let digits = layout.digits(a)?;
            entries.push((((digits[i] as u64) << log_t) | j as u64, Goldilocks::ONE));
        }
        ra_dims.push(SparseFactor { entries, var_map });
    }
    let positions: Vec<u64> = (0..t)
        .map(|j| (read_addresses[j] << log_t) | j as u64)
        .collect();
    let eq_j = DenseFactor::Table(DenseMle::eq_extension(&rcycle), (log_k..log_k + log_t).collect());
    let table_m = DenseMle::new(table.to_vec())?;
    let val_f = DenseFactor::Table(table_m, (0..log_k).collect());
    let inst = SparseInstance {
        num_vars: log_k + log_t,
        sparse: ra_dims,
        dense: vec![eq_j, val_f],
        terms: vec![SparseTerm {
            coeff: Goldilocks::ONE,
            positions,
            sparse: (0..d).collect(),
            dense: vec![0, 1],
        }],
    };
    let out = prove_sparse_sumcheck(&inst, claim, transcript)?;
    // Dim claims at the native (k_i, j) terminal points.
    let rho = &out.challenges;
    let mut claims: Vec<FactorClaim> = Vec::with_capacity(d);
    for i in 0..d {
        let mut native: Vec<Goldilocks> = rho[i * log_n..(i + 1) * log_n].to_vec();
        native.extend(rho[log_k..log_k + log_t].iter().copied());
        claims.push((FactorId::Ra(i), native, out.sparse_claims[i]));
    }
    Ok((
        crate::shout::ShoutProof {
            read_checking: out.proof,
        },
        claims,
    ))
}

// ---------------------------------------------------------------------------
// Sparse one-hot constraint PIOP (Figs 6/8).
// ---------------------------------------------------------------------------

/// Evaluate a sparse factor at a point over its own variables.
fn sparse_eval(
    entries: &[(u64, Goldilocks)],
    var_map: &[usize],
    point: &[Goldilocks],
) -> Goldilocks {
    let len = var_map.len();
    let mut acc = Goldilocks::ZERO;
    for &(own, val) in entries {
        let mut w = val;
        for (i, &v) in var_map.iter().enumerate() {
            if v >= point.len() {
                continue;
            }
            let bit = (own >> (len - 1 - i)) & 1;
            w = w.mul(&eq_lerp(point[v], bit));
        }
        acc = acc.add(&w);
    }
    acc
}

/// Sparse one-hot constraint prover over a per-cycle address column.
///
/// Transcript flow identical to [`crate::onehot_check::prove_onehot`].
/// Returns the proof plus the dim-factor claims the verifier will query
/// (booleanity terminals, the 2^-1 weight points, raf terminals).
pub fn prove_onehot_sparse(
    addresses: &[u64],
    log_k: usize,
    log_t: usize,
    d: usize,
    side: crate::onehot_check::OneHotSide,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<(crate::onehot_check::OneHotProof, Vec<FactorClaim>), PiopError> {
    let layout = OneHotLayout::new(log_k, log_t, d, usize::MAX)?;
    let log_n = layout.log_n();
    let t = layout.t();
    if addresses.len() != t {
        return Err(PiopError::Shape {
            expected: t,
            got: addresses.len(),
        });
    }
    let y_factor = side.addr_column();
    transcript
        .append_field_slice(
            b"onehot-meta",
            &[
                Goldilocks::from_u64(log_k as u64),
                Goldilocks::from_u64(log_t as u64),
                Goldilocks::from_u64(d as u64),
                Goldilocks::from_u64(y_factor.discriminant()),
            ],
        )
        .map_err(PiopError::Transcript)?;
    let r_prime = transcript.challenge_fields(b"onehot-rprime", log_t)?;
    let y = resolver.eval(y_factor, &r_prime)?;
    transcript.append_field(b"onehot-y", &y)?;

    // Per-dim digit entries (shared across legs).
    let mut digits_per_dim: Vec<Vec<u32>> = Vec::with_capacity(d);
    for i in 0..d {
        let mut col = Vec::with_capacity(t);
        for &a in addresses {
            col.push(layout.digits(a)?[i]);
        }
        digits_per_dim.push(col);
    }
    let positions: Vec<u64> = (0..t).map(|j| (addresses[j] << log_t) | j as u64).collect();

    // 1. Booleanity per dimension (over the dim's own (k_i, j) space).
    let mut booleanity = Vec::with_capacity(d);
    let mut dim_claims: Vec<(Vec<Goldilocks>, Goldilocks)> = Vec::with_capacity(d);
    for i in 0..d {
        let r_i = transcript.challenge_fields(b"onehot-bool-r", log_n)?;
        let mut point = r_i.clone();
        point.extend(r_prime.iter().copied());
        let eq_m = DenseMle::eq_extension(&point);
        // Dim-space numbering: address vars [0..log_n), then cycle vars.
        let mut var_map: Vec<usize> = (0..log_n).collect();
        var_map.extend(log_n..log_n + log_t);
        let eq_f = DenseFactor::Table(eq_m, var_map);
        let mut entries = Vec::with_capacity(t);
        for j in 0..t {
            entries.push((
                ((digits_per_dim[i][j] as u64) << log_t) | j as u64,
                Goldilocks::ONE,
            ));
        }
        let ra_f = SparseFactor {
            entries,
            var_map: (0..log_n + log_t).collect(),
        };
        // Terms: +ra·ra·eq − ra·eq (over the DIM space: num_vars = log_n + log_t).
        let dim_positions: Vec<u64> = (0..t)
            .map(|j| ((digits_per_dim[i][j] as u64) << log_t) | j as u64)
            .collect();
        let inst = SparseInstance {
            num_vars: log_n + log_t,
            sparse: vec![ra_f],
            dense: vec![eq_f],
            terms: vec![
                SparseTerm {
                    coeff: Goldilocks::ONE,
                    positions: dim_positions.clone(),
                    sparse: vec![0, 0],
                    dense: vec![0],
                },
                SparseTerm {
                    coeff: Goldilocks::ONE.neg(),
                    positions: dim_positions,
                    sparse: vec![0],
                    dense: vec![0],
                },
            ],
        };
        let out = prove_sparse_sumcheck(&inst, Goldilocks::ZERO, transcript)?;
        dim_claims.push((out.challenges.clone(), out.sparse_claims[0]));
        booleanity.push(out.proof);
    }

    // 3. raf-evaluation over the full (k, j) space.
    let eq_j = DenseFactor::Table(DenseMle::eq_extension(&r_prime), (log_k..log_k + log_t).collect());
    let w_evals: Vec<Goldilocks> = (0..layout.k())
        .map(|k| Goldilocks::from_u64(k as u64))
        .collect();
    let w_f = DenseFactor::Table(DenseMle::new(w_evals)?, (0..log_k).collect());
    let mut ra_dims = Vec::with_capacity(d);
    for i in 0..d {
        let mut var_map: Vec<usize> = (i * log_n..(i + 1) * log_n).collect();
        var_map.extend(log_k..log_k + log_t);
        let entries: Vec<(u64, Goldilocks)> = (0..t)
            .map(|j| {
                (
                    ((digits_per_dim[i][j] as u64) << log_t) | j as u64,
                    Goldilocks::ONE,
                )
            })
            .collect();
        ra_dims.push(SparseFactor { entries, var_map });
    }
    let inst = SparseInstance {
        num_vars: log_k + log_t,
        sparse: ra_dims,
        dense: vec![eq_j, w_f],
        terms: vec![SparseTerm {
            coeff: Goldilocks::ONE,
            positions,
            sparse: (0..d).collect(),
            dense: vec![0, 1],
        }],
    };
    let raf_out = prove_sparse_sumcheck(&inst, y, transcript)?;
    // Assemble the dim claims: booleanity terminals, the 2^-1 weight
    // points, and the raf terminals (native (k_i, j) points).
    let inv2 = Goldilocks::TWO.inverse().ok_or(PiopError::InverseOfTwo)?;
    let mut claims: Vec<FactorClaim> = Vec::new();
    let raf_point = &raf_out.challenges;
    for i in 0..d {
        // The side's matrix factor (Ra for reads, Wa for writes).
        let mf = side.matrix(i);
        let (bool_pt, bool_v) = &dim_claims[i];
        claims.push((mf, bool_pt.clone(), *bool_v));
        let mut wpt = vec![inv2; log_n];
        wpt.extend(r_prime.iter().copied());
        // The weight point lives in the DIM's own coordinate space.
        let dim_var_map: Vec<usize> = (0..log_n + log_t).collect();
        let wv = sparse_eval(&inst.sparse[i].entries, &dim_var_map, &wpt);
        claims.push((mf, wpt, wv));
        let mut native: Vec<Goldilocks> = raf_point[i * log_n..(i + 1) * log_n].to_vec();
        native.extend(raf_point[log_k..log_k + log_t].iter().copied());
        claims.push((mf, native, raf_out.sparse_claims[i]));
    }
    Ok((
        crate::onehot_check::OneHotProof {
            booleanity,
            raf: raf_out.proof,
        },
        claims,
    ))
}

// ---------------------------------------------------------------------------
// Twist over read/write ports (per-cycle read AND write).
// ---------------------------------------------------------------------------

/// A Twist witness with per-cycle read and write ports: every cycle has a
/// read address (the read port observes the current value — the value
/// BEFORE this cycle's write) and a write address (a dummy write to the
/// current value keeps the one-hot weight discipline at non-write cycles
/// with a zero increment).
pub struct TwistPortsWitness {
    pub log_k: usize,
    pub log_t: usize,
    /// Read addresses per cycle (T entries, each < K).
    pub read_addr: Vec<u64>,
    /// Write addresses per cycle (T entries, each < K).
    pub write_addr: Vec<u64>,
    /// The materialized current-value matrix `Val(k, j)` — dense `K·T`
    /// (row-major `k·T + j`), the value of address k BEFORE cycle j's
    /// write. This is the caller's scaling decision (the virtual-Val
    /// route of the paper replaces it; see docs).
    pub val: Vec<Goldilocks>,
    /// Sparse increment entries: `(full index (k<<log_t)|j, value)`.
    pub inc: Vec<(u64, Goldilocks)>,
    /// Public initial state (K entries).
    pub init: Vec<Goldilocks>,
    /// Public final state (K entries).
    pub final_state: Vec<Goldilocks>,
    /// The virtual-Val route: when set, the prover computes the Val
    /// factors from this O(K + T) spec instead of the materialized
    /// `val` table (the container-scale cap lift). The proofs are
    /// byte-identical (pinned by the equivalence tests).
    pub val_spec: Option<VirtualValSpec>,
}

/// Build the ports witness. `write_val[j]` is the value written at cycle
/// j (dummy cycles carry the current value of the dummy address).
#[allow(clippy::too_many_arguments)]
pub fn build_twist_ports(
    read_addr: &[u64],
    write_addr: &[u64],
    write_val: &[Goldilocks],
    init: &[Goldilocks],
    log_k: usize,
    log_t: usize,
) -> Result<TwistPortsWitness, PiopError> {
    let k = 1usize << log_k;
    let t = 1usize << log_t;
    if read_addr.len() != t || write_addr.len() != t || write_val.len() != t || init.len() != k {
        return Err(PiopError::Shape {
            expected: t,
            got: read_addr.len(),
        });
    }
    for (a, b) in read_addr.iter().zip(write_addr.iter()) {
        if *a >= k as u64 || *b >= k as u64 {
            return Err(PiopError::AddressOutOfRange {
                address: (*a).max(*b),
                k,
            });
        }
    }
    let mut running = init.to_vec();
    let mut val = vec![Goldilocks::ZERO; k * t];
    let mut inc: Vec<(u64, Goldilocks)> = Vec::new();
    for j in 0..t {
        for kk in 0..k {
            val[kk * t + j] = running[kk];
        }
        let wa = write_addr[j] as usize;
        let delta = write_val[j].sub(&running[wa]);
        if !delta.is_zero() {
            inc.push((((wa as u64) << log_t) | j as u64, delta));
        }
        running[wa] = write_val[j];
    }
    Ok(TwistPortsWitness {
        log_k,
        log_t,
        read_addr: read_addr.to_vec(),
        write_addr: write_addr.to_vec(),
        val,
        inc,
        init: init.to_vec(),
        final_state: running,
        val_spec: None,
    })
}

/// Build the ports witness on the VIRTUAL-VAL route: the O(K·T)
/// materialized `val` matrix is replaced by the O(K + T) write-event
/// spec — the container-scale cap (memory) is lifted while the proofs
/// stay byte-identical. The `read`/`write` streams and the increment
/// entries are the same as `build_twist_ports`.
#[allow(clippy::too_many_arguments)]
pub fn build_twist_ports_virtual(
    read_addr: &[u64],
    write_addr: &[u64],
    write_val: &[Goldilocks],
    init: &[Goldilocks],
    log_k: usize,
    log_t: usize,
) -> Result<TwistPortsWitness, PiopError> {
    let k = 1usize << log_k;
    let t = 1usize << log_t;
    if read_addr.len() != t
        || write_addr.len() != t
        || write_val.len() != t
        || init.len() != k
    {
        return Err(PiopError::Shape { expected: t, got: read_addr.len() });
    }
    let val_spec = VirtualValSpec::from_ports(write_addr, write_val, init, log_k, log_t)?;
    let mut running = init.to_vec();
    let mut inc: Vec<(u64, Goldilocks)> = Vec::new();
    for j in 0..t {
        let wa = write_addr[j] as usize;
        let delta = write_val[j].sub(&running[wa]);
        if !delta.is_zero() {
            inc.push((((wa as u64) << log_t) | j as u64, delta));
        }
        running[wa] = write_val[j];
    }
    Ok(TwistPortsWitness {
        log_k,
        log_t,
        read_addr: read_addr.to_vec(),
        write_addr: write_addr.to_vec(),
        val: Vec::new(), // never materialized on this route
        inc,
        init: init.to_vec(),
        final_state: running,
        val_spec: Some(val_spec),
    })
}

/// Claims produced by the sparse Twist prover, ready for the envelope's
/// claim table: `(factor, point, value)`.
pub type FactorClaim = (FactorId, Vec<Goldilocks>, Goldilocks);

/// The sparse Twist prover (ports model). Transcript flow identical to
/// [`crate::twist::prove_twist`].
///
/// `wv_col` is the committed write-value column (log_t vars); the
/// read-value claim at `rcycle` is resolved through `resolver`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::vec_init_then_push)]
pub fn prove_twist_ports_sparse(
    w: &TwistPortsWitness,
    wv_col: &DenseMle,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<(TwistProof, Vec<FactorClaim>), PiopError> {
    let log_k = w.log_k;
    let log_t = w.log_t;
    // Per-bit dimensions: d = log_k (each dimension is one address bit).
    let d = log_k.max(1);
    let layout = OneHotLayout::new(log_k, log_t, d, usize::MAX)?;
    let log_n = layout.log_n();
    let t = layout.t();
    let k = layout.k();
    let _ = k;
    // twist-meta absorb (identical bytes to twist.rs).
    transcript
        .append_bytes(
            b"twist-meta",
            &[
                (log_k as u64).to_le_bytes(),
                (log_t as u64).to_le_bytes(),
                (d as u64).to_le_bytes(),
            ]
            .concat(),
        )
        .map_err(PiopError::Transcript)?;

    let read_positions: Vec<u64> = (0..t)
        .map(|j| (w.read_addr[j] << log_t) | j as u64)
        .collect();
    let write_positions: Vec<u64> = (0..t)
        .map(|j| (w.write_addr[j] << log_t) | j as u64)
        .collect();

    // ---- Leg 1: read-checking at rcycle. ----
    let rcycle = transcript.challenge_fields(b"twist-rcycle", log_t)?;
    let rv_claim = resolver.eval(FactorId::ReadValues, &rcycle)?;
    transcript.append_field(b"twist-rv", &rv_claim)?;
    let mut ra_dims = Vec::with_capacity(d);
    for i in 0..d {
        let mut var_map: Vec<usize> = (i * log_n..(i + 1) * log_n).collect();
        var_map.extend(log_k..log_k + log_t);
        let entries: Vec<(u64, Goldilocks)> = (0..t)
            .map(|j| {
                let digits = layout.digits(w.read_addr[j])?;
                Ok((((digits[i] as u64) << log_t) | j as u64, Goldilocks::ONE))
            })
            .collect::<Result<Vec<_>, PiopError>>()?;
        ra_dims.push(SparseFactor { entries, var_map });
    }
    let eq_j = DenseFactor::Table(DenseMle::eq_extension(&rcycle), (log_k..log_k + log_t).collect());
    // The Val factor: the virtual route (O(K + T) state, the container-
    // scale cap lift) or the materialized table — byte-identical proofs.
    let val_f = match &w.val_spec {
        Some(spec) => DenseFactor::VirtualVal(spec.clone()),
        None => {
            let val_mle = DenseMle::new(w.val.clone())?;
            DenseFactor::Table(val_mle, (0..log_k + log_t).collect())
        }
    };
    let inst1 = SparseInstance {
        num_vars: log_k + log_t,
        sparse: ra_dims.clone(),
        dense: vec![eq_j, val_f],
        terms: vec![SparseTerm {
            coeff: Goldilocks::ONE,
            positions: read_positions,
            sparse: (0..d).collect(),
            dense: vec![0, 1],
        }],
    };
    let out1 = prove_sparse_sumcheck(&inst1, rv_claim, transcript)?;
    let read_checking = out1.proof;
    let rho1 = &out1.challenges;

    // ---- Leg 2: Inc definition over (k, j). ----
    let r_inc = transcript.challenge_fields(b"twist-rinc", log_k + log_t)?;
    let eq_full = DenseFactor::Table(DenseMle::eq_extension(&r_inc), (0..log_k + log_t).collect());
    let wv_proj = DenseFactor::Table(wv_col.clone(), (log_k..log_k + log_t).collect());
    let val_f2 = match &w.val_spec {
        Some(spec) => DenseFactor::VirtualVal(spec.clone()),
        None => {
            let val_mle2 = DenseMle::new(w.val.clone())?;
            DenseFactor::Table(val_mle2, (0..log_k + log_t).collect())
        }
    };
    let inc_positions: Vec<u64> = w.inc.iter().map(|&(p, _)| p).collect();
    let inc_entries: Vec<(u64, Goldilocks)> = w.inc.clone();
    let inc_f = SparseFactor {
        entries: inc_entries.clone(),
        var_map: (0..log_k + log_t).collect(),
    };
    let mut wa_dims = Vec::with_capacity(d);
    for i in 0..d {
        let mut var_map: Vec<usize> = (i * log_n..(i + 1) * log_n).collect();
        var_map.extend(log_k..log_k + log_t);
        let entries: Vec<(u64, Goldilocks)> = (0..t)
            .map(|j| {
                let digits = layout.digits(w.write_addr[j])?;
                Ok((((digits[i] as u64) << log_t) | j as u64, Goldilocks::ONE))
            })
            .collect::<Result<Vec<_>, PiopError>>()?;
        wa_dims.push(SparseFactor { entries, var_map });
    }
    // Pool: sparse [0..d]=ra (unused here), then wa dims, then Inc.
    // Rebuild a clean pool for leg 2: [wa dims..., Inc] + dense [eq, wv, val].
    let mut leg2_sparse: Vec<SparseFactor> = Vec::new();
    for f in &wa_dims {
        leg2_sparse.push(f.clone());
    }
    leg2_sparse.push(inc_f);
    let mut leg2_terms: Vec<SparseTerm> = Vec::with_capacity(3);
    // + eq·Inc
    leg2_terms.push(SparseTerm {
        coeff: Goldilocks::ONE,
        positions: inc_positions.clone(),
        sparse: vec![d], // Inc is pool index d (after d wa dims)
        dense: vec![0],
    });
    // − eq·(Π_i wa_i)·wv  +  eq·(Π_i wa_i)·Val — the COMBINED one-hot
    // selects the written cell (the doc's Inc definition; the per-dim
    // expansion would double-count for d > 1).
    leg2_terms.push(SparseTerm {
        coeff: Goldilocks::ONE.neg(),
        positions: write_positions.clone(),
        sparse: (0..d).collect(),
        dense: vec![0, 1],
    });
    leg2_terms.push(SparseTerm {
        coeff: Goldilocks::ONE,
        positions: write_positions.clone(),
        sparse: (0..d).collect(),
        dense: vec![0, 2],
    });
    let inst2 = SparseInstance {
        num_vars: log_k + log_t,
        sparse: leg2_sparse,
        dense: vec![eq_full, wv_proj, val_f2],
        terms: leg2_terms,
    };
    let out2 = prove_sparse_sumcheck(&inst2, Goldilocks::ZERO, transcript)?;
    let inc_definition = out2.proof;
    let rho2 = &out2.challenges;

    // ---- Leg 3: telescoping. ----
    let r_tel = transcript.challenge_fields(b"twist-rtel", log_k + log_t)?;
    let r_k: Vec<Goldilocks> = r_tel.iter().take(log_k).copied().collect();
    let eq_k = DenseFactor::Table(DenseMle::eq_extension(&r_k), (0..log_k).collect());
    let inc_f3 = SparseFactor {
        entries: inc_entries,
        var_map: (0..log_k + log_t).collect(),
    };
    let mut claim = Goldilocks::ZERO;
    let eq_k_evals = DenseMle::eq_extension(&r_k);
    for (kk, dv) in w.final_state.iter().zip(w.init.iter()).enumerate() {
        let wt = eq_k_evals.evaluations[kk];
        claim = claim.add(&wt.mul(&dv.0.sub(dv.1)));
    }
    let inst3 = SparseInstance {
        num_vars: log_k + log_t,
        sparse: vec![inc_f3],
        dense: vec![eq_k],
        terms: vec![SparseTerm {
            coeff: Goldilocks::ONE,
            positions: inc_positions.clone(),
            sparse: vec![0],
            dense: vec![0],
        }],
    };
    let out3 = prove_sparse_sumcheck(&inst3, claim, transcript)?;
    let telescoping = out3.proof;
    let _rho3 = &out3.challenges;

    // ---- Factor claims for the envelope claim table. ----
    let mut claims: Vec<FactorClaim> = Vec::new();
    // The read-value claim at rcycle (rv column evaluation).
    claims.push((FactorId::ReadValues, rcycle.clone(), rv_claim));
    // ra dims at leg-1 native points.
    for i in 0..d {
        let mut point: Vec<Goldilocks> = rho1[i * log_n..(i + 1) * log_n].to_vec();
        point.extend(rho1[log_k..log_k + log_t].iter().copied());
        claims.push((FactorId::Ra(i), point, out1.sparse_claims[i]));
    }
    // Val at leg-1 point and leg-2 point.
    claims.push((FactorId::Val, rho1.to_vec(), out1.dense_claims[1]));
    claims.push((FactorId::Val, rho2.to_vec(), out2.dense_claims[2]));
    // Inc at the leg-2 and leg-3 sumcheck points (rho2 / rho3).
    claims.push((
        FactorId::Inc,
        out2.challenges.clone(),
        out2.sparse_claims[d],
    ));
    claims.push((
        FactorId::Inc,
        out3.challenges.clone(),
        out3.sparse_claims[0],
    ));
    // wa dims at leg-2 native points.
    for i in 0..d {
        let mut point: Vec<Goldilocks> = rho2[i * log_n..(i + 1) * log_n].to_vec();
        point.extend(rho2[log_k..log_k + log_t].iter().copied());
        claims.push((FactorId::Wa(i), point, out2.sparse_claims[i]));
    }
    // wv at the leg-2 cycle point.
    let wv_point: Vec<Goldilocks> = rho2[log_k..log_k + log_t].to_vec();
    claims.push((FactorId::WriteValues, wv_point, out2.dense_claims[1]));

    Ok((
        TwistProof {
            read_checking,
            inc_definition,
            telescoping,
        },
        claims,
    ))
}

/// Verify the ports Twist proof, checking the caller-bound terminal
/// identities against resolver-supplied factor claims (the pipeline
/// authenticates them through grouped openings).
#[allow(clippy::too_many_arguments)]
pub fn verify_twist_ports_checked(
    proof: &TwistProof,
    init: &[Goldilocks],
    final_state: &[Goldilocks],
    log_k: usize,
    log_t: usize,
    d: usize,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<(), PiopError> {
    let layout = OneHotLayout::new(log_k, log_t, d, usize::MAX)?;
    let log_n = layout.log_n();
    transcript
        .append_bytes(
            b"twist-meta",
            &[
                (log_k as u64).to_le_bytes(),
                (log_t as u64).to_le_bytes(),
                (d as u64).to_le_bytes(),
            ]
            .concat(),
        )
        .map_err(PiopError::Transcript)?;

    // Leg 1.
    let rcycle = transcript.challenge_fields(b"twist-rcycle", log_t)?;
    let rv_at_rcycle = resolver.eval(FactorId::ReadValues, &rcycle)?;
    transcript.append_field(b"twist-rv", &rv_at_rcycle)?;
    let v1 = proof
        .read_checking
        .verify(log_k + log_t, d + 2, rv_at_rcycle, transcript, None)?;
    let rho1 = &v1.point;
    let rho1_j = &rho1[log_k..];
    let eq_v = DenseMle::eq_eval(&rcycle, rho1_j)?;
    let val_v = resolver.eval(FactorId::Val, rho1)?;
    let mut prod = eq_v.mul(&val_v);
    for i in 0..d {
        let mut native = rho1[i * log_n..(i + 1) * log_n].to_vec();
        native.extend(rho1_j.iter().copied());
        let ra_v = resolver.eval(FactorId::Ra(i), &native)?;
        prod = prod.mul(&ra_v);
    }
    if v1.final_claim != prod {
        return Err(PiopError::FinalCheckFailed("twist leg-1 terminal"));
    }

    // Leg 2.
    let r_inc = transcript.challenge_fields(b"twist-rinc", log_k + log_t)?;
    let v2 =
        proof
            .inc_definition
            .verify(log_k + log_t, d + 2, Goldilocks::ZERO, transcript, None)?;
    let rho2 = &v2.point;
    let rho2_j = &rho2[log_k..];
    let eq2 = DenseMle::eq_eval(&r_inc, rho2)?;
    let inc_v = resolver.eval(FactorId::Inc, rho2)?;
    let wv_v = resolver.eval(FactorId::WriteValues, rho2_j)?;
    let val2 = resolver.eval(FactorId::Val, rho2)?;
    let mut wa_prod = Goldilocks::ONE;
    for i in 0..d {
        let mut native = rho2[i * log_n..(i + 1) * log_n].to_vec();
        native.extend(rho2_j.iter().copied());
        let wa_v = resolver.eval(FactorId::Wa(i), &native)?;
        wa_prod = wa_prod.mul(&wa_v);
    }
    // eq·[Inc − (Π_i wa_i)·wv + (Π_i wa_i)·Val]
    let expect = eq2
        .mul(&inc_v)
        .sub(&eq2.mul(&wa_prod).mul(&wv_v))
        .add(&eq2.mul(&wa_prod).mul(&val2));
    if v2.final_claim != expect {
        return Err(PiopError::FinalCheckFailed("twist leg-2 terminal"));
    }

    // Leg 3.
    let r_tel = transcript.challenge_fields(b"twist-rtel", log_k + log_t)?;
    let r_k: Vec<Goldilocks> = r_tel.iter().take(log_k).copied().collect();
    let eq_k_evals = DenseMle::eq_extension(&r_k);
    let mut claim = Goldilocks::ZERO;
    for (kk, dv) in final_state.iter().zip(init.iter()).enumerate() {
        let wt = eq_k_evals.evaluations[kk];
        claim = claim.add(&wt.mul(&dv.0.sub(dv.1)));
    }
    let v3 = proof
        .telescoping
        .verify(log_k + log_t, 2, claim, transcript, None)?;
    let rho3 = &v3.point;
    let eq3 = DenseMle::eq_eval(&r_k, &rho3[..log_k])?;
    let inc3 = resolver.eval(FactorId::Inc, rho3)?;
    if v3.final_claim != eq3.mul(&inc3) {
        return Err(PiopError::FinalCheckFailed("twist leg-3 terminal"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onehot::one_hot_dim_matrix;
    use crate::onehot_check::{prove_onehot, verify_onehot, OneHotSide};
    use crate::shout::{prove_shout, verify_shout};
    use crate::WitnessResolver;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn sparse_shout_matches_dense_proof_exactly() {
        // K = 8 table, T = 4 reads, d = 1.
        let table: Vec<Goldilocks> = [10u64, 21, 32, 43, 54, 65, 76, 87].map(fe).to_vec();
        let reads = vec![2u64, 7, 0, 2];
        let log_k = 3usize;
        let log_t = 2usize;
        let matrix = one_hot_dim_matrix(
            &reads.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            log_k,
            log_t,
        )
        .ok()
        .unwrap();
        let mut rv: Vec<Goldilocks> = reads
            .iter()
            .map(|&a| fe(table[a as usize].to_canonical_u64()))
            .collect();
        rv.resize(1 << log_t, Goldilocks::ZERO);
        let rv_col = DenseMle::new(rv).ok().unwrap();
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_values: Some(&rv_col),
            ..Default::default()
        };
        // Dense proof.
        let mut t1 = Transcript::new_default(b"shout-test");
        let dense_proof = prove_shout(
            &table,
            std::slice::from_ref(&matrix),
            log_k,
            log_t,
            &resolver,
            &mut t1,
        )
        .ok()
        .unwrap();
        // Sparse proof.
        let mut t2 = Transcript::new_default(b"shout-test");
        let (sparse_proof, _) =
            prove_shout_sparse(&table, &reads, log_k, log_t, 1, &resolver, &mut t2)
                .ok()
                .unwrap();
        assert_eq!(
            dense_proof.read_checking.rounds,
            sparse_proof.read_checking.rounds
        );
        // The sparse proof verifies through the standard verifier.
        let mut t3 = Transcript::new_default(b"shout-test");
        assert!(verify_shout(&sparse_proof, &table, log_k, log_t, 1, &resolver, &mut t3).is_ok());
    }

    #[test]
    fn sparse_shout_d2_matches_dense() {
        let log_k = 4usize;
        let log_t = 2usize;
        let table: Vec<Goldilocks> = (0..16u64).map(|i| fe(i * 7 + 3)).collect();
        let reads = vec![0u64, 7, 15, 3];
        let layout = OneHotLayout::new(log_k, log_t, 2, usize::MAX).ok().unwrap();
        let digits: Vec<Vec<u32>> = reads
            .iter()
            .map(|&a| layout.digits(a).ok().unwrap())
            .collect();
        let m0 = one_hot_dim_matrix(&digits.iter().map(|d| d[0]).collect::<Vec<_>>(), 2, log_t)
            .ok()
            .unwrap();
        let m1 = one_hot_dim_matrix(&digits.iter().map(|d| d[1]).collect::<Vec<_>>(), 2, log_t)
            .ok()
            .unwrap();
        let mut rv: Vec<Goldilocks> = reads.iter().map(|&a| fe(a * 7 + 3)).collect();
        rv.resize(4, Goldilocks::ZERO);
        let rv_col = DenseMle::new(rv).ok().unwrap();
        let resolver = WitnessResolver {
            ra: vec![Some(&m0), Some(&m1)],
            read_values: Some(&rv_col),
            ..Default::default()
        };
        let mut t1 = Transcript::new_default(b"shout-test");
        let dense_proof = prove_shout(
            &table,
            &[m0.clone(), m1.clone()],
            log_k,
            log_t,
            &resolver,
            &mut t1,
        )
        .ok()
        .unwrap();
        let mut t2 = Transcript::new_default(b"shout-test");
        let (sparse_proof, _) =
            prove_shout_sparse(&table, &reads, log_k, log_t, 2, &resolver, &mut t2)
                .map_err(|e| panic!("d2 sparse err: {e:?}"))
                .ok()
                .unwrap();
        assert_eq!(
            dense_proof.read_checking.rounds,
            sparse_proof.read_checking.rounds
        );
        let mut t3 = Transcript::new_default(b"shout-test");
        assert!(verify_shout(&sparse_proof, &table, log_k, log_t, 2, &resolver, &mut t3).is_ok());
    }

    #[test]
    fn sparse_onehot_matches_dense() {
        let log_k = 3usize;
        let log_t = 3usize;
        let addr = vec![1u64, 5, 2, 7, 0, 3, 6, 4];
        let matrix = one_hot_dim_matrix(
            &addr.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            log_k,
            log_t,
        )
        .ok()
        .unwrap();
        let mut col: Vec<Goldilocks> = addr.iter().map(|&a| fe(a)).collect();
        col.resize(8, Goldilocks::ZERO);
        let col_m = DenseMle::new(col).ok().unwrap();
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_addr: Some(&col_m),
            ..Default::default()
        };
        let mut t1 = Transcript::new_default(b"onehot-test");
        let dense_proof = prove_onehot(
            std::slice::from_ref(&matrix),
            log_k,
            log_t,
            OneHotSide::Read,
            &resolver,
            &mut t1,
        )
        .ok()
        .unwrap();
        let mut t2 = Transcript::new_default(b"onehot-test");
        let (sparse_proof, _cl) =
            prove_onehot_sparse(&addr, log_k, log_t, 1, OneHotSide::Read, &resolver, &mut t2)
                .map_err(|e| panic!("onehot sparse err: {e:?}"))
                .ok()
                .unwrap();
        assert_eq!(dense_proof.booleanity.len(), sparse_proof.booleanity.len());
        for (a, b) in dense_proof
            .booleanity
            .iter()
            .zip(sparse_proof.booleanity.iter())
        {
            assert_eq!(a.rounds, b.rounds);
        }
        assert_eq!(dense_proof.raf.rounds, sparse_proof.raf.rounds);
        let mut t3 = Transcript::new_default(b"onehot-test");
        assert!(verify_onehot(
            &sparse_proof,
            log_k,
            log_t,
            OneHotSide::Read,
            &resolver,
            &mut t3
        )
        .is_ok());
    }

    #[test]
    fn twist_ports_honest_proves_and_verifies() {
        // 4-address memory, 8 cycles: each cycle reads AND writes.
        let log_k = 2usize;
        let log_t = 3usize;
        let k = 4usize;
        let init = vec![fe(0); k];
        // (read_addr, write_addr, write_val) per cycle.
        let plan: [(u64, u64, u64); 8] = [
            (0, 0, 10),
            (1, 1, 0),
            (0, 2, 30),
            (2, 0, 10),
            (2, 1, 5),
            (1, 1, 5),
            (1, 3, 0),
            (3, 2, 30),
        ];
        let read_addr: Vec<u64> = plan.iter().map(|p| p.0).collect();
        let write_addr: Vec<u64> = plan.iter().map(|p| p.1).collect();
        // Write values: cycles with a nonzero plan value write it; the
        // others are dummy writes carrying the current value (zero inc).
        let mut running = init.clone();
        let mut write_val: Vec<Goldilocks> = Vec::with_capacity(8);
        for p in plan.iter() {
            let cur = running[p.1 as usize];
            let v = if p.2 != 0 { fe(p.2) } else { cur };
            write_val.push(v);
            running[p.1 as usize] = v;
        }
        let w = build_twist_ports(&read_addr, &write_addr, &write_val, &init, log_k, log_t)
            .ok()
            .unwrap();
        // rv column: the observed read values = Val(read_addr[j], j).
        let t = 8;
        let mut rv_vals = Vec::with_capacity(t);
        for j in 0..t {
            rv_vals.push(w.val[read_addr[j] as usize * t + j]);
        }
        let rv_col = DenseMle::new(rv_vals).ok().unwrap();
        let wv_col = DenseMle::new(write_val.clone()).ok().unwrap();
        let resolver = WitnessResolver {
            read_values: Some(&rv_col),
            write_values: Some(&wv_col),
            ..Default::default()
        };
        let mut tr = Transcript::new_default(b"lzx-twist-ports");
        let (proof, claims) = prove_twist_ports_sparse(&w, &wv_col, &resolver, &mut tr)
            .map_err(|e| panic!("prove err: {e:?}"))
            .ok()
            .unwrap();
        assert!(!claims.is_empty());
        // Verifier with a claim-table-backed resolver.
        struct TableResolver<'a> {
            claims: &'a [FactorClaim],
        }
        impl<'a> FactorResolver for TableResolver<'a> {
            fn eval(
                &self,
                factor: FactorId,
                point: &[Goldilocks],
            ) -> Result<Goldilocks, PiopError> {
                for (f, p, v) in self.claims {
                    if *f == factor && p.as_slice() == point {
                        return Ok(*v);
                    }
                }
                Err(PiopError::MissingFactor { factor })
            }
        }
        let table_res = TableResolver { claims: &claims };
        let mut vt = Transcript::new_default(b"lzx-twist-ports");
        let res = verify_twist_ports_checked(
            &proof,
            &w.init,
            &w.final_state,
            log_k,
            log_t,
            log_k,
            &table_res,
            &mut vt,
        );
        assert!(
            res.is_ok(),
            "honest ports twist must verify: {:?}",
            res.err()
        );
        // Tampered final state must fail the telescoping claim.
        let mut bad_final = w.final_state.clone();
        bad_final[0] = bad_final[0].add(&fe(1));
        let mut vt2 = Transcript::new_default(b"lzx-twist-ports");
        assert!(verify_twist_ports_checked(
            &proof, &w.init, &bad_final, log_k, log_t, log_k, &table_res, &mut vt2
        )
        .is_err());
    }

    #[test]
    fn zeros_are_free_large_shout() {
        // K = 2^10 table, T = 2^10 reads, d = 10 (bit dims): the sparse
        // engine proves in O(T·n) — the dense route would need 2^20 cells.
        let log_k = 10usize;
        let log_t = 10usize;
        let table: Vec<Goldilocks> = (0..(1u64 << log_k)).map(|i| fe(i * 11 + 5)).collect();
        let reads: Vec<u64> = (0..(1usize << log_t))
            .map(|j| ((j.wrapping_mul(2654435761) >> 22) as u64) & ((1u64 << log_k) - 1))
            .collect();
        let mut rv: Vec<Goldilocks> = reads.iter().map(|&a| table[a as usize]).collect();
        let rv_col = DenseMle::new(rv.clone()).ok().unwrap();
        rv.clear();
        let resolver = WitnessResolver {
            read_values: Some(&rv_col),
            ..Default::default()
        };
        let mut tr = Transcript::new_default(b"shout-big");
        let start = std::time::Instant::now();
        let (proof, _) =
            prove_shout_sparse(&table, &reads, log_k, log_t, log_k, &resolver, &mut tr)
                .ok()
                .unwrap();
        let elapsed = start.elapsed();
        // 2^20 dense cells would be the dense route; here we touched T
        // entries per round: assert the proof exists and completes fast.
        assert!(!proof.read_checking.rounds.is_empty());
        assert_eq!(proof.read_checking.rounds.len(), log_k + log_t);
        assert!(
            elapsed.as_secs() < 60,
            "sparse shout must be fast, took {elapsed:?}"
        );
    }
}

#[cfg(test)]
mod identity_differential {
    use super::*;
    use crate::WitnessResolver;
    use lattice_sumcheck::sumcheck;
    use lattice_sumcheck::VirtualPolynomial;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// Byte-identity of the sparse engine against the dense engine over
    /// an arbitrary mixed instance: identity-var_map dense factors, full
    /// boolean 0/1 sparse columns, terms with sparse+dense mixes and an
    /// all-dense term — the constraint-family (shift) shape.
    ///
    /// THE MULTI-SPARSE DISCIPLINE (a real correctness finding this test
    /// pins): a term with TWO sparse factors CANNOT filter their entries
    /// to the boolean intersection of their supports — the multilinear
    /// products have suffix-level cross terms OUTSIDE the boolean
    /// intersection (the round polynomials at t >= 2 sample the
    /// extensions at non-boolean points), so intersection filtering
    /// silently drops nonzero contributions and the round messages
    /// diverge from the dense engine at every t >= 2 while still
    /// agreeing at t in {0,1} (the sum checks pass!). The correct
    /// constructions are: (a) at most ONE sparse factor per term with
    /// its OWN full support (the constraint families' route —
    /// selectors ride as dense factors), or (b) multiple sparse factors
    /// whose entries are aligned over the UNION of their supports,
    /// zero-padded where a factor has no nonzero row. Both are
    /// exercised here.
    #[test]
    fn mixed_sparse_instance_matches_dense_engine() {
        for n in [4usize, 6] {
            let r: Vec<Goldilocks> = (0..n).map(|i| fe(0x1234 + i as u64)).collect();
            let eq = DenseMle::eq_extension(&r);
            let row_a = DenseMle::random(n, b"row-a");
            let row_b = DenseMle::random(n, b"row-b");
            let mut rng: u64 = 99;
            let mut next = || {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                rng
            };
            let mut mkcol = |count: u64| -> (DenseMle, Vec<(u64, Goldilocks)>) {
                let mut vals = vec![0u64; 1 << n];
                let mut placed = 0u64;
                while placed < count {
                    let idx = (next() % (1u64 << n)) as usize;
                    if vals[idx] == 0 {
                        vals[idx] = 1;
                        placed += 1;
                    }
                }
                let mle = DenseMle::new(vals.iter().map(|v| fe(*v)).collect())
                    .ok()
                    .unwrap();
                let entries = vals
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v != 0)
                    .map(|(i, v)| (i as u64, fe(*v)))
                    .collect();
                (mle, entries)
            };
            let (col0_mle, col0_entries) = mkcol(5);
            let (col1_mle, col1_entries) = mkcol(7);

            // ---- Dense reference ----
            let mut vp = VirtualPolynomial::new(n);
            let f_eq = vp.add_factor(eq.clone()).ok().unwrap();
            let f_a = vp.add_factor(row_a.clone()).ok().unwrap();
            let f_b = vp.add_factor(row_b.clone()).ok().unwrap();
            let f_c0 = vp.add_factor(col0_mle.clone()).ok().unwrap();
            let f_c1 = vp.add_factor(col1_mle.clone()).ok().unwrap();
            vp.add_term(fe(3), vec![f_c0, f_eq]).ok().unwrap();
            vp.add_term(fe(5).neg(), vec![f_a, f_b, f_eq]).ok().unwrap();
            // The multi-sparse term (union construction).
            vp.add_term(fe(7), vec![f_c0, f_c1, f_a, f_eq])
                .ok()
                .unwrap();
            // The true total sum over the cube (the claimed sum).
            let cube = 1usize << n;
            let mut total = Goldilocks::ZERO;
            for x in 0..cube {
                let pt: Vec<Goldilocks> = (0..n)
                    .map(|i| fe(((x >> (n - 1 - i)) & 1) as u64))
                    .collect();
                let e = eq.evaluate(&pt).ok().unwrap();
                let av = row_a.evaluate(&pt).ok().unwrap();
                let bv = row_b.evaluate(&pt).ok().unwrap();
                let c0 = col0_mle.evaluate(&pt).ok().unwrap();
                let c1 = col1_mle.evaluate(&pt).ok().unwrap();
                total = total
                    .add(&fe(3).mul(&c0).mul(&e))
                    .add(&fe(5).neg().mul(&av).mul(&bv).mul(&e))
                    .add(&fe(7).mul(&c0).mul(&c1).mul(&av).mul(&e));
            }
            let mut tr_d = Transcript::new_default(b"sc-diff");
            let out_d = sumcheck::prove(&vp, total, &mut tr_d).ok().unwrap();

            // ---- Sparse instance (the correct constructions) ----
            let var_map: Vec<usize> = (0..n).collect();
            // The UNION of the two columns' supports with zero-padded
            // aligned entries (discipline (b) for multi-sparse terms).
            let mut union_rows: Vec<u64> = col0_entries
                .iter()
                .chain(col1_entries.iter())
                .map(|e| e.0)
                .collect();
            union_rows.sort_unstable();
            union_rows.dedup();
            let padded = |entries: &[(u64, Goldilocks)]| -> Vec<(u64, Goldilocks)> {
                union_rows
                    .iter()
                    .map(|&row| {
                        let v = entries
                            .iter()
                            .find(|e| e.0 == row)
                            .map(|e| e.1)
                            .unwrap_or(Goldilocks::ZERO);
                        (row, v)
                    })
                    .collect()
            };
            let inst = SparseInstance {
                num_vars: n,
                sparse: vec![
                    // col0 (single-sparse term's own support).
                    SparseFactor {
                        entries: col0_entries.clone(),
                        var_map: var_map.clone(),
                    },
                    // col0 and col1 aligned over the union (zero-padded).
                    SparseFactor {
                        entries: padded(&col0_entries),
                        var_map: var_map.clone(),
                    },
                    SparseFactor {
                        entries: padded(&col1_entries),
                        var_map: var_map.clone(),
                    },
                ],
                dense: vec![
                    DenseFactor::Table(eq.clone(), var_map.clone()),
                    DenseFactor::Table(row_a.clone(), var_map.clone()),
                    DenseFactor::Table(row_b.clone(), var_map.clone()),
                ],
                terms: vec![
                    SparseTerm {
                        coeff: fe(3),
                        positions: col0_entries.iter().map(|e| e.0).collect(),
                        sparse: vec![0],
                        dense: vec![0],
                    },
                    SparseTerm {
                        coeff: fe(5).neg(),
                        positions: (0..(1u64 << n)).collect(),
                        sparse: vec![],
                        dense: vec![1, 2, 0],
                    },
                    SparseTerm {
                        coeff: fe(7),
                        positions: union_rows.clone(),
                        sparse: vec![1, 2],
                        dense: vec![1, 0],
                    },
                ],
            };
            let mut tr_s = Transcript::new_default(b"sc-diff");
            let out_s = prove_sparse_sumcheck(&inst, total, &mut tr_s).ok().unwrap();

            assert_eq!(out_d.proof.rounds.len(), out_s.proof.rounds.len());
            for (ri, (rd, rs)) in out_d
                .proof
                .rounds
                .iter()
                .zip(out_s.proof.rounds.iter())
                .enumerate()
            {
                assert_eq!(rd, rs, "round {ri} (n={n})");
            }
            assert_eq!(out_d.final_claim, out_s.final_claim);
            assert_eq!(out_d.challenges, out_s.challenges);
        }
    }

    // ---- The virtual-Val route (the container-scale cap lift). ----

    /// The read values from the streams (the running map — no Val matrix).
    fn rv_from_streams(
        read_addr: &[u64],
        write_addr: &[u64],
        write_val: &[Goldilocks],
        init: &[Goldilocks],
    ) -> Vec<Goldilocks> {
        let mut running = init.to_vec();
        let mut out = Vec::with_capacity(read_addr.len());
        for j in 0..read_addr.len() {
            out.push(running[read_addr[j] as usize]);
            running[write_addr[j] as usize] = write_val[j];
        }
        out
    }

    #[test]
    fn twist_ports_virtual_matches_materialized_exactly() {
        // The SAME streams proven through the materialized table and the
        // virtual (init, writes) spec must produce byte-identical proofs —
        // the pin that the virtual route changes nothing on the wire.
        let log_k = 3usize;
        let log_t = 3usize;
        let k = 8usize;
        let t = 8usize;
        let init: Vec<Goldilocks> = (0..k).map(|i| fe((i as u64) * 3 + 1)).collect();
        // A read-write plan with repeated addresses (stale-read structure).
        let read_addr: Vec<u64> = vec![0, 5, 2, 5, 0, 7, 2, 5];
        let write_addr: Vec<u64> = vec![5, 2, 0, 5, 7, 2, 5, 0];
        let write_val: Vec<Goldilocks> =
            (0..t).map(|j| fe((j as u64 * 17 + 3) % 97)).collect();
        let rv_vals =
            rv_from_streams(&read_addr, &write_addr, &write_val, &init);
        let rv_col = DenseMle::new(rv_vals).ok().unwrap();
        let wv_col = DenseMle::new(write_val.clone()).ok().unwrap();
        let resolver = WitnessResolver {
            read_values: Some(&rv_col),
            write_values: Some(&wv_col),
            ..Default::default()
        };
        // Materialized.
        let w_mat = build_twist_ports(&read_addr, &write_addr, &write_val, &init, log_k, log_t)
            .ok().unwrap();
        let mut t1 = Transcript::new_default(b"lzx-twist-ports");
        let (proof_mat, _) = prove_twist_ports_sparse(&w_mat, &wv_col, &resolver, &mut t1)
            .map_err(|e| panic!("materialized prove err: {e:?}")).ok().unwrap();
        // Virtual.
        let w_virt = build_twist_ports_virtual(
            &read_addr, &write_addr, &write_val, &init, log_k, log_t,
        )
        .ok().unwrap();
        assert!(w_virt.val.is_empty(), "the virtual route never materializes");
        let mut t2 = Transcript::new_default(b"lzx-twist-ports");
        let (proof_virt, claims_virt) = prove_twist_ports_sparse(&w_virt, &wv_col, &resolver, &mut t2)
            .map_err(|e| panic!("virtual prove err: {e:?}")).ok().unwrap();
        // Byte-identical proofs.
        assert_eq!(proof_mat.read_checking.rounds, proof_virt.read_checking.rounds);
        assert_eq!(proof_mat.inc_definition.rounds, proof_virt.inc_definition.rounds);
        assert_eq!(proof_mat.telescoping.rounds, proof_virt.telescoping.rounds);
        // The virtual proof verifies through the SAME verifier.
        struct TableResolver<'a> {
            claims: &'a [FactorClaim],
        }
        impl<'a> FactorResolver for TableResolver<'a> {
            fn eval(&self, factor: FactorId, point: &[Goldilocks]) -> Result<Goldilocks, PiopError> {
                for (f, p, v) in self.claims {
                    if *f == factor && p.as_slice() == point {
                        return Ok(*v);
                    }
                }
                Err(PiopError::MissingFactor { factor })
            }
        }
        let table_res = TableResolver { claims: &claims_virt };
        let mut vt = Transcript::new_default(b"lzx-twist-ports");
        let res = verify_twist_ports_checked(
            &proof_virt, &w_virt.init, &w_virt.final_state, log_k, log_t, log_k,
            &table_res, &mut vt,
        );
        assert!(res.is_ok(), "virtual ports twist must verify: {:?}", res.err());
    }

    #[test]
    fn virtual_val_container_scale() {
        // K = 2^12, T = 2^12 — the MATERIALIZED route would allocate the
        // 2^24-cell Val matrix (128 MB of field elements); the virtual
        // route carries O(K + T) state. Prove + verify end-to-end.
        let log_k = 12usize;
        let log_t = 12usize;
        let k = 1usize << log_k;
        let t = 1usize << log_t;
        let init: Vec<Goldilocks> = (0..k).map(|i| fe((i as u64 * 7 + 1) % 1009)).collect();
        // Sparse-address traffic: reads and writes scattered over the
        // container (the regime the dispatch heuristic selects virtual).
        let read_addr: Vec<u64> = (0..t)
            .map(|j| (j as u64).wrapping_mul(0x9E3779B97F4A7C15) >> (64 - log_k))
            .collect();
        let write_addr: Vec<u64> = (0..t)
            .map(|j| ((j as u64).wrapping_mul(0x632BE59BD9B4E019)) >> (64 - log_k))
            .collect();
        let write_val: Vec<Goldilocks> =
            (0..t).map(|j| fe((j as u64 * 13 + 5) % 251)).collect();
        let rv_vals = rv_from_streams(&read_addr, &write_addr, &write_val, &init);
        let rv_col = DenseMle::new(rv_vals).ok().unwrap();
        let wv_col = DenseMle::new(write_val.clone()).ok().unwrap();
        let resolver = WitnessResolver {
            read_values: Some(&rv_col),
            write_values: Some(&wv_col),
            ..Default::default()
        };
        let w = build_twist_ports_virtual(
            &read_addr, &write_addr, &write_val, &init, log_k, log_t,
        )
        .ok().unwrap();
        assert!(w.val.is_empty());
        let mut tr = Transcript::new_default(b"lzx-twist-ports");
        let start = std::time::Instant::now();
        let (proof, claims) = prove_twist_ports_sparse(&w, &wv_col, &resolver, &mut tr)
            .map_err(|e| panic!("container-scale virtual prove err: {e:?}"))
            .ok().unwrap();
        let elapsed = start.elapsed();
        assert_eq!(proof.read_checking.rounds.len(), log_k + log_t);
        // Verify through the standard verifier with the claim table.
        struct TableResolver<'a> {
            claims: &'a [FactorClaim],
        }
        impl<'a> FactorResolver for TableResolver<'a> {
            fn eval(&self, factor: FactorId, point: &[Goldilocks]) -> Result<Goldilocks, PiopError> {
                for (f, p, v) in self.claims {
                    if *f == factor && p.as_slice() == point {
                        return Ok(*v);
                    }
                }
                Err(PiopError::MissingFactor { factor })
            }
        }
        let table_res = TableResolver { claims: &claims };
        let mut vt = Transcript::new_default(b"lzx-twist-ports");
        let res = verify_twist_ports_checked(
            &proof, &w.init, &w.final_state, log_k, log_t, log_k, &table_res, &mut vt,
        );
        assert!(res.is_ok(), "container-scale virtual twist must verify: {:?}", res.err());
        assert!(
            elapsed.as_secs() < 120,
            "container-scale virtual twist must be tractable, took {elapsed:?}"
        );
    }
}
