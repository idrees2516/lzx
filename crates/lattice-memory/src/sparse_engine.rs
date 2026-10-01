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
    pub dense: Vec<ProjectedDense>,
    pub terms: Vec<SparseTerm>,
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

/// A dense factor's partial value at (bound prefix, t, suffix).
#[allow(clippy::too_many_arguments)]
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
pub fn prove_sparse_sumcheck(
    inst: &SparseInstance,
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

    // Sparse factor working state: accumulated eq weights per entry and
    // the per-variable own-position lookup.
    let mut weights: Vec<Vec<Goldilocks>> = inst
        .sparse
        .iter()
        .map(|f| f.entries.iter().map(|_| Goldilocks::ONE).collect())
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
    // Dense factor working state.
    let mut dense_state: Vec<DenseMle> = inst.dense.iter().map(|f| f.mle.clone()).collect();
    let mut dense_bound: Vec<usize> = vec![0; inst.dense.len()];
    // Per-term entry order sorted by bit-reversed position: entries sharing
    // the same remaining-variable suffix stay contiguous at every round
    // (suffix groups nest as the rounds advance).
    let term_orders: Vec<Vec<usize>> = inst
        .terms
        .iter()
        .map(|t| {
            let mut idx: Vec<usize> = (0..t.positions.len()).collect();
            idx.sort_by_key(|&j| reverse_bits(t.positions[j], n));
            idx
        })
        .collect();

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
        // Round polynomial values g(0..=deg), computed per suffix group:
        // for the entries sharing a remaining suffix s, every one-hot
        // factor contributes the SUM over its entries and the dense
        // factors contribute their partial value at s — the product of
        // sums reproduces the true round polynomial including the cross
        // terms between entries of different one-hot factors that share
        // a suffix (the per-entry product would drop them).
        let mut evals_at: Vec<Goldilocks> = vec![Goldilocks::ZERO; deg + 1];
        for (ti, term) in inst.terms.iter().enumerate() {
            if term.sparse.is_empty() {
                return Err(PiopError::Shape { expected: 1, got: 0 });
            }
            let order = &term_orders[ti];
            let mut seg = 0usize;
            while seg < order.len() {
                let suffix = term.positions[order[seg]] & suffix_mask;
                let mut end = seg + 1;
                while end < order.len()
                    && (term.positions[order[end]] & suffix_mask) == suffix
                {
                    end += 1;
                }
                for (t, ev) in evals_at.iter_mut().enumerate() {
                    let t_fe = Goldilocks::from_u64(t as u64);
                    let mut prod = term.coeff;
                    for &di in &term.dense {
                        let v = dense_partial_value(
                            &dense_state[di],
                            dense_bound[di],
                            &inst.dense[di].var_map,
                            ell,
                            n,
                            t_fe,
                            suffix,
                        );
                        prod = prod.mul(&v);
                    }
                    for &fi in &term.sparse {
                        let f = &inst.sparse[fi];
                        let mut sum = Goldilocks::ZERO;
                        for k in seg..end {
                            let j = order[k];
                            let (_, val) = f.entries[j];
                            let base = weights[fi][j].mul(&val);
                            let p = sparse_pos[fi][ell];
                            if p == usize::MAX {
                                sum = sum.add(&base);
                            } else {
                                let bit = (term.positions[j] >> (n - 1 - ell)) & 1;
                                sum = sum.add(&base.mul(&eq_lerp(t_fe, bit)));
                            }
                        }
                        prod = prod.mul(&sum);
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

        // Bind factors to r.
        for (fi, f) in inst.sparse.iter().enumerate() {
            let p = sparse_pos[fi][ell];
            if p != usize::MAX {
                let shift = f.var_map.len() - 1 - p;
                for (j, &(own, _)) in f.entries.iter().enumerate() {
                    let bit = (own >> shift) & 1;
                    weights[fi][j] = weights[fi][j].mul(&eq_point(&r, bit));
                }
            }
        }
        for (di, df) in inst.dense.iter().enumerate() {
            let len = df.var_map.len();
            if dense_bound[di] < len && df.var_map[dense_bound[di]] == ell {
                dense_state[di] = dense_state[di]
                    .fix_variables(&[r])
                    .map_err(PiopError::Mle)?;
                dense_bound[di] += 1;
            }
        }
    }

    // Terminal: per-factor claims and the final combined claim.
    let mut sparse_claims = Vec::with_capacity(inst.sparse.len());
    for (fi, f) in inst.sparse.iter().enumerate() {
        let mut acc = Goldilocks::ZERO;
        for (j, &(_, val)) in f.entries.iter().enumerate() {
            acc = acc.add(&weights[fi][j].mul(&val));
        }
        sparse_claims.push(acc);
    }
    let mut dense_claims = Vec::with_capacity(inst.dense.len());
    for dstate in &dense_state {
        dense_claims.push(dstate.evaluations[0]);
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
) -> Result<crate::shout::ShoutProof, PiopError> {
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
    let eq_j = ProjectedDense {
        mle: DenseMle::eq_extension(&rcycle),
        var_map: (log_k..log_k + log_t).collect(),
    };
    let table_m = DenseMle::new(table.to_vec())?;
    let val_f = ProjectedDense {
        mle: table_m,
        var_map: (0..log_k).collect(),
    };
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
    Ok(crate::shout::ShoutProof {
        read_checking: out.proof,
    })
}

// ---------------------------------------------------------------------------
// Sparse one-hot constraint PIOP (Figs 6/8).
// ---------------------------------------------------------------------------

/// Sparse one-hot constraint prover over a per-cycle address column.
///
/// Transcript flow identical to [`crate::onehot_check::prove_onehot`].
pub fn prove_onehot_sparse(
    addresses: &[u64],
    log_k: usize,
    log_t: usize,
    d: usize,
    side: crate::onehot_check::OneHotSide,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<crate::onehot_check::OneHotProof, PiopError> {
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
    let positions: Vec<u64> = (0..t)
        .map(|j| (addresses[j] << log_t) | j as u64)
        .collect();

    // 1. Booleanity per dimension (over the dim's own (k_i, j) space).
    let mut booleanity = Vec::with_capacity(d);
    for i in 0..d {
        let r_i = transcript.challenge_fields(b"onehot-bool-r", log_n)?;
        let mut point = r_i.clone();
        point.extend(r_prime.iter().copied());
        let eq_m = DenseMle::eq_extension(&point);
        let mut var_map: Vec<usize> = (i * log_n..(i + 1) * log_n).collect();
        var_map.extend(log_k..log_k + log_t);
        let eq_f = ProjectedDense {
            mle: eq_m,
            var_map,
        };
        let mut entries = Vec::with_capacity(t);
        for j in 0..t {
            entries.push((((digits_per_dim[i][j] as u64) << log_t) | j as u64, Goldilocks::ONE));
        }
        let ra_f = SparseFactor {
            entries,
            var_map: {
                let mut vm: Vec<usize> = (i * log_n..(i + 1) * log_n).collect();
                vm.extend(log_k..log_k + log_t);
                vm
            },
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
        booleanity.push(out.proof);
    }

    // 3. raf-evaluation over the full (k, j) space.
    let eq_j = ProjectedDense {
        mle: DenseMle::eq_extension(&r_prime),
        var_map: (log_k..log_k + log_t).collect(),
    };
    let w_evals: Vec<Goldilocks> = (0..layout.k())
        .map(|k| Goldilocks::from_u64(k as u64))
        .collect();
    let w_f = ProjectedDense {
        mle: DenseMle::new(w_evals)?,
        var_map: (0..log_k).collect(),
    };
    let mut ra_dims = Vec::with_capacity(d);
    for i in 0..d {
        let mut var_map: Vec<usize> = (i * log_n..(i + 1) * log_n).collect();
        var_map.extend(log_k..log_k + log_t);
        let entries: Vec<(u64, Goldilocks)> = (0..t)
            .map(|j| (((digits_per_dim[i][j] as u64) << log_t) | j as u64, Goldilocks::ONE))
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

    Ok(crate::onehot_check::OneHotProof { booleanity, raf: raf_out.proof })
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
    if read_addr.len() != t
        || write_addr.len() != t
        || write_val.len() != t
        || init.len() != k
    {
        return Err(PiopError::Shape { expected: t, got: read_addr.len() });
    }
    for (a, b) in read_addr.iter().zip(write_addr.iter()) {
        if *a >= k as u64 || *b >= k as u64 {
            return Err(PiopError::AddressOutOfRange { address: (*a).max(*b), k });
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

    let read_positions: Vec<u64> =
        (0..t).map(|j| (w.read_addr[j] << log_t) | j as u64).collect();
    let write_positions: Vec<u64> =
        (0..t).map(|j| (w.write_addr[j] << log_t) | j as u64).collect();

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
    let eq_j = ProjectedDense {
        mle: DenseMle::eq_extension(&rcycle),
        var_map: (log_k..log_k + log_t).collect(),
    };
    let val_mle = DenseMle::new(w.val.clone())?;
    let val_f = ProjectedDense {
        mle: val_mle,
        var_map: (0..log_k + log_t).collect(),
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
    let eq_full = ProjectedDense {
        mle: DenseMle::eq_extension(&r_inc),
        var_map: (0..log_k + log_t).collect(),
    };
    let wv_proj = ProjectedDense {
        mle: wv_col.clone(),
        var_map: (log_k..log_k + log_t).collect(),
    };
    let val_mle2 = DenseMle::new(w.val.clone())?;
    let val_f2 = ProjectedDense {
        mle: val_mle2,
        var_map: (0..log_k + log_t).collect(),
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
    let mut leg2_terms: Vec<SparseTerm> = Vec::new();
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
    let eq_k = ProjectedDense {
        mle: DenseMle::eq_extension(&r_k),
        var_map: (0..log_k).collect(),
    };
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
    claims.push((FactorId::Inc, out2.challenges.clone(), out2.sparse_claims[d]));
    claims.push((FactorId::Inc, out3.challenges.clone(), out3.sparse_claims[0]));
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
    let v2 = proof
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
    use crate::shout::{prove_shout, verify_shout};
    use crate::onehot_check::{prove_onehot, verify_onehot, OneHotSide};
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
        let mut rv: Vec<Goldilocks> = reads.iter().map(|&a| fe(table[a as usize].to_canonical_u64())).collect();
        rv.resize(1 << log_t, Goldilocks::ZERO);
        let rv_col = DenseMle::new(rv).ok().unwrap();
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_values: Some(&rv_col),
            ..Default::default()
        };
        // Dense proof.
        let mut t1 = Transcript::new_default(b"shout-test");
        let dense_proof = prove_shout(&table, &[matrix.clone()], log_k, log_t, &resolver, &mut t1)
            .ok().unwrap();
        // Sparse proof.
        let mut t2 = Transcript::new_default(b"shout-test");
        let sparse_proof =
            prove_shout_sparse(&table, &reads, log_k, log_t, 1, &resolver, &mut t2)
                .ok().unwrap();
        assert_eq!(dense_proof.read_checking.rounds, sparse_proof.read_checking.rounds);
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
        let digits: Vec<Vec<u32>> = reads.iter().map(|&a| layout.digits(a).ok().unwrap()).collect();
        let m0 = one_hot_dim_matrix(&digits.iter().map(|d| d[0]).collect::<Vec<_>>(), 2, log_t)
            .ok().unwrap();
        let m1 = one_hot_dim_matrix(&digits.iter().map(|d| d[1]).collect::<Vec<_>>(), 2, log_t)
            .ok().unwrap();
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
        let sparse_proof = prove_shout_sparse(&table, &reads, log_k, log_t, 2, &resolver, &mut t2)
            .map_err(|e| panic!("d2 sparse err: {e:?}")).ok().unwrap();
        assert_eq!(dense_proof.read_checking.rounds, sparse_proof.read_checking.rounds);
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
        let dense_proof = prove_onehot(&[matrix.clone()], log_k, log_t, OneHotSide::Read, &resolver, &mut t1)
            .ok().unwrap();
        let mut t2 = Transcript::new_default(b"onehot-test");
        let sparse_proof =
            prove_onehot_sparse(&addr, log_k, log_t, 1, OneHotSide::Read, &resolver, &mut t2)
                .map_err(|e| panic!("onehot sparse err: {e:?}")).ok().unwrap();
        assert_eq!(dense_proof.booleanity.len(), sparse_proof.booleanity.len());
        for (a, b) in dense_proof.booleanity.iter().zip(sparse_proof.booleanity.iter()) {
            assert_eq!(a.rounds, b.rounds);
        }
        assert_eq!(dense_proof.raf.rounds, sparse_proof.raf.rounds);
        let mut t3 = Transcript::new_default(b"onehot-test");
        assert!(verify_onehot(&sparse_proof, log_k, log_t, OneHotSide::Read, &resolver, &mut t3).is_ok());
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
            .ok().unwrap();
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
            .ok().unwrap();
        assert!(!claims.is_empty());
        // Verifier with a claim-table-backed resolver.
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
        assert!(res.is_ok(), "honest ports twist must verify: {:?}", res.err());
        // Tampered final state must fail the telescoping claim.
        let mut bad_final = w.final_state.clone();
        bad_final[0] = bad_final[0].add(&fe(1));
        let mut vt2 = Transcript::new_default(b"lzx-twist-ports");
        assert!(
            verify_twist_ports_checked(&proof, &w.init, &bad_final, log_k, log_t, log_k, &table_res, &mut vt2).is_err()
        );
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
        let mut rv: Vec<Goldilocks> =
            reads.iter().map(|&a| table[a as usize]).collect();
        let rv_col = DenseMle::new(rv.clone()).ok().unwrap();
        rv.clear();
        let resolver = WitnessResolver {
            read_values: Some(&rv_col),
            ..Default::default()
        };
        let mut tr = Transcript::new_default(b"shout-big");
        let start = std::time::Instant::now();
        let proof = prove_shout_sparse(&table, &reads, log_k, log_t, log_k, &resolver, &mut tr)
            .ok().unwrap();
        let elapsed = start.elapsed();
        // 2^20 dense cells would be the dense route; here we touched T
        // entries per round: assert the proof exists and completes fast.
        assert!(!proof.read_checking.rounds.is_empty());
        assert_eq!(proof.read_checking.rounds.len(), log_k + log_t);
        assert!(elapsed.as_secs() < 60, "sparse shout must be fast, took {elapsed:?}");
    }
}
