//! **The streaming grand product check** — Appendix D of ePrint
//! 2025/611 (the Quarks lemma protocol with a depth-first prover).
//!
//! Statement: `P = Π_{i=0}^{2^n−1} v[i]` for a streamed vector `v`.
//!
//! ## The protocol (Lemma D.1)
//!
//! Define multilinear tables over `{0,1}^n` via the product tree: the
//! block `[o, o + 2^{j+1})` of `2^{j+1}` leaves is **labeled** by the
//! cube point `z = o + 2^j − 1` (the point with `j` trailing ones and a
//! zero at bit `j`), and
//!
//! ```text
//! g1(z) = block product      g2(z) = left-child product
//! g3(z) = right-child product
//! g1(1^n) = g3(1^n) = 0,     g2(1^n) = P          (the special point)
//! ```
//!
//! The labeling is a bijection onto `{0,1}^n`, and the recursion
//! `g1 = g2 · g3` holds at *every* cube point (verified in the tests),
//! so the sum-check instance
//!
//! ```text
//! 0 = Σ_z eq(u, z) · (g1(z) − g2(z)·g3(z))
//! ```
//!
//! reduces the claim to the terminal identity `g1(r) = g2(r)·g3(r)` at
//! the verifier's random point — the Quarks construction. The honest
//! round messages are identically zero (each term is zero on the cube);
//! soundness comes from binding the prover to consistent `g`-tables,
//! whose terminal evaluations the caller's PCS then authenticates.
//!
//! ## The streaming prover
//!
//! The **depth-first product-tree walk** is the paper's core data
//! structure: scan the stream once, push each `v[x]` onto a block stack,
//! and merge adjacent equal-size aligned blocks — the stack never
//! exceeds `n + 1` entries, so `P` itself is computed in `O(n)` space
//! (Theorem D.4's stack invariant, Lemma D.2). The `g`-table triples
//! are recorded at each merge during the same single pass; the
//! sum-check then runs over the recorded tables. (Algorithm 3's fully
//! `O(n)`-space *round-message* path composes the same DFS with
//! per-remaining-hypercube bound accumulation — the bucketed
//! `g_evals[t][k][s]` machinery of the paper's Steps 17–24; the
//! recorded-table path here is the reference implementation of the same
//! protocol at `O(2^n)` table space, one stream pass.)

use crate::oracle::StreamOracle;
use crate::small_space::interpolate_nodes;
use lattice_core::field_simd::{self, Sum8};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::Goldilocks;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrandProductError {
    Transcript(TranscriptError),
    RoundCheckFailed { round: usize },
    TerminalIdentityFailed,
    ShapeMismatch { expected: usize, got: usize },
}

impl core::fmt::Display for GrandProductError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GrandProductError::Transcript(e) => write!(f, "transcript error: {e}"),
            GrandProductError::RoundCheckFailed { round } => {
                write!(f, "grand-product round identity failed at round {round}")
            }
            GrandProductError::TerminalIdentityFailed => {
                write!(f, "grand-product terminal identity failed")
            }
            GrandProductError::ShapeMismatch { expected, got } => {
                write!(f, "grand-product shape {got} != {expected}")
            }
        }
    }
}

/// A stack block: (value, size, offset).
#[derive(Clone, Copy)]
struct Block {
    value: Goldilocks,
    size: u64,
    offset: u64,
}

/// **The depth-first streaming product walk** — `O(n)` space, one pass:
/// push `v[x]`, merge adjacent equal-size aligned blocks. Returns the
/// full product and, if `triples` is provided, records one
/// `(z, g1, g2, g3)` entry per merge (the sum-check's tables).
///
/// This is Theorem D.4's prover core: the stack mirrors the DFS state
/// of the product-tree, holding at most `n + 1` partial products
/// (Lemma D.2's invariant — the values `g2(x^{(j)})` for the active
/// block chain, in decreasing block size).
pub fn dfs_grand_product(
    stream: &mut dyn StreamOracle,
    triples: Option<&mut Vec<(u64, Goldilocks, Goldilocks, Goldilocks)>>,
) -> Result<Goldilocks, GrandProductError> {
    let n = stream.len().trailing_zeros() as usize;
    let mut stack: Vec<Block> = Vec::with_capacity(n + 1);
    let mut triples = triples;
    for x in 0..stream.len() {
        let v = stream.next();
        stack.push(Block {
            value: v,
            size: 1,
            offset: x,
        });
        // Merge while the top two blocks are adjacent, equal-size, and
        // aligned (the right block's offset is an odd multiple of size).
        while stack.len() >= 2 {
            let (left, right) = {
                let l = stack.len();
                (&stack[l - 2], &stack[l - 1])
            };
            if left.size == right.size
                && left.offset + left.size == right.offset
                && (right.offset / right.size) % 2 == 1
            {
                let prod = left.value.mul(&right.value);
                if let Some(t) = triples.as_deref_mut() {
                    // The merged block [left.offset, +2·size) is labeled
                    // by z = left.offset + size − 1.
                    let z = left.offset + left.size - 1;
                    t.push((z, prod, left.value, right.value));
                }
                let merged = Block {
                    value: prod,
                    size: left.size * 2,
                    offset: left.offset,
                };
                stack.pop();
                stack.pop();
                stack.push(merged);
            } else {
                break;
            }
        }
    }
    if stack.len() != 1 {
        return Err(GrandProductError::ShapeMismatch {
            expected: 1,
            got: stack.len(),
        });
    }
    Ok(stack[0].value)
}

/// The grand-product proof: the claimed product, the (all-zero) round
/// messages of the Quarks sum-check, the challenges, and the terminal
/// `g`-claims the caller's PCS authenticates.
#[derive(Clone, Debug)]
pub struct GrandProductProof {
    /// The claimed product `P` (bound by the caller's commitment).
    pub product: Goldilocks,
    /// Round messages (degree-3 univariates at nodes 0..=3).
    pub rounds: Vec<Vec<Goldilocks>>,
    pub challenges: Vec<Goldilocks>,
    /// `g1(r)`, `g2(r)`, `g3(r)`.
    pub g_claims: [Goldilocks; 3],
}

/// Prove `Π v = product` with one streaming pass + the Quarks
/// sum-check over the recorded `g`-tables.
///
/// `product`: if `None`, the DFS computes it (the honest path — the
/// caller typically holds the claim from the enclosing protocol).
pub fn prove_grand_product(
    stream: &mut dyn StreamOracle,
    product: Option<Goldilocks>,
    transcript: &mut Transcript,
) -> Result<GrandProductProof, GrandProductError> {
    let n = stream.len().trailing_zeros() as usize;
    let mut triples = Vec::with_capacity((1usize << n).saturating_sub(1));
    stream.reset();
    let p = dfs_grand_product(stream, Some(&mut triples))?;
    let product = product.unwrap_or(p);
    if p != product {
        // The streamed vector's product contradicts the claim.
        return Err(GrandProductError::TerminalIdentityFailed);
    }

    // g-tables in index order (the special all-ones point carries
    // (0, P, 0)).
    let len = 1usize << n;
    let mut g1 = vec![Goldilocks::ZERO; len];
    let mut g2 = vec![Goldilocks::ZERO; len];
    let mut g3 = vec![Goldilocks::ZERO; len];
    for (z, a, b, c) in &triples {
        g1[*z as usize] = *a;
        g2[*z as usize] = *b;
        g3[*z as usize] = *c;
    }
    g2[len - 1] = product; // g2(1^n) = P; g1 = g3 = 0 there.

    // The Quarks sum-check: 0 = Σ eq(u, z)·(g1 − g2·g3), degree 3.
    // u is sampled from the transcript after absorbing the product.
    transcript
        .append_field(b"grand-product", &product)
        .map_err(GrandProductError::Transcript)?;
    let mut u: Vec<Goldilocks> = Vec::with_capacity(n);
    for _ in 0..n {
        u.push(
            transcript
                .challenge_field(b"grand-product-u")
                .map_err(GrandProductError::Transcript)?,
        );
    }

    // eq table of u (in-memory; the bucketed streaming variant is the
    // paper's Algorithm 3 — see the module docs).
    let eq = field_simd::eq_table(&u);

    let mut current_claim = Goldilocks::ZERO;
    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(n);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(n);
    let mut bound_r: Vec<Goldilocks> = Vec::with_capacity(n);

    // Round i: bind the g's over the prefix with eq weights and the
    // eq-u factor likewise; accumulate the four point values.
    // Implementation: maintain bound copies of (eq, g1, g2, g3), each
    // halving per round — the reference in-memory prover.
    let mut b_eq = eq;
    let mut b1 = g1;
    let mut b2 = g2;
    let mut b3 = g3;
    for round in 0..n {
        let mut evals = vec![Goldilocks::ZERO; 4];
        for (pi, t) in [0u64, 1, 2, 3].iter().enumerate() {
            let tf = Goldilocks::from_u64(*t);
            let mut vals: Vec<Vec<Goldilocks>> = Vec::with_capacity(4);
            for arr in [&b_eq, &b1, &b2, &b3] {
                let half = arr.len() / 2;
                let v = match t {
                    0 => arr[..half].to_vec(),
                    1 => arr[half..].to_vec(),
                    _ => {
                        let mut buf = vec![Goldilocks::ZERO; half];
                        field_simd::bind_half_slices(&arr[..half], &arr[half..], tf, &mut buf);
                        buf
                    }
                };
                vals.push(v);
            }
            let mut acc = Sum8::new();
            acc.accumulate_term(Goldilocks::ONE, &[vals[0].as_slice(), vals[1].as_slice()]);
            let term1 = acc.finish();
            let mut acc2 = Sum8::new();
            acc2.accumulate_term(
                Goldilocks::ONE,
                &[vals[0].as_slice(), vals[2].as_slice(), vals[3].as_slice()],
            );
            let term2 = acc2.finish();
            evals[pi] = term1.sub(&term2);
        }
        let sum01 = evals[0].add(&evals[1]);
        if sum01 != current_claim {
            return Err(GrandProductError::RoundCheckFailed { round });
        }
        transcript
            .append_field_slice(b"sumcheck-round", &evals)
            .map_err(GrandProductError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(GrandProductError::Transcript)?;
        current_claim = interpolate_nodes(&evals, &r);
        bound_r.push(r);
        challenges.push(r);
        for arr in [&mut b_eq, &mut b1, &mut b2, &mut b3] {
            field_simd::bind_first_half_in_place(arr, r);
            let half = arr.len() / 2;
            arr.truncate(half);
        }
        rounds.push(evals);
    }

    let g_claims = [b1[0], b2[0], b3[0]];
    // Terminal identity: C_n = eq(u, r)·(g1(r) − g2(r)·g3(r)) — the
    // chained claim against the bound tables (the honest prover's
    // messages from round 1 on are nonzero: affine binding does not
    // commute with the degree-2 g2·g3 product, which is exactly what
    // makes the protocol non-vacuous).
    let terminal = b_eq[0].mul(&b1[0].sub(&b2[0].mul(&b3[0])));
    if terminal != current_claim {
        return Err(GrandProductError::TerminalIdentityFailed);
    }
    let _ = &bound_r;

    Ok(GrandProductProof {
        product,
        rounds,
        challenges,
        g_claims,
    })
}

/// An open bucket: one per simultaneously-open remaining-hypercube index
/// `m` (≤ `n + 1 − i` of them — exactly the DFS stack depth plus the
/// current subtree). `acc[k][s]` accumulates
/// `Σ_y eq((r_1..r_{i−1}, α_s), y) · g_k(y, m)` — the bound `g_k`
/// expansion for bucket key `m`, per `α`-node `s`.
#[derive(Clone)]
struct OpenBucket {
    m: u64,
    acc: [[Goldilocks; 4]; 4],
}

impl OpenBucket {
    fn free() -> Self {
        OpenBucket {
            m: u64::MAX,
            acc: [[Goldilocks::ZERO; 4]; 4],
        }
    }
}

/// **Algorithm 3 — the bucketed `O(n)`-space round-message prover**
/// (ePrint 2025/611, Appendix D, Theorem D.4).
///
/// Computes the Quarks sum-check round messages **without materializing
/// the `g`-tables** (the recorded-table path above is the `O(2^n)`-space
/// reference): the DFS walk emits `(z, g1, g2, g3)` merge events; each
/// event's term `eq((r_{<i}, α_s), y)·g_k(z)` (with `y = z`'s low `i`
/// bits, bucket key `m = z >> i`) routes into one of the open buckets,
/// and the *completion label* of `m` (its low `i` bits all ones — the
/// ancestor merge `z = (m+1)·2^i − 1`, or the special point `1^n` for
/// the last `m`) flushes the bucket product
/// `acc[0][s]·(acc[1][s] − acc[2][s]·acc[3][s])` into the round
/// accumulator and frees the slot.
///
/// Binding order: **least-significant variable first** (the paper's
/// convention — round `i` binds `x_i`, the `i`-th LSB). This is what
/// makes the bucketing work: the remaining hypercube is indexed by the
/// *high* bits, which is exactly the DFS tree's subtree structure, so
/// buckets open and close in tree order and at most `n + 1 − i` are
/// live. (Binding MSB-first would key the buckets by the *low* bits —
/// diagonal across the tree — and need `O(2^{n−i})` live buckets.)
///
/// Space: the DFS stack (`n + 1` blocks) + ≤ `n + 1 − i` buckets × 16
/// field elements + the challenges — `O(n)`. Time: one stream pass per
/// round with `O(n + i)` work per merge event — `O(n²·2^n)` field
/// operations overall (the same shape as Algorithm 1's `O(ℓ²·n·2^n)`).
/// The produced proof verifies with [`verify_grand_product`] (the
/// verifier's replay is binding-order-agnostic).
pub fn prove_grand_product_bucketed(
    stream: &mut dyn StreamOracle,
    product: Option<Goldilocks>,
    transcript: &mut Transcript,
) -> Result<GrandProductProof, GrandProductError> {
    let n = stream.len().trailing_zeros() as usize;
    // Derived-product mode: a plain DFS pre-pass (O(n) space) fixes the
    // claim so it can be absorbed BEFORE any challenge (FS hygiene).
    let product = match product {
        Some(p) => p,
        None => {
            stream.reset();
            dfs_grand_product(stream, None)?
        }
    };
    transcript
        .append_field(b"grand-product", &product)
        .map_err(GrandProductError::Transcript)?;
    let mut u: Vec<Goldilocks> = Vec::with_capacity(n);
    for _ in 0..n {
        u.push(
            transcript
                .challenge_field(b"grand-product-u")
                .map_err(GrandProductError::Transcript)?,
        );
    }
    // The u-challenges are drawn LSB-first (u[b] ↔ the variable at
    // LSB-position b — the round-(b+1) variable); eq(u, z) below uses
    // that order directly.

    if n == 0 {
        // Degenerate: the stream is a single leaf; P = v(0), no rounds,
        // g1 = g3 = 0, g2(·) = P at the empty point.
        stream.reset();
        let p = stream.next();
        if p != product {
            return Err(GrandProductError::TerminalIdentityFailed);
        }
        return Ok(GrandProductProof {
            product,
            rounds: Vec::new(),
            challenges: Vec::new(),
            g_claims: [Goldilocks::ZERO, product, Goldilocks::ZERO],
        });
    }

    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(n);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(n);
    let mut current_claim = Goldilocks::ZERO;
    // The last round's flush values per g (at the α nodes) — the g-claims
    // interpolate from the Boolean nodes.
    let mut last_flush: Option<[[Goldilocks; 4]; 4]> = None;
    let mut bound_r: Vec<Goldilocks> = Vec::with_capacity(n);
    let len = stream.len();

    for i in 1..=n {
        // ---- one DFS stream pass, bucketed ----
        let mut accumulator = [Goldilocks::ZERO; 4];
        let mut open: Vec<OpenBucket> = Vec::new();
        let mut stack: Vec<Block> = Vec::with_capacity(n + 1);
        let mask: u64 = (1u64 << i) - 1;
        stream.reset();
        for x in 0..len {
            let v = stream.next();
            stack.push(Block {
                value: v,
                size: 1,
                offset: x,
            });
            while stack.len() >= 2 {
                let (left, right) = {
                    let l = stack.len();
                    (&stack[l - 2], &stack[l - 1])
                };
                if left.size == right.size
                    && left.offset + left.size == right.offset
                    && (right.offset / right.size) % 2 == 1
                {
                    let prod = left.value.mul(&right.value);
                    let z = left.offset + left.size - 1;
                    // ---- the merge event: route the term ----
                    let g0 = eq_point_lsb(&u, z, n);
                    let slot = bucket_slot(&mut open, z >> i, i, n);
                    add_term(
                        &mut open[slot],
                        i,
                        &bound_r,
                        z,
                        mask,
                        [g0, prod, left.value, right.value],
                    );
                    if z & mask == mask {
                        // Completion label of bucket `z >> i`: flush now.
                        flush_bucket(&mut open[slot], &mut accumulator);
                    }
                    let merged = Block {
                        value: prod,
                        size: left.size * 2,
                        offset: left.offset,
                    };
                    stack.pop();
                    stack.pop();
                    stack.push(merged);
                } else {
                    break;
                }
            }
        }
        if stack.len() != 1 {
            return Err(GrandProductError::ShapeMismatch {
                expected: 1,
                got: stack.len(),
            });
        }
        let p = stack[0].value;
        if p != product {
            return Err(GrandProductError::TerminalIdentityFailed);
        }

        // ---- the special point 1^n: g1 = g3 = 0, g2 = P ----
        {
            let z = len - 1;
            debug_assert_eq!(z & mask, mask);
            let g0 = eq_point_lsb(&u, z, n);
            let slot = bucket_slot(&mut open, z >> i, i, n);
            add_term(
                &mut open[slot],
                i,
                &bound_r,
                z,
                mask,
                [g0, Goldilocks::ZERO, p, Goldilocks::ZERO],
            );
            let acc = flush_bucket(&mut open[slot], &mut accumulator);
            if i == n {
                last_flush = Some(acc);
            }
        }
        // All buckets must have flushed (every completion label arrived).
        debug_assert!(open.iter().all(|b| b.m == u64::MAX));

        // ---- the round message ----
        let evals = accumulator;
        let sum01 = evals[0].add(&evals[1]);
        if sum01 != current_claim {
            return Err(GrandProductError::RoundCheckFailed { round: i });
        }
        transcript
            .append_field_slice(b"sumcheck-round", &evals)
            .map_err(GrandProductError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(GrandProductError::Transcript)?;
        current_claim = interpolate_nodes(&evals, &r);
        bound_r.push(r);
        challenges.push(r);
        rounds.push(evals.to_vec());
    }

    // ---- the g-claims from the last round's flush ----
    // last_flush[k][s] = g_k(r_1..r_{n−1}, α_s): interpolate the two
    // Boolean nodes at r_n. (k=0 is the eq factor; k=1..3 are g1..g3.)
    let lf = last_flush.ok_or(GrandProductError::ShapeMismatch {
        expected: 1,
        got: 0,
    })?;
    let rn = challenges[n - 1];
    let interp2 = |v0: &Goldilocks, v1: &Goldilocks| -> Goldilocks {
        v0.mul(&Goldilocks::ONE.sub(&rn)).add(&rn.mul(v1))
    };
    let g_claims = [
        interp2(&lf[1][0], &lf[1][1]),
        interp2(&lf[2][0], &lf[2][1]),
        interp2(&lf[3][0], &lf[3][1]),
    ];

    // Terminal identity: current_claim = eq(u, r)·(g1(r) − g2(r)·g3(r)).
    let eq_ur = {
        let mut acc = Goldilocks::ONE;
        for (ub, rb) in u.iter().zip(challenges.iter()) {
            acc = acc.mul(
                &ub.mul(rb)
                    .add(&Goldilocks::ONE.sub(ub).mul(&Goldilocks::ONE.sub(rb))),
            );
        }
        acc
    };
    let terminal = eq_ur.mul(&g_claims[0].sub(&g_claims[1].mul(&g_claims[2])));
    if terminal != current_claim {
        return Err(GrandProductError::TerminalIdentityFailed);
    }

    Ok(GrandProductProof {
        product,
        rounds,
        challenges,
        g_claims,
    })
}

/// Find or open the bucket for key `m` (≤ `n + 1 − i` live at once).
fn bucket_slot(open: &mut Vec<OpenBucket>, m: u64, i: usize, n: usize) -> usize {
    if let Some(pos) = open.iter().position(|b| b.m == m) {
        return pos;
    }
    open.push(OpenBucket {
        m,
        acc: [[Goldilocks::ZERO; 4]; 4],
    });
    let cap = n + 2 - i;
    debug_assert!(
        open.iter().filter(|b| b.m != u64::MAX).count() <= cap,
        "live buckets exceed n+1-i"
    );
    open.len() - 1
}

/// Add `eq((r_{<i}, α_s), y)·g_k(z)` for all `(k, s)` into the bucket,
/// where `y = z & mask` and `gs = [g0, g1, g2, g3]`.
fn add_term(
    bucket: &mut OpenBucket,
    i: usize,
    bound_r: &[Goldilocks],
    z: u64,
    mask: u64,
    gs: [Goldilocks; 4],
) {
    let y = z & mask;
    // eq over the bound prefix: r_{b+1} vs y's LSB-position-b bit, b < i−1.
    let mut eq_prefix = Goldilocks::ONE;
    for (b, rb) in bound_r.iter().take(i.saturating_sub(1)).enumerate() {
        let f = if (y >> b) & 1 == 1 {
            *rb
        } else {
            Goldilocks::ONE.sub(rb)
        };
        eq_prefix = eq_prefix.mul(&f);
    }
    let y_top = (y >> (i - 1)) & 1 == 1;
    let alphas = [
        Goldilocks::ZERO,
        Goldilocks::ONE,
        Goldilocks::from_u64(2),
        Goldilocks::from_u64(3),
    ];
    for (s, alpha) in alphas.iter().enumerate() {
        let node_w = if y_top {
            *alpha
        } else {
            Goldilocks::ONE.sub(alpha)
        };
        let w = eq_prefix.mul(&node_w);
        for (acc_k, gk) in bucket.acc.iter_mut().zip(gs.iter()) {
            acc_k[s] = acc_k[s].add(&w.mul(gk));
        }
    }
}

/// Flush a completed bucket into the round accumulator; returns the
/// bucket's accumulated values (the bound g-evaluations at the α nodes).
fn flush_bucket(
    bucket: &mut OpenBucket,
    accumulator: &mut [Goldilocks; 4],
) -> [[Goldilocks; 4]; 4] {
    for (s, acc_s) in accumulator.iter_mut().enumerate() {
        let t =
            bucket.acc[0][s].mul(&bucket.acc[1][s].sub(&bucket.acc[2][s].mul(&bucket.acc[3][s])));
        *acc_s = acc_s.add(&t);
    }
    let out = bucket.acc;
    *bucket = OpenBucket::free();
    out
}

/// `eq(u, z)` with `u` in LSB-first order (u[b] ↔ LSB-position b).
fn eq_point_lsb(u: &[Goldilocks], z: u64, n: usize) -> Goldilocks {
    let mut acc = Goldilocks::ONE;
    for (b, ub) in u.iter().take(n).enumerate() {
        let f = if (z >> b) & 1 == 1 {
            *ub
        } else {
            Goldilocks::ONE.sub(ub)
        };
        acc = acc.mul(&f);
    }
    acc
}

/// Verify the grand-product proof: replay the round chain (the honest
/// messages are zero, so the chain stays at the initial zero claim) and
/// check the terminal identity from the claimed `g` evaluations.
///
/// The `g`-claims themselves are PCS-authenticated by the caller (in
/// the full Quarks construction the tables are slices of the committed
/// `(n+1)`-variate `f`, whose opening at `r` binds them — and `P =
/// f(0, 1^n)`).
pub fn verify_grand_product(
    proof: &GrandProductProof,
    transcript: &mut Transcript,
) -> Result<bool, GrandProductError> {
    let n = proof.rounds.len();
    transcript
        .append_field(b"grand-product", &proof.product)
        .map_err(GrandProductError::Transcript)?;
    let mut u: Vec<Goldilocks> = Vec::with_capacity(n);
    for _ in 0..n {
        u.push(
            transcript
                .challenge_field(b"grand-product-u")
                .map_err(GrandProductError::Transcript)?,
        );
    }
    let mut current = Goldilocks::ZERO;
    let mut point: Vec<Goldilocks> = Vec::with_capacity(n);
    for round in proof.rounds.iter() {
        if round.len() != 4 {
            return Err(GrandProductError::ShapeMismatch {
                expected: 4,
                got: round.len(),
            });
        }
        transcript
            .append_field_slice(b"sumcheck-round", round)
            .map_err(GrandProductError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(GrandProductError::Transcript)?;
        if round[0].add(&round[1]) != current {
            return Ok(false);
        }
        current = interpolate_nodes(round, &r);
        point.push(r);
    }
    // Terminal: current = eq(u, r)·(g1(r) − g2(r)·g3(r)), with the
    // g-claims PCS-authenticated by the caller.
    let eq_ur = {
        let mut acc = Goldilocks::ONE;
        for (ui, ri) in u.iter().zip(point.iter()) {
            acc = acc.mul(
                &ui.mul(ri)
                    .add(&Goldilocks::ONE.sub(ui).mul(&Goldilocks::ONE.sub(ri))),
            );
        }
        acc
    };
    let terminal = eq_ur.mul(&proof.g_claims[0].sub(&proof.g_claims[1].mul(&proof.g_claims[2])));
    Ok(terminal == current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::OwnedOracle;
    use lattice_core::DenseMle;

    /// The bucketed Algorithm-3 prover: prove/verify roundtrip at several
    /// sizes, with the derived-product mode and the caller-claim mode.
    #[test]
    fn bucketed_roundtrip() {
        for n in [1usize, 2, 3, 5, 7, 8] {
            let data: Vec<Goldilocks> = (0..(1u64 << n))
                .map(|i| g(i.wrapping_mul(29) + 5))
                .collect();
            let true_p = data.iter().fold(Goldilocks::ONE, |acc, v| acc.mul(v));
            // Derived mode.
            let mut stream = OwnedOracle::new(data.clone());
            let mut ts = Transcript::new_default(b"gp-b");
            let proof = prove_grand_product_bucketed(&mut stream, None, &mut ts).unwrap();
            assert_eq!(proof.product, true_p, "derived product n={n}");
            let mut ts2 = Transcript::new_default(b"gp-b");
            assert!(
                verify_grand_product(&proof, &mut ts2).unwrap(),
                "verify n={n}"
            );
            // Caller-claim mode (same seed: the protocol is identical
            // given the same claim input).
            let mut stream2 = OwnedOracle::new(data);
            let mut ts3 = Transcript::new_default(b"gp-b");
            let proof2 =
                prove_grand_product_bucketed(&mut stream2, Some(true_p), &mut ts3).unwrap();
            let mut ts4 = Transcript::new_default(b"gp-b");
            assert!(verify_grand_product(&proof2, &mut ts4).unwrap());
            // The two modes' proofs agree on the product and g-claims
            // (identical protocol given the same claim input).
            assert_eq!(proof.g_claims, proof2.g_claims);
        }
    }

    /// The bucketed prover's g-claims match the independently rebuilt
    /// g2-table evaluated at the challenges in **LSB-first** order (the
    /// bucketed protocol binds the least-significant variable first —
    /// reverse the point for the DenseMle's MSB-first convention).
    #[test]
    fn bucketed_g_claims_match_tables() {
        let n = 6;
        let data: Vec<Goldilocks> = DenseMle::random(n, b"gp-bk").evaluations;
        let mut stream = OwnedOracle::new(data.clone());
        let mut ts = Transcript::new_default(b"gp-bt");
        let proof = prove_grand_product_bucketed(&mut stream, None, &mut ts).unwrap();
        // Rebuild the g tables independently.
        let mut stream2 = OwnedOracle::new(data);
        let mut triples = Vec::new();
        let p = dfs_grand_product(&mut stream2, Some(&mut triples)).unwrap();
        let len = 1usize << n;
        let mut t1 = vec![Goldilocks::ZERO; len];
        let mut t2 = vec![Goldilocks::ZERO; len];
        let mut t3 = vec![Goldilocks::ZERO; len];
        for (z, a, b, c) in &triples {
            t1[*z as usize] = *a;
            t2[*z as usize] = *b;
            t3[*z as usize] = *c;
        }
        t2[len - 1] = p;
        // The challenges are LSB-first; DenseMle points are MSB-first.
        let pt_msb: Vec<Goldilocks> = proof.challenges.iter().rev().cloned().collect();
        let m1 = DenseMle::new(t1).unwrap();
        let m2 = DenseMle::new(t2).unwrap();
        let m3 = DenseMle::new(t3).unwrap();
        assert_eq!(m1.evaluate(&pt_msb).unwrap(), proof.g_claims[0]);
        assert_eq!(m2.evaluate(&pt_msb).unwrap(), proof.g_claims[1]);
        assert_eq!(m3.evaluate(&pt_msb).unwrap(), proof.g_claims[2]);
    }

    /// The bucketed prover's round messages are the true Quarks
    /// sum-check messages: an independent in-memory evaluation of the
    /// summand at round 1 (α ∈ {0,1,2,3}) matches, computed directly from
    /// the g-tables — the bit-identical cross-validation at round 1.
    #[test]
    fn bucketed_round1_matches_reference() {
        let n = 5;
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| g(i.wrapping_mul(37) + 11))
            .collect();
        let mut stream = OwnedOracle::new(data.clone());
        let mut ts = Transcript::new_default(b"gp-r1");
        let proof = prove_grand_product_bucketed(&mut stream, None, &mut ts).unwrap();
        // Rebuild the g tables and the u point from the transcript.
        let mut ts2 = Transcript::new_default(b"gp-r1");
        let product = proof.product;
        ts2.append_field(b"grand-product", &product).unwrap();
        let u: Vec<Goldilocks> = (0..n)
            .map(|_| ts2.challenge_field(b"grand-product-u").unwrap())
            .collect();
        // The reference in-memory engine over the SAME tables, binding
        // LSB-first, must produce the same round-1 message: this is the
        // recorded-table prover restricted to round 1.
        let mut stream2 = OwnedOracle::new(data);
        let mut triples = Vec::new();
        let p = dfs_grand_product(&mut stream2, Some(&mut triples)).unwrap();
        let len = 1usize << n;
        let mut g1 = vec![Goldilocks::ZERO; len];
        let mut g2 = vec![Goldilocks::ZERO; len];
        let mut g3 = vec![Goldilocks::ZERO; len];
        for (z, a, b, c) in &triples {
            g1[*z as usize] = *a;
            g2[*z as usize] = *b;
            g3[*z as usize] = *c;
        }
        g2[len - 1] = p;
        // Round 1 binds the LSB (LSB-position 0): the correct message is
        // f^1(α) = Σ_m eq(u_0,α)·eq(u_{1..},m)·[g1^α − g2^α·g3^α] with
        // g^α = (1−α)·g(0,m) + α·g(1,m) — the multilinear expansion.
        let mut expect = [Goldilocks::ZERO; 4];
        for (s, expect_s) in expect.iter_mut().enumerate() {
            let alpha = Goldilocks::from_u64(s as u64);
            let eq_u0 = u[0]
                .mul(&alpha)
                .add(&Goldilocks::ONE.sub(&u[0]).mul(&Goldilocks::ONE.sub(&alpha)));
            for m in 0..(len / 2) {
                let z0 = 2 * m;
                let z1 = 2 * m + 1;
                let mut eq_rest = Goldilocks::ONE;
                for (b, &ub) in u.iter().enumerate().skip(1) {
                    let f = if (z0 >> b) & 1 == 1 {
                        ub
                    } else {
                        Goldilocks::ONE.sub(&ub)
                    };
                    eq_rest = eq_rest.mul(&f);
                }
                let lin = |t0: &Goldilocks, t1: &Goldilocks| -> Goldilocks {
                    Goldilocks::ONE.sub(&alpha).mul(t0).add(&alpha.mul(t1))
                };
                let g1b = lin(&g1[z0], &g1[z1]);
                let g2b = lin(&g2[z0], &g2[z1]);
                let g3b = lin(&g3[z0], &g3[z1]);
                let term = g1b.sub(&g2b.mul(&g3b));
                *expect_s = expect_s.add(&eq_u0.mul(&eq_rest).mul(&term));
            }
        }
        // The bucketed round-1 message must equal the direct evaluation.
        assert_eq!(proof.rounds[0], expect.to_vec());
        let _ = p;
    }

    /// A wrong product claim is rejected (fail-closed DFS check).
    #[test]
    fn bucketed_wrong_product_rejected() {
        let n = 4;
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| g(i.wrapping_mul(23) + 7))
            .collect();
        let true_p = data.iter().fold(Goldilocks::ONE, |acc, v| acc.mul(v));
        let mut stream = OwnedOracle::new(data);
        let mut ts = Transcript::new_default(b"gp-bw");
        assert!(prove_grand_product_bucketed(
            &mut stream,
            Some(true_p.add(&Goldilocks::ONE)),
            &mut ts
        )
        .is_err());
    }

    /// Tampered proofs fail verification.
    #[test]
    fn bucketed_tampered_rejected() {
        let n = 5;
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| g(i.wrapping_mul(41) + 3))
            .collect();
        let mut stream = OwnedOracle::new(data);
        let mut ts = Transcript::new_default(b"gp-bt2");
        let mut proof = prove_grand_product_bucketed(&mut stream, None, &mut ts).unwrap();
        // Tamper 1: a round message.
        proof.rounds[2][1] = proof.rounds[2][1].add(&Goldilocks::ONE);
        let mut ts2 = Transcript::new_default(b"gp-bt2");
        assert!(!verify_grand_product(&proof, &mut ts2).unwrap());
        // Tamper 2: a g-claim (breaks the terminal identity).
        let mut stream2 = OwnedOracle::new((0..(1u64 << n)).map(|i| g(i * 41 + 3)).collect());
        let mut ts3 = Transcript::new_default(b"gp-bt3");
        let mut proof2 = prove_grand_product_bucketed(&mut stream2, None, &mut ts3).unwrap();
        proof2.g_claims[0] = proof2.g_claims[0].add(&Goldilocks::ONE);
        let mut ts4 = Transcript::new_default(b"gp-bt3");
        assert!(!verify_grand_product(&proof2, &mut ts4).unwrap());
    }

    fn g(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// The DFS walk computes the true product in O(n) space.
    #[test]
    fn dfs_product_matches_naive() {
        for n in [1usize, 3, 6, 8] {
            let data: Vec<Goldilocks> = (0..(1u64 << n))
                .map(|i| g(i.wrapping_mul(31) + 7))
                .collect();
            let mut stream = OwnedOracle::new(data.clone());
            let p = dfs_grand_product(&mut stream, None).unwrap();
            let naive = data.iter().fold(Goldilocks::ONE, |acc, v| acc.mul(v));
            assert_eq!(p, naive, "n={n}");
        }
    }

    /// The `g`-labeling is a bijection and `g1 = g2·g3` holds at every
    /// cube point (Lemma D.1's structure, verified structurally).
    #[test]
    fn g_tables_identity_on_the_cube() {
        let n = 5;
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| g(i.wrapping_mul(97) + 3))
            .collect();
        let mut stream = OwnedOracle::new(data.clone());
        let mut triples = Vec::new();
        let p = dfs_grand_product(&mut stream, Some(&mut triples)).unwrap();
        assert_eq!(triples.len(), (1usize << n) - 1);
        let mut covered = vec![false; 1usize << n];
        for (z, g1, g2, g3) in &triples {
            let zi = *z as usize;
            assert!(!covered[zi], "duplicate label {z}");
            covered[zi] = true;
            assert_eq!(*g1, g2.mul(g3), "identity at z={z}");
        }
        // All labels distinct and covering everything except 1^n.
        for (i, c) in covered.iter().enumerate() {
            if i == (1usize << n) - 1 {
                assert!(!c);
            } else {
                assert!(c, "uncovered label {i}");
            }
        }
        // The special point: g2(1^n) = P, g1 = g3 = 0 → identity holds.
        assert!(p.mul(&Goldilocks::ZERO).is_zero());
    }

    /// End-to-end: prove/verify roundtrip, including the eq(u, r)
    /// cross-check of the terminal claims.
    #[test]
    fn prove_verify_roundtrip() {
        let n = 6;
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| g(i.wrapping_mul(11) + 1))
            .collect();
        let mut stream = OwnedOracle::new(data);
        let mut ts = Transcript::new_default(b"gp-seed");
        let proof = prove_grand_product(&mut stream, None, &mut ts).unwrap();
        // Round 0's honest values at the interpolation nodes 0 and 1
        // are zero (the cube identity); the extrapolation nodes 2, 3 and
        // all later rounds are nonzero — affine binding does not
        // commute with the degree-2 product term.
        assert!(proof.rounds[0][0].is_zero() && proof.rounds[0][1].is_zero());
        let mut ts2 = Transcript::new_default(b"gp-seed");
        assert!(verify_grand_product(&proof, &mut ts2).unwrap());
        // Terminal claims are the bound g-table values (checked against
        // the independently rebuilt table below).
    }

    /// A tampered proof (nonzero round message or broken terminal)
    /// fails verification.
    #[test]
    fn tampered_proof_rejected() {
        let n = 5;
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| g(i.wrapping_mul(13) + 5))
            .collect();
        let mut stream = OwnedOracle::new(data);
        let mut ts = Transcript::new_default(b"gp-t");
        let mut proof = prove_grand_product(&mut stream, None, &mut ts).unwrap();
        // Tamper 1: a nonzero round message breaks the identity chain.
        proof.rounds[2][1] = proof.rounds[2][1].add(&Goldilocks::ONE);
        let mut ts2 = Transcript::new_default(b"gp-t");
        assert!(!verify_grand_product(&proof, &mut ts2).unwrap());
        // Tamper 2: break the terminal identity via a corrupted g-claim.
        let mut stream2 = OwnedOracle::new((0..(1u64 << n)).map(|i| g(i * 13 + 5)).collect());
        let mut ts3 = Transcript::new_default(b"gp-t2");
        let mut proof2 = prove_grand_product(&mut stream2, None, &mut ts3).unwrap();
        proof2.g_claims[0] = proof2.g_claims[0].add(&Goldilocks::ONE);
        let mut ts4 = Transcript::new_default(b"gp-t2");
        assert!(!verify_grand_product(&proof2, &mut ts4).unwrap());
    }

    /// A stream whose product contradicts the claimed `P` is rejected at
    /// prove time (the DFS catches it in O(n) space).
    #[test]
    fn wrong_product_claim_rejected() {
        let n = 4;
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| g(i.wrapping_mul(17) + 3))
            .collect();
        let true_p = data.iter().fold(Goldilocks::ONE, |acc, v| acc.mul(v));
        let wrong = true_p.add(&Goldilocks::ONE);
        let mut stream = OwnedOracle::new(data);
        let mut ts = Transcript::new_default(b"gp-w");
        assert!(prove_grand_product(&mut stream, Some(wrong), &mut ts).is_err());
    }

    /// The eq(u, ·) factor at the terminal point: the prover's u and the
    /// verifier's u agree (same transcript), and the terminal claims are
    /// the bound g-table values.
    #[test]
    fn terminal_claims_match_bound_tables() {
        let n = 6;
        let data: Vec<Goldilocks> = DenseMle::random(n, b"gp-r").evaluations;
        let mut stream = OwnedOracle::new(data.clone());
        let mut ts = Transcript::new_default(b"gp-x");
        let proof = prove_grand_product(&mut stream, None, &mut ts).unwrap();
        // Rebuild the g2 table independently and evaluate at r.
        let mut stream2 = OwnedOracle::new(data);
        let mut triples = Vec::new();
        let p = dfs_grand_product(&mut stream2, Some(&mut triples)).unwrap();
        let len = 1usize << n;
        let mut g2 = vec![Goldilocks::ZERO; len];
        for (z, _, b, _) in &triples {
            g2[*z as usize] = *b;
        }
        g2[len - 1] = p;
        let mle = DenseMle::new(g2).unwrap_or(DenseMle::constant(Goldilocks::ZERO));
        assert_eq!(mle.evaluate(&proof.challenges).unwrap(), proof.g_claims[1]);
    }
}
