//! Ring-valued product-claim sumcheck engine over R_q — the shared
//! substrate for the SALSAA A2–A5 protocol stack (Wave 7 item 7.11) and the
//! RoKoko Π^lin linearisation (Wave 7 item 7.13).
//!
//! Design (ported from the lattice-zk-lab reference, adapted to this
//! workspace's conventions):
//! * A **product claim** is `Σ_{z∈{0,1}^μ} Π_i MLE[T_i](z) = value` with
//!   `T_i : {0,1}^μ → R_q` flat tables of ring elements.
//! * Round messages are **ring-valued**: the prover sends
//!   `[g(0), g(1), …, g(deg)]` (evaluations of the round polynomial at the
//!   integer points 0..=deg, where `g(X) = Σ_{z'∈{0,1}^{μ-1}} Π_i
//!   MLE[T_i](…, X, z')`). The verifier checks `g(0)+g(1) = claim`,
//!   Lagrange-interpolates at a fresh `Z_q` challenge and binds every table
//!   by the affine combination `(1−c)·T_lo + c·T_hi` (scalar
//!   multiplication — no ring product in the binding path).
//! * Round challenges are sampled in `Z_q` (unbiased rejection-sampled
//!   u64, the pikkufold_lrp discipline). The paper's CRT-slot
//!   ring-valued challenges are a size optimisation; scalar challenges
//!   preserve the protocol logic at kernel scale (documented deviation,
//!   same choice as `lattice-folding/src/pikkufold_lrp.rs`).
//! * The engine returns the **final combined claim**; callers check the
//!   terminal identity `Σ_g comb_g · Π_i MLE[T_i](r)` with their private
//!   tables replaced by the prover's sent openings (the Π^lin / Π_air
//!   pattern).
//!
//! Helpers exported for the protocol modules: the σ⁻¹ conjugation
//! automorphism on R_q (`conj(a) = a(X^{-1})`, coefficient map
//! `(a_0, −a_{n−1}, …, −a_1)`), the conjugate inner product
//! `⟨w, w̄⟩ = Σ_j conj(w_j)·w_j` (its constant term is exactly ‖cf(w)‖² —
//! Remark 3's power-of-two shortcut), and the balanced trace
//! `Tr(t) = n·c_0` in the centred representative.

use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_ring::{RingConfig, RingElement};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RingScError {
    Transcript(TranscriptError),
    Ring(lattice_ring::RingError),
    /// Table lengths are not a power of two / disagree across factors.
    Shape { expected: usize, got: usize },
    /// A round message failed the sumcheck recurrence.
    RoundCheckFailed { round: usize },
    /// The terminal identity failed (caller-side check helper).
    TerminalFailed,
}

impl From<TranscriptError> for RingScError {
    fn from(e: TranscriptError) -> Self {
        RingScError::Transcript(e)
    }
}
impl From<lattice_ring::RingError> for RingScError {
    fn from(e: lattice_ring::RingError) -> Self {
        RingScError::Ring(e)
    }
}

// ---------------------------------------------------------------------------
// Ring helpers
// ---------------------------------------------------------------------------

/// The σ⁻¹ conjugation automorphism on R_q = Z_q[X]/(X^n+1):
/// `conj(a)(X) = a(X^{-1})`, coefficient map `(a_0, −a_{n−1}, …, −a_1)`.
/// Satisfies `conj(a·b) = conj(a)·conj(b)` and `Tr(conj(a)·b) = n·⟨a,b⟩`.
pub fn conj(a: &RingElement) -> RingElement {
    let ring = a.config();
    let n = ring.n();
    let q = ring.modulus.q as i128;
    let mut coeffs = vec![0u32; n];
    coeffs[0] = a.coeff(0);
    for i in 1..n {
        let v = -(i128::from(a.coeff(n - i)));
        coeffs[i] = v.rem_euclid(q) as u32;
    }
    RingElement::from_coeffs(ring, coeffs)
}

/// Ring inner product `Σ_i a_i·b_i` (schoolbook; kernel scale).
pub fn ring_dot(a: &[RingElement], b: &[RingElement]) -> Result<RingElement, RingScError> {
    if a.len() != b.len() {
        return Err(RingScError::Shape {
            expected: a.len(),
            got: b.len(),
        });
    }
    let ring = a
        .first()
        .map(|e| e.config().clone())
        .ok_or(RingScError::Shape {
            expected: 1,
            got: 0,
        })?;
    let mut acc = ring.zero();
    for (x, y) in a.iter().zip(b.iter()) {
        acc = acc.add(&x.mul(y)?)?;
    }
    Ok(acc)
}

/// The conjugate inner product `t = ⟨w, w̄⟩ = Σ_j conj(w_j)·w_j`.
/// Constant term = ‖cf(w)‖² exactly (wraparound-free regime: gate with
/// `trace_balanced`).
pub fn norm_conjugate_inner(w: &[RingElement]) -> Result<RingElement, RingScError> {
    let wbar: Vec<RingElement> = w.iter().map(conj).collect();
    ring_dot(&wbar, w)
}

/// The balanced trace `Tr(t) = n·t_0` with `t_0` in the centred
/// representative — the SALSAA Lemma-4.11-style integer readout gate.
pub fn trace_balanced(t: &RingElement) -> i64 {
    let q = t.config().modulus.q as i64;
    let c0 = t.coeff(0) as i64;
    let balanced = if c0 > q / 2 { c0 - q } else { c0 };
    balanced * t.config().n() as i64
}

/// Evaluate the multilinear extension of a ring-element table at a point of
/// `Z_q^μ` (iterative affine binding — O(2^μ·n) ring adds).
pub fn mle_eval_ring(
    table: &[RingElement],
    point: &[u32],
) -> Result<RingElement, RingScError> {
    // A length-1 table is a constant: its MLE at ANY point is the value
    // itself (the Π^lin/Π_air terminal's private-slot substitution).
    if table.len() == 1 {
        return Ok(table[0].clone());
    }
    if point.is_empty() {
        return table
            .first()
            .cloned()
            .ok_or(RingScError::Shape { expected: 1, got: 0 });
    }
    let ring = table
        .first()
        .map(|e| e.config().clone())
        .ok_or(RingScError::Shape {
            expected: 1,
            got: 0,
        })?;
    let mut cur: Vec<RingElement> = table.to_vec();
    for &c in point {
        if cur.len() < 2 || cur.len() % 2 != 0 {
            return Err(RingScError::Shape {
                expected: 2,
                got: cur.len(),
            });
        }
        let q = ring.modulus.q as i128;
        let half = cur.len() / 2;
        // bind the FIRST variable (MSB-first layout, the lab eq-table
        // endianness): lo = bit 0 block, hi = bit 1 block.
        let mut next = Vec::with_capacity(half);
        for j in 0..half {
            let lo = &cur[j];
            let hi = &cur[half + j];
            // (1-c)*lo + c*hi  — as centred scalar arithmetic mod q
            let mut coeffs = vec![0u32; ring.n()];
            for i in 0..ring.n() {
                let l = i128::from(lo.coeff(i));
                let h = i128::from(hi.coeff(i));
                let v = ((1 - i128::from(c)) * l + i128::from(c) * h).rem_euclid(q);
                coeffs[i] = v as u32;
            }
            next.push(RingElement::from_coeffs(&ring, coeffs));
        }
        cur = next;
    }
    cur.into_iter()
        .next()
        .ok_or(RingScError::Shape { expected: 1, got: 0 })
}

/// The eq-tensor table `eq(bin(z), r)` over the μ-dimensional cube as ring
/// CONSTANT elements (public tables for Π_mle row appends / boundary
/// selectors). `r` is a `Z_q^μ` point.
pub fn eq_table_ring(ring: &RingConfig, r: &[u32]) -> Vec<RingElement> {
    let q = ring.modulus.q as i128;
    let total = 1usize << r.len();
    let mut out = Vec::with_capacity(total);
    for z in 0..total {
        let mut v: i128 = 1;
        for (b, &c) in r.iter().enumerate() {
            let bit = (z >> (r.len() - 1 - b)) & 1;
            let factor = if bit == 1 { i128::from(c) } else { 1 - i128::from(c) };
            v = (v * factor).rem_euclid(q);
        }
        out.push(ring.constant(v.rem_euclid(q) as u32));
    }
    out
}

/// Uniform `Z_q` challenge from the transcript (rejection-sampled u64 —
/// unbiased; the pikkufold_lrp discipline).
pub fn challenge_zq(
    transcript: &mut Transcript,
    label: &[u8],
    q: u32,
) -> Result<u32, RingScError> {
    let limit = u64::from(q);
    let bound = u64::MAX - (u64::MAX % limit) - 1;
    for _ in 0..16 {
        let bytes = transcript.challenge_bytes(label, 8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes[..8]);
        let v = u64::from_le_bytes(arr);
        if v <= bound {
            return Ok((v % limit) as u32);
        }
    }
    Err(RingScError::Transcript(
        TranscriptError::RejectionBudgetExceeded,
    ))
}

/// Uniform ring element from the transcript (4 bytes per coefficient).
pub fn challenge_ring_elt(
    transcript: &mut Transcript,
    label: &[u8],
    ring: &RingConfig,
) -> Result<RingElement, RingScError> {
    let phi = ring.n();
    let bytes = transcript.challenge_bytes(label, 4 * phi)?;
    let mut coeffs = Vec::with_capacity(phi);
    for c in bytes.chunks(4) {
        let mut arr = [0u8; 4];
        let take = 4.min(c.len());
        arr[..take].copy_from_slice(&c[..take]);
        coeffs.push(u32::from_le_bytes(arr) % ring.modulus.q);
    }
    Ok(RingElement::from_coeffs(ring, coeffs))
}

// ---------------------------------------------------------------------------
// Product claims + the engine
// ---------------------------------------------------------------------------

/// One sumcheck claim: `Σ_{z} Π_i MLE[T_i](z) = value` over R_q.
#[derive(Clone, Debug)]
pub struct ProductClaim {
    pub tables: Vec<Vec<RingElement>>,
    pub value: RingElement,
}

/// The proof: per-round evaluation vectors `[g(0), …, g(deg)]`, the final
/// challenge point, and the per-factor bound values at that point.
#[derive(Clone, Debug)]
pub struct RingScProof {
    pub rounds: Vec<Vec<RingElement>>,
    pub point: Vec<u32>,
    /// MLE[T_i](point) for every factor of every claim, claim-major,
    /// factor-minor (the caller's terminal openings substitute these).
    pub final_values: Vec<Vec<RingElement>>,
}

/// Bind a table's first variable to the scalar `c`: `T ← (1−c)·lo + c·hi`.
fn bind_first(ring: &RingConfig, table: &[RingElement], c: u32) -> Vec<RingElement> {
    let q = ring.modulus.q as i128;
    let half = table.len() / 2;
    let mut next = Vec::with_capacity(half);
    for j in 0..half {
        let lo = &table[j];
        let hi = &table[half + j];
        let mut coeffs = vec![0u32; ring.n()];
        for i in 0..ring.n() {
            let l = i128::from(lo.coeff(i));
            let h = i128::from(hi.coeff(i));
            let v = ((1 - i128::from(c)) * l + i128::from(c) * h).rem_euclid(q);
            coeffs[i] = v as u32;
        }
        next.push(RingElement::from_coeffs(ring, coeffs));
    }
    next
}

/// Evaluate the round polynomial at integer point `t` (0..=deg) for the
/// current bound tables: `g(t) = Σ_{z'} Π_i MLE[T_i](…, t, z')` computed as
/// the product-sum over the affine-combined sub-tables.
fn round_eval(
    ring: &RingConfig,
    tables: &[Vec<RingElement>],
    combiner: &RingElement,
    t: i64,
) -> Result<RingElement, RingScError> {
    let q = ring.modulus.q as i128;
    // affine-combine each factor: (1-t)*lo + t*hi
    let mut combined: Vec<Vec<RingElement>> = Vec::with_capacity(tables.len());
    for table in tables {
        let half = table.len() / 2;
        let mut row = Vec::with_capacity(half);
        for j in 0..half {
            let lo = &table[j];
            let hi = &table[half + j];
            let mut coeffs = vec![0u32; ring.n()];
            for i in 0..ring.n() {
                let l = i128::from(lo.coeff(i));
                let h = i128::from(hi.coeff(i));
                let v = ((1 - i128::from(t)) * l + i128::from(t) * h).rem_euclid(q);
                coeffs[i] = v as u32;
            }
            row.push(RingElement::from_coeffs(ring, coeffs));
        }
        combined.push(row);
    }
    // product-sum over the sub-cube, weighted by the claim combiner
    let mut acc = ring.zero();
    if combined.is_empty() {
        return Ok(acc);
    }
    let len = combined[0].len();
    for z in 0..len {
        let mut prod = ring.one();
        for f in &combined {
            if f.len() != len {
                return Err(RingScError::Shape {
                    expected: len,
                    got: f.len(),
                });
            }
            prod = prod.mul(&f[z])?;
        }
        acc = acc.add(&prod.mul(combiner)?)?;
    }
    Ok(acc)
}

/// Lagrange interpolation of a degree-`d` polynomial at `c`, given its
/// evaluations at the integer points 0..=d (Newton/vandermonde-free: the
/// points are fixed and small — precomputed Lagrange basis in i128).
fn lagrange_at(ring: &RingConfig, evals: &[RingElement], c: u32) -> Result<RingElement, RingScError> {
    let d = evals.len() - 1;
    let q = ring.modulus.q as i128;
    let mut acc = ring.zero();
    for j in 0..=d {
        // basis_j(c) = Π_{m≠j} (c - m)/(j - m)
        let mut num: i128 = 1;
        let mut den: i128 = 1;
        for m in 0..=d {
            if m == j {
                continue;
            }
            num = (num * ((c as i128 - m as i128) % q)) % q;
            den = (den * ((j as i128 - m as i128) % q)) % q;
        }
        // modular inverse of den (q prime in this workspace's rings)
        let den_inv = mod_inv(den.rem_euclid(q), q);
        if den_inv == 0 {
            return Err(RingScError::TerminalFailed);
        }
        let w = (num * den_inv).rem_euclid(q);
        // acc += w * evals[j]
        let scaled = evals[j].scale_i64(w as i64);
        acc = acc.add(&scaled)?;
    }
    Ok(acc)
}

fn mod_inv(a: i128, q: i128) -> i128 {
    // extended Euclid; q is prime in this workspace (Q_32), so a != 0 inverts
    if a == 0 {
        return 0;
    }
    let (mut old_r, mut r) = (a, q);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let quot = old_r / r;
        (old_r, r) = (r, old_r - quot * r);
        (old_s, s) = (s, old_s - quot * s);
    }
    old_s.rem_euclid(q)
}

/// Prove the batched claim
/// `Σ_g comb_g · ( Σ_z Π_i MLE[T_{g,i}](z) ) = Σ_g comb_g · value_g`.
///
/// The transcript must already contain the statement. Round messages are
/// absorbed before each challenge. Returns the proof with the final point
/// and per-factor bound values.
pub fn ring_sc_prove(
    ring: &RingConfig,
    claims: &[ProductClaim],
    combiners: &[RingElement],
    transcript: &mut Transcript,
) -> Result<RingScProof, RingScError> {
    if claims.len() != combiners.len() || claims.is_empty() {
        return Err(RingScError::Shape {
            expected: combiners.len(),
            got: claims.len(),
        });
    }
    let num_vars = claims
        .first()
        .and_then(|c| c.tables.first())
        .map(|t| t.len().trailing_zeros() as usize)
        .ok_or(RingScError::Shape {
            expected: 1,
            got: 0,
        })?;
    // all factor tables must share the cube size
    for claim in claims {
        for table in &claim.tables {
            if table.len() != 1usize << num_vars {
                return Err(RingScError::Shape {
                    expected: 1 << num_vars,
                    got: table.len(),
                });
            }
        }
    }
    let degree: usize = claims
        .iter()
        .map(|c| c.tables.len())
        .max()
        .unwrap_or(0);
    let q = ring.modulus.q;

    let mut bound: Vec<Vec<Vec<RingElement>>> =
        claims.iter().map(|c| c.tables.clone()).collect();
    let mut rounds: Vec<Vec<RingElement>> = Vec::with_capacity(num_vars);
    let mut point: Vec<u32> = Vec::with_capacity(num_vars);

    // flatten for round_eval: all factors of all claims in one slice with
    // per-claim combiners — round_eval consumes (tables, combiners) where
    // tables = the factor list of ONE claim at a time.
    for _round in 0..num_vars {
        // evaluations at t = 0..=degree
        let mut evals: Vec<RingElement> = Vec::with_capacity(degree + 1);
        for t in 0..=degree as i64 {
            let mut acc = ring.zero();
            for (ci, claim_tables) in bound.iter().enumerate() {
                let g = round_eval(ring, claim_tables, &combiners[ci], t)?;
                acc = acc.add(&g)?;
            }
            evals.push(acc);
        }
        // absorb the round message
        for e in &evals {
            transcript.append_bytes(b"ring-sc-round", &e.to_bytes())?;
        }
        let c = challenge_zq(transcript, b"ring-sc-chal", q)?;
        // bind every factor's first variable
        bound = bound
            .iter()
            .map(|claim_tables| {
                claim_tables
                    .iter()
                    .map(|table| bind_first(ring, table, c))
                    .collect()
            })
            .collect();
        rounds.push(evals);
        point.push(c);
    }

    // final per-factor values at the point (length-1 tables)
    let mut final_values: Vec<Vec<RingElement>> = Vec::with_capacity(bound.len());
    for claim_tables in &bound {
        let mut vals = Vec::with_capacity(claim_tables.len());
        for table in claim_tables {
            vals.push(
                table
                    .first()
                    .cloned()
                    .ok_or(RingScError::Shape { expected: 1, got: 0 })?,
            );
        }
        final_values.push(vals);
    }
    Ok(RingScProof {
        rounds,
        point,
        final_values,
    })
}

/// Verify the round recurrence and derive the final combined claim.
///
/// Checks, per round: (a) shape `[deg+1]` ring elements; (b)
/// `g(0) + g(1) == current_claim`; then interpolates at the transcript
/// challenge. Returns `(Ok(final_claim))` — the caller checks the terminal
/// identity `final_claim == Σ_g comb_g · Π openings_g` itself.
pub fn ring_sc_verify(
    ring: &RingConfig,
    degree: usize,
    num_vars: usize,
    target: &RingElement,
    proof: &RingScProof,
    transcript: &mut Transcript,
) -> Result<RingElement, RingScError> {
    if proof.rounds.len() != num_vars {
        return Err(RingScError::Shape {
            expected: num_vars,
            got: proof.rounds.len(),
        });
    }
    let q = ring.modulus.q;
    let mut current = target.clone();
    for (r, evals) in proof.rounds.iter().enumerate() {
        if evals.len() != degree + 1 {
            return Err(RingScError::Shape {
                expected: degree + 1,
                got: evals.len(),
            });
        }
        let g0 = evals[0].add(&evals[1])?;
        if g0 != current {
            return Err(RingScError::RoundCheckFailed { round: r });
        }
        for e in evals {
            transcript.append_bytes(b"ring-sc-round", &e.to_bytes())?;
        }
        let c = challenge_zq(transcript, b"ring-sc-chal", q)?;
        current = lagrange_at(ring, evals, c)?;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingConfig {
        lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, 4)
            .ok()
            .unwrap()
    }

    fn rand_vec(ring: &RingConfig, m: usize, tag: &[u8], span: u32) -> Vec<RingElement> {
        (0..m)
            .map(|i| {
                let bytes = Transcript::xof(b"ring-sc-test", &[tag, &(i as u32).to_le_bytes()].concat(), 4 * ring.n());
                let coeffs: Vec<u32> = bytes
                    .chunks(4)
                    .take(ring.n())
                    .map(|c| {
                        let mut a = [0u8; 4];
                        a.copy_from_slice(&c[..4]);
                        u32::from_le_bytes(a) % (2 * span + 1)
                    })
                    .collect();
                RingElement::from_coeffs(ring, coeffs)
            })
            .collect()
    }

    #[test]
    fn conj_is_an_automorphism() {
        let ring = ring();
        let a = rand_vec(&ring, 1, b"a", 100)[0].clone();
        let b = rand_vec(&ring, 1, b"b", 100)[0].clone();
        // conj(a*b) == conj(a)*conj(b)
        let lhs = conj(&a.mul(&b).ok().unwrap());
        let rhs = conj(&a).mul(&conj(&b)).ok().unwrap();
        assert_eq!(lhs, rhs);
        // Tr(conj(a)*b) == n * <coeff(a), coeff(b)>  (balanced inner product)
        let q = ring.modulus.q as i64;
        let n = ring.n() as i64;
        let mut ip: i64 = 0;
        for i in 0..ring.n() {
            let x = a.coeff(i) as i64;
            let y = b.coeff(i) as i64;
            ip += balanced(x, q) * balanced(y, q);
        }
        let tr = trace_balanced(&conj(&a).mul(&b).ok().unwrap());
        // exact integer identity when no wraparound: tr == n*ip
        assert_eq!(tr, n * ip);
    }

    fn balanced(x: i64, q: i64) -> i64 {
        let v = x;
        if v > q / 2 {
            v - q
        } else {
            v
        }
    }

    #[test]
    fn norm_constant_term_is_integer_norm() {
        let ring = ring();
        let w = rand_vec(&ring, 5, b"w", 8);
        let t = norm_conjugate_inner(&w).ok().unwrap();
        let q = ring.modulus.q as i64;
        let mut norm_sq: i64 = 0;
        for e in &w {
            for i in 0..ring.n() {
                let v = balanced(i64::from(e.coeff(i)), q);
                norm_sq += v * v;
            }
        }
        // wraparound-free at this size: ct(t) == norm_sq mod q and the
        // balanced trace == n * norm_sq
        assert_eq!(trace_balanced(&t), (ring.n() as i64) * norm_sq);
    }

    #[test]
    fn mle_eval_matches_direct() {
        let ring = ring();
        let mu = 3;
        let table = rand_vec(&ring, 1 << mu, b"tbl", 50);
        let point: Vec<u32> = vec![3, 7, 11];
        let got = mle_eval_ring(&table, &point).ok().unwrap();
        // direct MLE evaluation
        let q = ring.modulus.q as i64;
        let mut acc = vec![0i64; ring.n()];
        for z in 0..(1usize << mu) {
            let mut w: i64 = 1;
            for (b, &c) in point.iter().enumerate() {
                let bit = (z >> (mu - 1 - b)) & 1;
                w = (w * if bit == 1 { c as i64 } else { 1 - c as i64 }) % q;
            }
            for i in 0..ring.n() {
                acc[i] = (acc[i] + w * (table[z].coeff(i) as i64)) % q;
            }
        }
        for i in 0..ring.n() {
            assert_eq!(got.coeff(i), acc[i].rem_euclid(q) as u32);
        }
    }

    #[test]
    fn sumcheck_honest_and_tampered() {
        let ring = ring();
        let mu = 4;
        let a = rand_vec(&ring, 1 << mu, b"sa", 30);
        let b = rand_vec(&ring, 1 << mu, b"sb", 30);
        let value = ring_dot(&a, &b).ok().unwrap();
        let claim = ProductClaim {
            tables: vec![a, b],
            value: value.clone(),
        };
        let mut t = Transcript::new_default(b"lzx-ring-sc");
        let proof =
            ring_sc_prove(&ring, &[claim], &[ring.one()], &mut t).ok().unwrap();
        // verify: replay and derive the final claim
        let mut vt = Transcript::new_default(b"lzx-ring-sc");
        let last = ring_sc_verify(&ring, 2, mu, &value, &proof, &mut vt)
            .ok()
            .unwrap();
        // terminal: prod of final values == last
        let fv = &proof.final_values[0];
        let prod = fv[0].mul(&fv[1]).ok().unwrap();
        assert_eq!(prod, last);
        // tampered round message rejected
        let mut bad = proof.clone();
        if bad.rounds[0][0] == ring.zero() {
            bad.rounds[0][0] = ring.one();
        } else {
            bad.rounds[0][0] = ring.zero();
        }
        let mut vt2 = Transcript::new_default(b"lzx-ring-sc");
        assert!(ring_sc_verify(&ring, 2, mu, &value, &bad, &mut vt2).is_err());
    }

    #[test]
    fn sumcheck_multiple_claims_with_combiners() {
        let ring = ring();
        let mu = 3;
        let a = rand_vec(&ring, 1 << mu, b"ma", 20);
        let b = rand_vec(&ring, 1 << mu, b"mb", 20);
        let b2 = b.clone();
        let c = rand_vec(&ring, 1 << mu, b"mc", 20);
        let v1 = ring_dot(&a, &b).ok().unwrap();
        let v2 = ring_dot(&b, &c).ok().unwrap();
        let alpha = ring.constant(5);
        // combined target: v1 + alpha*v2
        let target = v1.add(&v2.scale_i64(5)).ok().unwrap();
        let claims = vec![
            ProductClaim {
                tables: vec![a, b],
                value: v1,
            },
            ProductClaim {
                tables: vec![b2, c],
                value: v2,
            },
        ];
        let mut t = Transcript::new_default(b"lzx-ring-sc-multi");
        let proof =
            ring_sc_prove(&ring, &claims, &[ring.one(), alpha], &mut t)
                .ok()
                .unwrap();
        let mut vt = Transcript::new_default(b"lzx-ring-sc-multi");
        let last = ring_sc_verify(&ring, 2, mu, &target, &proof, &mut vt)
            .ok()
            .unwrap();
        // terminal: fv0[0]*fv0[1] + alpha*fv1[0]*fv1[1] == last
        let f = &proof.final_values;
        let t0 = f[0][0].mul(&f[0][1]).ok().unwrap();
        let t1 = f[1][0].mul(&f[1][1]).ok().unwrap();
        assert_eq!(t0.add(&t1.scale_i64(5)).ok().unwrap(), last);
    }

    #[test]
    fn eq_table_selects_row() {
        let ring = ring();
        let mu = 3;
        let point = vec![1u32, 0, 1];
        let table = eq_table_ring(&ring, &point);
        // MLE[eq](point) == 1; at the integer index of point it is 1
        let idx = 0b101usize;
        assert_eq!(table[idx], ring.one());
        // sum over the cube == eq(., 0 selected) etc: total sum = Π (1) = 1?
        // eq sums to Π_j [(1-r_j) + r_j] = 1
        let q = ring.modulus.q as i64;
        let mut s = ring.zero();
        for e in &table {
            s = s.add(e).ok().unwrap();
        }
        assert_eq!(s, ring.one());
        let _ = (q, mu);
    }
}
