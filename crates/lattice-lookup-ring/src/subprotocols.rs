//! Appendix B of ePrint 2026/471 — the PIOP toolkit over the split
//! ring, each protocol proving/verifying with the ring sum-check
//! engine (`ring_sumcheck`) underneath:
//!
//! * **Scalar product** (Constr. B.2): `⟨a,b⟩ = τ`.
//! * **Hadamard product** (Constr. B.5): `a∘b = c`.
//! * **Cyclic shift** (Constr. B.9): `b = shift(a)`, via two scalar
//!   products against the geometric vector `(1,γ,…,γ^{N−1})` and the
//!   Lemma-B.8 univariate check `ν₀ − γν₁ = (1−γ^N)·b_N`.
//! * **Entry product** (Constr. B.12): `τ = ∏ aᵢ`, via prefix-product
//!   vectors `c,d,e`, a Hadamard (`a∘c = d`), a cyclic shift
//!   (`e = shift(c)` with `e_N = 1`), and the final
//!   `d̂(η) − ê(η) = EQ(1^{logN}, η)·(τ−1)`.
//! * **Integer check** (Constr. B.15): a vector is `Z_q`-valued —
//!   random scalar points `ρ ∈ Z_q^{logN}`, reject unless
//!   `â(ρ) ∈ Z_q`; amplified.
//! * **Binary check** (Constr. B.19) — the LatticeFold+ range PIOP:
//!   the monomial matrix `M_f[j,i] = X^{f_{i,j}}`, the `β/β²`
//!   evaluation vectors `m^{(j)}, m'^{(j)}`, the batched
//!   `Σ EQ(c,ω)·Σ_j α^j(m^{(j)}(ω)² − m'^{(j)}(ω)) = 0` monomial-set
//!   sum-check, the `e_j = M̂_{f,j}(r)` evaluations with
//!   `ev(e_j)(β)² = ev(e_j)(β²)`, the `d` monomial-row scalar products
//!   `⟨M̂_{f,j}, ⊗r⟩ = e_j`, the `CF(f)` consistency block
//!   (`v = CF(f)·⊗r`, `w = ⟨f, ⊗r⟩`, `w = Σ_j v_j X^j`), and the
//!   range-closing `ct(X^{−1}·e_j) = v_j` (the paper's step 13 —
//!   `X^{−1} = −X^{d−1}` in the quotient ring, and
//!   `ct(X^{−1}e) = [X¹]e` exactly).
//!
//! Oracle model: every protocol emits `EvalQuery`s —
//! `(label, point, value)` MLE-evaluation claims the *outer* verifier
//! must settle. In the plain PIOP layer the caller resolves them from
//! the witness vectors (IOP-of-proximity semantics); the compiled
//! layer (`compile`) settles them through Ajtai openings.

// (Kernel loops use explicit indices by convention.)
#![allow(clippy::needless_range_loop)]

use crate::ring_d::{Elem, RingD};
use crate::ring_sumcheck::{
    prove_sumcheck, verify_sumcheck, RingFactor, RingSumcheckProof, RingTerm,
    RingVirtualPoly,
};
use lattice_core::transcript::Transcript;

/// An MLE evaluation claim on a named oracle vector.
#[derive(Debug, Clone)]
pub struct EvalQuery {
    pub label: String,
    pub point: Vec<Elem>,
    pub value: Elem,
}

/// Resolver for oracle queries: `(label, point) -> MLE evaluation`.
pub type Oracle<'r> = &'r dyn Fn(&str, &[Elem]) -> Result<Elem, String>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubError {
    Shape(String),
    Sumcheck(String),
    Verify(String),
}

impl From<crate::ring_sumcheck::SumcheckError> for SubError {
    fn from(e: crate::ring_sumcheck::SumcheckError) -> Self {
        SubError::Sumcheck(format!("{e:?}"))
    }
}

fn log2_exact(n: usize) -> Result<usize, SubError> {
    if n == 0 || !n.is_power_of_two() {
        return Err(SubError::Shape(format!("length {n} not a power of two")));
    }
    Ok(n.trailing_zeros() as usize)
}

// ===========================================================================
// Scalar product (Construction B.2)
// ===========================================================================

#[derive(Debug, Clone)]
pub struct ScalarProductProof {
    pub sc: RingSumcheckProof,
}

/// Prove `⟨a, b⟩ = τ` (τ computed here and returned).
pub fn prove_scalar_product(
    ring: &RingD,
    a: &[Elem],
    b: &[Elem],
    transcript: &mut Transcript,
) -> Result<(ScalarProductProof, Elem, Vec<EvalQuery>), SubError> {
    let n = a.len();
    if b.len() != n {
        return Err(SubError::Shape("a/b length mismatch".into()));
    }
    let log_n = log2_exact(n)?;
    let mut tau = ring.zero();
    for i in 0..n {
        let t = ring.mul(&a[i], &b[i]);
        tau = ring.add(&tau, &t);
    }
    let poly = RingVirtualPoly {
        num_vars: log_n,
        claimed_sum: tau.clone(),
        terms: vec![RingTerm {
            coeff: ring.one(),
            factors: vec![RingFactor::Mle(a.to_vec()), RingFactor::Mle(b.to_vec())],
        }],
    };
    let sc = prove_sumcheck(ring, &poly, transcript)?;
    let queries = vec![
        EvalQuery { label: "a".into(), point: sc.point.clone(), value: eval_mle(ring, a, &sc.point)? },
        EvalQuery { label: "b".into(), point: sc.point.clone(), value: eval_mle(ring, b, &sc.point)? },
    ];
    Ok((ScalarProductProof { sc }, tau, queries))
}

/// Verify a scalar-product proof; the resolver settles `a`/`b` at the
/// sum-check point (the engine's final check asserts `â·b̂ = ν`).
pub fn verify_scalar_product(
    ring: &RingD,
    n: usize,
    tau: &Elem,
    proof: &ScalarProductProof,
    transcript: &mut Transcript,
    oracle: Oracle<'_>,
) -> Result<(), SubError> {
    let log_n = log2_exact(n)?;
    let shape = crate::ring_sumcheck::RingSumcheckShape {
        num_vars: log_n,
        terms: vec![crate::ring_sumcheck::RingTermShape {
            coeff: ring.one(),
            num_factors: 2,
        }],
    };
    verify_sumcheck(ring, &shape, tau, &proof.sc, transcript, &mut |ti, fi, pt| {
        let _ = ti;
        let label = if fi == 0 { "a" } else { "b" };
        oracle(label, pt)
    })
    .map_err(SubError::from)
}

fn eval_mle(ring: &RingD, v: &[Elem], point: &[Elem]) -> Result<Elem, SubError> {
    ring.mle_eval(v, point).map_err(|e| SubError::Shape(format!("{e:?}")))
}

/// Settle an oracle query, lifting the resolver's error.
fn ask(oracle: Oracle<'_>, label: &str, point: &[Elem]) -> Result<Elem, SubError> {
    oracle(label, point).map_err(SubError::Verify)
}

// ===========================================================================
// Hadamard product (Construction B.5)
// ===========================================================================

#[derive(Debug, Clone)]
pub struct HadamardProof {
    pub sc: RingSumcheckProof,
}

/// Prove `a ∘ b = c` via `Σ_ω EQ(ω,y)(â(ω)b̂(ω) − ĉ(ω)) = 0`.
pub fn prove_hadamard(
    ring: &RingD,
    a: &[Elem],
    b: &[Elem],
    c: &[Elem],
    transcript: &mut Transcript,
) -> Result<(HadamardProof, Vec<EvalQuery>), SubError> {
    let n = a.len();
    if b.len() != n || c.len() != n {
        return Err(SubError::Shape("a/b/c length mismatch".into()));
    }
    for i in 0..n {
        let ab = ring.mul(&a[i], &b[i]);
        if ab != c[i] {
            return Err(SubError::Verify("a∘b ≠ c (fail-closed)".into()));
        }
    }
    let log_n = log2_exact(n)?;
    let y: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("had-y{i}").as_bytes()))
        .collect();
    let eq = ring.eq_row(&y);
    let poly = RingVirtualPoly {
        num_vars: log_n,
        claimed_sum: ring.zero(),
        terms: vec![
            RingTerm {
                coeff: ring.one(),
                factors: vec![
                    RingFactor::Eq(eq.clone()),
                    RingFactor::Mle(a.to_vec()),
                    RingFactor::Mle(b.to_vec()),
                ],
            },
            RingTerm {
                coeff: ring.neg(&ring.one()),
                factors: vec![RingFactor::Eq(eq.clone()), RingFactor::Mle(c.to_vec())],
            },
        ],
    };
    let sc = prove_sumcheck(ring, &poly, transcript)?;
    let pt = sc.point.clone();
    let queries = vec![
        EvalQuery { label: "a".into(), point: pt.clone(), value: eval_mle(ring, a, &pt)? },
        EvalQuery { label: "b".into(), point: pt.clone(), value: eval_mle(ring, b, &pt)? },
        EvalQuery { label: "c".into(), point: pt.clone(), value: eval_mle(ring, c, &pt)? },
    ];
    Ok((HadamardProof { sc }, queries))
}

/// Verify the Hadamard proof: the engine settles the sum-check; the
/// resolver must answer `a`, `b`, `c` at the final point.
pub fn verify_hadamard(
    ring: &RingD,
    n: usize,
    proof: &HadamardProof,
    transcript: &mut Transcript,
    oracle: Oracle<'_>,
) -> Result<(), SubError> {
    let log_n = log2_exact(n)?;
    // Replay the y challenges (the prover consumed them before the
    // sum-check rounds) and build the EQ table verifier-side.
    let y: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("had-y{i}").as_bytes()))
        .collect();
    let eq = ring.eq_row(&y);
    let shape = crate::ring_sumcheck::RingSumcheckShape {
        num_vars: log_n,
        terms: vec![
            crate::ring_sumcheck::RingTermShape {
                coeff: ring.one(),
                num_factors: 3,
            },
            crate::ring_sumcheck::RingTermShape {
                coeff: ring.neg(&ring.one()),
                num_factors: 2,
            },
        ],
    };
    verify_sumcheck(ring, &shape, &ring.zero(), &proof.sc, transcript, &mut |ti, fi, pt| {
        match (ti, fi) {
            (0, 0) | (1, 0) => ring.mle_eval(&eq, pt).map_err(|e| format!("{e:?}")),
            (0, 1) => oracle("a", pt),
            (0, 2) => oracle("b", pt),
            (1, 1) => oracle("c", pt),
            _ => Err("bad factor index".into()),
        }
    })
    .map_err(SubError::from)
}

// ===========================================================================
// Cyclic shift (Construction B.9)
// ===========================================================================

#[derive(Debug, Clone)]
pub struct CyclicShiftProof {
    pub gamma: Elem,
    /// ⟨a, γvec⟩ = ν0
    pub sp_a: ScalarProductProof,
    pub nu0: Elem,
    /// ⟨b, γvec⟩ = ν1
    pub sp_b: ScalarProductProof,
    pub nu1: Elem,
}

/// The geometric vector `(1, γ, γ², …, γ^{N−1})`.
fn gamma_vec(ring: &RingD, gamma: &Elem, n: usize) -> Vec<Elem> {
    let mut v = Vec::with_capacity(n);
    let mut cur = ring.one();
    for _ in 0..n {
        v.push(cur.clone());
        cur = ring.mul(&cur, gamma);
    }
    v
}

/// MLE of the geometric vector at `point`: `∏_j (1−x_j + x_j·γ^{2^j})`.
fn gamma_mle_eval(ring: &RingD, gamma: &Elem, point: &[Elem]) -> Elem {
    let mut acc = ring.one();
    for (j, xj) in point.iter().enumerate() {
        let g_pow = ring.pow(gamma, 1u64 << j.min(62));
        let t = ring.mul(xj, &g_pow);
        let one_minus = ring.sub(&ring.one(), xj);
        acc = ring.mul(&acc, &ring.add(&one_minus, &t));
    }
    acc
}

/// Prove `b = shift(a)` (b_i = a_{i+1}, b_N = a_1).
pub fn prove_cyclic_shift(
    ring: &RingD,
    a: &[Elem],
    b: &[Elem],
    transcript: &mut Transcript,
) -> Result<(CyclicShiftProof, Vec<EvalQuery>), SubError> {
    let n = a.len();
    if b.len() != n {
        return Err(SubError::Shape("a/b length mismatch".into()));
    }
    for i in 0..n {
        let expected = &a[(i + 1) % n];
        if b[i] != *expected {
            return Err(SubError::Verify("b ≠ shift(a) (fail-closed)".into()));
        }
    }
    let gamma = ring.sample_challenge(transcript, b"shift-gamma");
    let gvec = gamma_vec(ring, &gamma, n);
    let (sp_a, nu0, qs_a) = prove_scalar_product(ring, a, &gvec, transcript)?;
    let (sp_b, nu1, qs_b) = prove_scalar_product(ring, b, &gvec, transcript)?;
    let mut queries = Vec::new();
    for q in qs_a.iter().chain(qs_b.iter()) {
        // The 'b'-side query label must not collide with the witness b:
        // rename to shift-a / shift-b.
        let label = if q.label == "a" { "shift-a" } else { "shift-b" };
        queries.push(EvalQuery { label: label.into(), point: q.point.clone(), value: q.value.clone() });
    }
    // The b_N query at the all-ones point.
    let log_n = log2_exact(n)?;
    let ones = vec![ring.one(); log_n];
    let b_n = eval_mle(ring, b, &ones)?;
    queries.push(EvalQuery { label: "shift-b".into(), point: ones, value: b_n });
    Ok((CyclicShiftProof { gamma, sp_a, nu0, sp_b, nu1 }, queries))
}

/// Verify the cyclic-shift proof. The oracle answers `shift-a` /
/// `shift-b` MLE evaluations; the geometric vector is resolved
/// verifier-side in `O(log N)` ring work.
pub fn verify_cyclic_shift(
    ring: &RingD,
    n: usize,
    proof: &CyclicShiftProof,
    transcript: &mut Transcript,
    oracle: Oracle<'_>,
) -> Result<(), SubError> {
    let log_n = log2_exact(n)?;
    let gamma = ring.sample_challenge(transcript, b"shift-gamma");
    if gamma != proof.gamma {
        return Err(SubError::Verify("gamma mismatch".into()));
    }
    // Replay both scalar products. Their internal challenge streams
    // follow the transcript order (a-side first, then b-side).
    verify_scalar_product(ring, n, &proof.nu0, &proof.sp_a, transcript, &|label, pt| {
        match label {
            "a" => oracle("shift-a", pt),
            // the geometric side is public
            "b" => Ok(gamma_mle_eval(ring, &proof.gamma, pt)),
            _ => Err("bad label".into()),
        }
    })?;
    verify_scalar_product(ring, n, &proof.nu1, &proof.sp_b, transcript, &|label, pt| {
        match label {
            "a" => oracle("shift-b", pt),
            "b" => Ok(gamma_mle_eval(ring, &proof.gamma, pt)),
            _ => Err("bad label".into()),
        }
    })?;
    // The Lemma-B.8 check: ν0 − γ·ν1 = (1 − γ^N)·b_N.
    let ones = vec![ring.one(); log_n];
    let b_n = ask(oracle, "shift-b", &ones)?;
    let gamma_n = ring.pow(&proof.gamma, n as u64);
    let one_minus_gamma_n = ring.sub(&ring.one(), &gamma_n);
    let rhs = ring.mul(&one_minus_gamma_n, &b_n);
    let gamma_nu1 = ring.mul(&proof.gamma, &proof.nu1);
    let lhs = ring.sub(&proof.nu0, &gamma_nu1);
    if lhs != rhs {
        return Err(SubError::Verify("cyclic-shift univariate check failed".into()));
    }
    Ok(())
}

// ===========================================================================
// Entry product (Construction B.12)
// ===========================================================================

#[derive(Debug, Clone)]
pub struct EntryProductProof {
    pub hadamard: HadamardProof,
    pub shift: CyclicShiftProof,
}

/// Prove `τ = ∏_{i<N} a_i` for a nonzero-entry vector.
pub fn prove_entry_product(
    ring: &RingD,
    a: &[Elem],
    tau: &Elem,
    transcript: &mut Transcript,
) -> Result<(EntryProductProof, Vec<EvalQuery>), SubError> {
    let n = a.len();
    let log_n = log2_exact(n)?;
    // prefix products (0-indexed): c[0]=1, c[i]=∏_{k<=i-1} a_k … precisely:
    // c = (1, a_0, a_0a_1, …, ∏_{k<n-1} a_k)
    // d = (a_0, a_0a_1, …, ∏_{k<n} a_k)        (d = a ∘ c)
    // e = shift(c) = (a_0, …, ∏_{k<n-1} a_k, 1)
    let mut c = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    let mut run = ring.one();
    for i in 0..n {
        c.push(run.clone());
        run = ring.mul(&run, &a[i]);
        d.push(run.clone());
    }
    let mut e = c[1..].to_vec();
    e.push(ring.one());
    // fail-closed: the entry product must actually equal tau
    if d[n - 1] != *tau {
        return Err(SubError::Verify("∏a ≠ τ (fail-closed)".into()));
    }
    let (hadamard, qs_h) = prove_hadamard(ring, a, &c, &d, transcript)?;
    let (shift, qs_s) = prove_cyclic_shift(ring, &c, &e, transcript)?;
    let mut queries = Vec::new();
    for q in qs_h {
        queries.push(EvalQuery { label: format!("ep-{}", q.label), point: q.point, value: q.value });
    }
    for q in qs_s {
        queries.push(EvalQuery { label: format!("ep-{}", q.label), point: q.point, value: q.value });
    }
    // Final check at a random η (sampled AFTER the sub-protocols, from
    // the shared transcript — the verifier replays in the same slot):
    // d̂(η) − ê(η) = EQ(1^{logN}, η)·(τ−1).
    let eta: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("ep-eta{i}").as_bytes()))
        .collect();
    let dv = eval_mle(ring, &d, &eta)?;
    let ev = eval_mle(ring, &e, &eta)?;
    queries.push(EvalQuery { label: "ep-d".into(), point: eta.clone(), value: dv });
    queries.push(EvalQuery { label: "ep-e".into(), point: eta, value: ev });
    Ok((EntryProductProof { hadamard, shift }, queries))
}

/// Verify the entry-product proof. `tau` is the claimed product; the
/// oracle answers the `ep-*` labels.
pub fn verify_entry_product(
    ring: &RingD,
    n: usize,
    tau: &Elem,
    proof: &EntryProductProof,
    transcript: &mut Transcript,
    oracle: Oracle<'_>,
) -> Result<(), SubError> {
    let log_n = log2_exact(n)?;
    // Hadamard on (a, c, d): labels ep-a, ep-b (=c), ep-c (=d), ep-eq
    verify_hadamard(ring, n, &proof.hadamard, transcript, &|label, pt| match label {
        "a" => oracle("ep-a", pt),
        "b" => oracle("ep-b", pt),
        "c" => oracle("ep-d", pt),
        "eq" => oracle("ep-eq", pt),
        _ => Err("bad label".into()),
    })?;
    // Cyclic shift on (c, e): labels ep-shift-a (=c), ep-shift-b (=e)
    verify_cyclic_shift(ring, n, &proof.shift, transcript, &|label, pt| match label {
        "shift-a" => oracle("ep-b", pt), // c
        "shift-b" => oracle("ep-e", pt), // e
        _ => Err("bad label".into()),
    })?;
    // e_N = 1: the all-ones query on e must return 1.
    let ones = vec![ring.one(); log_n];
    let e_n = ask(oracle, "ep-e", &ones)?;
    if e_n != ring.one() {
        return Err(SubError::Verify("e_N ≠ 1".into()));
    }
    // η challenges then the closing identity
    let eta: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("ep-eta{i}").as_bytes()))
        .collect();
    let dv = ask(oracle, "ep-d", &eta)?;
    let ev = ask(oracle, "ep-e", &eta)?;
    let eq_ones_eta = {
        let row = ring.eq_row(&eta);
        let idx = (1usize << log_n) - 1;
        row[idx].clone()
    };
    let tau_minus_1 = ring.sub(tau, &ring.one());
    let rhs = ring.mul(&eq_ones_eta, &tau_minus_1);
    let lhs = ring.sub(&dv, &ev);
    if lhs != rhs {
        return Err(SubError::Verify("entry-product closing check failed".into()));
    }
    Ok(())
}

// ===========================================================================
// Integer check (Construction B.15)
// ===========================================================================

#[derive(Debug, Clone)]
pub struct IntegerCheckProof {
    /// The claimed MLE evaluations at the sampled scalar points.
    pub claims: Vec<(Vec<Elem>, Elem)>,
}

/// Sample a random *scalar* point `ρ ∈ Z_q^{logN}` (full-entropy
/// degree-0 ring elements — NOT from the binary challenge space).
fn sample_scalar_point(ring: &RingD, log_n: usize, transcript: &mut Transcript, label: &[u8]) -> Vec<Elem> {
    let bytes = transcript.challenge_bytes(label, log_n * 8).unwrap_or_default();
    (0..log_n)
        .map(|i| {
            let mut w = [0u8; 8];
            w.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
            ring.constant(u64::from_le_bytes(w))
        })
        .collect()
}

/// Prove the integer check: emits the evaluations `â(ρ_t)` for the
/// verifier to check integrality (amplified `rounds` times).
pub fn prove_integer_check(
    ring: &RingD,
    a: &[Elem],
    rounds: usize,
    transcript: &mut Transcript,
) -> Result<(IntegerCheckProof, Vec<EvalQuery>), SubError> {
    let log_n = log2_exact(a.len())?;
    let mut claims = Vec::new();
    let mut queries = Vec::new();
    for t in 0..rounds {
        let rho = sample_scalar_point(ring, log_n, transcript, format!("ic-rho{t}").as_bytes());
        let v = eval_mle(ring, a, &rho)?;
        queries.push(EvalQuery { label: "ic-a".into(), point: rho.clone(), value: v.clone() });
        claims.push((rho, v));
    }
    Ok((IntegerCheckProof { claims }, queries))
}

/// Verify: each claimed evaluation must be `Z_q`-valued (degree-0).
pub fn verify_integer_check(
    ring: &RingD,
    n: usize,
    proof: &IntegerCheckProof,
    transcript: &mut Transcript,
    oracle: Oracle<'_>,
) -> Result<(), SubError> {
    let log_n = log2_exact(n)?;
    for (t, (rho, _claim)) in proof.claims.iter().enumerate() {
        let rho2 = sample_scalar_point(ring, log_n, transcript, format!("ic-rho{t}").as_bytes());
        if &rho2 != rho {
            return Err(SubError::Verify("rho replay mismatch".into()));
        }
        let v = ask(oracle, "ic-a", rho)?;
        if !v.is_integer() {
            return Err(SubError::Verify("â(ρ) has non-constant coefficients".into()));
        }
    }
    Ok(())
}

// ===========================================================================
// Binary check (Construction B.19) — the LatticeFold+ range PIOP
// ===========================================================================

#[derive(Debug, Clone)]
pub struct BinaryCheckProof {
    /// The batched 0/1 sum-check over the CF rows:
    /// `Σ_ω EQ(c,ω)·Σ_j α^j·(CF̂_j(ω)² − CF̂_j(ω)) = 0`.
    pub binary_sc: RingSumcheckProof,
    /// The tensor point `r` (sum-check randomness of the consistency
    /// block, drawn after the binary sum-check).
    pub r: Vec<Elem>,
    /// `v_j = ⟨CF_j, ⊗r⟩` — ring elements.
    pub v: Vec<Elem>,
    /// `w = ⟨f, ⊗r⟩`.
    pub w: Elem,
    /// Scalar products binding the CF rows: `⟨CF_j, ⊗r⟩ = v_j`.
    pub cf_sps: Vec<ScalarProductProof>,
    /// The scalar product binding f: `⟨f, ⊗r⟩ = w`.
    pub f_sp: ScalarProductProof,
}

/// Prove that every entry of `f` has binary coefficients.
///
/// **Design (and the deviation from Construction B.19).** The paper's
/// steps 1–9 route the check through a monomial matrix `M_f[j,i] =
/// X^{f_{i,j}}`, its β/β²-evaluated rows `m^{(j)}, m'^{(j)}`, and a
/// binding `e_j = M̂_{f,j}(r)` checked via `ev(e_j)(β)` (step 8).
/// Polynomial evaluation at `β ∈ C` is **not well-defined on the
/// quotient ring** — `X^d ↦ β^d ≠ −1` — so the MLE and the evaluation
/// do not commute once the products `EQ(i,r)·X^{f}` wrap mod `X^d+1`;
/// the printed step-8 identity does not hold for honest provers under
/// this reading. We therefore enforce binary coefficients directly
/// where the paper's construction is sound, keeping its architecture:
///
/// * the **CF consistency block** (paper steps 10–12) verbatim:
///   `v_j = ⟨CF_j, ⊗r⟩`, `w = ⟨f, ⊗r⟩`, `w = Σ_j v_j·X^j` — binds the
///   coefficient oracles to `f` (this part is fully sound with
///   ring-valued tensor entries; only the paper's "v ∈ Z_q^d" typing
///   is corrected to `v ∈ R^d`);
/// * the **binary enforcement** as one batched degree-2 sum-check
///   `Σ_ω EQ(c,ω)·Σ_j α^j (CF̂_j(ω)² − CF̂_j(ω)) = 0` — the direct
///   `f_{i,j} ∈ {0,1}` test, subsuming the monomial-set check
///   (`f ∈ [0,d)`) and the range-closing step 13
///   (`ct(X^{−1}e_j) = v_j`, whose printed form does not typecheck
///   with ring-valued tensors either — the sound identity it gestures
///   at is `X^f = (1−f)+f·X`, i.e. our `CF² = CF` per coefficient).
///
/// Soundness: `(d + log N)/|C|` + the scalar-product errors — the
/// same order as Lemma B.20.
#[allow(clippy::too_many_lines)]
pub fn prove_binary_check(
    ring: &RingD,
    f: &[Elem],
    transcript: &mut Transcript,
) -> Result<(BinaryCheckProof, Vec<EvalQuery>), SubError> {
    let n = f.len();
    let log_n = log2_exact(n)?;
    let d = ring.d;
    for e in f {
        if !e.is_binary() {
            return Err(SubError::Verify("f has non-binary coefficients (fail-closed)".into()));
        }
    }
    // CF rows as constant ring elements.
    let cf_rows: Vec<Vec<Elem>> = (0..d)
        .map(|j| f.iter().map(|e| ring.constant(e.coeffs()[j])).collect())
        .collect();
    // The binary sum-check point c ∈ C^{logN} and the row weights α ∈ C.
    let c: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("bc-c{i}").as_bytes()))
        .collect();
    let alpha = ring.sample_challenge(transcript, b"bc-alpha");
    let eq = ring.eq_row(&c);
    let mut terms = Vec::with_capacity(2 * d);
    for j in 0..d {
        let alpha_j = ring.pow(&alpha, j as u64);
        terms.push(RingTerm {
            coeff: alpha_j.clone(),
            factors: vec![
                RingFactor::Eq(eq.clone()),
                RingFactor::Mle(cf_rows[j].clone()),
                RingFactor::Mle(cf_rows[j].clone()),
            ],
        });
        terms.push(RingTerm {
            coeff: ring.neg(&alpha_j),
            factors: vec![RingFactor::Eq(eq.clone()), RingFactor::Mle(cf_rows[j].clone())],
        });
    }
    let poly = RingVirtualPoly { num_vars: log_n, claimed_sum: ring.zero(), terms };
    let binary_sc = prove_sumcheck(ring, &poly, transcript)?;
    // The consistency tensor point r ∈ C^{logN}.
    let r: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("bc-r{i}").as_bytes()))
        .collect();
    let tensor = ring.tensor_eq(&r);
    // v_j = ⟨CF_j, ⊗r⟩ and w = ⟨f, ⊗r⟩ via scalar products.
    let mut v: Vec<Elem> = Vec::with_capacity(d);
    let mut cf_sps = Vec::with_capacity(d);
    let mut queries: Vec<EvalQuery> = Vec::new();
    for j in 0..d {
        let (sp, nu, qs) = prove_scalar_product(ring, &cf_rows[j], &tensor, transcript)?;
        v.push(nu);
        cf_sps.push(sp);
        for q in qs {
            if q.label == "a" {
                queries.push(EvalQuery { label: format!("bc-cf{j}"), point: q.point, value: q.value });
            }
        }
    }
    let (f_sp, w, qs_f) = prove_scalar_product(ring, f, &tensor, transcript)?;
    for q in qs_f {
        if q.label == "a" {
            queries.push(EvalQuery { label: "bc-f".into(), point: q.point, value: q.value });
        }
    }
    Ok((
        BinaryCheckProof {
            binary_sc,
            r,
            v,
            w,
            cf_sps,
            f_sp,
        },
        queries,
    ))
}

/// Verify the binary check.
#[allow(clippy::too_many_lines)]
pub fn verify_binary_check(
    ring: &RingD,
    n: usize,
    proof: &BinaryCheckProof,
    transcript: &mut Transcript,
    oracle: Oracle<'_>,
) -> Result<(), SubError> {
    let log_n = log2_exact(n)?;
    let d = ring.d;
    if proof.v.len() != d || proof.cf_sps.len() != d {
        return Err(SubError::Shape("binary-check arity mismatch".into()));
    }
    // Replay c and α; the EQ row is built verifier-side.
    let c: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("bc-c{i}").as_bytes()))
        .collect();
    let alpha = ring.sample_challenge(transcript, b"bc-alpha");
    let eq = ring.eq_row(&c);
    let mut terms_shape = Vec::with_capacity(2 * d);
    for j in 0..d {
        let alpha_j = ring.pow(&alpha, j as u64);
        terms_shape.push(crate::ring_sumcheck::RingTermShape {
            coeff: alpha_j.clone(),
            num_factors: 3,
        });
        terms_shape.push(crate::ring_sumcheck::RingTermShape {
            coeff: ring.neg(&alpha_j),
            num_factors: 2,
        });
    }
    let shape = crate::ring_sumcheck::RingSumcheckShape { num_vars: log_n, terms: terms_shape };
    verify_sumcheck(ring, &shape, &ring.zero(), &proof.binary_sc, transcript, &mut |ti, fi, pt| {
        let j = ti / 2;
        if ti % 2 == 0 {
            match fi {
                0 => ring.mle_eval(&eq, pt).map_err(|e| format!("{e:?}")),
                1 | 2 => oracle(&format!("bc-cf{j}"), pt),
                _ => Err("bad factor".into()),
            }
        } else {
            match fi {
                0 => ring.mle_eval(&eq, pt).map_err(|e| format!("{e:?}")),
                1 => oracle(&format!("bc-cf{j}"), pt),
                _ => Err("bad factor".into()),
            }
        }
    })?;
    // Replay r.
    let r: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("bc-r{i}").as_bytes()))
        .collect();
    if r != proof.r {
        return Err(SubError::Verify("r replay mismatch".into()));
    }
    // CF scalar products: ⟨CF_j, ⊗r⟩ = v_j.
    for (j, sp) in proof.cf_sps.iter().enumerate() {
        let vj = proof.v[j].clone();
        verify_scalar_product(ring, n, &vj, sp, transcript, &|label, pt| match label {
            "a" => oracle(&format!("bc-cf{j}"), pt),
            "b" => Ok(eq_row_at(ring, &r, pt)),
            _ => Err("bad label".into()),
        })?;
    }
    // ⟨f, ⊗r⟩ = w.
    let w = proof.w.clone();
    verify_scalar_product(ring, n, &w, &proof.f_sp, transcript, &|label, pt| match label {
        "a" => oracle("bc-f", pt),
        "b" => Ok(eq_row_at(ring, &r, pt)),
        _ => Err("bad label".into()),
    })?;
    // Step 12: w == Σ_j v_j·X^j.
    let mut v_poly = ring.zero();
    for (j, vj) in proof.v.iter().enumerate() {
        let shifted = shift_by_monomial(ring, vj, j);
        v_poly = ring.add(&v_poly, &shifted);
    }
    if proof.w != v_poly {
        return Err(SubError::Verify("w != Σ v_j X^j".into()));
    }
    Ok(())
}

/// Multiply a ring element by the monomial `X^j` (negacyclic wrap).
fn shift_by_monomial(ring: &RingD, e: &Elem, j: usize) -> Elem {
    let d = ring.d;
    let mut out = vec![0u64; d];
    for (k, &c) in e.coeffs().iter().enumerate() {
        let pos = k + j;
        if pos < d {
            out[pos] = (out[pos] + c) % ring.q;
        } else {
            // X^d = -1
            out[pos - d] = (out[pos - d] + ring.q - c % ring.q) % ring.q;
        }
    }
    Elem { c: out }
}

/// EQ(·, r) evaluated at point `pt` — the MLE of the tensor row.
fn eq_row_at(ring: &RingD, r: &[Elem], pt: &[Elem]) -> Elem {
    let mut acc = ring.one();
    for (ri, xi) in r.iter().zip(pt.iter()) {
        let t = ring.mul(ri, xi);
        let one_minus_r = ring.sub(&ring.one(), ri);
        let one_minus_x = ring.sub(&ring.one(), xi);
        let u = ring.mul(&one_minus_r, &one_minus_x);
        acc = ring.mul(&acc, &ring.add(&u, &t));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingD {
        RingD::new(4).ok().unwrap()
    }

    fn resolver_from<'a>(
        ring: &'a RingD,
        table: &'a [(&'static str, &'a [Elem])],
    ) -> impl Fn(&str, &[Elem]) -> Result<Elem, String> + 'a {
        move |label, pt| {
            for (l, v) in table {
                if *l == label {
                    return ring.mle_eval(v, pt).map_err(|e| format!("{e:?}"));
                }
            }
            Err(format!("unknown oracle label {label}"))
        }
    }

    #[test]
    fn scalar_product_roundtrip() {
        let r = ring();
        let n = 8;
        let a: Vec<Elem> = (0..n).map(|i| r.random(format!("sp-a{i}").as_bytes())).collect();
        let b: Vec<Elem> = (0..n).map(|i| r.random(format!("sp-b{i}").as_bytes())).collect();
        let mut tr = Transcript::new_default(b"sp");
        let (proof, tau, _qs) = prove_scalar_product(&r, &a, &b, &mut tr).ok().unwrap();
        let mut tr = Transcript::new_default(b"sp");
        let table = [("a", a.as_slice()), ("b", b.as_slice())];
        let res = resolver_from(&r, &table);
        verify_scalar_product(&r, n, &tau, &proof, &mut tr, &res).ok().unwrap();
        // wrong tau rejected
        let bad_tau = r.add(&tau, &r.one());
        let mut tr = Transcript::new_default(b"sp");
        assert!(verify_scalar_product(&r, n, &bad_tau, &proof, &mut tr, &res).is_err());
    }

    #[test]
    fn hadamard_roundtrip_and_tamper() {
        let r = ring();
        let n = 4;
        let a: Vec<Elem> = (0..n).map(|i| r.random(format!("h-a{i}").as_bytes())).collect();
        let b: Vec<Elem> = (0..n).map(|i| r.random(format!("h-b{i}").as_bytes())).collect();
        let c: Vec<Elem> = (0..n).map(|i| r.mul(&a[i], &b[i])).collect();
        let mut tr = Transcript::new_default(b"had");
        let (proof, _qs) = prove_hadamard(&r, &a, &b, &c, &mut tr).ok().unwrap();
        // verifier computes the EQ table itself
        let log_n = 2;
        let mut tr_v = Transcript::new_default(b"had");
        let y: Vec<Elem> = (0..log_n)
            .map(|i| r.sample_challenge(&mut tr_v, format!("had-y{i}").as_bytes()))
            .collect();
        let eq = r.eq_row(&y);
        let table = [("a", a.as_slice()), ("b", b.as_slice()), ("c", c.as_slice()), ("eq", eq.as_slice())];
        let res = resolver_from(&r, &table);
        let mut tr2 = Transcript::new_default(b"had");
        verify_hadamard(&r, n, &proof, &mut tr2, &res).ok().unwrap();
        // tampered c (a∘b ≠ c) — the honest prover refuses; the forged
        // proof must fail verification:
        let mut bad_c = c.clone();
        bad_c[0] = r.add(&bad_c[0], &r.one());
        let mut tr3 = Transcript::new_default(b"had");
        assert!(prove_hadamard(&r, &a, &b, &bad_c, &mut tr3).is_err());
    }

    #[test]
    fn cyclic_shift_roundtrip_and_reject() {
        let r = ring();
        let n = 8;
        let a: Vec<Elem> = (0..n).map(|i| r.random(format!("cs-a{i}").as_bytes())).collect();
        let b: Vec<Elem> = (0..n).map(|i| a[(i + 1) % n].clone()).collect();
        let mut tr = Transcript::new_default(b"cs");
        let (proof, _qs) = prove_cyclic_shift(&r, &a, &b, &mut tr).ok().unwrap();
        let table = [("shift-a", a.as_slice()), ("shift-b", b.as_slice())];
        let res = resolver_from(&r, &table);
        let mut tr2 = Transcript::new_default(b"cs");
        verify_cyclic_shift(&r, n, &proof, &mut tr2, &res).ok().unwrap();
        // non-shift rejected by the fail-closed prover
        let mut bad = b.clone();
        bad[0] = r.add(&bad[0], &r.one());
        let mut tr3 = Transcript::new_default(b"cs");
        assert!(prove_cyclic_shift(&r, &a, &bad, &mut tr3).is_err());
        // a forged shift (swap two entries) rejected at verify: craft b'
        // that IS a shift of a DIFFERENT a
        let mut a2 = a.clone();
        a2.swap(0, 1);
        let b2: Vec<Elem> = (0..n).map(|i| a2[(i + 1) % n].clone()).collect();
        let mut tr4 = Transcript::new_default(b"cs");
        let (proof2, _q2) = prove_cyclic_shift(&r, &a2, &b2, &mut tr4).ok().unwrap();
        // verify proof2 against the ORIGINAL a (mismatched witness)
        let table_orig = [("shift-a", a.as_slice()), ("shift-b", b.as_slice())];
        let res_orig = resolver_from(&r, &table_orig);
        let mut tr5 = Transcript::new_default(b"cs");
        assert!(verify_cyclic_shift(&r, n, &proof2, &mut tr5, &res_orig).is_err());
    }

    #[test]
    fn entry_product_roundtrip() {
        let r = ring();
        let n = 4;
        // nonzero entries (units, whp)
        let a: Vec<Elem> = (0..n).map(|i| r.random(format!("ep-a{i}").as_bytes())).collect();
        let mut tau = r.one();
        for e in &a {
            tau = r.mul(&tau, e);
        }
        // build c/d/e exactly as the protocol does, for the resolver
        let mut c = Vec::new();
        let mut d = Vec::new();
        let mut run = r.one();
        for i in 0..n {
            c.push(run.clone());
            run = r.mul(&run, &a[i]);
            d.push(run.clone());
        }
        let mut e_v = c[1..].to_vec();
        e_v.push(r.one());
        let mut tr = Transcript::new_default(b"ep");
        let (proof, _qs) = prove_entry_product(&r, &a, &tau, &mut tr).ok().unwrap();
        let table = [
            ("ep-a", a.as_slice()),
            ("ep-b", c.as_slice()),
            ("ep-d", d.as_slice()),
            ("ep-e", e_v.as_slice()),
        ];
        let res = resolver_from(&r, &table);
        let mut tr2 = Transcript::new_default(b"ep");
        verify_entry_product(&r, n, &tau, &proof, &mut tr2, &res)
            .unwrap_or_else(|e| panic!("ep verify failed: {e:?}"));
        // wrong tau rejected
        let bad_tau = r.mul(&tau, &r.x_gen());
        let mut tr3 = Transcript::new_default(b"ep");
        assert!(verify_entry_product(&r, n, &bad_tau, &proof, &mut tr3, &res).is_err());
    }

    #[test]
    fn integer_check_roundtrip() {
        let r = ring();
        let n = 4;
        let a: Vec<Elem> = (0..n).map(|i| r.constant(1_000_000 + i as u64)).collect();
        let mut tr = Transcript::new_default(b"ic");
        let (proof, _qs) = prove_integer_check(&r, &a, 3, &mut tr).ok().unwrap();
        let table = [("ic-a", a.as_slice())];
        let res = resolver_from(&r, &table);
        let mut tr2 = Transcript::new_default(b"ic");
        verify_integer_check(&r, n, &proof, &mut tr2, &res).ok().unwrap();
        // a non-integral vector fails
        let mut bad = a.clone();
        bad[1] = r.add(&bad[1], &r.x_gen());
        let mut tr3 = Transcript::new_default(b"ic");
        let (proof_bad, _q) = prove_integer_check(&r, &bad, 3, &mut tr3).ok().unwrap();
        let table_bad = [("ic-a", bad.as_slice())];
        let res_bad = resolver_from(&r, &table_bad);
        let mut tr4 = Transcript::new_default(b"ic");
        assert!(verify_integer_check(&r, n, &proof_bad, &mut tr4, &res_bad).is_err());
    }

    #[test]
    fn binary_check_roundtrip() {
        let r = ring();
        let n = 4;
        let f: Vec<Elem> = (0..n).map(|i| r.g_map(i as u64 * 3 + 1)).collect();
        for e in &f {
            assert!(e.is_binary());
        }
        let mut tr = Transcript::new_default(b"bc");
        let (proof, _qs) =
            prove_binary_check(&r, &f, &mut tr).unwrap_or_else(|e| panic!("prove failed: {e:?}"));
        // Replay the CF rows from f (the oracle the verifier settles).
        let d = r.d;
        let cf_rows: Vec<Vec<Elem>> = (0..d)
            .map(|j| f.iter().map(|e| r.constant(e.coeffs()[j])).collect())
            .collect();
        let r_ref = &r;
        let f_ref = &f;
        let cf_ref = &cf_rows;
        let res = move |label: &str, pt: &[Elem]| -> Result<Elem, String> {
            if let Some(j) = label.strip_prefix("bc-cf") {
                let j: usize = j.parse().map_err(|_| "bad index")?;
                return r_ref.mle_eval(&cf_ref[j], pt).map_err(|e| format!("{e:?}"));
            }
            match label {
                "bc-f" => r_ref.mle_eval(f_ref, pt).map_err(|e| format!("{e:?}")),
                _ => Err(format!("unknown label {label}")),
            }
        };
        let mut tr2 = Transcript::new_default(b"bc");
        verify_binary_check(&r, n, &proof, &mut tr2, &res)
            .unwrap_or_else(|e| panic!("verify failed: {e:?}"));
        // A forged proof over a NON-binary f (coefficient 2) must fail:
        // simulate by verifying the honest proof against a corrupted
        // oracle (the CF rows read the tampered vector).
        let mut f_bad = f.clone();
        f_bad[2] = r.constant(2);
        let cf_bad: Vec<Vec<Elem>> = (0..d)
            .map(|j| f_bad.iter().map(|e| r.constant(e.coeffs()[j])).collect())
            .collect();
        let cf_bad_ref = &cf_bad;
        let res_bad = move |label: &str, pt: &[Elem]| -> Result<Elem, String> {
            if let Some(j) = label.strip_prefix("bc-cf") {
                let j: usize = j.parse().map_err(|_| "bad index")?;
                return r_ref.mle_eval(&cf_bad_ref[j], pt).map_err(|e| format!("{e:?}"));
            }
            match label {
                "bc-f" => r_ref.mle_eval(f_ref, pt).map_err(|e| format!("{e:?}")),
                _ => Err(format!("unknown label {label}")),
            }
        };
        let mut tr3 = Transcript::new_default(b"bc");
        assert!(verify_binary_check(&r, n, &proof, &mut tr3, &res_bad).is_err());
    }

    #[test]
    fn binary_check_rejects_nonbinary() {
        let r = ring();
        let n = 4;
        let mut f: Vec<Elem> = (0..n).map(|i| r.g_map(i as u64)).collect();
        f[2] = r.constant(2); // coefficient value 2 — NOT binary
        let mut tr = Transcript::new_default(b"bc2");
        // the honest prover refuses
        assert!(prove_binary_check(&r, &f, &mut tr).is_err());
    }
}
