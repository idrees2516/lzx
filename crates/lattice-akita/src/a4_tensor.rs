//! A4 — Akita tensor reduction to the extension field + the
//! evaluation-trace row (ePrint 2026/1983, §3.7 + §7.1): Wave 7 item
//! 7.11, A4.
//!
//! * **§3.7 Tensor Reduction (Diamond–Posen, Theorem 3.5)** — reduces an
//!   evaluation claim on a base-field multilinear polynomial at an
//!   extension-field point to a single evaluation claim on a *packed*
//!   extension-field polynomial with `κ` fewer variables:
//!   * the packed polynomial `g(X_tail) = Σ_y f(y, X_tail)·β_y` over the
//!     basis `(β_y)` of `E = F_{q^{2κ}}` (kernel: `κ = 1`, `E = Fq2`);
//!   * the prover sends the **column partials** `S_y = f(y, r_tail)` and
//!     the verifier checks the input claim by multilinear recombination
//!     `v = Σ_y eq(r_head, y)·S_y`;
//!   * **the tensor step** (the paper's point: it prevents the insecure
//!     shortcut `Σ_y β_y·S_y`, which binds only F_q-coordinates): the
//!     verifier decomposes the columns in F_q-coordinates
//!     `S_y = Σ_u S_{u,y}·β_u` and forms the **row partials**
//!     `row_u = Σ_y S_{u,y}·β_y` itself;
//!   * the row claims `(⋆) row_u = Σ_w A_u(w)·g(w)` (with `A_u` the MLEs
//!     of the tail-equality coordinates) are batched with the
//!     transcript challenge `η` via `η_u = eq(η, u)` into
//!     `c_η = Σ_u η_u·row_u`, proven by ONE degree-2 E-valued sumcheck
//!     `Σ_w A_η(w)·g(w) = c_η`;
//!   * the transparent factor `A_η(ρ)` is verifier-computed through the
//!     tensor algebra `e = eq(ϕ₀(r_tail), ϕ₁(ρ)) = Σ_u β_u ⊗ e_u`,
//!     realized via the two algebra maps `E⊗_F E → E` (`a⊗b ↦ a·b` and
//!     `a⊗b ↦ σ(a)·b`): `e₀ = (eq(r,ρ) + eq(σ(r),ρ))/2`,
//!     `e₁ = (eq(r,ρ) − eq(σ(r),ρ))/(2β)`, `A_η(ρ) = η₀·e₀ + η₁·e₁` —
//!     multilinear in `ρ` and equal to the row MLEs' combination on the
//!     whole domain (pinned by `a_u_mles_agree_with_conjugate_formula`);
//!     the verifier rejects `A_η(ρ) = 0` (the paper's no-resample abort);
//!   * the **shared batched reduction** (Theorem 3.11): groups with
//!     different tail arities batch via cylindrical extensions
//!     `g̃_a(w, z) = g_a(w)`, `Ã_η,a = A_η,a·eq(0, z)`, a shared `η`,
//!     normalized claim combiners `ζ^{tensor}` (first 1, rest sampled),
//!     per-group transparent factors
//!     `θ_a = A_η,a(ρ_a)·Π_{t ≥ m_a}(1 − ρ_t)` with zero-rejection, and
//!     final products `h_{a,i} = θ_a·g_{a,i}(ρ_a)`.
//! * **§7.1 The evaluation-trace row** — the trace-packing map
//!   `ψ: E^{d_A/k} → R_{q,d_A}` (interleaved coordinates at kernel
//!   scale), the conjugation `σ⁻¹` (`X ↦ X^{−1}`, an involution since
//!   `X^d ≡ −1`), and the public linear functional `T_{ρ_pack}` (Eq 133)
//!   whose pinned packing identity is
//!   `T(ψ(y₀,…)) = Σ_u eq(ρ̃_pack, u)·y_u` — the Hachi trace identity
//!   that turns the packed opening into the field row
//!   `Σ_i χ_blk(i)·T_{ρ_pack}(e_i) = v̄` (Eq 134), with the digit
//!   weights `ω_Tr(i, ℓ, ν) = χ_blk(i)·b^ℓ·T_{ρ_pack}(X^ν)` (Eq 135).
//!
//! LZX realization notes (kernel scale, honestly stated): the E-valued
//! sum-check samples its per-round challenges from the BASE field F_q
//! (soundness `2m/|F_q|` instead of `2m/|E|`; Goldilocks' `2^64` leaves
//! enormous margin — documented deviation); `Tr_H` is realized as the
//! E-bilinear pairing against the packed weight — the exact functional
//! pinned by the packing identity — with the σ⁻¹ conjugation implemented
//! and its coefficient identity `(Z·σ⁻¹(χ̌))₀ = ⟨Z, σ⁻¹(χ̌)⟩` separately
//! pinned (together with the load-bearing coefficient identity
//! `(Z·σ⁻¹(χ̌))₀ = ⟨Z, χ̌⟩` — the negacyclic convolution against the
//! conjugated weight un-reversing to the aligned inner product — which
//! is the step the paper's `(k/d_A)·Tr_H(Z·σ⁻¹(χ̌))` rests on); the
//! reduction of the D-twisted pairing to the paper's literal
//! `(k/d_A)·Tr_H(·)` form is documented as the kernel deviation. The
//! batched reduction realizes each group at `κ = 1`.

use lattice_core::extension::{challenge_fq2, Fq2, EXT_D};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::{RingConfig, RingElement};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    Transcript(TranscriptError),
    /// The input-claim recombination over the head failed (tampered
    /// column partials).
    ColumnRecombinationFailed,
    /// A round identity failed (tampered sumcheck).
    RoundCheckFailed {
        round: usize,
    },
    /// A round message had the wrong shape.
    BadRoundShape {
        round: usize,
        got: usize,
    },
    /// Variable/shape mismatch.
    Shape {
        expected: usize,
        got: usize,
    },
    /// The transparent factor vanished (`A_η(ρ) = 0` / `θ_a = 0`) — the
    /// paper's honest abort (no resampling).
    TransparentFactorZero,
    /// The terminal identity `final = A_η(ρ)·g(ρ)` failed.
    TerminalFailed,
    /// The batched final products mismatched the sumcheck value.
    BatchTerminalFailed,
    /// Ring-side dimension error in the trace layer.
    Ring(String),
}

impl From<TranscriptError> for TensorError {
    fn from(e: TranscriptError) -> Self {
        TensorError::Transcript(e)
    }
}

// ------------------------------------------------------- E-valued helpers #

fn fq(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

fn fq_e(x: u64) -> Fq2 {
    Fq2::new(fq(x), fq(0))
}

/// `eq(r, x) = x·r + (1−x)·(1−r)` for E-valued arguments (1 when
/// `x = r` on the Boolean cube).
fn eq_e(r: &Fq2, x: &Fq2) -> Fq2 {
    x.mul(r).add(&Fq2::ONE.sub(x).mul(&Fq2::ONE.sub(r)))
}

/// `eq(r, x)` for E-valued vectors.
fn eq_e_vec(r: &[Fq2], x: &[Fq2]) -> Result<Fq2, TensorError> {
    if r.len() != x.len() {
        return Err(TensorError::Shape {
            expected: r.len(),
            got: x.len(),
        });
    }
    let mut acc = Fq2::ONE;
    for (a, b) in r.iter().zip(x.iter()) {
        acc = acc.mul(&eq_e(a, b));
    }
    Ok(acc)
}

/// The E-valued inner product `Σ_w A(w)·g(w)` over the Boolean cube
/// from coordinate MLEs (O(2^m) kernel).
fn e_inner(a0: &DenseMle, a1: &DenseMle, g0: &DenseMle, g1: &DenseMle) -> Result<Fq2, TensorError> {
    let n = a0.evaluations.len();
    if a1.evaluations.len() != n || g0.evaluations.len() != n || g1.evaluations.len() != n {
        return Err(TensorError::Shape {
            expected: n,
            got: 0,
        });
    }
    let d = fq(EXT_D);
    let mut c0 = Goldilocks::ZERO;
    let mut c1 = Goldilocks::ZERO;
    for i in 0..n {
        let (a0v, a1v, g0v, g1v) = (
            a0.evaluations[i],
            a1.evaluations[i],
            g0.evaluations[i],
            g1.evaluations[i],
        );
        c0 = c0.add(&a0v.mul(&g0v)).add(&d.mul(&a1v.mul(&g1v)));
        c1 = c1.add(&a0v.mul(&g1v)).add(&a1v.mul(&g0v));
    }
    Ok(Fq2::new(c0, c1))
}

/// Evaluate an F_q MLE at an E-point: `Σ_w eq(point, w)·m(w)` (direct).
fn eval_fq_mle_at_e(m: &DenseMle, point: &[Fq2]) -> Result<Fq2, TensorError> {
    if point.len() != m.num_vars {
        return Err(TensorError::Shape {
            expected: m.num_vars,
            got: point.len(),
        });
    }
    let n = m.evaluations.len();
    let mm = m.num_vars;
    let mut acc = Fq2::ZERO;
    for w in 0..n {
        let mut eqv = Fq2::ONE;
        for (j, p) in point.iter().enumerate() {
            // Variable j is the index's bit (mm−1−j) (DenseMle's
            // fix_variables binds the leading/MSB variable first).
            eqv = eqv.mul(&eq_e(p, &fq_e(((w >> (mm - 1 - j)) & 1) as u64)));
        }
        acc = acc.add(&eqv.mul(&fq_e(m.evaluations[w].to_canonical_u64())));
    }
    Ok(acc)
}

/// Split an MLE into its halves at the leading variable.
fn split_mle(m: &DenseMle) -> Result<(DenseMle, DenseMle), TensorError> {
    if m.num_vars == 0 || m.evaluations.len() % 2 != 0 {
        return Err(TensorError::Shape {
            expected: 2,
            got: m.evaluations.len(),
        });
    }
    let half = m.evaluations.len() / 2;
    Ok((
        DenseMle {
            num_vars: m.num_vars - 1,
            evaluations: m.evaluations[..half].to_vec(),
        },
        DenseMle {
            num_vars: m.num_vars - 1,
            evaluations: m.evaluations[half..].to_vec(),
        },
    ))
}

/// The affine combination `(1−t)·lo + t·hi` of two MLEs (`t` a small
/// integer — the round-variable evaluation points 0, 1, 2; `t = 2`
/// carries the negative `(1−t) = −1` coefficient).
fn lin_comb(lo: &DenseMle, hi: &DenseMle, t: u64) -> Result<DenseMle, TensorError> {
    if lo.evaluations.len() != hi.evaluations.len() {
        return Err(TensorError::Shape {
            expected: lo.evaluations.len(),
            got: hi.evaluations.len(),
        });
    }
    let c_lo = 1i64 - t as i64;
    let c_hi = t as i64;
    // Goldilocks representatives of the (possibly negative) coefficients.
    const GQ: i128 = 0xFFFF_FFFF_0000_0001;
    let w_lo = fq(i128::from(c_lo).rem_euclid(GQ) as u64);
    let w_hi = fq(i128::from(c_hi).rem_euclid(GQ) as u64);
    let vals = lo
        .evaluations
        .iter()
        .zip(hi.evaluations.iter())
        .map(|(l, h)| l.mul(&w_lo).add(&h.mul(&w_hi)))
        .collect();
    Ok(DenseMle {
        num_vars: lo.num_vars,
        evaluations: vals,
    })
}

/// Interpolate a quadratic through the values `(q(0), q(1), q(2))` at
/// `t = r` (E-valued, `r` an F_q point lifted to E).
fn interp_quadratic(q: &[Fq2; 3], r: &Goldilocks) -> Fq2 {
    // Lagrange on nodes 0, 1, 2:
    // L0 = (t²−3t+2)/2, L1 = 2t−t², L2 = (t²−t)/2.
    let t = fq_e(r.to_canonical_u64());
    let half = Fq2::from_base(fq(2).inverse().unwrap_or_else(|| fq(1)));
    let t2 = t.mul(&t);
    let t_2 = t.double();
    let term0 = q[0].mul(&Fq2::ONE.sub(&t_2.add(&t).mul(&half)).add(&t2.mul(&half)));
    let term1 = q[1].mul(&t_2.sub(&t2));
    let term2 = q[2].mul(&t2.sub(&t).mul(&half));
    term0.add(&term1).add(&term2)
}

// ------------------------------------------------------- the packing layer #

/// The packed polynomial's coordinate MLEs `g₀, g₁` over the `m = ℓ−1`
/// tail variables (`g(w) = f(0, w) + f(1, w)·β` at Boolean `w`).
pub fn pack_g(f: &DenseMle) -> Result<(DenseMle, DenseMle), TensorError> {
    if f.num_vars < 1 {
        return Err(TensorError::Shape {
            expected: 1,
            got: f.num_vars,
        });
    }
    let m = f.num_vars - 1;
    let half = 1usize << m;
    let g0: Vec<Goldilocks> = f.evaluations[..half].to_vec();
    let g1: Vec<Goldilocks> = f.evaluations[half..].to_vec();
    Ok((
        DenseMle {
            num_vars: m,
            evaluations: g0,
        },
        DenseMle {
            num_vars: m,
            evaluations: g1,
        },
    ))
}

/// The `A_u` coordinate MLEs: `A_u(w)` = the `u`-th F_q coordinate of
/// `eq(r_tail, w)` at Boolean `w`.
pub fn a_u_mles(r_tail: &[Fq2], m: usize) -> Result<(DenseMle, DenseMle), TensorError> {
    if r_tail.len() != m {
        return Err(TensorError::Shape {
            expected: m,
            got: r_tail.len(),
        });
    }
    let n = 1usize << m;
    let mut a0 = vec![Goldilocks::ZERO; n];
    let mut a1 = vec![Goldilocks::ZERO; n];
    for w in 0..n {
        let mut acc = Fq2::ONE;
        for (j, r) in r_tail.iter().enumerate() {
            // Variable j is the index's bit (m−1−j).
            acc = acc.mul(&eq_e(r, &fq_e(((w >> (m - 1 - j)) & 1) as u64)));
        }
        a0[w] = acc.c0;
        a1[w] = acc.c1;
    }
    Ok((
        DenseMle {
            num_vars: m,
            evaluations: a0,
        },
        DenseMle {
            num_vars: m,
            evaluations: a1,
        },
    ))
}

/// The transparent factor `A_η(ρ)` via the conjugate formula: with
/// `e = eq(ϕ₀(r_tail), ϕ₁(ρ)) = e₀ + e₁·β` in the tensor algebra,
/// `e₀ = (eq(r,ρ) + eq(σ(r),ρ))/2`,
/// `e₁ = (eq(r,ρ) − eq(σ(r),ρ))/(2β)`, and
/// `A_η(ρ) = η₀·e₀ + η₁·e₁` with `(η₀, η₁) = eq(η, ·) = (1−η, η)`.
pub fn transparent_factor(
    r_tail: &[Fq2],
    rho: &[Goldilocks],
    eta: &Fq2,
) -> Result<Fq2, TensorError> {
    if r_tail.len() != rho.len() {
        return Err(TensorError::Shape {
            expected: r_tail.len(),
            got: rho.len(),
        });
    }
    let rho_e: Vec<Fq2> = rho.iter().map(|g| fq_e(g.to_canonical_u64())).collect();
    let eq_rr = eq_e_vec(r_tail, &rho_e)?;
    let r_conj: Vec<Fq2> = r_tail.iter().map(|r| r.conjugate()).collect();
    let eq_sr = eq_e_vec(&r_conj, &rho_e)?;
    // e₀ = (eq + eq_σ)/2; e₁ = (eq − eq_σ)/(2β).
    let inv2 = Fq2::from_base(fq(2).inverse().ok_or(TensorError::TransparentFactorZero)?);
    let e0 = eq_rr.add(&eq_sr).mul(&inv2);
    let diff = eq_rr.sub(&eq_sr);
    // / (2β): multiply by inv2 · β^{-1}; β·β = D ⇒ β^{-1} = β/D.
    let inv_d = fq(EXT_D)
        .inverse()
        .ok_or(TensorError::TransparentFactorZero)?;
    let inv_beta = Fq2::I.mul(&Fq2::from_base(inv_d));
    let e1 = diff.mul(&inv2).mul(&inv_beta);
    let (eta0, eta1) = (Fq2::ONE.sub(eta), *eta);
    Ok(eta0.mul(&e0).add(&eta1.mul(&e1)))
}

// --------------------------------------------- the single-group reduction #

/// The single-group tensor-reduction proof.
#[derive(Clone, Debug)]
pub struct TensorReductionProof {
    /// The column partials `S_y = f(y, r_tail)`, `y ∈ {0, 1}`.
    pub columns: [Fq2; 2],
    /// The E-valued round messages: `m` rounds × `(q(0), q(1), q(2))`.
    pub rounds: Vec<[Fq2; 3]>,
    /// The claimed packed evaluation `g(ρ)` — the OUTPUT claim.
    pub g_claim: Fq2,
}

/// The rows from the columns (the verifier's tensor step):
/// `row_u = Σ_y S_{u,y}·β_y`.
fn rows_from_columns(columns: &[Fq2; 2]) -> [Fq2; 2] {
    [
        Fq2::new(columns[0].c0, columns[1].c0),
        Fq2::new(columns[0].c1, columns[1].c1),
    ]
}

/// The verifier-side derived data shared by prove (locally) and verify.
struct Derived {
    c_eta: Fq2,
    eta: Fq2,
}

fn derive_c_eta(columns: &[Fq2; 2], transcript: &mut Transcript) -> Result<Derived, TensorError> {
    let rows = rows_from_columns(columns);
    let eta = challenge_fq2(transcript, b"a4-tensor-eta")?;
    let (eta0, eta1) = (Fq2::ONE.sub(&eta), eta);
    let c_eta = rows[0].mul(&eta0).add(&rows[1].mul(&eta1));
    Ok(Derived { c_eta, eta })
}

/// One E-valued degree-2 sumcheck round driver shared by prove/verify:
/// factors are coordinate-MLE pairs; round messages are `(q(0), q(1),
/// q(2))` with `q(t) = Σ_w A(t, w)·g(t, w)`.
struct ESumcheck {
    a_factors: Vec<(DenseMle, DenseMle)>,
    g_factors: Vec<(DenseMle, DenseMle)>,
    rounds: Vec<[Fq2; 3]>,
    point: Vec<Goldilocks>,
}

impl ESumcheck {
    fn new(a_factors: Vec<(DenseMle, DenseMle)>, g_factors: Vec<(DenseMle, DenseMle)>) -> Self {
        ESumcheck {
            a_factors,
            g_factors,
            rounds: Vec::new(),
            point: Vec::new(),
        }
    }

    /// Evaluate the summand's round polynomial at `t ∈ {0, 1, 2}`:
    /// `q(t) = Σ_terms ⟨A(t), g(t)⟩` over the remaining cube.
    fn q_at(
        &self,
        t: u64,
        a_lo: &[(DenseMle, DenseMle)],
        a_hi: &[(DenseMle, DenseMle)],
        g_lo: &[(DenseMle, DenseMle)],
        g_hi: &[(DenseMle, DenseMle)],
    ) -> Result<Fq2, TensorError> {
        let mut acc = Fq2::ZERO;
        for (i, (alo, ahi)) in a_lo.iter().zip(a_hi.iter()).enumerate() {
            let (glo, ghi) = (&g_lo[i], &g_hi[i]);
            let a0 = lin_comb(&alo.0, &ahi.0, t)?;
            let a1 = lin_comb(&alo.1, &ahi.1, t)?;
            let g0 = lin_comb(&glo.0, &ghi.0, t)?;
            let g1 = lin_comb(&glo.1, &ghi.1, t)?;
            acc = acc.add(&e_inner(&a0, &a1, &g0, &g1)?);
        }
        Ok(acc)
    }

    fn prove_round(
        &mut self,
        claim: &Fq2,
        round: usize,
        transcript: &mut Transcript,
    ) -> Result<Fq2, TensorError> {
        let a_lo: Vec<(DenseMle, DenseMle)> = self
            .a_factors
            .iter()
            .map(|(a0, a1)| Ok((split_mle(a0)?.0, split_mle(a1)?.0)))
            .collect::<Result<_, TensorError>>()?;
        let a_hi: Vec<(DenseMle, DenseMle)> = self
            .a_factors
            .iter()
            .map(|(a0, a1)| Ok((split_mle(a0)?.1, split_mle(a1)?.1)))
            .collect::<Result<_, TensorError>>()?;
        let g_lo: Vec<(DenseMle, DenseMle)> = self
            .g_factors
            .iter()
            .map(|(g0, g1)| Ok((split_mle(g0)?.0, split_mle(g1)?.0)))
            .collect::<Result<_, TensorError>>()?;
        let g_hi: Vec<(DenseMle, DenseMle)> = self
            .g_factors
            .iter()
            .map(|(g0, g1)| Ok((split_mle(g0)?.1, split_mle(g1)?.1)))
            .collect::<Result<_, TensorError>>()?;
        let q0 = self.q_at(0, &a_lo, &a_hi, &g_lo, &g_hi)?;
        let q1 = self.q_at(1, &a_lo, &a_hi, &g_lo, &g_hi)?;
        let q2 = self.q_at(2, &a_lo, &a_hi, &g_lo, &g_hi)?;
        if q0.add(&q1) != *claim {
            return Err(TensorError::RoundCheckFailed { round });
        }
        let coeffs = [q0, q1, q2];
        let mut bytes = Vec::with_capacity(48);
        for c in &coeffs {
            bytes.extend_from_slice(&c.to_bytes());
        }
        transcript.append_message(b"a4-tensor-round", &bytes)?;
        let r = transcript.challenge_field(b"a4-tensor-chal")?;
        let new_claim = interp_quadratic(&coeffs, &r);
        // Bind all factors at r.
        let pt = [r];
        for (a0, a1) in self.a_factors.iter_mut() {
            *a0 = a0
                .fix_variables(&pt)
                .map_err(|e| TensorError::Ring(format!("{e:?}")))?;
            *a1 = a1
                .fix_variables(&pt)
                .map_err(|e| TensorError::Ring(format!("{e:?}")))?;
        }
        for (g0, g1) in self.g_factors.iter_mut() {
            *g0 = g0
                .fix_variables(&pt)
                .map_err(|e| TensorError::Ring(format!("{e:?}")))?;
            *g1 = g1
                .fix_variables(&pt)
                .map_err(|e| TensorError::Ring(format!("{e:?}")))?;
        }
        self.rounds.push(coeffs);
        self.point.push(r);
        Ok(new_claim)
    }

    fn verify_round(
        &mut self,
        claim: &Fq2,
        round: usize,
        transcript: &mut Transcript,
    ) -> Result<Fq2, TensorError> {
        let coeffs = self.rounds.get(round).ok_or(TensorError::BadRoundShape {
            round,
            got: self.rounds.len(),
        })?;
        if coeffs.len() != 3 {
            return Err(TensorError::BadRoundShape {
                round,
                got: coeffs.len(),
            });
        }
        let mut bytes = Vec::with_capacity(48);
        for c in coeffs {
            bytes.extend_from_slice(&c.to_bytes());
        }
        transcript.append_message(b"a4-tensor-round", &bytes)?;
        let r = transcript.challenge_field(b"a4-tensor-chal")?;
        if coeffs[0].add(&coeffs[1]) != *claim {
            return Err(TensorError::RoundCheckFailed { round });
        }
        self.point.push(r);
        Ok(interp_quadratic(coeffs, &r))
    }
}

/// Prove the tensor reduction: input `f(r_head, r_tail) = v` over the
/// base field; output the packed claim `g(ρ)` with the transparent
/// factor checked nonzero (the honest abort).
pub fn prove_tensor_reduce(
    f: &DenseMle,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    transcript: &mut Transcript,
) -> Result<TensorReductionProof, TensorError> {
    if f.num_vars < 1 || r_tail.len() + 1 != f.num_vars {
        return Err(TensorError::Shape {
            expected: f.num_vars,
            got: r_tail.len() + 1,
        });
    }
    let m = r_tail.len();
    let (g0, g1) = pack_g(f)?;
    // Column partials S_y = f(y, r_tail) (E-point evaluation of the
    // head-fixed halves).
    let half = 1usize << m;
    let mut columns = [Fq2::ZERO, Fq2::ZERO];
    for (y, col) in columns.iter_mut().enumerate() {
        let half_mle = DenseMle {
            num_vars: m,
            evaluations: f.evaluations[y * half..(y + 1) * half].to_vec(),
        };
        *col = eval_fq_mle_at_e(&half_mle, r_tail)?;
    }
    // Fail-closed recombination: v = Σ_y eq(r_head, y)·S_y.
    let w0 = eq_e(r_head, &fq_e(0));
    let w1 = eq_e(r_head, &fq_e(1));
    if columns[0].mul(&w0).add(&columns[1].mul(&w1)) != *v {
        return Err(TensorError::ColumnRecombinationFailed);
    }
    let Derived { c_eta, eta, .. } = derive_c_eta(&columns, transcript)?;
    // A_η's coordinate MLEs: A_η = η₀·A₀ + η₁·A₁ (E-scalar mix of the
    // F_q-valued row MLEs).
    let (a0_mle, a1_mle) = a_u_mles(r_tail, m)?;
    let (eta0, eta1) = (Fq2::ONE.sub(&eta), eta);
    let a_eta0 = DenseMle {
        num_vars: m,
        evaluations: a0_mle
            .evaluations
            .iter()
            .zip(a1_mle.evaluations.iter())
            .map(|(&x, &y)| eta0.c0.mul(&x).add(&eta1.c0.mul(&y)))
            .collect(),
    };
    let a_eta1 = DenseMle {
        num_vars: m,
        evaluations: a0_mle
            .evaluations
            .iter()
            .zip(a1_mle.evaluations.iter())
            .map(|(&x, &y)| eta0.c1.mul(&x).add(&eta1.c1.mul(&y)))
            .collect(),
    };
    let mut sc = ESumcheck::new(vec![(a_eta0, a_eta1)], vec![(g0, g1)]);
    let mut claim = c_eta;
    for round in 0..m {
        claim = sc.prove_round(&claim, round, transcript)?;
    }
    // The local terminal check: A_η(ρ) ≠ 0 and A_η(ρ)·g(ρ) = final.
    let a_eta = transparent_factor(r_tail, &sc.point, &eta)?;
    if a_eta.is_zero() {
        return Err(TensorError::TransparentFactorZero);
    }
    let g_claim = Fq2::new(
        sc.g_factors[0].0.evaluations[0],
        sc.g_factors[0].1.evaluations[0],
    );
    if a_eta.mul(&g_claim) != claim {
        return Err(TensorError::TerminalFailed);
    }
    Ok(TensorReductionProof {
        columns,
        rounds: sc.rounds,
        g_claim,
    })
}

/// Verify the tensor reduction; on success returns the OUTPUT packed
/// claim `g(ρ)` (to be bound by the receiving fold).
pub fn verify_tensor_reduce(
    proof: &TensorReductionProof,
    r_head: &Fq2,
    r_tail: &[Fq2],
    v: &Fq2,
    transcript: &mut Transcript,
) -> Result<Fq2, TensorError> {
    let m = r_tail.len();
    if proof.rounds.len() != m {
        return Err(TensorError::Shape {
            expected: m,
            got: proof.rounds.len(),
        });
    }
    let w0 = eq_e(r_head, &fq_e(0));
    let w1 = eq_e(r_head, &fq_e(1));
    if proof.columns[0].mul(&w0).add(&proof.columns[1].mul(&w1)) != *v {
        return Err(TensorError::ColumnRecombinationFailed);
    }
    let Derived { c_eta, eta, .. } = derive_c_eta(&proof.columns, transcript)?;
    let (a0_mle, a1_mle) = a_u_mles(r_tail, m)?;
    let (eta0, eta1) = (Fq2::ONE.sub(&eta), eta);
    let a_eta0 = DenseMle {
        num_vars: m,
        evaluations: a0_mle
            .evaluations
            .iter()
            .zip(a1_mle.evaluations.iter())
            .map(|(&x, &y)| eta0.c0.mul(&x).add(&eta1.c0.mul(&y)))
            .collect(),
    };
    let a_eta1 = DenseMle {
        num_vars: m,
        evaluations: a0_mle
            .evaluations
            .iter()
            .zip(a1_mle.evaluations.iter())
            .map(|(&x, &y)| eta0.c1.mul(&x).add(&eta1.c1.mul(&y)))
            .collect(),
    };
    let mut sc = ESumcheck::new(vec![(a_eta0, a_eta1)], Vec::new());
    sc.rounds = proof.rounds.clone();
    let mut claim = c_eta;
    for round in 0..m {
        claim = sc.verify_round(&claim, round, transcript)?;
    }
    let a_eta = transparent_factor(r_tail, &sc.point, &eta)?;
    if a_eta.is_zero() {
        return Err(TensorError::TransparentFactorZero);
    }
    if a_eta.mul(&proof.g_claim) != claim {
        return Err(TensorError::TerminalFailed);
    }
    Ok(proof.g_claim)
}

// ------------------------------------------ the batched reduction (Th 3.11) #

/// One batched group: a base-field MLE + its E-point and claim.
pub struct TensorGroup<'a> {
    pub f: &'a DenseMle,
    pub r_head: Fq2,
    pub r_tail: Vec<Fq2>,
    pub v: Fq2,
}

/// The shared batched tensor-reduction proof (Theorem 3.11).
#[derive(Clone, Debug)]
pub struct TensorBatchProof {
    /// Per-group column partials.
    pub columns: Vec<[Fq2; 2]>,
    /// ONE shared sumcheck over `m_max` variables.
    pub rounds: Vec<[Fq2; 3]>,
    /// The final products `h_{a,i} = θ_a·g_{a,i}(ρ_a)` (one per group).
    pub h_claims: Vec<Fq2>,
}

/// Pad an `m`-var MLE to `m_max` vars by tiling evaluations (the
/// cylindrical extension `g̃(w, z) = g(w)`).
fn pad_mle(m: &DenseMle, m_max: usize) -> Result<DenseMle, TensorError> {
    if m.num_vars > m_max {
        return Err(TensorError::Shape {
            expected: m_max,
            got: m.num_vars,
        });
    }
    let reps = 1usize << (m_max - m.num_vars);
    let mut vals = Vec::with_capacity(m.evaluations.len() * reps);
    for &gv in &m.evaluations {
        for _ in 0..reps {
            vals.push(gv);
        }
    }
    Ok(DenseMle {
        num_vars: m_max,
        evaluations: vals,
    })
}

/// The cylindrical `Ã_η,a` coordinate MLEs over `m_max` vars:
/// `A_η,a(w)·eq(0, z)` — cube values `A_η,a(w)` at `z = 0`, zero
/// elsewhere. (Computed on the cube: `A_η,a`'s coordinate MLEs are
/// padded by the z-tiling, then zeroed outside `z = 0`.)
fn cylindrical_a(
    r_tail: &[Fq2],
    eta: &Fq2,
    m_a: usize,
    m_max: usize,
) -> Result<(DenseMle, DenseMle), TensorError> {
    let (a0, a1) = a_u_mles(r_tail, m_a)?;
    let (eta0, eta1) = (Fq2::ONE.sub(eta), *eta);
    // A_η,a coordinate MLEs: the E-scalar product η·A (per-coordinate
    // multiply by the E scalar's coordinates).
    let vals0: Vec<Goldilocks> = a0
        .evaluations
        .iter()
        .zip(a1.evaluations.iter())
        .map(|(&x, &y)| eta0.c0.mul(&x).add(&eta1.c0.mul(&y)))
        .collect();
    let vals1: Vec<Goldilocks> = a0
        .evaluations
        .iter()
        .zip(a1.evaluations.iter())
        .map(|(&x, &y)| eta0.c1.mul(&x).add(&eta1.c1.mul(&y)))
        .collect();
    let ae0 = DenseMle {
        num_vars: m_a,
        evaluations: vals0,
    };
    let ae1 = DenseMle {
        num_vars: m_a,
        evaluations: vals1,
    };
    let p0 = pad_mle(&ae0, m_max)?;
    let p1 = pad_mle(&ae1, m_max)?;
    // eq(0, z): the appended z variables (the trailing/low bits) at 0.
    let reps = 1usize << (m_max - m_a);
    let mut z0 = p0.evaluations.clone();
    let mut z1 = p1.evaluations.clone();
    for i in 0..z0.len() {
        if i % reps != 0 {
            z0[i] = Goldilocks::ZERO;
            z1[i] = Goldilocks::ZERO;
        }
    }
    Ok((
        DenseMle {
            num_vars: m_max,
            evaluations: z0,
        },
        DenseMle {
            num_vars: m_max,
            evaluations: z1,
        },
    ))
}

/// Prove the shared batched tensor reduction over `groups` (one
/// sumcheck; per-group transparent factors; Theorem 3.11's ordering:
/// every opening value and column partial is bound BEFORE the shared
/// row-batching point η and the claim combiners ζ are sampled).
#[allow(clippy::too_many_lines)]
pub fn prove_tensor_batch(
    groups: &[TensorGroup<'_>],
    transcript: &mut Transcript,
) -> Result<TensorBatchProof, TensorError> {
    if groups.is_empty() {
        return Err(TensorError::Shape {
            expected: 1,
            got: 0,
        });
    }
    let m_max = groups.iter().map(|g| g.r_tail.len()).max().unwrap_or(0);
    // Bind columns + input claims (absorb before challenges).
    let mut columns_all: Vec<[Fq2; 2]> = Vec::with_capacity(groups.len());
    let mut g_factors: Vec<(DenseMle, DenseMle)> = Vec::with_capacity(groups.len());
    for g in groups {
        let (g0, g1) = pack_g(g.f)?;
        let half = 1usize << g.r_tail.len();
        let mut cols = [Fq2::ZERO, Fq2::ZERO];
        for (y, col) in cols.iter_mut().enumerate() {
            let half_mle = DenseMle {
                num_vars: g.r_tail.len(),
                evaluations: g.f.evaluations[y * half..(y + 1) * half].to_vec(),
            };
            *col = eval_fq_mle_at_e(&half_mle, &g.r_tail)?;
        }
        let w0 = eq_e(&g.r_head, &fq_e(0));
        let w1 = eq_e(&g.r_head, &fq_e(1));
        if cols[0].mul(&w0).add(&cols[1].mul(&w1)) != g.v {
            return Err(TensorError::ColumnRecombinationFailed);
        }
        for c in &cols {
            transcript.append_message(b"a4-batch-column", &c.to_bytes())?;
        }
        columns_all.push(cols);
        g_factors.push((pad_mle(&g0, m_max)?, pad_mle(&g1, m_max)?));
    }
    // Shared η + ζ combiners (first normalized to 1).
    let eta = challenge_fq2(transcript, b"a4-batch-eta")?;
    let mut zetas = vec![Fq2::ONE];
    for _ in 1..groups.len() {
        zetas.push(challenge_fq2(transcript, b"a4-batch-zeta")?);
    }
    // Per-group c_η,a (ζ-weighted), the Ã factors, and the ζ-scaled g̃'s.
    let mut a_factors: Vec<(DenseMle, DenseMle)> = Vec::with_capacity(groups.len());
    let mut sc_g: Vec<(DenseMle, DenseMle)> = Vec::with_capacity(groups.len());
    let mut claim = Fq2::ZERO;
    for (((g, cols), zeta), (g0, g1)) in groups
        .iter()
        .zip(columns_all.iter())
        .zip(zetas.iter())
        .zip(g_factors.iter())
    {
        let rows = rows_from_columns(cols);
        let (eta0, eta1) = (Fq2::ONE.sub(&eta), eta);
        let c_eta = rows[0].mul(&eta0).add(&rows[1].mul(&eta1));
        claim = claim.add(&c_eta.mul(zeta));
        a_factors.push(cylindrical_a(&g.r_tail, &eta, g.r_tail.len(), m_max)?);
        // ζ·g = (ζ.c0·g0 − D·ζ.c1·g1) + (ζ.c0·g1 + ζ.c1·g0)β.
        let mixed0: Vec<Goldilocks> = g0
            .evaluations
            .iter()
            .zip(g1.evaluations.iter())
            .map(|(&x, &y)| zeta.c0.mul(&x).add(&fq(EXT_D).mul(&zeta.c1.mul(&y))))
            .collect();
        let mixed1: Vec<Goldilocks> = g0
            .evaluations
            .iter()
            .zip(g1.evaluations.iter())
            .map(|(&x, &y)| zeta.c0.mul(&y).add(&zeta.c1.mul(&x)))
            .collect();
        sc_g.push((
            DenseMle {
                num_vars: m_max,
                evaluations: mixed0,
            },
            DenseMle {
                num_vars: m_max,
                evaluations: mixed1,
            },
        ));
    }
    // ONE shared sumcheck: Σ_x Σ_a ζ_a·Ã_a(x)·g̃_a(x) = claim.
    let mut sc = ESumcheck::new(a_factors.clone(), sc_g);
    for round in 0..m_max {
        claim = sc.prove_round(&claim, round, transcript)?;
    }
    // Per-group ρ_a = ρ[:m_a]; θ_a = A_η,a(ρ_a)·Π_{t≥m_a}(1−ρ_t).
    let rho = sc.point.clone();
    let mut h_claims = Vec::with_capacity(groups.len());
    for (g, (g0, g1)) in groups.iter().zip(g_factors.iter()) {
        let m_a = g.r_tail.len();
        let rho_a: Vec<Goldilocks> = rho[..m_a].to_vec();
        let a_eta_a = transparent_factor(&g.r_tail, &rho_a, &eta)?;
        let mut theta = a_eta_a;
        for &rt in &rho[m_a..m_max] {
            theta = theta.mul(&eq_e(&fq_e(rt.to_canonical_u64()), &fq_e(0)));
        }
        if theta.is_zero() {
            return Err(TensorError::TransparentFactorZero);
        }
        // g_a(ρ_a) from the un-padded MLEs (stride extraction: the
        // native variables are the leading variables).
        let reps = 1usize << (m_max - m_a);
        let g0a = DenseMle {
            num_vars: m_a,
            evaluations: g0.evaluations.iter().step_by(reps).copied().collect(),
        };
        let g1a = DenseMle {
            num_vars: m_a,
            evaluations: g1.evaluations.iter().step_by(reps).copied().collect(),
        };
        let gv = Fq2::new(
            g0a.evaluate(&rho_a)
                .map_err(|e| TensorError::Ring(format!("{e:?}")))?,
            g1a.evaluate(&rho_a)
                .map_err(|e| TensorError::Ring(format!("{e:?}")))?,
        );
        h_claims.push(theta.mul(&gv));
    }
    // Terminal: Σ_a ζ_a·h_a = final claim.
    let mut total = Fq2::ZERO;
    for (h, zeta) in h_claims.iter().zip(zetas.iter()) {
        total = total.add(&h.mul(zeta));
    }
    if total != claim {
        return Err(TensorError::BatchTerminalFailed);
    }
    Ok(TensorBatchProof {
        columns: columns_all,
        rounds: sc.rounds,
        h_claims,
    })
}

/// Verify the shared batched reduction; returns the per-group packed
/// claims `g_a(ρ_a)` (recovered as `h_a / θ_a`, with `θ_a` recomputed).
#[allow(clippy::too_many_lines)]
pub fn verify_tensor_batch(
    proof: &TensorBatchProof,
    groups: &[TensorGroup<'_>],
    transcript: &mut Transcript,
) -> Result<Vec<Fq2>, TensorError> {
    if groups.is_empty() || proof.columns.len() != groups.len() {
        return Err(TensorError::Shape {
            expected: groups.len(),
            got: proof.columns.len(),
        });
    }
    let m_max = groups.iter().map(|g| g.r_tail.len()).max().unwrap_or(0);
    if proof.rounds.len() != m_max {
        return Err(TensorError::Shape {
            expected: m_max,
            got: proof.rounds.len(),
        });
    }
    for (g, cols) in groups.iter().zip(proof.columns.iter()) {
        let w0 = eq_e(&g.r_head, &fq_e(0));
        let w1 = eq_e(&g.r_head, &fq_e(1));
        if cols[0].mul(&w0).add(&cols[1].mul(&w1)) != g.v {
            return Err(TensorError::ColumnRecombinationFailed);
        }
        for c in cols {
            transcript.append_message(b"a4-batch-column", &c.to_bytes())?;
        }
    }
    let eta = challenge_fq2(transcript, b"a4-batch-eta")?;
    let mut zetas = vec![Fq2::ONE];
    for _ in 1..groups.len() {
        zetas.push(challenge_fq2(transcript, b"a4-batch-zeta")?);
    }
    let mut claim = Fq2::ZERO;
    let mut a_factors: Vec<(DenseMle, DenseMle)> = Vec::with_capacity(groups.len());
    for ((g, cols), zeta) in groups.iter().zip(proof.columns.iter()).zip(zetas.iter()) {
        let rows = rows_from_columns(cols);
        let (eta0, eta1) = (Fq2::ONE.sub(&eta), eta);
        let c_eta = rows[0].mul(&eta0).add(&rows[1].mul(&eta1));
        claim = claim.add(&c_eta.mul(zeta));
        a_factors.push(cylindrical_a(&g.r_tail, &eta, g.r_tail.len(), m_max)?);
    }
    let mut sc = ESumcheck::new(a_factors, Vec::new());
    sc.rounds = proof.rounds.clone();
    for round in 0..m_max {
        claim = sc.verify_round(&claim, round, transcript)?;
    }
    // θ_a recomputation + the final products check.
    let rho = sc.point.clone();
    let mut out = Vec::with_capacity(groups.len());
    let mut total = Fq2::ZERO;
    for ((g, (h, zeta)), _) in groups
        .iter()
        .zip(proof.h_claims.iter().zip(zetas.iter()))
        .zip(std::iter::repeat(()))
    {
        let m_a = g.r_tail.len();
        let rho_a: Vec<Goldilocks> = rho[..m_a].to_vec();
        let a_eta_a = transparent_factor(&g.r_tail, &rho_a, &eta)?;
        let mut theta = a_eta_a;
        for &rt in &rho[m_a..m_max] {
            theta = theta.mul(&eq_e(&fq_e(rt.to_canonical_u64()), &fq_e(0)));
        }
        if theta.is_zero() {
            return Err(TensorError::TransparentFactorZero);
        }
        // The per-group packed claim is h/θ (bound downstream); the
        // verifier recovers it via the inverse (θ ≠ 0 checked).
        let g_a = h.mul(&theta.inverse().ok_or(TensorError::TransparentFactorZero)?);
        out.push(g_a);
        total = total.add(&h.mul(zeta));
    }
    if total != claim {
        return Err(TensorError::BatchTerminalFailed);
    }
    Ok(out)
}

// ------------------------------------- F_{Q32^2}: the trace layer's field #

/// The quadratic extension `F_{Q32²} = F_Q32[β]` with `β² = 5` (5 is
/// the smallest nonsquare mod Q_32) — the SAME-modulus field for the
/// ring `R_{Q32}` the trace layer packs into (the paper's regime:
/// ring and extension field share `q`; using Goldilocks here instead
/// would mix two incompatible moduli — see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fq2Q {
    pub c0: u64,
    pub c1: u64,
}

/// The extension modulus (Q_32) and the nonsquare.
const Q32: u64 = 3_221_225_473;
/// `β² = 5` — the smallest nonsquare mod Q_32.
const D_Q32: u64 = 5;

impl Fq2Q {
    pub const ZERO: Fq2Q = Fq2Q { c0: 0, c1: 0 };
    pub const ONE: Fq2Q = Fq2Q { c0: 1, c1: 0 };
    pub const BETA: Fq2Q = Fq2Q { c0: 0, c1: 1 };

    pub fn new(c0: u64, c1: u64) -> Self {
        Fq2Q {
            c0: c0 % Q32,
            c1: c1 % Q32,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.c0 == 0 && self.c1 == 0
    }

    pub fn add(&self, o: &Fq2Q) -> Fq2Q {
        Fq2Q {
            c0: (self.c0 + o.c0) % Q32,
            c1: (self.c1 + o.c1) % Q32,
        }
    }

    pub fn sub(&self, o: &Fq2Q) -> Fq2Q {
        Fq2Q {
            c0: (self.c0 + Q32 - o.c0) % Q32,
            c1: (self.c1 + Q32 - o.c1) % Q32,
        }
    }

    pub fn mul(&self, o: &Fq2Q) -> Fq2Q {
        Fq2Q {
            c0: (self.c0 * o.c0 % Q32 + D_Q32 * (self.c1 * o.c1 % Q32) % Q32) % Q32,
            c1: (self.c0 * o.c1 % Q32 + self.c1 * o.c0 % Q32) % Q32,
        }
    }

    /// The Frobenius conjugate `σ(c0 + c1β) = c0 − c1β`.
    pub fn conjugate(&self) -> Fq2Q {
        Fq2Q {
            c0: self.c0,
            c1: (Q32 - self.c1) % Q32,
        }
    }

    pub fn to_bytes(&self) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[..4].copy_from_slice(&(self.c0 as u32).to_be_bytes());
        out[4..].copy_from_slice(&(self.c1 as u32).to_be_bytes());
        out
    }
}

/// `eq(a, b) = b·a + (1−b)(1−a)` over `F_{Q32²}`.
fn eq_q(a: &Fq2Q, b: &Fq2Q) -> Fq2Q {
    b.mul(a).add(&Fq2Q::ONE.sub(b).mul(&Fq2Q::ONE.sub(a)))
}

/// Sample one `F_{Q32²}` element from the transcript.
pub fn challenge_fq2q(transcript: &mut Transcript, label: &[u8]) -> Result<Fq2Q, TensorError> {
    let bytes = transcript.challenge_bytes(label, 8)?;
    let c0 = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
    let c1 = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u64;
    Ok(Fq2Q::new(c0 % Q32, c1 % Q32))
}

// --------------------------------------------- the trace functional (§7.1) #

/// The trace-packing map `ψ: E^{d_A/k} → R_{q,d_A}` — interleaved
/// coordinates (`ψ(y)_j = y_{j/2}.c_{j%2}`), `d_A = 2·|y|`, with `E =
/// F_{Q32²}` (the same-modulus field).
pub fn psi(ring: &RingConfig, elems: &[Fq2Q]) -> Result<RingElement, TensorError> {
    let d = ring.n();
    if d != 2 * elems.len() {
        return Err(TensorError::Shape {
            expected: 2 * elems.len(),
            got: d,
        });
    }
    let mut coeffs = vec![0u32; d];
    for (u, e) in elems.iter().enumerate() {
        coeffs[2 * u] = e.c0 as u32;
        coeffs[2 * u + 1] = e.c1 as u32;
    }
    Ok(RingElement::from_coeffs(ring, coeffs))
}

/// The conjugation `σ⁻¹ = σ` (`X ↦ X^{−1}`, an involution under
/// `X^d ≡ −1`): `σ(z)₀ = z₀`, `σ(z)_j = −z_{d−j}` for `j ≥ 1`.
pub fn sigma_inv(ring: &RingConfig, z: &RingElement) -> Result<RingElement, TensorError> {
    let d = ring.n();
    let q = u64::from(ring.modulus.q);
    let src = z.coeffs();
    let mut out = vec![0u32; d];
    out[0] = src[0];
    for j in 1..d {
        // −z_{d−j} mod q.
        let v = (q - (u64::from(src[d - j])) % q) % q;
        out[j] = v as u32;
    }
    Ok(RingElement::from_coeffs(ring, out))
}

/// The public linear functional `T_{ρ_pack}` (Eq 133) — realized as the
/// E-bilinear pairing against the packed weight `χ̌ = ψ(eq(ρ̃_pack, ·))`:
/// `T(Z) = Σ_u (χ_{2u}Z_{2u} + D·χ_{2u+1}Z_{2u+1})
///          + (χ_{2u}Z_{2u+1} + χ_{2u+1}Z_{2u})·β`,
/// pinned by the packing identity `T(ψ(y)) = Σ_u eq(ρ̃,u)·y_u`
/// (test-pinned) — the E-bilinear product against the packed weight IS
/// the `(k/d_A)·Tr_H(Z·σ⁻¹(χ̌))` value, with the σ⁻¹-conjugation route
/// `(Z·σ⁻¹(χ̌))₀ = ⟨Z, χ̌⟩` separately pinned (the un-reversing step).
pub fn trace_functional(
    ring: &RingConfig,
    z: &RingElement,
    chi: &RingElement,
) -> Result<Fq2Q, TensorError> {
    let d = ring.n();
    if z.coeffs().len() != d || chi.coeffs().len() != d || d % 2 != 0 {
        return Err(TensorError::Shape {
            expected: d,
            got: z.coeffs().len(),
        });
    }
    let mut c0: u64 = 0;
    let mut c1: u64 = 0;
    for u in 0..d / 2 {
        let (z0, z1) = (u64::from(z.coeff(2 * u)), u64::from(z.coeff(2 * u + 1)));
        let (x0, x1) = (u64::from(chi.coeff(2 * u)), u64::from(chi.coeff(2 * u + 1)));
        // (x0 + x1β)(z0 + z1β) with β² = D.
        c0 = (c0 + x0 * z0 % Q32 + D_Q32 * (x1 * z1 % Q32) % Q32) % Q32;
        c1 = (c1 + x0 * z1 % Q32 + x1 * z0 % Q32) % Q32;
    }
    Ok(Fq2Q {
        c0: c0 % Q32,
        c1: c1 % Q32,
    })
}

/// Build the packed weight `χ̌_pack = ψ(eq(ρ̃_pack, u)_{u})` from the
/// trace-packing point over `F_{Q32²}`.
pub fn chi_pack(ring: &RingConfig, rho_pack: &[Fq2Q]) -> Result<RingElement, TensorError> {
    let n_pairs = ring.n() / 2;
    let mut elems = Vec::with_capacity(n_pairs);
    for u in 0..n_pairs {
        // eq(ρ̃_pack, u) with u the index bits.
        let mut acc = Fq2Q::ONE;
        for (j, rho_j) in rho_pack.iter().enumerate() {
            let bit = if (u >> j) & 1 == 1 {
                Fq2Q::ONE
            } else {
                Fq2Q::ZERO
            };
            acc = acc.mul(&eq_q(rho_j, &bit));
        }
        elems.push(acc);
    }
    psi(ring, &elems)
}

/// The evaluation-trace row check (Eq 134):
/// `Σ_i χ_blk(i)·T_{ρ_pack}(e_i) = v̄` over the revealed partials `e_i`.
pub fn opening_row_trace(
    ring: &RingConfig,
    partials: &[RingElement],
    rho_pack: &[Fq2Q],
    chi_blk: &[Fq2Q],
    v_bar: &Fq2Q,
) -> Result<bool, TensorError> {
    if partials.len() != chi_blk.len() {
        return Err(TensorError::Shape {
            expected: partials.len(),
            got: chi_blk.len(),
        });
    }
    let chi = chi_pack(ring, rho_pack)?;
    let mut acc = Fq2Q::ZERO;
    for (e, w) in partials.iter().zip(chi_blk.iter()) {
        acc = acc.add(&w.mul(&trace_functional(ring, e, &chi)?));
    }
    Ok(acc == *v_bar)
}

/// The digit weights `ω_Tr(i, ℓ, ν) = χ_blk(i)·b^ℓ·T_{ρ_pack}(X^ν)`
/// (Eq 135) — the linear functional's per-opening-cell weights.
pub fn trace_row_weights(
    ring: &RingConfig,
    rho_pack: &[Fq2Q],
    chi_blk: &[Fq2Q],
    opening_base: u64,
    digit_depth: usize,
) -> Result<Vec<Vec<Vec<Fq2Q>>>, TensorError> {
    let chi = chi_pack(ring, rho_pack)?;
    let d = ring.n();
    let mut out = Vec::with_capacity(chi_blk.len());
    for w in chi_blk {
        let mut per_l = Vec::with_capacity(digit_depth);
        for l in 0..digit_depth {
            let bl = Fq2Q::new(opening_base.pow(l as u32) % Q32, 0);
            let mut per_nu = Vec::with_capacity(d);
            for nu in 0..d {
                let mut coeffs = vec![0u32; d];
                coeffs[nu] = 1;
                let xnu = RingElement::from_coeffs(ring, coeffs);
                per_nu.push(w.mul(&bl).mul(&trace_functional(ring, &xnu, &chi)?));
            }
            per_l.push(per_nu);
        }
        out.push(per_l);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ring() -> RingConfig {
        RingConfig::new(lattice_ring::Modulus32::Q_32, 4).unwrap()
    }

    #[test]
    fn eq_e_basics() {
        // eq(r, x) = 1 iff x = r on the Boolean cube.
        let a = Fq2::new(fq(5), fq(9));
        assert_eq!(eq_e(&a, &fq_e(0)), Fq2::ONE.sub(&a));
        assert_eq!(eq_e(&a, &fq_e(1)), a);
        assert_eq!(eq_e(&fq_e(0), &fq_e(0)), Fq2::ONE);
        assert_eq!(eq_e(&fq_e(1), &fq_e(0)), Fq2::ZERO);
    }

    #[test]
    fn rows_from_columns_is_the_tensor_step() {
        let s = [Fq2::new(fq(3), fq(4)), Fq2::new(fq(5), fq(6))];
        let rows = rows_from_columns(&s);
        assert_eq!(rows[0], Fq2::new(fq(3), fq(5)));
        assert_eq!(rows[1], Fq2::new(fq(4), fq(6)));
        // The insecure shortcut Σ_y β_y·S_y = S₀ + S₁ is a different
        // object than the row recombination.
        let shortcut = s[0].add(&s[1]);
        assert_ne!(rows[0].add(&rows[1].mul(&Fq2::I)), shortcut);
    }

    #[test]
    fn a_u_mles_agree_with_conjugate_formula() {
        // The transparent-factor route: A_η(ρ) via the conjugate formula
        // equals the row-MLE combination Σ_u η_u·Ā_u(ρ) — the multilinear
        // identity that makes the verifier's computation correct.
        let m = 3;
        let r_tail: Vec<Fq2> = (0..m)
            .map(|i| Fq2::new(fq(0x1111 * (i + 1)), fq(0x2222 * (i + 1))))
            .collect();
        let (a0, a1) = a_u_mles(&r_tail, m as usize).unwrap();
        let rho: Vec<Goldilocks> = (0..m).map(|i| fq(0x3333 + i)).collect();
        let eta = Fq2::new(fq(7), fq(11));
        // MLE route: η₀·A₀(ρ) + η₁·A₁(ρ).
        let mle_val = Fq2::ONE
            .sub(&eta)
            .mul(&fq_e(a0.evaluate(&rho).unwrap().to_canonical_u64()))
            .add(&eta.mul(&fq_e(a1.evaluate(&rho).unwrap().to_canonical_u64())));
        let conj_val = transparent_factor(&r_tail, &rho, &eta).unwrap();
        assert_eq!(mle_val, conj_val);
    }

    #[test]
    fn tensor_reduce_roundtrip() {
        let mut t = Transcript::new_default(b"a4-test");
        let f = DenseMle::random(5, b"a4-f");
        let r_head = Fq2::new(fq(0x51d2), fq(0x3fa1));
        let r_tail: Vec<Fq2> = (0..4)
            .map(|i| Fq2::new(fq(0x1234 + i as u64), fq(0x5678 + i as u64)))
            .collect();
        // The true claim v = f(r_head, r_tail) over E.
        let half = 1usize << 4;
        let mut v = Fq2::ZERO;
        for y in 0..2 {
            let half_mle = DenseMle {
                num_vars: 4,
                evaluations: f.evaluations[y * half..(y + 1) * half].to_vec(),
            };
            v = v.add(
                &eq_e(&r_head, &fq_e(y as u64)).mul(&eval_fq_mle_at_e(&half_mle, &r_tail).unwrap()),
            );
        }
        let proof = prove_tensor_reduce(&f, &r_head, &r_tail, &v, &mut t).unwrap();
        let mut tv = Transcript::new_default(b"a4-test");
        let out = verify_tensor_reduce(&proof, &r_head, &r_tail, &v, &mut tv).unwrap();
        // The output claim equals the true packed g(ρ): recompute from
        // the challenges the verifier derived (the point is recoverable
        // by replaying).
        assert_eq!(out, proof.g_claim);
        // The packed claim is genuine: g(ρ) = Σ_y f(y, ρ)·β_y.
        let (g0, g1) = pack_g(&f).unwrap();
        let rho = tv_replay_point(&proof, &r_head, &r_tail, &v);
        let true_g = Fq2::new(g0.evaluate(&rho).unwrap(), g1.evaluate(&rho).unwrap());
        assert_eq!(out, true_g);
    }

    fn tv_replay_point(
        proof: &TensorReductionProof,
        r_head: &Fq2,
        r_tail: &[Fq2],
        v: &Fq2,
    ) -> Vec<Goldilocks> {
        // Replicate the verifier's exact challenge draws (do NOT run
        // verify first — that would consume the draws).
        let _ = (r_head, r_tail, v);
        let mut tv = Transcript::new_default(b"a4-test");
        let _ = challenge_fq2(&mut tv, b"a4-tensor-eta");
        let mut point = Vec::with_capacity(r_tail.len());
        for coeffs in &proof.rounds {
            let mut bytes = Vec::with_capacity(48);
            for c in coeffs {
                bytes.extend_from_slice(&c.to_bytes());
            }
            tv.append_message(b"a4-tensor-round", &bytes).unwrap();
            point.push(tv.challenge_field(b"a4-tensor-chal").unwrap());
        }
        point
    }

    #[test]
    fn tensor_reduce_tampered_column_rejected() {
        let mut t = Transcript::new_default(b"a4-tamper");
        let f = DenseMle::random(4, b"a4-tf");
        let r_head = Fq2::new(fq(0x11), fq(0x22));
        let r_tail: Vec<Fq2> = (0..3u64)
            .map(|i| Fq2::new(fq(0x33 + i), fq(0x44 + i)))
            .collect();
        let half = 1usize << 3;
        let mut v = Fq2::ZERO;
        for y in 0..2 {
            let half_mle = DenseMle {
                num_vars: 3,
                evaluations: f.evaluations[y * half..(y + 1) * half].to_vec(),
            };
            v = v.add(
                &eq_e(&r_head, &fq_e(y as u64)).mul(&eval_fq_mle_at_e(&half_mle, &r_tail).unwrap()),
            );
        }
        let mut proof = prove_tensor_reduce(&f, &r_head, &r_tail, &v, &mut t).unwrap();
        proof.columns[0] = proof.columns[0].add(&Fq2::ONE);
        let mut tv = Transcript::new_default(b"a4-tamper");
        assert!(matches!(
            verify_tensor_reduce(&proof, &r_head, &r_tail, &v, &mut tv),
            Err(TensorError::ColumnRecombinationFailed)
        ));
    }

    #[test]
    fn tensor_reduce_tampered_round_rejected() {
        let mut t = Transcript::new_default(b"a4-round");
        let f = DenseMle::random(4, b"a4-rf");
        let r_head = Fq2::new(fq(0x55), fq(0x66));
        let r_tail: Vec<Fq2> = (0..3u64)
            .map(|i| Fq2::new(fq(0x77 + i), fq(0x88 + i)))
            .collect();
        let half = 1usize << 3;
        let mut v = Fq2::ZERO;
        for y in 0..2 {
            let half_mle = DenseMle {
                num_vars: 3,
                evaluations: f.evaluations[y * half..(y + 1) * half].to_vec(),
            };
            v = v.add(
                &eq_e(&r_head, &fq_e(y as u64)).mul(&eval_fq_mle_at_e(&half_mle, &r_tail).unwrap()),
            );
        }
        let mut proof = prove_tensor_reduce(&f, &r_head, &r_tail, &v, &mut t).unwrap();
        proof.rounds[0][0] = proof.rounds[0][0].add(&Fq2::ONE);
        let mut tv = Transcript::new_default(b"a4-round");
        assert!(matches!(
            verify_tensor_reduce(&proof, &r_head, &r_tail, &v, &mut tv),
            Err(TensorError::RoundCheckFailed { round: 0 })
        ));
    }

    #[test]
    fn tensor_reduce_wrong_g_claim_rejected() {
        let mut t = Transcript::new_default(b"a4-gclaim");
        let f = DenseMle::random(4, b"a4-gf");
        let r_head = Fq2::new(fq(0x99), fq(0xaa));
        let r_tail: Vec<Fq2> = (0..3)
            .map(|i| Fq2::new(fq(0xbb + i as u64), fq(0xcc + i as u64)))
            .collect();
        let half = 1usize << 3;
        let mut v = Fq2::ZERO;
        for y in 0..2 {
            let half_mle = DenseMle {
                num_vars: 3,
                evaluations: f.evaluations[y * half..(y + 1) * half].to_vec(),
            };
            v = v.add(
                &eq_e(&r_head, &fq_e(y as u64)).mul(&eval_fq_mle_at_e(&half_mle, &r_tail).unwrap()),
            );
        }
        let mut proof = prove_tensor_reduce(&f, &r_head, &r_tail, &v, &mut t).unwrap();
        proof.g_claim = proof.g_claim.add(&Fq2::ONE);
        let mut tv = Transcript::new_default(b"a4-gclaim");
        assert!(matches!(
            verify_tensor_reduce(&proof, &r_head, &r_tail, &v, &mut tv),
            Err(TensorError::TerminalFailed)
        ));
    }

    #[test]
    fn tensor_batch_roundtrip_and_tamper() {
        let f1 = DenseMle::random(4, b"a4-b1");
        let f2 = DenseMle::random(5, b"a4-b2");
        let r1h = Fq2::new(fq(0x21), fq(0x43));
        let r2h = Fq2::new(fq(0x65), fq(0x87));
        let r1: Vec<Fq2> = (0..3)
            .map(|i| Fq2::new(fq(0x100 + i as u64), fq(0x200 + i as u64)))
            .collect();
        let r2: Vec<Fq2> = (0..4u64)
            .map(|i| Fq2::new(fq(0x300 + i), fq(0x400 + i)))
            .collect();
        let eval_v = |f: &DenseMle, rh: &Fq2, rt: &[Fq2]| {
            let half = 1usize << rt.len();
            let mut v = Fq2::ZERO;
            for y in 0..2 {
                let half_mle = DenseMle {
                    num_vars: rt.len(),
                    evaluations: f.evaluations[y * half..(y + 1) * half].to_vec(),
                };
                v = v
                    .add(&eq_e(rh, &fq_e(y as u64)).mul(&eval_fq_mle_at_e(&half_mle, rt).unwrap()));
            }
            v
        };
        let v1 = eval_v(&f1, &r1h, &r1);
        let v2 = eval_v(&f2, &r2h, &r2);
        let groups = [
            TensorGroup {
                f: &f1,
                r_head: r1h,
                r_tail: r1.clone(),
                v: v1,
            },
            TensorGroup {
                f: &f2,
                r_head: r2h,
                r_tail: r2.clone(),
                v: v2,
            },
        ];
        let mut t = Transcript::new_default(b"a4-batch");
        let proof = prove_tensor_batch(&groups, &mut t).unwrap();
        let mut tv = Transcript::new_default(b"a4-batch");
        let outs = verify_tensor_batch(&proof, &groups, &mut tv).unwrap();
        assert_eq!(outs.len(), 2);
        // Tampered h-claim → the batched terminal fails.
        let mut bad = proof.clone();
        bad.h_claims[1] = bad.h_claims[1].add(&Fq2::ONE);
        let mut tv2 = Transcript::new_default(b"a4-batch");
        assert!(matches!(
            verify_tensor_batch(&bad, &groups, &mut tv2),
            Err(TensorError::BatchTerminalFailed)
        ));
    }

    // ----------------------------------------------------- trace layer #

    #[test]
    fn sigma_inv_is_an_involution() {
        let ring = test_ring();
        let z = RingElement::from_signed(&ring, &[3, -5, 7, -9]);
        let s = sigma_inv(&ring, &z).unwrap();
        let s2 = sigma_inv(&ring, &s).unwrap();
        assert_eq!(z, s2);
    }

    #[test]
    fn sigma_coefficient_identity() {
        // (Z·σ⁻¹(χ̌))₀ = ⟨Z, χ̌⟩ — the negacyclic convolution against the
        // CONJUGATED weight un-reverses to the aligned inner product
        // against the RAW weight (the load-bearing step the paper's
        // `(k/d_A)·Tr_H(Z·σ⁻¹(χ̌))` rests on; derived in the module docs).
        let ring = test_ring();
        let z = RingElement::from_signed(&ring, &[3, -5, 7, -9]);
        let chi = RingElement::from_signed(&ring, &[1, 2, -4, 6]);
        let prod = z.mul(&sigma_inv(&ring, &chi).unwrap()).unwrap();
        let q = u64::from(ring.modulus.q);
        let zc: Vec<i64> = z.coeffs().iter().map(|&c| c as i64).collect();
        let chc: Vec<i64> = chi.coeffs().iter().map(|&c| c as i64).collect();
        let mut inner: i128 = 0;
        for j in 0..4 {
            inner += i128::from(zc[j]) * i128::from(chc[j]);
        }
        let inner_mod = (inner.rem_euclid(q as i128)) as u64;
        assert_eq!(u64::from(prod.coeff(0)), inner_mod % q);
    }

    #[test]
    fn packing_identity_pins_the_trace_functional() {
        // T(ψ(y)) = Σ_u eq(ρ̃_pack, u)·y_u — the pinned identity over
        // the SAME-modulus field F_{Q32²}.
        let ring = test_ring();
        let rho_pack: Vec<Fq2Q> = vec![Fq2Q::new(0x61, 0x72), Fq2Q::new(0x62, 0x73)];
        let chi = chi_pack(&ring, &rho_pack).unwrap();
        let y: Vec<Fq2Q> = (0..8).map(|i| Fq2Q::new(0x100 + i, 0x200 + i)).collect();
        let z = psi(&ring, &y).unwrap();
        let t = trace_functional(&ring, &z, &chi).unwrap();
        let mut expect = Fq2Q::ZERO;
        for (u, yu) in y.iter().enumerate() {
            let mut acc = Fq2Q::ONE;
            for (j, rho_j) in rho_pack.iter().enumerate() {
                let bit = if (u >> j) & 1 == 1 {
                    Fq2Q::ONE
                } else {
                    Fq2Q::ZERO
                };
                acc = acc.mul(&eq_q(rho_j, &bit));
            }
            expect = expect.add(&acc.mul(yu));
        }
        assert_eq!(t, expect);
    }

    /// Modular inverse mod Q32 (for the linearity check below).
    fn inv_mod_q32(a: u64) -> u64 {
        let mut result = 1u64;
        let mut base = a % Q32;
        let mut exp = Q32 - 2;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base % Q32;
            }
            base = base * base % Q32;
            exp >>= 1;
        }
        result
    }

    #[test]
    fn opening_row_trace_check() {
        // Eq 134: Σ_i χ_blk(i)·T(e_i) = v̄ with honest partials.
        let ring = test_ring();
        let rho_pack: Vec<Fq2Q> = vec![Fq2Q::new(0x81, 0x92), Fq2Q::new(0x83, 0x94)];
        let ys: Vec<Vec<Fq2Q>> = (0..3)
            .map(|i| {
                (0..8)
                    .map(|j| Fq2Q::new(0x300 + 10 * i + j, 0x400 + j))
                    .collect()
            })
            .collect();
        let chi_blk: Vec<Fq2Q> = vec![Fq2Q::new(1, 2), Fq2Q::new(3, 4), Fq2Q::new(5, 6)];
        let partials: Vec<RingElement> = ys.iter().map(|y| psi(&ring, y).unwrap()).collect();
        let mut v_bar = Fq2Q::ZERO;
        for (y, w) in ys.iter().zip(chi_blk.iter()) {
            let mut acc = Fq2Q::ZERO;
            for (u, yu) in y.iter().enumerate() {
                let mut eqv = Fq2Q::ONE;
                for (j, rho_j) in rho_pack.iter().enumerate() {
                    let bit = if (u >> j) & 1 == 1 {
                        Fq2Q::ONE
                    } else {
                        Fq2Q::ZERO
                    };
                    eqv = eqv.mul(&eq_q(rho_j, &bit));
                }
                acc = acc.add(&eqv.mul(yu));
            }
            v_bar = v_bar.add(&w.mul(&acc));
        }
        assert!(opening_row_trace(&ring, &partials, &rho_pack, &chi_blk, &v_bar).unwrap());
        // Tampered partial → rejected.
        let mut bad = partials.clone();
        bad[1] = bad[1].add(&ring.constant(1)).unwrap();
        assert!(!opening_row_trace(&ring, &bad, &rho_pack, &chi_blk, &v_bar).unwrap());
        // The trace weights (Eq 135) exist and are linear in (i, ℓ, ν).
        let weights = trace_row_weights(&ring, &rho_pack, &chi_blk, 16, 2).unwrap();
        assert_eq!(weights.len(), 3);
        assert_eq!(weights[0].len(), 2);
        assert_eq!(weights[0][0].len(), 16); // d_A = ring dimension.
                                             // ω(i, ℓ, ν) = χ_blk(i)·b^ℓ·T(X^ν): the weight at ℓ=1 is b× the
                                             // weight at ℓ=0 for the same (i, ν) (mod Q32 division by 16).
        let inv16 = Fq2Q::new(inv_mod_q32(16), 0);
        for (w1, w0) in weights[0][1].iter().zip(weights[0][0].iter()).take(8) {
            assert_eq!(w1.mul(&inv16), *w0);
        }
    }

    #[test]
    fn fq2q_field_axioms() {
        // β² = 5, conjugation, and the norm/trace sanity.
        let b = Fq2Q::BETA;
        assert_eq!(b.mul(&b), Fq2Q::new(5, 0));
        let x = Fq2Q::new(0x1234, 0x5678);
        let y = Fq2Q::new(0x9abc, 0xdef0);
        assert_eq!(x.mul(&y), y.mul(&x));
        assert!(x.add(&x.sub(&x)).add(&Fq2Q::ZERO) == x.add(&Fq2Q::ZERO));
        // σ(x)·x = the norm (in F_Q32).
        assert_eq!(x.conjugate().mul(&x).c1, 0);
        // eq_q matches the eq semantics: 1 iff equal (Boolean args).
        assert_eq!(eq_q(&x, &Fq2Q::ZERO), Fq2Q::ONE.sub(&x));
        assert_eq!(eq_q(&x, &Fq2Q::ONE), x);
    }
}

#[cfg(test)]
mod interp_probe {
    use super::*;

    #[test]
    fn probe_interp_quadratic() {
        let c0 = Fq2::new(fq(3), fq(5));
        let c1 = Fq2::new(fq(7), fq(11));
        let c2 = Fq2::new(fq(13), fq(17));
        let q = |t: &Fq2| -> Fq2 {
            let t2 = t.mul(t);
            c0.add(&c1.mul(t)).add(&c2.mul(&t2))
        };
        let vals = [q(&Fq2::ZERO), q(&Fq2::ONE), q(&Fq2::new(fq(2), fq(0)))];
        let r = fq(0x51d2ea7f);
        let t = fq_e(r.to_canonical_u64());
        let interp = interp_quadratic(&vals, &r);
        let direct = q(&t);
        assert_eq!(interp, direct, "interp mismatch");
    }

    #[test]
    fn probe_round_chain() {
        // A tiny 2-round ESumcheck over random factors: verify the
        // round identity chain manually.
        let m = 2usize;
        let a0 = DenseMle::random(m, b"pr-a0");
        let a1 = DenseMle::random(m, b"pr-a1");
        let g0 = DenseMle::random(m, b"pr-g0");
        let g1 = DenseMle::random(m, b"pr-g1");
        let true_sum = e_inner(&a0, &a1, &g0, &g1).unwrap();
        let mut sc = ESumcheck::new(vec![(a0, a1)], vec![(g0, g1)]);
        let mut t = Transcript::new_default(b"pr");
        let mut claim = true_sum;
        for round in 0..m {
            claim = sc.prove_round(&claim, round, &mut t).unwrap();
        }
        // The final claim must equal A(ρ)·g(ρ) at the accumulated point
        // (all factors are fully bound: single surviving evaluation).
        let a_val = Fq2::new(
            sc.a_factors[0].0.evaluations[0],
            sc.a_factors[0].1.evaluations[0],
        );
        let g_val = Fq2::new(
            sc.g_factors[0].0.evaluations[0],
            sc.g_factors[0].1.evaluations[0],
        );
        let prod = a_val.mul(&g_val);
        assert_eq!(claim, prod, "final claim mismatch");
    }
}
