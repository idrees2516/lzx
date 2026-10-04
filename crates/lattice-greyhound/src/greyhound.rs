//! The Greyhound polynomial commitment scheme (§4, Figure 4): Setup / Commit /
//! Open / Eval over `Z_q[X]` with degree bound `N = m·r·d`, composed with the
//! LaBRADOR engine as the Π¹ sub-proof (§4.3 — the R1 relation compiled into
//! the principal relation).
//!
//! Structure (following the reference's `greyhound.c`, which is the paper's
//! Figure 4 restructured for the Fiat-Shamir composition):
//!
//! * **Commit**: the polynomial's `len = N/d` ring elements are viewed as an
//!   `n × m` matrix (`s[i·m + j]`, the witness digit-decomposed per row);
//!   inner commitments `t_i = A·sx_i` (rank κ), digit-decomposed (fu × bu);
//!   one outer commitment `u1 = B·t̃` (rank κ1).
//! * **Eval** at a scalar `x ∈ Z_q`: `w_i = Σ_j x^{dj}·s[i·m + j]` (the row
//!   folded with powers of `x^d`), `ŵ = G^{-1}(w)` (fu × bu digits per row),
//!   the "v commitment" `u2 = D·ŵ` (the paper's first message), then the
//!   amortized opening `z = Σ_i c_i·sx_i` over the `n` blocks with challenges
//!   from the transcript. The witness `[z^(0..f-1), t̃, ŵ]` and the 5-constraint
//!   statement go to the LaBRADOR engine.
//! * The 5 constraints (§4.3's 3n+2 at n=1, κ ≠ κ1 allowed — the reference's
//!   `polcom_reduce`):
//!   1. `B·t̃ = u1` (κ1 equations),
//!   2. `D·ŵ = u2` (κ1),
//!   3. `Σ_i c_i w_i = ⟨a, z⟩` (1 — the quadratic-relations check: a = the
//!      x-power matrix),
//!   4. `A·z = Σ_i c_i t_i` (κ),
//!   5. `⟨ŵ, σ^{-1}(x)-powers⟩ = y` (1 — the evaluation claim, the §4.1
//!      translation `ct(σ^{-1}(x)·f(x^d)) = y`).
//!
//! The evaluation point `x` is a SCALAR in `Z_q` (the polynomial is over
//! `Z_q[X]`, degree < m·r·d); the ring elements encode `d = 64` coefficients
//! each and `X^d ↔ x^d` via the σ^{-1} constant-term trick.

use crate::challenge::challenge_vec;
use crate::recursion::{prove, verify, LabradorProof};
use crate::relation::{DotCnst, PrincipalStatement, PrincipalWitness, Term, VectorSpec};
use crate::ring::{cmod, pow_mod, Poly, LOGQ, N};
use crate::sis::{ComKey, ComParams, SLACK, T};
use crate::transcript::Transcript;

/// The Greyhound commitment parameters for one polynomial length (the
/// reference's `polcomctx` derived values + the paper's Table 4 shape).
#[derive(Clone, Copy, Debug)]
pub struct PcsParams {
    /// Number of ring elements (N/d).
    pub len: usize,
    /// Row width (the m of the paper's matrix form).
    pub m: usize,
    /// Row count (n in the reference; the r· folding count).
    pub n: usize,
    pub cpp: ComParams,
    /// The predicted witness norm² (the LaBRADOR input bound).
    pub normsq: u64,
}

impl PcsParams {
    /// The reference's `init_polcomctx` search: f/b/fu/bu/kappa/kappa1 with the
    /// m/n split minimizing `2·m·f + (kappa+1)·fu·n`.
    pub fn new(len: usize) -> Result<Self, String> {
        let mut best: Option<PcsParams> = None;
        'outer: for f in 2..=8usize {
            let b = ((LOGQ + f / 2) / f) as u32;
            for kappa in 1..=32usize {
                let m = ((len as f64 * (kappa + 1) as f64 / 2.0).sqrt()).round() as usize;
                let m = m.max(1);
                let n = len.div_ceil(m);
                let varz = 2f64.powi(2 * b as i32) / 12.0 * n as f64 * (32.0 + 4.0 * 8.0);
                let bu = (0.25 * (12.0 * varz).log2()).round().max(1.0) as u32;
                let fu = ((LOGQ as f64) / bu as f64).round().max(1.0) as usize;
                let mut normsq = (2f64.powi(2 * bu as i32) / 12.0
                    + varz / 2f64.powi(2 * bu as i32))
                    * (m * f) as f64;
                normsq += (2f64.powi(2 * bu as i32) * (fu - 1) as f64
                    + 2f64.powi(2 * (LOGQ as i32 - (fu as i32 - 1) * bu as i32)))
                    / 12.0
                    * (kappa + 1) as f64
                    * n as f64;
                normsq *= N as f64;
                if crate::sis::sis_secure(
                    kappa,
                    6.0 * T * SLACK * 2f64.powi(bu as i32) * normsq.sqrt(),
                ) {
                    // kappa1
                    let mut kappa1 = 33usize;
                    for k1 in 1..=32usize {
                        if crate::sis::sis_secure(k1, 2.0 * SLACK * normsq.sqrt()) {
                            kappa1 = k1;
                            break;
                        }
                    }
                    if kappa1 <= 32 {
                        best = Some(PcsParams {
                            len,
                            m,
                            n,
                            cpp: ComParams {
                                f,
                                fu,
                                fg: 0,
                                b,
                                bu,
                                bg: b,
                                kappa,
                                kappa1,
                                u1len: kappa * fu * n,
                                u2len: fu * n,
                            },
                            normsq: normsq as u64,
                        });
                        break 'outer;
                    }
                }
            }
        }
        best.ok_or_else(|| "cannot make commitments secure".into())
    }
}

/// The committed polynomial's private state.
pub struct Committed {
    pub params: PcsParams,
    /// The polynomial as ring elements (len = N/64).
    pub s: Vec<Poly>,
    /// The digit-decomposed rows: sx[i] = the f digits of s[i·m .. i·m+m]
    /// concatenated (m·f ring elements, the small-coefficient form).
    pub sx: Vec<Vec<Poly>>,
    /// The inner commitments t_i = A·sx_i (κ each, n blocks).
    pub t: Vec<Vec<Poly>>,
    /// The digit decomposition of t (fu × bu), flat [i][j][ρ].
    pub t_digits: Vec<Poly>,
    /// The outer commitment u1 (κ1).
    pub u1: Vec<Poly>,
    /// The statement digest.
    pub h: [u8; 16],
}

/// The key windows for the PCS layer: A at 0, B after (disjoint).
struct PcsWindows {
    a_off: usize,
    b_off: usize,
    d_off: usize,
    total: usize,
}

fn pcs_windows(cpp: &ComParams, n: usize, m: usize, f: usize) -> PcsWindows {
    let a_len = cpp.kappa * m * f;
    let b_len = cpp.kappa1 * cpp.fu * cpp.kappa * n;
    let d_len = cpp.kappa1 * cpp.fu * n;
    PcsWindows {
        a_off: 0,
        b_off: a_len,
        d_off: a_len + b_len,
        total: a_len + b_len + d_len,
    }
}

/// Evaluate the polynomial at the scalar x ∈ Z_q (Horner over all coefficients
/// — the reference's `polzvec_eval`).
pub fn eval_polynomial(s: &[Poly], x: i64) -> i64 {
    let mut y: i128 = 0;
    for p in s.iter().rev() {
        for &c in p.0.iter().rev() {
            y = cmod(y * x as i128 + c as i128) as i128;
        }
    }
    cmod(y)
}

/// Commit to a polynomial given as `len` ring elements.
pub fn commit(s: &[Poly], key: &ComKey) -> Result<Committed, String> {
    let params = PcsParams::new(s.len())?;
    let win = pcs_windows(&params.cpp, params.n, params.m, params.cpp.f);
    if win.total > key.len {
        return Err(format!(
            "key too short: need {}, have {}",
            win.total, key.len
        ));
    }
    let cpp = &params.cpp;
    let (m, n) = (params.m, params.n);

    // digit-decompose the rows: sx[i] = concat over digits j of s[i·m..][k]
    let mut sx: Vec<Vec<Poly>> = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<Poly> = (0..m)
            .map(|j| {
                if i * m + j < s.len() {
                    s[i * m + j]
                } else {
                    Poly::zero()
                }
            })
            .collect();
        let mut dig: Vec<Poly> = Vec::with_capacity(m * cpp.f);
        for j in 0..cpp.f {
            for k in 0..m {
                dig.push(row[k].decompose(cpp.f, cpp.b)[j]);
            }
        }
        sx.push(dig);
    }

    // inner commitments + digit decomposition
    let mut t: Vec<Vec<Poly>> = Vec::with_capacity(n);
    let mut t_digits: Vec<Poly> = Vec::new();
    for i in 0..n {
        let ti = key.mul_window(&sx[i], win.a_off, cpp.kappa);
        let dec: Vec<Vec<Poly>> = ti.iter().map(|p| p.decompose(cpp.fu, cpp.bu)).collect();
        for j in 0..cpp.fu {
            for rho in 0..cpp.kappa {
                t_digits.push(dec[rho][j]);
            }
        }
        t.push(ti);
    }

    // outer commitment u1 = B·t̃
    let u1 = key.mul_window(&t_digits, win.b_off, cpp.kappa1);

    // statement digest binding (u1, the parameters)
    let mut tr = Transcript::new(b"greyhound/commit", &[0u8; 0]);
    tr.absorb(&(s.len() as u64).to_le_bytes());
    tr.absorb(&params.h_bytes());
    tr.absorb_polys(&u1);
    let h = tr.h;

    Ok(Committed {
        params,
        s: s.to_vec(),
        sx,
        t,
        t_digits,
        u1,
        h,
    })
}

impl PcsParams {
    fn h_bytes(&self) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[..8].copy_from_slice(&(self.len as u64).to_le_bytes());
        b[8..16].copy_from_slice(&(self.m as u64).to_le_bytes());
        b[16..24].copy_from_slice(&(self.n as u64).to_le_bytes());
        b[24..28].copy_from_slice(&(self.cpp.kappa as u32).to_le_bytes());
        b[28..32].copy_from_slice(&(self.cpp.kappa1 as u32).to_le_bytes());
        b
    }
}

impl Committed {
    /// The public commitment: (u1, digest, params).
    pub fn commitment(&self) -> (Vec<Poly>, [u8; 16], PcsParams) {
        (self.u1.clone(), self.h, self.params)
    }
}

/// The evaluation proof: (u2, the LaBRADOR proof of the 5-constraint statement).
#[derive(Clone, Debug)]
pub struct EvalProof {
    pub u2: Vec<Poly>,
    pub labrador: LabradorProof,
}

/// Produce an evaluation proof for `f(x) = y` (Greyhound Figure 4's Eval.P,
/// with the LaBRADOR sub-proof attached).
pub fn eval_prove(com: &Committed, key: &ComKey, x: i64, y: i64) -> Result<EvalProof, String> {
    let params = &com.params;
    let cpp = &params.cpp;
    let (m, n) = (params.m, params.n);
    let win = pcs_windows(cpp, n, m, cpp.f);
    // verify the claim first (the prover knows f)
    let actual = eval_polynomial(&com.s, x);
    if actual != y {
        return Err(format!("f(x) = {actual} != the claimed {y}"));
    }

    // w_i = Σ_j x^{64j}·s[i·m + j], then ŵ = G^{-1}(w) (fu × bu per block)
    let xd = pow_mod(x, N as u64); // x^64
    let mut w_hat: Vec<Poly> = Vec::with_capacity(cpp.fu * n); // [i][j] digits
    for i in 0..n {
        // w = Σ_j x^{64j}·s[i·m+j]
        let mut w = Poly::zero();
        let mut xpow = Poly::constant(1);
        for j in 0..m {
            if i * m + j < com.s.len() {
                w.add_assign(&com.s[i * m + j].mul(&xpow));
            }
            xpow = xpow.scale(xd);
        }
        for d in w.decompose(cpp.fu, cpp.bu) {
            w_hat.push(d);
        }
    }
    // u2 = D·ŵ (the paper's v commitment)
    let u2 = key.mul_window(&w_hat, win.d_off, cpp.kappa1);

    // the amortization challenges from the transcript (after u1, x, y, u2)
    let mut tr = Transcript::from_state(com.h);
    tr.absorb(&x.to_le_bytes());
    tr.absorb(&y.to_le_bytes());
    tr.absorb_polys(&u2);
    let c = challenge_vec(n, &tr.challenge_seed(), 0);

    // z = Σ_i c_i·sx_i (rank m·f), decomposed into f_z parts of b_z bits by the
    // LaBRADOR level (init_proof chooses them) — we hand the LaBRADOR engine
    // the 5-constraint statement with the witness [z (as one vector), t̃, ŵ].
    let mut z = vec![Poly::zero(); m * cpp.f];
    for i in 0..n {
        for (k, s) in com.sx[i].iter().enumerate() {
            z[k].add_assign(&c[i].mul(s));
        }
    }

    // ---- build the 5-constraint principal statement ----
    // witness vectors: [z (rank m·f), t̃ (rank n·fu·κ), ŵ (rank n·fu)]
    let t_len = n * cpp.fu * cpp.kappa;
    let w_len = n * cpp.fu;
    let vectors = vec![
        VectorSpec::plain(m * cpp.f),
        VectorSpec::plain(t_len),
        VectorSpec::plain(w_len),
    ];
    let mut cnst: Vec<DotCnst> = Vec::with_capacity(2 * cpp.kappa1 + cpp.kappa + 3);
    let v_t = 1;
    let v_w = 2;

    // E1 (κ1): B·t̃ = u1
    for j in 0..cpp.kappa1 {
        let phi = (0..t_len)
            .map(|k| key.rows[win.b_off + j * t_len + k])
            .collect();
        cnst.push(DotCnst {
            terms: vec![Term {
                idx: v_t,
                off: 0,
                phi,
            }],
            a: vec![],
            b: Some(com.u1[j]),
            ct_only: false,
        });
    }
    // E2 (κ1): D·ŵ = u2
    for j in 0..cpp.kappa1 {
        let phi = (0..w_len)
            .map(|k| key.rows[win.d_off + j * w_len + k])
            .collect();
        cnst.push(DotCnst {
            terms: vec![Term {
                idx: v_w,
                off: 0,
                phi,
            }],
            a: vec![],
            b: Some(u2[j]),
            ct_only: false,
        });
    }
    // E3: Σ_i c_i w_i = ⟨a, z⟩ — a_{j·f+d} = 2^{db}·x^{64j} over the m·f
    // digit layout of z; and the w-terms: −c_i·2^{j·bu}·ŵ digits.
    {
        // sx is digit-major: sx[d*m + j] = digit d of row j
        let mut phi_z = vec![Poly::zero(); m * cpp.f];
        for d in 0..cpp.f {
            for j in 0..m {
                phi_z[d * m + j] =
                    Poly::constant(pow_mod(xd, j as u64)).scale(1i64 << (d as u32 * cpp.b));
            }
        }
        let mut phi_w = vec![Poly::zero(); w_len];
        for i in 0..n {
            for j in 0..cpp.fu {
                phi_w[i * cpp.fu + j] = c[i].neg().scale(1i64 << (j as u32 * cpp.bu));
            }
        }
        cnst.push(DotCnst::homogeneous(vec![
            Term {
                idx: 0,
                off: 0,
                phi: phi_z,
            },
            Term {
                idx: v_w,
                off: 0,
                phi: phi_w,
            },
        ]));
    }
    // E4 (κ): A·z = Σ_i c_i t_i
    for rho in 0..cpp.kappa {
        let phi =
            key.rows[win.a_off + rho * (m * cpp.f)..win.a_off + (rho + 1) * (m * cpp.f)].to_vec();
        let mut phi_t = vec![Poly::zero(); t_len];
        for i in 0..n {
            for j in 0..cpp.fu {
                phi_t[i * cpp.fu * cpp.kappa + j * cpp.kappa + rho] =
                    c[i].neg().scale(1i64 << (j as u32 * cpp.bu));
            }
        }
        cnst.push(DotCnst::homogeneous(vec![
            Term {
                idx: 0,
                off: 0,
                phi,
            },
            Term {
                idx: v_t,
                off: 0,
                phi: phi_t,
            },
        ]));
    }
    // E5: ⟨ŵ, σ^{-1}(x)·(x^{64m})^i·2^{j·bu}⟩ = y — the evaluation claim.
    // x̄ = σ^{-1}(x) as a ring element: coefficients [1, x^{63}, x^{62}, ..., x]
    // (the negacyclic X ↦ X^{-1}); the powers (x^{64m})^i between blocks.
    {
        let xd = pow_mod(x, N as u64);
        let xbar = {
            let mut p = [0i64; N];
            p[0] = 1;
            for k in 1..N {
                p[N - k] = -pow_mod(x, k as u64);
            }
            Poly(p)
        };
        let xm = pow_mod(xd, m as u64); // x^{64m}
        let mut phi_w = vec![Poly::zero(); w_len];
        for i in 0..n {
            let scale = pow_mod(xm, i as u64);
            for j in 0..cpp.fu {
                phi_w[i * cpp.fu + j] = xbar.scale(scale).scale(1i64 << (j as u32 * cpp.bu));
            }
        }
        // the evaluation claim is an F'-type constraint (the paper's ct(ȳ) = y)
        cnst.push(DotCnst {
            terms: vec![Term {
                idx: v_w,
                off: 0,
                phi: phi_w,
            }],
            a: vec![],
            b: Some(Poly::constant(y)),
            ct_only: true,
        });
    }

    // assemble the witness
    let wit: Vec<Vec<Poly>> = vec![z, com.t_digits.clone(), w_hat.clone()];

    // hmm — the E5 check must hold of the honest witness: verify completeness
    // of the 5 constraints before handing to LaBRADOR
    // the E5 evaluation claim is an F' constraint — it goes to ct_cnst (the
    // LIFTS family); E1–E4 are the F family
    let mut ct_cnst: Vec<DotCnst> = Vec::new();
    if let Some(e5) = cnst.pop() {
        debug_assert!(e5.ct_only);
        ct_cnst.push(e5);
    }
    let stmt = PrincipalStatement::new(vectors, cnst.clone(), ct_cnst.clone(), params.normsq);
    // widen the norm bound to cover the actual witness (the heuristic prediction
    // may be off at toy scale; the announced bound is what the verifier checks)
    let actual_norm: u64 = wit.iter().flat_map(|v| v.iter().map(|p| p.normsq())).sum();
    let bound = params.normsq.max(actual_norm);
    let stmt = PrincipalStatement::new(stmt.vectors.clone(), stmt.cnst.clone(), ct_cnst, bound);
    if let Err(e) = stmt.check_all(&wit) {
        return Err(format!("PCS statement incomplete: {e}"));
    }
    let wit = PrincipalWitness::new(wit);

    // run the LaBRADOR engine (the paper's Π¹)
    let labrador = prove(&stmt, &wit, key)?;
    Ok(EvalProof { u2, labrador })
}

/// Verify an evaluation proof: reconstruct the 5-constraint statement from
/// (u1, x, y, u2), replay the challenges, and verify the LaBRADOR proof.
pub fn eval_verify(
    u1: &[Poly],
    h: &[u8; 16],
    params: &PcsParams,
    key: &ComKey,
    x: i64,
    y: i64,
    proof: &EvalProof,
) -> Result<(), String> {
    let cpp = &params.cpp;
    let (m, n) = (params.m, params.n);
    let win = pcs_windows(cpp, n, m, cpp.f);
    if win.total > key.len {
        return Err("key too short".into());
    }
    if proof.u2.len() != cpp.kappa1 {
        return Err(format!("u2 length {} != κ1 {}", proof.u2.len(), cpp.kappa1));
    }
    if u1.len() != cpp.kappa1 {
        return Err("u1 length mismatch".into());
    }

    // SIS security at the announced norm (the reference's reduce checks)
    if !crate::sis::sis_secure(
        cpp.kappa,
        6.0 * T * SLACK * 2f64.powi(cpp.bu as i32) * (params.normsq as f64).sqrt(),
    ) {
        return Err("inner commitments not secure".into());
    }
    if !crate::sis::sis_secure(cpp.kappa1, 2.0 * SLACK * (params.normsq as f64).sqrt()) {
        return Err("outer commitments not secure".into());
    }

    // challenges
    let mut tr = Transcript::from_state(*h);
    tr.absorb(&x.to_le_bytes());
    tr.absorb(&y.to_le_bytes());
    tr.absorb_polys(&proof.u2);
    let c = challenge_vec(n, &tr.challenge_seed(), 0);

    // rebuild the 5-constraint statement (mirrors eval_prove)
    let t_len = n * cpp.fu * cpp.kappa;
    let w_len = n * cpp.fu;
    let vectors = vec![
        VectorSpec::plain(m * cpp.f),
        VectorSpec::plain(t_len),
        VectorSpec::plain(w_len),
    ];
    let mut cnst: Vec<DotCnst> = Vec::with_capacity(2 * cpp.kappa1 + cpp.kappa + 3);
    let v_t = 1;
    let v_w = 2;
    for j in 0..cpp.kappa1 {
        let phi = (0..t_len)
            .map(|k| key.rows[win.b_off + j * t_len + k])
            .collect();
        cnst.push(DotCnst {
            terms: vec![Term {
                idx: v_t,
                off: 0,
                phi,
            }],
            a: vec![],
            b: Some(u1[j]),
            ct_only: false,
        });
    }
    for j in 0..cpp.kappa1 {
        let phi = (0..w_len)
            .map(|k| key.rows[win.d_off + j * w_len + k])
            .collect();
        cnst.push(DotCnst {
            terms: vec![Term {
                idx: v_w,
                off: 0,
                phi,
            }],
            a: vec![],
            b: Some(proof.u2[j]),
            ct_only: false,
        });
    }
    {
        let xd = pow_mod(x, N as u64);
        // sx is digit-major: sx[d*m + j] = digit d of row j
        let mut phi_z = vec![Poly::zero(); m * cpp.f];
        for d in 0..cpp.f {
            for j in 0..m {
                phi_z[d * m + j] =
                    Poly::constant(pow_mod(xd, j as u64)).scale(1i64 << (d as u32 * cpp.b));
            }
        }
        let mut phi_w = vec![Poly::zero(); w_len];
        for i in 0..n {
            for j in 0..cpp.fu {
                phi_w[i * cpp.fu + j] = c[i].neg().scale(1i64 << (j as u32 * cpp.bu));
            }
        }
        cnst.push(DotCnst::homogeneous(vec![
            Term {
                idx: 0,
                off: 0,
                phi: phi_z,
            },
            Term {
                idx: v_w,
                off: 0,
                phi: phi_w,
            },
        ]));
    }
    for rho in 0..cpp.kappa {
        let phi =
            key.rows[win.a_off + rho * (m * cpp.f)..win.a_off + (rho + 1) * (m * cpp.f)].to_vec();
        let mut phi_t = vec![Poly::zero(); t_len];
        for i in 0..n {
            for j in 0..cpp.fu {
                phi_t[i * cpp.fu * cpp.kappa + j * cpp.kappa + rho] =
                    c[i].neg().scale(1i64 << (j as u32 * cpp.bu));
            }
        }
        cnst.push(DotCnst::homogeneous(vec![
            Term {
                idx: 0,
                off: 0,
                phi,
            },
            Term {
                idx: v_t,
                off: 0,
                phi: phi_t,
            },
        ]));
    }
    {
        let xd = pow_mod(x, N as u64);
        let xbar = {
            let mut p = [0i64; N];
            p[0] = 1;
            for k in 1..N {
                p[N - k] = -pow_mod(x, k as u64);
            }
            Poly(p)
        };
        let xm = pow_mod(xd, m as u64);
        let mut phi_w = vec![Poly::zero(); w_len];
        for i in 0..n {
            let scale = pow_mod(xm, i as u64);
            for j in 0..cpp.fu {
                phi_w[i * cpp.fu + j] = xbar.scale(scale).scale(1i64 << (j as u32 * cpp.bu));
            }
        }
        // the evaluation claim is an F'-type constraint (the paper's ct(ȳ) = y)
        cnst.push(DotCnst {
            terms: vec![Term {
                idx: v_w,
                off: 0,
                phi: phi_w,
            }],
            a: vec![],
            b: Some(Poly::constant(y)),
            ct_only: true,
        });
    }

    // the E5 claim is F' (the last pushed constraint); E1–E4 are F
    let mut ct_cnst: Vec<DotCnst> = Vec::new();
    if let Some(e5) = cnst.pop() {
        debug_assert!(e5.ct_only);
        ct_cnst.push(e5);
    }
    // the bound: the prover may have widened it — the first level's announced
    // normsq is the bound the prover actually used (§5.4's dynamic remedy);
    // the digest binds the constraints and the levels check the SIS security
    // at the announced norm
    let bound = params.normsq.max(
        proof
            .labrador
            .levels
            .first()
            .map(|l| l.normsq)
            .unwrap_or(params.normsq),
    );
    let stmt = PrincipalStatement::new(vectors, cnst, ct_cnst, bound);
    verify(&stmt, &proof.labrador, key)
}

/// The analytic proof size of an evaluation proof: Greyhound's contribution
/// (u1 + u2 + the challenge seeds) + the LaBRADOR sub-proof (§5's accounting).
pub fn eval_proof_size_bytes(proof: &EvalProof) -> u64 {
    let gh_bits = ((proof.u2.len()) as u64) * (N * LOGQ) as u64 + 128;
    let lab = crate::recursion::proof_size_bytes(&proof.labrador);
    (gh_bits).div_ceil(8) + lab
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_poly(seed: u64, i: usize) -> Poly {
        let mut p = [0i64; N];
        for (j, c) in p.iter_mut().enumerate() {
            *c = (((i * 37 + j * 17 + seed as usize * 13) % 7) as i64) - 3;
        }
        Poly(p)
    }

    #[test]
    fn eval_polynomial_horner() {
        // f = 1 + 2X + 3X^2 (packed in one ring element); f(5) = 1 + 10 + 75 = 86
        let mut p = [0i64; N];
        p[0] = 1;
        p[1] = 2;
        p[2] = 3;
        let s = vec![Poly(p)];
        assert_eq!(eval_polynomial(&s, 5), 86);
        assert_eq!(eval_polynomial(&s, 0), 1);
        // X^64 = -1: f = X^64 (as the ring product) — evaluate the COEFFICIENT
        // view: coefficient 64 lives in the second ring element
        let s2 = vec![Poly::zero(), Poly::constant(1)];
        // f(X) = X^64 → f(x) = x^64 mod q
        assert_eq!(eval_polynomial(&s2, 3), pow_mod(3, 64));
    }

    #[test]
    fn pcs_roundtrip_small() {
        // a small polynomial: 64 ring elements (4096 coefficients)
        let len = 64;
        let s: Vec<Poly> = (0..len).map(|i| small_poly(1, i)).collect();
        let params = PcsParams::new(len).unwrap();
        let win = pcs_windows(&params.cpp, params.n, params.m, params.cpp.f);
        // the LaBRADOR sub-proof needs windows beyond the PCS layer's
        let key = ComKey::expand(win.total + 8192, &[9u8; 32]);
        let com = commit(&s, &key).unwrap();
        let (u1, h, pub_params) = com.commitment();
        let x = 43;
        let y = eval_polynomial(&s, x);
        let proof = eval_prove(&com, &key, x, y).unwrap();
        eval_verify(&u1, &h, &pub_params, &key, x, y, &proof).unwrap();
        // a wrong claim is rejected (the prover refuses to prove it)
        assert!(eval_prove(&com, &key, x, y.wrapping_add(1)).is_err());
        // a tampered proof is rejected
        let mut bad = proof.clone();
        bad.u2[0] = bad.u2[0].add(&Poly::constant(1));
        assert!(eval_verify(&u1, &h, &pub_params, &key, x, y, &bad).is_err());
        // a wrong y in the verification
        assert!(eval_verify(&u1, &h, &pub_params, &key, x, y.wrapping_add(1), &proof).is_err());
    }
}
