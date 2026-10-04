//! **The §7 bridge: R1CS over F_q → the principal linear relation over
//! R_q** (Cyclo, Garreta–Lipmaa–Luhääär–Osadnik, ePrint 2026/359, §7 —
//! the paper's raison d'être).
//!
//! The construction, following the paper line by line:
//!
//! 1. **The θ_k digit map** (§7): `θ_k : R_q → F_q, f(X) ↦ f(k) mod q`
//!    — an F_q-**module** morphism (not a ring homomorphism; that is
//!    not needed). Its right-inverse `θ_k^{-1}` maps `c ∈ F_q` to the
//!    base-`k` digit polynomial `p_c(X) = Σ c_i X^i` (`ℓ_k(q) =
//!    ⌊log_k q⌋` digits, degree < φ) — so `‖p_c‖∞ < k`: **the lift is
//!    small-norm by construction**, which is what lets an arbitrary
//!    F_q-witness be Ajtai-committed bindingly.
//! 2. **The committed hybrid R1CS relation** `Ξ^{com-hyb-R1CS}`: the
//!    witness `z = (x, 1, w)` lifts through `θ_k^{-1}` onto `z' ∈
//!    R_q^m` (committed as `y = A·z'`), while the R1CS constraint
//!    `(M₀·z) ∘ (M₁·z) = M₂·z` **stays over F_q** via `θ_k(z') = z`.
//! 3. **The HyperNova-style linearized sum-check over F_{q²}** (the
//!    paper's F_{q^e}; the LZX realization at e = 2 over the
//!    commitment ring's own modulus `q = 3·2^30+1` — the
//!    bridge-local [`Fq2Q32`] field, `u² = 5`, 5 being the smallest
//!    quadratic non-residue): the R1CS satisfaction becomes
//!    `Σ_{b∈{0,1}^{log m}} eq(b; r)·(Q₀(b)·Q₁(b) − Q₂(b)) = 0` at a
//!    random `r ∈ F_{q²}^{log m}`, with `Q_i = M_i·z` as dense cube
//!    factors (the matrix-vector product — the engine holds them
//!    directly). Individual degree 3 (the `eq·Q₀·Q₁` term).
//! 4. **The terminal linearization**: the prover publishes
//!    `d_i = Q_i(u) ∈ F_{q²}` (the sum-check's factor claims) and the
//!    verifier asserts `(d₀·d₁ − d₂)·eq(u; r) = c` — the paper's
//!    Eq. (2)→terminal check, exactly the [`Fq2SumcheckProof::verify`]
//!    `expected_final` discipline.
//! 5. **The ring lifts**: `d'_i ∈ R_q²` (the componentwise discipline —
//!    the paper's single `R_{q^e}` element realized as the PAIR of
//!    ring elements carrying the two F_q-components' base-k digits;
//!    the LatticeBlindFold wave's rank-doubling pattern) with
//!    `θ_k(d'_i^{(b)}) = d_i^{(b)}` — **verifier-checked here**. The
//!    linear claims (4) (`Σ_{b'} MLE[M_i](u, b')·w'_{b'} = d'_i`)
//!    **ride the fold** — they are the principal-linear-relation
//!    instance data the folding layer's decider eventually consumes
//!    (the paper's architecture verbatim; the bridge records them in
//!    [`PrincipalLinearClaim`]).
//! 6. **The public-prefix elimination**: the verifier samples
//!    `v ∈ F_{q²}^{log(ℓ+1)}` and computes `e = MLE[(x, 1)](v)`; the
//!    claim `MLE[w'](v, 0) = e` (binding `w'`'s prefix to `(x, 1)`
//!    except with probability `≤ log(ℓ+1)/q² ≈ 2^{-121}`) rides the
//!    fold. The paper requires `ℓ+1` to be a power of two — enforced
//!    fail-closed here.
//! 7. **Skipping Π^ext when `k ≤ b`** (the remark that makes Cyclo the
//!    lightweight folder for F_q-relations): the lifted witness has
//!    norm `< k`; with `k` at or below the chunk base `b =
//!    2^{chunk_log−1}`, the extension-commitment step is SKIPPED and
//!    the folded witness feeds the range test directly. The
//!    end-to-end test wires the bridge's output straight into
//!    [`crate::cyclo::CycloAccumulator::new`].
//!
//! # Honest deviations (the LZX ledger)
//!
//! * `e = 2` (the paper suggests larger e for its 50-bit q; at LZX's
//!   31.6-bit q the quadratic extension gives the 2^{-121}–2^{-122}
//!   Schwartz–Zippel floor the paper's λ = 128 posture needs — the
//!   honest gap is the ~7 bits, documented).
//! * The `d'_i` ride as R_q PAIRS rather than the paper's single
//!   `R_{q^e}` tensor elements (the componentwise discipline — the
//!   same rank-doubling the LatticeBlindFold wave uses for R_K
//!   relations; the algebra is exact, the wire shape differs).
//! * The (4) linear claims and the prefix claim are recorded, not
//!   decided, by the bridge — exactly the paper's architecture (the
//!   principal linear relation's decider owns them).

#[cfg(test)]
use crate::cyclo::chunk_element;
#[cfg(test)]
use lattice_ring::Modulus32;
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_ring::{RingConfig, RingElement};

/// The commitment ring's modulus (q = 3·2^30 + 1 — `Modulus32::Q_32`).
const Q: u64 = 3221225473;

// ---------------------------------------------------------------------------
// F_{q^2} = F_q[u]/(u^2 - 5) — the bridge-local extension field
// ---------------------------------------------------------------------------

/// An element `c0 + c1·u` of `F_{q²}` over the commitment ring's
/// modulus (both components reduced `< q`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fq2Q32 {
    pub c0: u64,
    pub c1: u64,
}

impl Fq2Q32 {
    pub const ZERO: Fq2Q32 = Fq2Q32 { c0: 0, c1: 0 };
    pub const ONE: Fq2Q32 = Fq2Q32 { c0: 1, c1: 0 };

    /// The non-residue (5 is the smallest quadratic non-residue mod q).
    const D: u64 = 5;

    pub fn from_u64(x: u64) -> Self {
        Fq2Q32 {
            c0: x % Q,
            c1: 0,
        }
    }

    pub fn from_pair(c0: u64, c1: u64) -> Self {
        Fq2Q32 {
            c0: c0 % Q,
            c1: c1 % Q,
        }
    }

    fn reduce128(x: u128) -> u64 {
        (x % Q as u128) as u64
    }

    pub fn add(&self, o: &Self) -> Self {
        Fq2Q32 {
            c0: (self.c0 + o.c0) % Q,
            c1: (self.c1 + o.c1) % Q,
        }
    }

    pub fn sub(&self, o: &Self) -> Self {
        Fq2Q32 {
            c0: (self.c0 + Q - o.c0) % Q,
            c1: (self.c1 + Q - o.c1) % Q,
        }
    }

    pub fn neg(&self) -> Self {
        Fq2Q32 {
            c0: (Q - self.c0) % Q,
            c1: (Q - self.c1) % Q,
        }
    }

    /// `(a0 + a1 u)(b0 + b1 u) = (a0 b0 + 5 a1 b1) + (a0 b1 + a1 b0) u`.
    pub fn mul(&self, o: &Self) -> Self {
        let a0 = self.c0 as u128;
        let a1 = self.c1 as u128;
        let b0 = o.c0 as u128;
        let b1 = o.c1 as u128;
        Fq2Q32 {
            c0: Self::reduce128(a0 * b0 + Self::D as u128 * a1 * b1),
            c1: Self::reduce128(a0 * b1 + a1 * b0),
        }
    }

    /// The conjugate `c0 − c1 u` (the norm's partner).
    pub fn conj(&self) -> Self {
        Fq2Q32 {
            c0: self.c0,
            c1: (Q - self.c1) % Q,
        }
    }

    /// The norm `c0² − 5 c1² ∈ F_q` (nonzero iff invertible).
    fn norm(&self) -> u64 {
        let a = self.c0 as u128;
        let b = self.c1 as u128;
        Self::reduce128(
            (a * a + Q as u128 - Self::D as u128 * b * b % Q as u128) % Q as u128,
        )
    }

    pub fn inverse(&self) -> Option<Self> {
        let n = self.norm();
        if n == 0 {
            return None;
        }
        // n^{q-2} by square-and-multiply.
        let mut result: u64 = 1;
        let mut base = n % Q;
        let mut exp = Q - 2;
        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % Q as u128) as u64;
            }
            base = ((base as u128 * base as u128) % Q as u128) as u64;
            exp >>= 1;
        }
        let inv = result;
        Some(Fq2Q32 {
            c0: ((self.c0 as u128 * inv as u128) % Q as u128) as u64,
            c1: (((Q - self.c1) % Q) as u128 * inv as u128 % Q as u128) as u64,
        })
    }

    pub fn is_zero(&self) -> bool {
        self.c0 == 0 && self.c1 == 0
    }

    /// Canonical wire form (16 LE bytes).
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.c0.to_le_bytes());
        out[8..].copy_from_slice(&self.c1.to_le_bytes());
        out
    }
}

// ---------------------------------------------------------------------------
// The bridge-local sum-check over F_{q²} (the fq2_sumcheck pattern,
// field-swapped to the commitment ring's own extension)
// ---------------------------------------------------------------------------

/// A virtual polynomial over `F_{q²}`: dense cube factors + product
/// terms with coefficients.
#[derive(Clone, Debug, Default)]
pub struct Q2VirtualPoly {
    pub num_vars: usize,
    pub factors: Vec<Vec<Fq2Q32>>,
    pub terms: Vec<(Fq2Q32, Vec<usize>)>,
}

impl Q2VirtualPoly {
    pub fn new(num_vars: usize) -> Self {
        Q2VirtualPoly {
            num_vars,
            factors: Vec::new(),
            terms: Vec::new(),
        }
    }

    pub fn add_factor(&mut self, evals: Vec<Fq2Q32>) -> Result<usize, String> {
        if evals.len() != 1usize << self.num_vars {
            return Err(format!(
                "factor shape: {} vs 2^{}",
                evals.len(),
                self.num_vars
            ));
        }
        self.factors.push(evals);
        Ok(self.factors.len() - 1)
    }

    pub fn add_term(&mut self, coeff: Fq2Q32, ids: Vec<usize>) -> Result<(), String> {
        if ids.iter().any(|id| *id >= self.factors.len()) {
            return Err("term references unknown factor".into());
        }
        self.terms.push((coeff, ids));
        Ok(())
    }

    pub fn max_degree(&self) -> usize {
        self.terms.iter().map(|(_, ids)| ids.len()).max().unwrap_or(1)
    }
}

/// The round-polynomial evaluations (degree D ⇒ D+1 values at nodes
/// 0..=D).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Q2SumcheckProof {
    pub rounds: Vec<Vec<Fq2Q32>>,
}

fn half_bind_q2(evals: &[Fq2Q32], t: &Fq2Q32) -> Vec<Fq2Q32> {
    let pts = evals.len() / 2;
    let mut out = Vec::with_capacity(pts);
    for p in 0..pts {
        let a = evals[p];
        let b = evals[p + pts];
        out.push(a.add(&b.sub(&a).mul(t)));
    }
    out
}

fn sum_products_q2(bound: &[Vec<Fq2Q32>], terms: &[(Fq2Q32, Vec<usize>)]) -> Fq2Q32 {
    let mut acc = Fq2Q32::ZERO;
    for (coeff, ids) in terms {
        if ids.is_empty() {
            continue;
        }
        let pts = bound[ids[0]].len();
        #[allow(clippy::needless_range_loop)]
        for p in 0..pts {
            let mut prod = *coeff;
            for fi in ids {
                prod = prod.mul(&bound[*fi][p]);
            }
            acc = acc.add(&prod);
        }
    }
    acc
}

/// Lagrange-evaluate the round polynomial at `r` (nodes 0..n−1).
fn interpolate_q2(evals: &[Fq2Q32], r: &Fq2Q32) -> Fq2Q32 {
    let n = evals.len();
    let mut acc = Fq2Q32::ZERO;
    for (i, v) in evals.iter().enumerate() {
        let xi = Fq2Q32::from_u64(i as u64);
        let mut weight = Fq2Q32::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = Fq2Q32::from_u64(j as u64);
            let num = r.sub(&xj);
            let den = xi.sub(&xj);
            if let Some(inv) = den.inverse() {
                weight = weight.mul(&num.mul(&inv));
            }
        }
        acc = acc.add(&v.mul(&weight));
    }
    acc
}

fn absorb_q2_slice(
    transcript: &mut Transcript,
    label: &[u8],
    values: &[Fq2Q32],
) -> Result<(), String> {
    let mut buf = Vec::with_capacity(values.len() * 16);
    for v in values {
        buf.extend_from_slice(&v.to_bytes());
    }
    transcript
        .append_bytes(label, &buf)
        .map_err(|e| format!("{e:?}"))
}

/// A challenge from the transcript (16 bytes → the pair).
fn challenge_q2(transcript: &mut Transcript, label: &[u8]) -> Result<Fq2Q32, String> {
    let b = transcript
        .challenge_bytes(label, 16)
        .map_err(|e| format!("{e:?}"))?;
    let c0 = u64::from_le_bytes(b[..8].try_into().map_err(|e| format!("{e:?}"))?);
    let c1 = u64::from_le_bytes(b[8..].try_into().map_err(|e| format!("{e:?}"))?);
    Ok(Fq2Q32::from_pair(c0, c1))
}

/// Prover-side output.
pub struct Q2SumcheckOutput {
    pub proof: Q2SumcheckProof,
    pub challenges: Vec<Fq2Q32>,
    pub final_claim: Fq2Q32,
    /// Per-factor evaluations at the terminal point (the paper's `d_i`
    /// live here for matrix factors).
    pub factor_claims: Vec<Fq2Q32>,
}

/// Prove `Σ_{x ∈ {0,1}^n} P(x) = claim` over `F_{q²}`.
pub fn q2_sumcheck_prove(
    vp: &Q2VirtualPoly,
    claim: Fq2Q32,
    transcript: &mut Transcript,
) -> Result<Q2SumcheckOutput, String> {
    let n = vp.num_vars;
    let d = vp.max_degree();
    if vp.terms.is_empty() {
        if !claim.is_zero() {
            return Err("claim mismatch (zero polynomial)".into());
        }
        let mut challenges = Vec::with_capacity(n);
        let mut rounds = Vec::with_capacity(n);
        for _ in 0..n {
            let evals = vec![Fq2Q32::ZERO, Fq2Q32::ZERO];
            absorb_q2_slice(transcript, b"q2-sc-round", &evals)?;
            let r = challenge_q2(transcript, b"q2-sc-chal")?;
            challenges.push(r);
            rounds.push(evals);
        }
        return Ok(Q2SumcheckOutput {
            proof: Q2SumcheckProof { rounds },
            challenges,
            final_claim: Fq2Q32::ZERO,
            factor_claims: Vec::new(),
        });
    }
    let mut bound: Vec<Vec<Fq2Q32>> = vp.factors.clone();
    let mut current_claim = claim;
    let mut rounds: Vec<Vec<Fq2Q32>> = Vec::with_capacity(n);
    let mut challenges: Vec<Fq2Q32> = Vec::with_capacity(n);
    for _round in 0..n {
        let mut evals_at = Vec::with_capacity(d + 1);
        for t in 0..=d {
            let t_fe = Fq2Q32::from_u64(t as u64);
            let bound_at: Vec<Vec<Fq2Q32>> =
                bound.iter().map(|f| half_bind_q2(f, &t_fe)).collect();
            evals_at.push(sum_products_q2(&bound_at, &vp.terms));
        }
        absorb_q2_slice(transcript, b"q2-sc-round", &evals_at)?;
        let r = challenge_q2(transcript, b"q2-sc-chal")?;
        challenges.push(r);
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err("claim mismatch (round consistency)".into());
        }
        current_claim = interpolate_q2(&evals_at, &r);
        for b in bound.iter_mut() {
            *b = half_bind_q2(b, &r);
        }
        rounds.push(evals_at);
    }
    let factor_claims: Vec<Fq2Q32> = bound
        .iter()
        .map(|f| f.first().copied().unwrap_or(Fq2Q32::ZERO))
        .collect();
    let mut final_claim = Fq2Q32::ZERO;
    for (coeff, ids) in &vp.terms {
        let mut prod = *coeff;
        for fi in ids {
            prod = prod.mul(&factor_claims[*fi]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current_claim {
        return Err("final check failed".into());
    }
    Ok(Q2SumcheckOutput {
        proof: Q2SumcheckProof { rounds },
        challenges,
        final_claim,
        factor_claims,
    })
}

/// Verifier state for protocol-layer terminal checks.
pub struct Q2SumcheckVerifier {
    pub point: Vec<Fq2Q32>,
    pub final_claim: Fq2Q32,
}

impl Q2SumcheckProof {
    /// Verify against a claimed sum; `expected_final` (if provided) is
    /// the caller's locally-computable `P(r)` — the paper's terminal
    /// `(d₀·d₁ − d₂)·eq(u; r) = c` check lands here.
    pub fn verify(
        &self,
        num_vars: usize,
        max_degree: usize,
        claim: Fq2Q32,
        transcript: &mut Transcript,
        expected_final: Option<Fq2Q32>,
    ) -> Result<Q2SumcheckVerifier, String> {
        if self.rounds.len() != num_vars {
            return Err(format!(
                "round shape: {} vs {num_vars}",
                self.rounds.len()
            ));
        }
        let mut current_claim = claim;
        let mut point = Vec::with_capacity(num_vars);
        for (j, evals) in self.rounds.iter().enumerate() {
            if evals.len() != max_degree + 1 {
                return Err(format!(
                    "round {j}: {} values vs degree {max_degree}",
                    evals.len()
                ));
            }
            absorb_q2_slice(transcript, b"q2-sc-round", evals)?;
            let r = challenge_q2(transcript, b"q2-sc-chal")?;
            let sum01 = evals[0].add(&evals[1]);
            if sum01 != current_claim {
                return Err(format!("round {j}: g(0)+g(1) mismatch"));
            }
            current_claim = interpolate_q2(evals, &r);
            point.push(r);
        }
        if let Some(exp) = expected_final {
            if exp != current_claim {
                return Err("terminal: expected final mismatch".into());
            }
        }
        Ok(Q2SumcheckVerifier {
            point,
            final_claim: current_claim,
        })
    }
}

/// The eq table `eq(b, r)` over the cube — BIG-ENDIAN (variable 0 at
/// the high bit, the engine's round order; the `cyclo_protocols`
/// convention).
pub fn eq_table_q2(r: &[Fq2Q32]) -> Vec<Fq2Q32> {
    let mut evals = vec![Fq2Q32::ONE; 1usize << r.len()];
    for (var, e) in r.iter().enumerate() {
        let shift = r.len() - 1 - var;
        for (idx, val) in evals.iter_mut().enumerate() {
            let bit = (idx >> shift) & 1;
            let term = if bit == 1 { *e } else { Fq2Q32::ONE.sub(e) };
            *val = val.mul(&term);
        }
    }
    evals
}

/// Evaluate the MLE of an F_q vector at an F_{q²} point — BIG-ENDIAN
/// (point[0] at the high bit, pairing with the engine's challenges).
pub fn mle_eval_q2(x: &[u64], v: &[Fq2Q32]) -> Fq2Q32 {
    let num_vars = x.len().trailing_zeros() as usize;
    let mut acc = Fq2Q32::ZERO;
    for (idx, &xj) in x.iter().enumerate() {
        let mut w = Fq2Q32::ONE;
        for (var, pv) in v.iter().enumerate() {
            let bit = (idx >> (num_vars - 1 - var)) & 1;
            let term = if bit == 1 { *pv } else { Fq2Q32::ONE.sub(pv) };
            w = w.mul(&term);
        }
        acc = acc.add(&w.mul(&Fq2Q32::from_u64(xj)));
    }
    acc
}

// ---------------------------------------------------------------------------
// The θ_k digit map (§7)
// ---------------------------------------------------------------------------

/// The base-`k` digit embedding `θ_k^{-1} : F_q → R_q` and its
/// (left-inverse) projection `θ_k : R_q → F_q, f ↦ f(k) mod q`.
///
/// `ℓ_k(q) = ⌊log_k q⌋` digits; the lift has `‖p_c‖∞ < k` (the
/// small-norm-by-construction property the whole bridge rests on).
#[derive(Clone, Debug)]
pub struct ThetaK {
    pub k: u64,
    /// The digit count `ℓ_k(q)`.
    pub digits: usize,
}

impl ThetaK {
    /// Construct for base `k` — the digit count is the SMALLEST `ℓ`
    /// with `k^ℓ ≥ q` (every field element is representable; the
    /// paper's `ℓ_k(q)` with the ceiling — `⌊log_k q⌋` under-covers the
    /// values in `(k^⌊log_k q⌋, q)` at non-power-of-k moduli), and
    /// `ℓ ≤ 64` (the ring's coefficient slots).
    pub fn new(k: u64) -> Result<Self, String> {
        if k < 2 {
            return Err("k must be ≥ 2".into());
        }
        let mut digits = 0usize;
        let mut pow = 1u128;
        while pow < Q as u128 {
            pow *= k as u128;
            digits += 1;
            if digits > 64 {
                return Err(format!("digit count {digits} outside the ring's slots"));
            }
        }
        Ok(ThetaK { k, digits })
    }

    /// `θ_k^{-1}(c)`: the base-k digit polynomial — UNSIGNED digits in
    /// `[0, k)` (the paper's formulation; the norm bound `B > k`
    /// covers every digit).
    pub fn embed(&self, ring: &RingConfig, c: u64) -> RingElement {
        let mut coeffs = vec![0u32; ring.n()];
        let mut rem = (c % Q) as u128;
        for slot in coeffs.iter_mut().take(self.digits) {
            *slot = ((rem % self.k as u128) % Q as u128) as u32;
            rem /= self.k as u128;
        }
        // The remainder must be zero (the digit budget covers q).
        debug_assert_eq!(rem, 0);
        RingElement::from_coeffs(ring, coeffs)
    }

    /// `θ_k(f) = f(k) mod q` — the module projection (i128 accumulation:
    /// the balanced coefficients are negative).
    pub fn project(&self, _ring: &RingConfig, f: &RingElement) -> u64 {
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &c in f.coeffs() {
            let bal = if u64::from(c) > Q / 2 {
                i128::from(c) - Q as i128
            } else {
                i128::from(c)
            };
            acc = (acc + bal * pow).rem_euclid(Q as i128);
            pow = (pow * self.k as i128) % Q as i128;
        }
        acc as u64
    }

    /// `θ_k^{-1}` on an F_{q²} element — the componentwise pair (the
    /// rank-doubling discipline).
    pub fn embed_pair(&self, ring: &RingConfig, f: &Fq2Q32) -> [RingElement; 2] {
        [self.embed(ring, f.c0), self.embed(ring, f.c1)]
    }

    /// `θ_k` on a component pair — the inverse check.
    pub fn project_pair(&self, ring: &RingConfig, pair: &[RingElement; 2]) -> Fq2Q32 {
        Fq2Q32::from_pair(self.project(ring, &pair[0]), self.project(ring, &pair[1]))
    }
}

// ---------------------------------------------------------------------------
// The R1CS shape over F_q
// ---------------------------------------------------------------------------

/// An R1CS shape `(M₀·z) ∘ (M₁·z) = M₂·z` with `z = (x, 1, w) ∈ F_q^m`,
/// `x ∈ F_q^ℓ`, matrices row-major `m×m`. `m` and `ℓ+1` must be powers
/// of two (the sum-check cube and the prefix MLE respectively).
#[derive(Clone, Debug)]
pub struct R1csQ32 {
    pub ell: usize,
    pub m: usize,
    /// Row-major `m·m` entries per matrix.
    pub mats: [Vec<u64>; 3],
}

impl R1csQ32 {
    /// The matrix-vector product `M_i·z` over F_q.
    pub fn matvec(&self, i: usize, z: &[u64]) -> Result<Vec<u64>, String> {
        if z.len() != self.m {
            return Err(format!("z length {} vs m {}", z.len(), self.m));
        }
        let mat = &self.mats[i];
        let mut out = Vec::with_capacity(self.m);
        for r in 0..self.m {
            let mut acc: u128 = 0;
            for c in 0..self.m {
                acc += mat[r * self.m + c] as u128 * z[c] as u128;
                acc %= Q as u128;
            }
            out.push(acc as u64);
        }
        Ok(out)
    }

    /// Check R1CS satisfaction (the prover's fail-closed self-check).
    pub fn satisfied(&self, z: &[u64]) -> Result<bool, String> {
        let a = self.matvec(0, z)?;
        let b = self.matvec(1, z)?;
        let c = self.matvec(2, z)?;
        Ok((0..self.m).all(|j| {
            let prod = (a[j] as u128 * b[j] as u128) % Q as u128;
            prod as u64 == c[j]
        }))
    }

    /// The statement digest (the matrices + shape — absorbed before the
    /// challenges so the sum-check is bound to THE matrices).
    pub fn digest(&self) -> [u8; 32] {
        let mut buf = Vec::with_capacity(24 + 3 * self.mats[0].len() * 8);
        buf.extend_from_slice(&(self.ell as u64).to_le_bytes());
        buf.extend_from_slice(&(self.m as u64).to_le_bytes());
        for mat in &self.mats {
            for e in mat {
                buf.extend_from_slice(&e.to_le_bytes());
            }
        }
        Transcript::hash_domain(b"cyclo-r1cs-shape", &buf)
    }
}

// ---------------------------------------------------------------------------
// The bridge protocol
// ---------------------------------------------------------------------------

/// The ride-the-fold principal-linear claim data (the paper's Eqs. (3)
/// and (4) plus the prefix claim — the folding layer's decider owns
/// their terminal checks).
#[derive(Clone, Debug)]
pub struct PrincipalLinearClaim {
    /// The Ajtai commitment `y = A·z'` (serialized rows).
    pub commitment: Vec<u8>,
    /// The sum-check terminal point `u ∈ F_{q²}^{log m}`.
    pub u: Vec<Fq2Q32>,
    /// The published `d_i = Q_i(u) ∈ F_{q²}` (i ∈ [3]).
    pub d: [Fq2Q32; 3],
    /// The ring lifts `d'_i ∈ R_q²` (the componentwise discipline) with
    /// `θ_k(d'_i^{(b)}) = d_i^{(b)}` — serialized coefficient vectors.
    pub d_lift: [[Vec<u32>; 2]; 3],
    /// The prefix-elimination point `v ∈ F_{q²}^{log(ℓ+1)}`.
    pub v: Vec<Fq2Q32>,
    /// `e = MLE[(x, 1)](v)` — binds `w'`'s prefix to the public input.
    pub e: Fq2Q32,
    /// The θ_k parameters (k — the digit base).
    pub theta_k: u64,
    /// The commitment's row count k (for the verifier's re-parse).
    pub commit_k: usize,
}

/// The bridge proof artifact.
#[derive(Clone, Debug)]
pub struct R1csBridgeProof {
    /// The linearized sum-check (claim 0 over F_{q²}).
    pub sumcheck: Q2SumcheckProof,
    /// The ride-the-fold claim data.
    pub claim: PrincipalLinearClaim,
}

/// Prove the §7 bridge: reduce `(x, w) ∈ Ξ^{R1CS}` to the principal
/// linear claim over R_q.
///
/// * `pk` — the Ajtai key at `m` ring elements (the commitment of the
///   lifted witness);
/// * `theta` — the digit base (with `k ≤ 2^{chunk_log−1}` the output
///   feeds Cyclo's range test directly — the skip-Π^ext remark).
#[allow(clippy::too_many_arguments)]
pub fn prove_r1cs_bridge(
    ring: &RingConfig,
    pk: &AjtaiPublicKey,
    shape: &R1csQ32,
    x: &[u64],
    w: &[u64],
    theta: &ThetaK,
    transcript: &mut Transcript,
) -> Result<R1csBridgeProof, String> {
    if pk.params.ring.modulus.q != ring.modulus.q
        || pk.params.ring.log_n != ring.log_n
        || pk.params.m != shape.m
    {
        return Err("the Ajtai key must match the ring and witness width".into());
    }
    if x.len() != shape.ell || w.len() != shape.m - shape.ell - 1 {
        return Err(format!(
            "shape: |x| = {} (want {}), |w| = {} (want {})",
            x.len(),
            shape.ell,
            w.len(),
            shape.m - shape.ell - 1
        ));
    }
    if shape.m.count_ones() != 1 || (shape.ell + 1).count_ones() != 1 {
        return Err("m and ell+1 must be powers of two".into());
    }
    // 1. z = (x, 1, w) — the R1CS satisfaction self-check (fail-closed:
    //    a bridge proof of a UNSATISFIED instance cannot even start).
    let mut z = x.to_vec();
    z.push(1);
    z.extend_from_slice(w);
    if !shape.satisfied(&z)? {
        return Err("the witness does not satisfy the R1CS instance".into());
    }
    // 2. The lift z' = θ_k^{-1}(z) ∈ R_q^m — small-norm by construction.
    let z_lift: Vec<RingElement> = z.iter().map(|&c| theta.embed(ring, c)).collect();
    for e in &z_lift {
        if e.infinity_norm() as u64 >= theta.k {
            return Err("lift norm gate (k) exceeded".into());
        }
    }
    // 3. The Ajtai commitment y = A·z'.
    let commitment: AjtaiCommitment =
        pk.commit(&z_lift).map_err(|e| format!("{e:?}"))?;
    // 4. The statement absorption (FS hygiene — everything public
    //    before any challenge): the shape digest, x, y, θ.
    absorb_bridge_statement(transcript, shape, x, &commitment, theta)?;
    // 5. The eq point r ∈ F_{q²}^{log m}.
    let log_m = shape.m.trailing_zeros() as usize;
    let r: Vec<Fq2Q32> = (0..log_m)
        .map(|_| challenge_q2(transcript, b"cyclo-r1cs-r"))
        .collect::<Result<Vec<_>, _>>()?;
    // 6. The linearized sum-check: factors [eq, M₀z, M₁z, M₂z], the
    //    claim Σ_b eq(b,r)·(q₀[b]q₁[b] − q₂[b]) = 0.
    let eq = eq_table_q2(&r);
    let q0 = shape.matvec(0, &z)?;
    let q1 = shape.matvec(1, &z)?;
    let q2 = shape.matvec(2, &z)?;
    let mut vp = Q2VirtualPoly::new(log_m);
    let f_eq = vp.add_factor(eq)?;
    let f0 = vp.add_factor(q0.iter().map(|&v| Fq2Q32::from_u64(v)).collect())?;
    let f1 = vp.add_factor(q1.iter().map(|&v| Fq2Q32::from_u64(v)).collect())?;
    let f2 = vp.add_factor(q2.iter().map(|&v| Fq2Q32::from_u64(v)).collect())?;
    vp.add_term(Fq2Q32::ONE, vec![f_eq, f0, f1])?;
    vp.add_term(Fq2Q32::ONE.neg(), vec![f_eq, f2])?;
    let out = q2_sumcheck_prove(&vp, Fq2Q32::ZERO, transcript)?;
    // factor_claims = [eq(u,r), d0, d1, d2]
    let d = [
        out.factor_claims[f0],
        out.factor_claims[f1],
        out.factor_claims[f2],
    ];
    // 7. The ring lifts d'_i (the componentwise discipline).
    let d_lift: [[Vec<u32>; 2]; 3] = d
        .iter()
        .map(|di| {
            let pair = theta.embed_pair(ring, di);
            [
                pair[0].coeffs().to_vec(),
                pair[1].coeffs().to_vec(),
            ]
        })
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "lift shape")?;
    // 8. The prefix elimination: v ∈ F_{q²}^{log(ℓ+1)}, e = MLE[(x,1)](v).
    let log_prefix = (shape.ell + 1).trailing_zeros() as usize;
    let v: Vec<Fq2Q32> = (0..log_prefix)
        .map(|_| challenge_q2(transcript, b"cyclo-r1cs-v"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut prefix = x.to_vec();
    prefix.push(1);
    let e = mle_eval_q2(&prefix, &v);
    // The ride-the-fold claim (the MLE[w'](v, 0) = e binding).
    Ok(R1csBridgeProof {
        sumcheck: out.proof,
        claim: PrincipalLinearClaim {
            commitment: commitment.to_bytes(),
            u: out.challenges.clone(),
            d,
            d_lift,
            v,
            e,
            theta_k: theta.k,
            commit_k: pk.params.k,
        },
    })
}

/// Verify the bridge proof: the sum-check replay (claim 0 with the
/// terminal `(d₀·d₁ − d₂)·eq(u; r) = c` check — the eq factor's claim
/// RECOMPUTED by the verifier, never trusted from the proof), the
/// θ-lift consistencies, and the prefix evaluation — everything the
/// verifier can check WITHOUT the folding layer; the rest rides the
/// fold (returned in the claim).
pub fn verify_r1cs_bridge(
    ring: &RingConfig,
    shape: &R1csQ32,
    x: &[u64],
    proof: &R1csBridgeProof,
    transcript: &mut Transcript,
) -> Result<PrincipalLinearClaim, String> {
    if x.len() != shape.ell {
        return Err("public input length".into());
    }
    let claim = &proof.claim;
    let theta = ThetaK::new(claim.theta_k)?;
    // 1. The statement replay (the commitment re-parsed for the
    //    absorption — byte-identical to the prover's).
    let commitment = AjtaiCommitment::from_bytes(ring, claim.commit_k, &claim.commitment)
        .map_err(|e| format!("{e:?}"))?;
    absorb_bridge_statement(transcript, shape, x, &commitment, &theta)?;
    // 2. The r replay (the eq point — the verifier's own draw).
    let log_m = shape.m.trailing_zeros() as usize;
    let r: Vec<Fq2Q32> = (0..log_m)
        .map(|_| challenge_q2(transcript, b"cyclo-r1cs-r"))
        .collect::<Result<Vec<_>, _>>()?;
    // 3. The sum-check verification (continues the transcript) — the
    //    terminal point u is the verifier's derivation.
    let verdict = proof.sumcheck.verify(log_m, 3, Fq2Q32::ZERO, transcript, None)?;
    let u = verdict.point;
    if u.len() != log_m {
        return Err("terminal point arity".into());
    }
    // 4. The terminal check: eq(u, r)·(d₀·d₁ − d₂) = c — the eq value
    //    RECOMPUTED from the verifier's own (u, r); the d_i are the
    //    proof's publications (the paper's Eq. (2) terminal).
    let eq_ur = {
        let mut acc = Fq2Q32::ONE;
        for (ui, ri) in u.iter().zip(r.iter()) {
            // (1−u)(1−r) + u·r
            let one_minus_u = Fq2Q32::ONE.sub(ui);
            let one_minus_r = Fq2Q32::ONE.sub(ri);
            let term = one_minus_u.mul(&one_minus_r).add(&ui.mul(ri));
            acc = acc.mul(&term);
        }
        acc
    };
    let expected_final = eq_ur.mul(&claim.d[0].mul(&claim.d[1]).sub(&claim.d[2]));
    if expected_final != verdict.final_claim {
        return Err("terminal: (d0·d1 − d2)·eq(u, r) ≠ c".into());
    }
    // 5. The v replay (the prefix point — after the sum-check, matching
    //    the prover's draw order) + the prefix evaluation check.
    let log_prefix = (shape.ell + 1).trailing_zeros() as usize;
    let v: Vec<Fq2Q32> = (0..log_prefix)
        .map(|_| challenge_q2(transcript, b"cyclo-r1cs-v"))
        .collect::<Result<Vec<_>, _>>()?;
    if v != claim.v {
        return Err("prefix point replay mismatch".into());
    }
    let mut prefix = x.to_vec();
    prefix.push(1);
    let e = mle_eval_q2(&prefix, &claim.v);
    if e != claim.e {
        return Err("prefix evaluation mismatch".into());
    }
    // 6. The θ-lift consistencies: θ_k(d'_i^{(b)}) = d_i^{(b)}.
    for i in 0..3 {
        let lift = [
            RingElement::from_coeffs(ring, claim.d_lift[i][0].clone()),
            RingElement::from_coeffs(ring, claim.d_lift[i][1].clone()),
        ];
        let back = theta.project_pair(ring, &lift);
        if back != claim.d[i] {
            return Err(format!("θ-lift consistency failed for d_{i}"));
        }
    }
    // The rest rides the fold (the paper's architecture): the linear
    // claims (4) pair MLE[M_i](u, b') with the hidden w' — the folding
    // layer's decider owns their terminal checks.
    Ok(claim.clone())
}

fn absorb_bridge_statement(
    transcript: &mut Transcript,
    shape: &R1csQ32,
    x: &[u64],
    commitment: &AjtaiCommitment,
    theta: &ThetaK,
) -> Result<(), String> {
    let digest = shape.digest();
    transcript
        .append_bytes(b"cyclo-r1cs-shape", &digest)
        .map_err(|e| format!("{e:?}"))?;
    let mut xbuf = Vec::with_capacity(8 * x.len());
    for &c in x {
        xbuf.extend_from_slice(&c.to_le_bytes());
    }
    transcript
        .append_bytes(b"cyclo-r1cs-x", &xbuf)
        .map_err(|e| format!("{e:?}"))?;
    transcript
        .append_bytes(b"cyclo-r1cs-y", &commitment.to_bytes())
        .map_err(|e| format!("{e:?}"))?;
    transcript
        .append_bytes(
            b"cyclo-r1cs-theta",
            &[theta.k as u8, theta.digits as u8],
        )
        .map_err(|e| format!("{e:?}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_commitment::ajtai::AjtaiParams;

    fn ring() -> RingConfig {
        RingConfig::new(Modulus32::Q_32, 6).ok().unwrap()
    }

    /// A small random-ish R1CS shape with a KNOWN satisfying witness:
    /// M₀ = M₁ = I (identity), M₂ = diag(z∘z) — trivially satisfiable
    /// for any z; plus a shuffled variant to make it non-trivial.
    fn shape_with_witness(m: usize, ell: usize, seed: u64) -> (R1csQ32, Vec<u64>, Vec<u64>) {
        let mut xs = seed;
        let mut next = || {
            xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (xs >> 33) % Q
        };
        // M0, M1: random sparse (2 nonzeros per row); the witness is
        // then SOLVED: pick w freely, compute a = M0 z, b = M1 z, and
        // set M2's row j = the row that maps z to a[j]·b[j] (one
        // nonzero at the argmax position of a·b support).
        let m0: Vec<u64> = (0..m * m)
            .map(|i| {
                let r = i / m;
                let c = i % m;
                if c == r || c == (r + 1) % m {
                    1 + next() % 5
                } else {
                    0
                }
            })
            .collect();
        let m1: Vec<u64> = (0..m * m)
            .map(|i| {
                let r = i / m;
                let c = i % m;
                if c == r || c == (m + r - 1) % m {
                    1 + next() % 5
                } else {
                    0
                }
            })
            .collect();
        let x: Vec<u64> = (0..ell).map(|_| next() % 1000).collect();
        let w: Vec<u64> = (0..(m - ell - 1)).map(|_| next() % 1000).collect();
        let mut z = x.clone();
        z.push(1);
        z.extend_from_slice(&w);
        // Compute a = M0·z, b = M1·z, then build M2 rows mapping z ↦ a∘b:
        // row j: one nonzero at position c_j = j (the diagonal), value
        // (a[j]·b[j])·z[j]^{-1}.
        let a = {
            let shape0 = R1csQ32 {
                ell,
                m,
                mats: [m0.clone(), m1.clone(), vec![0; m * m]],
            };
            shape0.matvec(0, &z).unwrap()
        };
        let b = {
            let shape0 = R1csQ32 {
                ell,
                m,
                mats: [m0.clone(), m1.clone(), vec![0; m * m]],
            };
            shape0.matvec(1, &z).unwrap()
        };
        let mut m2 = vec![0u64; m * m];
        for j in 0..m {
            let target = (a[j] as u128 * b[j] as u128) % Q as u128;
            // find a nonzero z entry to hang the value on
            let mut c = j;
            while z[c] == 0 {
                c = (c + 1) % m;
            }
            let inv = mod_inv(z[c]);
            m2[j * m + c] = ((target as u128 * inv as u128) % Q as u128) as u64;
        }
        (
            R1csQ32 {
                ell,
                m,
                mats: [m0, m1, m2],
            },
            x,
            w,
        )
    }

    fn mod_inv(a: u64) -> u64 {
        let mut result: u64 = 1;
        let mut base = a % Q;
        let mut exp = Q - 2;
        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % Q as u128) as u64;
            }
            base = ((base as u128 * base as u128) % Q as u128) as u64;
            exp >>= 1;
        }
        result
    }

    #[test]
    fn theta_roundtrip_and_norm() {
        let ring = ring();
        for k in [2u64, 4, 16, 256] {
            let theta = ThetaK::new(k).unwrap();
            assert!(theta.digits <= 64);
            // The roundtrip θ_k ∘ θ_k^{-1} = id over sample values.
            for c in [0u64, 1, 2, Q - 1, Q / 2, 12345, 3_000_000_000] {
                let lift = theta.embed(&ring, c);
                assert!(
                    (lift.infinity_norm() as u64) < k,
                    "k={k}: lift norm {} >= {k}",
                    lift.infinity_norm()
                );
                let back = theta.project(&ring, &lift);
                assert_eq!(back, c % Q, "k={k} roundtrip failed at c={c}");
            }
        }
    }

    #[test]
    fn fq2_q32_field_axioms() {
        let a = Fq2Q32::from_pair(3, 5);
        let b = Fq2Q32::from_pair(7, 11);
        let c = Fq2Q32::from_pair(13, 2);
        // Associativity / commutativity / distributivity spot checks.
        assert_eq!(a.mul(&b), b.mul(&a));
        assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));
        assert_eq!(
            a.mul(&b.add(&c)),
            a.mul(&b).add(&a.mul(&c))
        );
        // Inverse.
        let inv = a.inverse().unwrap();
        assert_eq!(a.mul(&inv), Fq2Q32::ONE);
        // Norm/conjugate identity: z·conj(z) = norm (a base element).
        assert_eq!(a.mul(&a.conj()).c1, 0);
        // The non-residue: 5 is a non-square mod q.
        assert_eq!(
            {
                let mut acc: u64 = 1;
                for _ in 0..((Q - 1) / 2) {
                    acc = (acc * 5) % Q;
                }
                acc
            },
            Q - 1
        );
    }

    #[test]
    fn q2_sumcheck_differential() {
        // The engine vs the naive cube semantics: the proof verifies
        // against the naive sum, and the terminal factor claims equal
        // the factors' MLE evaluations at the derived point (the
        // big-endian convention — the engine's round order).
        let mut xs = 99u64;
        let mut next = || {
            xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1);
            (xs >> 33) % Q
        };
        let n = 4usize;
        let f0: Vec<Fq2Q32> = (0..1 << n).map(|_| Fq2Q32::from_pair(next(), next())).collect();
        let f1: Vec<Fq2Q32> = (0..1 << n).map(|_| Fq2Q32::from_pair(next(), next())).collect();
        let mut vp = Q2VirtualPoly::new(n);
        let i0 = vp.add_factor(f0.clone()).unwrap();
        let i1 = vp.add_factor(f1.clone()).unwrap();
        vp.add_term(Fq2Q32::from_u64(3), vec![i0, i1]).unwrap();
        // Naive sum over the cube.
        let naive: Fq2Q32 = (0..1 << n)
            .map(|p| Fq2Q32::from_u64(3).mul(&f0[p]).mul(&f1[p]))
            .fold(Fq2Q32::ZERO, |a, b| a.add(&b));
        let mut tr = Transcript::new_default(b"q2-test");
        let out = q2_sumcheck_prove(&vp, naive, &mut tr).unwrap();
        // The terminal factor claims equal the direct MLE evaluations
        // at the engine's point (the big-endian pairing).
        let mle_at = |table: &[Fq2Q32], point: &[Fq2Q32]| -> Fq2Q32 {
            let nv = point.len();
            let mut acc = Fq2Q32::ZERO;
            for (idx, v) in table.iter().enumerate() {
                let mut w = Fq2Q32::ONE;
                for (var, pv) in point.iter().enumerate() {
                    let bit = (idx >> (nv - 1 - var)) & 1;
                    let term = if bit == 1 { *pv } else { Fq2Q32::ONE.sub(pv) };
                    w = w.mul(&term);
                }
                acc = acc.add(&v.mul(&w));
            }
            acc
        };
        assert_eq!(out.factor_claims[i0], mle_at(&f0, &out.challenges));
        assert_eq!(out.factor_claims[i1], mle_at(&f1, &out.challenges));
        // The proof verifies against the naive claim with the
        // terminal check derived from the factor claims.
        let mut vt = Transcript::new_default(b"q2-test");
        let expected = Fq2Q32::from_u64(3)
            .mul(&out.factor_claims[i0])
            .mul(&out.factor_claims[i1]);
        let verdict = out
            .proof
            .verify(n, 2, naive, &mut vt, Some(expected))
            .unwrap();
        assert_eq!(verdict.point, out.challenges);
        // A wrong claim rejects.
        let mut vt2 = Transcript::new_default(b"q2-test");
        assert!(out
            .proof
            .verify(n, 2, naive.add(&Fq2Q32::ONE), &mut vt2, None)
            .is_err());
    }

    #[test]
    fn bridge_honest_roundtrip() {
        let ring = ring();
        let (shape, x, w) = shape_with_witness(8, 3, 42);
        let theta = ThetaK::new(4).unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: shape.m,
            norm_bound: 1 << 20,
        };
        let pk = AjtaiPublicKey::from_seed(params, [55u8; 32]).unwrap();
        let mut tr = Transcript::new_default(b"cyclo-r1cs-bridge");
        let proof = prove_r1cs_bridge(&ring, &pk, &shape, &x, &w, &theta, &mut tr).unwrap();
        // The honest shape invariants:
        // (a) the θ-lifts project back to the d_i.
        {
            let lift0 = [
                RingElement::from_coeffs(&ring, proof.claim.d_lift[0][0].clone()),
                RingElement::from_coeffs(&ring, proof.claim.d_lift[0][1].clone()),
            ];
            let back = theta.project_pair(&ring, &lift0);
            assert_eq!(back, proof.claim.d[0]);
        }
        // (b) the prefix evaluation: e = MLE[(x,1)](v) — recompute.
        {
            let mut prefix = x.clone();
            prefix.push(1);
            assert_eq!(mle_eval_q2(&prefix, &proof.claim.v), proof.claim.e);
        }
        // (c) the sum-check verifies with the terminal check.
        let mut vt = Transcript::new_default(b"cyclo-r1cs-bridge");
        let claim = verify_r1cs_bridge(&ring, &shape, &x, &proof, &mut vt).unwrap();
        let _ = claim;
    }

    #[test]
    fn bridge_unsatisfied_witness_rejected() {
        let ring = ring();
        let (shape, x, w) = shape_with_witness(8, 3, 7);
        let theta = ThetaK::new(4).unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: shape.m,
            norm_bound: 1 << 20,
        };
        let pk = AjtaiPublicKey::from_seed(params, [55u8; 32]).unwrap();
        // Corrupt the witness — the prove fails closed at the
        // satisfaction self-check.
        let mut w_bad = w.clone();
        w_bad[0] = (w_bad[0] + 1) % Q;
        let mut tr = Transcript::new_default(b"cyclo-r1cs-bridge");
        assert!(prove_r1cs_bridge(&ring, &pk, &shape, &x, &w_bad, &theta, &mut tr).is_err());
    }

    /// The skip-Π^ext wiring: the bridge's lifted witness has norm < k
    /// ≤ b (the chunk base), so it feeds `CycloAccumulator::new`
    /// (the range check ≤ 2^20 passes trivially) — the extension
    /// commitment step is SKIPPED (the paper's remark).
    #[test]
    fn bridge_output_feeds_cyclo_accumulator() {
        use crate::cyclo::CycloAccumulator;
        let ring = ring();
        let (shape, x, w) = shape_with_witness(8, 3, 11);
        // k = 4 ≤ b = 2^{8-1} = 128 (chunk_log 8): the skip regime.
        let theta = ThetaK::new(4).unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: shape.m,
            norm_bound: 1 << 26,
        };
        let pk = AjtaiPublicKey::from_seed(params, [55u8; 32]).unwrap();
        let mut tr = Transcript::new_default(b"cyclo-r1cs-bridge");
        let _proof = prove_r1cs_bridge(&ring, &pk, &shape, &x, &w, &theta, &mut tr).unwrap();
        // The lifted witness (recomputed — the prover's z'):
        let mut z = x.clone();
        z.push(1);
        z.extend_from_slice(&w);
        let z_lift: Vec<RingElement> = z.iter().map(|&c| theta.embed(&ring, c)).collect();
        // The norm is already below the chunk base b = 128 — no
        // extension commitment needed.
        for e in &z_lift {
            assert!((e.infinity_norm() as u64) < 4);
        }
        // It initializes the Cyclo accumulator directly (the range
        // gate ≤ 2^20 and the norm budget β* both pass).
        let acc = CycloAccumulator::new(&pk, &z_lift);
        assert!(acc.is_ok(), "the bridge's output must feed the accumulator");
    }

    /// The θ_k embedding's digit-level consistency with the chunk
    /// machinery: at `k = 2^{chunk_log}` the lift's coefficients ARE
    /// the base-k digits (both substrates agree per-coefficient), and
    /// the chunk roundtrip on the lift recovers it exactly.
    #[test]
    fn theta_embeds_like_chunks() {
        let ring = ring();
        let theta = ThetaK::new(256).unwrap();
        for c in [0u64, 1, 255, 256, 65_535, 1_000_000, Q - 1] {
            let lift = theta.embed(&ring, c);
            let back = theta.project(&ring, &lift);
            assert_eq!(back, c % Q);
            // The per-coefficient digits: coefficient j (j < digits) is
            // the j-th base-256 digit of c.
            let mut rem = c % Q;
            for j in 0..theta.digits {
                let digit = (rem % 256) as u32;
                assert_eq!(lift.coeffs()[j], digit, "digit {j} of {c}");
                rem /= 256;
            }
            assert_eq!(rem, 0);
            // The chunk roundtrip on the lift (the shared substrate —
            // chunk_element/unchunk agree on the digit decomposition).
            let chunks = chunk_element(&ring, &lift, 8);
            let rec = crate::cyclo::unchunk_elements(&ring, &chunks, 8);
            assert_eq!(rec.coeffs(), lift.coeffs());
        }
    }
}
