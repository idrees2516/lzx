//! SALSAA A5 (ePrint 2025/2124): the committed-AIR application (Fig. 7)
//! and the folding step (§7) — Wave 7 item 7.11, part 2.
//!
//! * **Π_air prover/verifier (Fig. 7)**: the tiny committed AIR has m
//!   rows (power of two), t trace columns, the degree-2 transition
//!   `f(W_i) = W_i[1] − W_i[0]²` (Fibonacci-style) and a boundary set C.
//!   The column tables are `V = [W, shift(W)]` (2t tables over the m-row
//!   hypercube, cyclic shift). The prover draws (η, α, θ), builds the
//!   three claim families and proves them in ONE combined degree-3 ring
//!   sumcheck:
//!   * transition: `Σ_z eq(η,z)·(1 − eq(z,1))·f_col(z) = 0` with
//!     `f_col = MLE[V_1] − MLE[V_0]²` (two groups);
//!   * shift: `Σ_z [θ̃(z)·V0_α(z) − corr(z)·V1_α(z)] = 0` where
//!     `θ̃ = (θ^i)_i`, `corr = (θ, …, θ^{m−1}, 1)` with the
//!     `(θ^m − 1)·eq(z,1)` wrap correction folded into the last row;
//!   * boundary: `Σ_z eq(z, bin(i_k))·V_{j_k}(z) = u_k`.
//!
//!   The verifier recomputes every public table from (η, α, θ) and the
//!   terminal identity substitutes the prover's 2t column openings.
//! * **folding step (§7, Lova-style)**: sample r; fold
//!   `w* = w1 + r·w2`, `C* = C1 + r·C2`, `rows* = rows1` (shared public
//!   rows), `targets* = t1 + r·t2` (bilinearity), norm growth
//!   `β* = ⌈√(2(β1² + β2²))⌉`; the b-decomposition re-shortening is the
//!   caller's gadget layer.

use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_ring::{RingConfig, RingElement};

use crate::ring_sc::{
    challenge_ring_elt, eq_table_ring, mle_eval_ring, ring_dot, ring_sc_prove, ring_sc_verify,
    ProductClaim, RingScError,
};
use crate::salsaa::{SalsaInstance, SalsaaError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AirError {
    Salsaa(SalsaaError),
    RingSc(RingScError),
    Transcript(TranscriptError),
    /// The transition constraint failed on the trace (not honest).
    TraceNotHonest,
    /// A boundary constraint failed.
    BoundaryFailed,
    /// The terminal identity failed.
    TerminalFailed,
    /// Challenge replay mismatch (tampered proof).
    ChallengeMismatch,
    Shape {
        expected: usize,
        got: usize,
    },
}

impl From<SalsaaError> for AirError {
    fn from(e: SalsaaError) -> Self {
        AirError::Salsaa(e)
    }
}
impl From<RingScError> for AirError {
    fn from(e: RingScError) -> Self {
        AirError::RingSc(e)
    }
}
impl From<lattice_ring::RingError> for AirError {
    fn from(e: lattice_ring::RingError) -> Self {
        AirError::RingSc(RingScError::Ring(e))
    }
}
impl From<TranscriptError> for AirError {
    fn from(e: TranscriptError) -> Self {
        AirError::Transcript(e)
    }
}

// ---------------------------------------------------------------------------
// The AIR statement
// ---------------------------------------------------------------------------

/// A boundary constraint `(row i_k, column j_k, value u_k)`.
#[derive(Clone, Debug)]
pub struct BoundaryConstraint {
    pub row: usize,
    pub col: usize,
    pub value: RingElement,
}

/// A tiny committed AIR: m rows (power of two), t trace columns, the
/// degree-2 transition `f(W_i) = W_i[1] − W_i[0]²`, boundary set C.
#[derive(Clone, Debug)]
pub struct AirParams {
    pub m: usize,
    pub t: usize,
    pub boundary: Vec<BoundaryConstraint>,
}

impl AirParams {
    pub fn mu(&self) -> usize {
        self.m.trailing_zeros() as usize
    }
}

/// Check the transition constraint on one trace row pair:
/// `f(W_i) = W_i[1] − W_i[0]²` evaluated against the NEXT row's W[0]
/// (the trace generator maintains `W_{i+1}[0] = W_i[0]²`).
fn transition_ok(row: &RingElement, next_w0: &RingElement) -> Result<bool, AirError> {
    let lhs = row.sub(next_w0)?;
    Ok(lhs.is_zero())
}

/// Generate an honest trace: `x_{i+1} = x_i²` in column 0 from the first
/// boundary value (or a small seed value when no boundary binds col 0).
pub fn air_gen_trace(
    ring: &RingConfig,
    params: &AirParams,
    seed: &[u8],
) -> Result<Vec<Vec<RingElement>>, AirError> {
    let b0 = params
        .boundary
        .iter()
        .find(|b| b.row == 0 && b.col == 0)
        .map(|b| b.value.clone());
    let mut x = match b0 {
        Some(v) => v,
        None => {
            let bytes = Transcript::xof(b"air-seed-x", seed, 4 * ring.n());
            let coeffs: Vec<u32> = bytes
                .chunks(4)
                .take(ring.n())
                .map(|c| {
                    let mut a = [0u8; 4];
                    a.copy_from_slice(&c[..4]);
                    u32::from_le_bytes(a) % 9
                })
                .collect();
            RingElement::from_coeffs(ring, coeffs)
        }
    };
    let mut w = Vec::with_capacity(params.m);
    for _ in 0..params.m {
        // column 1: a per-row small value; the transition compares
        // W_i[1] with W_{i+1}[0] = W_i[0]^2 — generate the trace so
        // W_i[1] = x^2 of the CURRENT row (checked against next row's W0)
        let next = x.mul(&x)?;
        w.push(vec![x.clone(), next.clone()]);
        x = next;
    }
    Ok(w)
}

/// `V = [W, shift(W)]` as 2t column tables over the m-row hypercube.
pub fn air_build_tables(params: &AirParams, w: &[Vec<RingElement>]) -> Vec<Vec<RingElement>> {
    let m = params.m;
    let t = params.t;
    let mut cols = Vec::with_capacity(2 * t);
    for j in 0..t {
        cols.push((0..m).map(|i| w[i][j].clone()).collect());
    }
    for j in 0..t {
        cols.push((0..m).map(|i| w[(i + 1) % m][j].clone()).collect());
    }
    cols
}

/// The Π_air proof: the combined degree-3 sumcheck + the challenges + the
/// 2t column openings at the final point.
#[derive(Clone, Debug)]
pub struct AirProof {
    pub sumcheck: crate::ring_sc::RingScProof,
    pub point: Vec<u32>,
    pub eta: Vec<u32>,
    pub alpha: Vec<RingElement>,
    pub theta: RingElement,
    pub col_openings: Vec<RingElement>,
}

/// The public tables derived from the challenges (shared by prove and
/// verify so the two cannot drift).
struct AirPublicTables {
    trans_weight: Vec<RingElement>,
    neg_trans_weight: Vec<RingElement>,
    theta_pow: Vec<RingElement>,
    neg_corr: Vec<RingElement>,
    boundary_sels: Vec<Vec<RingElement>>,
}

fn air_public_tables(
    ring: &RingConfig,
    params: &AirParams,
    eta: &[u32],
    theta: &RingElement,
) -> Result<AirPublicTables, AirError> {
    let m = params.m;
    let mu = params.mu();
    // eq(eta, z) and eq(z, 1) (the last-row selector)
    let eq_eta = eq_table_ring(ring, eta);
    let ones: Vec<u32> = vec![1; mu];
    let last_row_sel = eq_table_ring(ring, &ones);
    // trans_weight = eq_eta − eq_eta·eq(z,1)
    let mut trans_weight = Vec::with_capacity(m);
    for z in 0..m {
        trans_weight.push(eq_eta[z].sub(&eq_eta[z].mul(&last_row_sel[z])?)?);
    }
    let neg_trans_weight: Vec<RingElement> = trans_weight.iter().map(|x| x.neg()).collect();
    // theta powers and the cyclic correction table
    let mut theta_pow = Vec::with_capacity(m);
    let mut cur = ring.one();
    for _ in 0..m {
        theta_pow.push(cur.clone());
        cur = cur.mul(theta)?;
    }
    let mut corr: Vec<RingElement> = (0..m).map(|z| theta_pow[(z + 1) % m].clone()).collect();
    corr[m - 1] = ring.one();
    let neg_corr: Vec<RingElement> = corr.iter().map(|x| x.neg()).collect();
    // boundary selectors
    let mut boundary_sels = Vec::with_capacity(params.boundary.len());
    for b in &params.boundary {
        let mut sel = vec![ring.zero(); m];
        sel[b.row] = ring.one();
        boundary_sels.push(sel);
    }
    Ok(AirPublicTables {
        trans_weight,
        neg_trans_weight,
        theta_pow,
        neg_corr,
        boundary_sels,
    })
}

/// Π_air prover (Fig. 7).
pub fn air_prove(
    ring: &RingConfig,
    params: &AirParams,
    w: &[Vec<RingElement>],
    transcript: &mut Transcript,
) -> Result<AirProof, AirError> {
    let (m, t, mu) = (params.m, params.t, params.mu());
    let cols = air_build_tables(params, w);
    // honesty of the trace: transition + boundary. The transition applies
    // to rows 0..m-2 ONLY — the paper's eq(η,z)·(1 − eq(z,1)) mask zeroes
    // the last row where the cyclic shift wraps.
    for i in 0..m - 1 {
        let next_w0 = &cols[t][i]; // shift(W)[0] at row i = W[(i+1)%m][0]
        if !transition_ok(&w[i][1], next_w0)? {
            return Err(AirError::TraceNotHonest);
        }
    }
    for b in &params.boundary {
        if cols[b.col][b.row] != b.value {
            return Err(AirError::BoundaryFailed);
        }
    }
    // challenges
    let eta = (0..mu)
        .map(|_| crate::ring_sc::challenge_zq(transcript, b"salsaa:eta", ring.modulus.q))
        .collect::<Result<Vec<_>, _>>()?;
    let alpha = (0..t)
        .map(|_| challenge_ring_elt(transcript, b"salsaa:alpha", ring))
        .collect::<Result<Vec<_>, _>>()?;
    let theta = challenge_ring_elt(transcript, b"salsaa:theta", ring)?;
    // public tables
    let pubt = air_public_tables(ring, params, &eta, &theta)?;
    // alpha-weighted column sums
    let v0_alpha: Vec<RingElement> = {
        let mut out = Vec::with_capacity(m);
        for i in 0..m {
            let mut acc = ring.zero();
            for j in 0..t {
                acc = acc.add(&cols[j][i].mul(&alpha[j])?)?;
            }
            out.push(acc);
        }
        out
    };
    let v1_alpha: Vec<RingElement> = {
        let mut out = Vec::with_capacity(m);
        for i in 0..m {
            let mut acc = ring.zero();
            for j in 0..t {
                acc = acc.add(&cols[t + j][i].mul(&alpha[j])?)?;
            }
            out.push(acc);
        }
        out
    };
    // groups (degree 3: the transition's quadratic term)
    let mut groups: Vec<ProductClaim> = Vec::new();
    let mut combiners: Vec<RingElement> = Vec::new();
    // transition: Σ_z trans_weight·(V1 − V0²) = 0
    groups.push(ProductClaim {
        tables: vec![pubt.trans_weight.clone(), cols[1].clone()],
        value: ring.zero(),
    });
    combiners.push(ring.one());
    groups.push(ProductClaim {
        tables: vec![
            pubt.neg_trans_weight.clone(),
            cols[0].clone(),
            cols[0].clone(),
        ],
        value: ring.zero(),
    });
    combiners.push(ring.one());
    // shift: Σ_z theta_pow·V0_alpha − corr·V1_alpha = 0
    groups.push(ProductClaim {
        tables: vec![pubt.theta_pow.clone(), v0_alpha],
        value: ring.zero(),
    });
    combiners.push(ring.one());
    groups.push(ProductClaim {
        tables: vec![pubt.neg_corr.clone(), v1_alpha],
        value: ring.zero(),
    });
    combiners.push(ring.one());
    // boundary claims
    for (bidx, b) in params.boundary.iter().enumerate() {
        groups.push(ProductClaim {
            tables: vec![pubt.boundary_sels[bidx].clone(), cols[b.col].clone()],
            value: b.value.clone(),
        });
        combiners.push(ring.one());
    }
    let sumcheck = ring_sc_prove(ring, &groups, &combiners, transcript)?;
    let point = sumcheck.point.clone();
    let col_openings = cols
        .iter()
        .map(|c| mle_eval_ring(c, &point))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AirProof {
        sumcheck,
        point,
        eta,
        alpha,
        theta,
        col_openings,
    })
}

/// Π_air verifier: re-derive challenges and public tables, check the
/// combined sumcheck and the terminal identity with the sent column
/// openings.
pub fn air_verify(
    ring: &RingConfig,
    params: &AirParams,
    proof: &AirProof,
    transcript: &mut Transcript,
) -> Result<(), AirError> {
    let (m, t, mu) = (params.m, params.t, params.mu());
    // replay the challenges
    let eta = (0..mu)
        .map(|_| crate::ring_sc::challenge_zq(transcript, b"salsaa:eta", ring.modulus.q))
        .collect::<Result<Vec<_>, _>>()?;
    let alpha = (0..t)
        .map(|_| challenge_ring_elt(transcript, b"salsaa:alpha", ring))
        .collect::<Result<Vec<_>, _>>()?;
    let theta = challenge_ring_elt(transcript, b"salsaa:theta", ring)?;
    if eta != proof.eta || alpha != proof.alpha || theta != proof.theta {
        return Err(AirError::ChallengeMismatch);
    }
    let pubt = air_public_tables(ring, params, &eta, &theta)?;
    // combined target: all group values are zero except the boundaries
    let mut target = ring.zero();
    for b in &params.boundary {
        target = target.add(&b.value)?;
    }
    let last = ring_sc_verify(ring, 3, mu, &target, &proof.sumcheck, transcript)?;
    // terminal: recompute the combined product at the point with the
    // sent column openings substituting the private tables
    let point = &proof.point;
    let co = &proof.col_openings;
    if co.len() != 2 * t {
        return Err(AirError::Shape {
            expected: 2 * t,
            got: co.len(),
        });
    }
    let mut total = ring.zero();
    // group 0: trans_weight · V1
    let tw = mle_eval_ring(&pubt.trans_weight, point)?;
    total = total.add(&tw.mul(&co[1])?)?;
    // group 1: neg_trans_weight · V0 · V0
    let ntw = mle_eval_ring(&pubt.neg_trans_weight, point)?;
    total = total.add(&ntw.mul(&co[0])?.mul(&co[0])?)?;
    // group 2: theta_pow · V0_alpha
    let tp = mle_eval_ring(&pubt.theta_pow, point)?;
    let v0a = {
        let mut acc = ring.zero();
        for j in 0..t {
            acc = acc.add(&co[j].mul(&alpha[j])?)?;
        }
        acc
    };
    total = total.add(&tp.mul(&v0a)?)?;
    // group 3: neg_corr · V1_alpha
    let nc = mle_eval_ring(&pubt.neg_corr, point)?;
    let v1a = {
        let mut acc = ring.zero();
        for j in 0..t {
            acc = acc.add(&co[t + j].mul(&alpha[j])?)?;
        }
        acc
    };
    total = total.add(&nc.mul(&v1a)?)?;
    // boundary groups
    for (bidx, b) in params.boundary.iter().enumerate() {
        let sel = mle_eval_ring(&pubt.boundary_sels[bidx], point)?;
        total = total.add(&sel.mul(&co[b.col])?)?;
    }
    if total != last {
        return Err(AirError::TerminalFailed);
    }
    let _ = m;
    Ok(())
}

// ---------------------------------------------------------------------------
// The folding step (§7)
// ---------------------------------------------------------------------------

/// The folded instance (witness + linear data).
#[derive(Clone, Debug)]
pub struct FoldedInstance {
    pub rows: Vec<Vec<RingElement>>,
    pub targets: Vec<RingElement>,
    pub com: Vec<RingElement>,
    pub beta: u64,
    pub w: Option<Vec<RingElement>>,
}

/// One folding step (§7, Lova-style): sample r; fold
/// `w* = w1 + r·w2`, `C* = C1 + r·C2`, `targets* = t1 + r·t2`
/// (so `⟨row, w*⟩ = t*` by bilinearity when both inputs are honest),
/// norm growth `β* = ⌈√(2(β1² + β2²))⌉`. The cross-term cancellation
/// belongs to quadratic R1CS folding (handled at the AIR level via the
/// relaxed-witness error term — see the paper's §7 and the lab gap
/// ledger).
pub fn salsa_fold(
    ring: &RingConfig,
    inst1: &SalsaInstance,
    inst2: &SalsaInstance,
    transcript: &mut Transcript,
) -> Result<FoldedInstance, AirError> {
    let r = challenge_ring_elt(transcript, b"salsaa:fold-r", ring)?;
    let w1 = inst1.w.as_ref().ok_or(AirError::Shape {
        expected: 1,
        got: 0,
    })?;
    let w2 = inst2.w.as_ref().ok_or(AirError::Shape {
        expected: 1,
        got: 0,
    })?;
    if w1.len() != w2.len() || inst1.rows.len() != inst2.rows.len() {
        return Err(AirError::Shape {
            expected: w1.len(),
            got: w2.len(),
        });
    }
    let w_star: Vec<RingElement> = w1
        .iter()
        .zip(w2.iter())
        .map(|(a, b)| a.add(&b.mul(&r)?))
        .collect::<Result<Vec<_>, _>>()?;
    let com_star: Vec<RingElement> = inst1
        .com
        .iter()
        .zip(inst2.com.iter())
        .map(|(a, b)| a.add(&b.mul(&r)?))
        .collect::<Result<Vec<_>, _>>()?;
    let targets_star: Vec<RingElement> = inst1
        .targets
        .iter()
        .zip(inst2.targets.iter())
        .map(|(a, b)| a.add(&b.mul(&r)?))
        .collect::<Result<Vec<_>, _>>()?;
    let beta_sq = 2u64
        .saturating_mul(inst1.beta.saturating_mul(inst1.beta))
        .saturating_add(2u64.saturating_mul(inst2.beta.saturating_mul(inst2.beta)));
    let beta_star = (beta_sq as f64).sqrt().ceil() as u64;
    Ok(FoldedInstance {
        rows: inst1.rows.clone(),
        targets: targets_star,
        com: com_star,
        beta: beta_star,
        w: Some(w_star),
    })
}

/// Check the folded instance's constraints against the folded witness
/// (completeness of the fold).
pub fn salsa_fold_honest(folded: &FoldedInstance) -> Result<bool, AirError> {
    let w = folded.w.as_ref().ok_or(AirError::Shape {
        expected: 1,
        got: 0,
    })?;
    for (row, t) in folded.rows.iter().zip(&folded.targets) {
        if ring_dot(row, w)? != *t {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingConfig {
        lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, 3)
            .ok()
            .unwrap()
    }

    fn small_vec(ring: &RingConfig, m: usize, tag: &[u8], span: u32) -> Vec<RingElement> {
        (0..m)
            .map(|i| {
                let bytes = Transcript::xof(
                    b"air-test",
                    &[tag, &(i as u32).to_le_bytes()].concat(),
                    4 * ring.n(),
                );
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
    fn air_end_to_end_and_tampered() {
        let ring = ring();
        let params = AirParams {
            m: 8,
            t: 2,
            boundary: vec![BoundaryConstraint {
                row: 0,
                col: 0,
                value: ring.constant(3),
            }],
        };
        let w = air_gen_trace(&ring, &params, b"air-seed").ok().unwrap();
        let mut t = Transcript::new_default(b"lzx-salsaa-air");
        let proof = air_prove(&ring, &params, &w, &mut t)
            .map_err(|e| panic!("prove: {:?}", e))
            .unwrap();
        let mut vt = Transcript::new_default(b"lzx-salsaa-air");
        assert!(air_verify(&ring, &params, &proof, &mut vt).is_ok());
        // tampered column opening rejected at the terminal
        let mut bad = proof.clone();
        bad.col_openings[0] = bad.col_openings[0].add(&ring.one()).ok().unwrap();
        let mut vt2 = Transcript::new_default(b"lzx-salsaa-air");
        assert!(air_verify(&ring, &params, &bad, &mut vt2).is_err());
        // tampered sumcheck rounds rejected
        let mut bad2 = proof.clone();
        let r0 = bad2.sumcheck.rounds[0][0].clone();
        bad2.sumcheck.rounds[0][0] = r0.add(&ring.one()).ok().unwrap();
        let mut vt3 = Transcript::new_default(b"lzx-salsaa-air");
        assert!(air_verify(&ring, &params, &bad2, &mut vt3).is_err());
        // dishonest trace refused by the prover
        let mut w_bad = w.clone();
        w_bad[2][0] = w_bad[2][0].add(&ring.one()).ok().unwrap();
        let mut t4 = Transcript::new_default(b"lzx-salsaa-air");
        assert!(air_prove(&ring, &params, &w_bad, &mut t4).is_err());
    }

    #[test]
    fn folding_completeness_and_norm_growth() {
        let ring = ring();
        let m = 4;
        let rows = vec![small_vec(&ring, m, b"fr0", 6)];
        let w1 = small_vec(&ring, m, b"fw1", 4);
        let w2 = small_vec(&ring, m, b"fw2", 4);
        let (inst1, _pk1) = SalsaInstance::create(&ring, &w1, &rows, 128, [11u8; 32])
            .ok()
            .unwrap();
        let (inst2, _pk2) = SalsaInstance::create(&ring, &w2, &rows, 128, [12u8; 32])
            .ok()
            .unwrap();
        let mut t = Transcript::new_default(b"lzx-salsaa-fold");
        let folded = salsa_fold(&ring, &inst1, &inst2, &mut t).ok().unwrap();
        assert!(salsa_fold_honest(&folded).ok().unwrap());
        // commitment homomorphism: com* = com1 + r·com2 held by construction;
        // norm growth is tracked
        assert!(folded.beta > inst1.beta);
    }
}
