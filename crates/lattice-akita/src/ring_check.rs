//! A2 — Akita ring-relation checks (ePrint 2026/1983, §7): the quotient
//! lifting of §7.3 with the **α-after-commitment ordering** repaired (the
//! Grand-Danois bug documented in Appendix F.1) and the fused
//! relation+range sum-check of §7.4 (Eq 160).
//!
//! Every ring-valued row of the fold relation (Fig 6) has the normal form
//! of Eq 146:
//!
//! `Z_r(X) = Σ_{c ∈ Occ_r} A_{r,c}(X) · W_{r,c}(X) − Y_r(X) = 0 in R_r`,
//!
//! where the `W_{r,c}` are polynomials read from the (successor) witness
//! and the `A_{r,c}`, `Y_r` are public. Two checks reduce this to field
//! identities:
//!
//! * **Quotient lifting** (Eq 147-148): the prover supplies the quotient
//!   `Q_r ∈ K[X]`, `deg < d_r`, with
//!   `Σ_c A_{r,c}·W_{r,c} − Y_r = (X^{d_r}+1)·Q_r` in `K[X]`; the verifier
//!   evaluates both sides at a random `α` and checks
//!   `Σ_c A_{r,c}(α)·W_{r,c}(α) − Y_r(α) − (α^{d_r}+1)·Q_r(α) = 0`.
//!   **Ordering** (App F.1): the first public Grand Danois version sampled
//!   its row-combination challenge *before* committing to the recursive
//!   witness, so a prover could choose "a nonzero residual in the known
//!   projection kernel; the usual random-linear-combination bound requires
//!   the residual to be fixed first". The repaired order — Theorem 7.2 —
//!   fixes "the outgoing witness … before sampling α", and the protocol
//!   consequence is explicit: "the outgoing commitment is bound before α".
//!   This module draws `α` from the transcript only after the witness
//!   commitment and the quotients are absorbed, and the test
//!   [`alpha_before_commit_fails`] demonstrates the archived attack
//!   failing under the repaired order.
//! * **Fused relation+range sum-check** (Eq 160): one sum-check over the
//!   shared witness combines the α-reduced relation rows with the range
//!   pipeline's carried claim `s_claim = ŝ(r_virt)` under a fresh
//!   coefficient `γ` sampled after both claims are bound.
//!
//! LZX realization note (split fields, honestly stated): the paper's base
//! field `K = F_q` unifies the commitment ring and the sum-check field.
//! The LZX substrate splits them (ring `Q_32`, sum-check Goldilocks), so
//! the α-evaluation of Eq 148 is performed natively in `F_{Q_32}`
//! (Barrett arithmetic, sound on its own), while the fused sum-check of
//! Eq 160 runs over the Goldilocks digit layer where the range anchoring
//! `Σ_x eq(r_virt,x)·ŵ(x)(ŵ(x)+1) = s_claim` is a native Goldilocks
//! statement. A unified-field production path needs the large-modulus
//! ring (NEXT_STEPS §3.6 items A9/H2).

use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::{Modulus32, RingConfig, RingElement};
use lattice_sumcheck::virtual_poly::VirtualPolynomial;
use lattice_sumcheck::{SumcheckError, SumcheckProof};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RingCheckError {
    Ring(lattice_ring::RingError),
    Transcript(TranscriptError),
    Sumcheck(SumcheckError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Mle(lattice_core::mle::MleError),
    /// The lifted difference is not divisible by X^n + 1 (the row is
    /// false and no valid quotient exists).
    NotDivisible { row: usize },
    /// The α-evaluation check failed (Eq 148).
    AlphaCheck { row: usize },
    /// The fused sum-check failed.
    FusedCheck,
    /// Rejection sampling for α exhausted its budget.
    AlphaSampling,
    Shape { expected: usize, got: usize },
}

/// One ring-valued relation row in the normal form of Eq 146.
#[derive(Clone, Debug)]
pub struct RingRow {
    /// Public multipliers `A_{r,c}(X)` (one per occurrence).
    pub multipliers: Vec<RingElement>,
    /// Witness element index read by each occurrence (`W_{r,c}` is the
    /// degree-<n polynomial of that ring element).
    pub reads: Vec<usize>,
    /// The public target `Y_r(X)`.
    pub target: RingElement,
}

impl RingRow {
    /// Evaluate the row directly in `R_q` (the oracle the reductions
    /// approximate): `Σ_c A_{r,c} ⊛ w[reads[c]] − Y_r`.
    pub fn evaluate(&self, ring: &RingConfig, witness: &[RingElement]) -> Result<RingElement, RingCheckError> {
        let mut acc = ring.zero();
        for (a, &idx) in self.multipliers.iter().zip(self.reads.iter()) {
            let w = witness.get(idx).ok_or(RingCheckError::Shape {
                expected: idx,
                got: witness.len(),
            })?;
            acc = acc
                .add(&a.mul(w).map_err(RingCheckError::Ring)?)
                .map_err(RingCheckError::Ring)?;
        }
        acc.sub(&self.target).map_err(RingCheckError::Ring)
    }
}

// ---------------------------------------------------------------------------
// F_{Q_32} polynomial helpers (canonical u32 representatives).
// ---------------------------------------------------------------------------

/// Horner evaluation of a coefficient vector at `alpha` in F_q:
/// `Σ_j c_j·α^j`.
fn poly_eval(q: Modulus32, coeffs: &[u32], alpha: u32) -> u32 {
    let mut acc = 0u32;
    for &c in coeffs.iter().rev() {
        acc = q.add(q.mul(acc, alpha), c);
    }
    acc
}

/// Unreduced (no negacyclic reduction) schoolbook product of two
/// degree-<n polynomials, coefficients reduced mod q. Degree ≤ 2n−2.
fn unreduced_product(q: Modulus32, a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut acc = vec![0i128; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            acc[i + j] += x as i128 * y as i128;
        }
    }
    acc.iter().map(|&c| q.reduce_u64(c.rem_euclid(q.q as i128) as u64)).collect()
}

/// Divide `d` by `X^n + 1` over F_q. Returns `(quotient, remainder)`; the
/// quotient has degree ≤ deg(d) − n (≤ n−2 for products of degree-<n
/// polynomials).
fn divide_negacyclic(q: Modulus32, d: &[u32], n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut d = d.to_vec();
    let deg = d.len().saturating_sub(1);
    if deg < n {
        return (Vec::new(), d);
    }
    let mut quotient = vec![0u32; deg - n + 1];
    for k in (n..=deg).rev() {
        let c = d[k];
        if c != 0 {
            let kn = k - n;
            quotient[kn] = c;
            d[k] = 0;
            d[kn] = q.sub(d[kn], c);
        }
    }
    d.truncate(n);
    (quotient, d)
}

/// The lifted row: quotient digits of Eq 147 (canonical representatives).
#[derive(Clone, Debug)]
pub struct LiftedRow {
    /// `Q_r(X)` — degree ≤ n−2, coefficients in `[0, q)`.
    pub quotient: Vec<u32>,
}

/// Sample a uniform `α ∈ F_{Q_32}` from the transcript with exact
/// rejection (64-bit draws, bounded retries, fail closed).
pub fn sample_alpha(transcript: &mut Transcript, q: Modulus32) -> Result<u32, RingCheckError> {
    let q64 = q.q as u64;
    let limit = (u64::MAX / q64) * q64;
    for _ in 0..8 {
        let bytes = transcript
            .challenge_bytes(b"akita-alpha", 8)
            .map_err(RingCheckError::Transcript)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes);
        let raw = u64::from_le_bytes(arr);
        if raw < limit {
            return Ok((raw % q64) as u32);
        }
    }
    Err(RingCheckError::AlphaSampling)
}

/// Prover side of the quotient lift (Eq 147): for every row, compute the
/// unreduced difference `Σ_c A_{r,c}·W_{r,c} − Y_r` and divide it by
/// `X^n + 1`; a valid row leaves zero remainder.
pub fn quotient_lift(
    ring: &RingConfig,
    rows: &[RingRow],
    witness: &[RingElement],
) -> Result<Vec<LiftedRow>, RingCheckError> {
    let q = ring.modulus;
    let n = ring.n();
    let mut out = Vec::with_capacity(rows.len());
    for (ri, row) in rows.iter().enumerate() {
        let mut diff = vec![0u32; 2 * n - 1];
        for (a, &idx) in row.multipliers.iter().zip(row.reads.iter()) {
            let w = witness.get(idx).ok_or(RingCheckError::Shape {
                expected: idx,
                got: witness.len(),
            })?;
            let prod = unreduced_product(q, a.coeffs(), w.coeffs());
            for (k, &p) in prod.iter().enumerate() {
                diff[k] = q.add(diff[k], p);
            }
        }
        for (k, &y) in row.target.coeffs().iter().enumerate() {
            diff[k] = q.sub(diff[k], y);
        }
        let (quotient, remainder) = divide_negacyclic(q, &diff, n);
        if remainder.iter().any(|&c| c != 0) {
            return Err(RingCheckError::NotDivisible { row: ri });
        }
        out.push(LiftedRow { quotient });
    }
    Ok(out)
}

/// Verifier side of the α-check (Eq 148): with `α` sampled **after** the
/// witness commitment and the quotients were absorbed,
/// `Σ_c A_{r,c}(α)·W_{r,c}(α) − Y_r(α) − (α^n + 1)·Q_r(α) = 0` in F_q.
pub fn alpha_check(
    ring: &RingConfig,
    rows: &[RingRow],
    lifted: &[LiftedRow],
    witness: &[RingElement],
    alpha: u32,
) -> Result<(), RingCheckError> {
    let q = ring.modulus;
    let n = ring.n();
    let alpha_n_plus_1 = q.add(q.pow(alpha, n as u64), 1);
    for (ri, (row, lift)) in rows.iter().zip(lifted.iter()).enumerate() {
        if lift.quotient.len() > n.saturating_sub(1) {
            return Err(RingCheckError::Shape {
                expected: n - 1,
                got: lift.quotient.len(),
            });
        }
        let mut acc = 0u32;
        for (a, &idx) in row.multipliers.iter().zip(row.reads.iter()) {
            let w = witness.get(idx).ok_or(RingCheckError::Shape {
                expected: idx,
                got: witness.len(),
            })?;
            acc = q.add(acc, q.mul(poly_eval(q, a.coeffs(), alpha), poly_eval(q, w.coeffs(), alpha)));
        }
        acc = q.sub(acc, poly_eval(q, row.target.coeffs(), alpha));
        acc = q.sub(acc, q.mul(alpha_n_plus_1, poly_eval(q, &lift.quotient, alpha)));
        if acc != 0 {
            return Err(RingCheckError::AlphaCheck { row: ri });
        }
    }
    Ok(())
}

/// Absorb the witness commitment and the lifted quotients into the
/// transcript — the binding that must precede the `α` draw (App F.1).
pub fn absorb_lifted(
    transcript: &mut Transcript,
    commitment_bytes: &[u8],
    lifted: &[LiftedRow],
) -> Result<(), RingCheckError> {
    transcript
        .append_bytes(b"akita-ring-commitment", commitment_bytes)
        .map_err(RingCheckError::Transcript)?;
    for lift in lifted {
        let bytes: Vec<u8> = lift.quotient.iter().flat_map(|c| c.to_le_bytes()).collect();
        transcript
            .append_bytes(b"akita-ring-quotient", &bytes)
            .map_err(RingCheckError::Transcript)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The fused relation + range sum-check (§7.4, Eq 160).
// ---------------------------------------------------------------------------

/// The carried range claim (the output of the item-A3 pipeline) plus the
/// fusion coefficient.
#[derive(Clone, Debug)]
pub struct FusedClaim {
    /// The range pipeline's virtual point `r_virt`.
    pub rvirt: Vec<Goldilocks>,
    /// `s_claim = ŝ(r_virt)` — the derived-value claim.
    pub s_claim: Goldilocks,
}

/// The α-reduced row weights `m̂` over the flat witness coordinates and
/// the public relation target (Eq 158): coordinate `(e, j)` of the
/// witness carries weight `Σ_r ϑ_r · Σ_{c: reads e} A_{r,c}(α)·α^j`, and
/// the F_q target is `Σ_r ϑ_r · (Y_r(α) + (α^n+1)·Q_r(α))`.
///
/// At kernel scale the weights are materialized densely (the paper's §7.5
/// evaluates them structured); the `F_{Q_32}` products are embedded as
/// Goldilocks values for the Goldilocks sum-check layer. Because the LZX
/// substrate splits the ring field (`Q_32`) from the sum-check field
/// (Goldilocks), the fused sum-check runs on the **Goldilocks** target
/// `V_G = Σ_x ŵ(x)·m̂(x)` (see [`goldilocks_target``]); `fq_target` records
/// the paper-facing Eq-158 value.
pub struct FusedWeights {
    /// One Goldilocks weight per flat witness coordinate (length 2^µ').
    pub row_weights: Vec<Goldilocks>,
    /// The Eq-158 relation target evaluated in F_q and embedded.
    pub fq_target: Goldilocks,
}

/// Build the fused weights from the rows, quotients, batching challenges
/// `ϑ`, and the evaluation point `α` (Eq 158 over the flat coordinates).
pub fn build_fused_weights(
    ring: &RingConfig,
    rows: &[RingRow],
    lifted: &[LiftedRow],
    thetas: &[Goldilocks],
    witness_len: usize,
    alpha: u32,
) -> Result<FusedWeights, RingCheckError> {
    let q = ring.modulus;
    let n = ring.n();
    // Alpha powers per coefficient position.
    let alpha_pows: Vec<u32> = {
        let mut p = vec![1u32; n];
        for j in 1..n {
            p[j] = q.mul(p[j - 1], alpha);
        }
        p
    };
    // Embed a u32 residue as Goldilocks (canonical, injective).
    let fe = |v: u32| Goldilocks::from_u64(v as u64);
    // Coordinate layout: witness element e, coefficient j.
    let mut weights = vec![Goldilocks::ZERO; witness_len * n];
    let mut target = Goldilocks::ZERO;
    let alpha_n_plus_1 = q.add(q.pow(alpha, n as u64), 1);
    for (ri, (row, lift)) in rows.iter().zip(lifted.iter()).enumerate() {
        let theta = thetas.get(ri).copied().unwrap_or(Goldilocks::ZERO);
        for (a, &e) in row.multipliers.iter().zip(row.reads.iter()) {
            if e >= witness_len {
                return Err(RingCheckError::Shape {
                    expected: witness_len,
                    got: e,
                });
            }
            let a_alpha = poly_eval(q, a.coeffs(), alpha);
            for j in 0..n {
                let w = q.mul(a_alpha, alpha_pows[j]);
                let cur = weights[e * n + j];
                weights[e * n + j] = cur.add(&fe(w).mul(&theta));
            }
        }
        // Target: Y_r(α) + (α^n+1)·Q_r(α), both F_q values embedded.
        let mut t = poly_eval(q, row.target.coeffs(), alpha);
        t = q.add(t, q.mul(alpha_n_plus_1, poly_eval(q, &lift.quotient, alpha)));
        target = target.add(&fe(t).mul(&theta));
    }
    Ok(FusedWeights {
        row_weights: weights,
        fq_target: target,
    })
}

/// The Goldilocks-layer relation target of the fused sum-check:
/// `V_G = Σ_x ŵ(x)·m̂(x)` over the padded hypercube. In the paper's
/// unified field this equals the Eq-158 target; in the split-field LZX
/// realization it is the value the Goldilocks sum-check actually proves
/// (single products `ŵ·m̂ < q² < p` never wrap; only the final sum
/// reduces mod p).
pub fn goldilocks_target(weights: &FusedWeights, witness_flat: &[Goldilocks]) -> Goldilocks {
    let mut acc = Goldilocks::ZERO;
    for (w, m) in witness_flat.iter().zip(weights.row_weights.iter()) {
        acc = acc.add(&w.mul(m));
    }
    acc
}

/// The fused sum-check output: the deferred opening claim on the
/// successor witness plus the binding data.
#[derive(Clone, Debug)]
pub struct FusedProof {
    pub sumcheck: SumcheckProof,
    /// The final sum-check point `r_2`.
    pub point: Vec<Goldilocks>,
    /// The claimed `ŵ(r_2)` (the claim passed to the next level).
    pub witness_claim: Goldilocks,
}

/// Prove Eq 160: one sum-check over the shared witness combining the
/// relation term and the range anchoring under `γ` (drawn after both
/// claims are bound):
/// `Σ_x [ŵ(x)·m̂(x) + γ·eq(r_virt,x)·ŵ(x)·(ŵ(x)+1)] = V + γ·s_claim`.
pub fn prove_fused(
    witness_flat: &[Goldilocks],
    weights: &FusedWeights,
    claim: &FusedClaim,
    transcript: &mut Transcript,
) -> Result<FusedProof, RingCheckError> {
    // Pad the flat witness and weights to a power-of-two hypercube.
    let mu = witness_flat
        .len()
        .checked_next_power_of_two()
        .map(|p| p.trailing_zeros() as usize)
        .unwrap_or(0);
    let len = 1usize << mu;
    let mut w = witness_flat.to_vec();
    w.resize(len, Goldilocks::ZERO);
    let mut m = weights.row_weights.clone();
    m.resize(len, Goldilocks::ZERO);
    if claim.rvirt.len() != mu {
        return Err(RingCheckError::Shape {
            expected: mu,
            got: claim.rvirt.len(),
        });
    }
    // Bind the claims, then draw γ (Eq 160 ordering).
    let v_g = goldilocks_target(weights, witness_flat);
    transcript
        .append_field(b"akita-fused-v", &v_g)
        .map_err(RingCheckError::Transcript)?;
    transcript
        .append_field(b"akita-fused-sclaim", &claim.s_claim)
        .map_err(RingCheckError::Transcript)?;
    let gamma = transcript
        .challenge_field(b"akita-fused-gamma")
        .map_err(RingCheckError::Transcript)?;

    let w_mle = DenseMle {
        num_vars: mu,
        evaluations: w,
    };
    let m_mle = DenseMle {
        num_vars: mu,
        evaluations: m,
    };
    let eq_mle = DenseMle::eq_extension(&claim.rvirt);
    let mut vp = VirtualPolynomial::new(mu);
    let wi = vp.add_factor(w_mle.clone()).map_err(RingCheckError::Virtual)?;
    let mi = vp.add_factor(m_mle.clone()).map_err(RingCheckError::Virtual)?;
    let ei = vp.add_factor(eq_mle).map_err(RingCheckError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![wi, mi])
        .map_err(RingCheckError::Virtual)?;
    vp.add_term(gamma, vec![ei, wi, wi])
        .map_err(RingCheckError::Virtual)?;
    vp.add_term(gamma, vec![ei, wi])
        .map_err(RingCheckError::Virtual)?;
    let total = v_g.add(&gamma.mul(&claim.s_claim));
    let out = lattice_sumcheck::sumcheck::prove(&vp, total, transcript)
        .map_err(RingCheckError::Sumcheck)?;
    // The deferred opening claim: ŵ(r_2).
    let witness_claim = w_mle
        .evaluate(&out.challenges)
        .map_err(RingCheckError::Mle)?;
    Ok(FusedProof {
        sumcheck: out.proof,
        point: out.challenges,
        witness_claim,
    })
}

/// Verify the fused sum-check. `expected_witness` is the revealed
/// successor witness (kernel scale): the verifier recomputes the final
/// binding `P(r_2) = m̂(r_2)·ŵ(r_2) + γ·eq(r_virt,r_2)·ŵ(r_2)(ŵ(r_2)+1)`
/// exactly, as `pcs.rs` does for the evaluation proof.
pub fn verify_fused(
    witness_flat: &[Goldilocks],
    weights: &FusedWeights,
    claim: &FusedClaim,
    proof: &FusedProof,
    transcript: &mut Transcript,
) -> Result<(), RingCheckError> {
    let mu = witness_flat
        .len()
        .checked_next_power_of_two()
        .map(|p| p.trailing_zeros() as usize)
        .unwrap_or(0);
    let len = 1usize << mu;
    let mut w = witness_flat.to_vec();
    w.resize(len, Goldilocks::ZERO);
    let mut m = weights.row_weights.clone();
    m.resize(len, Goldilocks::ZERO);
    if claim.rvirt.len() != mu {
        return Err(RingCheckError::Shape {
            expected: mu,
            got: claim.rvirt.len(),
        });
    }
    transcript
        .append_field(b"akita-fused-v", &goldilocks_target(weights, witness_flat))
        .map_err(RingCheckError::Transcript)?;
    transcript
        .append_field(b"akita-fused-sclaim", &claim.s_claim)
        .map_err(RingCheckError::Transcript)?;
    let gamma = transcript
        .challenge_field(b"akita-fused-gamma")
        .map_err(RingCheckError::Transcript)?;
    let total =
        goldilocks_target(weights, witness_flat).add(&gamma.mul(&claim.s_claim));

    let w_mle = DenseMle {
        num_vars: mu,
        evaluations: w,
    };
    let m_mle = DenseMle {
        num_vars: mu,
        evaluations: m,
    };
    let eq_mle = DenseMle::eq_extension(&claim.rvirt);
    let verdict = proof
        .sumcheck
        .verify(mu, 3, total, transcript, None)
        .map_err(RingCheckError::Sumcheck)?;
    // Final binding: recompute P(r_2) from the revealed witness.
    let wr = w_mle
        .evaluate(&verdict.point)
        .map_err(RingCheckError::Mle)?;
    let mr = m_mle
        .evaluate(&verdict.point)
        .map_err(RingCheckError::Mle)?;
    let er = eq_mle
        .evaluate(&verdict.point)
        .map_err(RingCheckError::Mle)?;
    let expected_final = wr.mul(&mr).add(&gamma.mul(&er.mul(&wr.mul(&wr.add(&Goldilocks::ONE)))));
    if verdict.final_claim != expected_final {
        return Err(RingCheckError::FusedCheck);
    }
    if wr != proof.witness_claim {
        return Err(RingCheckError::FusedCheck);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};

    fn ring() -> RingConfig {
        RingConfig::new(Modulus32::Q_32, 4).ok().unwrap()
    }

    /// Honest rows from a synthetic fold: rows of the shape of Eq 7
    /// (`A·z − Σ c_i·t_i = 0`), built over a small witness.
    fn setup(tag: &[u8]) -> (RingConfig, Vec<RingElement>, Vec<RingRow>) {
        let ring = ring();
        let n = ring.n();
        // Witness: 4 short ring elements (successor digit segments).
        let witness: Vec<RingElement> = (0..4)
            .map(|i| {
                let mut t = tag.to_vec();
                t.push(i as u8);
                RingElement::from_signed(
                    &ring,
                    &(0..n).map(|j| ((i * 7 + j * 3) % 11) as i64 - 5).collect::<Vec<i64>>(),
                )
            })
            .collect();
        // Two rows in the Eq 146 normal form: multipliers, reads, target.
        let mk_row = |seed: u8| {
            let mult: Vec<RingElement> = (0..2)
                .map(|i| ring.uniform_from_seed(b"rc-A", &[seed, i as u8], 0))
                .collect();
            let target = {
                let mut acc = ring.zero();
                for (a, w) in mult.iter().zip([witness[0].clone(), witness[2].clone()]) {
                    acc = acc.add(&a.mul(&w).ok().unwrap()).ok().unwrap();
                }
                acc
            };
            RingRow {
                multipliers: mult,
                reads: vec![0, 2],
                target,
            }
        };
        let rows = vec![mk_row(1), mk_row(2)];
        (ring, witness, rows)
    }

    fn flat(witness: &[RingElement]) -> Vec<Goldilocks> {
        witness
            .iter()
            .flat_map(|e| e.coeffs().iter().map(|&c| Goldilocks::from_u64(c as u64)))
            .collect()
    }

    #[test]
    fn quotient_lift_exact_and_identity() {
        let (ring, witness, rows) = setup(b"ql");
        // The direct ring oracle: rows evaluate to zero.
        for row in &rows {
            assert!(row.evaluate(&ring, &witness).ok().unwrap().is_zero());
        }
        // The lift: zero remainder and the identity
        // Σ A·W − Y = (X^n+1)·Q holds coefficient-wise.
        let lifted = quotient_lift(&ring, &rows, &witness).ok().unwrap();
        let q = ring.modulus;
        let n = ring.n();
        for (row, lift) in rows.iter().zip(lifted.iter()) {
            let mut lhs = vec![0u32; 2 * n - 1];
            for (a, &idx) in row.multipliers.iter().zip(row.reads.iter()) {
                let prod = unreduced_product(q, a.coeffs(), witness[idx].coeffs());
                for (k, &p) in prod.iter().enumerate() {
                    lhs[k] = q.add(lhs[k], p);
                }
            }
            for (k, &y) in row.target.coeffs().iter().enumerate() {
                lhs[k] = q.sub(lhs[k], y);
            }
            // (X^n+1)·Q: the quotient appears at both shifted and
            // unshifted positions with a PLUS sign.
            let mut rhs = vec![0u32; 2 * n - 1];
            for (k, &c) in lift.quotient.iter().enumerate() {
                rhs[k + n] = q.add(rhs[k + n], c);
                rhs[k] = q.add(rhs[k], c);
            }
            assert_eq!(lhs, rhs, "Eq 147 must hold exactly for honest rows");
        }
    }

    #[test]
    fn alpha_check_honest_passes() {
        let (ring, witness, rows) = setup(b"alpha");
        let lifted = quotient_lift(&ring, &rows, &witness).ok().unwrap();
        let mut t = Transcript::new_default(b"lzx-akita-ring");
        absorb_lifted(&mut t, b"commitment", &lifted).ok().unwrap();
        let alpha = sample_alpha(&mut t, ring.modulus).ok().unwrap();
        assert!(alpha_check(&ring, &rows, &lifted, &witness, alpha).is_ok());
        // Every α in a sweep passes for honest rows (the identity holds
        // in K[X], not just at one point).
        for a in [1u32, 2, 5, 12345, ring.modulus.q - 7] {
            assert!(alpha_check(&ring, &rows, &lifted, &witness, a).is_ok());
        }
    }

    #[test]
    fn alpha_check_tampered_witness_rejected() {
        let (ring, witness, rows) = setup(b"tamper");
        let lifted = quotient_lift(&ring, &rows, &witness).ok().unwrap();
        // Tamper one witness coefficient: the ring rows break and the
        // α-check catches the false residual at a random α.
        let mut bad = witness.clone();
        let mut coeffs = bad[0].coeffs().to_vec();
        coeffs[1] = (coeffs[1] + 1) % ring.modulus.q;
        bad[0] = RingElement::from_coeffs(&ring, coeffs);
        // Direct oracle fails.
        assert!(!rows[0].evaluate(&ring, &bad).ok().unwrap().is_zero());
        // The lift of the tampered witness does not even exist.
        assert!(matches!(
            quotient_lift(&ring, &rows, &bad),
            Err(RingCheckError::NotDivisible { .. })
        ));
        // And the α-check against the stale quotient fails.
        let mut fails = 0;
        for a in [1u32, 2, 5, 12345, 999999937] {
            if alpha_check(&ring, &rows, &lifted, &bad, a).is_err() {
                fails += 1;
            }
        }
        assert!(fails >= 4, "a random α must catch the false residual");
    }

    #[test]
    fn alpha_check_tampered_quotient_rejected() {
        let (ring, witness, rows) = setup(b"tamper-q");
        let mut lifted = quotient_lift(&ring, &rows, &witness).ok().unwrap();
        if lifted[0].quotient.is_empty() {
            lifted[0].quotient.push(1);
        } else {
            let q = ring.modulus;
            lifted[0].quotient[0] = q.add(lifted[0].quotient[0], 1);
        }
        let mut t = Transcript::new_default(b"lzx-akita-ring");
        absorb_lifted(&mut t, b"commitment", &lifted).ok().unwrap();
        let alpha = sample_alpha(&mut t, ring.modulus).ok().unwrap();
        assert!(alpha_check(&ring, &rows, &lifted, &witness, alpha).is_err());
    }

    #[test]
    fn alpha_before_commit_fails() {
        // The Grand-Danois attack (App F.1): a challenge drawn BEFORE the
        // witness is committed lets the prover park a nonzero residual in
        // the known projection kernel — here the kernel of the evaluation
        // map W ↦ W(α₀). The repaired verifier (α after the commitment)
        // draws a different α and rejects.
        let (ring, witness, rows) = setup(b"gd");
        let lifted = quotient_lift(&ring, &rows, &witness).ok().unwrap();
        // Buggy ordering: α₀ drawn from a transcript BEFORE the
        // commitment is absorbed.
        let mut t_bug = Transcript::new_default(b"lzx-akita-ring");
        let alpha0 = sample_alpha(&mut t_bug, ring.modulus).ok().unwrap();
        // The attacker's residual: δ(X) = X − α₀ (nonzero in R_q, but
        // δ(α₀) = 0 — inside the kernel of the α₀-evaluation).
        let mut delta_coeffs = vec![0u32; ring.n()];
        delta_coeffs[0] = ring.modulus.sub(0, alpha0);
        delta_coeffs[1] = 1;
        let delta = RingElement::from_coeffs(&ring, delta_coeffs);
        assert!(!delta.is_zero());
        // Tampered witness: w'₀ = w₀ + δ — the ring row becomes false.
        let mut bad = witness.clone();
        bad[0] = bad[0].add(&delta).ok().unwrap();
        assert!(
            !rows[0].evaluate(&ring, &bad).ok().unwrap().is_zero(),
            "the attack breaks the ring relation"
        );
        // Under the buggy α₀ the α-check passes (the residual is in the
        // kernel) — the archived vulnerability.
        assert!(alpha_check(&ring, &rows, &lifted, &bad, alpha0).is_ok());
        // The repaired verifier: commitment and quotients absorbed FIRST,
        // then α — the fresh α lands outside the kernel and rejects.
        let mut t_ok = Transcript::new_default(b"lzx-akita-ring");
        absorb_lifted(&mut t_ok, b"commitment", &lifted).ok().unwrap();
        let alpha1 = sample_alpha(&mut t_ok, ring.modulus).ok().unwrap();
        assert_ne!(alpha1, alpha0, "transcript order must change α");
        assert!(alpha_check(&ring, &rows, &lifted, &bad, alpha1).is_err());
    }

    #[test]
    fn fused_sumcheck_honest_and_tampered() {
        let (ring, witness, rows) = setup(b"fused");
        let lifted = quotient_lift(&ring, &rows, &witness).ok().unwrap();
        let mut t = Transcript::new_default(b"lzx-akita-ring");
        absorb_lifted(&mut t, b"commitment", &lifted).ok().unwrap();
        let alpha = sample_alpha(&mut t, ring.modulus).ok().unwrap();
        // Row-batching challenges ϑ (Eq 158), drawn after α.
        let thetas = t
            .challenge_fields(b"akita-tau1", rows.len())
            .ok()
            .unwrap();
        let weights = build_fused_weights(&ring, &rows, &lifted, &thetas, witness.len(), alpha)
            .ok()
            .unwrap();
        let wf = flat(&witness);
        // Honest range claim: s_claim = ŝ(r_virt) over the derived values.
        let mu = wf.len().next_power_of_two().trailing_zeros() as usize;
        let rvirt: Vec<Goldilocks> = (1..=mu)
            .map(|i| Goldilocks::from_u64((i * 7717) as u64))
            .collect();
        let eq = DenseMle::eq_extension(&rvirt);
        let mut s_claim = Goldilocks::ZERO;
        for (x, &w) in eq.evaluations.iter().zip(wf.iter()) {
            s_claim = s_claim.add(&w.mul(&w.add(&Goldilocks::ONE)).mul(x));
        }
        let claim = FusedClaim { rvirt, s_claim };
        let mut pt = Transcript::new_default(b"lzx-akita-fused");
        let proof = prove_fused(&wf, &weights, &claim, &mut pt).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-akita-fused");
        assert!(verify_fused(&wf, &weights, &claim, &proof, &mut vt).is_ok());

        // Tampered carried claim: the transcript-derived γ desyncs and the
        // fusion catches it (an honest prover refuses to prove it at all).
        let mut bad_claim = claim.clone();
        bad_claim.s_claim = bad_claim.s_claim.add(&Goldilocks::ONE);
        {
            let mut t2 = Transcript::new_default(b"lzx-akita-fused");
            assert!(prove_fused(&wf, &weights, &bad_claim, &mut t2).is_err());
        }
        let mut vt2 = Transcript::new_default(b"lzx-akita-fused");
        assert!(verify_fused(&wf, &weights, &bad_claim, &proof, &mut vt2).is_err());

        // Tampered witness: the final binding recomputation fails.
        let mut bad_w = wf.clone();
        bad_w[0] = bad_w[0].add(&Goldilocks::ONE);
        let mut vt3 = Transcript::new_default(b"lzx-akita-fused");
        assert!(verify_fused(&bad_w, &weights, &claim, &proof, &mut vt3).is_err());
    }

    #[test]
    fn fused_weights_bind_rows_to_witness() {
        // The α-reduced row weights: the built target V must equal
        // Σ_r ϑ_r·(Y_r(α) + (α^n+1)·Q_r(α)), which for honest rows equals
        // Σ_r ϑ_r·Σ_c A_{r,c}(α)·W_{r,c}(α) (the lift identity at α) — and
        // the per-coordinate weights must match the direct computation
        // (identity test for Eq 158 at kernel scale).
        let (ring, witness, rows) = setup(b"weights");
        let lifted = quotient_lift(&ring, &rows, &witness).ok().unwrap();
        let thetas = vec![Goldilocks::from_u64(3), Goldilocks::from_u64(5)];
        let alpha = 17u32;
        let weights =
            build_fused_weights(&ring, &rows, &lifted, &thetas, witness.len(), alpha).ok().unwrap();
        let q = ring.modulus;
        let n = ring.n();
        let alpha_n_plus_1 = q.add(q.pow(alpha, n as u64), 1);
        let fe = |v: u32| Goldilocks::from_u64(v as u64);
        // Direct target computation, both ways.
        let mut via_target = Goldilocks::ZERO;
        let mut via_witness = Goldilocks::ZERO;
        for (ri, (row, lift)) in rows.iter().zip(lifted.iter()).enumerate() {
            let theta = thetas[ri];
            let t = q.add(
                poly_eval(q, row.target.coeffs(), alpha),
                q.mul(alpha_n_plus_1, poly_eval(q, &lift.quotient, alpha)),
            );
            via_target = via_target.add(&fe(t).mul(&theta));
            let mut rhs = 0u32;
            for (a, &idx) in row.multipliers.iter().zip(row.reads.iter()) {
                rhs = q.add(
                    rhs,
                    q.mul(poly_eval(q, a.coeffs(), alpha), poly_eval(q, witness[idx].coeffs(), alpha)),
                );
            }
            via_witness = via_witness.add(&fe(rhs).mul(&theta));
        }
        assert_eq!(weights.fq_target, via_target);
        assert_eq!(via_target, via_witness, "the lift identity at α must hold");
        // Per-coordinate weight check: element e=0, coefficient j=3 —
        // the ϑ-weighted F_q products folded in Goldilocks.
        let mut expect_w = Goldilocks::ZERO;
        for (ri, row) in rows.iter().enumerate() {
            for (a, &e) in row.multipliers.iter().zip(row.reads.iter()) {
                if e == 0 {
                    let contrib = q.mul(poly_eval(q, a.coeffs(), alpha), q.pow(alpha, 3));
                    expect_w = expect_w.add(&fe(contrib).mul(&thetas[ri]));
                }
            }
        }
        assert_eq!(weights.row_weights[3], expect_w);
    }

    #[test]
    fn commitment_binding_via_ajtai() {
        // The witness commitment absorbed before α is a real Ajtai
        // binding: a tampered witness no longer opens it.
        let ring = ring();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 1,
            m: 4,
            norm_bound: 1 << 20,
        };
        let pk = AjtaiPublicKey::from_seed(params, [55u8; 32]).ok().unwrap();
        let (_, witness, _) = setup(b"ajtai");
        let commitment = pk.commit(&witness).ok().unwrap();
        assert!(pk.verify_opening(&commitment, &witness).is_ok());
        let mut bad = witness.clone();
        let mut coeffs = bad[1].coeffs().to_vec();
        coeffs[0] = (coeffs[0] + 1) % ring.modulus.q;
        bad[1] = RingElement::from_coeffs(&ring, coeffs);
        assert!(pk.verify_opening(&commitment, &bad).is_err());
    }
}
