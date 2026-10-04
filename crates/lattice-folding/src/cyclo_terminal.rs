//! **The Cyclo §7 bridge's compact-PCS terminal** — the decider that
//! stops opening the witness.
//!
//! # What this replaces
//!
//! `cyclo_r1cs::decide_principal_linear` decides the ride-the-fold
//! claims on the OPENED lift `z'` (the decider model: the full witness
//! in the clear — `(D1)` re-runs `verify_opening`, `(D2)`/`(D3)` consume
//! `z'` coefficient-by-coefficient). This module replaces the opened
//! witness with the **ring-functional width fold**
//! (`lattice_widthfold::ring_fold`): the natural composition the
//! NEXT_STEPS ledger pointed at — the short-opening route through the
//! width fold's machinery.
//!
//! # The terminal's shape
//!
//! The lift `z' ∈ R_q^m` is never transmitted. In its place:
//!
//! * **(W0)** the fold's part images sum to the COMMITMENT `y = A·z'`
//!   — the exact replacement of `(D1)`'s `verify_opening` (the parts
//!   are pinned to the public commitment; the norm accounting rides
//!   the fold's extraction posture instead of `verify_opening`'s
//!   gate);
//! * **(D2)** the six `(4)` linear claims ride the fold's RING
//!   functional layer — the per-part values `U^t_i` with the exact
//!   `(W0R)` check `Σ_i U^{(i,b)}_i = d'_i^{(b)}` (the public lifts)
//!   and the `(W3R)` superposition identities binding them to the
//!   folded response;
//! * **(D3)** the two prefix claims ride the same layer as PROJECTED
//!   functionals (no exact ring target exists — the paper's check is
//!   `θ_k(Λ(v)) = e`): the decider computes the fold's public sums
//!   `Σ_i U^{(D3,b)}_i` and applies the `θ_k` projection itself;
//! * **(W2)** the short MSIS instance `[A₂ | −T]` — the estimator-gated
//!   binding (fail-closed at prove AND verify; the digit gate
//!   `β₁ = k − 1` sits deep in the sound regime, so the single stage
//!   covers the bridge's `m` at every test scale);
//! * **(W1)/(W3R)** the consistency carriers (the exact fold identity
//!   and the functional superpositions).
//!
//! # The honest ledger
//!
//! * The binding posture is the width fold's own (the per-stage
//!   estimator floor + the `(W0)` target threading); the full
//!   multi-fork LaBRADOR extraction across the fold is the documented
//!   open analysis (the same residual the zkvm Sound profile carries).
//! * The lift's norm precondition `∥z'∥∞ < k` enters as the fold's
//!   `β₁` parameter (the estimator's bound accounting) — the terminal
//!   verifies the FOLDED response's gate `(W4)` directly and the
//!   parts' norm story rides the `(W2)` extraction discipline (the
//!   honest deviation vs the clear decider's explicit per-element
//!   gate, recorded in `cyclo.md`).
//! * The `(D3)` projected checks bind through `θ_k`'s F_q-linearity
//!   exactly as the paper's extraction argument requires (the carries
//!   the digit embedding introduces do not survive the projection).

use crate::cyclo_r1cs::{eq_table_q2, Fq2Q32, PrincipalLinearClaim, R1csQ32, ThetaK};
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_ring::ring::{RingConfig, RingElement};
use lattice_widthfold::fold::WidthFoldParams;
use lattice_widthfold::ring_fold::{
    prove_ring_fold, ring_fold_functional_sum, verify_ring_fold, RingFoldProof, RingFunctional,
};

/// The compact terminal: the ring-functional width fold over the lift
/// (replaces the opened witness in the decider's interface).
#[derive(Clone, Debug)]
pub struct CompactTerminal {
    /// The fold over `z'` under the bridge's Ajtai key.
    pub fold: RingFoldProof,
}

/// Build the decider's EIGHT public functionals over `z'` from the
/// claim (both sides derive identically):
///
/// * `0..6` — the `(D2)` family: `Λ^{(i,b)}(x) =
///   Σ_{b'} MLE[M_i](u, b')^{(b)}·x_{b'}` (the matrix-MLE rows);
/// * `6..8` — the `(D3)` family: `Λ^{(D3,b)}(x) =
///   Σ_j eq(v, j)^{(b)}·x_j` (the prefix-elimination weights).
fn bridge_functionals(
    shape: &R1csQ32,
    claim: &PrincipalLinearClaim,
) -> Result<Vec<RingFunctional>, String> {
    let log_m = shape.m.trailing_zeros() as usize;
    if claim.u.len() != log_m {
        return Err("claim point arity".into());
    }
    let eq_u = eq_table_q2(&claim.u);
    let mut out: Vec<RingFunctional> = Vec::with_capacity(8);
    // (D2): the matrix-MLE rows per (i, b).
    for mat in shape.mats.iter() {
        for b in 0..2usize {
            let weights: Vec<u64> = (0..shape.m)
                .map(|bp| {
                    let mut m_bp = Fq2Q32::ZERO;
                    for (r, eu) in eq_u.iter().enumerate() {
                        m_bp = m_bp.add(&eu.mul(&Fq2Q32::from_u64(mat[r * shape.m + bp])));
                    }
                    if b == 0 {
                        m_bp.c0
                    } else {
                        m_bp.c1
                    }
                })
                .collect();
            out.push(RingFunctional { weights });
        }
    }
    // (D3): the prefix eq weights per component.
    let log_prefix = (shape.ell + 1).trailing_zeros() as usize;
    if claim.v.len() != log_prefix {
        return Err("prefix point arity".into());
    }
    let eq_v = eq_table_q2(&claim.v);
    for b in 0..2usize {
        let mut weights = vec![0u64; shape.m];
        for (j, evj) in eq_v.iter().enumerate() {
            weights[j] = if b == 0 { evj.c0 } else { evj.c1 };
        }
        out.push(RingFunctional { weights });
    }
    Ok(out)
}

/// The (D2) family's public targets: the claim's ring lifts `d'_i^{(b)}`
/// (the EXACT checks); the (D3) family carries `None` (the projected
/// `θ_k` checks are the decider's own).
fn bridge_targets(ring: &RingConfig, claim: &PrincipalLinearClaim) -> Vec<Option<RingElement>> {
    let mut out: Vec<Option<RingElement>> = Vec::with_capacity(8);
    for i in 0..3usize {
        for b in 0..2usize {
            out.push(Some(RingElement::from_coeffs(
                ring,
                claim.d_lift[i][b].clone(),
            )));
        }
    }
    out.push(None);
    out.push(None);
    out
}

/// The bridge key's column blocks (both sides regenerate from the pk).
fn pk_blocks(ring: &RingConfig, pk: &AjtaiPublicKey, m: usize) -> Vec<Vec<RingElement>> {
    let k = pk.params.k;
    (0..m)
        .map(|c| {
            (0..k)
                .map(|rr| pk.entry(rr, c).cloned().unwrap_or_else(|| ring.zero()))
                .collect()
        })
        .collect()
}

/// The commitment's rows as the fold's public image target.
fn commitment_target(
    ring: &RingConfig,
    claim: &PrincipalLinearClaim,
) -> Result<Vec<RingElement>, String> {
    let parsed = AjtaiCommitment::from_bytes(ring, claim.commit_k, &claim.commitment)
        .map_err(|e| format!("{e:?}"))?;
    Ok(parsed.rows)
}

/// The terminal's FS absorption: the bridge statement (the shape
/// digest + the claim bytes + θ) BEFORE any fold material — the decider
/// domain's hygiene (both sides identical).
fn absorb_terminal_statement(
    transcript: &mut Transcript,
    shape: &R1csQ32,
    claim: &PrincipalLinearClaim,
    theta: &ThetaK,
) -> Result<(), String> {
    let digest = shape.digest();
    transcript
        .append_bytes(b"cyclo-terminal-shape", &digest)
        .map_err(|e| format!("{e:?}"))?;
    transcript
        .append_bytes(b"cyclo-terminal-claim", &claim.commitment)
        .map_err(|e| format!("{e:?}"))?;
    let mut dbuf = Vec::new();
    for i in 0..3usize {
        for b in 0..2usize {
            for &c in &claim.d_lift[i][b] {
                dbuf.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
    transcript
        .append_bytes(b"cyclo-terminal-dlift", &dbuf)
        .map_err(|e| format!("{e:?}"))?;
    transcript
        .append_bytes(
            b"cyclo-terminal-theta",
            &[theta.k as u8, theta.digits as u8],
        )
        .map_err(|e| format!("{e:?}"))?;
    Ok(())
}

/// Prove the compact terminal over the lifted witness `z_lift` (the
/// prover holds the lift; the fold replaces its transmission).
#[allow(clippy::too_many_arguments)]
pub fn prove_compact_terminal(
    ring: &RingConfig,
    pk: &AjtaiPublicKey,
    shape: &R1csQ32,
    x: &[u64],
    claim: &PrincipalLinearClaim,
    theta: &ThetaK,
    z_lift: &[RingElement],
    transcript: &mut Transcript,
) -> Result<CompactTerminal, String> {
    if z_lift.len() != shape.m {
        return Err(format!("lift length {} vs m {}", z_lift.len(), shape.m));
    }
    let functionals = bridge_functionals(shape, claim)?;
    let targets = bridge_targets(ring, claim);
    // The lift-norm precondition (the paper's ∥z'∥ < k): the fold's
    // β₁ (the estimator's bound accounting).
    for e in z_lift {
        if e.infinity_norm() as u64 >= theta.k {
            return Err("lift norm gate (k) exceeded".into());
        }
    }
    let beta1 = theta.k.max(2);
    let q = u64::from(ring.modulus.q);
    // The fold's profile: the cheap sound row at the digit gate (the
    // bridge's m sits in the single-stage coverage; fail-closed beyond).
    let params = WidthFoldParams::sound_profile_for(shape.m, beta1, q, ring.n() as u64)?;
    let blocks = pk_blocks(ring, pk, shape.m);
    let t_target = commitment_target(ring, claim)?;
    absorb_terminal_statement(transcript, shape, claim, theta)?;
    let fold = prove_ring_fold(
        ring,
        z_lift,
        &t_target,
        &functionals,
        &targets,
        &blocks,
        pk.params.k,
        params,
        beta1,
        // The terminal's key domain: the claim commitment binds the
        // seed choice (both sides re-derive identically from the claim).
        terminal_seed(claim),
        transcript,
    )?;
    let _ = x;
    Ok(CompactTerminal { fold })
}

/// The terminal's fold-key seed: derived from the claim's commitment
/// (public — both sides identical; the bridge pk's own seed stays
/// with the key regeneration).
fn terminal_seed(claim: &PrincipalLinearClaim) -> [u8; 32] {
    let mut st = Transcript::new_default(b"lzx-cyclo-terminal-seed");
    let _ = st.append_bytes(b"commitment", &claim.commitment);
    let mut s = [0u8; 32];
    if let Ok(b) = st.challenge_bytes(b"seed", 32) {
        s.copy_from_slice(&b);
    }
    s
}

/// **The compact decider**: decide the ride-the-fold claims WITHOUT the
/// opened witness — the ring-functional width fold carries `(D1)`
/// (the (W0) commitment binding), `(D2)` (the exact (W0R) checks
/// against the public lifts), and `(D3)` (the projected `θ_k` checks
/// on the fold's public functional sums), with `(W1)`/`(W2)`/`(W3R)`
/// as the consistency carriers and the estimator-gated binding.
pub fn decide_principal_linear_compact(
    ring: &RingConfig,
    pk: &AjtaiPublicKey,
    shape: &R1csQ32,
    x: &[u64],
    claim: &PrincipalLinearClaim,
    terminal: &CompactTerminal,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let theta = ThetaK::new(claim.theta_k)?;
    if terminal.fold.n_bar != shape.m {
        return Err(format!(
            "terminal width {} vs m {}",
            terminal.fold.n_bar, shape.m
        ));
    }
    let functionals = bridge_functionals(shape, claim)?;
    let targets = bridge_targets(ring, claim);
    let beta1 = theta.k.max(2);
    let blocks = pk_blocks(ring, pk, shape.m);
    let t_target = commitment_target(ring, claim)?;
    absorb_terminal_statement(transcript, shape, claim, &theta)?;
    // (D1)+(D2)+(W1)/(W2)/(W3R)/(W4): the fold's full check suite.
    verify_ring_fold(
        ring,
        &t_target,
        &functionals,
        &targets,
        &blocks,
        pk.params.k,
        beta1,
        terminal_seed(claim),
        &terminal.fold,
        transcript,
    )?;
    // (D3) The projected prefix checks: θ_k(Σ_i U^{(D3,b)}_i) = e_b.
    for b in 0..2usize {
        let t = 6 + b;
        let sum = ring_fold_functional_sum(ring, &terminal.fold, t)?;
        let projected = theta.project(ring, &sum);
        let want = if b == 0 { claim.e.c0 } else { claim.e.c1 };
        if projected != want {
            return Err(format!(
                "(D3): the projected prefix claim (component {b}) failed"
            ));
        }
    }
    let _ = x;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cyclo_r1cs::{prove_r1cs_bridge, verify_r1cs_bridge};
    use lattice_commitment::ajtai::AjtaiParams;

    fn ring() -> RingConfig {
        lattice_widthfold::codec::q32_ring().unwrap()
    }

    /// The test shape from cyclo_r1cs's own suite: a random sparse R1CS
    /// with a satisfying witness.
    fn shape_with_witness(m: usize, ell: usize, seed: u64) -> (R1csQ32, Vec<u64>, Vec<u64>) {
        // A diagonal-dominant shape with a known satisfying witness: A·z = z (identity rows on the wire
        // segment), B·z = 1 (the prefix constant), C·z = z∘z —
        // satisfied by z with z_i ∈ {0, 1} on the wire.
        let mut mats = vec![vec![0u64; m * m]; 3];
        for r in 0..m {
            // M0: the row r picks z_r (identity).
            mats[0][r * m + r] = 1;
            // M1: the row r sums the prefix (z_ell = 1 contributes).
            mats[1][r * m + ell] = 1;
            // M2: the row r picks z_r again (so C·z = z must equal
            // (A·z)∘(B·z) = z_r·1 = z_r ✓ for any z with z[ell] = 1).
            mats[2][r * m + r] = 1;
        }
        let mut z_next = seed;
        let mut step = || {
            z_next = z_next
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            z_next
        };
        let mut z = vec![0u64; m];
        for v in z.iter_mut().take(ell) {
            *v = step() % 1000;
        }
        z[ell] = 1;
        for v in z.iter_mut().take(m).skip(ell + 1) {
            *v = step() % 100;
        }
        let w = z[ell + 1..].to_vec();
        let shape = R1csQ32 {
            ell,
            m,
            mats: [mats[0].clone(), mats[1].clone(), mats[2].clone()],
        };
        (shape, z[..ell].to_vec(), w)
    }

    /// The full compact-terminal pipeline: prove ⟶ verify ⟶ DECIDE
    /// (no opened witness anywhere) + the tamper suite.
    #[test]
    fn bridge_compact_terminal_end_to_end() {
        let ring = ring();
        let (shape, x, w) = shape_with_witness(8, 3, 5);
        let theta = ThetaK::new(4).unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: shape.m,
            norm_bound: 1 << 20,
        };
        let pk = AjtaiPublicKey::from_seed(params, [55u8; 32]).unwrap();
        // Prove + verify the bridge.
        let mut tr = Transcript::new_default(b"cyclo-r1cs-bridge");
        let proof = prove_r1cs_bridge(&ring, &pk, &shape, &x, &w, &theta, &mut tr).unwrap();
        let mut vt = Transcript::new_default(b"cyclo-r1cs-bridge");
        let claim = verify_r1cs_bridge(&ring, &shape, &x, &proof, &mut vt).unwrap();
        // The lift (prover-side).
        let mut z = x.clone();
        z.push(1);
        z.extend_from_slice(&w);
        let z_lift: Vec<RingElement> = z.iter().map(|&c| theta.embed(&ring, c)).collect();

        // The compact terminal: prove (the fold replaces the witness).
        let mut tt = Transcript::new_default(b"cyclo-terminal");
        let terminal =
            prove_compact_terminal(&ring, &pk, &shape, &x, &claim, &theta, &z_lift, &mut tt)
                .expect("the honest terminal proves");
        // DECIDE without the witness.
        let mut dt = Transcript::new_default(b"cyclo-terminal");
        decide_principal_linear_compact(&ring, &pk, &shape, &x, &claim, &terminal, &mut dt)
            .expect("the honest terminal decides every claim");

        // ---- The compact decider's tamper suite ----
        // (a) A WRONG lift (a different witness's digits): the terminal
        //     PROOF fails at (W0R) — the (D2) exact targets mismatch.
        let mut z_wrong = z.clone();
        z_wrong[shape.ell + 1] += 1;
        let z_lift_wrong: Vec<RingElement> =
            z_wrong.iter().map(|&c| theta.embed(&ring, c)).collect();
        let mut tt2 = Transcript::new_default(b"cyclo-terminal");
        assert!(prove_compact_terminal(
            &ring,
            &pk,
            &shape,
            &x,
            &claim,
            &theta,
            &z_lift_wrong,
            &mut tt2
        )
        .is_err());

        // (b) A corrupted claim (the d_lift bytes): (W0R) rejects.
        let mut claim_bad = claim.clone();
        claim_bad.d_lift[0][0][3] ^= 0x40;
        let mut dt3 = Transcript::new_default(b"cyclo-terminal");
        assert!(decide_principal_linear_compact(
            &ring, &pk, &shape, &x, &claim_bad, &terminal, &mut dt3
        )
        .is_err());

        // (c) A corrupted prefix evaluation e: the (D3) projected check
        //     rejects.
        let mut claim_bad2 = claim.clone();
        claim_bad2.e = claim_bad2.e.add(&Fq2Q32::ONE);
        let mut dt4 = Transcript::new_default(b"cyclo-terminal");
        assert!(decide_principal_linear_compact(
            &ring,
            &pk,
            &shape,
            &x,
            &claim_bad2,
            &terminal,
            &mut dt4
        )
        .is_err());

        // (d) A swapped commitment: the (W0) binding rejects.
        let mut claim_bad3 = claim.clone();
        if claim_bad3.commitment.len() > 4 {
            claim_bad3.commitment[4] ^= 0x80;
        }
        let mut dt5 = Transcript::new_default(b"cyclo-terminal");
        assert!(decide_principal_linear_compact(
            &ring,
            &pk,
            &shape,
            &x,
            &claim_bad3,
            &terminal,
            &mut dt5
        )
        .is_err());

        // (e) A tampered fold response (the folded z): (W1)/(W2) reject.
        let mut term_bad = terminal.clone();
        {
            let mut coeffs =
                lattice_widthfold::codec::decode_response(&term_bad.fold.response).unwrap();
            assert!(!coeffs.is_empty());
            coeffs[0] = coeffs[0].wrapping_add(1);
            term_bad.fold.response = lattice_widthfold::codec::encode_response(&coeffs).unwrap();
        }
        let mut dt6 = Transcript::new_default(b"cyclo-terminal");
        assert!(decide_principal_linear_compact(
            &ring, &pk, &shape, &x, &claim, &term_bad, &mut dt6
        )
        .is_err());

        // (f) Tampered functional values (the per-part U's): (W3R) or
        //     the derived checks reject.
        let mut term_bad2 = terminal.clone();
        assert!(!term_bad2.fold.func_values.is_empty());
        term_bad2.fold.func_values[3] ^= 0x20;
        let mut dt7 = Transcript::new_default(b"cyclo-terminal");
        assert!(decide_principal_linear_compact(
            &ring, &pk, &shape, &x, &claim, &term_bad2, &mut dt7
        )
        .is_err());
    }
}
