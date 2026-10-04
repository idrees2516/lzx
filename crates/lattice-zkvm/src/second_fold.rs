//! **The second-level fold (P4 / DESIGN_50KB Stage 5.2)** — the LaBRADOR
//! decider role for the compact opening: amortize the first fold's
//! responses over their PUBLIC targets into ONE short response bound
//! by a fresh Ajtai key, taking the binding MSIS instance from the
//! estimator's broken regime (~2^12 at the benchmark response lengths)
//! to the sound regime (`κ = 4, n̄ ∈ {2, 4}, r = 4, A ≤ 2^8` — 329+
//! classical bits).
//!
//! # The construction (the "response vector per part, γ/δ wiring")
//!
//! The first fold of each of `r` bundles produces a response
//! `v_i ∈ R^n̄` (prover-side, NEVER transmitted) with the PUBLIC target
//! `t_i = F̄·v_i` (the verifier computes it from the bundle's
//! `Σ_j d_j·y_j`) and the functional claim `u_i = Φ(v_i)` (the
//! Goldilocks carrier — the two-characteristic discipline). The
//! bundles share the level-1 key `F̄` (the column-uniform family).
//!
//! The second fold:
//!
//! 1. **Inner commitments** (transmitted BEFORE the challenges): each
//!    response committed under a FRESH seed-derived Ajtai key
//!    `A₂ ∈ R^{κ×n̄}`: `T_i = A₂·v_i ∈ R^κ` with κ = 4 — the
//!    estimator's sound row count.
//! 2. **Scalar challenges** `γ_i ∈ [−A, A]` (the γ wiring — scalar so
//!    the functional commutes exactly, the level-1 discipline).
//! 3. **The amortized response** `z = Σ_i γ_i·v_i ∈ R^n̄` with
//!    `|z|_∞ ≤ r·A·β₁` (the fail-closed gate; β₁ the level-1 gate).
//! 4. **The checks** — every one EXACT and LINEAR, with NO cross terms
//!    (the targets are public, so the key applies once to the folded
//!    response — this is precisely the LaBRADOR amortized-opening
//!    shape):
//!    * `(L1)` `F̄·z = Σ_i γ_i·t_i` — the fold consistency against the
//!      PUBLIC targets (the verifier computes both sides);
//!    * `(L2)` `A₂·z = Σ_i γ_i·T_i` — the fold consistency binding `z`
//!      to the inner commitments: **the short MSIS instance the whole
//!      construction exists for**;
//!    * `(L3)` `Φ(z) = Σ_i γ_i·u_i` — the commuting functional;
//!    * `(L4)` the norm gates on `z` (fail-closed).
//!
//! # Soundness (the extraction)
//!
//! A cheater fixes the `T_i` before `γ` and picks `z` after. Two
//! accepting transcripts with `γ ≠ γ'` give
//! `A₂·Δz = Σ_i Δγ_i·T_i` and `F̄·Δz = Σ_i Δγ_i·t_i` — subtracting the
//! honest relations, `(Δz, Δγ)` solves MSIS on `[A₂ | −T]` at rank
//! `n̄ + r` with the amplitude bound — **exactly the instance shape the
//! estimator's fold_security_table models** (`m = (n̄ + r)·64`, bound
//! `2·r·A·β₁` at the relaxed 2× factor). The fail-closed profile
//! re-runs the estimator and refuses any configuration below the
//! security floor.
//!
//! # The honest width-reduction finding
//!
//! The DESIGN doc's "~5 KB at the benchmark response lengths" requires
//! REDUCING the response width `n̄` (hundreds at benchmark scale) to
//! `{2, 4}`. That reduction splits ONE response into parts and folds
//! them — but the parts' link images `p_i = F̄_i·s_i` are PROVER data,
//! and binding them to the parts through the fold necessarily produces
//! the cross terms `F̄_i·s_j (i ≠ j)` — the QUADRATIC GARBAGE the
//! LaBRADOR level commits before the challenges (the `h_ij` machinery).
//! There is no garbage-free width-reducing fold: the construction here
//! amortizes over PUBLIC targets instead (sound, exact, no garbage)
//! and leaves the width reduction — the estimator-mandated Stage 5.2
//! completion at benchmark scale — to the LaBRADOR tail (the documented
//! follow-up). The profile below gates every shape fail-closed: the
//! sound regime covers `n̄ ≤ 4` (the bits bundle at small traces, where
//! the byte-packed stream fits a handful of ring elements).

use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_ring::{RingConfig, RingElement};

/// The security floor (classical bits) the profile enforces.
pub const SECURITY_FLOOR_BITS: f64 = 128.0;

/// The second-fold parameters (public shape, transmitted in the
/// opening).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecondFoldParams {
    /// The instance count r (one level-1 response per instance).
    pub r: usize,
    /// The inner-commitment key rows κ (the estimator's sound row
    /// count: 4).
    pub kappa: usize,
    /// The level-1 key rows k (F̄'s rank — the link image width).
    pub k: usize,
    /// The scalar challenge amplitude (γ_i ∈ [−A, A]).
    pub amplitude: u32,
    /// The response width n̄ (the final response rank).
    pub n_bar: usize,
}

impl SecondFoldParams {
    /// The estimator-tuned sound profile: `κ = 4, r = 4, A = 2^8,
    /// n̄ = 2` — the SECURITY.md table's sound row (329+ classical
    /// bits at the byte-packed level-1 gate β₁ = 255), re-verified
    /// fail-closed by `assert_sound` at construction.
    pub fn sound_defaults() -> Self {
        SecondFoldParams {
            r: 4,
            kappa: 4,
            k: 4,
            amplitude: 1 << 8,
            n_bar: 2,
        }
    }

    /// The final response bound `β₂ = r·A·β₁` (the completeness gate).
    pub fn beta2(&self, beta1: u64) -> u64 {
        self.r as u64 * self.amplitude as u64 * beta1
    }
}

/// The estimator verdict for a candidate profile at a given level-1
/// gate: `(classical_bits, quantum_bits)` of the FINAL instance
/// `m = (n̄ + r)·64`, bound `2·β₂` (the extraction's relaxed 2× factor,
/// matching `fold_security_table`'s convention).
pub fn profile_bits(
    params: &SecondFoldParams,
    beta1: u64,
    q: u64,
    ring_dim: u64,
) -> Result<(f64, f64), String> {
    let beta2 = params.beta2(beta1);
    // The extraction's relaxed bound (the 2× factor).
    let bound = 2 * beta2;
    if bound == 0 || bound >= q / 2 {
        return Err(format!(
            "gate exceeds q/2: bound {bound} vs q {q}"
        ));
    }
    let width = (params.n_bar + params.r) as u64;
    let p = lattice_sis_estimator::scalar_sis_from_ring(
        ring_dim,
        params.kappa as u64,
        width,
        q as u128,
        bound,
        lattice_sis_estimator::SisNorm::Infinity,
    )
    .map_err(|e| format!("estimator: {e:?}"))?;
    lattice_sis_estimator::sis_security_bits(&p).map_err(|e| format!("bits: {e:?}"))
}

/// The fail-closed profile check: refuse any configuration whose final
/// instance sits below the security floor. This is the
/// "estimator-gated SecurityProfiles" discipline applied to the
/// second fold — the gate that makes the construction's binding claim
/// load-bearing instead of aspirational.
pub fn assert_sound(
    params: &SecondFoldParams,
    beta1: u64,
    q: u64,
    ring_dim: u64,
) -> Result<(f64, f64), String> {
    let (cl, qm) = profile_bits(params, beta1, q, ring_dim)?;
    if cl < SECURITY_FLOOR_BITS {
        return Err(format!(
            "second-fold profile below the security floor: {cl:.1} classical bits \
             (params r={}, kappa={}, n_bar={}, A=2^{}, beta1={beta1})",
            params.r,
            params.kappa,
            params.n_bar,
            params.amplitude.trailing_zeros()
        ));
    }
    Ok((cl, qm))
}

/// The second-fold proof artifact.
#[derive(Clone, Debug)]
pub struct SecondFoldProof {
    pub params: SecondFoldParams,
    /// The inner commitments T_i (r × κ ring elements, flat).
    pub t_inner: Vec<u8>,
    /// The amortized response z ∈ R^{n̄} (rANS-coded coefficients).
    pub response: crate::compact::ResponseWire,
    /// The profile verdict at prove time (the recorded classical bits —
    /// the posture marker: the verifier re-derives and compares).
    pub classical_bits: f64,
}

/// Prove the second fold over `r` level-1 responses `v_i ∈ R^{n̄}` with
/// their PUBLIC targets:
///
/// * `t_targets[i] = F̄·v_i` — each instance's level-1 fold target (the
///   verifier computes it from the instance's `Σ_j d_j·y_j`);
/// * `u_targets[i] = Φ(v_i)` — each instance's functional claim;
/// * `f_bar_blocks` — the SHARED level-1 key's column blocks (the
///   verifier regenerates them from the seed);
/// * `psi_weights` — the functional's per-coefficient Goldilocks
///   weights (the Φ of the folded response is checked against the
///   γ-weighted target sum).
///
/// Fails closed when the profile is below the security floor.
#[allow(clippy::too_many_arguments)]
pub fn prove_second_fold(
    ring: &RingConfig,
    v_parts: &[Vec<RingElement>],
    t_targets: &[Vec<RingElement>],
    u_targets: &[Goldilocks],
    f_bar_blocks: &[Vec<RingElement>],
    psi_weights: &[Goldilocks],
    params: SecondFoldParams,
    beta1: u64,
    seed: [u8; 32],
    transcript: &mut Transcript,
) -> Result<SecondFoldProof, String> {
    let q = u64::from(ring.modulus.q);
    let n = ring.n();
    let r = params.r;
    let n_bar = params.n_bar;
    if v_parts.len() != r || t_targets.len() != r || u_targets.len() != r {
        return Err(format!(
            "shape: {} parts / {} targets / {} functionals (want {r})",
            v_parts.len(),
            t_targets.len(),
            u_targets.len()
        ));
    }
    for p in v_parts {
        if p.len() != n_bar {
            return Err(format!("shape: part width {} != n_bar {n_bar}", p.len()));
        }
    }
    if f_bar_blocks.len() != n_bar {
        return Err(format!(
            "shape: f_bar blocks {} != n_bar {n_bar}",
            f_bar_blocks.len()
        ));
    }
    // The fail-closed profile gate.
    let (cl, _qm) = assert_sound(&params, beta1, q, n as u64)?;
    // The inner commitments under the fresh key A₂ (derived from the
    // second-fold domain so it is independent of the level-1 key).
    let a2_params = AjtaiParams {
        ring: ring.clone(),
        k: params.kappa,
        m: n_bar,
        norm_bound: u32::try_from(params.beta2(beta1).max(1)).map_err(|e| format!("{e:?}"))?,
    };
    let a2_seed = derive_a2_seed(seed);
    let a2 = AjtaiPublicKey::from_seed(a2_params, a2_seed).map_err(|e| format!("{e:?}"))?;
    let t_inner: Vec<Vec<RingElement>> = v_parts
        .iter()
        .map(|p| a2.commit(p).map(|c| c.rows))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{e:?}"))?;
    // Absorb the pre-challenge material (FS hygiene).
    let flat_inner: Vec<RingElement> = t_inner.iter().flatten().cloned().collect();
    let inner_bytes = crate::compact::serialize_elements(ring, &flat_inner);
    transcript
        .append_bytes(b"sf-inner", &inner_bytes)
        .map_err(|e| format!("{e:?}"))?;
    // The scalar challenges γ_i (the γ wiring).
    let gammas: Vec<i64> = (0..r)
        .map(|_i| -> Result<i64, String> {
            let b = transcript
                .challenge_bytes(b"sf-gamma", 2)
                .map_err(|e| format!("{e:?}"))?;
            let raw = u16::from_le_bytes([b[0], b[1]]) as u64;
            let m = 2 * params.amplitude as u64 + 1;
            Ok((raw % m) as i64 - params.amplitude as i64)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // The amortized response z = Σ γ_i v_i.
    let mut z: Vec<RingElement> = vec![ring.zero(); n_bar];
    for (i, g) in gammas.iter().enumerate() {
        if *g == 0 {
            continue;
        }
        for c in 0..n_bar {
            let prod = v_parts[i][c].scale_i64(*g);
            z[c] = z[c].add(&prod).map_err(|e| format!("{e:?}"))?;
        }
    }
    // The norm gate: |z|_∞ ≤ β₂ (fail-closed completeness).
    let beta2 = params.beta2(beta1);
    if beta2 >= q / 2 {
        return Err(format!("beta2 exceeds q/2: {beta2} vs {}", q / 2));
    }
    let mut z_coeffs: Vec<i32> = Vec::with_capacity(n_bar * n);
    for e in &z {
        for &c in e.coeffs() {
            let balanced = if u64::from(c) > q / 2 {
                i64::from(c) - q as i64
            } else {
                i64::from(c)
            };
            if balanced.unsigned_abs() > beta2 {
                return Err(format!("second-fold gate: |{balanced}| > {beta2}"));
            }
            z_coeffs.push(balanced as i32);
        }
    }
    // The self-checks (L1)-(L3) — the prover verifies its own
    // construction before transmitting.
    {
        // (L1) F̄·z = Σ γ_i t_i.
        let fz = apply_key(ring, f_bar_blocks, &z, params.k);
        let mut rhs = vec![ring.zero(); params.k];
        for (i, g) in gammas.iter().enumerate() {
            if *g == 0 {
                continue;
            }
            for rr in 0..params.k {
                let prod = t_targets[i][rr].scale_i64(*g);
                rhs[rr] = rhs[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
            }
        }
        for (rr, e) in fz.iter().enumerate() {
            if e.coeffs() != rhs[rr].coeffs() {
                return Err("self-check L1 (public-target fold)".into());
            }
        }
        // (L2) A₂·z = Σ γ_i T_i.
        let az = a2
            .commit(&z)
            .map(|c| c.rows)
            .map_err(|e| format!("{e:?}"))?;
        let mut rhs2 = vec![ring.zero(); params.kappa];
        for (i, g) in gammas.iter().enumerate() {
            if *g == 0 {
                continue;
            }
            for rr in 0..params.kappa {
                let prod = t_inner[i][rr].scale_i64(*g);
                rhs2[rr] = rhs2[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
            }
        }
        for (rr, e) in az.iter().enumerate() {
            if e.coeffs() != rhs2[rr].coeffs() {
                return Err("self-check L2 (inner commitments)".into());
            }
        }
        // (L3) Φ(z) = Σ γ_i u_i.
        let phi_z = functional_of(ring, &z, psi_weights, q);
        let mut rhs3 = Goldilocks::ZERO;
        for (i, g) in gammas.iter().enumerate() {
            if *g != 0 {
                let term = u_targets[i].mul(&Goldilocks::from_u64(g.unsigned_abs()));
                rhs3 = if *g < 0 { rhs3.sub(&term) } else { rhs3.add(&term) };
            }
        }
        if phi_z != rhs3 {
            return Err("self-check L3 (commuting functional)".into());
        }
    }
    // Absorb the amortized response.
    transcript
        .append_bytes(b"sf-z", &crate::compact::serialize_elements(ring, &z))
        .map_err(|e| format!("{e:?}"))?;
    let response = crate::compact::encode_response(&z_coeffs).map_err(|e| e)?;
    Ok(SecondFoldProof {
        params: params.clone(),
        t_inner: inner_bytes,
        response,
        classical_bits: cl,
    })
}

/// Verify a second-fold proof given the PUBLIC targets `(t_i, u_i)`,
/// the shared level-1 key blocks (regenerated from the seed), and the
/// ψ weights. Replays the transcript, checks (L1)-(L4), and re-derives
/// the profile verdict (the fail-closed posture gate — the verifier
/// refuses proofs recorded below the floor).
#[allow(clippy::too_many_arguments)]
pub fn verify_second_fold(
    ring: &RingConfig,
    t_targets: &[Vec<RingElement>],
    u_targets: &[Goldilocks],
    f_bar_blocks: &[Vec<RingElement>],
    psi_weights: &[Goldilocks],
    beta1: u64,
    seed: [u8; 32],
    proof: &SecondFoldProof,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let q = u64::from(ring.modulus.q);
    let n = ring.n();
    let params = &proof.params;
    let r = params.r;
    let n_bar = params.n_bar;
    if t_targets.len() != r || u_targets.len() != r {
        return Err(format!(
            "shape: {} targets / {} functionals (want {r})",
            t_targets.len(),
            u_targets.len()
        ));
    }
    if f_bar_blocks.len() != n_bar {
        return Err(format!(
            "shape: f_bar blocks {} != n_bar {n_bar}",
            f_bar_blocks.len()
        ));
    }
    // The fail-closed profile re-derivation.
    let (cl, _) = assert_sound(params, beta1, q, n as u64)?;
    if (cl - proof.classical_bits).abs() > 1.0 {
        return Err(format!(
            "posture marker mismatch: proof {} vs verifier {cl:.1}",
            proof.classical_bits
        ));
    }
    // Deserialize the inner commitments.
    let t_inner =
        crate::compact::deserialize_elements(ring, &proof.t_inner).map_err(|e| e)?;
    if t_inner.len() != r * params.kappa {
        return Err(format!(
            "inner commitment count {} != {}",
            t_inner.len(),
            r * params.kappa
        ));
    }
    // The transcript replay (pre-challenge material).
    transcript
        .append_bytes(b"sf-inner", &proof.t_inner)
        .map_err(|e| format!("{e:?}"))?;
    // The challenges.
    let gammas: Vec<i64> = (0..r)
        .map(|_i| -> Result<i64, String> {
            let b = transcript
                .challenge_bytes(b"sf-gamma", 2)
                .map_err(|e| format!("{e:?}"))?;
            let raw = u16::from_le_bytes([b[0], b[1]]) as u64;
            let m = 2 * params.amplitude as u64 + 1;
            Ok((raw % m) as i64 - params.amplitude as i64)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Decode + gate z.
    let z_coeffs = crate::compact::decode_response(&proof.response).map_err(|e| e)?;
    if z_coeffs.len() != n_bar * n {
        return Err("response length".into());
    }
    let beta2 = params.beta2(beta1);
    if beta2 >= q / 2 {
        return Err("beta2 exceeds q/2".into());
    }
    let mut z: Vec<RingElement> = Vec::with_capacity(n_bar);
    for chunk in z_coeffs.chunks(n) {
        let mut coeffs = vec![0u32; n];
        for (i, &c) in chunk.iter().enumerate() {
            if (c as i64).abs() > beta2 as i64 {
                return Err("norm gate".into());
            }
            coeffs[i] = (c as i64).rem_euclid(q as i64) as u32;
        }
        z.push(RingElement::from_coeffs(ring, coeffs));
    }
    transcript
        .append_bytes(b"sf-z", &crate::compact::serialize_elements(ring, &z))
        .map_err(|e| format!("{e:?}"))?;
    // (L1) F̄·z = Σ γ_i t_i — both sides verifier-computable.
    {
        let fz = apply_key(ring, f_bar_blocks, &z, params.k);
        let mut rhs = vec![ring.zero(); params.k];
        for (i, g) in gammas.iter().enumerate() {
            if *g == 0 {
                continue;
            }
            for rr in 0..params.k {
                let prod = t_targets[i][rr].scale_i64(*g);
                rhs[rr] = rhs[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
            }
        }
        for (rr, e) in fz.iter().enumerate() {
            if e.coeffs() != rhs[rr].coeffs() {
                return Err("L1: the public-target fold".into());
            }
        }
    }
    // (L2) A₂·z = Σ γ_i T_i — the short MSIS binding.
    {
        let a2_params = AjtaiParams {
            ring: ring.clone(),
            k: params.kappa,
            m: n_bar,
            norm_bound: u32::try_from(params.beta2(beta1).max(1))
                .map_err(|e| format!("{e:?}"))?,
        };
        let a2_seed = derive_a2_seed(seed);
        let a2 = AjtaiPublicKey::from_seed(a2_params, a2_seed).map_err(|e| format!("{e:?}"))?;
        let az = a2
            .commit(&z)
            .map(|c| c.rows)
            .map_err(|e| format!("{e:?}"))?;
        let mut rhs = vec![ring.zero(); params.kappa];
        for (i, g) in gammas.iter().enumerate() {
            if *g == 0 {
                continue;
            }
            for rr in 0..params.kappa {
                let prod = t_inner[i * params.kappa + rr].scale_i64(*g);
                rhs[rr] = rhs[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
            }
        }
        for (rr, e) in az.iter().enumerate() {
            if e.coeffs() != rhs[rr].coeffs() {
                return Err("L2: inner-commitment binding (the short MSIS instance)".into());
            }
        }
    }
    // (L3) Φ(z) = Σ γ_i u_i.
    {
        let phi_z = functional_of(ring, &z, psi_weights, q);
        let mut rhs = Goldilocks::ZERO;
        for (i, g) in gammas.iter().enumerate() {
            if *g != 0 {
                let term = u_targets[i].mul(&Goldilocks::from_u64(g.unsigned_abs()));
                rhs = if *g < 0 { rhs.sub(&term) } else { rhs.add(&term) };
            }
        }
        if phi_z != rhs {
            return Err("L3: the commuting functional".into());
        }
    }
    Ok(())
}

/// The A₂ key seed (the second-fold domain, independent of level-1).
fn derive_a2_seed(seed: [u8; 32]) -> [u8; 32] {
    let mut st = Transcript::new_default(b"lzx-second-fold-key");
    let _ = st.append_bytes(b"seed", &seed);
    let mut s = [0u8; 32];
    if let Ok(b) = st.challenge_bytes(b"key", 32) {
        s.copy_from_slice(&b);
    }
    s
}

/// Apply a column-block key to a response vector: rows = Σ_c block_c·v_c.
/// (Shared with the width fold — the level-1 key-group application.)
pub(crate) fn apply_key(
    ring: &RingConfig,
    blocks: &[Vec<RingElement>],
    v: &[RingElement],
    k: usize,
) -> Vec<RingElement> {
    let mut out = vec![ring.zero(); k];
    for (c, blk) in blocks.iter().enumerate() {
        if c >= v.len() {
            break;
        }
        for (rr, e) in blk.iter().enumerate().take(k) {
            if let Ok(prod) = v[c].mul(e) {
                if let Ok(s) = out[rr].add(&prod) {
                    out[rr] = s;
                }
            }
        }
    }
    out
}

/// The Goldilocks functional of a response: Φ(v) = Σ ψ_m · balanced(v_m).
/// (Shared with the width fold — the ψ-slice functional.)
pub(crate) fn functional_of(
    ring: &RingConfig,
    v: &[RingElement],
    psi_weights: &[Goldilocks],
    q: u64,
) -> Goldilocks {
    let n = ring.n();
    let mut acc = Goldilocks::ZERO;
    for (c, e) in v.iter().enumerate() {
        for d in 0..n {
            let w = psi_weights
                .get(c * n + d)
                .copied()
                .unwrap_or(Goldilocks::ZERO);
            if w != Goldilocks::ZERO {
                let term = crate::compact::phi_term(&w, e.coeffs()[d], q as u32);
                acc = acc.add(&term);
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingConfig {
        crate::compact::column_ring().unwrap()
    }

    /// Synthetic level-1 instances: r responses of byte-valued width-
    /// n_bar elements, a shared key, and the derived public targets.
    fn synth(
        ring: &RingConfig,
        r: usize,
        n_bar: usize,
        k: usize,
    ) -> (
        Vec<Vec<RingElement>>,
        Vec<Vec<RingElement>>,
        Vec<Goldilocks>,
        Vec<Vec<RingElement>>,
        Vec<Goldilocks>,
    ) {
        let n = ring.n();
        let mut rng: u64 = 0x9E3779B97F4A7C15;
        // The shared key blocks.
        let mut blocks = Vec::with_capacity(n_bar);
        for _ in 0..n_bar {
            let mut blk = Vec::with_capacity(k);
            for _ in 0..k {
                let mut coeffs = vec![0u32; n];
                for c in coeffs.iter_mut() {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *c = ((rng >> 33) % 97) as u32;
                }
                blk.push(RingElement::from_coeffs(ring, coeffs));
            }
            blocks.push(blk);
        }
        // The responses + targets.
        let mut parts = Vec::with_capacity(r);
        let mut targets = Vec::with_capacity(r);
        let mut us = Vec::with_capacity(r);
        for _ in 0..r {
            let mut v = Vec::with_capacity(n_bar);
            for _ in 0..n_bar {
                let mut coeffs = vec![0u32; n];
                for c in coeffs.iter_mut() {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *c = ((rng >> 33) % 256) as u32;
                }
                v.push(RingElement::from_coeffs(ring, coeffs));
            }
            let t = apply_key(ring, &blocks, &v, k);
            // The functional weights (shared across instances).
            let mut psi = Vec::with_capacity(n_bar * n);
            for _ in 0..n_bar * n {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                psi.push(Goldilocks::from_u64((rng >> 33) % 1_000_003));
            }
            let u = functional_of(ring, &v, &psi, u64::from(ring.modulus.q));
            parts.push(v);
            targets.push(t);
            us.push(u);
            // (keep one psi for the fold — the verifier's weights)
            if targets.len() == 1 {
                // store the weights: re-derive below
            }
        }
        // The functional weights (fixed, shared).
        let mut psi = Vec::with_capacity(n_bar * n);
        for _ in 0..n_bar * n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            psi.push(Goldilocks::from_u64((rng >> 33) % 1_000_003));
        }
        // Recompute the targets' functionals with the shared psi.
        let us: Vec<Goldilocks> = parts
            .iter()
            .map(|v| functional_of(ring, v, &psi, u64::from(ring.modulus.q)))
            .collect();
        (parts, targets, us, blocks, psi)
    }

    #[test]
    fn second_fold_honest_roundtrip() {
        let ring = ring();
        let params = SecondFoldParams::sound_defaults();
        let (parts, targets, us, blocks, psi) =
            synth(&ring, params.r, params.n_bar, params.k);
        let beta1 = 255u64; // the byte-packed level-1 gate
        let seed = [7u8; 32];
        let mut tr = Transcript::new_default(b"sf-test");
        let proof = prove_second_fold(
            &ring, &parts, &targets, &us, &blocks, &psi, params.clone(), beta1, seed, &mut tr,
        )
        .unwrap();
        // Verify with a fresh transcript.
        let mut vt = Transcript::new_default(b"sf-test");
        assert!(verify_second_fold(
            &ring, &targets, &us, &blocks, &psi, beta1, seed, &proof, &mut vt
        )
        .is_ok());
        // The profile verdict is recorded and above the floor.
        assert!(proof.classical_bits >= SECURITY_FLOOR_BITS);
    }

    #[test]
    fn second_fold_tampered_z_rejected() {
        let ring = ring();
        let params = SecondFoldParams::sound_defaults();
        let (parts, targets, us, blocks, psi) =
            synth(&ring, params.r, params.n_bar, params.k);
        let beta1 = 255u64;
        let seed = [7u8; 32];
        let mut tr = Transcript::new_default(b"sf-test");
        let proof = prove_second_fold(
            &ring, &parts, &targets, &us, &blocks, &psi, params.clone(), beta1, seed, &mut tr,
        )
        .unwrap();
        // Tamper: flip a coefficient of the decoded response.
        let mut bad = proof.clone();
        let mut coeffs = crate::compact::decode_response(&bad.response).unwrap();
        if let Some(c0) = coeffs.first_mut() {
            *c0 = c0.wrapping_add(1);
        }
        bad.response = crate::compact::encode_response(&coeffs).unwrap();
        let mut vt = Transcript::new_default(b"sf-test");
        assert!(verify_second_fold(
            &ring, &targets, &us, &blocks, &psi, beta1, seed, &bad, &mut vt
        )
        .is_err());
    }

    #[test]
    fn second_fold_tampered_inner_rejected() {
        let ring = ring();
        let params = SecondFoldParams::sound_defaults();
        let (parts, targets, us, blocks, psi) =
            synth(&ring, params.r, params.n_bar, params.k);
        let beta1 = 255u64;
        let seed = [7u8; 32];
        let mut tr = Transcript::new_default(b"sf-test");
        let proof = prove_second_fold(
            &ring, &parts, &targets, &us, &blocks, &psi, params.clone(), beta1, seed, &mut tr,
        )
        .unwrap();
        let mut bad = proof.clone();
        if let Some(b) = bad.t_inner.first_mut() {
            *b ^= 0x01;
        }
        let mut vt = Transcript::new_default(b"sf-test");
        assert!(verify_second_fold(
            &ring, &targets, &us, &blocks, &psi, beta1, seed, &bad, &mut vt
        )
        .is_err());
    }

    #[test]
    fn second_fold_wrong_target_rejected() {
        let ring = ring();
        let params = SecondFoldParams::sound_defaults();
        let (parts, targets, us, blocks, psi) =
            synth(&ring, params.r, params.n_bar, params.k);
        let beta1 = 255u64;
        let seed = [7u8; 32];
        let mut tr = Transcript::new_default(b"sf-test");
        let proof = prove_second_fold(
            &ring, &parts, &targets, &us, &blocks, &psi, params.clone(), beta1, seed, &mut tr,
        )
        .unwrap();
        // A different public target (the wrong statement).
        let mut t2 = targets.clone();
        if let Some(e0) = t2[0].first_mut() {
            let mut coeffs = e0.coeffs().to_vec();
            if let Some(c0) = coeffs.first_mut() {
                *c0 = (*c0 + 1) % ring.modulus.q;
            }
            *e0 = RingElement::from_coeffs(&ring, coeffs);
        }
        let mut vt = Transcript::new_default(b"sf-test");
        assert!(verify_second_fold(
            &ring, &t2, &us, &blocks, &psi, beta1, seed, &proof, &mut vt
        )
        .is_err());
    }

    #[test]
    fn second_fold_wrong_functional_rejected() {
        let ring = ring();
        let params = SecondFoldParams::sound_defaults();
        let (parts, targets, us, blocks, psi) =
            synth(&ring, params.r, params.n_bar, params.k);
        let beta1 = 255u64;
        let seed = [7u8; 32];
        let mut tr = Transcript::new_default(b"sf-test");
        let proof = prove_second_fold(
            &ring, &parts, &targets, &us, &blocks, &psi, params.clone(), beta1, seed, &mut tr,
        )
        .unwrap();
        let mut u2 = us.clone();
        if let Some(u0) = u2.first_mut() {
            *u0 = u0.add(&Goldilocks::ONE);
        }
        let mut vt = Transcript::new_default(b"sf-test");
        assert!(verify_second_fold(
            &ring, &targets, &u2, &blocks, &psi, beta1, seed, &proof, &mut vt
        )
        .is_err());
    }

    #[test]
    fn profile_gate_rejects_broken_shapes() {
        // The fail-closed gate: a profile whose final instance is below
        // the floor must be refused (the estimator verdict, not an
        // assertion).
        let params = SecondFoldParams {
            r: 16,
            kappa: 2,
            k: 4,
            amplitude: 1 << 12,
            n_bar: 64,
        };
        let beta1 = 255u64;
        let ring = ring();
        let res = assert_sound(&params, beta1, u64::from(ring.modulus.q), ring.n() as u64);
        assert!(res.is_err());
    }

    #[test]
    fn profile_sound_defaults_pass() {
        let ring = ring();
        let params = SecondFoldParams::sound_defaults();
        let beta1 = 255u64;
        let (cl, qm) =
            assert_sound(&params, beta1, u64::from(ring.modulus.q), ring.n() as u64).unwrap();
        // The SECURITY.md sound row: 329+ classical bits.
        assert!(cl > 300.0, "classical {cl}");
        assert!(qm > 100.0, "quantum {qm}");
    }
}
