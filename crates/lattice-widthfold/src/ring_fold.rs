//! **The ring-functional width fold** — the compact-PCS terminal layer
//! for the Cyclo §7 bridge: the width-reducing fold carrying EXACT
//! mod-q linear identities (R_q-valued functionals) instead of the
//! Goldilocks coefficient functional.
//!
//! # Why a separate functional layer
//!
//! The zkvm Sound profile binds its response through a GOLDILOCKS
//! functional (`Φ(v) = u`, the carrier's claim) — the
//! two-characteristic discipline: the claims live in the sumchecks'
//! field. The Cyclo bridge's decider claims are RING identities over
//! `R_q` (the (4) linear claims `Σ_{b'} MLE[M_i](u,b')^{(b)}·z'_{b'} =
//! d'_i^{(b)}` and the prefix claim's eq-weighted sums) whose public
//! scalars are full-range `F_q` values — no homomorphism carries them
//! into Goldilocks. This module keeps the width fold's binding
//! structure ((W0)–(W2), (W4) — identical to `fold.rs`) and swaps the
//! functional layer for exact ring arithmetic:
//!
//! * the per-part functional values `U^t_i = Λ_t^{(i)}(s_i) ∈ R_q`
//!   and the functional garbage `G^t_ij = Λ_t^{(i)}(s_j)` are
//!   committed BEFORE the challenges (the LaBRADOR pre-challenge
//!   discipline);
//! * `(W3R)` per functional `t` and slice `i`:
//!   `Λ_t^{(i)}(z) = γ_i·U^t_i + Σ_{j≠i} γ_j·G^t_ij` — the exact
//!   ring superposition (the same identity shape as (W3), over R_q);
//! * `(W0R)` the functional consistency: `Σ_i U^t_i = Λ_t(v)` — the
//!   CALLER's public claim (the bridge's `d'_i` lifts and the projected
//!   prefix value) checked against the transmitted per-part values.
//!
//! The binding story is the width fold's own: `(W0)` pins the parts'
//! images to the public commitment target, `(W2)` is the short MSIS
//! instance `[A₂ | −T]` (estimator-gated fail-closed), `(W1)`/`(W3R)`
//! are the consistency carriers that close the hole a folded response
//! unrelated to the pre-committed material would open. The honest
//! residual (the multi-fork LaBRADOR extraction) is documented in the
//! consuming module.

use crate::codec::{decode_response, encode_response, serialize_elements, ResponseWire};
use crate::fold::WidthFoldParams;
use crate::helpers::apply_key;

use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_ring::ring::{RingConfig, RingElement};

/// An R_q-valued linear functional over a response: one F_q scalar per
/// ring element (`Λ(v) = Σ_c λ_c·v_c ∈ R_q`). The scalars are PUBLIC
/// (caller-derived: the bridge's matrix-MLE rows and eq tables).
#[derive(Clone, Debug)]
pub struct RingFunctional {
    /// The per-ring-element scalars (mod q; length ≥ the response
    /// width — weights beyond the response are ZERO).
    pub weights: Vec<u64>,
}

impl RingFunctional {
    /// The functional restricted to part `i` (weights
    /// `[i·w, (i+1)·w)`), applied to a width-`w` vector.
    fn slice_eval(
        &self,
        ring: &RingConfig,
        x: &[RingElement],
        part_i: usize,
        w: usize,
    ) -> RingElement {
        let q = u64::from(ring.modulus.q);
        let mut acc = ring.zero();
        for c in 0..w {
            let lam = self.weights.get(part_i * w + c).copied().unwrap_or(0);
            if lam == 0 {
                continue;
            }
            // λ ∈ F_q as a balanced scalar (|λ| ≤ q/2 keeps the
            // products' magnitudes honest for the norm accounting).
            let lam_bal = if lam > q / 2 {
                lam as i64 - q as i64
            } else {
                lam as i64
            };
            if lam_bal == 0 {
                continue;
            }
            let term = x[c].scale_i64(lam_bal);
            acc = acc.add(&term).unwrap_or(acc);
        }
        acc
    }

    /// The full functional over a response.
    pub fn eval(&self, ring: &RingConfig, v: &[RingElement]) -> RingElement {
        let q = u64::from(ring.modulus.q);
        let mut acc = ring.zero();
        for (c, e) in v.iter().enumerate() {
            let lam = self.weights.get(c).copied().unwrap_or(0);
            if lam == 0 {
                continue;
            }
            let lam_bal = if lam > q / 2 {
                lam as i64 - q as i64
            } else {
                lam as i64
            };
            if lam_bal == 0 {
                continue;
            }
            let term = e.scale_i64(lam_bal);
            acc = acc.add(&term).unwrap_or(acc);
        }
        acc
    }
}

/// The ring-functional width-fold proof (the bridge terminal's
/// artifact — replaces the opened witness).
#[derive(Clone, Debug)]
pub struct RingFoldProof {
    pub params: WidthFoldParams,
    /// The input response width (before zero padding to `w·r₂`).
    pub n_bar: usize,
    /// The part images `p_i` (r₂·k ring elements, serialized).
    pub p_images: Vec<u8>,
    /// The quadratic garbage `G_ij`, i≠j (r₂·(r₂−1)·k ring elements).
    pub garbage: Vec<u8>,
    /// The inner commitments `T_i` (r₂·κ ring elements, serialized).
    pub t_inner: Vec<u8>,
    /// The per-functional per-part values `U^t_i` (T_count·r₂ ring
    /// elements, functional-major, serialized).
    pub func_values: Vec<u8>,
    /// The per-functional garbage `G^t_ij` (T_count·r₂·(r₂−1) ring
    /// elements, functional-major then row-major over (i, j≠i)).
    pub func_garbage: Vec<u8>,
    /// The folded response `z ∈ R^w` (rANS-coded coefficients).
    pub response: ResponseWire,
    /// The estimator verdict at prove time (the posture marker).
    pub classical_bits: f64,
}

/// The A₂ key seed (the same domain derivation as `fold.rs`).
fn derive_w2_seed(seed: [u8; 32], params: &WidthFoldParams) -> [u8; 32] {
    let mut st = Transcript::new_default(b"lzx-width-fold-key");
    let _ = st.append_bytes(b"seed", &seed);
    let shape = [
        params.r2 as u32,
        params.kappa as u32,
        params.amplitude,
        params.w as u32,
    ];
    let _ = st.append_bytes(
        b"shape",
        &shape.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    );
    let mut s = [0u8; 32];
    if let Ok(b) = st.challenge_bytes(b"key", 32) {
        s.copy_from_slice(&b);
    }
    s
}

fn split_parts(
    ring: &RingConfig,
    v: &[RingElement],
    w: usize,
    r2: usize,
) -> Vec<Vec<RingElement>> {
    (0..r2)
        .map(|i| {
            let lo = i * w;
            let mut part: Vec<RingElement> = Vec::with_capacity(w);
            for c in 0..w {
                part.push(v.get(lo + c).cloned().unwrap_or_else(|| ring.zero()));
            }
            part
        })
        .collect()
}

fn key_groups(
    ring: &RingConfig,
    f_bar_blocks: &[Vec<RingElement>],
    k: usize,
    w: usize,
    r2: usize,
) -> Vec<Vec<Vec<RingElement>>> {
    (0..r2)
        .map(|i| {
            (0..w)
                .map(|c| {
                    f_bar_blocks
                        .get(i * w + c)
                        .cloned()
                        .unwrap_or_else(|| vec![ring.zero(); k])
                })
                .collect()
        })
        .collect()
}

/// Prove the ring-functional width fold: the response `v ∈ R^{n̄}`
/// under the key blocks `f_bar_blocks` (rank k), the PUBLIC image
/// target `t_target`, and the functional targets — `Some(target)` for
/// the EXACT claims (`Λ_t(v) = target`, checked by the fold's (W0R)),
/// `None` for the PROJECTED claims (the caller checks its own
/// projection of the fold's public per-part sums — e.g. the Cyclo
/// bridge's `θ_k(Λ_t(v)) = e`).
///
/// Fails closed on: the profile floor, shape mismatches, the prover's
/// self-checks ((W0), the functional targets, (W1), (W2), (W3R)), the
/// β₂ < q/2 gate, and the response norm gate.
#[allow(clippy::too_many_arguments)]
pub fn prove_ring_fold(
    ring: &RingConfig,
    v: &[RingElement],
    t_target: &[RingElement],
    functionals: &[RingFunctional],
    functional_targets: &[Option<RingElement>],
    f_bar_blocks: &[Vec<RingElement>],
    k: usize,
    params: WidthFoldParams,
    beta1: u64,
    seed: [u8; 32],
    transcript: &mut Transcript,
) -> Result<RingFoldProof, String> {
    let q = u64::from(ring.modulus.q);
    let n = ring.n();
    let n_bar = v.len();
    let r2 = params.r2;
    let w = params.w;
    let t_count = functionals.len();
    if functional_targets.len() != t_count {
        return Err("functional target count".into());
    }
    if f_bar_blocks.len() < n_bar {
        return Err(format!(
            "shape: key blocks {} < response width {n_bar}",
            f_bar_blocks.len()
        ));
    }
    if t_target.len() != k {
        return Err(format!("shape: target rank {} != k {k}", t_target.len()));
    }
    if w * r2 < n_bar {
        return Err(format!(
            "shape: w·r₂ = {} < n̄ = {n_bar}",
            w * r2
        ));
    }
    // The fail-closed profile gate.
    let (cl, _qm) = params.assert_sound(beta1, q, n as u64)?;

    // The prover's self-checks: the EXACT targets are the caller's
    // public claims (a wrong input cannot even start). The projected
    // targets are the caller's to check on the fold's public sums.
    for (t, func) in functionals.iter().enumerate() {
        if let Some(target) = &functional_targets[t] {
            let val = func.eval(ring, v);
            if val.coeffs() != target.coeffs() {
                return Err(format!(
                    "self-check: Λ_{t}(v) ≠ the public functional target"
                ));
            }
        }
    }

    let parts = split_parts(ring, v, w, r2);
    let groups = key_groups(ring, f_bar_blocks, k, w, r2);

    // The pre-challenge material.
    let p: Vec<Vec<RingElement>> = (0..r2)
        .map(|i| apply_key(ring, &groups[i], &parts[i], k))
        .collect();
    let mut garbage: Vec<RingElement> = Vec::with_capacity(r2 * (r2 - 1) * k);
    for i in 0..r2 {
        for j in 0..r2 {
            if i != j {
                garbage.extend_from_slice(&apply_key(ring, &groups[i], &parts[j], k));
            }
        }
    }
    // T_i = A₂·s_i — the inner commitments.
    let a2_params = AjtaiParams {
        ring: ring.clone(),
        k: params.kappa,
        m: w,
        norm_bound: u32::try_from(params.beta2(beta1).max(1)).map_err(|e| format!("{e:?}"))?,
    };
    let a2_seed = derive_w2_seed(seed, &params);
    let a2 = AjtaiPublicKey::from_seed(a2_params, a2_seed).map_err(|e| format!("{e:?}"))?;
    let t_inner: Vec<Vec<RingElement>> = parts
        .iter()
        .map(|s| a2.commit(s).map(|c| c.rows))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{e:?}"))?;
    // The functional layer: U^t_i and G^t_ij.
    let mut func_values: Vec<RingElement> = Vec::with_capacity(t_count * r2);
    let mut func_garbage: Vec<RingElement> = Vec::with_capacity(t_count * r2 * (r2 - 1));
    for func in functionals {
        for i in 0..r2 {
            func_values.push(func.slice_eval(ring, &parts[i], i, w));
        }
        for i in 0..r2 {
            for j in 0..r2 {
                if i != j {
                    func_garbage.push(func.slice_eval(ring, &parts[j], i, w));
                }
            }
        }
    }

    // (W0) Σ p_i = t — the public image consistency self-check.
    {
        let mut sum = vec![ring.zero(); k];
        for p_i in &p {
            for rr in 0..k {
                sum[rr] = sum[rr].add(&p_i[rr]).map_err(|e| format!("{e:?}"))?;
            }
        }
        for rr in 0..k {
            if sum[rr].coeffs() != t_target[rr].coeffs() {
                return Err("self-check W0: Σ p_i ≠ t".into());
            }
        }
    }
    // (W0R) Σ_i U^t_i = Λ_t(v) — the functional consistency self-check
    // (the EXACT family only; the projected family's sums are the
    // caller's public data through the (W3R)-bound per-part values).
    for (t, target) in functional_targets.iter().enumerate() {
        let Some(target) = target else { continue };
        let mut sum = ring.zero();
        for i in 0..r2 {
            sum = sum
                .add(&func_values[t * r2 + i])
                .map_err(|e| format!("{e:?}"))?;
        }
        if sum.coeffs() != target.coeffs() {
            return Err(format!("self-check W0R: functional {t}"));
        }
    }

    // The FS absorption (pre-challenge material).
    let p_bytes = serialize_elements(ring, &p.iter().flatten().cloned().collect::<Vec<_>>());
    let garbage_bytes = serialize_elements(ring, &garbage);
    let t_bytes = serialize_elements(
        ring,
        &t_inner.iter().flatten().cloned().collect::<Vec<_>>(),
    );
    let fv_bytes = serialize_elements(ring, &func_values);
    let fg_bytes = serialize_elements(ring, &func_garbage);
    transcript
        .append_bytes(b"rf-p", &p_bytes)
        .and_then(|_| transcript.append_bytes(b"rf-g", &garbage_bytes))
        .and_then(|_| transcript.append_bytes(b"rf-t", &t_bytes))
        .and_then(|_| transcript.append_bytes(b"rf-fv", &fv_bytes))
        .and_then(|_| transcript.append_bytes(b"rf-fg", &fg_bytes))
        .map_err(|e| format!("{e:?}"))?;

    // The scalar challenges γ_i ∈ [−A, A].
    let gammas: Vec<i64> = (0..r2)
        .map(|_i| -> Result<i64, String> {
            let b = transcript
                .challenge_bytes(b"rf-gamma", 2)
                .map_err(|e| format!("{e:?}"))?;
            let raw = u16::from_le_bytes([b[0], b[1]]) as u64;
            let m = 2 * params.amplitude as u64 + 1;
            Ok((raw % m) as i64 - params.amplitude as i64)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // The folded response z = Σ_i γ_i·s_i.
    let beta2 = params.beta2(beta1);
    if beta2 >= q / 2 {
        return Err(format!("beta2 exceeds q/2: {beta2} vs {}", q / 2));
    }
    let mut z: Vec<RingElement> = vec![ring.zero(); w];
    for (i, g) in gammas.iter().enumerate() {
        if *g == 0 {
            continue;
        }
        for c in 0..w {
            let prod = parts[i][c].scale_i64(*g);
            z[c] = z[c].add(&prod).map_err(|e| format!("{e:?}"))?;
        }
    }
    let mut z_coeffs: Vec<i32> = Vec::with_capacity(w * n);
    for e in &z {
        for &c in e.coeffs() {
            let balanced = if u64::from(c) > q / 2 {
                i64::from(c) - q as i64
            } else {
                i64::from(c)
            };
            if balanced.unsigned_abs() > beta2 {
                return Err(format!("ring-fold gate: |{balanced}| > {beta2}"));
            }
            z_coeffs.push(balanced as i32);
        }
    }

    // (W1) Σ_i γ_i·(F̄_(i)·z) = Σ_i γ_i²·p_i + Σ_{i≠j} γ_iγ_j·G_ij.
    {
        let mut lhs = vec![ring.zero(); k];
        for (i, g) in gammas.iter().enumerate() {
            if *g == 0 {
                continue;
            }
            let fzi = apply_key(ring, &groups[i], &z, k);
            for rr in 0..k {
                let prod = fzi[rr].scale_i64(*g);
                lhs[rr] = lhs[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
            }
        }
        let mut rhs = vec![ring.zero(); k];
        for (i, gi) in gammas.iter().enumerate() {
            if *gi == 0 {
                continue;
            }
            for rr in 0..k {
                let prod = p[i][rr].scale_i64(gi * gi);
                rhs[rr] = rhs[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
            }
        }
        let mut idx = 0usize;
        for (i, gi) in gammas.iter().enumerate() {
            for (j, gj) in gammas.iter().enumerate() {
                if i != j && *gi != 0 && *gj != 0 {
                    let w_ij = gi * gj;
                    for rr in 0..k {
                        let prod = garbage[idx + rr].scale_i64(w_ij);
                        rhs[rr] = rhs[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
                    }
                }
                if i != j {
                    idx += k;
                }
            }
        }
        for rr in 0..k {
            if lhs[rr].coeffs() != rhs[rr].coeffs() {
                return Err("self-check W1: the exact fold identity".into());
            }
        }
    }
    // (W2) A₂·z = Σ_i γ_i·T_i.
    {
        let az = a2.commit(&z).map(|c| c.rows).map_err(|e| format!("{e:?}"))?;
        let mut rhs = vec![ring.zero(); params.kappa];
        for (i, g) in gammas.iter().enumerate() {
            if *g == 0 {
                continue;
            }
            for rr in 0..params.kappa {
                let prod = t_inner[i][rr].scale_i64(*g);
                rhs[rr] = rhs[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
            }
        }
        for (rr, e) in az.iter().enumerate() {
            if e.coeffs() != rhs[rr].coeffs() {
                return Err("self-check W2: the inner-commitment binding".into());
            }
        }
    }
    // (W3R) Λ_t^{(i)}(z) = γ_i·U^t_i + Σ_{j≠i} γ_j·G^t_ij.
    {
        for t in 0..t_count {
            let func = &functionals[t];
            for (i, gi) in gammas.iter().enumerate() {
                let lhs = func.slice_eval(ring, &z, i, w);
                let mut rhs = ring.zero();
                if *gi != 0 {
                    let term = func_values[t * r2 + i].scale_i64(*gi);
                    rhs = rhs.add(&term).map_err(|e| format!("{e:?}"))?;
                }
                let mut idx = t * r2 * (r2 - 1) + i * (r2 - 1);
                for (j, gj) in gammas.iter().enumerate() {
                    if i != j {
                        if *gj != 0 {
                            let term = func_garbage[idx].scale_i64(*gj);
                            rhs = rhs.add(&term).map_err(|e| format!("{e:?}"))?;
                        }
                        idx += 1;
                    }
                }
                if lhs.coeffs() != rhs.coeffs() {
                    return Err(format!(
                        "self-check W3R: functional {t} slice {i} superposition"
                    ));
                }
            }
        }
    }

    transcript
        .append_bytes(b"rf-z", &serialize_elements(ring, &z))
        .map_err(|e| format!("{e:?}"))?;
    let response = encode_response(&z_coeffs)?;
    Ok(RingFoldProof {
        params,
        n_bar,
        p_images: p_bytes,
        garbage: garbage_bytes,
        t_inner: t_bytes,
        func_values: fv_bytes,
        func_garbage: fg_bytes,
        response,
        classical_bits: cl,
    })
}

/// The fold's public per-part functional sum `Σ_i U^t_i` (the
/// projected family's caller-side check input — verifier-computable
/// from the transmitted per-part values).
pub fn ring_fold_functional_sum(
    ring: &RingConfig,
    proof: &RingFoldProof,
    t: usize,
) -> Result<RingElement, String> {
    let r2 = proof.params.r2;
    let func_values = crate::codec::deserialize_elements(ring, &proof.func_values)?;
    if func_values.len() <= t * r2 {
        return Err("functional index out of range".into());
    }
    let mut sum = ring.zero();
    for i in 0..r2 {
        sum = sum
            .add(&func_values[t * r2 + i])
            .map_err(|e| format!("{e:?}"))?;
    }
    Ok(sum)
}

/// Verify the ring-functional width fold: replays the transcript,
/// checks (W0), (W0R) against the PUBLIC functional targets, (W1),
/// (W2), (W3R), (W4), and re-derives the estimator posture.
#[allow(clippy::too_many_arguments)]
pub fn verify_ring_fold(
    ring: &RingConfig,
    t_target: &[RingElement],
    functionals: &[RingFunctional],
    functional_targets: &[Option<RingElement>],
    f_bar_blocks: &[Vec<RingElement>],
    k: usize,
    beta1: u64,
    seed: [u8; 32],
    proof: &RingFoldProof,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let q = u64::from(ring.modulus.q);
    let n = ring.n();
    let params = &proof.params;
    let r2 = params.r2;
    let w = params.w;
    let t_count = functionals.len();
    if functional_targets.len() != t_count {
        return Err("functional target count".into());
    }
    if t_target.len() != k {
        return Err(format!("shape: target rank {} != k {k}", t_target.len()));
    }
    if w * r2 < proof.n_bar {
        return Err("shape: w·r₂ < n̄".into());
    }
    // The posture re-derivation + marker check.
    let (cl, _) = params.assert_sound(beta1, q, n as u64)?;
    if (cl - proof.classical_bits).abs() > 1.0 {
        return Err(format!(
            "posture marker mismatch: proof {} vs verifier {cl:.1}",
            proof.classical_bits
        ));
    }
    // Deserialize the pre-challenge material.
    let p_flat = crate::codec::deserialize_elements(ring, &proof.p_images)?;
    if p_flat.len() != r2 * k {
        return Err(format!("p count {} != {}", p_flat.len(), r2 * k));
    }
    let garbage = crate::codec::deserialize_elements(ring, &proof.garbage)?;
    if garbage.len() != r2 * (r2 - 1) * k {
        return Err(format!(
            "garbage count {} != {}",
            garbage.len(),
            r2 * (r2 - 1) * k
        ));
    }
    let t_inner = crate::codec::deserialize_elements(ring, &proof.t_inner)?;
    if t_inner.len() != r2 * params.kappa {
        return Err(format!(
            "inner count {} != {}",
            t_inner.len(),
            r2 * params.kappa
        ));
    }
    let func_values = crate::codec::deserialize_elements(ring, &proof.func_values)?;
    if func_values.len() != t_count * r2 {
        return Err(format!(
            "functional values {} != {}",
            func_values.len(),
            t_count * r2
        ));
    }
    let func_garbage = crate::codec::deserialize_elements(ring, &proof.func_garbage)?;
    if func_garbage.len() != t_count * r2 * (r2 - 1) {
        return Err(format!(
            "functional garbage {} != {}",
            func_garbage.len(),
            t_count * r2 * (r2 - 1)
        ));
    }
    // (W0) Σ p_i = t.
    {
        let mut sum = vec![ring.zero(); k];
        for i in 0..r2 {
            for rr in 0..k {
                sum[rr] = sum[rr]
                    .add(&p_flat[i * k + rr])
                    .map_err(|e| format!("{e:?}"))?;
            }
        }
        for rr in 0..k {
            if sum[rr].coeffs() != t_target[rr].coeffs() {
                return Err("W0: the public image consistency".into());
            }
        }
    }
    // (W0R) Σ_i U^t_i = the public functional target (the EXACT
    // family; the projected family's sums are the caller's).
    for (t, target) in functional_targets.iter().enumerate() {
        let Some(target) = target else { continue };
        let mut sum = ring.zero();
        for i in 0..r2 {
            sum = sum
                .add(&func_values[t * r2 + i])
                .map_err(|e| format!("{e:?}"))?;
        }
        if sum.coeffs() != target.coeffs() {
            return Err(format!("W0R: functional {t} consistency"));
        }
    }
    // The transcript replay.
    transcript
        .append_bytes(b"rf-p", &proof.p_images)
        .and_then(|_| transcript.append_bytes(b"rf-g", &proof.garbage))
        .and_then(|_| transcript.append_bytes(b"rf-t", &proof.t_inner))
        .and_then(|_| transcript.append_bytes(b"rf-fv", &proof.func_values))
        .and_then(|_| transcript.append_bytes(b"rf-fg", &proof.func_garbage))
        .map_err(|e| format!("{e:?}"))?;
    let gammas: Vec<i64> = (0..r2)
        .map(|_i| -> Result<i64, String> {
            let b = transcript
                .challenge_bytes(b"rf-gamma", 2)
                .map_err(|e| format!("{e:?}"))?;
            let raw = u16::from_le_bytes([b[0], b[1]]) as u64;
            let m = 2 * params.amplitude as u64 + 1;
            Ok((raw % m) as i64 - params.amplitude as i64)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Decode + gate z.
    let z_coeffs = decode_response(&proof.response)?;
    if z_coeffs.len() != w * n {
        return Err("response length".into());
    }
    let beta2 = params.beta2(beta1);
    if beta2 >= q / 2 {
        return Err("beta2 exceeds q/2".into());
    }
    let mut z: Vec<RingElement> = Vec::with_capacity(w);
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
        .append_bytes(b"rf-z", &serialize_elements(ring, &z))
        .map_err(|e| format!("{e:?}"))?;

    let groups = key_groups(ring, f_bar_blocks, k, w, r2);

    // (W1) Σ_i γ_i·(F̄_(i)·z) = Σ_i γ_i²·p_i + Σ_{i≠j} γ_iγ_j·G_ij.
    {
        let mut lhs = vec![ring.zero(); k];
        for (i, g) in gammas.iter().enumerate() {
            if *g == 0 {
                continue;
            }
            let fzi = apply_key(ring, &groups[i], &z, k);
            for rr in 0..k {
                let prod = fzi[rr].scale_i64(*g);
                lhs[rr] = lhs[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
            }
        }
        let mut rhs = vec![ring.zero(); k];
        for (i, gi) in gammas.iter().enumerate() {
            if *gi == 0 {
                continue;
            }
            for rr in 0..k {
                let prod = p_flat[i * k + rr].scale_i64(gi * gi);
                rhs[rr] = rhs[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
            }
        }
        let mut idx = 0usize;
        for (i, gi) in gammas.iter().enumerate() {
            for (j, gj) in gammas.iter().enumerate() {
                if i != j {
                    if *gi != 0 && *gj != 0 {
                        let w_ij = gi * gj;
                        for rr in 0..k {
                            let prod = garbage[idx + rr].scale_i64(w_ij);
                            rhs[rr] = rhs[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
                        }
                    }
                    idx += k;
                }
            }
        }
        for rr in 0..k {
            if lhs[rr].coeffs() != rhs[rr].coeffs() {
                return Err("W1: the exact fold identity".into());
            }
        }
    }
    // (W2) A₂·z = Σ_i γ_i·T_i.
    {
        let a2_params = AjtaiParams {
            ring: ring.clone(),
            k: params.kappa,
            m: w,
            norm_bound: u32::try_from(beta2.max(1)).map_err(|e| format!("{e:?}"))?,
        };
        let a2_seed = derive_w2_seed(seed, params);
        let a2 = AjtaiPublicKey::from_seed(a2_params, a2_seed).map_err(|e| format!("{e:?}"))?;
        let az = a2.commit(&z).map(|c| c.rows).map_err(|e| format!("{e:?}"))?;
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
                return Err("W2: the inner-commitment binding (the short MSIS instance)".into());
            }
        }
    }
    // (W3R) Λ_t^{(i)}(z) = γ_i·U^t_i + Σ_{j≠i} γ_j·G^t_ij.
    {
        for t in 0..t_count {
            let func = &functionals[t];
            for (i, gi) in gammas.iter().enumerate() {
                let lhs = func.slice_eval(ring, &z, i, w);
                let mut rhs = ring.zero();
                if *gi != 0 {
                    let term = func_values[t * r2 + i].scale_i64(*gi);
                    rhs = rhs.add(&term).map_err(|e| format!("{e:?}"))?;
                }
                let mut idx = t * r2 * (r2 - 1) + i * (r2 - 1);
                for (j, gj) in gammas.iter().enumerate() {
                    if i != j {
                        if *gj != 0 {
                            let term = func_garbage[idx].scale_i64(*gj);
                            rhs = rhs.add(&term).map_err(|e| format!("{e:?}"))?;
                        }
                        idx += 1;
                    }
                }
                if lhs.coeffs() != rhs.coeffs() {
                    return Err(format!(
                        "W3R: functional {t} slice {i} superposition"
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingConfig {
        crate::codec::q32_ring().unwrap()
    }

    fn synth_response(ring: &RingConfig, n_bar: usize, seed: u64) -> Vec<RingElement> {
        let n = ring.n();
        let q = ring.modulus.q;
        (0..n_bar)
            .map(|i| {
                let coeffs: Vec<u32> = (0..n)
                    .map(|j| {
                        let mut x = seed
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .wrapping_add((i as u64 + 1) * 0x2545_F491_4F6C_DD1D)
                            .wrapping_add((j as u64 + 1) * 0x9E37_79B9_7F4A_7C15);
                        x ^= x >> 12;
                        x ^= x << 25;
                        x ^= x >> 27;
                        let b = (x % 256) as i64;
                        let v = if b > 127 { b - 256 } else { b };
                        (v.rem_euclid(q as i64)) as u32
                    })
                    .collect();
                RingElement::from_coeffs(ring, coeffs)
            })
            .collect()
    }

    fn key_blocks(
        ring: &RingConfig,
        k: usize,
        n_bar: usize,
        seed: [u8; 32],
    ) -> (AjtaiPublicKey, Vec<Vec<RingElement>>) {
        let params = AjtaiParams {
            ring: ring.clone(),
            k,
            m: n_bar,
            norm_bound: 255,
        };
        let key = AjtaiPublicKey::from_seed(params, seed).unwrap();
        let blocks = (0..n_bar)
            .map(|c| (0..k).map(|rr| key.entry(rr, c).cloned().unwrap()).collect())
            .collect();
        (key, blocks)
    }

    /// The honest roundtrip with three ring functionals at n̄ = 16.
    #[test]
    fn ring_fold_honest_roundtrip() {
        let ring = ring();
        let n_bar = 16usize;
        let k = 4usize;
        let v = synth_response(&ring, n_bar, 0x77);
        let (_, blocks) = key_blocks(&ring, k, n_bar, [9u8; 32]);
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        // Three functionals with full-range F_q weights.
        let funcs: Vec<RingFunctional> = (0..3)
            .map(|t| RingFunctional {
                weights: (0..n_bar)
                    .map(|c| {
                        let mut x = (t as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .wrapping_add(c as u64);
                        x ^= x >> 33;
                        x ^= x << 19;
                        x % q
                    })
                    .collect(),
            })
            .collect();
        let targets: Vec<Option<RingElement>> =
            funcs.iter().map(|f| Some(f.eval(&ring, &v))).collect();
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(n_bar, beta1, q, ring.n() as u64)
            .expect("sound profile");
        let mut tr = Transcript::new_default(b"test-ring-fold");
        let proof = prove_ring_fold(
            &ring, &v, &t, &funcs, &targets, &blocks, k, params, beta1, [9u8; 32], &mut tr,
        )
        .expect("honest prove");
        let mut tr2 = Transcript::new_default(b"test-ring-fold");
        verify_ring_fold(
            &ring, &t, &funcs, &targets, &blocks, k, beta1, [9u8; 32], &proof, &mut tr2,
        )
        .expect("honest verify");
    }

    /// Tampering the folded response fails (W1/W2).
    #[test]
    fn ring_fold_tampered_response_rejected() {
        let ring = ring();
        let n_bar = 16usize;
        let k = 4usize;
        let v = synth_response(&ring, n_bar, 0x78);
        let (_, blocks) = key_blocks(&ring, k, n_bar, [9u8; 32]);
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let funcs: Vec<RingFunctional> = vec![RingFunctional {
            weights: (0..n_bar).map(|c| ((c * 2654435761) as u64) % q).collect(),
        }];
        let targets: Vec<Option<RingElement>> =
            funcs.iter().map(|f| Some(f.eval(&ring, &v))).collect();
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(n_bar, beta1, q, ring.n() as u64)
            .expect("sound profile");
        let mut tr = Transcript::new_default(b"test-ring-fold");
        let mut proof = prove_ring_fold(
            &ring, &v, &t, &funcs, &targets, &blocks, k, params, beta1, [9u8; 32], &mut tr,
        )
        .expect("prove");
        let mut coeffs = decode_response(&proof.response).unwrap();
        coeffs[0] = coeffs[0].wrapping_add(1);
        proof.response = encode_response(&coeffs).unwrap();
        let mut tr2 = Transcript::new_default(b"test-ring-fold");
        assert!(
            verify_ring_fold(&ring, &t, &funcs, &targets, &blocks, k, beta1, [9u8; 32], &proof, &mut tr2)
                .is_err()
        );
    }

    /// A wrong PUBLIC functional target fails (W0R).
    #[test]
    fn ring_fold_wrong_functional_target_rejected() {
        let ring = ring();
        let n_bar = 16usize;
        let k = 4usize;
        let v = synth_response(&ring, n_bar, 0x79);
        let (_, blocks) = key_blocks(&ring, k, n_bar, [9u8; 32]);
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let funcs: Vec<RingFunctional> = vec![RingFunctional {
            weights: (0..n_bar).map(|c| ((c * 40503) as u64) % q).collect(),
        }];
        let targets: Vec<Option<RingElement>> =
            funcs.iter().map(|f| Some(f.eval(&ring, &v))).collect();
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(n_bar, beta1, q, ring.n() as u64)
            .expect("sound profile");
        let mut tr = Transcript::new_default(b"test-ring-fold");
        let proof = prove_ring_fold(
            &ring, &v, &t, &funcs, &targets, &blocks, k, params, beta1, [9u8; 32], &mut tr,
        )
        .expect("prove");
        // A wrong target: the W0R check rejects.
        let mut bad = targets.clone();
        bad[0] = Some(bad[0].clone().unwrap().add(&ring.one()).unwrap());
        let mut tr2 = Transcript::new_default(b"test-ring-fold");
        assert!(
            verify_ring_fold(&ring, &t, &funcs, &bad, &blocks, k, beta1, [9u8; 32], &proof, &mut tr2)
                .is_err()
        );
    }

    /// Tampering the functional garbage fails (W3R).
    #[test]
    fn ring_fold_tampered_functional_garbage_rejected() {
        let ring = ring();
        let n_bar = 16usize;
        let k = 4usize;
        let v = synth_response(&ring, n_bar, 0x80);
        let (_, blocks) = key_blocks(&ring, k, n_bar, [9u8; 32]);
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let funcs: Vec<RingFunctional> = vec![RingFunctional {
            weights: (0..n_bar).map(|c| ((c * 1103515245 + 12345) as u64) % q).collect(),
        }];
        let targets: Vec<Option<RingElement>> =
            funcs.iter().map(|f| Some(f.eval(&ring, &v))).collect();
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(n_bar, beta1, q, ring.n() as u64)
            .expect("sound profile");
        let mut tr = Transcript::new_default(b"test-ring-fold");
        let mut proof = prove_ring_fold(
            &ring, &v, &t, &funcs, &targets, &blocks, k, params, beta1, [9u8; 32], &mut tr,
        )
        .expect("prove");
        assert!(!proof.func_garbage.is_empty());
        proof.func_garbage[5] ^= 0x08;
        let mut tr2 = Transcript::new_default(b"test-ring-fold");
        assert!(
            verify_ring_fold(&ring, &t, &funcs, &targets, &blocks, k, beta1, [9u8; 32], &proof, &mut tr2)
                .is_err()
        );
    }
}
