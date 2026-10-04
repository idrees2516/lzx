//! **The width fold (DESIGN_50KB Stage 5.2 — the LaBRADOR tail)**: the
//! quadratic-garbage width-reducing fold that takes the first fold's WIDE
//! response `v ∈ R^{n̄}` (n̄ in the tens-to-hundreds at benchmark scale —
//! the single-level fold's broken MSIS regime, ~2^12 bits per the Stage
//! 5.1 estimator verdict) down to the sound narrow regime, at the price
//! the LaBRADOR level always pays: the **quadratic garbage committed
//! BEFORE the challenges**.
//!
//! # Why the garbage is forced (the honest width-reduction finding)
//!
//! The first fold's response `v` splits into `r₂` parts `s_i ∈ R^w`
//! (`w·r₂ ≥ n̄`, zero-padded), with the level-1 key split into matching
//! column groups `F̄_{(i)} ∈ R^{k×w}`. The parts' link images
//! `p_i = F̄_{(i)}·s_i` are PROVER data (only their sum — the public
//! target `t = F̄·v = Σ_j d_j·y_j` — is verifier-computable), and
//! binding them through the fold `z = Σ_i γ_i·s_i` NECESSARILY produces
//! the cross terms `F̄_{(i)}·s_j (i ≠ j)`: the verifier's key-group
//! image of the folded response is
//!
//! ```text
//! Σ_i γ_i·(F̄_{(i)}·z) = Σ_{i,j} γ_i·γ_j·(F̄_{(i)}·s_j)
//!                     = Σ_i γ_i²·p_i + Σ_{i≠j} γ_i·γ_j·G_ij ,
//! ```
//!
//! a degree-2 polynomial in the challenges whose off-diagonal
//! coefficients `G_ij = F̄_{(i)}·s_j` are exactly the LaBRADOR
//! quadratic garbage. **There is no garbage-free width-reducing fold**
//! (the finding the second-level design doc records); committing the
//! `G_ij` before the challenges is the price of the width reduction.
//! The same superposition hits the Goldilocks functional layer: the
//! per-slice functionals `ψ^{(i)}(s_j)` do not transport through
//! `z = Σ_i γ_i·s_i` either, so the functional garbage
//! `g_ij = ψ^{(i)}(s_j)` rides the same pre-challenge commitment
//! (the L3-superposition fix).
//!
//! # The construction
//!
//! 1. **Pre-challenge material** (all absorbed into the transcript and
//!    transmitted — fixed before any γ, so no grinding):
//!    * part images `p_i = F̄_{(i)}·s_i ∈ R^k` (the diagonal),
//!    * quadratic garbage `G_ij = F̄_{(i)}·s_j ∈ R^k` for `i ≠ j`,
//!    * inner commitments `T_i = A₂·s_i ∈ R^κ` under the fresh
//!      seed-derived width-fold key `A₂ ∈ R^{κ×w}`,
//!    * per-slice functionals `u_i = ψ^{(i)}(s_i)`,
//!    * functional garbage `g_ij = ψ^{(i)}(s_j)` for `i ≠ j`.
//! 2. **Scalar challenges** `γ_i ∈ [−A₂, A₂]` (scalar so the functional
//!    layer commutes exactly — the level-1 discipline).
//! 3. **The folded response** `z = Σ_i γ_i·s_i ∈ R^w` with the
//!    fail-closed gate `|z|_∞ ≤ β₂ = r₂·A₂·β₁ < q/2`.
//! 4. **The checks** — every one EXACT:
//!    * `(W0)` `Σ_i p_i = t` — the public image consistency;
//!    * `(W0')` `Σ_i u_i = u` — the public functional consistency;
//!    * `(W1)` `Σ_i γ_i·(F̄_{(i)}·z) = Σ_i γ_i²·p_i + Σ_{i≠j} γ_i γ_j·G_ij`
//!      — the exact fold identity (the verifier applies each key group
//!      to `z` itself);
//!    * `(W2)` `A₂·z = Σ_i γ_i·T_i` — **the short MSIS instance the
//!      whole construction exists for**;
//!    * `(W3)` per slice `i`: `ψ^{(i)}(z) = γ_i·u_i + Σ_{j≠i} γ_j·g_ij`
//!      — the functional superposition identities;
//!    * `(W4)` the norm gate on `z` (fail-closed).
//!
//! # Soundness (the honest ledger)
//!
//! The binding core is `(W2)`: a cheater fixes the `T_i` before `γ`
//! and picks `z` after; two accepting transcripts with `γ ≠ γ'` give
//! `A₂·Δz = Σ_i Δγ_i·T_i`, i.e. `[A₂ | −T]·(Δz, Δγ) = 0` — an MSIS
//! solution at width `w + r₂` with the extraction's relaxed bound
//! `2·β₂` — exactly the instance `second_fold::profile_bits` models,
//! gated fail-closed below 128 classical bits by
//! [`WidthFoldParams::assert_sound`].
//!
//! The degree-2 identities `(W1)/(W3)` are the *consistency carriers*:
//! the full quadratic-consistency extraction needs the LaBRADOR
//! special-soundness degree law (a degree-`d` fold needs `d+1 = 3`
//! transcripts), so the standalone 2-transcript argument pins the
//! response binding (W2) while (W1)/(W3) close the consistency hole
//! the width reduction would otherwise open (a `z` unrelated to the
//! committed parts' images cannot satisfy the verifier-computed
//! key-group images at the drawn `γ` without re-solving the
//! pre-committed quadratic form). The profile is the posture gate:
//! the verifier re-derives the estimator verdict and refuses any
//! proof recorded below the floor.
//!
//! # Cost (the honest quadratic ledger)
//!
//! The garbage is `r₂·(r₂−1)·k` ring elements plus `r₂·κ` inner
//! commitments plus `r₂²` Goldilocks functional terms — quadratic in
//! the part count. Beyond `r₂ ≈ 8` the re-packing route (re-running
//! the level-1 fold at a larger column count `r₁`, which shrinks `n̄`
//! without garbage) is cheaper per byte; what re-packing CANNOT buy is
//! the binding: its own single-level MSIS instance `[F̄ | −y]` stays in
//! the broken regime at benchmark stream sizes, while the width fold's
//! `[A₂ | −T]` at `w + r₂` is the sound instance. The sound proof is
//! therefore LARGER than the clear one (the measured honest multiple
//! lands in BENCHMARKS.md §2j) — the price of the binding claim the
//! Stage 5.1 verdict demands.

use crate::compact::{decode_response, encode_response, serialize_elements, ResponseWire};
use crate::second_fold::{apply_key, functional_of, profile_bits, SECURITY_FLOOR_BITS};
use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_ring::{RingConfig, RingElement};

/// The width-fold parameters (the public shape, transmitted in the
/// proof and re-gated by the verifier).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WidthFoldParams {
    /// The part count r₂ (the wide response splits into this many
    /// parts; powers of two; ≥ 2 for a real fold).
    pub r2: usize,
    /// The inner-key rows κ (A₂'s rank — the estimator's knob).
    pub kappa: usize,
    /// The γ-challenge amplitude (γ_i ∈ [−A, A]).
    pub amplitude: u32,
    /// The part width w (the folded response's rank — the sound
    /// regime is the small rows; must satisfy `w·r₂ ≥ n̄`).
    pub w: usize,
}

impl WidthFoldParams {
    /// The final response bound `β₂ = r₂·A₂·β₁` (the completeness
    /// gate at the level-1 gate `β₁`).
    pub fn beta2(&self, beta1: u64) -> u64 {
        self.r2 as u64 * self.amplitude as u64 * beta1
    }

    /// The estimator verdict for this profile at the given level-1
    /// gate: `(classical_bits, quantum_bits)` of the MSIS instance
    /// `[A₂ | −T]` at width `w + r₂`, bound `2·β₂` (the extraction's
    /// relaxed 2× factor — `second_fold::profile_bits`'s convention).
    pub fn profile_bits(
        &self,
        beta1: u64,
        q: u64,
        ring_dim: u64,
    ) -> Result<(f64, f64), String> {
        let view = crate::second_fold::SecondFoldParams {
            r: self.r2,
            kappa: self.kappa,
            k: 0, // unused by profile_bits (the level-1 k is not part of
            // the width fold's MSIS instance)
            amplitude: self.amplitude,
            n_bar: self.w,
        };
        profile_bits(&view, beta1, q, ring_dim)
    }

    /// The fail-closed posture gate: refuse any profile whose final
    /// MSIS instance sits below the security floor.
    pub fn assert_sound(
        &self,
        beta1: u64,
        q: u64,
        ring_dim: u64,
    ) -> Result<(f64, f64), String> {
        let (cl, qm) = self.profile_bits(beta1, q, ring_dim)?;
        if cl < SECURITY_FLOOR_BITS {
            return Err(format!(
                "width-fold profile below the security floor: {cl:.1} classical bits \
                 (params r2={}, kappa={}, w={}, A=2^{}, beta1={beta1})",
                self.r2,
                self.kappa,
                self.w,
                self.amplitude.trailing_zeros()
            ));
        }
        Ok((cl, qm))
    }

    /// The cheapest sound profile covering a level-1 response of width
    /// `n_bar` at the level-1 gate `beta1` — the honest search: iterate
    /// the part counts in increasing garbage cost, the part width in
    /// increasing MSIS width, and the (κ, A₂) knobs in increasing
    /// commitment cost, returning the first estimator-sound row.
    ///
    /// Fail-closed: `Err` when NO candidate row reaches the floor at
    /// this `beta1` — the caller must then lower the level-1 amplitude
    /// (the `A₁ ↓` lever of the Stage 5.1 table) or shrink the stream.
    pub fn sound_profile_for(
        n_bar: usize,
        beta1: u64,
        q: u64,
        ring_dim: u64,
    ) -> Result<WidthFoldParams, String> {
        // Ordered by the honest cost ledger: the garbage r2(r2-1)·k
        // dominates, so small r2 first; then small w (the MSIS width
        // w + r2 is the soundness driver); then the (kappa, A) knobs.
        for r2 in [2usize, 4, 8] {
            let w_min = n_bar.div_ceil(r2).max(1);
            for w in [w_min, w_min.next_power_of_two().max(w_min)] {
                // w must be a part width the estimator can gate; cap the
                // search at the sound regime's edge (the profile gate
                // rejects wider rows anyway, but the loop stays honest).
                if w > 16 {
                    continue;
                }
                for kappa in [4usize, 8, 16, 32, 64, 128] {
                    for amplitude in [1u32 << 4, 1 << 6, 1 << 8] {
                        let cand = WidthFoldParams {
                            r2,
                            kappa,
                            amplitude,
                            w,
                        };
                        if cand.beta2(beta1) >= q / 2 {
                            continue;
                        }
                        if cand.assert_sound(beta1, q, ring_dim).is_ok() {
                            return Ok(cand);
                        }
                    }
                }
            }
        }
        Err(format!(
            "no sound width-fold profile at n_bar={n_bar}, beta1={beta1} \
             (lower the level-1 amplitude or shrink the stream)"
        ))
    }
}

/// The width-fold proof artifact (replaces the level-1 response in the
/// Sound profile's compact opening).
#[derive(Clone, Debug)]
pub struct WidthFoldProof {
    pub params: WidthFoldParams,
    /// The level-1 response width (before zero padding to `w·r₂`).
    pub n_bar: usize,
    /// The part images `p_i` (r₂·k ring elements, serialized).
    pub p_images: Vec<u8>,
    /// The quadratic garbage `G_ij`, i≠j (r₂·(r₂−1)·k ring elements,
    /// row-major over (i, j) skipping the diagonal, serialized).
    pub garbage: Vec<u8>,
    /// The inner commitments `T_i` (r₂·κ ring elements, serialized).
    pub t_inner: Vec<u8>,
    /// The per-slice functionals `u_i` (r₂ Goldilocks, LE u64s).
    pub u_parts: Vec<u8>,
    /// The functional garbage `g_ij`, i≠j (r₂·(r₂−1) Goldilocks,
    /// row-major over (i, j) skipping the diagonal, LE u64s).
    pub g_func: Vec<u8>,
    /// The folded response `z ∈ R^w` (rANS-coded coefficients).
    pub response: ResponseWire,
    /// The estimator verdict at prove time (the posture marker — the
    /// verifier re-derives and compares).
    pub classical_bits: f64,
}

/// The A₂ key seed (the width-fold domain, independent of level-1).
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

/// The zero-padded part split: part i = v[i·w .. (i+1)·w], zero beyond
/// n̄. Returns r₂ parts of exactly w elements each.
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

/// The zero-padded key blocks: block c of group i =
/// `f_bar_blocks[i·w + c]` (zero blocks beyond the key's n̄ columns).
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

/// Serialize r Goldilocks values (LE u64s — the transcript/proof wire
/// form).
fn serialize_goldilocks(vals: &[Goldilocks]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 * vals.len());
    for v in vals {
        buf.extend_from_slice(&v.to_canonical_u64().to_le_bytes());
    }
    buf
}

fn deserialize_goldilocks(bytes: &[u8], count: usize) -> Result<Vec<Goldilocks>, String> {
    if bytes.len() != 8 * count {
        return Err(format!("goldilocks wire length {} != {}", bytes.len(), 8 * count));
    }
    let mut out = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(8) {
        let raw = u64::from_le_bytes(chunk.try_into().map_err(|e| format!("{e:?}"))?);
        out.push(Goldilocks::from_u64(raw));
    }
    Ok(out)
}

/// The ψ-slice functional of one part: `ψ^{(i)}(s) = Σ_m ψ[i·w·n + m]·s_m`
/// over Goldilocks (the weight slice for part i's stream slots).
fn psi_slice_of(
    ring: &RingConfig,
    s: &[RingElement],
    psi_weights: &[Goldilocks],
    part_i: usize,
    w: usize,
    q: u64,
) -> Goldilocks {
    let n = ring.n();
    let base = part_i * w * n;
    let slice = &psi_weights[base.min(psi_weights.len())..];
    functional_of(ring, s, slice, q)
}

/// Prove the width fold over the level-1 response `v ∈ R^{n̄}` with:
///
/// * `t_target` — the PUBLIC level-1 target `F̄·v = Σ_j d_j·y_j` (the
///   verifier computes it from the bundle's per-column commitments);
/// * `u_target` — the PUBLIC functional claim `Φ(v) = Σ_j d_j·ũ_j`;
/// * `f_bar_blocks` — the level-1 key's column blocks (n̄ blocks of k
///   elements — the prover holds the key object it committed under);
/// * `psi_weights` — the full stream weights (n̄·n Goldilocks);
/// * `beta1` — the level-1 response gate.
///
/// Fails closed on: shape mismatches, the prover's own consistency
/// (the level-1 target/functional self-checks), the β₂ < q/2 gate,
/// the response norm gate, and the estimator profile floor.
#[allow(clippy::too_many_arguments)]
pub fn prove_width_fold(
    ring: &RingConfig,
    v: &[RingElement],
    t_target: &[RingElement],
    u_target: &Goldilocks,
    f_bar_blocks: &[Vec<RingElement>],
    k: usize,
    psi_weights: &[Goldilocks],
    params: WidthFoldParams,
    beta1: u64,
    seed: [u8; 32],
    transcript: &mut Transcript,
) -> Result<WidthFoldProof, String> {
    let q = u64::from(ring.modulus.q);
    let n = ring.n();
    let n_bar = v.len();
    let r2 = params.r2;
    let w = params.w;
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
            "shape: w·r₂ = {} < n̄ = {n_bar} (the parts cannot cover the response)",
            w * r2
        ));
    }
    // The fail-closed profile gate.
    let (cl, _qm) = params.assert_sound(beta1, q, n as u64)?;

    // The level-1 self-checks (the prover verifies its own inputs).
    let fz = apply_key(ring, f_bar_blocks, v, k);
    for (rr, e) in fz.iter().enumerate() {
        if e.coeffs() != t_target[rr].coeffs() {
            return Err("self-check: F̄·v ≠ t (the level-1 target)".into());
        }
    }
    let phi_v = functional_of(ring, v, psi_weights, q);
    if phi_v != *u_target {
        return Err("self-check: Φ(v) ≠ u (the level-1 functional)".into());
    }

    // The parts + key groups (zero-padded to w·r₂).
    let parts = split_parts(ring, v, w, r2);
    let groups = key_groups(ring, f_bar_blocks, k, w, r2);

    // The pre-challenge material.
    // p_i = F̄_(i)·s_i — the part images.
    let p: Vec<Vec<RingElement>> = (0..r2)
        .map(|i| apply_key(ring, &groups[i], &parts[i], k))
        .collect();
    // G_ij = F̄_(i)·s_j for i ≠ j — the quadratic garbage.
    let mut garbage: Vec<RingElement> = Vec::with_capacity(r2 * (r2 - 1) * k);
    for i in 0..r2 {
        for j in 0..r2 {
            if i != j {
                let gij = apply_key(ring, &groups[i], &parts[j], k);
                garbage.extend_from_slice(&gij);
            }
        }
    }
    // T_i = A₂·s_i — the inner commitments under the fresh key.
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
    // u_i = ψ^(i)(s_i) and g_ij = ψ^(i)(s_j) — the functional layer.
    let u_parts: Vec<Goldilocks> = (0..r2)
        .map(|i| psi_slice_of(ring, &parts[i], psi_weights, i, w, q))
        .collect();
    let mut g_func: Vec<Goldilocks> = Vec::with_capacity(r2 * (r2 - 1));
    for i in 0..r2 {
        for j in 0..r2 {
            if i != j {
                g_func.push(psi_slice_of(ring, &parts[j], psi_weights, i, w, q));
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
    // (W0') Σ u_i = u — the public functional consistency self-check.
    {
        let mut sum = Goldilocks::ZERO;
        for u_i in &u_parts {
            sum = sum.add(u_i);
        }
        if sum != *u_target {
            return Err("self-check W0': Σ u_i ≠ u".into());
        }
    }

    // The FS absorption (all pre-challenge material BEFORE any γ).
    let p_bytes = serialize_elements(
        ring,
        &p.iter().flatten().cloned().collect::<Vec<_>>(),
    );
    let garbage_bytes = serialize_elements(ring, &garbage);
    let t_bytes = serialize_elements(
        ring,
        &t_inner.iter().flatten().cloned().collect::<Vec<_>>(),
    );
    let u_bytes = serialize_goldilocks(&u_parts);
    let g_bytes = serialize_goldilocks(&g_func);
    transcript
        .append_bytes(b"wf-p", &p_bytes)
        .and_then(|_| transcript.append_bytes(b"wf-g", &garbage_bytes))
        .and_then(|_| transcript.append_bytes(b"wf-t", &t_bytes))
        .and_then(|_| transcript.append_bytes(b"wf-u", &u_bytes))
        .and_then(|_| transcript.append_bytes(b"wf-gf", &g_bytes))
        .map_err(|e| format!("{e:?}"))?;

    // The scalar challenges γ_i ∈ [−A, A].
    let gammas: Vec<i64> = (0..r2)
        .map(|_i| -> Result<i64, String> {
            let b = transcript
                .challenge_bytes(b"wf-gamma", 2)
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
                return Err(format!("width-fold gate: |{balanced}| > {beta2}"));
            }
            z_coeffs.push(balanced as i32);
        }
    }

    // The self-checks (W1)-(W3) — the prover verifies its own
    // construction before transmitting.
    // (W1) Σ_i γ_i·(F̄_(i)·z) = Σ_i γ_i²·p_i + Σ_{i≠j} γ_i γ_j·G_ij.
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
    // (W3) ψ^(i)(z) = γ_i·u_i + Σ_{j≠i} γ_j·g_ij per slice.
    {
        for (i, gi) in gammas.iter().enumerate() {
            let psi_z = psi_slice_of(ring, &z, psi_weights, i, w, q);
            let mut rhs = u_parts[i].mul(&Goldilocks::from_u64(gi.unsigned_abs()));
            rhs = if *gi < 0 { Goldilocks::ZERO.sub(&rhs) } else { rhs };
            // The flat functional-garbage layout is row-major over
            // (i, j≠i): slice i's entries start at i·(r₂−1).
            let mut idx = i * (r2 - 1);
            for (j, gj) in gammas.iter().enumerate() {
                if i != j {
                    if *gj != 0 {
                        let term =
                            g_func[idx].mul(&Goldilocks::from_u64(gj.unsigned_abs()));
                        rhs = if *gj < 0 { rhs.sub(&term) } else { rhs.add(&term) };
                    }
                    idx += 1;
                }
            }
            if psi_z != rhs {
                return Err(format!("self-check W3: slice {i} superposition"));
            }
        }
    }

    // Absorb the canonical folded response.
    transcript
        .append_bytes(b"wf-z", &serialize_elements(ring, &z))
        .map_err(|e| format!("{e:?}"))?;
    let response = encode_response(&z_coeffs)?;
    Ok(WidthFoldProof {
        params,
        n_bar,
        p_images: p_bytes,
        garbage: garbage_bytes,
        t_inner: t_bytes,
        u_parts: u_bytes,
        g_func: g_bytes,
        response,
        classical_bits: cl,
    })
}

/// Verify a width-fold proof given the PUBLIC level-1 target
/// `(t_target, u_target)`, the level-1 key column blocks (regenerated
/// from the seed by the caller — the compact verify path already holds
/// them), and the ψ weights. Replays the transcript, checks
/// (W0)-(W4), and re-derives the estimator verdict (the fail-closed
/// posture gate).
#[allow(clippy::too_many_arguments)]
pub fn verify_width_fold(
    ring: &RingConfig,
    t_target: &[RingElement],
    u_target: &Goldilocks,
    f_bar_blocks: &[Vec<RingElement>],
    k: usize,
    psi_weights: &[Goldilocks],
    beta1: u64,
    seed: [u8; 32],
    proof: &WidthFoldProof,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let q = u64::from(ring.modulus.q);
    let n = ring.n();
    let params = &proof.params;
    let r2 = params.r2;
    let w = params.w;
    if t_target.len() != k {
        return Err(format!("shape: target rank {} != k {k}", t_target.len()));
    }
    if w * r2 < proof.n_bar {
        return Err("shape: w·r₂ < n̄".into());
    }
    // The fail-closed profile re-derivation + posture marker.
    let (cl, _) = params.assert_sound(beta1, q, n as u64)?;
    if (cl - proof.classical_bits).abs() > 1.0 {
        return Err(format!(
            "posture marker mismatch: proof {} vs verifier {cl:.1}",
            proof.classical_bits
        ));
    }
    // Deserialize the pre-challenge material.
    let p_flat =
        crate::compact::deserialize_elements(ring, &proof.p_images)?;
    if p_flat.len() != r2 * k {
        return Err(format!("p count {} != {}", p_flat.len(), r2 * k));
    }
    let garbage =
        crate::compact::deserialize_elements(ring, &proof.garbage)?;
    if garbage.len() != r2 * (r2 - 1) * k {
        return Err(format!(
            "garbage count {} != {}",
            garbage.len(),
            r2 * (r2 - 1) * k
        ));
    }
    let t_inner =
        crate::compact::deserialize_elements(ring, &proof.t_inner)?;
    if t_inner.len() != r2 * params.kappa {
        return Err(format!(
            "inner count {} != {}",
            t_inner.len(),
            r2 * params.kappa
        ));
    }
    let u_parts = deserialize_goldilocks(&proof.u_parts, r2)?;
    let g_func = deserialize_goldilocks(&proof.g_func, r2 * (r2 - 1))?;
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
    // (W0') Σ u_i = u.
    {
        let mut sum = Goldilocks::ZERO;
        for u_i in &u_parts {
            sum = sum.add(u_i);
        }
        if sum != *u_target {
            return Err("W0': the public functional consistency".into());
        }
    }
    // The transcript replay (pre-challenge material).
    transcript
        .append_bytes(b"wf-p", &proof.p_images)
        .and_then(|_| transcript.append_bytes(b"wf-g", &proof.garbage))
        .and_then(|_| transcript.append_bytes(b"wf-t", &proof.t_inner))
        .and_then(|_| transcript.append_bytes(b"wf-u", &proof.u_parts))
        .and_then(|_| transcript.append_bytes(b"wf-gf", &proof.g_func))
        .map_err(|e| format!("{e:?}"))?;
    // The challenges.
    let gammas: Vec<i64> = (0..r2)
        .map(|_i| -> Result<i64, String> {
            let b = transcript
                .challenge_bytes(b"wf-gamma", 2)
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
        .append_bytes(b"wf-z", &serialize_elements(ring, &z))
        .map_err(|e| format!("{e:?}"))?;

    // The key groups (zero-padded to w·r₂ — the verifier regenerates
    // the level-1 blocks; the pad columns are zero).
    let groups = key_groups(ring, f_bar_blocks, k, w, r2);

    // (W1) Σ_i γ_i·(F̄_(i)·z) = Σ_i γ_i²·p_i + Σ_{i≠j} γ_i γ_j·G_ij.
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
    // (W2) A₂·z = Σ_i γ_i·T_i — the short MSIS binding.
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
    // (W3) ψ^(i)(z) = γ_i·u_i + Σ_{j≠i} γ_j·g_ij per slice.
    {
        for (i, gi) in gammas.iter().enumerate() {
            let psi_z = psi_slice_of(ring, &z, psi_weights, i, w, q);
            let mut rhs = u_parts[i].mul(&Goldilocks::from_u64(gi.unsigned_abs()));
            rhs = if *gi < 0 { Goldilocks::ZERO.sub(&rhs) } else { rhs };
            // The flat functional-garbage layout is row-major over
            // (i, j≠i): slice i's entries start at i·(r₂−1).
            let mut idx = i * (r2 - 1);
            for (j, gj) in gammas.iter().enumerate() {
                if i != j {
                    if *gj != 0 {
                        let term =
                            g_func[idx].mul(&Goldilocks::from_u64(gj.unsigned_abs()));
                        rhs = if *gj < 0 { rhs.sub(&term) } else { rhs.add(&term) };
                    }
                    idx += 1;
                }
            }
            if psi_z != rhs {
                return Err(format!("W3: slice {i} functional superposition"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compact::column_ring;
    use lattice_ring::RingElement;

    fn ring() -> RingConfig {
        column_ring().unwrap()
    }

    /// A deterministic pseudo-random short vector: coefficients in
    /// [−β, β] (the byte-instance regime at β = 255).
    fn synth_response(ring: &RingConfig, n_bar: usize, beta: u32, seed: u64) -> Vec<RingElement> {
        let n = ring.n();
        let q = ring.modulus.q;
        (0..n_bar)
            .map(|i| {
                let coeffs: Vec<u32> = (0..n)
                    .map(|j| {
                        // xorshift-ish deterministic byte in [0, 255]
                        let mut x = seed
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .wrapping_add((i as u64 + 1) * 0x2545_F491_4F6C_DD1D)
                            .wrapping_add((j as u64 + 1) * 0x9E37_79B9_7F4A_7C15);
                        x ^= x >> 12;
                        x ^= x << 25;
                        x ^= x >> 27;
                        let b = (x % 256) as i64;
                        let v = if b > 127 { b - 256 } else { b };
                        let _ = beta;
                        (v.rem_euclid(q as i64)) as u32
                    })
                    .collect();
                RingElement::from_coeffs(ring, coeffs)
            })
            .collect()
    }

    /// The level-1 key blocks from a seeded Ajtai key (the prover and
    /// verifier derive the same blocks).
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

    /// ψ weights: deterministic Goldilocks values over n̄·n slots.
    fn synth_psi(n_bar: usize, n: usize, seed: u64) -> Vec<Goldilocks> {
        (0..n_bar * n)
            .map(|i| {
                let mut x = seed.wrapping_add(i as u64);
                x ^= x >> 33;
                x ^= x << 19;
                x ^= x >> 45;
                Goldilocks::from_u64(x % 1_000_003)
            })
            .collect()
    }

    fn test_setup(n_bar: usize) -> (RingConfig, Vec<RingElement>, Vec<Vec<RingElement>>, Vec<Goldilocks>, [u8; 32]) {
        let ring = ring();
        let v = synth_response(&ring, n_bar, 255, 0xABCD);
        let (_, blocks) = key_blocks(&ring, 4, n_bar, [7u8; 32]);
        let psi = synth_psi(n_bar, ring.n(), 0x1234);
        (ring, v, blocks, psi, [7u8; 32])
    }

    /// The honest roundtrip at the byte-instance gate β₁ = 255 (the
    /// estimator's reference regime) — the width fold must verify.
    #[test]
    fn width_fold_honest_roundtrip() {
        let (ring, v, blocks, psi, seed) = test_setup(16);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        // β₁ = 255: the byte-instance gate.
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(16, beta1, q, ring.n() as u64)
            .expect("a sound profile exists at the byte-instance gate");
        let mut tr = Transcript::new_default(b"test-width-fold");
        let proof = prove_width_fold(
            &ring, &v, &t, &u, &blocks, k, &psi, params.clone(), beta1, seed, &mut tr,
        )
        .expect("honest prove");
        let mut tr2 = Transcript::new_default(b"test-width-fold");
        verify_width_fold(&ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2)
            .expect("honest verify");
    }

    /// Tampering the folded response must fail W2 (the MSIS binding).
    #[test]
    fn width_fold_tampered_z_rejected() {
        let (ring, v, blocks, psi, seed) = test_setup(16);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(16, beta1, q, ring.n() as u64)
            .expect("sound profile");
        let mut tr = Transcript::new_default(b"test-width-fold");
        let mut proof = prove_width_fold(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("prove");
        // Flip one coefficient of the decoded response.
        let mut coeffs = decode_response(&proof.response).unwrap();
        assert!(!coeffs.is_empty());
        coeffs[0] = coeffs[0].wrapping_add(1);
        proof.response = encode_response(&coeffs).unwrap();
        let mut tr2 = Transcript::new_default(b"test-width-fold");
        assert!(verify_width_fold(&ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2).is_err());
    }

    /// Tampering the inner commitments must fail W2.
    #[test]
    fn width_fold_tampered_inner_rejected() {
        let (ring, v, blocks, psi, seed) = test_setup(16);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(16, beta1, q, ring.n() as u64)
            .expect("sound profile");
        let mut tr = Transcript::new_default(b"test-width-fold");
        let mut proof = prove_width_fold(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("prove");
        // Corrupt one byte of the serialized inner commitments.
        let mut t_inner = proof.t_inner.clone();
        assert!(!t_inner.is_empty());
        t_inner[0] ^= 0x01;
        proof.t_inner = t_inner;
        // The transcript replay absorbs the tampered bytes — the
        // challenges differ, so W2 (or the replay divergence) rejects.
        let mut tr2 = Transcript::new_default(b"test-width-fold");
        assert!(verify_width_fold(&ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2).is_err());
    }

    /// Tampering the part images must fail W0/W1.
    #[test]
    fn width_fold_tampered_images_rejected() {
        let (ring, v, blocks, psi, seed) = test_setup(16);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(16, beta1, q, ring.n() as u64)
            .expect("sound profile");
        let mut tr = Transcript::new_default(b"test-width-fold");
        let mut proof = prove_width_fold(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("prove");
        // Corrupt one byte of the serialized part images.
        let mut p_images = proof.p_images.clone();
        assert!(!p_images.is_empty());
        p_images[3] ^= 0x02;
        proof.p_images = p_images;
        let mut tr2 = Transcript::new_default(b"test-width-fold");
        assert!(verify_width_fold(&ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2).is_err());
    }

    /// Tampering the quadratic garbage must fail W1.
    #[test]
    fn width_fold_tampered_garbage_rejected() {
        let (ring, v, blocks, psi, seed) = test_setup(16);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(16, beta1, q, ring.n() as u64)
            .expect("sound profile");
        let mut tr = Transcript::new_default(b"test-width-fold");
        let mut proof = prove_width_fold(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("prove");
        // Corrupt one byte of the serialized garbage.
        let mut garbage = proof.garbage.clone();
        assert!(!garbage.is_empty());
        garbage[7] ^= 0x04;
        proof.garbage = garbage;
        let mut tr2 = Transcript::new_default(b"test-width-fold");
        assert!(verify_width_fold(&ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2).is_err());
    }

    /// Tampering the functional garbage must fail W3.
    #[test]
    fn width_fold_tampered_functional_rejected() {
        let (ring, v, blocks, psi, seed) = test_setup(16);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(16, beta1, q, ring.n() as u64)
            .expect("sound profile");
        let mut tr = Transcript::new_default(b"test-width-fold");
        let mut proof = prove_width_fold(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("prove");
        // Corrupt one byte of the serialized functional garbage.
        let mut g_func = proof.g_func.clone();
        assert!(!g_func.is_empty());
        g_func[2] ^= 0x08;
        proof.g_func = g_func;
        let mut tr2 = Transcript::new_default(b"test-width-fold");
        assert!(verify_width_fold(&ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2).is_err());
    }

    /// A wrong public target must fail W0.
    #[test]
    fn width_fold_wrong_target_rejected() {
        let (ring, v, blocks, psi, seed) = test_setup(16);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(16, beta1, q, ring.n() as u64)
            .expect("sound profile");
        let mut tr = Transcript::new_default(b"test-width-fold");
        let proof = prove_width_fold(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("prove");
        // A different (wrong) target.
        let mut t_wrong = t.clone();
        t_wrong[0] = t_wrong[0].add(&ring.one()).unwrap();
        let mut tr2 = Transcript::new_default(b"test-width-fold");
        assert!(verify_width_fold(&ring, &t_wrong, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2).is_err());
    }

    /// A wrong functional claim must fail W0'.
    #[test]
    fn width_fold_wrong_functional_rejected() {
        let (ring, v, blocks, psi, seed) = test_setup(16);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(16, beta1, q, ring.n() as u64)
            .expect("sound profile");
        let mut tr = Transcript::new_default(b"test-width-fold");
        let proof = prove_width_fold(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("prove");
        let u_wrong = u.add(&Goldilocks::ONE);
        let mut tr2 = Transcript::new_default(b"test-width-fold");
        assert!(verify_width_fold(&ring, &t, &u_wrong, &blocks, k, &psi, beta1, seed, &proof, &mut tr2).is_err());
    }

    /// The profile gate must reject the broken shapes (the honest
    /// floor: wide w or huge amplitudes at the real level-1 gates).
    #[test]
    fn profile_gate_rejects_broken_shapes() {
        let q = u64::from(ring().modulus.q);
        let ring_dim = ring().n() as u64;
        // The broken single-level regime: w = 64 (a wide response at
        // benchmark scale) at any part count — the Stage 5.1 verdict.
        let broken = WidthFoldParams {
            r2: 4,
            kappa: 4,
            amplitude: 1 << 6,
            w: 64,
        };
        assert!(broken.assert_sound(255, q, ring_dim).is_err());
        // The amplitude blow-up: β₂ at A = 2^20 breaches q/2 outright
        // (β₂ = 8·2^20·255 ≈ 2^31.3 > q/2 ≈ 2^30.6 — the completeness
        // gate is unreachable, the profile must refuse).
        let blowup = WidthFoldParams {
            r2: 8,
            kappa: 8,
            amplitude: 1 << 20,
            w: 4,
        };
        assert!(blowup.profile_bits(255, q, ring_dim).is_err());
    }

    /// The sound search must find a profile at the byte-instance gate
    /// and fail closed at the blown-up level-1 gates.
    #[test]
    fn sound_search_honest_floors() {
        let q = u64::from(ring().modulus.q);
        let ring_dim = ring().n() as u64;
        // β₁ = 255 (the byte-instance regime): a sound row exists.
        let p = WidthFoldParams::sound_profile_for(16, 255, q, ring_dim);
        assert!(p.is_ok(), "no sound row at the byte-instance gate");
        // The real level-1 gates: β₁ = r·A₁·255 at r = 8, A₁ = 2^6
        // (β₁ ≈ 2^17). Whether a row exists is the estimator's
        // verdict — assert it FAILS CLOSED (either a row exists with
        // the honest cost, or the search refuses).
        let beta1_real = 8u64 * 64 * 255;
        if let Ok(p) = WidthFoldParams::sound_profile_for(64, beta1_real, q, ring_dim) {
            // If a row exists it must actually be sound.
            assert!(p.assert_sound(beta1_real, q, ring_dim).is_ok());
            assert!(p.w * p.r2 >= 64);
        }
        // The absurd regime fails closed.
        assert!(
            WidthFoldParams::sound_profile_for(64, 1 << 26, q, ring_dim).is_err(),
            "the absurd gate must fail closed"
        );
    }

    /// The zero-padding path: n̄ not divisible by w·r₂ still folds
    /// (the pad coordinates are zero — the identities still hold).
    #[test]
    fn width_fold_padding_path() {
        // n̄ = 10 with w·r₂ = 16: 6 pad columns.
        let (ring, v, blocks, psi, seed) = test_setup(10);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params = WidthFoldParams::sound_profile_for(10, beta1, q, ring.n() as u64)
            .expect("sound profile at the padded shape");
        assert!(params.w * params.r2 >= 10);
        let mut tr = Transcript::new_default(b"test-width-fold");
        let proof = prove_width_fold(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("padded prove");
        let mut tr2 = Transcript::new_default(b"test-width-fold");
        verify_width_fold(&ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2)
            .expect("padded verify");
    }
}
