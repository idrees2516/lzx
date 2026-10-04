//! **The recursive width-collapse staging** — the log-stages of the
//! sound width-fold rows that take the Sound profile's coverage from
//! `n̄ ≤ 16` (the single-stage boundary) to the benchmark streams.
//!
//! # Why staging is needed (the honest boundary this module moves)
//!
//! The single width fold's sound coverage is bounded by TWO ceilings:
//!
//! 1. the **re-packing ceiling** — the level-1 column count `r₁` is
//!    capped (the packing bound `r₁ ≤ min(128, factor length)` and the
//!    `β₁ = r₁·A₁·255 ≤ 2^20` gate), so streams beyond ~128 KB cannot
//!    re-pack their way down to `n̄ ≤ 16`;
//! 2. the **single-stage MSIS ceiling** — one fold from a wide `n̄`
//!    needs `w·r₂ ≥ n̄` with the instance `[A₂|−T]` at width
//!    `w + r₂`; the estimator's sound rows live at `w + r₂ ≤ ~20`, so
//!    a single stage cannot both cover a wide response and stay sound.
//!
//! The staging breaks ceiling 2: each stage halves (or quarters) the
//! width with its OWN estimator-sound instance, and the per-stage
//! OUTPUT gate grows geometrically (`β_{ℓ+1} = r₂·A₂·β_ℓ`) — the
//! completeness budget `β_final < q/2` is the remaining ceiling (the
//! honest Q_32 boundary: the Modulus-50 class is the follow-up).
//!
//! # The stage threading (what chains the folds together)
//!
//! Stage `ℓ` consumes the previous stage's OUTPUT as its input
//! response, with PUBLIC claims both sides derive identically:
//!
//! * the **image target** `t_{ℓ+1} = Σ_i γ^{(ℓ)}_i·T^{(ℓ)}_i` — the
//!   stage-ℓ inner commitments combined at the stage-ℓ challenges
//!   (exactly stage ℓ's `(W2)` RHS: the verifier computes it from the
//!   transmitted `T`'s and the transcript-derived `γ`'s);
//! * the **functional claim** `u_{ℓ+1} = Σ_i (γ_i)²·u_i +
//!   Σ_{i≠j} γ_iγ_j·g_ij` — the γ-weighted superposition the stage-ℓ
//!   `(W3)` identities certify (the folded functional's value);
//! * the **folded functional weights** `Ψ_{ℓ+1}[m] =
//!   Σ_i γ_i·Ψ_ℓ[i·w_ℓ·n + m]` — the γ-weighted column-fold of the
//!   weight matrix: the functional the NEXT stage's `(W3)` proves
//!   over `z_ℓ` (the γ-weighted sum of the slice functionals);
//! * the **key blocks** — the stage-ℓ inner key `A₂^{(ℓ)}`'s columns
//!   (both sides regenerate it from the salted seed and the stage
//!   params); the next stage's `k` is the stage's `κ`.
//!
//! Every stage is verified by the UNMODIFIED `verify_width_fold`
//! ((W0)–(W4) + the fail-closed estimator posture gate); the chain
//! adds only the public-claim derivation between stages.
//!
//! # The sound posture (the honest ledger)
//!
//! * **Per-stage fail-closed gating**: every stage's `[A₂^{(ℓ)} |
//!   −T^{(ℓ)}]` instance is re-derived by the verifier and must clear
//!   `SECURITY_FLOOR_BITS + the chain's grinding allowance` — the
//!   estimator verdict, never an assertion.
//! * **The AND-composition**: the chain's binding is the conjunction
//!   of the per-stage bindings (an accepting fork at ANY stage yields
//!   a short kernel on THAT stage's instance — all gated).
//! * **The honest residual** (documented, not claimed): the full
//!   multi-stage extraction — the LaBRADOR special-soundness degree
//!   law composed ACROSS stages, unwinding the chain to the level-1
//!   response — is the open analysis; the conservative posture here is
//!   the per-stage estimator floor + the (W0) target threading (each
//!   stage's public target is the previous stage's committed `T`
//!   combination, so no stage can detach from its predecessor).
//! * **The grinding allowance**: the per-stage challenge space
//!   `|C_ℓ| = (2·A₂+1)^{r₂}` admits replay grinding of
//!   `log₂|C_ℓ|` bits; the chain charges the TOTAL against every
//!   stage's floor (the 32-bit allowance `CHAIN_GRINDING_BITS`).

use crate::fold::{prove_width_fold_ex, verify_width_fold_ex, WidthFoldParams, WidthFoldProof};
#[allow(unused_imports)]
use crate::helpers::functional_of;
use crate::SECURITY_FLOOR_BITS;
use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_ring::ring::{RingConfig, RingElement};

/// The chain-level replay-grinding allowance (bits) charged on top of
/// the per-stage estimator floor: `Σ_ℓ log₂(2·A₂^{(ℓ)}+1)^{r₂^{(ℓ)}}`
/// must stay under this for the schedule to ship.
pub const CHAIN_GRINDING_BITS: f64 = 32.0;

/// The staged schedule: `stages[0]` folds the level-1 response,
/// `stages[ℓ]` folds stage `ℓ−1`'s output. The LAST stage must land
/// the final width in the single-stage sound regime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WidthChainParams {
    pub stages: Vec<WidthFoldParams>,
}

impl WidthChainParams {
    /// The final response width (the last stage's `w`).
    pub fn final_width(&self) -> usize {
        self.stages.last().map(|s| s.w).unwrap_or(0)
    }

    /// The per-stage input gates `β_ℓ` (β₁ given; `β_{ℓ+1} =
    /// r₂^{(ℓ)}·A₂^{(ℓ)}·β_ℓ`).
    pub fn stage_gates(&self, beta1: u64) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.stages.len());
        let mut beta = beta1;
        for s in &self.stages {
            out.push(beta);
            beta = s.beta2(beta);
        }
        out
    }

    /// The total replay-grinding loss (bits) of the schedule.
    pub fn grinding_bits(&self) -> f64 {
        self.stages
            .iter()
            .map(|s| {
                let space = (2.0 * s.amplitude as f64 + 1.0).powi(s.r2 as i32);
                space.log2()
            })
            .sum()
    }

    /// The honest cost model (transmitted ring elements): per stage
    /// `r₂·(r₂−1)·k_in` garbage + `r₂·k_in` images + `r₂·κ` inner
    /// commitments, where `k_in` is the stage's INPUT key rank (the
    /// level-1 `k` for stage 0, the previous stage's `κ` after).
    pub fn cost_elements(&self, k_level1: usize) -> usize {
        let mut k_in = k_level1;
        let mut total = 0usize;
        for s in &self.stages {
            total += s.r2 * (s.r2 - 1) * k_in + 2 * s.r2 * k_in + s.r2 * s.kappa + s.w;
            k_in = s.kappa;
        }
        total
    }

    /// The per-stage estimator verdicts (fail-closed below
    /// `SECURITY_FLOOR_BITS + CHAIN_GRINDING_BITS`).
    pub fn assert_sound_chain(
        &self,
        beta1: u64,
        q: u64,
        ring_dim: u64,
    ) -> Result<Vec<(f64, f64)>, String> {
        if self.stages.is_empty() {
            return Err("empty chain".into());
        }
        let floor = SECURITY_FLOOR_BITS + CHAIN_GRINDING_BITS;
        let gates = self.stage_gates(beta1);
        let mut verdicts = Vec::with_capacity(self.stages.len());
        for (s, &beta) in self.stages.iter().zip(gates.iter()) {
            let (cl, qm) = s
                .profile_bits(beta, q, ring_dim)
                .map_err(|e| format!("stage gate: {e}"))?;
            if cl < floor {
                return Err(format!(
                    "chain stage below the floor+grinding allowance: {cl:.1} < {floor:.1} \
                     (r2={}, w={}, kappa={}, A=2^{}, beta=2^{:.0})",
                    s.r2,
                    s.w,
                    s.kappa,
                    s.amplitude.trailing_zeros(),
                    (beta as f64).log2()
                ));
            }
            verdicts.push((cl, qm));
        }
        if self.grinding_bits() > CHAIN_GRINDING_BITS {
            return Err(format!(
                "chain grinding {0:.1} bits exceeds the {CHAIN_GRINDING_BITS}-bit allowance",
                self.grinding_bits()
            ));
        }
        // The multi-stage extraction ledger (the degree-law unwind
        // posture, laws E1–E5): every sound chain is now ALSO
        // extraction-sound at prove AND verify time — the honest
        // residual ("the full multi-stage extraction is the open
        // analysis") is closed as an enforced artifact, not prose.
        let ledger = crate::extraction::chain_extraction_ledger(self, beta1, q, ring_dim)
            .map_err(|e| format!("extraction ledger: {e}"))?;
        ledger.assert_extraction_sound(q)?;
        Ok(verdicts)
    }

    /// The staged sound schedule for a level-1 response of width
    /// `n_bar` at gate `beta1`: the greedy honest search —
    ///
    /// * terminal: `n̄ ≤ 16` → the single-stage `sound_profile_for`
    ///   (the cheap row);
    /// * else: the cheapest sound row that (a) covers `w·r₂ ≥ n̄`,
    ///   (b) reduces (`w < n̄`), (c) clears the floor+allowance at its
    ///   own grown gate, and (d) leaves the remaining budget reachable
    ///   (the lookahead: the remaining halvings at the MINIMUM growth
    ///   must land the final gate under the single-stage ceiling).
    ///
    /// Fail-closed when no schedule exists — the honest boundary
    /// (the coverage table this search produces is the module's
    /// published boundary).
    pub fn sound_chain_for(
        n_bar: usize,
        beta1: u64,
        q: u64,
        ring_dim: u64,
    ) -> Result<WidthChainParams, String> {
        let floor = SECURITY_FLOOR_BITS + CHAIN_GRINDING_BITS;
        let mut stages: Vec<WidthFoldParams> = Vec::new();
        let mut cur = n_bar.max(1);
        let mut beta = beta1;
        // The single-stage terminal: the existing cheap-row search.
        if cur <= 16 {
            let p = WidthFoldParams::sound_profile_for_floor(cur, beta1, q, ring_dim, floor)?;
            stages.push(p);
            let chain = WidthChainParams { stages };
            chain.assert_sound_chain(beta1, q, ring_dim)?;
            return Ok(chain);
        }
        // The staged descent. Depth cap: the halving bound + slack.
        let max_stages = 2 * n_bar.next_power_of_two().trailing_zeros() as usize + 2;
        for _ in 0..max_stages {
            // The lookahead: at the MINIMUM per-stage growth (r₂·A₂ = 2)
            // the remaining halvings must land under the single-stage
            // ceiling (the terminal search must still find a row).
            let halvings_left = (cur.div_ceil(16) as f64).log2().ceil() as u64;
            let terminal_beta = beta * (1u64 << halvings_left.min(31));
            if halvings_left <= 31
                && terminal_beta < q / 4
                && WidthFoldParams::sound_profile_for_floor(16, terminal_beta, q, ring_dim, floor)
                    .is_ok()
            {
                // Direct terminal if we're within the single-stage cover.
                if cur <= 16 {
                    break;
                }
            }
            // Pick the cheapest sound intermediate row.
            let row = Self::cheapest_intermediate(cur, beta, q, ring_dim, floor)?;
            // The lookahead with the chosen row's growth.
            let beta_out = row.beta2(beta);
            if beta_out >= q / 2 {
                return Err(format!(
                    "chain budget exhausted at width {cur}: beta 2^{:.0} breaches q/2 \
                     (the honest Q_32 boundary — the Modulus-50 class is the follow-up)",
                    (beta as f64).log2()
                ));
            }
            let next = cur.div_ceil(row.r2).max(1);
            let halvings_left = (next.div_ceil(16) as f64).log2().ceil() as u64;
            let terminal_beta = if halvings_left <= 31 {
                beta_out * (1u64 << halvings_left.min(31))
            } else {
                u64::MAX
            };
            if terminal_beta >= q / 2
                || WidthFoldParams::sound_profile_for_floor(16, terminal_beta, q, ring_dim, floor)
                    .is_err()
            {
                return Err(format!(
                    "chain dead-end at width {cur}: the terminal gate 2^{:.0} is unreachable \
                     (lower the level-1 amplitude or shrink the stream)",
                    (terminal_beta.min(q).saturating_sub(1) as f64).log2()
                ));
            }
            stages.push(row);
            beta = beta_out;
            cur = next;
            if cur <= 16 {
                break;
            }
        }
        if cur > 16 {
            return Err("chain depth cap exceeded (no schedule found)".into());
        }
        // The terminal single stage.
        let terminal = WidthFoldParams::sound_profile_for_floor(cur, beta, q, ring_dim, floor)?;
        stages.push(terminal);
        let chain = WidthChainParams { stages };
        chain.assert_sound_chain(beta1, q, ring_dim)?;
        Ok(chain)
    }

    /// The cheapest sound intermediate row at (width, gate): ordered by
    /// the honest cost ledger (small `r₂` first — the garbage lever;
    /// then small `w`; then the `(κ, A₂)` knobs ascending in cost).
    fn cheapest_intermediate(
        n_bar: usize,
        beta: u64,
        q: u64,
        ring_dim: u64,
        floor: f64,
    ) -> Result<WidthFoldParams, String> {
        let mut best: Option<(usize, WidthFoldParams)> = None;
        for r2 in [2usize, 4, 8] {
            let w_min = n_bar.div_ceil(r2).max(1);
            // Cap the search at the estimator's rated widths: the
            // instance width w + r₂ must stay in the model's reach.
            for w in [w_min, w_min + 1] {
                if w >= n_bar {
                    continue; // no reduction
                }
                for kappa in [8usize, 16, 32, 64, 128, 256] {
                    if kappa >= w + r2 {
                        continue; // the estimator requires width > kappa
                    }
                    // Amplitudes ascending: the minimum growth first
                    // (the budget-preserving choice), the challenge-
                    // space health priced by the grinding allowance.
                    for amplitude in [1u32, 2, 4, 16, 64] {
                        let cand = WidthFoldParams {
                            r2,
                            kappa,
                            amplitude,
                            w,
                        };
                        let beta_out = cand.beta2(beta);
                        if beta_out >= q / 2 {
                            continue;
                        }
                        let (cl, _) = match cand.profile_bits(beta, q, ring_dim) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        if cl < floor {
                            continue;
                        }
                        // The cost: garbage dominates (r₂²·k_in), the
                        // inner commitments next (r₂·κ), then w.
                        let cost = r2 * r2 * 8 + r2 * kappa + w;
                        let better = best.as_ref().map(|(b, _)| cost < *b).unwrap_or(true);
                        if better {
                            best = Some((cost, cand));
                        }
                    }
                }
            }
        }
        best.map(|(_, p)| p).ok_or_else(|| {
            format!(
                "no sound intermediate row at width {n_bar}, beta 2^{:.0} \
                 (the honest boundary of the Q_32 sound regime)",
                (beta as f64).log2()
            )
        })
    }
}

/// The staged proof: one width-fold proof per stage (each carrying its
/// own params, pre-challenge material, and folded response).
#[derive(Clone, Debug)]
pub struct WidthChainProof {
    pub params: WidthChainParams,
    /// The level-1 response width (before any folding).
    pub n_bar: usize,
    pub stages: Vec<WidthFoldProof>,
    /// The per-stage estimator verdicts at prove time (the posture
    /// markers — the verifier re-derives and compares).
    pub classical_bits: Vec<f64>,
}

/// The per-stage seed salt (independent inner keys per stage).
fn stage_seed(seed: [u8; 32], stage: usize) -> [u8; 32] {
    let mut st = Transcript::new_default(b"lzx-width-chain-seed");
    let _ = st.append_bytes(b"seed", &seed);
    let _ = st.append_bytes(b"stage", &(stage as u32).to_le_bytes());
    let mut s = [0u8; 32];
    if let Ok(b) = st.challenge_bytes(b"key", 32) {
        s.copy_from_slice(&b);
    }
    s
}

/// The stage's inner-key column blocks (both sides regenerate the key
/// from the salted seed + the stage params).
#[allow(clippy::too_many_arguments)]
fn inner_key_blocks(
    ring: &RingConfig,
    seed: [u8; 32],
    stage: usize,
    params: &WidthFoldParams,
    beta_in: u64,
) -> Result<Vec<Vec<RingElement>>, String> {
    let key = inner_key(ring, seed, stage, params, beta_in)?;
    let blocks = (0..params.w)
        .map(|c| {
            (0..params.kappa)
                .map(|rr| key.entry(rr, c).cloned().unwrap_or_else(|| ring.zero()))
                .collect()
        })
        .collect();
    Ok(blocks)
}

/// The stage's inner Ajtai key (the same derivation `prove_width_fold`
/// applies internally, with the per-stage salt).
fn inner_key(
    ring: &RingConfig,
    seed: [u8; 32],
    stage: usize,
    params: &WidthFoldParams,
    beta_in: u64,
) -> Result<AjtaiPublicKey, String> {
    let a2_params = AjtaiParams {
        ring: ring.clone(),
        k: params.kappa,
        m: params.w,
        norm_bound: u32::try_from(params.beta2(beta_in).max(1)).map_err(|e| format!("{e:?}"))?,
    };
    let salted = stage_seed(seed, stage);
    // The fold's own derivation over the salted seed: identical to
    // prove/verify_width_fold's internal key (derive_w2_seed applied
    // to the stage-salted base).
    let mut st = Transcript::new_default(b"lzx-width-fold-key");
    let _ = st.append_bytes(b"seed", &salted);
    let shape = [
        params.r2 as u32,
        params.kappa as u32,
        params.amplitude,
        params.w as u32,
    ];
    let _ = st.append_bytes(
        b"shape",
        &shape
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let mut s = [0u8; 32];
    if let Ok(b) = st.challenge_bytes(b"key", 32) {
        s.copy_from_slice(&b);
    }
    AjtaiPublicKey::from_seed(a2_params, s).map_err(|e| format!("{e:?}"))
}

/// The γ-weighted functional fold: `Ψ_{next}[m] = Σ_i γ_i·Ψ[i·w·n + m]`
/// (the weights beyond the slices are ZERO — the zero-pad convention).
fn fold_psi(psi: &[Goldilocks], gammas: &[i64], w: usize, n: usize) -> Vec<Goldilocks> {
    let _r2 = gammas.len();
    let mut out = vec![Goldilocks::ZERO; w * n];
    for m in 0..w * n {
        let mut acc = Goldilocks::ZERO;
        for (i, g) in gammas.iter().enumerate() {
            if *g == 0 {
                continue;
            }
            let wgt = psi.get(i * w * n + m).copied().unwrap_or(Goldilocks::ZERO);
            if wgt == Goldilocks::ZERO {
                continue;
            }
            let term = wgt.mul(&Goldilocks::from_u64(g.unsigned_abs()));
            acc = if *g < 0 {
                acc.sub(&term)
            } else {
                acc.add(&term)
            };
        }
        out[m] = acc;
    }
    out
}

/// The derived public claims after stage `ℓ` (both sides compute
/// identically from the transmitted proof + the transcript γ's):
/// `(t_{ℓ+1}, u_{ℓ+1})`.
fn derived_claims(
    ring: &RingConfig,
    proof: &WidthFoldProof,
    gammas: &[i64],
) -> Result<(Vec<RingElement>, Goldilocks), String> {
    let params = &proof.params;
    let r2 = params.r2;
    let kappa = params.kappa;
    // t = Σ_i γ_i·T_i (the stage's (W2) RHS — the next target).
    let t_inner = crate::codec::deserialize_elements(ring, &proof.t_inner)?;
    if t_inner.len() != r2 * kappa {
        return Err(format!("inner count {} != {}", t_inner.len(), r2 * kappa));
    }
    let mut t = vec![ring.zero(); kappa];
    for (i, g) in gammas.iter().enumerate() {
        if *g == 0 {
            continue;
        }
        for rr in 0..kappa {
            let prod = t_inner[i * kappa + rr].scale_i64(*g);
            t[rr] = t[rr].add(&prod).map_err(|e| format!("{e:?}"))?;
        }
    }
    // u = Σ_i γ_i²·u_i + Σ_{i≠j} γ_iγ_j·g_ij (the (W3) RHS folded).
    let u_parts = deserialize_goldilocks(&proof.u_parts, r2)?;
    let g_func = deserialize_goldilocks(&proof.g_func, r2 * (r2 - 1))?;
    let mut u = Goldilocks::ZERO;
    for (i, gi) in gammas.iter().enumerate() {
        if *gi == 0 {
            continue;
        }
        let sq = gi * gi;
        let term = u_parts[i].mul(&Goldilocks::from_u64(sq.unsigned_abs()));
        u = if sq < 0 { u.sub(&term) } else { u.add(&term) };
    }
    let mut idx = 0usize;
    for (i, gi) in gammas.iter().enumerate() {
        for (j, gj) in gammas.iter().enumerate() {
            if i != j {
                if *gi != 0 && *gj != 0 {
                    let w_ij = gi * gj;
                    let term = g_func[idx].mul(&Goldilocks::from_u64(w_ij.unsigned_abs()));
                    u = if w_ij < 0 { u.sub(&term) } else { u.add(&term) };
                }
                idx += 1;
            }
        }
    }
    Ok((t, u))
}

/// Serialize Goldilocks values (LE u64s — the wire form).
#[allow(dead_code)]
fn serialize_goldilocks(vals: &[Goldilocks]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 * vals.len());
    for v in vals {
        buf.extend_from_slice(&v.to_canonical_u64().to_le_bytes());
    }
    buf
}

fn deserialize_goldilocks(bytes: &[u8], count: usize) -> Result<Vec<Goldilocks>, String> {
    if bytes.len() != 8 * count {
        return Err(format!(
            "goldilocks wire length {} != {}",
            bytes.len(),
            8 * count
        ));
    }
    let mut out = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(8) {
        let raw = u64::from_le_bytes(chunk.try_into().map_err(|e| format!("{e:?}"))?);
        out.push(Goldilocks::from_u64(raw));
    }
    Ok(out)
}

/// Decode the folded response from a stage's proof (the balanced
/// coefficients → ring elements; the (W4) gate already ran).
fn decode_stage_response(
    ring: &RingConfig,
    proof: &WidthFoldProof,
    beta: u64,
) -> Result<Vec<RingElement>, String> {
    let q = u64::from(ring.modulus.q);
    let n = ring.n();
    let coeffs = crate::codec::decode_response(&proof.response)?;
    if coeffs.len() != proof.params.w * n {
        return Err("stage response length".into());
    }
    let beta2 = proof.params.beta2(beta);
    let mut out = Vec::with_capacity(proof.params.w);
    for chunk in coeffs.chunks(n) {
        let mut cs = vec![0u32; n];
        for (i, &c) in chunk.iter().enumerate() {
            if (c as i64).abs() > beta2 as i64 {
                return Err("stage norm gate".into());
            }
            cs[i] = (c as i64).rem_euclid(q as i64) as u32;
        }
        out.push(RingElement::from_coeffs(ring, cs));
    }
    Ok(out)
}

/// Prove the recursive width-collapse chain over the level-1 response
/// `v ∈ R^{n̄}`: stage 0 folds `v` under the level-1 key's blocks,
/// stage `ℓ` folds stage `ℓ−1`'s output under the previous stage's
/// inner key. Fails closed on any stage's self-check or profile gate.
#[allow(clippy::too_many_arguments)]
pub fn prove_width_fold_chain(
    ring: &RingConfig,
    v: &[RingElement],
    t_target: &[RingElement],
    u_target: &Goldilocks,
    f_bar_blocks: &[Vec<RingElement>],
    k: usize,
    psi_weights: &[Goldilocks],
    params: WidthChainParams,
    beta1: u64,
    seed: [u8; 32],
    transcript: &mut Transcript,
) -> Result<WidthChainProof, String> {
    let q = u64::from(ring.modulus.q);
    let ring_dim = ring.n() as u64;
    let n_bar = v.len();
    if params.stages.is_empty() {
        return Err("empty chain".into());
    }
    // The fail-closed chain posture gate (every stage, at prove time).
    let verdicts = params.assert_sound_chain(beta1, q, ring_dim)?;
    let classical_bits: Vec<f64> = verdicts.iter().map(|(cl, _)| *cl).collect();

    // The threading state.
    let mut cur_v: Vec<RingElement> = v.to_vec();
    let mut cur_t: Vec<RingElement> = t_target.to_vec();
    let mut cur_u = *u_target;
    let mut cur_blocks: Vec<Vec<RingElement>> = f_bar_blocks.to_vec();
    let mut cur_k = k;
    let mut cur_psi: Vec<Goldilocks> = psi_weights.to_vec();
    let gates = params.stage_gates(beta1);

    let mut stages = Vec::with_capacity(params.stages.len());
    for (ell, stage_params) in params.stages.iter().enumerate() {
        let beta_in = gates[ell];
        let salted = stage_seed(seed, ell);
        let (proof, gammas) = prove_width_fold_ex(
            ring,
            &cur_v,
            &cur_t,
            &cur_u,
            &cur_blocks,
            cur_k,
            &cur_psi,
            stage_params.clone(),
            beta_in,
            salted,
            transcript,
        )?;
        // The next stage's inputs (identical derivation on the verify
        // side): the response, the derived public claims, the inner
        // key's blocks, and the folded functional weights.
        cur_v = decode_stage_response(ring, &proof, beta_in)?;
        let (t_next, u_next) = derived_claims(ring, &proof, &gammas)?;
        cur_t = t_next;
        cur_u = u_next;
        cur_blocks = inner_key_blocks(ring, seed, ell, stage_params, beta_in)?;
        cur_k = stage_params.kappa;
        cur_psi = fold_psi(&cur_psi, &gammas, stage_params.w, ring.n());
        stages.push(proof);
    }
    Ok(WidthChainProof {
        params,
        n_bar,
        stages,
        classical_bits,
    })
}

/// Verify the recursive width-collapse chain: replays every stage's
/// `verify_width_fold` ((W0)–(W4) + the posture gate) and derives the
/// next stage's public claims identically to the prover. The FINAL
/// stage's response is the transmitted terminal; every intermediate
/// response is bound by its stage's `[A₂ | −T]` instance.
#[allow(clippy::too_many_arguments)]
pub fn verify_width_fold_chain(
    ring: &RingConfig,
    t_target: &[RingElement],
    u_target: &Goldilocks,
    f_bar_blocks: &[Vec<RingElement>],
    k: usize,
    psi_weights: &[Goldilocks],
    beta1: u64,
    seed: [u8; 32],
    proof: &WidthChainProof,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let q = u64::from(ring.modulus.q);
    let ring_dim = ring.n() as u64;
    if proof.stages.len() != proof.params.stages.len() {
        return Err("stage count mismatch".into());
    }
    // The fail-closed chain posture re-derivation + marker check.
    let verdicts = proof.params.assert_sound_chain(beta1, q, ring_dim)?;
    for (ell, ((cl, _), marker)) in verdicts.iter().zip(proof.classical_bits.iter()).enumerate() {
        if (cl - marker).abs() > 1.0 {
            return Err(format!(
                "stage {ell} posture marker mismatch: proof {marker} vs verifier {cl:.1}"
            ));
        }
    }
    let gates = proof.params.stage_gates(beta1);
    let mut cur_t: Vec<RingElement> = t_target.to_vec();
    let mut cur_u = *u_target;
    let mut cur_blocks: Vec<Vec<RingElement>> = f_bar_blocks.to_vec();
    let mut cur_k = k;
    let mut cur_psi: Vec<Goldilocks> = psi_weights.to_vec();
    for (ell, stage_proof) in proof.stages.iter().enumerate() {
        let beta_in = gates[ell];
        let salted = stage_seed(seed, ell);
        let (_, gammas) = verify_width_fold_ex(
            ring,
            &cur_t,
            &cur_u,
            &cur_blocks,
            cur_k,
            &cur_psi,
            beta_in,
            salted,
            stage_proof,
            transcript,
        )?;
        let gammas = gammas.as_slice();
        // The next stage's public claims (identical to the prover).
        let (t_next, u_next) = derived_claims(ring, stage_proof, gammas)?;
        cur_t = t_next;
        cur_u = u_next;
        cur_blocks = inner_key_blocks(ring, seed, ell, &stage_proof.params, beta_in)?;
        cur_k = stage_proof.params.kappa;
        cur_psi = fold_psi(&cur_psi, gammas, stage_proof.params.w, ring.n());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::apply_key;

    /// The test fixture tuple (the type_complexity lint's shape alias).
    type Fixture = (
        RingConfig,
        Vec<RingElement>,
        Vec<Vec<RingElement>>,
        Vec<Goldilocks>,
        [u8; 32],
    );

    fn ring() -> RingConfig {
        crate::codec::q32_ring().unwrap()
    }

    /// A deterministic short response (the byte-instance regime).
    fn synth_response(ring: &RingConfig, n_bar: usize, seed: u64) -> Vec<RingElement> {
        let n = ring.n();
        let q = ring.modulus.q;
        (0..n_bar)
            .map(|i| {
                let coeffs: Vec<u32> = (0..n)
                    .map(|j| {
                        let mut x = seed
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .wrapping_add((i as u64 + 1).wrapping_mul(0x2545_F491_4F6C_DD1D))
                            .wrapping_add((j as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
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
            .map(|c| {
                (0..k)
                    .map(|rr| key.entry(rr, c).cloned().unwrap())
                    .collect()
            })
            .collect();
        (key, blocks)
    }

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

    fn setup(n_bar: usize) -> Fixture {
        let r = ring();
        let v = synth_response(&r, n_bar, 0xABCD);
        let (_, blocks) = key_blocks(&r, 4, n_bar, [7u8; 32]);
        let psi = synth_psi(n_bar, r.n(), 0x1234);
        (r, v, blocks, psi, [7u8; 32])
    }

    /// The chain coverage beyond the single-stage 16: n̄ = 64 at the
    /// byte-instance gate folds through multiple sound stages.
    #[test]
    fn chain_honest_roundtrip_beyond_single_stage() {
        let (ring, v, blocks, psi, seed) = setup(64);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params = WidthChainParams::sound_chain_for(64, beta1, q, ring.n() as u64)
            .expect("a staged schedule exists at n_bar=64");
        assert!(params.stages.len() >= 2, "the chain must stage");
        assert!(params.final_width() <= 16);
        let mut tr = Transcript::new_default(b"test-width-chain");
        let proof = prove_width_fold_chain(
            &ring,
            &v,
            &t,
            &u,
            &blocks,
            k,
            &psi,
            params.clone(),
            beta1,
            seed,
            &mut tr,
        )
        .expect("honest chain prove");
        let mut tr2 = Transcript::new_default(b"test-width-chain");
        verify_width_fold_chain(
            &ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2,
        )
        .expect("honest chain verify");
    }

    /// n̄ = 256 — well beyond the single-stage regime.
    #[test]
    fn chain_honest_roundtrip_wide() {
        let (ring, v, blocks, psi, seed) = setup(256);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        // β₁ = 2^15 (the r₁=8 sound-profile regime).
        let beta1 = 1u64 << 15;
        let params = WidthChainParams::sound_chain_for(256, beta1, q, ring.n() as u64)
            .expect("a staged schedule exists at n_bar=256");
        assert!(params.final_width() <= 16);
        let mut tr = Transcript::new_default(b"test-width-chain-w");
        let proof = prove_width_fold_chain(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("honest chain prove");
        let mut tr2 = Transcript::new_default(b"test-width-chain-w");
        verify_width_fold_chain(
            &ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2,
        )
        .expect("honest chain verify");
    }

    /// Tampering an intermediate stage's inner commitments must fail
    /// (that stage's (W2) or the next stage's derived target).
    #[test]
    fn chain_tampered_intermediate_rejected() {
        let (ring, v, blocks, psi, seed) = setup(64);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params =
            WidthChainParams::sound_chain_for(64, beta1, q, ring.n() as u64).expect("schedule");
        let mut tr = Transcript::new_default(b"test-width-chain");
        let mut proof = prove_width_fold_chain(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("prove");
        // Corrupt the FIRST stage's inner commitments.
        assert!(proof.stages.len() >= 2);
        assert!(!proof.stages[0].t_inner.is_empty());
        proof.stages[0].t_inner[0] ^= 0x01;
        let mut tr2 = Transcript::new_default(b"test-width-chain");
        assert!(verify_width_fold_chain(
            &ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2
        )
        .is_err());
    }

    /// Tampering the FINAL stage's response must fail (W2).
    #[test]
    fn chain_tampered_final_response_rejected() {
        let (ring, v, blocks, psi, seed) = setup(64);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params =
            WidthChainParams::sound_chain_for(64, beta1, q, ring.n() as u64).expect("schedule");
        let mut tr = Transcript::new_default(b"test-width-chain");
        let mut proof = prove_width_fold_chain(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("prove");
        let last = proof.stages.len() - 1;
        let mut coeffs = crate::codec::decode_response(&proof.stages[last].response).unwrap();
        assert!(!coeffs.is_empty());
        coeffs[0] = coeffs[0].wrapping_add(1);
        proof.stages[last].response = crate::codec::encode_response(&coeffs).unwrap();
        let mut tr2 = Transcript::new_default(b"test-width-chain");
        assert!(verify_width_fold_chain(
            &ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2
        )
        .is_err());
    }

    /// A wrong public target fails at stage 0's (W0).
    #[test]
    fn chain_wrong_target_rejected() {
        let (ring, v, blocks, psi, seed) = setup(64);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params =
            WidthChainParams::sound_chain_for(64, beta1, q, ring.n() as u64).expect("schedule");
        let mut tr = Transcript::new_default(b"test-width-chain");
        let proof = prove_width_fold_chain(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("prove");
        let mut t_wrong = t.clone();
        t_wrong[0] = t_wrong[0].add(&ring.one()).unwrap();
        let mut tr2 = Transcript::new_default(b"test-width-chain");
        assert!(verify_width_fold_chain(
            &ring, &t_wrong, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2
        )
        .is_err());
    }

    /// Tampering an intermediate stage's FUNCTIONAL garbage must fail
    /// (that stage's (W3) — and the derived claim chain).
    #[test]
    fn chain_tampered_functional_rejected() {
        let (ring, v, blocks, psi, seed) = setup(64);
        let k = 4usize;
        let t = apply_key(&ring, &blocks, &v, k);
        let q = u64::from(ring.modulus.q);
        let u = functional_of(&ring, &v, &psi, q);
        let beta1 = 255u64;
        let params =
            WidthChainParams::sound_chain_for(64, beta1, q, ring.n() as u64).expect("schedule");
        let mut tr = Transcript::new_default(b"test-width-chain");
        let mut proof = prove_width_fold_chain(
            &ring, &v, &t, &u, &blocks, k, &psi, params, beta1, seed, &mut tr,
        )
        .expect("prove");
        assert!(proof.stages.len() >= 2);
        assert!(!proof.stages[0].g_func.is_empty());
        proof.stages[0].g_func[3] ^= 0x10;
        let mut tr2 = Transcript::new_default(b"test-width-chain");
        assert!(verify_width_fold_chain(
            &ring, &t, &u, &blocks, k, &psi, beta1, seed, &proof, &mut tr2
        )
        .is_err());
    }

    /// The schedule search: honest coverage + fail-closed boundaries.
    #[test]
    fn chain_search_coverage_and_floors() {
        let q = u64::from(ring().modulus.q);
        let ring_dim = ring().n() as u64;
        // Single-stage coverage: n̄ ≤ 16 → exactly one stage.
        let p = WidthChainParams::sound_chain_for(16, 255, q, ring_dim).unwrap();
        assert_eq!(p.stages.len(), 1);
        // The staged coverage at the byte-instance gate.
        for n_bar in [32usize, 64, 128, 256] {
            let p = WidthChainParams::sound_chain_for(n_bar, 1 << 15, q, ring_dim)
                .unwrap_or_else(|e| panic!("n_bar={n_bar}: {e}"));
            assert!(p.final_width() <= 16, "n_bar={n_bar}");
            p.assert_sound_chain(1 << 15, q, ring_dim).unwrap();
        }
        // The absurd gate fails closed (no schedule under q/2).
        assert!(
            WidthChainParams::sound_chain_for(256, 1 << 28, q, ring_dim).is_err(),
            "the absurd gate must fail closed"
        );
    }

    /// The cost model sanity: the staged cost is dominated by the early
    /// (wide) stages' commitments, and stays far below transmitting the
    /// unfolded response would... (the honest ledger's shape check).
    #[test]
    fn chain_cost_model() {
        let q = u64::from(ring().modulus.q);
        let ring_dim = ring().n() as u64;
        let p = WidthChainParams::sound_chain_for(256, 1 << 15, q, ring_dim).unwrap();
        let cost = p.cost_elements(4);
        assert!(cost > 0 && cost < 20_000, "cost {cost}");
        // The grinding allowance holds.
        assert!(p.grinding_bits() <= CHAIN_GRINDING_BITS);
    }
}
