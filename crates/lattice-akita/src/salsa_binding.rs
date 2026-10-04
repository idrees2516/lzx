//! **SALSA D4 — the binding closure and the r-column capacity split**
//! (the two honest-ledger follow-ups to the response-layer swap).
//!
//! # The closure (what this module closes)
//!
//! `salsa_response`'s documented outer-layer gap: "the Ajtai binding of
//! the byte-witness to the commitment — the authenticated opening at
//! the challenge — is not realized here; the binding-complete polylog
//! route is the compact-mode fold." This module lands that route:
//!
//! ```text
//! claims ──RLC carrier──> f(r_sc) ──(W0')/(W3) functional thread──> Φ(v) = f(r_sc)
//!                                              │
//! commitment t = F̄·v <──(W0) part-image sum────┘── the width-collapse chain
//!                                              │      (per-stage [A₂ | −T] ≥ floor)
//! D1 norm proof: every byte-coefficient ≤ 255 ─┘      (the β₁ = 255 gate's certificate)
//! ```
//!
//! `prove_grouped_salsa_bound` replaces the ψ-functional SUMCHECK with
//! the recursive width-collapse chain over the SAME byte-packed
//! witness: the chain's `(W0)` pins the part images to the commitment,
//! `(W0')`/`(W3)` carry the functional claim `Φ(v) = f(r_sc)`, and the
//! per-stage `[A₂ | −T]` instances (estimator-gated ≥ 128 + 32 bits)
//! are the binding — the level-1 `[F̄]` equation is *proven*, not
//! assumed, and the D1 norm proof certifies the shortness that
//! justifies the fold's `β₁ = 255` byte gate. Nothing intra-response
//! changes; the OUTER layer closes.
//!
//! # The split (what this module scales)
//!
//! The byte-packed D1 regime's per-commitment capacity is capped by the
//! Lemma-4 gate `m·n·B² < q/2` at `B = 255` — [`byte_capacity`] computes
//! the exact power-of-two boundary (2,048 values at ring dim 16 / Q_32;
//! the prior "~1,200" prose note was the margin-rounded statement of
//! this same cap). Beyond it, [`prove_grouped_salsa_split`] applies the
//! compact mode's discipline — the r-column split: the flat byte stream
//! splits into `r` columns (each within the gate, each under its own
//! domain-separated Ajtai key with its own D1 certificate and its own
//! binding chain), and the ψ-functional decomposes over the column
//! variables:
//!
//! ```text
//! f(r_sc) = Σ_j μ_j·u_j,   μ_j = eq(r_sc_tail, bin(j)),
//! u_j     = Φ_j(v_j)       (column j's functional claim, bound by its chain)
//! ```
//!
//! The response grows as `r·(D1 + chain)` — the honest O(r) price of
//! scaling past the per-commitment cap at fixed modulus (the Modulus-50
//! class remains the headroom route; see NEXT_STEPS).

use crate::pcs::{AkitaPcs, Commitment, GroupedOpening};
use crate::salsa_response::{byte_pack_witness, eq_at, psi_weights_at, SalsaResponseError};
use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_salsa::ring_norm::{prove_ring_norm, verify_ring_norm, RingNormProof};
use lattice_widthfold::chain::{
    prove_width_fold_chain, verify_width_fold_chain, WidthChainParams, WidthChainProof,
};
use lattice_widthfold::helpers::functional_of;

/// The byte gate: every byte-packed coefficient is bounded by 255 (the
/// D1 Lemma-4 regime — one byte per ring coefficient).
pub const BYTE_GATE: u64 = 255;

/// The exact per-commitment capacity (VALUES) of the byte-packed D1
/// regime under the Lemma-4 gate: the largest power-of-two slot count
/// `m` with `m·n·B² < q/2` at `B = 255`, converted to values
/// (`m·n/8`). Fail-closed (returns 0) if even one slot fails the gate.
pub fn byte_capacity(ring: &lattice_ring::RingConfig) -> usize {
    let n = ring.n();
    let mut m = 1usize;
    while lattice_salsa::ring_norm::wraparound_gate(m * 2, ring, BYTE_GATE).is_ok() {
        m *= 2;
    }
    m * n / 8
}

/// The column count for `values` values at the given per-column
/// capacity: the smallest power of two `r` with `values/r ≤ capacity`
/// (1 within capacity — the split degenerates to the bound response).
pub fn column_count_for(values: usize, capacity: usize) -> usize {
    if capacity == 0 || values <= capacity {
        return 1;
    }
    let mut r = 1usize;
    while values.div_ceil(r) > capacity {
        r *= 2;
    }
    r.min(values.next_power_of_two())
}

/// The fold-key seed, domain-separated from the (public) commitment
/// bytes — both sides derive it identically from the statement.
fn fold_seed(commitment_bytes: &[u8]) -> [u8; 32] {
    let mut st = Transcript::new_default(b"akita-salsa-fold-seed");
    let _ = st.append_bytes(b"com", commitment_bytes);
    let mut s = [0u8; 32];
    if let Ok(b) = st.challenge_bytes(b"seed", 32) {
        s.copy_from_slice(&b);
    }
    s
}

/// The per-column Ajtai key seed (public randomness, domain-separated
/// by the split shape — the reference's honest posture: production
/// setups derive column keys from a ceremony, not a public domain).
fn column_seed(num_columns: usize, col: usize) -> [u8; 32] {
    let mut st = Transcript::new_default(b"akita-salsa-split-key");
    let _ = st.append_bytes(b"r", &(num_columns as u32).to_le_bytes());
    let _ = st.append_bytes(b"j", &(col as u32).to_le_bytes());
    let mut s = [0u8; 32];
    if let Ok(b) = st.challenge_bytes(b"key", 32) {
        s.copy_from_slice(&b);
    }
    s
}

/// The level-1 key's column blocks (`n̄` blocks of `k` elements — the
/// fold's `f_bar_blocks` input, both sides derive from the public key).
fn key_blocks(pk: &AjtaiPublicKey, n_bar: usize) -> Vec<Vec<lattice_ring::RingElement>> {
    let ring = &pk.params.ring;
    let k = pk.params.k;
    (0..n_bar)
        .map(|c| {
            (0..k)
                .map(|rr| pk.entry(rr, c).cloned().unwrap_or_else(|| ring.zero()))
                .collect()
        })
        .collect()
}

/// The binding-closed SALSAA response: the carrier + D1 + the
/// width-collapse chain (the authenticated opening at the challenge).
#[derive(Clone, Debug)]
pub struct SalsaBoundResponse {
    /// The grouped carrier sumcheck (the RLC over the claims).
    pub sumcheck: lattice_sumcheck::SumcheckProof,
    /// The first claim's point (the shape carrier).
    pub point: Vec<Goldilocks>,
    /// The combined RLC claim `Σ_i ρ^i·f(r_i)`.
    pub value: Goldilocks,
    /// The D1 norm proof over the byte-packed witness (the β₁ = 255
    /// gate's certificate — the shortness precondition the fold's
    /// parameters honestly rest on).
    pub chain: RingNormProof,
    /// The D1 challenge evaluation `z(r)`.
    pub z_r: Goldilocks,
    /// The carrier terminal claim `f(r_sc)` — the fold's functional
    /// target (the (W0')/(W3) thread).
    pub f_term: Goldilocks,
    /// THE BINDING: the recursive width-collapse chain over the padded
    /// byte-witness (per-stage `[A₂ | −T]`, estimator-gated).
    pub fold: WidthChainProof,
}

impl AkitaPcs {
    /// Prove the grouped SALSAA response WITH the byte-witness↔
    /// commitment binding closed: the carrier, D1's norm certificate,
    /// and the width-collapse chain over the padded byte-witness (the
    /// functional target `f(r_sc)` rides the chain's (W0')/(W3)
    /// thread; the commitment rides (W0); the per-stage instances are
    /// the binding). The ψ-functional sumcheck of the open mode is
    /// SUBSUMED by the chain — the response carries strictly more
    /// binding at one fewer sumcheck.
    pub fn prove_grouped_salsa_bound(
        &self,
        mle: &DenseMle,
        claims: &[GroupedOpening],
        transcript: &mut Transcript,
    ) -> Result<SalsaBoundResponse, SalsaResponseError> {
        if claims.is_empty() {
            return Err(SalsaResponseError::Shape {
                expected: 1,
                got: 0,
            });
        }
        // 1. The grouped carrier (identical to the open mode's RLC).
        let rhos = transcript
            .challenge_fields(b"akita-group-rho", claims.len())
            .map_err(SalsaResponseError::Transcript)?;
        let mut vp = lattice_sumcheck::VirtualPolynomial::new(mle.num_vars);
        let fi = vp.add_factor(mle.clone())?;
        let mut combined_claim = Goldilocks::ZERO;
        for (i, claim) in claims.iter().enumerate() {
            let eq = DenseMle::eq_extension(&claim.point);
            let ei = vp.add_factor(eq)?;
            vp.add_term(rhos[i], vec![fi, ei])?;
            combined_claim = combined_claim.add(&rhos[i].mul(&claim.value));
        }
        let out = lattice_sumcheck::sumcheck::prove(&vp, combined_claim, transcript)?;
        let f_term = out
            .factor_claims
            .first()
            .copied()
            .ok_or(SalsaResponseError::Shape {
                expected: 1,
                got: 0,
            })?;

        // 2. The byte-packed witness, padded to the key's m slots.
        let ring = self.pk.params.ring.clone();
        let packed = byte_pack_witness(&ring, &mle.evaluations);
        let padded = self
            .pk
            .pad_to_m(&packed)
            .map_err(|_| SalsaResponseError::Shape {
                expected: self.pk.params.m,
                got: packed.len(),
            })?;
        let n_bar = padded.len();

        // 3. D1: the norm certificate (the β₁ = 255 gate's justification).
        let norm = prove_ring_norm(&padded, &ring, BYTE_GATE, transcript)?;
        let z_r = norm.z_at_challenge;

        // 4. THE BINDING: the width-collapse chain over the padded
        //    byte-witness. The ψ-weights at the carrier's terminal
        //    point r_sc (verifier-computable); the target is the
        //    commitment; the functional target is f(r_sc).
        let commitment = self
            .pk
            .commit(&padded)
            .map_err(|e| SalsaResponseError::Ajtai(format!("{e:?}")))?;
        let t_target: Vec<lattice_ring::RingElement> = commitment.rows.clone();
        let total_coeffs = n_bar * ring.n();
        let psi = psi_weights_at(&out.challenges, mle.evaluations.len(), total_coeffs);
        let blocks = key_blocks(&self.pk, n_bar);
        let q = u64::from(ring.modulus.q);
        let dim = ring.n() as u64;
        let params = WidthChainParams::sound_chain_for(n_bar, BYTE_GATE, q, dim)
            .map_err(|e| SalsaResponseError::Ajtai(format!("chain schedule: {e}")))?;
        let seed = fold_seed(&commitment.to_bytes());
        let fold = prove_width_fold_chain(
            &ring,
            &padded,
            &t_target,
            &f_term,
            &blocks,
            self.pk.params.k,
            &psi,
            params,
            BYTE_GATE,
            seed,
            transcript,
        )
        .map_err(|e| SalsaResponseError::Ajtai(format!("fold chain: {e}")))?;

        Ok(SalsaBoundResponse {
            sumcheck: out.proof,
            point: claims[0].point.clone(),
            value: combined_claim,
            chain: norm,
            z_r,
            f_term,
            fold,
        })
    }

    /// Verify the binding-closed SALSAA response: the carrier with the
    /// terminal binding, D1's norm sumcheck with the Lemma-4 gate, and
    /// the width-collapse chain — the (W0) part-image sum against the
    /// TRANSMITTED commitment, the (W0')/(W3) functional thread pinned
    /// to `f(r_sc)`, and every stage's estimator verdict re-derived.
    pub fn verify_grouped_salsa_bound(
        &self,
        commitment: &Commitment,
        claims: &[GroupedOpening],
        proof: &SalsaBoundResponse,
        transcript: &mut Transcript,
    ) -> Result<(), SalsaResponseError> {
        if claims.is_empty() || proof.point.len() != commitment.num_vars {
            return Err(SalsaResponseError::Shape {
                expected: commitment.num_vars,
                got: proof.point.len(),
            });
        }
        let rhos = transcript
            .challenge_fields(b"akita-group-rho", claims.len())
            .map_err(SalsaResponseError::Transcript)?;
        let mut combined = Goldilocks::ZERO;
        for (rho, c) in rhos.iter().zip(claims.iter()) {
            combined = combined.add(&rho.mul(&c.value));
        }
        // 1. The carrier: the RLC sumcheck; the terminal binds f_term.
        let verdict = proof
            .sumcheck
            .verify(commitment.num_vars, 2, combined, transcript, None)?;
        let mut expected_final = Goldilocks::ZERO;
        for (i, claim) in claims.iter().enumerate() {
            let eq_factor = eq_at(&claim.point, &verdict.point);
            expected_final = expected_final.add(&rhos[i].mul(&eq_factor.mul(&proof.f_term)));
        }
        if verdict.final_claim != expected_final {
            return Err(SalsaResponseError::TerminalBindingFailed);
        }
        // 2. D1: the norm sumcheck (bound reconstructed from the
        //    claimed norm; the Lemma-4 gate re-runs inside).
        let ring = self.pk.params.ring.clone();
        let total = (proof.chain.num_elements * proof.chain.ring_dim) as u64;
        let bound = ((proof.chain.claimed_norm_sq as f64 / total.max(1) as f64)
            .sqrt()
            .ceil() as u64)
            .max(1);
        verify_ring_norm(&proof.chain, &ring, bound, proof.z_r, transcript)?;
        // 3. THE BINDING: the chain against the transmitted commitment
        //    with the verifier's own ψ-weights at r_sc.
        let n_bar = proof.fold.n_bar;
        let total_coeffs = n_bar * ring.n();
        let psi = psi_weights_at(&verdict.point, 1usize << commitment.num_vars, total_coeffs);
        let blocks = key_blocks(&self.pk, n_bar);
        let seed = fold_seed(&commitment.commitment.to_bytes());
        verify_width_fold_chain(
            &ring,
            &commitment.commitment.rows,
            &proof.f_term,
            &blocks,
            self.pk.params.k,
            &psi,
            BYTE_GATE,
            seed,
            &proof.fold,
            transcript,
        )
        .map_err(|e| SalsaResponseError::Ajtai(format!("fold chain: {e}")))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The r-column capacity split (the compact mode's discipline)
// ---------------------------------------------------------------------------

/// One column's proof: the commitment, the functional claim, the D1
/// norm certificate, and the binding chain.
#[derive(Clone, Debug)]
pub struct SalsaColumnProof {
    /// The column's serialized Ajtai commitment (the public statement).
    pub commitment: Vec<u8>,
    /// The column's functional claim `u_j = Φ_j(v_j)` (the μ-weighted
    /// decomposition's term — checked against `f(r_sc)` by the
    /// verifier's own column weights).
    pub u_j: Goldilocks,
    /// The column's D1 norm proof (within the Lemma-4 gate by
    /// construction — the planner sizes the columns).
    pub d1: RingNormProof,
    /// The column's D1 challenge evaluation.
    pub z_r: Goldilocks,
    /// The column's binding chain (its own estimator-gated instances).
    pub fold: WidthChainProof,
}

/// The r-column split response: the carrier over the FULL MLE plus the
/// per-column proofs. Scales the byte-packed regime past the
/// per-commitment Lemma-4 cap at `r` columns (the compact mode's
/// discipline: the flat domain's top `log₂r` value bits index the
/// columns; the eq factor splits `eq(r_sc, x) = μ_j·eq(r_sc_head, m')`).
#[derive(Clone, Debug)]
pub struct SalsaSplitResponse {
    /// The grouped carrier sumcheck over the full MLE.
    pub sumcheck: lattice_sumcheck::SumcheckProof,
    /// The first claim's point (the shape carrier).
    pub point: Vec<Goldilocks>,
    /// The combined RLC claim.
    pub value: Goldilocks,
    /// The carrier terminal claim `f(r_sc)` — the μ-decomposition's
    /// target.
    pub f_term: Goldilocks,
    /// The per-column proofs (r = a power of two).
    pub columns: Vec<SalsaColumnProof>,
}

/// The column selector weight `μ_j = eq(r_sc[col_bits], bin(j))` — the
/// eq factor over the TOP `log₂r` coordinates of `r_sc` (MSB-first,
/// matching `psi_weights_at`'s convention: coordinate 0 ↔ the value
/// index's most significant bit, which is the column selector).
fn mu_weight(r_sc: &[Goldilocks], num_columns: usize, j: usize) -> Goldilocks {
    let col_bits = num_columns.trailing_zeros() as usize;
    let mut acc = Goldilocks::ONE;
    for (i, &r) in r_sc.iter().take(col_bits).enumerate() {
        let bit = (j >> (col_bits - 1 - i)) & 1;
        let sel = if bit == 1 { r } else { Goldilocks::ONE.sub(&r) };
        acc = acc.mul(&sel);
    }
    acc
}

impl AkitaPcs {
    /// Prove the grouped SALSAA response ACROSS r columns (the capacity
    /// split): the carrier runs over the full MLE; the byte stream
    /// splits into `r` columns of ≤ [`byte_capacity`] values, each with
    /// its own domain-separated key, commitment, D1 certificate, and
    /// binding chain; the functional decomposes as
    /// `f(r_sc) = Σ_j μ_j·u_j` with the verifier's own column weights.
    pub fn prove_grouped_salsa_split(
        &self,
        mle: &DenseMle,
        claims: &[GroupedOpening],
        transcript: &mut Transcript,
    ) -> Result<SalsaSplitResponse, SalsaResponseError> {
        if claims.is_empty() {
            return Err(SalsaResponseError::Shape {
                expected: 1,
                got: 0,
            });
        }
        // 1. The grouped carrier over the FULL MLE (unchanged layer).
        let rhos = transcript
            .challenge_fields(b"akita-group-rho", claims.len())
            .map_err(SalsaResponseError::Transcript)?;
        let mut vp = lattice_sumcheck::VirtualPolynomial::new(mle.num_vars);
        let fi = vp.add_factor(mle.clone())?;
        let mut combined_claim = Goldilocks::ZERO;
        for (i, claim) in claims.iter().enumerate() {
            let eq = DenseMle::eq_extension(&claim.point);
            let ei = vp.add_factor(eq)?;
            vp.add_term(rhos[i], vec![fi, ei])?;
            combined_claim = combined_claim.add(&rhos[i].mul(&claim.value));
        }
        let out = lattice_sumcheck::sumcheck::prove(&vp, combined_claim, transcript)?;
        let f_term = out
            .factor_claims
            .first()
            .copied()
            .ok_or(SalsaResponseError::Shape {
                expected: 1,
                got: 0,
            })?;
        let r_sc = out.challenges.clone();

        // 2. The split plan (fail-closed on the geometry).
        let ring = self.pk.params.ring.clone();
        let n = ring.n();
        let q = u64::from(ring.modulus.q);
        let dim = n as u64;
        let values = mle.evaluations.len();
        if !values.is_power_of_two() || values < 2 {
            return Err(SalsaResponseError::Shape {
                expected: 2,
                got: values,
            });
        }
        let r = column_count_for(values, byte_capacity(&ring));
        let vpc = values / r; // values per column (a power of two)
        let col_bits = r.trailing_zeros() as usize;

        // 3. Per column: pack, commit, D1, and the binding chain.
        let mut columns = Vec::with_capacity(r);
        for j in 0..r {
            let col_evals = &mle.evaluations[j * vpc..(j + 1) * vpc];
            let packed = byte_pack_witness(&ring, col_evals);
            let m_col = packed.len().next_power_of_two().max(1);
            // The planner's invariant: the column fits the Lemma-4 gate.
            if lattice_salsa::ring_norm::wraparound_gate(m_col, &ring, BYTE_GATE).is_err() {
                return Err(SalsaResponseError::Ajtai(format!(
                    "column {j}: m_col={m_col} breaches the Lemma-4 gate (planner bug)"
                )));
            }
            let key = AjtaiPublicKey::from_seed(
                AjtaiParams {
                    ring: ring.clone(),
                    k: 2,
                    m: m_col,
                    norm_bound: 1 << 20,
                },
                column_seed(r, j),
            )
            .map_err(|e| SalsaResponseError::Ajtai(format!("{e:?}")))?;
            let padded = key
                .pad_to_m(&packed)
                .map_err(|_| SalsaResponseError::Shape {
                    expected: m_col,
                    got: packed.len(),
                })?;
            let n_bar = padded.len();
            let commitment = key
                .commit(&padded)
                .map_err(|e| SalsaResponseError::Ajtai(format!("{e:?}")))?;
            // D1: the column's norm certificate.
            let d1 = prove_ring_norm(&padded, &ring, BYTE_GATE, transcript)?;
            let z_r = d1.z_at_challenge;
            // The column-local ψ-weights: the head coordinates of r_sc
            // (the low value bits) at the column-local index.
            let r_head = &r_sc[col_bits..];
            let psi = psi_weights_at(r_head, vpc, n_bar * n);
            let u_j = functional_of(&ring, &padded, &psi, q);
            // The binding chain.
            let blocks = key_blocks(&key, n_bar);
            let params = WidthChainParams::sound_chain_for(n_bar, BYTE_GATE, q, dim)
                .map_err(|e| SalsaResponseError::Ajtai(format!("col {j} schedule: {e}")))?;
            let seed = fold_seed(&commitment.to_bytes());
            let fold = prove_width_fold_chain(
                &ring,
                &padded,
                &commitment.rows,
                &u_j,
                &blocks,
                2,
                &psi,
                params,
                BYTE_GATE,
                seed,
                transcript,
            )
            .map_err(|e| SalsaResponseError::Ajtai(format!("col {j} fold: {e}")))?;
            columns.push(SalsaColumnProof {
                commitment: commitment.to_bytes(),
                u_j,
                d1,
                z_r,
                fold,
            });
        }
        Ok(SalsaSplitResponse {
            sumcheck: out.proof,
            point: claims[0].point.clone(),
            value: combined_claim,
            f_term,
            columns,
        })
    }

    /// Verify the r-column split response: the carrier, the
    /// μ-decomposition `Σ_j μ_j·u_j = f(r_sc)` with the verifier's own
    /// column weights, and every column's D1 + binding chain against
    /// the transmitted commitment under the re-derived column key.
    pub fn verify_grouped_salsa_split(
        &self,
        num_vars: usize,
        claims: &[GroupedOpening],
        proof: &SalsaSplitResponse,
        transcript: &mut Transcript,
    ) -> Result<(), SalsaResponseError> {
        if claims.is_empty() || proof.point.len() != num_vars {
            return Err(SalsaResponseError::Shape {
                expected: num_vars,
                got: proof.point.len(),
            });
        }
        let rhos = transcript
            .challenge_fields(b"akita-group-rho", claims.len())
            .map_err(SalsaResponseError::Transcript)?;
        let mut combined = Goldilocks::ZERO;
        for (rho, c) in rhos.iter().zip(claims.iter()) {
            combined = combined.add(&rho.mul(&c.value));
        }
        // 1. The carrier → r_sc + the terminal binding.
        let verdict = proof
            .sumcheck
            .verify(num_vars, 2, combined, transcript, None)?;
        let mut expected_final = Goldilocks::ZERO;
        for (i, claim) in claims.iter().enumerate() {
            let eq_factor = eq_at(&claim.point, &verdict.point);
            expected_final = expected_final.add(&rhos[i].mul(&eq_factor.mul(&proof.f_term)));
        }
        if verdict.final_claim != expected_final {
            return Err(SalsaResponseError::TerminalBindingFailed);
        }
        // 2. The split geometry + the μ-decomposition.
        let ring = self.pk.params.ring.clone();
        let n = ring.n();
        let values = 1usize << num_vars;
        let r = proof.columns.len();
        if r == 0 || !r.is_power_of_two() || r > values {
            return Err(SalsaResponseError::Shape {
                expected: values,
                got: r,
            });
        }
        let vpc = values / r;
        let mut decomp = Goldilocks::ZERO;
        for (j, col) in proof.columns.iter().enumerate() {
            let mu = mu_weight(&verdict.point, r, j);
            decomp = decomp.add(&mu.mul(&col.u_j));
        }
        if decomp != proof.f_term {
            return Err(SalsaResponseError::TerminalBindingFailed);
        }
        // 3. Per column: the re-derived key, the transmitted
        //    commitment, D1, and the binding chain.
        for (j, col) in proof.columns.iter().enumerate() {
            let n_bar = col.fold.n_bar;
            let key = AjtaiPublicKey::from_seed(
                AjtaiParams {
                    ring: ring.clone(),
                    k: 2,
                    m: n_bar,
                    norm_bound: 1 << 20,
                },
                column_seed(r, j),
            )
            .map_err(|e| SalsaResponseError::Ajtai(format!("{e:?}")))?;
            let commitment =
                lattice_commitment::ajtai::AjtaiCommitment::from_bytes(&ring, 2, &col.commitment)
                    .map_err(|e| SalsaResponseError::Ajtai(format!("col {j} com: {e:?}")))?;
            // D1 (bound reconstructed from the claimed norm).
            let total = (col.d1.num_elements * col.d1.ring_dim) as u64;
            let bound = ((col.d1.claimed_norm_sq as f64 / total.max(1) as f64)
                .sqrt()
                .ceil() as u64)
                .max(1);
            verify_ring_norm(&col.d1, &ring, bound, col.z_r, transcript)?;
            // The binding chain: the verifier's own head weights.
            let col_bits = r.trailing_zeros() as usize;
            let r_head = &verdict.point[col_bits..];
            let psi = psi_weights_at(r_head, vpc, n_bar * n);
            let blocks = key_blocks(&key, n_bar);
            let seed = fold_seed(&commitment.to_bytes());
            verify_width_fold_chain(
                &ring,
                &commitment.rows,
                &col.u_j,
                &blocks,
                2,
                &psi,
                BYTE_GATE,
                seed,
                &col.fold,
                transcript,
            )
            .map_err(|e| SalsaResponseError::Ajtai(format!("col {j} fold: {e}")))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod bound_tests {
    use super::*;
    use crate::pcs::GroupedOpening;
    use lattice_ring::{Modulus32, RingConfig};

    fn setup(log_n: u32, m_slots: usize) -> AkitaPcs {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams {
            ring,
            k: 2,
            m: m_slots,
            norm_bound: 1 << 20,
        };
        let pk = AjtaiPublicKey::from_seed(params, [23u8; 32]).ok().unwrap();
        AkitaPcs { pk }
    }

    fn mle(num_vars: usize, tag: &[u8]) -> DenseMle {
        let n = 1usize << num_vars;
        let bytes = Transcript::xof(b"akita-bound-mle", tag, n);
        let evals: Vec<Goldilocks> = bytes
            .iter()
            .take(n)
            .map(|&b| Goldilocks::from_u64(u64::from(b) % 1024))
            .collect();
        DenseMle::new(evals).ok().unwrap()
    }

    fn claims_for(f: &DenseMle, num_vars: usize, count: usize) -> Vec<GroupedOpening> {
        (0..count)
            .map(|i| {
                let point: Vec<Goldilocks> = (0..num_vars)
                    .map(|j| Goldilocks::from_u64(0x4000_0000 + (i * 16 + j) as u64))
                    .collect();
                let value = f.evaluate(&point).ok().unwrap();
                GroupedOpening { point, value }
            })
            .collect()
    }

    /// The closure end-to-end: prove → verify against the byte-packed
    /// commitment; the binding is the chain, not an opened witness.
    #[test]
    fn bound_salsa_honest_and_tamper() {
        let (pcs, _) = (setup(4, 64), ());
        let f = mle(4, b"bound-1");
        let com = pcs.commit_bytes(&f).ok().unwrap();
        let claims = claims_for(&f, 4, 3);
        let mut t = Transcript::new_default(b"akita-salsa-bound");
        let proof = pcs
            .prove_grouped_salsa_bound(&f, &claims, &mut t)
            .ok()
            .unwrap();
        let mut vt = Transcript::new_default(b"akita-salsa-bound");
        let vr = pcs.verify_grouped_salsa_bound(&com, &claims, &proof, &mut vt);
        assert!(vr.is_ok(), "the honest bound response must verify: {vr:?}");

        // THE CLOSURE TEST: a commitment to a DIFFERENT witness must
        // fail the chain's (W0) — the authenticated opening at the
        // challenge. This is the binding the open mode lacked.
        let g = mle(4, b"bound-2");
        let com_wrong = pcs.commit_bytes(&g).ok().unwrap();
        let mut vt2 = Transcript::new_default(b"akita-salsa-bound");
        assert!(pcs
            .verify_grouped_salsa_bound(&com_wrong, &claims, &proof, &mut vt2)
            .is_err());

        // Tampered f_term: the carrier terminal AND the fold's
        // functional thread reject.
        let mut bad = proof.clone();
        bad.f_term = bad.f_term.add(&Goldilocks::ONE);
        let mut vt3 = Transcript::new_default(b"akita-salsa-bound");
        assert!(pcs
            .verify_grouped_salsa_bound(&com, &claims, &bad, &mut vt3)
            .is_err());

        // Tampered D1 z_r: the norm reconstruction fails.
        let mut bad2 = proof.clone();
        bad2.z_r = bad2.z_r.add(&Goldilocks::ONE);
        let mut vt4 = Transcript::new_default(b"akita-salsa-bound");
        assert!(pcs
            .verify_grouped_salsa_bound(&com, &claims, &bad2, &mut vt4)
            .is_err());

        // Tampered fold stage material: the chain's (W0) rejects.
        let mut bad3 = proof.clone();
        if let Some(s0) = bad3.fold.stages.first_mut() {
            s0.p_images[0] ^= 0x01;
        }
        let mut vt5 = Transcript::new_default(b"akita-salsa-bound");
        assert!(pcs
            .verify_grouped_salsa_bound(&com, &claims, &bad3, &mut vt5)
            .is_err());

        // A tampered carrier round: the RLC sumcheck rejects.
        let mut bad4 = proof.clone();
        if let Some(r0) = bad4.sumcheck.rounds.first_mut() {
            if let Some(v) = r0.first_mut() {
                *v = v.add(&Goldilocks::ONE);
            }
        }
        let mut vt6 = Transcript::new_default(b"akita-salsa-bound");
        assert!(pcs
            .verify_grouped_salsa_bound(&com, &claims, &bad4, &mut vt6)
            .is_err());

        // A tampered claim value: the RLC carrier rejects.
        let mut claims_bad = claims.clone();
        claims_bad[0].value = claims_bad[0].value.add(&Goldilocks::ONE);
        let mut vt7 = Transcript::new_default(b"akita-salsa-bound");
        assert!(pcs
            .verify_grouped_salsa_bound(&com, &claims_bad, &proof, &mut vt7)
            .is_err());
    }

    /// The capacity law: the exact Lemma-4 boundary at the reference
    /// shape (ring dim 16, Q_32): 2,048 values per commitment.
    #[test]
    fn byte_capacity_is_the_lemma4_boundary() {
        let ring = RingConfig::new(Modulus32::Q_32, 4).ok().unwrap();
        assert_eq!(byte_capacity(&ring), 2048);
        // The gate itself: 2048 values fit, 4096 do not.
        let m_fit = 2048 * 8 / ring.n();
        let m_over = 4096 * 8 / ring.n();
        assert!(lattice_salsa::ring_norm::wraparound_gate(m_fit, &ring, BYTE_GATE).is_ok());
        assert!(lattice_salsa::ring_norm::wraparound_gate(m_over, &ring, BYTE_GATE).is_err());
        // The column planner: within capacity → 1 column; beyond → the
        // power-of-two split.
        assert_eq!(column_count_for(2048, byte_capacity(&ring)), 1);
        assert_eq!(column_count_for(2049, byte_capacity(&ring)), 2);
        assert_eq!(column_count_for(1 << 14, byte_capacity(&ring)), 8);
    }

    /// The r-column split BEYOND the per-commitment cap (2^12 values →
    /// 2 columns at ring dim 16): prove → verify with the μ-weighted
    /// functional decomposition + per-column D1/binding; tamper
    /// coverage on every layer.
    #[test]
    fn split_salsa_beyond_capacity_honest_and_tamper() {
        let pcs = setup(4, 4); // the split manages its own column keys
        let f = mle(12, b"split-1"); // 4,096 values > the 2,048 cap
        let claims = claims_for(&f, 12, 3);
        let mut t = Transcript::new_default(b"akita-salsa-split");
        let proof = pcs
            .prove_grouped_salsa_split(&f, &claims, &mut t)
            .ok()
            .unwrap();
        assert_eq!(
            proof.columns.len(),
            2,
            "the planner must split into 2 columns"
        );
        let mut vt = Transcript::new_default(b"akita-salsa-split");
        let vr = pcs.verify_grouped_salsa_split(12, &claims, &proof, &mut vt);
        assert!(vr.is_ok(), "the honest split response must verify: {vr:?}");

        // Tampered u_j: the μ-decomposition against f_term fails.
        let mut bad = proof.clone();
        bad.columns[0].u_j = bad.columns[0].u_j.add(&Goldilocks::ONE);
        let mut vt2 = Transcript::new_default(b"akita-salsa-split");
        assert!(pcs
            .verify_grouped_salsa_split(12, &claims, &bad, &mut vt2)
            .is_err());

        // A tampered column commitment: the chain's (W0) rejects (the
        // per-column authenticated opening).
        let mut bad2 = proof.clone();
        if let Some(b) = bad2.columns[0].commitment.first_mut() {
            *b ^= 0x01;
        }
        let mut vt3 = Transcript::new_default(b"akita-salsa-split");
        assert!(pcs
            .verify_grouped_salsa_split(12, &claims, &bad2, &mut vt3)
            .is_err());

        // Tampered D1 z_r (column 1): the norm reconstruction fails.
        let mut bad3 = proof.clone();
        bad3.columns[1].z_r = bad3.columns[1].z_r.add(&Goldilocks::ONE);
        let mut vt4 = Transcript::new_default(b"akita-salsa-split");
        assert!(pcs
            .verify_grouped_salsa_split(12, &claims, &bad3, &mut vt4)
            .is_err());

        // Tampered fold stage material (column 0): the chain rejects.
        let mut bad4 = proof.clone();
        if let Some(s0) = bad4.columns[0].fold.stages.first_mut() {
            s0.p_images[0] ^= 0x01;
        }
        let mut vt5 = Transcript::new_default(b"akita-salsa-split");
        assert!(pcs
            .verify_grouped_salsa_split(12, &claims, &bad4, &mut vt5)
            .is_err());

        // A tampered carrier round: the RLC sumcheck rejects.
        let mut bad5 = proof.clone();
        if let Some(r0) = bad5.sumcheck.rounds.first_mut() {
            if let Some(v) = r0.first_mut() {
                *v = v.add(&Goldilocks::ONE);
            }
        }
        let mut vt6 = Transcript::new_default(b"akita-salsa-split");
        assert!(pcs
            .verify_grouped_salsa_split(12, &claims, &bad5, &mut vt6)
            .is_err());
    }

    /// The degenerate split (values within capacity → r = 1): the split
    /// API reduces to the single-column bound path and verifies.
    #[test]
    fn split_degenerates_within_capacity() {
        let pcs = setup(4, 4);
        let f = mle(6, b"split-0"); // 64 values << the 2,048 cap
        let claims = claims_for(&f, 6, 2);
        let mut t = Transcript::new_default(b"akita-salsa-split-d");
        let proof = pcs
            .prove_grouped_salsa_split(&f, &claims, &mut t)
            .ok()
            .unwrap();
        assert_eq!(proof.columns.len(), 1);
        let mut vt = Transcript::new_default(b"akita-salsa-split-d");
        let vr = pcs.verify_grouped_salsa_split(6, &claims, &proof, &mut vt);
        assert!(vr.is_ok(), "the degenerate split must verify: {vr:?}");
    }
}
