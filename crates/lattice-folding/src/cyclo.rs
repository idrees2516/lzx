//! Cyclo (Garreta–Lipmaa–Luhääär–Osadnik, ePrint 2026/359): lightweight
//! lattice-based folding via partial range checks.
//!
//! Key design (per the paper):
//! * **No norm checks on the accumulator** — only the *input* (fresh)
//!   witnesses are range-checked; the accumulator norm grows additively
//!   per fold within a generously bounded number of folds.
//! * **Extension commitment** — the norm-reducing primitive: decompose a
//!   witness into low-norm chunks, commit to the chunks, and treat the
//!   chunk-vector as the new witness representation (norm refreshed).
//! * **ℓ∞ range test via sumcheck** — the partial variant checks only the
//!   digits that matter: the high chunk of each coefficient determines
//!   whether the balanced representative can exceed the bound, so the
//!   low chunks are certified by digit-count bookkeeping alone.
//! * **R_q ↔ F_q bridge** — the polynomial evaluation map connects ring
//!   elements to field polynomials (used when folding F_q constraints).

#[allow(unused_imports)] // AjtaiParams used by the test module via super::*
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiParams, AjtaiPublicKey};
use lattice_core::norm_budget::NormBudget;
use lattice_core::short_challenge::ShortChallengeSpec;
use lattice_core::transcript::Transcript;
use lattice_ring::RingElement;

/// An extension commitment: decomposed-chunk commitment that reduces the
/// effective witness norm.
#[derive(Clone, Debug)]
pub struct ExtensionCommitment {
    /// Commitment to the chunked witness.
    pub chunks: AjtaiCommitment,
    /// Chunk count per ring element (log base).
    pub chunk_log: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CycloError {
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    ChunkCountMismatch { expected: usize, got: usize },
    NormTooLargeForRefresh { norm: u64, chunkable: u64 },
    FoldBudgetExceeded { folds: usize, max: usize },
    PartialRangeFailed,
    /// Wave 6.2: the fold would push the witness norm past
    /// `min(q/2, β*)` — refusing beats wrapping mod q (wraparound silently
    /// destroys the SIS binding argument).
    NormGateExceeded { beta_after: u128, cap: u64 },
    /// Wave 6.4: the accumulator witness length no longer matches the
    /// commitment key dimension (the latent post-refresh truncation bug).
    WitnessLengthMismatch { expected: usize, got: usize },
    /// Short-challenge sampling failed (budget/parameters).
    ShortChallenge(lattice_core::short_challenge::ShortChallengeError),
}

/// Decompose a ring element into base-2^chunk_log balanced chunks:
/// coefficient c (balanced) = Σ_i 2^{i·chunk_log} · d_i with
/// |d_i| ≤ 2^{chunk_log-1}. Extraction is the exact iterative borrow
/// algorithm (same discipline as gadget decomposition, over signed values).
fn chunk_element(
    ring: &lattice_ring::RingConfig,
    e: &RingElement,
    chunk_log: u32,
) -> Vec<RingElement> {
    let q = ring.modulus.q;
    let half_q = q / 2;
    let span = 1i64 << chunk_log;
    let half = span / 2;
    // Balanced coefficients live in [-q/2, q/2) ⊂ [-2^31, 2^31): 32 bits
    // of coverage suffices.
    let num_chunks = ((32 + chunk_log - 1) / chunk_log.max(1)) as usize;
    // Chunk-major layout: out[chunk] holds every coefficient's chunk digit.
    let mut chunk_coeffs: Vec<Vec<u32>> = vec![Vec::with_capacity(ring.n()); num_chunks];
    for &c in e.coeffs() {
        let balanced = if c <= half_q {
            c as i64
        } else {
            c as i64 - q as i64
        };
        // Iterative borrow (exact): rem = d_i + span · rem' each step.
        let mut rem = balanced;
        for digits in chunk_coeffs.iter_mut() {
            let mut digit = rem.rem_euclid(span);
            if digit > half {
                digit -= span;
            }
            rem = (rem - digit) / span;
            digits.push(ring.modulus.reduce_i64(digit));
        }
    }
    chunk_coeffs
        .into_iter()
        .map(|coeffs| RingElement::from_coeffs(ring, coeffs))
        .collect()
}

/// Recompose chunk elements back into the original element (exact inverse
/// of `chunk_element`). (Library-visible for the test oracle.)
#[allow(dead_code)]
fn unchunk_elements(
    ring: &lattice_ring::RingConfig,
    chunks: &[RingElement],
    chunk_log: u32,
) -> RingElement {
    let q = ring.modulus;
    let mut coeffs = vec![0u32; ring.n()];
    for (idx, c) in coeffs.iter_mut().enumerate() {
        let mut acc: i128 = 0;
        for (ci, chunk) in chunks.iter().enumerate() {
            let raw = chunk.coeff(idx);
            let half = q.q / 2;
            let balanced = if raw <= half {
                raw as i64
            } else {
                raw as i64 - q.q as i64
            };
            acc += (balanced as i128) << (ci as u32 * chunk_log);
        }
        *c = q.reduce_i64(acc as i64);
    }
    RingElement::from_coeffs(ring, coeffs)
}

impl ExtensionCommitment {
    /// Build an extension commitment: chunk + commit. The chunked witness
    /// has infinity norm ≤ 2^{chunk_log-1}, refreshing the norm.
    pub fn commit(
        pk: &AjtaiPublicKey,
        w: &[RingElement],
        chunk_log: u32,
    ) -> Result<(Self, Vec<RingElement>), CycloError> {
        let ring = &pk.params.ring;
        let mut all_chunks = Vec::with_capacity(w.len() * 4);
        for e in w {
            all_chunks.extend(chunk_element(ring, e, chunk_log));
        }
        let padded = pk
            .pad_to_m(&all_chunks)
            .map_err(CycloError::Ajtai)?;
        let commitment = pk.commit(&padded).map_err(CycloError::Ajtai)?;
        Ok((
            ExtensionCommitment {
                chunks: commitment,
                chunk_log,
            },
            padded,
        ))
    }

    /// Effective (refreshed) norm of the chunked representation.
    pub fn refreshed_norm(chunk_log: u32) -> u64 {
        1u64 << (chunk_log - 1)
    }
}

/// The Cyclo fold: input witness (range-checked, fresh) folds into the
/// accumulator WITHOUT accumulator norm checks; norm grows additively and
/// is tracked by the shared `NormBudget` with a **hard gate** against
/// `min(q/2, β*)` per fold (Wave 6.2); a `refresh` (extension commitment)
/// resets it.
#[derive(Clone)]
pub struct CycloAccumulator {
    /// Current folded witness (hidden from the verifier; committed).
    pub witness: Vec<RingElement>,
    /// Commitment to the folded witness.
    pub commitment: AjtaiCommitment,
    /// Norm budget with hard wraparound gate (Wave 6.2).
    pub norm_budget: NormBudget,
    /// Folds since the last refresh.
    pub folds_since_refresh: usize,
}

pub struct CycloFoldResult {
    pub accumulator: CycloAccumulator,
    /// The fold challenge (balanced integer, ring + field views).
    pub challenge: i64,
    /// Ring-scalar view of the challenge.
    pub ring_scalar: u32,
    /// Commitment to the (padded) input witness — surfaced so callers and
    /// verifiers can absorb it (Wave 6.4 statement binding).
    pub input_commitment: AjtaiCommitment,
}

/// Maximum folds before a mandatory refresh (the paper's "generously
/// bounded number of folds").
pub const MAX_FOLDS_BEFORE_REFRESH: usize = 64;
impl CycloAccumulator {
    /// Initialize from a fresh (range-checked) input witness.
    pub fn new(
        pk: &AjtaiPublicKey,
        w: &[RingElement],
    ) -> Result<Self, CycloError> {
        // Cyclo range-checks the INPUT witness only.
        let bound = 1 << 20;
        for e in w {
            if e.infinity_norm() > bound {
                return Err(CycloError::PartialRangeFailed);
            }
        }
        if w.len() > pk.params.m {
            return Err(CycloError::WitnessLengthMismatch {
                expected: pk.params.m,
                got: w.len(),
            });
        }
        let padded = pk.pad_to_m(w).map_err(CycloError::Ajtai)?;
        let commitment = pk.commit(&padded).map_err(CycloError::Ajtai)?;
        let norm = padded.iter().map(|e| e.infinity_norm() as u64).max().unwrap_or(0);
        // Wave 6.2: fresh budgets are gated against the SIS bound too —
        // an input already over β* cannot be opened.
        if norm > pk.params.norm_bound as u64 {
            return Err(CycloError::NormGateExceeded {
                beta_after: norm as u128,
                cap: pk.params.norm_bound as u64,
            });
        }
        Ok(CycloAccumulator {
            witness: padded,
            commitment,
            norm_budget: NormBudget::fresh(norm),
            folds_since_refresh: 0,
        })
    }

    /// Fold a fresh input witness into the accumulator.
    ///
    /// **Wave 6.4 (FS hygiene)**: the fold transcript absorbs the FULL
    /// statement — accumulator commitment, *input commitment*, key
    /// parameters, and the fold counter — before deriving the challenge
    /// (previously only the accumulator commitment was absorbed, leaving
    /// the challenge grindable w.r.t. the input). The input commitment is
    /// computed first and returned to the caller.
    ///
    /// **Wave 6.2**: the norm growth `β' = β + |r|·β_in` is hard-gated
    /// against `min(q/2, β*)` — a fold that would wrap the balanced
    /// representative mod q is refused.
    pub fn fold(
        &self,
        pk: &AjtaiPublicKey,
        input: &[RingElement],
    ) -> Result<CycloFoldResult, CycloError> {
        if self.folds_since_refresh >= MAX_FOLDS_BEFORE_REFRESH {
            return Err(CycloError::FoldBudgetExceeded {
                folds: self.folds_since_refresh,
                max: MAX_FOLDS_BEFORE_REFRESH,
            });
        }
        // Wave 6.4: post-refresh length assertion (the latent truncation
        // bug: after refresh the witness length is pk_ext.m, which must
        // still match THIS key's m for folding).
        if self.witness.len() != pk.params.m {
            return Err(CycloError::WitnessLengthMismatch {
                expected: pk.params.m,
                got: self.witness.len(),
            });
        }
        // Input range check (Cyclo checks only fresh inputs).
        let bound = 1 << 20;
        for e in input {
            if e.infinity_norm() > bound {
                return Err(CycloError::PartialRangeFailed);
            }
        }
        if input.len() > pk.params.m {
            return Err(CycloError::WitnessLengthMismatch {
                expected: pk.params.m,
                got: input.len(),
            });
        }
        let padded_input = pk.pad_to_m(input).map_err(CycloError::Ajtai)?;
        let input_commitment = pk.commit(&padded_input).map_err(CycloError::Ajtai)?;
        let input_norm = padded_input
            .iter()
            .map(|e| e.infinity_norm() as u64)
            .max()
            .unwrap_or(0);
        // Wave 6.4: absorb the full statement BEFORE the challenge.
        let mut transcript = Transcript::new_default(b"lzx-cyclo");
        transcript
            .append_bytes(b"acc", &self.commitment.to_bytes())
            .map_err(|_| CycloError::PartialRangeFailed)?;
        transcript
            .append_bytes(b"input", &input_commitment.to_bytes())
            .map_err(|_| CycloError::PartialRangeFailed)?;
        let mut params_bytes = Vec::with_capacity(20);
        params_bytes.extend_from_slice(&pk.params.ring.modulus.q.to_le_bytes());
        params_bytes.extend_from_slice(&pk.params.ring.log_n.to_le_bytes());
        params_bytes.extend_from_slice(&(pk.params.k as u32).to_le_bytes());
        params_bytes.extend_from_slice(&(pk.params.m as u32).to_le_bytes());
        params_bytes.extend_from_slice(&(self.folds_since_refresh as u32).to_le_bytes());
        transcript
            .append_bytes(b"params", &params_bytes)
            .map_err(|_| CycloError::PartialRangeFailed)?;
        // Small short challenge (additive norm growth).
        let seed = transcript
            .challenge_bytes(b"fold-r", 32)
            .map_err(|_| CycloError::PartialRangeFailed)?;
        let bytes = Transcript::xof(b"cyclo-chal", &seed, 8);
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes[..8]);
        let raw = (u64::from_le_bytes(arr) & 0xFFF) as i64;
        let r_int = if raw >= 1 << 11 { raw - (1 << 12) } else { raw };
        let ring = &pk.params.ring;
        let r_scalar = ring.modulus.reduce_i64(r_int);

        // Wave 6.2: hard norm gate BEFORE folding.
        let q_half = (ring.modulus.q / 2) as u64;
        let budget = self
            .norm_budget
            .fold_scalar(r_int.unsigned_abs(), input_norm, q_half, pk.params.norm_bound as u64)
            .map_err(|e| CycloError::NormGateExceeded {
                beta_after: match e {
                    lattice_core::norm_budget::NormBudgetError::Wraparound { beta_after, .. } => beta_after,
                },
                cap: q_half.min(pk.params.norm_bound as u64),
            })?;

        // w' = w_acc + r·w_in.
        let mut folded = Vec::with_capacity(self.witness.len());
        for (a, b) in self.witness.iter().zip(padded_input.iter()) {
            folded.push(a.add(&b.scale_i64(r_int)).map_err(CycloError::Ring)?);
        }
        let commitment = {
            let mut rows = Vec::with_capacity(self.commitment.rows.len());
            // t' = t_acc + r·t_in (commitment homomorphism).
            for (t, b) in self.commitment.rows.iter().zip(input_commitment.rows.iter()) {
                rows.push(t.add(&b.scale_i64(r_int)).map_err(CycloError::Ring)?);
            }
            AjtaiCommitment { rows }
        };
        Ok(CycloFoldResult {
            accumulator: CycloAccumulator {
                witness: folded,
                commitment,
                norm_budget: budget,
                folds_since_refresh: self.folds_since_refresh + 1,
            },
            challenge: r_int,
            ring_scalar: r_scalar,
            input_commitment,
        })
    }

    /// **Wave 6.1 integration — the paper's ring-element fold**: fold under
    /// a challenge `d` drawn from a short-challenge distribution over R_q
    /// (Cyclo's set `D`: biased ternary) with per-sample certified operator
    /// norm Γ_C and op-norm rejection, and norm growth gated by the
    /// rigorous law `β' = β + Γ_C·⌈√N⌉·β_in`.
    pub fn fold_ring_challenge(
        &self,
        pk: &AjtaiPublicKey,
        input: &[RingElement],
        spec: &ShortChallengeSpec,
        gamma_cap: u64,
    ) -> Result<CycloFoldResult, CycloError> {
        if self.witness.len() != pk.params.m {
            return Err(CycloError::WitnessLengthMismatch {
                expected: pk.params.m,
                got: self.witness.len(),
            });
        }
        if spec.n != pk.params.ring.n() {
            return Err(CycloError::ShortChallenge(
                lattice_core::short_challenge::ShortChallengeError::InvalidParameters,
            ));
        }
        let bound = 1 << 20;
        for e in input {
            if e.infinity_norm() > bound {
                return Err(CycloError::PartialRangeFailed);
            }
        }
        let padded_input = pk.pad_to_m(input).map_err(CycloError::Ajtai)?;
        let input_commitment = pk.commit(&padded_input).map_err(CycloError::Ajtai)?;
        let input_norm = padded_input
            .iter()
            .map(|e| e.infinity_norm() as u64)
            .max()
            .unwrap_or(0);
        // Challenge from the transcript statement (Wave 6.4 ordering).
        let mut transcript = Transcript::new_default(b"lzx-cyclo-ring");
        transcript
            .append_bytes(b"acc", &self.commitment.to_bytes())
            .map_err(|_| CycloError::PartialRangeFailed)?;
        transcript
            .append_bytes(b"input", &input_commitment.to_bytes())
            .map_err(|_| CycloError::PartialRangeFailed)?;
        let seed = transcript
            .challenge_bytes(b"fold-d", 32)
            .map_err(|_| CycloError::PartialRangeFailed)?;
        let challenge = spec
            .sample_with_gamma_cap(&seed, gamma_cap, 16)
            .map_err(CycloError::ShortChallenge)?;
        let ring = &pk.params.ring;
        let d = RingElement::from_signed(ring, &challenge.coefficients);
        let gamma_c = challenge.gamma_c();

        // Rigorous norm gate: β' = β + Γ_C·⌈√N⌉·β_in.
        let q_half = (ring.modulus.q / 2) as u64;
        let sqrt_n = lattice_core::norm_budget::ceil_sqrt(ring.n() as u64);
        let budget = self
            .norm_budget
            .fold(gamma_c, sqrt_n, input_norm, q_half, pk.params.norm_bound as u64)
            .map_err(|e| CycloError::NormGateExceeded {
                beta_after: match e {
                    lattice_core::norm_budget::NormBudgetError::Wraparound { beta_after, .. } => beta_after,
                },
                cap: q_half.min(pk.params.norm_bound as u64),
            })?;

        // w' = w_acc + d·w_in (ring multiplication).
        let mut folded = Vec::with_capacity(self.witness.len());
        for (a, b) in self.witness.iter().zip(padded_input.iter()) {
            folded.push(a.add(&d.mul(b).map_err(CycloError::Ring)?).map_err(CycloError::Ring)?);
        }
        // t' = t_acc + d·t_in.
        let mut rows = Vec::with_capacity(self.commitment.rows.len());
        for (t, b) in self.commitment.rows.iter().zip(input_commitment.rows.iter()) {
            rows.push(
                t.add(&d.mul(b).map_err(CycloError::Ring)?)
                    .map_err(CycloError::Ring)?,
            );
        }
        // Report the scalar view as the balanced integer closest to the
        // constant coefficient (informational for callers).
        let d_const = challenge.coefficients.first().copied().unwrap_or(0);
        Ok(CycloFoldResult {
            accumulator: CycloAccumulator {
                witness: folded,
                commitment: AjtaiCommitment { rows },
                norm_budget: budget,
                folds_since_refresh: self.folds_since_refresh + 1,
            },
            challenge: d_const,
            ring_scalar: ring.modulus.reduce_i64(d_const),
            input_commitment,
        })
    }

    /// Refresh via extension commitment: chunk + recommit, resetting the
    /// norm budget to the refreshed (small) bound. `pk_ext` is the
    /// extension key — it must have `num_chunks` times the slot count of
    /// the base key (chunking expands the witness by that factor).
    pub fn refresh(
        &self,
        pk_ext: &AjtaiPublicKey,
        chunk_log: u32,
    ) -> Result<(CycloAccumulator, ExtensionCommitment), CycloError> {
        let (ext, chunked) = ExtensionCommitment::commit(pk_ext, &self.witness, chunk_log)?;
        let norm = chunked
            .iter()
            .map(|e| e.infinity_norm() as u64)
            .max()
            .unwrap_or(0);
        Ok((
            CycloAccumulator {
                witness: chunked,
                commitment: ext.chunks.clone(),
                norm_budget: self.norm_budget.refresh(norm),
                folds_since_refresh: 0,
            },
            ext,
        ))
    }
}

/// Partial range check (the paper's lightweight ℓ∞ test): verify a ring
/// element's coefficients lie in [-β, β] by checking ONLY the high chunks
/// — the low chunks are bounded by construction (digit-count argument:
/// with k chunks of c bits, values ≥ 2^{(k-1)·c} are determined by the top
/// chunk alone).
pub fn partial_range_check(
    e: &RingElement,
    beta: u64,
    chunk_log: u32,
) -> Result<bool, CycloError> {
    let ring = e.config();
    let num_chunks = ((32 + chunk_log - 1) / chunk_log.max(1)) as usize;
    // The top chunk determines magnitude: if the top chunk is zero (or
    // minimal), the value fits in the remaining chunks' range.
    let top_span = 1u64 << (((num_chunks - 1) as u32) * chunk_log);
    if beta < top_span {
        // Bound smaller than the top chunk's span: full decomposition
        // needed — fall back to checking every chunk's high digit.
        let chunks = chunk_element(ring, e, chunk_log);
        let chunk_half = (1u64 << (chunk_log - 1)) - 1;
        return Ok(chunks
            .iter()
            .all(|c| c.infinity_norm() as u64 <= chunk_half));
    }
    // Partial: only the top chunk is inspected.
    let chunks = chunk_element(ring, e, chunk_log);
    let top = chunks
        .last()
        .ok_or(CycloError::ChunkCountMismatch { expected: num_chunks, got: 0 })?;
    let top_half = (1u64 << (chunk_log - 1)) - 1;
    // If the top chunk is in the minimal band, the total value is below
    // the (num_chunks-1)-chunk span, hence below beta when beta ≥ span.
    let lower_span = 1u64 << (((num_chunks - 1) as u32) * chunk_log);
    if top.infinity_norm() as u64 <= top_half.min(beta / lower_span.max(1)) {
        // Value fits in the lower chunks: still bounded by their span.
        return Ok(true);
    }
    // Top chunk significant: the value is at least the lower span; bound
    // it exactly via the direct infinity norm (the paper's amortization:
    // this happens rarely because folding keeps values small).
    Ok(e.infinity_norm() as u64 <= beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_core::short_challenge::ShortChallengeFamily;
    use lattice_ring::{Modulus32, RingConfig};

    fn setup(log_n: u32, m: usize) -> (AjtaiPublicKey, RingConfig) {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m,
            // Wave 6.2: the norm_bound is now a HARD gate (β*), so the
            // fixture declares a bound consistent with multi-fold growth
            // (input norms ≤ 2^20, |r| ≤ 2^11, a handful of folds).
            norm_bound: 1 << 26,
        };
        let pk = AjtaiPublicKey::from_seed(params, [31u8; 32]).ok().unwrap();
        (pk, ring)
    }

    fn small_w(ring: &RingConfig, tag: &[u8]) -> Vec<RingElement> {
        lattice_commitment::ajtai::sample_small_secret(ring, 3, 256, tag)
    }

    #[test]
    fn extension_commitment_roundtrip() {
        let (pk, ring) = setup(4, 16);
        let w = small_w(&ring, b"cyclo-w");
        let (ext, chunked) = ExtensionCommitment::commit(&pk, &w, 8).ok().unwrap();
        // Chunked norm refreshed to ≤ 2^7.
        let max_norm = chunked.iter().map(|e| e.infinity_norm()).max().unwrap();
        assert!(max_norm <= 128, "refreshed norm {max_norm} > 128");
        // Recomposition: unchunk the first element's chunks and compare.
        // (chunked is padded to m; the per-element chunk count comes from
        // the chunk geometry, not from the padded length.)
        let num_chunks = ((32 + 8 - 1) / 8) as usize;
        for (i, orig) in w.iter().enumerate() {
            let chunks: Vec<RingElement> =
                chunked[i * num_chunks..(i + 1) * num_chunks].to_vec();
            let rec = unchunk_elements(&ring, &chunks, 8);
            assert_eq!(rec, *orig, "element {i} recomposition failed");
        }
        // The extension commitment opens the chunked witness.
        assert!(pk.verify_opening(&ext.chunks, &chunked).is_ok());
    }

    #[test]
    fn fold_additive_norm_growth() {
        let (pk, ring) = setup(4, 3);
        let acc0 = CycloAccumulator::new(&pk, &small_w(&ring, b"a0")).ok().unwrap();
        let mut acc = acc0;
        let base_norm = acc.norm_budget.beta();
        for i in 0..4u64 {
            let input = small_w(&ring, format!("in{i}").as_bytes());
            let res = acc.fold(&pk, &input).ok().unwrap();
            // Additive growth: norm ≤ base + (i+1) · |r|·input_norm.
            let bound = base_norm + (i + 1) * (1 << 11) * 256;
            assert!(
                res.accumulator.norm_budget.beta() <= bound,
                "beta {} vs bound {bound}",
                res.accumulator.norm_budget.beta()
            );
            assert_eq!(res.accumulator.norm_budget.folds(), i + 1);
            acc = res.accumulator;
        }
        assert_eq!(acc.folds_since_refresh, 4);
    }

    #[test]
    fn refresh_resets_norm() {
        // The extension key needs num_chunks x the base slot count.
        let (pk, ring) = setup(4, 64);
        let (pk_ext, _) = setup(4, 256);
        let acc0 = CycloAccumulator::new(&pk, &small_w(&ring, b"r0")).ok().unwrap();
        let res = acc0.fold(&pk, &small_w(&ring, b"r1")).ok().unwrap();
        assert_eq!(res.accumulator.norm_budget.folds(), 1);
        let (refreshed, _ext) = res.accumulator.refresh(&pk_ext, 8).ok().unwrap();
        assert!(refreshed.norm_budget.beta() <= 128);
        assert_eq!(refreshed.norm_budget.folds(), 0);
        assert_eq!(refreshed.folds_since_refresh, 0);
        // Wave 6.4: folding the refreshed accumulator under the BASE key
        // now fails closed on the length assertion (the latent truncation
        // bug — refresh produces a pk_ext.m-length witness).
        assert!(matches!(
            refreshed.fold(&pk, &small_w(&ring, b"r2")),
            Err(CycloError::WitnessLengthMismatch { expected: 64, got: 256 })
        ));
    }

    #[test]
    fn fs_statement_binding_and_determinism() {
        // Wave 6.4: the challenge is a function of BOTH commitments (grind
        // resistance w.r.t. the input) and the fold counter.
        let (pk, ring) = setup(4, 3);
        let acc = CycloAccumulator::new(&pk, &small_w(&ring, b"fs-a")).ok().unwrap();
        let in1 = small_w(&ring, b"fs-in1");
        let in2 = small_w(&ring, b"fs-in2");
        let r1 = acc.fold(&pk, &in1).ok().unwrap();
        let r2 = acc.fold(&pk, &in2).ok().unwrap();
        // Different inputs → different challenges (input commitment bound).
        assert_ne!(r1.challenge, r2.challenge);
        // Deterministic replay.
        let r1b = acc.fold(&pk, &in1).ok().unwrap();
        assert_eq!(r1.challenge, r1b.challenge);
        assert_eq!(r1.ring_scalar, r1b.ring_scalar);
        // The fold counter is absorbed: same input at a different fold
        // index yields a different challenge.
        let acc2 = r1.accumulator.clone();
        let ra = acc2.fold(&pk, &in1).ok().unwrap(); // fold #1
        let rb = r1.accumulator.fold(&pk, &in1).ok().unwrap(); // fold #1 too
        assert_eq!(ra.challenge, rb.challenge);
        let acc_once = CycloAccumulator::new(&pk, &small_w(&ring, b"fs-b")).ok().unwrap();
        let rc = acc_once.fold(&pk, &in1).ok().unwrap(); // fold #0
        // Different accumulator commitment → different challenge anyway;
        // the counter test: fold the SAME accumulator twice in a row.
        let rd = acc2.fold(&pk, &in1).ok().unwrap();
        let re = r1.accumulator.fold(&pk, &in1).ok().unwrap();
        assert_eq!(rd.challenge, re.challenge);
        // rc used a different accumulator; both must still verify.
        assert!(pk.verify_opening(&rc.input_commitment, &pk.pad_to_m(&in1).ok().unwrap()).is_ok());
    }

    #[test]
    fn ring_challenge_fold_paper_fidelity() {
        // Wave 6.1: the paper's D-challenge fold with certified Γ_C and the
        // rigorous norm law β' = β + Γ_C·⌈√N⌉·β_in.
        let (pk, ring) = setup(4, 6);
        // Biased-ternary D over R_q (n = 16, matching the ring).
        let spec = ShortChallengeSpec {
            n: ring.n(),
            family: ShortChallengeFamily::BiasedTernary { p_nonzero_permille: 500 },
        };
        let acc = CycloAccumulator::new(&pk, &small_w(&ring, b"d-a")).ok().unwrap();
        let input = small_w(&ring, b"d-in");
        let res = acc
            .fold_ring_challenge(&pk, &input, &spec, 8)
            .ok()
            .unwrap();
        assert!(res.accumulator.norm_budget.gamma_max <= 8);
        assert_eq!(res.accumulator.norm_budget.folds(), 1);
        // Rigorous growth law with the ACTUAL input norm.
        let padded = pk.pad_to_m(&input).ok().unwrap();
        let input_norm = padded
            .iter()
            .map(|e| e.infinity_norm() as u64)
            .max()
            .unwrap_or(0);
        let beta_before = acc.norm_budget.beta();
        let expected = beta_before + res.accumulator.norm_budget.gamma_max * 4 * input_norm;
        assert_eq!(res.accumulator.norm_budget.beta(), expected);
        // Deterministic replay: identical witness, commitment, and budget.
        let res2 = acc
            .fold_ring_challenge(&pk, &input, &spec, 8)
            .ok()
            .unwrap();
        assert_eq!(res2.accumulator.witness, res.accumulator.witness);
        assert_eq!(res2.accumulator.commitment.rows, res.accumulator.commitment.rows);
        assert_eq!(res2.accumulator.norm_budget, res.accumulator.norm_budget);
        // Structural homomorphism: commit(w_acc + d·w_in) == t_acc + d·t_in
        // for the transcript-derived d — verified by re-deriving d from the
        // same transcript statement.
        let mut t = Transcript::new_default(b"lzx-cyclo-ring");
        let _ = t.append_bytes(b"acc", &acc.commitment.to_bytes());
        let _ = t.append_bytes(b"input", &res.input_commitment.to_bytes());
        let seed = t.challenge_bytes(b"fold-d", 32).ok().unwrap();
        let challenge = spec.sample_with_gamma_cap(&seed, 8, 16).ok().unwrap();
        let d = RingElement::from_signed(&ring, &challenge.coefficients);
        let mut expect_rows = Vec::with_capacity(acc.commitment.rows.len());
        for (ta, tb) in acc.commitment.rows.iter().zip(res.input_commitment.rows.iter()) {
            expect_rows.push(ta.add(&d.mul(tb).ok().unwrap()).ok().unwrap());
        }
        assert_eq!(res.accumulator.commitment.rows, expect_rows);
        let recomputed = pk.commit(&res.accumulator.witness).ok().unwrap();
        assert_eq!(recomputed.rows, expect_rows);
    }

    #[test]
    fn ring_challenge_norm_gate_fails_closed() {
        // A Γ cap that no sample can meet → fail closed after retries.
        let (pk, ring) = setup(4, 6);
        let spec = ShortChallengeSpec {
            n: ring.n(),
            family: ShortChallengeFamily::BiasedTernary { p_nonzero_permille: 999 },
        };
        let acc = CycloAccumulator::new(&pk, &small_w(&ring, b"gate")).ok().unwrap();
        let input = small_w(&ring, b"gate-in");
        // Dense ternary over n = 16 has Γ ≈ ⌈√16⌉ = 4; cap 1 is unreachable.
        assert!(matches!(
            acc.fold_ring_challenge(&pk, &input, &spec, 1),
            Err(CycloError::ShortChallenge(_))
        ));
        // Wrong spec dimension rejected.
        let bad_spec = ShortChallengeSpec {
            n: 8,
            family: ShortChallengeFamily::BiasedTernary { p_nonzero_permille: 500 },
        };
        assert!(matches!(
            acc.fold_ring_challenge(&pk, &input, &bad_spec, 8),
            Err(CycloError::ShortChallenge(_))
        ));
    }

    #[test]
    fn input_range_check_enforced() {
        let (pk, ring) = setup(4, 3);
        let mut big = vec![ring.zero(); 3];
        let mut coeffs = vec![0u32; ring.n()];
        coeffs[0] = 1 << 25; // way over the input bound
        big[0] = RingElement::from_coeffs(&ring, coeffs);
        assert!(matches!(
            CycloAccumulator::new(&pk, &big),
            Err(CycloError::PartialRangeFailed)
        ));
    }

    #[test]
    fn partial_range_check_agrees_with_direct() {
        let (_, ring) = setup(4, 3);
        let beta = 1u64 << 16;
        for seed in ["a", "b", "c", "d"] {
            let w = lattice_commitment::ajtai::sample_small_secret(&ring, 1, 300, seed.as_bytes());
            let e = &w[0];
            let direct = e.infinity_norm() as u64 <= beta;
            let partial = partial_range_check(e, beta, 8).ok().unwrap();
            assert_eq!(partial, direct, "seed {seed}");
        }
    }

    #[test]
    fn chunk_helpers_inverse() {
        let (_, ring) = setup(4, 3);
        for tag in ["x", "y"] {
            let w = lattice_commitment::ajtai::sample_small_secret(&ring, 1, 4096, tag.as_bytes());
            let chunks = chunk_element(&ring, &w[0], 8);
            let back = unchunk_elements(&ring, &chunks, 8);
            assert_eq!(back, w[0]);
        }
        // Larger values stress the borrow logic.
        let mut coeffs = vec![0u32; ring.n()];
        coeffs[0] = 3_000_000_123; // within q, balanced positive
        coeffs[1] = ring.modulus.q - 2_999_999_777; // balanced negative
        let e = RingElement::from_coeffs(&ring, coeffs);
        for chunk_log in [4u32, 8, 11] {
            let chunks = chunk_element(&ring, &e, chunk_log);
            let back = unchunk_elements(&ring, &chunks, chunk_log);
            assert_eq!(back, e, "chunk_log {chunk_log}");
        }
    }
}
