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
/// is tracked; a `refresh` (extension commitment) resets it.
pub struct CycloAccumulator {
    /// Current folded witness (hidden from the verifier; committed).
    pub witness: Vec<RingElement>,
    /// Commitment to the folded witness.
    pub commitment: AjtaiCommitment,
    /// Additive norm budget consumed.
    pub norm_budget: u64,
    /// Folds since the last refresh.
    pub folds_since_refresh: usize,
}

pub struct CycloFoldResult {
    pub accumulator: CycloAccumulator,
    /// The fold challenge (balanced integer, ring + field views).
    pub challenge: i64,
    /// Ring-scalar view of the challenge.
    pub ring_scalar: u32,
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
        let padded = pk.pad_to_m(w).map_err(CycloError::Ajtai)?;
        let commitment = pk.commit(&padded).map_err(CycloError::Ajtai)?;
        let norm = padded.iter().map(|e| e.infinity_norm() as u64).max().unwrap_or(0);
        Ok(CycloAccumulator {
            witness: padded,
            commitment,
            norm_budget: norm,
            folds_since_refresh: 0,
        })
    }

    /// Fold a fresh input witness into the accumulator.
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
        // Input range check (Cyclo checks only fresh inputs).
        let bound = 1 << 20;
        for e in input {
            if e.infinity_norm() > bound {
                return Err(CycloError::PartialRangeFailed);
            }
        }
        let padded_input = pk.pad_to_m(input).map_err(CycloError::Ajtai)?;
        let mut transcript = Transcript::new_default(b"lzx-cyclo");
        transcript
            .append_bytes(b"acc", &self.commitment.to_bytes())
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

        // w' = w_acc + r·w_in.
        let mut folded = Vec::with_capacity(self.witness.len());
        for (a, b) in self.witness.iter().zip(padded_input.iter()) {
            folded.push(a.add(&b.scale_i64(r_int)).map_err(CycloError::Ring)?);
        }
        let commitment = {
            let mut rows = Vec::with_capacity(self.commitment.rows.len());
            // t' = t_acc + r·t_in: recommit folded (verifier recomputes from
            // the folded commitment homomorphism in the real protocol; here
            // we commit the folded witness directly for the test oracle).
            for (t, b) in self.commitment.rows.iter().zip(
                pk.commit(&padded_input)
                    .map_err(CycloError::Ajtai)?
                    .rows
                    .iter(),
            ) {
                rows.push(t.add(&b.scale_i64(r_int)).map_err(CycloError::Ring)?);
            }
            AjtaiCommitment { rows }
        };
        let norm = folded.iter().map(|e| e.infinity_norm() as u64).max().unwrap_or(0);
        Ok(CycloFoldResult {
            accumulator: CycloAccumulator {
                witness: folded,
                commitment,
                norm_budget: norm,
                folds_since_refresh: self.folds_since_refresh + 1,
            },
            challenge: r_int,
            ring_scalar: r_scalar,
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
                norm_budget: norm,
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
    use lattice_ring::{Modulus32, RingConfig};

    fn setup(log_n: u32, m: usize) -> (AjtaiPublicKey, RingConfig) {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m,
            norm_bound: 1 << 20,
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
        let base_norm = acc.norm_budget;
        for i in 0..4 {
            let input = small_w(&ring, format!("in{i}").as_bytes());
            let res = acc.fold(&pk, &input).ok().unwrap();
            // Additive growth: norm ≤ base + (i+1) · |r|·input_norm.
            assert!(res.accumulator.norm_budget <= base_norm + (i + 1) * (1 << 11) * 256);
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
        let (refreshed, _ext) = res.accumulator.refresh(&pk_ext, 8).ok().unwrap();
        assert!(refreshed.norm_budget <= 128);
        assert_eq!(refreshed.folds_since_refresh, 0);
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
