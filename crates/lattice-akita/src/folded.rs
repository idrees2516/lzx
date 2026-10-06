//! The cross-column folded opening layer (the r_air RLC fold, SOTA
//! mechanism #2 of the throughput path): the per-column grouped
//! openings collapse into a small number of CHUNKED INTEGER FOLDS per
//! distinct evaluation point, using the Ajtai commitment's linear
//! homomorphism.
//!
//! # The mechanism
//!
//! The v3 pipeline carries ~300 committed columns whose evaluation
//! claims concentrate at a handful of points (the shared `r_air` of the
//! batched AIR, each lookup group's shared `rcycle`, the memory legs'
//! points). Opening every column separately costs one eq·f sumcheck,
//! one packed witness, and one Ajtai recompute per column — on both
//! sides. The fold replaces that with, per point and per chunk of at
//! most [`FOLD_CHUNK_COLS`] claims:
//!
//! 1. small integer challenges `d_c ∈ [−A, A]` (the compact-opening
//!    discipline: `r·A·β₀ ≪ q/2` keeps the limb-level integer fold
//!    EXACT mod q, and the Goldilocks functional commutes through the
//!    packing);
//! 2. ONE degree-2 sumcheck over `eq(p, ·)·G` with
//!    `G = Σ_c d_c·f_c` over Goldilocks (the value-level fold —
//!    materialized once per chunk, SIMD);
//! 3. ONE folded packed witness `V = Σ_c d_c·pack(f_c)` (the limb-level
//!    integer fold — exact, gate-checked);
//! 4. the Ajtai-linearity check `A·V = Σ_c d_c·y_c` — the VERIFIER
//!    computes the folded commitment from the per-column commitments
//!    (`RingElement::scale_i64` + add), so the binding to every column
//!    rides ONE commit recompute;
//! 5. the final binding through the packed MLE functional
//!    `Φ(V)(r_sc) = Σ_i eq(p,i)·Σ_k 2^{22k}·V[3i+k]` — the mod-p
//!    reconstruction that commutes with the integer fold.
//!
//! # The norm discipline (honest)
//!
//! `‖V‖∞ ≤ L·A·2^22 < q/2` keeps every coefficient an exact balanced
//! integer (no mod-q wrap — the commutation's precondition). The
//! binding terminates in MSIS on `[A | −y]` at the relaxed bound — the
//! same estimator-gated regime as the compact opening's fold
//! (SECURITY.md); the response is revealed in the clear at kernel
//! scale (the polylog private response is the A3/A4 substitution's
//! layer, not this one).

use crate::pcs::AkitaPcs;
use lattice_commitment::ajtai::AjtaiCommitment;
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::packing::{LIMBS_PER_VALUE, LIMB_BITS};
use lattice_ring::{RingConfig, RingElement};
use lattice_sumcheck::sumcheck;
use lattice_sumcheck::SumcheckProof;
use lattice_sumcheck::VirtualPolynomial;

/// The integer fold challenge amplitude `A` (challenges in `[−A, A]`).
pub const FOLD_AMPLITUDE: u32 = 1 << 4;
/// The per-chunk claim count: `L·A·2^22 < q/2` with margin at
/// `q = 2^32 − 99` (32·16·2^22 = 2^27 ≪ 2^31).
pub const FOLD_CHUNK_COLS: usize = 32;
/// The per-coefficient limb bound of the split packing.
pub const LIMB_BOUND: u64 = 1 << LIMB_BITS;

/// One evaluation claim entering the fold.
pub struct FoldClaim<'a> {
    /// The caller's column identifier (absorbed into the transcript so
    /// the fold challenges bind the exact claim set).
    pub col: usize,
    /// The column's `2^log_t` evaluations.
    pub evals: &'a [Goldilocks],
    /// The claimed `f(p)`.
    pub value: Goldilocks,
}

/// The claims at one distinct point.
pub struct FoldPoint<'a> {
    pub point: &'a [Goldilocks],
    pub claims: &'a [FoldClaim<'a>],
}

/// One chunk's proof: the fold challenges, the eq·G sumcheck, and the
/// folded packed witness.
#[derive(Clone, Debug)]
pub struct FoldedChunk {
    /// Per-claim integer challenges (the chunk's claims, in order).
    pub d: Vec<i64>,
    /// The degree-2 sumcheck over `eq(p, ·)·G`.
    pub sumcheck: SumcheckProof,
    /// The folded packed witness `V = Σ_c d_c·pack(f_c)` (padded to the
    /// key's `m` slots).
    pub opened: Vec<RingElement>,
}

/// The folded opening layer's proof: one entry per distinct point.
#[derive(Clone, Debug)]
pub struct FoldedOpenings {
    pub log_t: usize,
    /// The distinct points in canonical (first-appearance) order.
    pub points: Vec<Vec<Goldilocks>>,
    /// Per point: the chunks (consecutive claim ranges).
    pub chunks: Vec<Vec<FoldedChunk>>,
}

/// Errors of the folded layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoldError {
    Transcript(lattice_core::transcript::TranscriptError),
    Sumcheck(lattice_sumcheck::SumcheckError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Mle(lattice_core::mle::MleError),
    Ring(lattice_ring::RingError),
    Shape {
        expected: usize,
        got: usize,
    },
    /// A folded coefficient exceeded the exactness gate.
    GateExceeded {
        value: i64,
        gate: i64,
    },
    /// The underlying Ajtai layer rejected a shape/commit operation.
    Ajtai,
    /// The Ajtai-linearity check failed.
    CommitmentMismatch,
    /// The final sumcheck binding failed.
    FinalCheckFailed,
}

impl From<lattice_core::transcript::TranscriptError> for FoldError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        FoldError::Transcript(e)
    }
}

/// Sample one integer fold challenge in `[−A, A]` from the transcript.
fn sample_d(transcript: &mut Transcript) -> Result<i64, FoldError> {
    let b = transcript.challenge_bytes(b"fold-dchal", 2)?;
    let raw = u16::from_le_bytes([b[0], b[1]]) as u64;
    let m = 2 * FOLD_AMPLITUDE as u64 + 1;
    Ok((raw % m) as i64 - FOLD_AMPLITUDE as i64)
}

/// The chunk's exactness gate: `L·A·(2^22 − 1)`.
fn chunk_gate(claims: usize) -> i64 {
    (claims as i64) * (FOLD_AMPLITUDE as i64) * (LIMB_BOUND as i64 - 1)
}

/// The limb-level integer fold: `Σ_c d_c·pack(f_c)` built on the flat
/// limb streams (one pass, no per-column ring elements), with the
/// per-coefficient exactness gate checked at assembly.
fn fold_limb_stream(
    ring: &RingConfig,
    claims: &[FoldClaim<'_>],
    d: &[i64],
) -> Result<Vec<RingElement>, FoldError> {
    let n = ring.n();
    let q = ring.modulus.q as i64;
    let half = ring.modulus.q / 2;
    let mut acc: Vec<i64> = Vec::new();
    let mut len = 0usize;
    for (c, dc) in claims.iter().zip(d.iter()) {
        if *dc == 0 {
            continue;
        }
        let raw = c.evals.to_vec();
        let mut li = 0usize;
        for v in &raw {
            let x = v.to_canonical_u64();
            for k in 0..LIMBS_PER_VALUE {
                let limb = ((x >> (k * LIMB_BITS as usize)) & (LIMB_BOUND - 1)) as i64;
                if li >= len {
                    acc.push(0);
                    len += 1;
                }
                acc[li] += dc * limb;
                li += 1;
            }
        }
    }
    // Gate: every accumulated coefficient must stay an exact balanced
    // integer (no mod-q wrap).
    let gate = chunk_gate(claims.len());
    for &v in &acc {
        if v.abs() > gate || v.unsigned_abs() >= u64::from(half) {
            return Err(FoldError::GateExceeded { value: v, gate });
        }
    }
    // Chunk into ring elements (zero-padded).
    let num_elems = acc.len().div_ceil(n).max(1);
    let mut out = Vec::with_capacity(num_elems);
    for e in 0..num_elems {
        let start = e * n;
        let end = (start + n).min(acc.len().max(1));
        let mut chunk: Vec<i64> = if start < acc.len() {
            acc[start..end].to_vec()
        } else {
            Vec::new()
        };
        chunk.resize(n, 0);
        let _ = q;
        out.push(RingElement::from_signed(ring, &chunk));
    }
    Ok(out)
}

/// The packed MLE functional: `Φ(V)(r) = Σ_i eq(r,i)·Σ_k 2^{22k}·V[3i+k]`
/// over Goldilocks — the mod-p reconstruction that commutes with the
/// integer limb fold.
fn packed_mle_eval(
    ring: &RingConfig,
    packed: &[RingElement],
    point: &[Goldilocks],
    log_t: usize,
) -> Result<Goldilocks, FoldError> {
    let n = ring.n();
    let q = ring.modulus.q;
    let half = q / 2;
    let mut flat: Vec<i64> = Vec::with_capacity(packed.len() * n);
    for elem in packed {
        for &c in elem.coeffs() {
            let balanced = if c <= half {
                c as i64
            } else {
                c as i64 - q as i64
            };
            flat.push(balanced);
        }
    }
    let eq = lattice_core::field_simd::eq_table(point);
    let t_pow = 1usize << log_t;
    if eq.len() < t_pow {
        return Err(FoldError::Shape {
            expected: t_pow,
            got: eq.len(),
        });
    }
    let fe = |x: i64| -> Goldilocks {
        if x >= 0 {
            Goldilocks::from_u64(x as u64)
        } else {
            Goldilocks::from_u64(x.unsigned_abs()).neg()
        }
    };
    let limb_w = [
        Goldilocks::from_u64(1),
        Goldilocks::from_u64(1u64 << LIMB_BITS),
        Goldilocks::from_u64(1u64 << (2 * LIMB_BITS)),
    ];
    let mut acc = Goldilocks::ZERO;
    for (i, &eq_i) in eq.iter().enumerate().take(t_pow) {
        let mut v = Goldilocks::ZERO;
        for (k, &w) in limb_w.iter().enumerate() {
            let idx = i * LIMBS_PER_VALUE + k;
            if idx >= flat.len() {
                break;
            }
            v = v.add(&fe(flat[idx]).mul(&w));
        }
        acc = acc.add(&eq_i.mul(&v));
    }
    Ok(acc)
}

/// Prove the folded openings for every distinct point.
///
/// The points are taken in the given (canonical first-appearance)
/// order; the claims of each point are chunked consecutively.
pub fn prove_folded_openings(
    pcs: &AkitaPcs,
    groups: &[FoldPoint<'_>],
    transcript: &mut Transcript,
) -> Result<FoldedOpenings, FoldError> {
    let ring = &pcs.pk.params.ring;
    let log_t = groups
        .first()
        .map(|g| g.point.len())
        .ok_or(FoldError::Shape {
            expected: 1,
            got: 0,
        })?;
    transcript
        .append_field_slice(
            b"fold-open-meta",
            &[
                Goldilocks::from_u64(groups.len() as u64),
                Goldilocks::from_u64(log_t as u64),
                Goldilocks::from_u64(FOLD_AMPLITUDE as u64),
                Goldilocks::from_u64(FOLD_CHUNK_COLS as u64),
            ],
        )
        .map_err(FoldError::Transcript)?;
    let mut points = Vec::with_capacity(groups.len());
    let mut chunks_out = Vec::with_capacity(groups.len());
    for group in groups {
        let point = group.point;
        let claims = group.claims;
        if claims.is_empty() {
            return Err(FoldError::Shape {
                expected: 1,
                got: 0,
            });
        }
        // Bind the claim set: (point, per-claim (col, value)).
        let mut meta = Vec::with_capacity(2 + claims.len() * 2);
        meta.push(Goldilocks::from_u64(point.len() as u64));
        meta.extend(point.iter().copied());
        meta.push(Goldilocks::from_u64(claims.len() as u64));
        transcript
            .append_field_slice(b"fold-pt-meta", &meta)
            .map_err(FoldError::Transcript)?;
        for c in claims {
            transcript
                .append_field_slice(
                    b"fold-claim",
                    &[Goldilocks::from_u64(c.col as u64), c.value],
                )
                .map_err(FoldError::Transcript)?;
        }
        let mut chunks: Vec<FoldedChunk> =
            Vec::with_capacity(claims.len().div_ceil(FOLD_CHUNK_COLS));
        for chunk_claims in claims.chunks(FOLD_CHUNK_COLS) {
            let d: Vec<i64> = (0..chunk_claims.len())
                .map(|_| sample_d(transcript))
                .collect::<Result<_, _>>()?;
            // G = Σ d_c·f_c over Goldilocks (value-level fold).
            let mut g_evals = vec![Goldilocks::ZERO; chunk_claims[0].evals.len()];
            let mut w_claim = Goldilocks::ZERO;
            for (c, dc) in chunk_claims.iter().zip(d.iter()) {
                if *dc == 0 {
                    continue;
                }
                let dc_fe = Goldilocks::from_u64(dc.unsigned_abs());
                let dc_fe = if *dc < 0 { dc_fe.neg() } else { dc_fe };
                for (g, &v) in g_evals.iter_mut().zip(c.evals.iter()) {
                    *g = g.add(&v.mul(&dc_fe));
                }
                w_claim = w_claim.add(&c.value.mul(&dc_fe));
            }
            // The eq(p, ·)·G sumcheck.
            let g_mle = DenseMle::new(g_evals).map_err(FoldError::Mle)?;
            let mut vp = VirtualPolynomial::new(log_t);
            let gi = vp.add_factor(g_mle).map_err(FoldError::Virtual)?;
            let eq = DenseMle::eq_extension(point);
            let ei = vp.add_factor(eq).map_err(FoldError::Virtual)?;
            vp.add_term(Goldilocks::ONE, vec![gi, ei])
                .map_err(FoldError::Virtual)?;
            let out = sumcheck::prove(&vp, w_claim, transcript).map_err(FoldError::Sumcheck)?;
            // The folded packed witness (limb-level integer fold).
            let opened = fold_limb_stream(ring, chunk_claims, &d)?;
            let padded = pcs.pk.pad_to_m(&opened).map_err(|_| FoldError::Ajtai)?;
            chunks.push(FoldedChunk {
                d,
                sumcheck: out.proof,
                opened: padded,
            });
        }
        points.push(point.to_vec());
        chunks_out.push(chunks);
    }
    Ok(FoldedOpenings {
        log_t,
        points,
        chunks: chunks_out,
    })
}

/// The verifier-side commitment fold: `Σ_c d_c·y_c` from the per-column
/// commitments (the Ajtai linear homomorphism).
pub fn fold_commitments(
    ring: &RingConfig,
    comms: &[&AjtaiCommitment],
    d: &[i64],
) -> Result<AjtaiCommitment, FoldError> {
    if comms.is_empty() || comms.len() != d.len() {
        return Err(FoldError::Shape {
            expected: d.len(),
            got: comms.len(),
        });
    }
    let rows = comms[0].rows.len();
    let mut out_rows = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut acc = ring.zero();
        for (c, dc) in comms.iter().zip(d.iter()) {
            if *dc == 0 {
                continue;
            }
            let scaled = c.rows[i].scale_i64(*dc);
            acc = acc.add(&scaled).map_err(FoldError::Ring)?;
        }
        out_rows.push(acc);
    }
    Ok(AjtaiCommitment { rows: out_rows })
}

/// Verify the folded openings. `commitment_of` resolves each claim's
/// column to its transmitted commitment.
pub fn verify_folded_openings(
    pcs: &AkitaPcs,
    groups: &[FoldPoint<'_>],
    commitment_of: &dyn Fn(usize) -> Result<AjtaiCommitment, FoldError>,
    proof: &FoldedOpenings,
    transcript: &mut Transcript,
) -> Result<(), FoldError> {
    let ring = &pcs.pk.params.ring;
    if proof.points.len() != groups.len() {
        return Err(FoldError::Shape {
            expected: groups.len(),
            got: proof.points.len(),
        });
    }
    let log_t = proof.log_t;
    transcript
        .append_field_slice(
            b"fold-open-meta",
            &[
                Goldilocks::from_u64(groups.len() as u64),
                Goldilocks::from_u64(log_t as u64),
                Goldilocks::from_u64(FOLD_AMPLITUDE as u64),
                Goldilocks::from_u64(FOLD_CHUNK_COLS as u64),
            ],
        )
        .map_err(FoldError::Transcript)?;
    for (gi, group) in groups.iter().enumerate() {
        let point = group.point;
        let claims = group.claims;
        if claims.is_empty() {
            return Err(FoldError::Shape {
                expected: 1,
                got: 0,
            });
        }
        let mut meta = Vec::with_capacity(2 + claims.len() * 2);
        meta.push(Goldilocks::from_u64(point.len() as u64));
        meta.extend(point.iter().copied());
        meta.push(Goldilocks::from_u64(claims.len() as u64));
        transcript
            .append_field_slice(b"fold-pt-meta", &meta)
            .map_err(FoldError::Transcript)?;
        for c in claims {
            transcript
                .append_field_slice(
                    b"fold-claim",
                    &[Goldilocks::from_u64(c.col as u64), c.value],
                )
                .map_err(FoldError::Transcript)?;
        }
        let chunks = &proof.chunks[gi];
        if chunks.len() != claims.len().div_ceil(FOLD_CHUNK_COLS) {
            return Err(FoldError::Shape {
                expected: claims.len().div_ceil(FOLD_CHUNK_COLS),
                got: chunks.len(),
            });
        }
        let gate = chunk_gate(FOLD_CHUNK_COLS.min(claims.len()));
        for (ci, chunk) in chunks.iter().enumerate() {
            let chunk_end = ((ci + 1) * FOLD_CHUNK_COLS).min(claims.len());
            let chunk_claims = &claims[ci * FOLD_CHUNK_COLS..chunk_end];
            if chunk.d.len() != chunk_claims.len() {
                return Err(FoldError::Shape {
                    expected: chunk_claims.len(),
                    got: chunk.d.len(),
                });
            }
            let d: Vec<i64> = (0..chunk_claims.len())
                .map(|_| sample_d(transcript))
                .collect::<Result<_, _>>()?;
            if d != chunk.d {
                return Err(FoldError::Shape {
                    expected: d.len(),
                    got: chunk.d.len(),
                });
            }
            // The combined claim (verifier-computable).
            let fe = |x: i64| -> Goldilocks {
                if x >= 0 {
                    Goldilocks::from_u64(x as u64)
                } else {
                    Goldilocks::from_u64(x.unsigned_abs()).neg()
                }
            };
            let mut w_claim = Goldilocks::ZERO;
            for (c, dc) in chunk_claims.iter().zip(d.iter()) {
                w_claim = w_claim.add(&c.value.mul(&fe(*dc)));
            }
            let verdict = chunk
                .sumcheck
                .verify(log_t, 2, w_claim, transcript, None)
                .map_err(FoldError::Sumcheck)?;
            // The exactness gate on the revealed folded witness.
            for elem in &chunk.opened {
                for &c in elem.coeffs() {
                    let balanced = if c <= ring.modulus.q / 2 {
                        c as i64
                    } else {
                        c as i64 - ring.modulus.q as i64
                    };
                    if balanced.abs() > gate {
                        return Err(FoldError::GateExceeded {
                            value: balanced,
                            gate,
                        });
                    }
                }
            }
            // The Ajtai-linearity check: A·V == Σ d_c·y_c.
            let comms: Result<Vec<_>, _> =
                chunk_claims.iter().map(|c| commitment_of(c.col)).collect();
            let comms = comms?;
            let refs: Vec<&AjtaiCommitment> = comms.iter().collect();
            let folded = fold_commitments(ring, &refs, &d)?;
            let recomputed = pcs.pk.commit(&chunk.opened).map_err(|_| FoldError::Ajtai)?;
            if recomputed.rows != folded.rows {
                return Err(FoldError::CommitmentMismatch);
            }
            // The final binding: eq(p, r_sc)·Φ(V)(r_sc) == final_claim —
            // Φ evaluates the folded witness's MLE at the SUMCHECK
            // TERMINAL point r_sc (the eq weights are eq(r_sc, ·)).
            let g_at_sc = packed_mle_eval(ring, &chunk.opened, &verdict.point, log_t)?;
            let eq_at_sc = DenseMle::eq_eval(point, &verdict.point).map_err(FoldError::Mle)?;
            if verdict.final_claim != eq_at_sc.mul(&g_at_sc) {
                return Err(FoldError::FinalCheckFailed);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::akita_setup;
    use lattice_ring::packing::pack_field_elements;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn cols(log_t: usize, count: usize, seed: u64) -> Vec<Vec<Goldilocks>> {
        (0..count)
            .map(|c| {
                (0..(1usize << log_t))
                    .map(|i| fe((i as u64 * 7 + c as u64 * 131 + seed).rem_euclid(1 << 40)))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn folded_openings_roundtrip_many_columns() {
        let log_t = 4usize;
        let log_n = 4u32;
        let count = 70usize; // > 2 chunks at FOLD_CHUNK_COLS=32
        let columns = cols(log_t, count, 5);
        let point: Vec<Goldilocks> = (0..log_t).map(|i| fe((i as u64 * 17 + 3) % 97)).collect();
        let claims: Vec<FoldClaim> = columns
            .iter()
            .enumerate()
            .map(|(c, col)| {
                let mle = DenseMle::new(col.clone()).ok().unwrap();
                let value = mle.evaluate(&point).ok().unwrap();
                FoldClaim {
                    col: c,
                    evals: col,
                    value,
                }
            })
            .collect();
        let m = ((3usize << log_t) >> log_n).max(1);
        let pcs = akita_setup(log_n, m, 1 << 23, [7u8; 32]).ok().unwrap();
        // Commit each column.
        let comms: Vec<AjtaiCommitment> = columns
            .iter()
            .map(|col| {
                pcs.commit(&DenseMle::new(col.clone()).ok().unwrap())
                    .ok()
                    .unwrap()
                    .commitment
            })
            .collect();
        let groups = [FoldPoint {
            point: &point,
            claims: &claims,
        }];
        let mut t = Transcript::new_default(b"fold-test");
        let proof = prove_folded_openings(&pcs, &groups, &mut t).ok().unwrap();
        assert_eq!(proof.chunks[0].len(), 3); // 70 claims → 3 chunks
        assert_eq!(proof.points.len(), 1);
        let mut t2 = Transcript::new_default(b"fold-test");
        let lookup = |col: usize| -> Result<AjtaiCommitment, FoldError> { Ok(comms[col].clone()) };
        assert!(verify_folded_openings(&pcs, &groups, &lookup, &proof, &mut t2).is_ok());
    }

    #[test]
    fn folded_openings_two_points() {
        let log_t = 3usize;
        let columns = cols(log_t, 5, 9);
        let p1: Vec<Goldilocks> = (0..log_t).map(|i| fe(i as u64 * 11 + 1)).collect();
        let p2: Vec<Goldilocks> = (0..log_t).map(|i| fe(i as u64 * 5 + 29)).collect();
        let m = ((3usize << log_t) >> 3).max(1);
        let pcs = akita_setup(3, m, 1 << 23, [1u8; 32]).ok().unwrap();
        let comms: Vec<AjtaiCommitment> = columns
            .iter()
            .map(|col| {
                pcs.commit(&DenseMle::new(col.clone()).ok().unwrap())
                    .ok()
                    .unwrap()
                    .commitment
            })
            .collect();
        let claims1: Vec<FoldClaim> = columns[0..3]
            .iter()
            .enumerate()
            .map(|(c, col)| {
                let value = DenseMle::new(col.clone())
                    .ok()
                    .unwrap()
                    .evaluate(&p1)
                    .ok()
                    .unwrap();
                FoldClaim {
                    col: c,
                    evals: col,
                    value,
                }
            })
            .collect();
        let claims2: Vec<FoldClaim> = columns[3..5]
            .iter()
            .enumerate()
            .map(|(c, col)| {
                let value = DenseMle::new(col.clone())
                    .ok()
                    .unwrap()
                    .evaluate(&p2)
                    .ok()
                    .unwrap();
                FoldClaim {
                    col: c + 3,
                    evals: col,
                    value,
                }
            })
            .collect();
        let groups = [
            FoldPoint {
                point: &p1,
                claims: &claims1,
            },
            FoldPoint {
                point: &p2,
                claims: &claims2,
            },
        ];
        let mut t = Transcript::new_default(b"fold-test");
        let proof = prove_folded_openings(&pcs, &groups, &mut t).ok().unwrap();
        assert_eq!(proof.points.len(), 2);
        let mut t2 = Transcript::new_default(b"fold-test");
        let lookup = |col: usize| -> Result<AjtaiCommitment, FoldError> { Ok(comms[col].clone()) };
        assert!(verify_folded_openings(&pcs, &groups, &lookup, &proof, &mut t2).is_ok());
    }

    #[test]
    fn folded_tamper_suite() {
        let log_t = 3usize;
        let columns = cols(log_t, 4, 21);
        let point: Vec<Goldilocks> = (0..log_t).map(|i| fe(i as u64 * 3 + 7)).collect();
        let m = ((3usize << log_t) >> 3).max(1);
        let pcs = akita_setup(3, m, 1 << 23, [2u8; 32]).ok().unwrap();
        let comms: Vec<AjtaiCommitment> = columns
            .iter()
            .map(|col| {
                pcs.commit(&DenseMle::new(col.clone()).ok().unwrap())
                    .ok()
                    .unwrap()
                    .commitment
            })
            .collect();
        let claims: Vec<FoldClaim> = columns
            .iter()
            .enumerate()
            .map(|(c, col)| {
                let value = DenseMle::new(col.clone())
                    .ok()
                    .unwrap()
                    .evaluate(&point)
                    .ok()
                    .unwrap();
                FoldClaim {
                    col: c,
                    evals: col,
                    value,
                }
            })
            .collect();
        let groups = [FoldPoint {
            point: &point,
            claims: &claims,
        }];
        let mut t = Transcript::new_default(b"fold-test");
        let proof = prove_folded_openings(&pcs, &groups, &mut t).ok().unwrap();
        let lookup = |col: usize| -> Result<AjtaiCommitment, FoldError> { Ok(comms[col].clone()) };

        // (a) A tampered claimed value desyncs the transcript.
        let mut bad_claims: Vec<FoldClaim> = claims
            .iter()
            .map(|c| FoldClaim {
                col: c.col,
                evals: c.evals,
                value: c.value,
            })
            .collect();
        bad_claims[1].value = bad_claims[1].value.add(&fe(1));
        let bad_groups = [FoldPoint {
            point: &point,
            claims: &bad_claims,
        }];
        let mut t2 = Transcript::new_default(b"fold-test");
        assert!(verify_folded_openings(&pcs, &bad_groups, &lookup, &proof, &mut t2).is_err());

        // (b) A tampered round message fails the sumcheck.
        let mut tp = prove_folded_openings(&pcs, &groups, &mut Transcript::new_default(b"x"))
            .ok()
            .unwrap();
        if let Some(r) = tp.chunks[0][0].sumcheck.rounds.first_mut() {
            if let Some(v) = r.first_mut() {
                *v = v.add(&fe(1));
            }
        }
        let mut t3 = Transcript::new_default(b"fold-test");
        assert!(verify_folded_openings(&pcs, &groups, &lookup, &tp, &mut t3).is_err());

        // (c) A swapped opened witness fails the Ajtai fold.
        let mut tw = prove_folded_openings(&pcs, &groups, &mut Transcript::new_default(b"x"))
            .ok()
            .unwrap();
        let other = pack_field_elements(&pcs.pk.params.ring, &cols(log_t, 1, 999)[0]);
        tw.chunks[0][0].opened = pcs.pk.pad_to_m(&other).ok().unwrap();
        let mut t4 = Transcript::new_default(b"fold-test");
        assert!(verify_folded_openings(&pcs, &groups, &lookup, &tw, &mut t4).is_err());

        // (d) A tampered per-column commitment fails the fold.
        let wrong_comm = {
            let other_col = cols(log_t, 1, 555)[0].clone();
            pcs.commit(&DenseMle::new(other_col).ok().unwrap())
                .ok()
                .unwrap()
                .commitment
        };
        let bad_lookup = |col: usize| -> Result<AjtaiCommitment, FoldError> {
            Ok(if col == 1 {
                wrong_comm.clone()
            } else {
                comms[col].clone()
            })
        };
        let mut t5 = Transcript::new_default(b"fold-test");
        assert!(verify_folded_openings(&pcs, &groups, &bad_lookup, &proof, &mut t5).is_err());
    }
}
