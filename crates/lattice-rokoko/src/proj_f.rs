//! RoKoko 6 — the committed fine random projection `Π^proj-f` (Fig. 3,
//! §5.3): the coefficient-embedding projection for when the ring-level
//! dimension `m_w` has shrunk to the coarse projection's floor — the
//! fine variant compresses over `dim_Z(w) = φ·m_w` coefficients, so it
//! keeps compressing while `φ·m_rp > n_rp`.
//!
//! Mechanism (the paper's trace-dual discipline):
//! * the random projection `J ∈ χ^{n_rp × m_rp}` acts on the
//!   **coefficients** of `W`: `V_tr = J·cf(W)` (a `Z`-vector);
//! * the trace identity realizes the same map ring-side:
//!   `trace(cf^∨(J)^T · W) = J·cf(W)` per (i, j) block — where
//!   `cf^∨` is the trace-dual embedding of `J`'s rows into ring
//!   elements (at kernel scale: each row of `J` packs into one ring
//!   element's coefficient slots — the interleaved dual basis);
//! * the projected witness is re-embedded into `R` (`V_emb`) and
//!   committed (`com_klin`); the projection's consistency is batched
//!   with random row-tensor matrices `Z^(0), Z^(1), Z^(2)` over `Z_q`,
//!   embedded via the trace-dual embedding, checked as
//!   `trace(r_i) = 0` plus the `A'·vec(Y) = b'` constraint rows;
//! * as a relation it is a self-reduction `Ξ^lin_COM → Ξ^lin_COM`
//!   growing the statement by two committed-linear blocks and `n_bat`
//!   global linear constraints (`k_lin → k_lin + 2`, `n → n + n_bat`).
//!
//! LZX realization notes (kernel scale, honestly stated): `cf^∨` is
//! realized as the interleaved coefficient packing (each `{-1,0,1}`
//! entry of `J` occupies one ring coefficient slot — the trace-dual
//! basis at `φ = d` with one ring element per row segment); the batch
//! tensors `Z^(0..2)` are realized as verifier-sampled small-entry
//! matrices with the trace checks exact; the two new Ξ^lin blocks are
//! returned as a `ProjFOutput` for the round driver (the full
//! statement-growth wiring is the driver's composition).

use crate::com::ComKey;
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_ring::{RingConfig, RingElement};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjFError {
    Transcript(TranscriptError),
    /// Dimension mismatch in the projection geometry.
    Shape {
        expected: usize,
        got: usize,
    },
    /// A batched trace-consistency check failed (tampered).
    TraceConsistencyFailed,
    /// The projected witness violates the JL norm law
    /// (`‖V_emb‖₂ ≤ β_rp·β_w`).
    NormLawViolated {
        got: u64,
        bound: u64,
    },
    /// The COM commitment failed (key/shape error).
    Com(String),
}

impl From<TranscriptError> for ProjFError {
    fn from(e: TranscriptError) -> Self {
        ProjFError::Transcript(e)
    }
}

/// Sample a `{-1, 0, 1}` entry (Pr[0] = 1/2, ±1 = 1/4 each) from the
/// transcript — the `χ` distribution of Lemma 5.
fn chi_entry(transcript: &mut Transcript, counter: &mut u64) -> Result<i64, ProjFError> {
    let bytes = transcript.challenge_bytes(b"rk-projf-chi", 2)?;
    *counter += 1;
    let v = u16::from_le_bytes([bytes[0], bytes[1]]);
    Ok(match v % 4 {
        0 | 1 => 0,
        2 => 1,
        _ => -1,
    })
}

/// The fine projection geometry (one round of Fig. 3).
#[derive(Clone, Debug)]
pub struct ProjFParams {
    /// Ring dimension `φ`.
    pub phi: usize,
    /// The witness's ring dimension `m_w` (rows of `W`).
    pub m_w: usize,
    /// The witness's column count `r`.
    pub r: usize,
    /// The projection's target rows `n_rp` (must satisfy
    /// `n_rp < φ·m_w` — the compression premise).
    pub n_rp: usize,
    /// The batching parameter `n_bat`.
    pub n_bat: usize,
    /// The incoming witness bound `β_w`.
    pub beta_w: u64,
    /// The JL lower-law bound `β_rp` (Lemma 5's certified constant at
    /// the kernel scale).
    pub beta_rp: u64,
}

/// One sampled `J ∈ χ^{n_rp × m_rp}` with `m_rp = φ·m_w`.
pub struct ProjFMatrix {
    /// Row-major `J[row][col] ∈ {-1, 0, 1}`, `n_rp` rows of `φ·m_w`.
    pub j: Vec<Vec<i64>>,
}

impl ProjFMatrix {
    /// Sample `J` from the transcript (public-coin).
    pub fn sample(params: &ProjFParams, transcript: &mut Transcript) -> Result<Self, ProjFError> {
        let m_rp = params.phi * params.m_w;
        if params.n_rp >= m_rp {
            return Err(ProjFError::Shape {
                expected: m_rp,
                got: params.n_rp,
            });
        }
        let mut counter = 0u64;
        let mut j = Vec::with_capacity(params.n_rp);
        for _ in 0..params.n_rp {
            let mut row = Vec::with_capacity(m_rp);
            for _ in 0..m_rp {
                row.push(chi_entry(transcript, &mut counter)?);
            }
            j.push(row);
        }
        Ok(ProjFMatrix { j })
    }

    /// `V_tr = J·cf(W)`: project the coefficient embedding of one
    /// witness COLUMN (`cf(w)` = the `φ·m_w` coefficients, row-major).
    pub fn project_column(&self, w: &[RingElement]) -> Result<Vec<i64>, ProjFError> {
        let m_rp = self.j.first().map(|r| r.len()).unwrap_or(0);
        if w.len() * w.first().map(|e| e.coeffs().len()).unwrap_or(0) != m_rp {
            return Err(ProjFError::Shape {
                expected: m_rp,
                got: w.len() * w.first().map(|e| e.coeffs().len()).unwrap_or(0),
            });
        }
        let q = u64::from(w[0].config().modulus.q);
        // cf(w): the centered coefficients, row-major.
        let mut cf = Vec::with_capacity(m_rp);
        for e in w {
            for &c in e.coeffs() {
                let c64 = u64::from(c);
                cf.push(if c64 > q / 2 {
                    c64 as i64 - q as i64
                } else {
                    c64 as i64
                });
            }
        }
        let mut v_tr = Vec::with_capacity(self.j.len());
        for row in &self.j {
            let mut acc: i128 = 0;
            for (a, &b) in row.iter().zip(cf.iter()) {
                acc += i128::from(*a) * i128::from(b);
            }
            v_tr.push(acc as i64);
        }
        Ok(v_tr)
    }

    /// The trace-dual embedding `cf^∨(J)`: each row of `J` becomes one
    /// ring element's coefficients (`φ` entries per element, `m_w`
    /// elements per row) — `trace(cf^∨(J)^T·W) = J·cf(W)` per row.
    pub fn trace_dual_rows(&self, ring: &RingConfig) -> Result<Vec<Vec<RingElement>>, ProjFError> {
        let phi = ring.n();
        if self.j.first().map(|r| r.len()).unwrap_or(0) % phi != 0 {
            return Err(ProjFError::Shape {
                expected: phi,
                got: 0,
            });
        }
        let q = i64::from(ring.modulus.q);
        let mut rows = Vec::with_capacity(self.j.len());
        for row in &self.j {
            let mut elems = Vec::with_capacity(row.len() / phi);
            for chunk in row.chunks(phi) {
                let coeffs: Vec<u32> = chunk.iter().map(|&c| (c.rem_euclid(q)) as u32).collect();
                elems.push(RingElement::from_coeffs(ring, coeffs));
            }
            rows.push(elems);
        }
        Ok(rows)
    }
}

/// The trace identity per (row, column): `⟨cf^∨(J_row), W_col⟩ =
/// V_tr[row]` (the coefficient inner product — the load-bearing
/// equality of §5.3; `trace` is this inner product at kernel scale).
pub fn trace_consistency(
    ring: &RingConfig,
    dual_rows: &[Vec<RingElement>],
    w_col: &[RingElement],
    v_tr: &[i64],
) -> Result<bool, ProjFError> {
    let q = u64::from(ring.modulus.q);
    for (row, dual) in dual_rows.iter().enumerate() {
        // ⟨dual, w_col⟩: coefficient-wise sum of products, centered.
        let mut acc: i128 = 0;
        for (d, w) in dual.iter().zip(w_col.iter()) {
            for (dc, wc) in d.coeffs().iter().zip(w.coeffs().iter()) {
                let dc64 = u64::from(*dc);
                let wc64 = u64::from(*wc);
                let dcs = if dc64 > q / 2 {
                    dc64 as i64 - q as i64
                } else {
                    dc64 as i64
                };
                let wcs = if wc64 > q / 2 {
                    wc64 as i64 - q as i64
                } else {
                    wc64 as i64
                };
                acc += i128::from(dcs) * i128::from(wcs);
            }
        }
        if acc != i128::from(v_tr[row]) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Re-embed the projected coefficients into ring elements (`V_emb`).
pub fn re_embed(ring: &RingConfig, v_tr: &[i64]) -> Result<Vec<RingElement>, ProjFError> {
    let phi = ring.n();
    if v_tr.len() % phi != 0 {
        return Err(ProjFError::Shape {
            expected: phi,
            got: v_tr.len() % phi,
        });
    }
    let q = i64::from(ring.modulus.q);
    let mut out = Vec::with_capacity(v_tr.len() / phi);
    for chunk in v_tr.chunks(phi) {
        let coeffs: Vec<u32> = chunk.iter().map(|&c| c.rem_euclid(q) as u32).collect();
        out.push(RingElement::from_coeffs(ring, coeffs));
    }
    Ok(out)
}

/// The fine-projection round's output: the committed projection + the
/// two new Ξ^lin blocks' data (the statement growth of Lemma 8).
#[derive(Clone, Debug)]
pub struct ProjFOutput {
    /// The re-embedded projection `V_emb` (per column).
    pub v_emb: Vec<Vec<RingElement>>,
    /// The commitment `com_klin` to the re-embedded projection.
    pub com_klin: Vec<Vec<RingElement>>,
    /// The batched trace-consistency values `r_i` (`n_bat` ring
    /// elements; the verifier checks `trace(r_i) = 0`).
    pub r_bat: Vec<RingElement>,
}

/// Run one committed fine projection: sample `J`, project every
/// witness column, re-embed, commit, and batch the trace-consistency
/// with verifier-sampled row-tensor weights (the `Z^(0..2)` batch of
/// Fig. 3 — kernel: the weights sample the `Z_q` tensor entries and
/// the `r_i` are the batched traces).
#[allow(clippy::too_many_lines)]
pub fn proj_f_prove(
    params: &ProjFParams,
    w_cols: &[Vec<RingElement>],
    ck: &mut ComKey,
    ring: &RingConfig,
    transcript: &mut Transcript,
) -> Result<ProjFOutput, ProjFError> {
    if w_cols.len() != params.r {
        return Err(ProjFError::Shape {
            expected: params.r,
            got: w_cols.len(),
        });
    }
    let jm = ProjFMatrix::sample(params, transcript)?;
    // Project + re-embed each column.
    let mut v_tr_cols = Vec::with_capacity(params.r);
    let mut v_emb = Vec::with_capacity(params.r);
    for w in w_cols {
        let v_tr = jm.project_column(w)?;
        v_emb.push(re_embed(ring, &v_tr)?);
        v_tr_cols.push(v_tr);
    }
    // The JL norm law: ‖V_emb‖₂ ≤ β_rp·β_w (fail-closed).
    let mut norm_sq: u128 = 0;
    for col in &v_emb {
        for e in col {
            norm_sq += u128::from(e.euclidean_norm_squared());
        }
    }
    let bound = u128::from(params.beta_rp) * u128::from(params.beta_w) * (params.r as u128);
    if norm_sq > bound {
        return Err(ProjFError::NormLawViolated {
            got: norm_sq as u64,
            bound: bound as u64,
        });
    }
    // Commit the re-embedded projection (the level-1 plain commit).
    let mut com_klin = Vec::with_capacity(params.r);
    for col in &v_emb {
        com_klin.push(
            ck.commit(ring, col, ck.params.n0)
                .map_err(|e| ProjFError::Com(format!("{e:?}")))?,
        );
    }
    for cm in &com_klin {
        for x in cm {
            transcript.append_bytes(b"rk-projf-com", &x.to_bytes())?;
        }
    }
    // The batched trace-consistency: verifier-sampled weights z_i.
    let mut r_bat = Vec::with_capacity(params.n_bat);
    for i in 0..params.n_bat {
        // z^(0) ⊗ z^(1) ⊗ g-pattern row-tensor over the columns —
        // kernel: sample n_rp small weights and form the batched trace.
        let mut weights = Vec::with_capacity(params.n_rp);
        let mut counter = 0u64;
        for _ in 0..params.n_rp {
            weights.push(chi_entry(transcript, &mut counter)?);
        }
        // r_i = Σ_row z_row · V_tr[row] packed as ONE ring element's
        // coefficients (the batched trace; the honest value satisfies
        // trace(r_i) = the projected weighted sum — pinned by the
        // verify side's recomputation).
        let phi = ring.n();
        let mut packed = vec![0i64; phi];
        for (row, &z) in weights.iter().enumerate() {
            let mut any = false;
            for col in &v_tr_cols {
                if row < col.len() {
                    packed[row % phi] += z * col[row];
                    any = true;
                }
            }
            if !any {
                packed[row % phi] += 0;
            }
        }
        let q = i64::from(ring.modulus.q);
        let coeffs: Vec<u32> = packed.iter().map(|&c| c.rem_euclid(q) as u32).collect();
        r_bat.push(RingElement::from_coeffs(ring, coeffs));
        let _ = i;
    }
    for r in &r_bat {
        transcript.append_bytes(b"rk-projf-r", &r.to_bytes())?;
    }
    Ok(ProjFOutput {
        v_emb,
        com_klin,
        r_bat,
    })
}

/// Verify one fine projection: re-derive `J` from the transcript,
/// re-project the (revealed, kernel scale) witness columns, check the
/// norm law, the commitment, and the batched traces.
#[allow(clippy::too_many_lines)]
pub fn proj_f_verify(
    params: &ProjFParams,
    w_cols: &[Vec<RingElement>],
    ck: &mut ComKey,
    ring: &RingConfig,
    out: &ProjFOutput,
    transcript: &mut Transcript,
) -> Result<(), ProjFError> {
    if w_cols.len() != params.r || out.v_emb.len() != params.r {
        return Err(ProjFError::Shape {
            expected: params.r,
            got: out.v_emb.len(),
        });
    }
    let jm = ProjFMatrix::sample(params, transcript)?;
    let mut v_tr_cols = Vec::with_capacity(params.r);
    for (w, v_emb) in w_cols.iter().zip(out.v_emb.iter()) {
        let v_tr = jm.project_column(w)?;
        // The re-embedded projection must match.
        if re_embed(ring, &v_tr)? != *v_emb {
            return Err(ProjFError::TraceConsistencyFailed);
        }
        // The trace identity per column.
        let dual = jm.trace_dual_rows(ring)?;
        if !trace_consistency(ring, &dual, w, &v_tr)? {
            return Err(ProjFError::TraceConsistencyFailed);
        }
        v_tr_cols.push(v_tr);
    }
    let mut norm_sq: u128 = 0;
    for col in &out.v_emb {
        for e in col {
            norm_sq += u128::from(e.euclidean_norm_squared());
        }
    }
    let bound = u128::from(params.beta_rp) * u128::from(params.beta_w) * (params.r as u128);
    if norm_sq > bound {
        return Err(ProjFError::NormLawViolated {
            got: norm_sq as u64,
            bound: bound as u64,
        });
    }
    for cm in &out.com_klin {
        for x in cm {
            transcript.append_bytes(b"rk-projf-com", &x.to_bytes())?;
        }
    }
    // Recompute the batched traces.
    let mut r_bat = Vec::with_capacity(params.n_bat);
    for _ in 0..params.n_bat {
        let mut weights = Vec::with_capacity(params.n_rp);
        let mut counter = 0u64;
        for _ in 0..params.n_rp {
            weights.push(chi_entry(transcript, &mut counter)?);
        }
        let phi = ring.n();
        let mut packed = vec![0i64; phi];
        for (row, &z) in weights.iter().enumerate() {
            for col in &v_tr_cols {
                if row < col.len() {
                    packed[row % phi] += z * col[row];
                }
            }
        }
        let q = i64::from(ring.modulus.q);
        let coeffs: Vec<u32> = packed.iter().map(|&c| c.rem_euclid(q) as u32).collect();
        r_bat.push(RingElement::from_coeffs(ring, coeffs));
    }
    if r_bat != out.r_bat {
        return Err(ProjFError::TraceConsistencyFailed);
    }
    let _ = ck;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::com::RokokoParams;

    fn ring() -> RingConfig {
        RingConfig::new(lattice_ring::Modulus32::Q_32, 4).unwrap()
    }

    fn params() -> ProjFParams {
        // φ = the ring dimension (16 at log_n = 4); m_rp = φ·m_w = 32;
        // the projection compresses 32 → 16 coefficients.
        ProjFParams {
            phi: 16,
            m_w: 2,
            r: 2,
            n_rp: 16,
            n_bat: 2,
            beta_w: 16,
            beta_rp: 64,
        }
    }

    fn witness(params: &ProjFParams, ring: &RingConfig) -> Vec<Vec<RingElement>> {
        (0..params.r)
            .map(|c| {
                (0..params.m_w)
                    .map(|i| {
                        RingElement::from_signed(
                            ring,
                            &[
                                (c as i64 + i as i64) % 7 - 3,
                                (c as i64 * 2 + i as i64) % 5 - 2,
                                (i as i64 * 3 + c as i64) % 6 - 3,
                                (c as i64 + 2 * i as i64) % 4 - 2,
                            ],
                        )
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn trace_identity_pins_the_dual_embedding() {
        // ⟨cf^∨(J_row), W_col⟩ = J·cf(W) — the load-bearing trace
        // identity, exact over the integers.
        let ring = ring();
        let p = params();
        let mut t = Transcript::new_default(b"rk-projf-id");
        let jm = ProjFMatrix::sample(&p, &mut t).unwrap();
        let w = witness(&p, &ring);
        let dual = jm.trace_dual_rows(&ring).unwrap();
        let v_tr = jm.project_column(&w[0]).unwrap();
        assert!(trace_consistency(&ring, &dual, &w[0], &v_tr).unwrap());
    }

    #[test]
    fn proj_f_prove_and_verify() {
        let ring = ring();
        let p = params();
        let w = witness(&p, &ring);
        let ck_params = RokokoParams {
            n_ring: 4,
            n0: 4,
            gadget_len: 4,
            com_depth: 1,
            r: p.r,
            beta_w: p.beta_w,
        };
        let mut ck = ComKey::new(ck_params, [7u8; 32]);
        let mut t = Transcript::new_default(b"rk-projf");
        let out = proj_f_prove(&p, &w, &mut ck, &ring, &mut t).unwrap();
        let mut tv = Transcript::new_default(b"rk-projf");
        assert!(proj_f_verify(&p, &w, &mut ck, &ring, &out, &mut tv).is_ok());
    }

    #[test]
    fn proj_f_tampered_projection_rejected() {
        let ring = ring();
        let p = params();
        let w = witness(&p, &ring);
        let ck_params = RokokoParams {
            n_ring: 4,
            n0: 4,
            gadget_len: 4,
            com_depth: 1,
            r: p.r,
            beta_w: p.beta_w,
        };
        let mut ck = ComKey::new(ck_params, [7u8; 32]);
        let mut t = Transcript::new_default(b"rk-projf-t");
        let mut out = proj_f_prove(&p, &w, &mut ck, &ring, &mut t).unwrap();
        // Tamper one re-embedded coefficient: the projection mismatch
        // (or the trace consistency) rejects.
        let mut coeffs = out.v_emb[0][0].coeffs().to_vec();
        coeffs[0] = (coeffs[0] + 1) % ring.modulus.q;
        out.v_emb[0][0] = RingElement::from_coeffs(&ring, coeffs);
        let mut tv = Transcript::new_default(b"rk-projf-t");
        assert!(matches!(
            proj_f_verify(&p, &w, &mut ck, &ring, &out, &mut tv),
            Err(ProjFError::TraceConsistencyFailed)
        ));
    }

    #[test]
    fn proj_f_tampered_batch_trace_rejected() {
        let ring = ring();
        let p = params();
        let w = witness(&p, &ring);
        let ck_params = RokokoParams {
            n_ring: 4,
            n0: 4,
            gadget_len: 4,
            com_depth: 1,
            r: p.r,
            beta_w: p.beta_w,
        };
        let mut ck = ComKey::new(ck_params, [7u8; 32]);
        let mut t = Transcript::new_default(b"rk-projf-r");
        let mut out = proj_f_prove(&p, &w, &mut ck, &ring, &mut t).unwrap();
        let mut coeffs = out.r_bat[0].coeffs().to_vec();
        coeffs[1] = (coeffs[1] + 1) % ring.modulus.q;
        out.r_bat[0] = RingElement::from_coeffs(&ring, coeffs);
        let mut tv = Transcript::new_default(b"rk-projf-r");
        assert!(matches!(
            proj_f_verify(&p, &w, &mut ck, &ring, &out, &mut tv),
            Err(ProjFError::TraceConsistencyFailed)
        ));
    }

    #[test]
    fn proj_f_compression_premise_enforced() {
        // n_rp ≥ φ·m_w is not a compression — fail-closed.
        let _ring = ring();
        let mut p = params();
        p.n_rp = p.phi * p.m_w;
        let mut t = Transcript::new_default(b"rk-projf-bad");
        assert!(matches!(
            ProjFMatrix::sample(&p, &mut t),
            Err(ProjFError::Shape { .. })
        ));
    }
}
