//! The polynomial commitment instantiation for ePrint 2026/538: **vector
//! Pedersen over BN254 G1** (via `lattice_pcd`), with evaluation proofs as
//! **linear openings**.
//!
//! The paper keeps the PC abstract ("compatible with any PC that supports
//! evaluation proof accumulation — KZG, Pedersen [BCMS20], and their
//! variants"); this module is the concrete drop-in:
//!
//! * `PcKey::commit_vec` / `commit_matrix` — commitments to the
//!   polynomial's encoding (cube values for multilinear, the Lagrange
//!   coefficients for univariate — both are the natural n-element encodings
//!   over `Domain`), homomorphic over `F_r`-scalars.
//! * `LinearOpening` — the evaluation proof: the full encoding + the
//!   blinding scalar. **Transparent long openings** (O(n) proof size):
//!   binding (Pedersen binding), complete, and O(n) verify. The honest
//!   deviation ledger records that swapping in an IPA (the BCMS20
//!   "Pedersen PCS" route) or KZG gives O(log n)/O(1) proofs — the swap is
//!   local to `open`/`verify`.
//! * `batch_prove_pce` / `batch_verify_pcep` — `Π_provePCE` and
//!   `Π_batchPCEP`: batch same-point claims via the homomorphic
//!   combination `Σ ηᵢ·[cᵢ]` + ONE combined opening (sound by binding:
//!   a valid batch opening implies each claim up to a Pedersen opening
//!   collision).

use crate::poly::{lambda_eval, Domain};
use crate::Fp256;
use lattice_core::transcript::Transcript;
use lattice_pcd::pedersen::{msm, PedersenCommitment, PedersenKey};
use lattice_pcd::G1Affine;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PcError {
    Shape(&'static str),
    Poly(crate::poly::PolyError),
    Pedersen(lattice_pcd::pedersen::PedersenError),
    Transcript(lattice_core::transcript::TranscriptError),
    VerificationFailed,
}

impl core::fmt::Display for PcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PcError::Shape(s) => write!(f, "pc shape: {s}"),
            PcError::Poly(e) => write!(f, "poly: {e}"),
            PcError::Pedersen(e) => write!(f, "pedersen: {e}"),
            PcError::Transcript(e) => write!(f, "transcript: {e}"),
            PcError::VerificationFailed => write!(f, "opening verification failed"),
        }
    }
}

impl From<crate::poly::PolyError> for PcError {
    fn from(e: crate::poly::PolyError) -> Self {
        PcError::Poly(e)
    }
}

impl From<lattice_pcd::pedersen::PedersenError> for PcError {
    fn from(e: lattice_pcd::pedersen::PedersenError) -> Self {
        PcError::Pedersen(e)
    }
}

impl From<lattice_core::transcript::TranscriptError> for PcError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        PcError::Transcript(e)
    }
}

/// A polynomial commitment (one Pedersen point).
pub type PcCommitment = PedersenCommitment;

/// The PC key: the vector-Pedersen key (sized to the domain).
pub struct PcKey {
    pub key: PedersenKey,
    pub domain: Domain,
}

impl PcKey {
    pub fn new(domain: Domain, seed: &[u8]) -> Result<PcKey, PcError> {
        Ok(PcKey {
            key: PedersenKey::derive(seed, domain.size())?,
            domain,
        })
    }

    /// Commit a polynomial given by its domain encoding (cube values /
    /// Lagrange coefficients). Returns the commitment + the witness
    /// (encoding + blinding) for later openings.
    pub fn commit_encoding(
        &self,
        encoding: &[Fp256],
        transcript: &mut Transcript,
    ) -> Result<(PcCommitment, PcWitness), PcError> {
        if encoding.len() != self.domain.size() {
            return Err(PcError::Shape("encoding length vs domain"));
        }
        let (c, blind) = self.key.commit_fresh(encoding, transcript)?;
        Ok((
            c,
            PcWitness {
                encoding: encoding.to_vec(),
                blind,
            },
        ))
    }

    /// Commit a matrix polynomial `M(X,Y)` — the encoding is the n² matrix
    /// entries in row-major order (the coefficient vector of the
    /// λ(Y)-by-λ(X) expansion).
    pub fn commit_matrix(
        &self,
        domain: &Domain,
        m: &[Vec<Fp256>],
    ) -> Result<PcCommitment, PcError> {
        let n = domain.size();
        if m.len() != n || m.iter().any(|r| r.len() != n) {
            return Err(PcError::Shape("matrix shape"));
        }
        // The matrix commitment is over 2ν variables → the encoding is n².
        // Reuse the key by expanding a dedicated key derivation: we commit
        // with a key sized n² derived from the same seed.
        let key2 = PedersenKey::derive(
            &self.key.seed,
            n * n,
        )?;
        let flat: Vec<Fp256> = m.concat();
        // Deterministic blinding (the matrices are public structure — the
        // commitment is binding-only here; hiding is not required for the
        // holographic index).
        let blind = Fp256::ZERO;
        Ok(key2.commit(&flat, &blind)?)
    }

    /// Open a matrix commitment (the long opening of the n² encoding).
    pub fn open_matrix(
        &self,
        domain: &Domain,
        m: &[Vec<Fp256>],
    ) -> Result<LinearOpening, PcError> {
        let n = domain.size();
        if m.len() != n || m.iter().any(|r| r.len() != n) {
            return Err(PcError::Shape("matrix shape"));
        }
        let key2 = PedersenKey::derive(&self.key.seed, n * n)?;
        let flat: Vec<Fp256> = m.concat();
        let c = key2.commit(&flat, &Fp256::ZERO)?;
        Ok(LinearOpening {
            commitment: c,
            encoding: flat,
            blind: Fp256::ZERO,
        })
    }

    /// Verify a matrix opening (re-commit and compare).
    pub fn verify_matrix_opening(
        &self,
        domain: &Domain,
        opening: &LinearOpening,
    ) -> Result<bool, PcError> {
        let n = domain.size();
        if opening.encoding.len() != n * n {
            return Err(PcError::Shape("matrix opening length"));
        }
        let key2 = PedersenKey::derive(&self.key.seed, n * n)?;
        let c = key2.commit(&opening.encoding, &opening.blind)?;
        Ok(c == opening.commitment)
    }

    /// Evaluate the polynomial at a domain point (ν coordinates):
    /// multilinear → the MLE at the point; univariate → the Lagrange
    /// combination.
    pub fn eval_encoding(&self, encoding: &[Fp256], point: &[Fp256]) -> Result<Fp256, PcError> {
        if encoding.len() != self.domain.size() || point.len() != self.domain.nu() {
            return Err(PcError::Shape("eval shape"));
        }
        let mut acc = Fp256::ZERO;
        for (i, v) in encoding.iter().enumerate() {
            if v.is_zero() {
                continue;
            }
            let lam = lambda_eval(&self.domain, i, point)?;
            acc = acc.add(&v.mul(&lam));
        }
        Ok(acc)
    }

    /// Evaluate a matrix polynomial at `(x, y)`.
    pub fn eval_matrix(
        &self,
        domain: &Domain,
        m: &[Vec<Fp256>],
        x: &[Fp256],
        y: &[Fp256],
    ) -> Result<Fp256, PcError> {
        crate::poly::matrix_poly_eval(domain, m, x, y).map_err(PcError::Poly)
    }
}

/// The committed polynomial's witness (for openings).
#[derive(Clone, Debug)]
pub struct PcWitness {
    pub encoding: Vec<Fp256>,
    pub blind: Fp256,
}

/// A transparent linear opening: the full encoding + blinding.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearOpening {
    pub commitment: PcCommitment,
    pub encoding: Vec<Fp256>,
    pub blind: Fp256,
}

impl LinearOpening {
    /// Verify: the commitment opens the encoding, and the evaluation at
    /// `point` equals `claim`.
    pub fn verify(
        &self,
        key: &PcKey,
        point: &[Fp256],
        claim: &Fp256,
    ) -> Result<bool, PcError> {
        if self.encoding.len() != key.domain.size() {
            return Ok(false);
        }
        if !key
            .key
            .verify_opening(&self.commitment, &self.encoding, &self.blind)?
        {
            return Ok(false);
        }
        let v = key.eval_encoding(&self.encoding, point)?;
        Ok(v == *claim)
    }
}

/// Π_provePCE: batch same-point evaluation claims into ONE opening.
/// Inputs: `(commitment, encoding, blind, claimed_eval)` triples at the
/// SAME point α. Output: a batched `PceBatchProof`.
#[derive(Clone, Debug, PartialEq)]
pub struct PceBatchProof {
    /// The combined commitment `Σ ηᵢ·[cᵢ]`.
    pub combined: PcCommitment,
    /// The combined encoding `Σ ηᵢ·pᵢ` (the long opening).
    pub encoding: Vec<Fp256>,
    /// The combined blinding.
    pub blind: Fp256,
    /// The η challenges (rederived by the verifier).
    pub etas: Vec<Fp256>,
}

/// The statement for batching: (commitment, claimed eval) at one point.
#[derive(Clone, Debug, PartialEq)]
pub struct PceClaim {
    pub poly_id: usize,
    pub commitment: PcCommitment,
    pub claimed: Fp256,
}

#[allow(clippy::type_complexity)]
pub fn batch_prove_pce(
    key: &PcKey,
    point: &[Fp256],
    claims: &[PceClaim],
    witnesses: &[(Vec<Fp256>, Fp256)], // (encoding, blind) per claim
    _transcript: &mut Transcript,
) -> Result<PceBatchProof, PcError> {
    if claims.len() != witnesses.len() {
        return Err(PcError::Shape("claims vs witnesses"));
    }
    // η ← RO over the point + commitments + claims — a SELF-CONTAINED
    // derivation (the proof is portable: verified at any later transcript
    // state, e.g. by the next fold).
    let mut t = Transcript::new_default(b"pcep-batch");
    lattice_pcd::util::absorb_fp_slice(&mut t, b"pcep-point", point)?;
    for c in claims {
        t.append_message(b"pcep-com", &c.commitment.to_bytes())?;
        lattice_pcd::util::absorb_fp(&mut t, b"pcep-val", &c.claimed)?;
    }
    let etas = lattice_pcd::util::challenge_fp_vec(&mut t, b"pcep-eta", claims.len())?;
    // Combined encoding/blind/commitment.
    let mut enc = vec![Fp256::ZERO; key.domain.size()];
    let mut blind = Fp256::ZERO;
    let mut coms: Vec<(Fp256, PedersenCommitment)> = Vec::with_capacity(claims.len());
    for (i, (e, b)) in witnesses.iter().enumerate() {
        if e.len() != key.domain.size() {
            return Err(PcError::Shape("witness encoding length"));
        }
        for (j, v) in e.iter().enumerate() {
            enc[j] = enc[j].add(&etas[i].mul(v));
        }
        blind = blind.add(&etas[i].mul(b));
        coms.push((etas[i], claims[i].commitment));
    }
    let combined = PedersenCommitment::linear_combine(&coms);
    Ok(PceBatchProof {
        combined,
        encoding: enc,
        blind,
        etas,
    })
}

/// Π_batchPCEP verification: rederive η, recombine, and check the single
/// opening + the claimed combined value `Σ ηᵢ·vᵢ`.
pub fn batch_verify_pcep(
    key: &PcKey,
    point: &[Fp256],
    claims: &[PceClaim],
    claimed_sum: &Fp256,
    proof: &PceBatchProof,
    _transcript: &mut Transcript,
) -> Result<bool, PcError> {
    // Self-contained η rederivation (mirrors the prover).
    let mut t = Transcript::new_default(b"pcep-batch");
    lattice_pcd::util::absorb_fp_slice(&mut t, b"pcep-point", point)?;
    for c in claims {
        t.append_message(b"pcep-com", &c.commitment.to_bytes())?;
        lattice_pcd::util::absorb_fp(&mut t, b"pcep-val", &c.claimed)?;
    }
    let etas = lattice_pcd::util::challenge_fp_vec(&mut t, b"pcep-eta", claims.len())?;
    if etas != proof.etas {
        return Ok(false);
    }
    let combined = PedersenCommitment::linear_combine(
        &claims
            .iter()
            .enumerate()
            .map(|(i, c)| (etas[i], c.commitment))
            .collect::<Vec<_>>(),
    );
    if combined != proof.combined {
        return Ok(false);
    }
    // The single opening check.
    let opening = LinearOpening {
        commitment: proof.combined,
        encoding: proof.encoding.clone(),
        blind: proof.blind,
    };
    if !opening.verify(key, point, claimed_sum)? {
        return Ok(false);
    }
    Ok(true)
}

/// The verifier-side combined claim: `Σ ηᵢ·vᵢ`.
pub fn batch_claimed_sum(etas: &[Fp256], claims: &[PceClaim]) -> Fp256 {
    etas.iter()
        .zip(claims.iter())
        .fold(Fp256::ZERO, |acc, (e, c)| acc.add(&e.mul(&c.claimed)))
}

/// A convenience: the homomorphic matrix-commitment linear combination used
/// by the decider (§5.3): `[M] = Σ Σ ηⱼ·[Mⱼ^{(i)}]` over the per-function
/// matrix commitments.
pub fn combine_matrix_commitments(
    items: &[(Fp256, PcCommitment)],
) -> PcCommitment {
    PedersenCommitment::linear_combine(items)
}

/// Expose the raw MSM for callers building custom combinations.
pub fn scalar_mul_point(bases: &[G1Affine], scalars: &[Fp256]) -> G1Affine {
    msm(bases, scalars)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> Fp256 {
        Fp256::from_canonical_u64(v)
    }

    #[test]
    fn commit_eval_open_roundtrip() {
        let domain = Domain::Multivariate { num_vars: 3 };
        let key = PcKey::new(domain.clone(), &[51u8; 32]).ok().unwrap();
        let encoding: Vec<Fp256> = (0..8).map(|i| fr(i as u64 * 3 + 1)).collect();
        let mut t = Transcript::new_default(b"pc-test");
        let (c, wit) = key.commit_encoding(&encoding, &mut t).ok().unwrap();
        let point = vec![fr(3), fr(5), fr(7)];
        let val = key.eval_encoding(&encoding, &point).ok().unwrap();
        let opening = LinearOpening {
            commitment: c,
            encoding: wit.encoding.clone(),
            blind: wit.blind,
        };
        assert!(opening.verify(&key, &point, &val).ok().unwrap());
        // Wrong claim rejected.
        assert!(!opening.verify(&key, &point, &fr(999)).ok().unwrap());
        // Tampered encoding rejected (binding).
        let mut bad = opening.clone();
        bad.encoding[0] = bad.encoding[0].add(&fr(1));
        assert!(!bad.verify(&key, &point, &val).ok().unwrap());
    }

    #[test]
    fn batch_pce_roundtrip_and_tamper() {
        let domain = Domain::Multivariate { num_vars: 2 };
        let key = PcKey::new(domain.clone(), &[52u8; 32]).ok().unwrap();
        let enc1: Vec<Fp256> = (0..4).map(|i| fr(i as u64 + 5)).collect();
        let enc2: Vec<Fp256> = (0..4).map(|i| fr(i as u64 * 7 + 2)).collect();
        let mut t = Transcript::new_default(b"pc-batch");
        let (c1, w1) = key.commit_encoding(&enc1, &mut t).ok().unwrap();
        let (c2, w2) = key.commit_encoding(&enc2, &mut t).ok().unwrap();
        let point = vec![fr(3), fr(5)];
        let v1 = key.eval_encoding(&enc1, &point).ok().unwrap();
        let v2 = key.eval_encoding(&enc2, &point).ok().unwrap();
        let claims = vec![
            PceClaim {
                poly_id: 0,
                commitment: c1,
                claimed: v1,
            },
            PceClaim {
                poly_id: 1,
                commitment: c2,
                claimed: v2,
            },
        ];
        let witnesses = vec![(w1.encoding, w1.blind), (w2.encoding, w2.blind)];
        let mut t2 = Transcript::new_default(b"pc-batch");
        let proof =
            batch_prove_pce(&key, &point, &claims, &witnesses, &mut t2).ok().unwrap();
        let claimed_sum = batch_claimed_sum(&proof.etas, &claims);
        let mut t3 = Transcript::new_default(b"pc-batch");
        assert!(batch_verify_pcep(&key, &point, &claims, &claimed_sum, &proof, &mut t3)
            .ok()
            .unwrap());
        // Tampered combined encoding → reject.
        let mut bad = proof.clone();
        bad.encoding[1] = bad.encoding[1].add(&fr(1));
        let mut t4 = Transcript::new_default(b"pc-batch");
        assert!(!batch_verify_pcep(&key, &point, &claims, &claimed_sum, &bad, &mut t4)
            .ok()
            .unwrap());
        // Wrong claimed sum → reject.
        let mut t5 = Transcript::new_default(b"pc-batch");
        assert!(
            !batch_verify_pcep(&key, &point, &claims, &fr(12345), &proof, &mut t5)
                .ok()
                .unwrap()
        );
    }

    #[test]
    fn matrix_commitment_and_opening() {
        let domain = Domain::Multivariate { num_vars: 2 };
        let key = PcKey::new(domain.clone(), &[53u8; 32]).ok().unwrap();
        let m = crate::poly::fp_matrix(b"mpc", b"seed", 4);
        let c = key.commit_matrix(&domain, &m).ok().unwrap();
        let opening = key.open_matrix(&domain, &m).ok().unwrap();
        assert_eq!(c, opening.commitment);
        assert!(key.verify_matrix_opening(&domain, &opening).ok().unwrap());
        // Tampering the opening → reject.
        let mut bad = opening.clone();
        bad.encoding[0] = bad.encoding[0].add(&fr(1));
        // (re-commit check catches it only through the encoding mismatch:)
        let key2 = PedersenKey::derive(&key.key.seed, 16).ok().unwrap();
        let c_bad = key2.commit(&bad.encoding, &bad.blind).ok().unwrap();
        assert_ne!(c_bad, bad.commitment);
    }
}
