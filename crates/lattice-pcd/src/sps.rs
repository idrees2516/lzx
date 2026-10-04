//! The special-sound-protocol framework of §5.1 (ePrint 2026/289).
//!
//! A `(2µ−1)`-move special-sound protocol `Π_sps` for relation `R` has a
//! **homogeneous algebraic verifier map** (the paper's Eq. (2)):
//!
//! ```text
//! V_sps(x, [mᵢ]ᵢ∈[µ], [rᵢ]ᵢ∈[µ−1]) := Σ_{k=0..d} f_k^V(x, [m], [r]) ∈ Fⁿ
//! ```
//!
//! The committed-message NARK `FS[Π_sps^cm]` commits every prover message:
//! the proof is `π = (π.x = [Cᵢ]ᵢ∈[µ], [rᵢ]ᵢ∈[µ−1]), π.w = ([mᵢ]ᵢ∈[µ])` with
//! the predicate `Φ = V_NARK` checking (i) challenge derivation
//! `rᵢ = ρ_NARK(rᵢ₋₁, Cᵢ)`, (ii) `Cᵢ = Com(mᵢ)`, (iii) the map evaluates to
//! zero.
//!
//! Instances (the paper's Table 1 set, protocol designs from [BC23]):
//! * **R1CS** (d = 2, µ = 1): `m₁ = z = (1, x, w)`, map `Az∘Bz − Cz`, plus
//!   the constant-coordinate consistency `z₀ − 1` (whose relaxation at the
//!   interpolated point is exactly Nova's `u` drift — absorbed by the
//!   committed error term).
//! * **CCS** (d = q, µ = 1): `m₁ = z`, map `Σᵢ cᵢ ∘ⱼ∈Sᵢ (Mⱼ z)` — the
//!   high-degree headline case (ZK without the O(2^d) blowup).
//! * **Permutation / grand product** (d = n, µ = 2 — a
//!   challenge-*dependent* map): `m₁ = a`, challenge `ρ`, `m₂ = b`, map
//!   `[∏ᵢ(aᵢ+ρ) − ∏ᵢ(b_{π(i)}+ρ)]` — both products are homogeneous of
//!   degree n in `(a, b, ρ)`.
//!
//! **Map evaluation as a black box**: the accumulation prover only needs
//! `V_sps` at *concrete interpolated points* — `x(X), m(X), r(X)` are
//! eq-combinations of the parties' data, so `F(X) = V_sps(x(X), m(X),
//! r(X))` is evaluated by interpolating the inputs and applying the map.
//! Round polynomials of the sum-check (per-variable degree ≤ d) are then
//! interpolated from `d+2` point evaluations — the standard
//! `(d+2)·2^{L−j}` sum-check work.

use crate::pedersen::{PedersenCommitment, PedersenKey};
use crate::util::absorb_fp_slice;
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpsError {
    Shape(&'static str),
    Pedersen(crate::pedersen::PedersenError),
    Transcript(lattice_core::transcript::TranscriptError),
}

impl core::fmt::Display for SpsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SpsError::Shape(s) => write!(f, "shape error: {s}"),
            SpsError::Pedersen(e) => write!(f, "pedersen error: {e}"),
            SpsError::Transcript(e) => write!(f, "transcript error: {e}"),
        }
    }
}

impl From<crate::pedersen::PedersenError> for SpsError {
    fn from(e: crate::pedersen::PedersenError) -> Self {
        SpsError::Pedersen(e)
    }
}

impl From<lattice_core::transcript::TranscriptError> for SpsError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        SpsError::Transcript(e)
    }
}

/// A special-sound relation: the homogeneous algebraic map + shape data.
pub trait SpsRelation {
    /// Output length `n` of the map.
    fn num_outputs(&self) -> usize;
    /// Number of prover rounds `µ`.
    fn num_rounds(&self) -> usize;
    /// Message length of round `i` (0-based).
    fn msg_len(&self, round: usize) -> usize;
    /// Public instance length `s`.
    fn inst_len(&self) -> usize;
    /// Maximum total degree `d` of the map (sizes masks and round polys).
    fn degree(&self) -> usize;
    /// The algebraic map `V_sps` at concrete inputs.
    fn eval_map(
        &self,
        x: &[Fp256],
        msgs: &[Vec<Fp256>],
        challenges: &[Fp256],
    ) -> Result<Vec<Fp256>, SpsError>;
}

// ---------------------------------------------------------------------------
// R1CS
// ---------------------------------------------------------------------------

/// R1CS instance shape: `A z ∘ B z = C z` with `z = (1, x, w)`,
/// `x ∈ F^s`, `w ∈ F^t` — the paper's §2.2 running example.
#[derive(Clone, Debug)]
pub struct R1csSps {
    pub s: usize,
    pub t: usize,
    pub rows: usize,
    pub a: Vec<Vec<Fp256>>,
    pub b: Vec<Vec<Fp256>>,
    pub c: Vec<Vec<Fp256>>,
}

impl R1csSps {
    /// Random instance with the given density (for tests/benches).
    pub fn random(s: usize, t: usize, rows: usize, seed: &[u8]) -> R1csSps {
        let n = 1 + s + t;
        let gen = |label: &[u8]| -> Vec<Vec<Fp256>> {
            (0..rows)
                .map(|i| {
                    crate::util::fp_vec_from_seed(
                        label,
                        &[seed, &(i as u64).to_le_bytes()].concat(),
                        n,
                    )
                })
                .collect()
        };
        R1csSps {
            s,
            t,
            rows,
            a: gen(b"r1cs-a"),
            b: gen(b"r1cs-b"),
            c: gen(b"r1cs-c"),
        }
    }

    /// A relation with a large solution space: `A = 0`, `C = 0`, `B`
    /// random — every `z` with `z₀ = 1` satisfies (the map is `0 ∘ Bz − 0`
    /// plus the `z₀ − 1` coordinate). Used by multi-step/multi-instance
    /// protocol tests that need many distinct satisfying witnesses under a
    /// *fixed* index (the elementary-row `sample_satisfying` fixes exactly
    /// one witness per call).
    pub fn many_solutions(s: usize, t: usize, rows: usize, seed: &[u8]) -> R1csSps {
        let n = 1 + s + t;
        let b: Vec<Vec<Fp256>> = (0..rows)
            .map(|i| {
                crate::util::fp_vec_from_seed(
                    b"r1cs-b",
                    &[seed, &(i as u64).to_le_bytes()].concat(),
                    n,
                )
            })
            .collect();
        R1csSps {
            s,
            t,
            rows,
            a: vec![vec![Fp256::ZERO; n]; rows],
            b,
            c: vec![vec![Fp256::ZERO; n]; rows],
        }
    }

    /// Draw a random satisfying witness for a `many_solutions` relation.
    pub fn draw_solution(&self, seed: &[u8]) -> (Vec<Fp256>, Vec<Fp256>) {
        let n = 1 + self.s + self.t;
        let mut z = crate::util::fp_vec_from_seed(b"r1cs-sol", seed, n);
        z[0] = Fp256::from_canonical_u64(1);
        (z[1..1 + self.s].to_vec(), z[1 + self.s..].to_vec())
    }

    fn mat_vec(m: &[Vec<Fp256>], z: &[Fp256]) -> Vec<Fp256> {
        m.iter()
            .map(|row| {
                row.iter()
                    .zip(z.iter())
                    .fold(Fp256::ZERO, |acc, (a, b)| acc.add(&a.mul(b)))
            })
            .collect()
    }

    /// Sample a satisfying witness for a *random* instance by construction:
    /// draw `z` with `z₀ = 1`, derive `C z` from `A z ∘ B z` (the C rows are
    /// overwritten so the system is satisfied by construction). Returns
    /// `(x, w)`; `self.c` is mutated to the consistent matrix.
    pub fn sample_satisfying(&mut self, seed: &[u8]) -> Result<(Vec<Fp256>, Vec<Fp256>), SpsError> {
        let n = 1 + self.s + self.t;
        let mut z = crate::util::fp_vec_from_seed(b"r1cs-witness", seed, n);
        z[0] = Fp256::from_canonical_u64(1);
        let az = Self::mat_vec(&self.a, &z);
        let bz = Self::mat_vec(&self.b, &z);
        // C := diag(Az∘Bz) · ... — set each C row to pick out the product:
        // row i of C becomes e_{argmax} scaled? Simplest: C row i = the
        // elementary vector for a column j(i) times (Az∘Bz)_i / z_{j(i)}.
        for i in 0..self.rows {
            let prod = az[i].mul(&bz[i]);
            // pick a nonzero z coordinate to hang the product on.
            let mut placed = false;
            for j in 0..n {
                if !z[j].is_zero() {
                    let ratio = prod.mul(
                        &z[j]
                            .inverse()
                            .ok_or(SpsError::Shape("inverse of zero witness coordinate"))?,
                    );
                    let mut row = vec![Fp256::ZERO; n];
                    row[j] = ratio;
                    self.c[i] = row;
                    placed = true;
                    break;
                }
            }
            if !placed {
                // z is entirely zero except z₀=1: hang on z₀.
                let mut row = vec![Fp256::ZERO; n];
                row[0] = prod;
                self.c[i] = row;
            }
        }
        let x = z[1..1 + self.s].to_vec();
        let w = z[1 + self.s..].to_vec();
        Ok((x, w))
    }
}

impl SpsRelation for R1csSps {
    fn num_outputs(&self) -> usize {
        // rows + 1 + s: the map also carries the z₀ = 1 consistency
        // coordinate (the relaxed-u drift, absorbed by the committed
        // error) and the public-input consistency z[1..1+s] − x.
        self.rows + 1 + self.s
    }
    fn num_rounds(&self) -> usize {
        1
    }
    fn msg_len(&self, _round: usize) -> usize {
        1 + self.s + self.t
    }
    fn inst_len(&self) -> usize {
        self.s
    }
    fn degree(&self) -> usize {
        2
    }
    fn eval_map(
        &self,
        x: &[Fp256],
        msgs: &[Vec<Fp256>],
        _challenges: &[Fp256],
    ) -> Result<Vec<Fp256>, SpsError> {
        if msgs.len() != 1 || msgs[0].len() != 1 + self.s + self.t {
            return Err(SpsError::Shape("r1cs message shape"));
        }
        if x.len() != self.s {
            return Err(SpsError::Shape("r1cs instance shape"));
        }
        let z = &msgs[0];
        let az = Self::mat_vec(&self.a, z);
        let bz = Self::mat_vec(&self.b, z);
        let cz = Self::mat_vec(&self.c, z);
        let mut out = Vec::with_capacity(self.rows + 1);
        for i in 0..self.rows {
            out.push(az[i].mul(&bz[i]).sub(&cz[i]));
        }
        // z₀ − 1 (the u-consistency coordinate).
        out.push(z[0].sub(&Fp256::from_canonical_u64(1)));
        // z[1..1+s] − x (the public-input consistency).
        for i in 0..self.s {
            out.push(z[1 + i].sub(&x[i]));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// CCS — the high-degree case
// ---------------------------------------------------------------------------

/// Customizable Constraint System shape (Setty–Thaler–Wahby): the witness
/// constraint `Σ_{i∈[q]} cᵢ ∘_{j∈Sᵢ} (Mⱼ z) = 0` with `z = (x, w)`.
#[derive(Clone, Debug)]
pub struct CcsSps {
    pub s: usize,
    pub t: usize,
    pub rows: usize,
    /// Matrices `M₁..M_{t_M}`.
    pub matrices: Vec<Vec<Vec<Fp256>>>,
    /// Constants `c₁..c_q`.
    pub constants: Vec<Fp256>,
    /// Multisets `S₁..S_q` over matrix indices.
    pub sets: Vec<Vec<usize>>,
}

impl CcsSps {
    /// A random CCS with `t_M` matrices and max set size `d`.
    pub fn random(
        s: usize,
        t: usize,
        rows: usize,
        t_m: usize,
        q: usize,
        d: usize,
        seed: &[u8],
    ) -> CcsSps {
        let n = s + t;
        let matrices: Vec<Vec<Vec<Fp256>>> = (0..t_m)
            .map(|j| {
                (0..rows)
                    .map(|i| {
                        crate::util::fp_vec_from_seed(
                            b"ccs-matrix",
                            &[seed, &(j as u64).to_le_bytes(), &(i as u64).to_le_bytes()].concat(),
                            n,
                        )
                    })
                    .collect()
            })
            .collect();
        let constants = crate::util::fp_vec_from_seed(b"ccs-const", seed, q);
        let sets: Vec<Vec<usize>> = (0..q)
            .map(|i| {
                let bytes = Transcript::xof(
                    b"ccs-set",
                    &[seed, &(i as u64).to_le_bytes()].concat(),
                    d * 4,
                );
                (0..d)
                    .map(|k| {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(&bytes[k * 4..k * 4 + 4]);
                        (u32::from_le_bytes(b) as usize) % t_m.max(1)
                    })
                    .collect()
            })
            .collect();
        CcsSps {
            s,
            t,
            rows,
            matrices,
            constants,
            sets,
        }
    }

    /// Mutating sampler: draws a witness and zeroes each set's first matrix
    /// so the Hadamard products vanish; returns `(x, w)`.
    pub fn sample_satisfying_mut(&mut self, seed: &[u8]) -> (Vec<Fp256>, Vec<Fp256>) {
        let n = self.s + self.t;
        let mut z = crate::util::fp_vec_from_seed(b"ccs-witness", seed, n);
        if z.iter().all(|v| v.is_zero()) {
            z[0] = Fp256::from_canonical_u64(1);
        }
        let firsts: Vec<usize> = {
            let mut f: Vec<usize> = self
                .sets
                .iter()
                .filter(|s| !s.is_empty())
                .map(|s| s[0])
                .collect();
            f.sort_unstable();
            f.dedup();
            f
        };
        for j in firsts {
            for row in self.matrices[j].iter_mut() {
                for v in row.iter_mut() {
                    *v = Fp256::ZERO;
                }
            }
        }
        (z[..self.s].to_vec(), z[self.s..].to_vec())
    }
}

impl SpsRelation for CcsSps {
    fn num_outputs(&self) -> usize {
        // rows + s: the constraint rows plus the public-input consistency
        // z[..s] − x.
        self.rows + self.s
    }
    fn num_rounds(&self) -> usize {
        1
    }
    fn msg_len(&self, _round: usize) -> usize {
        self.s + self.t
    }
    fn inst_len(&self) -> usize {
        self.s
    }
    fn degree(&self) -> usize {
        self.sets.iter().map(|s| s.len()).max().unwrap_or(1)
    }
    fn eval_map(
        &self,
        x: &[Fp256],
        msgs: &[Vec<Fp256>],
        _challenges: &[Fp256],
    ) -> Result<Vec<Fp256>, SpsError> {
        if msgs.len() != 1 || msgs[0].len() != self.s + self.t {
            return Err(SpsError::Shape("ccs message shape"));
        }
        if x.len() != self.s {
            return Err(SpsError::Shape("ccs instance shape"));
        }
        let z = &msgs[0];
        // Per-matrix products Mⱼz.
        let mzs: Vec<Vec<Fp256>> = self
            .matrices
            .iter()
            .map(|m| {
                m.iter()
                    .map(|row| {
                        row.iter()
                            .zip(z.iter())
                            .fold(Fp256::ZERO, |acc, (a, b)| acc.add(&a.mul(b)))
                    })
                    .collect()
            })
            .collect();
        let mut out = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            let mut total = Fp256::ZERO;
            for (i, s_i) in self.sets.iter().enumerate() {
                if s_i.is_empty() {
                    continue;
                }
                let mut prod = Fp256::from_canonical_u64(1);
                for &j in s_i {
                    prod = prod.mul(&mzs[j][r]);
                }
                total = total.add(&self.constants[i].mul(&prod));
            }
            out.push(total);
        }
        // z[..s] − x.
        for i in 0..self.s {
            out.push(z[i].sub(&x[i]));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Permutation / grand product — the challenge-dependent µ = 2 case
// ---------------------------------------------------------------------------

/// Prove `a = π(b)` for a public permutation `π` via the grand-product
/// check `∏(aᵢ + ρ) = ∏(b_{π(i)} + ρ)` at a random challenge `ρ`
/// (degree n — the high-degree, challenge-dependent showcase).
#[derive(Clone, Debug)]
pub struct PermutationSps {
    pub n: usize,
    /// `perm[i]` = the source index in `b` for output position `i`.
    pub perm: Vec<usize>,
}

impl PermutationSps {
    pub fn random(n: usize, seed: &[u8]) -> PermutationSps {
        let bytes = Transcript::xof(b"perm", seed, n * 4);
        let mut idx: Vec<usize> = (0..n).collect();
        // Fisher–Yates from the XOF stream.
        for i in (1..n).rev() {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[(n - 1 - i) * 4..(n - 1 - i) * 4 + 4]);
            let j = (u32::from_le_bytes(b) as usize) % (i + 1);
            idx.swap(i, j);
        }
        PermutationSps { n, perm: idx }
    }

    /// Apply the permutation: `out[i] = b[perm[i]]`.
    pub fn apply(&self, b: &[Fp256]) -> Result<Vec<Fp256>, SpsError> {
        if b.len() != self.n {
            return Err(SpsError::Shape("permutation vector length"));
        }
        Ok(self.perm.iter().map(|&i| b[i]).collect())
    }
}

impl SpsRelation for PermutationSps {
    fn num_outputs(&self) -> usize {
        1
    }
    fn num_rounds(&self) -> usize {
        2
    }
    fn msg_len(&self, _round: usize) -> usize {
        self.n
    }
    fn inst_len(&self) -> usize {
        0
    }
    fn degree(&self) -> usize {
        self.n
    }
    fn eval_map(
        &self,
        _x: &[Fp256],
        msgs: &[Vec<Fp256>],
        challenges: &[Fp256],
    ) -> Result<Vec<Fp256>, SpsError> {
        if msgs.len() != 2 || challenges.len() != 1 {
            return Err(SpsError::Shape("permutation transcript shape"));
        }
        let (a, b) = (&msgs[0], &msgs[1]);
        if a.len() != self.n || b.len() != self.n {
            return Err(SpsError::Shape("permutation vector length"));
        }
        let rho = &challenges[0];
        let mut lhs = Fp256::from_canonical_u64(1);
        for v in a {
            lhs = lhs.mul(&v.add(rho));
        }
        let mut rhs = Fp256::from_canonical_u64(1);
        for &i in &self.perm {
            rhs = rhs.mul(&b[i].add(rho));
        }
        Ok(vec![lhs.sub(&rhs)])
    }
}

// ---------------------------------------------------------------------------
// The committed-message SPS transcript (NARK proof structure)
// ---------------------------------------------------------------------------

/// A predicate instance `qx = (x, [Cᵢ]^µ, [rᵢ]^{µ−1})`.
#[derive(Clone, Debug)]
pub struct SpsInstance {
    pub x: Vec<Fp256>,
    pub commitments: Vec<PedersenCommitment>,
    pub challenges: Vec<Fp256>,
}

/// A predicate witness `qw = ([mᵢ]^µ, blinds)`.
#[derive(Clone, Debug)]
pub struct SpsWitness {
    pub messages: Vec<Vec<Fp256>>,
    pub blinds: Vec<Fp256>,
}

/// Derive the NARK challenge chain `r₁ = ρ(x)`, `rᵢ = ρ(rᵢ₋₁, Cᵢ)` —
/// the predicate's check (i) of §5.2. **Stateless** (a pure function of the
/// instance): the random oracle is queried on the serialized inputs only,
/// so re-derivation in any context agrees.
pub fn derive_challenges(inst: &SpsInstance) -> Result<Vec<Fp256>, SpsError> {
    let mu = inst.commitments.len();
    let mut out = Vec::with_capacity(mu.saturating_sub(1));
    let mut prev: Vec<u8> = Vec::new();
    for i in 0..mu.saturating_sub(1) {
        let mut input = Vec::new();
        if i == 0 {
            for v in &inst.x {
                input.extend_from_slice(&v.from_mont().canon_bytes());
            }
        } else {
            input.extend_from_slice(&prev);
        }
        input.extend_from_slice(&inst.commitments[i].to_bytes());
        let h = Transcript::xof(b"nark-challenge", &input, 32);
        let mut b = [0u8; 32];
        b.copy_from_slice(&h);
        let r = crate::util::fp_from_be32(&b);
        prev = r.from_mont().canon_bytes().to_vec();
        out.push(r);
    }
    Ok(out)
}

/// Check the predicate `Φ = V_NARK` for a given relation: challenge
/// consistency + commitments + map = 0.
pub fn check_predicate<R: SpsRelation>(
    rel: &R,
    inst: &SpsInstance,
    wit: &SpsWitness,
    key: &PedersenKey,
    _transcript: &mut Transcript,
) -> Result<bool, SpsError> {
    if inst.commitments.len() != rel.num_rounds()
        || wit.messages.len() != rel.num_rounds()
        || wit.blinds.len() != rel.num_rounds()
    {
        return Ok(false);
    }
    let derived = derive_challenges(inst)?;
    if derived != inst.challenges {
        return Ok(false);
    }
    for i in 0..rel.num_rounds() {
        if !key.verify_opening(&inst.commitments[i], &wit.messages[i], &wit.blinds[i])? {
            return Ok(false);
        }
    }
    let map = rel.eval_map(&inst.x, &wit.messages, &inst.challenges)?;
    Ok(map.iter().all(|v| v.is_zero()))
}

/// Produce the committed-message transcript for a prover with messages in
/// hand: commitments with fresh transcript-derived blinds, then the
/// challenge chain.
pub fn commit_messages(
    messages: &[Vec<Fp256>],
    key: &PedersenKey,
    transcript: &mut Transcript,
) -> Result<(SpsInstance, SpsWitness), SpsError> {
    let mut commitments = Vec::with_capacity(messages.len());
    let mut blinds = Vec::with_capacity(messages.len());
    let mut inst_coms = Vec::with_capacity(messages.len());
    for m in messages.iter() {
        absorb_fp_slice(transcript, b"sps-msg", m)?;
        let (c, r) = key.commit_fresh(m, transcript)?;
        commitments.push(c);
        inst_coms.push(c);
        blinds.push(r);
        transcript.append_message(b"sps-com", &c.to_bytes())?;
    }
    Ok((
        SpsInstance {
            x: Vec::new(), // caller fills
            commitments,
            challenges: Vec::new(), // caller fills via derive
        },
        SpsWitness {
            messages: messages.to_vec(),
            blinds,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> Fp256 {
        Fp256::from_canonical_u64(v)
    }

    #[test]
    fn r1cs_satisfaction_and_violation() {
        let mut rel = R1csSps::random(2, 6, 4, b"seed-r1cs");
        let (x, w) = rel.sample_satisfying(b"w0").ok().unwrap();
        let mut z = vec![fr(1)];
        z.extend(x.iter().cloned());
        z.extend(w.iter().cloned());
        let map = rel.eval_map(&x, &[z.clone()], &[]).ok().unwrap();
        assert!(map.iter().all(|v| v.is_zero()));
        // Tampered witness:
        let mut z2 = z.clone();
        z2[3] = z2[3].add(&fr(1));
        let map2 = rel.eval_map(&x, &[z2], &[]).ok().unwrap();
        assert!(map2.iter().any(|v| !v.is_zero()));
    }

    #[test]
    fn ccs_satisfaction() {
        let mut ccs = CcsSps::random(2, 6, 4, 3, 3, 3, b"seed-ccs");
        let (x, w) = ccs.sample_satisfying_mut(b"w0");
        let mut z = x.clone();
        z.extend(w.iter().cloned());
        let map = ccs.eval_map(&x, &[z], &[]).ok().unwrap();
        assert!(map.iter().all(|v| v.is_zero()), "map = {:?}", map);
        assert_eq!(ccs.degree(), 3);
    }

    #[test]
    fn permutation_map() {
        let perm = PermutationSps {
            n: 4,
            perm: vec![2, 0, 3, 1],
        };
        let b = vec![fr(10), fr(20), fr(30), fr(40)];
        let a = perm.apply(&b).ok().unwrap(); // a = (30, 10, 40, 20)
        let rho = fr(7);
        // a = π(b): map = 0.
        let ok_map = perm
            .eval_map(&[], &[a.clone(), b.clone()], &[rho])
            .ok()
            .unwrap();
        assert!(ok_map.iter().all(|v| v.is_zero()));
        // wrong a: nonzero whp.
        let mut bad = a.clone();
        bad[0] = fr(999);
        let bad_map = perm.eval_map(&[], &[bad, b], &[rho]).ok().unwrap();
        assert!(bad_map.iter().any(|v| !v.is_zero()));
    }

    #[test]
    fn challenge_derivation_consistent() {
        let key = PedersenKey::derive(&[3u8; 32], 16).ok().unwrap();
        let mut t1 = Transcript::new_default(b"sps-test");
        let mut t2 = Transcript::new_default(b"sps-test");
        let msgs = vec![vec![fr(1), fr(2), fr(3)], vec![fr(4), fr(5)]];
        let (mut inst, wit) = commit_messages(&msgs, &key, &mut t1).ok().unwrap();
        inst.x = vec![fr(9)];
        inst.challenges = derive_challenges(&inst).ok().unwrap();
        // Stateless re-derivation must agree.
        let again = derive_challenges(&inst).ok().unwrap();
        assert_eq!(inst.challenges, again);
        let _ = &mut t2;
        // Predicate holds when the map does (identity-like messages on the
        // permutation relation with a satisfying pair):
        let perm = PermutationSps {
            n: 3,
            perm: vec![1, 2, 0],
        };
        let b = vec![fr(5), fr(6), fr(7)];
        let a = perm.apply(&b).ok().unwrap();
        let mut t3 = Transcript::new_default(b"sps-test2");
        let (mut inst2, wit2) = commit_messages(&[a, b], &key, &mut t3).ok().unwrap();
        inst2.challenges = derive_challenges(&inst2).ok().unwrap();
        assert!(check_predicate(&perm, &inst2, &wit2, &key, &mut t3)
            .ok()
            .unwrap());
        let _ = wit;
    }
}
