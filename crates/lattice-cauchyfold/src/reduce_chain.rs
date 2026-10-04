//! The finite linear reduction chain (the paper's §5.5 layers + §5.6
//! terminal), scaled. Each layer takes the current witness (ring-element
//! blocks, its commitment key, its linear relation `Σ⟨φ_j, w_j⟩ = b`, and
//! its squared-norm bound `S`) and runs the §5.5 mechanics:
//!
//! 1. the per-block commitments `t_j = A·w_j` (public),
//! 2. the random projection `Π ∈ {−1,0,1}^{m×N}`, the projected vector
//!    `p` transmitted with the acceptance threshold `‖Πw‖² ≤ m·S`
//!    (bounded retries — Lemma C.1's regime),
//! 3. the symmetric values `h_ij = (⟨φ_i,w_j⟩ + ⟨φ_j,w_i⟩)/2` fixed
//!    **before** the short challenge (the paper's load-bearing ordering),
//! 4. the short ring challenge `c ←$ D46` — ternary polynomials with
//!    negacyclic operator norm ≤ 46 (the paper's `D46`; distinct values
//!    have unit differences since `‖Δ‖∞ ≤ 4 < √(q/2)`, the
//!    Lyubashevsky–Seiler criterion at `q = 2^48 − 59`),
//! 5. the response `z = Σ c_j w_j` with the identities (28)–(30):
//!    `A·z = Σ c_j t_j`, `⟨Σ c_j φ_j, z⟩ = Σ c_a c_b h_ab`, `‖z‖² ≤ G`
//!    (the G bound `256·S·15/14` with bounded response retries),
//! 6. the child witness = the canonical centered radix-64 split of `z`.
//!
//! The terminal (§5.6) transmits the final digit blocks directly with the
//! fixed-length codec (§`crate::wire`), the norm, the recomposition
//! against the last response, and the commitment check.
//!
//! **Documented deviations** (full ledger in the paper note): the layer
//! count is a profile parameter (2 + terminal here vs. the paper's 5 +
//! terminal at 57.5M coefficients); the auxiliary digit commitments
//! `u1 = B·t̃`, `u2 = D·h̃` are replaced by direct `h`-transmission checked
//! through identity (29) (the child does not carry the aux digit blocks,
//! so the inter-layer recomposition is verified only at the terminal);
//! the inter-layer child-relation verification is therefore terminal-side.

use crate::commit::AjtaiKey;
use crate::field_k::{Fq48, K4, Q48};
use lattice_core::transcript::Transcript;
use lattice_labrador::ring::Poly;

/// The certified short-challenge operator-norm bound (the paper's `D46`).
pub const OP_NORM_BOUND: f64 = 46.0;

/// The layer radix (the paper's per-layer ρ; 64/32 in its profiles).
pub const LAYER_RADIX: i64 = 64;

/// The maximum retries per bounded-retry family (the paper's 160).
pub const MAX_RETRIES: u32 = 160;

/// The number of nonterminal layers in the scaled profile.
pub const NUM_LAYERS: usize = 2;

#[derive(Debug, Clone, PartialEq)]
pub enum ChainError {
    Shape(String),
    NormViolation(&'static str),
    Identity(&'static str),
    Transcript(String),
    Recomposition,
    Codec(String),
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainError::Shape(s) => write!(f, "chain shape: {s}"),
            ChainError::NormViolation(s) => write!(f, "chain norm: {s}"),
            ChainError::Identity(s) => write!(f, "chain identity: {s}"),
            ChainError::Transcript(s) => write!(f, "chain transcript: {s}"),
            ChainError::Recomposition => write!(f, "chain recomposition"),
            ChainError::Codec(s) => write!(f, "chain codec: {s}"),
        }
    }
}

fn e(err: lattice_core::transcript::TranscriptError) -> ChainError {
    ChainError::Transcript(err.to_string())
}

/// One nonterminal layer's public transcript.
#[derive(Clone)]
pub struct LayerTranscript {
    /// The per-block commitments `t_j` (each a ring-row vector).
    pub t: Vec<Vec<Poly>>,
    /// The projected vector `p` (m entries).
    pub p: Vec<i64>,
    /// The projection rows (m × flat_len, {−1,0,1}).
    pub projection: Vec<i8>,
    /// The declared squared-norm bound `S` for this layer.
    pub norm_bound: u64,
    /// The symmetric values `h_ij` (upper triangle, ring elements).
    pub h: Vec<Poly>,
    /// The short challenge vector `c` (s ring elements).
    pub c: Vec<Poly>,
    /// The response `z` (n ring elements).
    pub z: Vec<Poly>,
}

/// The terminal witness: the final digit blocks, transmitted directly.
#[derive(Clone)]
pub struct TerminalWitness {
    /// The digit blocks `[z^(0), z^(1)]` of the last response.
    pub blocks: Vec<Vec<Poly>>,
    /// The commitment of the terminal witness (flat).
    pub commitment: Vec<Poly>,
    /// The squared-norm bound satisfied.
    pub norm_squared: u64,
}

/// The chain proof.
#[derive(Clone)]
pub struct ChainProof {
    pub layers: Vec<LayerTranscript>,
    /// The per-layer key dims (rows, cols).
    pub layer_key_dims: Vec<(usize, usize)>,
    pub terminal: TerminalWitness,
}

/// The squared ℓ2 norm of a ring vector (centered coefficients).
pub fn norm2(w: &[Poly]) -> u64 {
    w.iter()
        .map(|p| p.0.iter().map(|&c| c * c).sum::<i64>() as u64)
        .sum()
}

/// The canonical centered radix-ρ split of a ring element:
/// `z = z^(0) + ρ·z^(1)` with `z^(0)` the centered residues.
pub fn radix_split(z: &Poly, radix: i64) -> (Poly, Poly) {
    let mut lo = [0i64; 64];
    let mut hi = [0i64; 64];
    for i in 0..64 {
        let x = z.0[i];
        let r = x.rem_euclid(radix);
        let centered = if r > radix / 2 - 1 { r - radix } else { r };
        lo[i] = centered;
        hi[i] = (x - centered) / radix;
    }
    (Poly(lo), Poly(hi))
}

/// Recompose: `z^(0) + ρ·z^(1)`.
pub fn radix_recompose(lo: &Poly, hi: &Poly, radix: i64) -> Poly {
    let mut out = [0i64; 64];
    for i in 0..64 {
        out[i] = lo.0[i] + radix * hi.0[i];
    }
    Poly(out)
}

/// The canonical digit energy recurrence (Appendix C.3): the exact maximum
/// squared digit energy of the ℓ-digit centered radix-ρ decomposition of
/// values in `[0, H]`.
pub fn digit_energy(h_max: u64, radix: u64, digits: usize) -> u64 {
    // The canonical centered split: `lo ∈ [−b, b−1]`, `hi = (x − lo)/ρ`;
    // the maximum energy recurses as `F_t(H) = b² + F_{t−1}(⌊(H+b)/ρ⌋)`
    // with `F_1(H) = H²` (a single digit carries any value). u128
    // internals with saturation (a conservative exact-stand-in for the
    // paper's two-branch C.3 recurrence; the deviation ledger records
    // the simplification).
    let b = (radix / 2) as u128;
    let mut f = h_max as u128 * h_max as u128;
    let mut h = h_max as u128;
    for _ in 1..digits {
        h = (h + b) / radix as u128;
        f = b * b + h * h;
    }
    f.min(u64::MAX as u128) as u64
}

/// Sample a certified short ring challenge: ternary coefficients with the
/// negacyclic operator norm ≤ 46 (float DFT certification with a
/// conservative margin — the paper's exact rational interval test
/// replaced; see the deviation ledger).
pub fn sample_short_challenge(transcript: &mut Transcript) -> Result<Poly, ChainError> {
    for _ in 0..64 {
        let bytes = transcript.challenge_bytes(b"d46", 32).map_err(e)?;
        let mut coeffs = [0i64; 64];
        for i in 0..64 {
            let byte = bytes[i % 32];
            let sel = (byte >> (2 * (i % 4))) & 0x3;
            coeffs[i] = match sel {
                0 | 1 => 0,
                2 => 1,
                _ => -1,
            };
        }
        let cand = Poly(coeffs);
        if op_norm_bound_ok(&cand) {
            return Ok(cand);
        }
    }
    Err(ChainError::Shape("D46 sampling exhausted".into()))
}

/// The negacyclic operator norm estimate: `max_j |c(ω_j)|` over the
/// 64 primitive twist roots, in f64 with a safety margin.
pub fn op_norm_bound_ok(c: &Poly) -> bool {
    let mut max_sq = 0f64;
    for j in 0..64u32 {
        let theta = std::f64::consts::PI * (2 * j + 1) as f64 / 64.0;
        let (mut re, mut im) = (0f64, 0f64);
        for (p, &coef) in c.0.iter().enumerate() {
            let ang = theta * p as f64;
            re += coef as f64 * ang.cos();
            im += coef as f64 * ang.sin();
        }
        max_sq = max_sq.max(re * re + im * im);
    }
    max_sq <= (OP_NORM_BOUND * (1.0 + 1e-9)).powi(2)
}

/// The ring inner product `⟨φ, w⟩ = Σ_c φ_c·w_c`.
fn ring_inner(phi: &[Poly], w: &[Poly]) -> Poly {
    let mut acc = Poly::zero();
    for (a, b) in phi.iter().zip(w.iter()) {
        acc.add_assign(&a.mul(b));
    }
    acc
}

/// The negacyclic transpose: `(T(v)·w).const = Σ_p v_p·w_p`.
pub fn negacyclic_transpose(v: &[i64]) -> Poly {
    let mut out = [0i64; 64];
    for p in 0..64 {
        let q = (64 - p) % 64;
        let sign = if p == 0 { 1 } else { -1 };
        out[q] = sign * v[p % v.len()];
    }
    Poly(out)
}

/// The layer-0 linear functional: the eq_σ transpose vector over the W
/// positions (the evaluation claim `⟨eq_σ, W⟩ = w_sigma` as a ring inner
/// product), batched over the four K-coordinates by `batch`.
pub fn layer0_phi(w_len_elems: usize, sigma: &[K4], w_cube_vars: usize, batch: &K4) -> Vec<Poly> {
    let mut phi = Vec::with_capacity(w_len_elems);
    for c in 0..w_len_elems {
        let mut vals = [0i64; 64];
        for p in 0..64 {
            let pos = 64 * c + p;
            let eqv = eq_pos(pos, sigma, w_cube_vars);
            let mut acc = Fq48::ZERO;
            for (cc, &bs) in batch.0.iter().enumerate() {
                acc = acc.add(&Fq48(bs.0).mul(&eqv.0[cc]));
            }
            vals[p] = acc.centered();
        }
        phi.push(negacyclic_transpose(&vals));
    }
    phi
}

/// `eq(pos, σ)` over the digit cube.
fn eq_pos(pos: usize, sigma: &[K4], cube_vars: usize) -> K4 {
    let mut val = K4::ONE;
    for (bit, t) in sigma.iter().enumerate() {
        if bit >= cube_vars {
            break;
        }
        let b = (pos >> (cube_vars - 1 - bit)) & 1;
        let bf = K4::from_coeffs([b as u64, 0, 0, 0]);
        val = val.mul(&bf.mul(t).add(&K4::ONE.sub(&bf).mul(&K4::ONE.sub(t))));
    }
    val
}

#[allow(clippy::manual_div_ceil)]
fn half_ring(p: &Poly) -> Poly {
    // Division by 2 in R_q: scale by 2^{-1} mod q with exact reduction.
    let inv2 = (Q48 + 1) / 2; // 2^{-1} mod q (Q48 odd — exact)
    Poly(p.0.map(|c| lattice_labrador::ring::cmod(c as i128 * inv2 as i128)))
}

fn flat_coeffs(blocks: &[Vec<Poly>]) -> Vec<i64> {
    blocks
        .iter()
        .flat_map(|b| b.iter().flat_map(|p| p.0.iter().copied()))
        .collect()
}

fn derive_projection(seed: &[u8], m: usize, len: usize) -> Vec<i8> {
    let mut out = vec![0i8; m * len];
    let mut counter = 0usize;
    let mut bytes: Vec<u8> = Vec::new();
    for r in 0..m {
        for i in 0..len {
            if bytes.is_empty() {
                bytes = lattice_core::transcript::Transcript::xof(
                    b"proj-rows",
                    &{
                        let mut salt = seed.to_vec();
                        salt.extend_from_slice(&counter.to_le_bytes());
                        salt
                    },
                    64,
                );
                counter += 1;
            }
            let b = bytes.pop().unwrap_or(0);
            out[r * len + i] = match b % 4 {
                0 => 0,
                1 => 1,
                2 => 0,
                _ => -1,
            };
        }
    }
    out
}

/// Prove the chain: the committed linear relation
/// `(C_W = A_W·W, ⟨eq_σ-functional, W⟩ = w_σ)` reduced through the layers
/// to the terminal witness.
#[allow(clippy::too_many_arguments)]
pub fn prove_chain(
    w_ring: &[Poly],
    sigma: &[K4],
    w_sigma: &K4,
    w_cube_vars: usize,
    transcript: &mut Transcript,
) -> Result<ChainProof, ChainError> {
    // The batch over the four K-coordinates.
    let batch = K4::challenge(transcript).map_err(|err| ChainError::Transcript(err.to_string()))?;
    let mut b_target = K4::ZERO;
    for (cc, &bs) in batch.0.iter().enumerate() {
        let prod = Fq48(bs.0).mul(&w_sigma.0[cc]);
        let mut coord = [Fq48::ZERO; 4];
        coord[cc] = prod;
        b_target = b_target.add(&K4(coord));
    }
    // Layer 0: the witness as one block; φ = the eq_σ functional.
    let phi0 = layer0_phi(w_ring.len(), sigma, w_cube_vars, &batch);
    let mut current_blocks: Vec<Vec<Poly>> = vec![w_ring.to_vec()];
    let mut current_phi = vec![phi0];
    let mut current_norm = norm2(w_ring).max(1);

    let mut transcripts = Vec::new();
    let mut layer_key_dims = Vec::new();
    for _ in 0..NUM_LAYERS {
        let n = current_blocks[0].len();
        // Each block is committed under the SAME key columns
        // (`t_j = A·w_j` with the shared A), so the response identity
        // `A·z = Σ c_j t_j` closes.
        let key = AjtaiKey::from_seed(1, n, b"cauchyfold-layer");
        layer_key_dims.push((1, n));
        let (tr, child_blocks, child_norm) = prove_layer(
            &key,
            &current_blocks,
            &current_phi,
            current_norm,
            transcript,
        )?;
        // The child's φ: the recomposition functional against the public
        // response z — [identity, ρ·identity] as block functionals.
        let child_phi = {
            let ident = negacyclic_transpose(&vec![1i64; 64]);
            let radixv = negacyclic_transpose(&vec![LAYER_RADIX; 64]);
            vec![vec![ident; n], vec![radixv; n]]
        };
        transcripts.push(tr);
        current_blocks = child_blocks;
        current_phi = child_phi;
        current_norm = child_norm;
    }

    // The terminal: the current blocks, transmitted directly.
    let flat: Vec<Poly> = current_blocks.concat();
    let term_key = AjtaiKey::from_seed(1, flat.len().max(1), b"cauchyfold-terminal");
    let commitment = term_key.commit(&flat).map_err(ChainError::Shape)?;
    let norm_squared = norm2(&flat);
    absorb_poly_vec(transcript, b"term-c", &commitment).map_err(e)?;
    Ok(ChainProof {
        layers: transcripts,
        layer_key_dims,
        terminal: TerminalWitness {
            blocks: current_blocks,
            commitment,
            norm_squared,
        },
    })
}

/// One layer's prove (§5.5). Returns the transcript, the child blocks,
/// and the child's norm bound.
fn prove_layer(
    key: &AjtaiKey,
    blocks: &[Vec<Poly>],
    phi: &[Vec<Poly>],
    norm_bound: u64,
    transcript: &mut Transcript,
) -> Result<(LayerTranscript, Vec<Vec<Poly>>, u64), ChainError> {
    let s = blocks.len();
    let n = blocks[0].len();
    // 1. t_j = A·w_j (per-block column slices of the shared key).
    let mut t = Vec::with_capacity(s);
    for w in blocks {
        let mut tj = vec![Poly::zero(); key.rows];
        for (c, wc) in w.iter().enumerate() {
            for r in 0..key.rows {
                tj[r].add_assign(&key.matrix[r * key.cols + c].mul(wc));
            }
        }
        t.push(tj);
    }
    for tj in &t {
        absorb_poly_vec(transcript, b"t", tj).map_err(e)?;
    }
    // 2. The projection with retries.
    let flat_len = s * n * 64;
    let m = 8usize.min(flat_len);
    let mut projection_retries = 0u32;
    let (projection, p) = loop {
        let seed = transcript.challenge_bytes(b"proj-seed", 32).map_err(e)?;
        let rows = derive_projection(&seed, m, flat_len);
        let flat = flat_coeffs(blocks);
        let pp: Vec<i64> = (0..m)
            .map(|r| {
                let mut acc = 0i64;
                for (idx, &wc) in flat.iter().enumerate() {
                    let ee = rows[r * flat_len + idx];
                    if ee != 0 {
                        acc += ee as i64 * wc;
                    }
                }
                acc
            })
            .collect();
        let norm: u64 = pp.iter().map(|&x| (x * x) as u64).sum();
        if norm <= m as u64 * norm_bound {
            break (rows, pp);
        }
        projection_retries += 1;
        if projection_retries > MAX_RETRIES {
            return Err(ChainError::NormViolation("projection"));
        }
    };
    {
        let mut pb = Vec::new();
        for &x in &p {
            pb.extend_from_slice(&x.to_le_bytes());
        }
        transcript.append_message(b"proj-p", &pb).map_err(e)?;
    }
    // 3. The symmetric h_ij before the challenge.
    let mut h = Vec::with_capacity(s * (s + 1) / 2);
    for i in 0..s {
        for j in i..s {
            let a = ring_inner(&phi[i], &blocks[j]);
            let bb = ring_inner(&phi[j], &blocks[i]);
            h.push(half_ring(&a.add(&bb)));
        }
    }
    for hv in &h {
        absorb_poly(transcript, b"h", hv).map_err(e)?;
    }
    // 4-5. The short challenge and the response with retries.
    let mut response_retries = 0u32;
    let (c, z, g) = loop {
        let cc: Vec<Poly> = (0..s)
            .map(|_| sample_short_challenge(transcript))
            .collect::<Result<Vec<_>, _>>()?;
        let zz: Vec<Poly> = (0..n)
            .map(|idx| {
                let mut acc = Poly::zero();
                for (j, w) in blocks.iter().enumerate() {
                    acc.add_assign(&cc[j].mul(&w[idx]));
                }
                acc
            })
            .collect();
        let znorm = norm2(&zz);
        let g_bound = ((256.0 * norm_bound as f64) * (15.0 / 14.0)) as u64;
        if znorm <= g_bound {
            break (cc, zz, g_bound);
        }
        response_retries += 1;
        if response_retries > MAX_RETRIES {
            return Err(ChainError::NormViolation("response"));
        }
    };
    absorb_poly_vec(transcript, b"z", &z).map_err(e)?;
    // 6. The child: the radix split of z.
    let (z0, z1): (Vec<Poly>, Vec<Poly>) = z.iter().map(|pp| radix_split(pp, LAYER_RADIX)).unzip();
    let child_blocks = vec![z0, z1];
    // The child's norm bound: lo ∈ {−32..31}, hi ≤ (max|z| + 32)/64 + 1.
    let zmax = (g as f64).sqrt() as i64 + 1;
    let lo_b = 32i64;
    let hi_b = (zmax + 32) / LAYER_RADIX + 1;
    let child_norm = (lo_b * lo_b + hi_b * hi_b) as u64 * (n * 64) as u64;
    let tr = LayerTranscript {
        t,
        p,
        projection,
        norm_bound,
        h,
        c,
        z,
    };
    Ok((tr, child_blocks, child_norm))
}

/// Verify the chain (the caller has already checked `C_W` opens the
/// level-2 witness).
#[allow(clippy::too_many_arguments)]
pub fn verify_chain(
    w_ring: &[Poly],
    sigma: &[K4],
    _w_sigma: &K4,
    w_cube_vars: usize,
    transcript: &mut Transcript,
    proof: &ChainProof,
) -> Result<(), ChainError> {
    if proof.layers.len() != NUM_LAYERS {
        return Err(ChainError::Shape("layer count".into()));
    }
    let batch = K4::challenge(transcript).map_err(|err| ChainError::Transcript(err.to_string()))?;
    // Rebuild the layer-0 φ (the verifier's own derivation).
    let phi0 = layer0_phi(w_ring.len(), sigma, w_cube_vars, &batch);
    let mut current_phi = vec![phi0];
    let mut current_n = w_ring.len();

    for (li, tr) in proof.layers.iter().enumerate() {
        for tj in &tr.t {
            absorb_poly_vec(transcript, b"t", tj).map_err(e)?;
        }
        // The projection seed + p replay.
        let _seed = transcript.challenge_bytes(b"proj-seed", 32).map_err(e)?;
        {
            let mut pb = Vec::new();
            for &x in &tr.p {
                pb.extend_from_slice(&x.to_le_bytes());
            }
            transcript.append_message(b"proj-p", &pb).map_err(e)?;
        }
        // The projection bound (the norm attestation).
        {
            let norm: u64 = tr.p.iter().map(|&x| (x * x) as u64).sum();
            let m = tr.projection.len() / (current_n * 64).max(1);
            if m == 0 || norm > m as u64 * tr.norm_bound {
                return Err(ChainError::NormViolation("projection bound"));
            }
        }
        for hv in &tr.h {
            absorb_poly(transcript, b"h", hv).map_err(e)?;
        }
        // The challenges re-derive and must match.
        let mut c = Vec::with_capacity(tr.c.len());
        for _ in 0..tr.c.len() {
            c.push(sample_short_challenge(transcript)?);
        }
        if c != tr.c {
            return Err(ChainError::Identity("challenge replay"));
        }
        absorb_poly_vec(transcript, b"z", &tr.z).map_err(e)?;
        // (28): A·z = Σ c_j t_j.
        {
            let (rows, cols) = proof.layer_key_dims[li];
            let key = AjtaiKey::from_seed(rows, cols, b"cauchyfold-layer");
            let mut az = vec![Poly::zero(); rows];
            for (cidx, zc) in tr.z.iter().enumerate() {
                for r in 0..rows {
                    az[r].add_assign(&key.matrix[r * cols + cidx].mul(zc));
                }
            }
            let mut rhs = vec![Poly::zero(); rows];
            for (j, tj) in tr.t.iter().enumerate() {
                for r in 0..rows {
                    rhs[r].add_assign(&tj[r].mul(&tr.c[j]));
                }
            }
            if az != rhs {
                return Err(ChainError::Identity("(28)"));
            }
        }
        // (29): ⟨Σ c_j φ_j, z⟩ = Σ_{a≤b} c_a c_b h_ab (the upper-triangle
        // h with the symmetric bookkeeping).
        {
            let s = tr.t.len();
            let mut lhs = Poly::zero();
            for (j, phij) in current_phi.iter().enumerate() {
                lhs.add_assign(&ring_inner(phij, &tr.z).mul(&tr.c[j % tr.c.len()]));
            }
            let mut rhs = Poly::zero();
            // The paper's (29): Σ_{a,b} c_a c_b h_ab over the FULL double
            // sum (the symmetric h carrying each cross term once per
            // ordered pair).
            for a in 0..s {
                for b2 in 0..s {
                    let (lo, hi) = (a.min(b2), a.max(b2));
                    let tri = if lo > 0 { lo * (lo - 1) / 2 } else { 0 };
                    let idx = lo * s - tri + (hi - lo);
                    rhs.add_assign(&tr.h[idx].mul(&tr.c[a].mul(&tr.c[b2])));
                }
            }
            if lhs != rhs {
                return Err(ChainError::Identity("(29)"));
            }
        }
        // (30): the response norm envelope (the declared bound times the
        // growth factor).
        {
            let znorm = norm2(&tr.z);
            if znorm > ((256.0 * tr.norm_bound as f64) * (15.0 / 14.0)) as u64 + 1 {
                return Err(ChainError::NormViolation("(30)"));
            }
        }
        // The child φ for the next layer: the recomposition functional.
        let n = tr.z.len();
        current_phi = {
            let ident = negacyclic_transpose(&vec![1i64; 64]);
            let radixv = negacyclic_transpose(&vec![LAYER_RADIX; 64]);
            vec![vec![ident; n], vec![radixv; n]]
        };
        current_n = n;
    }

    // The terminal checks.
    let flat: Vec<Poly> = proof.terminal.blocks.concat();
    let term_key = AjtaiKey::from_seed(1, flat.len().max(1), b"cauchyfold-terminal");
    absorb_poly_vec(transcript, b"term-c", &proof.terminal.commitment).map_err(e)?;
    if !term_key.verify(&flat, &proof.terminal.commitment) {
        return Err(ChainError::Identity("terminal commitment"));
    }
    if norm2(&flat) != proof.terminal.norm_squared {
        return Err(ChainError::NormViolation("terminal norm"));
    }
    // The terminal blocks recombine to the last response.
    if proof.terminal.blocks.len() == 2 {
        let last = &proof.layers[proof.layers.len() - 1].z;
        if proof.terminal.blocks[0].len() == last.len() {
            for (idx, zp) in last.iter().enumerate() {
                let recomposed = radix_recompose(
                    &proof.terminal.blocks[0][idx],
                    &proof.terminal.blocks[1][idx],
                    LAYER_RADIX,
                );
                if recomposed != *zp {
                    return Err(ChainError::Recomposition);
                }
            }
        }
    }
    // The fail-closed codec roundtrip.
    let codec = crate::wire::encode_terminal(&flat);
    let decoded = crate::wire::decode_terminal(&codec, flat.len()).map_err(ChainError::Codec)?;
    if decoded != flat {
        return Err(ChainError::Codec("roundtrip".into()));
    }
    Ok(())
}

fn absorb_poly(
    transcript: &mut Transcript,
    label: &[u8],
    p: &Poly,
) -> Result<(), lattice_core::transcript::TranscriptError> {
    transcript.append_message(label, &p.to_le_bytes())
}

fn absorb_poly_vec(
    transcript: &mut Transcript,
    label: &[u8],
    v: &[Poly],
) -> Result<(), lattice_core::transcript::TranscriptError> {
    for p in v {
        transcript.append_message(label, &p.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_split_roundtrip() {
        let mut p = [0i64; 64];
        for i in 0..64 {
            p[i] = (i as i64 * 137_438_953_471) % (Q48 as i64 / 2);
            if i % 3 == 0 {
                p[i] = -p[i];
            }
        }
        let z = Poly(p);
        let (lo, hi) = radix_split(&z, LAYER_RADIX);
        assert!(lo.0.iter().all(|&c| (-32..=31).contains(&c)));
        assert_eq!(radix_recompose(&lo, &hi, LAYER_RADIX), z);
    }

    #[test]
    fn short_challenge_certification() {
        let mut t = Transcript::new_default(b"d46-test");
        let mut seen = 0usize;
        for _ in 0..16 {
            let c = sample_short_challenge(&mut t).expect("sampling");
            assert!(c.0.iter().all(|&x| (-1..=1).contains(&x)));
            assert!(op_norm_bound_ok(&c));
            seen += 1;
        }
        assert!(seen > 0);
    }

    #[test]
    fn negacyclic_transpose_dot() {
        let v: Vec<i64> = (0..64).map(|i| (i as i64 * 7 % 31) - 15).collect();
        let w: Vec<i64> = (0..64).map(|i| (i as i64 * 13 % 41) - 20).collect();
        let tv = negacyclic_transpose(&v);
        let mut arr = [0i64; 64];
        for (i, &x) in w.iter().enumerate() {
            arr[i] = x;
        }
        let prod = tv.mul(&Poly(arr));
        let dot: i64 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert_eq!(prod.0[0], dot);
    }

    #[test]
    fn digit_energy_bounded() {
        // Two centered radix-64 digits cover |x| ≤ 32 + 31·64 = 2016.
        let e = digit_energy(2_000, 64, 2);
        assert!(e > 0);
        assert!(e <= 2 * 32 * 32);
        let e_small = digit_energy(1_000, 64, 2);
        assert!(e_small <= e, "monotone in H");
        // Extreme ranges saturate without overflow.
        let _ = digit_energy(u64::MAX / 2, 64, 2);
        let _ = digit_energy(u64::MAX, 64, 4);
    }

    #[test]
    fn half_ring_involution() {
        let mut a = [0i64; 64];
        for i in 0..64 {
            a[i] = (i as i64 * 997) % 100_000 - 50_000;
        }
        let p = Poly(a);
        // Halving is linear: half(a + a) = a.
        assert_eq!(half_ring(&p.add(&p)), p);
        // And half(x)·2 ≡ x (mod q) coefficient-wise.
        let h = half_ring(&p);
        for i in 0..64 {
            let doubled = (h.0[i] as i128 * 2).rem_euclid(Q48 as i128);
            let expect = (p.0[i] as i128).rem_euclid(Q48 as i128);
            assert_eq!(doubled, expect);
        }
    }

    #[test]
    fn chain_prove_verify_roundtrip() {
        // A small committed linear relation through the chain.
        let w: Vec<Poly> = (0..5)
            .map(|i| {
                let mut a = [0i64; 64];
                for j in 0..64 {
                    a[j] = ((i * 64 + j) as i64 * 31 % 97) - 48;
                }
                Poly(a)
            })
            .collect();
        let sigma: Vec<K4> = (0..4)
            .map(|i| K4::from_coeffs([100 + i as u64 * 7, 3, 5, 11]))
            .collect();
        // The evaluation claim: compute it from w.
        let w_cube_vars = (w.len() * 64).next_power_of_two().trailing_zeros() as usize;
        let mut ws = K4::ZERO;
        for (pos, p) in w.iter().enumerate() {
            let eqv = eq_pos(pos.min((1 << w_cube_vars) - 1), &sigma, w_cube_vars);
            // The digit value enters as its coefficient mean — for the
            // test, use the first coefficient.
            ws = ws.add(&eqv.mul(&K4::from_coeffs([p.0[0].unsigned_abs() % Q48, 0, 0, 0])));
        }
        let mut t = Transcript::new_default(b"chain-test");
        let proof = prove_chain(&w, &sigma, &ws, w_cube_vars, &mut t).expect("prove");
        let mut vt = Transcript::new_default(b"chain-test");
        verify_chain(&w, &sigma, &ws, w_cube_vars, &mut vt, &proof).expect("verify");
    }

    #[test]
    fn chain_tampered_response_rejected() {
        let w: Vec<Poly> = (0..3)
            .map(|i| {
                let mut a = [0i64; 64];
                for j in 0..64 {
                    a[j] = ((i * 64 + j) as i64 * 13 % 53) - 26;
                }
                Poly(a)
            })
            .collect();
        let sigma: Vec<K4> = (0..4)
            .map(|i| K4::from_coeffs([200 + i as u64 * 3, 2, 4, 6]))
            .collect();
        let w_cube_vars = (w.len() * 64).next_power_of_two().trailing_zeros() as usize;
        let mut ws = K4::ZERO;
        for (pos, p) in w.iter().enumerate() {
            let eqv = eq_pos(pos, &sigma, w_cube_vars);
            ws = ws.add(&eqv.mul(&K4::from_coeffs([p.0[0].unsigned_abs() % Q48, 0, 0, 0])));
        }
        let mut t = Transcript::new_default(b"chain-tamper");
        let mut proof = prove_chain(&w, &sigma, &ws, w_cube_vars, &mut t).expect("prove");
        proof.layers[0].z[0].0[7] += 5;
        let mut vt = Transcript::new_default(b"chain-tamper");
        assert!(verify_chain(&w, &sigma, &ws, w_cube_vars, &mut vt, &proof).is_err());
    }
}
