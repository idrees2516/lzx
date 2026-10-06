//! The block-commit batched opening (the "next fold" after the compact
//! column fold — docs/DESIGN_50KB.md Stage 5; the commitment-layer
//! bottleneck).
//!
//! # The problem
//!
//! `compact.rs` commits the packed column universe **per column**: r Ajtai
//! commitments `y_j = F̄·w_j ∈ R^k` under the column-uniform key. The
//! transmitted commitment layer is `r·k` ring elements — at the benchmark
//! scale that is ~16 KB of a 33 KB proof (the single largest component,
//! BENCHMARKS.md §2b'), and it grows linearly in the column count while
//! legs and claims shrink under batching. At richer-workload scale the
//! per-column commitments (hundreds of them) dominate the proof.
//!
//! # The construction (one packed block commitment + the fused binding)
//!
//! A single linear Ajtai commitment `y = F·vec(W)` over the whole universe
//! **cannot** be checked against the folded short response
//! `v = Σ_j d_j·w_j` by a purely linear relation: for any verifier maps,
//! `⟨ρ, y⟩ = ⟨Fᵀρ, vec(W)⟩` and `⟨μ, v⟩ = ⟨d⊗μ, vec(W)⟩` live in
//! generically-transversal subspaces — the compact check
//! `F̄·v = Σ_j d_j·y_j` works precisely because each column has its own
//! committed vector. The escape is to let a **sumcheck carry the binding**
//! (Akita's fused Eq-160 pattern — the "outer layer" the research
//! consensus flagged as the documented gap):
//!
//! 1. **Commit once**: `y = F·vec(W) ∈ R^k` with the wide seeded key
//!    `F ∈ R^{k×m}`, `m = r·n̄^pad` ring elements (the whole universe,
//!    zero-padded to the power-of-two cube). Transmitted: `k` ring
//!    elements — **r× smaller than the r·k layer**.
//! 2. The compact machinery is unchanged: the carrier terminal
//!    `(r_sc, w)`, the per-column values `ũ_j`, the MLE interpolation
//!    check, the fold challenges `d_j ∈ [−A, A]`, the response
//!    `v = Σ_j d_j·w_j`, the rANS coding, the norm gate, and the
//!    Goldilocks functional commute `Φ(v) = Σ_j d_j·ũ_j`.
//! 3. **The fused binding sumcheck** (NEW): the verifier derives
//!    `ρ ∈ Z_q^k`, `h ∈ R^{n̄}`, `β ∈ R`, computes
//!    `g := Σ_l ρ_l·F[l] ∈ R^m` (one pass over the seeded key) and
//!    `c_y := Σ_l ρ_l·y_l ∈ R`, and checks the single ring equation
//!    `⟨β, Σ_p c_p·w_p⟩ = ⟨β, c_y + ⟨h, v⟩⟩` where
//!    `c_p = g_p + d_{j(p)}·h_{i(p)}` is **public** (verifier-computable).
//!    The g-half `Σ_p g_p·w_p = ⟨ρ, y⟩` is the commitment binding (the
//!    W-side is pinned to the ONE block commitment); the h-half
//!    `Σ_p d_j·h_i·w_p = ⟨h, v⟩` is the response-fold binding (the
//!    transmitted v is pinned to the W-side).
//!    The check runs as a **degree-2 sumcheck over the m-entry cube**
//!    (`log₂m` rounds, three Z_q values per round: the public C̃ factor
//!    times the prover's W̃ factor). Terminal: the eq-interpolated
//!    `C̃(τ)` (verifier, one pass) against the eq-interpolated witness
//!    `ŵ ∈ R` (prover, one ring element): `⟨β, C̃(τ)·ŵ⟩ = final claim`.
//!
//! # Soundness chain (the honest statement)
//!
//! The soundness chain: carrier `w` ← interpolation check ← `ũ_j` ←
//! functional commute `Φ(v) = Σ d_j·ũ_j` ← the h-half of the fused
//! sumcheck pins v ← the W-side round messages + terminal `ŵ` ← the
//! g-half pins y (`⟨g, W⟩ = ⟨ρ, y⟩` for the random ρ). Knowledge
//! soundness terminates in MSIS on the wide `[F | −y]` key with the
//! byte-bounded preimage (the honest W has coefficients ≤ 255 by the
//! packing); the response norm gate keeps the extractor's folded
//! difference short. The composite extractor (sumcheck rewinding over
//! (ρ, h, β, τ) composed with the Ajtai relation) is recorded in
//! SECURITY.md; the wide m-column key strengthens the estimator's m/n
//! regime vs the column-uniform F̄.
//!
//! # Cost profile (measured in BENCHMARKS.md §2d)
//!
//! * Proof: `k` ring elements of commitment (vs `r·k`) + `3·log₂m` Z_q
//!   values + `ŵ` (one ring element) — at the fib scale ~16 KB → ~1.5 KB
//!   per bundle.
//! * Prover: ONE commitment pass (`k·m` ring mults — the same ring work
//!   as the r per-column commits, but one serialization and one absorb)
//!   plus the fused sumcheck (`~3m` ring mults, the per-round product
//!   passes with the sum-lerp trick).
//! * Verifier: one key pass for `g`, one pass for `C̃(τ)`'s binding, and
//!   the sumcheck replay — `O(k·m + m·log m)` coefficient ops.

use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::ring::{RingConfig, RingElement};

use crate::compact::{
    column_ring, decode_response, encode_response, pack_columns, phi_term,
    psi_weights_goldilocks, serialize_elements, Fq, PackShape, PackedWitness, ResponseWire,
};
use crate::ledger::LedgerError;

// ---------------------------------------------------------------------------
// Parameters and artifacts
// ---------------------------------------------------------------------------

/// Fold parameters for the block mode (public shape, transmitted).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockFoldParams {
    /// Number of columns r (a power of two, ≤ every factor length).
    pub r: usize,
    /// Commitment rows k (the ONE block commitment is k ring elements).
    pub k: usize,
    /// Scalar challenge amplitude A (challenges in [−A, A]).
    pub amplitude: u32,
    /// The response norm gate (per coefficient; must be < q/2).
    pub gate: u32,
}

impl BlockFoldParams {
    /// The estimator-tuned default, matching `compact::FoldParams::new`.
    pub fn new(r: usize, k: usize) -> Self {
        let amplitude = 1u32 << 6;
        let gate = (r as u64) * (amplitude as u64) * 255;
        BlockFoldParams {
            r,
            k,
            amplitude,
            gate: gate.min((lattice_ring::Modulus32::Q_32.q / 2 - 1) as u64) as u32,
        }
    }

    /// The worst-case fold bound r·A·β₀ (the completeness guarantee).
    pub fn worst_bound(&self, beta0: u64) -> u64 {
        (self.r as u64) * (self.amplitude as u64) * beta0
    }
}

/// The block opening artifact (one per bundle).
#[derive(Clone, Debug)]
pub struct BlockOpening {
    /// Fold parameters (public shape).
    pub params: BlockFoldParams,
    /// Per-factor byte widths (the packing shape).
    pub widths: Vec<u8>,
    /// Per-column Goldilocks values ũ_j (r field elements).
    pub u_tilde: Vec<Goldilocks>,
    /// The rANS-coded response v = Σ_j d_j·w_j.
    pub response: ResponseWire,
    /// The fused binding sumcheck's round messages (degree-2: three Z_q
    /// values [h(0), h(1), h(2)] per round over the m-cube).
    pub binding_rounds: Vec<[u64; 3]>,
    /// The eq-interpolated witness ring element ŵ ∈ R (the terminal).
    pub w_hat: Vec<u32>,
}

/// The prover-side block bundle state.
pub struct BlockBundleProver {
    pub ring: RingConfig,
    pub packed: PackedWitness,
    pub key: AjtaiPublicKey,
    pub params: BlockFoldParams,
    /// Per-column ring elements (unpadded, the same byte→ring packing as
    /// compact).
    pub columns: Vec<Vec<RingElement>>,
    /// The flat secret vector vec(W) zero-padded to m = r·n̄^pad.
    pub secret: Vec<RingElement>,
    /// The ONE block commitment y = F·vec(W) ∈ R^k.
    pub y: Vec<RingElement>,
    /// Ring elements per column (unpadded).
    pub n_bar: usize,
    /// The padded cube size m = r · next_pow2(n̄).
    pub m: usize,
    pub seed: [u8; 32],
}

/// Commit a bundle as ONE packed block: the wide key F ∈ R^{k×m} over the
/// whole column universe and the single commitment y = F·vec(W).
pub fn block_bundle_commit(
    entries: &[(u32, &DenseMle)],
    seed: [u8; 32],
    r: usize,
    k: usize,
) -> Result<BlockBundleProver, LedgerError> {
    let ring = column_ring()?;
    let params = BlockFoldParams::new(r, k);
    let packed = pack_columns(entries, r)?;
    let stream = packed.shape.stream_len();
    let n = ring.n();
    let n_bar = stream.div_ceil(n).max(1);
    let n_bar_pad = n_bar.next_power_of_two();
    let m = r * n_bar_pad;
    let mut columns: Vec<Vec<RingElement>> = Vec::with_capacity(r);
    for j in 0..r {
        let mut col: Vec<RingElement> = Vec::with_capacity(n_bar);
        for i in 0..n_bar {
            let mut coeffs = vec![0u32; n];
            for (kk, c) in coeffs.iter_mut().enumerate() {
                if let Some(&b) = packed.columns[j].get(i * n + kk) {
                    *c = b as u32;
                }
            }
            col.push(RingElement::from_coeffs(&ring, coeffs));
        }
        columns.push(col);
    }
    // vec(W) = [w_1 | ... | w_r], each column zero-padded to n̄^pad.
    let mut secret: Vec<RingElement> = Vec::with_capacity(m);
    for col in &columns {
        secret.extend_from_slice(col);
        for _ in n_bar..n_bar_pad {
            secret.push(ring.zero());
        }
    }
    let ajtai = AjtaiParams {
        ring: ring.clone(),
        k,
        m,
        norm_bound: params.gate,
    };
    let key = AjtaiPublicKey::from_seed(ajtai, seed).map_err(LedgerError::Ajtai)?;
    let commitment = key.commit(&secret).map_err(LedgerError::Ajtai)?;
    Ok(BlockBundleProver {
        ring,
        packed,
        key,
        params,
        columns,
        secret,
        y: commitment.rows,
        n_bar,
        m,
        seed,
    })
}

// ---------------------------------------------------------------------------
// Shared challenge/key helpers (prover and verifier run the same code)
// ---------------------------------------------------------------------------

/// The ring functional ⟨β, x⟩ = Σ_c β[c]·x[c] mod q (coefficient-wise, no
/// convolution): scalarizes one ring equation into a Z_q equation.
fn beta_functional(beta: &RingElement, x: &RingElement) -> Fq {
    let bc = beta.coeffs();
    let xc = x.coeffs();
    let mut acc = Fq::ZERO;
    for (b, xx) in bc.iter().zip(xc.iter()) {
        acc = acc.add(Fq::mul(Fq::from_u64(*b as u64), Fq::from_u64(*xx as u64)));
    }
    acc
}

/// The public C-array: `c_p = g_p + d_{j(p)}·h_{i(p)}` over the padded
/// m-cube, from the verifier's key pass `g`, the fold challenges `d` and
/// the binding vector `h`.
#[allow(clippy::expect_used)] // same-ring adds cannot fail (one RingConfig)
fn build_c_array(
    ring: &RingConfig,
    g: &[RingElement],
    d: &[i64],
    h: &[RingElement],
    r: usize,
    n_bar_pad: usize,
) -> Vec<RingElement> {
    let m = r * n_bar_pad;
    let mut c = Vec::with_capacity(m);
    for j in 0..r {
        let dj = d[j];
        for i in 0..n_bar_pad {
            let p = j * n_bar_pad + i;
            let mut cp = g[p].clone();
            if dj != 0 {
                cp = cp.add(&h[i].scale_i64(dj)).expect("same-ring add");
            }
            c.push(cp);
        }
    }
    let _ = ring;
    c
}

/// The key pass: `g = Σ_l ρ_l·F[l] ∈ R^m`.
#[allow(clippy::expect_used)] // same-ring adds cannot fail (one RingConfig)
fn key_pass(
    ring: &RingConfig,
    key: &AjtaiPublicKey,
    rho: &[Fq],
    m: usize,
) -> Vec<RingElement> {
    let mut g = Vec::with_capacity(m);
    for p in 0..m {
        let mut acc = ring.zero();
        for (l, &rl) in rho.iter().enumerate() {
            if let Some(elem) = key.entry(l, p) {
                acc = acc.add(&elem.scale_i64(rl.0 as i64)).expect("same-ring add");
            }
        }
        g.push(acc);
    }
    g
}

/// Bind one array: `next[i] = (1−τ)·cur[i] + τ·cur[half+i]`.
#[allow(clippy::expect_used)] // same-ring adds cannot fail (one RingConfig)
fn bind_ring_array(cur: &[RingElement], tau: &Fq) -> Vec<RingElement> {
    let half = cur.len() / 2;
    let one_minus = Fq::ONE.sub(*tau);
    let mut next = Vec::with_capacity(half);
    for i in 0..half {
        let a = &cur[i];
        let b = &cur[half + i];
        let mut bound = a.scale_i64(one_minus.0 as i64);
        let tb = b.scale_i64(tau.0 as i64);
        bound = bound.add(&tb).expect("same-ring add");
        next.push(bound);
    }
    next
}

/// The Lagrange basis evaluation of the degree-2 round polynomial at τ:
/// `h(τ) = h0·ℓ0 + h1·ℓ1 + h2·ℓ2` over nodes {0, 1, 2}.
#[allow(clippy::expect_used)] // inv(2) exists at the odd prime q
fn lagrange2(h0: Fq, h1: Fq, h2: Fq, tau: Fq) -> Fq {
    let one = Fq::ONE;
    let two = Fq::from_u64(2);
    let inv2 = two.inv().expect("2 invertible mod q");
    // ℓ0(τ) = (τ−1)(τ−2)/2, ℓ1(τ) = τ(2−τ), ℓ2(τ) = τ(τ−1)/2.
    let l0 = tau.sub(one).mul(tau.sub(two)).mul(inv2);
    let l1 = tau.mul(two.sub(tau));
    let l2 = tau.mul(tau.sub(one)).mul(inv2);
    h0.mul(l0).add(h1.mul(l1)).add(h2.mul(l2))
}

// ---------------------------------------------------------------------------
// The block opening (prover)
// ---------------------------------------------------------------------------

impl BlockBundleProver {
    /// The serialized ONE-block commitment (k ring elements).
    pub fn commitment_bytes(&self) -> Vec<u8> {
        serialize_elements(&self.ring, &self.y)
    }

    /// The flat MLE over Goldilocks (the carrier's polynomial — the SAME
    /// one the compact bundle uses).
    pub fn flat_mle(&self) -> &DenseMle {
        &self.packed.flat
    }

    /// Prove the block opening given the carrier terminal (r_sc, w).
    #[allow(clippy::too_many_lines)]
    pub fn prove_block_opening(
        &self,
        r_sc: &[Goldilocks],
        w: &Goldilocks,
        transcript: &mut Transcript,
    ) -> Result<BlockOpening, LedgerError> {
        let flat_log = self.packed.shape.flat_log;
        let log_r = self.params.r.trailing_zeros() as usize;
        if r_sc.len() != flat_log {
            return Err(LedgerError::Layout("r_sc arity".into()));
        }
        let r_head = &r_sc[..flat_log - log_r];
        let r_tail = &r_sc[flat_log - log_r..];
        let r = self.params.r;
        let n_bar = self.n_bar;
        let n_bar_pad = n_bar.next_power_of_two();
        let m = self.m;
        let n = self.ring.n();

        // ---- 1. Per-column values ũ_j + the interpolation self-check ----
        let psi = psi_weights_goldilocks(&self.packed.shape, r_head);
        let mut u_tilde: Vec<Goldilocks> = Vec::with_capacity(r);
        for j in 0..r {
            let mut acc = Goldilocks::ZERO;
            for mm in 0..psi.len() {
                if let Some(&b) = self.packed.columns[j].get(mm) {
                    acc = acc.add(&psi[mm].mul(&Goldilocks::from_u64(b as u64)));
                }
            }
            u_tilde.push(acc);
        }
        {
            let mut table = vec![Goldilocks::ONE];
            for rr in r_tail.iter().rev() {
                let one_minus = Goldilocks::ONE.sub(rr);
                let mut next = Vec::with_capacity(table.len() * 2);
                for t in &table {
                    next.push(t.mul(&one_minus));
                }
                for t in &table {
                    next.push(t.mul(rr));
                }
                table = next;
            }
            let mut check = Goldilocks::ZERO;
            for (j, uj) in u_tilde.iter().enumerate() {
                check = check.add(&table[j].mul(uj));
            }
            if check != *w {
                return Err(LedgerError::Layout(
                    "interpolation self-check failed".into(),
                ));
            }
        }
        let mut ubuf: Vec<u8> = Vec::with_capacity(8 * u_tilde.len());
        for u in &u_tilde {
            ubuf.extend_from_slice(&u.to_canonical_u64().to_le_bytes());
        }
        transcript
            .append_bytes(b"fold-utilde", &ubuf)
            .map_err(LedgerError::Transcript)?;

        // ---- 2. The scalar fold challenges d_j ∈ [−A, A] ----
        let d: Vec<i64> = (0..r)
            .map(|_| {
                let b = transcript
                    .challenge_bytes(b"fold-dchal", 2)
                    .map_err(LedgerError::Transcript)?;
                let raw = u16::from_le_bytes([b[0], b[1]]) as u64;
                let mm = 2 * self.params.amplitude as u64 + 1;
                Ok((raw % mm) as i64 - self.params.amplitude as i64)
            })
            .collect::<Result<_, _>>()?;

        // ---- 3. The response v = Σ_j d_j·w_j + the norm gate ----
        let gate = self.params.gate as i64;
        if gate >= (self.ring.modulus.q / 2) as i64 {
            return Err(LedgerError::Layout("gate exceeds q/2".into()));
        }
        let mut v: Vec<RingElement> = vec![self.ring.zero(); n_bar];
        let mut v_coeffs: Vec<i32> = Vec::with_capacity(n_bar * n);
        for (j, &dj) in d.iter().enumerate() {
            if dj == 0 {
                continue;
            }
            for i in 0..n_bar {
                let prod = self.columns[j][i].scale_i64(dj);
                v[i] = v[i]
                    .add(&prod)
                    .map_err(|e| LedgerError::Layout(format!("ring add: {e:?}")))?;
            }
        }
        for elem in &v {
            for &c in elem.coeffs() {
                let balanced = if c > self.ring.modulus.q / 2 {
                    c as i64 - self.ring.modulus.q as i64
                } else {
                    c as i64
                };
                if balanced.abs() > gate {
                    return Err(LedgerError::Layout(format!(
                        "norm gate: |{balanced}| > {gate}"
                    )));
                }
                v_coeffs.push(balanced as i32);
            }
        }

        // ---- 4. The Goldilocks functional commute Φ(v) = Σ d_j·ũ_j ----
        {
            let q = self.ring.modulus.q;
            let mut phi_v = Goldilocks::ZERO;
            for (i, elem) in v.iter().enumerate() {
                let mut elem_sum = Goldilocks::ZERO;
                for (kk, &c) in elem.coeffs().iter().enumerate() {
                    let weight = psi.get(i * n + kk).copied().unwrap_or(Goldilocks::ZERO);
                    if weight != Goldilocks::ZERO {
                        let t = phi_term(&weight, c, q);
                        elem_sum = elem_sum.add(&t);
                    }
                }
                phi_v = phi_v.add(&elem_sum);
            }
            let mut rhs = Goldilocks::ZERO;
            for (j, &dj) in d.iter().enumerate() {
                if dj != 0 {
                    let term = u_tilde[j].mul(&Goldilocks::from_u64(dj.unsigned_abs()));
                    rhs = if dj < 0 { rhs.sub(&term) } else { rhs.add(&term) };
                }
            }
            if phi_v != rhs {
                return Err(LedgerError::Layout("functional self-check failed".into()));
            }
        }
        transcript
            .append_bytes(b"fold-v", &serialize_elements(&self.ring, &v))
            .map_err(LedgerError::Transcript)?;
        let response = encode_response(&v_coeffs)
            .map_err(|e| LedgerError::Layout(format!("response encode: {e}")))?;

        // ---- 5. The fused binding challenges: ρ, h, β ----
        let rho_seed = transcript
            .challenge_bytes(b"block-rho", 32)
            .map_err(LedgerError::Transcript)?;
        let h_seed = transcript
            .challenge_bytes(b"block-h", 32)
            .map_err(LedgerError::Transcript)?;
        let beta_seed = transcript
            .challenge_bytes(b"block-beta", 32)
            .map_err(LedgerError::Transcript)?;
        let rho: Vec<Fq> = (0..self.params.k)
            .map(|l| {
                Fq::from_u64(
                    self.ring
                        .uniform_from_seed(b"block-rho", &rho_seed, l as u64)
                        .coeffs()[0] as u64,
                )
            })
            .collect();
        let h: Vec<RingElement> = (0..n_bar_pad)
            .map(|i| self.ring.uniform_from_seed(b"block-h", &h_seed, i as u64))
            .collect();
        let beta = self.ring.uniform_from_seed(b"block-beta", &beta_seed, 0);

        // The verifier's key pass g and the public C-array.
        let g = key_pass(&self.ring, &self.key, &rho, m);
        let c = build_c_array(&self.ring, &g, &d, &h, r, n_bar_pad);

        // The claim: ⟨β, c_y + ⟨h, v⟩⟩ (verifier-computable from y and v).
        let claim = {
            let mut c_y = self.ring.zero();
            for (l, &rl) in rho.iter().enumerate() {
                let scaled = self.y[l].scale_i64(rl.0 as i64);
                c_y = c_y
                    .add(&scaled)
                    .map_err(|e| LedgerError::Layout(format!("ring add: {e:?}")))?;
            }
            let mut hv = self.ring.zero();
            for i in 0..n_bar_pad {
                let vi = if i < n_bar { &v[i] } else { &self.ring.zero() };
                let term = h[i]
                    .mul(vi)
                    .map_err(|e| LedgerError::Layout(format!("ring mul: {e:?}")))?;
                hv = hv
                    .add(&term)
                    .map_err(|e| LedgerError::Layout(format!("ring add: {e:?}")))?;
            }
            beta_functional(&beta, &c_y.add(&hv).map_err(|e| {
                LedgerError::Layout(format!("ring add: {e:?}"))
            })?)
        };

        // ---- 6. The degree-2 fused sumcheck over the m-cube ----
        // Prover state: the W-array (the padded secret) and the C-array,
        // bound per round. Round messages:
        //   h(0) = Σ_{i<half} ⟨β, C[i]·W[i]⟩
        //   h(1) = Σ_{i<half} ⟨β, C[half+i]·W[half+i]⟩
        //   h(2) = 6·h(1) + 3·h(0) − 2·S,
        //     S = Σ_{i<half} ⟨β, (C[i]+C[half+i])·(W[i]+W[half+i])⟩
        // (the sum-lerp identity for the degree-2 node t = 2.)
        let log_m = m.trailing_zeros() as usize;
        let mut c_cur = c;
        let mut w_cur = self.secret.clone();
        let mut running = claim;
        let mut binding_rounds: Vec<[u64; 3]> = Vec::with_capacity(log_m);
        let q = self.ring.modulus.q as u64;
        let three = Fq::from_u64(3);
        let six = Fq::from_u64(6);
        for _ in 0..log_m {
            let half = c_cur.len() / 2;
            let mut h0 = Fq::ZERO;
            let mut h1 = Fq::ZERO;
            let mut s_acc = Fq::ZERO;
            for i in 0..half {
                let p00 = c_cur[i]
                    .mul(&w_cur[i])
                    .map_err(|e| LedgerError::Layout(format!("ring mul: {e:?}")))?;
                h0 = h0.add(beta_functional(&beta, &p00));
                let p11 = c_cur[half + i]
                    .mul(&w_cur[half + i])
                    .map_err(|e| LedgerError::Layout(format!("ring mul: {e:?}")))?;
                h1 = h1.add(beta_functional(&beta, &p11));
                let cs = c_cur[i]
                    .add(&c_cur[half + i])
                    .map_err(|e| LedgerError::Layout(format!("ring add: {e:?}")))?;
                let ws = w_cur[i]
                    .add(&w_cur[half + i])
                    .map_err(|e| LedgerError::Layout(format!("ring add: {e:?}")))?;
                let ps = cs
                    .mul(&ws)
                    .map_err(|e| LedgerError::Layout(format!("ring mul: {e:?}")))?;
                s_acc = s_acc.add(beta_functional(&beta, &ps));
            }
            let h2 = six.mul(h1).add(three.mul(h0)).sub(s_acc).sub(s_acc);
            // Prover-side guard: g(0) + g(1) == running claim.
            if h0.add(h1) != running {
                return Err(LedgerError::Layout(
                    "binding sumcheck claim mismatch".into(),
                ));
            }
            binding_rounds.push([h0.0, h1.0, h2.0]);
            let mut rbuf = Vec::with_capacity(24);
            rbuf.extend_from_slice(&h0.0.to_le_bytes());
            rbuf.extend_from_slice(&h1.0.to_le_bytes());
            rbuf.extend_from_slice(&h2.0.to_le_bytes());
            transcript
                .append_bytes(b"block-round", &rbuf)
                .map_err(LedgerError::Transcript)?;
            let rc = transcript
                .challenge_bytes(b"block-rchal", 8)
                .map_err(LedgerError::Transcript)?;
            let tau = Fq(u64::from_le_bytes([
                rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6], rc[7],
            ]) % q);
            running = lagrange2(h0, h1, h2, tau);
            c_cur = bind_ring_array(&c_cur, &tau);
            w_cur = bind_ring_array(&w_cur, &tau);
        }

        // ---- 7. The terminal: ŵ = the eq-interp of vec(W) at τ ----
        let w_hat = w_cur[0].clone();
        let terminal = {
            let prod = c_cur[0]
                .mul(&w_hat)
                .map_err(|e| LedgerError::Layout(format!("ring mul: {e:?}")))?;
            beta_functional(&beta, &prod)
        };
        if terminal != running {
            return Err(LedgerError::Layout("binding terminal self-check".into()));
        }
        transcript
            .append_bytes(b"block-what", &elem_coeffs_bytes(&w_hat))
            .map_err(LedgerError::Transcript)?;

        Ok(BlockOpening {
            params: self.params.clone(),
            widths: self.packed.shape.widths.clone(),
            u_tilde,
            response,
            binding_rounds,
            w_hat: w_hat.coeffs().to_vec(),
        })
    }
}

fn elem_coeffs_bytes(elem: &RingElement) -> Vec<u8> {
    let mut out = Vec::with_capacity(elem.coeffs().len() * 4);
    for c in elem.coeffs() {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// The block opening (verifier)
// ---------------------------------------------------------------------------

/// Verify a block bundle opening given the carrier terminal (r_sc, w) and
/// the ONE block commitment's bytes.
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
pub fn verify_block_opening(
    seed: [u8; 32],
    commitment_bytes: &[u8],
    factor_lens: &[usize],
    flat_log: usize,
    r_sc: &[Goldilocks],
    w: &Goldilocks,
    opening: &BlockOpening,
    transcript: &mut Transcript,
) -> Result<(), LedgerError> {
    let ring = column_ring()?;
    let params = &opening.params;
    let shape = PackShape {
        factor_lens: factor_lens.to_vec(),
        widths: opening.widths.clone(),
        r: params.r,
        flat_log,
    };
    let log_r = params.r.trailing_zeros() as usize;
    if flat_log < log_r || r_sc.len() != flat_log {
        return Err(LedgerError::Layout("r_sc arity".into()));
    }
    let r_head = &r_sc[..flat_log - log_r];
    let r_tail = &r_sc[flat_log - log_r..];

    // 1. The ũ's + interpolation check (identical to the compact verify).
    if opening.u_tilde.len() != params.r {
        return Err(LedgerError::Layout("u count".into()));
    }
    let mut ubuf: Vec<u8> = Vec::with_capacity(8 * params.r);
    for u in &opening.u_tilde {
        ubuf.extend_from_slice(&u.to_canonical_u64().to_le_bytes());
    }
    transcript
        .append_bytes(b"fold-utilde", &ubuf)
        .map_err(LedgerError::Transcript)?;
    {
        let mut table = vec![Goldilocks::ONE];
        for rr in r_tail.iter().rev() {
            let one_minus = Goldilocks::ONE.sub(rr);
            let mut next = Vec::with_capacity(table.len() * 2);
            for t in &table {
                next.push(t.mul(&one_minus));
            }
            for t in &table {
                next.push(t.mul(rr));
            }
            table = next;
        }
        let mut check = Goldilocks::ZERO;
        for (j, uj) in opening.u_tilde.iter().enumerate() {
            check = check.add(&table[j].mul(uj));
        }
        if check != *w {
            return Err(LedgerError::DerivedMismatch);
        }
    }

    // 2. The challenge replay.
    let d: Vec<i64> = (0..params.r)
        .map(|_| {
            let b = transcript
                .challenge_bytes(b"fold-dchal", 2)
                .map_err(LedgerError::Transcript)?;
            let raw = u16::from_le_bytes([b[0], b[1]]) as u64;
            let mm = 2 * params.amplitude as u64 + 1;
            Ok((raw % mm) as i64 - params.amplitude as i64)
        })
        .collect::<Result<_, _>>()?;

    // 3. Decode + gate the response (identical to the compact verify).
    let psi = psi_weights_goldilocks(&shape, r_head);
    let n = ring.n();
    let stream = shape.stream_len();
    let n_bar = stream.div_ceil(n).max(1);
    let n_bar_pad = n_bar.next_power_of_two();
    let m = params.r * n_bar_pad;
    let v_coeffs = decode_response(&opening.response)
        .map_err(|e| LedgerError::Layout(format!("response: {e}")))?;
    if v_coeffs.len() != n_bar * n {
        return Err(LedgerError::Layout("response length".into()));
    }
    let gate = params.gate as i64;
    if gate >= (ring.modulus.q / 2) as i64 {
        return Err(LedgerError::Layout("gate exceeds q/2".into()));
    }
    let mut v: Vec<RingElement> = Vec::with_capacity(n_bar);
    for chunk in v_coeffs.chunks(n) {
        let mut coeffs = vec![0u32; n];
        for (i, &c) in chunk.iter().enumerate() {
            if (c as i64).abs() > gate {
                return Err(LedgerError::Layout("norm gate".into()));
            }
            coeffs[i] = (c as i64).rem_euclid(ring.modulus.q as i64) as u32;
        }
        v.push(RingElement::from_coeffs(&ring, coeffs));
    }
    transcript
        .append_bytes(b"fold-v", &serialize_elements(&ring, &v))
        .map_err(LedgerError::Transcript)?;

    // 4. The functional commute: Φ(v) = Σ_j d_j·ũ_j over Goldilocks.
    let q32 = ring.modulus.q;
    let mut phi_v = Goldilocks::ZERO;
    for (i, elem) in v.iter().enumerate() {
        let mut elem_sum = Goldilocks::ZERO;
        for (kk, &c) in elem.coeffs().iter().enumerate() {
            let weight = psi.get(i * n + kk).copied().unwrap_or(Goldilocks::ZERO);
            if weight != Goldilocks::ZERO {
                let t = phi_term(&weight, c, q32);
                elem_sum = elem_sum.add(&t);
            }
        }
        phi_v = phi_v.add(&elem_sum);
    }
    let mut rhs = Goldilocks::ZERO;
    for (j, &dj) in d.iter().enumerate() {
        if dj != 0 {
            let term = opening.u_tilde[j].mul(&Goldilocks::from_u64(dj.unsigned_abs()));
            rhs = if dj < 0 { rhs.sub(&term) } else { rhs.add(&term) };
        }
    }
    if phi_v != rhs {
        return Err(LedgerError::DerivedMismatch);
    }

    // 5. The fused binding: rebuild the wide key, the challenges, the
    //    public C-array, and the claim from the ONE transmitted y.
    let y = crate::compact::deserialize_elements(&ring, commitment_bytes)
        .map_err(|e| LedgerError::Layout(format!("y: {e}")))?;
    if y.len() != params.k {
        return Err(LedgerError::Layout("y count".into()));
    }
    let rho_seed = transcript
        .challenge_bytes(b"block-rho", 32)
        .map_err(LedgerError::Transcript)?;
    let h_seed = transcript
        .challenge_bytes(b"block-h", 32)
        .map_err(LedgerError::Transcript)?;
    let beta_seed = transcript
        .challenge_bytes(b"block-beta", 32)
        .map_err(LedgerError::Transcript)?;
    let rho: Vec<Fq> = (0..params.k)
        .map(|l| {
            Fq::from_u64(
                ring.uniform_from_seed(b"block-rho", &rho_seed, l as u64).coeffs()[0] as u64,
            )
        })
        .collect();
    let h: Vec<RingElement> = (0..n_bar_pad)
        .map(|i| ring.uniform_from_seed(b"block-h", &h_seed, i as u64))
        .collect();
    let beta = ring.uniform_from_seed(b"block-beta", &beta_seed, 0);

    let ajtai = AjtaiParams {
        ring: ring.clone(),
        k: params.k,
        m,
        norm_bound: params.gate,
    };
    let key = AjtaiPublicKey::from_seed(ajtai, seed).map_err(LedgerError::Ajtai)?;
    let g = key_pass(&ring, &key, &rho, m);
    let mut c_cur = build_c_array(&ring, &g, &d, &h, params.r, n_bar_pad);

    let claim = {
        let mut c_y = ring.zero();
        for (l, &rl) in rho.iter().enumerate() {
            let scaled = y[l].scale_i64(rl.0 as i64);
            c_y = c_y
                .add(&scaled)
                .map_err(|e| LedgerError::Layout(format!("ring add: {e:?}")))?;
        }
        let mut hv = ring.zero();
        for i in 0..n_bar_pad {
            let vi = if i < n_bar { &v[i] } else { &ring.zero() };
            let term = h[i]
                .mul(vi)
                .map_err(|e| LedgerError::Layout(format!("ring mul: {e:?}")))?;
            hv = hv
                .add(&term)
                .map_err(|e| LedgerError::Layout(format!("ring add: {e:?}")))?;
        }
        beta_functional(&beta, &c_y.add(&hv).map_err(|e| {
            LedgerError::Layout(format!("ring add: {e:?}"))
        })?)
    };

    // 6. The sumcheck replay (the verifier binds its own public C-array).
    let log_m = m.trailing_zeros() as usize;
    if opening.binding_rounds.len() != log_m {
        return Err(LedgerError::Layout("binding rounds".into()));
    }
    let q = ring.modulus.q as u64;
    let mut running = claim;
    for round in &opening.binding_rounds {
        let h0 = Fq(round[0]);
        let h1 = Fq(round[1]);
        let h2 = Fq(round[2]);
        if h0.0 >= q || h1.0 >= q || h2.0 >= q {
            return Err(LedgerError::Layout("round value out of range".into()));
        }
        if h0.add(h1) != running {
            return Err(LedgerError::DerivedMismatch);
        }
        let mut rbuf = Vec::with_capacity(24);
        rbuf.extend_from_slice(&h0.0.to_le_bytes());
        rbuf.extend_from_slice(&h1.0.to_le_bytes());
        rbuf.extend_from_slice(&h2.0.to_le_bytes());
        transcript
            .append_bytes(b"block-round", &rbuf)
            .map_err(LedgerError::Transcript)?;
        let rc = transcript
            .challenge_bytes(b"block-rchal", 8)
            .map_err(LedgerError::Transcript)?;
        let tau = Fq(u64::from_le_bytes([
            rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6], rc[7],
        ]) % q);
        running = lagrange2(h0, h1, h2, tau);
        c_cur = bind_ring_array(&c_cur, &tau);
    }

    // 7. The terminal: ⟨β, C̃(τ)·ŵ⟩ == the final running claim.
    if opening.w_hat.len() != n {
        return Err(LedgerError::Layout("w_hat length".into()));
    }
    transcript
        .append_bytes(b"block-what", &{
            let mut out = Vec::with_capacity(n * 4);
            for c in &opening.w_hat {
                if *c >= ring.modulus.q {
                    return Err(LedgerError::Layout("w_hat coefficient".into()));
                }
                out.extend_from_slice(&c.to_le_bytes());
            }
            out
        })
        .map_err(LedgerError::Transcript)?;
    let w_hat = RingElement::from_coeffs(&ring, opening.w_hat.clone());
    let terminal = {
        let prod = c_cur[0]
            .mul(&w_hat)
            .map_err(|e| LedgerError::Layout(format!("ring mul: {e:?}")))?;
        beta_functional(&beta, &prod)
    };
    if terminal != running {
        return Err(LedgerError::DerivedMismatch);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// Three byte-valued factors (r=4 columns).
    fn entries_small() -> Vec<(u32, DenseMle)> {
        vec![
            (7, DenseMle { num_vars: 3, evaluations: vec![fe(1), fe(2), fe(3), fe(4), fe(250), fe(255), fe(0), fe(9)] }),
            (8, DenseMle { num_vars: 3, evaluations: vec![fe(17), fe(0), fe(200), fe(128), fe(7), fe(66), fe(99), fe(1)] }),
            (9, DenseMle { num_vars: 3, evaluations: vec![fe(0); 8] }),
        ]
    }

    fn run_block(
        r: usize,
        k: usize,
        entries: &[(u32, DenseMle)],
    ) -> Result<(), LedgerError> {
        let seed = [7u8; 32];
        let prover = block_bundle_commit(
            &entries.iter().map(|(i, m)| (*i, m)).collect::<Vec<_>>(),
            seed,
            r,
            k,
        )
        .expect("commit");
        let commitment = prover.commitment_bytes();
        // A carrier terminal: evaluate the flat MLE at a fixed point.
        let flat_log = prover.packed.shape.flat_log;
        let point: Vec<Goldilocks> = (0..flat_log)
            .map(|i| fe((i as u64 * 37 + 11) % 65521))
            .collect();
        let w = prover.flat_mle().evaluate(&point).map_err(|e| {
            LedgerError::Layout(format!("eval: {e:?}"))
        })?;
        let mut transcript = Transcript::new_default(b"lzx-block-test");
        let opening = prover.prove_block_opening(&point, &w, &mut transcript)?;
        // Verify with a fresh transcript (full replay).
        let mut vt = Transcript::new_default(b"lzx-block-test");
        let factor_lens: Vec<usize> = entries.iter().map(|(_, m)| m.evaluations.len()).collect();
        verify_block_opening(
            seed,
            &commitment,
            &factor_lens,
            flat_log,
            &point,
            &w,
            &opening,
            &mut vt,
        )
    }

    #[test]
    fn block_roundtrip_small() {
        let entries = entries_small();
        assert!(run_block(4, 2, &entries).is_ok());
    }

    #[test]
    fn block_roundtrip_r2() {
        // r=2 columns over 8-entry factors.
        let entries = entries_small();
        assert!(run_block(2, 4, &entries).is_ok());
    }

    #[test]
    fn block_roundtrip_r_equals_len() {
        // r = 8 = the factor length (one byte per column).
        let entries = entries_small();
        assert!(run_block(8, 2, &entries).is_ok());
    }

    #[test]
    fn block_tampered_commitment_rejected() {
        let entries = entries_small();
        let seed = [7u8; 32];
        let prover = block_bundle_commit(
            &entries.iter().map(|(i, m)| (*i, m)).collect::<Vec<_>>(),
            seed,
            4,
            2,
        )
        .expect("commit");
        let mut commitment = prover.commitment_bytes();
        commitment[3] ^= 0x40;
        let flat_log = prover.packed.shape.flat_log;
        let point: Vec<Goldilocks> = (0..flat_log)
            .map(|i| fe((i as u64 * 37 + 11) % 65521))
            .collect();
        let w = prover.flat_mle().evaluate(&point).expect("eval");
        let mut transcript = Transcript::new_default(b"lzx-block-test");
        let opening = prover.prove_block_opening(&point, &w, &mut transcript).expect("prove");
        let mut vt = Transcript::new_default(b"lzx-block-test");
        let factor_lens: Vec<usize> = entries.iter().map(|(_, m)| m.evaluations.len()).collect();
        assert!(verify_block_opening(
            seed,
            &commitment,
            &factor_lens,
            flat_log,
            &point,
            &w,
            &opening,
            &mut vt
        )
        .is_err());
    }

    #[test]
    fn block_tampered_terminal_rejected() {
        let entries = entries_small();
        let seed = [7u8; 32];
        let prover = block_bundle_commit(
            &entries.iter().map(|(i, m)| (*i, m)).collect::<Vec<_>>(),
            seed,
            4,
            2,
        )
        .expect("commit");
        let commitment = prover.commitment_bytes();
        let flat_log = prover.packed.shape.flat_log;
        let point: Vec<Goldilocks> = (0..flat_log)
            .map(|i| fe((i as u64 * 37 + 11) % 65521))
            .collect();
        let w = prover.flat_mle().evaluate(&point).expect("eval");
        let mut transcript = Transcript::new_default(b"lzx-block-test");
        let mut opening = prover.prove_block_opening(&point, &w, &mut transcript).expect("prove");
        opening.w_hat[0] = opening.w_hat[0].wrapping_add(1);
        let mut vt = Transcript::new_default(b"lzx-block-test");
        let factor_lens: Vec<usize> = entries.iter().map(|(_, m)| m.evaluations.len()).collect();
        assert!(verify_block_opening(
            seed,
            &commitment,
            &factor_lens,
            flat_log,
            &point,
            &w,
            &opening,
            &mut vt
        )
        .is_err());
    }

    #[test]
    fn block_tampered_round_rejected() {
        let entries = entries_small();
        let seed = [7u8; 32];
        let prover = block_bundle_commit(
            &entries.iter().map(|(i, m)| (*i, m)).collect::<Vec<_>>(),
            seed,
            4,
            2,
        )
        .expect("commit");
        let commitment = prover.commitment_bytes();
        let flat_log = prover.packed.shape.flat_log;
        let point: Vec<Goldilocks> = (0..flat_log)
            .map(|i| fe((i as u64 * 37 + 11) % 65521))
            .collect();
        let w = prover.flat_mle().evaluate(&point).expect("eval");
        let mut transcript = Transcript::new_default(b"lzx-block-test");
        let mut opening = prover.prove_block_opening(&point, &w, &mut transcript).expect("prove");
        // Corrupt one round's h(2) value (keeps g(0)+g(1) consistent with
        // nothing — the terminal check must catch it).
        if let Some(rr) = opening.binding_rounds.first_mut() {
            rr[2] = (rr[2] + 1) % crate::compact::Q;
        }
        let mut vt = Transcript::new_default(b"lzx-block-test");
        let factor_lens: Vec<usize> = entries.iter().map(|(_, m)| m.evaluations.len()).collect();
        assert!(verify_block_opening(
            seed,
            &commitment,
            &factor_lens,
            flat_log,
            &point,
            &w,
            &opening,
            &mut vt
        )
        .is_err());
    }

    #[test]
    fn block_tampered_response_rejected() {
        let entries = entries_small();
        let seed = [7u8; 32];
        let prover = block_bundle_commit(
            &entries.iter().map(|(i, m)| (*i, m)).collect::<Vec<_>>(),
            seed,
            4,
            2,
        )
        .expect("commit");
        let commitment = prover.commitment_bytes();
        let flat_log = prover.packed.shape.flat_log;
        let point: Vec<Goldilocks> = (0..flat_log)
            .map(|i| fe((i as u64 * 37 + 11) % 65521))
            .collect();
        let w = prover.flat_mle().evaluate(&point).expect("eval");
        let mut transcript = Transcript::new_default(b"lzx-block-test");
        let mut opening = prover.prove_block_opening(&point, &w, &mut transcript).expect("prove");
        // Corrupt the first ũ: breaks the interpolation + commute checks.
        if let Some(u) = opening.u_tilde.first_mut() {
            *u = fe(u.to_canonical_u64().wrapping_add(1));
        }
        let mut vt = Transcript::new_default(b"lzx-block-test");
        let factor_lens: Vec<usize> = entries.iter().map(|(_, m)| m.evaluations.len()).collect();
        assert!(verify_block_opening(
            seed,
            &commitment,
            &factor_lens,
            flat_log,
            &point,
            &w,
            &opening,
            &mut vt
        )
        .is_err());
    }

    #[test]
    fn block_wrong_seed_rejected() {
        let entries = entries_small();
        let seed = [7u8; 32];
        let prover = block_bundle_commit(
            &entries.iter().map(|(i, m)| (*i, m)).collect::<Vec<_>>(),
            seed,
            4,
            2,
        )
        .expect("commit");
        let commitment = prover.commitment_bytes();
        let flat_log = prover.packed.shape.flat_log;
        let point: Vec<Goldilocks> = (0..flat_log)
            .map(|i| fe((i as u64 * 37 + 11) % 65521))
            .collect();
        let w = prover.flat_mle().evaluate(&point).expect("eval");
        let mut transcript = Transcript::new_default(b"lzx-block-test");
        let opening = prover.prove_block_opening(&point, &w, &mut transcript).expect("prove");
        let mut vt = Transcript::new_default(b"lzx-block-test");
        let factor_lens: Vec<usize> = entries.iter().map(|(_, m)| m.evaluations.len()).collect();
        assert!(verify_block_opening(
            [8u8; 32],
            &commitment,
            &factor_lens,
            flat_log,
            &point,
            &w,
            &opening,
            &mut vt
        )
        .is_err());
    }

    #[test]
    fn block_commitment_is_one_k_vector() {
        // The transmitted commitment is k ring elements — r× smaller than
        // the compact mode's r·k layer.
        let entries = entries_small();
        let prover = block_bundle_commit(
            &entries.iter().map(|(i, m)| (*i, m)).collect::<Vec<_>>(),
            [7u8; 32],
            4,
            2,
        )
        .expect("commit");
        let n = prover.ring.n();
        // 4-byte count header + k·n·4 bytes — ONE k-vector of ring
        // elements (vs the compact mode's r·k).
        assert_eq!(prover.commitment_bytes().len(), 4 + 2 * n * 4);
        assert_eq!(prover.y.len(), 2);
    }
}
