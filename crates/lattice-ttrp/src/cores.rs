//! Tensor-train cores and the `χ_TT` sampler (ePrint 2026/2146, §2–3).
//!
//! A tensor-train (TT) row is a chain of 3rd-order core tensors
//! `M_1 ∈ Z^{1×d×c}, M_i ∈ Z^{c×d×c} (1 < i < µ), M_µ ∈ Z^{c×d×1}`
//! whose entries are i.i.d. from the distribution
//! `D_ghl = Bin(2, 1/2) − 1` (0 with probability 1/2, ±1 with probability
//! 1/4 each — Definition 4's adaptation of the Gaussian/Rademacher cores of
//! [RR20] to modular arithmetic).
//!
//! The materialised row is `Mat(M_1 ⊗ ⋯ ⊗ M_µ) ∈ Z^{1×d^µ}` where the
//! Kronecker product of 3rd-order tensors acts slice-wise (Eq. (6)):
//! `(A ⊗ B)(i, j) = A(i)·B(j)`, and `Mat` horizontally concatenates the
//! matrix slices. Entry `n` of the row (n's base-d digits `n_1..n_µ`,
//! most-significant first) is the matrix chain product
//! `M_1(n_1)·M_2(n_2)⋯M_µ(n_µ)`.
//!
//! Sampling is deterministic from a 32-byte seed via SHAKE-256 streams
//! (`χ_TT(k, m̄r·φ, d, µ, c)` in the paper's notation): prover and verifier
//! regenerate identical cores from the same transcript state.

use lattice_core::keccak::shake256;

/// One 3rd-order core tensor, flat layout `[slice][row][col]` with shape
/// `(r0, d, r1)`: `M[slice]` is the `slice`-th `r0×r1` matrix slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreTensor {
    pub r0: usize,
    pub d: usize,
    pub r1: usize,
    /// Flat signed entries in {-1, 0, +1}, length `r0*d*r1`.
    pub entries: Vec<i8>,
}

impl CoreTensor {
    /// The `(a, b)` entry of slice `s`: `M[s][a][b]`.
    #[inline]
    pub fn at(&self, s: usize, a: usize, b: usize) -> i8 {
        self.entries[s * self.r0 * self.r1 + a * self.r1 + b]
    }

    /// `Mat(M)` — the `r0 × (d·r1)` matrix obtained by horizontally
    /// concatenating the matrix slices (paper §2).
    pub fn mat(&self) -> Vec<i64> {
        let mut out = vec![0i64; self.r0 * self.d * self.r1];
        for a in 0..self.r0 {
            for s in 0..self.d {
                for b in 0..self.r1 {
                    out[a * self.d * self.r1 + s * self.r1 + b] = self.at(s, a, b) as i64;
                }
            }
        }
        out
    }

    /// Total entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Full parameter set of a TTRP instance (Definition 4 + Figure 1).
///
/// Invariants enforced by [`TtrpParams::validate`]:
/// * `m̄r = 2^nu` — the witness vector length over `R`;
/// * `phi = 2^phi_log` — the ring degree;
/// * `d = 2^ell` — the core slice dimension;
/// * `ell * (mu1 + mu2) = nu + phi_log` — the spatial cores index the
///   `m̄r` ring positions and the coefficient cores index the `φ`
///   coefficients, together covering the `m̄r·φ = d^µ` integer columns;
/// * `k` projection rows, internal rank `c`, `k1` aggregation rows for the
///   challenge matrix `Γ ∈ Z_q^{k1×k}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TtrpParams {
    /// log2 of the ring degree φ.
    pub phi_log: u32,
    /// ν: log2 of the witness length `m̄r` over R.
    pub nu: usize,
    /// ℓ: log2 of the core dimension d.
    pub ell: usize,
    /// Number of spatial cores (index the ring positions).
    pub mu1: usize,
    /// Number of coefficient cores (index the coefficient of each ring elt).
    pub mu2: usize,
    /// Internal TT rank c ≥ 1.
    pub c: usize,
    /// Projection row count k ≥ 1.
    pub k: usize,
    /// Aggregation row count k' ≥ 1 (challenge matrix Γ rows).
    pub k1: usize,
}

impl TtrpParams {
    pub fn phi(&self) -> usize {
        1usize << self.phi_log
    }
    /// `m̄r` — the witness vector length over R.
    pub fn m_bar(&self) -> usize {
        1usize << self.nu
    }
    /// d — the core slice dimension.
    pub fn d(&self) -> usize {
        1usize << self.ell
    }
    /// µ = µ1 + µ2 — total core count.
    pub fn mu(&self) -> usize {
        self.mu1 + self.mu2
    }
    /// Total integer column count `m̄r·φ = d^µ`.
    pub fn cols(&self) -> usize {
        self.m_bar() * self.phi()
    }

    pub fn validate(&self) -> Result<(), TtrpCoreError> {
        if self.phi_log == 0 || self.phi_log > 16 {
            return Err(TtrpCoreError::BadParams {
                why: "phi_log must be in [1, 16]".into(),
            });
        }
        if self.ell == 0 || self.ell > 8 {
            return Err(TtrpCoreError::BadParams {
                why: "ell must be in [1, 8]".into(),
            });
        }
        if self.mu1 == 0 || self.mu2 == 0 {
            return Err(TtrpCoreError::BadParams {
                why: "mu1 and mu2 must be >= 1".into(),
            });
        }
        if self.ell * self.mu() != self.nu + self.phi_log as usize {
            return Err(TtrpCoreError::BadParams {
                why: format!(
                    "ell*(mu1+mu2) = {} must equal nu + phi_log = {}",
                    self.ell * self.mu(),
                    self.nu + self.phi_log as usize
                ),
            });
        }
        if self.c == 0 || self.k == 0 || self.k1 == 0 {
            return Err(TtrpCoreError::BadParams {
                why: "c, k, k1 must be >= 1".into(),
            });
        }
        Ok(())
    }

    /// Total core-tensor entry count (the TT-format representation size):
    /// per row `d·(2c + (µ−2)·c²)`, over k rows — Table 4's "Repr." column.
    pub fn representation_entries(&self) -> usize {
        let per_row = self.d() * (2 * self.c + self.mu().saturating_sub(2) * self.c * self.c);
        per_row * self.k
    }

    /// The shape (r0, r1) of core `i` (0-indexed).
    pub fn core_shape(&self, i: usize) -> (usize, usize) {
        let mu = self.mu();
        if i == 0 {
            (1, self.c)
        } else if i + 1 == mu {
            (self.c, 1)
        } else {
            (self.c, self.c)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtrpCoreError {
    BadParams { why: String },
}

impl core::fmt::Display for TtrpCoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TtrpCoreError::BadParams { why } => write!(f, "ttrp params: {why}"),
        }
    }
}

/// Sample the full core stack `tM_i^{(j)}u_{i∈[µ], j∈[k]} ← χ_TT` from a
/// 32-byte seed. Deterministic; consumed by both prover and verifier.
///
/// Each entry draws 2 bits from a SHAKE-256 stream: 0 or 3 → 0 (prob. 1/2),
/// 1 → +1 (prob. 1/4), 2 → −1 (prob. 1/4) — exactly `D_ghl`.
pub fn sample_cores(params: &TtrpParams, seed: &[u8]) -> Vec<Vec<CoreTensor>> {
    let total: usize = params.k
        * params
            .mu()
            .saturating_sub(2)
            .saturating_mul(params.c * params.c * params.d())
        + params.k * 2 * params.c * params.d();
    // 2 bits per entry; over-allocate the stream generously.
    let bytes_needed = total.div_ceil(4) + 64;
    let mut input = b"lattice-ttrp/chi-tt/core-stream\x00".to_vec();
    input.extend_from_slice(seed);
    let stream = shake256(&input, bytes_needed);
    let mut bit_pos = 0usize;
    let mut next_bits = |n: usize| -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = stream[bit_pos / 8];
            let bit = (byte >> (bit_pos % 8)) & 1;
            v = (v << 1) | bit as u32;
            bit_pos += 1;
        }
        v
    };
    let mut rows = Vec::with_capacity(params.k);
    for _ in 0..params.k {
        let mut cores = Vec::with_capacity(params.mu());
        for i in 0..params.mu() {
            let (r0, r1) = params.core_shape(i);
            let len = r0 * params.d() * r1;
            let mut entries = Vec::with_capacity(len);
            for _ in 0..len {
                entries.push(match next_bits(2) {
                    1 => 1i8,
                    2 => -1i8,
                    _ => 0i8,
                });
            }
            cores.push(CoreTensor {
                r0,
                d: params.d(),
                r1,
                entries,
            });
        }
        rows.push(cores);
    }
    rows
}

/// Materialise the full TT row `Mat(M_1 ⊗ ⋯ ⊗ M_µ) ∈ Z^{1×d^µ}` by the
/// left-to-right mixed-product chain (Fact 1):
/// `R_1 = Mat(M_1)`, `R_{p+1} = R_p · (I_{d^p} ⊗ Mat(M_{p+1}))`.
///
/// Only used for testing / small instances (the protocol never
/// materialises the row).
pub fn materialize_row(cores: &[CoreTensor]) -> Vec<i64> {
    let mu = cores.len();
    let d = cores[0].d;
    // R is the partial chain product as a 1 × (d^p · c_p) row.
    let (mut r, mut r1_dim, mut dd) = {
        let m0 = cores[0].mat(); // 1 × (d*c)
        (m0, cores[0].r1, d)
    };
    for core in &cores[1..] {
        // (I_{dd} ⊗ Mat(core)): block-diagonal with dd blocks of Mat(core).
        // R (1 × dd*r1_dim) times it: R viewed as dd chunks of length
        // r1_dim; each chunk multiplies Mat(core) (r1_dim × d*core.r1),
        // and the outputs concatenate horizontally per chunk row.
        let mc = core.mat(); // r1_dim × (d * core.r1)
        let block_cols = d * core.r1;
        let mut next = vec![0i64; dd * block_cols];
        for blk in 0..dd {
            for s in 0..d {
                for b in 0..core.r1 {
                    // entry (row 0 of R? no — R is 1×(dd*r1_dim):
                    // out[blk][s][b] = Σ_a R[blk*r1_dim + a] * mc[a][s][b]
                    let mut acc = 0i64;
                    for a in 0..r1_dim {
                        acc += r[blk * r1_dim + a] * mc[a * block_cols + s * core.r1 + b];
                    }
                    next[blk * block_cols + s * core.r1 + b] = acc;
                }
            }
        }
        r = next;
        r1_dim = core.r1;
        dd *= d;
    }
    debug_assert_eq!(r.len(), dd.max(1) * r1_dim.max(1));
    let _ = mu;
    r
}

/// Naive TT row entry at column `n` (digits most-significant first) via the
/// chain product of slices — the ground-truth for
/// [`materialize_row`] tests and for small cross-checks.
#[allow(clippy::needless_range_loop)]
pub fn tt_entry_naive(cores: &[CoreTensor], n: usize) -> i64 {
    let mu = cores.len();
    let d = cores[0].d;
    // The first core has shape 1×d×c: the running accumulator is its
    // n_1-th slice, a 1×c row.
    let n1 = n / d.pow((mu - 1) as u32);
    let (mut acc, mut acc_w) = {
        let r1 = cores[0].r1;
        let mut v = vec![0i64; r1];
        for b in 0..r1 {
            v[b] = cores[0].at(n1, 0, b) as i64;
        }
        (v, r1)
    };
    let mut rem = n % d.pow((mu - 1) as u32);
    for (i, core) in cores.iter().enumerate().skip(1) {
        let digits_left = mu - 1 - i;
        let s = rem / d.pow(digits_left as u32);
        rem %= d.pow(digits_left as u32);
        // acc (1×acc_w) times slice M_i(s) (acc_w × r1): result 1×r1.
        let mut next = vec![0i64; core.r1];
        for b in 0..core.r1 {
            let mut sum = 0i64;
            for a in 0..core.r0 {
                sum += acc[a] * core.at(s, a, b) as i64;
            }
            next[b] = sum;
        }
        acc = next;
        acc_w = core.r1;
    }
    debug_assert_eq!(acc.len(), 1);
    let _ = acc_w;
    acc[0]
}
