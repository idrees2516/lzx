//! Native pure-std Rust port of the LaBRADOR (Dachshund) proof system from the lattice-dogs
//! reference (as vendored by osdnk/labinius at `LOGQ = 48`).
//!
//! Ring `Z_Q[X]/(X^64 + 1)`, `Q = 2^48 - 59`. Upstream represents ring elements as `polx`
//! (8-prime RNS NTT images in i16 lanes, AVX-512); this port uses exact centered `i64`
//! coefficients with `i128` schoolbook negacyclic products — same algebra, no RNS.
//!
//! Statement language (the "Dachshund simple statement"): `r` witness vectors — vector `i` of
//! rank `n[i]` with either an exact l2 bound `betasq[i]` or a binary requirement — plus `k`
//! dot-product constraints `sum_j <phi_j, s[idx_j][off_j..off_j+len_j]> = b` (plain negacyclic
//! inner products; upstream's truncated *extension products* of degree `deg` are expressed here
//! as `deg` separate degree-1 constraints against a proportionally expanded key — see the
//! crate docs for the trade-off).
//!
//! Protocol (from `labrador.c` + `dachshund.c`): parameter selection by the SIS rule, witness
//! expansion with binary norm-slack, the sigma/flip conjugate construction with `2^BL`-digit
//! lifting, inner/outer commitments with quadratic garbage, the JL projection, challenge
//! aggregation (alpha/beta/gamma/delta), amortization into one short opening `z` with digit
//! decomposition, and the full verifier.

// Upstream kernel structure: loops index with strides and table positions
// (`batches[c * nr + i]`, `lut[3 * k + r]`), which the range-loop lint's iterator
// suggestions cannot express. The patterns are verbatim from the ported reference.
#![allow(clippy::needless_range_loop)]
pub mod core;
pub mod ring;

pub use ring::{Poly, Q, Q_INV, N};

use lattice_core::keccak::{sha3_256, KeccakSponge};

/// `LOGQ`.
pub const LOGQ: usize = 48;
/// Largest witness coefficient the exact accumulator handles (upstream's `WITNESS_COEFF_MAX`).
pub const WITNESS_COEFF_MAX: i64 = 23170;

/// One witness vector: rank `n` (ring elements), l2 bound or binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorSpec {
    pub n: usize,
    pub betasq: u64,
    pub binary: bool,
}

impl VectorSpec {
    pub fn norm_bounded(n: usize, betasq: u64) -> Self {
        Self { n, betasq, binary: false }
    }
    pub fn binary(n: usize) -> Self {
        Self { n, betasq: 0, binary: true }
    }
}

/// A contiguous slice `s[idx][off .. off + len]` of one witness vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub idx: usize,
    pub off: usize,
    pub len: usize,
}

impl Block {
    pub fn new(idx: usize, off: usize, len: usize) -> Self {
        Self { idx, off, len }
    }
}

/// One dot-product constraint: `sum_j <phi_j, s[block_j]> = b` (one ring element; degree 1).
#[derive(Clone, Debug)]
pub struct Constraint {
    pub blocks: Vec<Block>,
    /// `phi` blocks, each `len` ring elements (coefficient form, centered).
    pub phi: Vec<Vec<Poly>>,
    pub b: Option<Poly>,
}

impl Constraint {
    pub fn new(blocks: Vec<Block>, phi: Vec<Vec<Poly>>, b: Option<Poly>) -> Self {
        Self { blocks, phi, b }
    }
}

/// A Dachshund simple statement.
#[derive(Clone, Debug)]
pub struct Statement {
    pub vectors: Vec<VectorSpec>,
    pub constraints: Vec<Constraint>,
    /// The 32-byte digest binding the whole statement (upstream: caller-supplied).
    pub digest: [u8; 32],
}

impl Statement {
    pub fn new(vectors: Vec<VectorSpec>, constraints: Vec<Constraint>) -> Self {
        let mut st = Self {
            vectors,
            constraints,
            digest: [0u8; 32],
        };
        st.digest = st.content_digest();
        st
    }

    pub fn with_digest(vectors: Vec<VectorSpec>, constraints: Vec<Constraint>, digest: [u8; 32]) -> Self {
        Self { vectors, constraints, digest }
    }

    pub fn total_rank(&self) -> usize {
        self.vectors.iter().map(|v| v.n).sum()
    }

    /// A digest binding every input: vector specs and, per constraint, blocks, phi and b.
    pub fn content_digest(&self) -> [u8; 32] {
        let mut h = KeccakSponge::new_sha3_256();
        fn upd(h: &mut KeccakSponge, bytes: &[u8]) {
            h.update(&(bytes.len() as u64).to_le_bytes());
            h.update(bytes);
        }
        h.update(b"labrador/statement/v1");
        h.update(&((LOGQ) as u64).to_le_bytes());
        h.update(&(self.vectors.len() as u64).to_le_bytes());
        for v in &self.vectors {
            h.update(&(v.n as u64).to_le_bytes());
            h.update(&v.betasq.to_le_bytes());
            h.update(&[u8::from(v.binary)]);
        }
        h.update(&(self.constraints.len() as u64).to_le_bytes());
        for c in &self.constraints {
            h.update(&(c.blocks.len() as u64).to_le_bytes());
            for blk in &c.blocks {
                h.update(&(blk.idx as u64).to_le_bytes());
                h.update(&(blk.off as u64).to_le_bytes());
                h.update(&(blk.len as u64).to_le_bytes());
            }
            for phi in &c.phi {
                for p in phi {
                    for &x in p.0.iter() {
                        h.update(&x.to_le_bytes());
                    }
                }
            }
            match &c.b {
                None => h.update(b"hom"),
                Some(b) => {
                    h.update(b"b");
                    for &x in b.0.iter() {
                        h.update(&x.to_le_bytes());
                    }
                }
            }
        }
        let _ = &upd;
        let out = h.finalize(32);
        out.try_into().unwrap()
    }
}

/// A witness: one vector of `n * 64` centered i16 coefficients per statement vector.
#[derive(Clone, Debug, Default)]
pub struct Witness {
    pub vectors: Vec<Vec<i16>>,
}

impl Witness {
    pub fn new(vectors: Vec<Vec<i16>>) -> Self {
        Self { vectors }
    }

    pub fn normsq(&self, i: usize) -> u64 {
        self.vectors[i]
            .iter()
            .map(|&c| (c as i64 * c as i64) as u64)
            .sum()
    }
}

/// The produced proof (analytic size in KB reported alongside).
pub struct Proof {
    /// First outer commitment openings (kappa1 ring elements).
    pub u1: Vec<Poly>,
    /// Second outer commitment openings (kappa1 ring elements).
    pub u2: Vec<Poly>,
    /// The JL projection (256 signed sums) and its nonce.
    pub p: [i32; 256],
    pub jlnonce: u64,
    /// The amortized witness: `f` digit vectors of rank `n` plus the aux vector of rank `m`.
    pub digits: Vec<Vec<i16>>,
    pub aux: Vec<i16>,
    /// LaBRADOR's announced output norm bound.
    pub normsq: u64,
    /// Commitment of the simple statement (kappa1 ring elements, the "u0").
    pub com_u: Vec<Poly>,
}

/// Prove `wit` satisfies `stmt`. Returns the proof.
pub fn prove(stmt: &Statement, wit: &Witness) -> Result<Proof, String> {
    core::prove(stmt, wit)
}

/// Verify `proof` against `stmt`.
pub fn verify(stmt: &Statement, proof: &Proof) -> Result<(), String> {
    core::verify(stmt, proof)
}

/// The smallest commitment rank LaBRADOR's SIS rule calls secure for a norm (slack applied by
/// the caller).
pub fn sis_secure(rank: usize, norm: f64) -> bool {
    let log_delta = 1.00444f64.log2();
    let maxlog = 2.0 * (LOGQ as f64 * log_delta * N as f64).sqrt() * (rank as f64).sqrt();
    let maxlog = maxlog.min(LOGQ as f64);
    norm.log2() < maxlog
}

pub fn sis_rank(norm: f64) -> usize {
    (1..=32)
        .find(|&k| sis_secure(k, norm))
        .ok_or_else(|| "no commitment rank at or below 32 is SIS-secure for this norm".to_string())
        .unwrap()
}

/// Deterministic seed bytes for key expansion (SHAKE-based, replacing upstream's AES-CTR).
pub fn seed_bytes(label: &[u8], nonce: u64) -> [u8; 32] {
    let mut h = KeccakSponge::new_shake256();
    h.update(b"labrador/comkey/v1");
    h.update(label);
    h.update(&nonce.to_le_bytes());
    h.finalize(32).try_into().unwrap()
}

pub fn sha3(bytes: &[u8]) -> [u8; 32] {
    sha3_256(bytes)
}
