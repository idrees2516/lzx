//! The zkVM memory arguments (Wave 7.4 P0-4/P0-5): register file, RAM, and
//! instruction fetch as Twist & Shout instances over the *digit-bit*
//! representation (ePrint 2025/105, Figs 8-9).
//!
//! * The one-hot matrices `ra`/`wa` and the increment matrix `Inc` are
//!   **virtual** — pure functions of the committed digit-bit tensor (the
//!   address bits of each stream slot), the committed activity columns,
//!   and the committed offset-encoded increment column. Every evaluation
//!   claim on them is proven by a dedicated **matrix-evaluation
//!   sumcheck** that expands the one-hot selection as a product of affine
//!   digit-bit factors. The `K × T_s` matrices are materialized
//!   prover-side for the main legs' round polynomials only — never
//!   committed, so the committed universe stays `O(T)`.
//! * The current-value matrix `Val` is **virtual in the paper's sense**
//!   (Fig 9 Eq 11): `Val(k, j) = init(k) + Σ_{j' < j} Inc(k, j')`,
//!   materialized prover-side but pinned by the **Val-evaluation
//!   sumcheck** — `Val(r_a, r_c) = init̃(r_a) + Σ_{j'} Inc̃(r_a, j')·
//!   LT(j', r_c)` — with `LT` evaluated by the verifier itself. This
//!   closes the stale-read hole of a bare materialized-Val formulation.
//! * Runs are per-limb: four instances per memory (register file / RAM),
//!   each carrying 16-bit read/write values, so every value `< 2^16 < p`
//!   and the field encoding is canonical.
//! * Stream layout: slot-major `j = slot·T + t`, RAM slots
//!   `[read, write, pad, pad]`, register slots `[rs1, rs2, rd-write,
//!   pad]`; variable order everywhere is `(address/bit block MSB-first,
//!   then slot/cycle block)`.
//!
//! Leg order per read-write instance (the exact transcript flow both
//! sides replay): `B` booleanity → `R` raf → `C` read-checking → `Ma`
//! matrix-eval(ra at C's point) → `V0` Val-eval → `Mu0` matrix-eval(Inc
//! at V0's point) → `W` write-checking → `Mb` matrix-eval(wa) → `Mc`
//! matrix-eval(Inc at W's point) → `V1` Val-eval → `Mu1` matrix-eval(Inc
//! at V1's point) → `T` telescoping → `Md` matrix-eval(Inc at T's
//! point). Read-only (fetch) instances stop after `Ma`.

use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::sumcheck;
use lattice_sumcheck::sumcheck::SumcheckOutput;
use lattice_sumcheck::SumcheckProof;
use lattice_sumcheck::VirtualPolynomial;

use crate::ledger::{Factor, Ledger, LedgerError};

/// The offset applied to increment values: `inc' = inc + INC_OFFSET`
/// (non-negative, `< 2^18`).
pub const INC_OFFSET: u64 = 1 << 17;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    Ledger(LedgerError),
    Sumcheck(lattice_sumcheck::SumcheckError),
    Batch(lattice_sumcheck::batch::BatchError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Transcript(lattice_core::transcript::TranscriptError),
    Mle(lattice_core::mle::MleError),
    FinalCheck(&'static str),
    ClaimMismatch,
    Shape,
}

impl From<LedgerError> for MemoryError {
    fn from(e: LedgerError) -> Self {
        MemoryError::Ledger(e)
    }
}

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

/// Prover-side instance data.
#[derive(Clone, Debug)]
pub struct MemoryInstance {
    pub log_k: usize,
    pub log_ts: usize,
    /// Stream addresses (length T_s).
    pub addr: Vec<u64>,
    /// Read/write activity per slot.
    pub ractive: Vec<u8>,
    pub wactive: Vec<u8>,
    /// The read-value column (length T_s; 0 at read-inactive slots).
    pub rv: Vec<Goldilocks>,
    /// The write-value column (0 at write-inactive slots).
    pub wv: Vec<Goldilocks>,
    /// The offset-encoded increment column (`INC_OFFSET` at non-write
    /// slots).
    pub inc_off: Vec<Goldilocks>,
    /// Public initial state (length K).
    pub init: Vec<Goldilocks>,
    /// Final state (length K) — public via the statement.
    pub final_state: Vec<Goldilocks>,
    /// Read-only instances (fetch): the public table over K.
    pub table: Option<Vec<Goldilocks>>,
}

impl MemoryInstance {
    pub fn t_s(&self) -> usize {
        1usize << self.log_ts
    }

    pub fn k(&self) -> usize {
        1usize << self.log_k
    }

    pub fn read_only(&self) -> bool {
        self.table.is_some()
    }

    /// The digit-block row count: address bits padded to a power of two
    /// (the bit-block's variable count is `log_rows()`).
    pub fn log_rows(&self) -> usize {
        let rows = self.log_k.next_power_of_two();
        rows.trailing_zeros() as usize
    }

    /// The digit-bit tensor: MLE over `(log_rows + log_ts)`; row `i` is
    /// address bit `log_k − 1 − i` (MSB-first; rows `log_k..` are zero
    /// padding); index `i * T_s + j`.
    pub fn digit_tensor(&self) -> DenseMle {
        let t_s = self.t_s();
        let rows = 1usize << self.log_rows();
        let mut evals = Vec::with_capacity(rows * t_s);
        for i in 0..rows {
            let bit = self.log_k.wrapping_sub(1).wrapping_sub(i);
            for j in 0..t_s {
                let v = if i < self.log_k {
                    (self.addr[j] >> (self.log_k - 1 - i)) & 1
                } else {
                    0
                };
                let _ = bit;
                evals.push(fe(v));
            }
        }
        DenseMle {
            num_vars: self.log_rows() + self.log_ts,
            evaluations: evals,
        }
    }

    /// The raf weight table over the padded digit rows: row `i` weighs
    /// `2^(log_k − 1 − i)` for `i < log_k`, else 0.
    pub fn raf_weights(&self) -> DenseMle {
        let rows = 1usize << self.log_rows();
        let evals: Vec<Goldilocks> = (0..rows)
            .map(|i| {
                if i < self.log_k {
                    fe(1u64 << (self.log_k - 1 - i))
                } else {
                    Goldilocks::ZERO
                }
            })
            .collect();
        DenseMle {
            num_vars: self.log_rows(),
            evaluations: evals,
        }
    }

    /// The read/write activity stream column.
    pub fn activity_col(&self, write: bool) -> DenseMle {
        let active = if write { &self.wactive } else { &self.ractive };
        DenseMle {
            num_vars: self.log_ts,
            evaluations: (0..self.t_s()).map(|j| fe(active[j] as u64)).collect(),
        }
    }

    pub fn addr_col(&self) -> DenseMle {
        DenseMle {
            num_vars: self.log_ts,
            evaluations: self.addr.iter().map(|&a| fe(a)).collect(),
        }
    }

    pub fn rv_col(&self) -> DenseMle {
        DenseMle {
            num_vars: self.log_ts,
            evaluations: self.rv.clone(),
        }
    }

    pub fn wv_col(&self) -> DenseMle {
        DenseMle {
            num_vars: self.log_ts,
            evaluations: self.wv.clone(),
        }
    }

    pub fn inc_col(&self) -> DenseMle {
        DenseMle {
            num_vars: self.log_ts,
            evaluations: self.inc_off.clone(),
        }
    }

    /// The materialized one-hot matrix for a side: MLE over
    /// `(log_k + log_ts)`, index `k * T_s + j`.
    fn one_hot(&self, write: bool) -> DenseMle {
        let t_s = self.t_s();
        let active = if write { &self.wactive } else { &self.ractive };
        let mut evals = vec![Goldilocks::ZERO; self.k() * t_s];
        for j in 0..t_s {
            if active[j] != 0 && self.addr[j] < self.k() as u64 {
                evals[self.addr[j] as usize * t_s + j] = Goldilocks::ONE;
            }
        }
        DenseMle {
            num_vars: self.log_k + self.log_ts,
            evaluations: evals,
        }
    }

    /// The materialized increment matrix (the virtual `Inc`).
    fn inc_matrix(&self) -> DenseMle {
        let t_s = self.t_s();
        let mut evals = vec![Goldilocks::ZERO; self.k() * t_s];
        for j in 0..t_s {
            if self.wactive[j] != 0 && self.addr[j] < self.k() as u64 {
                let inc = self.inc_off[j].sub(&fe(INC_OFFSET));
                evals[self.addr[j] as usize * t_s + j] = inc;
            }
        }
        DenseMle {
            num_vars: self.log_k + self.log_ts,
            evaluations: evals,
        }
    }

    /// The materialized current-value matrix (prover-side; virtual to the
    /// verifier): `val[k·T_s + j]` = the value of address k before slot
    /// j's write.
    fn val_matrix(&self) -> DenseMle {
        let t_s = self.t_s();
        let mut current = self.init.clone();
        let mut evals = vec![Goldilocks::ZERO; self.k() * t_s];
        for j in 0..t_s {
            for k in 0..self.k() {
                evals[k * t_s + j] = current[k];
            }
            if self.wactive[j] != 0 && self.addr[j] < self.k() as u64 {
                current[self.addr[j] as usize] = self.wv[j];
            }
        }
        DenseMle {
            num_vars: self.log_k + self.log_ts,
            evaluations: evals,
        }
    }
}

/// One leg's proof: name + sumcheck + its claim value.
#[derive(Clone, Debug)]
pub struct LegProof {
    pub name: &'static str,
    pub sc: SumcheckProof,
    pub claim: Goldilocks,
}

/// The full memory proof for one instance (legs in protocol order).
#[derive(Clone, Debug)]
pub struct MemoryProof {
    pub legs: Vec<LegProof>,
}

pub(crate) fn absorb(
    transcript: &mut Transcript,
    inst: usize,
    leg: &str,
    log_k: usize,
    log_ts: usize,
) -> Result<(), MemoryError> {
    transcript
        .append_message(
            b"mem-leg",
            &[
                inst as u8,
                (inst >> 8) as u8,
                leg.as_bytes()[0],
                leg.as_bytes().get(1).copied().unwrap_or(0),
                log_k as u8,
                log_ts as u8,
            ],
        )
        .map_err(MemoryError::Transcript)
}

/// Factors for instance `inst`'s stream columns.
pub fn rv_factor(inst: usize) -> Factor {
    Factor::RvCol { inst }
}

pub fn wv_factor(inst: usize) -> Factor {
    Factor::WvCol { inst }
}

pub fn inc_factor(inst: usize) -> Factor {
    Factor::IncCol { inst }
}

pub fn addr_factor(inst: usize) -> Factor {
    Factor::AddrCol { inst }
}

pub fn activity_factor(inst: usize, write: bool) -> Factor {
    Factor::ActiveCol {
        id: 2 * inst + write as usize,
    }
}

/// Prove one instance's full memory argument. The legs are appended to
/// `legs` in protocol order.
pub fn prove_memory(
    inst: usize,
    m: &MemoryInstance,
    ledger: &mut Ledger<'_>,
    legs: &mut Vec<LegProof>,
    transcript: &mut Transcript,
) -> Result<(), MemoryError> {
    let log_k = m.log_k;
    let log_ts = m.log_ts;
    let digits = m.digit_tensor();

    // ---- B: digit booleanity: Σ eq·D·(D−1) = 0 over the padded
    //      (rows, j) space. ----
    let log_rows = m.log_rows();
    absorb(transcript, inst, "B", log_k, log_ts)?;
    let r_bool = transcript
        .challenge_fields(b"mem-bool-r", log_rows + log_ts)
        .map_err(MemoryError::Transcript)?;
    {
        let eq = DenseMle::eq_extension(&r_bool);
        let d_minus_1: Vec<Goldilocks> = digits
            .evaluations
            .iter()
            .map(|v| v.sub(&Goldilocks::ONE))
            .collect();
        let dm = DenseMle {
            num_vars: log_rows + log_ts,
            evaluations: d_minus_1,
        };
        let mut vp = VirtualPolynomial::new(log_rows + log_ts);
        let di = vp
            .add_factor(digits.clone())
            .map_err(MemoryError::Virtual)?;
        let dmi = vp.add_factor(dm).map_err(MemoryError::Virtual)?;
        let ei = vp.add_factor(eq).map_err(MemoryError::Virtual)?;
        vp.add_term(Goldilocks::ONE, vec![di, dmi, ei])
            .map_err(MemoryError::Virtual)?;
        let out =
            sumcheck::prove(&vp, Goldilocks::ZERO, transcript).map_err(MemoryError::Sumcheck)?;
        // Bind the factor claims: [D, D−1, eq] at the terminal point.
        let d_at = ledger.tensor_claim(Factor::DigitBits { inst }, &out.challenges)?;
        if d_at != out.factor_claims[0]
            || d_at.sub(&Goldilocks::ONE) != out.factor_claims[1]
            || DenseMle::eq_eval(&r_bool, &out.challenges).map_err(MemoryError::Mle)?
                != out.factor_claims[2]
        {
            return Err(MemoryError::FinalCheck("booleanity binding"));
        }
        legs.push(LegProof {
            name: "B",
            sc: out.proof,
            claim: Goldilocks::ZERO,
        });
    }

    // ---- R: raf: Σ eq(r'_j, j)·w(row)·D(row, j) = addr_col(r'_j). ----
    absorb(transcript, inst, "R", log_k, log_ts)?;
    let r_prime = transcript
        .challenge_fields(b"mem-raf-r", log_ts)
        .map_err(MemoryError::Transcript)?;
    let addr_claim = ledger.tensor_claim(addr_factor(inst), &r_prime)?;
    {
        let log_rows = m.log_rows();
        let eq_j = DenseMle::one(log_rows).tensor(&DenseMle::eq_extension(&r_prime));
        let w_ext = m.raf_weights().tensor(&DenseMle::one(log_ts));
        let mut vp = VirtualPolynomial::new(log_rows + log_ts);
        let ei = vp.add_factor(eq_j).map_err(MemoryError::Virtual)?;
        let wi = vp.add_factor(w_ext).map_err(MemoryError::Virtual)?;
        let di = vp
            .add_factor(digits.clone())
            .map_err(MemoryError::Virtual)?;
        vp.add_term(Goldilocks::ONE, vec![ei, wi, di])
            .map_err(MemoryError::Virtual)?;
        let out = sumcheck::prove(&vp, addr_claim, transcript).map_err(MemoryError::Sumcheck)?;
        // Factor claims: [eq_j, w, D] at the terminal point.
        let d_at = ledger.tensor_claim(Factor::DigitBits { inst }, &out.challenges)?;
        let eq_at =
            DenseMle::eq_eval(&r_prime, &out.challenges[log_rows..]).map_err(MemoryError::Mle)?;
        if d_at != out.factor_claims[2]
            || eq_at != out.factor_claims[0]
            || m.raf_weights()
                .evaluate(&out.challenges[..log_rows])
                .map_err(MemoryError::Mle)?
                != out.factor_claims[1]
        {
            return Err(MemoryError::FinalCheck("raf binding"));
        }
        legs.push(LegProof {
            name: "R",
            sc: out.proof,
            claim: addr_claim,
        });
    }

    // ---- C: read-checking: rv(r_c) = Σ eq(r_c,j)·ra·Val (fetch: table). ----
    absorb(transcript, inst, "C", log_k, log_ts)?;
    let r_c = transcript
        .challenge_fields(b"mem-read-r", log_ts)
        .map_err(MemoryError::Transcript)?;
    let rv_claim = ledger.tensor_claim(rv_factor(inst), &r_c)?;
    let ra = m.one_hot(false);
    let val = m.val_matrix();
    let read_point: Vec<Goldilocks>;
    let read_ra_claim: Goldilocks;
    let read_val_claim: Goldilocks;
    let read_final_claim: Goldilocks;
    {
        let eq_j = DenseMle::one(log_k).tensor(&DenseMle::eq_extension(&r_c));
        let val_factor = match &m.table {
            Some(table) => DenseMle::new(table.clone())
                .map_err(MemoryError::Mle)?
                .tensor(&DenseMle::one(log_ts)),
            None => val.clone(),
        };
        let mut vp = VirtualPolynomial::new(log_k + log_ts);
        let ei = vp.add_factor(eq_j).map_err(MemoryError::Virtual)?;
        let ri = vp.add_factor(ra.clone()).map_err(MemoryError::Virtual)?;
        let vi = vp.add_factor(val_factor).map_err(MemoryError::Virtual)?;
        vp.add_term(Goldilocks::ONE, vec![ei, ri, vi])
            .map_err(MemoryError::Virtual)?;
        let out = sumcheck::prove(&vp, rv_claim, transcript).map_err(MemoryError::Sumcheck)?;
        read_point = out.challenges.clone();
        read_ra_claim = out.factor_claims[1];
        read_val_claim = out.factor_claims[2];
        read_final_claim = out.final_claim;
        legs.push(LegProof {
            name: "C",
            sc: out.proof,
            claim: rv_claim,
        });
    }

    // ---- Ma: matrix-eval of ra at the read leg's point. ----
    let ra_at = prove_matrix_eval(inst, "Ma", m, ledger, &read_point, legs, transcript)?;
    if ra_at != read_ra_claim {
        return Err(MemoryError::FinalCheck("read ra binding"));
    }
    if m.read_only() {
        // Terminal check: rv = eq(r_c, ρ_j)·ra(ρ)·Table(ρ_k).
        let (rho_k, rho_j) = read_point.split_at(log_k);
        let eq_v = DenseMle::eq_eval(&r_c, rho_j).map_err(MemoryError::Mle)?;
        let table = m.table.as_ref().ok_or(MemoryError::Shape)?;
        let table_mle = DenseMle::new(table.clone()).map_err(MemoryError::Mle)?;
        let t_at = table_mle.evaluate(rho_k).map_err(MemoryError::Mle)?;
        if eq_v.mul(&ra_at).mul(&t_at) != read_final_claim {
            return Err(MemoryError::FinalCheck("fetch read terminal"));
        }
        return Ok(());
    }

    // ---- V0 + Mu0: Val-evaluation at the read point. ----
    let val_at_read = prove_val_eval(inst, "V0", m, ledger, &read_point, legs, transcript)?;
    if val_at_read != read_val_claim {
        return Err(MemoryError::FinalCheck("read val binding"));
    }

    // ---- W: write-checking (zero form). ----
    absorb(transcript, inst, "W", log_k, log_ts)?;
    let r_w = transcript
        .challenge_fields(b"mem-write-r", log_k + log_ts)
        .map_err(MemoryError::Transcript)?;
    let write_point: Vec<Goldilocks>;
    let write_inc_claim: Goldilocks;
    let write_wa_claim: Goldilocks;
    let write_wv_claim: Goldilocks;
    let write_val_claim: Goldilocks;
    {
        let eq = DenseMle::eq_extension(&r_w);
        let wa = m.one_hot(true);
        let inc = m.inc_matrix();
        let wv_lift = DenseMle::one(log_k).tensor(&m.wv_col());
        let mut vp = VirtualPolynomial::new(log_k + log_ts);
        let ei = vp.add_factor(eq).map_err(MemoryError::Virtual)?;
        let ii = vp.add_factor(inc).map_err(MemoryError::Virtual)?;
        vp.add_term(Goldilocks::ONE, vec![ei, ii])
            .map_err(MemoryError::Virtual)?;
        let wi = vp.add_factor(wa.clone()).map_err(MemoryError::Virtual)?;
        let wvi = vp.add_factor(wv_lift).map_err(MemoryError::Virtual)?;
        let vali = vp.add_factor(val.clone()).map_err(MemoryError::Virtual)?;
        vp.add_term(Goldilocks::ONE.neg(), vec![ei, wi, wvi])
            .map_err(MemoryError::Virtual)?;
        vp.add_term(Goldilocks::ONE, vec![ei, wi, vali])
            .map_err(MemoryError::Virtual)?;
        let out =
            sumcheck::prove(&vp, Goldilocks::ZERO, transcript).map_err(MemoryError::Sumcheck)?;
        write_point = out.challenges.clone();
        // Factor order in the W vp: [eq, inc, wa, wv-lift, val].
        write_inc_claim = out.factor_claims[1];
        write_wa_claim = out.factor_claims[2];
        write_wv_claim = out.factor_claims[3];
        write_val_claim = out.factor_claims[4];
        legs.push(LegProof {
            name: "W",
            sc: out.proof,
            claim: Goldilocks::ZERO,
        });
    }

    // ---- Mb / Mc + the wv column claim at the write point's j-part. ----
    let wa_at = prove_matrix_eval(inst, "Mb", m, ledger, &write_point, legs, transcript)?;
    if wa_at != write_wa_claim {
        return Err(MemoryError::FinalCheck("write wa binding"));
    }
    let inc_at_w = prove_matrix_eval(inst, "Mc", m, ledger, &write_point, legs, transcript)?;
    if inc_at_w != write_inc_claim {
        return Err(MemoryError::FinalCheck("write inc binding"));
    }
    let wv_at_w = ledger.tensor_claim(wv_factor(inst), &write_point[log_k..])?;
    if wv_at_w != write_wv_claim {
        return Err(MemoryError::FinalCheck("write wv binding"));
    }

    // ---- V1 + Mu1: Val-evaluation at the write point. ----
    let val_at_write = prove_val_eval(inst, "V1", m, ledger, &write_point, legs, transcript)?;
    if val_at_write != write_val_claim {
        return Err(MemoryError::FinalCheck("write val binding"));
    }

    // ---- T: telescoping: Σ_{k,j} eq(r_t,k)·Inc = Σ_k eq(r_t,k)(F−I). ----
    absorb(transcript, inst, "T", log_k, log_ts)?;
    let r_t = transcript
        .challenge_fields(b"mem-tel-r", log_k)
        .map_err(MemoryError::Transcript)?;
    let tel_claim = {
        let eq_k = DenseMle::eq_extension(&r_t);
        let mut acc = Goldilocks::ZERO;
        for (k, (f, i)) in m.final_state.iter().zip(m.init.iter()).enumerate() {
            acc = acc.add(&eq_k.evaluations[k].mul(&f.sub(i)));
        }
        acc
    };
    let tel_point: Vec<Goldilocks>;
    let tel_inc_claim: Goldilocks;
    {
        let eq_k = DenseMle::eq_extension(&r_t).tensor(&DenseMle::one(log_ts));
        let inc = m.inc_matrix();
        let mut vp = VirtualPolynomial::new(log_k + log_ts);
        let ei = vp.add_factor(eq_k).map_err(MemoryError::Virtual)?;
        let ii = vp.add_factor(inc).map_err(MemoryError::Virtual)?;
        vp.add_term(Goldilocks::ONE, vec![ei, ii])
            .map_err(MemoryError::Virtual)?;

        let out = sumcheck::prove(&vp, tel_claim, transcript).map_err(MemoryError::Sumcheck)?;
        tel_point = out.challenges.clone();
        tel_inc_claim = out.factor_claims[1];
        legs.push(LegProof {
            name: "T",
            sc: out.proof,
            claim: tel_claim,
        });
    }

    // ---- Md: matrix-eval of Inc at the telescoping point. ----
    let inc_at_tel = prove_matrix_eval(inst, "Md", m, ledger, &tel_point, legs, transcript)?;
    if inc_at_tel != tel_inc_claim {
        return Err(MemoryError::FinalCheck("tel inc binding"));
    }

    // The per-leg factor bindings above are the prover-side fail-closed
    // guarantees; the engine's round guards already pinned each sum.
    Ok(())
}

/// Verify one instance's memory argument (verifier side). Legs are
/// consumed from `iter` in protocol order.
pub fn verify_memory(
    inst: usize,
    m: &MemoryInstance,
    proof: &MemoryProof,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), MemoryError> {
    let log_k = m.log_k;
    let log_ts = m.log_ts;
    let mut iter = proof.legs.iter();

    // ---- B ----
    let log_rows = m.log_rows();
    absorb(transcript, inst, "B", log_k, log_ts)?;
    let r_bool = transcript
        .challenge_fields(b"mem-bool-r", log_rows + log_ts)
        .map_err(MemoryError::Transcript)?;
    {
        let leg = next_leg(&mut iter, "B")?;
        let verdict = leg
            .sc
            .verify(log_rows + log_ts, 3, Goldilocks::ZERO, transcript, None)
            .map_err(MemoryError::Sumcheck)?;

        if !leg.claim.is_zero() {
            return Err(MemoryError::ClaimMismatch);
        }
        let d_at = ledger.tensor_claim(Factor::DigitBits { inst }, &verdict.point)?;
        let eq_v = DenseMle::eq_eval(&r_bool, &verdict.point).map_err(MemoryError::Mle)?;
        let expect = eq_v.mul(&d_at.square().sub(&d_at));
        if expect != verdict.final_claim {
            return Err(MemoryError::FinalCheck("booleanity"));
        }
    }

    // ---- R ----
    absorb(transcript, inst, "R", log_k, log_ts)?;
    let r_prime = transcript
        .challenge_fields(b"mem-raf-r", log_ts)
        .map_err(MemoryError::Transcript)?;
    {
        let leg = next_leg(&mut iter, "R")?;
        let addr_claim = ledger.tensor_claim(addr_factor(inst), &r_prime)?;
        if leg.claim != addr_claim {
            return Err(MemoryError::ClaimMismatch);
        }
        let verdict = leg
            .sc
            .verify(log_rows + log_ts, 3, addr_claim, transcript, None)
            .map_err(MemoryError::Sumcheck)?;

        let (rho_rows, rho_j) = verdict.point.split_at(log_rows);
        let eq_v = DenseMle::eq_eval(&r_prime, rho_j).map_err(MemoryError::Mle)?;
        let w_v = m
            .raf_weights()
            .evaluate(rho_rows)
            .map_err(MemoryError::Mle)?;
        let d_at = ledger.tensor_claim(Factor::DigitBits { inst }, &verdict.point)?;
        if eq_v.mul(&w_v).mul(&d_at) != verdict.final_claim {
            return Err(MemoryError::FinalCheck("raf"));
        }
    }

    // ---- C ----
    absorb(transcript, inst, "C", log_k, log_ts)?;
    let r_c = transcript
        .challenge_fields(b"mem-read-r", log_ts)
        .map_err(MemoryError::Transcript)?;
    let (read_final, read_point);
    {
        let leg = next_leg(&mut iter, "C")?;
        let rv_claim = ledger.tensor_claim(rv_factor(inst), &r_c)?;
        if leg.claim != rv_claim {
            return Err(MemoryError::ClaimMismatch);
        }
        let verdict = leg
            .sc
            .verify(log_k + log_ts, 3, rv_claim, transcript, None)
            .map_err(MemoryError::Sumcheck)?;

        read_final = verdict.final_claim;
        read_point = verdict.point;
    }

    // ---- Ma ----
    let ra_at = verify_matrix_eval(
        inst,
        "Ma",
        m,
        ledger,
        &read_point,
        next_leg(&mut iter, "Ma")?,
        transcript,
    )?;

    if m.read_only() {
        let (rho_k, rho_j) = read_point.split_at(log_k);
        let eq_v = DenseMle::eq_eval(&r_c, rho_j).map_err(MemoryError::Mle)?;
        let table = m.table.as_ref().ok_or(MemoryError::Shape)?;
        let table_mle = DenseMle::new(table.clone()).map_err(MemoryError::Mle)?;
        let t_at = table_mle.evaluate(rho_k).map_err(MemoryError::Mle)?;
        if eq_v.mul(&ra_at).mul(&t_at) != read_final {
            return Err(MemoryError::FinalCheck("fetch read"));
        }
        if iter.next().is_some() {
            return Err(MemoryError::Shape);
        }
        return Ok(());
    }

    // ---- V0 + Mu0 ----
    let val_at_read = verify_val_eval(inst, "V0", m, ledger, &read_point, &mut iter, transcript)?;

    // ---- W ----
    absorb(transcript, inst, "W", log_k, log_ts)?;
    let r_w = transcript
        .challenge_fields(b"mem-write-r", log_k + log_ts)
        .map_err(MemoryError::Transcript)?;
    let (write_final, write_point);
    {
        let leg = next_leg(&mut iter, "W")?;
        let verdict = leg
            .sc
            .verify(log_k + log_ts, 3, Goldilocks::ZERO, transcript, None)
            .map_err(MemoryError::Sumcheck)?;
        write_final = verdict.final_claim;
        write_point = verdict.point;
    }

    // ---- Mb / Mc ----
    let wa_at = verify_matrix_eval(
        inst,
        "Mb",
        m,
        ledger,
        &write_point,
        next_leg(&mut iter, "Mb")?,
        transcript,
    )?;
    let inc_at_w = verify_matrix_eval(
        inst,
        "Mc",
        m,
        ledger,
        &write_point,
        next_leg(&mut iter, "Mc")?,
        transcript,
    )?;
    let wv_at_w = ledger.tensor_claim(wv_factor(inst), &write_point[log_k..])?;

    // ---- V1 + Mu1 ----
    let val_at_write = verify_val_eval(inst, "V1", m, ledger, &write_point, &mut iter, transcript)?;

    // ---- T ----
    absorb(transcript, inst, "T", log_k, log_ts)?;
    let r_t = transcript
        .challenge_fields(b"mem-tel-r", log_k)
        .map_err(MemoryError::Transcript)?;
    let (tel_final, tel_point);
    {
        let leg = next_leg(&mut iter, "T")?;
        let eq_k = DenseMle::eq_extension(&r_t);
        let mut acc = Goldilocks::ZERO;
        for (k, (f, i)) in m.final_state.iter().zip(m.init.iter()).enumerate() {
            acc = acc.add(&eq_k.evaluations[k].mul(&f.sub(i)));
        }
        if leg.claim != acc {
            return Err(MemoryError::ClaimMismatch);
        }
        let verdict = leg
            .sc
            .verify(log_k + log_ts, 2, acc, transcript, None)
            .map_err(MemoryError::Sumcheck)?;
        tel_final = verdict.final_claim;
        tel_point = verdict.point;
    }

    // ---- Md ----
    let inc_at_tel = verify_matrix_eval(
        inst,
        "Md",
        m,
        ledger,
        &tel_point,
        next_leg(&mut iter, "Md")?,
        transcript,
    )?;

    // ---- Terminals: the summand identities recomputed from the
    //      authenticated factor values (ra/wa/Inc from the matrix-evals,
    //      Val from the Val-evals, wv from the ledger). ----
    {
        let (_, rho_j) = read_point.split_at(log_k);
        let eq_v = DenseMle::eq_eval(&r_c, rho_j).map_err(MemoryError::Mle)?;

        if eq_v.mul(&ra_at).mul(&val_at_read) != read_final {
            return Err(MemoryError::FinalCheck("read-checking"));
        }
        let eq_v = DenseMle::eq_eval(&r_w, &write_point).map_err(MemoryError::Mle)?;
        let inner = inc_at_w.sub(&wa_at.mul(&wv_at_w.sub(&val_at_write)));
        if eq_v.mul(&inner) != write_final {
            return Err(MemoryError::FinalCheck("write-checking"));
        }
        let (rho_k_t, _) = tel_point.split_at(log_k);
        let eq_v = DenseMle::eq_eval(&r_t, rho_k_t).map_err(MemoryError::Mle)?;
        if eq_v.mul(&inc_at_tel) != tel_final {
            return Err(MemoryError::FinalCheck("telescoping"));
        }
    }
    if iter.next().is_some() {
        return Err(MemoryError::Shape);
    }
    Ok(())
}

fn next_leg<'b>(
    iter: &mut core::slice::Iter<'b, LegProof>,
    name: &str,
) -> Result<&'b LegProof, MemoryError> {
    let leg = iter.next().ok_or(MemoryError::Shape)?;
    if leg.name != name {
        return Err(MemoryError::Shape);
    }
    Ok(leg)
}

/// Which virtual matrix a matrix-eval leg addresses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MatrixKind {
    Ra,
    Wa,
    Inc,
}

pub(crate) fn matrix_kind(name: &str) -> MatrixKind {
    match name {
        "Ma" => MatrixKind::Ra,
        "Mb" => MatrixKind::Wa,
        _ => MatrixKind::Inc,
    }
}

// ---------------------------------------------------------------------------
// The leg-polynomial builders (shared between the per-instance protocol
// above and the Stage-4 leg batching in `legbatch.rs`).
// ---------------------------------------------------------------------------

/// B at a concrete `r_bool` (the eq factor is `eq(r_bool, ·)`): the
/// digit-booleanity virtual polynomial over the `(log_rows + log_ts)`
/// cube, `Σ eq(r_bool)·D·(D−1)`.
pub(crate) fn booleanity_vp_at(
    m: &MemoryInstance,
    r_bool: &[Goldilocks],
) -> Result<VirtualPolynomial, MemoryError> {
    let digits = m.digit_tensor();
    let log_rows = m.log_rows();
    let r_len = log_rows + m.log_ts;
    let d_minus_1: Vec<Goldilocks> = digits
        .evaluations
        .iter()
        .map(|v| v.sub(&Goldilocks::ONE))
        .collect();
    let dm = DenseMle {
        num_vars: r_len,
        evaluations: d_minus_1,
    };
    let eq = DenseMle::eq_extension(r_bool);
    let mut vp = VirtualPolynomial::new(r_len);
    let di = vp.add_factor(digits).map_err(MemoryError::Virtual)?;
    let dmi = vp.add_factor(dm).map_err(MemoryError::Virtual)?;
    let ei = vp.add_factor(eq).map_err(MemoryError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![di, dmi, ei])
        .map_err(MemoryError::Virtual)?;
    Ok(vp)
}

/// R: the raf virtual polynomial: `Σ eq(r′, j)·w(row)·D(row, j)`.
pub(crate) fn raf_vp(
    m: &MemoryInstance,
    r_prime: &[Goldilocks],
) -> Result<VirtualPolynomial, MemoryError> {
    let digits = m.digit_tensor();
    let log_rows = m.log_rows();
    let eq_j = DenseMle::one(log_rows).tensor(&DenseMle::eq_extension(r_prime));
    let w_ext = m.raf_weights().tensor(&DenseMle::one(m.log_ts));
    let mut vp = VirtualPolynomial::new(log_rows + m.log_ts);
    let ei = vp.add_factor(eq_j).map_err(MemoryError::Virtual)?;
    let wi = vp.add_factor(w_ext).map_err(MemoryError::Virtual)?;
    let di = vp.add_factor(digits).map_err(MemoryError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![ei, wi, di])
        .map_err(MemoryError::Virtual)?;
    Ok(vp)
}

/// C: the read-checking virtual polynomial: `Σ eq(r_c, j)·ra·Val`.
pub(crate) fn read_vp(
    m: &MemoryInstance,
    r_c: &[Goldilocks],
) -> Result<VirtualPolynomial, MemoryError> {
    let ra = m.one_hot(false);
    let val = m.val_matrix();
    let val_factor = match &m.table {
        Some(table) => DenseMle::new(table.clone())
            .map_err(MemoryError::Mle)?
            .tensor(&DenseMle::one(m.log_ts)),
        None => val,
    };
    let eq_j = DenseMle::one(m.log_k).tensor(&DenseMle::eq_extension(r_c));
    let mut vp = VirtualPolynomial::new(m.log_k + m.log_ts);
    let ei = vp.add_factor(eq_j).map_err(MemoryError::Virtual)?;
    let ri = vp.add_factor(ra).map_err(MemoryError::Virtual)?;
    let vi = vp.add_factor(val_factor).map_err(MemoryError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![ei, ri, vi])
        .map_err(MemoryError::Virtual)?;
    Ok(vp)
}

/// W: the write-checking virtual polynomial (zero form):
/// `Σ eq(r_w)·[Inc − wa·(wv − Val)]`.
pub(crate) fn write_vp(
    m: &MemoryInstance,
    r_w: &[Goldilocks],
) -> Result<VirtualPolynomial, MemoryError> {
    let eq = DenseMle::eq_extension(r_w);
    let wa = m.one_hot(true);
    let inc = m.inc_matrix();
    let wv_lift = DenseMle::one(m.log_k).tensor(&m.wv_col());
    let val = m.val_matrix();
    let mut vp = VirtualPolynomial::new(m.log_k + m.log_ts);
    let ei = vp.add_factor(eq).map_err(MemoryError::Virtual)?;
    let ii = vp.add_factor(inc).map_err(MemoryError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![ei, ii])
        .map_err(MemoryError::Virtual)?;
    let wi = vp.add_factor(wa).map_err(MemoryError::Virtual)?;
    let wvi = vp.add_factor(wv_lift).map_err(MemoryError::Virtual)?;
    let vali = vp.add_factor(val).map_err(MemoryError::Virtual)?;
    vp.add_term(Goldilocks::ONE.neg(), vec![ei, wi, wvi])
        .map_err(MemoryError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![ei, wi, vali])
        .map_err(MemoryError::Virtual)?;
    Ok(vp)
}

/// T: the telescoping virtual polynomial: `Σ eq(r_t, k)·Inc`.
pub(crate) fn telescoping_vp(
    m: &MemoryInstance,
    r_t: &[Goldilocks],
) -> Result<VirtualPolynomial, MemoryError> {
    let eq_k = DenseMle::eq_extension(r_t).tensor(&DenseMle::one(m.log_ts));
    let inc = m.inc_matrix();
    let mut vp = VirtualPolynomial::new(m.log_k + m.log_ts);
    let ei = vp.add_factor(eq_k).map_err(MemoryError::Virtual)?;
    let ii = vp.add_factor(inc).map_err(MemoryError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![ei, ii])
        .map_err(MemoryError::Virtual)?;
    Ok(vp)
}

/// The telescoping claim: `Σ_k eq(r_t, k)·(Final − Init)` — public.
pub(crate) fn tel_claim(m: &MemoryInstance, r_t: &[Goldilocks]) -> Result<Goldilocks, MemoryError> {
    let eq_k = DenseMle::eq_extension(r_t);
    let mut acc = Goldilocks::ZERO;
    for (k, (f, i)) in m.final_state.iter().zip(m.init.iter()).enumerate() {
        acc = acc.add(&eq_k.evaluations[k].mul(&f.sub(i)));
    }
    Ok(acc)
}

/// The matrix-eval virtual polynomial (Ma/Mb/Mc/Md/Mu0/Mu1 share this).
pub(crate) fn matrix_vp(
    m: &MemoryInstance,
    rho_k: &[Goldilocks],
    rho_j: &[Goldilocks],
    kind: MatrixKind,
) -> Result<VirtualPolynomial, MemoryError> {
    let inc_case = kind == MatrixKind::Inc;
    let write_activity = kind != MatrixKind::Ra;
    let log_ts = m.log_ts;
    let mut vp = VirtualPolynomial::new(log_ts);
    let eq = DenseMle::eq_extension(rho_j);
    let ei = vp.add_factor(eq).map_err(MemoryError::Virtual)?;
    let mut term = vec![ei];
    let activity = DenseMle {
        num_vars: log_ts,
        evaluations: (0..m.t_s())
            .map(|j| {
                let a = if write_activity {
                    m.wactive[j]
                } else {
                    m.ractive[j]
                };
                fe(a as u64)
            })
            .collect(),
    };
    let ai = vp.add_factor(activity).map_err(MemoryError::Virtual)?;
    term.push(ai);
    for b in 0..m.log_k {
        let rho_b = rho_k[b];
        let affine: Vec<Goldilocks> = (0..m.t_s())
            .map(|j| {
                let bit = fe((m.addr[j] >> (m.log_k - 1 - b)) & 1);
                bit.mul(&rho_b.double().sub(&Goldilocks::ONE))
                    .add(&Goldilocks::ONE.sub(&rho_b))
            })
            .collect();
        let fi = vp
            .add_factor(DenseMle {
                num_vars: log_ts,
                evaluations: affine,
            })
            .map_err(MemoryError::Virtual)?;
        term.push(fi);
    }
    if inc_case {
        let inc_part: Vec<Goldilocks> = m.inc_off.iter().map(|v| v.sub(&fe(INC_OFFSET))).collect();
        let ci = vp
            .add_factor(DenseMle {
                num_vars: log_ts,
                evaluations: inc_part,
            })
            .map_err(MemoryError::Virtual)?;
        term.push(ci);
    }
    vp.add_term(Goldilocks::ONE, term)
        .map_err(MemoryError::Virtual)?;
    Ok(vp)
}

/// The Val-evaluation virtual polynomial (V0/V1) + its claims.
/// Returns `(vp, val_at, sumcheck_claim)` with
/// `sumcheck_claim = val_at − init_at`.
pub(crate) fn val_vp(
    m: &MemoryInstance,
    point: &[Goldilocks],
) -> Result<(VirtualPolynomial, Goldilocks, Goldilocks), MemoryError> {
    let (r_a, r_c) = point.split_at(m.log_k);
    let init_mle = DenseMle::new(m.init.clone()).map_err(MemoryError::Mle)?;
    let init_at = init_mle.evaluate(r_a).map_err(MemoryError::Mle)?;
    let val = m.val_matrix();
    let val_at = val.evaluate(point).map_err(MemoryError::Mle)?;
    let u: Vec<Goldilocks> = (0..m.t_s())
        .map(|j| {
            let sel = eq_of_addr(r_a, m.addr[j], m.log_k);
            let inc = if m.wactive[j] != 0 {
                m.inc_off[j].sub(&fe(INC_OFFSET))
            } else {
                Goldilocks::ZERO
            };
            sel.mul(&inc)
        })
        .collect();
    let lt: Vec<Goldilocks> = (0..m.t_s())
        .map(|j| {
            let bits: Vec<Goldilocks> = (0..m.log_ts)
                .map(|b| fe((j >> (m.log_ts - 1 - b)) as u64 & 1))
                .collect();
            DenseMle::lt_extension(&bits, r_c).map_err(MemoryError::Mle)
        })
        .collect::<Result<_, _>>()?;
    let claim = val_at.sub(&init_at);
    let mut vp = VirtualPolynomial::new(m.log_ts);
    let ui = vp
        .add_factor(DenseMle {
            num_vars: m.log_ts,
            evaluations: u,
        })
        .map_err(MemoryError::Virtual)?;
    let li = vp
        .add_factor(DenseMle {
            num_vars: m.log_ts,
            evaluations: lt,
        })
        .map_err(MemoryError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![ui, li])
        .map_err(MemoryError::Virtual)?;
    Ok((vp, val_at, claim))
}

/// The (virtual) matrix's evaluation at `point` — the matrix-eval leg's
/// claimed sum (prover-side; the verifier never calls this).
pub(crate) fn matrix_claim(
    m: &MemoryInstance,
    point: &[Goldilocks],
    kind: MatrixKind,
) -> Result<Goldilocks, MemoryError> {
    let matrix = match kind {
        MatrixKind::Ra => m.one_hot(false),
        MatrixKind::Wa => m.one_hot(true),
        MatrixKind::Inc => m.inc_matrix(),
    };
    matrix.evaluate(point).map_err(MemoryError::Mle)
}

/// Prove a matrix-evaluation leg: the (virtual) matrix's evaluation at
/// `point`, expanded over the j'-cube from the committed columns.
fn prove_matrix_eval(
    inst: usize,
    name: &'static str,
    m: &MemoryInstance,
    ledger: &mut Ledger<'_>,
    point: &[Goldilocks],
    legs: &mut Vec<LegProof>,
    transcript: &mut Transcript,
) -> Result<Goldilocks, MemoryError> {
    absorb(transcript, inst, name, m.log_k, m.log_ts)?;
    let (rho_k, rho_j) = point.split_at(m.log_k);
    let kind = matrix_kind(name);
    let matrix = match kind {
        MatrixKind::Ra => m.one_hot(false),
        MatrixKind::Wa => m.one_hot(true),
        MatrixKind::Inc => m.inc_matrix(),
    };
    let claim = matrix.evaluate(point).map_err(MemoryError::Mle)?;
    let out = matrix_eval_sumcheck(m, rho_k, rho_j, claim, kind, transcript)?;
    bind_matrix_factors(
        inst,
        m,
        ledger,
        rho_k,
        rho_j,
        &out.challenges,
        &out.factor_claims,
        kind,
    )?;
    legs.push(LegProof {
        name,
        sc: out.proof,
        claim,
    });
    Ok(claim)
}

/// Verify a matrix-evaluation leg.
fn verify_matrix_eval(
    inst: usize,
    name: &'static str,
    m: &MemoryInstance,
    ledger: &mut Ledger<'_>,
    point: &[Goldilocks],
    leg: &LegProof,
    transcript: &mut Transcript,
) -> Result<Goldilocks, MemoryError> {
    absorb(transcript, inst, name, m.log_k, m.log_ts)?;
    let (rho_k, rho_j) = point.split_at(m.log_k);
    let kind = matrix_kind(name);
    let degree = 2 + m.log_k + (kind == MatrixKind::Inc) as usize;
    let verdict = leg
        .sc
        .verify(m.log_ts, degree, leg.claim, transcript, None)
        .map_err(MemoryError::Sumcheck)?;
    check_matrix_terminal(
        inst,
        m,
        ledger,
        rho_k,
        rho_j,
        &verdict.point,
        &verdict.final_claim,
        kind,
    )?;
    Ok(leg.claim)
}

/// The matrix-eval sumcheck body (prover).
fn matrix_eval_sumcheck(
    m: &MemoryInstance,
    rho_k: &[Goldilocks],
    rho_j: &[Goldilocks],
    claim: Goldilocks,
    kind: MatrixKind,
    transcript: &mut Transcript,
) -> Result<SumcheckOutput, MemoryError> {
    let inc_case = kind == MatrixKind::Inc;
    let write_activity = kind != MatrixKind::Ra;
    let log_ts = m.log_ts;
    let mut vp = VirtualPolynomial::new(log_ts);
    let eq = DenseMle::eq_extension(rho_j);
    let ei = vp.add_factor(eq).map_err(MemoryError::Virtual)?;
    let mut term = vec![ei];
    let activity = DenseMle {
        num_vars: log_ts,
        evaluations: (0..m.t_s())
            .map(|j| {
                let a = if write_activity {
                    m.wactive[j]
                } else {
                    m.ractive[j]
                };
                fe(a as u64)
            })
            .collect(),
    };
    let ai = vp.add_factor(activity).map_err(MemoryError::Virtual)?;
    term.push(ai);
    for b in 0..m.log_k {
        let rho_b = rho_k[b];
        let affine: Vec<Goldilocks> = (0..m.t_s())
            .map(|j| {
                let bit = fe((m.addr[j] >> (m.log_k - 1 - b)) & 1);
                // eq(rho_b, bit) = bit·(2ρ−1) + (1−ρ).
                bit.mul(&rho_b.double().sub(&Goldilocks::ONE))
                    .add(&Goldilocks::ONE.sub(&rho_b))
            })
            .collect();
        let fi = vp
            .add_factor(DenseMle {
                num_vars: log_ts,
                evaluations: affine,
            })
            .map_err(MemoryError::Virtual)?;
        term.push(fi);
    }
    if inc_case {
        let inc_part: Vec<Goldilocks> = m.inc_off.iter().map(|v| v.sub(&fe(INC_OFFSET))).collect();
        let ci = vp
            .add_factor(DenseMle {
                num_vars: log_ts,
                evaluations: inc_part,
            })
            .map_err(MemoryError::Virtual)?;
        term.push(ci);
    }
    vp.add_term(Goldilocks::ONE, term)
        .map_err(MemoryError::Virtual)?;
    sumcheck::prove(&vp, claim, transcript).map_err(MemoryError::Sumcheck)
}

/// Bind the matrix-eval sumcheck's factor claims to base claims (prover).
/// `terminal` is the sumcheck's own terminal point (where the factor
/// claims live).
#[allow(clippy::too_many_arguments)]
fn bind_matrix_factors(
    inst: usize,
    m: &MemoryInstance,
    ledger: &mut Ledger<'_>,
    rho_k: &[Goldilocks],
    rho_j: &[Goldilocks],
    terminal: &[Goldilocks],
    factor_claims: &[Goldilocks],
    kind: MatrixKind,
) -> Result<(), MemoryError> {
    // Factor order: [eq, activity, digit-affine_b..., (inc-part)?].
    let eq_claim = factor_claims[0];
    let expect_eq = DenseMle::eq_eval(rho_j, terminal).map_err(MemoryError::Mle)?;
    if eq_claim != expect_eq {
        return Err(MemoryError::ClaimMismatch);
    }
    let act_claim = factor_claims[1];
    let derived = ledger.tensor_claim(activity_factor(inst, kind != MatrixKind::Ra), terminal)?;
    if derived != act_claim {
        return Err(MemoryError::ClaimMismatch);
    }
    for b in 0..m.log_k {
        let claimed = factor_claims[2 + b];
        let mut point = crate::ledger::idx_point(m.log_rows(), b);
        point.extend_from_slice(terminal);
        let row = ledger.tensor_claim(Factor::DigitBits { inst }, &point)?;
        let affine = digit_affine(row, rho_k[b]);
        if affine != claimed {
            return Err(MemoryError::ClaimMismatch);
        }
    }
    if kind == MatrixKind::Inc {
        let inc_claim = *factor_claims.last().ok_or(MemoryError::Shape)?;
        let derived = ledger.tensor_claim(inc_factor(inst), terminal)?;
        if derived.sub(&fe(INC_OFFSET)) != inc_claim {
            return Err(MemoryError::ClaimMismatch);
        }
    }
    Ok(())
}

/// Check the matrix-eval terminal identity (verifier).
#[allow(clippy::too_many_arguments)]
fn check_matrix_terminal(
    inst: usize,
    m: &MemoryInstance,
    ledger: &mut Ledger<'_>,
    rho_k: &[Goldilocks],
    rho_j: &[Goldilocks],
    point: &[Goldilocks],
    final_claim: &Goldilocks,
    kind: MatrixKind,
) -> Result<(), MemoryError> {
    // The terminal identity: eq(ρ_j, point)·activity(point)·
    // Π_b digit-affine_b(point)·(inc-part(point)) == final_claim, with the
    // factor values derived from LEDGER claims at `point`.
    let mut expect = DenseMle::eq_eval(rho_j, point).map_err(MemoryError::Mle)?;
    let act = ledger.tensor_claim(activity_factor(inst, kind != MatrixKind::Ra), point)?;
    expect = expect.mul(&act);
    for b in 0..m.log_k {
        let mut pt = crate::ledger::idx_point(m.log_rows(), b);
        pt.extend_from_slice(point);
        let row = ledger.tensor_claim(Factor::DigitBits { inst }, &pt)?;
        expect = expect.mul(&digit_affine(row, rho_k[b]));
    }
    if kind == MatrixKind::Inc {
        let inc = ledger.tensor_claim(inc_factor(inst), point)?;
        expect = expect.mul(&inc.sub(&fe(INC_OFFSET)));
    }
    if expect != *final_claim {
        return Err(MemoryError::FinalCheck("matrix-eval"));
    }
    Ok(())
}

/// The affine digit factor: `eq(ρ, row) = row·(2ρ − 1) + (1 − ρ)`.
pub(crate) fn digit_affine(row: Goldilocks, rho_b: Goldilocks) -> Goldilocks {
    row.mul(&rho_b.double().sub(&Goldilocks::ONE))
        .add(&Goldilocks::ONE.sub(&rho_b))
}

/// Prove a Val-evaluation leg (Fig 9 Eq 11) at `point` (k-part, j-part).
fn prove_val_eval(
    inst: usize,
    name: &'static str,
    m: &MemoryInstance,
    ledger: &mut Ledger<'_>,
    point: &[Goldilocks],
    legs: &mut Vec<LegProof>,
    transcript: &mut Transcript,
) -> Result<Goldilocks, MemoryError> {
    absorb(transcript, inst, name, m.log_k, m.log_ts)?;
    let (r_a, r_c) = point.split_at(m.log_k);
    let init_mle = DenseMle::new(m.init.clone()).map_err(MemoryError::Mle)?;
    let init_at = init_mle.evaluate(r_a).map_err(MemoryError::Mle)?;
    let val = m.val_matrix();
    let val_at = val.evaluate(point).map_err(MemoryError::Mle)?;
    let u: Vec<Goldilocks> = (0..m.t_s())
        .map(|j| {
            let sel = eq_of_addr(r_a, m.addr[j], m.log_k);
            let inc = if m.wactive[j] != 0 {
                m.inc_off[j].sub(&fe(INC_OFFSET))
            } else {
                Goldilocks::ZERO
            };
            sel.mul(&inc)
        })
        .collect();
    let lt: Vec<Goldilocks> = (0..m.t_s())
        .map(|j| {
            let bits: Vec<Goldilocks> = (0..m.log_ts)
                .map(|b| fe((j >> (m.log_ts - 1 - b)) as u64 & 1))
                .collect();
            DenseMle::lt_extension(&bits, r_c).map_err(MemoryError::Mle)
        })
        .collect::<Result<_, _>>()?;
    let claim = val_at.sub(&init_at);
    let mut vp = VirtualPolynomial::new(m.log_ts);
    let ui = vp
        .add_factor(DenseMle {
            num_vars: m.log_ts,
            evaluations: u,
        })
        .map_err(MemoryError::Virtual)?;
    let li = vp
        .add_factor(DenseMle {
            num_vars: m.log_ts,
            evaluations: lt,
        })
        .map_err(MemoryError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![ui, li])
        .map_err(MemoryError::Virtual)?;
    let out = sumcheck::prove(&vp, claim, transcript).map_err(MemoryError::Sumcheck)?;
    legs.push(LegProof {
        name,
        sc: out.proof.clone(),
        claim: val_at,
    });
    // The u factor's claim = a matrix-eval of Inc at (r_a, V's point).
    let mut full = r_a.to_vec();
    full.extend_from_slice(&out.challenges);
    let u_claim = out.factor_claims[0];
    let mu_name: &'static str = if name == "V0" { "Mu0" } else { "Mu1" };
    let inc_at = prove_matrix_eval(inst, mu_name, m, ledger, &full, legs, transcript)?;
    if inc_at != u_claim {
        return Err(MemoryError::FinalCheck("val-eval u binding"));
    }
    let lt_at = DenseMle::lt_extension(&out.challenges, r_c).map_err(MemoryError::Mle)?;
    if out.factor_claims[1] != lt_at {
        return Err(MemoryError::FinalCheck("val-eval lt binding"));
    }
    Ok(val_at)
}

/// Verify a Val-evaluation leg.
fn verify_val_eval(
    inst: usize,
    name: &'static str,
    m: &MemoryInstance,
    ledger: &mut Ledger<'_>,
    point: &[Goldilocks],
    iter: &mut core::slice::Iter<'_, LegProof>,
    transcript: &mut Transcript,
) -> Result<Goldilocks, MemoryError> {
    absorb(transcript, inst, name, m.log_k, m.log_ts)?;
    let (r_a, r_c) = point.split_at(m.log_k);
    let init_mle = DenseMle::new(m.init.clone()).map_err(MemoryError::Mle)?;
    let init_at = init_mle.evaluate(r_a).map_err(MemoryError::Mle)?;
    let leg = next_leg(iter, name)?;
    // The leg's carried claim IS the Val evaluation; the sumcheck's claim
    // is (Val − init) at the point.
    let val_at = leg.claim;
    let verdict = leg
        .sc
        .verify(m.log_ts, 3, leg.claim.sub(&init_at), transcript, None)
        .map_err(MemoryError::Sumcheck)?;
    let mu_name: &'static str = if name == "V0" { "Mu0" } else { "Mu1" };
    let mut full = r_a.to_vec();
    full.extend_from_slice(&verdict.point);
    let mu_leg = next_leg(iter, mu_name)?;
    let u_claim = verify_matrix_eval(inst, mu_name, m, ledger, &full, mu_leg, transcript)?;
    let lt_at = DenseMle::lt_extension(&verdict.point, r_c).map_err(MemoryError::Mle)?;
    if u_claim.mul(&lt_at) != verdict.final_claim {
        return Err(MemoryError::FinalCheck("val-eval"));
    }
    Ok(val_at)
}

/// eq(r, bin(addr)) evaluated pointwise for the u-column.
fn eq_of_addr(r: &[Goldilocks], addr: u64, log_k: usize) -> Goldilocks {
    let mut acc = Goldilocks::ONE;
    for (b, rho) in r.iter().enumerate() {
        let bit = fe((addr >> (log_k - 1 - b)) & 1);
        let same = rho
            .mul(&bit)
            .add(&Goldilocks::ONE.sub(rho).mul(&Goldilocks::ONE.sub(&bit)));
        acc = acc.mul(&same);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny read-write memory: K=4 addresses, T_s=4 slots:
    /// slots: read a=1, write a=2 (5→), read a=2, read a=3.
    fn tiny_instance() -> MemoryInstance {
        let init = vec![fe(10), fe(20), fe(30), fe(40)];
        let final_state = vec![fe(10), fe(20), fe(5), fe(40)];
        let t_s = 4;
        let mut addr = vec![0u64; t_s];
        let mut ractive = vec![0u8; t_s];
        let mut wactive = vec![0u8; t_s];
        let mut rv = vec![Goldilocks::ZERO; t_s];
        let mut wv = vec![Goldilocks::ZERO; t_s];
        let mut inc_off = vec![fe(INC_OFFSET); t_s];
        let accesses: [(u64, bool, u64); 4] =
            [(1, false, 20), (2, true, 5), (2, false, 5), (3, false, 40)];
        let mut current = init.clone();
        for (j, (a, w, v)) in accesses.iter().enumerate() {
            addr[j] = *a;
            if *w {
                wactive[j] = 1;
                wv[j] = fe(*v);
                inc_off[j] = fe(*v).sub(&current[*a as usize]).add(&fe(INC_OFFSET));
                current[*a as usize] = fe(*v);
            } else {
                ractive[j] = 1;
                rv[j] = current[*a as usize];
            }
        }
        MemoryInstance {
            log_k: 2,
            log_ts: 2,
            addr,
            ractive,
            wactive,
            rv,
            wv,
            inc_off,
            init,
            final_state,
            table: None,
        }
    }

    #[test]
    fn honest_memory_proves_and_verifies() {
        let m = tiny_instance();
        // Prover.
        let tensor = m.digit_tensor();
        let rv_col = m.rv_col();
        let wv_col = m.wv_col();
        let inc_col = m.inc_col();
        let addr_col = m.addr_col();
        let wact_col = m.activity_col(true);
        let ract_col = m.activity_col(false);
        let table = vec![
            (Factor::DigitBits { inst: 0 }, &tensor),
            (rv_factor(0), &rv_col),
            (wv_factor(0), &wv_col),
            (inc_factor(0), &inc_col),
            (addr_factor(0), &addr_col),
            (activity_factor(0, true), &wact_col),
            (activity_factor(0, false), &ract_col),
        ];
        let mut ledger = Ledger::prover(table);
        let mut transcript = Transcript::new_default(b"mem-test");
        let mut legs = Vec::new();
        assert!(prove_memory(0, &m, &mut ledger, &mut legs, &mut transcript).is_ok());
        let proof = MemoryProof { legs };
        // Verifier.
        let claims: Vec<crate::ledger::BaseClaim> = ledger.claims().to_vec();
        let mut vledger = Ledger::verifier(claims);
        let mut vtranscript = Transcript::new_default(b"mem-test");
        assert!(verify_memory(0, &m, &proof, &mut vledger, &mut vtranscript).is_ok());
        // Tampered leg rejected.
        let mut bad = proof.clone();
        if let Some(leg) = bad.legs.first_mut() {
            if let Some(round) = leg.sc.rounds.first_mut() {
                if let Some(v) = round.first_mut() {
                    *v = v.add(&fe(1));
                }
            }
        }
        let mut vledger2 = Ledger::verifier(ledger_claims_against(&m));
        let mut vt2 = Transcript::new_default(b"mem-test");
        assert!(verify_memory(0, &m, &bad, &mut vledger2, &mut vt2).is_err());
    }

    fn ledger_claims_against(m: &MemoryInstance) -> Vec<crate::ledger::BaseClaim> {
        let tensor = m.digit_tensor();
        let rv_col = m.rv_col();
        let wv_col = m.wv_col();
        let inc_col = m.inc_col();
        let addr_col = m.addr_col();
        let wact_col = m.activity_col(true);
        let ract_col = m.activity_col(false);
        let table = vec![
            (Factor::DigitBits { inst: 0 }, &tensor),
            (rv_factor(0), &rv_col),
            (wv_factor(0), &wv_col),
            (inc_factor(0), &inc_col),
            (addr_factor(0), &addr_col),
            (activity_factor(0, true), &wact_col),
            (activity_factor(0, false), &ract_col),
        ];
        let mut ledger = Ledger::prover(table);
        let mut t = Transcript::new_default(b"mem-test");
        let mut legs = Vec::new();
        let _ = prove_memory(0, m, &mut ledger, &mut legs, &mut t);
        ledger.claims().to_vec()
    }

    #[test]
    fn stale_read_witness_rejected_by_prover() {
        // A stale read at slot 2 (claims 30 after the write set it to 5):
        // the honest build sets rv[2] = 5; corrupt it to 30 — the
        // read-checking terminal must fail closed.
        let mut m = tiny_instance();
        m.rv[2] = fe(30);
        let tensor = m.digit_tensor();
        let rv_col = m.rv_col();
        let wv_col = m.wv_col();
        let inc_col = m.inc_col();
        let addr_col = m.addr_col();
        let act_col = m.activity_col(true);
        let table = vec![
            (Factor::DigitBits { inst: 0 }, &tensor),
            (rv_factor(0), &rv_col),
            (wv_factor(0), &wv_col),
            (inc_factor(0), &inc_col),
            (addr_factor(0), &addr_col),
            (activity_factor(0, true), &act_col),
        ];
        let mut ledger = Ledger::prover(table);
        let mut t = Transcript::new_default(b"mem-test");
        let mut legs = Vec::new();
        assert!(prove_memory(0, &m, &mut ledger, &mut legs, &mut t).is_err());
    }

    #[test]
    fn read_only_fetch_proves_and_verifies() {
        // Fetch-like: K=2 words, table = [7, 9], reads of word 1, 0, 1, 1.
        let m = MemoryInstance {
            log_k: 1,
            log_ts: 2,
            addr: vec![1, 0, 1, 1],
            ractive: vec![1, 1, 1, 1],
            wactive: vec![0, 0, 0, 0],
            rv: vec![fe(9), fe(7), fe(9), fe(9)],
            wv: vec![Goldilocks::ZERO; 4],
            inc_off: vec![fe(INC_OFFSET); 4],
            init: vec![fe(7), fe(9)],
            final_state: vec![fe(7), fe(9)],
            table: Some(vec![fe(7), fe(9)]),
        };
        let tensor = m.digit_tensor();
        let rv_col = m.rv_col();
        let addr_col = m.addr_col();
        let wact_col = m.activity_col(true);
        let ract_col = m.activity_col(false);
        let table = vec![
            (Factor::DigitBits { inst: 1 }, &tensor),
            (rv_factor(1), &rv_col),
            (addr_factor(1), &addr_col),
            (activity_factor(1, true), &wact_col),
            (activity_factor(1, false), &ract_col),
        ];
        let mut ledger = Ledger::prover(table);
        let mut t = Transcript::new_default(b"fetch-test");
        let mut legs = Vec::new();
        assert!(prove_memory(1, &m, &mut ledger, &mut legs, &mut t).is_ok());
        let proof = MemoryProof { legs };
        let claims: Vec<crate::ledger::BaseClaim> = ledger.claims().to_vec();
        let mut vledger = Ledger::verifier(claims);
        let mut vt = Transcript::new_default(b"fetch-test");
        assert!(verify_memory(1, &m, &proof, &mut vledger, &mut vt).is_ok());
    }
}
