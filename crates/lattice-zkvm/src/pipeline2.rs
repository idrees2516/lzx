//! The v2 pipeline: the real Twist & Shout memory arguments in the live
//! path, the claim table batched through grouped openings, and a verifier
//! that NEVER re-executes the program.
//!
//! Soundness scope (honest): the v2 proof binds the statement
//! "there is a read/write access stream over the K-word RAM window,
//! starting from the public initial image (program + input) and ending
//! at the public final state, in which every read observes the value of
//! the most recent write" — the Twist & Shout memory argument, with
//! one-hot well-formedness checks and read-only table (Shout) bindings
//! for the fetch and input streams — AND the instruction-semantics AIR
//! (Stage 4.5: `semantics.rs` — the full constraint families over the
//! trace's witness: decode (with the coverage partition identities),
//! booleanity, selectors, flags, arithmetic, shifts, MUL, DIV,
//! comparisons, control, routing, and termination, every auxiliary
//! column range-linked to boolean bits). The verifier never re-executes.

use crate::pipeline::*;
use crate::semantics::{prove_instruction_semantics, SemanticsProof};
use lattice_akita::pcs::{AkitaPcs, GroupedOpening};
use lattice_akita::salsa_binding::SalsaBoundResponse;
use lattice_akita::salsa_response::SalsaGroupedResponse;
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_memory::onehot_check::OneHotSide as OHSide;
use lattice_memory::sparse_engine::{
    build_twist_ports, build_twist_ports_virtual, prove_onehot_sparse, prove_shout_sparse,
    prove_twist_ports_sparse, verify_twist_ports_checked, VirtualValSpec,
};
use lattice_memory::twist::TwistProof;
use lattice_memory::{FactorId, FactorResolver, OneHotProof, PiopError, ShoutProof};
use lattice_vm::MachineState;
use std::cell::RefCell;

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

/// A (column, point, value) claim authenticated by a grouped opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnClaim {
    pub col: usize,
    /// FactorId discriminant (disambiguates sentinel claims at one point).
    pub factor: u64,
    pub point: Vec<Goldilocks>,
    pub value: Goldilocks,
}

/// Recording resolver (prover side).
struct RecRes<'a> {
    cols: &'a [Vec<Goldilocks>],
    claims: RefCell<Vec<ColumnClaim>>,
    log_vars: usize,
    map: fn(FactorId) -> Option<usize>,
}

impl<'a> RecRes<'a> {
    fn new(
        cols: &'a [Vec<Goldilocks>],
        log_vars: usize,
        map: fn(FactorId) -> Option<usize>,
    ) -> Self {
        Self {
            cols,
            claims: RefCell::new(Vec::new()),
            log_vars,
            map,
        }
    }
}

impl<'a> FactorResolver for RecRes<'a> {
    fn eval(&self, factor: FactorId, point: &[Goldilocks]) -> Result<Goldilocks, PiopError> {
        let col = (self.map)(factor).ok_or(PiopError::MissingFactor { factor })?;
        let log_vars = self.log_vars;
        if point.len() != log_vars {
            return Err(PiopError::Shape {
                expected: log_vars,
                got: point.len(),
            });
        }
        let v = DenseMle {
            num_vars: log_vars,
            evaluations: self.cols[col].clone(),
        }
        .evaluate(point)?;
        self.claims.borrow_mut().push(ColumnClaim {
            col,
            factor: factor.discriminant(),
            point: point.to_vec(),
            value: v,
        });
        Ok(v)
    }
}

/// Claim-table resolver (verifier side).
struct TableRes<'a> {
    claims: &'a [ColumnClaim],
    map: fn(FactorId) -> Option<usize>,
}

impl<'a> FactorResolver for TableRes<'a> {
    fn eval(&self, factor: FactorId, point: &[Goldilocks]) -> Result<Goldilocks, PiopError> {
        if let Some(col) = (self.map)(factor) {
            for c in self.claims {
                if c.col == col && c.point.as_slice() == point {
                    return Ok(c.value);
                }
            }
        }
        // Sentinel claims (dims / Inc / Val — the P1 commitment layer):
        // resolve by (factor discriminant, point).
        for c in self.claims {
            if c.col == SENTINEL_COL
                && c.factor == factor.discriminant()
                && c.point.as_slice() == point
            {
                return Ok(c.value);
            }
        }
        Err(PiopError::MissingFactor { factor })
    }
}

/// The v2 proof: all leg proofs + the claim table + grouped openings.
#[derive(Clone, Debug)]
pub struct ProofV2 {
    pub log_t: usize,
    pub log_k_ram: usize,
    pub log_k_fetch: usize,
    pub num_fetch: usize,
    pub num_input: usize,
    pub public_state: PublicStateV2,
    /// Committed columns: [fetch_rv, fetch_ra, ram_rv, ram_wv, ram_ra, ram_wa,
    /// reg_rv_a, reg_rv_b, reg_wv, reg_ra_a, reg_ra_b, reg_wa].
    pub commitments: Vec<Vec<u8>>,
    pub fetch: ShoutProof,
    pub input_shout: ShoutProof,
    pub onehot_fetch: OneHotProof,
    pub onehot_ram_r: OneHotProof,
    pub onehot_ram_w: OneHotProof,
    pub onehot_reg_a: OneHotProof,
    pub onehot_reg_b: OneHotProof,
    pub onehot_reg_w: OneHotProof,
    pub twist_ram: TwistProof,
    pub twist_reg_a: TwistProof,
    pub twist_reg_b: TwistProof,
    /// The instruction-semantics layer (Stage 4.5): every constraint
    /// family over the trace's witness, bound through the bits/values
    /// bundles.
    pub semantics: SemanticsProof,
    pub claims: Vec<ColumnClaim>,
    /// The Stage-5 grouped openings — the response-layer mode:
    /// `Salsa` (D4 open: the byte-packed chain, polylog, the
    /// byte-witness-to-commitment binding the documented residual) or
    /// `Bound` (D4 closed: the compact-fold composition — every
    /// column's byte-witness bound to its commitment through the
    /// width-collapse chain, estimator-gated per stage).
    pub openings: Stage5Openings,
}

/// The Stage-5 response-layer mode.
#[derive(Clone, Debug)]
pub enum Stage5Openings {
    /// The open D4 response (the original swap): three O(log N)
    /// sumchecks, zero disclosure, the Ajtai binding the documented
    /// outer-layer gap.
    Salsa(Vec<SalsaGroupedResponse>),
    /// The binding-closed D4 response: the carrier + D1 + the
    /// width-collapse chain per column (the authenticated opening at
    /// the challenge — the compact-fold composition at this layer).
    Bound(Vec<SalsaBoundResponse>),
}

/// The Stage-5 mode selector for the prover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage5Mode {
    Salsa,
    Bound,
}

/// Column slots in the commitment list.
mod slot {
    pub const FETCH_RV: usize = 0;
    pub const FETCH_RA: usize = 1;
    pub const RAM_RV: usize = 2;
    pub const RAM_WV: usize = 3;
    pub const RAM_RA: usize = 4;
    pub const RAM_WA: usize = 5;
    pub const REG_RV_A: usize = 6;
    pub const REG_RV_B: usize = 7;
    pub const REG_WV: usize = 8;
    pub const REG_RA_A: usize = 9;
    pub const REG_RA_B: usize = 10;
    pub const REG_WA: usize = 11;
    pub const INPUT_RV: usize = 12;
    pub const INPUT_RA: usize = 13;
}

#[allow(clippy::too_many_lines)]
pub fn prove_v2(
    pcs: &AkitaPcs,
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
) -> Result<(PublicStateV2, ProofV2), PipelineError> {
    prove_v2_with_stage5(pcs, program, public_input, max_steps, Stage5Mode::Salsa)
}

/// The v2 prover with the Stage-5 response-layer mode selected (the
/// `Bound` mode = the compact-fold composition: every column's
/// byte-witness bound to its commitment through the width-collapse
/// chain — the D4 closure at the pipeline layer).
#[allow(clippy::too_many_lines)]
pub fn prove_v2_with_stage5(
    pcs: &AkitaPcs,
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
    stage5: Stage5Mode,
) -> Result<(PublicStateV2, ProofV2), PipelineError> {
    // ---- 1. Execute. ----
    let mut state = MachineState::new();
    state.load_program(0x1000, program);
    state.load_program(0x3000, public_input);
    state.regs[10] = public_input.len() as u64;
    state.pc = 0x1000;
    let rows = lattice_vm::run(&mut state, max_steps).map_err(PipelineError::Execution)?;
    if rows.is_empty() {
        return Err(PipelineError::BadShape("empty trace".into()));
    }
    let trace = build_trace(
        &state,
        &rows,
        &[(0x1000, program), (0x3000, public_input)],
    )?;
    let log_t = trace.log_t;
    // ---- 2. The access streams. ----
    // Fetch stream: every step's ((pc - 0x1000)/4 -> instruction word).
    let mut fetch_ra: Vec<u64> = trace.cols[Col::Pcq as usize]
        .iter()
        .map(|x| x.to_canonical_u64().saturating_sub(0x1000 / 4))
        .collect();
    let fetch_rv: Vec<Goldilocks> = trace.cols[Col::Iw as usize].clone();
    let log_k_fetch = {
        let max_q = fetch_ra.iter().copied().max().unwrap_or(1);
        (max_q + 1).next_power_of_two().max(2).trailing_zeros() as usize
    };
    fetch_ra.resize(1 << log_t, 0);
    // RAM ports.
    let ram_rv = trace.cols[Col::Mrv as usize].clone();
    let ram_wv = trace.cols[Col::Mwv as usize].clone();
    let mut ram_ra = trace.ram_read_addr.clone();
    let mut ram_wa = trace.ram_write_addr.clone();
    ram_ra.resize(1 << log_t, 0);
    ram_wa.resize(1 << log_t, 0);
    // Register ports (mirrors of rs1a/rs2a + write port).
    let reg_rv_a = trace.cols[Col::Rs1v as usize].clone();
    let reg_rv_b = trace.cols[Col::Rs2v as usize].clone();
    let reg_wv = trace.cols[Col::Wvr as usize].clone();
    let mut reg_ra_a = trace.reg_read_addr_a.clone();
    let mut reg_ra_b = trace.reg_read_addr_b.clone();
    let mut reg_wa = trace.reg_write_addr.clone();
    reg_ra_a.resize(1 << log_t, 0);
    reg_ra_b.resize(1 << log_t, 0);
    reg_wa.resize(1 << log_t, 0);
    // Input stream: the input words read at boot (a fixed small stream:
    // the guest's first reads of the input region; here: the full input
    // window read once, word by word, as its own read-only stream).
    let num_input_words = public_input.len().div_ceil(8).max(1);
    let log_t_in = num_input_words.next_power_of_two().max(2).trailing_zeros() as usize;
    let input_ra: Vec<u64> = (0..(1 << log_t_in))
        .map(|i| i.min(num_input_words - 1) as u64)
        .collect();
    let input_rv: Vec<Goldilocks> = (0..(1 << log_t_in))
        .map(|i| {
            let idx = i.min(num_input_words - 1);
            let mut w = 0u64;
            for b in 0..8 {
                let off = idx * 8 + b;
                if off < public_input.len() {
                    w |= (public_input[off] as u64) << (b * 8);
                }
            }
            fe(w)
        })
        .collect();
    // RAM window bound.
    let mut max_word = 0x3000 / 8 + num_input_words as u64;
    max_word = max_word.max(0x1000 / 8 + program.len().div_ceil(8) as u64);
    for r in &rows {
        if let Some((a, _, _)) = &r.mem_access {
            max_word = max_word.max(a / 8);
        }
    }
    let log_k_ram = (max_word + 1).next_power_of_two().max(2).trailing_zeros() as usize;
    let k_ram = 1usize << log_k_ram;
    // Public init/final RAM.
    let mut init_ram = vec![0u64; k_ram];
    for (i, chunk) in program.chunks(8).enumerate() {
        let mut w = 0u64;
        for (b, byte) in chunk.iter().enumerate() {
            w |= (*byte as u64) << (b * 8);
        }
        let idx = 0x1000 / 8 + i;
        if idx < k_ram {
            init_ram[idx] = w;
        }
    }
    for (i, chunk) in public_input.chunks(8).enumerate() {
        let mut w = 0u64;
        for (b, byte) in chunk.iter().enumerate() {
            w |= (*byte as u64) << (b * 8);
        }
        let idx = 0x3000 / 8 + i;
        if idx < k_ram {
            init_ram[idx] = w;
        }
    }
    let mut final_ram = init_ram.clone();
    for (addr, v) in state.memory.snapshot_pairs() {
        let idx = (addr / 8) as usize;
        if idx < k_ram {
            final_ram[idx] = v;
        }
    }
    let public_state = PublicStateV2 {
        final_regs: state.regs,
        final_ram: final_ram.clone(),
        num_steps: rows.len() as u64,
    };
    // ---- 3. Commit the columns. ----
    let t_cols: Vec<Vec<Goldilocks>> = vec![
        fetch_rv.clone(),
        fetch_ra.iter().map(|&v| fe(v)).collect(),
        ram_rv.clone(),
        ram_wv.clone(),
        ram_ra.iter().map(|&v| fe(v)).collect(),
        ram_wa.iter().map(|&v| fe(v)).collect(),
        reg_rv_a.clone(),
        reg_rv_b.clone(),
        reg_wv.clone(),
        reg_ra_a.iter().map(|&v| fe(v)).collect(),
        reg_ra_b.iter().map(|&v| fe(v)).collect(),
        reg_wa.iter().map(|&v| fe(v)).collect(),
        input_rv.clone(),
        input_ra.iter().map(|&v| fe(v)).collect(),
    ];
    let mut commitments = Vec::with_capacity(t_cols.len());
    for (ci, col) in t_cols.iter().enumerate() {
        let lv = col_log_vars(ci, log_t, log_t_in);
        let mle = DenseMle {
            num_vars: lv,
            evaluations: pad_to(col, lv),
        };
        // The D4 regime: the columns commit their BYTE-PACKED witnesses
        // (one byte per coefficient — the SALSAA chain's Lemma-4 gate).
        commitments.push(pcs.commit_bytes(&mle)?.commitment.to_bytes());
    }
    // ---- 4. Public tables. ----
    let mut fetch_table: Vec<Goldilocks> = (0..program.len() / 4)
        .map(|i| {
            let mut w = 0u32;
            for b in 0..4 {
                w |= (program[i * 4 + b] as u32) << (b * 8);
            }
            fe(w as u64)
        })
        .collect();
    fetch_table.resize(1 << log_k_fetch, Goldilocks::ZERO);
    // Input table: the public input words.
    let log_k_in = num_input_words.next_power_of_two().max(2).trailing_zeros() as usize;
    let mut input_table: Vec<Goldilocks> = (0..num_input_words)
        .map(|i| {
            let mut w = 0u64;
            for b in 0..8 {
                let off = i * 8 + b;
                if off < public_input.len() {
                    w |= (public_input[off] as u64) << (b * 8);
                }
            }
            fe(w)
        })
        .collect();
    input_table.resize(1 << log_k_in, Goldilocks::ZERO);
    // ---- 5. Twist witnesses. ----
    // The RAM instance takes the VIRTUAL-VAL route when the address
    // traffic is sparse relative to the container (the dispatch
    // heuristic from VirtualValSpec::pairwise_cost): the O(K·T)
    // materialized Val matrix — the container-scale cap — is never
    // allocated. The register instances (K = 32, every address hot)
    // stay on the materialized route: K·T is small there and the
    // pairwise cost would exceed it.
    let ram_witness = {
        let ram_init: Vec<Goldilocks> = init_ram.iter().map(|&v| fe(v)).collect();
        let pairwise = VirtualValSpec::pairwise_cost(&ram_ra, &ram_wa);
        let k_ram = 1u64 << log_k_ram;
        let t_pow = 1u64 << log_t;
        if pairwise * 8 < k_ram * t_pow {
            build_twist_ports_virtual(&ram_ra, &ram_wa, &ram_wv, &ram_init, log_k_ram, log_t)?
        } else {
            build_twist_ports(&ram_ra, &ram_wa, &ram_wv, &ram_init, log_k_ram, log_t)?
        }
    };
    let reg_init: Vec<Goldilocks> = {
        let mut v = vec![Goldilocks::ZERO; 32];
        v[10] = fe(public_input.len() as u64);
        v
    };
    let rega_witness = build_twist_ports(&reg_ra_a, &reg_wa, &reg_wv, &reg_init, 5, log_t)?;
    let regb_witness = build_twist_ports(&reg_ra_b, &reg_wa, &reg_wv, &reg_init, 5, log_t)?;
    // ---- 6. Transcript. ----
    let mut transcript = Transcript::new_default(b"lzx-zkvm-v2");
    let prog_digest = crate::program_digest(program);
    let input_digest = crate::public_input_digest(public_input);
    transcript.append_bytes(b"program", &prog_digest)?;
    transcript.append_bytes(b"public-input", &input_digest)?;
    transcript.append_field_slice(
        b"v2-meta",
        &[
            fe(log_t as u64),
            fe(log_k_ram as u64),
            fe(log_k_fetch as u64),
            fe(rows.len() as u64),
            fe(log_t_in as u64),
        ],
    )?;
    for (i, c) in commitments.iter().enumerate() {
        let mut b = (i as u32).to_le_bytes().to_vec();
        b.extend_from_slice(c);
        transcript.append_bytes(b"col-commit", &b)?;
    }
    let mut claims: Vec<ColumnClaim> = Vec::new();
    // ---- Stage 1: fetch Shout + one-hot. ----
    let fetch = {
        let res = RecRes::new(&t_cols, log_t, |f| match f {
            FactorId::ReadValues => Some(slot::FETCH_RV),
            FactorId::ReadAddr => Some(slot::FETCH_RA),
            _ => None,
        });
        let (p, dim_claims) = prove_shout_sparse(
            &fetch_table,
            &fetch_ra,
            log_k_fetch,
            log_t,
            log_k_fetch,
            &res,
            &mut transcript,
        )?;
        claims.extend(res.claims.borrow().iter().cloned());
        claims.extend(dim_claims.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL,
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    let onehot_fetch = {
        let res = RecRes::new(&t_cols, log_t, |f| match f {
            FactorId::ReadAddr => Some(slot::FETCH_RA),
            _ => None,
        });
        let (p, dim_claims) = prove_onehot_sparse(
            &fetch_ra,
            log_k_fetch,
            log_t,
            log_k_fetch,
            OHSide::Read,
            &res,
            &mut transcript,
        )?;
        claims.extend(res.claims.borrow().iter().cloned());
        claims.extend(dim_claims.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL,
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    // ---- Stage 2: input Shout. ----
    let input_shout = {
        let res = RecRes::new(&t_cols, log_t_in, |f| match f {
            FactorId::ReadValues => Some(slot::INPUT_RV),
            FactorId::ReadAddr => Some(slot::INPUT_RA),
            _ => None,
        });
        let (p, dim_claims) = prove_shout_sparse(
            &input_table,
            &input_ra,
            log_k_in,
            log_t_in,
            log_k_in,
            &res,
            &mut transcript,
        )?;
        claims.extend(res.claims.borrow().iter().cloned());
        claims.extend(dim_claims.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL,
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    // ---- Stage 3: RAM one-hots + Twist. ----
    let onehot_ram_r = {
        let res = RecRes::new(&t_cols, log_t, |f| match f {
            FactorId::ReadAddr => Some(slot::RAM_RA),
            _ => None,
        });
        let (p, dim_claims) = prove_onehot_sparse(
            &ram_ra,
            log_k_ram,
            log_t,
            log_k_ram,
            OHSide::Read,
            &res,
            &mut transcript,
        )?;
        claims.extend(res.claims.borrow().iter().cloned());
        claims.extend(dim_claims.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL,
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    let onehot_ram_w = {
        let res = RecRes::new(&t_cols, log_t, |f| match f {
            FactorId::WriteAddr => Some(slot::RAM_WA),
            _ => None,
        });
        let (p, dim_claims) = prove_onehot_sparse(
            &ram_wa,
            log_k_ram,
            log_t,
            log_k_ram,
            OHSide::Write,
            &res,
            &mut transcript,
        )?;
        claims.extend(res.claims.borrow().iter().cloned());
        claims.extend(dim_claims.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL,
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    let twist_ram = {
        // The wv column resolver supplies ReadValues/WriteValues.
        let res = RamPortsResolver {
            t_cols: &t_cols,
            log_t,
        };
        let (p, cl) = prove_twist_ports_sparse(
            &ram_witness,
            &wv_mle(&t_cols[slot::RAM_WV], log_t),
            &res,
            &mut transcript,
        )?;
        claims.extend(cl.iter().map(|(f, point, v)| ColumnClaim {
            col: twist_factor_slot(&ram_map(), *f),
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    // ---- Stage 4: register twists + one-hots. ----
    let onehot_reg_a = {
        let res = RecRes::new(&t_cols, log_t, |f| match f {
            FactorId::ReadAddr => Some(slot::REG_RA_A),
            _ => None,
        });
        let (p, dim_claims) =
            prove_onehot_sparse(&reg_ra_a, 5, log_t, 5, OHSide::Read, &res, &mut transcript)?;
        claims.extend(res.claims.borrow().iter().cloned());
        claims.extend(dim_claims.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL,
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    let onehot_reg_b = {
        let res = RecRes::new(&t_cols, log_t, |f| match f {
            FactorId::ReadAddr => Some(slot::REG_RA_B),
            _ => None,
        });
        let (p, dim_claims) =
            prove_onehot_sparse(&reg_ra_b, 5, log_t, 5, OHSide::Read, &res, &mut transcript)?;
        claims.extend(res.claims.borrow().iter().cloned());
        claims.extend(dim_claims.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL,
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    let onehot_reg_w = {
        let res = RecRes::new(&t_cols, log_t, |f| match f {
            FactorId::WriteAddr => Some(slot::REG_WA),
            _ => None,
        });
        let (p, dim_claims) =
            prove_onehot_sparse(&reg_wa, 5, log_t, 5, OHSide::Write, &res, &mut transcript)?;
        claims.extend(res.claims.borrow().iter().cloned());
        claims.extend(dim_claims.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL,
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    let twist_reg_a = {
        let res = RegPortsResolver {
            t_cols: &t_cols,
            log_t,
            rv_slot: slot::REG_RV_A,
        };
        let (p, cl) = prove_twist_ports_sparse(
            &rega_witness,
            &wv_mle(&t_cols[slot::REG_WV], log_t),
            &res,
            &mut transcript,
        )?;
        claims.extend(cl.iter().map(|(f, point, v)| ColumnClaim {
            col: twist_factor_slot(&reg_map_a(), *f),
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    let twist_reg_b = {
        let res = RegPortsResolver {
            t_cols: &t_cols,
            log_t,
            rv_slot: slot::REG_RV_B,
        };
        let (p, cl) = prove_twist_ports_sparse(
            &regb_witness,
            &wv_mle(&t_cols[slot::REG_WV], log_t),
            &res,
            &mut transcript,
        )?;
        claims.extend(cl.iter().map(|(f, point, v)| ColumnClaim {
            col: twist_factor_slot(&reg_map_b(), *f),
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    // ---- Stage 4.5: the instruction-semantics constraint families. ----
    // The trace's witness (the executed instructions' decode/ALU/
    // control/routing/termination constraints), committed through the
    // bits/values bundles with its own statement transcript. The
    // verifier consumes only public material (the statement's final
    // registers come from the public state).
    let (semantics, _) =
        prove_instruction_semantics(program, public_input, max_steps, log_k_ram, log_k_fetch)
            .map_err(|e| PipelineError::BadShape(format!("semantics: {e:?}")))?;
    // ---- Stage 5: grouped openings (the stage-4 leg batching). ----
    let mut openings_salsa = Vec::new();
    let mut openings_bound = Vec::new();
    for (ci, col) in t_cols.iter().enumerate() {
        let col_claims: Vec<GroupedOpening> = claims
            .iter()
            .filter(|c| c.col == ci)
            .map(|c| GroupedOpening {
                point: c.point.clone(),
                value: c.value,
            })
            .collect();
        let lv = col_log_vars(ci, log_t, log_t_in);
        if col_claims.is_empty() {
            placebo_opening_salsa(
                pcs,
                col,
                lv,
                stage5,
                &mut openings_salsa,
                &mut openings_bound,
            )?;
        } else {
            let mle = DenseMle {
                num_vars: lv,
                evaluations: pad_to(col, lv),
            };
            match stage5 {
                Stage5Mode::Salsa => {
                    let (resp, _packed) =
                        pcs.prove_grouped_salsa(&mle, &col_claims, &mut transcript)?;
                    openings_salsa.push(resp);
                }
                Stage5Mode::Bound => {
                    let resp = pcs.prove_grouped_salsa_bound(&mle, &col_claims, &mut transcript)?;
                    openings_bound.push(resp);
                }
            }
        }
    }
    let openings = match stage5 {
        Stage5Mode::Salsa => Stage5Openings::Salsa(openings_salsa),
        Stage5Mode::Bound => Stage5Openings::Bound(openings_bound),
    };
    Ok((
        public_state.clone(),
        ProofV2 {
            log_t,
            log_k_ram,
            log_k_fetch,
            num_fetch: fetch_ra.len(),
            num_input: public_input.len(),
            public_state: public_state.clone(),
            commitments,
            fetch,
            input_shout,
            onehot_fetch,
            onehot_ram_r,
            onehot_ram_w,
            onehot_reg_a,
            onehot_reg_b,
            onehot_reg_w,
            twist_ram,
            twist_reg_a,
            twist_reg_b,
            semantics,
            claims,
            openings,
        },
    ))
}

fn col_log_vars(ci: usize, log_t: usize, log_t_in: usize) -> usize {
    if ci == slot::INPUT_RV || ci == slot::INPUT_RA {
        log_t_in
    } else {
        log_t
    }
}

fn wv_mle(col: &[Goldilocks], log_t: usize) -> DenseMle {
    DenseMle {
        num_vars: log_t,
        evaluations: pad_to(col, log_t),
    }
}

fn pad_to(col: &[Goldilocks], log_vars: usize) -> Vec<Goldilocks> {
    let mut v = col.to_vec();
    v.resize(1usize << log_vars, Goldilocks::ZERO);
    v
}

fn placebo_opening_salsa(
    pcs: &AkitaPcs,
    col: &[Goldilocks],
    log_vars: usize,
    stage5: Stage5Mode,
    openings_salsa: &mut Vec<SalsaGroupedResponse>,
    openings_bound: &mut Vec<SalsaBoundResponse>,
) -> Result<(), PipelineError> {
    let mle = DenseMle {
        num_vars: log_vars,
        evaluations: pad_to(col, log_vars),
    };
    let mut t = Transcript::new_default(b"lzx-placebo");
    let point: Vec<Goldilocks> = (0..log_vars).map(|i| fe(i as u64 + 1)).collect();
    let claims = [GroupedOpening {
        point,
        value: fe(0),
    }];
    match stage5 {
        Stage5Mode::Salsa => {
            let (resp, _packed) = pcs.prove_grouped_salsa(&mle, &claims, &mut t)?;
            openings_salsa.push(resp);
        }
        Stage5Mode::Bound => {
            let resp = pcs.prove_grouped_salsa_bound(&mle, &claims, &mut t)?;
            openings_bound.push(resp);
        }
    }
    Ok(())
}

fn ram_map() -> fn(FactorId) -> Option<usize> {
    |f| match f {
        FactorId::ReadValues => Some(slot::RAM_RV),
        FactorId::WriteValues => Some(slot::RAM_WV),
        _ => None,
    }
}
fn reg_map_a() -> fn(FactorId) -> Option<usize> {
    |f| match f {
        FactorId::ReadValues => Some(slot::REG_RV_A),
        FactorId::WriteValues => Some(slot::REG_WV),
        _ => None,
    }
}
fn reg_map_b() -> fn(FactorId) -> Option<usize> {
    |f| match f {
        FactorId::ReadValues => Some(slot::REG_RV_B),
        FactorId::WriteValues => Some(slot::REG_WV),
        _ => None,
    }
}

fn twist_factor_slot(map: &fn(FactorId) -> Option<usize>, f: FactorId) -> usize {
    map(f).unwrap_or(usize::MAX)
}

/// Sentinel column for factor claims that ride on the (P1) dim/Inc/Val
/// commitment layer: resolvable by the verifier, skipped by openings.
pub const SENTINEL_COL: usize = usize::MAX;

/// The RAM twist's prover-side resolver over the committed columns.
struct RamPortsResolver<'a> {
    t_cols: &'a [Vec<Goldilocks>],
    log_t: usize,
}

impl<'a> FactorResolver for RamPortsResolver<'a> {
    fn eval(&self, factor: FactorId, point: &[Goldilocks]) -> Result<Goldilocks, PiopError> {
        let col = match factor {
            FactorId::ReadValues => slot::RAM_RV,
            FactorId::WriteValues => slot::RAM_WV,
            _ => return Err(PiopError::MissingFactor { factor }),
        };
        DenseMle {
            num_vars: self.log_t,
            evaluations: self.t_cols[col].clone(),
        }
        .evaluate(point)
        .map_err(PiopError::Mle)
    }
}

/// The register twists' prover-side resolver.
struct RegPortsResolver<'a> {
    t_cols: &'a [Vec<Goldilocks>],
    log_t: usize,
    rv_slot: usize,
}

impl<'a> FactorResolver for RegPortsResolver<'a> {
    fn eval(&self, factor: FactorId, point: &[Goldilocks]) -> Result<Goldilocks, PiopError> {
        let col = match factor {
            FactorId::ReadValues => self.rv_slot,
            FactorId::WriteValues => slot::REG_WV,
            _ => return Err(PiopError::MissingFactor { factor }),
        };
        DenseMle {
            num_vars: self.log_t,
            evaluations: self.t_cols[col].clone(),
        }
        .evaluate(point)
        .map_err(PiopError::Mle)
    }
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify a v2 proof. Never re-executes the program.
#[allow(clippy::too_many_lines)]
pub fn verify_v2(
    pcs: &AkitaPcs,
    program: &[u8],
    public_input: &[u8],
    public_state: &PublicStateV2,
    proof: &ProofV2,
    max_steps: u64,
) -> Result<(), PipelineError> {
    if public_state.num_steps > max_steps || public_state.num_steps == 0 {
        return Err(PipelineError::VerificationFailed);
    }
    let log_t = proof.log_t;
    let log_t_in = public_input
        .len()
        .div_ceil(8)
        .max(1)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let num_input_words = public_input.len().div_ceil(8).max(1);
    let log_k_in = num_input_words.next_power_of_two().max(2).trailing_zeros() as usize;
    // Recompute the public tables.
    let mut fetch_table: Vec<Goldilocks> = (0..program.len() / 4)
        .map(|i| {
            let mut w = 0u32;
            for b in 0..4 {
                w |= (program[i * 4 + b] as u32) << (b * 8);
            }
            fe(w as u64)
        })
        .collect();
    fetch_table.resize(1 << proof.log_k_fetch, Goldilocks::ZERO);
    let mut input_table: Vec<Goldilocks> = (0..num_input_words)
        .map(|i| {
            let mut w = 0u64;
            for b in 0..8 {
                let off = i * 8 + b;
                if off < public_input.len() {
                    w |= (public_input[off] as u64) << (b * 8);
                }
            }
            fe(w)
        })
        .collect();
    input_table.resize(1 << log_k_in, Goldilocks::ZERO);
    // Public init/final RAM.
    let log_k_ram = proof.log_k_ram;
    let k_ram = 1usize << log_k_ram;
    if public_state.final_ram.len() != k_ram {
        return Err(PipelineError::VerificationFailed);
    }
    let mut init_ram = vec![0u64; k_ram];
    for (i, chunk) in program.chunks(8).enumerate() {
        let mut w = 0u64;
        for (b, byte) in chunk.iter().enumerate() {
            w |= (*byte as u64) << (b * 8);
        }
        let idx = 0x1000 / 8 + i;
        if idx < k_ram {
            init_ram[idx] = w;
        }
    }
    for (i, chunk) in public_input.chunks(8).enumerate() {
        let mut w = 0u64;
        for (b, byte) in chunk.iter().enumerate() {
            w |= (*byte as u64) << (b * 8);
        }
        let idx = 0x3000 / 8 + i;
        if idx < k_ram {
            init_ram[idx] = w;
        }
    }
    // Transcript replay.
    let mut transcript = Transcript::new_default(b"lzx-zkvm-v2");
    let prog_digest = crate::program_digest(program);
    let input_digest = crate::public_input_digest(public_input);
    transcript.append_bytes(b"program", &prog_digest)?;
    transcript.append_bytes(b"public-input", &input_digest)?;
    transcript.append_field_slice(
        b"v2-meta",
        &[
            fe(log_t as u64),
            fe(log_k_ram as u64),
            fe(proof.log_k_fetch as u64),
            fe(public_state.num_steps),
            fe(log_t_in as u64),
        ],
    )?;
    for (i, c) in proof.commitments.iter().enumerate() {
        let mut b = (i as u32).to_le_bytes().to_vec();
        b.extend_from_slice(c);
        transcript.append_bytes(b"col-commit", &b)?;
    }
    let claims = &proof.claims;
    // Stage 1: fetch.
    let fetch_res = TableRes {
        claims,
        map: |f| match f {
            FactorId::ReadValues => Some(slot::FETCH_RV),
            FactorId::ReadAddr => Some(slot::FETCH_RA),
            _ => None,
        },
    };
    lattice_memory::verify_shout(
        &proof.fetch,
        &fetch_table,
        proof.log_k_fetch,
        log_t,
        proof.log_k_fetch,
        &fetch_res,
        &mut transcript,
    )?;
    let oh_fetch_res = TableRes {
        claims,
        map: |f| match f {
            FactorId::ReadAddr => Some(slot::FETCH_RA),
            _ => None,
        },
    };
    lattice_memory::verify_onehot(
        &proof.onehot_fetch,
        proof.log_k_fetch,
        log_t,
        OHSide::Read,
        &oh_fetch_res,
        &mut transcript,
    )?;
    // Stage 2: input.
    let in_res = TableRes {
        claims,
        map: |f| match f {
            FactorId::ReadValues => Some(slot::INPUT_RV),
            FactorId::ReadAddr => Some(slot::INPUT_RA),
            _ => None,
        },
    };
    lattice_memory::verify_shout(
        &proof.input_shout,
        &input_table,
        log_k_in,
        log_t_in,
        log_k_in,
        &in_res,
        &mut transcript,
    )?;
    // Stage 3: RAM.
    let ram_oh_r = TableRes {
        claims,
        map: |f| match f {
            FactorId::ReadAddr => Some(slot::RAM_RA),
            _ => None,
        },
    };
    lattice_memory::verify_onehot(
        &proof.onehot_ram_r,
        log_k_ram,
        log_t,
        OHSide::Read,
        &ram_oh_r,
        &mut transcript,
    )?;
    let ram_oh_w = TableRes {
        claims,
        map: |f| match f {
            FactorId::WriteAddr => Some(slot::RAM_WA),
            _ => None,
        },
    };
    lattice_memory::verify_onehot(
        &proof.onehot_ram_w,
        log_k_ram,
        log_t,
        OHSide::Write,
        &ram_oh_w,
        &mut transcript,
    )?;
    let ram_tw = TableRes {
        claims,
        map: ram_map_pub(),
    };
    verify_twist_ports_checked(
        &proof.twist_ram,
        &init_ram.iter().map(|&v| fe(v)).collect::<Vec<_>>(),
        &public_state
            .final_ram
            .iter()
            .map(|&v| fe(v))
            .collect::<Vec<_>>(),
        log_k_ram,
        log_t,
        log_k_ram,
        &ram_tw,
        &mut transcript,
    )?;
    // Stage 4: registers.
    let reg_init: Vec<Goldilocks> = {
        let mut v = vec![Goldilocks::ZERO; 32];
        v[10] = fe(public_input.len() as u64);
        v
    };
    let reg_final: Vec<Goldilocks> = public_state.final_regs.iter().map(|&x| fe(x)).collect();
    let oh_ra = TableRes {
        claims,
        map: |f| match f {
            FactorId::ReadAddr => Some(slot::REG_RA_A),
            _ => None,
        },
    };
    lattice_memory::verify_onehot(
        &proof.onehot_reg_a,
        5,
        log_t,
        OHSide::Read,
        &oh_ra,
        &mut transcript,
    )?;
    let oh_rb = TableRes {
        claims,
        map: |f| match f {
            FactorId::ReadAddr => Some(slot::REG_RA_B),
            _ => None,
        },
    };
    lattice_memory::verify_onehot(
        &proof.onehot_reg_b,
        5,
        log_t,
        OHSide::Read,
        &oh_rb,
        &mut transcript,
    )?;
    let oh_rw = TableRes {
        claims,
        map: |f| match f {
            FactorId::WriteAddr => Some(slot::REG_WA),
            _ => None,
        },
    };
    lattice_memory::verify_onehot(
        &proof.onehot_reg_w,
        5,
        log_t,
        OHSide::Write,
        &oh_rw,
        &mut transcript,
    )?;
    let rega_tw = TableRes {
        claims,
        map: reg_map_a(),
    };
    verify_twist_ports_checked(
        &proof.twist_reg_a,
        &reg_init,
        &reg_final,
        5,
        log_t,
        5,
        &rega_tw,
        &mut transcript,
    )?;
    let regb_tw = TableRes {
        claims,
        map: reg_map_b(),
    };
    verify_twist_ports_checked(
        &proof.twist_reg_b,
        &reg_init,
        &reg_final,
        5,
        log_t,
        5,
        &regb_tw,
        &mut transcript,
    )?;
    // Stage 4.5: the instruction semantics (no re-execution; the
    // statement's digests + the public final registers).
    crate::semantics::verify_instruction_semantics(&proof.semantics, program, public_input)
        .map_err(|e| PipelineError::BadShape(format!("semantics: {e:?}")))?;
    // Stage 5: grouped openings per committed column (the mode's own
    // verifier — the Bound mode's chains re-derive every stage's
    // estimator verdict against the transmitted commitments).
    let ring = &pcs.pk.params.ring;
    let n_openings = match &proof.openings {
        Stage5Openings::Salsa(v) => v.len(),
        Stage5Openings::Bound(v) => v.len(),
    };
    for ci in 0..n_openings {
        let col_claims: Vec<GroupedOpening> = claims
            .iter()
            .filter(|c| c.col == ci)
            .map(|c| GroupedOpening {
                point: c.point.clone(),
                value: c.value,
            })
            .collect();
        if col_claims.is_empty() {
            continue; // placebo column (separate transcript at prove time)
        }
        let commitment = lattice_commitment::ajtai::AjtaiCommitment::from_bytes(
            ring,
            pcs.pk.params.k,
            &proof.commitments[ci],
        )
        .map_err(|_| lattice_akita::pcs::AkitaPcsError::VerificationFailed)?;
        let comm = lattice_akita::pcs::Commitment {
            commitment,
            num_packed: 0,
            num_vars: col_log_vars(ci, log_t, log_t_in),
        };
        match &proof.openings {
            Stage5Openings::Salsa(v) => {
                pcs.verify_grouped_salsa(&comm, &col_claims, &v[ci], &mut transcript)?;
            }
            Stage5Openings::Bound(v) => {
                pcs.verify_grouped_salsa_bound(&comm, &col_claims, &v[ci], &mut transcript)?;
            }
        }
    }
    Ok(())
}

fn ram_map_pub() -> fn(FactorId) -> Option<usize> {
    |f| match f {
        FactorId::ReadValues => Some(slot::RAM_RV),
        FactorId::WriteValues => Some(slot::RAM_WV),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> AkitaPcs {
        lattice_akita::akita_setup(4, 64, 1 << 23, [91u8; 32])
            .ok()
            .unwrap()
    }

    fn demo_program() -> Vec<u8> {
        // addi x1, x0, 8; addi x2, x0, 7; add x3, x1, x2; sd x3, 0(x1);
        // ld x4, 0(x1); ecall
        let words = [
            0x0080_0093u32,
            0x0070_0113,
            0x0020_81b3,
            0x0030_b023,
            0x0000_b203,
            0x0000_0073,
        ];
        let mut v = Vec::new();
        for w in words {
            v.extend_from_slice(&w.to_le_bytes());
        }
        v
    }

    #[test]
    fn v2_prove_and_verify_happy_path() {
        let pcs = setup();
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, proof) = prove_v2(&pcs, &program, &input, 64).ok().unwrap();
        assert_eq!(state.num_steps, 6, "regs={:?}", &state.final_regs[..8]);
        assert_eq!(state.final_regs[4], 15);
        // The stored word is visible in the final RAM window.
        assert_eq!(state.final_ram[8 / 8], 15); // word at absolute address 8
        match verify_v2(&pcs, &program, &input, &state, &proof, 64) {
            Ok(_) => {}
            Err(e) => panic!("verify err: {e:?}"),
        }
    }

    #[test]
    fn v2_tampered_final_state_rejected() {
        let pcs = setup();
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (mut state, proof) = prove_v2(&pcs, &program, &input, 64).ok().unwrap();
        state.final_ram[0x201] = state.final_ram[0x201].wrapping_add(1);
        assert!(verify_v2(&pcs, &program, &input, &state, &proof, 64).is_err());
    }

    #[test]
    fn v2_tampered_claim_rejected() {
        let pcs = setup();
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, mut proof) = prove_v2(&pcs, &program, &input, 64).ok().unwrap();
        if let Some(c) = proof.claims.first_mut() {
            c.value = c.value.add(&Goldilocks::ONE);
        }
        assert!(verify_v2(&pcs, &program, &input, &state, &proof, 64).is_err());
    }

    /// The D4 response-layer swap's tamper coverage at the pipeline
    /// level: a tampered SALSAA opening (f_term / z_r / carrier round)
    /// must reject at Stage 5.
    #[test]
    fn v2_tampered_salsa_opening_rejected() {
        let pcs = setup();
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, mut proof) = prove_v2(&pcs, &program, &input, 64).ok().unwrap();
        // Tamper the first non-placebo opening's f_term: the terminal
        // binding fails.
        if let Stage5Openings::Salsa(ops) = &mut proof.openings {
            if let Some(op) = ops.iter_mut().find(|o| o.chain.num_elements > 0) {
                op.f_term = op.f_term.add(&Goldilocks::ONE);
            }
        }
        assert!(verify_v2(&pcs, &program, &input, &state, &proof, 64).is_err());

        // Tamper z_r on a fresh proof: the D1 reconstruction fails.
        let (state2, mut proof2) = prove_v2(&pcs, &program, &input, 64).ok().unwrap();
        if let Stage5Openings::Salsa(ops) = &mut proof2.openings {
            if let Some(op) = ops.iter_mut().find(|o| o.chain.num_elements > 0) {
                op.z_r = op.z_r.add(&Goldilocks::ONE);
            }
        }
        assert!(verify_v2(&pcs, &program, &input, &state2, &proof2, 64).is_err());

        // The proof carries NO opened witness anywhere (the disclosure
        // removal): the openings' wire is two sumchecks + O(1) claims.
        let (_, proof3) = prove_v2(&pcs, &program, &input, 64).ok().unwrap();
        if let Stage5Openings::Salsa(ops) = &proof3.openings {
            for op in ops {
                assert!(std::mem::size_of_val(&op.chain) < 4096);
            }
        }
    }

    /// The D4 BINDING-CLOSED composition at the pipeline layer: the
    /// v2 pipeline with `Stage5Mode::Bound` — every committed column's
    /// byte-witness bound through its width-collapse chain (the
    /// authenticated opening at the challenge). Prove → verify; a
    /// tampered COMMITMENT (the binding the open mode lacked) rejects.
    #[test]
    fn v2_stage5_bound_composition_honest_and_tampered() {
        let pcs = setup();
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, proof) = prove_v2_with_stage5(&pcs, &program, &input, 64, Stage5Mode::Bound)
            .ok()
            .unwrap();
        match verify_v2(&pcs, &program, &input, &state, &proof, 64) {
            Ok(_) => {}
            Err(e) => panic!("the bound-mode v2 pipeline must verify: {e:?}"),
        }

        // THE PIPELINE-LEVEL CLOSURE TEST: swap one column's commitment
        // for another column's — the chain's (W0) part-image sum no
        // longer matches the (re-derived) commitment, Stage 5 rejects.
        let (state2, mut proof2) =
            prove_v2_with_stage5(&pcs, &program, &input, 64, Stage5Mode::Bound)
                .ok()
                .unwrap();
        if proof2.commitments.len() >= 2 {
            proof2.commitments.swap(0, 1);
            assert!(verify_v2(&pcs, &program, &input, &state2, &proof2, 64).is_err());
        }

        // A tampered bound opening's f_term: the carrier terminal AND
        // the fold's functional thread reject.
        let (state3, mut proof3) =
            prove_v2_with_stage5(&pcs, &program, &input, 64, Stage5Mode::Bound)
                .ok()
                .unwrap();
        if let Stage5Openings::Bound(ops) = &mut proof3.openings {
            if let Some(op) = ops.iter_mut().find(|o| o.fold.n_bar > 0) {
                op.f_term = op.f_term.add(&Goldilocks::ONE);
            }
        }
        assert!(verify_v2(&pcs, &program, &input, &state3, &proof3, 64).is_err());
    }

    #[test]
    fn v2_wrong_program_rejected() {
        let pcs = setup();
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, proof) = prove_v2(&pcs, &program, &input, 64).ok().unwrap();
        let mut other = program.clone();
        other[0] ^= 0x01;
        assert!(verify_v2(&pcs, &other, &input, &state, &proof, 64).is_err());
    }
}

#[cfg(test)]
mod semantics_pipeline_tests {
    use super::*;

    fn setup() -> AkitaPcs {
        lattice_akita::akita_setup(4, 64, 1 << 23, [91u8; 32])
            .ok()
            .unwrap()
    }

    fn enc_addi(rd: u8, rs1: u8, imm: i64) -> u32 {
        ((imm as u32 & 0xFFF) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0x13
    }
    fn enc_r(f7: u32, rs2: u8, rs1: u8, f3: u32, rd: u8, op: u32) -> u32 {
        (f7 << 25)
            | ((rs2 as u32) << 20)
            | ((rs1 as u32) << 15)
            | (f3 << 12)
            | ((rd as u32) << 7)
            | op
    }

    /// A full-semantics program through the COMPLETE v2 pipeline:
    /// memory + arithmetic + shifts + MUL + DIV + branches.
    fn full_semantics_program() -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&enc_addi(1, 0, -7).to_le_bytes()); // x1 = -7
        p.extend_from_slice(&enc_addi(2, 0, 3).to_le_bytes()); // x2 = 3
        p.extend_from_slice(&enc_addi(10, 0, 40).to_le_bytes()); // shamt source
        p.extend_from_slice(&enc_r(1, 2, 1, 0, 3, 0x33).to_le_bytes()); // mul x3
        p.extend_from_slice(&enc_r(1, 2, 1, 1, 4, 0x33).to_le_bytes()); // mulh x4
        p.extend_from_slice(&enc_r(1, 2, 1, 4, 5, 0x33).to_le_bytes()); // div x5
        p.extend_from_slice(&enc_r(1, 2, 1, 6, 6, 0x33).to_le_bytes()); // rem x6
        p.extend_from_slice(&enc_r(0, 10, 1, 1, 7, 0x33).to_le_bytes()); // sll x7
        p.extend_from_slice(&enc_r(0x20, 10, 1, 5, 8, 0x33).to_le_bytes()); // sra x8
                                                                            // Memory: store x3 then load it back.
        p.extend_from_slice(&enc_addi(20, 0, 64).to_le_bytes());
        let sd: u32 = (3u32 << 20) | (20 << 15) | (3 << 12) | 0x23;
        p.extend_from_slice(&sd.to_le_bytes());
        let ld: u32 = (20 << 15) | (3 << 12) | (21 << 7) | 0x03;
        p.extend_from_slice(&ld.to_le_bytes());
        p.extend_from_slice(&0x73u32.to_le_bytes());
        p
    }

    #[test]
    fn v2_full_semantics_prove_and_verify() {
        let pcs = setup();
        let program = full_semantics_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, proof) =
            prove_v2(&pcs, &program, &input, 128).unwrap_or_else(|e| panic!("prove: {e:?}"));
        // The semantics roundtrip: mul(-7*3) = -21 (wrapping), div
        // truncating, the load returns the stored word.
        assert_eq!(state.final_regs[3], (-21i64) as u64);
        assert_eq!(state.final_regs[5], (-7i64 / 3i64) as u64);
        assert_eq!(state.final_regs[21], (-21i64) as u64);
        match verify_v2(&pcs, &program, &input, &state, &proof, 128) {
            Ok(_) => {}
            Err(e) => panic!("verify err: {e:?}"),
        }
    }

    #[test]
    fn v2_full_semantics_tampered_final_rejected() {
        let pcs = setup();
        let program = full_semantics_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (mut state, proof) = prove_v2(&pcs, &program, &input, 128).ok().unwrap();
        // Tamper a register the CONSTRAINT layer pins (the mul result).
        state.final_regs[3] = state.final_regs[3].wrapping_add(1);
        assert!(verify_v2(&pcs, &program, &input, &state, &proof, 128).is_err());
    }

    #[test]
    fn v2_full_semantics_tampered_leg_rejected() {
        let pcs = setup();
        let program = full_semantics_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, mut proof) = prove_v2(&pcs, &program, &input, 128).ok().unwrap();
        // Corrupt the semantics layer's mul leg: the family identity
        // must fail at verify.
        if let Some(leg) = proof.semantics.legs.iter_mut().find(|l| l.name == "mul") {
            if let Some(r0) = leg.sc.rounds.first_mut() {
                if let Some(e0) = r0.first_mut() {
                    *e0 = e0.add(&Goldilocks::ONE);
                }
            }
        }
        assert!(verify_v2(&pcs, &program, &input, &state, &proof, 128).is_err());
    }

    #[test]
    fn v2_full_semantics_tampered_claim_rejected() {
        let pcs = setup();
        let program = full_semantics_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, mut proof) = prove_v2(&pcs, &program, &input, 128).ok().unwrap();
        if let Some(c) = proof.semantics.claims.first_mut() {
            c.value = c.value.add(&Goldilocks::ONE);
        }
        assert!(verify_v2(&pcs, &program, &input, &state, &proof, 128).is_err());
    }
}
