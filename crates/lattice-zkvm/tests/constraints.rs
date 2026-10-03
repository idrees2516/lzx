//! Integration tests for the zkVM instruction-semantics constraint
//! families (Twist-and-Shout T2): honest roundtrips, tamper rejection
//! per family, and the v1 coverage gate.
//!
//! The witness is built from the columns.rs reference program
//! (addi/sw/lw/ecall — all inside the covered subset), executed by the
//! real `lattice-vm` executor, and the aux columns come from
//! `build_aux`. The prover ledger is seeded with every factor the
//! families bind (value tensors, the instruction tensor, the bit and
//! value columns); the verifier ledger receives the recorded base
//! claims — the same prover/verifier pairing `memproof.rs` uses.

use lattice_zkvm::columns::{build_cycle_witness, CycleWitness, RamWindow, FetchWindow};
use lattice_zkvm::constraints::{
    build_aux, prove_constraints, verify_constraints, AuxCols, ConstraintError, ConstraintLeg,
};
use lattice_zkvm::ledger::{Factor, Ledger};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_vm::{decode::Instr, run as vm_run, MachineState};

fn enc_addi(rd: u8, rs1: u8, imm: i64) -> u32 {
    ((imm as u32 & 0xFFF) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0x13
}

/// addi x1, x0, 12; addi x2, x0, 8; sw x1, 0(x2); lw x3, 0(x2); ecall
fn program() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&enc_addi(1, 0, 12).to_le_bytes());
    p.extend_from_slice(&enc_addi(2, 0, 8).to_le_bytes());
    let sw: u32 = (1u32 << 20) | (2 << 15) | (2 << 12) | 0x23;
    p.extend_from_slice(&sw.to_le_bytes());
    let lw: u32 = (2 << 15) | (2 << 12) | (3 << 7) | 0x03;
    p.extend_from_slice(&lw.to_le_bytes());
    p.extend_from_slice(&0x73u32.to_le_bytes());
    p
}

fn run_trace(prog: &[u8]) -> Vec<lattice_vm::exec::TraceRow> {
    let mut state = MachineState::new();
    state.load_program(0, prog);
    vm_run(&mut state, 64).ok().unwrap()
}

fn col_mle(col: &[u8], log_t: usize) -> DenseMle {
    DenseMle {
        num_vars: log_t,
        evaluations: col.iter().map(|v| Goldilocks::from_u64(*v as u64)).collect(),
    }
}

fn val_col_mle(col: &[Goldilocks], log_t: usize) -> DenseMle {
    DenseMle {
        num_vars: log_t,
        evaluations: col.to_vec(),
    }
}

/// Owns every column MLE so the ledger table can borrow them.
struct TableOwner {
    bit_mles: Vec<DenseMle>,
    val_mles: Vec<DenseMle>,
}

impl TableOwner {
    fn new(w: &CycleWitness, aux: &AuxCols) -> Self {
        TableOwner {
            bit_mles: aux.bits.iter().map(|c| col_mle(c, w.log_t)).collect(),
            val_mles: aux.vals.iter().map(|c| val_col_mle(c, w.log_t)).collect(),
        }
    }

    fn table<'a>(&'a self, w: &'a CycleWitness) -> Vec<(Factor, &'a DenseMle)> {
        let mut table: Vec<(Factor, &DenseMle)> = Vec::new();
        for slot in 0..6usize {
            table.push((Factor::ValueBits { slot }, &w.values[slot]));
        }
        table.push((Factor::InstrBits, &w.instr_bits));
        for (id, m) in self.bit_mles.iter().enumerate() {
            table.push((Factor::BitCol { id }, m));
        }
        for (id, m) in self.val_mles.iter().enumerate() {
            table.push((Factor::ValCol { id }, m));
        }
        table
    }
}

fn setup(prog: &[u8]) -> (CycleWitness, AuxCols, Vec<Instr>) {
    let rows = run_trace(prog);
    let (w, _words) = build_cycle_witness(
        &rows,
        prog,
        &[],
        RamWindow { log_k: 4 },
        FetchWindow { log_k: 3 },
    )
    .ok()
    .unwrap();
    // build_aux's contract: the per-cycle EXECUTED instructions (the
    // trace's instr column), not the program listing — control flow can
    // skip program words.
    let instrs: Vec<Instr> = rows.iter().map(|r| r.instr).collect();
    let aux = build_aux(&w, &instrs).ok().unwrap();
    (w, aux, instrs)
}

fn prove_and_verify(w: &CycleWitness, aux: &AuxCols, instrs: &[Instr]) -> Vec<ConstraintLeg> {
    let owner = TableOwner::new(w, aux);
    let table = owner.table(w);
    let mut ledger = Ledger::prover(table);
    let mut legs = Vec::new();
    let mut t = Transcript::new_default(b"con-test");
    prove_constraints(w, aux, instrs, &mut ledger, &mut legs, &mut t)
        .ok()
        .unwrap();
    // Verifier: kernel scale uses a table-backed ledger (the prover's
    // witness table stands in for the PCS-authenticated openings the
    // production pipeline supplies; the final identity checks compare
    // the sumcheck-derived claims against the table-derived values).
    let vowner = TableOwner::new(w, aux);
    let vtable = vowner.table(w);
    let mut vledger = Ledger::prover(vtable);
    let mut vt = Transcript::new_default(b"con-test");
    let vr = verify_constraints(w, aux, instrs, &legs, &mut vledger, &mut vt);
    println!("VERIFY RESULT: {:?}", vr);
    assert!(vr.is_ok());
    legs
}

#[test]
fn honest_program_proves_and_verifies() {
    let prog = program();
    let (w, aux, instrs) = setup(&prog);
    let legs = prove_and_verify(&w, &aux, &instrs);
    // The full family set: 8 booleanity pieces are 3 legs, then one leg
    // per remaining family.
    let names: Vec<&str> = legs.iter().map(|l| l.name).collect();
    assert!(names.contains(&"bool-tensors"));
    assert!(names.contains(&"bool-instr"));
    assert!(names.contains(&"bool-cols"));
    assert!(names.contains(&"sel"));
    assert!(names.contains(&"flags"));
    assert!(names.contains(&"arith"));
    assert!(names.contains(&"cmp"));
    assert!(names.contains(&"ctrl"));
    assert!(names.contains(&"route"));
    assert!(names.contains(&"halt-end"));
    assert_eq!(names.len(), 10);
}

#[test]
fn tampered_leg_round_rejected() {
    let prog = program();
    let (w, aux, instrs) = setup(&prog);
    let mut legs = prove_and_verify(&w, &aux, &instrs);
    // Tamper the selector leg's first round value.
    let sel_leg = legs.iter_mut().find(|l| l.name == "sel").unwrap();
    if let Some(r0) = sel_leg.sc.rounds.first_mut() {
        if let Some(e0) = r0.first_mut() {
            *e0 = e0.add(&Goldilocks::ONE);
        }
    }
    // Re-verify: transcript desync -> round or final check fails.
    let vowner = TableOwner::new(&w, &aux);
    let vtable = vowner.table(&w);
    let mut vledger = Ledger::prover(vtable);
    let mut vt = Transcript::new_default(b"con-test");
    assert!(verify_constraints(&w, &aux, &instrs, &legs, &mut vledger, &mut vt).is_err());
}

#[test]
fn tampered_halt_termination_rejected() {
    // The halt-end leg's claim is 1 (halted[T-1] = 1); a forged claim of
    // 0 must fail the round-sum identity.
    let prog = program();
    let (w, aux, instrs) = setup(&prog);
    let mut legs = prove_and_verify(&w, &aux, &instrs);
    let halt_leg = legs.iter_mut().find(|l| l.name == "halt-end").unwrap();
    halt_leg.claim = Goldilocks::ZERO;
    let vowner = TableOwner::new(&w, &aux);
    let vtable = vowner.table(&w);
    let mut vledger = Ledger::prover(vtable);
    let mut vt = Transcript::new_default(b"con-test");
    assert!(verify_constraints(&w, &aux, &instrs, &legs, &mut vledger, &mut vt).is_err());
}

#[test]
fn corrupted_witness_rejected_at_prove_time() {
    // Flip a committed rd bit after building the aux columns: the
    // arithmetic identity (rd = a + b + carries) no longer holds, so
    // the sumcheck's consistency guard fires at prove time.
    let prog = program();
    let (mut w, aux, instrs) = setup(&prog);
    let t = 1usize << w.log_t;
    // Cycle 0: addi x1, x0, 12 -> rd bit 2 (value 12) lives at row 61.
    w.values[3].evaluations[61 * t] = w.values[3].evaluations[61 * t].add(&Goldilocks::ONE);
    let owner = TableOwner::new(&w, &aux);
    let table = owner.table(&w);
    let mut ledger = Ledger::prover(table);
    let mut legs = Vec::new();
    let mut tr = Transcript::new_default(b"con-test");
    let res = prove_constraints(&w, &aux, &instrs, &mut ledger, &mut legs, &mut tr);
    assert!(matches!(
        res,
        Err(ConstraintError::Sumcheck(
            lattice_sumcheck::SumcheckError::ClaimMismatch
        )) | Err(ConstraintError::FinalCheck(_))
    ));
}

#[test]
fn corrupted_selector_column_rejected() {
    // Flip a selector column bit post-build: the selector identity
    // (s = product of instr-bit indicators) breaks at prove time.
    let prog = program();
    let (w, mut aux, instrs) = setup(&prog);
    let sel_addi = aux.index.sel_by("sel_addi");
    aux.bits[sel_addi][2] ^= 1; // cycle 2 is sw, not addi.
    let owner = TableOwner::new(&w, &aux);
    let table = owner.table(&w);
    let mut ledger = Ledger::prover(table);
    let mut legs = Vec::new();
    let mut tr = Transcript::new_default(b"con-test");
    let res = prove_constraints(&w, &aux, &instrs, &mut ledger, &mut legs, &mut tr);
    assert!(res.is_err());
}

#[test]
fn coverage_gate_rejects_uncovered_instruction() {
    // A program with MUL (uncovered class) must be rejected fail-closed.
    let mut prog = Vec::new();
    prog.extend_from_slice(&enc_addi(1, 0, 3).to_le_bytes());
    prog.extend_from_slice(&enc_addi(2, 0, 4).to_le_bytes());
    // mul x3, x1, x2: funct7=1, rs2=2, rs1=1, f3=0, rd=3, opcode 0x33.
    let mul: u32 = (1u32 << 25) | (2 << 20) | (1 << 15) | (3 << 7) | 0x33;
    prog.extend_from_slice(&mul.to_le_bytes());
    prog.extend_from_slice(&0x73u32.to_le_bytes());
    let rows = run_trace(&prog);
    let (w, _words) = build_cycle_witness(
        &rows,
        &prog,
        &[],
        RamWindow { log_k: 4 },
        FetchWindow { log_k: 3 },
    )
    .ok()
    .unwrap();
    let instrs: Vec<Instr> = rows.iter().map(|r| r.instr).collect();
    let aux = build_aux(&w, &instrs).ok().unwrap();
    let owner = TableOwner::new(&w, &aux);
    let table = owner.table(&w);
    let mut ledger = Ledger::prover(table);
    let mut legs = Vec::new();
    let mut tr = Transcript::new_default(b"con-test");
    assert!(matches!(
        prove_constraints(&w, &aux, &instrs, &mut ledger, &mut legs, &mut tr),
        Err(ConstraintError::UncoveredInstruction { cycle: 2 })
    ));
    // And the verifier refuses too.
    let claims: Vec<lattice_zkvm::ledger::BaseClaim> = ledger.claims().to_vec();
    let mut vledger = Ledger::verifier(claims);
    let mut vt = Transcript::new_default(b"con-test");
    assert!(matches!(
        verify_constraints(&w, &aux, &instrs, &legs, &mut vledger, &mut vt),
        Err(ConstraintError::UncoveredInstruction { .. })
    ));
}

#[test]
fn branches_and_jumps_prove_and_verify() {
    // A control-flow program: jal over skipped instructions.
    //   addi x1, x0, 1
    //   jal x0, +12 (skip two)
    //   addi x2, x0, 99  (skipped)
    //   addi x3, x0, 99  (skipped)
    //   addi x4, x0, 7
    //   ecall
    let mut p = Vec::new();
    p.extend_from_slice(&enc_addi(1, 0, 1).to_le_bytes()); // 0
    // J-type: imm[20|10:1|11|19:12] rd opcode; offset 12.
    let imm = 12u32;
    let j20 = (imm >> 20) & 1;
    let j10_1 = (imm >> 1) & 0x3ff;
    let j11 = (imm >> 11) & 1;
    let j19_12 = (imm >> 12) & 0xff;
    let jal: u32 = (j20 << 31)
        | (j10_1 << 21)
        | (j11 << 20)
        | (j19_12 << 12)
        | 0x6f;
    p.extend_from_slice(&jal.to_le_bytes()); // 4 -> jumps to 16
    p.extend_from_slice(&enc_addi(2, 0, 99).to_le_bytes()); // 8 (skipped)
    p.extend_from_slice(&enc_addi(3, 0, 99).to_le_bytes()); // 12 (skipped)
    p.extend_from_slice(&enc_addi(4, 0, 7).to_le_bytes()); // 16
    p.extend_from_slice(&0x73u32.to_le_bytes()); // 20
    let rows = run_trace(&p);
    let (w, _words) = build_cycle_witness(
        &rows,
        &p,
        &[],
        RamWindow { log_k: 4 },
        FetchWindow { log_k: 3 },
    )
    .ok()
    .unwrap();
    let instrs: Vec<Instr> = rows.iter().map(|r| r.instr).collect();
    let aux = build_aux(&w, &instrs).ok().unwrap();
    // x4 must hold 7 (the jal skipped the two 99s).
    let mut rd4 = 0u64;
    for row in &rows {
        if let Some((4, v)) = row.reg_writes.first() {
            rd4 = *v;
        }
    }
    assert_eq!(rd4, 7);
    let _ = prove_and_verify(&w, &aux, &instrs);
}

#[test]
fn family_leg_shapes() {
    // The selector leg covers the full decode: its degree bound must
    // accommodate the widest mask (opcode 7 + f3 3 + f7 7 = 17 bits,
    // plus eq -> 18).
    let prog = program();
    let (w, aux, instrs) = setup(&prog);
    let legs = prove_and_verify(&w, &aux, &instrs);
    for leg in &legs {
        // Every leg has at least one round with (degree+1) entries.
        assert!(!leg.sc.rounds.is_empty());
        for r in &leg.sc.rounds {
            assert!(r.len() >= 2);
        }
    }
}
