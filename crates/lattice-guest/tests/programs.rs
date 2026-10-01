//! Every guest program runs on lattice-vm and matches its reference.

use lattice_guest::asm::run_on_vm;
use lattice_guest::programs::suite;

#[test]
fn suite_programs_match_references() {
    let s = suite();
    if let Err(e) = &s {
        panic!("suite build failed: {:?}", e);
    }
    for prog in s.ok().unwrap() {
        // Build a single image: program at 0 (its inline data travels
        // with it via data_base), public input at 0x1000.
        let mut image = prog.image.clone();
        // The assembler's inline data is already appended to the code
        // image; place_data_at was not used, so segments() covers it.
        let asm_prog = lattice_guest::asm::AssembledProgram {
            code: image.clone(),
            data: vec![],
            labels: Default::default(),
            data_base: prog.data_base,
        };
        let run = run_on_vm(&asm_prog, &prog.public_input, 2_000_000)
            .unwrap_or_else(|e| panic!("{}: {:?}", prog.name, e));
        assert!(run.state.halted, "{} did not halt", prog.name);
        for (reg, want) in &prog.result_regs {
            let got = run.state.reg(*reg);
            assert_eq!(
                got, *want,
                "{}: register x{} mismatch (got {:?}, cycles {})",
                prog.name, reg, got, run.steps
            );
        }
        let _ = &mut image;
    }
}
