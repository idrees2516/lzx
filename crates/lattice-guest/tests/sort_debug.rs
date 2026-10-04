#[test]
fn sort_debug() {
    use lattice_guest::asm::{run_on_vm, AssembledProgram};
    use lattice_guest::programs::{reference_sorting, sorting};
    let vals: Vec<u64> = vec![3, 1, 2];
    let prog = sorting(&vals).ok().unwrap();
    println!("expected {:x}", reference_sorting(&vals));
    let asm_prog = AssembledProgram {
        code: prog.image.clone(),
        data: vec![],
        labels: Default::default(),
        data_base: prog.data_base,
    };
    let run = run_on_vm(&asm_prog, &prog.public_input, 100000)
        .ok()
        .unwrap();
    println!("got reg x10 = {:x}", run.state.reg(10));
    for r in [8u8, 9, 10, 12, 13, 14, 15] {
        println!("x{} = {:x}", r, run.state.reg(r));
    }
    println!(
        "image len = {} data_base = {:x}",
        prog.image.len(),
        prog.data_base
    );
    // dump raw image around data_base
    let db = prog.data_base as usize;
    for off in [0usize, 4, 12, 20] {
        let b = &prog.image[db + off..db + off + 8];
        println!(
            "image[{:x}+{}] = {:x}",
            db,
            off,
            u64::from_le_bytes(b.try_into().ok().unwrap())
        );
    }
    // dump the data region: data_base.. + 4 + 3*8
    for i in 0..3 {
        let addr = prog.data_base + 4 + i * 8;
        println!("mem[{}] = {}", i, run.state.memory.load_u64(addr));
    }
}
