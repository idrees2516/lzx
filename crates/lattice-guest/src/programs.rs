//! Jolt-style benchmark guest programs, written against the crate's own
//! RV64IM assembler. Every program is a real algorithm that terminates
//! with `ecall`, leaves its primary result in `a0` (x10), reads its
//! public input from [`crate::PUBLIC_INPUT_BASE`] (0x1000) as aligned
//! words, and uses only aligned `lw`/`lwu`/`ld`/`sw`/`sd` accesses.
//!
//! Jolt lineage (the a16z/jolt guest set each program models):
//! `fibonacci`→fibonacci, `sha2_256`→sha2/sha2-chain, `sha3_keccak`→
//! keccak/sha3, `modinv`→modinv, `muldiv`→muldiv, `collatz`→collatz,
//! `memory_ops`→memory-ops, `regex`→the regexp benchmark's flavor,
//! `matrix_mul`/`sorting`→the standard zkVM compute benchmarks.

use crate::asm::{Assembler, AsmError};

/// Public statement of one benchmark instance.
#[derive(Clone, Debug)]
pub struct GuestProgram {
    pub name: &'static str,
    pub description: &'static str,
    /// The full byte image (code + inline data) loaded at 0.
    pub image: Vec<u8>,
    pub public_input: Vec<u8>,
    /// Result registers and their expected values.
    pub result_regs: Vec<(u8, u64)>,
    /// The data base (0 for inline data).
    pub data_base: u64,
}

// Register conventions.
const A0: u8 = 10;
const A1: u8 = 11;
const A2: u8 = 12;
const A3: u8 = 13;
const A4: u8 = 14;
const A5: u8 = 15;
const A6: u8 = 16;
const A7: u8 = 17;
const T0: u8 = 5;
const T1: u8 = 6;
const T2: u8 = 7;
const T3: u8 = 28;
const S0: u8 = 8;

/// Register x10 (a0) — the primary result register.
pub const RESULT_REG: u8 = A0;

// ---------------------------------------------------------------------------
// fibonacci(n)
// ---------------------------------------------------------------------------

pub fn reference_fibonacci(n: u64) -> u64 {
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        let t = a.wrapping_add(b);
        a = b;
        b = t;
    }
    a
}

pub fn fibonacci(n: u64) -> Result<GuestProgram, AsmError> {
    let expected = reference_fibonacci(n);
    let mut a = Assembler::new();
    a.li(A0, 0)?;
    a.li(A1, 1)?;
    a.li(T0, n as i64)?;
    let lp = a.label("lp");
    let done = a.label("done");
    a.bind(lp)?;
    a.beqz(T0, done.into())?;
    a.add(T2, A0, A1)?;
    a.mv(A0, A1)?;
    a.mv(A1, T2)?;
    a.addi(T0, T0, -1)?;
    a.j(lp.into())?;
    a.bind(done)?;
    a.ecall()?;
    Ok(GuestProgram {
        name: "fibonacci",
        description: "iterative wrapping-u64 fibonacci (jolt: fibonacci)",
        image: a.finish()?.code,
        public_input: vec![],
        result_regs: vec![(A0, expected)],
        data_base: 0,
    })
}

// ---------------------------------------------------------------------------
// collatz(n)
// ---------------------------------------------------------------------------

pub fn reference_collatz(n: u64) -> u64 {
    let mut total = 0u64;
    for i in 1..=n {
        let mut v = i;
        while v != 1 {
            v = if v % 2 == 0 { v / 2 } else { 3 * v + 1 };
            total += 1;
        }
    }
    total
}

pub fn collatz(n: u64) -> Result<GuestProgram, AsmError> {
    let expected = reference_collatz(n);
    let mut a = Assembler::new();
    a.li(A0, 0)?; // total
    a.li(A1, 1)?; // i
    a.li(T0, n as i64)?;
    let outer = a.label("outer");
    let inner = a.label("inner");
    let even = a.label("even");
    let next_i = a.label("next_i");
    let done = a.label("done");
    a.bind(outer)?;
    // if i > n: done (branch when n < i).
    a.blt(T0, A1, done.into())?;
    a.mv(T1, A1)?;
    a.bind(inner)?;
    a.li(T2, 1)?;
    a.beq(T1, T2, next_i.into())?;
    a.andi(A2, T1, 1)?;
    a.beqz(A2, even.into())?;
    a.li(T2, 3)?;
    a.mul(T1, T1, T2)?;
    a.addi(T1, T1, 1)?;
    a.addi(A0, A0, 1)?;
    a.j(inner.into())?;
    a.bind(even)?;
    a.srli(T1, T1, 1)?;
    a.addi(A0, A0, 1)?;
    a.j(inner.into())?;
    a.bind(next_i)?;
    a.addi(A1, A1, 1)?;
    a.j(outer.into())?;
    a.bind(done)?;
    a.ecall()?;
    Ok(GuestProgram {
        name: "collatz",
        description: "3n+1 iteration count over 1..=n (jolt: collatz)",
        image: a.finish()?.code,
        public_input: vec![],
        result_regs: vec![(A0, expected)],
        data_base: 0,
    })
}

// ---------------------------------------------------------------------------
// sorting(n): insertion sort of u64s from the input, XOR checksum.
// Input: [count u32][count u64s] at 0x1000 (word-aligned).
// ---------------------------------------------------------------------------

pub fn reference_sorting(values: &[u64]) -> u64 {
    let mut v = values.to_vec();
    for i in 1..v.len() {
        let mut j = i;
        while j > 0 && v[j] < v[j - 1] {
            v.swap(j, j - 1);
            j -= 1;
        }
    }
    let mut x = 0u64;
    for val in v {
        x ^= val;
    }
    x
}

pub fn sorting(values: &[u64]) -> Result<GuestProgram, AsmError> {
    let expected = reference_sorting(values);
    let n = values.len();
    let mut input = (n as u32).to_le_bytes().to_vec();
    for v in values {
        input.extend_from_slice(&v.to_le_bytes());
    }
    let mut a = Assembler::new();
    let data = a.label("data");
    a.la(A2, data)?;
    a.addi(A2, A2, 8)?; // &values[0] (the u64 count word precedes)
    a.ld(A3, A2, -8)?; // count
    a.li(A4, 1)?; // i
    let outer = a.label("outer");
    let inner = a.label("inner");
    let next_i = a.label("next_i");
    let done_outer = a.label("done_outer");
    let csum = a.label("csum");
    let cdone = a.label("cdone");
    let done = a.label("done");
    a.bind(outer)?;
    a.bge(A4, A3, done_outer.into())?;
    a.mv(A5, A4)?; // j = i
    a.bind(inner)?;
    a.beqz(A5, next_i.into())?;
    a.slli(T0, A5, 3)?;
    a.add(T1, A2, T0)?;
    a.ld(T2, T1, 0)?;
    a.ld(T3, T1, -8)?;
    a.bltu(T3, T2, next_i.into())?; // a[j-1] <u a[j]: inner done
    a.sd(T1, T3, 0)?;
    a.sd(T1, T2, -8)?;
    a.addi(A5, A5, -1)?;
    a.j(inner.into())?;
    a.bind(next_i)?;
    a.addi(A4, A4, 1)?;
    a.j(outer.into())?;
    a.bind(done_outer)?;
    // Checksum: XOR-fold the sorted array.
    a.li(A0, 0)?;
    a.li(A4, 0)?;
    a.bind(csum)?;
    a.bge(A4, A3, cdone.into())?;
    a.slli(T0, A4, 3)?;
    a.add(T1, A2, T0)?;
    a.ld(T2, T1, 0)?;
    a.xor(A0, A0, T2)?;
    a.addi(A4, A4, 1)?;
    a.j(csum.into())?;
    a.bind(cdone)?;
    a.ecall()?;
    a.bind(done)?; // unused exit label (kept for clarity)
    a.data();
    a.bind(data)?;
    a.word64(values.len() as u64)?;
    for v in values {
        a.word64(*v)?;
    }
    let prog = a.finish()?;
    Ok(GuestProgram {
        name: "sorting",
        description: "insertion sort of n u64s + XOR checksum (zkVM compute benchmark)",
        image: prog.code.clone(),
        public_input: vec![],
        result_regs: vec![(A0, expected)],
        data_base: prog.data_base,
    })
}

// ---------------------------------------------------------------------------
// memory_ops: strided stores + pointer chase over a 256-word region.
// ---------------------------------------------------------------------------

pub fn reference_memory_ops(seed: u64) -> u64 {
    let mut mem = vec![0u64; 256];
    for i in 0..128u64 {
        mem[(i * 7 % 256) as usize] = i.wrapping_mul(2654435761);
    }
    let mut p = (seed % 256) as usize;
    let mut x = 0u64;
    for _ in 0..64 {
        x ^= mem[p];
        p = (mem[p] % 256) as usize;
    }
    x
}

pub fn memory_ops(seed: u64) -> Result<GuestProgram, AsmError> {
    let expected = reference_memory_ops(seed);
    let mut a = Assembler::new();
    a.li(A2, 0x2000)?;
    a.li(A3, 0)?;
    let sloop = a.label("sloop");
    let sdone = a.label("sdone");
    let cloop = a.label("cloop");
    let cdone = a.label("cdone");
    a.bind(sloop)?;
    a.li(T0, 128)?;
    a.bge(A3, T0, sdone.into())?;
    a.li(T1, 7)?;
    a.mul(T2, A3, T1)?;
    a.li(T1, 256)?;
    a.remu(T2, T2, T1)?;
    a.slli(T2, T2, 3)?;
    a.add(T2, A2, T2)?;
    a.li(T3, 2654435761)?;
    a.mul(A4, A3, T3)?;
    a.sd(T2, A4, 0)?;
    a.addi(A3, A3, 1)?;
    a.j(sloop.into())?;
    a.bind(sdone)?;
    // Chase.
    a.li(A0, 0)?;
    a.li(A3, seed as i64)?;
    a.li(T1, 256)?;
    a.remu(A3, A3, T1)?;
    a.slli(A3, A3, 3)?;
    a.add(A3, A2, A3)?;
    a.li(A4, 0)?;
    a.bind(cloop)?;
    a.li(T0, 64)?;
    a.bge(A4, T0, cdone.into())?;
    a.ld(T2, A3, 0)?;
    a.xor(A0, A0, T2)?;
    a.li(T1, 256)?;
    a.remu(T2, T2, T1)?;
    a.slli(T2, T2, 3)?;
    a.add(A3, A2, T2)?;
    a.addi(A4, A4, 1)?;
    a.j(cloop.into())?;
    a.bind(cdone)?;
    a.ecall()?;
    Ok(GuestProgram {
        name: "memory_ops",
        description: "strided stores + pointer chase over 256 words (jolt: memory-ops)",
        image: a.finish()?.code,
        public_input: vec![],
        result_regs: vec![(A0, expected)],
        data_base: 0,
    })
}

// ---------------------------------------------------------------------------
// regex: table-driven DFA over the input matching (ab|ba)*c.
// Input layout: [count u32][count u32s, one byte each] in inline data.
// ---------------------------------------------------------------------------

pub fn reference_regex(bytes: &[u8]) -> u64 {
    let mut state = 0u8;
    let mut count = 0u64;
    for &b in bytes {
        state = match (state, b) {
            (0, b'a') => 1,
            (0, b'b') => 2,
            (1, b'b') => 0,
            (2, b'a') => 0,
            (0, b'c') => {
                count += 1;
                0
            }
            _ => 3,
        };
        if state == 3 {
            break;
        }
    }
    count
}

pub fn regex(bytes: &[u8]) -> Result<GuestProgram, AsmError> {
    let expected = reference_regex(bytes);
    let mut a = Assembler::new();
    let inp = a.label("input");
    let cls = a.label("class");
    let trans = a.label("trans");
    a.la(A2, inp)?;
    a.addi(A2, A2, 8)?; // &input[0] (the u64 count word precedes)
    a.ld(A3, A2, -8)?; // count
    a.li(A0, 0)?; // matches
    a.li(A1, 0)?; // state
    a.li(A4, 0)?; // i
    let lp = a.label("lp");
    let maybe = a.label("maybe");
    let accept = a.label("accept");
    let advance = a.label("advance");
    let done = a.label("done");
    a.bind(lp)?;
    a.bge(A4, A3, done.into())?;
    // Dead-state check.
    a.li(T0, 3)?;
    a.beq(A1, T0, done.into())?;
    // byte = input[i] (u32 word).
    a.slli(T0, A4, 2)?;
    a.add(T0, A2, T0)?;
    a.lw(T1, T0, 0)?;
    // cc = class[byte].
    a.la(T2, cls)?;
    a.slli(T3, T1, 2)?;
    a.add(T3, T2, T3)?;
    a.lw(T2, T3, 0)?; // cc in 0..4
    a.beqz(A1, maybe.into())?;
    a.j(advance.into())?;
    a.bind(maybe)?;
    a.li(T3, 2)?;
    a.beq(T2, T3, accept.into())?;
    a.j(advance.into())?;
    a.bind(accept)?;
    a.addi(A0, A0, 1)?;
    a.bind(advance)?;
    // state = trans[state*4 + cc].
    a.slli(T0, A1, 2)?;
    a.add(T0, T0, T2)?;
    a.slli(T0, T0, 2)?;
    a.la(T3, trans)?;
    a.add(T3, T3, T0)?;
    a.lw(A1, T3, 0)?;
    a.addi(A4, A4, 1)?;
    a.j(lp.into())?;
    a.bind(done)?;
    a.ecall()?;
    a.data();
    a.bind(inp)?;
    a.word64(bytes.len() as u64)?;
    for b in bytes {
        a.word(u32::from(*b))?;
    }
    a.bind(cls)?;
    for i in 0..256u32 {
        let c = match i as u8 {
            b'a' => 0,
            b'b' => 1,
            b'c' => 2,
            _ => 3,
        };
        a.word(c)?;
    }
    a.bind(trans)?;
    // states x classes: 0:[1,2,0,3] 1:[3,0,3,3] 2:[0,3,3,3] 3:[3,3,3,3]
    for row in [[1u32, 2, 0, 3], [3, 0, 3, 3], [0, 3, 3, 3], [3, 3, 3, 3]] {
        for v in row {
            a.word(v)?;
        }
    }
    let _ = &mut a;
    let prog = a.finish()?;
    Ok(GuestProgram {
        name: "regex",
        description: "DFA matcher for (ab|ba)*c over the input (jolt: regexp flavor)",
        image: prog.code.clone(),
        public_input: vec![],
        result_regs: vec![(A0, expected)],
        data_base: prog.data_base,
    })
}

// ---------------------------------------------------------------------------
// matrix_mul(n): n×n u64-wrapping matmul over three matrices in data
// (A, B inputs), checksum = XOR-fold of C. Result in a0.
// ---------------------------------------------------------------------------

pub fn reference_matrix_mul(n: usize, a: &[u64], b: &[u64]) -> u64 {
    let mut c = vec![0u64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0u64;
            for k in 0..n {
                acc = acc.wrapping_add(a[i * n + k].wrapping_mul(b[k * n + j]));
            }
            c[i * n + j] = acc;
        }
    }
    let mut x = 0u64;
    for v in c {
        x ^= v;
    }
    x
}

pub fn matrix_mul(n: usize, a: &[u64], b: &[u64]) -> Result<GuestProgram, AsmError> {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), n * n);
    let expected = reference_matrix_mul(n, a, b);
    let mut asm = Assembler::new();
    let la = asm.label("A");
    let lb = asm.label("B");
    let lc = asm.label("C");
    asm.la(A2, la)?;
    asm.la(A3, lb)?;
    asm.la(A4, lc)?;
    asm.li(A5, n as i64)?; // n
    // C = 0.
    asm.li(A6, 0)?;
    let zloop = asm.label("zloop");
    let zdone = asm.label("zdone");
    asm.bind(zloop)?;
    asm.li(T0, 0)?;
    let cells = (n * n) as i64;
    asm.li(T0, cells)?;
    asm.bge(A6, T0, zdone.into())?;
    asm.slli(T1, A6, 3)?;
    asm.add(T1, A4, T1)?;
    asm.sd(T1, 0, 0)?; // (x0 = 0)
    asm.addi(A6, A6, 1)?;
    asm.j(zloop.into())?;
    asm.bind(zdone)?;
    // Triple loop.
    asm.li(A6, 0)?; // i
    let ilp = asm.label("ilp");
    let jlp = asm.label("jlp");
    let klp = asm.label("klp");
    let idone = asm.label("idone");
    let jdone = asm.label("jdone");
    let kdone = asm.label("kdone");
    let csum = asm.label("csum");
    let cdone = asm.label("cdone");
    let done = asm.label("done");
    asm.bind(ilp)?;
    asm.bge(A6, A5, idone.into())?;
    asm.li(A7, 0)?; // j
    asm.bind(jlp)?;
    asm.bge(A7, A5, jdone.into())?;
    asm.li(S0, 0)?; // k
    asm.bind(klp)?;
    asm.bge(S0, A5, kdone.into())?;
    // t0 = A[i][k]; t1 = B[k][j]; t2 = C[i][j] += t0*t1.
    asm.mul(T0, A6, A5)?;
    asm.add(T0, T0, S0)?;
    asm.slli(T0, T0, 3)?;
    asm.add(T0, A2, T0)?;
    asm.ld(T0, T0, 0)?;
    asm.mul(T1, S0, A5)?;
    asm.add(T1, T1, A7)?;
    asm.slli(T1, T1, 3)?;
    asm.add(T1, A3, T1)?;
    asm.ld(T1, T1, 0)?;
    asm.mul(T0, T0, T1)?;
    asm.mul(T1, A6, A5)?;
    asm.add(T1, T1, A7)?;
    asm.slli(T1, T1, 3)?;
    asm.add(T1, A4, T1)?;
    asm.ld(T2, T1, 0)?;
    asm.add(T2, T2, T0)?;
    asm.sd(T1, T2, 0)?;
    asm.addi(S0, S0, 1)?;
    asm.j(klp.into())?;
    asm.bind(kdone)?;
    asm.addi(A7, A7, 1)?;
    asm.j(jlp.into())?;
    asm.bind(jdone)?;
    asm.addi(A6, A6, 1)?;
    asm.j(ilp.into())?;
    asm.bind(idone)?;
    // Checksum.
    asm.li(A6, 0)?;
    asm.li(A0, 0)?;
    asm.bind(csum)?;
    asm.li(T0, cells)?;
    asm.bge(A6, T0, cdone.into())?;
    asm.slli(T1, A6, 3)?;
    asm.add(T1, A4, T1)?;
    asm.ld(T2, T1, 0)?;
    asm.xor(A0, A0, T2)?;
    asm.addi(A6, A6, 1)?;
    asm.j(csum.into())?;
    asm.bind(cdone)?;
    asm.ecall()?;
    asm.bind(done)?;
    asm.data();
    asm.bind(la)?;
    for v in a {
        asm.word64(*v)?;
    }
    asm.bind(lb)?;
    for v in b {
        asm.word64(*v)?;
    }
    asm.bind(lc)?;
    for _ in 0..n * n {
        asm.word64(0)?;
    }
    let prog = asm.finish()?;
    Ok(GuestProgram {
        name: "matrix_mul",
        description: "n×n wrapping-u64 matmul + XOR checksum (zkVM compute benchmark)",
        image: prog.code.clone(),
        public_input: vec![],
        result_regs: vec![(A0, expected)],
        data_base: prog.data_base,
    })
}

// ---------------------------------------------------------------------------
// modinv(count): batched binary extended Euclid inverses mod p = 2^61−1
// over input values; checksum = XOR-fold. Input in data.
// ---------------------------------------------------------------------------

pub const MODINV_P: u64 = (1 << 61) - 1;

pub fn reference_modinv(values: &[u64]) -> u64 {
    // Fermat: a^(p-2) mod p.
    let mut x = 0u64;
    for &v in values {
        let a = v % MODINV_P;
        let e = MODINV_P - 2;
        let mut acc = 1u64;
        let mut base = a;
        let mut exp = e;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = acc.wrapping_mul(base) % MODINV_P;
            }
            base = base.wrapping_mul(base) % MODINV_P;
            exp >>= 1;
        }
        x ^= acc;
    }
    x
}

pub fn modinv(values: &[u64]) -> Result<GuestProgram, AsmError> {
    let expected = reference_modinv(values);
    let mut asm = Assembler::new();
    let data = asm.label("data");
    asm.la(A2, data)?;
    asm.addi(A2, A2, 8)?; // &values[0] (the u64 count word precedes)
    asm.ld(A3, A2, -8)?; // count
    asm.li(T2, MODINV_P as i64)?; // p
    asm.li(A0, 0)?; // checksum
    asm.li(A4, 0)?; // i
    let lp = asm.label("lp");
    let done = asm.label("done");
    asm.bind(lp)?;
    asm.bge(A4, A3, done.into())?;
    // a = values[i] mod p.
    asm.slli(T0, A4, 3)?;
    asm.add(T0, A2, T0)?;
    asm.ld(T1, T0, 0)?;
    asm.remu(T1, T1, T2)?;
    // acc = 1; base = a; exp = p-2.
    asm.li(A5, 1)?; // acc
    asm.mv(A6, T1)?; // base
    asm.li(A7, MODINV_P as i64 - 2)?; // exp
    let ilp = asm.label("ilp");
    let idone = asm.label("idone");
    let mul_step = asm.label("mul_step");
    let square = asm.label("square");
    asm.bind(ilp)?;
    asm.beqz(A7, idone.into())?;
    asm.andi(T0, A7, 1)?;
    asm.bnez(T0, mul_step.into())?;
    asm.j(square.into())?;
    asm.bind(mul_step)?;
    asm.mul(A5, A5, A6)?;
    asm.remu(A5, A5, T2)?;
    asm.bind(square)?;
    asm.mul(A6, A6, A6)?;
    asm.remu(A6, A6, T2)?;
    asm.srli(A7, A7, 1)?;
    asm.j(ilp.into())?;
    asm.bind(idone)?;
    // checksum ^= acc; i++.
    asm.xor(A0, A0, A5)?;
    asm.addi(A4, A4, 1)?;
    asm.j(lp.into())?;
    asm.bind(done)?;
    asm.ecall()?;
    asm.data();
    asm.bind(data)?;
    asm.word64(values.len() as u64)?;
    for v in values {
        asm.word64(*v)?;
    }
    let prog = asm.finish()?;
    Ok(GuestProgram {
        name: "modinv",
        description: "batched Fermat inverses mod 2^61-1 (jolt: modinv)",
        image: prog.code.clone(),
        public_input: vec![],
        result_regs: vec![(A0, expected)],
        data_base: prog.data_base,
    })
}

// ---------------------------------------------------------------------------
// muldiv(count): batched mul + divu + remu over input pairs; checksum in
// a0. (jolt: muldiv — the 64-bit flavor.)
// ---------------------------------------------------------------------------

pub fn reference_muldiv(values: &[(u64, u64, u64)]) -> u64 {
    let mut x = 0u64;
    for (a, b, c) in values {
        let lo = a.wrapping_mul(*b);
        let q = if *c == 0 { 0 } else { lo / c };
        let r = if *c == 0 { 0 } else { lo % c };
        x ^= lo ^ q.rotate_left(17) ^ r.rotate_left(33);
    }
    x
}

pub fn muldiv(values: &[(u64, u64, u64)]) -> Result<GuestProgram, AsmError> {
    let expected = reference_muldiv(values);
    let mut asm = Assembler::new();
    let data = asm.label("data");
    asm.la(A2, data)?;
    asm.addi(A2, A2, 8)?; // &values[0] (the u64 count word precedes)
    asm.ld(A3, A2, -8)?; // count
    asm.li(A0, 0)?;
    asm.li(A4, 0)?; // i
    let lp = asm.label("lp");
    let c_zero = asm.label("c_zero");
    let after = asm.label("after");
    let done = asm.label("done");
    asm.bind(lp)?;
    asm.bge(A4, A3, done.into())?;
    // a, b, c = values[i].0/1/2 (24 bytes per item).
    asm.slli(T0, A4, 3)?;
    asm.li(T1, 3)?;
    asm.mul(T0, T0, T1)?;
    asm.add(T0, A2, T0)?;
    asm.ld(T1, T0, 0)?; // a
    asm.ld(T2, T0, 8)?; // b
    asm.ld(T3, T0, 16)?; // c
    asm.mul(T1, T1, T2)?; // lo = a*b
    // if c == 0: q = r = 0.
    asm.beqz(T3, c_zero.into())?;
    asm.divu(T2, T1, T3)?; // q
    asm.remu(T3, T1, T3)?; // r — careful: c is in T3; remu destroys it.
    asm.j(after.into())?;
    asm.bind(c_zero)?;
    asm.li(T2, 0)?;
    asm.li(T3, 0)?;
    asm.bind(after)?;
    // x ^= lo ^ rotl(q, 17) ^ rotl(r, 33).
    asm.xor(A0, A0, T1)?;
    asm.slli(T0, T2, 17)?;
    asm.srli(T2, T2, 47)?;
    asm.or(T0, T0, T2)?; // rotl(q, 17)
    asm.xor(A0, A0, T0)?;
    // r is in T3 — reload from memory? T3 was overwritten by remu's
    // result... rotl(r, 33):
    asm.slli(T0, T3, 33)?;
    asm.srli(T3, T3, 31)?;
    asm.or(T0, T0, T3)?;
    asm.xor(A0, A0, T0)?;
    asm.addi(A4, A4, 1)?;
    asm.j(lp.into())?;
    asm.bind(done)?;
    asm.ecall()?;
    asm.data();
    asm.bind(data)?;
    asm.word64(values.len() as u64)?;
    for (a, b, c) in values {
        asm.word64(*a)?;
        asm.word64(*b)?;
        asm.word64(*c)?;
    }
    let prog = asm.finish()?;
    Ok(GuestProgram {
        name: "muldiv",
        description: "batched mul + divu + remu with rotations (jolt: muldiv, 64-bit flavor)",
        image: prog.code.clone(),
        public_input: vec![],
        result_regs: vec![(A0, expected)],
        data_base: prog.data_base,
    })
}

// ---------------------------------------------------------------------------
// The suites.
// ---------------------------------------------------------------------------

/// Pseudo-random helper for reproducible inputs.
pub fn lcg(seed: u64) -> impl Iterator<Item = u64> {
    let mut s = seed;
    core::iter::repeat_with(move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s
    })
}

/// The default (small) suite: every program at a size that keeps the
/// end-to-end memory-argument proof tractable (T ≤ ~2^12 cycles).
pub fn suite() -> Result<Vec<GuestProgram>, AsmError> {
    let mut out = Vec::new();
    out.push(fibonacci(30)?);
    out.push(collatz(1000)?);
    let vals: Vec<u64> = lcg(0xC0FFEE).take(64).collect();
    out.push(sorting(&vals)?);
    out.push(memory_ops(12345)?);
    let bytes: Vec<u8> = b"ababcbaabbaccababacabbacabbaabcabc".to_vec();
    out.push(regex(&bytes)?);
    let n = 8usize;
    let a: Vec<u64> = lcg(0xACE).take(n * n).collect();
    let b: Vec<u64> = lcg(0xBEEF).take(n * n).collect();
    out.push(matrix_mul(n, &a, &b)?);
    let invs: Vec<u64> = lcg(0x5EED).take(16).collect();
    out.push(modinv(&invs)?);
    let md: Vec<(u64, u64, u64)> = lcg(0xD1CE)
        .take(48)
        .collect::<Vec<_>>()
        .chunks(3)
        .map(|c| (c[0], c[1], c[2] | 1))
        .collect();
    out.push(muldiv(&md)?);
    Ok(out)
}

/// The large suite (heavier sizes).
pub fn suite_large() -> Result<Vec<GuestProgram>, AsmError> {
    let mut out = Vec::new();
    out.push(fibonacci(90)?);
    out.push(collatz(4000)?);
    let vals: Vec<u64> = lcg(0xF00D).take(128).collect();
    out.push(sorting(&vals)?);
    out.push(memory_ops(987654321)?);
    let n = 12usize;
    let a: Vec<u64> = lcg(0x1234).take(n * n).collect();
    let b: Vec<u64> = lcg(0x5678).take(n * n).collect();
    out.push(matrix_mul(n, &a, &b)?);
    let invs: Vec<u64> = lcg(0x9ABC).take(64).collect();
    out.push(modinv(&invs)?);
    let md: Vec<(u64, u64, u64)> = lcg(0xDEF0)
        .take(96)
        .collect::<Vec<_>>()
        .chunks(3)
        .map(|c| (c[0], c[1], c[2] | 1))
        .collect();
    out.push(muldiv(&md)?);
    Ok(out)
}
