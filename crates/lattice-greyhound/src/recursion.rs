//! The LaBRADOR recursion driver (§5.3 / the reference's `pack.c` composite):
//! iterate `prove_level` until the level no longer shrinks, then one tail level
//! (§5.6) whose openings are transmitted directly; the final witness is sent in
//! the clear and verified by [`verify_tail`].
//!
//! The proof-size model (§5.7 + the reference's entropy accounting):
//! * a non-tail level transmits `(u1len + u2len + LIFTS)·N·LOGQ` bits of
//!   commitments and lift polynomials;
//! * the JL vector p: `(log2‖p‖₂ − 4 + 2.05)·256` bits (256 near-Gaussian
//!   entries at σ = ‖p‖₂/16, entropy log2(σ) + log2(√(2πe)) ≈ log2 σ + 2.05);
//! * the final witness: `Σ_i (log2(normsq_i/(N·n_i))/2 + 2.05)·N·n_i` bits;
//! * challenges expand from 128-bit seeds.

use crate::protocol::{prove_level, reduce_level, verify_tail, LevelProof};
use crate::relation::{PrincipalStatement, PrincipalWitness};
use crate::ring::{LOGQ, N};
use crate::sis::{ComKey, LIFTS};

/// The full recursively-composed proof.
#[derive(Clone, Debug)]
pub struct LabradorProof {
    pub levels: Vec<LevelProof>,
    pub tail: LevelProof,
    pub final_witness: PrincipalWitness,
}

/// Prove a principal statement with the full recursion.
pub fn prove(
    stmt: &PrincipalStatement,
    wit: &PrincipalWitness,
    key: &ComKey,
) -> Result<LabradorProof, String> {
    let mut levels: Vec<LevelProof> = Vec::new();
    let mut cur_stmt = stmt.clone();
    let mut cur_wit = wit.clone();

    // iterate until the level no longer shrinks (the reference's rule: stop
    // when the next proof + witness is no smaller than the current one)
    let max_levels = 16;
    loop {
        if levels.len() >= max_levels {
            return Err("recursion did not terminate".into());
        }
        // predicted size of the current witness (entropy bits)
        let cur_size = witness_size_bits(&cur_wit);
        let (proof, next_stmt, next_wit) = prove_level(&cur_stmt, &cur_wit, key, false).map_err(|e| { eprintln!("[prove] level {} FAILED: {e}", levels.len()); e })?;
        let next_size = witness_size_bits(next_wit.as_ref().unwrap());
        levels.push(proof);
        if next_size >= cur_size {
            // no shrink — the last level should have been the tail; rewind
            // (the reference checks before accepting the level)
            let last = levels.pop().unwrap();
            // re-run as tail on the current statement
            let (tail, _, final_wit) = prove_level(&cur_stmt, &cur_wit, key, true)?;
            let _ = last;
            return Ok(LabradorProof {
                levels,
                tail,
                final_witness: final_wit.unwrap(),
            });
        }
        cur_stmt = next_stmt.unwrap();
        cur_wit = next_wit.unwrap();
        // try the tail now: if it would shrink further, keep recursing
        let (tail_test, _, tail_wit) = prove_level(&cur_stmt, &cur_wit, key, true)?;
        let tail_size = witness_size_bits(tail_wit.as_ref().unwrap()) + level_size_bits(&tail_test);
        if tail_size < witness_size_bits(&cur_wit) {
            return Ok(LabradorProof {
                levels,
                tail: tail_test,
                final_witness: tail_wit.unwrap(),
            });
        }
    }
}

/// Verify a full proof against the original statement.
pub fn verify(
    stmt: &PrincipalStatement,
    proof: &LabradorProof,
    key: &ComKey,
) -> Result<(), String> {
    let mut cur = stmt.clone();
    for (i, lp) in proof.levels.iter().enumerate() {
        if lp.tail() {
            return Err(format!("level {i} is a tail inside the chain"));
        }
        cur = reduce_level(&cur, lp, key)?;
    }
    if !proof.tail.tail() {
        return Err("the last level is not a tail".into());
    }
    verify_tail(&cur, &proof.tail, &proof.final_witness, key)
}

/// Entropy bits of a witness (the reference's `print_witness_pp`).
pub fn witness_size_bits(wit: &PrincipalWitness) -> u64 {
    wit.s
        .iter()
        .map(|v| {
            let normsq: u64 = v.iter().map(|p| p.normsq()).sum();
            let n = v.len();
            if n == 0 || normsq == 0 {
                0u64
            } else {
                let ent = ((normsq as f64 / (N * n) as f64).log2() / 2.0 + 2.05).max(1.0);
                ((N * n) as f64 * ent) as u64
            }
        })
        .sum()
}

/// Bits of one level's transmitted proof (the reference's `print_proof_pp`).
pub fn level_size_bits(lp: &LevelProof) -> u64 {
    let jl_bits: u64 = {
        let psq: u64 = lp.p.iter().map(|&x| (x * x) as u64).sum();
        if psq == 0 {
            0u64
        } else {
            (((psq as f64).sqrt().log2() - 4.0 + 2.05).max(1.0) * 256.0) as u64
        }
    };
    ((lp.u1.len() + lp.u2.len() + LIFTS) * N * LOGQ) as u64 + jl_bits
        + 128 // the challenge seeds
}

/// Total proof size in bytes (the analytic model — matches the serialized
/// pieces the verifier needs).
pub fn proof_size_bytes(proof: &LabradorProof) -> u64 {
    let levels: u64 = proof.levels.iter().map(level_size_bits).sum();
    let tail = level_size_bits(&proof.tail);
    let witness = witness_size_bits(&proof.final_witness);
    (levels + tail + witness).div_ceil(8)
}

/// A measured (serialized) size — the honest byte count of the wire format:
/// ring elements bit-packed at LOGQ bits each, the JL vector at
/// 8·⌈log2 bound⌉ bits per entry hmm — see `serialize` for the exact packing.
pub fn serialize_size_bytes(proof: &LabradorProof) -> u64 {
    let mut bits: u64 = 0;
    for lp in proof.levels.iter().chain(std::iter::once(&proof.tail)) {
        bits += (lp.u1.len() + lp.u2.len() + lp.bb.len()) as u64 * (N * LOGQ) as u64;
        // p: 256 values, each at ~17 bits (the bound 4√normsq < 2^17 in the
        // sound regime) — use the announced bound
        bits += 256 * 18;
        bits += 128; // challenge seed
    }
    // the final witness: entropy-coded digits
    bits += witness_size_bits(&proof.final_witness);
    bits.div_ceil(8)
}

/// The per-level table (the paper's Table 3 shape).
pub fn level_table(proof: &LabradorProof) -> Vec<LevelRow> {
    proof
        .levels
        .iter()
        .chain(std::iter::once(&proof.tail))
        .map(|lp| LevelRow {
            n: lp.nn,
            r: lp.r,
            kappa: lp.cpp.kappa,
            kappa1: lp.cpp.kappa1,
            f: lp.cpp.f,
            fu: lp.cpp.fu,
            fg: lp.cpp.fg,
            bits: level_size_bits(lp),
            tail: lp.tail(),
        })
        .collect()
}

/// One row of the level table.
#[derive(Clone, Copy, Debug)]
pub struct LevelRow {
    pub n: usize,
    pub r: usize,
    pub kappa: usize,
    pub kappa1: usize,
    pub f: usize,
    pub fu: usize,
    pub fg: usize,
    pub bits: u64,
    pub tail: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::protocol::prove_level;
    use crate::ring::Poly;

    fn toy_statement() -> (PrincipalStatement, PrincipalWitness) {
        // 3 vectors of rank 32, one linear constraint + one ct constraint.
        // LaBRADOR witnesses are SHORT — small ternary coefficients.
        let mk = |seed: u64| -> Vec<Poly> {
            (0..32)
                .map(|i| {
                    let mut p = [0i64; 64];
                    for (j, c) in p.iter_mut().enumerate() {
                        *c = ((i * 37 + j * 17 + seed as usize * 13) % 7) as i64 - 3;
                    }
                    Poly(p)
                })
                .collect()
        };
        let s0 = mk(1);
        let s1 = mk(2);
        let s2 = mk(3);
        let phi = mk(4);
        let phi2 = mk(5);
        let b = crate::ring::sprod(&phi, &s0).add(&crate::ring::sprod(&phi2, &s1));
        // an F' constraint: only the ct of b matters; scramble the rest
        let mut b_ct = crate::ring::sprod(&phi2, &s2);
        if b_ct.0.len() > 1 {
            b_ct.0[1] = 12345; // arbitrary higher coefficient — still valid F'
        }
        let st = PrincipalStatement::new(
            vec![
                crate::relation::VectorSpec::plain(32),
                crate::relation::VectorSpec::plain(32),
                crate::relation::VectorSpec::plain(32),
            ],
            vec![crate::relation::DotCnst::with_b(
                vec![
                    crate::relation::Term { idx: 0, off: 0, phi },
                    crate::relation::Term { idx: 1, off: 0, phi: phi2.clone() },
                ],
                b,
            )],
            vec![crate::relation::DotCnst {
                terms: vec![crate::relation::Term { idx: 2, off: 0, phi: phi2 }],
                a: vec![],
                b: Some(b_ct),
                ct_only: true,
            }],
            u64::MAX / 4,
        );
        (st, PrincipalWitness::new(vec![s0, s1, s2]))
    }

    #[test]
    fn full_recursion_roundtrip() {
        let (st, wit) = toy_statement();
        let key = ComKey::expand(1 << 18, &[7u8; 32]);
        let proof = prove(&st, &wit, &key).unwrap();
        assert!(verify(&st, &proof, &key).is_ok(), "verification failed");
        // tamper the final witness
        let mut bad = proof.clone();
        if let Some(v) = bad.final_witness.s.first_mut() {
            if let Some(p) = v.first_mut() {
                *p = p.add(&Poly::constant(1));
            }
        }
        assert!(verify(&st, &bad, &key).is_err(), "tampered witness accepted");
    }

    #[test]
    fn single_level_roundtrip() {
        let (st, wit) = toy_statement();
        let key = ComKey::expand(1 << 18, &[7u8; 32]);
        let t0 = std::time::Instant::now();
        let (proof, target, target_wit) = prove_level(&st, &wit, &key, false).unwrap();
        eprintln!("[diag] prove_level took {:?}", t0.elapsed());
        let t1 = std::time::Instant::now();
        let rebuilt = reduce_level(&st, &proof, &key).unwrap();
        eprintln!("[diag] reduce_level took {:?}", t1.elapsed());
        // the target statement's constraints hold of the target witness
        rebuilt.check_all(&target_wit.as_ref().unwrap().s).unwrap();
        // the digest binds: prover-constructed and verifier-reconstructed agree
        assert_eq!(rebuilt.digest, target.as_ref().unwrap().digest, "reduce must reproduce the prover's target statement");
        let _ = proof;
    }

    #[test]
    fn sizes_are_sane() {
        let (st, wit) = toy_statement();
        let key = ComKey::expand(1 << 18, &[7u8; 32]);
        let proof = prove(&st, &wit, &key).unwrap();
        let bytes = proof_size_bytes(&proof);
        assert!(bytes > 0 && bytes < 1 << 20, "size {bytes} insane");
        let table = level_table(&proof);
        assert!(!table.is_empty());
        assert!(table.last().unwrap().tail);
    }
}
