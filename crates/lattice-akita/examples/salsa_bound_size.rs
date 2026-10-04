//! **The D4 binding-closure + r-column split's size evidence**: the
//! open SALSAA response (polylog, the binding the documented gap) vs
//! the BOUND response (the width-collapse chain composition — the
//! authenticated opening) vs the Clear mode (the opened witness), and
//! the SPLIT response beyond the per-commitment Lemma-4 cap
//! (2,048 values at ring dim 16 / Q_32 — the prior "~1,200" prose note
//! was this same cap, margin-rounded).

use lattice_akita::pcs::{AkitaPcs, GroupedOpening};
use lattice_akita::salsa_binding::{byte_capacity, SalsaBoundResponse, SalsaSplitResponse};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::{Modulus32, RingConfig};

fn bound_bytes(p: &SalsaBoundResponse) -> usize {
    let carrier: usize = p.sumcheck.rounds.iter().map(|r| r.len() * 8).sum();
    let d1: usize = p.chain.sumcheck.rounds.iter().map(|r| r.len() * 8).sum();
    let fold: usize = p
        .fold
        .stages
        .iter()
        .map(|s| {
            s.p_images.len()
                + s.garbage.len()
                + s.t_inner.len()
                + s.u_parts.len()
                + s.g_func.len()
                + s.response.hist.len()
                + s.response.payload.len()
                + s.response.raw.len()
        })
        .sum();
    carrier + d1 + fold + 8 * 4 + 32
}

fn split_bytes(p: &SalsaSplitResponse) -> usize {
    let carrier: usize = p.sumcheck.rounds.iter().map(|r| r.len() * 8).sum();
    let cols: usize = p
        .columns
        .iter()
        .map(|c| {
            let d1: usize = c.d1.sumcheck.rounds.iter().map(|r| r.len() * 8).sum();
            let fold: usize = c
                .fold
                .stages
                .iter()
                .map(|s| {
                    s.p_images.len()
                        + s.garbage.len()
                        + s.t_inner.len()
                        + s.u_parts.len()
                        + s.g_func.len()
                        + s.response.hist.len()
                        + s.response.payload.len()
                        + s.response.raw.len()
                })
                .sum();
            c.commitment.len() + d1 + fold + 8 * 2
        })
        .sum();
    carrier + cols + 8 * 3
}

fn main() {
    let ring = RingConfig::new(Modulus32::Q_32, 4).ok().unwrap();
    let cap = byte_capacity(&ring);
    println!(
        "== D4: the binding closure + the r-column split (ring dim {}, Q_32) ==",
        ring.n()
    );
    println!("per-commitment Lemma-4 capacity: {cap} values (the r-column split beyond it)");
    println!();
    println!("| values | Clear (B) | Salsa open (B) | Salsa BOUND (B) | bound/open | bound prove (ms) | bound verify (ms) |");
    println!("|---|---|---|---|---|---|---|");
    for log_n in [6usize, 8, 10] {
        let n = 1usize << log_n;
        let m_slots = (n * 8).div_ceil(16).next_power_of_two().max(1);
        let params = lattice_commitment::ajtai::AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: m_slots,
            norm_bound: 1 << 20,
        };
        let pk = lattice_commitment::ajtai::AjtaiPublicKey::from_seed(params, [91u8; 32])
            .ok()
            .unwrap();
        let pcs = AkitaPcs { pk: pk.clone() };
        let evals: Vec<Goldilocks> = (0..n)
            .map(|i| Goldilocks::from_u64((i as u64 * 2654435761) % 65_536))
            .collect();
        let f = DenseMle::new(evals).ok().unwrap();
        let claims: Vec<GroupedOpening> = (0..4)
            .map(|c| {
                let point: Vec<Goldilocks> = (0..log_n)
                    .map(|j| Goldilocks::from_u64(0x1000_0000 + (c * 16 + j) as u64))
                    .collect();
                let value = f.evaluate(&point).ok().unwrap();
                GroupedOpening { point, value }
            })
            .collect();
        // The Clear mode (the witness-revealing baseline).
        let mut t1 = Transcript::new_default(b"salsa-size");
        let clear = match pcs.prove_grouped(&f, &claims, &mut t1) {
            Ok(c) => c,
            Err(e) => {
                println!("| 2^{log_n} | prove failed: {e:?} | — | — | — |");
                continue;
            }
        };
        let digits: usize = clear.norm_proof.digits.iter().map(|d| d.len() * 8).sum();
        let clear_bytes = clear.opened_witness.len() * 16 * 4
            + digits
            + clear
                .sumcheck
                .rounds
                .iter()
                .map(|r| r.len() * 8)
                .sum::<usize>();
        // The open SALSAA response.
        let mut t2 = Transcript::new_default(b"salsa-size");
        let (salsa, _packed) = pcs.prove_grouped_salsa(&f, &claims, &mut t2).ok().unwrap();
        let salsa_bytes = {
            let sc = salsa
                .sumcheck
                .rounds
                .iter()
                .map(|r| r.len() * 8)
                .sum::<usize>();
            let d1 = salsa
                .chain
                .sumcheck
                .rounds
                .iter()
                .map(|r| r.len() * 8)
                .sum::<usize>();
            let func = salsa
                .functional
                .rounds
                .iter()
                .map(|r| r.len() * 8)
                .sum::<usize>();
            sc + d1 + func + 8 * 4 + 32
        };
        // The BOUND response (the closure) + the timings.
        let mut t3 = Transcript::new_default(b"salsa-size");
        let t0 = std::time::Instant::now();
        let bound = pcs
            .prove_grouped_salsa_bound(&f, &claims, &mut t3)
            .ok()
            .unwrap();
        let prove_ms = t0.elapsed().as_millis();
        let b_bytes = bound_bytes(&bound);
        let mut vt = Transcript::new_default(b"salsa-size");
        let com = pcs.commit_bytes(&f).ok().unwrap();
        let t1 = std::time::Instant::now();
        assert!(
            pcs.verify_grouped_salsa_bound(&com, &claims, &bound, &mut vt)
                .is_ok(),
            "the bound response must verify at 2^{log_n}"
        );
        let verify_ms = t1.elapsed().as_millis();
        println!(
            "| 2^{log_n} | {clear_bytes} | {salsa_bytes} | {b_bytes} | {:.1}x | {prove_ms} | {verify_ms} |",
            b_bytes as f64 / salsa_bytes.max(1) as f64
        );
    }
    println!();
    println!("== Beyond the cap: the r-column split ==");
    println!("| values | columns | split response (B) | vs Clear | prove/verify |");
    println!("|---|---|---|---|---|");
    for log_n in [12usize, 13] {
        let n = 1usize << log_n;
        // The split manages its own per-column keys; the pcs shape only
        // fixes the ring.
        let params = lattice_commitment::ajtai::AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: 4,
            norm_bound: 1 << 20,
        };
        let pk = lattice_commitment::ajtai::AjtaiPublicKey::from_seed(params, [91u8; 32])
            .ok()
            .unwrap();
        let pcs = AkitaPcs { pk };
        let evals: Vec<Goldilocks> = (0..n)
            .map(|i| Goldilocks::from_u64((i as u64 * 2654435761) % 65_536))
            .collect();
        let f = DenseMle::new(evals).ok().unwrap();
        let claims: Vec<GroupedOpening> = (0..4)
            .map(|c| {
                let point: Vec<Goldilocks> = (0..log_n)
                    .map(|j| Goldilocks::from_u64(0x1000_0000 + (c * 16 + j) as u64))
                    .collect();
                let value = f.evaluate(&point).ok().unwrap();
                GroupedOpening { point, value }
            })
            .collect();
        // The Clear baseline at this scale (the opened witness).
        let m_slots = (n * 8).div_ceil(16).next_power_of_two().max(1);
        let clear_params = lattice_commitment::ajtai::AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: m_slots,
            norm_bound: 1 << 20,
        };
        let clear_pk =
            lattice_commitment::ajtai::AjtaiPublicKey::from_seed(clear_params, [91u8; 32])
                .ok()
                .unwrap();
        let clear_pcs = AkitaPcs { pk: clear_pk };
        let mut t1 = Transcript::new_default(b"salsa-size");
        let clear_bytes = match clear_pcs.prove_grouped(&f, &claims, &mut t1) {
            Ok(c) => {
                let digits: usize = c.norm_proof.digits.iter().map(|d| d.len() * 8).sum();
                c.opened_witness.len() * 16 * 4
                    + digits
                    + c.sumcheck.rounds.iter().map(|r| r.len() * 8).sum::<usize>()
            }
            Err(_) => 0,
        };
        // The split + the timings.
        let mut t2 = Transcript::new_default(b"salsa-size");
        let t0 = std::time::Instant::now();
        let split = pcs
            .prove_grouped_salsa_split(&f, &claims, &mut t2)
            .ok()
            .unwrap();
        let prove_ms = t0.elapsed().as_millis();
        let s_bytes = split_bytes(&split);
        let mut vt = Transcript::new_default(b"salsa-size");
        let t1 = std::time::Instant::now();
        assert!(
            pcs.verify_grouped_salsa_split(log_n, &claims, &split, &mut vt)
                .is_ok(),
            "the split response must verify at 2^{log_n}"
        );
        let verify_ms = t1.elapsed().as_millis();
        let ratio = if clear_bytes > 0 {
            format!("{:.1}x", clear_bytes as f64 / s_bytes.max(1) as f64)
        } else {
            "—".to_string()
        };
        println!(
            "| 2^{log_n} | {} | {s_bytes} | {ratio} | {prove_ms} ms / {verify_ms} ms |",
            split.columns.len()
        );
    }
}
