//! **The D4 response-layer swap's size evidence**: the grouped SALSAA
//! response (polylog, no disclosure) vs the Clear grouped opening
//! (the opened witness + the digit-devealing NormProof) at the
//! pipeline's pk shape (ring dim 16).
//!
//! The honest regime note: the byte-packed D1 chain's Lemma-4 gate
//! (`m·n·B² < q/2` at `B = 255`) caps the per-commitment capacity at
//! ~1,200 values (the r-column split — the compact mode's discipline —
//! is the scaling route beyond it).

use lattice_akita::pcs::{AkitaPcs, GroupedOpening};
use lattice_akita::salsa_response::SalsaGroupedResponse;
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::{Modulus32, RingConfig};

fn wire_bytes(p: &SalsaGroupedResponse) -> usize {
    let sc = p.sumcheck.rounds.iter().map(|r| r.len() * 8).sum::<usize>();
    let d1 = p.chain.sumcheck.rounds.iter().map(|r| r.len() * 8).sum::<usize>();
    let func = p.functional.rounds.iter().map(|r| r.len() * 8).sum::<usize>();
    sc + d1 + func + 8 * 4 + 32
}

fn main() {
    println!("| column values | Clear response (B) | SALSAA response (B) | reduction |");
    println!("|---|---|---|---|");
    for log_n in [6usize, 8, 10] {
        let n = 1usize << log_n;
        // The byte-packed witness length (one byte per coefficient):
        // n·8 bytes / 16 coefficients per element.
        let m_slots = (n * 8).div_ceil(16).next_power_of_two().max(1);
        let ring = RingConfig::new(Modulus32::Q_32, 4).ok().unwrap();
        let params = lattice_commitment::ajtai::AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: m_slots,
            norm_bound: 1 << 20,
        };
        let pk =
            lattice_commitment::ajtai::AjtaiPublicKey::from_seed(params, [91u8; 32]).ok().unwrap();
        let pcs = AkitaPcs { pk: pk.clone() };
        // A trace-like column with bounded values (the Clear mode's
        // NormProof regime; full-range values exceed ANY sound bound —
        // the SALSAA paper's motivating regime).
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
                println!("| 2^{log_n} | prove failed: {e:?} | — | — |");
                continue;
            }
        };
        let digits: usize = clear.norm_proof.digits.iter().map(|d| d.len() * 8).sum();
        let clear_bytes = clear.opened_witness.len() * 16 * 4 // the opened packed witness
            + digits
            + clear.sumcheck.rounds.iter().map(|r| r.len() * 8).sum::<usize>();
        // The SALSAA response (D4).
        let mut t2 = Transcript::new_default(b"salsa-size");
        let (salsa, _packed) = pcs.prove_grouped_salsa(&f, &claims, &mut t2).ok().unwrap();
        let salsa_bytes = wire_bytes(&salsa);
        let mut vt = Transcript::new_default(b"salsa-size");
        let com = pcs.commit_bytes(&f).ok().unwrap();
        assert!(pcs.verify_grouped_salsa(&com, &claims, &salsa, &mut vt).is_ok());
        println!(
            "| 2^{log_n} | {clear_bytes} | {salsa_bytes} | {:.1}x |",
            clear_bytes as f64 / salsa_bytes.max(1) as f64
        );
    }
}
