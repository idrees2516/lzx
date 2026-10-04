//! Probe: the sound frontier for the RECURSIVE width-fold chain (Stage 5.2+).
//!
//! The chain's stage ℓ terminates in MSIS on `[A_ℓ | −T_ℓ]` at
//! `w_ℓ + r₂` columns with the extraction bound `2·r₂·A₂·β_ℓ`. The
//! single-stage sound coverage is `n̄ ≤ 16` (the cheap row (8,2,8,2^2));
//! the chain extends coverage iff INTERMEDIATE rows exist that are
//! estimator-sound at their own (grown) gates. This probe scans the
//! frontier honestly: for every (width, bound) pair the estimator can
//! rate, the classical-bit verdict.

use lattice_sis_estimator::{scalar_sis_from_ring, sis_security_bits, SisNorm};

const Q: u128 = 3221225473; // 3·2^30 + 1 (Q_32)
const RING_DIM: u64 = 64;

fn bits(kappa: u64, width: u64, bound: u64) -> Option<f64> {
    scalar_sis_from_ring(RING_DIM, kappa, width, Q, bound, SisNorm::Infinity)
        .ok()
        .and_then(|p| sis_security_bits(&p).ok())
        .map(|(cl, _)| cl)
}

fn main() {
    for kappa in [16u64, 32, 64, 128] {
        println!("# The sound frontier (q = 3·2^30+1, N = 64, kappa = {kappa})");
        println!("# rows: classical bits at (width, bound); SOUND >= 128");
        println!();
        print!("| bound\\\\width |");
        for width in [10u64, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 130] {
            print!(" {width} |");
        }
        println!();
        print!("|---|");
        for _ in 0..12 {
            print!("---|");
        }
        println!();
        for log_b in (16u32..=29).rev() {
            let bound: u64 = 1 << log_b;
            if u128::from(bound) >= Q / 2 {
                continue;
            }
            print!("| 2^{log_b} |");
            for width in [10u64, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 130] {
                match bits(kappa, width, bound) {
                    Some(cl) => {
                        let mark = if cl >= 128.0 { "**" } else { "  " };
                        print!(" {mark}{cl: >5.0}{mark} |");
                    }
                    None => print!("   —  |"),
                }
            }
            println!();
        }
        println!();
    }
}
