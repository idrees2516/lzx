//! **The width fold's MSIS security table** (DESIGN_50KB.md Stage 5.2 — the
//! LaBRADOR tail: the quadratic-garbage width-reducing fold that takes the
//! first fold's WIDE response `v ∈ R^{n̄}` to the estimator's sound regime).
//!
//! The width fold's binding terminates in MSIS on `[A₂ | −T]` (the fresh
//! inner key against the pre-challenge inner commitments), the same
//! instance shape `second_fold::profile_bits` models: `m = (w + r₂)·64`
//! columns at the extraction's relaxed bound `2·r₂·A₂·β₁`, where
//!
//! * `w` — the folded (narrow) response width the fold produces;
//! * `r₂` — the part count (one wide-response slice per part);
//! * `κ` — the inner key A₂'s row count;
//! * `A₂` — the γ-challenge amplitude;
//! * `β₁` — the FIRST fold's response gate `r·A₁·255` (the width fold's
//!   input bound — byte-packed columns through the level-1 scalar fold).
//!
//! The honest question this table answers: at the first fold's REAL gates
//! (`β₁` in the 2^15–2^17 range at benchmark column counts), which
//! `(w, r₂, κ, A₂)` rows reach 128 classical bits — and what they cost
//! (the quadratic garbage `r₂·(r₂−1)·κ` ring elements committed
//! pre-challenge, the LaBRADOR ledger's honest price).

use lattice_sis_estimator::{scalar_sis_from_ring, sis_security_bits, SisNorm};

const Q: u128 = 3221225473; // 3·2^30 + 1 (Q_32)
const RING_DIM: u64 = 64;
const BETA0: u64 = 255; // byte-packed coefficient bound

fn bits(kappa: u64, width: u64, bound: u64) -> Option<(f64, f64)> {
    scalar_sis_from_ring(RING_DIM, kappa, width, Q, bound, SisNorm::Infinity)
        .ok()
        .and_then(|p| sis_security_bits(&p).ok())
}

fn main() {
    println!("# The width fold's MSIS security table (q = 3·2^30+1, N = 64)");
    println!("# instance [A2 | -T]: kappa rows x (w + r2) cols, bound 2·r2·A2·beta1");
    println!("# beta1 = the level-1 gate r·A1·255 (A1 = 2^6, the shipped amplitude)");
    println!();
    println!("## The landscape at the real level-1 gates");
    println!();
    println!("| beta1 (r) | w | r2 | kappa | A2 | bound | classical | quantum | garbage elems |");
    println!("|---|---|---|---|---|---|---|---|---|");
    // The first fold's gate at the shipped amplitude A1 = 2^6.
    for (r1, beta1) in [(4u64, 4 * 64 * BETA0), (8, 8 * 64 * BETA0)] {
        let tag = format!("2^{:.0} (r={r1})", (beta1 as f64).log2());
        for w in [2u64, 4, 8] {
            for r2 in [2u64, 4, 8] {
                for kappa in [4u64, 8, 16, 32] {
                    for a2 in [1u64 << 4, 1 << 6, 1 << 8] {
                        let bound = 2 * r2 * a2 * beta1;
                        if u128::from(bound) >= Q / 2 {
                            continue;
                        }
                        if let Some((cl, qm)) = bits(kappa, w + r2, bound) {
                            if cl >= 128.0 {
                                let garbage = r2 * (r2 - 1) * kappa;
                                println!(
                                    "| {tag} | {w} | {r2} | {kappa} | 2^{} | {bound} | {cl:.1} | {qm:.1} | {garbage} |",
                                    a2.trailing_zeros()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    println!();
    println!("## The minimal-sound rows (the cheapest profile per (w, beta1))");
    println!();
    println!("| beta1 (r) | w | r2 | kappa | A2 | classical | garbage elems | proof bytes* |");
    println!("|---|---|---|---|---|---|---|---|");
    for (r1, beta1) in [(2u64, 2 * 64 * BETA0), (4, 4 * 64 * BETA0), (8, 8 * 64 * BETA0)] {
        let tag = format!("2^{:.0} (r={r1})", (beta1 as f64).log2());
        for w in [2u64, 4, 8] {
            // find the min-cost sound row: cost = r2^2·kappa (garbage) — the
            // quadratic ledger; tie-break on r2 first (smaller r2 = smaller
            // proof), then kappa.
            let mut best: Option<(u64, u64, u64, f64, u64)> = None;
            for r2 in [2u64, 4, 8, 16] {
                for kappa in [4u64, 8, 16, 32, 64] {
                    for a2 in [1u64 << 4, 1 << 6, 1 << 8] {
                        let bound = 2 * r2 * a2 * beta1;
                        if u128::from(bound) >= Q / 2 {
                            continue;
                        }
                        if let Some((cl, _)) = bits(kappa, w + r2, bound) {
                            if cl >= 128.0 {
                                let cost = r2 * r2 * kappa;
                                let better = best
                                    .map(|(_, _, _, _, bg)| cost < bg)
                                    .unwrap_or(true);
                                if better {
                                    best = Some((r2, kappa, a2, cl, cost));
                                }
                            }
                        }
                    }
                }
            }
            if let Some((r2, kappa, a2, cl, cost)) = best {
                // garbage r2(r2-1)kappa + images r2·kappa + inner r2·kappa
                // ring elements, 4 B/coeff × 64 coeffs = 256 B/element.
                let elems = r2 * (r2 - 1) * kappa + 2 * r2 * kappa;
                let bytes = elems * 256;
                println!(
                    "| {tag} | {w} | {r2} | {kappa} | 2^{} | {cl:.1} | {cost} | ~{bytes} |",
                    a2.trailing_zeros()
                );
            } else {
                println!("| {tag} | {w} | — | — | — | none | — | — |");
            }
        }
    }
    println!();
    println!("\\* proof bytes = (garbage + images + inner commitments) at 256 B/element;");
    println!("  the folded response itself adds w·256 B (replacing the level-1 response's");
    println!("  n̄·256 B — the width-reduction win: n̄ -> w).");
    println!();
    println!("## The re-packed regime (the Sound profile's shipping shape)");
    println!();
    println!("The level-1 fold re-packed at a large column count r₁ shrinks n̄ to the");
    println!("single digits at the price of the gate beta1 = r₁·A₁·255 (A₁ = 2^6; the");
    println!("A₁ = 2^4 lever halves it per doubling). The width fold's [A2 | -T] is the");
    println!("binding instance of the whole Sound profile — the level-1 [F̄ | -y]");
    println!("instance never arises (the response is not transmitted; the y_j enter");
    println!("only through the public target t = Σ d_j·y_j).");
    println!();
    println!("| beta1 | r1 (A1=2^6) | w | r2 | kappa | A2 | classical | quantum | garbage |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for (log_b1, r1) in [(18u32, 256u64), (20, 1024), (22, 4096), (24, 16384)] {
        let beta1: u64 = 1 << log_b1;
        for (w, r2) in [(8u64, 2u64), (8, 4), (4, 4), (2, 8)] {
            for kappa in [8u64, 16, 32] {
                for a2 in [1u64 << 2, 1 << 4, 1 << 6] {
                    let bound = 2 * r2 * a2 * beta1;
                    if u128::from(bound) >= Q / 2 {
                        continue;
                    }
                    if let Some((cl, qm)) = bits(kappa, w + r2, bound) {
                        if cl >= 128.0 {
                            let garbage = r2 * (r2 - 1) * 4; // level-1 k = 4
                            println!(
                                "| 2^{log_b1} | {r1} | {w} | {r2} | {kappa} | 2^{} | {cl:.1} | {qm:.1} | {garbage} |",
                                a2.trailing_zeros()
                            );
                        }
                    }
                }
            }
        }
    }
    println!();
    println!("## The unsound rows the table rules out (the honest floor)");
    println!();
    println!("| w | r2 | kappa | A2 | beta1 | classical |");
    println!("|---|---|---|---|---|---|");
    for (w, r2, kappa, a2, beta1) in [
        (4u64, 4u64, 4u64, 1u64 << 6, 8 * 64 * BETA0),
        (4, 8, 4, 1 << 6, 8 * 64 * BETA0),
        (8, 4, 4, 1 << 8, 8 * 64 * BETA0),
        (2, 4, 2, 1 << 8, 4 * 64 * BETA0),
    ] {
        let bound = 2 * r2 * a2 * beta1;
        if let Some((cl, _)) = bits(kappa, w + r2, bound) {
            println!(
                "| {w} | {r2} | {kappa} | 2^{} | 2^{:.0} | {cl:.1} |",
                a2.trailing_zeros(),
                (beta1 as f64).log2()
            );
        }
    }
}
