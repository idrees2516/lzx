//! **The compact fold's MSIS security table** (DESIGN_50KB.md Stage 5.1:
//! "run lattice-sis-estimator on the fold's instances and publish the
//! security table").
//!
//! The compact opening's soundness terminates in MSIS on `[F̄ | −y]`
//! (compact.rs's extraction: two relaxed openings give
//! `A·(Δw − Δσ·s₀) = 0` — plain MSIS on the Ajtai key) with the
//! mixed-moduli constraint lattice. The instance coordinates (the
//! repo's scalar convention, `scalar_sis_from_ring`):
//!
//! * `q = 3·2^30 + 1` (Q_32), ring dimension `N = 64`;
//! * module rank `k` (the fold's commitment rows — `k = 2` is the
//!   shipped configuration; `k = 4, 8` the tightening knobs);
//! * `m = (n̄ + r)·64` — the folded response columns plus the `y`-block;
//! * the coefficient bound: the **worst-case gate** `2·r·A·255` (the
//!   fail-closed completeness gate at the extraction's relaxed `2×`
//!   factor) versus the **statistical 6σ** bound
//!   `2·6·√r·A·255/3` of the honest fold.
//!
//! ## The honest finding (the estimator's verdict)
//!
//! The pre-estimator assertion ("the lattice covering radius at the
//! chosen (k, n̄) far above the gate") does NOT survive the run at the
//! single-level fold's shapes: with `n̄ = M/(64r)` in the hundreds, the
//! instance is massively overdetermined (`m/n` in the tens), and at the
//! folded bound `β ≈ 2·r·A·255` the combinatorial/regime attacks solve
//! in ~12–50 bits. **The single-level fold buys the 27 KB size, not the
//! MSIS binding** — the sound posture needs the knobs and, at these
//! response lengths, ultimately the second-level fold (the LaBRADOR
//! decider, Stage 5.2), which is precisely the `(k, n̄)` tension the
//! design doc named. The table below maps the knob levers: `A↓`
//! (amplitude), `k↑` (commitment rows), `r↓` (columns), and the
//! statistical gate; the search section reports the minimal secure
//! configurations at the short-response end (`n̄ ≤ 8`) that a
//! second-level fold produces.

use lattice_sis_estimator::{scalar_sis_from_ring, sis_security_bits, SisNorm};

const Q: u128 = 3221225473; // 3·2^30 + 1 (Q_32)
const RING_DIM: u64 = 64;
const BETA0: u64 = 255; // byte-packed coefficient bound

fn bits(k: u64, m_ring: u64, bound: u64) -> Option<(f64, f64)> {
    scalar_sis_from_ring(RING_DIM, k, m_ring, Q, bound, SisNorm::Infinity)
        .ok()
        .and_then(|p| sis_security_bits(&p).ok())
}

fn main() {
    println!("# The compact fold's MSIS security table (q = 3·2^30+1, N = 64)");
    println!("# bound regimes: gate = 2·r·A·255 (worst-case), 6σ = 2·6·√r·A·255/3 (statistical)");
    println!();
    println!("## The shipped shape (k = 2, A = 2^12) at benchmark response lengths");
    println!();
    println!("| k | A | n_bar | r | bound | beta | classical | quantum |");
    println!("|---|---|---|---|---|---|---|---|");
    for (n_bar, r) in [(128u64, 16u64), (256, 8), (512, 4), (1024, 2)] {
        for (label, a) in [("A=2^12", 1u64 << 12), ("A=2^6", 1u64 << 6)] {
            let gate = 2 * r * a * BETA0;
            let sigma = (r as f64).sqrt() * (a as f64) * (BETA0 as f64) / 3.0;
            let six_sigma = (12.0 * sigma) as u64;
            for (regime, bound) in [("gate", gate), ("6-sigma", six_sigma)] {
                if let Some((cl, qm)) = bits(2, n_bar + r, bound) {
                    println!("| 2 | {label} | {n_bar} | {r} | {regime} | {bound} | {cl:.1} | {qm:.1} |");
                }
            }
        }
    }
    println!();
    println!("## The knob search: minimal k for >= 128 classical bits at the gate bound");
    println!();
    println!("| n_bar | r | A | min k (gate) | min k (6σ) |");
    println!("|---|---|---|---|---|");
    for n_bar in [2u64, 4, 8, 16, 32, 64, 128] {
        for r in [4u64, 8, 16] {
            for a in [1u64 << 6, 1 << 8, 1 << 12] {
                let gate = 2 * r * a * BETA0;
                let sigma = (r as f64).sqrt() * (a as f64) * (BETA0 as f64) / 3.0;
                let six_sigma = (12.0 * sigma) as u64;
                let find = |bound: u64| -> u64 {
                    for k in [2u64, 4, 8, 16, 32, 64, 128, 256] {
                        if let Some((cl, _)) = bits(k, n_bar + r, bound) {
                            if cl >= 128.0 {
                                return k;
                            }
                        }
                    }
                    0 // no k up to 256 reaches 128 bits
                };
                let kg = find(gate);
                let ks = find(six_sigma);
                if kg > 0 || ks > 0 {
                    println!("| {n_bar} | {r} | 2^{a:#x} | {kg} | {ks} |");
                }
            }
        }
    }
    println!();
    println!("## The reference: short responses (the second-level fold's regime)");
    println!();
    println!("| k | n_bar | r | A | bound | classical | quantum |");
    println!("|---|---|---|---|---|---|---|");
    for k in [2u64, 4] {
        for n_bar in [2u64, 4, 8] {
            for (r, a) in [(4u64, 1u64 << 6), (8, 1 << 6), (4, 1 << 8)] {
                let gate = 2 * r * a * BETA0;
                if let Some((cl, qm)) = bits(k, n_bar + r, gate) {
                    println!("| {k} | {n_bar} | {r} | 2^{a:#x} | gate {gate} | {cl:.1} | {qm:.1} |");
                }
            }
        }
    }
}
