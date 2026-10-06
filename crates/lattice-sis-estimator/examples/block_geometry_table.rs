//! **The block-commit geometry's MSIS security table** (the honest
//! ledger's first follow-up: "the estimator table has not been re-run
//! for the block geometry — the follow-up is lattice-sis-estimator at
//! (k, m, n̄_pad) with the byte bound").
//!
//! `blockfold.rs` replaces the r per-column Ajtai commitments with ONE
//! packed block commitment `y = F·vec(W)` under the wide seeded key
//! `F ∈ R_q^{k×m}`, `m = r·n̄^pad` ring columns, q = Q_32, ring dim 64.
//! Knowledge soundness terminates in MSIS on the WIDE `[F | −y]` — and
//! the block's distinctive property is that the preimage's W-side is
//! **byte-bounded** (the honest W is byte-packed; two openings differ by
//! at most 255 per coefficient), INDEPENDENT of r and A.
//!
//! Instance coordinates (`scalar_sis_from_ring`, the repo convention):
//! * homogeneous binding (`F` alone — two W-side openings of one y):
//!   rank k, width m, bound 255;
//! * composed extraction (`[F | −y]` — Δw vs the response difference):
//!   rank k, width m+1, bound max(255, 2·r·A·255) (the fail-closed
//!   completeness gate at the extraction's relaxed 2×) and the 6σ
//!   statistical form;
//! * Akita's exact-norm discipline: BOTH ℓ∞ (byte bound) and ℓ2
//!   (worst-case 255·√(m·64); the 6σ statistical form 6·(255/√12)·√(m·64)).
//!
//! Run: `cargo run --release -p lattice-sis-estimator --example
//! block_geometry_table`.

use lattice_sis_estimator::{scalar_sis_from_ring, sis_security_bits, SisNorm};

const Q: u128 = 3221225473; // 3·2^30 + 1 (Q_32)
const RING_DIM: u64 = 64;
const BETA0: u64 = 255; // byte-packed coefficient bound
const AMPLITUDE: u64 = 1 << 6; // the shipped interim hardening (Stage 5.1)

fn bits(k: u64, m_ring: u64, bound: u64, norm: SisNorm) -> Option<(f64, f64)> {
    scalar_sis_from_ring(RING_DIM, k, m_ring, Q, bound, norm)
        .ok()
        .and_then(|p| sis_security_bits(&p).ok())
}

fn main() {
    println!("# The block-commit geometry's MSIS table (q = 3·2^30+1, N = 64, A = 2^6)");
    println!("# m = r·n̄^pad ring columns — the WHOLE universe (vs the compact fold's n̄+r)");
    println!("# the byte bound 255 is the block's distinctive preimage: W-side differences");
    println!("# of two openings, independent of r and A");
    println!();

    // ---- 1. The shipped benchmark shapes (m·64 ≈ the byte-stream M). ----
    println!("## 1. The benchmark bundle shapes — the byte-bound homogeneous instance (F)");
    println!();
    println!("| k | n_bar | r | m_ring | m/n | bound | classical | quantum |");
    println!("|---|---|---|---|---|---|---|---|");
    for (n_bar, r) in [(128u64, 16u64), (256, 8), (512, 4), (1024, 2)] {
        let n_bar_pad = n_bar.next_power_of_two();
        let m_ring = r * n_bar_pad;
        for k in [2u64, 4, 8, 16, 32, 64] {
            let m_over_n = (m_ring as f64) / (k as f64);
            if let Some((cl, qm)) = bits(k, m_ring, BETA0, SisNorm::Infinity) {
                println!(
                    "| {k} | {n_bar} | {r} | {m_ring} | {m_over_n:.0} | byte {BETA0} | {cl:.1} | {qm:.1} |"
                );
            }
        }
    }
    println!();

    // ---- 2. The composed [F | −y] instance at the gate and 6σ bounds. ----
    println!("## 2. The composed extraction [F | −y] — gate = 2·r·A·255 and 6σ");
    println!();
    println!("| k | n_bar | r | regime | bound | classical | quantum |");
    println!("|---|---|---|---|---|---|---|");
    for (n_bar, r) in [(128u64, 16u64), (256, 8), (512, 4), (1024, 2)] {
        let n_bar_pad = n_bar.next_power_of_two();
        let m_ring = r * n_bar_pad + 1; // + the y-block
        let gate = 2 * r * AMPLITUDE * BETA0;
        let sigma = (r as f64).sqrt() * (AMPLITUDE as f64) * (BETA0 as f64) / 3.0;
        let six_sigma = (6.0 * sigma) as u64;
        for (regime, bound) in [("gate", gate), ("6-sigma", six_sigma)] {
            for k in [2u64, 8, 32] {
                if let Some((cl, qm)) = bits(k, m_ring, bound, SisNorm::Infinity) {
                    println!(
                        "| {k} | {n_bar} | {r} | {regime} | {bound} | {cl:.1} | {qm:.1} |"
                    );
                }
            }
        }
    }
    println!();

    // ---- 3. Akita's exact-norm discipline: the ℓ2 paths. ----
    println!("## 3. The Euclidean paths (exact-ℓ2, the Akita discipline)");
    println!();
    println!("| k | m_ring | regime | l2 bound | classical | quantum |");
    println!("|---|---|---|---|---|---|");
    for (n_bar, r) in [(128u64, 16u64), (512, 4)] {
        let n_bar_pad = n_bar.next_power_of_two();
        let m_ring = r * n_bar_pad;
        let coeffs = m_ring * RING_DIM;
        // worst-case: every coefficient difference at the byte bound.
        let l2_worst = (BETA0 as f64) * (coeffs as f64).sqrt();
        // 6σ: byte-valued differences, uniform-ish — σ = 255/√12 per coord.
        let l2_six_sigma = 6.0 * (BETA0 as f64) / 12.0f64.sqrt() * (coeffs as f64).sqrt();
        for k in [2u64, 8, 32] {
            for (regime, bound) in [
                ("l2-worst", l2_worst.ceil() as u64),
                ("l2-6sigma", l2_six_sigma.ceil() as u64),
            ] {
                if let Some((cl, qm)) = bits(k, m_ring, bound, SisNorm::Euclidean) {
                    println!(
                        "| {k} | {m_ring} | {regime} | {bound} | {cl:.1} | {qm:.1} |"
                    );
                }
            }
        }
    }
    println!();

    // ---- 4. The k-ladder search: minimal k for ≥ 128 classical bits. ----
    println!("## 4. The k-ladder: minimal k reaching 128 classical bits (byte bound)");
    println!();
    println!("| n_bar | r | m_ring | min k (byte) | min k (gate) | min k (l2-6σ) |");
    println!("|---|---|---|---|---|---|");
    for (n_bar, r) in [
        (128u64, 16u64),
        (256, 8),
        (512, 4),
        (1024, 2),
        (2048, 2),
        (4096, 1),
    ] {
        let n_bar_pad = n_bar.next_power_of_two();
        let m_ring = r * n_bar_pad;
        let gate = 2 * r * AMPLITUDE * BETA0;
        let coeffs = m_ring * RING_DIM;
        let l2_six_sigma = 6.0 * (BETA0 as f64) / 12.0f64.sqrt() * (coeffs as f64).sqrt();
        let find = |bound: u64, norm: SisNorm| -> u64 {
            for k in [2u64, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
                if let Some((cl, _)) = bits(k, m_ring, bound, norm) {
                    if cl >= 128.0 {
                        return k;
                    }
                }
            }
            0
        };
        let kb = find(BETA0, SisNorm::Infinity);
        let kg = find(gate, SisNorm::Infinity);
        let kl = find(l2_six_sigma.ceil() as u64, SisNorm::Euclidean);
        println!("| {n_bar} | {r} | {m_ring} | {kb} | {kg} | {kl} |");
    }
    println!();

    // ---- 5. The compact-vs-block verdict at one bundle. ----
    println!("## 5. The honest comparison: the block's wide key vs the compact's [F̄ | −y]");
    println!();
    println!("The block instance is MUCH wider (m = r·n̄^pad vs n̄+r) but its W-side");
    println!("preimage is byte-bounded (255) instead of gate-bounded (2·r·A·255).");
    println!("The wider instance gives the attack MORE freedom (larger m/n) — the");
    println!("security must come from the byte bound and k, not the width.");
    println!();
    println!("| mode | k | m_ring | bound | classical | quantum |");
    println!("|---|---|---|---|---|---|");
    {
        // The benchmark bundle (n_bar=128, r=16), compact gate vs block byte.
        let (n_bar, r) = (128u64, 16u64);
        let compact_gate = 2 * r * AMPLITUDE * BETA0;
        let compact_sigma = (r as f64).sqrt() * (AMPLITUDE as f64) * (BETA0 as f64) / 3.0;
        let compact_6s = (6.0 * compact_sigma) as u64;
        for k in [2u64, 8, 32] {
            if let Some((cl, qm)) = bits(k, n_bar + r, compact_gate, SisNorm::Infinity) {
                println!("| compact gate | {k} | {} | {compact_gate} | {cl:.1} | {qm:.1} |", n_bar + r);
            }
            if let Some((cl, qm)) = bits(k, n_bar + r, compact_6s, SisNorm::Infinity) {
                println!("| compact 6σ | {k} | {} | {compact_6s} | {cl:.1} | {qm:.1} |", n_bar + r);
            }
        }
        let m_ring = r * n_bar.next_power_of_two();
        for k in [2u64, 8, 32] {
            if let Some((cl, qm)) = bits(k, m_ring, BETA0, SisNorm::Infinity) {
                println!("| block byte | {k} | {m_ring} | {BETA0} | {cl:.1} | {qm:.1} |");
            }
        }
    }
}
