//! **The multi-stage LaBRADOR extraction ledger's published table** —
//! the degree-law unwind analysis as measured output (every numeric
//! claim in `docs/analysis/MULTISTAGE_EXTRACTION.md` is this program's
//! stdout — no hand-typed numbers).
//!
//! Rows: the shipped schedule shapes (the coverage table's boundary
//! rows) with their composed extraction ledgers: the rewind tree, the
//! unwind degree 2L, the grinding ledger, the composed knowledge gap,
//! the unwind norm law, and the per-stage kernel verdicts.

use lattice_widthfold::chain::WidthChainParams;
use lattice_widthfold::extraction::{chain_extraction_ledger, ledger_report};

fn main() {
    let ring = lattice_widthfold::codec::q32_ring().unwrap();
    let q = u64::from(ring.modulus.q);
    let dim = ring.n() as u64;
    println!(
        "== The multi-stage LaBRADOR extraction ledger (Q_32, ring dim {}) ==",
        dim
    );
    println!();
    // The shipped coverage-table boundary rows + the benchmark shapes.
    for (n_bar, beta, tag) in [
        (
            16usize,
            255u64,
            "single-stage terminal (the n_bar<=16 boundary row)",
        ),
        (512, 255, "the D4 byte-witness scale (2^10 values)"),
        (4096, 255, "the byte-gate coverage boundary"),
        (
            512,
            1 << 15,
            "the benchmark-stream schedule (BENCHMARKS 2k)",
        ),
        (2048, 1 << 15, "the r1=8 sound-profile boundary"),
        (128, 522240, "the r1=128 packing-cap row"),
    ] {
        match WidthChainParams::sound_chain_for(n_bar, beta, q, dim) {
            Ok(params) => match chain_extraction_ledger(&params, beta, q, dim) {
                Ok(ledger) => {
                    println!(
                        "-- n_bar={n_bar}, beta=2^{:.0} ({tag}) --",
                        (beta as f64).log2()
                    );
                    println!("{}", ledger_report(&ledger, q));
                    println!();
                }
                Err(e) => println!("n_bar={n_bar}: ledger FAILED (fail-closed): {e}"),
            },
            Err(e) => println!(
                "n_bar={n_bar}, beta=2^{:.0}: no schedule ({e})",
                (beta as f64).log2()
            ),
        }
    }
}
