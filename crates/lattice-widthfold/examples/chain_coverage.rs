//! **The recursive width-collapse chain's coverage table** — the
//! measured boundary this wave publishes (NEXT_STEPS's honest ledger:
//! "takes the Sound profile's coverage from n̄ ≤ 16 to the benchmark
//! streams").
//!
//! Rows: the largest staged n̄ per level-1 gate (the search
//! fail-closes beyond it — the honest Q_32 ceiling), the stream
//! ceiling at the r₁ = 128 packing cap, and the schedule shapes at
//! the benchmark scales.

use lattice_widthfold::chain::WidthChainParams;
fn main() {
    let ring = lattice_widthfold::codec::q32_ring().unwrap();
    let q = u64::from(ring.modulus.q);
    let dim = ring.n() as u64;
    // The coverage boundary per level-1 gate: the largest n̄ that stages.
    for (beta, tag) in [(255u64, "byte gate"), (1 << 15, "r1=8 sound profile"), (1 << 17, "r1=32"), (522240, "r1=128 (the packing cap)")] {
        let mut best = 0usize;
        for log_n in (4usize..=14).rev() {
            let n_bar = 1usize << log_n;
            if WidthChainParams::sound_chain_for(n_bar, beta, q, dim).is_ok() {
                best = n_bar;
                break;
            }
        }
        // The stream ceiling at r1=128: stream = n̄·64·128 bytes.
        let stream_mb = best as f64 * 64.0 * 128.0 / 1e6;
        println!("beta=2^{:.0} ({tag}): max staged n̄ = {best} (stream ceiling ~{stream_mb:.1} MB at r1=128)", (beta as f64).log2());
    }
    // The schedule shapes at the benchmark scale.
    for (n_bar, beta) in [(256usize, 1u64 << 15), (512, 1 << 15), (1024, 522240)] {
        match WidthChainParams::sound_chain_for(n_bar, beta, q, dim) {
            Ok(p) => {
                let stages: Vec<String> = p.stages.iter().map(|s| format!("(r2={},w={},k={},A=2^{})", s.r2, s.w, s.kappa, s.amplitude.trailing_zeros())).collect();
                println!("n̄={n_bar} beta=2^{:.0}: {} stages: {} | cost {} elems | grinding {:.1}b", (beta as f64).log2(), p.stages.len(), stages.join(" -> "), p.cost_elements(4), p.grinding_bits());
            }
            Err(e) => println!("n̄={n_bar} beta=2^{:.0}: ERR {e}", (beta as f64).log2()),
        }
    }
}
