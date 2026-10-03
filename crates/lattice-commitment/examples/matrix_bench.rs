//! The seeded matrix derivation microbenchmark (the verify-side carrier
//! factoring): per-element cost of `AjtaiPublicKey::from_seed` at the
//! zkvm bundles' benchmark-scale dimensions.

use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
use lattice_ring::{Modulus32, RingConfig};
use std::time::Instant;

fn main() {
    let ring = RingConfig::new(Modulus32::Q_32, 6).ok().unwrap();
    for m in [500usize, 2000, 8000] {
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m,
            norm_bound: 1 << 22,
        };
        let t = Instant::now();
        let _pk = AjtaiPublicKey::from_seed(params, [7u8; 32]).ok().unwrap();
        let el = t.elapsed();
        println!(
            "m={m} k=2: from_seed {:.0} ms ({:.2} us/element)",
            el.as_secs_f64() * 1e3,
            el.as_secs_f64() * 1e6 / (2.0 * m as f64)
        );
    }
}
