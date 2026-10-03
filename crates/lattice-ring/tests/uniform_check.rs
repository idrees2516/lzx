//! The uniform-from-seed derivation: distinctness + the fast-path
//! differential against the original counter-indexed loop.

/// The pre-fast-path reference: per-coefficient XOF with the growing
/// squeeze (counter-indexed windows). Kept as the bit-exact ground truth.
fn uniform_reference(
    c: &lattice_ring::RingConfig,
    domain: &[u8],
    seed: &[u8],
    index: u64,
) -> Vec<u32> {
    let mut salt = Vec::with_capacity(domain.len() + seed.len() + 8);
    salt.extend_from_slice(domain);
    salt.extend_from_slice(seed);
    salt.extend_from_slice(&index.to_le_bytes());
    let q = c.modulus.q as u64;
    let limit = (u32::MAX as u64 + 1) - ((u32::MAX as u64 + 1) % q);
    let mut coeffs = Vec::with_capacity(c.n());
    let mut counter = 0u64;
    while coeffs.len() < c.n() {
        let bytes = lattice_core::transcript::Transcript::xof(
            b"uniform",
            &salt,
            8 + (counter as usize) * 4 + 4,
        );
        let off = bytes.len() - 4;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(&bytes[off..]);
        let cand = u32::from_le_bytes(arr) as u64;
        if cand < limit {
            coeffs.push((cand % q) as u32);
        }
        counter += 1;
    }
    coeffs
}

#[test]
fn uniform_coeffs_are_distinct() {
    let c = lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, 6)
        .ok()
        .unwrap();
    let u = c.uniform_from_seed(b"ajtai-A", &[7u8; 32], 3);
    let coeffs = u.coeffs();
    let distinct: std::collections::HashSet<u32> = coeffs.iter().copied().collect();
    eprintln!(
        "n={} distinct_coeffs={} first8={:?}",
        coeffs.len(),
        distinct.len(),
        &coeffs[..8.min(coeffs.len())]
    );
    // A uniform ring element over R_q with n=64 should have ~64 distinct
    // coefficients (collision probability for q~2^31.6 is negligible).
    assert!(
        distinct.len() > 32,
        "uniform_from_seed produces (near-)constant ring elements: {} distinct of {}",
        distinct.len(),
        coeffs.len()
    );
}

#[test]
fn uniform_fast_path_is_byte_identical_to_reference() {
    for log_n in [3u32, 5, 6] {
        let c = lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, log_n)
            .ok()
            .unwrap();
        for idx in [0u64, 1, 7, 255, 4096, 1 << 20] {
            for seed_tag in [0u8, 1, 42, 255] {
                let seed = [seed_tag; 32];
                let got = c.uniform_from_seed(b"ajtai-A", &seed, idx);
                let want = uniform_reference(&c, b"ajtai-A", &seed, idx);
                assert_eq!(
                    got.coeffs(),
                    &want[..],
                    "log_n={log_n} idx={idx} seed_tag={seed_tag}"
                );
            }
        }
    }
}
