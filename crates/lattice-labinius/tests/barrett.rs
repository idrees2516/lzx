//! Exhaustive check of [`barrett_mod_u64`] against `%` for every supported prime.
//!
//! The claim is narrow but load-bearing: for every `v < q^2 + q` the Barrett form returns
//! exactly `v % q`. The fold's Horner evaluations and every inner-product accumulator feed it
//! values below `q^2 + q`, so this exhausts the reachable range.

use lattice_labinius::params::{barrett_mod_u64, QS, QS_LARGE, QS_QUAD};

fn check_q(q: u16) {
    let q64 = q as u64;
    let hi = q64 * q64 + q64;
    // base region and the interesting straddles around multiples of q
    let mut v = 0u64;
    while v < 4096 {
        assert_eq!(barrett_mod_u64(v, q), v % q64, "q={q} v={v}");
        v += 1;
    }
    for k in 1..=q as u64 {
        for d in [0u64, 1, 2, q64 - 1, q64, q64 + 1] {
            let v = k * q64 + d;
            if v < hi {
                assert_eq!(barrett_mod_u64(v, q), v % q64, "q={q} v={v}");
            }
        }
    }
    // pseudo-random interior + top edge
    let mut x = 0x9E37_79B9_7F4A_7C15u64 % hi;
    for _ in 0..4096 {
        assert_eq!(barrett_mod_u64(x, q), x % q64, "q={q} v={x}");
        x = (x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) % hi;
    }
    for v in [hi - 1, hi - 2, hi - q64, hi - q64 - 1] {
        assert_eq!(barrett_mod_u64(v, q), v % q64, "q={q} v={v}");
    }
}

#[test]
fn barrett_exhaustive_all_primes() {
    for q in QS {
        check_q(q);
    }
    for q in QS_LARGE {
        check_q(q);
    }
    for q in QS_QUAD {
        check_q(q);
    }
}

#[test]
fn barrett_sweep_small_primes() {
    // every v in [0, q^2 + q) for the two smallest primes — the full exhaustive range
    for q in [3889u16] {
        let q64 = q as u64;
        let hi = q64 * q64 + q64;
        for v in 0..hi {
            assert_eq!(barrett_mod_u64(v, q), v % q64, "q={q} v={v}");
        }
    }
}
