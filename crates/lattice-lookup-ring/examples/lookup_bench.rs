//! Ring-lookup benchmarks: the two PIOPs at several (M, N), the
//! windowed engine's binding pass (ring vs Fp256), the compiled
//! Ring-LogUp, and the RAM batch verification.
//!
//! Run: `cargo run -p lattice-lookup-ring --example lookup_bench`

use lattice_core::transcript::Transcript;
use lattice_lookup_ring::compile::{prove_logup_committed, verify_logup_committed};
use lattice_lookup_ring::fp256_port;
use lattice_lookup_ring::ram::{prove_ram_batch, verify_ram_batch, RamOp};
use lattice_lookup_ring::ring_d::{Elem, RingD};
use lattice_lookup_ring::ring_logup;
use lattice_lookup_ring::ring_plookup;
use lattice_lookup_ring::windowed;
use lattice_lookup_ring::carrier::{CarrierKey, CarrierParams};
use std::time::Instant;

fn build_case(r: &RingD, m: usize, n: usize, seed: &str) -> (Vec<Elem>, Vec<Elem>, Vec<Elem>) {
    let b: Vec<Elem> = (0..n).map(|j| r.random(format!("{seed}-b{j}").as_bytes())).collect();
    let mut a = Vec::with_capacity(m);
    let mut c = Vec::with_capacity(m);
    for i in 0..m {
        let j = (i * 7 + 1) % n;
        a.push(b[j].clone());
        c.push(r.g_map(j as u64));
    }
    (a, b, c)
}

fn bench(label: &str, f: impl FnOnce()) {
    let t = Instant::now();
    f();
    println!("{label:<42} {:>9.1} ms", t.elapsed().as_secs_f64() * 1000.0);
}

fn main() {
    println!("=== Ring-LogUp / Ring-Plookup (d=8, q ~ 2^32) ===");
    for (m, n) in [(4usize, 4usize), (8, 8), (16, 16)] {
        let r = RingD::new(8).ok().unwrap();
        let (a, b, c) = build_case(&r, m, n, "bench");
        bench(&format!("Ring-LogUp  prove+verify M={m} N={n}"), || {
            let mut tr = Transcript::new_default(b"bench");
            let (p, o) = ring_logup::prove_ring_logup(&r, &a, &b, &c, &mut tr).ok().unwrap();
            let mut tr2 = Transcript::new_default(b"bench");
            ring_logup::verify_ring_logup(&r, m, n, &p, &o, &mut tr2).ok().unwrap();
        });
        if m + n <= 16 {
            bench(&format!("Ring-Plookup prove+verify M={m} N={n}"), || {
                let mut tr = Transcript::new_default(b"bench2");
                let (p, o) = ring_plookup::prove_ring_plookup(&r, &a, &b, &c, &mut tr).ok().unwrap();
                let mut tr2 = Transcript::new_default(b"bench2");
                ring_plookup::verify_ring_plookup(&r, m, n, &p, &o, &mut tr2).ok().unwrap();
            });
        }
    }
    println!();
    println!("=== The windowed engine's binding pass (ring carrier) ===");
    for n in [8usize, 32, 128] {
        let r = RingD::new(8).ok().unwrap();
        let v: Vec<Elem> = (0..n).map(|i| r.random(format!("w{i}").as_bytes())).collect();
        let params = windowed::carrier_params_for(&r, n, 2, 1 << 14);
        let key = CarrierKey::from_seed(params, [9u8; 32]);
        bench(&format!("windowed commit+bind N={n}"), || {
            let (slots, commitment) = windowed::windowed_commit(&key, &v).ok().unwrap();
            let mut tr = Transcript::new_default(b"pt");
            let point: Vec<Elem> = (0..n.trailing_zeros())
                .map(|i| r.sample_challenge(&mut tr, format!("p{i}").as_bytes()))
                .collect();
            let y = r.mle_eval(&v, &point).ok().unwrap();
            let proof =
                windowed::prove_windowed_eval(&r, &key, &slots, &point, &y, &commitment, b"s").ok().unwrap();
            windowed::verify_windowed_eval(&r, &key, &commitment, &point, &y, &proof).ok().unwrap();
        });
    }
    println!();
    println!("=== The Fp256 binding pass (BN254 Fr, CIOS grid) ===");
    for n in [8usize, 32, 128] {
        use lattice_projsumcheck::fp256::Fp256;
        let v: Vec<Fp256> = (0..n)
            .map(|i| {
                let bytes = Transcript::xof(b"fpv", format!("{i}").as_bytes(), 8);
                Fp256::from_canonical_u64(u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]))
            })
            .collect();
        let slots = fp256_port::FpSlots::from_vector(&v);
        let carrier = fp256_port::FpCarrier::from_seed(64, n * fp256_port::FP_WINDOWS, b"fpkey");
        bench(&format!("Fp256 binding pass  N={n}"), || {
            let commitment = carrier.commit(&slots.mont).ok().unwrap();
            let point: Vec<Fp256> = (0..n.trailing_zeros())
                .map(|i| {
                    let bytes = Transcript::xof(b"fpp", format!("{i}").as_bytes(), 32);
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(&bytes);
                    Fp256::sample_upper_limb(&arr)
                })
                .collect();
            let y = fp256_port::fp_linear_image(&slots, &point).ok().unwrap();
            let proof =
                fp256_port::prove_fp_binding(&carrier, &slots, &point, &y, &commitment, b"s").ok().unwrap();
            fp256_port::verify_fp_binding(&carrier, n, &point, &y, &commitment, b"s", &proof)
                .ok()
                .unwrap();
        });
    }
    println!();
    println!("=== The compiled Ring-LogUp (commit + binding passes) ===");
    {
        let r = RingD::new(4).ok().unwrap();
        let (a, b, c) = build_case(&r, 4, 4, "cmp");
        let params = windowed::carrier_params_for(&r, 4, 2, 1 << 16);
        let key = CarrierKey::from_seed(params, [21u8; 32]);
        bench("compiled LogUp prove+verify M=N=4", || {
            let proof = prove_logup_committed(&r, &key, &a, &b, &c).ok().unwrap();
            verify_logup_committed(&r, &key, 4, 4, &proof).ok().unwrap();
        });
    }
    println!();
    println!("=== The RAM batch verification (Section 6) ===");
    for (m, k) in [(4usize, 8usize), (8, 16)] {
        // d=8: the record count 2M+k needs |C| = 2^d distinct tags
        let r = RingD::new(8).ok().unwrap();
        let initial: Vec<Elem> = (0..m).map(|i| r.random(format!("ri{i}").as_bytes())).collect();
        let mut ops = Vec::new();
        let mut cur = initial.clone();
        for i in 0..k {
            let addr = ((i * 3 + 1) % m) as u64;
            let val = r.random(format!("rv{i}").as_bytes());
            let write = i % 2 == 0;
            ops.push(RamOp {
                write,
                addr,
                value: if write { val.clone() } else { cur[addr as usize].clone() },
            });
            if write {
                cur[addr as usize] = val;
            }
        }
        bench(&format!("RAM batch verify M={m} k={k}"), || {
            let mut tr = Transcript::new_default(b"ram");
            let (p, o) = prove_ram_batch(&r, &initial, &ops, &cur, &mut tr).ok().unwrap();
            let mut tr2 = Transcript::new_default(b"ram");
            verify_ram_batch(&r, m, k, &initial, &cur, &p, &o, &mut tr2).ok().unwrap();
        });
    }
}
