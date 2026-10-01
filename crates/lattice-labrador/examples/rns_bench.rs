//! The PERFORMANCE.md §4 "LaBRADOR RNS" item, measured: the i128 schoolbook baseline vs the
//! AVX-512 split-2^24 convolution (`src/conv.rs`) that answers it at this ring's N=64 regime.
//!
//! Upstream's 8-prime RNS + per-prime NTT wins at N >= 1024 (the O(N^2) -> O(N log N)
//! transition); at N=64 the schoolbook is only 4096 MACs and the CRT reconstruction would eat
//! the win, so the exact split convolution (no RNS, no CRT, branchless windows) is the right
//! instrument here.
use lattice_labrador::ring::{Poly, N};
use std::time::Instant;

fn median(v: &mut [u128]) -> f64 { v.sort(); v[v.len()/2] as f64 }

fn main() {
    let mut r = 0x1234_5678_9ABC_DEF0u64;
    let next = |r: &mut u64| { *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((*r >> 16) as i64).rem_euclid(1 << 47) };
    let mk = |r: &mut u64| { let mut p = [0i64; N]; for x in p.iter_mut() { *x = next(r); } Poly(p) };
    let a: Vec<Poly> = (0..16).map(|_| mk(&mut r)).collect();
    let b: Vec<Poly> = (0..16).map(|_| mk(&mut r)).collect();
    let mut t = vec![];
    for _ in 0..15 { let s = Instant::now(); for i in 0..16 { std::hint::black_box(a[i].mul_schoolbook(&b[i])); } t.push(s.elapsed().as_nanos()); }
    println!("schoolbook mul x16        {:>10.2} us  ({:.2} us each)", median(&mut t)/1e3, median(&mut t)/1e3/16.0);
    let mut t = vec![];
    for _ in 0..15 { let s = Instant::now(); std::hint::black_box(Poly::sprod_schoolbook(&a, &b)); t.push(s.elapsed().as_nanos()); }
    println!("schoolbook sprod k=16 {:>10.2} us", median(&mut t)/1e3);
    // the vectorized paths
    let mut t = vec![];
    for _ in 0..15 { let s = Instant::now(); for i in 0..16 { std::hint::black_box(a[i].mul(&b[i])); } t.push(s.elapsed().as_nanos()); }
    println!("conv mul x16          {:>10.2} us  ({:.2} us each)", median(&mut t)/1e3, median(&mut t)/1e3/16.0);
    let mut t = vec![];
    for _ in 0..15 { let s = Instant::now(); std::hint::black_box(Poly::sprod(&a, &b)); t.push(s.elapsed().as_nanos()); }
    println!("conv sprod k=16       {:>10.2} us", median(&mut t)/1e3);
}
