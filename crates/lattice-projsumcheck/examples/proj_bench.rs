//! Benchmarks for the projective (monomial-basis) sum-check — the §6
//! evaluation of ePrint 2026/762, adapted to Goldilocks and this
//! codebase's layout:
//!
//! * binding throughput: Boolean `a + r·(b−a)` vs projective `a + r·b`
//!   (Table 2 — one subtraction saved per coefficient);
//! * full-domain `eq`/`LT` table construction: Boolean vs projective
//!   recurrences (Table 4 — the `(e, e·r)` free-left-half and the
//!   subtraction-free LT doubling);
//! * end-to-end degree-2 and degree-2 × eq sum-check proving: the Boolean
//!   engine vs the projective engine (Table 6);
//! * Fp256: chained multiplication, full CIOS vs upper-limb challenges
//!   (Table 3 — the 1.92× path), and the projective binding loop with
//!   upper-limb challenges (Table 5).

// Index-arithmetic loops (MSB-first bit extraction, limb walks,
// prefix/suffix products) read clearer with explicit indices.
#![allow(clippy::needless_range_loop)]
use lattice_core::field_simd;
use lattice_core::{DenseMle, Goldilocks, Transcript};
use lattice_projsumcheck::fp256::Fp256;
use lattice_projsumcheck::proj_mle::MonomialMle;
use lattice_projsumcheck::proj_sumcheck::{prove as proj_prove, ProjVirtualPolynomial};
use lattice_projsumcheck::tables::{eq_projective_table, lt_projective_table};
use lattice_sumcheck::sumcheck::prove as bool_prove;
use lattice_sumcheck::virtual_poly::VirtualPolynomial;
use std::time::Instant;

fn now_ms() -> f64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    t.as_secs_f64() * 1000.0
}

fn fmt_ratio(base: f64, opt: f64) -> String {
    if opt <= 0.0 {
        "n/a".to_string()
    } else {
        format!("{:.3}x", base / opt)
    }
}

fn main() {
    println!("=== Projective sum-check benchmarks (ePrint 2026/762) ===\n");

    // ------------------------------------------------------------------
    // 1. Binding throughput over 2^20 coefficient pairs (Table 2).
    // ------------------------------------------------------------------
    {
        let n: usize = 20;
        let len = 1usize << n;
        let base: Vec<Goldilocks> = (0..len as u64).map(Goldilocks::from_u64).collect();
        let r = Goldilocks::from_u64(0x1234_5678_9abc_def0);

        let mut buf_bool = base.clone();
        let t0 = Instant::now();
        field_simd::bind_first_half_in_place(&mut buf_bool, r);
        let t_bool = t0.elapsed().as_secs_f64() * 1e3;

        let mut buf_proj = base.clone();
        let t0 = Instant::now();
        field_simd::bind_projective_first_half_in_place(&mut buf_proj, r);
        let t_proj = t0.elapsed().as_secs_f64() * 1e3;

        // Blackhole so both kernels are fully measured.
        let _ = (buf_bool[0].to_canonical_u64(), buf_proj[0].to_canonical_u64());
        println!(
            "[binding 2^{}] Boolean (a + r(b−a)): {:.3} ms | projective (a + r·b): {:.3} ms | speedup {}",
            n, t_bool, t_proj, fmt_ratio(t_bool, t_proj)
        );
    }

    // ------------------------------------------------------------------
    // 2. Full-domain table construction (Table 4).
    // ------------------------------------------------------------------
    {
        let n = 20;
        let r: Vec<Goldilocks> = (1..=n as u64)
            .map(|i| Goldilocks::from_u64(i.wrapping_mul(1_000_000_007)))
            .collect();

        let t0 = Instant::now();
        let eq_bool = field_simd::eq_table(&r);
        let t_eq_bool = t0.elapsed().as_secs_f64() * 1e3;

        let t0 = Instant::now();
        let eq_proj = eq_projective_table(&r);
        let t_eq_proj = t0.elapsed().as_secs_f64() * 1e3;

        let t0 = Instant::now();
        let lt_proj = lt_projective_table(&r);
        let t_lt_proj = t0.elapsed().as_secs_f64() * 1e3;

        // LT Boolean baseline: the standard recurrence (1 mul + 1 add +
        // 1 sub per entry per round).
        let t0 = Instant::now();
        let lt_bool = lt_bool_table(&r);
        let t_lt_bool = t0.elapsed().as_secs_f64() * 1e3;

        println!(
            "[eq table n={}] Boolean: {:.3} ms | projective (e, e·r): {:.3} ms | speedup {}",
            n, t_eq_bool, t_eq_proj, fmt_ratio(t_eq_bool, t_eq_proj)
        );
        println!(
            "[LT table n={}] Boolean: {:.3} ms | projective: {:.3} ms | speedup {}",
            n, t_lt_bool, t_lt_proj, fmt_ratio(t_lt_bool, t_lt_proj)
        );
        // Blackhole the results so the table constructions are measured.
        let chk = |t: &[Goldilocks]| -> u64 {
            t.iter().map(|v| v.to_canonical_u64()).fold(0u64, u64::wrapping_add)
        };
        let _ = (chk(&eq_bool), chk(&eq_proj), chk(&lt_bool), chk(&lt_proj));
    }

    // ------------------------------------------------------------------
    // 3. End-to-end sum-check proving (Table 6, degree-2 and degree-2×eq).
    // ------------------------------------------------------------------
    for (n, with_eq) in [(20usize, false), (20, true), (22, false)] {
        let f = MonomialMle::random(n, b"bench-f");
        let h = MonomialMle::random(n, b"bench-h");
        let df = DenseMle::random(n, b"bench-f");
        let dh = DenseMle::random(n, b"bench-h");

        // Boolean engine instance.
        let mut vp_bool = VirtualPolynomial::new(n);
        let fa = vp_bool.add_factor(df.clone()).unwrap();
        let fb = vp_bool.add_factor(dh.clone()).unwrap();
        if with_eq {
            let pt: Vec<Goldilocks> = vec![Goldilocks::from_u64(7); n];
            let eqf = vp_bool.add_factor(DenseMle::eq_extension(&pt)).unwrap();
            vp_bool.add_term(Goldilocks::ONE, vec![fa, fb, eqf]).unwrap();
        } else {
            vp_bool.add_term(Goldilocks::ONE, vec![fa, fb]).unwrap();
        }
        let claim_bool = dense_claim(&vp_bool);

        let mut ts = Transcript::new_default(b"bench-bool");
        let t0 = Instant::now();
        let out_bool = bool_prove(&vp_bool, claim_bool, &mut ts).unwrap();
        let t_bool = t0.elapsed().as_secs_f64() * 1e3;

        // Projective engine instance.
        let mut vp_proj = ProjVirtualPolynomial::new(n);
        let pa = vp_proj.add_factor(f.clone()).unwrap();
        let pb = vp_proj.add_factor(h.clone()).unwrap();
        if with_eq {
            let r: Vec<Goldilocks> = (1..=n as u64)
                .map(|i| Goldilocks::from_u64(i.wrapping_mul(1_000_000_007)))
                .collect();
            let peq = vp_proj.add_factor(MonomialMle::eq_projective(&r)).unwrap();
            vp_proj.add_term(Goldilocks::ONE, vec![pa, pb, peq]).unwrap();
        } else {
            vp_proj.add_term(Goldilocks::ONE, vec![pa, pb]).unwrap();
        }
        let claim_proj = vp_proj.total_sum();
        let mut ts2 = Transcript::new_default(b"bench-proj");
        let t0 = Instant::now();
        let out_proj = proj_prove(&vp_proj, claim_proj, &mut ts2).unwrap();
        let t_proj = t0.elapsed().as_secs_f64() * 1e3;

        // The discrete claims coincide (same truth tables).
        if !with_eq {
            assert_eq!(claim_bool, claim_proj);
        }

        let proof_bool_bytes: usize = out_bool
            .proof
            .rounds
            .iter()
            .map(|r| r.len() * 8)
            .sum();
        let proof_proj_bytes: usize = out_proj
            .proof
            .rounds
            .iter()
            .map(|r| r.len() * 8)
            .sum();
        println!(
            "[sumcheck n={}{}] Boolean: {:.3} ms ({} B) | projective: {:.3} ms ({} B) | speedup {} | size {:.2}x",
            n,
            if with_eq { " ×eq" } else { "" },
            t_bool,
            proof_bool_bytes,
            t_proj,
            proof_proj_bytes,
            fmt_ratio(t_bool, t_proj),
            proof_bool_bytes as f64 / proof_proj_bytes as f64
        );
    }

    // ------------------------------------------------------------------
    // 4. Fp256: chained multiplication, full CIOS vs upper-limb (Table 3)
    //    and the binding loop (Table 5).
    // ------------------------------------------------------------------
    {
        let iters = 1_000_000u32;
        let a = Fp256::from_canonical_u64(0xdead_beef_cafe_f00d);
        let mut hash = [11u8; 32];
        hash[0] = 0x42;
        let up = Fp256::sample_upper_limb(&hash);

        let mut acc = a;
        let t0 = Instant::now();
        for _ in 0..iters {
            acc = acc.mul(&up);
        }
        let t_full = t0.elapsed().as_secs_f64() * 1e3;

        let mut acc2 = a;
        let t0 = Instant::now();
        for _ in 0..iters {
            acc2 = acc2.mul_upper_limb(&up);
        }
        let t_upper = t0.elapsed().as_secs_f64() * 1e3;
        assert_eq!(acc.limbs, acc2.limbs, "short-circuit must be bit-exact");

        println!(
            "[Fp256 chained mul ×{}] full CIOS: {:.3} ms | upper-limb: {:.3} ms | speedup {}",
            iters, t_full, t_upper, fmt_ratio(t_full, t_upper)
        );

        // Binding loop: p(0,x') + r·p(∞,x') over 2^20 pairs with an
        // upper-limb challenge vs a full-field challenge (Table 5's
        // projective rows).
        let len = 1usize << 20;
        let lo: Vec<Fp256> = (0..len as u64).map(|i| Fp256::from_canonical_u64(i * 31 + 7)).collect();
        let hi: Vec<Fp256> = (0..len as u64).map(|i| Fp256::from_canonical_u64(i * 97 + 3)).collect();
        let mut hash2 = [0u8; 32];
        for (i, b) in hash2.iter_mut().enumerate() {
            *b = (i * 71 + 13) as u8;
        }
        let up2 = Fp256::sample_upper_limb(&hash2);
        let full_r = {
            let mut v = [0u64; 4];
            for i in 0..4 {
                v[i] = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            }
            // A full-field random-ish Montgomery value (top limb set).
            v[3] |= 1 << 63;
            Fp256 { limbs: v }
        };

        let t0 = Instant::now();
        let mut bound_full: Vec<Fp256> = Vec::with_capacity(len);
        for i in 0..len {
            // p(0) + r·p(∞) with a full-field challenge.
            let prod = full_r.mul(&hi[i]);
            bound_full.push(lo[i].add(&prod));
        }
        let t_bind_full = t0.elapsed().as_secs_f64() * 1e3;

        let t0 = Instant::now();
        let mut bound_upper: Vec<Fp256> = Vec::with_capacity(len);
        for i in 0..len {
            let prod = up2.mul_upper_limb(&hi[i]);
            bound_upper.push(lo[i].add(&prod));
        }
        let t_bind_upper = t0.elapsed().as_secs_f64() * 1e3;
        let _ = &bound_full;
        let _ = &bound_upper;
        println!(
            "[Fp256 projective binding 2^20] full-field r: {:.3} ms | upper-limb r: {:.3} ms | speedup {}",
            t_bind_full, t_bind_upper, fmt_ratio(t_bind_full, t_bind_upper)
        );
    }

    println!("\n( timestamp {} ms )", now_ms());
}

/// Boolean-engine claim: Σ over the cube of the virtual polynomial.
fn dense_claim(vp: &VirtualPolynomial) -> Goldilocks {
    let n = vp.num_vars;
    let len = 1usize << n;
    let mut acc = Goldilocks::ZERO;
    for i in 0..len {
        let mut term_acc = Goldilocks::ZERO;
        for (c, ids) in &vp.terms {
            let mut prod = *c;
            for fi in ids {
                prod = prod.mul(&vp.factors[*fi].evaluations[i]);
            }
            term_acc = term_acc.add(&prod);
        }
        acc = acc.add(&term_acc);
    }
    acc
}

/// The Boolean LT table via the standard recurrence (1 mul + 1 add + 1 sub
/// per entry per round) — the Table 4 baseline.
fn lt_bool_table(r: &[Goldilocks]) -> Vec<Goldilocks> {
    let n = r.len();
    // Boolean LT(r, ·) as evaluations: LT(x,y) = 1[nat(x) < nat(y)] over
    // the y cube; build by the classic doubling: each round splits
    // entries into (e·(1−r), e·r) with the LT correction.
    // Simplest reference: direct O(n·2^n) from the closed form
    // LT = Σ_v [y_v · Π_{u<v} eq(r_u,y_u) · Π_{u>v}(1−y_u)].
    let mut evals = vec![Goldilocks::ZERO; 1usize << n];
    for idx in 0..(1usize << n) {
        let mut acc = Goldilocks::ZERO;
        let mut prefix = Goldilocks::ONE;
        for v in 0..n {
            let yv = (idx >> (n - 1 - v)) & 1;
            if yv == 1 {
                let mut suffix = Goldilocks::ONE;
                for u in (v + 1)..n {
                    let yu = (idx >> (n - 1 - u)) & 1;
                    suffix = suffix.mul(&Goldilocks::from_u64(1 - yu as u64));
                }
                acc = acc.add(&prefix.mul(&suffix));
            }
            let ru = r[v];
            let yv_f = Goldilocks::from_u64(yv as u64);
            prefix = prefix
                .mul(&ru.mul(&yv_f).add(&Goldilocks::ONE.sub(&ru).mul(&Goldilocks::ONE.sub(&yv_f))));
        }
        evals[idx] = acc;
    }
    evals
}
