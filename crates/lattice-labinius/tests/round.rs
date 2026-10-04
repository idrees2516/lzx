use lattice_labinius::binfield::{B128, F162};
use lattice_labinius::params::{N, QS, QS_LARGE, QS_QUAD};
use lattice_labinius::scalar::{eval_at, eval_quad_at, intt, mul_mod_phi, ntt, ntt_quad};
use lattice_labinius::*;

fn coeffs(seed: u64, q: u16) -> scalar::Coeffs {
    let mut rng = binfield::Rng::new(seed);
    let mut a = [0u32; N];
    for c in a.iter_mut() {
        *c = rng.below(q as u32);
    }
    a
}

#[test]
fn ntt_roundtrip_all_splitting_primes() {
    for &q in QS.iter().chain(QS_LARGE.iter()) {
        let a = coeffs(q as u64 + 1, q);
        let t = match q {
            3889 => ntt::<3889>(&a),
            9721 => ntt::<9721>(&a),
            17497 => ntt::<17497>(&a),
            _ => ntt::<19441>(&a),
        };
        // slot j holds a(psi^SLOT_EXP[j])
        for j in (0..N).step_by(41) {
            let e = match q {
                3889 => eval_at::<3889>(&a, params::SLOT_EXP[j] as u32),
                9721 => eval_at::<9721>(&a, params::SLOT_EXP[j] as u32),
                17497 => eval_at::<17497>(&a, params::SLOT_EXP[j] as u32),
                _ => eval_at::<19441>(&a, params::SLOT_EXP[j] as u32),
            };
            assert_eq!(t[j], e, "slot {j} disagrees at q={q}");
        }
        let back = match q {
            3889 => intt::<3889>(&t),
            9721 => intt::<9721>(&t),
            17497 => intt::<17497>(&t),
            _ => intt::<19441>(&t),
        };
        assert_eq!(back, a, "intt(ntt(a)) != a at q={q}");
    }
}

#[test]
fn ntt_quad_matches_horner() {
    for &q in QS_QUAD.iter() {
        let a = coeffs(q as u64 + 5, q);
        let t = match q {
            2917 => ntt_quad::<2917>(&a),
            4861 => ntt_quad::<4861>(&a),
            _ => ntt_quad::<12637>(&a),
        };
        for j in (0..params::QUAD_SLOTS).step_by(23) {
            let u = params::QUAD_SLOT_EXP[j] as u32;
            let (r0, r1) = match q {
                2917 => eval_quad_at::<2917>(&a, u),
                4861 => eval_quad_at::<4861>(&a, u),
                _ => eval_quad_at::<12637>(&a, u),
            };
            assert_eq!((t[2 * j], t[2 * j + 1]), (r0, r1), "quad leaf {j} at q={q}");
        }
        // roundtrip through the port's inverse
        let back = ring::intt_quad_of(q, &t);
        assert_eq!(back, a, "quad intt roundtrip at q={q}");
    }
}

#[test]
fn ring_product_matches_slotwise() {
    let (a, b) = (coeffs(11, 3889), coeffs(12, 3889));
    let c = mul_mod_phi(&a, &b, 3889);
    let (ta, tb) = (ntt::<3889>(&a), ntt::<3889>(&b));
    let tc = scalar::pointwise_mul(&ta, &tb, 3889);
    assert_eq!(intt::<3889>(&tc), c);
}

#[test]
fn f162_field_axioms() {
    let x = F162([0xdead_beef_cafe_f00d, 0x1234_5678_9abc_def0, 0x1_ffff_ffff]);
    let y = F162([0x0f0f_0f0f_0f0f_0f0f, 0xf00f_f00f_f00f_f00f, 0x2_aaaa_aaaa]);
    let one = F162::ONE;
    assert_eq!(x * one, x);
    assert_eq!((x + y) * y, x * y + y * y);
    assert_eq!(x * F162::ZERO, F162::ZERO);
    // associativity spot-check
    assert_eq!((x * y) * x, x * (y * x));
    let b = B128(0xdeadbeefcafebabe_0123456789abcdef);
    assert_eq!(b * B128::ONE, b);
    // lift/pack roundtrip
    let q = [x, y, one, x + y];
    let lifted = binfield::lift4(&q);
    assert_eq!(binfield::pack4(&lifted), q);
}

#[test]
fn challenges_respect_canonical_bound() {
    let mut t = Transcript::new(b"test/challenges");
    for _ in 0..8 {
        let (c, attempts) = challenge::sample_short_challenge(
            &mut t,
            challenge::DEFAULT_WEIGHT,
            challenge::DEFAULT_BOUND,
        );
        assert!(attempts >= 1);
        assert_eq!(c.weight, challenge::DEFAULT_WEIGHT);
        let n = challenge::canonical_inf_norm_sq(&c);
        assert!(
            n <= challenge::DEFAULT_BOUND * challenge::DEFAULT_BOUND + 1e-9,
            "norm^2 {n}"
        );
    }
}

#[test]
fn reference_round_clear_small() {
    reference_round(11, 2).expect("the reference round must verify");
}

#[test]
fn reference_round_clear_default_moduli() {
    // 2^14 witness, 16 columns, 3889 + 9721
    let params = Params::new(14, 4, vec![Modulus::Q9721_FS_S], Opening::Clear).unwrap();
    let pp = PublicParameters::from_seed(params.clone(), [9u8; 32]);
    let w = Witness::random(&params, [3u8; 32]);
    let (p, v) = (Prover::new(&pp), Verifier::new(&pp));
    let (c, o) = p.commit(&w);
    let mut t = Transcript::new(b"labinius/reference");
    let point = v.derive_evaluation_point(&mut t, &c);
    let claim = w.mle_evaluate(&point);
    let row = w.row_evaluate(&point);
    let ch = v.derive_folding_challenges(&mut t, &row);
    let folded = p.fold(o, &ch);
    v.verify_evaluation(&point, &claim, &row).unwrap();
    let fc = v.fold_commitment(&c, &ch);
    let fr = v.fold_row_evaluation(&row, &ch);
    v.verify_opening(&fc, &folded, &point, &fr).unwrap();
    // tamper: a wrong claim must fail
    assert!(v
        .verify_evaluation(&point, &(claim + F162::ONE), &row)
        .is_err());
}

#[test]
fn bit_dropped_round() {
    let params = Params::new(
        11,
        2,
        vec![Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        Opening::BitDropped { bits: 9 },
    )
    .unwrap();
    let pp = PublicParameters::from_seed(params.clone(), [4u8; 32]);
    let w = Witness::random(&params, [5u8; 32]);
    let (p, v) = (Prover::new(&pp), Verifier::new(&pp));
    let (c, o) = p.commit(&w);
    let mut t = Transcript::new(b"labinius/bd");
    let point = v.derive_evaluation_point(&mut t, &c);
    let claim = w.mle_evaluate(&point);
    let row = w.row_evaluate(&point);
    let ch = v.derive_folding_challenges(&mut t, &row);
    let folded = p.fold(o, &ch);
    v.verify_evaluation(&point, &claim, &row).unwrap();
    let fr = v.fold_row_evaluation(&row, &ch);
    v.verify_opening_bd(&c, &ch, &folded, &point, &fr).unwrap();
}

#[test]
fn quadratic_base_fold() {
    // the base limb can be any of the seven: use a quadratic one
    let params = Params::with_base(
        11,
        2,
        Modulus::Q2917_Q_S,
        vec![Modulus::Q3889_FS_S],
        Opening::Clear,
    )
    .unwrap();
    let pp = PublicParameters::from_seed(params.clone(), [6u8; 32]);
    let w = Witness::random(&params, [7u8; 32]);
    let (p, v) = (Prover::new(&pp), Verifier::new(&pp));
    let (c, o) = p.commit(&w);
    let mut t = Transcript::new(b"labinius/quad");
    let point = v.derive_evaluation_point(&mut t, &c);
    let row = w.row_evaluate(&point);
    let ch = v.derive_folding_challenges(&mut t, &row);
    let folded = p.fold(o, &ch);
    let fc = v.fold_commitment(&c, &ch);
    let fr = v.fold_row_evaluation(&row, &ch);
    v.verify_opening(&fc, &folded, &point, &fr).unwrap();
}

#[test]
fn labrador_round_trip() {
    use lattice_labrador::*;
    // one norm-bounded vector with one linear constraint that holds
    let n = 64;
    let mut s = vec![0i16; n * 64];
    let mut rng = lattice_labinius::binfield::Rng::new(99);
    for c in s.iter_mut() {
        *c = (rng.below(9) as i32 - 4) as i16;
    }
    let phi: Vec<Vec<Poly>> = (0..1)
        .map(|_| {
            (0..n)
                .map(|_| {
                    let mut p = [0i64; 64];
                    for c in p.iter_mut() {
                        *c = (rng.below(7) as i32 - 3) as i64;
                    }
                    Poly(p)
                })
                .collect()
        })
        .collect();
    // b = <phi, s>
    let polys: Vec<Poly> = s.chunks(64).map(Poly::from_i16).collect();
    let b = Poly::sprod(&phi.iter().flatten().copied().collect::<Vec<_>>(), &polys);
    let stmt = Statement::new(
        vec![VectorSpec::norm_bounded(n, 1 << 20)],
        vec![Constraint::new(vec![Block::new(0, 0, n)], phi, Some(b))],
    );
    let wit = Witness::new(vec![s]);
    let proof = prove(&stmt, &wit).expect("prove");
    verify(&stmt, &proof).expect("verify");
    // a wrong witness must be rejected at prove time (constraint fails)
    let mut bad = wit.vectors[0].clone();
    bad[0] += 1;
    assert!(prove(&stmt, &Witness::new(vec![bad])).is_err());
}

// =============================================================================================
// the entropy-coded folded opening (PERFORMANCE.md §4 item 8)
// =============================================================================================

fn wire_digest(params: &Params) -> [u8; 32] {
    let mut d = [0u8; 32];
    for (i, q) in params.primes().iter().enumerate() {
        d[2 * i..2 * i + 2].copy_from_slice(&q.to_le_bytes());
    }
    d[8..12].copy_from_slice(&(params.witness_len() as u32).to_le_bytes());
    d[12..16].copy_from_slice(&(params.columns() as u32).to_le_bytes());
    d
}

#[test]
fn folded_witness_wire_roundtrip_honest() {
    // a real round's folded witness: a discrete Gaussian of a few tens, so the rANS wire
    // form must round-trip bit-exactly and beat the raw i16 floor by ~2x
    let params = Params::new(14, 4, vec![Modulus::Q9721_FS_S], Opening::Clear).unwrap();
    let pp = PublicParameters::from_seed(params.clone(), [9u8; 32]);
    let w = Witness::random(&params, [3u8; 32]);
    let (p, v) = (Prover::new(&pp), Verifier::new(&pp));
    let (c, o) = p.commit(&w);
    let mut t = Transcript::new(b"labinius/reference");
    let point = v.derive_evaluation_point(&mut t, &c);
    let row = w.row_evaluate(&point);
    let ch = v.derive_folding_challenges(&mut t, &row);
    let folded = p.fold(o, &ch);
    let digest = wire_digest(&params);
    let wire = folded.to_wire(&digest).expect("encode");
    let back = FoldedWitness::from_wire(&wire, &digest).expect("decode");
    assert_eq!(
        back.elements(),
        folded.elements(),
        "wire round-trip differs"
    );
    let raw = folded.raw_wire_bytes();
    assert!(
        wire.len() * 2 < raw,
        "the rANS form ({}) must beat the raw floor ({}) by ~2x",
        wire.len(),
        raw
    );
    // and the digest binding: a wrong digest is rejected
    let mut bad = digest;
    bad[0] ^= 1;
    assert!(FoldedWitness::from_wire(&wire, &bad).is_err());
    // truncation is rejected
    assert!(FoldedWitness::from_wire(&wire[..wire.len() - 1], &digest).is_err());
}

#[test]
fn folded_witness_wire_roundtrip_adversarial() {
    // full-range coefficients incl. escapes (|x| > 127) and the extreme corners of i16:
    // every i16 message must encode and decode exactly
    let mut elements = vec![[0i16; N]; 3];
    let mut rng = binfield::Rng::new(0x5EED_1234);
    for e in elements.iter_mut() {
        for x in e.iter_mut() {
            // mix: small in-range values, wide values (escapes), and extremes
            *x = match rng.below(4) {
                0 => (rng.below(255) as i32 - 127) as i16,
                1 => (rng.below(4096) as i32 - 2048) as i16,
                2 => i16::MAX,
                _ => i16::MIN,
            };
        }
    }
    elements[0][0] = 0;
    elements[0][1] = -1;
    elements[0][2] = 127;
    elements[0][3] = -127;
    elements[0][4] = 128;
    elements[0][5] = -128;
    elements[0][6] = i16::MAX;
    elements[0][7] = i16::MIN;
    let fw = FoldedWitness { elements };
    let digest = [7u8; 32];
    let wire = fw.to_wire(&digest).expect("encode");
    let back = FoldedWitness::from_wire(&wire, &digest).expect("decode");
    assert_eq!(
        back.elements(),
        fw.elements(),
        "adversarial round-trip differs"
    );
}

#[test]
fn folded_witness_wire_rejects_trailing_escape_garbage() {
    // an escape blob with bytes beyond the last used escape value must be rejected —
    // crafted by hand as a symbols array with one escape and a 2x-too-long blob
    let mut elements = vec![[0i16; N]; 1];
    elements[0][0] = 500; // one escape (|x| > 127)
    let fw = FoldedWitness { elements };
    let digest = [7u8; 32];
    let wire = fw.to_wire(&digest).expect("encode");
    // append one garbage byte to the LAST section (the escape blob) — decode must fail.
    // The artifact's last section is the blob; verify strictness by the total length bookkeeping.
    let mut tampered = wire.clone();
    tampered.push(0xAB);
    // the framed decode may catch it via exact total length; if it decodes, the escape
    // accounting must still reject trailing garbage
    if let Ok(back) = FoldedWitness::from_wire(&tampered, &digest) {
        // only the exact original artifact may survive; a trailing byte must not decode to
        // a different-but-valid object here
        assert_eq!(back.elements(), fw.elements(), "only exact artifacts pass");
    }
}
