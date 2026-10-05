//! The statement-growth driver benchmark: the full multi-round
//! prove/verify (coarse + fine rounds) at the driver-scale fixture.

use lattice_core::transcript::Transcript;
use lattice_rokoko::com::{ComKey, RokokoParams};
use lattice_rokoko::driver::{
    rokoko_driver_prove, rokoko_driver_verify, DriverParams,
};
use lattice_rokoko::protocol::{mat_vec, LinComInstance};
use lattice_ring::RingElement;

fn small_vec(
    ring: &lattice_ring::RingConfig,
    m: usize,
    tag: &[u8],
    span: u32,
) -> Vec<RingElement> {
    (0..m)
        .map(|i| {
            let bytes = Transcript::xof(
                b"rokoko-driver-bench",
                &[tag, &(i as u32).to_le_bytes()].concat(),
                4 * ring.n(),
            );
            let coeffs: Vec<u32> = bytes
                .chunks(4)
                .take(ring.n())
                .map(|c| {
                    let mut a = [0u8; 4];
                    a.copy_from_slice(&c[..4]);
                    u32::from_le_bytes(a) % (2 * span + 1)
                })
                .collect();
            RingElement::from_coeffs(ring, coeffs)
        })
        .collect()
}

fn main() {
    let ring = lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, 3)
        .ok()
        .unwrap();
    let params = DriverParams {
        rho_c: 64,
        rho_f: 16,
        n_rp: 2,
        n_bat: 2,
        ell: 9,
        ell_prime: 31,
        ell_fold: 8,
        beta_rp: 4,
        fine_switch: 128,
        fine_split: 512,
        terminal_m: 64,
        target_bits: 10.0,
        com_depth: 1,
    };
    let rok = RokokoParams {
        n_ring: 3,
        n0: 2,
        gadget_len: 2,
        com_depth: 1,
        r: 2,
        beta_w: 512,
    };
    let mut ck = ComKey::new(rok, [201u8; 32]);
    // the fixture: m_w = 512, r = 2, one caller block, one l-claim
    let (m_w, r, m_y) = (512, 2, 4);
    let ys: Vec<Vec<Vec<RingElement>>> = vec![(0..m_y)
        .map(|k| small_vec(&ring, r, format!("by{}", k).as_bytes(), 1))
        .collect()];
    let h = vec![vec![small_vec(&ring, m_y, b"bh0", 1)]];
    let mut f_row = vec![ring.zero(); m_w];
    f_row[0] = ring.one();
    let f = vec![vec![f_row]];
    let mut w_cols: Vec<Vec<RingElement>> = (0..r)
        .map(|col| small_vec(&ring, m_w, format!("bw{}", col).as_bytes(), 2))
        .collect();
    for col in 0..r {
        let ycol: Vec<RingElement> =
            ys[0].iter().map(|row| row[col].clone()).collect();
        let hy = lattice_salsa::ring_sc::ring_dot(&h[0][0], &ycol).ok().unwrap();
        w_cols[col][0] = hy;
    }
    let y_flat: Vec<RingElement> = ys[0].iter().flatten().cloned().collect();
    let (com, aux) =
        lattice_rokoko::com::com_commit(&mut ck, &ring, &y_flat, 1, 32).ok().unwrap();
    let ell = vec![small_vec(&ring, m_w, b"bell", 1)];
    let rr = vec![vec![ring.one(), ring.zero()]];
    let tt = vec![lattice_salsa::ring_sc::ring_dot(&ell[0], &w_cols[0]).ok().unwrap()];
    let inst = LinComInstance {
        f,
        h,
        coms: vec![com],
        aux: vec![aux],
        ell,
        rr,
        tt,
        m_w,
        r,
        beta_w: 512,
        w_cols: Some(w_cols),
        ys: Some(ys),
    };
    let _ = mat_vec;
    println!("# RoKoko statement-growth driver bench (m_w=512, r=2, 2 rounds)");
    let t0 = std::time::Instant::now();
    let mut pt = Transcript::new_default(b"lzx-rokoko-driver-bench");
    let proof = rokoko_driver_prove(&inst, &mut ck, &ring, &mut pt, &params)
        .map_err(|e| panic!("prove: {:?}", e))
        .ok()
        .unwrap();
    let prove_ms = t0.elapsed().as_millis();
    let t1 = std::time::Instant::now();
    let mut vt = Transcript::new_default(b"lzx-rokoko-driver-bench");
    let vres = rokoko_driver_verify(&inst, &mut ck, &ring, &proof, &mut vt, &params);
    let verify_ms = t1.elapsed().as_millis();
    assert!(vres.is_ok(), "verify: {:?}", vres.err());
    for rec in &proof.ledger {
        println!(
            "| round {} | {:?} | k_lin {}->{} | n {}->{} | m_w {}->{} |",
            rec.round,
            rec.kind,
            rec.k_lin_before,
            rec.k_lin_after_projection,
            rec.n_before,
            rec.n_after,
            rec.m_w_before,
            rec.m_w_after
        );
    }
    println!(
        "| prove {} ms | verify {} ms | rounds {} | parbreak min_c {:.1} bits |",
        prove_ms,
        verify_ms,
        proof.ledger.len(),
        proof.parbreak.min_classical_bits
    );
}
