//! KAT manifest generator (audit gate G8 prep): emits a JSON manifest
//! of known-answer digests for the core cryptographic components.
//!
//! Every vector is deterministic (fixed seeds via
//! `SecretSeed::from_kat_label` / fixed transcript domains), so any
//! change to field arithmetic, Keccak, the transcript, NTT, packing,
//! or commitment derivation changes the manifest — the release
//! artifact must record this manifest's digest.

use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};

fn hex32(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn field_kats() -> Vec<(String, String)> {
    let mut out = Vec::new();
    // Arithmetic identities on canonical edge values.
    let edge = [
        Goldilocks::ZERO,
        Goldilocks::ONE,
        Goldilocks::from_u64(u64::MAX - 1),
        Goldilocks::from_u64(0x7FFF_FFFF_0000_0000),
    ];
    for (i, a) in edge.iter().enumerate() {
        for (j, b) in edge.iter().enumerate() {
            let add = a.add(b);
            let sub = a.sub(b);
            let mul = a.mul(b);
            let bytes = Transcript::hash_domain(
                b"kat-field",
                &[add.to_bytes(), sub.to_bytes(), mul.to_bytes()].concat(),
            );
            out.push((format!("field-ops-{i}-{j}"), hex32(&bytes)));
        }
    }
    out
}

fn transcript_kats() -> Vec<(String, String)> {
    let mut out = Vec::new();
    // Deterministic challenge sequence digest.
    let mut t = Transcript::new_default(b"lzx-kat-transcript");
    t.append_message(b"a", b"kat").ok();
    let chals = t.challenge_fields(b"r", 16).ok().unwrap_or_default();
    let mut bytes = Vec::new();
    for c in &chals {
        bytes.extend_from_slice(&c.to_bytes());
    }
    out.push((
        "transcript-challenges".into(),
        hex32(&Transcript::hash_domain(b"kat-chals", &bytes)),
    ));
    out.push((
        "keccak-sha3-256".into(),
        hex32(&Transcript::hash_domain(b"kat", b"abc")),
    ));
    out
}

fn ntt_kats() -> Vec<(String, String)> {
    use lattice_ring::ntt::NttTables;
    use lattice_ring::{Modulus32, RingConfig, RingElement};
    let mut out = Vec::new();
    for &log_n in &[4u32, 6, 8] {
        let cfg = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let n = 1usize << log_n;
        let bytes = Transcript::xof(b"kat-ntt", &[log_n as u8], n * 4);
        let coeffs: Vec<u32> = bytes
            .chunks(4)
            .take(n)
            .map(|c| {
                let mut a = [0u8; 4];
                a.copy_from_slice(&c[..4.min(c.len())]);
                u32::from_le_bytes(a) % Modulus32::Q_32.q
            })
            .collect();
        let elem = RingElement::from_coeffs(&cfg, coeffs);
        let tables = NttTables::new(Modulus32::Q_32, log_n).ok().unwrap();
        let mut v = elem.coeffs().to_vec();
        tables.forward_fast(&mut v).ok().unwrap();
        let mut fb: Vec<u8> = Vec::new();
        for c in &v {
            fb.extend_from_slice(&c.to_le_bytes());
        }
        out.push((
            format!("ntt-forward-logn{log_n}"),
            hex32(&Transcript::hash_domain(b"kat-ntt-fwd", &fb)),
        ));
        // Roundtrip.
        tables.inverse_fast(&mut v).ok().unwrap();
        let mut ib: Vec<u8> = Vec::new();
        for c in &v {
            ib.extend_from_slice(&c.to_le_bytes());
        }
        out.push((
            format!("ntt-roundtrip-logn{log_n}"),
            hex32(&Transcript::hash_domain(b"kat-ntt-inv", &ib)),
        ));
    }
    out
}

fn packing_kats() -> Vec<(String, String)> {
    use lattice_ring::packing::pack_field_elements;
    use lattice_ring::{Modulus32, RingConfig};
    let cfg = RingConfig::new(Modulus32::Q_32, 4).ok().unwrap();
    let bytes = Transcript::xof(b"kat-pack", b"v", 64 * 8);
    let values: Vec<Goldilocks> = (0..64)
        .map(|i| {
            Goldilocks::from_u64(u64::from_le_bytes(
                bytes[i * 8..(i + 1) * 8].try_into().unwrap_or([0u8; 8]),
            ))
        })
        .collect();
    let packed = pack_field_elements(&cfg, &values);
    let mut pb = Vec::new();
    for e in &packed {
        pb.extend_from_slice(&e.to_bytes());
    }
    vec![(
        "split-packing".into(),
        hex32(&Transcript::hash_domain(b"kat-pack", &pb)),
    )]
    // (roundtrip verified by the in-crate tests; the manifest pins the
    // packed representation.)
}

fn commitment_kats() -> Vec<(String, String)> {
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_ring::{Modulus32, RingConfig};
    let mut out = Vec::new();
    for &(log_n, m) in &[(4u32, 8usize), (6, 8)] {
        let params = AjtaiParams {
            ring: RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap(),
            k: 2,
            m,
            norm_bound: 1 << 23,
        };
        let pk = AjtaiPublicKey::from_seed(params, [0xAB; 32]).ok().unwrap();
        let s = lattice_commitment::ajtai::sample_small_secret(
            &pk.params.ring,
            m,
            1 << 12,
            b"kat-secret",
        );
        let t = pk.commit(&s).ok().unwrap();
        out.push((
            format!("ajtai-commitment-logn{log_n}-m{m}"),
            hex32(&Transcript::hash_domain(b"kat-ajtai", &t.to_bytes())),
        ));
    }
    out
}

fn zk_kats() -> Vec<(String, String)> {
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_ring::{Modulus32, RingConfig};
    use lattice_zk::entropy::{SecretSeed, ShakeStream};
    use lattice_zk::zk_sumcheck::{zk_prove, ZkSumcheckStatement};
    let num_vars = 3usize;
    let n = 1usize << num_vars;
    let m = lattice_zk::zk_sumcheck::required_slots(n);
    let params = AjtaiParams {
        ring: RingConfig::new(Modulus32::Q_32, 4).ok().unwrap(),
        k: 2,
        m,
        norm_bound: 1 << 24,
    };
    let pk = AjtaiPublicKey::from_seed(params, [0x5A; 32]).ok().unwrap();
    let bytes = Transcript::xof(b"kat-zk", b"f", n * 8);
    let f: Vec<Goldilocks> = (0..n)
        .map(|i| {
            Goldilocks::from_u64(u64::from_le_bytes(
                bytes[i * 8..(i + 1) * 8].try_into().unwrap_or([0u8; 8]),
            ))
        })
        .collect();
    let claim = f.iter().fold(Goldilocks::ZERO, |a, b| a.add(b));
    let mut stream = ShakeStream::new(SecretSeed::from_kat_label(b"kat-zk-proof"), b"m");
    let mut t = Transcript::new_default(b"lzx-zk-sumcheck");
    if let Ok((proof, _r, _v)) = zk_prove(&pk, &f, claim, &mut stream, &mut t) {
        let mut pb = Vec::new();
        pb.extend_from_slice(&proof.commitment.to_bytes());
        for [a, b] in &proof.rounds {
            pb.extend_from_slice(&a.to_bytes());
            pb.extend_from_slice(&b.to_bytes());
        }
        pb.extend_from_slice(&proof.blinded_final.to_bytes());
        return vec![(
            "zk-sumcheck-proof".into(),
            hex32(&Transcript::hash_domain(b"kat-zk-proof", &pb)),
        )];
    }
    let _ = ZkSumcheckStatement { num_vars, claim };
    vec![]
}

fn mle_kats() -> Vec<(String, String)> {
    let mle = DenseMle::random(5, b"kat-mle");
    let point: Vec<Goldilocks> = (1..=5)
        .map(|i| Goldilocks::from_u64((i * 7919) as u64))
        .collect();
    let eval = mle.evaluate(&point).ok().unwrap();
    vec![(
        "mle-evaluation".into(),
        hex32(&Transcript::hash_domain(b"kat-mle", &eval.to_bytes())),
    )]
}

fn main() {
    let mut manifest = String::from("{\n  \"lzx_kat_manifest_v1\": {\n");
    let groups: [(&str, Vec<(String, String)>); 7] = [
        ("field", field_kats()),
        ("transcript", transcript_kats()),
        ("ntt", ntt_kats()),
        ("packing", packing_kats()),
        ("commitment", commitment_kats()),
        ("zk", zk_kats()),
        ("mle", mle_kats()),
    ];
    let mut total = 0usize;
    for (gi, (group, entries)) in groups.iter().enumerate() {
        manifest.push_str(&format!("    \"{group}\": {{\n"));
        for (i, (name, digest)) in entries.iter().enumerate() {
            manifest.push_str(&format!("      \"{name}\": \"{digest}\""));
            if i + 1 < entries.len() {
                manifest.push(',');
            }
            manifest.push('\n');
            total += 1;
        }
        manifest.push_str("    }");
        if gi + 1 < groups.len() {
            manifest.push(',');
        }
        manifest.push('\n');
    }
    manifest.push_str("  },\n");
    manifest.push_str(&format!("  \"vectors\": {total}\n}}\n"));
    let _ = std::fs::create_dir_all("target/kat");
    let digest = Transcript::hash_domain(b"kat-manifest", manifest.as_bytes());
    println!("{manifest}");
    println!("manifest digest: {}", hex32(&digest));
    let _ = std::fs::write("target/kat/lzx-kat-manifest.json", manifest);
    let _ = std::fs::write("target/kat/lzx-kat-manifest.digest", hex32(&digest));
    println!("Artifacts: target/kat/lzx-kat-manifest.json (.digest)");
}
