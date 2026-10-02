//! The windowed engine: digit-window decompositions on the
//! `2^{h_r} × 2^{h_c}` grid with the tensor binding pass — the
//! Greyhound-style MLE evaluation argument over the Ajtai carrier
//! (follow-up (a) of `ring-lookups.md`).
//!
//! An oracle vector `v ∈ R^N` (`N = 2^{h_r+h_c}`) is laid out as the
//! GRID `S ∈ R^{2^{h_r}×2^{h_c}}` (row = the high `h_r` index bits,
//! column = the low `h_c` bits). The multilinear evaluation at
//! `r ∈ R^{log N}` factorizes through the tensor pair
//!
//! ```text
//! v̂(r) = a^T·S·b,   a = ⊗_{t≥h_c}(1−r_t, r_t) ∈ R^{2^{h_r}},
//!                  b = ⊗_{t<h_c}(1−r_t, r_t) ∈ R^{2^{h_c}},
//! ```
//!
//! exactly the §2.4 recipe of the paper (Greyhound's
//! `y = a^T·S·b` view of committed multilinear polynomials).
//!
//! Because Ajtai binding needs *short* openings while oracle entries
//! have norm up to `q/2`, each grid entry is decomposed into
//! **digit windows** (`S_{ij} = Σ_k 2^{wk}·D^{(k)}_{ij}`, slots with
//! ℓ∞ ≤ `2^w`) and the carrier commits the digit slots
//! (`lattice-lookup-ring/src/carrier.rs`).
//!
//! The **binding pass** is the Lyubashevsky-style linear-form
//! argument of the workspace's `lattice-commitment::linear_proof`,
//! grid-structured: mask `y_m` (short), `w = A·y_m`, the linear image
//! `img = ℓ(y_m) = Σ_k 2^{wk}·(a^T Y_m^{(k)} b)`, challenge
//! `c ← C` (binary ring element — the sampling space), response
//! `z = y_m + c·s` with rejection sampling, and the checks
//! `‖z‖∞ ≤ B`, `A·z = w + c·t`, `ℓ(z) = img + c·y`.
//!
//! Honest scope: the response is transmitted in the clear —
//! `O(N·K_w)` ring elements, matching the paper's PIOP proof length
//! `O(N + poly(d)·M)` — and the verifier's `A·z` recomputation is
//! linear in the slot count. Greyhound's `√N`-transmission
//! compression (the split-and-check response layer) is the documented
//! next optimization; the workspace's Serval module carries that
//! pattern for its own relation.


// (Kernel loops use explicit indices by convention.)
#![allow(clippy::needless_range_loop)]
use crate::carrier::{sample_short, CarrierCommitment, CarrierError, CarrierKey, CarrierParams};
use crate::ring_d::{Elem, RingD};
use lattice_core::transcript::Transcript;

/// The window bit width (slots bounded by `2^w`).
pub const WINDOW_BITS: u64 = 4;
/// Digit layers per ring element (`ceil(32 / w)` — q < 2^32).
pub const WINDOW_COUNT: usize = 8;

#[derive(Debug, Clone)]
pub enum WindowedError {
    Carrier(CarrierError),
    Shape(String),
    Norm,
    Retries,
    Binding,
}

impl From<CarrierError> for WindowedError {
    fn from(e: CarrierError) -> Self {
        WindowedError::Carrier(e)
    }
}

/// Decompose one ring element into `K` digit-layer elements
/// (`Σ_k 2^{wk}·D_k = e` coefficient-wise, `‖D_k‖∞ < 2^w`).
pub fn digit_windows(ring: &RingD, e: &Elem) -> Vec<Elem> {
    let mut out = Vec::with_capacity(WINDOW_COUNT);
    for _ in 0..WINDOW_COUNT {
        out.push(ring.zero());
    }
    for (pos, &c) in e.coeffs().iter().enumerate() {
        let mut rem = c;
        for k in 0..WINDOW_COUNT {
            let digit = rem & ((1u64 << WINDOW_BITS) - 1);
            out[k].c[pos] = digit;
            rem >>= WINDOW_BITS;
        }
    }
    out
}

/// Reconstruct an element from its digit layers.
pub fn digit_reconstruct(ring: &RingD, layers: &[Elem]) -> Elem {
    let mut e = ring.zero();
    for (k, layer) in layers.iter().enumerate() {
        for (pos, &c) in layer.coeffs().iter().enumerate() {
            e.c[pos] = (e.c[pos] + (c << (WINDOW_BITS * k as u64))) % ring.q;
        }
    }
    e
}

/// The grid decomposition of an oracle vector: rows `2^{h_r}`, columns
/// `2^{h_c}` (`h_r + h_c = log N`), plus digit layers per entry.
/// Returns the flattened carrier slots: `K·N` short ring elements
/// (layer-major, then row-major).
pub fn windowed_slots(ring: &RingD, v: &[Elem]) -> Result<Vec<Elem>, WindowedError> {
    let n = v.len();
    if !n.is_power_of_two() || n < 2 {
        return Err(WindowedError::Shape("vector length must be a power of two".into()));
    }
    let log_n = n.trailing_zeros() as usize;
    let h_r = log_n.div_ceil(2);
    let _h_c = log_n - h_r;
    let mut slots = Vec::with_capacity(n * WINDOW_COUNT);
    for k in 0..WINDOW_COUNT {
        for e in v {
            let layers = digit_windows(ring, e);
            slots.push(layers[k].clone());
        }
    }
    Ok(slots)
}

/// Reconstruct the oracle vector from the carrier slots.
pub fn slots_to_vector(ring: &RingD, slots: &[Elem], n: usize) -> Result<Vec<Elem>, WindowedError> {
    if slots.len() != n * WINDOW_COUNT {
        return Err(WindowedError::Shape("slot count mismatch".into()));
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let layers: Vec<Elem> = (0..WINDOW_COUNT).map(|k| slots[k * n + i].clone()).collect();
        v.push(digit_reconstruct(ring, &layers));
    }
    Ok(v)
}

/// The tensor pair `(a, b)` for an evaluation point: `a` over the HIGH
/// `h_r` bits' challenges, `b` over the LOW `h_c`.
pub fn grid_tensors(ring: &RingD, point: &[Elem]) -> (Vec<Elem>, Vec<Elem>) {
    let log_n = point.len();
    let h_r = log_n.div_ceil(2);
    let h_c = log_n - h_r;
    // a_i = ∏_{t in high} (i_t ? point[t] : 1−point[t]) with i over
    // the high-bit cube; b similarly over the low bits.
    let a = tensor_over(ring, &point[h_c..]);
    let b = tensor_over(ring, &point[..h_c]);
    (a, b)
}

fn tensor_over(ring: &RingD, challenges: &[Elem]) -> Vec<Elem> {
    let mut acc = vec![ring.one()];
    for ch in challenges {
        let one_minus = ring.sub(&ring.one(), ch);
        let mut next = Vec::with_capacity(acc.len() * 2);
        for e in &acc {
            next.push(ring.mul(e, &one_minus));
        }
        for e in &acc {
            next.push(ring.mul(e, ch));
        }
        acc = next;
    }
    acc
}

/// The linear-form coefficient at slot `(k, i, j)`:
/// `c_{kij} = a_i·b_j·2^{wk}`.
fn linear_coefficient(ring: &RingD, a: &[Elem], b: &[Elem], k: usize, i: usize, j: usize) -> Elem {
    let base = ring.mul(&a[i], &b[j]);
    let weight = (1u64 << (WINDOW_BITS * k as u64)) % ring.q;
    ring.scale(&base, weight)
}

/// The linear image `ℓ(slots) = Σ_{k,i,j} a_i b_j 2^{wk}·slots[k·N+i·C+j]`
/// — the grid-structured evaluation of the digit reconstruction.
pub fn linear_image(ring: &RingD, slots: &[Elem], point: &[Elem]) -> Result<Elem, WindowedError> {
    let n = slots.len() / WINDOW_COUNT;
    let (a, b) = grid_tensors(ring, point);
    let log_n = n.trailing_zeros() as usize;
    let h_r = log_n.div_ceil(2);
    let h_c = log_n - h_r;
    let rows = 1usize << h_r;
    let cols = 1usize << h_c;
    let mut acc = ring.zero();
    for k in 0..WINDOW_COUNT {
        for i in 0..rows {
            for j in 0..cols {
                let slot = &slots[k * n + i * cols + j];
                if slot.is_zero() {
                    continue;
                }
                let c = linear_coefficient(ring, &a, &b, k, i, j);
                acc = ring.add(&acc, &ring.mul(&c, slot));
            }
        }
    }
    Ok(acc)
}

/// A windowed evaluation proof (the binding pass artifact).
#[derive(Clone, Debug)]
pub struct WindowedEvalProof {
    pub mask_commitment: CarrierCommitment,
    pub mask_image: Elem,
    pub challenge: Elem,
    pub response: Vec<Elem>,
    pub retries: u32,
}

pub const MAX_RETRIES: u32 = 64;

/// Derive the challenge from the statement (the wave-3 FS-ordering
/// discipline: mask commitment and image absorbed BEFORE the
/// challenge).
fn derive_challenge(
    ring: &RingD,
    key: &CarrierKey,
    commitment: &CarrierCommitment,
    point: &[Elem],
    y: &Elem,
    mask: &CarrierCommitment,
    image: &Elem,
) -> Elem {
    let mut tr = Transcript::new_default(b"windowed-eval");
    let _ = tr.append_bytes(b"key-seed", &key.seed);
    let _ = tr.append_bytes(b"commitment", &commitment.to_bytes());
    for p in point {
        let _ = tr.append_bytes(b"point", &elem_bytes(ring, p));
    }
    let _ = tr.append_bytes(b"claim", &elem_bytes(ring, y));
    let _ = tr.append_bytes(b"mask", &mask.to_bytes());
    let _ = tr.append_bytes(b"image", &elem_bytes(ring, image));
    ring.sample_challenge(&mut tr, b"we-chal")
}

fn elem_bytes(ring: &RingD, e: &Elem) -> Vec<u8> {
    let mut buf = Vec::with_capacity(ring.d * 8);
    for &c in e.coeffs() {
        buf.extend_from_slice(&c.to_le_bytes());
    }
    buf
}

/// Prove `v̂(point) = y` for the committed oracle vector (the binding
/// pass). `slots` and `commitment` come from
/// [`windowed_slots`] + `CarrierKey::commit`.
#[allow(clippy::too_many_lines)]
pub fn prove_windowed_eval(
    ring: &RingD,
    key: &CarrierKey,
    slots: &[Elem],
    point: &[Elem],
    y: &Elem,
    commitment: &CarrierCommitment,
    prover_seed: &[u8],
) -> Result<WindowedEvalProof, WindowedError> {
    // fail-closed: the claim must hold
    let img_check = linear_image(ring, slots, point)?;
    if img_check != *y {
        return Err(WindowedError::Binding);
    }
    let bound = key.params.norm_bound;
    // rejection-sampled Lyubashevsky rounds
    for retry in 0..=MAX_RETRIES {
        let mut salt = Vec::with_capacity(prover_seed.len() + 8);
        salt.extend_from_slice(prover_seed);
        salt.extend_from_slice(&(retry as u64).to_le_bytes());
        let mask = sample_short(ring, key.params.m, bound / 4, &salt);
        let mask_commitment = key.commit(&mask)?;
        let image = linear_image(ring, &mask, point)?;
        let challenge = derive_challenge(ring, key, commitment, point, y, &mask_commitment, &image);
        // z = mask + c·slots
        let mut z = Vec::with_capacity(key.params.m);
        let mut ok = true;
        for (mi, si) in mask.iter().zip(slots.iter()) {
            let cs = ring.mul(&challenge, si);
            let zi = ring.add(mi, &cs);
            if zi.inf_norm(ring.q) > bound {
                ok = false;
                break;
            }
            z.push(zi);
        }
        if !ok {
            continue;
        }
        return Ok(WindowedEvalProof {
            mask_commitment,
            mask_image: image,
            challenge,
            response: z,
            retries: retry,
        });
    }
    Err(WindowedError::Retries)
}

/// Verify the binding pass: norms, the Ajtai identity `A·z = w + c·t`,
/// and the linear image `ℓ(z) = img + c·y`.
pub fn verify_windowed_eval(
    ring: &RingD,
    key: &CarrierKey,
    commitment: &CarrierCommitment,
    point: &[Elem],
    y: &Elem,
    proof: &WindowedEvalProof,
) -> Result<(), WindowedError> {
    if proof.response.len() != key.params.m {
        return Err(WindowedError::Carrier(CarrierError::Dimension {
            expected: key.params.m,
            got: proof.response.len(),
        }));
    }
    for z in &proof.response {
        if z.inf_norm(ring.q) > key.params.norm_bound {
            return Err(WindowedError::Norm);
        }
    }
    // FS replay
    let expected_chal =
        derive_challenge(ring, key, commitment, point, y, &proof.mask_commitment, &proof.mask_image);
    if expected_chal != proof.challenge {
        return Err(WindowedError::Binding);
    }
    // A·z = w + c·t
    let az = key.commit(&proof.response)?;
    for (i, az_i) in az.rows.iter().enumerate() {
        let ct = ring.mul(&proof.challenge, &commitment.rows[i]);
        let expect = ring.add(&proof.mask_commitment.rows[i], &ct);
        if *az_i != expect {
            return Err(WindowedError::Binding);
        }
    }
    // ℓ(z) = img + c·y
    let lz = linear_image(ring, &proof.response, point)?;
    let cy = ring.mul(&proof.challenge, y);
    let expect = ring.add(&proof.mask_image, &cy);
    if lz != expect {
        return Err(WindowedError::Binding);
    }
    Ok(())
}

/// Convenience: commit an oracle vector's digit windows and return
/// `(slots, commitment)`.
pub fn windowed_commit(
    key: &CarrierKey,
    v: &[Elem],
) -> Result<(Vec<Elem>, CarrierCommitment), WindowedError> {
    let ring = &key.params.ring;
    let slots = windowed_slots(ring, v)?;
    let commitment = key.commit(&slots)?;
    Ok((slots, commitment))
}

/// The carrier parameters for a given oracle-vector length.
pub fn carrier_params_for(ring: &RingD, n: usize, k: usize, bound: u64) -> CarrierParams {
    CarrierParams {
        ring: ring.clone(),
        k,
        m: n * WINDOW_COUNT,
        norm_bound: bound,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingD {
        RingD::new(4).ok().unwrap()
    }

    #[test]
    fn digit_windows_roundtrip() {
        let r = ring();
        for seed in ["a", "b", "c"] {
            let e = r.random(seed.as_bytes());
            let layers = digit_windows(&r, &e);
            assert_eq!(layers.len(), WINDOW_COUNT);
            for l in &layers {
                assert!(l.inf_norm(r.q) < (1 << WINDOW_BITS));
            }
            assert_eq!(digit_reconstruct(&r, &layers), e);
        }
    }

    #[test]
    fn grid_slots_roundtrip() {
        let r = ring();
        let v: Vec<Elem> = (0..8).map(|i| r.random(format!("g{i}").as_bytes())).collect();
        let slots = windowed_slots(&r, &v).ok().unwrap();
        assert_eq!(slots.len(), 8 * WINDOW_COUNT);
        let back = slots_to_vector(&r, &slots, 8).ok().unwrap();
        assert_eq!(back, v);
    }

    #[test]
    fn linear_image_equals_mle_eval() {
        let r = ring();
        let n = 16;
        let v: Vec<Elem> = (0..n).map(|i| r.random(format!("li{i}").as_bytes())).collect();
        let slots = windowed_slots(&r, &v).ok().unwrap();
        let mut tr = Transcript::new_default(b"pt");
        let point: Vec<Elem> = (0..4).map(|i| r.sample_challenge(&mut tr, format!("p{i}").as_bytes())).collect();
        // the linear image over digit slots must equal the direct MLE eval
        let li = linear_image(&r, &slots, &point).ok().unwrap();
        let mle = r.mle_eval(&v, &point).ok().unwrap();
        assert_eq!(li, mle);
    }

    #[test]
    fn windowed_eval_end_to_end() {
        let r = ring();
        let n = 8;
        let v: Vec<Elem> = (0..n).map(|i| r.random(format!("we{i}").as_bytes())).collect();
        let params = carrier_params_for(&r, n, 2, 1 << 14);
        let key = CarrierKey::from_seed(params, [11u8; 32]);
        let (slots, commitment) = windowed_commit(&key, &v).ok().unwrap();
        let mut tr = Transcript::new_default(b"pt2");
        let point: Vec<Elem> =
            (0..3).map(|i| r.sample_challenge(&mut tr, format!("q{i}").as_bytes())).collect();
        let y = r.mle_eval(&v, &point).ok().unwrap();
        let proof =
            prove_windowed_eval(&r, &key, &slots, &point, &y, &commitment, b"seed").ok().unwrap();
        verify_windowed_eval(&r, &key, &commitment, &point, &y, &proof).ok().unwrap();
        // wrong claim rejected
        let y_bad = r.add(&y, &r.one());
        assert!(prove_windowed_eval(&r, &key, &slots, &point, &y_bad, &commitment, b"s").is_err());
        // tampered response rejected
        let mut bad = proof.clone();
        bad.response[0] = r.add(&bad.response[0], &r.one());
        assert!(verify_windowed_eval(&r, &key, &commitment, &point, &y, &bad).is_err());
        // tampered commitment rejected
        let (proof2, commitment2) = {
            let (slots2, c2) = windowed_commit(&key, &v).ok().unwrap();
            let p2 = prove_windowed_eval(&r, &key, &slots2, &point, &y, &c2, b"s2").ok().unwrap();
            (p2, c2)
        };
        let mut c_bad = commitment2.clone();
        c_bad.rows[0] = r.add(&c_bad.rows[0], &r.one());
        assert!(verify_windowed_eval(&r, &key, &c_bad, &point, &y, &proof2).is_err());
    }

    #[test]
    fn wrong_point_rejected() {
        let r = ring();
        let n = 4;
        let v: Vec<Elem> = (0..n).map(|i| r.random(format!("wp{i}").as_bytes())).collect();
        let params = carrier_params_for(&r, n, 1, 1 << 14);
        let key = CarrierKey::from_seed(params, [5u8; 32]);
        let (slots, commitment) = windowed_commit(&key, &v).ok().unwrap();
        let mut tr = Transcript::new_default(b"pt3");
        let point: Vec<Elem> =
            (0..2).map(|i| r.sample_challenge(&mut tr, format!("q{i}").as_bytes())).collect();
        let other: Vec<Elem> =
            (0..2).map(|i| r.sample_challenge(&mut tr, format!("o{i}").as_bytes())).collect();
        let y = r.mle_eval(&v, &point).ok().unwrap();
        let proof =
            prove_windowed_eval(&r, &key, &slots, &point, &y, &commitment, b"seed").ok().unwrap();
        assert!(verify_windowed_eval(&r, &key, &commitment, &other, &y, &proof).is_err());
    }
}
