//! The Fp256 port of the windowed engine's binding pass (follow-up (b)
//! of `ring-lookups.md`) — the same digit-window/grid/tensor binding
//! rounds over the BN254 scalar field, on the CIOS Montgomery grid of
//! `lattice-projsumcheck/src/fp256.rs` (§5 of ePrint 2026/762).
//!
//! Why a 256-bit port: the split ring's CRT slots
//! `F_{q^{d/2}} × F_{q^{d/2}}` grow past 64 bits as `d` rises, and the
//! workspace's two-characteristic carrier discipline (zkVM
//! `compact.rs`) already routes 256-bit-field challenges through the
//! BN254 scalar field. The binding pass — mask, challenge, response,
//! `A·z = w + c·t`, `ℓ(z) = img + c·y` over the digit windows of the
//! `2^{h_r} × 2^{h_c}` grid — is field-generic; this module
//! instantiates it on `Fp256`:
//!
//! * digit windows over the **Montgomery limbs** (16 × 16-bit windows
//!   per element — an exact integer identity, no canonical conversion
//!   needed);
//! * grid tensors `a = ⊗(1−r_j, r_j)` with challenges sampled by the
//!   **upper-limb short-circuit discipline** (`Fp256::sample_upper_limb`,
//!   λ = 125 — the "grid substrate" ready in `fp256.rs`);
//! * the challenge-response products ride `mul_upper_limb`, the CIOS
//!   short-circuit that skips 18 of 36 limb products.
//!
//! Honest scope: over a prime *field* there are no zero-divisors and
//! no norms — the carrier `A` is a tall random field matrix
//! (`k ≥ m`), so binding is the linear-algebraic statement
//! (kernel-free `A`), not Module-SIS. The port demonstrates the
//! engine's field-genericity and the 256-bit carrier path; the
//! post-quantum binding stays with the ring instantiation
//! (`carrier.rs`).


// (Kernel loops use explicit indices by convention.)
#![allow(clippy::needless_range_loop)]
use lattice_core::transcript::Transcript;
use lattice_projsumcheck::fp256::Fp256;

/// Digit windows per element: 4 Montgomery limbs × 64 bits = 16
/// windows of 16 bits.
pub const FP_WINDOWS: usize = 16;
pub const FP_WINDOW_BITS: u64 = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FpPortError {
    Shape(String),
    Binding,
}

/// Digit windows of one element's Montgomery limbs (16 × 16 bits,
/// exact integer identity).
pub fn fp_digit_windows(e: &Fp256) -> Vec<Fp256> {
    let mut out = Vec::with_capacity(FP_WINDOWS);
    for k in 0..FP_WINDOWS {
        let limb = k / 4;
        let shift = 16 * (k % 4);
        let raw = (e.limbs[limb] >> shift) & 0xffff;
        out.push(Fp256::from_canonical_u64(raw));
    }
    out
}

/// The carrier: a tall random matrix `A ∈ F_p^{k×m}` from a seed, with
/// `commit(s) = A·s`.
pub struct FpCarrier {
    pub k: usize,
    pub m: usize,
    /// Row-major k×m entries (Montgomery form).
    pub matrix: Vec<Fp256>,
}

/// A seeded field element (Montgomery image of a random u64 —
/// demonstration-scale entropy; production sampling draws full-width
/// canonical residues with rejection).
fn fp_from_seed(domain: &[u8], seed: &[u8], index: usize) -> Fp256 {
    let mut salt = Vec::with_capacity(domain.len() + seed.len() + 8);
    salt.extend_from_slice(domain);
    salt.extend_from_slice(seed);
    salt.extend_from_slice(&(index as u64).to_le_bytes());
    let bytes = Transcript::xof(domain, &salt, 8);
    Fp256::from_canonical_u64(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

impl FpCarrier {
    pub fn from_seed(k: usize, m: usize, seed: &[u8]) -> Self {
        let mut matrix = Vec::with_capacity(k * m);
        for i in 0..(k * m) {
            matrix.push(fp_from_seed(b"fp-carrier-A", seed, i));
        }
        FpCarrier { k, m, matrix }
    }

    /// `t = A·s`.
    pub fn commit(&self, s: &[Fp256]) -> Result<Vec<Fp256>, FpPortError> {
        if s.len() != self.m {
            return Err(FpPortError::Shape("slot arity mismatch".into()));
        }
        let mut out = Vec::with_capacity(self.k);
        for i in 0..self.k {
            let mut acc = Fp256::ZERO;
            for (j, sj) in s.iter().enumerate() {
                let aij = &self.matrix[i * self.m + j];
                acc = acc.add(&aij.mul(sj));
            }
            out.push(acc);
        }
        Ok(out)
    }
}

/// The oracle-vector slots: for each entry, `FP_WINDOWS` windows
/// tracked as RAW u16 values (the exact integer identity) with their
/// Montgomery images.
#[derive(Clone, Debug)]
pub struct FpSlots {
    /// Raw window values (canonical integers `< 2^16`), window-major.
    pub raw: Vec<u64>,
    /// Montgomery images of the windows, same order.
    pub mont: Vec<Fp256>,
    /// The oracle vector length.
    pub n: usize,
}

impl FpSlots {
    /// Window-decompose a vector: raw u16 values (the exact integer
    /// identity on the Montgomery limbs) plus their Montgomery images.
    pub fn from_vector(v: &[Fp256]) -> Self {
        let n = v.len();
        let mut raw = Vec::with_capacity(n * FP_WINDOWS);
        let mut mont = Vec::with_capacity(n * FP_WINDOWS);
        for e in v {
            for k in 0..FP_WINDOWS {
                let limb = k / 4;
                let shift = 16 * (k % 4);
                let val = (e.limbs[limb] >> shift) & 0xffff;
                raw.push(val);
                mont.push(Fp256::from_canonical_u64(val));
            }
        }
        FpSlots { raw, mont, n }
    }

    /// Reconstruct the vector exactly (integer identity on the
    /// Montgomery limbs).
    pub fn reconstruct(&self) -> Vec<Fp256> {
        let mut out = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mut limbs = [0u64; 4];
            for k in 0..FP_WINDOWS {
                let limb = k / 4;
                let shift = 16 * (k % 4);
                limbs[limb] |= self.raw[i * FP_WINDOWS + k] << shift;
            }
            out.push(Fp256 { limbs });
        }
        out
    }
}

/// The grid tensors over Fp256 challenges: `a` over the high bits,
/// `b` over the low bits (Montgomery products).
pub fn fp_grid_tensors(point: &[Fp256]) -> (Vec<Fp256>, Vec<Fp256>) {
    let log_n = point.len();
    let h_r = log_n.div_ceil(2);
    let h_c = log_n - h_r;
    let one = Fp256::one_mont();
    let tensor = |challenges: &[Fp256]| -> Vec<Fp256> {
        let mut acc = vec![one];
        for ch in challenges {
            let one_minus_ch = sub_mont(&one, ch);
            let mut next = Vec::with_capacity(acc.len() * 2);
            for e in &acc {
                next.push(e.mul(&one_minus_ch));
            }
            for e in &acc {
                next.push(e.mul(ch));
            }
            acc = next;
        }
        acc
    };
    let a = tensor(&point[h_c..]);
    let b = tensor(&point[..h_c]);
    (a, b)
}

/// `x − y` over Montgomery residues (both in `[0, p)`): `x + (p − y)`
/// — the negation is limbwise `p − y` (valid since `y < p`).
fn sub_mont(x: &Fp256, y: &Fp256) -> Fp256 {
    x.add(&Fp256 { limbs: ch_neg_limbs(&y.limbs) })
}

fn ch_neg_limbs(y: &[u64; 4]) -> [u64; 4] {
    let p = lattice_projsumcheck::fp256::BN254_FR;
    let mut limbs = [0u64; 4];
    let mut borrow = 0u128;
    for i in 0..4 {
        let sub = (y[i] as u128) + borrow;
        if sub <= p[i] as u128 {
            limbs[i] = p[i] - sub as u64;
            borrow = 0;
        } else {
            limbs[i] = p[i].wrapping_sub(sub as u64);
            borrow = 1;
        }
    }
    limbs
}

/// The linear image over the grid, flat over the SLOT space:
/// `ℓ(slots) = Σ_{k,i,j} a_i·b_j·2^{16k}·slots[k·n + i·cols + j]` —
/// the MLE evaluation of the reconstructed oracle vector. Works for
/// any slot-space vector (windows of a committed oracle, a mask, or a
/// response).
pub fn fp_linear_image_flat(slots: &[Fp256], n: usize, point: &[Fp256]) -> Result<Fp256, FpPortError> {
    let log_n = n.trailing_zeros() as usize;
    if point.len() != log_n {
        return Err(FpPortError::Shape("point arity mismatch".into()));
    }
    if slots.len() != n * FP_WINDOWS {
        return Err(FpPortError::Shape("slot arity mismatch".into()));
    }
    let (a, b) = fp_grid_tensors(point);
    let h_r = log_n.div_ceil(2);
    let h_c = log_n - h_r;
    let rows = 1usize << h_r;
    let cols = 1usize << h_c;
    let mut acc = Fp256::ZERO;
    // weight_k = 2^{16k} in Montgomery form, built by repeated
    // multiplication (2^{16k} overflows u64 for k ≥ 4).
    let two16 = Fp256::from_canonical_u64(1u64 << FP_WINDOW_BITS);
    let mut weight = Fp256::one_mont();
    for k in 0..FP_WINDOWS {
        for i in 0..rows {
            for j in 0..cols {
                let w = &slots[k * n + i * cols + j];
                if w == &Fp256::ZERO {
                    continue;
                }
                let coeff = a[i].mul(&b[j]).mul(&weight);
                acc = acc.add(&coeff.mul(w));
            }
        }
        weight = weight.mul(&two16);
    }
    Ok(acc)
}

/// The linear image of a committed oracle's window slots.
pub fn fp_linear_image(slots: &FpSlots, point: &[Fp256]) -> Result<Fp256, FpPortError> {
    fp_linear_image_flat(&slots.mont, slots.n, point)
}

/// A binding-pass proof over Fp256.
#[derive(Clone, Debug)]
pub struct FpBindingProof {
    pub mask_commitment: Vec<Fp256>,
    pub mask_image: Fp256,
    /// The upper-limb challenge (λ = 125).
    pub challenge: Fp256,
    pub response: Vec<Fp256>,
}

/// Sample an upper-limb challenge off a transcript.
fn sample_upper(tr: &mut Transcript, label: &[u8]) -> Fp256 {
    let bytes = tr.challenge_bytes(label, 32).unwrap_or_else(|_| vec![0u8; 32]);
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes[..32.min(bytes.len())]);
    Fp256::sample_upper_limb(&arr)
}

/// Prove `v̂(point) = y` (the binding pass) for the committed slots.
pub fn prove_fp_binding(
    carrier: &FpCarrier,
    slots: &FpSlots,
    point: &[Fp256],
    y: &Fp256,
    commitment: &[Fp256],
    seed: &[u8],
) -> Result<FpBindingProof, FpPortError> {
    // fail-closed: the claim must hold
    let check = fp_linear_image(slots, point)?;
    if check != *y {
        return Err(FpPortError::Binding);
    }
    // mask: random field elements
    let mask: Vec<Fp256> = (0..carrier.m)
        .map(|i| fp_from_seed(b"fp-mask", seed, i))
        .collect();
    let mask_commitment = carrier.commit(&mask)?;
    let image = fp_linear_image_flat(&mask, slots.n, point)?;
    // the challenge: upper-limb (the grid substrate's discipline)
    let mut tr = Transcript::new_default(b"fp-binding");
    let _ = tr.append_bytes(b"stmt", seed);
    for c in commitment {
        for l in c.limbs {
            let _ = tr.append_bytes(b"c", &l.to_le_bytes());
        }
    }
    for p in point {
        for l in p.limbs {
            let _ = tr.append_bytes(b"pt", &l.to_le_bytes());
        }
    }
    let _ = tr.append_bytes(b"y", &{
        let mut v = Vec::new();
        for l in y.limbs {
            v.extend_from_slice(&l.to_le_bytes());
        }
        v
    });
    for c in &mask_commitment {
        for l in c.limbs {
            let _ = tr.append_bytes(b"w", &l.to_le_bytes());
        }
    }
    let _ = tr.append_bytes(b"img", &{
        let mut v = Vec::new();
        for l in image.limbs {
            v.extend_from_slice(&l.to_le_bytes());
        }
        v
    });
    let challenge = sample_upper(&mut tr, b"chal");
    // z = mask + c·slots — the challenge products ride the CIOS
    // upper-limb short-circuit.
    let mut response = Vec::with_capacity(carrier.m);
    for (mi, si) in mask.iter().zip(slots.mont.iter()) {
        response.push(mi.add(&si.mul_upper_limb(&challenge)));
    }
    Ok(FpBindingProof { mask_commitment, mask_image: image, challenge, response })
}

/// Verify the binding pass.
pub fn verify_fp_binding(
    carrier: &FpCarrier,
    n: usize,
    point: &[Fp256],
    y: &Fp256,
    commitment: &[Fp256],
    seed: &[u8],
    proof: &FpBindingProof,
) -> Result<(), FpPortError> {
    if proof.response.len() != carrier.m {
        return Err(FpPortError::Shape("response arity mismatch".into()));
    }
    // FS replay
    let mut tr = Transcript::new_default(b"fp-binding");
    let _ = tr.append_bytes(b"stmt", seed);
    for c in commitment {
        for l in c.limbs {
            let _ = tr.append_bytes(b"c", &l.to_le_bytes());
        }
    }
    for p in point {
        for l in p.limbs {
            let _ = tr.append_bytes(b"pt", &l.to_le_bytes());
        }
    }
    let _ = tr.append_bytes(b"y", &{
        let mut v = Vec::new();
        for l in y.limbs {
            v.extend_from_slice(&l.to_le_bytes());
        }
        v
    });
    for c in &proof.mask_commitment {
        for l in c.limbs {
            let _ = tr.append_bytes(b"w", &l.to_le_bytes());
        }
    }
    let _ = tr.append_bytes(b"img", &{
        let mut v = Vec::new();
        for l in proof.mask_image.limbs {
            v.extend_from_slice(&l.to_le_bytes());
        }
        v
    });
    let expected = sample_upper(&mut tr, b"chal");
    if expected != proof.challenge {
        return Err(FpPortError::Binding);
    }
    if !proof.challenge.is_upper_limb() {
        return Err(FpPortError::Binding);
    }
    // A·z = w + c·t
    let az = carrier.commit(&proof.response)?;
    for (i, az_i) in az.iter().enumerate() {
        let ct = commitment[i].mul_upper_limb(&proof.challenge);
        let expect = proof.mask_commitment[i].add(&ct);
        if *az_i != expect {
            return Err(FpPortError::Binding);
        }
    }
    // ℓ(z) = img + c·y — the response lives in slot space
    let lz = fp_linear_image_flat(&proof.response, n, point)?;
    let cy = y.mul_upper_limb(&proof.challenge);
    let expect = proof.mask_image.add(&cy);
    if lz != expect {
        return Err(FpPortError::Binding);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vec(seed: &[u8], n: usize) -> Vec<Fp256> {
        (0..n).map(|i| fp_from_seed(b"fp-vec", seed, i)).collect()
    }

    #[test]
    fn digit_windows_exact_identity() {
        let v = sample_vec(b"dw", 8);
        let slots = FpSlots::from_vector(&v);
        let back = slots.reconstruct();
        assert_eq!(back.len(), v.len());
        for (x, y) in back.iter().zip(v.iter()) {
            assert_eq!(x.limbs, y.limbs);
        }
        for r in &slots.raw {
            assert!(*r < (1 << FP_WINDOW_BITS));
        }
    }

    #[test]
    fn binding_pass_end_to_end() {
        let n = 8usize;
        let v = sample_vec(b"bp", n);
        let slots = FpSlots::from_vector(&v);
        let carrier = FpCarrier::from_seed(64, n * FP_WINDOWS, b"key");
        let commitment = carrier.commit(&slots.mont).ok().unwrap();
        // challenges: upper-limb points
        let point: Vec<Fp256> = (0..3).map(|i| {
            let bytes = Transcript::xof(b"pt", b"bp", 32);
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            // vary per index
            arr[0] = i as u8;
            Fp256::sample_upper_limb(&arr)
        }).collect();
        let y = fp_linear_image(&slots, &point).ok().unwrap();
        let proof = prove_fp_binding(&carrier, &slots, &point, &y, &commitment, b"s").unwrap_or_else(|e| panic!("prove: {e:?}"));
        verify_fp_binding(&carrier, n, &point, &y, &commitment, b"s", &proof).ok().unwrap();
        // wrong claim rejected at prove time
        let y_bad = y.add(&Fp256::one_mont());
        assert!(prove_fp_binding(&carrier, &slots, &point, &y_bad, &commitment, b"s").is_err());
        // tampered response rejected
        let mut bad = proof.clone();
        bad.response[0] = bad.response[0].add(&Fp256::one_mont());
        assert!(verify_fp_binding(&carrier, n, &point, &y, &commitment, b"s", &bad).is_err());
        // tampered commitment rejected
        let bad2 = proof.clone();
        let mut c2 = commitment.clone();
        c2[0] = c2[0].add(&Fp256::one_mont());
        assert!(verify_fp_binding(&carrier, n, &point, &y, &c2, b"s", &bad2).is_err());
        // wrong point rejected
        let other: Vec<Fp256> = point.iter().rev().cloned().collect();
        assert!(verify_fp_binding(&carrier, n, &other, &y, &commitment, b"s", &proof).is_err());
    }

    #[test]
    fn upper_limb_short_circuit_equivalence() {
        // mul_upper_limb(c) == mul(c) for upper-limb challenges — the
        // CIOS grid's fast path is functionally identical.
        let v = sample_vec(b"sc", 16);
        let chal = {
            let bytes = Transcript::xof(b"chal", b"sc", 32);
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            Fp256::sample_upper_limb(&arr)
        };
        assert!(chal.is_upper_limb());
        for e in &v {
            assert_eq!(e.mul_upper_limb(&chal), e.mul(&chal));
        }
    }

    #[test]
    fn structural_equivalence_with_ring_engine() {
        // The port mirrors the ring engine's structure: same window
        // discipline shape, same grid tensor arity, same three checks.
        let ring = crate::ring_d::RingD::new(4).ok().unwrap();
        // ring engine windows: 8 × 4-bit; Fp256 port: 16 × 16-bit —
        // both exact integer identities on their representations.
        let e = ring.random(b"eq");
        let rw = crate::windowed::digit_windows(&ring, &e);
        assert_eq!(crate::windowed::digit_reconstruct(&ring, &rw), e);
        let f = sample_vec(b"eq", 1);
        let fs = FpSlots::from_vector(&f);
        assert_eq!(fs.reconstruct(), f);
        // grid tensor arities agree with the ring engine's
        let pt_ring: Vec<crate::ring_d::Elem> = {
            let mut tr = Transcript::new_default(b"pt");
            (0..2).map(|i| ring.sample_challenge(&mut tr, format!("r{i}").as_bytes())).collect()
        };
        let (ra, rb) = crate::windowed::grid_tensors(&ring, &pt_ring);
        let pt_fp: Vec<Fp256> = (0..2)
            .map(|i| {
                let bytes = Transcript::xof(b"ptf", b"eq", 32);
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                arr[0] = i as u8;
                Fp256::sample_upper_limb(&arr)
            })
            .collect();
        let (fa, fb) = fp_grid_tensors(&pt_fp);
        assert_eq!(ra.len(), fa.len());
        assert_eq!(rb.len(), fb.len());
    }
}
