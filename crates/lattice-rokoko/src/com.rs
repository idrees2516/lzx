//! RoKoko COM — the recursive Ajtai commitment (paper Fig. 1, Wave 7 item
//! 7.13 component 3), ported from the lattice-zk-lab reference.
//!
//! `Com_{par_com}(ck, w)`:
//! * level 0: `y = A_{n0, m}·w` (a plain vSIS/Ajtai commitment);
//! * depth ≥ 2: `e = G^{−1}_l(y)` (binary digit layers), zero-padded to
//!   the next power of two, committed recursively — the deepest output is
//!   `com`;
//! * `Verify_{par_com, β}`: (b0) `‖w‖₂ ≤ β0`; (b1) the gadget
//!   recomposition `G_l(x[:l·n0]) == y` at every level; (b2) the
//!   recursive structure itself.
//!
//! Keys are seed-derived per (n, m) shape (transparent setup, cached).

use lattice_commitment::ajtai::{AjtaiParams, AjtaiError, AjtaiPublicKey};
use lattice_ring::{RingConfig, RingElement};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComError {
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    /// ‖w‖₂ exceeded the β0 gate (fail closed).
    NormBoundExceeded { norm_sq: u64, beta0: u64 },
    /// Gadget recomposition failed (b1).
    RecompositionFailed,
    /// Recursive structure mismatch (b2).
    StructureMismatch,
    Shape { expected: usize, got: usize },
}

impl From<AjtaiError> for ComError {
    fn from(e: AjtaiError) -> Self {
        ComError::Ajtai(e)
    }
}
impl From<lattice_ring::RingError> for ComError {
    fn from(e: lattice_ring::RingError) -> Self {
        ComError::Ring(e)
    }
}

/// RoKoko kernel parameters.
#[derive(Clone, Debug)]
pub struct RokokoParams {
    pub n_ring: u32,     // log2 ring dimension
    pub n0: usize,       // vSIS rows per commitment level
    pub gadget_len: usize, // l: binary gadget digits per element
    pub com_depth: usize,  // COM recursion depth d
    pub r: usize,        // witness column count (power of two)
    pub beta_w: u64,     // witness l2 bound per column
}

impl RokokoParams {
    pub fn ring(&self) -> Result<RingConfig, ComError> {
        lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, self.n_ring)
            .map_err(ComError::Ring)
    }
}

/// Seeded vSIS keys A_{n, m} for m a power of two (transparent, cached).
pub struct ComKey {
    pub params: RokokoParams,
    pub seed: [u8; 32],
    cache: std::collections::HashMap<(usize, usize), AjtaiPublicKey>,
}

impl ComKey {
    pub fn new(params: RokokoParams, seed: [u8; 32]) -> Self {
        ComKey {
            params,
            seed,
            cache: std::collections::HashMap::new(),
        }
    }

    /// The A_{n, m} key (derived on first use, cached after).
    pub fn key(&mut self, ring: &RingConfig, n: usize, m: usize) -> Result<&AjtaiPublicKey, ComError> {
        if !self.cache.contains_key(&(n, m)) {
            let mut seed = self.seed;
            for (i, b) in format!("|{}|{}", n, m).bytes().enumerate() {
                seed[i % 32] ^= b.wrapping_add(i as u8);
            }
            let params = AjtaiParams {
                ring: ring.clone(),
                k: n,
                m,
                norm_bound: 1 << 20,
            };
            let pk = AjtaiPublicKey::from_seed(params, seed)?;
            self.cache.insert((n, m), pk);
        }
        self.cache
            .get(&(n, m))
            .ok_or(ComError::Shape { expected: 1, got: 0 })
    }

    /// `A_{n, m}·w` — the plain level commitment.
    pub fn commit(
        &mut self,
        ring: &RingConfig,
        w: &[RingElement],
        n: usize,
    ) -> Result<Vec<RingElement>, ComError> {
        let m = w.len();
        let out = self.key(ring, n, m)?.commit(w)?;
        Ok(out.rows)
    }
}

/// aux = (x*, x) for depth ≥ 2; trivial for depth 1.
#[derive(Clone, Debug)]
pub struct ComOpening {
    /// level-0 commitment `y = A_{n0, m}·w`.
    pub y: Vec<RingElement>,
    /// recursive commitment output (None at depth 1).
    pub x_star: Option<Vec<RingElement>>,
    /// the decomposed+padded vector committed recursively (None at depth 1).
    pub x: Option<Vec<RingElement>>,
}

/// `G^{−1}_l(v)`: binary digit layers, component-major (output index
/// t·l + e = digit layer e of component t).
pub fn g_inv_vec(ring: &RingConfig, vec: &[RingElement], l: usize) -> Vec<RingElement> {
    let mut out = Vec::with_capacity(vec.len() * l);
    for elt in vec {
        let mut layers = vec![ring.zero(); l];
        for (i, &c) in elt.coeffs().iter().enumerate() {
            for e in 0..l {
                let mut coeffs = layers[e].coeffs().to_vec();
                coeffs[i] = (c >> e) & 1;
                layers[e] = RingElement::from_coeffs(ring, coeffs);
            }
        }
        out.extend(layers);
    }
    out
}

/// `G_l(v)`: recompose `count` elements from their digit layers.
pub fn g_vec(flat: &[RingElement], l: usize, count: usize) -> Result<Vec<RingElement>, ComError> {
    let ring = flat
        .first()
        .map(|e| e.config().clone())
        .ok_or(ComError::Shape {
            expected: 1,
            got: 0,
        })?;
    let q = i128::from(ring.modulus.q);
    let mut out = Vec::with_capacity(count);
    for t in 0..count {
        let mut acc = ring.zero();
        for e in 0..l {
            // 2^e mod q as an i64 scalar (reduced for large e)
            let scalar = ((1i128 << e) % q) as i64;
            let scaled = flat[t * l + e].scale_i64(scalar);
            acc = acc.add(&scaled)?;
        }
        out.push(acc);
    }
    Ok(out)
}

fn next_pow_two(v: usize) -> usize {
    if v <= 1 {
        1
    } else {
        1usize << ((v - 1).ilog2() + 1)
    }
}

/// `Com_{par_com}(ck, w)` — the recursive Ajtai commitment (Fig. 1).
/// Returns (com, aux). depth = 1: com = A_{n0, m}·w.
pub fn com_commit(
    ck: &mut ComKey,
    ring: &RingConfig,
    w: &[RingElement],
    depth: usize,
    l: usize,
) -> Result<(Vec<RingElement>, ComOpening), ComError> {
    let n0 = ck.params.n0;
    let y = ck.commit(ring, w, n0)?;
    if depth == 1 {
        return Ok((
            y.clone(),
            ComOpening {
                y,
                x_star: None,
                x: None,
            },
        ));
    }
    // e = G^{-1}_l(y); x = e || 0 padded to next_pow_two(l·n0)
    let e = g_inv_vec(ring, &y, l);
    let target = next_pow_two(l * n0);
    let mut x = e;
    x.resize(target, ring.zero());
    let (com_out, _) = com_commit(ck, ring, &x, depth - 1, l)?;
    Ok((
        com_out.clone(),
        ComOpening {
            y,
            x_star: Some(com_out),
            x: Some(x),
        },
    ))
}

/// The flattened integer squared norm of a ring-element vector (balanced
/// representatives).
pub fn l2_sq_int(w: &[RingElement]) -> u64 {
    let mut total: u64 = 0;
    for e in w {
        total = total.saturating_add(e.euclidean_norm_squared());
    }
    total
}

/// `Verify_{par_com, β}(ck, w, com, aux)` — Figure 1 right column.
pub fn com_verify(
    ck: &mut ComKey,
    ring: &RingConfig,
    w: &[RingElement],
    com: &[RingElement],
    aux: &ComOpening,
    depth: usize,
    l: usize,
    beta0: u64,
) -> Result<bool, ComError> {
    let n0 = ck.params.n0;
    // b0: ||w||_2 <= beta0
    if l2_sq_int(w) > beta0.saturating_mul(beta0) {
        eprintln!("DEBUG b0 fail: norm={} beta0={}", l2_sq_int(w), beta0);
        return Ok(false);
    }
    if depth == 1 {
        return Ok(ck.commit(ring, w, n0)? == com);
    }
    // b1: A_{n0, 2^mu} w == y  and  G_l(x[:l·n0]) == y
    if ck.commit(ring, w, n0)? != aux.y {
        return Ok(false);
    }
    let target = next_pow_two(l * n0);
    let x = aux.x.as_ref().ok_or(ComError::Shape {
        expected: target,
        got: 0,
    })?;
    if x.len() != target {
        eprintln!("DEBUG b1 x len {} != {}", x.len(), target);
        return Ok(false);
    }
    if g_vec(&x[..l * n0], l, n0)? != aux.y {
        return Ok(false);
    }
    // b2: the recursive commitment of x must equal the recorded x*
    let x_star = aux.x_star.as_ref().ok_or(ComError::Shape {
        expected: 1,
        got: 0,
    })?;
    let (recomputed, _) = com_commit(ck, ring, x, depth - 1, l)?;
    if &recomputed != x_star {
        return Ok(false);
    }
    if x_star != com { eprintln!("DEBUG final com mismatch"); }
    Ok(x_star == com)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> RokokoParams {
        RokokoParams {
            n_ring: 3,
            n0: 2,
            gadget_len: 4,
            com_depth: 2,
            r: 2,
            beta_w: 16,
        }
    }

    fn small_w(ring: &RingConfig, m: usize, tag: &[u8]) -> Vec<RingElement> {
        (0..m)
            .map(|i| {
                let bytes = lattice_core::transcript::Transcript::xof(
                    b"rokoko-com-test",
                    &[tag, &(i as u32).to_le_bytes()].concat(),
                    4 * ring.n(),
                );
                let coeffs: Vec<u32> = bytes
                    .chunks(4)
                    .take(ring.n())
                    .map(|c| {
                        let mut a = [0u8; 4];
                        a.copy_from_slice(&c[..4]);
                        u32::from_le_bytes(a) % 5
                    })
                    .collect();
                RingElement::from_coeffs(ring, coeffs)
            })
            .collect()
    }

    #[test]
    fn com_depth1_roundtrip_and_tamper() {
        let p = params();
        let ring = p.ring().ok().unwrap();
        let mut ck = ComKey::new(p, [41u8; 32]);
        let w = small_w(&ring, 4, b"w1");
        let (com, aux) = com_commit(&mut ck, &ring, &w, 1, 32).ok().unwrap();
        assert!(com_verify(&mut ck, &ring, &w, &com, &aux, 1, 32, 32)
            .ok()
            .unwrap());
        // tampered witness fails (norm or binding)
        let w2 = small_w(&ring, 4, b"w2");
        assert!(!com_verify(&mut ck, &ring, &w2, &com, &aux, 1, 32, 32)
            .ok()
            .unwrap());
    }

    #[test]
    fn com_depth2_roundtrip_and_tamper() {
        let p = params();
        let ring = p.ring().ok().unwrap();
        let mut ck = ComKey::new(p, [42u8; 32]);
        let w = small_w(&ring, 4, b"w");
        let (com, aux) = com_commit(&mut ck, &ring, &w, 2, 32).ok().unwrap();
        assert!(com_verify(&mut ck, &ring, &w, &com, &aux, 2, 32, 32)
            .ok()
            .unwrap());
        // tampered com fails
        let mut com_bad = com.clone();
        if !com_bad.is_empty() {
            let mut coeffs = com_bad[0].coeffs().to_vec();
            coeffs[0] = (coeffs[0] + 1) % ring.modulus.q;
            com_bad[0] = RingElement::from_coeffs(&ring, coeffs);
        }
        assert!(!com_verify(&mut ck, &ring, &w, &com_bad, &aux, 2, 32, 32)
            .ok()
            .unwrap());
        // inflated witness fails the beta0 gate
        let w_big: Vec<RingElement> = (0..4)
            .map(|_| RingElement::from_coeffs(&ring, vec![100; ring.n()]))
            .collect();
        assert!(!com_verify(&mut ck, &ring, &w_big, &com, &aux, 2, 32, 32)
            .ok()
            .unwrap());
    }

    #[test]
    fn gadget_roundtrip() {
        let p = params();
        let ring = p.ring().ok().unwrap();
        let v = small_w(&ring, 3, b"gv");
        let flat = g_inv_vec(&ring, &v, 4);
        assert_eq!(flat.len(), 12);
        let back = g_vec(&flat, 4, 3).ok().unwrap();
        assert_eq!(back, v);
    }
}
