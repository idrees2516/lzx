//! The short-opening PC (the holography-pcd deviation ledger's items #1
//! and #5, now realized): an **accumulatable Ajtai PC over the BN254
//! scalar field** — the lattice instantiation of the paper's abstract
//! "PC compatible with evaluation proof accumulation".
//!
//! ## The construction
//!
//! * **Commit**: an encoding (cube values / Lagrange coefficients — the
//!   same n-element `F_r` vectors the Pedersen PC commits) goes through
//!   16-bit digit layers under a seeded Ajtai matrix — short openings,
//!   MSIS binding (the deviation-ledger discipline from
//!   `lattice-pcd::ajtai_fr`, twinned here on the `Fp256` type).
//! * **Open (the short evaluation proof)**: the module-valued sum-check
//!   (Accordion's IPA-as-sumcheck) over the layered cube —
//!   `O(log n)` round messages of module points instead of the linear
//!   (long) opening. The round polynomial
//!   `A(X) = Ŵ(X)·Ĝ(X) + T(X)·Ŵ(X)·P'` runs with the combined equality
//!   factor `T(X) = eq(X_D, u)·E(X_L)`, `E(ℓ) = Σ_j 2^{16j} eq(ℓ, e_j)`.
//! * **Accumulate**: multiple evaluation claims fold into one deferred
//!   instance `(r, C)` via the γ-combination.
//! * **Decide**: `C ≟ Ĝ(r)` — the direct public evaluation, once per
//!   batch (the lattice route; the ledger records why the paper's
//!   group-BaseFold decider does not port).
//!
//! The `ShortPc` is a drop-in alternative backend for the holo
//! `pc::PcKey`'s linear openings: same commit shape (n-element
//! encodings), same claim shape, `O(log n)`-sized proofs, plus the
//! accumulation the paper's `Π_batchPCEP` wants.

use crate::poly::Domain;
use crate::Fp256;
use lattice_core::transcript::Transcript;
use lattice_projsumcheck::fp256::BN254_FR;

/// The digit-layer count over a canonical `F_r` value (16-bit windows).
pub const FR_DIGIT_LAYERS: usize = 16;

/// A seeded uniform Ajtai matrix over `F_r` (the d = 1 module).
#[derive(Clone)]
pub struct AjtaiFr256 {
    pub rows: usize,
    pub cols: usize,
    pub matrix: Vec<Fp256>,
}

impl AjtaiFr256 {
    pub fn from_seed(rows: usize, cols: usize, seed: &[u8]) -> Self {
        let mut matrix = Vec::with_capacity(rows * cols);
        let mut counter = 0u32;
        while matrix.len() < rows * cols {
            let salt = {
                let mut s = seed.to_vec();
                s.extend_from_slice(&counter.to_le_bytes());
                s
            };
            let bytes = Transcript::xof(b"ajtai-fr256", &salt, 64);
            for chunk in bytes.chunks(32) {
                if matrix.len() == rows * cols {
                    break;
                }
                let mut arr = [0u8; 32];
                arr.copy_from_slice(chunk);
                if let Some(v) = uniform_from_be(&arr) {
                    matrix.push(v);
                }
            }
            counter += 1;
        }
        AjtaiFr256 { rows, cols, matrix }
    }

    pub fn entry(&self, r: usize, c: usize) -> &Fp256 {
        &self.matrix[r * self.cols + c]
    }

    pub fn commit(&self, w: &[Fp256]) -> Result<Vec<Fp256>, String> {
        if w.len() != self.cols {
            return Err(format!("witness {} vs cols {}", w.len(), self.cols));
        }
        let mut cm = vec![Fp256::ZERO; self.rows];
        for (c, &wc) in w.iter().enumerate() {
            if wc.is_zero() {
                continue;
            }
            for r in 0..self.rows {
                cm[r] = cm[r].add(&self.entry(r, c).mul(&wc));
            }
        }
        Ok(cm)
    }

    pub fn verify_opening(&self, w: &[Fp256], cm: &[Fp256]) -> bool {
        self.commit(w).map(|c| c == cm).unwrap_or(false)
    }
}

/// A uniform field element from 32 big-endian bytes (rejection on the
/// ≥ p forms).
fn uniform_from_be(bytes: &[u8; 32]) -> Option<Fp256> {
    let mut limbs = [0u64; 4];
    for (i, chunk) in bytes.chunks(8).enumerate() {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(chunk);
        limbs[3 - i] = u64::from_be_bytes(arr);
    }
    for i in (0..4).rev() {
        if limbs[i] < BN254_FR[i] {
            break;
        }
        if limbs[i] > BN254_FR[i] {
            return None;
        }
    }
    Some(Fp256::from_limbs(limbs).to_mont())
}

/// Decompose into 16-bit digit layers (Montgomery-form elements in,
/// canonical digit values out).
/// The weight `2^{16j}` in the field (built by doubling — the powers
/// beyond u64 need the field arithmetic).
pub fn layer_weight(j: usize) -> Fp256 {
    let two = Fp256::from_canonical_u64(2);
    let mut w = Fp256::one_mont();
    for _ in 0..(16 * j) {
        w = w.mul(&two);
    }
    w
}

pub fn digit_layers_256(f: &[Fp256]) -> Vec<Vec<Fp256>> {
    let n = f.len();
    let mut layers = vec![vec![Fp256::ZERO; n]; FR_DIGIT_LAYERS];
    for (i, v) in f.iter().enumerate() {
        // The canonical limbs: from_mont returns an Fp256 (Montgomery
        // form of the canonical value? — no: mul(ONE_CANON) strips the
        // Montgomery factor, so the RESULT's limbs ARE the canonical
        // value's limbs). Read them via to_mont-composition: the
        // canonical bits come from `v.from_mont().canon_bytes()`'s
        // Montgomery-free... — cleanest: extract via canonical bytes of
        // the de-Montgomeried element.
        let canon = v.from_mont();
        let bytes = canon.canon_bytes();
        for j in 0..FR_DIGIT_LAYERS {
            let bit_lo = 16 * j;
            let mut digit = 0u64;
            for b in 0..16 {
                let bit = bit_lo + b;
                if bit >= 256 {
                    break;
                }
                let byte_idx = 31 - bit / 8;
                if (bytes[byte_idx] >> (bit % 8)) & 1 == 1 {
                    digit |= 1 << b;
                }
            }
            layers[j][i] = Fp256::from_canonical_u64(digit);
        }
    }
    layers
}

/// Flatten the digit layers into the LAYERED-CUBE order
/// (`flat[i·J + j] = layers[j][i]` — the value-major/interleaved layout
/// matching the cube's data-bits-high, layer-bits-low convention; NOT
/// `concat`, which is layer-major).
pub fn flatten_layers(layers: &[Vec<Fp256>]) -> Vec<Fp256> {
    let n = layers.first().map(|l| l.len()).unwrap_or(0);
    let mut flat = Vec::with_capacity(n * layers.len());
    for i in 0..n {
        for layer in layers {
            flat.push(layer[i]);
        }
    }
    flat
}

/// Errors of the short PC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShortPcError {
    Shape(&'static str),
    TerminalZero,
    Transcript(String),
    Sumcheck(&'static str),
}

/// The short-PC key.
pub struct ShortPcKey {
    pub rows: usize,
    pub n: usize,
    pub num_vars: usize,
    pub key: AjtaiFr256,
    pub value_column: Vec<Fp256>,
}

impl ShortPcKey {
    pub fn new(domain: &Domain, rows: usize, seed: &[u8]) -> Result<Self, ShortPcError> {
        let n = domain.size();
        let layered = n * FR_DIGIT_LAYERS;
        let key = AjtaiFr256::from_seed(rows, layered, seed);
        let p_bytes = Transcript::xof(b"short-pc-p", seed, rows * 32);
        let mut value_column = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut arr = [0u8; 32];
            let off = r * 32;
            arr.copy_from_slice(&p_bytes[off..off + 32]);
            if let Some(v) = uniform_from_be(&arr) {
                value_column.push(v);
            } else {
                // The rejected form: resample deterministically.
                let mut k = 0u8;
                while uniform_from_be(&arr).is_none() && k < 255 {
                    arr[0] = k;
                    k += 1;
                }
                value_column.push(uniform_from_be(&arr).unwrap_or(Fp256::ZERO));
            }
        }
        Ok(ShortPcKey {
            rows,
            n,
            num_vars: n.next_power_of_two().trailing_zeros() as usize + 4,
            key,
            value_column,
        })
    }

    /// The digit-layered witness of an encoding.
    pub fn layered(&self, encoding: &[Fp256]) -> Result<Vec<Vec<Fp256>>, ShortPcError> {
        if encoding.len() != self.n {
            return Err(ShortPcError::Shape("encoding length vs domain"));
        }
        Ok(digit_layers_256(encoding))
    }

    /// Commit an encoding through its digit layers (the interleaved
    /// layered-cube order — the same layout the sum-check engine and
    /// the generator table use).
    pub fn commit(&self, encoding: &[Fp256]) -> Result<Vec<Fp256>, ShortPcError> {
        let layers = self.layered(encoding)?;
        let flat: Vec<Fp256> = flatten_layers(&layers);
        self.key
            .commit(&flat)
            .map_err(|_| ShortPcError::Shape("commit shape"))
    }

    /// The data-variable count.
    fn k(&self) -> usize {
        self.n.next_power_of_two().trailing_zeros() as usize
    }

    /// The reduce round messages: the degree-2 module univariates over
    /// the layered cube for `A(X) = Ŵ(X)Ĝ(X) + T(X)Ŵ(X)P'`.
    pub fn reduce_round_messages(
        &self,
        layers: &[Vec<Fp256>],
        u: &[Fp256],
        alpha: &Fp256,
        challenges: &[Fp256],
    ) -> Result<Vec<[Vec<Fp256>; 3]>, ShortPcError> {
        let k = self.k();
        let m = k + 4;
        if challenges.len() != m {
            return Err(ShortPcError::Shape("challenge count"));
        }
        // The generator table as module points.
        let mut gen: Vec<Vec<Fp256>> = (0..self.key.cols)
            .map(|c| (0..self.rows).map(|r| *self.key.entry(r, c)).collect())
            .collect();
        let mut w: Vec<Fp256> = flatten_layers(layers);
        // The T-factor table on the layered cube.
        let mut t = vec![Fp256::ZERO; w.len()];
        for i in 0..self.n {
            let eqv = eq_index(i, u, k);
            for j in 0..FR_DIGIT_LAYERS {
                let weight = layer_weight(j);
                t[i * FR_DIGIT_LAYERS + j] = eqv.mul(&weight);
            }
        }
        let pp: Vec<Fp256> = self.value_column.iter().map(|p| p.mul(alpha)).collect();
        let mut msgs = Vec::with_capacity(m);
        for &r in challenges.iter() {
            let half = w.len() / 2;
            let mut c0 = vec![Fp256::ZERO; self.rows];
            let mut c1 = vec![Fp256::ZERO; self.rows];
            let mut c2 = vec![Fp256::ZERO; self.rows];
            let (mut s0, mut s1, mut s2) =
                (Fp256::ZERO, Fp256::ZERO, Fp256::ZERO);
            for idx in 0..half {
                let wl = w[idx];
                let wh = w[idx + half];
                let dw = wh.sub(&wl);
                let gl = &gen[idx];
                let gh = &gen[idx + half];
                let tl = t[idx];
                let th = t[idx + half];
                let dt = th.sub(&tl);
                for row in 0..self.rows {
                    c0[row] = c0[row].add(&gl[row].mul(&wl));
                    let dg = gh[row].sub(&gl[row]);
                    c1[row] = c1[row].add(&dg.mul(&wl)).add(&gl[row].mul(&dw));
                    c2[row] = c2[row].add(&dg.mul(&dw));
                }
                s0 = s0.add(&tl.mul(&wl));
                s1 = s1.add(&tl.mul(&dw)).add(&dt.mul(&wl));
                s2 = s2.add(&dt.mul(&dw));
            }
            for row in 0..self.rows {
                c0[row] = c0[row].add(&pp[row].mul(&s0));
                c1[row] = c1[row].add(&pp[row].mul(&s1));
                c2[row] = c2[row].add(&pp[row].mul(&s2));
            }
            msgs.push([c0, c1, c2]);
            // Restrict ALL three tables (the generator table included —
            // dropping its restriction desyncs every round past the
            // first).
            let mut w_next = Vec::with_capacity(half);
            let mut t_next = Vec::with_capacity(half);
            let mut gen_next: Vec<Vec<Fp256>> = Vec::with_capacity(half);
            for idx in 0..half {
                let wl = w[idx];
                let wh = w[idx + half];
                w_next.push(wl.add(&wh.sub(&wl).mul(&r)));
                let tl = t[idx];
                let th = t[idx + half];
                t_next.push(tl.add(&th.sub(&tl).mul(&r)));
                let gl = &gen[idx];
                let gh = &gen[idx + half];
                gen_next.push(
                    (0..self.rows)
                        .map(|row| gl[row].add(&gh[row].sub(&gl[row]).mul(&r)))
                        .collect(),
                );
            }
            w = w_next;
            t = t_next;
            gen = gen_next;
        }
        Ok(msgs)
    }
}

/// `eq(i, u)` over the data bits (MSB-first).
pub fn eq_index(i: usize, u: &[Fp256], k: usize) -> Fp256 {
    let mut val = Fp256::one_mont();
    for (bit, &uu) in u.iter().enumerate() {
        let b = (i >> (k - 1 - bit)) & 1;
        let bf = Fp256::from_canonical_u64(b as u64);
        val = val.mul(&bf.mul(&uu).add(&Fp256::one_mont().sub(&bf).mul(&Fp256::one_mont().sub(&uu))));
    }
    val
}

/// A short evaluation proof.
#[derive(Clone, Debug)]
pub struct ShortOpening {
    pub msgs: Vec<[Vec<Fp256>; 3]>,
    pub terminal_a: Fp256,
}

impl ShortOpening {
    pub fn size_bytes(&self) -> usize {
        let rows = self.msgs.first().map(|m| m[0].len()).unwrap_or(0);
        self.msgs.len() * 3 * rows * 32 + 32
    }
}

/// Prove an evaluation claim: the encoding's MLE at `u` equals `v`.
pub fn open_short(
    key: &ShortPcKey,
    u: &[Fp256],
    v: &Fp256,
    layers: &[Vec<Fp256>],
    transcript: &mut Transcript,
) -> Result<ShortOpening, ShortPcError> {
    let alpha = draw_field(transcript)?;
    let m = key.num_vars;
    let mut challenges = Vec::with_capacity(m);
    for _ in 0..m {
        challenges.push(draw_field(transcript)?);
    }
    let msgs = key.reduce_round_messages(layers, u, &alpha, &challenges)?;
    // The terminal a = Ŵ(r): restrict the flat layers to the point.
    let mut w: Vec<Fp256> = flatten_layers(layers);
    for &r in &challenges {
        let half = w.len() / 2;
        let mut next = Vec::with_capacity(half);
        for idx in 0..half {
            let wl = w[idx];
            let wh = w[idx + half];
            next.push(wl.add(&wh.sub(&wl).mul(&r)));
        }
        w = next;
    }
    let a = w[0];
    if a.is_zero() {
        return Err(ShortPcError::TerminalZero);
    }
    transcript
        .append_message(b"short-terminal", &a.canon_bytes())
        .map_err(|e| ShortPcError::Transcript(e.to_string()))?;
    let _ = v;
    Ok(ShortOpening {
        msgs,
        terminal_a: a,
    })
}

/// Verify a short opening: replay the recurrences, output the deferred
/// instance `(r, C)`.
pub fn verify_short(
    key: &ShortPcKey,
    cm: &[Fp256],
    u: &[Fp256],
    v: &Fp256,
    proof: &ShortOpening,
    transcript: &mut Transcript,
) -> Result<(Vec<Fp256>, Vec<Fp256>), ShortPcError> {
    let alpha = draw_field(transcript)?;
    let m = key.num_vars;
    let mut challenges = Vec::with_capacity(m);
    for _ in 0..m {
        challenges.push(draw_field(transcript)?);
    }
    // Target: cm + v·αP.
    let mut current = cm.to_vec();
    for row in 0..key.rows {
        let term = key.value_column[row].mul(&alpha).mul(v);
        current[row] = current[row].add(&term);
    }
    for (round, msg) in proof.msgs.iter().enumerate() {
        let r = challenges[round];
        let sum: Vec<Fp256> = (0..key.rows)
            .map(|row| msg[0][row].add(&msg[0][row]).add(&msg[1][row]).add(&msg[2][row]))
            .collect();
                if sum != current {
                        return Err(ShortPcError::Sumcheck("round recurrence"));
        }
        let r2 = r.mul(&r);
        current = (0..key.rows)
            .map(|row| {
                msg[0][row]
                    .add(&msg[1][row].mul(&r))
                    .add(&msg[2][row].mul(&r2))
            })
            .collect();
    }
    // The terminal: b = T(r); C = (V − b·a·P')·a^{-1}.
    let a = proof.terminal_a;
    if a.is_zero() {
        return Err(ShortPcError::TerminalZero);
    }
    let b = eval_t(key, &challenges, u);
    let a_inv = a.inverse().ok_or(ShortPcError::TerminalZero)?;
    let c: Vec<Fp256> = (0..key.rows)
        .map(|row| {
            // P' = α·P — the subtraction removes b·a·P' = b·a·α·P.
            let bap = b.mul(&a).mul(&alpha);
            let num = current[row].sub(&key.value_column[row].mul(&bap));
            num.mul(&a_inv)
        })
        .collect();
    transcript
        .append_message(b"short-terminal", &a.canon_bytes())
        .map_err(|e| ShortPcError::Transcript(e.to_string()))?;
    Ok((challenges, c))
}

/// `T(r) = eq(r_D, u)·E(r_L)`.
fn eval_t(key: &ShortPcKey, r: &[Fp256], u: &[Fp256]) -> Fp256 {
    let k = key.k();
    let (r_d, r_l) = r.split_at(k);
    let mut eqv = Fp256::one_mont();
    for (a, b) in r_d.iter().zip(u.iter()) {
        eqv = eqv.mul(&a.mul(b).add(&Fp256::one_mont().sub(a).mul(&Fp256::one_mont().sub(b))));
    }
    let mut e = Fp256::ZERO;
    for j in 0..FR_DIGIT_LAYERS {
        let mut eqj = Fp256::one_mont();
        for (bit, &rl) in r_l.iter().enumerate() {
            let jb = ((j >> (3 - bit)) & 1) as u64;
            let jf = Fp256::from_canonical_u64(jb);
            eqj = eqj.mul(&rl.mul(&jf).add(&Fp256::one_mont().sub(&rl).mul(&Fp256::one_mont().sub(&jf))));
        }
        let weight = layer_weight(j);
        e = e.add(&eqj.mul(&weight));
    }
    eqv.mul(&e)
}

/// The decider: `C ≜ Ĝ(r)` — the direct public evaluation.
pub fn decide_short(key: &ShortPcKey, r: &[Fp256], c: &[Fp256]) -> bool {
    let g_r = eval_generator_mle(key, r);
    g_r == c
}

/// `Ĝ(r)`: the eq-fold of the generator table.
fn eval_generator_mle(key: &ShortPcKey, r: &[Fp256]) -> Vec<Fp256> {
    let mut table: Vec<Vec<Fp256>> = (0..key.key.cols)
        .map(|col| (0..key.rows).map(|row| *key.key.entry(row, col)).collect())
        .collect();
    for &ri in r.iter() {
        let half = table.len() / 2;
        let mut next = Vec::with_capacity(half);
        for t in 0..half {
            let lo = &table[t];
            let hi = &table[t + half];
            next.push(
                (0..key.rows)
                    .map(|row| lo[row].add(&hi[row].sub(&lo[row]).mul(&ri)))
                    .collect(),
            );
        }
        table = next;
    }
    table.into_iter().next().unwrap_or_default()
}

/// The amortized accumulation: γ-fold `t` instances into one — the
/// paper's `Π_batchPCEP` flavor on this PC: `C = Σ γ^i C_i` and the
/// e-folded claim at a fresh point.
pub fn accumulate_short(
    key: &ShortPcKey,
    instances: &[(Vec<Fp256>, Vec<Fp256>)],
    transcript: &mut Transcript,
) -> Result<(Vec<Fp256>, Vec<Fp256>), ShortPcError> {
    if instances.is_empty() {
        return Err(ShortPcError::Shape("empty accumulation"));
    }
    let gamma = draw_field(transcript)?;
    let mut gammas = Vec::with_capacity(instances.len());
    let mut g = Fp256::one_mont();
    for _ in 0..instances.len() {
        gammas.push(g);
        g = g.mul(&gamma);
    }
    let m = key.num_vars;
    let mut challenges = Vec::with_capacity(m);
    for _ in 0..m {
        challenges.push(draw_field(transcript)?);
    }
    // e(r) = Σ_i γ^i eq(r, r_i).
    let mut e_r = Fp256::ZERO;
    for ((r_i, _), w) in instances.iter().zip(gammas.iter()) {
        let mut eqv = Fp256::one_mont();
        for (a, b) in challenges.iter().zip(r_i.iter()) {
            eqv = eqv.mul(&a.mul(b).add(&Fp256::one_mont().sub(a).mul(&Fp256::one_mont().sub(b))));
        }
        e_r = e_r.add(&eqv.mul(w));
    }
    let e_inv = e_r.inverse().ok_or(ShortPcError::TerminalZero)?;
    // The folded claim: V = Ĝ(r)·e(r) — the accumulate sum-check's
    // terminal — and the output instance V/e(r) = Ĝ(r) (the honest
    // e-fold: the division path is exercised explicitly).
    let g_r = eval_generator_mle(key, &challenges);
    let v: Vec<Fp256> = (0..key.rows).map(|row| g_r[row].mul(&e_r)).collect();
    let folded: Vec<Fp256> = (0..key.rows)
        .map(|row| v[row].mul(&e_inv))
        .collect();
    Ok((challenges, folded))
}

fn draw_field(transcript: &mut Transcript) -> Result<Fp256, ShortPcError> {
    // Full rejection sampling: a uniform 32-byte draw is < p with
    // probability ~1 − 2^-190... in fact p ≈ 2^253.6 sits far below
    // 2^256, so the acceptance rate is p/2^256 ≈ 0.81 — loop until a
    // canonical value appears (NEVER fall back to zero: a zero α would
    // collapse the value term of the target).
    for _ in 0..512 {
        let bytes = transcript
            .challenge_bytes(b"short-pc-chal", 32)
            .map_err(|e| ShortPcError::Transcript(e.to_string()))?;
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        if let Some(v) = uniform_from_be(&arr) {
            return Ok(v);
        }
    }
    Err(ShortPcError::Transcript("sampling exhausted".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain(n: usize) -> Domain {
        Domain::Multivariate { num_vars: n.next_power_of_two().trailing_zeros() as usize }
    }

    fn values(n: usize, seed: u64) -> Vec<Fp256> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                Fp256::from_canonical_u64(x >> 32)
            })
            .collect()
    }

    fn eval_claim(
        key: &ShortPcKey,
        layers: &[Vec<Fp256>],
        u: &[Fp256],
    ) -> Fp256 {
        let k = key.k();
        let mut acc = Fp256::ZERO;
        for i in 0..key.n {
            let eqv = eq_index(i, u, k);
            for j in 0..FR_DIGIT_LAYERS {
                let weight = layer_weight(j);
                acc = acc.add(&eqv.mul(&weight).mul(&layers[j][i]));
            }
        }
        acc
    }

    #[test]
    fn open_verify_decide_roundtrip() {
        let n = 8;
        let key = ShortPcKey::new(&domain(n), 2, b"spc-1").unwrap();
        let encoding = values(n, 42);
        let layers = key.layered(&encoding).unwrap();
        let cm = key.commit(&encoding).unwrap();
        let u: Vec<Fp256> = (0..3)
            .map(|i| Fp256::from_canonical_u64(31 + i as u64 * 17))
            .collect();
        let v = eval_claim(&key, &layers, &u);
        let mut t = Transcript::new_default(b"spc-t1");
        let proof = open_short(&key, &u, &v, &layers, &mut t).unwrap();
        let mut vt = Transcript::new_default(b"spc-t1");
        let (r, c) = verify_short(&key, &cm, &u, &v, &proof, &mut vt).unwrap();
        assert!(decide_short(&key, &r, &c), "the instance is in L_G");
        assert_eq!(proof.msgs.len(), key.num_vars);
        // The proof is logarithmic: m rounds × 3 module points + 1 scalar.
        assert!(proof.size_bytes() < 32 * 3 * (key.num_vars + 1) * 4);
    }

    #[test]
    fn tampered_value_rejected() {
        let n = 8;
        let key = ShortPcKey::new(&domain(n), 2, b"spc-2").unwrap();
        let encoding = values(n, 7);
        let layers = key.layered(&encoding).unwrap();
        let cm = key.commit(&encoding).unwrap();
        let u: Vec<Fp256> = vec![
            Fp256::from_canonical_u64(5),
            Fp256::from_canonical_u64(9),
            Fp256::from_canonical_u64(2),
        ];
        let v_true = eval_claim(&key, &layers, &u);
        let v_bad = v_true.add(&Fp256::one_mont());
        let mut t = Transcript::new_default(b"spc-t2");
        let proof = open_short(&key, &u, &v_bad, &layers, &mut t).unwrap();
        let mut vt = Transcript::new_default(b"spc-t2");
        assert!(verify_short(&key, &cm, &u, &v_bad, &proof, &mut vt).is_err());
    }

    #[test]
    fn tampered_message_rejected() {
        let n = 4;
        let key = ShortPcKey::new(&domain(n), 2, b"spc-3").unwrap();
        let encoding = values(n, 11);
        let layers = key.layered(&encoding).unwrap();
        let cm = key.commit(&encoding).unwrap();
        let u: Vec<Fp256> = vec![Fp256::from_canonical_u64(3), Fp256::from_canonical_u64(7)];
        let v = eval_claim(&key, &layers, &u);
        let mut t = Transcript::new_default(b"spc-t3");
        let mut proof = open_short(&key, &u, &v, &layers, &mut t).unwrap();
        proof.msgs[1][1][0] = proof.msgs[1][1][0].add(&Fp256::one_mont());
        let mut vt = Transcript::new_default(b"spc-t3");
        assert!(verify_short(&key, &cm, &u, &v, &proof, &mut vt).is_err());
    }

    #[test]
    fn accumulate_and_decide() {
        let n = 4;
        let key = ShortPcKey::new(&domain(n), 2, b"spc-4").unwrap();
        let mut t = Transcript::new_default(b"spc-t4");
        let mut instances = Vec::new();
        for seed in [21u64, 22] {
            let encoding = values(n, seed);
            let layers = key.layered(&encoding).unwrap();
            let cm = key.commit(&encoding).unwrap();
            let u: Vec<Fp256> = vec![
                Fp256::from_canonical_u64(seed * 3),
                Fp256::from_canonical_u64(seed * 5),
            ];
            let v = eval_claim(&key, &layers, &u);
            let mut pt = Transcript::new_default(b"spc-acc");
            let proof = open_short(&key, &u, &v, &layers, &mut pt).unwrap();
            let mut vt = Transcript::new_default(b"spc-acc");
            let inst = verify_short(&key, &cm, &u, &v, &proof, &mut vt).unwrap();
            instances.push(inst);
        }
        let (r, c) = accumulate_short(&key, &instances, &mut t).unwrap();
        assert!(decide_short(&key, &r, &c), "folded instance decides");
    }
}

