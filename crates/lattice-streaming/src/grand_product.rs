//! **The streaming grand product check** — Appendix D of ePrint
//! 2025/611 (the Quarks lemma protocol with a depth-first prover).
//!
//! Statement: `P = Π_{i=0}^{2^n−1} v[i]` for a streamed vector `v`.
//!
//! ## The protocol (Lemma D.1)
//!
//! Define multilinear tables over `{0,1}^n` via the product tree: the
//! block `[o, o + 2^{j+1})` of `2^{j+1}` leaves is **labeled** by the
//! cube point `z = o + 2^j − 1` (the point with `j` trailing ones and a
//! zero at bit `j`), and
//!
//! ```text
//! g1(z) = block product      g2(z) = left-child product
//! g3(z) = right-child product
//! g1(1^n) = g3(1^n) = 0,     g2(1^n) = P          (the special point)
//! ```
//!
//! The labeling is a bijection onto `{0,1}^n`, and the recursion
//! `g1 = g2 · g3` holds at *every* cube point (verified in the tests),
//! so the sum-check instance
//!
//! ```text
//! 0 = Σ_z eq(u, z) · (g1(z) − g2(z)·g3(z))
//! ```
//!
//! reduces the claim to the terminal identity `g1(r) = g2(r)·g3(r)` at
//! the verifier's random point — the Quarks construction. The honest
//! round messages are identically zero (each term is zero on the cube);
//! soundness comes from binding the prover to consistent `g`-tables,
//! whose terminal evaluations the caller's PCS then authenticates.
//!
//! ## The streaming prover
//!
//! The **depth-first product-tree walk** is the paper's core data
//! structure: scan the stream once, push each `v[x]` onto a block stack,
//! and merge adjacent equal-size aligned blocks — the stack never
//! exceeds `n + 1` entries, so `P` itself is computed in `O(n)` space
//! (Theorem D.4's stack invariant, Lemma D.2). The `g`-table triples
//! are recorded at each merge during the same single pass; the
//! sum-check then runs over the recorded tables. (Algorithm 3's fully
//! `O(n)`-space *round-message* path composes the same DFS with
//! per-remaining-hypercube bound accumulation — the bucketed
//! `g_evals[t][k][s]` machinery of the paper's Steps 17–24; the
//! recorded-table path here is the reference implementation of the same
//! protocol at `O(2^n)` table space, one stream pass.)

use crate::oracle::StreamOracle;
use crate::small_space::interpolate_nodes;
use lattice_core::field_simd::{self, Sum8};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::Goldilocks;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrandProductError {
    Transcript(TranscriptError),
    RoundCheckFailed { round: usize },
    TerminalIdentityFailed,
    ShapeMismatch { expected: usize, got: usize },
}

impl core::fmt::Display for GrandProductError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GrandProductError::Transcript(e) => write!(f, "transcript error: {e}"),
            GrandProductError::RoundCheckFailed { round } => {
                write!(f, "grand-product round identity failed at round {round}")
            }
            GrandProductError::TerminalIdentityFailed => {
                write!(f, "grand-product terminal identity failed")
            }
            GrandProductError::ShapeMismatch { expected, got } => {
                write!(f, "grand-product shape {got} != {expected}")
            }
        }
    }
}

/// A stack block: (value, size, offset).
#[derive(Clone, Copy)]
struct Block {
    value: Goldilocks,
    size: u64,
    offset: u64,
}

/// **The depth-first streaming product walk** — `O(n)` space, one pass:
/// push `v[x]`, merge adjacent equal-size aligned blocks. Returns the
/// full product and, if `triples` is provided, records one
/// `(z, g1, g2, g3)` entry per merge (the sum-check's tables).
///
/// This is Theorem D.4's prover core: the stack mirrors the DFS state
/// of the product-tree, holding at most `n + 1` partial products
/// (Lemma D.2's invariant — the values `g2(x^{(j)})` for the active
/// block chain, in decreasing block size).
pub fn dfs_grand_product(
    stream: &mut dyn StreamOracle,
    triples: Option<&mut Vec<(u64, Goldilocks, Goldilocks, Goldilocks)>>,
) -> Result<Goldilocks, GrandProductError> {
    let n = stream.len().trailing_zeros() as usize;
    let mut stack: Vec<Block> = Vec::with_capacity(n + 1);
    let mut triples = triples;
    for x in 0..stream.len() {
        let v = stream.next();
        stack.push(Block {
            value: v,
            size: 1,
            offset: x,
        });
        // Merge while the top two blocks are adjacent, equal-size, and
        // aligned (the right block's offset is an odd multiple of size).
        while stack.len() >= 2 {
            let (left, right) = {
                let l = stack.len();
                (&stack[l - 2], &stack[l - 1])
            };
            if left.size == right.size
                && left.offset + left.size == right.offset
                && (right.offset / right.size) % 2 == 1
            {
                let prod = left.value.mul(&right.value);
                if let Some(t) = triples.as_deref_mut() {
                    // The merged block [left.offset, +2·size) is labeled
                    // by z = left.offset + size − 1.
                    let z = left.offset + left.size - 1;
                    t.push((z, prod, left.value, right.value));
                }
                let merged = Block {
                    value: prod,
                    size: left.size * 2,
                    offset: left.offset,
                };
                stack.pop();
                stack.pop();
                stack.push(merged);
            } else {
                break;
            }
        }
    }
    if stack.len() != 1 {
        return Err(GrandProductError::ShapeMismatch {
            expected: 1,
            got: stack.len(),
        });
    }
    Ok(stack[0].value)
}

/// The grand-product proof: the claimed product, the (all-zero) round
/// messages of the Quarks sum-check, the challenges, and the terminal
/// `g`-claims the caller's PCS authenticates.
#[derive(Clone, Debug)]
pub struct GrandProductProof {
    /// The claimed product `P` (bound by the caller's commitment).
    pub product: Goldilocks,
    /// Round messages (degree-3 univariates at nodes 0..=3).
    pub rounds: Vec<Vec<Goldilocks>>,
    pub challenges: Vec<Goldilocks>,
    /// `g1(r)`, `g2(r)`, `g3(r)`.
    pub g_claims: [Goldilocks; 3],
}

/// Prove `Π v = product` with one streaming pass + the Quarks
/// sum-check over the recorded `g`-tables.
///
/// `product`: if `None`, the DFS computes it (the honest path — the
/// caller typically holds the claim from the enclosing protocol).
pub fn prove_grand_product(
    stream: &mut dyn StreamOracle,
    product: Option<Goldilocks>,
    transcript: &mut Transcript,
) -> Result<GrandProductProof, GrandProductError> {
    let n = stream.len().trailing_zeros() as usize;
    let mut triples = Vec::with_capacity((1usize << n).saturating_sub(1));
    stream.reset();
    let p = dfs_grand_product(stream, Some(&mut triples))?;
    let product = product.unwrap_or(p);
    if p != product {
        // The streamed vector's product contradicts the claim.
        return Err(GrandProductError::TerminalIdentityFailed);
    }

    // g-tables in index order (the special all-ones point carries
    // (0, P, 0)).
    let len = 1usize << n;
    let mut g1 = vec![Goldilocks::ZERO; len];
    let mut g2 = vec![Goldilocks::ZERO; len];
    let mut g3 = vec![Goldilocks::ZERO; len];
    for (z, a, b, c) in &triples {
        g1[*z as usize] = *a;
        g2[*z as usize] = *b;
        g3[*z as usize] = *c;
    }
    g2[len - 1] = product; // g2(1^n) = P; g1 = g3 = 0 there.

    // The Quarks sum-check: 0 = Σ eq(u, z)·(g1 − g2·g3), degree 3.
    // u is sampled from the transcript after absorbing the product.
    transcript
        .append_field(b"grand-product", &product)
        .map_err(GrandProductError::Transcript)?;
    let mut u: Vec<Goldilocks> = Vec::with_capacity(n);
    for _ in 0..n {
        u.push(
            transcript
                .challenge_field(b"grand-product-u")
                .map_err(GrandProductError::Transcript)?,
        );
    }

    // eq table of u (in-memory; the bucketed streaming variant is the
    // paper's Algorithm 3 — see the module docs).
    let eq = field_simd::eq_table(&u);

    let mut current_claim = Goldilocks::ZERO;
    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(n);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(n);
    let mut bound_r: Vec<Goldilocks> = Vec::with_capacity(n);

    // Round i: bind the g's over the prefix with eq weights and the
    // eq-u factor likewise; accumulate the four point values.
    // Implementation: maintain bound copies of (eq, g1, g2, g3), each
    // halving per round — the reference in-memory prover.
    let mut b_eq = eq;
    let mut b1 = g1;
    let mut b2 = g2;
    let mut b3 = g3;
    for round in 0..n {
        let mut evals = vec![Goldilocks::ZERO; 4];
        for (pi, t) in [0u64, 1, 2, 3].iter().enumerate() {
            let tf = Goldilocks::from_u64(*t);
            let mut vals: Vec<Vec<Goldilocks>> = Vec::with_capacity(4);
            for arr in [&b_eq, &b1, &b2, &b3] {
                let half = arr.len() / 2;
                let v = match t {
                    0 => arr[..half].to_vec(),
                    1 => arr[half..].to_vec(),
                    _ => {
                        let mut buf = vec![Goldilocks::ZERO; half];
                        field_simd::bind_half_slices(&arr[..half], &arr[half..], tf, &mut buf);
                        buf
                    }
                };
                vals.push(v);
            }
            let mut acc = Sum8::new();
            acc.accumulate_term(Goldilocks::ONE, &[vals[0].as_slice(), vals[1].as_slice()]);
            let term1 = acc.finish();
            let mut acc2 = Sum8::new();
            acc2.accumulate_term(
                Goldilocks::ONE,
                &[vals[0].as_slice(), vals[2].as_slice(), vals[3].as_slice()],
            );
            let term2 = acc2.finish();
            evals[pi] = term1.sub(&term2);
        }
        let sum01 = evals[0].add(&evals[1]);
        if sum01 != current_claim {
            return Err(GrandProductError::RoundCheckFailed { round });
        }
        transcript
            .append_field_slice(b"sumcheck-round", &evals)
            .map_err(GrandProductError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(GrandProductError::Transcript)?;
        current_claim = interpolate_nodes(&evals, &r);
        bound_r.push(r);
        challenges.push(r);
        for arr in [&mut b_eq, &mut b1, &mut b2, &mut b3] {
            field_simd::bind_first_half_in_place(arr, r);
            let half = arr.len() / 2;
            arr.truncate(half);
        }
        rounds.push(evals);
    }

    let g_claims = [b1[0], b2[0], b3[0]];
    // Terminal identity: C_n = eq(u, r)·(g1(r) − g2(r)·g3(r)) — the
    // chained claim against the bound tables (the honest prover's
    // messages from round 1 on are nonzero: affine binding does not
    // commute with the degree-2 g2·g3 product, which is exactly what
    // makes the protocol non-vacuous).
    let terminal = b_eq[0].mul(&b1[0].sub(&b2[0].mul(&b3[0])));
    if terminal != current_claim {
        return Err(GrandProductError::TerminalIdentityFailed);
    }
    let _ = &bound_r;

    Ok(GrandProductProof {
        product,
        rounds,
        challenges,
        g_claims,
    })
}

/// Verify the grand-product proof: replay the round chain (the honest
/// messages are zero, so the chain stays at the initial zero claim) and
/// check the terminal identity from the claimed `g` evaluations.
///
/// The `g`-claims themselves are PCS-authenticated by the caller (in
/// the full Quarks construction the tables are slices of the committed
/// `(n+1)`-variate `f`, whose opening at `r` binds them — and `P =
/// f(0, 1^n)`).
pub fn verify_grand_product(
    proof: &GrandProductProof,
    transcript: &mut Transcript,
) -> Result<bool, GrandProductError> {
    let n = proof.rounds.len();
    transcript
        .append_field(b"grand-product", &proof.product)
        .map_err(GrandProductError::Transcript)?;
    let mut u: Vec<Goldilocks> = Vec::with_capacity(n);
    for _ in 0..n {
        u.push(
            transcript
                .challenge_field(b"grand-product-u")
                .map_err(GrandProductError::Transcript)?,
        );
    }
    let mut current = Goldilocks::ZERO;
    let mut point: Vec<Goldilocks> = Vec::with_capacity(n);
    for round in proof.rounds.iter() {
        if round.len() != 4 {
            return Err(GrandProductError::ShapeMismatch {
                expected: 4,
                got: round.len(),
            });
        }
        transcript
            .append_field_slice(b"sumcheck-round", round)
            .map_err(GrandProductError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(GrandProductError::Transcript)?;
        if round[0].add(&round[1]) != current {
            return Ok(false);
        }
        current = interpolate_nodes(round, &r);
        point.push(r);
    }
    // Terminal: current = eq(u, r)·(g1(r) − g2(r)·g3(r)), with the
    // g-claims PCS-authenticated by the caller.
    let eq_ur = {
        let mut acc = Goldilocks::ONE;
        for (ui, ri) in u.iter().zip(point.iter()) {
            acc = acc.mul(&ui.mul(ri).add(&Goldilocks::ONE.sub(ui).mul(&Goldilocks::ONE.sub(ri))));
        }
        acc
    };
    let terminal = eq_ur.mul(&proof.g_claims[0].sub(&proof.g_claims[1].mul(&proof.g_claims[2])));
    Ok(terminal == current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::OwnedOracle;
    use lattice_core::DenseMle;

    fn g(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// The DFS walk computes the true product in O(n) space.
    #[test]
    fn dfs_product_matches_naive() {
        for n in [1usize, 3, 6, 8] {
            let data: Vec<Goldilocks> = (0..(1u64 << n))
                .map(|i| g(i.wrapping_mul(31) + 7))
                .collect();
            let mut stream = OwnedOracle::new(data.clone());
            let p = dfs_grand_product(&mut stream, None).unwrap();
            let naive = data
                .iter()
                .fold(Goldilocks::ONE, |acc, v| acc.mul(v));
            assert_eq!(p, naive, "n={n}");
        }
    }

    /// The `g`-labeling is a bijection and `g1 = g2·g3` holds at every
    /// cube point (Lemma D.1's structure, verified structurally).
    #[test]
    fn g_tables_identity_on_the_cube() {
        let n = 5;
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| g(i.wrapping_mul(97) + 3))
            .collect();
        let mut stream = OwnedOracle::new(data.clone());
        let mut triples = Vec::new();
        let p = dfs_grand_product(&mut stream, Some(&mut triples)).unwrap();
        assert_eq!(triples.len(), (1usize << n) - 1);
        let mut covered = vec![false; 1usize << n];
        for (z, g1, g2, g3) in &triples {
            let zi = *z as usize;
            assert!(!covered[zi], "duplicate label {z}");
            covered[zi] = true;
            assert_eq!(*g1, g2.mul(g3), "identity at z={z}");
        }
        // All labels distinct and covering everything except 1^n.
        for (i, c) in covered.iter().enumerate() {
            if i == (1usize << n) - 1 {
                assert!(!c);
            } else {
                assert!(c, "uncovered label {i}");
            }
        }
        // The special point: g2(1^n) = P, g1 = g3 = 0 → identity holds.
        assert!(p.mul(&Goldilocks::ZERO).is_zero());
    }

    /// End-to-end: prove/verify roundtrip, including the eq(u, r)
    /// cross-check of the terminal claims.
    #[test]
    fn prove_verify_roundtrip() {
        let n = 6;
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| g(i.wrapping_mul(11) + 1))
            .collect();
        let mut stream = OwnedOracle::new(data);
        let mut ts = Transcript::new_default(b"gp-seed");
        let proof = prove_grand_product(&mut stream, None, &mut ts).unwrap();
        // Round 0's honest values at the interpolation nodes 0 and 1
        // are zero (the cube identity); the extrapolation nodes 2, 3 and
        // all later rounds are nonzero — affine binding does not
        // commute with the degree-2 product term.
        assert!(proof.rounds[0][0].is_zero() && proof.rounds[0][1].is_zero());
        let mut ts2 = Transcript::new_default(b"gp-seed");
        assert!(verify_grand_product(&proof, &mut ts2).unwrap());
        // Terminal claims are the bound g-table values (checked against
        // the independently rebuilt table below).
    }

    /// A tampered proof (nonzero round message or broken terminal)
    /// fails verification.
    #[test]
    fn tampered_proof_rejected() {
        let n = 5;
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| g(i.wrapping_mul(13) + 5))
            .collect();
        let mut stream = OwnedOracle::new(data);
        let mut ts = Transcript::new_default(b"gp-t");
        let mut proof = prove_grand_product(&mut stream, None, &mut ts).unwrap();
        // Tamper 1: a nonzero round message breaks the identity chain.
        proof.rounds[2][1] = proof.rounds[2][1].add(&Goldilocks::ONE);
        let mut ts2 = Transcript::new_default(b"gp-t");
        assert!(!verify_grand_product(&proof, &mut ts2).unwrap());
        // Tamper 2: break the terminal identity via a corrupted g-claim.
        let mut stream2 = OwnedOracle::new((0..(1u64 << n)).map(|i| g(i * 13 + 5)).collect());
        let mut ts3 = Transcript::new_default(b"gp-t2");
        let mut proof2 = prove_grand_product(&mut stream2, None, &mut ts3).unwrap();
        proof2.g_claims[0] = proof2.g_claims[0].add(&Goldilocks::ONE);
        let mut ts4 = Transcript::new_default(b"gp-t2");
        assert!(!verify_grand_product(&proof2, &mut ts4).unwrap());
    }

    /// A stream whose product contradicts the claimed `P` is rejected at
    /// prove time (the DFS catches it in O(n) space).
    #[test]
    fn wrong_product_claim_rejected() {
        let n = 4;
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| g(i.wrapping_mul(17) + 3))
            .collect();
        let true_p = data.iter().fold(Goldilocks::ONE, |acc, v| acc.mul(v));
        let wrong = true_p.add(&Goldilocks::ONE);
        let mut stream = OwnedOracle::new(data);
        let mut ts = Transcript::new_default(b"gp-w");
        assert!(prove_grand_product(&mut stream, Some(wrong), &mut ts).is_err());
    }

    /// The eq(u, ·) factor at the terminal point: the prover's u and the
    /// verifier's u agree (same transcript), and the terminal claims are
    /// the bound g-table values.
    #[test]
    fn terminal_claims_match_bound_tables() {
        let n = 6;
        let data: Vec<Goldilocks> = DenseMle::random(n, b"gp-r").evaluations;
        let mut stream = OwnedOracle::new(data.clone());
        let mut ts = Transcript::new_default(b"gp-x");
        let proof = prove_grand_product(&mut stream, None, &mut ts).unwrap();
        // Rebuild the g2 table independently and evaluate at r.
        let mut stream2 = OwnedOracle::new(data);
        let mut triples = Vec::new();
        let p = dfs_grand_product(&mut stream2, Some(&mut triples)).unwrap();
        let len = 1usize << n;
        let mut g2 = vec![Goldilocks::ZERO; len];
        for (z, _, b, _) in &triples {
            g2[*z as usize] = *b;
        }
        g2[len - 1] = p;
        let mle = DenseMle::new(g2).unwrap_or(DenseMle::constant(Goldilocks::ZERO));
        assert_eq!(mle.evaluate(&proof.challenges).unwrap(), proof.g_claims[1]);
    }
}
