//! Serval (Zhang et al., PolyU HK): slack-free l2-sound polynomial
//! commitments — the split-and-fold IPA — ported from the lattice-zk-lab
//! reference implementation onto this crate's `Poly` ring
//! (Z_Q[X]/(X^64+1), Q = 2^48−59).
//!
//! Serval refines the LaBRADOR-family IPA with **exact l2 bookkeeping**
//! (slack-free extraction): each split-and-fold round maintains
//! `t = ⟨v, v̄⟩` as the TRUE conjugate inner product, not a slack bound.
//!
//! * **One round** (`split_fold_round`): split `v → (v_L, v_R)`; the
//!   prover sends the conjugate quartet `(L, M1, M2, R)` with
//!   `L = ⟨v̄_L, v_L⟩`, `M1 = ⟨v̄_L, v_R⟩`, `M2 = ⟨v̄_R, v_L⟩`,
//!   `R = ⟨v̄_R, v_R⟩` (conjugation = the `flip` automorphism); both
//!   parties sample the challenge pair `(c, c')` and fold
//!   `v_next = c·v_L + c'·v_R` with the exact norm update
//!   `t_next = conj(c)·c·L + conj(c)·c'·M1 + conj(c')·c·M2 + conj(c')·c'·R`
//!   (flip-linearity: ring challenges contribute conjugated factors) —
//!   the norm chain is preserved exactly (no extractor slack).
//! * **The full IPA** (`run_ipa`): log₂(|v|) rounds down to a single
//!   ring element whose revealed norm pins the whole chain.
//!
//! The verifier (`verify_ipa`) replays the transcript, checks every
//! round's norm-chain update and the final `‖v_final‖² == t_last`
//! equality (the slack-free terminal).

use crate::ring::{cmod_div, Poly, N as POLY_N};
use lattice_core::transcript::{Transcript, TranscriptError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServalError {
    Transcript(TranscriptError),
    /// The norm-chain update failed (tampered quartet).
    NormChainFailed {
        round: usize,
    },
    /// The terminal norm equality failed.
    TerminalFailed,
    /// The vector length is not a power of two.
    Shape {
        len: usize,
    },
}

impl From<TranscriptError> for ServalError {
    fn from(e: TranscriptError) -> Self {
        ServalError::Transcript(e)
    }
}

/// One round's quartet + the exact norm update.
#[derive(Clone, Debug)]
pub struct Quartet {
    pub l: Poly,
    pub m1: Poly,
    pub m2: Poly,
    pub r: Poly,
    /// the challenge pair (c, c').
    pub c: Poly,
    pub cp: Poly,
    /// the exact norm after the fold.
    pub t_next: Poly,
}

/// A small (ternary) challenge ring element from the transcript.
fn challenge_small(transcript: &mut Transcript, label: &[u8]) -> Result<Poly, ServalError> {
    let bytes = transcript.challenge_bytes(label, POLY_N)?;
    let mut coeffs = [0i64; POLY_N];
    for (i, &b) in bytes.iter().enumerate() {
        coeffs[i] = match b % 3 {
            0 => 0i64,
            1 => 1,
            _ => -1,
        };
    }
    Ok(Poly(coeffs))
}

/// The conjugate inner product `⟨v̄, v⟩` (constant term = ‖cf(v)‖² in the
/// wraparound-free regime; the exact l2 readout). Conjugation is the
/// `flip` automorphism `a(X) ↦ a(X^{−1})` (upstream `polx_flip`).
fn norm_conj(v: &[Poly]) -> Poly {
    let vbar: Vec<Poly> = v.iter().map(|p| p.flip()).collect();
    Poly::sprod(&vbar, v)
}

/// One split-and-fold round: `v → (v_L, v_R)`; the quartet with the
/// challenge pair; the quadratic fold; the exact norm bookkeeping.
pub fn split_fold_round(
    v: &[Poly],
    transcript: &mut Transcript,
) -> Result<(Vec<Poly>, Quartet), ServalError> {
    let half = v.len() / 2;
    let v_l = &v[..half];
    let v_r = &v[half..];
    // the conjugate quartet (left factors conjugated)
    let vbar_l: Vec<Poly> = v_l.iter().map(|p| p.flip()).collect();
    let vbar_r: Vec<Poly> = v_r.iter().map(|p| p.flip()).collect();
    let l = Poly::sprod(&vbar_l, v_l);
    let m1 = Poly::sprod(&vbar_l, v_r);
    let m2 = Poly::sprod(&vbar_r, v_l);
    let r = Poly::sprod(&vbar_r, v_r);
    // challenge pair (c, c') with the quadratic fold
    let c = challenge_small(transcript, b"serval:c")?;
    let cp = challenge_small(transcript, b"serval:cp")?;
    let v_next: Vec<Poly> = v_l
        .iter()
        .zip(v_r.iter())
        .map(|(a, b)| a.mul(&c).add(&b.mul(&cp)))
        .collect();
    // the exact norm bookkeeping with RING challenges — the conjugated
    // challenge factors the flip-linearity requires:
    // t' = conj(c)·c·L + conj(c)·c'·M1 + conj(c')·c·M2 + conj(c')·c'·R
    // (constants would commute past flip; ring elements do not).
    let c_conj = c.flip();
    let cp_conj = cp.flip();
    let t_next = l
        .mul(&c_conj)
        .mul(&c)
        .add(&m1.mul(&c_conj).mul(&cp))
        .add(&m2.mul(&cp_conj).mul(&c))
        .add(&r.mul(&cp_conj).mul(&cp));
    Ok((
        v_next,
        Quartet {
            l,
            m1,
            m2,
            r,
            c,
            cp,
            t_next,
        },
    ))
}

/// The full log-round IPA with the exact-norm chain: returns the final
/// ring element, the quartet history, and the initial norm claim.
pub fn run_ipa(
    v: &[Poly],
    transcript: &mut Transcript,
) -> Result<(Poly, Vec<Quartet>, Poly), ServalError> {
    if !v.len().is_power_of_two() || v.is_empty() {
        return Err(ServalError::Shape { len: v.len() });
    }
    let t_final = norm_conj(v);
    let mut history = Vec::new();
    let mut cur = v.to_vec();
    while cur.len() > 1 {
        let (v_next, quartet) = split_fold_round(&cur, transcript)?;
        history.push(quartet);
        cur = v_next;
    }
    Ok((cur[0], history, t_final))
}

/// The verifier: replay the challenges and walk the exact-norm chain:
/// * per round: (a) the parent-consistency `t == L + R` (the norm is
///   additive over the split: `⟨v̄, v⟩ = ⟨v̄_L, v_L⟩ + ⟨v̄_R, v_R⟩` as
///   full ring elements); (b) the fold update
///   `t' = conj(c)·c·L + conj(c)·c'·M1 + conj(c')·c·M2 + conj(c')·c'·R`;
/// * terminal: the revealed final element satisfies
///   `⟨v̄_final, v_final⟩ == t_last` (slack-free).
pub fn verify_ipa(
    t0: &Poly,
    history: &[Quartet],
    v_final: &Poly,
    transcript: &mut Transcript,
) -> Result<(), ServalError> {
    let mut t = *t0;
    for (round, q) in history.iter().enumerate() {
        // replay the challenge pair
        let c = challenge_small(transcript, b"serval:c")?;
        let cp = challenge_small(transcript, b"serval:cp")?;
        if c != q.c || cp != q.cp {
            return Err(ServalError::NormChainFailed { round });
        }
        // (a) parent consistency: t == L + R
        if t != q.l.add(&q.r) {
            return Err(ServalError::NormChainFailed { round });
        }
        // (b) the fold update
        let c_conj = c.flip();
        let cp_conj = cp.flip();
        let expect =
            q.l.mul(&c_conj)
                .mul(&c)
                .add(&q.m1.mul(&c_conj).mul(&cp))
                .add(&q.m2.mul(&cp_conj).mul(&c))
                .add(&q.r.mul(&cp_conj).mul(&cp));
        if expect != q.t_next {
            return Err(ServalError::NormChainFailed { round });
        }
        t = q.t_next;
    }
    // terminal: the revealed final element's norm equals the chain value
    let final_norm = norm_conj(&[*v_final]);
    if final_norm != t {
        return Err(ServalError::TerminalFailed);
    }
    Ok(())
}

/// The integer readout of a norm claim in the wraparound-free regime
/// (ct = ‖·‖² when the true norm < Q/2).
pub fn norm_readout(t: &Poly) -> i128 {
    i128::from(cmod_div(i128::from(t.0[0])))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_vec(m: usize, tag: &[u8], span: i64) -> Vec<Poly> {
        (0..m)
            .map(|i| {
                let bytes = Transcript::xof(
                    b"serval-test",
                    &[tag, &(i as u32).to_le_bytes()].concat(),
                    POLY_N,
                );
                let mut coeffs = [0i16; POLY_N];
                for (k, &b) in bytes.iter().enumerate() {
                    coeffs[k] = ((b as i64 % (2 * span + 1)) - span) as i16;
                }
                Poly::from_i16(&coeffs)
            })
            .collect()
    }

    #[test]
    fn split_fold_exact_norm_chain() {
        let v = small_vec(8, b"v", 4);
        let mut t = Transcript::new_default(b"lzx-serval");
        let t0 = norm_conj(&v);
        let (v_next, q) = split_fold_round(&v, &mut t).ok().unwrap();
        // the exact norm chain: t' == norm(v_next) directly
        let t_direct = norm_conj(&v_next);
        assert_eq!(q.t_next, t_direct);
        let _ = t0;
    }

    #[test]
    fn ipa_end_to_end_and_tampered() {
        let v = small_vec(8, b"iv", 4);
        let t0 = norm_conj(&v);
        let mut t = Transcript::new_default(b"lzx-serval");
        let (v_final, history, t_final) = run_ipa(&v, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-serval");
        assert!(verify_ipa(&t0, &history, &v_final, &mut vt).is_ok());
        let _ = &t_final;
        // tampered quartet norm update rejected
        let mut bad_history = history.clone();
        if !bad_history.is_empty() {
            bad_history[0].t_next = bad_history[0].t_next.add(&Poly::one());
        }
        let mut vt2 = Transcript::new_default(b"lzx-serval");
        assert!(verify_ipa(&t0, &bad_history, &v_final, &mut vt2).is_err());
        // tampered final element rejected at the terminal
        let v_bad = v_final.add(&Poly::one());
        let mut vt3 = Transcript::new_default(b"lzx-serval");
        assert!(verify_ipa(&t0, &history, &v_bad, &mut vt3).is_err());
        // tampered initial norm claim rejected
        let t0_bad = t0.add(&Poly::one());
        let mut vt4 = Transcript::new_default(b"lzx-serval");
        assert!(verify_ipa(&t0_bad, &history, &v_final, &mut vt4).is_err());
    }

    #[test]
    fn shape_requires_power_of_two() {
        let v = small_vec(6, b"sv", 4);
        let mut t = Transcript::new_default(b"lzx-serval");
        assert!(matches!(
            run_ipa(&v, &mut t),
            Err(ServalError::Shape { .. })
        ));
    }

    #[test]
    fn norm_readout_wraparound_free() {
        // small vectors: ct(t) == the true integer norm
        let v = small_vec(4, b"nr", 3);
        let t = norm_conj(&v);
        let mut expect: i128 = 0;
        for p in &v {
            for &c in p.0.iter() {
                expect += i128::from(c) * i128::from(c);
            }
        }
        assert_eq!(norm_readout(&t), expect);
    }
}
