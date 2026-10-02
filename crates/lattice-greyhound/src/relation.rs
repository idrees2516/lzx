//! The principal relation R of LaBRADOR §5.1 (Greyhound §2.4): short solutions
//! to dot-product constraint systems over R_q.
//!
//! A statement is a family `F` of *fully vanishing* quadratic dot-product
//! functions and a family `F'` of *constant-term-only* functions
//!
//! ```text
//! f(s_1..s_r)  = Σ_{i,j} a_ij ⟨s_i, s_j⟩ + Σ_i ⟨φ_i, s_i⟩ − b          (F)
//! ct(f'(s))    = 0                                                     (F')
//! Σ_i ‖s_i‖² ≤ β²
//! ```
//!
//! with a witness of `r` vectors of ranks `n_i`. This is the native language of
//! both papers: Greyhound's R1 relation (§4.3) and LaBRADOR's recursion target
//! (§5.3) are instances, and the §6 R1CS reductions compile into it.
//!
//! Constraints are sparse (term lists addressing `(vector, offset)` slices),
//! matching the reference's `sparsecnst`. Witness vectors carry a *digit
//! origin*: chunks of the decomposed amortized opening `z = Σ 2^{d·b} z^(d)`
//! are labelled with their digit `d`, which makes the quadratic structure
//! chunking-invariant (see `protocol.rs`).

use crate::ring::{sprod, Poly};

/// Spec of one witness vector: rank + digit origin (None = not part of a
/// decomposed z — e.g. the v-vector of outer-commitment openings).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorSpec {
    pub n: usize,
    pub digit: Option<usize>,
}

impl VectorSpec {
    pub fn plain(n: usize) -> Self {
        Self { n, digit: None }
    }
    pub fn z_part(n: usize, digit: usize) -> Self {
        Self { n, digit: Some(digit) }
    }
}

/// One linear term: `⟨phi, s_idx[off..off+phi.len()]⟩`.
#[derive(Clone, Debug)]
pub struct Term {
    pub idx: usize,
    pub off: usize,
    pub phi: Vec<Poly>,
}

/// One dot-product constraint (an element of F or F').
#[derive(Clone, Debug)]
pub struct DotCnst {
    pub terms: Vec<Term>,
    /// Quadratic entries `a_ij ⟨s_i, s_j⟩` for i ≤ j (symmetric extension with
    /// the (2 − [i=j]) factor at evaluation).
    pub a: Vec<(usize, usize, Poly)>,
    /// The target b; `None` for homogeneous constraints.
    pub b: Option<Poly>,
    /// F'-family: only the constant term must vanish.
    pub ct_only: bool,
}

impl DotCnst {
    pub fn homogeneous(terms: Vec<Term>) -> Self {
        Self { terms, a: Vec::new(), b: None, ct_only: false }
    }

    pub fn with_b(terms: Vec<Term>, b: Poly) -> Self {
        Self { terms, a: Vec::new(), b: Some(b), ct_only: false }
    }

    /// Evaluate at a witness (returns the full ring element; callers check = 0
    /// or ct = 0 depending on `ct_only`).
    pub fn eval(&self, s: &[Vec<Poly>]) -> Poly {
        let mut acc = Poly::zero();
        for t in &self.terms {
            let slice = &s[t.idx][t.off..t.off + t.phi.len()];
            acc.add_assign(&sprod(&t.phi, slice));
        }
        for &(i, j, ref coeff) in &self.a {
            let prod = sprod(&s[i], &s[j]);
            let scaled = if i == j { coeff.mul(&prod) } else { coeff.mul(&prod).scale(2) };
            acc.add_assign(&scaled);
        }
        if let Some(b) = &self.b {
            acc.sub_assign(b);
        }
        acc
    }

    /// Check the constraint at a witness.
    pub fn check(&self, s: &[Vec<Poly>]) -> bool {
        let v = self.eval(s);
        if self.ct_only {
            v.constant_term() == 0
        } else {
            v.is_zero()
        }
    }
}

/// A principal statement: (F, F', β) over `r` witness vectors.
#[derive(Clone, Debug)]
pub struct PrincipalStatement {
    pub vectors: Vec<VectorSpec>,
    /// The F family (fully vanishing).
    pub cnst: Vec<DotCnst>,
    /// The F' family (constant-term only).
    pub ct_cnst: Vec<DotCnst>,
    /// The squared norm bound β².
    pub betasq: u64,
    /// Statement digest (binds everything; the Fiat-Shamir root).
    pub digest: [u8; 32],
}

impl PrincipalStatement {
    pub fn new(
        vectors: Vec<VectorSpec>,
        cnst: Vec<DotCnst>,
        ct_cnst: Vec<DotCnst>,
        betasq: u64,
    ) -> Self {
        let mut st = Self { vectors, cnst, ct_cnst, betasq, digest: [0; 32] };
        st.digest = st.content_digest();
        st
    }

    pub fn total_rank(&self) -> usize {
        self.vectors.iter().map(|v| v.n).sum()
    }

    /// Check every structural invariant: term slices in range, a-indices valid.
    pub fn validate(&self) -> Result<(), String> {
        for (k, c) in self.cnst.iter().chain(self.ct_cnst.iter()).enumerate() {
            for t in &c.terms {
                if t.idx >= self.vectors.len() {
                    return Err(format!("cnst {k}: term idx {} out of range", t.idx));
                }
                if t.off + t.phi.len() > self.vectors[t.idx].n {
                    return Err(format!(
                        "cnst {k}: term slice [{}, {}) exceeds rank {} of vector {}",
                        t.off,
                        t.off + t.phi.len(),
                        self.vectors[t.idx].n,
                        t.idx
                    ));
                }
            }
            for &(i, j, _) in &c.a {
                if i > j || j >= self.vectors.len() {
                    return Err(format!("cnst {k}: bad a-index ({i},{j})"));
                }
            }
        }
        Ok(())
    }

    /// Full constraint check (F and F') at a witness — the *statement-side*
    /// check used by tests and the final verification path.
    pub fn check_all(&self, s: &[Vec<Poly>]) -> Result<(), String> {
        if s.len() != self.vectors.len() {
            return Err("witness multiplicity mismatch".into());
        }
        for (i, v) in self.vectors.iter().enumerate() {
            if s[i].len() != v.n {
                return Err(format!("witness vector {i} rank mismatch"));
            }
        }
        let normsq: u64 = s.iter().map(|v| v.iter().map(|p| p.normsq()).sum::<u64>()).sum();
        if normsq > self.betasq {
            return Err(format!("witness norm² {normsq} exceeds bound {}", self.betasq));
        }
        for (k, c) in self.cnst.iter().enumerate() {
            if !c.check(s) {
                return Err(format!("constraint {k} (F) violated"));
            }
        }
        for (k, c) in self.ct_cnst.iter().enumerate() {
            if !c.check(s) {
                return Err(format!("constraint {k} (F') violated"));
            }
        }
        Ok(())
    }

    fn content_digest(&self) -> [u8; 32] {
        use lattice_core::keccak::KeccakSponge;
        let mut h = KeccakSponge::new_sha3_256();
        h.update(b"greyhound/principal/v1");
        h.update(&(self.vectors.len() as u64).to_le_bytes());
        for v in &self.vectors {
            h.update(&(v.n as u64).to_le_bytes());
            h.update(&[v.digit.map(|d| d as u8).unwrap_or(255)]);
        }
        h.update(&self.betasq.to_le_bytes());
        let absorb_cnst = |h: &mut KeccakSponge, c: &DotCnst, tag: &[u8]| {
            h.update(tag);
            h.update(&(c.terms.len() as u64).to_le_bytes());
            for t in &c.terms {
                h.update(&(t.idx as u64).to_le_bytes());
                h.update(&(t.off as u64).to_le_bytes());
                for p in &t.phi {
                    h.update(&p.to_le_bytes());
                }
            }
            h.update(&(c.a.len() as u64).to_le_bytes());
            for (i, j, coeff) in &c.a {
                h.update(&(*i as u64).to_le_bytes());
                h.update(&(*j as u64).to_le_bytes());
                h.update(&coeff.to_le_bytes());
            }
            match &c.b {
                None => h.update(b"hom"),
                Some(b) => h.update(&b.to_le_bytes()),
            }
            h.update(&[u8::from(c.ct_only)]);
        };
        h.update(&(self.cnst.len() as u64).to_le_bytes());
        for c in &self.cnst {
            absorb_cnst(&mut h, c, b"F");
        }
        h.update(&(self.ct_cnst.len() as u64).to_le_bytes());
        for c in &self.ct_cnst {
            absorb_cnst(&mut h, c, b"Fp");
        }
        let out = h.finalize(32);
        out.try_into().unwrap()
    }
}

/// A witness: `r` vectors of ring elements.
#[derive(Clone, Debug, Default)]
pub struct PrincipalWitness {
    pub s: Vec<Vec<Poly>>,
}

impl PrincipalWitness {
    pub fn new(s: Vec<Vec<Poly>>) -> Self {
        Self { s }
    }

    pub fn rank(&self, i: usize) -> usize {
        self.s[i].len()
    }

    pub fn normsq(&self) -> u64 {
        self.s.iter().map(|v| v.iter().map(|p| p.normsq()).sum::<u64>()).sum()
    }

    pub fn per_vector_normsq(&self) -> Vec<u64> {
        self.s.iter().map(|v| v.iter().map(|p| p.normsq()).sum::<u64>()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_vec(n: usize, seed: u64) -> Vec<Poly> {
        (0..n)
            .map(|i| {
                let mut p = [0i64; 64];
                for (j, c) in p.iter_mut().enumerate() {
                    *c = (((i * 53 + j * 29 + seed as usize * 7) % 11) as i64) - 5;
                }
                Poly(p)
            })
            .collect()
    }

    #[test]
    fn constraint_eval_linear_and_quadratic() {
        // s = (s0, s1); f = ⟨φ, s0⟩ + a·⟨s1, s1⟩ − b with b = the honest value
        let s0 = small_vec(3, 1);
        let s1 = small_vec(2, 2);
        let phi = small_vec(3, 3);
        let a = Poly::constant(5);
        let honest = sprod(&phi, &s0).add(&a.mul(&sprod(&s1, &s1)));
        let c = DotCnst {
            terms: vec![Term { idx: 0, off: 0, phi: phi.clone() }],
            a: vec![(1, 1, a)],
            b: Some(honest),
            ct_only: false,
        };
        let s = vec![s0, s1];
        assert!(c.check(&s));
        // tamper b
        let mut c2 = c.clone();
        c2.b = Some(honest.add(&Poly::constant(1)));
        assert!(!c2.check(&s));
    }

    #[test]
    fn symmetric_a_evaluates_doubled() {
        // a_12 ⟨s1,s2⟩ with the symmetric extension = 2·a·⟨s1,s2⟩
        let s1 = small_vec(2, 10);
        let s2 = small_vec(2, 20);
        let a = Poly::constant(3);
        let honest = a.mul(&sprod(&s1, &s2)).scale(2);
        let c = DotCnst {
            terms: vec![],
            a: vec![(0, 1, a)],
            b: Some(honest),
            ct_only: false,
        };
        assert!(c.check(&[s1, s2]));
    }

    #[test]
    fn ct_only_constraint() {
        let s0 = small_vec(2, 30);
        let phi = small_vec(2, 31);
        let val = sprod(&phi, &s0);
        // b agrees with the honest value ONLY in the constant term — the F'
        // semantics (the higher coefficients are free)
        let mut b = small_vec(2, 99).pop().unwrap();
        b.0[0] = val.constant_term();
        let c = DotCnst {
            terms: vec![Term { idx: 0, off: 0, phi }],
            a: vec![],
            b: Some(b),
            ct_only: true,
        };
        assert!(c.check(std::slice::from_ref(&s0)));
        // tamper the b constant term — the ct check must fail
        let mut c2 = c.clone();
        c2.b = Some(Poly::constant(1).add(&b));
        assert!(!c2.check(std::slice::from_ref(&s0)));
        // a fully-vanishing reading must also fail on this b
        let mut c3 = c2.clone();
        c3.ct_only = false;
        assert!(!c3.check(std::slice::from_ref(&s0)));
    }

    #[test]
    fn statement_check_all_and_validate() {
        let s0 = small_vec(2, 40);
        let phi = small_vec(2, 41);
        let honest = sprod(&phi, &s0);
        let st = PrincipalStatement::new(
            vec![VectorSpec::plain(2)],
            vec![DotCnst::with_b(
                vec![Term { idx: 0, off: 0, phi: phi.clone() }],
                honest,
            )],
            vec![],
            u64::MAX / 4,
        );
        st.validate().unwrap();
        st.check_all(&[s0]).unwrap();
        // out-of-range term caught
        let bad = PrincipalStatement::new(
            vec![VectorSpec::plain(2)],
            vec![DotCnst::homogeneous(vec![Term { idx: 0, off: 5, phi }])],
            vec![],
            100,
        );
        assert!(bad.validate().is_err());
    }
}
