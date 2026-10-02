//! A generic `K`-valued multilinear sum-check engine over digit-cube
//! tables — the workhorse of CauchyFold's field front end (§5.3's
//! "sum-check and random aggregation") and the coefficient-and-range
//! check (§5.4's fingerprint polynomial).
//!
//! The statement is a function `g` evaluated on the current restriction of
//! a tuple of multilinear tables (all sized to the same cube); per round
//! the engine restricts the first remaining variable to the `D+1` probe
//! values `{0, 1, …, D}`, sums `g` over the tail, and interpolates the
//! degree-`D` round polynomial. Verifier cost is `O(D·m)`; prover cost
//! `O((D+1)·N)` overall (the tails telescope).

use crate::field_k::{K4, KPoly};

/// One round message: the `D+1` coefficients of the univariate
/// `A_i(X) = Σ_t c_t X^t`.
#[derive(Clone, Debug, PartialEq)]
pub struct KRoundMessage(pub Vec<K4>);

impl KRoundMessage {
    pub fn eval(&self, x: &K4) -> K4 {
        let mut acc = K4::ZERO;
        for c in self.0.iter().rev() {
            acc = acc.mul(x).add(c);
        }
        acc
    }

    /// `A_i(0) + A_i(1)` — for `A_i(X) = Σ_t c_t X^t` this is
    /// `2c₀ + Σ_{t≥1} c_t` (evaluate at `X ∈ {0,1}` and sum).
    pub fn cube_sum(&self) -> K4 {
        let mut acc = self.0[0];
        acc = acc.add(&self.0[0]);
        for c in self.0.iter().skip(1) {
            acc = acc.add(c);
        }
        acc
    }
}

/// Restrict a table (values on the remaining cube) to its first variable
/// taking the value `v`: `lo + v·(hi − lo)` per entry, halving the length.
pub fn restrict_table(table: &[K4], v: &K4) -> Vec<K4> {
    let half = table.len() / 2;
    let mut out = Vec::with_capacity(half);
    for t in 0..half {
        let lo = table[t];
        let hi = table[t + half];
        out.push(lo.add(&hi.sub(&lo).mul(v)));
    }
    out
}

/// The statement closure's signature.
pub type KStatement<'a> = Box<dyn Fn(&[Vec<K4>]) -> K4 + 'a>;

/// The sum-check statement: `g(restricted tables) = value`.
pub struct KSumcheck<'a> {
    pub num_vars: usize,
    pub degree: usize,
    /// The tables, all of length `2^{num_vars}`.
    pub tables: Vec<Vec<K4>>,
    /// The claimed cube-sum.
    pub claim: K4,
    /// The statement: evaluate on the CURRENT table tuple (the tail).
    pub g: KStatement<'a>,
}

impl<'a> KSumcheck<'a> {
    /// Prove: emit all round messages given the challenge sequence.
    pub fn prove(
        mut self,
        challenges: &[K4],
    ) -> Result<Vec<KRoundMessage>, String> {
        if challenges.len() != self.num_vars {
            return Err("challenge count".into());
        }
        let mut msgs = Vec::with_capacity(self.num_vars);
        for &r in challenges {
            let msg = self.round_message()?;
            msgs.push(msg);
            // Restrict all tables.
            self.tables = self
                .tables
                .iter()
                .map(|t| restrict_table(t, &r))
                .collect();
        }
        Ok(msgs)
    }

    /// The round message for the first remaining variable.
    fn round_message(&self) -> Result<KRoundMessage, String> {
        let d = self.degree;
        let probes: Vec<K4> = (0..=d as u64)
            .map(|v| K4::from_coeffs([v, 0, 0, 0]))
            .collect();
        let mut vals = Vec::with_capacity(d + 1);
        for v in &probes {
            let restricted: Vec<Vec<K4>> = self
                .tables
                .iter()
                .map(|t| restrict_table(t, v))
                .collect();
            // Sum g over the restricted tail cube: g is defined on tables,
            // so the "sum over the tail" is NOT Σ g(table) — the caller's g
            // must itself be the CUBE-SUM functional of its factors.
            // Contract: g(tables) = Σ_index g_pointwise(...). The node's
            // legs define g accordingly.
            vals.push((self.g)(&restricted));
        }
        // Interpolate the degree-d polynomial through (probe, value).
        #[cfg(test)]
        if self.num_vars > 0 && self.tables[0].len() > 4 {
            // Only for the outermost call (avoid recursion noise).
        }
        let pts: Vec<(K4, K4)> = probes.into_iter().zip(vals).collect();
        let poly = KPoly::interpolate(&pts);
        let mut coeffs: Vec<K4> = poly.coeffs.clone();
        coeffs.resize(d + 1, K4::ZERO);
        Ok(KRoundMessage(coeffs))
    }

    /// Verify a transcript: the recurrences. Returns the terminal
    /// evaluation `A_m(r_m)` and the point.
    pub fn verify(
        num_vars: usize,
        degree: usize,
        claim: &K4,
        msgs: &[KRoundMessage],
        challenges: &[K4],
    ) -> Result<(Vec<K4>, K4), String> {
        if msgs.len() != num_vars || challenges.len() != num_vars {
            return Err("transcript shape".into());
        }
        let mut current = *claim;
        for (i, (msg, r)) in msgs.iter().zip(challenges.iter()).enumerate() {
            if msg.0.len() != degree + 1 {
                return Err(format!("round {i} degree"));
            }
            if msg.cube_sum() != current {
                return Err(format!("round {i} recurrence"));
            }
            current = msg.eval(r);
        }
        Ok((challenges.to_vec(), current))
    }
}

/// The multilinear `z_α(x) = α^{int(x)}` = `Π_i (α^{2^{m−1−i}})^{x_i}` —
/// the §5.4 position-power factor (its cube values).
pub fn z_alpha_table(alpha: &K4, num_vars: usize) -> Vec<K4> {
    let n = 1usize << num_vars;
    let mut out = Vec::with_capacity(n);
    for idx in 0..n {
        let mut acc = K4::ONE;
        for i in 0..num_vars {
            if (idx >> (num_vars - 1 - i)) & 1 == 1 {
                acc = acc.mul(&alpha.pow(1u64 << (num_vars - 1 - i)));
            }
        }
        out.push(acc);
    }
    out
}

/// The range polynomial `R_Δ(X) = Π_{a=−δ}^{δ−1} (X − a)` over the signed
/// digit set `{−δ, …, δ−1}` — the paper's `R15` with `{−7..7}`; this
/// crate's level-2 digits are signed 4-bit `{−8..7}`, so `R16`.
pub fn range_poly(delta: i64) -> KPoly {
    let mut p = KPoly::constant(K4::ONE);
    for a in -delta..delta {
        let c = K4::from_coeffs([((-a).rem_euclid(crate::field_k::Q48 as i64)) as u64, 0, 0, 0]);
        p = p.mul(&KPoly::from_coeffs(vec![c, K4::ONE]));
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_k::Fq48;

    fn k4(v: u64) -> K4 {
        K4::from_coeffs([v % crate::field_k::Q48, v * 7 % crate::field_k::Q48, v * 3, v])
    }

    #[test]
    fn restrict_table_linearity() {
        let table: Vec<K4> = (0..8).map(|i| k4(i * 991)).collect();
        let v = k4(5);
        let r = restrict_table(&table, &v);
        assert_eq!(r.len(), 4);
        // lo + v(hi − lo).
        assert_eq!(r[0], table[0].add(&table[4].sub(&table[0]).mul(&v)));
        // Restricting to 0/1 returns the halves.
        let r0 = restrict_table(&table, &K4::ZERO);
        assert_eq!(r0, table[..4].to_vec());
        let r1 = restrict_table(&table, &K4::ONE);
        assert_eq!(r1, table[4..].to_vec());
    }

    #[test]
    fn sumcheck_inner_product_roundtrip() {
        // g(tables) = Σ_x A(x)·B(x) — the inner-product leg (degree 2).
        let m = 4;
        let a: Vec<K4> = (0..1usize << m).map(|i| k4(i as u64 * 31 + 1)).collect();
        let b: Vec<K4> = (0..1usize << m).map(|i| k4(i as u64 * 17 + 5)).collect();
        let claim: K4 = a.iter().zip(b.iter()).map(|(x, y)| x.mul(y)).fold(K4::ZERO, |acc, t| acc.add(&t));
        let g = move |tables: &[Vec<K4>]| -> K4 {
            let mut acc = K4::ZERO;
            for (x, y) in tables[0].iter().zip(tables[1].iter()) {
                acc = acc.add(&x.mul(y));
            }
            acc
        };
        let challenges: Vec<K4> = (0..m).map(|i| k4(100 + i as u64 * 13)).collect();
        let sc = KSumcheck {
            num_vars: m,
            degree: 2,
            tables: vec![a.clone(), b.clone()],
            claim,
            g: Box::new(g),
        };
        let msgs = sc.prove(&challenges).expect("prove");
        let (point, terminal) =
            KSumcheck::verify(m, 2, &claim, &msgs, &challenges).expect("verify");
        // The terminal: A(r)·B(r) with the restricted tables.
        let ra = challenges.iter().fold(a.clone(), |t, r| restrict_table(&t, r));
        let rb = challenges.iter().fold(b.clone(), |t, r| restrict_table(&t, r));
        assert_eq!(point, challenges);
        assert_eq!(terminal, ra[0].mul(&rb[0]));
    }

    #[test]
    fn sumcheck_degree17_fingerprint_shape() {
        // The fingerprint leg shape: Σ_x h(x)·W(x) + c·z(x)·R16(W(x)) —
        // individual degree 17.
        let m = 3;
        let n = 1usize << m;
        let w: Vec<K4> = (0..n).map(|i| k4(i as u64 * 5 + 3)).collect();
        let h: Vec<K4> = (0..n).map(|i| k4(i as u64 * 11 + 7)).collect();
        let alpha = k4(3);
        let z = z_alpha_table(&alpha, m);
        let r16 = range_poly(8);
        let c = k4(19);
        let r16c = r16.clone();
        let g = move |tables: &[Vec<K4>]| -> K4 {
            let mut acc = K4::ZERO;
            for i in 0..tables[0].len() {
                let term = tables[1][i].mul(&tables[0][i]);
                let r = r16c.eval(&tables[0][i]).mul(&tables[2][i]);
                acc = acc.add(&term.add(&r.scale(&c)));
            }
            acc
        };
        let claim = g(&[w.clone(), h.clone(), z.clone()]);
        let challenges: Vec<K4> = (0..m).map(|i| k4(50 + i as u64 * 3)).collect();
        let sc = KSumcheck {
            num_vars: m,
            degree: 17,
            tables: vec![w.clone(), h.clone(), z.clone()],
            claim,
            g: Box::new(g),
        };
        let msgs = sc.prove(&challenges).expect("prove");
        let (_, terminal) = KSumcheck::verify(m, 17, &claim, &msgs, &challenges).expect("verify");
        let rw = challenges.iter().fold(w.clone(), |t, r| restrict_table(&t, r));
        let rh = challenges.iter().fold(h.clone(), |t, r| restrict_table(&t, r));
        let rz = challenges.iter().fold(z.clone(), |t, r| restrict_table(&t, r));
        let expect = rh[0].mul(&rw[0]).add(&r16.eval(&rw[0]).mul(&rz[0]).scale(&c));
        assert_eq!(terminal, expect);
    }

    #[test]
    fn range_poly_roots() {
        let r = range_poly(8);
        for a in -8i64..8 {
            let x = K4::from_coeffs([a.rem_euclid(crate::field_k::Q48 as i64) as u64, 0, 0, 0]);
            assert!(r.eval(&x).is_zero(), "R16({a}) = 0");
        }
        // Outside the set: nonzero.
        let x = K4::from_coeffs([Fq48(9).0, 0, 0, 0]);
        assert!(!r.eval(&x).is_zero());
        let x = K4::from_coeffs([Fq48(Q48_OUTSIDE).0, 0, 0, 0]);
        assert!(!r.eval(&x).is_zero());
    }

    const Q48_OUTSIDE: u64 = 100;

    #[test]
    fn z_alpha_is_multilinear_exponential() {
        let alpha = k4(6);
        let m = 4;
        let z = z_alpha_table(&alpha, m);
        assert_eq!(z[0], K4::ONE);
        assert_eq!(z[5], alpha.pow(5));
        assert_eq!(z[9], alpha.pow(9));
        // Restriction consistency: z on the cube factors per-bit.
        assert_eq!(z[3], alpha.pow(3));
    }
}
