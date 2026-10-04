//! The binary shadow of the scheme over `F162 = GF(2)[x]/(x^162 + x^81 + 1)`. Port of
//! `labinius` `eval.rs`.
//!
//! `F162` is exactly `R_162 mod 2` under the plain lift, so the whole fold has a shadow over
//! `F`: the statement `t = sum eq(p1,j) eq(p0,i) W[i + wdim j]`, the prover's message
//! `u = B W`, the claim check `u . eq(p1) = t`, and the binary check `B v = u^T c`.

use crate::binfield::F162;
use crate::challenge::ShortChallenge;

/// `eq(ps, b) = prod_k (ps_k if bit k of b is 1 else 1 + ps_k)`, all `2^ps.len()` of them.
pub fn eq_table(ps: &[F162]) -> Vec<F162> {
    let mut t = vec![F162::ONE];
    for &p in ps {
        let n = t.len();
        t.resize(2 * n, F162::ZERO);
        for b in 0..n {
            let x = t[b];
            t[b + n] = x * p;
            t[b] = x * (F162::ONE + p);
        }
    }
    t
}

/// `u_j = sum_i eq(p0, i) W[i + wdim j]`: one `wdim`-term dot product per column.
pub fn row_evaluate(witness: &[F162], p0: &[F162]) -> Vec<F162> {
    let wdim = 1usize << p0.len();
    assert_eq!(
        witness.len() % wdim,
        0,
        "witness is not whole columns of {wdim}"
    );
    let eq = eq_table(p0);
    (0..witness.len() / wdim)
        .map(|j| dot(&eq, &witness[j * wdim..(j + 1) * wdim]))
        .collect()
}

/// `u^T eq(p1)`, the claim a row evaluation implies.
pub fn claim(u: &[F162], p1: &[F162]) -> F162 {
    assert_eq!(u.len(), 1 << p1.len());
    dot(&eq_table(p1), u)
}

/// `u^T c`, the binary side of the fold: `sum_j u_j (c_j mod 2)`.
pub fn fold_binary(u: &[F162], challenges: &[ShortChallenge]) -> F162 {
    assert_eq!(u.len(), challenges.len(), "one challenge per column");
    let c: Vec<F162> = challenges.iter().map(|c| c.to_f162()).collect();
    dot(&c, u)
}

/// The binary check `B v == u^T c`: `sum_i eq(p0, i) (v_i mod 2) == u_folded`.
pub fn binary_check(p0: &[F162], v_mod2: &[F162], u_folded: F162) -> bool {
    v_mod2.len() == 1usize << p0.len() && dot(&eq_table(p0), v_mod2) == u_folded
}

/// Plain dot product over `F162`.
pub fn dot(a: &[F162], b: &[F162]) -> F162 {
    assert_eq!(a.len(), b.len());
    let mut acc = F162::ZERO;
    for i in 0..a.len() {
        acc.add_assign(a[i] * b[i]);
    }
    acc
}
