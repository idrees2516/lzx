//! Section 4 of ePrint 2026/471: the attacks that break naive field
//! lookups when restated over rings. Each attack is implemented as an
//! executable demonstration over a toy composite ring, pinning **why**
//! Ring-Plookup's indexed tags and Ring-LogUp's tagged denominators are
//! necessary.
//!
//! 1. **CRT-component swapping** (`Z15 = Z3 × Z5`): an invalid lookup
//!    `a' = {7, 11}` into `b = {1, 2, 3}` — CRT `(1,2), (2,1)` — is a
//!    *valid* lookup inside each CRT component, but with **swapped
//!    index maps**. Every grand-product relation that does not pin the
//!    per-component index order therefore still holds over the full
//!    ring. Demonstrated on the original Plookup `F/G` relation
//!    (Eqs. 1–2) and on the Spartan/Lasso grand-product
//!    `WSt·WSh = RS·S` (Eq. 7) that the first version of
//!    [CCCFGSV25] relied on.
//! 2. **Zero-divisor attacks on LogUp**: over `Z6`, the cleared
//!    denominator sum rewritten over the whole ring
//!    `f(x) = Σ_v (m_a[v]−m_b[v])·∏_{z≠v}(x−z)` (Eq. 9) is
//!    **identically zero** — every `∏_{z≠v}(x−z)` contains a
//!    zero-divisor pair — so an invalid lookup passes. Over `Z12`, the
//!    LogUp relation polynomial `P(x)` itself (Eq. 8) vanishes for
//!    every `x` because `∏_j (x−b_j)` always contains a
//!    zero-divisor-triple.
//!
//! The ring-safe protocols in `ring_plookup` / `ring_logup` reject the
//! analogous forgeries (their own test suites pin that).


// (Kernel loops use explicit indices by convention.)
#![allow(clippy::needless_range_loop)]
/// Original Plookup grand products (Eqs. 1–2 of [GW20], as restated
/// over a ring): `F(β,γ)` and `G(β,γ)` with the `(1+β)` factor.
pub fn plookup_f(n: u64, a: &[u64], b: &[u64], beta: u64, gamma: u64) -> u64 {
    let mut f = pow_mod(1 + beta, a.len() as u64, n);
    for &ai in a {
        f = f * (gamma + ai) % n;
    }
    for j in 0..b.len().saturating_sub(1) {
        let t = (gamma * (1 + beta) + b[j] + beta * b[j + 1]) % n;
        f = f * t % n;
    }
    f % n
}

pub fn plookup_g(n: u64, s: &[u64], beta: u64, gamma: u64) -> u64 {
    let mut g = 1u64;
    for j in 0..s.len().saturating_sub(1) {
        let t = (gamma * (1 + beta) + s[j] + beta * s[j + 1]) % n;
        g = g * t % n;
    }
    g % n
}

fn pow_mod(mut base: u64, mut exp: u64, n: u64) -> u64 {
    let mut r = 1u64;
    base %= n;
    while exp > 0 {
        if exp & 1 == 1 {
            r = r * base % n;
        }
        base = base * base % n;
        exp >>= 1;
    }
    r
}

/// The Spartan/Lasso offline-memory grand product (Eq. 7):
/// `WSt·WSh = RS·S` where
/// `WSt = ∏_j (b_j − x2)`, `WSh = ∏_i (a_i + x1(read_ts_i+1) − x2)`,
/// `RS = ∏_i (a_i + x1·read_ts_i − x2)`,
/// `S = ∏_j (b_j + x1·final_cts_j − x2)`.
pub fn lasso_grand_product_holds(
    n: u64,
    a: &[u64],
    b: &[u64],
    read_ts: &[u64],
    final_cts: &[u64],
    x1: u64,
    x2: u64,
) -> bool {
    let mut wst = 1u64;
    for &bj in b {
        wst = wst * ((bj + n - x2 % n) % n) % n;
    }
    let mut wsh = 1u64;
    for (i, &ai) in a.iter().enumerate() {
        let t = (ai + x1 * (read_ts[i] + 1) % n + n - x2 % n) % n;
        wsh = wsh * t % n;
    }
    let mut rs = 1u64;
    for (i, &ai) in a.iter().enumerate() {
        let t = (ai + x1 * read_ts[i] % n + n - x2 % n) % n;
        rs = rs * t % n;
    }
    let mut s = 1u64;
    for (j, &bj) in b.iter().enumerate() {
        let t = (bj + x1 * final_cts[j] % n + n - x2 % n) % n;
        s = s * t % n;
    }
    let lhs = wst * wsh % n;
    let rhs = rs * s % n;
    lhs == rhs
}

/// The cleared-denominator LogUp sum over the whole ring (Eq. 9):
/// `f(x) = Σ_{v ∈ ring} (m_a[v] − m_b[v]) · ∏_{z ≠ v} (x − z)`.
pub fn logup_field_sum(n: u64, a: &[u64], b: &[u64], m: &[u64], x: u64) -> u64 {
    let mut ma = vec![0u64; n as usize];
    let mut mb = vec![0u64; n as usize];
    for &v in a {
        ma[(v % n) as usize] += 1;
    }
    for (j, &bj) in b.iter().enumerate() {
        let cnt = m.get(j).copied().unwrap_or(0);
        mb[(bj % n) as usize] += cnt;
    }
    let mut f = 0u64;
    for v in 0..n {
        let diff = (ma[v as usize] + n - mb[v as usize] % n) % n;
        let mut prod = 1u64;
        for z in 0..n {
            if z != v {
                prod = prod * ((x + n - z) % n) % n;
            }
        }
        f = (f + diff * prod) % n;
    }
    f % n
}

/// The full LogUp relation polynomial (Eq. 8):
/// `P(x) = (∏_j (x−b_j) · ∏_i Σ-ish)` — precisely:
/// `P = ∏_j(x−b_j)·∏_{i}(∏_{ℓ≠i}(x−a_ℓ)) − ∏_i(x−a_i)·Σ_j m_j·∏_{k≠j}(x−b_k)`.
pub fn logup_relation_poly(n: u64, a: &[u64], b: &[u64], m: &[u64], x: u64) -> u64 {
    // ∏_j (x − b_j)
    let mut prod_b = 1u64;
    for &bj in b {
        prod_b = prod_b * ((x + n - bj % n) % n) % n;
    }
    // ∏_i ∏_{ℓ≠i} (x − a_ℓ)  = ∏_ℓ (x−a_ℓ)^{M−1}
    let mut star = 1u64;
    for &ai in a {
        let factor = (x + n - ai % n) % n;
        star = star * pow_mod(factor, (a.len().saturating_sub(1)) as u64, n) % n;
    }
    let first = prod_b * star % n;
    // ∏_i (x − a_i)
    let mut prod_a = 1u64;
    for &ai in a {
        prod_a = prod_a * ((x + n - ai % n) % n) % n;
    }
    let mut sum = 0u64;
    for (j, _) in b.iter().enumerate() {
        let mj = m.get(j).copied().unwrap_or(0);
        let mut prod_skip = 1u64;
        for (k, &bk) in b.iter().enumerate() {
            if k != j {
                prod_skip = prod_skip * ((x + n - bk % n) % n) % n;
            }
        }
        sum = (sum + mj % n * prod_skip) % n;
    }
    let second = prod_a * sum % n;
    (first + n - second) % n
}

pub fn is_subset(a: &[u64], b: &[u64]) -> bool {
    let mut pool = b.to_vec();
    for &v in a {
        match pool.iter().position(|&x| x == v) {
            Some(i) => {
                pool.remove(i);
            }
            None => return false,
        }
    }
    true
}

/// CRT components of `Z15 = Z3 × Z5`.
pub fn crt15(v: u64) -> (u64, u64) {
    (v % 3, v % 5)
}

// --------------------------------------------------------------------------
// Attack 1: CRT swapping on the original Plookup relation over Z15.
// --------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// The paper's attack (Section 4, "Plookup"): the invalid lookup
    /// `a' = {7, 11}` over `Z15` (CRT `(1,2)` and `(2,1)`) satisfies the
    /// original Plookup `F ≡ G` relation at *every* random `(β,γ)`,
    /// because the relation holds per CRT component — with the two
    /// components using swapped index maps.
    #[test]
    fn attack_crt_swap_plookup_over_z15() {
        let n = 15u64;
        let b: Vec<u64> = vec![1, 2, 3];
        let a_bad: Vec<u64> = vec![7, 11];
        // the malicious merge vector w = {1,1,2,2,3}
        let w: Vec<u64> = vec![1, 1, 2, 2, 3];
        assert!(!is_subset(&a_bad, &b), "7 and 11 are not members of the table over Z15");
        // per-component the lookup IS valid — with swapped indices:
        let (a1p, a1m) = crt15(a_bad[0]);
        let (a2p, a2m) = crt15(a_bad[1]);
        assert_eq!((a1p, a2p), (1, 2), "Z3 components: 7→1, 11→2 (b has 1,2)");
        assert_eq!((a1m, a2m), (2, 1), "Z5 components: 7→2, 11→1 — SWAPPED order");
        // The Plookup relation F(β,γ) = G(β,γ) holds at random points.
        for trial in 0..40 {
            let beta = (7 * trial + 3) % n;
            let gamma = (11 * trial + 5) % n;
            let f = plookup_f(n, &a_bad, &b, beta, gamma);
            let g = plookup_g(n, &w, beta, gamma);
            assert_eq!(f, g, "attack holds: F==G at beta={beta}, gamma={gamma}");
        }
        // sanity: a genuinely valid lookup also passes (no false negative)
        let a_ok: Vec<u64> = vec![1, 2];
        let w_ok: Vec<u64> = vec![1, 1, 2, 2, 3];
        let f = plookup_f(n, &a_ok, &b, 2, 4);
        let g = plookup_g(n, &w_ok, 2, 4);
        assert_eq!(f, g);
    }

    /// The attack on the Spartan/Lasso grand product (Eq. 7) that the
    /// first version of [CCCFGSV25] used: same CRT swap, same
    /// read_ts/final_cts, and `WSt·WSh = RS·S` still holds over Z15.
    #[test]
    fn attack_crt_swap_lasso_grand_product_over_z15() {
        let n = 15u64;
        let b: Vec<u64> = vec![1, 2, 3];
        let a_bad: Vec<u64> = vec![7, 11];
        let read_ts: Vec<u64> = vec![0, 0];
        let final_cts: Vec<u64> = vec![1, 1, 0];
        assert!(!is_subset(&a_bad, &b));
        for trial in 0..40 {
            let x1 = (3 * trial + 1) % n;
            let x2 = (5 * trial + 2) % n;
            assert!(
                lasso_grand_product_holds(n, &a_bad, &b, &read_ts, &final_cts, x1, x2),
                "Eq.(7) holds for the invalid lookup at x1={x1}, x2={x2}"
            );
        }
        // Over Z15 the attack is a per-CRT-component polynomial identity:
        // it holds at EVERY (x1, x2) in the grid.
        let mut all_hold = true;
        for x1 in 0..n {
            for x2 in 0..n {
                if !lasso_grand_product_holds(n, &a_bad, &b, &read_ts, &final_cts, x1, x2) {
                    all_hold = false;
                }
            }
        }
        assert!(all_hold, "the CRT-swap attack is a polynomial identity over Z15");
        // Over a prime field (Z17) the same invalid lookup is NOT a
        // polynomial identity: it holds only at isolated points (the
        // difference polynomial's roots), a small fraction of the grid.
        let nf = 17u64;
        let bf: Vec<u64> = vec![1, 2, 3];
        let af: Vec<u64> = vec![7, 11];
        let mut holds = 0usize;
        let mut total = 0usize;
        for x1 in 0..nf {
            for x2 in 0..nf {
                total += 1;
                if lasso_grand_product_holds(nf, &af, &bf, &read_ts, &final_cts, x1, x2) {
                    holds += 1;
                }
            }
        }
        // ~48/289 accidental roots for these toy parameters — the
        // contrast with the 225/225 polynomial identity over Z15 is the
        // point: over a field the attack survives only by luck, over
        // the CRT-split ring it is an identity.
        assert!(
            holds * 4 < total,
            "over Z17 the identity fails generically (holds at {holds}/{total} points only)"
        );
    }

    /// The zero-divisor attack on LogUp's *proof technique* (Eq. 9) over
    /// Z6: `f(x)` is identically zero for an invalid lookup.
    #[test]
    fn attack_logup_zero_divisor_z6() {
        let n = 6u64;
        let b: Vec<u64> = vec![1, 2, 3, 4];
        let a_bad: Vec<u64> = vec![5];
        let m: Vec<u64> = vec![0, 0, 0, 1]; // any multiplicity vector
        assert!(!is_subset(&a_bad, &b));
        for x in 0..n {
            let f = logup_field_sum(n, &a_bad, &b, &m, x);
            assert_eq!(f, 0, "f({x}) = 0 over Z6 despite the invalid lookup");
        }
        // Over a field the same technique catches the invalid lookup:
        // f(x) is NOT identically zero over Z7.
        let nf = 7u64;
        let bf: Vec<u64> = vec![1, 2, 3, 4];
        let af: Vec<u64> = vec![5];
        let mut nonzero = false;
        for x in 0..nf {
            if logup_field_sum(nf, &af, &bf, &m, x) != 0 {
                nonzero = true;
            }
        }
        assert!(nonzero, "over the field Z7 the zero-test exposes the forgery");
    }

    /// The zero-divisor attack on the LogUp *relation itself* (Eq. 8)
    /// over Z12: `P(x) = 0` for every x, for ANY invalid lookup.
    #[test]
    fn attack_logup_relation_z12() {
        let n = 12u64;
        let b: Vec<u64> = vec![1, 2, 3, 4, 5, 6];
        let m: Vec<u64> = vec![1; 6];
        for &bad in &[7u64, 8, 9, 10, 11] {
            let a_bad: Vec<u64> = vec![bad];
            assert!(!is_subset(&a_bad, &b));
            for x in 0..n {
                let p = logup_relation_poly(n, &a_bad, &b, &m, x);
                assert_eq!(p, 0, "P(x)=0 over Z12 for invalid lookup value {bad} at x={x}");
            }
        }
    }

    /// The honest reference: over Z15 a *valid* lookup satisfies all the
    /// relations (completeness of the attack harness itself).
    #[test]
    fn honest_lookups_pass_the_field_relations() {
        let n = 15u64;
        let b: Vec<u64> = vec![1, 2, 3];
        let a: Vec<u64> = vec![2, 1];
        let w: Vec<u64> = vec![1, 1, 2, 2, 3];
        assert!(is_subset(&a, &b));
        let f = plookup_f(n, &a, &b, 3, 7);
        let g = plookup_g(n, &w, 3, 7);
        assert_eq!(f, g);
        // and over Z6 the zero-divisor function is zero for honest too
        let n6 = 6u64;
        let b6: Vec<u64> = vec![1, 2, 3, 4];
        let a6: Vec<u64> = vec![2];
        let m6: Vec<u64> = vec![0, 1, 0, 0];
        assert!(is_subset(&a6, &b6));
        for x in 0..n6 {
            assert_eq!(logup_field_sum(n6, &a6, &b6, &m6, x), 0);
        }
    }
}
