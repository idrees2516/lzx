//! The executable knowledge-soundness extraction (the paper's Lemma 5.2,
//! the 4-ary transcript tree + the two-`α` subtraction), over a rewindable
//! prover abstraction — with the lattice-specific outcome algebra.
//!
//! ## The harness
//!
//! `ReduceProver` models a deterministic-in-the-challenge-prefix prover
//! (the honest adapter computes from the witness; cheating adapters model
//! adversaries). The extractor drives it as the verifier:
//!
//! 1. Two runs with distinct `α¹ ≠ α²` (the paper's two subtrees).
//! 2. Within each run, the challenge grid: two seeded field values per
//!    variable, all `2^m` combinations — the paper's tree branches on field
//!    challenges (never on cube points: those are measure zero and a
//!    cheater could answer honestly exactly there).
//! 3. At every grid leaf, the prover's terminal scalar `a_p`; the
//!    multilinear witness is recovered by tensor interpolation through the
//!    grid (`Ŵ(b) = Σ_p a_p Π_i L_{p_i}(b_i)`).
//! 4. Outcome checks: `com(Ŵ) ≟ cm`, `Σ_b T(b)Ŵ(b) ≟ v`, the digit norm
//!    budget, and — the paper's terminal step — the two-`α` subtraction:
//!    if `a¹ ≠ a²`,
//!    `K = (a¹ − a², (v − t̂¹)α¹ − (v − t̂²)α²)` is a kernel relation on
//!    `[G | P]`.
//!
//! ## The lattice outcome algebra (the honest deviation)
//!
//! Over Pedersen groups, DLA forbids *any* nonzero
//! `f` with `⟨f, G⟩ = 0`, so Lemma 5.2's collision outcome is a direct
//! contradiction. Over Ajtai modules, kernel relations with **arbitrary
//! field coefficients exist trivially** (the matrix has huge mod-`q`
//! kernel); Module-SIS only rules out **short** kernels. Consequently:
//!
//! * for **consistent provers** (honest, or double-opening cheaters whose
//!   two witnesses are both short), the recovered `a¹, a²` are short and
//!   their difference is a *short* kernel — a genuine MSIS solution, the
//!   security-relevant outcome (demonstrated by the double-opening test on
//!   a crafted duplicated-column SRS);
//! * for **inconsistent provers**, the harness reports the failed
//!   consistency checks and any mod-`q` kernel it can form — which is
//!   field-valid but not necessarily short. The adversary-model closure
//!   (forcing shortness through response-norm checks) is the discipline of
//!   `lattice-commitment::linear_proof`, `lattice-labrador`, and
//!   `lattice-cauchyfold` — see the deviation ledger.

use crate::module::{Fq, LayeredCube, ModulePoint, Srs};
use crate::sumcheck::RoundMessage;

/// A rewindable reduce prover: deterministic messages given the challenge
/// prefix. `prefix` holds `r₁ … r_{i−1}` (the point before round `i`).
pub trait ReduceProver {
    /// The round-`i` message (1-indexed round) under `(α, prefix)`.
    fn round_message(&self, alpha: Fq, prefix: &[Fq]) -> RoundMessage;

    /// The terminal scalar `a = Ŵ(r)` claimed at the full point.
    fn terminal(&self, alpha: Fq, point: &[Fq]) -> Fq;
}

/// The honest prover adapter: computes every message from the true witness
/// by restricting the tables to the given prefix.
pub struct HonestReduceProver<'a> {
    pub srs: &'a Srs,
    pub cube: &'a LayeredCube,
    pub w: &'a [Fq],
    pub u: &'a [Fq],
}

impl<'a> HonestReduceProver<'a> {
    fn restricted_tables(
        &self,
        alpha: &Fq,
        prefix: &[Fq],
    ) -> crate::sumcheck::SummandTables {
        let p_prime = self.srs.value_column().scale(alpha);
        let t_table = self.cube.t_table(self.u);
        let mut tables =
            crate::sumcheck::SummandTables::for_reduce(self.w, self.srs, &t_table, p_prime);
        for r in prefix {
            tables.restrict(r);
        }
        tables
    }
}

impl<'a> ReduceProver for HonestReduceProver<'a> {
    fn round_message(&self, alpha: Fq, prefix: &[Fq]) -> RoundMessage {
        let tables = self.restricted_tables(&alpha, prefix);
        tables.round_message()
    }

    fn terminal(&self, alpha: Fq, point: &[Fq]) -> Fq {
        let tables = self.restricted_tables(&alpha, point);
        let (a, _) = tables.terminal_values();
        a
    }
}

/// The extraction outcome.
#[derive(Clone, Debug)]
pub struct ExtractionOutcome {
    /// The recovered cube values of the multilinear witness (per `α` run).
    pub witnesses: [Vec<Fq>; 2],
    /// `com(Ŵ^j) == cm` per run.
    pub commitment_consistent: [bool; 2],
    /// `Σ_b T(b) Ŵ^j(b) == v` per run.
    pub evaluation_consistent: [bool; 2],
    /// Both recovered witnesses pass the digit norm budget.
    pub short: [bool; 2],
    /// The two-`α` kernel relation on `[G | P]`, when the witnesses differ
    /// (nonzero coefficient vector) and the relation verifies.
    pub kernel: Option<(Vec<Fq>, Fq)>,
    /// Whether the kernel coefficients are short (an MSIS solution).
    pub kernel_short: bool,
}

/// Run the extraction: the two-`α` grid recovery over a rewindable prover.
pub fn extract(
    prover: &dyn ReduceProver,
    srs: &Srs,
    cube: &LayeredCube,
    cm: &ModulePoint,
    u: &[Fq],
    v: &Fq,
    seed: &[u8],
) -> ExtractionOutcome {
    let m = cube.num_vars();
    // Two distinct alphas and the per-variable grid values, all seeded and
    // distinct from 0/1 (field points, not cube points).
    let mut salt = Vec::new();
    salt.extend_from_slice(b"extract");
    salt.extend_from_slice(seed);
    let [alpha1, alpha2] = derive_alphas(&salt);
    let grid = derive_grid(&salt, m);
    let (g0, g1) = &grid;

    let t_table = cube.t_table(u);
    let mut witnesses = [Vec::new(), Vec::new()];
    let mut commit_ok = [false, false];
    let mut eval_ok = [false, false];
    let mut short_ok = [false, bool::default()];
    for (j, alpha) in [alpha1, alpha2].iter().enumerate() {
        // Walk the grid: leaf value per grid point.
        let mut leaf_vals = vec![Fq::ZERO; 1 << m];
        for idx in 0..(1usize << m) {
            let point: Vec<Fq> = (0..m)
                .map(|i| {
                    if (idx >> (m - 1 - i)) & 1 == 0 {
                        g0[i]
                    } else {
                        g1[i]
                    }
                })
                .collect();
            leaf_vals[idx] = prover.terminal(*alpha, &point);
        }
        // Tensor interpolation to the cube: W(b) = Σ_p a_p Π_i L_{p_i}(b_i).
        let w = interpolate_grid_to_cube(&grid, &leaf_vals, m);
        witnesses[j] = w.clone();
        commit_ok[j] = srs.commit_scalars(&w) == *cm;
        let mut t_sum = Fq::ZERO;
        for (wi, ti) in w.iter().zip(t_table.iter()) {
            t_sum = t_sum.add(&ti.mul(wi));
        }
        eval_ok[j] = t_sum == *v;
        short_ok[j] = srs.is_short(&w);
    }

    // The two-α subtraction: if the recovered witnesses differ, form the
    // [G | P] kernel and check it.
    let a1 = &witnesses[0];
    let a2 = &witnesses[1];
    let differ = a1 != a2;
    let mut kernel = None;
    let mut kernel_short = false;
    if differ {
        let mut k_g = vec![Fq::ZERO; a1.len()];
        for (k, (x, y)) in k_g.iter_mut().zip(a1.iter().zip(a2.iter())) {
            *k = x.sub(y);
        }
        // t̂^j = Σ_b T(b) a^j(b) — the P-slot coefficient:
        // (v − t̂¹)α¹ − (v − t̂²)α² ... sign-fixed by verification below.
        let t_hat = |w: &[Fq]| -> Fq {
            let mut s = Fq::ZERO;
            for (wi, ti) in w.iter().zip(t_table.iter()) {
                s = s.add(&ti.mul(wi));
            }
            s
        };
        let t1 = t_hat(a1);
        let t2 = t_hat(a2);
        let kp = v.sub(&t1).mul(&alpha1).sub(&v.sub(&t2).mul(&alpha2));
        if srs.is_extended_kernel(&k_g, &kp) {
            let all_g_short = srs.is_short(&k_g);
            let kp_short = kp.0 <= srs.digit_norm_bound()
                || (crate::module::Q_50 - kp.0) <= srs.digit_norm_bound();
            kernel_short = all_g_short && kp_short;
            kernel = Some((k_g, kp));
        } else {
            // Try the negated P coefficient (sign convention).
            let kp_neg = kp.neg();
            if srs.is_extended_kernel(&k_g, &kp_neg) {
                let all_g_short = srs.is_short(&k_g);
                let kp_short = kp_neg.0 <= srs.digit_norm_bound()
                    || (crate::module::Q_50 - kp_neg.0) <= srs.digit_norm_bound();
                kernel_short = all_g_short && kp_short;
                kernel = Some((k_g, kp_neg));
            }
        }
    }
    ExtractionOutcome {
        witnesses,
        commitment_consistent: commit_ok,
        evaluation_consistent: eval_ok,
        short: short_ok,
        kernel,
        kernel_short,
    }
}

/// Two distinct seeded alphas of opposite parity (the parity split matters
/// for parity-keyed cheaters — the extractor is free to choose its
/// challenges, so we pin the two runs to distinguishable keys).
fn derive_alphas(salt: &[u8]) -> [Fq; 2] {
    let b = lattice_core::transcript::Transcript::xof(b"alpha-pair", salt, 16);
    let mk = |off: usize| -> Fq {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&b[off..off + 8]);
        Fq::from_u64(u64::from_le_bytes(arr) & ((1 << 49) - 1)).add(&Fq::from_u64(11))
    };
    let a1 = Fq(mk(0).0 & !1u64); // even
    let mut a2 = Fq(mk(8).0 | 1u64); // odd
    if a1 == a2 {
        a2 = a2.add(&Fq::from_u64(2));
    }
    [a1, a2]
}

/// Two seeded grid values per variable (distinct, nonzero). Returns
/// `(s⁰, s¹)` with `s^bit[i]` the value of variable `i` at grid bit `bit`.
fn derive_grid(salt: &[u8], m: usize) -> (Vec<Fq>, Vec<Fq>) {
    let b = lattice_core::transcript::Transcript::xof(b"grid-pair", salt, 8 * 2 * m + 8);
    let mk = |off: usize| -> Fq {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&b[off..off + 8]);
        Fq::from_u64(u64::from_le_bytes(arr) & ((1 << 49) - 1)).add(&Fq::from_u64(3))
    };
    let mut s0 = Vec::with_capacity(m);
    let mut s1 = Vec::with_capacity(m);
    for i in 0..m {
        let v0 = mk(i * 16);
        let mut v1 = mk(i * 16 + 8);
        if v0 == v1 {
            v1 = v1.add(&Fq::ONE);
        }
        s0.push(v0);
        s1.push(v1);
    }
    (s0, s1)
}

/// Tensor interpolation: cube values of the multilinear through the grid
/// leaf values. `grid.0[i]` / `grid.1[i]` are the two grid values of
/// variable `i`.
fn interpolate_grid_to_cube(grid: &(Vec<Fq>, Vec<Fq>), leaves: &[Fq], m: usize) -> Vec<Fq> {
    // For each cube point b: W(b) = Σ_p leaves[p] Π_i L_{p_i}(b_i) with
    // L_0(x) = (x − s¹)/(s⁰ − s¹), L_1(x) = (x − s⁰)/(s¹ − s⁰).
    let (s0, s1) = grid;
    let mut out = vec![Fq::ZERO; 1 << m];
    for b in 0..(1usize << m) {
        let mut acc = Fq::ZERO;
        for p in 0..(1usize << m) {
            let mut coef = leaves[p];
            for i in 0..m {
                let (pv, other) = if (p >> (m - 1 - i)) & 1 == 0 {
                    (&s0[i], &s1[i])
                } else {
                    (&s1[i], &s0[i])
                };
                let bv = Fq::from_u64(((b >> (m - 1 - i)) & 1) as u64);
                // L(bv) = (bv − other) / (pv − other)
                let num = bv.sub(other);
                let den = pv.sub(other);
                coef = coef.mul(&num.mul(&den.inv().unwrap_or(Fq::ONE)));
            }
            acc = acc.add(&coef);
        }
        out[b] = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{eq_eval_index, Q_50};
    use lattice_core::transcript::Transcript;

    fn setup(k: usize, seed: &[u8]) -> (Srs, LayeredCube, Vec<Fq>, Vec<Fq>, Vec<Fq>, ModulePoint) {
        let cube = LayeredCube::new(k, 4);
        let srs = Srs::from_seed(1, 16, cube.size(), seed);
        let n = 1usize << k;
        let f: Vec<Fq> = (0..n)
            .map(|i| Fq::from_u64((i as u64).wrapping_mul(6364136223846793005).wrapping_add(7) & 0xFFFF_FFFF))
            .collect();
        let w = cube.digit_layers(&f);
        let u: Vec<Fq> = (0..k).map(|i| Fq::from_u64(100 + i as u64 * 31)).collect();
        let cm = srs.commit_scalars(&w);
        (srs, cube, f, w, u, cm)
    }

    fn eval_v(cube: &LayeredCube, w: &[Fq], u: &[Fq]) -> Fq {
        let t = cube.t_table(u);
        let mut acc = Fq::ZERO;
        for (wi, ti) in w.iter().zip(t.iter()) {
            acc = acc.add(&ti.mul(wi));
        }
        acc
    }

    #[test]
    fn honest_extraction_recovers_short_witness() {
        let (srs, cube, _f, w, u, cm) = setup(4, b"extract-honest");
        let v = eval_v(&cube, &w, &u);
        let prover = HonestReduceProver {
            srs: &srs,
            cube: &cube,
            w: &w,
            u: &u,
        };
        let out = extract(&prover, &srs, &cube, &cm, &u, &v, b"s1");
        for j in 0..2 {
            assert!(out.commitment_consistent[j], "com(w^j) == cm");
            assert!(out.evaluation_consistent[j], "eval claim holds");
            assert!(out.short[j], "recovered witness is short");
            assert_eq!(out.witnesses[j], w, "recovered the true digit witness");
        }
        assert!(out.kernel.is_none(), "consistent prover: no kernel");
    }

    #[test]
    fn wrong_value_cheater_fails_consistency() {
        let (srs, cube, _f, w, u, cm) = setup(3, b"extract-badval");
        // A cheater proving a WRONG claimed value v' != v: the honest
        // prover machinery runs against the true witness, so the extraction
        // recovers the true w — whose T-weighted sum is v, not v'.
        let v_true = eval_v(&cube, &w, &u);
        let v_bad = v_true.add(&Fq::from_u64(5));
        let prover = HonestReduceProver {
            srs: &srs,
            cube: &cube,
            w: &w,
            u: &u,
        };
        let out = extract(&prover, &srs, &cube, &cm, &u, &v_bad, b"s2");
        // The recovered witness is short and commitment-consistent, but the
        // claimed evaluation fails — exactly the knowledge-soundness
        // verdict: this prover does NOT know a witness for (cm, u, v').
        assert!(out.short[0] && out.commitment_consistent[0]);
        assert!(!out.evaluation_consistent[0]);
        assert!(!out.evaluation_consistent[1]);
    }

    /// A double-opening cheater: alternates between two short witnesses of
    /// the same commitment (only possible on a crafted SRS with a known
    /// short kernel — the duplicated-column construction).
    struct DoubleOpeningProver<'a> {
        srs: &'a Srs,
        cube: &'a LayeredCube,
        w1: Vec<Fq>,
        w2: Vec<Fq>,
        u: &'a [Fq],
    }

    impl<'a> ReduceProver for DoubleOpeningProver<'a> {
        fn round_message(&self, alpha: Fq, prefix: &[Fq]) -> RoundMessage {
            let w = if alpha.0 & 1 == 0 { &self.w1 } else { &self.w2 };
            let honest = HonestReduceProver {
                srs: self.srs,
                cube: self.cube,
                w,
                u: self.u,
            };
            honest.round_message(alpha, prefix)
        }

        fn terminal(&self, alpha: Fq, point: &[Fq]) -> Fq {
            let w = if alpha.0 & 1 == 0 { &self.w1 } else { &self.w2 };
            let honest = HonestReduceProver {
                srs: self.srs,
                cube: self.cube,
                w,
                u: self.u,
            };
            honest.terminal(alpha, point)
        }
    }

    #[test]
    fn double_opening_yields_short_kernel() {
        // Crafted SRS + query point making an ACCEPTING double-opener:
        // duplicate the generator pair (i=0,j=0) <-> (i=1,j=0) — layered
        // indices 0 and 4 — and set the first data query variable to 1/2
        // so that eq((0,...),u) = eq((1,...),u). Then e := e_0 − e_4 is a
        // short kernel of the crafted SRS AND has zero T-sum, so w and
        // w + e are two short openings of the same cm with the same
        // evaluation claim — the cheater alternates by α parity and BOTH
        // runs accept. Extraction must output their difference as a short
        // kernel (the MSIS-solution outcome; honest sampling rules such
        // pairs out — that IS Module-SIS).
        let k = 3;
        let cube = LayeredCube::new(k, 4);
        let base = Srs::from_seed(1, 16, cube.size(), b"dup-base");
        let mut columns: Vec<ModulePoint> =
            (0..cube.size() + 1).map(|c| base.generator(c).clone()).collect();
        columns[4] = columns[0].clone();
        let srs = Srs::from_columns(1, 16, columns);
        let n = 1usize << k;
        let f: Vec<Fq> = (0..n).map(|i| Fq::from_u64((i as u64 * 997 + 3) & 0xFFFF)).collect();
        let w = cube.digit_layers(&f);
        let mut w2 = w.clone();
        // w2 = w + e_0 − e_4 (still short: digit-sized entries).
        w2[0] = w2[0].add(&Fq::ONE);
        w2[4] = w2[4].sub(&Fq::ONE);
        assert_eq!(srs.commit_scalars(&w), srs.commit_scalars(&w2));
        let cm = srs.commit_scalars(&w);
        let half = Fq::from_u64(2).inv().expect("2 invertible");
        // i=0 vs i=1 differ in the LAST data bit (MSB-first layout), so
        // the half-point goes there.
        let u: Vec<Fq> = vec![Fq::from_u64(7), Fq::from_u64(9), half];
        // The two witnesses carry the SAME evaluation claim.
        let v = eval_v(&cube, &w, &u);
        assert_eq!(eval_v(&cube, &w2, &u), v);
        let prover = DoubleOpeningProver {
            srs: &srs,
            cube: &cube,
            w1: w,
            w2,
            u: &u,
        };
        let out = extract(&prover, &srs, &cube, &cm, &u, &v, b"s3");
        // The two alphas split the cheater: the two recovered witnesses
        // differ, both open cm, both carry the claim, and their difference
        // is a SHORT kernel on the crafted SRS.
        assert_ne!(out.witnesses[0], out.witnesses[1]);
        assert!(out.commitment_consistent[0] && out.commitment_consistent[1]);
        assert!(out.evaluation_consistent[0] && out.evaluation_consistent[1]);
        assert!(out.kernel.is_some(), "kernel relation must be found");
        assert!(out.kernel_short, "the kernel is short (MSIS solution)");
        let (k_g, k_p) = out.kernel.clone().unwrap();
        assert!(k_p.is_zero());
        // The kernel is supported exactly on the duplicated pair {0, 4}
        // with unit-magnitude coefficients (the sign depends on which run
        // saw which witness).
        let support: Vec<usize> = k_g
            .iter()
            .enumerate()
            .filter(|(_, x)| !x.is_zero())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(support, vec![0, 4]);
        let mags: Vec<u64> = support.iter().map(|&i| k_g[i].0.min(Q_50 - k_g[i].0)).collect();
        assert!(mags.iter().all(|&m| m == 1), "unit coefficients");
    }

    #[test]
    fn grid_interpolation_recovers_multilinear() {
        // The tensor interpolation through field grid values reproduces a
        // known multilinear's cube values.
        let m = 5;
        let cube_vals: Vec<Fq> = (0..1usize << m)
            .map(|i| Fq::from_u64((i as u64 * 7919 + 13) & 0xFFFF))
            .collect();
        let salt = b"grid-test";
        let grid = derive_grid(salt, m);
        let (s0, s1) = &grid;
        // Leaf values: the multilinear evaluated at the grid points.
        let leaves: Vec<Fq> = (0..1usize << m)
            .map(|p| {
                // Evaluate the multilinear with cube values cube_vals at
                // the grid point (MSB-first).
                let mut acc = Fq::ZERO;
                for b in 0..(1usize << m) {
                    let mut eqv = Fq::ONE;
                    for i in 0..m {
                        let bv = Fq::from_u64(((b >> (m - 1 - i)) & 1) as u64);
                        let pv = if (p >> (m - 1 - i)) & 1 == 0 {
                            s0[i]
                        } else {
                            s1[i]
                        };
                        eqv = eqv.mul(
                            &bv.mul(&pv).add(&Fq::ONE.sub(&bv).mul(&Fq::ONE.sub(&pv))),
                        );
                    }
                    acc = acc.add(&eqv.mul(&cube_vals[b]));
                }
                acc
            })
            .collect();
        let recovered = interpolate_grid_to_cube(&grid, &leaves, m);
        assert_eq!(recovered, cube_vals);
    }

    #[test]
    fn honest_prover_messages_match_protocol() {
        // The adapter's messages equal the in-protocol reduce messages for
        // the same challenge sequence.
        let (srs, cube, _f, w, u, cm) = setup(3, b"adapter");
        let v = eval_v(&cube, &w, &u);
        let prover = HonestReduceProver {
            srs: &srs,
            cube: &cube,
            w: &w,
            u: &u,
        };
        // Draw a challenge sequence.
        let mut t = Transcript::new_default(b"adapter-t");
        let alpha = Fq::challenge(&mut t, b"a").unwrap();
        let rs: Vec<Fq> = (0..cube.num_vars())
            .map(|_| Fq::challenge(&mut t, b"r").unwrap())
            .collect();
        let msg0 = prover.round_message(alpha, &[]);
        // Round 1's cube sum must equal cm + v·αP.
        let target = cm.axpy(&v, &srs.value_column().scale(&alpha));
        assert_eq!(msg0.cube_sum(), target);
        // Round 2 with prefix [r1].
        let msg1 = prover.round_message(alpha, &rs[..1]);
        assert_eq!(msg0.eval(&rs[0]), msg1.cube_sum());
        // The terminal at the full point: a = Ŵ(r) — cross-check against
        // direct MLE evaluation of the digit witness.
        let mut eqv = Fq::ZERO;
        for (b, &wb) in w.iter().enumerate() {
            eqv = eqv.add(&eq_eval_index(b, &rs, cube.num_vars()).mul(&wb));
        }
        assert_eq!(prover.terminal(alpha, &rs), eqv);
        let _ = v;
    }
}
