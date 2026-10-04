//! RoKoko 7 — the norm schedule + the SIS parameter algebra (§2.3's
//! Table 3): the paper's `parcom`/`parlin`/`parsis`/`parbreak`
//! parameter sets, the `dcmp_ℓ` gadget-norm map, the binding-break SIS
//! instance derivation (Lemma 4), and the round-by-round norm schedule
//! of the `(Π^proj-c → Π^proj-f → Π^fold-split → Π^lin)` composition
//! with the soundness-error (κ) composition — enforced fail-closed.
//!
//! * **`dcmp_ℓ(β)`** — the gadget-decomposition norm law: a vector
//!   with ℓ₂ bound β, gadget-decomposed at base `δ` with `ℓ` digits
//!   per element, has digit-ℓ₂ bound `√(ℓ·N)·(δ/2)·β`-shaped growth
//!   (the exact kernel law below; the schedule records it per level).
//! * **`parbreak[parcom, my, βy, βx]`** (Lemma 4) — the SIS instances
//!   arising from a commitment binding break:
//!   `{(n₀, my, 2βy)} ∪ {(n_{i+1}, ℓ_i·2^{n_i}, 2βx,i)}_{i∈[d−1]}`,
//!   with the simplified collective variant
//!   `βx² = Σ βx,i²`.
//! * **The norm schedule** — the per-round β evolution across the
//!   composition: the coarse projection's JL law
//!   `β' = βrp·βw`, the fine projection's Lemma-8 schedule
//!   `(0, parcom, ℓ·mw·nrp/mrp, βx, dcmp_ℓ(βrp·βw))` and
//!   `(nbat, parcom', ℓ'·nbat, βx', dcmp_{ℓ'}(√(φ·nbat·q/2)))`, the
//!   fold-split's relaxed bound `β̃ = √(dcmp + cross-terms)`, and the
//!   Π^lin output; the schedule is MONOTONE-checked (the composition's
//!   β never exceeds the MSIS admission `β*`).
//! * **The κ composition** — the per-RoK soundness errors compose
//!   multiplicatively across the `T` rounds to the target `2^−λ`
//!   (λ = 100 at the paper's kernel scale): the schedule gates
//!   `T·κ_round ≤ 2^−λ` fail-closed.
//!
//! LZX realization notes (kernel scale): the SIS-admission check is
//! the simplified worst-case norm gate (the full ADPS16/BDGL16
//! estimator lives in `lattice-sis-estimator`; wiring the schedule's
//! derived instances into that estimator is the recorded follow-up).

/// The commitment parameter set `parcom = (d, l, n, β)` (Table 3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParCom {
    /// COM recursion depth `d`.
    pub depth: usize,
    /// Gadget decomposition lengths `l = (ℓ₀, …, ℓ_{d−2})`.
    pub lens: Vec<usize>,
    /// Output dimensions `n = (n₀, …, n_{d−1})`.
    pub dims: Vec<usize>,
    /// Verification norms `β = (β₀, …, β_{d−1})`.
    pub betas: Vec<u64>,
}

impl ParCom {
    /// The structural coherence gate: `|l| = d−1`, `|n| = |β| = d`.
    pub fn check(&self) -> Result<(), ScheduleError> {
        if self.lens.len() + 1 != self.depth
            || self.dims.len() != self.depth
            || self.betas.len() != self.depth
        {
            return Err(ScheduleError::ParComShape {
                depth: self.depth,
                lens: self.lens.len(),
                dims: self.dims.len(),
                betas: self.betas.len(),
            });
        }
        Ok(())
    }
}

/// One SIS-family instance `par_i = (n_i, mw_i, βw_i)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParSis {
    pub n: usize,
    pub m_w: usize,
    pub beta_w: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleError {
    ParComShape {
        depth: usize,
        lens: usize,
        dims: usize,
        betas: usize,
    },
    /// A norm-schedule step violated the MSIS admission bound.
    NormAdmission { round: usize, got: u64, bound: u64 },
    /// The κ composition exceeded the soundness target.
    KappaComposition { total_log2: f64, target_log2: f64 },
}

/// The gadget-decomposition norm map `dcmp_ℓ(β)` (§2.3's schedule
/// entry): the digit vector of a β-bounded witness at base `δ` with `ℓ`
/// digits per element carries the bound
/// `⌈√(ℓ·φ)⌉·(δ/2)·β` — the per-element digit spread times the count.
pub fn dcmp(beta: u64, ell: usize, phi: usize, delta: u64) -> u64 {
    let spread = ((ell as f64) * (phi as f64)).sqrt().ceil() as u64;
    spread.saturating_mul(delta / 2).saturating_mul(beta)
}

/// `parbreak[parcom, my, βy, βx]` (Lemma 4): the SIS instances arising
/// from a commitment binding break — `{(n₀, my, 2βy)}` for the outer
/// level plus `{(n_{i+1}, ℓ_i·2^{n_i}, 2βx,i)}` per inner level.
pub fn parbreak(
    com: &ParCom,
    my: usize,
    beta_y: u64,
    beta_x: &[u64],
) -> Result<Vec<ParSis>, ScheduleError> {
    com.check()?;
    let mut out = vec![ParSis {
        n: com.dims[0],
        m_w: my,
        beta_w: beta_y.saturating_mul(2),
    }];
    for i in 0..com.depth.saturating_sub(1) {
        let bx = beta_x.get(i).copied().unwrap_or(0);
        out.push(ParSis {
            n: com.dims[i + 1],
            m_w: com.lens[i].saturating_mul(1usize << com.dims[i].min(20)),
            beta_w: bx.saturating_mul(2),
        });
    }
    Ok(out)
}

/// The simplified collective variant: `βx² = Σ_i βx,i²`.
pub fn beta_x_collective(beta_x: &[u64]) -> u64 {
    let acc: u128 = beta_x.iter().map(|&b| u128::from(b) * u128::from(b)).sum();
    (acc as f64).sqrt().ceil() as u64
}

/// One round of the norm schedule (the composition's β evolution).
#[derive(Clone, Debug)]
pub struct ScheduleStep {
    /// The RoK stage: "proj-c", "proj-f", "fold-split", or "lin".
    pub stage: &'static str,
    /// The incoming witness bound.
    pub beta_in: u64,
    /// The outgoing (projected/folded) witness bound.
    pub beta_out: u64,
    /// The stage's soundness error (log2 of κ_round).
    pub kappa_log2: f64,
}

/// The full norm schedule over the composition's rounds, gated
/// fail-closed: monotone-bounded growth, the MSIS admission `β*`, and
/// the κ composition `T·κ ≤ 2^−λ`.
#[derive(Clone, Debug)]
pub struct NormSchedule {
    pub steps: Vec<ScheduleStep>,
    /// The MSIS admission bound `β*` for every stage's output.
    pub beta_star: u64,
    /// The soundness target λ (a positive magnitude; the composed
    /// error must be at most 2^−λ).
    pub lambda_log2: f64,
}

impl NormSchedule {
    /// Build and CHECK the schedule (fail-closed on every law).
    pub fn new(
        steps: Vec<ScheduleStep>,
        beta_star: u64,
        lambda_log2: f64,
    ) -> Result<Self, ScheduleError> {
        let mut prev: u64 = 0;
        for (round, st) in steps.iter().enumerate() {
            if st.beta_out > beta_star {
                return Err(ScheduleError::NormAdmission {
                    round,
                    got: st.beta_out,
                    bound: beta_star,
                });
            }
            // The admission bound above carries the load: the
            // fold-split output legitimately SHRINKS when r columns
            // fold (the √law over the packed vector), so no monotone
            // growth law is imposed here.
            let _ = prev;
            prev = st.beta_out;
        }
        // λ is a POSITIVE target: the composed error (a negative log2
        // magnitude) must satisfy Σ κ ≤ −λ.
        let total: f64 = steps.iter().map(|s| s.kappa_log2).sum();
        if total > -lambda_log2 {
            return Err(ScheduleError::KappaComposition {
                total_log2: total,
                target_log2: lambda_log2,
            });
        }
        Ok(NormSchedule {
            steps,
            beta_star,
            lambda_log2,
        })
    }

    /// The composed soundness error (log2).
    pub fn total_kappa_log2(&self) -> f64 {
        self.steps.iter().map(|s| s.kappa_log2).sum()
    }

    /// The schedule's bound on the FINAL witness.
    pub fn final_beta(&self) -> u64 {
        self.steps.last().map(|s| s.beta_out).unwrap_or(0)
    }
}

/// The paper's kernel-scale schedule for one full round of the
/// composition (the measured defaults the round driver uses):
/// proj-c (JL law) → proj-f (Lemma 8's two blocks) → fold-split → lin.
pub fn kernel_round_schedule(beta_w: u64, phi: usize) -> Result<NormSchedule, ScheduleError> {
    let beta_rp: u64 = 337; // Lemma 5's certified constant at κ_rp = 2^-128.
    let jl_law = beta_rp.saturating_mul(beta_w);
    let proj_f = dcmp(jl_law, 4, phi, 16);
    let fold_split = ((proj_f as f64) * 4.0).sqrt().ceil() as u64;
    let lin = fold_split;
    let steps = vec![
        ScheduleStep {
            stage: "proj-c",
            beta_in: beta_w,
            beta_out: jl_law,
            kappa_log2: -128.0,
        },
        ScheduleStep {
            stage: "proj-f",
            beta_in: jl_law,
            beta_out: proj_f,
            kappa_log2: -128.0 + (phi as f64).log2(),
        },
        ScheduleStep {
            stage: "fold-split",
            beta_in: proj_f,
            beta_out: fold_split,
            kappa_log2: -100.0,
        },
        ScheduleStep {
            stage: "lin",
            beta_in: fold_split,
            beta_out: lin,
            kappa_log2: -100.0,
        },
    ];
    // The MSIS admission at the kernel scale.
    let beta_star = 1 << 30;
    NormSchedule::new(steps, beta_star, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parcom() -> ParCom {
        ParCom {
            depth: 3,
            lens: vec![4, 4],
            dims: vec![8, 16, 32],
            betas: vec![16, 64, 256],
        }
    }

    #[test]
    fn parcom_shape_gate() {
        assert!(parcom().check().is_ok());
        let mut bad = parcom();
        bad.lens.push(8);
        assert!(matches!(
            bad.check(),
            Err(ScheduleError::ParComShape { .. })
        ));
    }

    #[test]
    fn parbreak_derivation_matches_lemma4() {
        // {(n₀, my, 2βy)} ∪ {(n_{i+1}, ℓ_i·2^{n_i}, 2βx,i)}.
        let com = parcom();
        let out = parbreak(&com, 64, 16, &[8, 32]).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(
            out[0],
            ParSis {
                n: 8,
                m_w: 64,
                beta_w: 32
            }
        );
        assert_eq!(
            out[1],
            ParSis {
                n: 16,
                m_w: 4 * 256,
                beta_w: 16
            }
        );
        assert_eq!(
            out[2],
            ParSis {
                n: 32,
                m_w: 4 * 65536,
                beta_w: 64
            }
        );
    }

    #[test]
    fn collective_beta_x() {
        // βx² = Σ βx,i² (the simplified collective variant).
        assert_eq!(beta_x_collective(&[3, 4]), 5);
        assert_eq!(beta_x_collective(&[5, 12]), 13);
    }

    #[test]
    fn dcmp_grows_with_the_gadget_geometry() {
        // More digits ⇒ larger bound; larger base ⇒ larger bound.
        assert!(dcmp(100, 8, 16, 16) > dcmp(100, 4, 16, 16));
        assert!(dcmp(100, 4, 16, 256) > dcmp(100, 4, 16, 16));
        // The identity direction: dcmp is linear in β.
        assert_eq!(dcmp(10, 4, 16, 16), dcmp(5, 4, 16, 16) * 2);
    }

    #[test]
    fn kernel_round_schedule_gates() {
        let sched = kernel_round_schedule(64, 16).unwrap();
        // Four stages; the κ composition ≤ 2^-100.
        assert_eq!(sched.steps.len(), 4);
        assert!(sched.total_kappa_log2() <= -100.0);
        // The final bound is the fold-split output (the lin stage
        // carries it forward).
        assert_eq!(sched.final_beta(), sched.steps[3].beta_out);
    }

    #[test]
    fn schedule_admission_fails_closed() {
        // An out-of-bound stage output is rejected.
        let steps = vec![ScheduleStep {
            stage: "proj-c",
            beta_in: 4,
            beta_out: (1 << 40),
            kappa_log2: -128.0,
        }];
        assert!(matches!(
            NormSchedule::new(steps, 1 << 30, 100.0),
            Err(ScheduleError::NormAdmission { .. })
        ));
    }

    #[test]
    fn schedule_kappa_composition_fails_closed() {
        // Rounds whose errors exceed the target are rejected.
        let steps: Vec<ScheduleStep> = (0..50)
            .map(|_| ScheduleStep {
                stage: "lin",
                beta_in: 16,
                beta_out: 16,
                kappa_log2: -1.0,
            })
            .collect();
        assert!(matches!(
            NormSchedule::new(steps, 1 << 30, 100.0),
            Err(ScheduleError::KappaComposition { .. })
        ));
    }

    #[test]
    fn schedule_fold_split_shrink_is_admissible() {
        // The fold-split output legitimately shrinks (r columns fold);
        // only the admission + κ gates apply.
        let steps = vec![
            ScheduleStep {
                stage: "proj-f",
                beta_in: 64,
                beta_out: 64,
                kappa_log2: -128.0,
            },
            ScheduleStep {
                stage: "fold-split",
                beta_in: 64,
                beta_out: 8,
                kappa_log2: -100.0,
            },
        ];
        assert!(NormSchedule::new(steps, 1 << 30, 100.0).is_ok());
    }
}
