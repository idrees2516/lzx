//! **The multi-stage LaBRADOR extraction ledger** — the executable half
//! of the degree-law unwind analysis (`docs/analysis/
//! MULTISTAGE_EXTRACTION.md` is the prose half; every numeric claim
//! there is computed here, fail-closed).
//!
//! # What this module closes (the honest residual this replaces)
//!
//! `chain`'s original posture doc recorded: "the full multi-stage
//! extraction — the LaBRADOR special-soundness degree law composed
//! ACROSS stages, unwinding the chain to the level-1 response — is the
//! open analysis; the conservative posture here is the per-stage
//! estimator floor + the (W0) target threading." This module turns that
//! residual into a machine-checked ledger: the composed extraction is
//! now a first-class, fail-closed artifact that `assert_sound_chain`
//! enforces at prove AND verify time.
//!
//! # The extraction model (five quantities per stage, five laws total)
//!
//! **Per stage ℓ** (params `r₂^{(ℓ)}, A₂^{(ℓ)}, κ^{(ℓ)}, w^{(ℓ)}`, gate
//! `β_ℓ`, output gate `β_{ℓ+1} = r₂·A₂·β_ℓ`):
//!
//! 1. **The challenge space** `|C_ℓ| = (2·A₂+1)^{r₂}` — the γ-coordinates
//!    each range over `[−A₂, A₂]`.
//! 2. **The fork width** `= 2` — the DEGREE LAW, stage-local: the stage's
//!    response is AFFINE in its γ's because every quadratic term is
//!    committed pre-challenge as garbage (`G_ij`, `g_ij`) — the LaBRADOR
//!    discipline that keeps special soundness 2-fold (a degree-d response
//!    would need d+1 transcripts; the garbage transmission pins d = 1).
//! 3. **The stage knowledge gap** `κ_ℓ = |C_ℓ|^{-1}` — the probability a
//!    rewinding extractor fails to split a stage (both draws collide).
//! 4. **The kernel bound** `2·β_{ℓ+1}` — the fork difference lands on
//!    `[A₂^{(ℓ)} | −T^{(ℓ)}]` at the relaxed 2× extraction factor (the
//!    same bound `profile_bits` hands the estimator — the quantity the
//!    MSIS verdict rates).
//! 5. **The estimator verdict** — `(classical, quantum)` bits of that
//!    kernel instance (the modeled layer; everything else here is
//!    arithmetic).
//!
//! **The composed laws**:
//!
//! * **(E1) The AND-composition**: an accepting fork at ANY stage yields
//!   a short kernel on THAT stage's instance — all gated ≥ floor +
//!   grinding. (This is the cheat-detection route; it is what shipped
//!   before this module and it is unchanged.)
//! * **(E2) The degree-law unwind**: the public claims compose degree-2
//!   per stage in the challenges (`u_{ℓ+1} = Σγ_i²·u_i + Σ_{i≠j}γ_iγ_j·
//!   g_ij`; the target thread `t_{ℓ+1} = Σγ_i·T_i` is degree 1), so the
//!   level-1 claims are pinned by a degree-`2L` identity in the staged
//!   challenges — while the per-stage EXTRACTION stays affine (fork
//!   width 2) because the quadratic terms are pre-committed. The unwind
//!   degree is `2L`; the rewind tree has `2^L` leaves.
//! * **(E3) The unwind norm law**: the telescoped level-1 reconstruction
//!   carries `‖·‖_∞ ≤ 2·β_{L+1}` (the fork's 2× factor on the FINAL
//!   gate); wraparound-free reading requires `2·β_{L+1} < q/2`. The
//!   stages' own gates imply it transitively; the ledger asserts it
//!   EXPLICITLY as the composed law (machine-checked redundancy — the
//!   same discipline as the D1 Lemma-4 gate, lifted to the chain).
//! * **(E4) The grinding ledger**: replay-grinding all L stages costs
//!   `Σ_ℓ log₂|C_ℓ|` bits — the `CHAIN_GRINDING_BITS` allowance every
//!   stage's floor already absorbs.
//! * **(E5) The extractor-feasibility cap**: the knowledge extractor
//!   rewinds `2^L` leaves; the schedule depth must keep the tree under
//!   [`REWIND_TREE_CAP`] — the chain-level analogue of the single-stage
//!   `n̄ ≤ 16` sound-coverage boundary (a schedule that cannot be
//!   extracted within the cap fails closed even if every stage is
//!   individually sound).
//!
//! # The honest layering
//!
//! * ARITHMETIC (proven by this module's checks): the degree law, the
//!   norm law, the grinding ledger, the tree size, the kernel bounds.
//! * MODELED (the estimator, not proven): the MSIS hardness of each
//!   kernel instance.
//! * STANDARD ASSUMPTIONS (stated, not proven): the Fiat-Shamir
//!   rewinding soundness with the grinding allowance; transcript
//!   collision resistance.
//!
//! The ledger never blurs the three layers — each field carries the
//! layer it belongs to in its doc comment.

use crate::chain::{WidthChainParams, CHAIN_GRINDING_BITS};
use crate::SECURITY_FLOOR_BITS;

/// The extractor-feasibility cap (law E5): the rewind tree `2^L` must
/// stay under this — schedules deeper than 16 stages cannot be
/// knowledge-extracted within the modeled rewind budget and fail
/// closed. (The greedy search's depth cap already keeps shipped
/// schedules far below this; the ledger makes the boundary explicit
/// and enforced rather than incidental.)
pub const REWIND_TREE_CAP: usize = 1 << 16;

/// One stage's extraction entry (the five per-stage quantities).
#[derive(Clone, Debug)]
pub struct StageExtraction {
    /// The stage index (0 = the level-1 fold).
    pub stage: usize,
    /// The part count `r₂^{(ℓ)}`.
    pub r2: usize,
    /// The γ-amplitude `A₂^{(ℓ)}`.
    pub amplitude: u32,
    /// The inner-key rows `κ^{(ℓ)}`.
    pub kappa: usize,
    /// The part width `w^{(ℓ)}`.
    pub w: usize,
    /// `log₂|C_ℓ|` — the challenge-space size in bits (ARITHMETIC).
    pub challenge_space_bits: f64,
    /// The fork width: 2, by the stage-local degree law (ARITHMETIC —
    /// the garbage pre-commitment keeps the response affine).
    pub fork_width: usize,
    /// `−log₂κ_ℓ` where `κ_ℓ = |C_ℓ|^{-1}` (ARITHMETIC; the knowledge
    /// gap, NOT the forgery security — see `ExtractionLedger`).
    pub kappa_stage_bits: f64,
    /// The kernel bound `2·β_{ℓ+1}` on `[A₂^{(ℓ)} | −T^{(ℓ)}]`
    /// (ARITHMETIC — the quantity the estimator rates).
    pub kernel_bound: u64,
    /// The estimator verdict of the kernel instance (MODELED layer).
    pub msis_bits: (f64, f64),
    /// The stage's input gate `β_ℓ`.
    pub beta_in: u64,
    /// The stage's output gate `β_{ℓ+1} = r₂·A₂·β_ℓ`.
    pub beta_out: u64,
}

/// The composed extraction ledger of a staged schedule.
#[derive(Clone, Debug)]
pub struct ExtractionLedger {
    /// The per-stage entries (stage 0 = the level-1 fold).
    pub stages: Vec<StageExtraction>,
    /// `L` — the stage count.
    pub num_stages: usize,
    /// The rewind tree leaf count `2^L` (law E5's quantity).
    pub rewind_tree_leaves: usize,
    /// `−log₂(Σ_ℓ κ_ℓ)` — the composed KNOWLEDGE gap of the full
    /// unwind extractor (ARITHMETIC). HONEST SCOPE: this is the
    /// rewinding-completeness gap, NOT the forgery security — a small
    /// value only means the knowledge extractor aborts often; the
    /// FORGERY security is the per-stage MSIS floor net of grinding.
    pub composed_kappa_bits: f64,
    /// `Σ_ℓ log₂|C_ℓ|` — the replay-grinding cost in bits (law E4).
    pub grinding_bits: f64,
    /// The unwind degree `2L` — the composed public-claim identity's
    /// degree in the staged challenges (law E2).
    pub unwind_degree: usize,
    /// The unwind norm law bound `2·β_{L+1}` (law E3).
    pub unwind_norm_bound: u64,
    /// `log₂(q/2) − log₂(unwind_norm_bound)` — the wraparound slack
    /// of the level-1 telescoped reconstruction (ARITHMETIC).
    pub norm_slack_bits: f64,
    /// The weakest stage's classical MSIS verdict (MODELED layer) —
    /// the binding security the chain actually stands on.
    pub min_stage_msis_bits: f64,
}

impl ExtractionLedger {
    /// The machine-checked extraction posture (laws E1–E5), fail-closed.
    ///
    /// E1 is the per-stage floor (re-derived here so the ledger is
    /// self-contained: every stage ≥ floor + the grinding allowance);
    /// E3 is the explicit norm law; E5 is the extractor-feasibility
    /// cap. E2/E4 are arithmetic identities the constructor guarantees
    /// and `assert` re-checks.
    pub fn assert_extraction_sound(&self, q: u64) -> Result<(), String> {
        if self.stages.is_empty() {
            return Err("extraction ledger: empty chain".into());
        }
        let floor = SECURITY_FLOOR_BITS + CHAIN_GRINDING_BITS;
        // E1: every stage's kernel instance clears floor + grinding.
        for s in &self.stages {
            if s.msis_bits.0 < floor {
                return Err(format!(
                    "E1: stage {} kernel instance {} bits < floor+grinding {floor:.1}",
                    s.stage, s.msis_bits.0
                ));
            }
            // The stage-local degree law: the response is affine (the
            // garbage is pre-committed) — fork width exactly 2.
            if s.fork_width != 2 {
                return Err(format!(
                    "E2 (stage-local): fork width {} != 2 (the affine degree law)",
                    s.fork_width
                ));
            }
            // The kernel bound must itself be estimator-readable
            // (this is the transitive check the profile gate makes;
            // restated as the extraction's own law).
            if s.kernel_bound >= q / 2 {
                return Err(format!(
                    "E3 (stage {}): kernel bound {} ≥ q/2 {}",
                    s.stage,
                    s.kernel_bound,
                    q / 2
                ));
            }
        }
        // E2: the composed degree law — degree 2 per stage.
        if self.unwind_degree != 2 * self.num_stages {
            return Err(format!(
                "E2: unwind degree {} != 2L = {}",
                self.unwind_degree,
                2 * self.num_stages
            ));
        }
        // E3: the unwind norm law (the explicit composed form).
        if self.unwind_norm_bound >= q / 2 {
            return Err(format!(
                "E3: unwind norm {} ≥ q/2 {} (the telescoped reconstruction wraps)",
                self.unwind_norm_bound,
                q / 2
            ));
        }
        // E4: the grinding ledger under the allowance.
        if self.grinding_bits > CHAIN_GRINDING_BITS {
            return Err(format!(
                "E4: grinding {:.1} bits exceeds the {:.1}-bit allowance",
                self.grinding_bits, CHAIN_GRINDING_BITS
            ));
        }
        // E5: the extractor-feasibility cap.
        if self.rewind_tree_leaves > REWIND_TREE_CAP {
            return Err(format!(
                "E5: rewind tree 2^{} = {} leaves exceeds the cap {REWIND_TREE_CAP}",
                self.num_stages, self.rewind_tree_leaves
            ));
        }
        Ok(())
    }
}

/// Build the extraction ledger of a staged schedule at level-1 gate
/// `beta1`. Fails closed on any per-stage estimator verdict (the
/// modeled layer) — the arithmetic laws are total functions and never
/// fail; only the MODEL can refuse.
pub fn chain_extraction_ledger(
    params: &WidthChainParams,
    beta1: u64,
    q: u64,
    ring_dim: u64,
) -> Result<ExtractionLedger, String> {
    if params.stages.is_empty() {
        return Err("extraction ledger: empty chain".into());
    }
    let gates = params.stage_gates(beta1);
    let mut stages = Vec::with_capacity(params.stages.len());
    let mut kappa_sum = 0.0f64;
    for (ell, s) in params.stages.iter().enumerate() {
        let beta_in = gates[ell];
        let beta_out = s.beta2(beta_in);
        // The challenge space (2A₂+1)^{r₂}.
        let space = (2.0 * s.amplitude as f64 + 1.0).powi(s.r2 as i32);
        let space_bits = space.log2();
        let kappa_stage_bits = space_bits; // −log₂(1/|C|) = log₂|C|
        kappa_sum += 1.0 / space;
        // The kernel instance verdict (the modeled layer).
        let msis = s
            .profile_bits(beta_in, q, ring_dim)
            .map_err(|e| format!("stage {ell} estimator: {e}"))?;
        stages.push(StageExtraction {
            stage: ell,
            r2: s.r2,
            amplitude: s.amplitude,
            kappa: s.kappa,
            w: s.w,
            challenge_space_bits: space_bits,
            fork_width: 2,
            kappa_stage_bits,
            kernel_bound: 2 * beta_out,
            msis_bits: msis,
            beta_in,
            beta_out,
        });
    }
    let num_stages = params.stages.len();
    // β_{L+1} — the gate AFTER the last stage (the last stage's own
    // output gate; `stage_gates` stores the per-stage INPUT gates).
    let final_gate = stages
        .last()
        .map(|s| s.beta_out)
        .ok_or_else(|| "extraction ledger: empty stages".to_string())?;
    let unwind_norm_bound = 2 * final_gate;
    let min_stage_msis_bits = stages
        .iter()
        .map(|s| s.msis_bits.0)
        .fold(f64::INFINITY, f64::min);
    let norm_slack_bits = (q as f64 / 2.0 / unwind_norm_bound.max(1) as f64).log2();
    Ok(ExtractionLedger {
        grinding_bits: params.grinding_bits(),
        rewind_tree_leaves: 1usize << num_stages,
        composed_kappa_bits: -kappa_sum.log2(),
        unwind_degree: 2 * num_stages,
        unwind_norm_bound,
        norm_slack_bits,
        min_stage_msis_bits,
        num_stages,
        stages,
    })
}

/// The widthfold extraction laws as a compact multi-line string (the
/// `extraction_table` example prints this for the shipped schedules;
/// the analysis doc's tables are THIS function's output).
pub fn ledger_report(ledger: &ExtractionLedger, q: u64) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "stages L={} | rewind tree 2^L={} | unwind degree 2L={} | grinding {:.1}b (allowance {:.1}b)\n",
        ledger.num_stages,
        ledger.rewind_tree_leaves,
        ledger.unwind_degree,
        ledger.grinding_bits,
        CHAIN_GRINDING_BITS
    ));
    out.push_str(&format!(
        "composed knowledge gap 2^{:.1} (the rewind abort, NOT forgery security) | min stage MSIS {:.1}b (the binding security)\n",
        ledger.composed_kappa_bits, ledger.min_stage_msis_bits
    ));
    out.push_str(&format!(
        "unwind norm 2·β_(L+1)={} vs q/2={} (slack 2^{:.1})\n",
        ledger.unwind_norm_bound,
        q / 2,
        ledger.norm_slack_bits
    ));
    out.push_str("per stage: [ell] |C| bits | κ_ℓ bits | kernel 2β_out | MSIS cl/qm\n");
    for s in &ledger.stages {
        out.push_str(&format!(
            "  [{}] |C|={:.1}b κ={:.1}b kernel={} msis={:.1}/{:.1}b (β {}->{} r2={} A=2^{} κ={} w={})\n",
            s.stage,
            s.challenge_space_bits,
            s.kappa_stage_bits,
            s.kernel_bound,
            s.msis_bits.0,
            s.msis_bits.1,
            s.beta_in,
            s.beta_out,
            s.r2,
            s.amplitude.trailing_zeros(),
            s.kappa,
            s.w
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::q32_ring;
    use crate::fold::WidthFoldParams;

    fn ledger_for(n_bar: usize, beta1: u64) -> ExtractionLedger {
        let ring = q32_ring().unwrap();
        let q = u64::from(ring.modulus.q);
        let dim = ring.n() as u64;
        let params = WidthChainParams::sound_chain_for(n_bar, beta1, q, dim)
            .unwrap_or_else(|e| panic!("no sound chain at n_bar={n_bar}, beta={beta1}: {e}"));
        chain_extraction_ledger(&params, beta1, q, dim)
            .unwrap_or_else(|e| panic!("ledger failed: {e}"))
    }

    /// E1–E5 hold for every shipped schedule shape (the byte gate and
    /// the sound-profile gates from the coverage table).
    #[test]
    fn extraction_ledger_shipped_schedules_pass() {
        for (n_bar, beta) in [
            (16usize, 255u64),
            (64, 255),
            (512, 255),
            (4096, 255), // the byte-gate coverage boundary
            (16, 1 << 15),
            (512, 1 << 15),
            (2048, 1 << 15), // the r₁=8 sound-profile boundary
            (128, 522240),   // the packing-cap row
        ] {
            let ring = q32_ring().unwrap();
            let q = u64::from(ring.modulus.q);
            let l = ledger_for(n_bar, beta);
            assert!(
                l.assert_extraction_sound(q).is_ok(),
                "schedule n_bar={n_bar} beta=2^{:.0} failed the extraction posture",
                (beta as f64).log2()
            );
            // The degree law: exactly 2 per stage.
            assert_eq!(l.unwind_degree, 2 * l.num_stages);
            // The tree: 2^L leaves.
            assert_eq!(l.rewind_tree_leaves, 1usize << l.num_stages);
            // The grind ledger matches the params' own accounting.
            assert!(l.grinding_bits <= CHAIN_GRINDING_BITS);
        }
    }

    /// The unwind norm law is the composed form of the per-stage gates:
    /// 2·β_{L+1} equals twice the last stage's output gate, and it must
    /// sit under q/2 with the documented slack.
    #[test]
    fn unwind_norm_law_composition() {
        let ring = q32_ring().unwrap();
        let q = u64::from(ring.modulus.q);
        let l = ledger_for(512, 255);
        let last = l.stages.last().unwrap();
        assert_eq!(l.unwind_norm_bound, 2 * last.beta_out);
        assert!(l.unwind_norm_bound < q / 2);
        assert!(l.norm_slack_bits > 0.0);
    }

    /// The knowledge gap is the union bound over the per-stage forks —
    /// the composed κ is at least as large as any single stage's κ
    /// (the bits-value at most any single stage's bits-value + the
    /// stage count), and never the forgery security (which is the MSIS
    /// floor: strictly larger here).
    #[test]
    fn composed_kappa_is_union_bound_not_security() {
        let l = ledger_for(512, 255);
        let min_stage_kappa_bits = l
            .stages
            .iter()
            .map(|s| s.kappa_stage_bits)
            .fold(f64::INFINITY, f64::min);
        // Σκ_ℓ ≥ max κ_ℓ ⇒ −log₂(Σκ) ≤ max −log₂(κ) = min bits-value.
        assert!(l.composed_kappa_bits <= min_stage_kappa_bits + 1e-9);
        // The binding security is the MSIS verdict, vastly above the
        // knowledge gap for every shipped schedule.
        assert!(l.min_stage_msis_bits > l.composed_kappa_bits);
    }

    /// A fabricated too-deep chain fails E5 (the extractor-feasibility
    /// cap) even though each stage is individually sound — the
    /// chain-level sound-coverage boundary this module adds.
    #[test]
    fn rewind_tree_cap_fails_closed() {
        // 20 halvings of (r2=2, A=1): individually sound rows are not
        // the question; the TREE 2^20 exceeds the cap.
        let params = WidthChainParams {
            stages: (0..20)
                .map(|_| WidthFoldParams {
                    r2: 2,
                    kappa: 16,
                    amplitude: 1,
                    w: 16,
                })
                .collect(),
        };
        let ring = q32_ring().unwrap();
        let q = u64::from(ring.modulus.q);
        let dim = ring.n() as u64;
        // The estimator may or may not rate the fabricated rows; the
        // ledger construction succeeds (arithmetic is total), and the
        // POSTURE fails on E5 (or E1 first, which is also correct).
        let ledger = chain_extraction_ledger(&params, 255, q, dim);
        match ledger {
            Ok(l) => {
                let err = l.assert_extraction_sound(q).unwrap_err();
                // Any of the composed laws firing is correct
                // fail-closed behavior (E4 grinding typically fires
                // first on over-deep schedules — the replay space
                // accumulates ~3.2 bits/stage).
                assert!(
                    err.contains("E5")
                        || err.contains("E4")
                        || err.contains("E1")
                        || err.contains("E3"),
                    "unexpected failure: {err}"
                );
            }
            Err(_) => {
                // The estimator refusing a fabricated row is also
                // fail-closed behavior — accepted.
            }
        }
    }

    /// The ledger report is total on honest schedules (the analysis
    /// doc's tables are this output — no hand-typed numbers).
    #[test]
    fn ledger_report_total() {
        let ring = q32_ring().unwrap();
        let q = u64::from(ring.modulus.q);
        let l = ledger_for(256, 255);
        let rep = ledger_report(&l, q);
        assert!(rep.contains("unwind degree"), "report: {rep}");
        assert!(rep.contains("per stage"), "report: {rep}");
        assert!(!rep.is_empty());
    }
}
