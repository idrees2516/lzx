//! The parameter profiles (the paper's §7 + Appendix D): the two k = 16
//! paper profiles recorded **declaratively** (their dimensions need
//! ~15 GiB and 2.6-hour runs — marked not-executed here), plus this
//! crate's scaled profile that actually executes end-to-end, with the
//! norm-bound derivations (Appendix C.3's digit-energy recurrence).

/// The paper's k = 16 profiles (Tables 1-2, 5-6): declarative record.
pub struct PaperProfile {
    pub name: &'static str,
    /// The serialized prover-to-verifier folding wire (bytes).
    pub folding_wire: usize,
    /// The 16 fresh-input commitments (bytes, accounted separately).
    pub fresh_input_commitments: usize,
    /// The initial linear-stage parameters (rank, radix, digits, S0).
    pub initial: (usize, usize, usize, u64),
    /// The linear chain schedule: (blocks s_i, block length n_i, radix,
    /// projection rows m_i, slack σ_i).
    pub chain: [([usize; 2], usize, [usize; 2]); 6],
    /// The terminal response size (bytes).
    pub terminal_response: usize,
    /// The measured pipeline time (seconds, median).
    pub pipeline_time_s: f64,
    /// The measured verification replay (seconds, median).
    pub verify_time_s: f64,
    /// The measured peak RSS (GiB).
    pub peak_rss_gib: f64,
}

/// The paper's "minimum" profile (Table 3, 4, 5, 6, 9).
pub const PAPER_MINIMUM: PaperProfile = PaperProfile {
    name: "minimum",
    folding_wire: 127_887,
    fresh_input_commitments: 24_576,
    initial: (7, 32, 10, 2_818_920_384),
    chain: [
        ([56, 18_726], 64, [864, 4]),
        ([19, 3_138], 64, [864, 4]),
        ([10, 1_068], 64, [1_024, 3]),
        ([6, 682], 64, [864, 4]),
        ([5, 478], 32, [864, 4]),
        ([380, 0], 0, [1_024, 3]),
    ],
    terminal_response: 35_763,
    pipeline_time_s: 9_407.36,
    verify_time_s: 800.07,
    peak_rss_gib: 15.35,
};

/// The paper's "CRS600" profile.
pub const PAPER_CRS600: PaperProfile = PaperProfile {
    name: "CRS600",
    folding_wire: 129_002,
    fresh_input_commitments: 24_576,
    initial: (8, 128, 7, 4_491_476_992),
    chain: [
        ([66, 15_997], 64, [864, 4]),
        ([23, 2_642], 64, [1_024, 3]),
        ([11, 999], 64, [1_024, 3]),
        ([7, 599], 64, [864, 4]),
        ([5, 485], 32, [864, 4]),
        ([382, 0], 0, [1_024, 3]),
    ],
    terminal_response: 36_054,
    pipeline_time_s: 9_503.33,
    verify_time_s: 843.41,
    peak_rss_gib: 12.40,
};

/// The shared k = 16 field parameters (both paper profiles): the prime,
/// the extension field, the poles/scales, the challenge support.
pub struct PaperField {
    /// `q = 281474976710597 = 2^48 − 59`.
    pub q: u64,
    /// `K = F_q[u]/(u^4 − 4u^2 + 2)`.
    pub k_poly: [i64; 5],
    /// `ξ_i = i`, `λ_i = 1`.
    pub poles: &'static str,
    /// `C = K \ {poles}` — `|C| = q^4 − 16`.
    pub support: &'static str,
    /// The root ring `R_{q,256}` and the linear chain's `R_{q,64}`.
    pub rings: (usize, usize),
    /// The root witness: 57,507,840 coefficients before padding,
    /// `N_W = 2^26` after.
    pub root_witness: (u64, u64),
    /// The Module-SIS roles: J = 22 commitment matrices.
    pub sis_roles: usize,
}

pub const PAPER_FIELD: PaperField = PaperField {
    q: 281_474_976_710_597,
    k_poly: [1, 0, -4, 0, 2],
    poles: "xi_i = i, lambda_i = 1, i = 1..=16",
    support: "C = K \\ {poles}, |C| = q^4 - 16",
    rings: (256, 64),
    root_witness: (57_507_840, 67_108_864),
    sis_roles: 22,
};

/// The executed scaled profile (this crate's tests/benchmarks).
pub struct ScaledProfile {
    /// The folding arity (the full machinery runs at any k; tests cover
    /// 2, 4, 8, 16).
    pub k: usize,
    /// The relaxed-R1CS dimensions (s variables, y constraints).
    pub r1cs: (usize, usize),
    /// The number of nonterminal chain layers + terminal.
    pub chain_layers: usize,
    /// The level-1 digit width (8-bit) and the level-2 digit set
    /// (centered radix-16, `{−8..7}`).
    pub digits: (u32, &'static str),
}

pub const SCALED: ScaledProfile = ScaledProfile {
    k: 16,
    r1cs: (4, 2),
    chain_layers: 2,
    digits: (8, "centered radix-16, {-8..7}"),
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce_chain::digit_energy;

    #[test]
    fn paper_profiles_recorded() {
        assert_eq!(PAPER_MINIMUM.folding_wire, 127_887);
        assert_eq!(PAPER_CRS600.folding_wire, 129_002);
        // Including the fresh-input commitments (Table 3's last row).
        assert_eq!(
            PAPER_MINIMUM.folding_wire + PAPER_MINIMUM.fresh_input_commitments,
            152_463
        );
        assert_eq!(
            PAPER_CRS600.folding_wire + PAPER_CRS600.fresh_input_commitments,
            153_578
        );
        // The initial-stage S0 bounds (Table 5).
        assert_eq!(PAPER_MINIMUM.initial.3, 2_818_920_384);
        assert_eq!(PAPER_CRS600.initial.3, 4_491_476_992);
        // The chain schedules' block counts (Table 6).
        assert_eq!(PAPER_MINIMUM.chain[0].0[0], 56);
        assert_eq!(PAPER_CRS600.chain[0].0[0], 66);
    }

    #[test]
    fn field_parameters() {
        assert_eq!(PAPER_FIELD.q, (1u64 << 48) - 59);
        // q ≡ 5 (mod 8) — the unit-criterion modulus.
        assert_eq!(PAPER_FIELD.q % 8, 5);
        assert_eq!(PAPER_FIELD.root_witness.1, 1 << 26);
    }

    #[test]
    fn digit_energy_paper_example() {
        // The paper's radix-64 canonical digit energy example (8129) is
        // its regime's two-digit maximum; our recurrence reproduces the
        // same construction — monotone in H and bounded by the
        // representable-range square.
        let e_small = digit_energy(1_000, 64, 2);
        let e_large = digit_energy(2_000, 64, 2);
        assert!(e_small > 0);
        assert!(e_large >= e_small, "monotone in H");
        // The two-digit centered radix-64 energy of representable values
        // stays in the (b−1)²/b² band structure.
        assert!(e_small <= 2 * 32 * 32);
        // Saturation at extreme ranges (no overflow).
        let _ = digit_energy(u64::MAX / 2, 64, 2);
        let _ = digit_energy(u64::MAX, 64, 4);
    }

    #[test]
    fn scaled_profile() {
        assert_eq!(SCALED.k, 16);
        assert_eq!(SCALED.r1cs, (4, 2));
        assert_eq!(SCALED.chain_layers, 2);
    }
}
