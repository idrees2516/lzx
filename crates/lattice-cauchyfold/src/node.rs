//! The CauchyFold node protocol (the paper's §5, Protocol 1 / Figure 1):
//! the transcript order (source + carrier commitments **before** the
//! Cauchy challenge, the output commitment after), the fold, the
//! field-level checks (§5.3's random aggregation + sum-check reducing to
//! evaluation claims at a shared query point `τ`), and the root reduction
//! (§5.3-5.4: the level-2 digit witness `W`, the linear system
//! `ΓW = Y` including the commitment rows, the range polynomial `R16`,
//! the fingerprint sum-check) producing the initial committed linear
//! relation consumed by the finite reduction chain
//! ([`crate::reduce_chain`]).
//!
//! ## The 19 root objects
//!
//! Matching the paper's `k + 3 = 19` persistent table commitments at
//! `k = 16`: the accumulator + 16 fresh states, the carrier, and the
//! folded output. Each is committed as its level-1 digit table under a
//! shared root matrix (the paper samples per-object matrices — a
//! documented deviation; the γ-combination then rides one homomorphic
//! system).
//!
//! ## The field-check legs
//!
//! One `K`-valued sum-check over the shared digit cube carries, after
//! random batching: the Booleanity legs (`x(x−1) = 0` on the designated
//! positions of every state in `I = {0..k+1}`), the output's linear
//! bindings to the claimed vectors `Az*, Bz*, Cz*` (whose quadratic
//! combination `⟨ρ, Az* ⊙ Bz*⟩ − u⟨ρ, Cz*⟩ = ⟨ρ, E*⟩` is then public
//! arithmetic), and the carrier-evaluation leg binding the claimed `H(c)`
//! to the committed carrier table. The residual update
//! `E* = E_0 + Σ a_i² E_i + H(c)/D(c)` is checked as public arithmetic.

use crate::cauchy::{Carrier, CauchyParams, QuadraticMap};
use crate::commit::{AjtaiKey, Level1Encoding, Level2Encoding};
use crate::field_k::{Fq48, K4, Q48};
use crate::sumcheck_k::{restrict_table, KRoundMessage, KSumcheck};
use lattice_core::transcript::Transcript;
use lattice_labrador::ring::Poly;

/// Errors of the node protocol.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeError {
    Shape(String),
    /// The Cauchy challenge hit a pole (completeness error `≤ k/|K|`).
    PoleHit,
    Transcript(String),
    Sumcheck(String),
    /// The public arithmetic of the fold failed.
    FoldArithmetic(&'static str),
    RangeViolation,
    Chain(String),
}

/// The node's structural parameters (the scaled profile; see
/// `crate::params` for the paper's k=16 profiles).
#[derive(Clone, Debug)]
pub struct NodeParams {
    pub cauchy: CauchyParams,
    pub relation: QuadraticMap,
    /// Whether the fixture sources carry Boolean payload positions.
    pub boolean_payload: bool,
    /// The root commitment matrix rows (ring rows).
    pub root_rows: usize,
}

impl NodeParams {
    /// The scaled test profile: `k = 16`, `s = 4` variables, `y = 2`
    /// constraints.
    pub fn scaled(k: usize) -> Self {
        NodeParams {
            cauchy: CauchyParams::paper(k),
            relation: QuadraticMap::benchmark(4, 2, 0xC0FFEE),
            boolean_payload: true,
            root_rows: 1,
        }
    }

    /// The shared digit-cube size: the largest table is the carrier's
    /// (`24·y·k` level-1 digits); all tables pad to this power of two.
    pub fn cube_vars(&self) -> usize {
        let k = self.cauchy.k();
        let max_digits = 24 * self.relation.y * k;
        let state_digits = 24 * self.relation.s;
        let m = max_digits.max(state_digits);
        m.next_power_of_two().trailing_zeros() as usize
    }

    pub fn root_key(&self) -> AjtaiKey {
        // Enough columns for the largest table (the carrier's ring vector).
        let max_digits = 24 * self.relation.y * self.cauchy.k();
        let cols = max_digits.div_ceil(64);
        AjtaiKey::from_seed(self.root_rows, cols, b"cauchyfold-root")
    }
}

/// A committed state: the level-1 digit table + its commitment.
#[derive(Clone)]
pub struct CommittedTable {
    pub encoding: Level1Encoding,
    /// The commitment `A_R · w` (ring rows).
    pub commitment: Vec<Poly>,
}

/// Encode + commit one table under the root key (the ring vector is
/// zero-padded to the key's column count — the key is sized for the
/// largest table, the carrier's).
fn commit_table(key: &AjtaiKey, values: &[K4], boolean: bool) -> CommittedTable {
    let encoding = Level1Encoding::encode(values, boolean);
    let mut w = encoding.to_ring_vector();
    while w.len() < key.cols {
        w.push(Poly::zero());
    }
    let commitment = key.commit(&w).expect("shaped");
    CommittedTable {
        encoding,
        commitment,
    }
}

/// The node witness (prover-side): the source records and the fold.
pub struct NodeWitness {
    pub sources: Vec<Vec<K4>>,
    pub residuals: Vec<Vec<K4>>,
}

/// Generate an honest witness: relaxed records `Q(z_i) = E_i` with the
/// designated Boolean payload positions satisfied.
pub fn honest_witness(params: &NodeParams, seed: u64) -> NodeWitness {
    let k = params.cauchy.k();
    let s = params.relation.s;
    let mut nxt = seed;
    let mut rnd = move || {
        nxt = nxt.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        nxt >> 33
    };
    let sources: Vec<Vec<K4>> = (0..k + 1)
        .map(|_i| {
            (0..s)
                .map(|l| {
                    if params.boolean_payload && l % 4 == 0 {
                        // Boolean payload coefficient: the low digit of
                        // coefficient 0 is 0/1.
                        let bit = rnd() % 2;
                        K4::from_coeffs([(rnd() % (Q48 >> 8)) << 8 | bit, rnd() % Q48, rnd() % Q48, rnd() % Q48])
                    } else {
                        K4::from_coeffs([rnd() % Q48, rnd() % Q48, rnd() % Q48, rnd() % Q48])
                    }
                })
                .collect()
        })
        .collect();
    let residuals: Vec<Vec<K4>> = sources.iter().map(|z| params.relation.eval(z)).collect();
    NodeWitness {
        sources,
        residuals,
    }
}

/// The prover-to-verifier node proof (everything before the chain).
#[derive(Clone)]
pub struct NodeProof {
    /// `C_H` — the carrier commitment (fixed before `c`).
    pub carrier_commitment: Vec<Poly>,
    /// `C_*` — the output commitment (after `c`).
    pub output_commitment: Vec<Poly>,
    /// The claimed folded residual `E*` (public).
    pub folded_residual: Vec<K4>,
    /// The claimed `H(c)` (public, K^y).
    pub claimed_hc: Vec<K4>,
    /// The claimed linear images `Az*`, `Bz*`, `Cz*` (public, K^y each).
    pub az: Vec<K4>,
    pub bz: Vec<K4>,
    pub cz: Vec<K4>,
    /// The field-check sum-check transcript.
    pub field_msgs: Vec<KRoundMessage>,
    /// The field-check query point `τ` (re-derivable; stored for
    /// convenience).
    pub tau: Vec<K4>,
    /// The per-table evaluation claims at `τ` (19 entries: 17 states,
    /// carrier, output).
    pub eval_claims: Vec<K4>,
    /// The root `γ`.
    pub gamma: K4,
    /// The level-2 witness `W` (the prover side; the commitment is
    /// transmitted).
    pub level2: Level2Encoding,
    /// `C_W`.
    pub w_commitment: Vec<Poly>,
    /// The Γ right-hand side `Y` (public).
    pub gamma_rows: usize,
    /// The fingerprint sum-check transcript.
    pub fingerprint_msgs: Vec<KRoundMessage>,
    /// The fingerprint query point.
    pub sigma: Vec<K4>,
    /// The terminal claim `W(σ)` (the K-value).
    pub w_sigma: K4,
    /// The chain proof (see `reduce_chain`).
    pub chain: crate::reduce_chain::ChainProof,
}

/// The public node instance (what the verifier sees beyond the proof).
#[derive(Clone)]
pub struct NodeInstance {
    /// The source commitments `C_0..C_k` (fixed before the protocol).
    pub source_commitments: Vec<Vec<Poly>>,
    /// The claimed residuals `E_0..E_k` (public).
    pub residuals: Vec<Vec<K4>>,
}

/// Prove one node. Returns `(instance, proof)`.
pub fn prove(
    params: &NodeParams,
    witness: &NodeWitness,
    _seed: &[u8],
) -> Result<(NodeInstance, NodeProof), NodeError> {
    let k = params.cauchy.k();
    let y = params.relation.y;
    let key = params.root_key();
    let mut transcript = Transcript::new_default(b"cauchyfold-node");

    // ---- Commit the sources (inputs; their commitments are the instance).
    let source_tables: Vec<CommittedTable> = witness
        .sources
        .iter()
        .map(|z| commit_table(&key, z, params.boolean_payload))
        .collect();
    for t in &source_tables {
        absorb_ring(&mut transcript, b"src-c", &t.commitment).map_err(e)?;
    }
    let instance = NodeInstance {
        source_commitments: source_tables.iter().map(|t| t.commitment.clone()).collect(),
        residuals: witness.residuals.clone(),
    };

    // ---- Stage 1: the carrier (before c).
    let carrier = Carrier::direct(&params.cauchy, &params.relation, &witness.sources);
    let carrier_values: Vec<K4> = carrier.coeffs.concat();
    let carrier_table = commit_table(&key, &carrier_values, false);
    absorb_ring(&mut transcript, b"carrier-c", &carrier_table.commitment).map_err(e)?;

    // ---- c ←$ C = K \ {poles}.
    let c = draw_challenge_off_poles(&params.cauchy, &mut transcript)?;

    // ---- The fold.
    let z_star = params
        .cauchy
        .fold_z(&witness.sources, &c)
        .ok_or(NodeError::PoleHit)?;
    let hc = carrier.eval(&c);
    let e_star = params
        .cauchy
        .fold_e(&witness.residuals, &carrier, &c)
        .ok_or(NodeError::PoleHit)?;
    let output_table = commit_table(&key, &z_star, params.boolean_payload);
    absorb_ring(&mut transcript, b"out-c", &output_table.commitment).map_err(e)?;

    // ---- Stage 2: the field checks.
    // Public claims: Az*, Bz*, Cz*.
    let az = image(&params.relation.a, &z_star);
    let bz = image(&params.relation.b, &z_star);
    let cz = image(&params.relation.c, &z_star);
    absorb_k_vec(&mut transcript, b"az", &az).map_err(e)?;
    absorb_k_vec(&mut transcript, b"bz", &bz).map_err(e)?;
    absorb_k_vec(&mut transcript, b"cz", &cz).map_err(e)?;
    absorb_k_vec(&mut transcript, b"hc", &hc).map_err(e)?;
    absorb_k_vec(&mut transcript, b"estar", &e_star).map_err(e)?;

    // The batching challenges.
    let rho = draw_k_vec(&mut transcript, y)?;
    let alphas = draw_k_vec(&mut transcript, 5)?;

    // Build the shared-cube tables (19): [output, sources 0..k, carrier].
    let cube_vars = params.cube_vars();
    let cube_len = 1usize << cube_vars;
    let mut tables: Vec<Vec<K4>> = Vec::with_capacity(19);
    let push_padded = |tables: &mut Vec<Vec<K4>>, digits: &[u16]| {
        let mut t = vec![K4::ZERO; cube_len];
        for (i, &d) in digits.iter().enumerate() {
            t[i] = K4::from_coeffs([d as u64, 0, 0, 0]);
        }
        tables.push(t);
    };
    push_padded(&mut tables, &output_table.encoding.digits);
    for t in &source_tables {
        push_padded(&mut tables, &t.encoding.digits);
    }
    push_padded(&mut tables, &carrier_table.encoding.digits);
    debug_assert_eq!(tables.len(), k + 3);

    // The weight tables (verifier-computable; both sides derive them).
    let weights = FieldWeights::build(params, &c, &rho, cube_vars);
    let tables_with_weights = {
        let mut tw = tables.clone();
        tw.extend(weights.weight_tables());
        tw
    };

    // The claimed total (public arithmetic).
    let total = field_check_total(params, &weights, &alphas, &rho, &az, &bz, &cz, &hc, &e_star);

    // The g-closure over the full table tuple.
    let g = build_field_g(cube_vars, weights.clone(), alphas.clone(), k);

    let challenges: Vec<K4> = (0..cube_vars)
        .map(|_| draw_k(&mut transcript).map_err(NodeError::Transcript))
        .collect::<Result<Vec<_>, _>>()?;
    // Individual degree 3: the Booleanity legs `Z·(Z−1)·ind` are cubic
    // in the cube variables.
    let sc = KSumcheck {
        num_vars: cube_vars,
        degree: 3,
        tables: tables_with_weights,
        claim: total,
        g: Box::new(g),
    };
    let field_msgs = sc.prove(&challenges).map_err(NodeError::Sumcheck)?;
            let tau = challenges;

    // ---- The evaluation claims at τ (the 19 tables).
    let eval_claims: Vec<K4> = tables
        .iter()
        .map(|t| {
            let mut restricted = t.clone();
            for r in &tau {
                restricted = restrict_table(&restricted, r);
            }
            restricted[0]
        })
        .collect();

    // ---- Stage 3: the root reduction.
    // γ drawn in the FIRST K-coordinate (a plain F_q scalar): the
    // Γ commitment rows are per-coefficient equations, so the
    // γ-combination must ride the scalar slot (the paper's ϑ-embedding
    // for full-K γs would need ring-structured row coefficients — the
    // deviation ledger).
    let gamma = {
        let g = Fq48::challenge(&mut transcript).map_err(NodeError::Transcript)?;
        K4([g, Fq48::ZERO, Fq48::ZERO, Fq48::ZERO])
    };
    let mut combined_c = vec![Poly::zero(); params.root_rows];
    for (i, t) in std::iter::once(&output_table)
        .chain(source_tables.iter())
        .chain(std::iter::once(&carrier_table))
        .enumerate()
    {
        let g_i = gamma.pow(i as u64);
        for (r, c_elem) in combined_c.iter_mut().enumerate() {
            c_elem.add_assign(&scale_ring_by_k4(&t.commitment[r], &g_i));
        }
    }
    let mut v_combined = K4::ZERO;
    for (i, &v) in eval_claims.iter().enumerate() {
        v_combined = v_combined.add(&v.scale(&gamma.pow(i as u64)));
    }
    // The combined level-1 digit stream: each table's digits PADDED to
    // key.cols·64 positions (the per-table ring-vector layout), so the
    // Γ commitment rows and the claim functional share one indexing.
    let padded_len = key.cols * 64;
    let real_lens: Vec<usize> = std::iter::once(&output_table)
        .chain(source_tables.iter())
        .chain(std::iter::once(&carrier_table))
        .map(|t| t.encoding.digits.len())
        .collect();
    let table_digits: Vec<Vec<u16>> = std::iter::once(&output_table)
        .chain(source_tables.iter())
        .chain(std::iter::once(&carrier_table))
        .map(|t| {
            let mut d = t.encoding.digits.clone();
            d.resize(padded_len, 0);
            d
        })
        .collect();
    let w_all: Vec<u16> = table_digits.concat();
    // The combined functional f: γ^i·eq(p, τ) on table i's REAL digit
    // positions, zero on the padding.
    let mut f_weights = vec![K4::ZERO; w_all.len()];
    for (i, _digits) in table_digits.iter().enumerate() {
        let g_i = gamma.pow(i as u64);
        for p in 0..real_lens[i] {
            let eqv = eq_index_at(p, &tau, cube_vars);
            f_weights[i * padded_len + p] = eqv.mul(&g_i);
        }
    }
    // Sanity: the combined claim equals ⟨f, w_all⟩ exactly.
    let check_v: K4 = f_weights
        .iter()
        .zip(w_all.iter())
        .map(|(f, &d)| f.mul(&K4::from_coeffs([d as u64, 0, 0, 0])))
        .fold(K4::ZERO, |a, b| a.add(&b));
    debug_assert_eq!(check_v, v_combined);

    // ---- The level-2 witness W and the Γ system.
    let level2 = Level2Encoding::encode(&w_all);
    let w_ring = level2.to_ring_vector();
    let w_key = AjtaiKey::from_seed(params.root_rows, w_ring.len().max(1), b"cauchyfold-w");
    let w_commitment = w_key.commit(&w_ring).map_err(NodeError::Shape)?;
    absorb_ring(&mut transcript, b"cw", &w_commitment).map_err(e)?;

    // ΓW = Y: the commitment rows (A_R·Rec(W) = C_all, ring-structured) +
    // the claim row (⟨f∘Rec, W⟩ = V).
    let gammas: Vec<K4> = (0..k + 3).map(|i| gamma.pow(i as u64)).collect();
    let gamma_system = GammaSystem::build(
        &key,
        &gammas,
        &combined_c,
        &f_weights,
        v_combined,
        &level2,
        k + 3,
    );

    // The fingerprint sum-check over the W cube.
        let w_cube_vars = level2
        .digits
        .len()
        .next_power_of_two()
        .trailing_zeros() as usize;
    let w_cube_len = 1usize << w_cube_vars;
    let mut w_table = vec![K4::ZERO; w_cube_len];
    for (i, &d) in level2.digits.iter().enumerate() {
        w_table[i] = K4::from_coeffs([(d as i64).rem_euclid(Q48 as i64) as u64, 0, 0, 0]);
    }
    let fp_alpha = draw_k(&mut transcript).map_err(NodeError::Transcript)?;
    let h_alpha = gamma_system.h_alpha_table(&fp_alpha, w_cube_len);
    let z_alpha = crate::sumcheck_k::z_alpha_table(&fp_alpha, w_cube_vars);
    let r16 = crate::sumcheck_k::range_poly(8);
    // The claimed total: α^row-weighted Y; the range part contributes 0.
    let fp_total = gamma_system.alpha_weighted_y(&fp_alpha);
    let fp_degree = 17;
    let c_leg = fp_alpha.pow(gamma_system.rows as u64);
    let g_fp = move |ts: &[Vec<K4>]| -> K4 {
        let mut acc = K4::ZERO;
        for i in 0..ts[0].len() {
            let lin = ts[1][i].mul(&ts[0][i]);
            let rng = r16.eval(&ts[0][i]).mul(&ts[2][i]);
            acc = acc.add(&lin.add(&rng.scale(&c_leg)));
        }
        acc
    };
    let fp_challenges: Vec<K4> = (0..w_cube_vars)
        .map(|_| draw_k(&mut transcript).map_err(NodeError::Transcript))
        .collect::<Result<Vec<_>, _>>()?;
    let fp_sc = KSumcheck {
        num_vars: w_cube_vars,
        degree: fp_degree,
        tables: vec![w_table.clone(), h_alpha, z_alpha],
        claim: fp_total,
        g: Box::new(g_fp),
    };
    let fingerprint_msgs = fp_sc.prove(&fp_challenges).map_err(NodeError::Sumcheck)?;
    let sigma = fp_challenges;

    // The terminal claim W(σ).
    let w_sigma = {
        let mut t = w_table.clone();
        for r in &sigma {
            t = restrict_table(&t, r);
        }
        t[0]
    };
    absorb_k(&mut transcript, b"wsigma", &w_sigma).map_err(e)?;

    // ---- Stage 4: the chain over the committed linear relation
    // (C_W, W, the 4 Fq-scalar equations of the terminal evaluation).
    let chain = crate::reduce_chain::prove_chain(&w_ring, &sigma, &w_sigma, w_cube_vars, &mut transcript)
        .map_err(|e| NodeError::Chain(e.to_string()))?;

    Ok((
        instance,
        NodeProof {
            carrier_commitment: carrier_table.commitment,
            output_commitment: output_table.commitment,
            folded_residual: e_star,
            claimed_hc: hc,
            az,
            bz,
            cz,
            field_msgs,
            tau,
            eval_claims,
            gamma,
            level2,
            w_commitment,
            gamma_rows: gamma_system.rows,
            fingerprint_msgs,
            sigma,
            w_sigma,
            chain,
        },
    ))
}

/// Verify one node against its instance. Returns `Ok(())` on acceptance.
pub fn verify(
    params: &NodeParams,
    instance: &NodeInstance,
    proof: &NodeProof,
) -> Result<(), NodeError> {
    let k = params.cauchy.k();
    let y = params.relation.y;
    let key = params.root_key();
    let mut transcript = Transcript::new_default(b"cauchyfold-node");

    for cmt in &instance.source_commitments {
        absorb_ring(&mut transcript, b"src-c", cmt).map_err(e)?;
    }
    absorb_ring(&mut transcript, b"carrier-c", &proof.carrier_commitment).map_err(e)?;
    let c = draw_challenge_off_poles(&params.cauchy, &mut transcript)?;
    absorb_ring(&mut transcript, b"out-c", &proof.output_commitment).map_err(e)?;

    // The public claims.
    absorb_k_vec(&mut transcript, b"az", &proof.az).map_err(e)?;
    absorb_k_vec(&mut transcript, b"bz", &proof.bz).map_err(e)?;
    absorb_k_vec(&mut transcript, b"cz", &proof.cz).map_err(e)?;
    absorb_k_vec(&mut transcript, b"hc", &proof.claimed_hc).map_err(e)?;
    absorb_k_vec(&mut transcript, b"estar", &proof.folded_residual).map_err(e)?;

    // The residual update: E* = E_0 + Σ a_i² E_i + H(c)/D(c) — public.
    {
        let a_s: Vec<Option<K4>> = (0..k)
            .map(|i| params.cauchy.a_i(i, &c))
            .collect();
        if a_s.iter().any(|a| a.is_none()) {
            return Err(NodeError::PoleHit);
        }
        let d_inv = params
            .cauchy
            .d_eval(&c)
            .inv()
            .ok_or(NodeError::PoleHit)?;
        let mut expect = instance.residuals[0].clone();
        for i in 0..k {
            let a = a_s[i].unwrap();
            let a2 = a.mul(&a);
            for j in 0..y {
                expect[j] = expect[j].add(&instance.residuals[i + 1][j].scale(&a2));
            }
        }
        for j in 0..y {
            expect[j] = expect[j].add(&proof.claimed_hc[j].scale(&d_inv));
        }
        if expect != proof.folded_residual {
            return Err(NodeError::FoldArithmetic("residual update"));
        }
    }

    // The quadratic check (the homogeneous form): ⟨ρ, az⊙bz⟩ = ⟨ρ, E*⟩ —
    // public arithmetic on the claimed images.
    let rho = draw_k_vec(&mut transcript, y)?;
    let alphas = draw_k_vec(&mut transcript, 5)?;
    {
        let mut lhs = K4::ZERO;
        let mut ers = K4::ZERO;
        for j in 0..y {
            lhs = lhs.add(&proof.az[j].mul(&proof.bz[j]).mul(&rho[j]));
            ers = ers.add(&proof.folded_residual[j].mul(&rho[j]));
        }
        if lhs != ers {
            return Err(NodeError::FoldArithmetic("quadratic identity"));
        }
    }

    // The field-check sum-check: rebuild the weight tables and the claim.
    let cube_vars = params.cube_vars();
    let weights = FieldWeights::build(params, &c, &rho, cube_vars);
    let total = field_check_total(params, &weights, &alphas, &rho, &proof.az, &proof.bz, &proof.cz, &proof.claimed_hc, &proof.folded_residual);
    let challenges: Vec<K4> = (0..cube_vars)
        .map(|_| draw_k(&mut transcript).map_err(NodeError::Transcript))
        .collect::<Result<Vec<_>, _>>()?;
    let (_, terminal) = KSumcheck::verify(
        cube_vars,
        3,
        &total,
        &proof.field_msgs,
        &challenges,
    )
    .map_err(|e| NodeError::Sumcheck(format!("field: {e}")))?;
    // The terminal must equal the pointwise g at the claimed table values.
    {
        // Restrict the weight tables to τ.
        let wt: Vec<Vec<K4>> = weights
            .weight_tables()
            .into_iter()
            .map(|t| {
                let mut r = t;
                for x in &challenges {
                    r = restrict_table(&r, x);
                }
                r
            })
            .collect();
        // The claimed table values at τ (the eval claims).
        let vals = &proof.eval_claims;
        let v_out = vals[0];
        let v_srcs = &vals[1..k + 2];
        let v_carrier = vals[k + 2];
        let g_point = field_g_pointwise(
            &wt,
            &alphas,
            &v_out,
            v_srcs,
            &v_carrier,
            cube_vars,
        );
        if g_point != terminal {
            return Err(NodeError::Sumcheck("terminal consistency".into()));
        }
    }

    // The root: γ (first-coordinate), C_all, the level-2 system.
    let gamma = {
        let g = Fq48::challenge(&mut transcript).map_err(NodeError::Transcript)?;
        K4([g, Fq48::ZERO, Fq48::ZERO, Fq48::ZERO])
    };
    // C_all = Σ γ^i C_i — recompute from the instance/proof commitments.
    let mut combined_c = vec![Poly::zero(); params.root_rows];
    {
        let cmts: Vec<&Vec<Poly>> = std::iter::once(&proof.output_commitment)
            .chain(instance.source_commitments.iter())
            .chain(std::iter::once(&proof.carrier_commitment))
            .collect();
        for (i, cmt) in cmts.iter().enumerate() {
            let g_i = gamma.pow(i as u64);
            for (r, elem) in combined_c.iter_mut().enumerate() {
                elem.add_assign(&scale_ring_by_k4(&cmt[r], &g_i));
            }
        }
    }
    // The combined functional f over the PADDED per-table digit streams
    // (verifier-side; the real lengths are fixed by the params: states
    // 24·s digits, the carrier 24·y·k, the output 24·s).
    let padded_len = key.cols * 64;
    let digit_lens: Vec<usize> = {
        let mut v = Vec::new();
        v.push(24 * params.relation.s); // output
        for _ in 0..k + 1 {
            v.push(24 * params.relation.s);
        }
        v.push(24 * y * k);
        v
    };
    let mut f_weights = vec![K4::ZERO; padded_len * (k + 3)];
    for (i, &dl) in digit_lens.iter().enumerate() {
        let g_i = gamma.pow(i as u64);
        for p in 0..dl.min(padded_len) {
            let eqv = eq_index_at(p, &proof.tau, cube_vars);
            f_weights[i * padded_len + p] = eqv.mul(&g_i);
        }
    }
    // V = Σ γ^i v_i.
    let mut v_combined = K4::ZERO;
    for (i, &v) in proof.eval_claims.iter().enumerate() {
        v_combined = v_combined.add(&v.scale(&gamma.pow(i as u64)));
    }
    // The combined commitment check: A_R·Rec(W) = C_all and the claim row
    // — verified through the Γ/fingerprint machinery + the chain.
    let w_len = proof.level2.digits.len();
    let gammas: Vec<K4> = (0..k + 3).map(|i| gamma.pow(i as u64)).collect();
    let gamma_system =
        GammaSystem::build_checked(&key, &gammas, &combined_c, &f_weights, v_combined, &proof.level2, k + 3)?;

    // C_W binding: re-derive the W key and check the commitment.
    let w_ring = proof.level2.to_ring_vector();
    let w_key = AjtaiKey::from_seed(params.root_rows, w_ring.len().max(1), b"cauchyfold-w");
    absorb_ring(&mut transcript, b"cw", &proof.w_commitment).map_err(e)?;
    if !w_key.verify(&w_ring, &proof.w_commitment) {
        return Err(NodeError::Shape("C_W does not open".into()));
    }

    // The fingerprint sum-check.
    let w_cube_vars = w_len.next_power_of_two().trailing_zeros() as usize;
    let w_cube_len = 1usize << w_cube_vars;
    let fp_alpha = draw_k(&mut transcript).map_err(NodeError::Transcript)?;
    let h_alpha = gamma_system.h_alpha_table(&fp_alpha, w_cube_len);
    let z_alpha = crate::sumcheck_k::z_alpha_table(&fp_alpha, w_cube_vars);
    let r16 = crate::sumcheck_k::range_poly(8);
    let fp_total = gamma_system.alpha_weighted_y(&fp_alpha);
    let fp_challenges: Vec<K4> = (0..w_cube_vars)
        .map(|_| draw_k(&mut transcript).map_err(NodeError::Transcript))
        .collect::<Result<Vec<_>, _>>()?;
        let (_, fp_terminal) = KSumcheck::verify(
        w_cube_vars,
        17,
        &fp_total,
        &proof.fingerprint_msgs,
        &fp_challenges,
    )
    .map_err(|e| NodeError::Sumcheck(format!("fingerprint: {e}")))?;
    // Terminal consistency: h_α(σ)·W(σ) + α^{mΓ}·z_α(σ)·R16(W(σ)).
    {
        let mut ha = h_alpha;
        let mut za = z_alpha;
        for x in &fp_challenges {
            ha = restrict_table(&ha, x);
            za = restrict_table(&za, x);
        }
        let expect = ha[0]
            .mul(&proof.w_sigma)
            .add(&r16.eval(&proof.w_sigma).mul(&za[0]).scale(&fp_alpha.pow(gamma_system.rows as u64)));
        if expect != fp_terminal {
            return Err(NodeError::Sumcheck("fingerprint terminal".into()));
        }
    }
    absorb_k(&mut transcript, b"wsigma", &proof.w_sigma).map_err(e)?;

    // The chain.
    crate::reduce_chain::verify_chain(
        &w_ring,
        &proof.sigma,
        &proof.w_sigma,
        w_cube_vars,
        &mut transcript,
        &proof.chain,
    )
    .map_err(|e| NodeError::Chain(e.to_string()))
}

// ---------------------------------------------------------------------------
// The field-check weight machinery
// ---------------------------------------------------------------------------

/// The verifier/prover-shared weight tables for the field-check legs.
#[derive(Clone)]
struct FieldWeights {
    _cube_vars: usize,
    /// The output's binding weights for az, bz, cz (three tables).
    bind_a: Vec<K4>,
    bind_b: Vec<K4>,
    bind_c: Vec<K4>,
    /// The carrier's H(c) binding weights (one table).
    bind_h: Vec<K4>,
    /// The designated-position indicators: output + k+1 sources.
    bool_indicators: Vec<Vec<K4>>,
}

impl FieldWeights {
    fn build(params: &NodeParams, c: &K4, rho: &[K4], cube_vars: usize) -> Self {
        let k = params.cauchy.k();
        let s = params.relation.s;
        let y = params.relation.y;
        let cube_len = 1usize << cube_vars;
        let mk = |f: &dyn Fn(usize) -> K4| -> Vec<K4> {
            (0..cube_len).map(f).collect()
        };
        // The digit-position packing weights: position i maps to
        // (value l, coefficient cc, digit j) with j = i % 6,
        // cc = (i/6) % 4, l = i/24; weight 2^{8(5−j)}.
        let pack_w = |i: usize| -> K4 {
            let j = i % 6;
            K4::from_coeffs([1u64 << (8 * (5 - j)), 0, 0, 0])
        };
        // az-binding: μ_A(x) = Σ_j (ρ_j·A[j][l])·u^{cc}·pack_w(x)·[x
        // belongs to (l, cc)] — full K-arithmetic (the products mix
        // coordinates) with the basis embedding u^{cc} selecting the
        // K-coefficient slot.
        let basis = |cc: usize| -> K4 {
            match cc {
                0 => K4::ONE,
                1 => K4([Fq48::ZERO, Fq48::ONE, Fq48::ZERO, Fq48::ZERO]),
                2 => K4([Fq48::ZERO, Fq48::ZERO, Fq48::ONE, Fq48::ZERO]),
                _ => K4([Fq48::ZERO, Fq48::ZERO, Fq48::ZERO, Fq48::ONE]),
            }
        };
        let bind = |mat: &Vec<Vec<K4>>| -> Vec<K4> {
            mk(&|i| {
                let l = i / 24;
                let cc = (i / 6) % 4;
                if l >= s {
                    return K4::ZERO;
                }
                let w = pack_w(i).mul(&basis(cc));
                let mut acc = K4::ZERO;
                for j in 0..y {
                    acc = acc.add(&rho[j].mul(&mat[j][l]).mul(&w));
                }
                acc
            })
        };
        let bind_a = bind(&params.relation.a);
        let bind_b = bind(&params.relation.b);
        let bind_c = bind(&params.relation.c);
        // The carrier H(c) binding: the carrier's digit stream is
        // (t, j, cc, digit) with length 24·y·k — position i maps to
        // (flat = i/6, cc = (i/6)%4, j = (i/24)%y, t = i/(24y)); weight
        // c^t · pack_w(i); the target: Σ_i Z_H(i)·w(i) = Σ_j hc_j... the
        // claimed H(c) is a K^y vector — the leg's target = the ρ-batched
        // claimed value (see field_check_total). The weight batches with
        // the SAME ρ: η(x) = Σ_j ρ_j·[j(x) = j]·c^{t(x)}·pack_w(x).
        let bind_h = mk(&|i| {
            let per = 24 * y;
            if i >= per * k {
                return K4::ZERO;
            }
            let t = i / per;
            let j = (i / 24) % y;
            let cc = (i / 6) % 4;
            let w = pack_w(i).mul(&basis(cc));
            let ct = c.pow(t as u64);
            rho[j].mul(&ct).mul(&w)
        });
        // The Boolean indicators: the designated positions of each table.
        let bool_indicators = {
            let mut out = Vec::new();
            // The sources' designated Boolean slots: the low digit of
            // coefficient 0 of every 4th value (the encoder's rule; the
            // honest generator constructs boolean payloads there). The
            // scaled profile checks I = {0..k} — the sources — the folded
            // output's payload discipline is the re-encoding follow-up
            // (deviation ledger).
            let state_ind = mk(&|i| {
                let l = i / 24;
                let cc = (i / 6) % 4;
                let j = i % 6;
                if l % 4 == 0 && cc == 0 && j == 5 && l < s {
                    K4::ONE
                } else {
                    K4::ZERO
                }
            });
            out.push(mk(&|_| K4::ZERO)); // output: not in I (scaled)
            for _ in 0..k + 1 {
                out.push(state_ind.clone());
            }
            out
        };
        FieldWeights {
            _cube_vars: cube_vars,
            bind_a,
            bind_b,
            bind_c,
            bind_h,
            bool_indicators,
        }
    }

    fn weight_tables(&self) -> Vec<Vec<K4>> {
        let mut v = vec![
            self.bind_a.clone(),
            self.bind_b.clone(),
            self.bind_c.clone(),
            self.bind_h.clone(),
        ];
        v.extend(self.bool_indicators.iter().cloned());
        v
    }
}

/// The claimed total of the field-check sum-check (public arithmetic).
#[allow(clippy::too_many_arguments)]
fn field_check_total(
    params: &NodeParams,
    _weights: &FieldWeights,
    alphas: &[K4],
    rho: &[K4],
    az: &[K4],
    bz: &[K4],
    cz: &[K4],
    hc: &[K4],
    _e_star: &[K4],
) -> K4 {
    let y = params.relation.y;
    // Binding targets: Σ_j ρ_j az_j, Σ_j ρ_j bz_j, Σ_j ρ_j cz_j (the
    // az/bz/cz batches), and the H(c) target Σ_j ρ_j hc_j.
    let batch = |v: &[K4]| -> K4 {
        let mut acc = K4::ZERO;
        for j in 0..y {
            acc = acc.add(&v[j].mul(&rho[j]));
        }
        acc
    };
    // The α-weights: [bindA, bindB, bindC, bool, Hc].
    alphas[0]
        .mul(&batch(az))
        .add(&alphas[1].mul(&batch(bz)))
        .add(&alphas[2].mul(&batch(cz)))
        .add(&alphas[4].mul(&batch(hc)))
    // The booleanity legs claim 0.
}

/// Build the field-check `g` closure over the table tuple:
/// [0] = output, [1..k+2] = sources, [k+2] = carrier,
/// [k+3] = bind_a, [k+4] = bind_b, [k+5] = bind_c, [k+6] = bind_h,
/// [k+7..] = the boolean indicators.
fn build_field_g(
    _cube_vars: usize,
    _weights: FieldWeights,
    alphas: Vec<K4>,
    k: usize,
) -> impl Fn(&[Vec<K4>]) -> K4 {
    // g(tables) = Σ_x [
    //   α_A·Z_out(x)·bindA(x) + α_B·Z_out(x)·bindB(x) + α_C·Z_out(x)·bindC(x)
    //   + α_bool·Σ_tables Z_t(x)(Z_t(x)−1)·ind_t(x)
    //   + α_H·Z_H(x)·bindH(x) ]
    let bool_alpha = alphas[3];
    let ind_count = k + 2;
    move |ts: &[Vec<K4>]| -> K4 {
        let mut acc = K4::ZERO;
        let n = ts[0].len();
        for i in 0..n {
            let z_out = ts[0][i];
            let mut term = alphas[0].mul(&z_out.mul(&ts[k + 3][i]));
            term = term.add(&alphas[1].mul(&z_out.mul(&ts[k + 4][i])));
            term = term.add(&alphas[2].mul(&z_out.mul(&ts[k + 5][i])));
            term = term.add(&alphas[4].mul(&ts[k + 2][i].mul(&ts[k + 6][i])));
            // Booleanity over the sources (the output's indicator is all
            // zero in the scaled profile — kept in the table for layout
            // stability).
            for t in 0..ind_count {
                let z = ts[t][i];
                let ind = ts[k + 7 + t][i];
                let zm1 = z.sub(&K4::ONE);
                term = term.add(&bool_alpha.mul(&z.mul(&zm1).mul(&ind)));
            }
            acc = acc.add(&term);
        }
        acc
    }
}

/// The pointwise terminal check of the field sum-check at `τ`.
#[allow(clippy::too_many_arguments)]
fn field_g_pointwise(
    restricted_weights: &[Vec<K4>],
    alphas: &[K4],
    v_out: &K4,
    v_srcs: &[K4],
    v_carrier: &K4,
    _cube_vars: usize,
) -> K4 {
    let mut term = alphas[0].mul(&v_out.mul(&restricted_weights[0][0]));
    term = term.add(&alphas[1].mul(&v_out.mul(&restricted_weights[1][0])));
    term = term.add(&alphas[2].mul(&v_out.mul(&restricted_weights[2][0])));
    term = term.add(&alphas[4].mul(&v_carrier.mul(&restricted_weights[3][0])));
    let bool_alpha = alphas[3];
    // The sources' booleanity at the point (the output's indicator is
    // zero in the scaled profile).
    {
        for (t, &vs) in v_srcs.iter().enumerate() {
            let ind = restricted_weights[5 + t][0];
            term = term.add(&bool_alpha.mul(&vs.mul(&vs.sub(&K4::ONE)).mul(&ind)));
        }
    }
    term
}

// ---------------------------------------------------------------------------
// The Γ system (the root linear relation)
// ---------------------------------------------------------------------------

/// The Γ system: `ΓW = Y` with the ring-structured commitment rows
/// (`A_R·Rec(W) = C_all`) and the claim row (`⟨f∘Rec, W⟩ = V`).
pub struct GammaSystem {
    /// The number of rows `m_Γ`.
    pub rows: usize,
    /// Sparse rows: (w-position, coefficient).
    pub row_coeffs: Vec<Vec<(usize, K4)>>,
    /// The right-hand side `Y`.
    pub y: Vec<K4>,
}

impl GammaSystem {
    /// Build from the root key, the per-table γ powers, the combined
    /// commitment, the combined functional (over the padded per-table
    /// digit streams), and the level-2 witness.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        key: &AjtaiKey,
        gammas: &[K4],
        combined_c: &[Poly],
        f: &[K4],
        v: K4,
        level2: &Level2Encoding,
        num_tables: usize,
    ) -> GammaSystem {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        // The commitment rows: (A_R·Σ_i γ^i·w_i)[r][m] = C_all[r][m],
        // with each table's ring vector padded to key.cols slots and the
        // level-2 recomposition inside.
        for r in 0..key.rows {
            for m in 0..64 {
                let mut coeffs: Vec<(usize, K4)> = Vec::new();
                for i in 0..num_tables {
                    let g_i = gammas[i];
                    for c in 0..key.cols {
                        let a_rc = &key.matrix[r * key.cols + c];
                        for p in 0..64 {
                            // q candidates: p+q = m or p+q = m+64.
                            let mut add_coeff = K4::ZERO;
                            if m >= p {
                                let q = m - p;
                                add_coeff = K4::from_coeffs([
                                    a_rc.0[q].rem_euclid(Q48 as i64) as u64,
                                    0,
                                    0,
                                    0,
                                ]);
                            }
                            let mut sub_coeff = K4::ZERO;
                            if m + 64 >= p && p + 63 >= m + 64 {
                                let q = m + 64 - p;
                                if q < 64 {
                                    sub_coeff = K4::from_coeffs([
                                        a_rc.0[q].rem_euclid(Q48 as i64) as u64,
                                        0,
                                        0,
                                        0,
                                    ]);
                                }
                            }
                            if add_coeff.is_zero() && sub_coeff.is_zero() {
                                continue;
                            }
                            for j in 0..3 {
                                let wpos = 3 * (64 * (i * key.cols + c) + p) + j;
                                if wpos >= level2.digits.len() {
                                    continue;
                                }
                                let base = add_coeff.sub(&sub_coeff);
                                // The centered radix-16 split: value =
                                // 256·d0 + 16·d1 + d2 — weights 16^(2−j).
                                let w16 = K4::from_coeffs([1u64 << (4 * (2 - j)), 0, 0, 0]);
                                let coeff = base.mul(&w16).mul(&g_i);
                                if !coeff.is_zero() {
                                    coeffs.push((wpos, coeff));
                                }
                            }
                        }
                    }
                }
                rows.push(coeffs);
                y.push(K4::from_coeffs([
                    combined_c[r].0[m].rem_euclid(Q48 as i64) as u64,
                    0,
                    0,
                    0,
                ]));
            }
        }
        // The claim row: ⟨f∘Rec, W⟩ = V — coefficient on W[3i+j] =
        // f[i]·16^j.
        {
            let mut coeffs: Vec<(usize, K4)> = Vec::new();
            for (i, &fi) in f.iter().enumerate() {
                if fi.is_zero() {
                    continue;
                }
                for j in 0..3 {
                    let wpos = 3 * i + j;
                    if wpos >= level2.digits.len() {
                        continue;
                    }
                    let w16 = K4::from_coeffs([1u64 << (4 * (2 - j)), 0, 0, 0]);
                    let coeff = fi.mul(&w16);
                    if !coeff.is_zero() {
                        coeffs.push((wpos, coeff));
                    }
                }
            }
            rows.push(coeffs);
            y.push(v);
        }
        GammaSystem {
            rows: rows.len(),
            row_coeffs: rows,
            y,
        }
    }

    /// The verifier-side build: same rows from the public data.
    #[allow(clippy::too_many_arguments)]
    pub fn build_checked(
        key: &AjtaiKey,
        gammas: &[K4],
        combined_c: &[Poly],
        f: &[K4],
        v: K4,
        level2: &Level2Encoding,
        num_tables: usize,
    ) -> Result<GammaSystem, NodeError> {
        let sys = Self::build(key, gammas, combined_c, f, v, level2, num_tables);
        let w_len = level2.digits.len();
        if sys
            .row_coeffs
            .iter()
            .any(|r| r.iter().any(|(p, _)| *p >= w_len))
        {
            return Err(NodeError::Shape("row support".into()));
        }
        Ok(sys)
    }

    /// The α-weighted row-functional table `h_α` on the W cube.
    pub fn h_alpha_table(&self, alpha: &K4, cube_len: usize) -> Vec<K4> {
        let mut t = vec![K4::ZERO; cube_len];
        for (row, coeffs) in self.row_coeffs.iter().enumerate() {
            let w = alpha.pow(row as u64);
            for (p, c) in coeffs {
                if *p < cube_len {
                    t[*p] = t[*p].add(&c.mul(&w));
                }
            }
        }
        t
    }

    /// `T_α = Σ_row α^row·Y_row` (the range part contributes 0).
    pub fn alpha_weighted_y(&self, alpha: &K4) -> K4 {
        let mut acc = K4::ZERO;
        for (row, yv) in self.y.iter().enumerate() {
            acc = acc.add(&yv.scale(&alpha.pow(row as u64)));
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn e(err: lattice_core::transcript::TranscriptError) -> NodeError {
    NodeError::Transcript(err.to_string())
}

fn absorb_ring(
    transcript: &mut Transcript,
    label: &[u8],
    elems: &[Poly],
) -> Result<(), lattice_core::transcript::TranscriptError> {
    for p in elems {
        transcript.append_message(label, &p.to_le_bytes())?;
    }
    Ok(())
}

fn absorb_k(
    transcript: &mut Transcript,
    label: &[u8],
    v: &K4,
) -> Result<(), lattice_core::transcript::TranscriptError> {
    transcript.append_message(label, &v.to_bytes())
}

fn absorb_k_vec(
    transcript: &mut Transcript,
    label: &[u8],
    v: &[K4],
) -> Result<(), lattice_core::transcript::TranscriptError> {
    for x in v {
        transcript.append_message(label, &x.to_bytes())?;
    }
    Ok(())
}

fn draw_k(transcript: &mut Transcript) -> Result<K4, String> {
    K4::challenge(transcript)
}

fn draw_k_vec(transcript: &mut Transcript, n: usize) -> Result<Vec<K4>, NodeError> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(K4::challenge(transcript).map_err(NodeError::Transcript)?);
    }
    Ok(out)
}

/// Draw the Cauchy challenge `c ←$ C = K \ {poles}`.
fn draw_challenge_off_poles(
    params: &CauchyParams,
    transcript: &mut Transcript,
) -> Result<K4, NodeError> {
    for _ in 0..64 {
        let c = K4::challenge(transcript).map_err(NodeError::Transcript)?;
        if params.poles.iter().all(|xi| *xi != c) {
            return Ok(c);
        }
    }
    Err(NodeError::PoleHit)
}

/// Scale a ring element by a full K4 field element (per-coordinate ring
/// scaling with the K-embedding `ϑ` from Appendix B.1).
fn scale_ring_by_k4(p: &Poly, v: &K4) -> Poly {
    // The paper's B.1 embedding: ϑ_j = X^{d/8} − X^{d−d/8}. For d = 64:
    // ϑ_j = X^{8j} − X^{64−8j}... the exact map matters only for the
    // verifier reproducing the same combination — both sides use this
    // function, so consistency holds; the algebraic role is "a public
    // scalar in R_q derived from the K element".
    // Simpler faithful choice: fold the four K-coordinates into the ring
    // via ϑ powers with the negacyclic twists:
    let mut acc = p.scale(v.0[0].centered());
    let theta1 = Poly::monomial(1, 8).sub(&Poly::monomial(1, 56));
    let theta2 = Poly::monomial(1, 16).sub(&Poly::monomial(1, 48));
    let theta3 = Poly::monomial(1, 24).sub(&Poly::monomial(1, 40));
    acc.add_assign(&p.scale(v.0[1].centered()).mul(&theta1));
    acc.add_assign(&p.scale(v.0[2].centered()).mul(&theta2));
    acc.add_assign(&p.scale(v.0[3].centered()).mul(&theta3));
    acc
}

fn image(mat: &[Vec<K4>], z: &[K4]) -> Vec<K4> {
    let y = mat.len();
    let s = z.len();
    let mut out = Vec::with_capacity(y);
    for j in 0..y {
        let mut acc = K4::ZERO;
        for l in 0..s {
            acc = acc.add(&mat[j][l].mul(&z[l]));
        }
        out.push(acc);
    }
    out
}

/// `eq(p, τ)` for the cube index `p` over `cube_vars` variables.
fn eq_index_at(p: usize, tau: &[K4], cube_vars: usize) -> K4 {
    let mut val = K4::ONE;
    for (bit, &t) in tau.iter().enumerate() {
        let b = (p >> (cube_vars - 1 - bit)) & 1;
        let bf = K4::from_coeffs([b as u64, 0, 0, 0]);
        val = val.mul(&bf.mul(&t).add(&K4::ONE.sub(&bf).mul(&K4::ONE.sub(&t))));
    }
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_roundtrip_scaled_k4() {
        let params = NodeParams::scaled(4);
        let witness = honest_witness(&params, 7);
        let (instance, proof) = prove(&params, &witness, b"s1").expect("prove");
        verify(&params, &instance, &proof).expect("verify");
    }

    #[test]
    fn node_roundtrip_k16() {
        let params = NodeParams::scaled(16);
        let witness = honest_witness(&params, 11);
        let (instance, proof) = prove(&params, &witness, b"s2").expect("prove");
        verify(&params, &instance, &proof).expect("verify");
    }

    #[test]
    fn tampered_residual_rejected() {
        let params = NodeParams::scaled(4);
        let witness = honest_witness(&params, 13);
        let (mut instance, proof) = prove(&params, &witness, b"s3").expect("prove");
        instance.residuals[2][0] = instance.residuals[2][0].add(&K4::ONE);
        assert!(verify(&params, &instance, &proof).is_err());
    }

    #[test]
    fn tampered_output_commitment_rejected() {
        let params = NodeParams::scaled(4);
        let witness = honest_witness(&params, 17);
        let (instance, mut proof) = prove(&params, &witness, b"s4").expect("prove");
        // Corrupt the output commitment: the transcript changes, so every
        // derived challenge shifts — rejected.
        proof.output_commitment[0].0[3] += 1;
        assert!(verify(&params, &instance, &proof).is_err());
    }

    #[test]
    fn tampered_claimed_value_rejected() {
        let params = NodeParams::scaled(4);
        let witness = honest_witness(&params, 19);
        let (instance, mut proof) = prove(&params, &witness, b"s5").expect("prove");
        proof.az[0] = proof.az[0].add(&K4::ONE);
        assert!(verify(&params, &instance, &proof).is_err());
    }

    #[test]
    fn tampered_eval_claim_rejected() {
        let params = NodeParams::scaled(4);
        let witness = honest_witness(&params, 23);
        let (instance, mut proof) = prove(&params, &witness, b"s6").expect("prove");
        proof.eval_claims[3] = proof.eval_claims[3].add(&K4::ONE);
        assert!(verify(&params, &instance, &proof).is_err());
    }

    #[test]
    fn tampered_level2_rejected() {
        let params = NodeParams::scaled(4);
        let witness = honest_witness(&params, 29);
        let (instance, mut proof) = prove(&params, &witness, b"s7").expect("prove");
        // Flip a level-2 digit: the W commitment check or the fingerprint
        // fails.
        proof.level2.digits[10] = 7 - proof.level2.digits[10];
        assert!(verify(&params, &instance, &proof).is_err());
    }
}
