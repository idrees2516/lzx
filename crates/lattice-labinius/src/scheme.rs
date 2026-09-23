//! The scheme: parameters, a prover, a verifier, and the values that pass between them.
//! Port of `labinius` `scheme/`.
//!
//! One round is
//!
//! ```text
//!     (C, opening) = prover.commit(w)
//!     p = (p0, p1)   = verifier.derive_evaluation_point(transcript, C)
//!     t              = w.mle_evaluate(p)          the statement
//!     u              = w.row_evaluate(p)          the prover's message
//!     c              = verifier.derive_folding_challenges(transcript, u)
//!     v              = prover.fold(opening, c)
//! ```
//!
//! and the verifier accepts when `u . eq(p1) = t`, `v` is short, `A v = sum_j c_j C_j` modulo
//! every modulus, and `eq(p0) . (v mod 2) = sum_j u_j (c_j mod 2)` over `F162`. With
//! [`Opening::Recursive`] the opening is replaced by a LaBRADOR proof of the recursive
//! statement (see [`crate::recursion`]).

use crate::bd;
use crate::binfield::{B128, F162};
use crate::challenge::{
    sample_short_challenge, ShortChallenge, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT,
};
use crate::eval;
use crate::key::{random_witness, AuxData, CommitmentKey, CommitmentMatrix};
use crate::params::N;
use crate::ring::{Modulus, PowerOfThreeRing, N162};

/// Why a [`Params`] was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParamError {
    ColumnsExceedWitness,
    TooFewColumns,
    ColumnTooShort,
    DuplicateModulus(Modulus),
    BaseIsAlsoExtra(Modulus),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::ColumnsExceedWitness => write!(f, "column_log_len exceeds witness_log_len"),
            ParamError::TooFewColumns => write!(f, "column_log_len must be at least 1"),
            ParamError::ColumnTooShort => write!(f, "a column must hold at least 128 F162 elements"),
            ParamError::DuplicateModulus(m) => write!(f, "the modulus {m:?} is listed twice"),
            ParamError::BaseIsAlsoExtra(m) => write!(f, "the base modulus {m:?} is listed again as an extra one"),
        }
    }
}
impl std::error::Error for ParamError {}

/// The shape of one round: a witness of `2^witness_log_len` `F162` read as `2^column_log_len`
/// columns, committed modulo `base` and every entry of `extra_moduli`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Params {
    pub witness_log_len: u32,
    pub column_log_len: u32,
    pub base: Modulus,
    pub extra_moduli: Vec<Modulus>,
    pub opening: Opening,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Opening {
    Clear,
    BitDropped { bits: u32 },
    /// Recurse the folded opening into LaBRADOR (see [`crate::recursion`]).
    Recursive,
}

/// A benchmark suite. This port scales the witness to what a scalar (non-AVX-512) prover on a
/// small machine handles: `m` mirrors upstream `sizes` (2^18) and is the default.
pub struct Suite {
    pub name: &'static str,
    pub witness_log_len: u32,
    pub column_log_len_clear: u32,
    pub column_log_len_recursive: u32,
    pub moduli: &'static [Modulus],
    pub moduli_bd: &'static [Modulus],
    pub dropped_bits: u32,
}

pub const SUITES: [Suite; 4] = [
    Suite {
        name: "sizexs",
        witness_log_len: 14,
        column_log_len_clear: 4,
        column_log_len_recursive: 5,
        moduli: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S],
        moduli_bd: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        dropped_bits: 10,
    },
    Suite {
        name: "sizes",
        witness_log_len: 16,
        column_log_len_clear: 6,
        column_log_len_recursive: 7,
        moduli: &[Modulus::Q3889_FS_S, Modulus::Q9721_FS_S],
        moduli_bd: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        dropped_bits: 9,
    },
    Suite {
        name: "sizem",
        witness_log_len: 18,
        column_log_len_clear: 7,
        column_log_len_recursive: 8,
        moduli: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S],
        moduli_bd: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        dropped_bits: 9,
    },
    Suite {
        name: "sizel",
        witness_log_len: 20,
        column_log_len_clear: 8,
        column_log_len_recursive: 9,
        moduli: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S],
        moduli_bd: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        dropped_bits: 8,
    },
];

impl Suite {
    pub fn from_flag(flag: &str) -> Option<&'static Suite> {
        SUITES.iter().find(|r| r.name == format!("size{flag}"))
    }
}

/// The default shape: 2^18 F162 in 128 columns, moduli 3889 + 2917, clear opening.
pub fn basic() -> Params {
    Params::new(18, 7, vec![Modulus::Q2917_Q_S], Opening::Clear).expect("basic parameters are valid")
}

impl Params {
    pub fn new(
        witness_log_len: u32,
        column_log_len: u32,
        extra_moduli: Vec<Modulus>,
        opening: Opening,
    ) -> Result<Params, ParamError> {
        Params::with_base(witness_log_len, column_log_len, Modulus::BASE, extra_moduli, opening)
    }

    pub fn with_base(
        witness_log_len: u32,
        column_log_len: u32,
        base: Modulus,
        extra_moduli: Vec<Modulus>,
        opening: Opening,
    ) -> Result<Params, ParamError> {
        if column_log_len > witness_log_len {
            return Err(ParamError::ColumnsExceedWitness);
        }
        if column_log_len < 1 {
            return Err(ParamError::TooFewColumns);
        }
        if witness_log_len - column_log_len < 7 {
            return Err(ParamError::ColumnTooShort);
        }
        for (i, m) in extra_moduli.iter().enumerate() {
            if *m == base {
                return Err(ParamError::BaseIsAlsoExtra(*m));
            }
            if extra_moduli[..i].contains(m) {
                return Err(ParamError::DuplicateModulus(*m));
            }
        }
        Ok(Params {
            witness_log_len,
            column_log_len,
            base,
            extra_moduli,
            opening,
        })
    }

    pub fn sized(suite: &Suite, opening: Opening) -> Params {
        let column_log_len = match opening {
            Opening::Recursive => suite.column_log_len_recursive,
            _ => suite.column_log_len_clear,
        };
        let list = match opening {
            Opening::BitDropped { .. } => suite.moduli_bd,
            _ => suite.moduli,
        };
        Params::with_base(suite.witness_log_len, column_log_len, list[0], list[1..].to_vec(), opening)
            .expect("the sized parameters are valid")
    }

    pub fn recursion(&self) -> bool {
        self.opening == Opening::Recursive
    }

    pub fn dropped_bits(&self) -> u32 {
        match self.opening {
            Opening::BitDropped { bits } => bits,
            _ => 0,
        }
    }

    pub fn witness_len(&self) -> usize {
        1usize << self.witness_log_len
    }

    /// The cap on `|v|^2` at this shape: `FOLD_CAP * (witness_len / 4) * N` (per ring element
    /// and challenge, times the number of coefficients).
    pub fn fold_cap(&self) -> u64 {
        (crate::recursion::FOLD_CAP * (self.witness_len() / 4 * N) as f64).ceil() as u64
    }

    pub fn columns(&self) -> usize {
        1usize << self.column_log_len
    }

    pub fn row_log_len(&self) -> u32 {
        self.witness_log_len - self.column_log_len
    }

    pub fn column_len(&self) -> usize {
        1usize << self.row_log_len()
    }

    pub fn primes(&self) -> Vec<u16> {
        core::iter::once(self.base.prime())
            .chain(self.extra_moduli.iter().map(|m| m.prime()))
            .collect()
    }

    pub fn bd_cap(&self) -> u64 {
        bd::cap(self.columns(), self.dropped_bits(), DEFAULT_WEIGHT)
    }
}

/// The private input: `witness_len()` elements of `F162`, read as `witness_len()/4` binary ring
/// elements of `R_648` four `F162` at a time.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Witness {
    pub params: Params,
    pub elements: Vec<F162>,
}

impl Witness {
    /// The trace lifted into `F162` bit for bit.
    pub fn lifted(params: &Params, trace: &[B128]) -> Result<Witness, ()> {
        if trace.len() != params.witness_len() {
            return Err(());
        }
        Ok(Witness {
            params: params.clone(),
            elements: trace.iter().map(|&x| F162::from_b128(x)).collect(),
        })
    }

    pub fn from_elements(params: &Params, elements: Vec<F162>) -> Result<Witness, ()> {
        if elements.len() != params.witness_len() {
            return Err(());
        }
        Ok(Witness {
            params: params.clone(),
            elements,
        })
    }

    /// A uniform witness from an XOF seed (three LE u64 per element, top 30 bits cleared).
    pub fn random(params: &Params, seed: [u8; 32]) -> Witness {
        let _ = &seed;
        Witness {
            params: params.clone(),
            elements: random_witness(params.witness_len(), seed_u64(&seed)),
        }
    }

    pub fn elements(&self) -> &[F162] {
        &self.elements
    }

    /// The multilinear extension of the witness at the point — the statement being proved.
    pub fn mle_evaluate(&self, point: &EvaluationPoint) -> F162 {
        self.check(point);
        eval::claim(&eval::row_evaluate(&self.elements, &point.p0), &point.p1)
    }

    /// `u = B W`, one field element per column: the prover's message.
    pub fn row_evaluate(&self, point: &EvaluationPoint) -> RowEvaluation {
        self.check(point);
        RowEvaluation {
            values: eval::row_evaluate(&self.elements, &point.p0),
        }
    }

    fn check(&self, point: &EvaluationPoint) {
        assert_eq!(
            (point.p0.len(), point.p1.len()),
            (
                self.params.row_log_len() as usize,
                self.params.column_log_len as usize
            ),
            "the evaluation point does not match the witness"
        );
    }
}

fn seed_u64(seed: &[u8; 32]) -> u64 {
    u64::from_le_bytes(seed[..8].try_into().unwrap())
}

/// A point of `F162^nu` split the way the witness is: `p0` over the row variables, `p1` over
/// the column variables.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EvaluationPoint {
    pub p0: Vec<F162>,
    pub p1: Vec<F162>,
}

impl EvaluationPoint {
    pub fn of(p0: Vec<F162>, p1: Vec<F162>) -> Self {
        EvaluationPoint { p0, p1 }
    }

    pub fn p0(&self) -> &[F162] {
        &self.p0
    }
    pub fn p1(&self) -> &[F162] {
        &self.p1
    }
}

/// `u_j = sum_i eq(p0, i) W[i, j]`, one field element per column.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RowEvaluation {
    pub values: Vec<F162>,
}

impl RowEvaluation {
    pub fn values(&self) -> &[F162] {
        &self.values
    }
    pub fn values_mut(&mut self) -> &mut [F162] {
        &mut self.values
    }
}

/// The `columns()` short challenges of one round.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FoldingChallenges {
    pub challenges: Vec<ShortChallenge>,
}

impl FoldingChallenges {
    /// Absorb the source (`u`), then derive the challenges: weight 28, canonical bound 12.
    pub fn derive(params: &Params, transcript: &mut Transcript, row: &RowEvaluation) -> Self {
        transcript.absorb_bytes(b"labinius/row-evaluation");
        let mut bytes = Vec::with_capacity(24 * row.values.len());
        for x in &row.values {
            bytes.extend_from_slice(&x.to_le24());
        }
        transcript.absorb_bytes(&bytes);
        FoldingChallenges {
            challenges: (0..params.columns())
                .map(|_| sample_short_challenge(transcript, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
                .collect(),
        }
    }

    pub fn challenges(&self) -> &[ShortChallenge] {
        &self.challenges
    }
    pub fn len(&self) -> usize {
        self.challenges.len()
    }
    pub fn is_empty(&self) -> bool {
        self.challenges.is_empty()
    }
}

/// The amortised witness `v = sum_j c_j W_j`: one column's worth of `R_648` elements in
/// coefficient form, centered, and genuinely small.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FoldedWitness {
    pub elements: Vec<[i16; N]>,
}

impl FoldedWitness {
    pub fn elements(&self) -> &[[i16; N]] {
        &self.elements
    }
    pub fn elements_mut(&mut self) -> &mut [[i16; N]] {
        &mut self.elements
    }
    /// The `4 * len` parities, as F162 elements (the binary shadow of the fold).
    pub fn mod_2(&self) -> Vec<F162> {
        crate::fold::components_mod_2(&self.elements)
    }
    /// Exact squared l2 norm over all coefficients.
    pub fn normsq(&self) -> u64 {
        self.elements
            .iter()
            .flat_map(|e| e.iter())
            .map(|&x| (x as i64 * x as i64) as u64)
            .sum()
    }
}

/// `sum_j c_j C_j`: four `R_162` elements per modulus.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FoldedCommitment {
    pub primes: Vec<u16>,
    pub rows: [Vec<PowerOfThreeRing>; 4],
}

impl FoldedCommitment {
    pub fn element(&self, row: usize, modulus_index: usize) -> &PowerOfThreeRing {
        &self.rows[row][modulus_index]
    }
    pub fn moduli(&self) -> &[u16] {
        &self.primes
    }
}

/// The commitment in its two clear forms: the matrix, or bit-dropped.
#[derive(Clone)]
pub enum CommitmentValue {
    Matrix(CommitmentMatrix),
    Dropped(bd::Dropped),
}

#[derive(Clone)]
pub struct Commitment {
    pub primes: Vec<u16>,
    pub columns: usize,
    pub value: CommitmentValue,
}

impl Commitment {
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn moduli(&self) -> &[u16] {
        &self.primes
    }
    pub fn matrix(&self) -> &CommitmentMatrix {
        match &self.value {
            CommitmentValue::Matrix(m) => m,
            _ => panic!("this commitment is not the matrix"),
        }
    }
    pub fn dropped(&self) -> &bd::Dropped {
        match &self.value {
            CommitmentValue::Dropped(d) => d,
            _ => panic!("this commitment is not bit-dropped"),
        }
    }
    /// Wire bytes: `ceil(log2 q)`-bit slots per matrix element (dropped: its own packing).
    pub fn wire_bytes(&self) -> usize {
        match &self.value {
            CommitmentValue::Matrix(m) => {
                let bits: u32 = self.primes.iter().map(|&q| bd::residue_bits(q)).sum();
                4 * m.cols() * N162 * bits as usize / 8
            }
            CommitmentValue::Dropped(d) => d.wire_bytes(),
        }
    }
}

/// What the prover keeps from a commitment and the fold consumes.
#[derive(Clone)]
pub struct CommitmentOpening {
    pub(crate) aux: AuxData,
}

/// The prover.
pub struct Prover {
    params: Params,
    key: CommitmentKey,
}

/// The verifier: the public parameters and nothing else.
pub struct Verifier {
    params: Params,
    key: CommitmentKey,
    matrix_seed: [u8; 32],
}

/// [`Params`] together with the public matrix `A` expanded from a seed.
pub struct PublicParameters {
    pub(crate) params: Params,
    pub(crate) matrix_seed: [u8; 32],
    pub(crate) key: CommitmentKey,
}

impl PublicParameters {
    /// Expand `A` from the seed (deterministic; upstream hashes the seed with blake3 first —
    /// this port derives the 64-bit key seed with SHA3-256 instead).
    pub fn from_seed(params: Params, matrix_seed: [u8; 32]) -> PublicParameters {
        let digest = lattice_core::keccak::sha3_256(&matrix_seed);
        let seed = u64::from_le_bytes(digest[..8].try_into().unwrap());
        let key = CommitmentKey::random(params.column_len(), seed, params.base, &params.extra_moduli);
        PublicParameters {
            params,
            matrix_seed,
            key,
        }
    }

    pub fn params(&self) -> &Params {
        &self.params
    }
    pub fn key(&self) -> &CommitmentKey {
        &self.key
    }
}

impl Prover {
    pub fn new(pp: &PublicParameters) -> Prover {
        Prover {
            params: pp.params.clone(),
            key: pp.key.clone(),
        }
    }

    pub fn commit(&self, witness: &Witness) -> (Commitment, CommitmentOpening) {
        assert_eq!(witness.params, self.params, "the witness was built for other parameters");
        let (matrix, aux) = self.key.commit(&witness.elements, self.params.columns());
        let primes = self.params.primes();
        let value = match self.params.dropped_bits() {
            0 => CommitmentValue::Matrix(matrix),
            bits => CommitmentValue::Dropped(bd::drop_bits(&matrix, &primes, bits)),
        };
        (
            Commitment {
                primes,
                columns: self.params.columns(),
                value,
            },
            CommitmentOpening { aux },
        )
    }

    /// `v = sum_j c_j W_j`, in coefficient form modulo the base prime, centered.
    pub fn fold(&self, opening: CommitmentOpening, challenges: &FoldingChallenges) -> FoldedWitness {
        let elements = crate::fold::fold_witness(&opening.aux, &challenges.challenges, self.key.prime(0));
        FoldedWitness { elements }
    }
}

impl Verifier {
    pub fn new(pp: &PublicParameters) -> Verifier {
        Verifier {
            params: pp.params.clone(),
            key: pp.key.clone(),
            matrix_seed: pp.matrix_seed,
        }
    }

    /// The shape, the moduli and the key seed, absorbed before anything the prover chooses.
    fn absorb_parameters(&self, transcript: &mut Transcript) {
        transcript.absorb_bytes(b"labinius/parameters");
        transcript.absorb_u64(self.params.witness_log_len as u64);
        transcript.absorb_u64(self.params.column_log_len as u64);
        for q in self.params.primes() {
            transcript.absorb_u64(q as u64);
        }
        transcript.absorb_u64(u64::from(self.params.recursion()));
        if self.params.dropped_bits() > 0 {
            transcript.absorb_bytes(b"labinius/dropped-bits");
            transcript.absorb_u64(self.params.dropped_bits() as u64);
        }
        transcript.absorb_bytes(&self.matrix_seed);
    }

    /// Absorb the commitment, then derive `p = (p0, p1)`: one uniform `F162` per variable.
    pub fn derive_evaluation_point(
        &self,
        transcript: &mut Transcript,
        commitment: &Commitment,
    ) -> EvaluationPoint {
        self.absorb_parameters(transcript);
        transcript.absorb_bytes(b"labinius/commitment");
        transcript.absorb_u64(commitment.columns() as u64);
        match &commitment.value {
            CommitmentValue::Matrix(m) => {
                for j in 0..commitment.columns() {
                    for row in 0..4 {
                        for k in 0..self.key.limbs() {
                            transcript.absorb_bytes(b"elem");
                            transcript.absorb_u64(row as u64);
                            transcript.absorb_u64(j as u64);
                            transcript.absorb_u64(k as u64);
                            transcript.absorb_elements(std::slice::from_ref(m.element(row, j, k)));
                        }
                    }
                }
            }
            CommitmentValue::Dropped(d) => {
                transcript.absorb_bytes(b"labinius/dropped-commitment");
                transcript.absorb_bytes(&(d.top.len() as u64).to_le_bytes());
                for &t in &d.top {
                    transcript.absorb_u64(t as u64);
                }
                for dig in &d.digits {
                    for &x in dig {
                        transcript.absorb_u64(x as u64);
                    }
                }
            }
        }
        let (rows, cols) = (
            self.params.row_log_len() as usize,
            self.params.column_log_len as usize,
        );
        let mut bytes = vec![0u8; 24 * (rows + cols)];
        transcript.fill(b"labinius/evaluation-point", &mut bytes);
        let element = |n: usize| F162::from_le24(&bytes[24 * n..24 * n + 24]);
        EvaluationPoint {
            p0: (0..rows).map(element).collect(),
            p1: (0..cols).map(|k| element(rows + k)).collect(),
        }
    }

    pub fn derive_folding_challenges(
        &self,
        transcript: &mut Transcript,
        row: &RowEvaluation,
    ) -> FoldingChallenges {
        FoldingChallenges::derive(&self.params, transcript, row)
    }

    /// `sum_j c_j C_j` per modulus.
    pub fn fold_commitment(
        &self,
        commitment: &Commitment,
        challenges: &FoldingChallenges,
    ) -> FoldedCommitment {
        assert_eq!(challenges.len(), commitment.columns(), "one challenge per column");
        let matrix = commitment.matrix();
        let mut rows: [Vec<PowerOfThreeRing>; 4] = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        for k in 0..self.key.limbs() {
            let q = self.key.prime(k);
            let columns: Vec<Vec<PowerOfThreeRing>> = (0..matrix.cols())
                .map(|j| (0..4).map(|row| *matrix.element(row, j, k)).collect())
                .collect();
            let folded = crate::fold::fold_commitment(q, &challenges.challenges, &columns);
            for row in 0..4 {
                rows[row].push(folded[row]);
            }
        }
        FoldedCommitment {
            primes: self.params.primes(),
            rows,
        }
    }

    /// `u' = sum_j u_j (c_j mod 2)` over `F162`.
    pub fn fold_row_evaluation(
        &self,
        row_evaluation: &RowEvaluation,
        challenges: &FoldingChallenges,
    ) -> F162 {
        eval::fold_binary(&row_evaluation.values, &challenges.challenges)
    }

    /// The claim check: `u . eq(p1) == t`.
    pub fn verify_evaluation(
        &self,
        point: &EvaluationPoint,
        claimed_value: &F162,
        row_evaluation: &RowEvaluation,
    ) -> Result<(), VerificationError> {
        if row_evaluation.values.len() != self.params.columns()
            || point.p1.len() != self.params.column_log_len as usize
        {
            return Err(VerificationError::Rejected);
        }
        if eval::claim(&row_evaluation.values, &point.p1) == *claimed_value {
            Ok(())
        } else {
            Err(VerificationError::Rejected)
        }
    }

    /// The opening check, recomputed from `v` alone: `v` is centered modulo the base modulus,
    /// `A v` equals the folded commitment on every modulus, and `eq(p0) . (v mod 2) == u'`.
    pub fn verify_opening(
        &self,
        folded_commitment: &FoldedCommitment,
        folded_witness: &FoldedWitness,
        point: &EvaluationPoint,
        folded_row_value: &F162,
    ) -> Result<(), VerificationError> {
        let v = &folded_witness.elements;
        let half = ((self.key.prime(0) - 1) / 2) as i32;
        if v.len() != self.key.len_ring() || folded_commitment.primes != self.params.primes() {
            return Err(VerificationError::Rejected);
        }
        let mut normsq = 0u64;
        let mut worst = 0i32;
        for e in v {
            for &x in e.iter() {
                normsq += (x as i64 * x as i64) as u64;
                worst = worst.max(x as i32);
            }
        }
        if worst > half || normsq > self.params.fold_cap() {
            return Err(VerificationError::Rejected);
        }
        for k in 0..self.key.limbs() {
            let q = self.key.prime(k);
            let comps = crate::fold::a_times_v_components(q, &self.key.a[k], v);
            for (row, c) in comps.iter().enumerate() {
                if *c != folded_commitment.rows[row][k] {
                    return Err(VerificationError::Rejected);
                }
            }
        }
        if eval::binary_check(point.p0(), &folded_witness.mod_2(), *folded_row_value) {
            Ok(())
        } else {
            Err(VerificationError::Rejected)
        }
    }

    /// The bit-dropped opening check.
    pub fn verify_opening_bd(
        &self,
        commitment: &Commitment,
        challenges: &FoldingChallenges,
        folded_witness: &FoldedWitness,
        point: &EvaluationPoint,
        folded_row_value: &F162,
    ) -> Result<(), VerificationError> {
        let v = &folded_witness.elements;
        let half = ((self.key.prime(0) - 1) / 2) as i32;
        if v.len() != self.key.len_ring() || commitment.moduli() != self.params.primes() {
            return Err(VerificationError::Rejected);
        }
        let dropped = match &commitment.value {
            CommitmentValue::Dropped(d) => d,
            _ => return Err(VerificationError::Rejected),
        };
        if dropped.dropped_bits != self.params.dropped_bits() {
            return Err(VerificationError::Rejected);
        }
        let mut normsq = 0u64;
        let mut worst = 0i32;
        for e in v {
            for &x in e.iter() {
                normsq += (x as i64 * x as i64) as u64;
                worst = worst.max(x as i32);
            }
        }
        if worst > half || normsq > self.params.fold_cap() {
            return Err(VerificationError::Rejected);
        }
        let residual = bd::residual(&self.key, dropped, &challenges.challenges, v)
            .ok_or(VerificationError::Rejected)?;
        if residual > self.params.bd_cap() as u128 {
            return Err(VerificationError::Rejected);
        }
        if eval::binary_check(point.p0(), &folded_witness.mod_2(), *folded_row_value) {
            Ok(())
        } else {
            Err(VerificationError::Rejected)
        }
    }
}

/// The verifier rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VerificationError {
    Rejected,
}

impl std::fmt::Display for VerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the opening was rejected")
    }
}
impl std::error::Error for VerificationError {}
