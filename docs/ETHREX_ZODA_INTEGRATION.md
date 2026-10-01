# ETHREX × ZODA × lzx — Integration Wire Formats & Adapter Design

**Task:** research document for building zero-dependency (pure-std) Rust adapters in the `lzx`
workspace that (a) package zkVM-proved CLOB transaction batches into ethrex L2 blocks,
(b) prepare proof-submission payloads for ethrex's L1 verifier path, and (c) post data blobs
to zoda for data availability.

**Sources analyzed (read-only):**
- `/home/z/my-project/repos/ethrex` — LambdaClass' Ethereum execution client + L2 rollup stack (clone).
- `/home/z/my-project/repos/zoda` — ZODA tensor-code DA protocol implementation (clone; note: this is
  the `idrees2516/zoda` repo — same owner as `lzx`, zero-dependency pure-Rust, 159 tests).
- `/home/z/my-project/lzx` — the target workspace (lattice-based zkVM proving RV64IMAC executions).

All citations are `file:line` relative to the repo roots above. Everything below was read directly
from the clones; nothing was executed. No code was changed by this task.

---

## 1. The ethrex L2 stack: crates, block format, L1 submission

### 1.1 Where the L2 lives

The L2 is **not a separate repo**; it is the `crates/l2` subtree plus shared types in `crates/common/types/l2/`:

| crate | role |
|---|---|
| `crates/l2/sequencer` | the L2 node: block producer, L1 committer, proof coordinator, proof sender/verifier (`crates/l2/sequencer/mod.rs`) |
| `crates/l2/common` | shared L2 types: messages, privileged txs, prover input, calldata `Value` model, merkle tree (`crates/l2/common/src/lib.rs`) |
| `crates/l2/contracts` | Solidity: `OnChainProposer`, `CommonBridge`, `Timelock`, `Router`, based variants (`crates/l2/contracts/src/l1/OnChainProposer.sol`) |
| `crates/l2/sdk` | Rust SDK to talk to L1/L2: `build_generic_tx`, calldata encode/decode (`crates/l2/sdk/src/sdk.rs`, 1826 lines) |
| `crates/l2/prover` | thin launcher that pulls work from the proof coordinator (`crates/l2/prover/src/prover.rs:9-23`) |
| `crates/prover` | backend-agnostic prover: Exec / SP1 / RISC0 / Zisk / OpenVM (`crates/prover/src/backend/mod.rs:48-60`) |
| `crates/guest-program` | the zkVM guest that validates a batch (L2 stateless validator) + its `ProgramOutput` (`crates/guest-program/src/l2/output.rs:6-28`) |
| `crates/l2/storage` | rollup store: batches, proofs, fee configs (`crates/l2/storage/src/store.rs`) |
| `crates/l2/networking/rpc` | L2 JSON-RPC surface incl. `ethrex_*` methods (`crates/l2/networking/rpc/rpc.rs:407-424`) |
| `crates/l2/based`, `crates/l2/tee` | based (L1-synchronized) sequencing + TDX quote tooling |

### 1.2 The key architectural fact: there is no separate "L2 block" type

ethrex L2 reuses **the full L1 `Block` type** (`crates/common/types/block.rs:41-44`):
an L2 block = `Block { header: BlockHeader, body: BlockBody }`. Batches of these blocks are
rolled up to L1. The unit committed on L1 is a **batch** (`crates/common/types/l2/batch.rs:7-21`):

```rust
pub struct Batch {
    pub number: u64,                       // batch number, 1-indexed
    pub first_block: u64,
    pub last_block: u64,
    pub state_root: H256,                  // state root after last block of batch
    pub l1_in_messages_rolling_hash: H256, // deposits processed (see §5)
    pub l2_in_message_rolling_hashes: Vec<(u64, H256)>,
    pub l1_out_message_hashes: Vec<H256>,  // withdrawals (see §4)
    pub non_privileged_transactions: u64,
    pub balance_diffs: Vec<BalanceDiff>,
    pub blobs_bundle: BlobsBundle,         // the blob(s) posted to L1 for DA
    pub commit_tx: Option<H256>,
    pub verify_tx: Option<H256>,
}
```

### 1.3 L2 block RLP structure (exact field order)

`Block` RLP = list of 4 fields (`crates/common/types/block.rs:56-64`):

```
rlp([ header, transactions, ommers, withdrawals? ])
```
- `transactions`: list of canonical tx encodings (`tx_type_byte || rlp(inner)` for typed txs,
  bare `rlp(inner)` for legacy — `crates/common/types/transaction.rs:2753-2774`).
- `ommers`: always empty list on L2.
- `withdrawals`: **optional** field (omitted entirely when `None`; `encode_optional_field`).
  L2 bodies typically set `withdrawals: Some(vec![])` post-Shanghai (`block.rs:330-337`).

`BlockHeader` RLP = 15 mandatory + 10 optional fields **in this exact order**
(`crates/common/types/block.rs:225-253`):

```
parent_hash            H256      (32B)
ommers_hash            H256      (keccak(rlp([])) = 0x1dcc4de8... for empty ommers)
coinbase               Address   (20B) — L2 coinbase/fee vault
state_root             H256
transactions_root      H256      (trie over canonical tx encodings, keyed by rlp(index))
receipts_root          H256
logs_bloom             Bloom     (256B)
difficulty             U256      (0 on L2)
number                 u64
gas_limit              u64
gas_used               u64
timestamp              u64
extra_data             Bytes
prev_randao            H256      (0 on L2 PoS-less chains)
nonce                  u64       (encoded as 8 raw BE bytes, block.rs:242)
--- optional, each present only if Some: ---
base_fee_per_gas       u64
withdrawals_root       H256
blob_gas_used          u64
excess_blob_gas        u64
parent_beacon_block_root H256
requests_hash          H256
block_access_list_hash H256      (Amsterdam, EIP-7928)
slot_number            u64
burned_fees            u64       (LStar, EIP-8079)
```

`block_hash = keccak256(rlp(header))` (`block.rs:436-441`).

### 1.4 How a batch is submitted to L1 (the committer)

`crates/l2/sequencer/l1_committer.rs` is the whole commit pipeline:

1. **Batch production** (`produce_batch`, `l1_committer.rs:684-795`): pulls consecutive L2 blocks
   from the store until a batch gas limit / privileged-tx budget (`PRIVILEGED_TX_BUDGET = 300`,
   `crates/l2/common/src/privileged_transactions.rs:7`) / blob-size limit is hit; accumulates
   L1/L2 in/out messages and the new state root.
2. **Blob construction** (`generate_blobs_bundle`, `l1_committer.rs:1481-1519`) — the exact blob
   payload layout:

   ```
   blob_data = [u64 BE block_count]
             || rlp(Block_1) || rlp(Block_2) || ... || rlp(Block_n)     (concatenated)
             || fee_config_1 || fee_config_2 || ... || fee_config_n     (FeeConfig::to_vec)
   ```
   `FeeConfig::to_vec` (`crates/common/types/l2/fee_config.rs:89-121`) is a custom (non-RLP)
   encoding: `[version:u8=0][type_bitmask:u8][optional vault addresses (20B) +
   fee values (u64 BE) in bitmask order]`. Type bits: base-fee-vault=1, operator-fee=2, l1-fee=4.

   The blob is zero-padded to 131,072 bytes by `blob_from_bytes`
   (`crates/common/types/blobs_bundle.rs:43-52`; safety cap `SAFE_BYTES_PER_BLOB = 126,976`
   = 131072·31/32, `crates/common/types/constants.rs:24-28`), then KZG-committed:
   `BlobsBundle::create_from_blobs` produces `commitments[48B]`, `proofs[48B]`
   (`blobs_bundle.rs:94-123`); `blobKZGVersionedHash = 0x01 || sha256(commitment)[1..32]`
   (`blobs_bundle.rs:76-80`).
3. **Commit transaction** (`send_commitment`, `l1_committer.rs:1297-1472`): an **EIP-4844 blob
   transaction** (rollup mode; EIP-1559 in validium mode, `l1_committer.rs:1399-1448`) sent to
   the `OnChainProposer` via the `Timelock` (or directly in based mode, `l1_committer.rs:1390-1397`)
   with calldata `commitBatch(...)` (§1.5).

### 1.5 The `commitBatch` L1 call (both signature variants)

`crates/l2/sequencer/l1_committer.rs:74-76`:

```rust
const COMMIT_FUNCTION_SIGNATURE_BASED: &str =
    "commitBatch(uint256,bytes32,bytes32,bytes32,bytes32,uint256,bytes32,bytes[])";
const COMMIT_FUNCTION_SIGNATURE: &str =
    "commitBatch(uint256,bytes32,bytes32,bytes32,bytes32,uint256,bytes32,(uint256,uint256,(address,address,address,uint256)[],bytes32[])[],(uint256,bytes32)[])";
```

Solidity (`crates/l2/contracts/src/l1/OnChainProposer.sol:226-236`):

```solidity
function commitBatch(
    uint256 batchNumber,
    bytes32 newStateRoot,
    bytes32 withdrawalsLogsMerkleRoot,
    bytes32 processedPrivilegedTransactionsRollingHash,
    bytes32 lastBlockHash,
    uint256 nonPrivilegedTransactions,
    bytes32 commitHash,                                   // keccak(git commit string)
    ICommonBridge.BalanceDiff[] calldata balanceDiffs,    // (chainId, value, AssetDiff[], hashes[])
    ICommonBridge.L2MessageRollingHash[] calldata l2MessageRollingHashes // (chainId, rollingHash)
) external onlyOwner whenNotPaused
```

Semantics (OnChainProposer.sol:237-308): batch number must equal `lastCommittedBatch + 1`;
the batch data itself rides in the **blob of the EIP-4844 tx** (`blobhash(0)` is read at
OnChainProposer.sol:273 and stored as `blobKZGVersionedHash`; rollup mode reverts if absent,
validium reverts if present). Based mode instead passes the RLP blocks **in calldata** as
`bytes[]` (`l1_committer.rs:1311-1330`).

Calldata encoding rules (needed for byte-exact adapters): 4-byte selector =
`keccak256("commitBatch(...)")[0..4]` (`crates/l2/sdk/src/calldata.rs:63-102`), then standard
ABI tuple encoding of the arguments (`encode_calldata`, `calldata.rs:79-102`). Rust `Value`
model: `Value::Uint` (32-byte ABI uint), `Value::FixedBytes` (right-padded to 32B),
`Value::Address`, `Value::Bytes` (dynamic, offset+length+data), `Value::Array`, `Value::Tuple`
(`crates/l2/common/src/calldata.rs`).

---

## 2. The ethrex L2 transaction format

`crates/common/types/transaction.rs:66-77` — `Transaction` is an enum: Legacy, EIP-2930,
EIP-1559, EIP-4844 (not allowed on L2: `rpc.rs:380-390`), EIP-7702, **PrivilegedL2Transaction**,
FeeToken, Frame.

### 2.1 Regular transactions — RLP field order

`EIP1559Transaction` (`transaction.rs:268-293`, encode at `670-687`) — the natural carrier for
CLOB order transactions:

```
rlp([ chain_id, nonce, max_priority_fee_per_gas, max_fee_per_gas, gas_limit,
      to (20B address | 0x80 for create), value, data, access_list,
      signature_y_parity, signature_r, signature_s ])
```
(`LegacyTransaction` order: `nonce, gas_price, gas, to, value, data, v, r, s` — `transaction.rs:636-650`;
EIP-2930 at `652-668`; EIP-7702 at `710-728`.)

Canonical envelope for typed txs = `0x02 || rlp(inner)` (`encode_canonical`,
`transaction.rs:2753-2774`); inside a block body, `Transaction::encode` wraps the canonical
bytes in an RLP byte-string (`transaction.rs:548-561`).

`TxKind::Call(addr)` encodes as the 20-byte address; `TxKind::Create` as RLP null `0x80`
(`transaction.rs:617-624`).

### 2.2 Privileged (deposit-type) transactions — `0x7e`

Struct (`transaction.rs:355-377`), RLP encode (`transaction.rs:730-745`):

```
type byte: 0x7e (TxType::Privileged, transaction.rs:379-392 — Optimism-style deposit prefix)
rlp([ chain_id, nonce, max_priority_fee_per_gas, max_fee_per_gas, gas_limit,
      to, value, data, access_list, from ])
```

**No signature fields.** `from: Address` is carried explicitly (`transaction.rs:369-370`) —
the L1 watcher mints these directly from `PrivilegedTxSent` events, using the L1
`transactionId` as the L2 nonce (`crates/l2/sdk/src/privileged_data.rs:86-121`). The
`data: Bytes` field carries arbitrary calldata (deposit payloads) exactly like a normal tx.
Privileged txs are never gossiped over P2P (`transaction.rs:144-149`).

### 2.3 How a client submits a tx to the L2

Standard JSON-RPC `eth_sendRawTransaction` to the L2 node
(`crates/l2/networking/rpc/rpc.rs:380-394`) or sponsored `ethrex_sendTransaction`
(`rpc.rs:409`). EIP-4844 is rejected on L2.

---

## 3. The proof path: prover backends, coordinator protocol, L1 verification

### 3.1 Backends

`crates/prover/src/backend/mod.rs:48-60`: `BackendType = Exec | SP1 | RISC0 | ZisK | OpenVM`
(feature-gated). Unified trait `ProverBackend` (`backend/mod.rs:87-153`):
`serialize_input / execute / prove(input, ProofFormat) / verify / to_proof_bytes`.
`ProofFormat = Groth16 | Compressed` (`crates/common/types/prover.rs:102-110`).
`ProverType = Exec | RISC0 | SP1 | TDX` (`prover.rs:6-12`) — note TDX proofs are produced by
the TEE quote path, not the `crates/prover` backends.

### 3.2 What gets proven, and the coordinator wire protocol

Input to the guest (`crates/l2/common/src/prover.rs:26-37`):

```rust
pub struct ProverInputData {
    pub blocks: Vec<Block>,                  // the L2 blocks of the batch
    pub execution_witness: ExecutionWitness, // stateless witness (trie nodes, codes)
    pub elasticity_multiplier: u64,
    pub blob_commitment: [u8; 48],           // KZG commitment of the batch blob
    pub blob_proof: [u8; 48],
    pub fee_configs: Vec<FeeConfig>,
}
```

The guest program outputs `ProgramOutput` (`crates/guest-program/src/l2/output.rs:6-28`) whose
`encode()` (`output.rs:31-71`) is **the public-values byte layout** the L1 contract reconstructs:

```
bytes 0..32    initial_state_hash        (state root before batch)
bytes 32..64   final_state_hash          (state root after batch)
bytes 64..96   l1_out_messages_merkle_root (withdrawalsLogsMerkleRoot)
bytes 96..128  l1_in_messages_rolling_hash
bytes 128..160 blob_versioned_hash
bytes 160..192 last_block_hash
bytes 192..224 chain_id (U256 BE)
bytes 224..256 non_privileged_count (U256 BE)
--- variable part ---
per BalanceDiff:  chain_id(32) value(32) [tokenL1(20) tokenL2(20) destTokenL2(20) value(32)]* message_hashes(32)*
per (chain_id, rolling_hash): chain_id(32 BE) rolling_hash(32)
```

This mirrors `OnChainProposer._getPublicInputsFromCommitment` **exactly**
(`OnChainProposer.sol:563-652`; the fixed part is `abi.encodePacked` of 8 values,
OnChainProposer.sol:596-605). That function is the single source of truth for the
"public inputs" a proof must attest to.

**Coordinator protocol** (provers pull work over TCP; one JSON message per connection —
`read_to_end` then `serde_json::from_slice`, `crates/l2/sequencer/proof_coordinator.rs:405-415`):
`ProofData<I>` enum (`crates/common/types/prover.rs:118-159`):
`ProverSetup{prover_type, payload}` / `ProverSetupACK` /
`InputRequest{commit_hash, prover_type}` / `VersionMismatch` / `ProverTypeNotNeeded` /
`InputResponse{id, input, format}` / `ProofSubmit{id, proof: ProverOutput}` / `ProofSubmitACK{id}`.
Server side: `crates/l2/sequencer/proof_coordinator.rs:109` (`TcpListener::bind`), dispatch at
`proof_coordinator.rs:417-441`. `ProverOutput = Proof(ProofBytes) |
ProofWithPublicValues{proof_bytes, public_values}` (`prover.rs:63-70`) — the second variant is
the Aligned mode payload.

### 3.3 On-chain verification — exact interfaces

`OnChainProposer.verifyBatches` (`OnChainProposer.sol:415-434`; interface
`IOnChainProposer.sol:136-141`):

```solidity
function verifyBatches(
    uint256 firstBatchNumber,      // must be lastVerifiedBatch + 1
    bytes[] calldata risc0BlockProofs,   // one per batch (empty bytes if not required)
    bytes[] calldata sp1ProofsBytes,     // one per batch
    bytes[] calldata tdxSignatures       // one per batch
) external onlyOwner whenNotPaused;
```

Rust-side calldata signature: `"verifyBatches(uint256,bytes[],bytes[],bytes[])"`
(`crates/l2/sequencer/l1_proof_sender.rs:54`, encoding at `539-549`: three parallel `bytes[]`
arrays; absent proofs are empty `Value::Bytes(vec![])`).

Per-batch dispatch inside `_verifyBatchInternal` (`OnChainProposer.sol:310-412`):
- `REQUIRE_RISC0_PROOF` → `IRiscZeroVerifier(RISC0_VERIFIER_ADDRESS).verify(risc0BlockProof,
  risc0Vk, sha256(publicInputs))` (OnChainProposer.sol:359-373). Interface
  (`crates/l2/contracts/src/l1/interfaces/IRiscZeroVerifier.sol:39-49`):
  `function verify(bytes calldata seal, bytes32 imageId, bytes32 journalDigest) external view;`
- `REQUIRE_SP1_PROOF` → `ISP1Verifier(SP1_VERIFIER_ADDRESS).verifyProof(sp1Vk, publicInputs,
  sp1ProofBytes)` (OnChainProposer.sol:376-388). Interface
  (`.../ISP1Verifier.sol:8-20`): `function verifyProof(bytes32 programVKey, bytes calldata
  publicValues, bytes calldata proofBytes) external view;`
- `REQUIRE_TDX_PROOF` → `ITDXVerifier(TDX_VERIFIER_ADDRESS).verify(publicInputs, tdxSignature)`
  (OnChainProposer.sol:390-399). Interface (`.../ITDXVerifier.sol:7-15`):
  `function verify(bytes calldata payload, bytes memory signature) external view;`
  (implementation: `crates/l2/tee/contracts/src/TDXVerifier.sol:50-53`, ECDSA-authorized-signer
  over the payload.)
- Aligned mode: `verifyBatchesAligned(uint256 firstBatchNumber, uint256 lastBatchNumber,
  bytes32[][] sp1MerkleProofsList, bytes32[][] risc0MerkleProofsList)`
  (`OnChainProposer.sol:437-441`), delegating to `isProofVerified(bytes32[],uint16,bytes32,bytes)`
  on `ALIGNEDPROOFAGGREGATOR` (`OnChainProposer.sol:543-561`).

**Honest constraint:** the deployed verifier set accepts **only SP1 / RISC Zero / TDX / Aligned
proofs**. Verification keys are keyed by `(commitHash, verifierId)` (`OnChainProposer.sol:113-115`,
`verificationKeys`), so even an SP1-format proof must come from *the ethrex guest program image*.
A lattice proof from lzx **cannot** be verified by the stock contract. (Design response: §10.3.)

Supporting state: `REQUIRE_RISC0_PROOF/REQUIRE_SP1_PROOF/REQUIRE_TDX_PROOF`, `VALIDIUM`,
`ALIGNED_MODE` flags set in `initialize(...)` (`OnChainProposer.sol:125-199`); the contract is
`onlyOwner` (owner = Timelock) for commit and verify.

### 3.4 Proof sender (Rust)

`crates/l2/sequencer/l1_proof_sender.rs`: builds the three parallel proof arrays
(`523-537`), encodes `verifyBatches` calldata (`539-549`), sends via the Timelock
(`551-555`); falls back to single-batch sends on revert (`651-689`); invalid-proof custom
errors (`InvalidRisc0Proof 0x14add973`, `InvalidSp1Proof 0x7ff849b5`, `InvalidTdxProof
0x62013a95`, `IOnChainProposer.sol:41-43`) trigger deletion of the stored proof (`566-576`).

---

## 4. L1 ↔ L2 messaging: deposits and withdrawals

### 4.1 Deposits (L1 → L2)

Entry points on `CommonBridge` (`crates/l2/contracts/src/l1/CommonBridge.sol`):
- `deposit(address l2Recipient) payable` (CommonBridge.sol:296-311) → builds
  `mintETH(l2Recipient)` calldata for the L2 bridge predeploy (`BRIDGE_ADDRESS =
  0x0000...ffff`, `crates/l2/common/src/messages.rs:28-31`).
- `depositERC20(tokenL1, tokenL2, destination, amount)` (CommonBridge.sol:317-338).
- generic `sendToL2(SendValues{to, gasLimit, value, data})` (CommonBridge.sol:288-293;
  struct `ICommonBridge.sol:46-51`).

Both funnel into `_sendToL2` (CommonBridge.sol:253-285), which computes the
**privileged-tx hash** (the exact concatenation an adapter must reproduce):

```
l2MintTxHash = keccak256( bytes32(chainId) || bytes20(from) || bytes20(to)
                         || bytes32(transactionId) || bytes32(value)
                         || bytes32(gasLimit) || bytes32(keccak256(data)) )
```
pushes it to `pendingTxHashes`, and emits:

```solidity
event PrivilegedTxSent(address indexed l1From, address from, address to,
                       uint256 transactionId, uint256 value, uint256 gasLimit, bytes data);
```
(`ICommonBridge.sol:18-26`).

The L2 `l1_watcher` parses the event data (layout documented at
`crates/l2/sdk/src/privileged_data.rs:24-39`: from[0..32], to[32..64], transactionId[64..96],
value[96..128], gasLimit[128..160], data-offset[160..192], data-len[192..224], data[224..]) and
injects a `PrivilegedL2Transaction` (type `0x7e`, §2.2) with `from` = event `from` and nonce =
`transactionId` (`privileged_data.rs:86-121`).

The rolling hash consumed by `commitBatch` is
`compute_privileged_transactions_hash` (`crates/l2/common/src/privileged_transactions.rs:71-97`):
for `n` hashes, `H256 = be16(n) || keccak256(concat(h_1..h_n))[2..32]` — i.e. the first 2 bytes
are the big-endian count, the last 30 bytes are a truncated keccak. The L1 contract recomputes
this via `getPendingTransactionsVersionedHash(uint16 number)` (CommonBridge.sol:341-359) and
rejects mismatches (`InvalidPrivilegedTransactionLogs`, OnChainProposer.sol:243-252).

### 4.2 Withdrawals (L2 → L1)

L2 side: a withdrawal is a message emitted through the `Messenger` predeploy
(`MESSENGER_ADDRESS = 0x0000...fffe`, `messages.rs:12-15`). The `L1Message` wire struct
(`messages.rs:47-66`):

```rust
pub struct L1Message { pub from: Address, pub data_hash: H256, pub message_id: U256 }
// encode() = from(20) || data_hash(32) || message_id(32 BE); hash = keccak(encode())
```

Extracted from receipts by matching `L1Message(address,bytes32,uint256)` events
(`messages.rs:76-96`). The batch commitment carries the **Merkle root of these message
hashes** (`compute_merkle_root(&batch.l1_out_message_hashes)`, `l1_committer.rs:1298`; the
tree is `crates/l2/common/src/merkle_tree.rs`). On verify, `CommonBridge.publishWithdrawals(
batchNumber, withdrawalsLogsMerkleRoot)` is called (OnChainProposer.sol:265-270; event
`WithdrawalsPublished(uint256 indexed, bytes32 indexed)` `ICommonBridge.sol:32-35`). Users then
prove-and-claim via `claimWithdrawal(...)` with an `L1MessageProof { batch_number, message_id,
message_hash, merkle_proof }` (`messages.rs:39-45`; produced by the L2 RPC
`ethrex_getL1MessageProof`, `rpc.rs:410`).

L2↔L2 messaging uses the `L2Message` struct (`messages.rs:98-131`) with
`L2Message(uint256,address,address,uint256,uint256,uint256,bytes)` events and per-chain
rolling hashes (`L2MessageRollingHash {chainId, rollingHash}`, `ICommonBridge.sol:69-72`);
`get_balance_diffs` (`messages.rs:177-244`) aggregates the `BalanceDiff` /
`AssetDiff{tokenL1, tokenL2, destTokenL2, value}` values (`ICommonBridge.sol:53-67`)
committed in `commitBatch` and republished by `publishL2Messages` on verify
(OnChainProposer.sol:401-403).

---

## 5. State diffs and commitments — the full commitment set

ethrex L2 today commits **full state roots** (not sparse state diffs) plus message-level diffs.
The complete on-chain commitment record `BatchCommitmentInfo` (OnChainProposer.sol:34-44):

| field | meaning | source |
|---|---|---|
| `newStateRoot` | MPT state root after last block of batch | `l1_committer.rs:1055-1060` (`state_trie(...).hash_no_commit`) |
| `blobKZGVersionedHash` | `0x01‖sha256(KZG C)[1..]` of the batch blob (from `blobhash(0)`) | `blobs_bundle.rs:76-80` |
| `processedPrivilegedTransactionsRollingHash` | `be16(n)‖keccak(concat)[2..32]` over deposit hashes | `privileged_transactions.rs:71-97` |
| `withdrawalsLogsMerkleRoot` | Merkle root over `L1Message` hashes | `l1_committer.rs:1298` |
| `lastBlockHash` | `keccak(rlp(header))` of last block | `l1_committer.rs:1521-1531` |
| `nonPrivilegedTransactions` | count of non-0x7e txs in batch | `l1_committer.rs:1048-1053` |
| `balanceDiffs` | per-chain ETH + per-token diffs | `messages.rs:177-244` |
| `commitHash` | `keccak(git_commit_string)` binding the prover vk | `l1_committer.rs:1300` |
| `l2InMessageRollingHashes` | per-source-chain L2-in message rolling hashes | `l1_committer.rs:1100-1106` |

The **data availability commitment is the blob's KZG versioned hash** — the prover input pins
`blob_commitment`/`blob_proof` (`prover.rs:32-35`) and the guest verifies the blob against it
(`crates/guest-program/src/l2/program.rs`). The batch blob content (§1.4) is the on-chain DA:
full RLP blocks, not state diffs. (A separate `AccountUpdate`-based state-diff path exists for
witness generation — `crates/common/types/account_update.rs` — but it feeds the prover, not L1.)

---

## 6. ZODA architecture: what it is and is not

**What it is** (`repos/zoda/README.md:1-50`, `docs/ARCHITECTURE.md`): a zero-dependency,
pure-Rust **library** implementing the ZODA tensor-code DA protocol (eprint 2025/034) with
Ethereum EIP-4844/EIP-7594 integration and a post-quantum lattice PCS variant. Validity
design: data is arranged in an `m×k` matrix over BLS12-381 `Fr`; a systematic RS tensor code
extends it to `2m×2k`; rows and columns are Merkle-committed; Fiat–Shamir-derived random
projections (`g_r`, `g_r2`, `z_r`, `z_r2`) make **every sampled row/column its own proof**
(verified with one O(width) inner product) — "zero-overhead" sampling
(`crates/zoda-core/src/lib.rs:19-32`, `crates/zoda-core/src/params.rs:104-134`).

Crate layout (README.md:30-50): `zoda-math` (fields/NTT/Merkle/RNG) → `zoda-bls` (full
BLS12-381) → `zoda-kzg` (EIP-4844 KZG + mainnet trusted-setup parser), `zoda-core` (tensor
DA), `zoda-pq` (BDLOP/Lyubashevsky lattice PCS) → services: `zoda-edas` (EIP-7594 cells +
FK20), `zoda-das` (2D sampling sessions), `zoda-sybils` (BLS sortition), `zoda-rda`
(adaptive confidence), `zoda-archival` (custody storage), `zoda-bridges` (light-client
proofs), `zoda-ethrex` (ethrex-compatible sidecar types), `zoda` (facade).

**What it is NOT:** there is **no node binary, no JSON-RPC server, no P2P stack**.
`docs/DEPLOYMENT.md:62-66`: *"A node binary wiring these together against a real P2P stack
(libp2p) is intentionally not in scope yet — the layers are trait-based (`SampleOracle`,
custody stores) so the network binding is a thin adapter."* Consequence for us: "posting a
blob to zoda" means (a) producing the byte-exact artifacts zoda's API defines (sidecar /
cells / grid commitments), and/or (b) embedding zoda's algorithms; a network submission
endpoint does not exist to call. This is stated plainly because the task asked for JSON-RPC
schemas — there are none, by design.

---

## 7. ZODA blob/data format, commitments, and "certificates"

### 7.1 The EIP-4844 path (primary, ethrex-compatible)

Blob unit: 131,072 bytes; each 32-byte chunk interpreted as a big-endian field element must be
`< r` (canonicalization: `chunk[0] &= 0x3f`, `crates/zoda-ethrex/src/lib.rs:113-115`).

The submission artifact is `BlobTransactionSidecar`
(`crates/zoda-ethrex/src/lib.rs:24-30`) — **byte-identical to ethrex's
`BlobTransactionSidecar`** (type mapping table: `docs/ETHEX_INTEGRATION.md:8-16`):

```rust
pub struct BlobTransactionSidecar {
    pub blob_versioned_hashes: Vec<[u8; 32]>, // 0x01 || sha256(C)[1..]
    pub kzg_commitments:      Vec<[u8; 48]>,  // G1 compressed (c-kzg layout)
    pub blobs:                Vec<[u8; 131072]>,
    pub kzg_proofs:           Vec<[u8; 48]>,
}
```

Builder: `BlobTransactionSidecar::from_blobs(&blobs, &setup)` (`lib.rs:72-89`) =
per blob `blob_to_kzg_commitment` (`crates/zoda-kzg/src/eip4844.rs:178`) +
`compute_blob_kzg_proof` (`eip4844.rs:477`) + `kzg_to_versioned_hash` (`eip4844.rs:553`).
Verification (what a DA node runs on ingest): `validate` / `validate_batch`
(`zoda-ethrex/src/lib.rs:35-69`) = `verify_blob_kzg_proof` (`eip4844.rs:493`) per blob /
one pairing for the batch (`eip4844.rs:517`). The "certificate" on this path is the pair
(KZG commitment, per-blob proof) — verified inside an EIP-4844 blob transaction by Ethereum
consensus itself, which is also **where zoda's Ethereum-flavoured data settles** (§9).

### 7.2 The EIP-7594 / PeerDAS path (chunking + per-cell certificates)

`crates/zoda-edas/src/lib.rs`:
- Constants (spec-exact): `FIELD_ELEMENTS_PER_CELL = 64`, `CELLS_PER_EXT_BLOB = 128`,
  `BYTES_PER_CELL = 2048`, `NUMBER_OF_CUSTODY_GROUPS = 128`, `CUSTODY_REQUIREMENT = 4`
  (`lib.rs:18-25`). `pub type Cell = [u8; 2048]` (`lib.rs:28`).
- Chunking: `compute_cells(blob, setup)` — IFFT to monomial, FFT to the 8192-domain,
  bit-reverse, slice into 128 cells of 64 big-endian Fr elements (`lib.rs:88-101`).
- Certificate per cell: `compute_cells_and_kzg_proofs(blob, setup) -> (Vec<Cell>,
  Vec<[u8;48]>)` — FK20 proofs, O(n log n) for all 128 (`lib.rs:104-124`, `147-150`).
- Batch verify (the "validity proof attests cells are correct extensions of the committed
  blob"): `verify_cell_kzg_proof_batch(&commitments, &indices, &cells, &proofs, &setup)`
  (`lib.rs:420`) — one pairing for any number of cells.
- Recovery: `recover_cells_and_kzg_proofs(&known_idx, &known_cells, &setup)` (`lib.rs:589`)
  from any 64 of 128 cells; custody assignment `get_custody_groups(node_id, count)`
  (`lib.rs:741`) / `compute_columns_for_custody_group` (`lib.rs:770`).

### 7.3 The ZODA tensor path (the protocol's own commitment)

`ZodaParams::new(m, k)` (powers of two) → `commit(&Matrix)` (`crates/zoda-core/src/params.rs:51-79`):
tensor-encode, build row/column Merkle trees (leaves: `0x01/0x02 ‖ index_u64le ‖ Fr_le32*`,
`params.rs:82-100`), derive challenges from
`sha256("ZODA-TENSOR-FIAT-SHAMIR-V1" ‖ row_root ‖ col_root ‖ m ‖ k)` (`params.rs:104-134`).
The **certificate** is `ZodaPublic` (`params.rs:211-225`): `{g_r, g_r2, z_r, z_r2, row_root,
col_root}` — roots + projections. There is no SNARK here: soundness is information-theoretic
(sampling catches withholding); "verification" of a line is `verify_row_sample /
verify_column_sample` (`crates/zoda-core/src/sample.rs`, re-exported `lib.rs:44`).

### 7.4 Post-quantum commitment option (relevant to lzx)

`zoda-pq` (`docs/POST_QUANTUM.md`, `crates/zoda-pq/src/pcs.rs`): BDLOP Module-LWE/SIS
commitment `C = A·r + B·m` over `R_q = Z_q[X]/(X^n+1)`, q = 8380417, with Lyubashevsky
σ-protocol openings at admissible ring points ζ = t·X, ζ^n = −1 (POST_QUANTUM.md:22-65).
API: `LatticePcs::commit/open/verify` (`pcs.rs:240`), `ring_point_from_seed` (`pcs.rs:209`).
Sizes at L1 (n=1024): commitment ≈ 5.9 KB, proof ≈ 32 KB (POST_QUANTUM.md:91-97) — the
documented gap a LaBRADOR-style folding (i.e. exactly lzx's stack) is meant to close
(POST_QUANTUM.md:99-105).

---

## 8. Querying / retrieving data from zoda

Library-level retrieval surfaces (all local, trait-based — a network node would expose these
over RPC):

- **Sampling** (availability queries): `zoda-das::sample_session` /
  `run_attested_session` (`crates/zoda-das/src/session.rs:208`), `sample_cell_session`
  (`crates/zoda-das/src/cell_das.rs:257`); exact sample-count planners in
  `zoda-das/src/availability.rs` (`cell_samples_for_target`, `line_miss_probability`).
  Adaptive confidence: `zoda-rda::RdaSession` (`crates/zoda-rda/src/lib.rs:39`) with
  `RdaVerdict` (`lib.rs:124`).
- **Full retrieval** (reconstruction): `zoda-core::reconstruct(public, kept_cols, matrix)`
  (`crates/zoda-core/src/reconstruct.rs`) from ≥ k of 2k columns; `reconstruct_2d /
  PartialGrid` for scattered cells (`crates/zoda-core/src/reconstruct2.rs`, re-export
  `lib.rs:43`); blob-level `recover_blob` (`cell_das.rs:390`) and
  `zoda-edas::recover_cells_and_kzg_proofs` (`lib.rs:589`) from ≥ 64 of 128 cells.
- **Custody storage**: `CellCustody` (`crates/zoda-archival/src/cells.rs:14-20`) — per-slot,
  per-column `(Cell, proof[48])` per blob with verify-on-insert
  (`put_column_verified`, `cells.rs:43-72`); `GridCustody` /
  `CustodyStore` (`crates/zoda-archival/src/store.rs:10,198`); disk formats `ZODACST1` /
  `ZODACEL1` (`cells.rs:8-9`, `crates/zoda-archival/src/format.rs:90,130`).
- **Inclusion proofs for light clients**: `RowInclusionProof / ColumnInclusionProof`
  (`crates/zoda-bridges/src/lib.rs:17-28`) = ZODA projection check + SHA-256 Merkle proof
  against `ZodaPublic.row_root/col_root`; `BridgeMessage {destination_chain: [u8;4], nonce:
  u64, payload}` with KZG-backed message commitments (`lib.rs:95-146`) — the natural shape
  for "prove the CLOB batch blob contains order X".

---

## 9. Relationship of zoda to ethrex and Bitcoin

- **To ethrex:** integration is explicit and typed — `zoda-ethrex` provides byte-compatible
  `BlobTransactionSidecar` so zoda DA tooling plugs into ethrex's blob import path
  (`docs/ETHEX_INTEGRATION.md:1-63`): ethrex's `verify_blob_kzg_proof` call site is replaced
  by `sidecar.validate(&setup)`; the DA-side path extends sidecars to EIP-7594 columns and
  custody/sampling (`ETHEX_INTEGRATION.md:35-61`). zoda does **not** consume ethrex blocks;
  it consumes **blobs** — i.e. it sits at the same layer as the ethrex batch blob from §1.4.
- **To Ethereum:** zoda's Ethereum-flavoured mode settles on Ethereum L1 consensus via
  EIP-4844 blob transactions (the KZG versioned hash is the on-chain anchor); EIP-7594 cells
  follow the Fulu consensus specs (320/320 official vectors pass bit-exactly, README.md:145).
- **To Bitcoin:** **no relationship.** Nothing in the repo mentions Bitcoin; zoda is not a
  Bitcoin L2 DA. Its only settlement anchors are Ethereum blob transactions (KZG) or, in the
  PQ variant, no chain at all (the lattice commitment is setup-free and would itself need a
  verification bridge — see §10.3).

---

## 10. Integration design for lzx

### 10.0 Assumptions about the lzx side (verified in the workspace)

- lzx proves RV64IMAC programs: `prove_program(pcs, program, public_input, max_steps) ->
  (PublicOutput, ProofEnvelope)` (`crates/lattice-zkvm/src/prove.rs:39-44`) with
  `PublicOutput { final_regs: [u64; 32], memory_digest: [u8; 32] }` (`prove.rs:22-26`).
- The proof envelope is a self-describing byte container: `{version u32 LE, program_digest
  [32], public_input_digest [32], public_output_digest [32], tagged length-prefixed sections
  (Commitment/Sumcheck/Witness/Norm)}` with 32 MiB total cap
  (`crates/lattice-zkvm/src/envelope.rs:21-35, 96-100`).
- lzx is pure-std, zero-dependency (worklog Task 11); crates.io is blocked in this
  environment. **Adapters therefore cannot link ethrex or zoda crates** — they must
  reimplement the byte formats below in `lzx/crates/*` with `std` only. Every format needed
  is fully specified above, which is exactly why this document exists.

### 10.1 Mapping: one proved CLOB batch = one ethrex L2 block (inside one batch)

**Recommended: each proved CLOB batch becomes ONE `Block`, and ONE `Batch` containing exactly
that block** (single-block batches are first-class — the committer loop naturally produces
them, `l1_committer.rs:841-1065`). Byte-level assembly plan for the adapter
(new crate `lzx/crates/lattice-bridge-ethrex`, pure std):

1. **CLOB order → transactions.** Encode each order as an EIP-1559 tx
   (`0x02 || rlp([chain_id, nonce, max_priority_fee, max_fee, gas_limit, to, value, data,
   access_list=[], y_parity, r, s])`, §2.1) where `data` = the order payload
   (fixed-width order struct serialized by the CLOB). `to` = the CLOB contract address on L2;
   `value` = order margin; `nonce` = order sequence. If the operator injects proved outputs
   authoritatively instead, use a `PrivilegedL2Transaction` (`0x7e`, no signature, explicit
   `from` — §2.2) — this is the natural carrier for "lzk operator applies settled batch",
   and it is excluded from `nonPrivilegedTransactions` by the committer
   (`l1_committer.rs:1048-1053`).
2. **Header.** Fill `parent_hash` = hash of previous lzx/ethrex block, `number` = height,
   `state_root` = **lzx state digest** (keccak of the CLOB position-tree root / or
   `PublicOutput.memory_digest`), `transactions_root` = trie root over canonical txs keyed by
   `rlp(index)` (`crates/common/types/block.rs:361-369`), `receipts_root` /
   `logs_bloom` = zeros-or-computed, `ommers_hash` = `keccak(rlp([]))`, `timestamp`,
   `gas_*` = measured. Encode with the exact 25-field order of §1.3 (optionals: include
   `base_fee_per_gas` + `withdrawals_root` post-Shanghai; omit the rest).
3. **Block RLP** = `rlp([header, txs, [], withdrawals?])` (§1.3).
4. **Batch blob** = `u64_be(1) ‖ rlp(Block) ‖ FeeConfig::to_vec()` zero-padded to 131,072
   bytes (§1.4). Compute the KZG commitment **only if** a trusted setup is available
   (deferred — see §10.5); the versioned-hash field of the payload can still be filled with
   `0x01 ‖ sha256(placeholder)` in dry-run mode since local tests never hit the chain.
5. **commitBatch calldata** = `keccak("commitBatch(uint256,bytes32,bytes32,bytes32,bytes32,
   uint256,bytes32,(uint256,uint256,(address,address,address,uint256)[],bytes32[])[],
   (uint256,bytes32)[])")[0..4] ‖ ABI(args)` with all message/rolling-hash fields zero
   (no deposits/withdrawals in the CLOB MVP) and `balanceDiffs = []`,
   `l2MessageRollingHashes = []` (§1.5). The adapter exposes:
   `fn build_commit_calldata(batch: &ClobBatchCommitment) -> Vec<u8>`.

### 10.2 lzx proof → ethrex public-inputs payload

Assemble the 256-byte fixed public-input block **in the exact order of
`_getPublicInputsFromCommitment` / `ProgramOutput::encode`** (§3.2):
`initial_state_hash ‖ final_state_hash ‖ withdrawals_merkle_root(=0) ‖
l1_in_rolling_hash(=0) ‖ blob_versioned_hash ‖ last_block_hash ‖ chain_id ‖
non_privileged_count`, with variable parts empty for the CLOB MVP. Mapping:

| ethrex public input | lzx source |
|---|---|
| `initial_state_hash` | previous batch's `PublicOutput.memory_digest` (keccak-wrapped) |
| `final_state_hash` | current `PublicOutput.memory_digest` |
| `last_block_hash` | `keccak(rlp(header))` of the assembled block |
| `blob_versioned_hash` | hash of the DA blob (§10.4) |
| `chain_id`, `non_privileged_count` | config, tx count |

The lzx proof additionally binds `program_digest`, `public_input_digest`,
`public_output_digest` (envelope header, `envelope.rs:21-27`); the adapter defines
`public_inputs_digest = keccak(the 256-byte block ‖ program_digest ‖ public_input_digest)`.

### 10.3 Verification path — honest assessment + concrete proposal

**Fact:** the stock `OnChainProposer` verifies only SP1 / RISC Zero / TDX / Aligned proofs
(§3.3); verification keys are image-bound by `(commitHash, verifierId)`. lzx lattice proofs
cannot pass `verifyBatches` as deployed. Options, in order of realism:

1. **No-proof dev mode (works with stock contracts today).** `initialize` accepts
   `requireRisc0Proof = requireSp1Proof = requireTdxProof = false` (no cross-flag requirement
   forces any of them true — `OnChainProposer.sol:125-199`). Then `verifyBatches` performs
   only bridge bookkeeping and advances `lastVerifiedBatch` — an **operator-attested
   (trusted/optimistic-with-revert-window) mode** guarded by `onlyOwner` (Timelock) +
   `revertBatch` while paused (OnChainProposer.sol:655-670). This is what the local e2e test
   simulates; lzx's proof is verified **off-chain** by the operator/verifier set before the
   attestation is sent. Payload = exactly §3.3's `verifyBatches(uint256,bytes[],bytes[],bytes[])`
   with three arrays of empty `bytes`.
2. **Custom verifier (the design we propose to add).** A new L1 contract pair:

   ```solidity
   // SPDX-License-Identifier: MIT
   pragma solidity =0.8.31;
   /// Lattice (lzx) verifier stub. First implementation: operator ECDSA attestation.
   /// Second implementation: on/off-chain lattice-verification bridge (committee or future
   /// EVM-friendly lattice argument).
   interface ILatticeVerifier {
       /// @dev Must revert on failure. proofBytes = lzx ProofEnvelope::to_bytes().
       function verify(bytes calldata publicInputs, bytes calldata proofBytes) external view;
   }
   ```

   and inside `OnChainProposer` (mirroring the SP1 branch at OnChainProposer.sol:376-388):

   ```solidity
   bool public REQUIRE_LATTICE_PROOF;
   address public LATTICE_VERIFIER_ADDRESS;
   uint8 internal constant LATTICE_VERIFIER_ID = 3;

   function verifyBatchesLattice(
       uint256 firstBatchNumber,
       bytes[] calldata latticeProofsBytes
   ) external onlyOwner whenNotPaused {
       /* same per-batch loop as verifyBatches, with: */
       // try ILatticeVerifier(LATTICE_VERIFIER_ADDRESS).verify(publicInputs, latticeProofsBytes[i])
       // {} catch { revert InvalidLatticeProof(); }
   }
   ```

   with `error InvalidLatticeProof();` and `verificationKeys[commitHash][LATTICE_VERIFIER_ID]`
   reused to bind the lzx proving-key digest. The lzx adapter then emits
   `verifyBatchesLattice(uint256,bytes[])` calldata: selector
   `keccak("verifyBatchesLattice(uint256,bytes[])")[0..4]` ‖ ABI head `(uint256, offset)`
   ‖ array (per batch: length ‖ `ProofEnvelope::to_bytes()`).
   **Honest note:** a full lattice argument on EVM is not realistic today (the envelope's
   ring arithmetic and norm checks are ~MiB-scale and pairing-free); the first
   `ILatticeVerifier` is therefore an ECDSA/committee attestation over
   `keccak256(publicInputs ‖ proofDigest)`, upgradeable to an Aligned-style aggregated
   verification service later — the same escalation path ethrex itself uses for SP1
   (`verifyBatchesAligned`, §3.3).
3. **lzx-as-guest inside SP1/RISC0** (longer-term): port the lzx verifier into an SP1 guest
   so the outer proof is stock-verifiable. Not in scope for the adapters; noted for honesty.

### 10.4 Data availability on zoda — exact payload

Because zoda has no network endpoint (§6), "submission" = producing the byte-exact artifacts
its ingest API consumes, which the DA node (when deployed) verifies with `validate` /
`put_column_verified`. The adapter (`lzx/crates/lattice-bridge-zoda`) emits:

```rust
/// Byte-compatible with zoda_ethrex::BlobTransactionSidecar (zoda-ethrex/src/lib.rs:24-30)
/// and ethrex's BlobsBundle.
pub struct ZodaBlobSubmission {
    pub blob_versioned_hashes: Vec<[u8; 32]>, // 0x01 || sha256(C)[1..]
    pub kzg_commitments:       Vec<[u8; 48]>,
    pub blobs:                 Vec<[u8; 131_072]>,
    pub kzg_proofs:            Vec<[u8; 48]>,
}
```

**Payload content (the CLOB batch data blob, before padding):**

```
magic "LZXBLOB1" (8B)
|| u32_le version
|| u64_le batch_number
|| [u8;32] lzx_program_digest        // ProofEnvelope.program_digest
|| [u8;32] lzx_public_input_digest
|| [u8;32] lzx_public_output_digest  // binds trades/state via memory_digest
|| u64_le order_count
|| order_i: [u8;32] order_id, u64 price, u64 qty, u8 side, [u8;20] trader
|| u64_le output_len || outputs (trades + position deltas, lzx canonical serialization)
|| u64_le envelope_len || ProofEnvelope::to_bytes()
```

Capped at 126,976 B (SAFE_BYTES_PER_BLOB); zero-padded to 131,072; each 32-byte chunk
canonicalized (`chunk[0] &= 0x3f`) exactly as zoda requires (`zoda-ethrex/src/lib.rs:118-122`,
`zoda-bridges/src/lib.rs:120-122`). KZG commitment/proof fields are filled when a setup file
is present; in pure-std test mode they are zero-filled and flagged by version byte — the
retrieval path below does not depend on them.

**Retrieval path (client side):**
- Fetch sidecar by `blob_versioned_hash` (EIP-4844 availability) or from custody columns
  (`CellCustody::get_column`, §8); strip padding; parse header; **verify the lzx proof
  locally** via `verify_program` against the digests in the blob header (this is the step
  that turns DA into validity — lzx's proof is the certificate, zoda only certifies
  *availability*).
- Optional ZODA-tensor overlay for sampling clients: arrange the (padded) blob's 4096 field
  elements into an `m×k` grid, commit per §7.3, serve row/column samples that verify with
  `ZodaPublic` — algorithm is re-implementable in pure std (SHA-256 Merkle + inner products),
  and `zoda-bridges::RowInclusionProof` (`zoda-bridges/src/lib.rs:17-28`) is the light-client
  proof shape to mirror.
- PQ option (fits lzx's thesis): commit the blob polynomial with a BDLOP-style lattice
  commitment (§7.4) instead of KZG — no trusted setup — and prove openings with lzx's own
  HyperWolf/Serval stack rather than zoda-pq's 32 KB σ-protocol proofs. This is the
  "post-quantum DA certificate" differentiator; defer until the base path is green.

### 10.5 What runs where (honest table for the parent's local e2e test)

| step | runs locally (no nodes)? | how |
|---|---|---|
| CLOB order → EIP-1559 / 0x7e tx RLP bytes | **yes** | std-only RLP encoder in `lattice-bridge-ethrex` matching §2.1/§2.2 field order exactly |
| L2 block RLP + batch blob layout assembly | **yes** | §1.3 header order + §1.4 blob layout; verify by decoding our own bytes with a strict decoder (round-trip property test) |
| `commitBatch` / `verifyBatches` / `verifyBatchesLattice` calldata bytes | **yes** | keccak (already in `lattice-core::keccak`) + minimal ABI encoder for `(uint,bytes32×5,uint,bytes32,(…)[],(…)[],bytes[])`; assert selector + layout vs. golden vectors |
| ethrex public-inputs block (256 B) | **yes** | §3.2 layout; assert against `ProgramOutput::encode` semantics |
| lzx proof generation + local verification | **yes** | existing `prove_program` / `verify_program` |
| KZG commitment/proof for the blob | **no** (needs trusted setup + BLS12-381) | zero-fill + version flag; or `Setup::from_seed_for_testing`-style small setup is NOT in scope for pure-std lzx |
| actual L1 tx signing/broadcast of commit/verify | **no** | needs ethrex L1 node + funded key (docker-compose-l2 exists at `crates/l2/docker-compose.yaml`) |
| on-chain SP1/RISC0/TDX verification of an lzx proof | **no — impossible with stock contracts** | §10.3: attestation mode or proposed `ILatticeVerifier`/`verifyBatchesLattice` |
| zoda sidecar byte-format + cell layout (128 cells × 2048 B) | **yes** | mirror §7.1/§7.2 shapes; chunk canonicalization; round-trip tests |
| zoda tensor commitment (Merkle roots + FS projections) | **yes** | pure-std reimplementation (SHA-256 + Fr-math subset) or digest-level stub |
| zoda KZG cell proofs / FK20 | **no** | requires BLS12-381 + trusted setup; leave 48-byte fields zeroed in test mode |
| posting to a live zoda network | **no — network does not exist yet** | §6; artifacts are the deliverable |
| custody column storage format (ZODACEL1) | **yes (format only)** | §8 `cells.rs` layout |

**Recommended e2e shape for the parent agent:** prove a CLOB batch with lzx → assemble the
ethrex block RLP + blob + commit/verify calldata → assemble the zoda sidecar + cell layout →
verify our own lzx proof over the digests embedded in both payloads → golden-vector tests
pinning every byte layout above. That exercises 100% of what is exercisable without nodes,
and every artifact is ready to be broadcast the day nodes/contracts exist.

### 10.6 Adapter API sketch (pure std, for the parent to implement)

```rust
// lzx/crates/lattice-bridge-ethrex
pub struct EthrexBatchCommitment { /* fields of §5 */ }
pub fn rlp_encode_block(header: &L2HeaderSpec, txs: &[L2TxSpec]) -> Vec<u8>;
pub fn build_batch_blob(blocks_rlp: &[Vec<u8>], fee_cfgs: &[u8]) -> [u8; 131_072];
pub fn build_commit_calldata(c: &EthrexBatchCommitment, based: bool) -> Vec<u8>;
pub fn build_verify_calldata(first_batch: u64, proofs: &[Vec<u8>]) -> Vec<u8>;        // verifyBatches
pub fn build_verify_lattice_calldata(first_batch: u64, proofs: &[Vec<u8>]) -> Vec<u8>; // §10.3 proposal
pub fn build_public_inputs(c: &EthrexBatchCommitment) -> [u8; 256];

// lzx/crates/lattice-bridge-zoda
pub struct ZodaBlobSubmission { /* §10.4 */ }
pub fn build_clob_blob(batch: &ClobBatch, envelope: &ProofEnvelope) -> [u8; 131_072];
pub fn canonicalize_blob_chunks(blob: &mut [u8; 131_072]);
pub fn split_cells(blob: &[u8; 131_072]) -> Vec<[u8; 2048]>; // layout only, no FK20
```

---

## Appendix A — quick selector/constant reference

| item | value | source |
|---|---|---|
| privileged tx type byte | `0x7e` | `transaction.rs:391` |
| EIP-1559 type byte | `0x02` | `transaction.rs:384` |
| blob size / safe size | 131,072 / 126,976 B | `constants.rs:24-28` |
| versioned hash prefix | `0x01` | `blobs_bundle.rs:79` |
| messenger / bridge predeploys | `0x…fffe` / `0x…ffff` | `messages.rs:12-15,28-31` |
| privileged-tx budget | 300/batch | `privileged_transactions.rs:7` |
| rolling hash format | `be16(n) ‖ keccak(concat)[2..32]` | `privileged_transactions.rs:80-96` |
| cells | 128 × 2048 B | `zoda-edas/lib.rs:18-28` |
| zoda FS domain | `"ZODA-TENSOR-FIAT-SHAMIR-V1"` | `zoda-core/params.rs:14` |
| verify errors | `InvalidRisc0Proof 0x14add973` / `InvalidSp1Proof 0x7ff849b5` / `InvalidTdxProof 0x62013a95` | `IOnChainProposer.sol:41-43` |
