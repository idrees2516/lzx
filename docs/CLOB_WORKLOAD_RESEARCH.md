# CLOB Workload Research — `perp-options-clob` → lzx zkVM Guest Port Blueprint

**Task ID:** 3-a · **Date:** 2026-09-29
**Subject repo:** `/home/z/my-project/repos/perp-options-clob` (git HEAD `f625aa0`, "v0.6: American-style options + zkLighter channel orderbook")
**Target:** a no_std, zero-dependency, integer-only RV64IMAC guest program in the lzx zkVM (`/home/z/my-project/lzx`, crates `lattice-vm` = canonical RV64IMAC decode/exec/state with per-step `TraceRow`, `lattice-zkvm` = proving envelope).

This document is the port blueprint: what the workload is, what its exact
algorithms are (transcribed with file:line citations), what it costs to
execute, and exactly what to port into the guest. All paths below are
relative to the repo root unless prefixed `lzx/`.

---

## 1. Workspace layout and dependency map

The workspace is **15 crates, zero external dependencies**. Root
`Cargo.toml:1-19` lists the members; `Cargo.lock` contains exactly the 15
`poc-*` packages and nothing else. `unsafe_code = "forbid"` workspace-wide
(`Cargo.toml:31`). The only external crate anywhere is `libfuzzer-sys 0.4`
in the out-of-workspace `fuzz/` crate (`fuzz/Cargo.toml:14`), which is a
test harness and irrelevant to a port. **There is no serde, no rustc-hash,
no rayon — every data structure is hand-rolled std.** This is the single
best property of the repo for a dependency-free port.

| Crate | LOC (src) | Purpose | Path deps | Port-relevant blockers |
|---|---|---|---|---|
| `poc-core` | ~1,050 | Domain primitives: `Order`, `Side`, `OrderType`, `TimeInForce`, `SelfTradePrevention`, `PerpMarket`, `OptionMarket`, `Instrument`, exact-integer money (`num.rs`) | none | `Symbol = String` (`core/src/types.rs:15`); `String` in markets; `format!` in errors (`core/src/errors.rs`) |
| `poc-orderbook` | 1,743 | The CLOB: zkLighter channel book, pure `match_taker`, auction uncross | `poc-core` | `BTreeMap`, `VecDeque`, `Vec`, `String` symbol |
| `poc-oracle` | 534 | Multi-provider median/staleness/qu quarantine mark, TWAP ring buffer | `poc-core` | `BTreeMap`, `VecDeque`, `String` provider names |
| `poc-economics` | ~2,650 | Fee ladder + rebates + volume ledger, funding (BitMEX premium+interest), 60/30/10 revenue router, reward pool, vaults, MM tiers, referral | `poc-core` | `HashMap` in `FeeSchedule` (`economics/src/fees.rs:113`); otherwise pure integer |
| `poc-margin` | ~2,440 | Positions/accounts, SFPM scenario portfolio margin, Black-Scholes, American (BAW/CRR/Merton) | `poc-core` | **`f64` analytics** in `blackscholes.rs`, `american.rs`, `portfolio.rs` (legs/scan in f64), `Mark::Option { iv: f64, tau_years: f64 }` |
| `poc-volsurface` | 1,240 | Governed IV surface: anchor → blend (EWMA) → clamp | `poc-core`, `poc-margin` | f64 IV inversion via `implied_vol`; integer EWMA blend core is fine |
| `poc-risk` | ~1,620 | Pre-trade gates, liquidation planner + queue + insurance fund, ADL, greeks caps | `poc-core`, `poc-margin` | `BinaryHeap`, `BTreeMap`, f64 in margin context |
| `poc-rfq` | 2,340 | RFQ multi-leg quotes, block trades | `poc-core` | `BTreeMap`/`Vec`; not needed in a proving workload |
| `poc-engine` | ~9,000 | The sequencer: `plan` (pure) → events → `apply_event` (single mutator), tick sweep, exercise, amend, auctions, collateral, MMP | core, rfq, volsurface, orderbook, oracle, economics, margin, risk | `BTreeMap` everywhere, `Vec<Event>` journal, `Box`ed events, `f64` config (`option_ivs`, `risk_free_rate`, `EngineConfig` `engine/src/engine.rs:69,73`) |
| `poc-settlement` | ~2,430 | SHA-256 merkle account tree, state-diff batches, withdrawal exit, proof-of-reserves, **provable book hash chain** | `poc-core`, `poc-engine`, `poc-orderbook` | SHA-256 implementation itself (`settlement/src/hash.rs`, 211 lines) — port it, it is pure integer |
| `poc-persist` | ~2,300 | LEB128 command codec, framed CRC chain-hashed WAL | `poc-core`, `poc-engine` | Codec is no_std-friendly already (pure `&[u8]`); I/O not needed |
| `poc-api` | ~2,720 | Hand-rolled JSON enum, FIX 4.4 codec + session state machine, auth nonces, token buckets | `poc-core`, `poc-engine` | Not needed in the guest (transport only) |
| `poc-governance` | 448 | Weighted multisig → timelock proposals | `poc-core` | Not needed |
| `poc-bench` | 998 | Deterministic micro/stress benchmarks | many | Host-side only; gives us measured perf anchors |
| `poc-demo` | 628 | Scripted end-to-end session | many | Host-side only |

**Hard blockers for a no_std/alloc-free port, ranked:**

1. **`f64` in the margin/analytics path** (`margin/src/blackscholes.rs:1-2`
   "f64 — never posted to a ledger", `margin/src/american.rs:1`,
   `margin/src/portfolio.rs:91-106` legs are `f64`, engine `tau_years`
   `engine/src/engine.rs:3579`). The *ledger* is 100% integer (`u128`/`i128`
   minor units through `mul_div` — `core/src/num.rs:31`); only scenario
   repricing and marks touch floats. → replace with fixed-point (§4, §8).
2. **Heap collections**: `BTreeMap` (levels, orders, accounts, positions,
   instruments), `VecDeque` (channels, mark samples, TWAP windows),
   `Vec` (fills, events, legs), `HashMap` (fee volume ledger). → static
   arrays with capacity + insertion-order semantics (§8).
3. **`String` symbols / provider names** everywhere (`core/src/types.rs:15`).
   → u16 instrument index (guest has ≤256 instruments; deterministic).
4. `Box<Event>`/`format!`-based errors → fixed discriminants, error codes.
5. `BinaryHeap` liquidation queue (`risk/src/liquidation.rs:117`) → static
   array + selection sort (deterministic, O(n²) but n ≤ 64).

Everything else — `mul_div` rounding discipline, position math, funding
math, fee math, matching, auction uncross, book hash chain — is already
exact integer arithmetic and ports 1:1.

---

## 2. The CLOB (central limit orderbook)

### 2.1 Data structures

`LimitOrderBook` (`orderbook/src/lib.rs:312-324`):

```rust
pub struct LimitOrderBook {
    symbol: Symbol,
    asks: BTreeMap<u64, ChannelQueue>,        // price ascending; first = best ask
    bids_inverted: BTreeMap<u64, ChannelQueue>, // key = INVERT - price; first = best bid
    orders: BTreeMap<OrderId, RestingOrder>,  // every resting order by id
    auction: bool,
}
const INVERT: u64 = u64::MAX;                 // lib.rs:326
```

Bids are keyed `u64::MAX − price_ticks` so both sides are ascending maps
whose *first* entry is the best quote — no reversed iteration anywhere
(`lib.rs:316-319`, `invert_bid` `lib.rs:328-330`).

Each price level is a **chain of zkLighter channels** (`lib.rs:85-228`):

```rust
pub const CHANNEL_CAPACITY: usize = 8;       // lib.rs:61
pub struct Channel {
    slots: [OrderId; 8],   // slot 0 value = tombstone sentinel; ids start at 1
    live: u8,              // count of non-tombstone slots
    visible_total: u64,    // Σ visible_lots of live slots (cached)
}
pub struct ChannelQueue {
    channels: VecDeque<Channel>,
    total_visible: u64,    // cached level aggregate
    live_orders: u64,      // cached level count
}
```

`RestingOrder` (`lib.rs:64-74`) = the live `Order` + `price_ticks: u64`
(copied for matching) + `visible_lots: u64` (iceberg display slice; plain
orders show full open qty).

Key channel semantics (each is a porting decision point):

- `Channel::push` appends **strictly after the last live slot**, never into
  an earlier tombstone gap (`lib.rs:115-129`) — this is the arrival-order
  invariant that makes price-time priority and iceberg re-queueing correct.
- `Channel::remove` tombstones the slot and subtracts visible qty
  (`lib.rs:132-141`); an emptied channel is dropped by
  `ChannelQueue::remove`'s `retain` (`lib.rs:182-195`).
- `adjust_visible` shrinks a partially-filled order's cached contribution
  **in place** — queue position is never changed by a partial fill
  (`lib.rs:200-211`).
- Order representation, price/qty: `Order` has 18 fields
  (`core/src/types.rs:150-196`): `id: u64`, `subaccount: u64`, `symbol:
  String`, `side: Side{Bid,Ask}`, `order_type` (`Limit`, `Market`,
  `StopMarket{trigger}`, `StopLimit{trigger,limit}`, `TrailingStopMarket{offset_ticks}`,
  `TrailingStopLimit{...}` — `types.rs:50-84`), `price_ticks: Option<u64>`,
  `qty_lots: u64`, `filled_lots: u64`, `tif` (`Gtc/Ioc/Fok/Gtd(ts)`
  `types.rs:115-125`), `post_only: bool`, `reduce_only: bool`, `stp`
  (`CancelNewest|CancelOldest|CancelBoth|DecrementAndCancel`
  `types.rs:131-144`), `display_lots: Option<u64>` (iceberg),
  `trailing_extreme_quote_minor: Option<u128>`, `oco_group: Option<u64>`,
  `client_ts`, `engine_ts`.

**Price/qty representation — all integers:**

- Price = integer **ticks**; money conversion `price_quote_minor = ticks ×
  tick_size_quote_minor` (`u128`, checked — `core/src/instrument.rs:402-404`).
- Qty = integer **lots** (`u64`).
- Notional = `(ticks × tick_size) × (lots × lot_size) / 10^base_decimals`
  via `mul_div(..., Rounding::NearestHalfUp)` (`instrument.rs:421-430`).
- BTC-PERP defaults: quote_decimals 2, base_decimals 5, tick 100 (=$1),
  lot 100 (=0.001 BTC) (`instrument.rs:66-83`).

### 2.2 The matching engine — exact transcription

`match_taker` is **pure** — reads book, returns `MatchOutcome { fills:
Vec<Fill>, stp: StpEffects, taker_remaining_lots }`, mutates nothing
(`lib.rs:802-921`). `Fill` = `{taker_order_id, maker_order_id,
taker_subaccount, maker_subaccount, maker_side, price_ticks, qty_lots}`
(`lib.rs:231-247`) — **fills always execute at the maker's price**
(`lib.rs:49-50`).

The exact loop (`lib.rs:818-921`), transcribed for the port:

```
match_taker(taker, price_limit, fok, stp):
  outcome.taker_remaining_lots = taker.open_qty()            // qty_lots - filled_lots, sat-sub
  if fok:
      reachable = available_within(taker.side, price_limit)  // O(levels within limit),
      if reachable < taker.open_qty(): return outcome        //   from cached level aggregates (lib.rs:603-629)
  levels = (taker.side == Bid) ? asks.iter() : bids_inverted.iter()   // ascending key = best first
  'levels: while let Some((key, queue)) = levels.next():
      price = (Bid taker) ? key : INVERT - key
      if price_limit is Some(limit):
          if taker.side == Bid  and price >  limit: break 'levels
          if taker.side == Ask  and price <  limit: break 'levels
      channel_iter = queue.iter_live()                       // channels in arrival order,
      while let Some(maker_id) = channel_iter.next():        //   tombstones skipped lazily
          if outcome.stp.taker_canceled: break 'levels
          if outcome.taker_remaining_lots == 0: break 'levels  // lazy: stop the moment taker fills
          r = orders.get(maker_id) or continue               // stale slot: skip
          (maker_open, maker_price, maker_sub) = (r.visible_lots, r.price_ticks, r.order.subaccount)
          if maker_sub == taker.subaccount:                  // ---- self-trade prevention
              match stp:
                CancelNewest:  stp.taker_canceled = true; break 'levels      // taker dies, maker rests
                CancelOldest:  stp.canceled_makers.push(maker_id); continue  // maker dies, keep walking
                CancelBoth:    stp.canceled_makers.push(maker_id); stp.taker_canceled = true; break 'levels
                DecrementAndCancel:
                    overlap = min(taker_remaining_lots, maker_open)
                    stp.decrements.push((maker_id, overlap))
                    stp.taker_consumed_lots += overlap
                    taker_remaining_lots -= overlap; continue
          fill_qty = min(outcome.taker_remaining_lots, maker_open)
          if fill_qty == 0: continue
          outcome.fills.push(Fill { taker id/sub, maker id/sub,
                                    maker_side: taker.side.opposite(),
                                    price_ticks: maker_price,        // maker's price
                                    qty_lots: fill_qty })
          outcome.taker_remaining_lots -= fill_qty
  return outcome
```

**Crossing detection** is implicit: a limit bid with `price ≥ best_ask`
walks the ask map and matches while `price ≤ limit`; a crossing order
never rests (the engine matches first, rests remainder). `would_cross`
(`lib.rs:396-401`) implements `bid: price ≥ best_ask` / `ask: price ≤
best_bid` for post-only enforcement. **Price-time priority** = map key
order (price priority) × channel slot arrival order (time priority).
**Partial fills** fall out of `min(taker_remaining, maker_open)`.
**Fills are emitted in execution order** (best price first, FIFO within
level). Book invariant: never crossed outside auction mode (`lib.rs:306-311`).

**Applying the outcome** (the only mutation path — same one replay uses):

- `apply_fill(fill)` → `reduce(maker_order_id, fill.qty_lots)`
  (`lib.rs:555-557`). `reduce` (`lib.rs:495-549`): bumps `filled_lots`
  (saturating), cancels at zero; plain orders shrink `visible_lots` to the
  new open qty and call `adjust_visible`; **icebergs whose visible slice is
  exhausted** re-reveal `min(open, display).max(1)` and *re-queue at the
  back of the level* (`queue.remove` + `queue.push`, `lib.rs:520-529`) —
  the Deribit "revealing size joins the queue anew" rule.
- `cancel(id)` (`lib.rs:471-484`): removes from `orders`, tombstones in the
  channel, drops the level if it empties.

### 2.3 Order lifecycle (engine side)

`plan_place` (`engine/src/engine.rs:1140-1304`) — validation cascade:
instrument/account exist → `validate_qty` (positive, ≤ max_order_lots) →
portfolio greeks caps (options only) → price on tick grid → reduce-only cap
(`risk/src/pretrade.rs:315-325`: cap at current position, 0 → reject) →
`build_marks` (oracle-anchored; missing mark → reject) → trailing anchor →
MMP freeze check → estimated fee → `poc_risk::check_order` full gate →
parked stop orders rest off-book → auction-mode orders rest unmatched with
reservation → otherwise `plan_match(order, now, reservation)`.

`plan_match` (`engine/src/engine.rs:1309-1526`) — for each fill: notional
(`mul_div` half-up), taker/maker fee tiers (`fee_schedule.tier_for`),
option fee caps `min(bps × underlying_notional, cap% × premium)`
(taker 12.5% / maker 2.5% of premium — `economics/src/fees.rs:49-68`),
MM-tier discount, `TradeExecuted` event, `OrderClosed` for fully-filled
makers; STP cancels event; MMP trip scan; taker lifecycle: remaining 0 →
`OrderClosed(Filled)`; GTC/GTD remainder → `OrderResting(reservation)`;
else → `OrderClosed(IocRemainder)`.

`apply_event` (`engine/src/engine.rs:1616+`): `OrderResting` →
`book.insert_resting` + `account.track_order` + reservation bump
(`engine.rs:1705-1736`); `OrderClosed` → `book.cancel` + untrack +
reservation release (`engine.rs:1737-1766`); `TradeExecuted` →
`apply_trade` (`engine.rs:2601-2731`): `book.apply_fill` (maker), both
accounts `apply_fill` + `apply_fee`, fee-volume ledger (day buckets),
MMP accumulation, revenue routing (60% house / 30% insurance / 10% buyback
— `economics/src/revenue.rs`), **reservation scaling** `reserved ×
remaining/before` floor (`engine.rs:2694-2718`), mark sample push for
funding TWAP.

**Amendment** (`engine/src/amend.rs:1-17`): price change or size increase
→ **cancel-and-replace** (new id, back of level — no queue privilege
survives); size reduction at same price → in-place `OrderAmended`, queue
priority kept. Batch placement = atomic validity gate + cumulative
order-margin simulation, sequential matching against the pre-batch book
(`amend.rs:19-29`, `engine.rs:599-608`).

**Auction uncrossing** (`orderbook/src/lib.rs:632-800`): prefix-sum
supply/demand curves; `bid_qty_at(p)`/`ask_qty_at(p)` via
`partition_point` binary searches (`lib.rs:711-728`); candidate prices =
all resting levels, sorted+deduped; clearing price maximizes
`min(bid_qty_at(p), ask_qty_at(p))`, ties toward indicative mid then lower
price (`lib.rs:734-747`); pairing walk matches price-time-priority bids
(desc) against asks (asc) at the uniform price, later-arriving order
(higher id) reported as taker (`lib.rs:765-796`). Measured: 10k orders /
121 levels uncross in 9.1 ms native (`docs/BENCHMARKS.md:78`).

---

## 3. Perpetuals

### 3.1 Position representation

`Position` (`margin/src/account.rs:17-28`):

```rust
pub struct Position {
    pub symbol: Symbol,
    pub signed_lots: i64,                  // >0 long, <0 short
    pub avg_entry_quote_minor: u128,       // VWAP entry, quote-minor per 1.0 base
    pub realized_pnl_quote_minor: i128,    // lifetime, signed
}
```

`MarginAccount` (`account.rs:174-194`): `cash_quote_minor: i128`
(signed — negative only transiently in bankruptcy), `positions:
BTreeMap<Symbol, Position>` (flat positions pruned, `account.rs:264-268`),
`order_margin_quote_minor: u128`, `open_orders: BTreeMap<OrderId,
OpenOrderInfo>`, `fees_paid`, `funding_pnl`.

### 3.2 Fill application / PnL (exact arithmetic)

`Position::apply_fill(instrument, side, qty_lots, price_quote_minor) -> i128`
(`account.rs:59-119`) — one algorithm for perps and premium-unpaid options:

1. `old_qty == 0` → **open**: `avg_entry = price`, return 0.
2. same direction → **extension**: VWAP update
   `avg = (|old|×avg + qty×price) / (|old|+qty)` (integer div,
   `account.rs:84-96`), return 0.
3. opposite → **reduction (possibly with flip)**: convert prices to
   per-lot (`quote_per_lot = price × lot_size / 10^base_decimals` half-up,
   `account.rs:157-171`), `reduced = min(|old|, qty)`,
   `realized = (fill_per_lot − entry_per_lot) × reduced` for longs,
   `(entry_per_lot − fill_per_lot) × reduced` for shorts (`account.rs:102-109`);
   flip carries the new-side portion at the fill price; flat → entry zeroed.

Entries never touch cash; only realized PnL moves cash
(variation-margin convention, `account.rs:245-270`). Equity identity
`equity = cash + Σ uPnL` holds by construction (`account.rs:5-10`).

**Unrealized PnL** (`instrument.rs:455-483`):
`pnl = sign(qty) × |qty| × lot_size / 10^d × (mark − entry)` — computed as
`mul_diff` per-base then sign, `Rounding::NearestHalfUp`. Position notional:
`mark × |lots| × lot_size / 10^d` (`instrument.rs:435-448`).

### 3.3 Funding

BitMEX premium + interest (`economics/src/funding.rs:69-96`):

```
premium_bps_raw = (mark_twap − index_twap) × 10_000 / index_twap     // i128
premium_bps     = clamp(premium_bps_raw, ±premium_clamp_bps=5)
rate_bps        = clamp(premium_bps + interest_bps(=1), ±rate_cap_bps=75)
```

Payment per lot (`funding.rs:108-129`): `notional_lot = ceil(mark ×
lot_size / 10^d)`; `per_lot = ceil_ceil(notional_lot × |rate| / 10_000) ×
sign(rate)` — magnitude rounds **up** (payer's expense). Sweep iterates
all accounts: `credit = −(lots × per_lot)`; longs pay shorts; **exactly
zero-sum** (`sweep.rs:526-538`, test `funding_conserves_over_closed_positions`
`funding.rs:229-235`). Mark TWAP from BBO-mid/impact-price samples pushed
on every trade + oracle tick (`engine.rs:3092-3149`); index TWAP from the
oracle's ring buffer (max 3,600 samples, `oracle/src/lib.rs:39`).

### 3.4 Margin requirements

Two-layer system:

- **Per-market ratios** (floors): `initial_margin_ratio_bps = 500` (5%),
  `maintenance_margin_ratio_bps = 375` (3.75%) on `PerpMarket`
  (`instrument.rs:54-57`) — these feed the SFPM scan range default.
- **SFPM portfolio margin** (`margin/src/portfolio.rs`): maintenance =
  worst-case loss of the whole book under a **scenario grid** —
  `scan_range_bps = 375`, `initial_multiplier_pct = 140`,
  `vol_shift_pct = 25` (`portfolio.rs:30-39`). The grid is 6 spot scales ×
  3 vol scales = **18 scenarios** (`portfolio.rs:319-346`): spot shocks
  `{−1.0, −0.5, −0.25, +0.25, +0.5, +1.0} × scan`, vol shocks `{−1, 0,
  +1} × vol_shift`. Perp leg: `pnl += signed_base × (spot_shock × spot)`.
  Option leg: full BS/BAW **reprice** at shocked spot/iv
  (`OptionLegView::reprice`, `blackscholes.rs:313-318`). Maintenance =
  `half_up(worst_loss + SOMC)`; initial = `maintenance × 140%`
  (`portfolio.rs:364-366`). SOMC = short-option minimum charge
  `short_base × spot × somc_bps/10_000` (default 500bps = 5% of spot per
  base shorted, `instrument.rs:178`, `portfolio.rs:350-362`).
- Health classification (`account.rs:358-372`): `equity < maintenance` →
  Liquidation; `< initial` → Restricted; else Healthy.
  `available = equity − initial − order_margin` (`account.rs:341-343`).

### 3.5 Liquidation

Planner is pure (`risk/src/liquidation.rs:272-296`): candidates =
accounts with `equity < maintenance`, ranked by deficit
(`maintenance − equity`, ties by sub id — `liquidation.rs:91-106`). Plan
strategy (`liquidation.rs:274-287`): rank underlyings by maintenance
contribution; within an underlying close by |notional| desc; close legs at
**penalized prices** — longs sell at `mark × (1 − p)`, shorts buy at
`mark × (1 + p)`, p = 125 bps (`liquidation.rs:32-39`); partial-first via
bisection to the smallest fraction that restores `maintenance × 1.2`
(restoration_buffer_bps 2000); bankrupt flag when emptied below the line.

Engine execution (`sweep.rs:771-1128`): **Phase A** — cross the book as an
aggressive IOC limit taker at the penalized price (fills only at better
prices, `sweep.rs:884-920`); **Phase B** — remainder to the insurance fund
as buyer of last resort at the penalized price (position booked into
insurance inventory, G-23), and when the fund cannot absorb the deficit,
**iterative ADL** (`adl_max_rounds = 4`, `adl_round_bps = 5000` — 50% per
round, `engine.rs:175-176`) force-closes the most profitable
counterparties at the bankruptcy price (`sweep.rs:923-960`,
`bankruptcy_price` `sweep.rs:1153`). Liquidation pulls the account's
resting orders on that instrument (`engine.rs:2750-2766`). Penalty flows
to the insurance fund; absorbed shortfalls debit it (`liquidation.rs:202-243`).

---

## 4. Options

### 4.1 Premium / pricing math

`OptionMarket` (`instrument.rs:219-254`): `kind: Call|Put`,
`strike_quote_minor: u128`, expiry (dated) or everlasting variant,
`exercise_style: European|American`, tick/lot sizes, margin params
(SOMC 500bps, liquidation fee 125bps). Option quantity = lots of
`lot_size_base_minor` base units; premium quoted in ticks of
`tick_size_quote_minor` per 1.0 base.

- **Marks are oracle-anchored, never book-derived** (README:101-109):
  perp marks at oracle spot; option marks = theoretical value at oracle
  spot with the governed surface's IV (`engine.rs:2918-2960`).
- **European: Black-Scholes in `f64`** (`margin/src/blackscholes.rs:70-93`):
  `d1 = (ln(S/K) + (r + σ²/2)τ) / (σ√τ)`, `d2 = d1 − σ√τ`, call =
  `S·N(d1) − K·e^(−rτ)·N(d2)`. Norm-CDF via **Abramowitz–Stegun 7.1.26
  erf approximation** (|ε| < 1.5e-7, 5 Horner coefficients,
  `blackscholes.rs:52-62`). Degenerate τ≤0 or σ≤0 collapse to discounted
  intrinsic (the scenario grid probes σ=0 vol crush). Native cost: 81 ns
  (`docs/BENCHMARKS.md:80`).
- **American: Barone-Adesi & Whaley quadratic approximation**
  (`margin/src/american.rs:98-157`): early-exercise premium
  `A·(S/S*)^q` added to the European value; optimal boundary `S*` solved
  by **bracketed bisection, 200 iterations** + geometric bracket
  expansion up to 40 doublings (`american.rs:179-247`). CRR binomial
  referee (400 steps, O(steps²)) and Merton perpetual closed form also
  provided (`american.rs:287-340`). Native: 10.7 µs at r=3%, **81 ns at
  the venue-default r=0** where BAW ≡ European exactly (asserted by test).
- **Implied vol inversion** (surface): Newton–Raphson 50 iters with
  bisection fallback 100 iters (`blackscholes.rs:198-254`).
- Zero-rate default (`risk_free_rate = 0`, `engine.rs:167`): American ==
  European; the whole option analytics stack collapses to one BS eval.

**The entire option pricing path is `f64` and must be replaced by
fixed-point in the guest.** The ledger side is integer (§4.2).

### 4.2 Payoff / exercise / settlement

- **Intrinsic (integer)**: `option_intrinsic_quote_minor = mul_div(max(0,
  S−K) or max(0, K−S), lot_size, 10^d, NearestHalfUp)`
  (`instrument.rs:488-501`) — cash-settled, per lot.
- **Expiry settlement** (`sweep.rs:615-675`): strike = **30-minute TWAP**
  ending at expiry (oracle ring buffer; fail-safe: no settlement without
  a defensible price). Per account: `payout = intrinsic_per_lot ×
  signed_lots`; `OptionExpiry` event applied as a forced close at the
  terminal per-base value through the same `account.apply_fill`
  (`engine.rs:1816-1850`) — realized PnL lands in cash, position
  disappears, instrument delists.
- **American early exercise** (`engine/src/exercise.rs`): holder tenders
  any portion of a long position → request parks for a TWAP window
  (30 min default, 5 bp fee on intrinsic proceeds) → settles at window
  TWAP intrinsic → matching short positions assigned **pro-rata with
  exact largest-remainder distribution** (`exercise.rs:183-219`):
  `base_i = floor(l_i × E / Σl)`, `rem_i = (l_i × E) mod Σl`, rank by
  rem desc then sub id, hand out the residual lots one each — `Σ assigned
  == settled` exactly (zero-sum ledger through settlement). Exercise fee
  routes through the 60/30/10 revenue router (`engine.rs:2868-2889`).
- **Option fees**: `min(bps_rate × underlying_notional, 12.5% × premium)`
  takers / `2.5%` makers (`fees.rs:16-30`, applied in `engine.rs:1348-1379`).

---

## 5. Risk / margin engine (aggregate view)

`PortfolioMarginEngine::margin_summary_ex` (`portfolio.rs:168-215`) is the
single aggregate entry point used by *pre-trade gates, order-margin
reservation, withdrawal checks, health classification, and liquidation*:

1. **Equity** (integer-exact): `cash + Σ unrealized_pnl(position, mark)`
   over all positions (`portfolio.rs:175-192`); + haircut collateral
   equity (pre-trade only, `pretrade.rs:263-265`).
2. **Maintenance/initial**: per-underlying SFPM scan (`scan_underlying_ex`
   `portfolio.rs:234-371`), summed across underlyings with **no
   cross-underlying offsetting** (SPAN additive). Each scan flattens
   positions into legs (perp: signed_base f64; option: leg view with BS/BAW
   reprice), evaluates the 18-scenario grid, adds the SOMC floor, converts
   back to u128 half-up, scales initial ×140%.
3. Spot-hedge collateral enters the grid as a linear perp-equivalent leg
   (G-20, `portfolio.rs:231-249`).

**Pre-trade gate** (`risk/src/pretrade.rs:164-276`) runs, in order: halted
→ price band `|price − mark| × 10_000 > mark × band_bps` (1,000 bps perps /
2,000 bps options) → post-only would cross (BBO compare) → reduce-only
semantics → position cap (10,000 lots) → open-order cap (200) → **margin
gate: clone account, apply the *hypothetical full fill at the order's own
limit price*, apply worst-case fee, re-run the full margin summary, reject
if `available < 0`**. `order_margin_increment` (`pretrade.rs:285-309`) =
`initial(after) − initial(before)` — two full margin scans per placement.

Greeks caps (vega/gamma portfolio limits, `risk/src/greeks_limits.rs`) are
disabled by default. Circuit breakers (G-21) and cascade-velocity
suspension gate the tick sweep (`sweep.rs:232-291`).

---

## 6. Serialization

- **No serde anywhere.** Commands use a **custom binary codec**
  (`persist/src/codec.rs:1-16`): LEB128 varints for every integer,
  length-prefixed strings/vectors, one-byte enum tags, **total** (a buffer
  either decodes to exactly one command consuming the whole buffer or
  fails). E.g. `encode_perp` (`codec.rs:42-57`) writes 13 varint fields.
  The WAL wraps frames with length + CRC + chain hash (`persist/src/wal.rs`).
- **Events are not serialized** in-process — the journal is a
  `Vec<Event>` (cloned, boxed variants); replay uses identical types
  (`engine.rs:253`). Bit-exact replay is asserted by tests
  (`engine::tests::determinism_and_replay`).
- **Settlement commitments**: account merkle tree + state-diff batches
  over SHA-256 leaf hashes (`settlement/src/merkle.rs:20-58`); the
  **provable book state** (`settlement/src/books.rs:122-134`) is a
  domain-separated SHA-256 hash chain over every resting order in
  canonical (symbol, order-id) order:
  `root₀ = H(LINK ‖ H(leaf₀))`, `root_{i+1} = H(LINK ‖ root_i ‖
  H(leaf_{i+1}))`, leaf = `H(LEAF ‖ sym_len ‖ sym ‖ order_id ‖ subaccount
  ‖ is_bid ‖ price_ticks ‖ open_lots ‖ visible_lots)` all big-endian
  (`books.rs:56-70`). Measured 1.2 ms per 1,000 resting orders native
  (`docs/BENCHMARKS.md:83`). **This is the natural public-output
  commitment for the guest.**
- API layer: hand-rolled JSON (`api/src/json.rs`) and FIX 4.4 codec
  (`api/src/fix.rs`) — transport-only, not ported.

---

## 7. Workload characterization for proving

### 7.1 Dominant computation loops (N-order batch, book depth D)

For each order in a batch of N against a book of depth D (resting orders,
L levels per side):

| Phase | Loop shape | Count |
|---|---|---|
| Validation + tick-grid check | O(1) per order | N |
| Mark build | O(#instruments on underlying); perp O(1), each option = 1 BS eval | N (re-built per place in `plan_place:1206`; host could cache) |
| Pre-trade margin gate | 2 × full SFPM scan = 2 × (P positions × 18 scenarios × per-leg reprice) + account clone | N |
| `match_taker` | O(levels crossed) map walk × O(channels/level) × ≤8 slot probes + O(1) per fill; FOK adds O(L) aggregate sum | ≤ N × D |
| Fill post-processing | per fill: notional mul_div, 2 fee mul_divs, tier lookups, 2 account apply_fills, volume ledger, revenue split | F total fills |
| Book apply | per fill: order lookup + channel ops (≤8 slot scan ×2) | F |
| Position update | per fill: VWAP mul+div or realized-PnL mul | F |
| Reservation scaling | per fill: mul_div | F |
| Liquidation sweep (per tick) | accounts × margin scans + book IOC crosses + ADL rounds | bursty |

### 7.2 Instruction mix (RV64IMAC)

- **Integer mul/div heavy.** Every notional, fee, PnL, funding payment,
  reservation scale is a `u128 mul_div` (checked mul + div + rem +
  rounding branch — `core/src/num.rs:31-64`). On RV64: `u128×u128` =
  `mul`+`mulh` pair (2 instr), `u128/u128` = compiler-rt `__udivti3`
  (~50-150 instructions, or restructure to keep divisors ≤ u64 so it is
  one `divu`). i64 signed arithmetic for positions/PnL (`mul`+`mulh`,
  sign fixups).
- **Branch heavy.** Side dispatch, price-band compares, level-walk loop
  conditions, STP policy match (4 arms), TIF/post-only/reduce-only gates,
  health classification, rounding-direction selects. The matching loop is
  a data-dependent nested loop (2-3 levels deep with early exits).
- **Memory access: mixed random/sequential.** Random: `orders.get(id)`
  (BTreeMap, O(log n) pointer chase — in the guest becomes O(1) array
  index), level lookup by price (BTreeMap → guest: sorted array, binary
  or linear scan), account/position lookup by id. Sequential: channel
  slot scan (8 × u64 contiguous), fills/events append, scenario-grid leg
  iteration, SHA-256 chain over canonical order.
- **No floats anywhere on the ledger path** — the only f64 is analytics
  (BS/BAW/IV-solve/tau), which becomes fixed-point ALU work in the guest
  (polynomial Horner + Newton sqrt + range reductions).

### 7.3 Dynamic instruction count per order (order-of-magnitude, justified)

Anchors: native p50 = 485 ns for a lazy taker vs a 2,000-order book;
539 ns place/cancel churn; 700 ns-1.2 µs tick sweep at low load; a full
place incl. risk+journal ≈ 0.5-0.7 µs (`docs/BENCHMARKS.md:20-29, 77-83`).
At ~3 GHz with IPC 2-3 that is ~1,500-2,000 cycles ≈ **3,000-6,000
retired instructions natively per full order placement**. A zkVM guest
(rv64i interpreter semantics, no OOO, flat arrays instead of BTreeMap but
with bounds checks and explicit u128 helpers) realistically runs 5-20×
more instructions; add margin-scan work that native amortorizes via f64:

| Workload | Est. dynamic instr / order | Dominated by |
|---|---|---|
| Perp-only place, no cross (rest) | ~10-20 k | margin gate (2 scans × 18 scenarios × P legs), VWAP, fees |
| Perp taker, k fills (k ≤ 16) | ~15-40 k | match walk + k × (fees + 2 apply_fills + book ops) |
| Option mark/pre-trade (1 option leg, fixed-pt BS) | +30-80 k per BS reprice set | 18 scenarios × BS × 2 scans ≈ 36 BS evals; each fixed-pt BS ≈ 800-2,000 instr (erf poly, exp, ln, sqrt) |
| American BAW (r>0, bisect boundary) | +100-300 k | 200-iteration bisection × residual (each = 1 BS + erf) |
| SHA-256 book commitment | ~1-2 k instr / resting order | 2 compressions per leaf (~64-byte leaf + link chain) |

**Recommended benchmark shape: a perp-only batch** (N=256 orders,
D=512 resting, 8-16 accounts, ≤8 fills/order) lands around
**5-15 M dynamic instructions ≈ T ≈ 10^7 trace steps** — squarely in
lzx's wheelhouse. Option-heavy workloads multiply T by 10-50× via
fixed-point BS repricing; do them as a second-stage guest.

### 7.4 What makes this workload *good* for a zkVM

- Deterministic, event-sourced, bit-exact replay by construction
  (pure `match_taker` + single `apply_event` mutator) — the guest program
  *is* the audit.
- Bounded per-step matching work (8-slot channels) — no unbounded inner
  loop inside one match step (`orderbook/src/lib.rs:26-44`).
- Everything integer-exact on the ledger → no fixed-point soundness
  pitfalls in money movement; only analytics need care.
- The book hash chain gives a natural public-output commitment.

---

## 8. Port plan (guest program)

### 8.1 What to port, from where

| Guest module | Ported from | Changes |
|---|---|---|
| `num.rs` (mul_div, apply_bps, Rounding) | `core/src/num.rs:14-98` | keep u128 (Rust u128 → `mul/mulh` + soft div); Option→explicit error codes; no_std identical |
| `instr.rs` (static instrument table: tick, lot, decimals, band, ratios, somc, fee caps) | `core/src/instrument.rs` (only the fields used: `tick_size`, `lot_size`, `base_decimals`, `price_band_bps`, `max_order_lots`, margin ratios) | `String`→u16 index; drop `validate_qty`'s `format!` → error codes |
| `order.rs` (`Order`, `Side`, `TimeInForce`, STP; **Limit+Market only**) | `core/src/types.rs` | drop stop/trailing/OCO/TWAP/iceberg fields (see §8.2); fixed 48-64 B record |
| `book.rs` (LimitOrderBook + Channel + ChannelQueue + match_taker + insert/cancel/reduce + apply_fill + invariants) | `orderbook/src/lib.rs` | BTreeMap→static arrays (§8.3); `Vec<Fill>`→fixed array; `VecDeque`→ring of channels; macro→two explicit fn copies (bid/ask) |
| `account.rs` (Position, MarginAccount, apply_fill, quote_per_lot, Health) | `margin/src/account.rs` | BTreeMap→static arrays |
| `margin.rs` (SFPM scan, margin_summary) | `margin/src/portfolio.rs` | f64 legs → i128/Q-fixed scan for perp-only stage; option stage adds fixed-point BS legs |
| `bs.rs` (fixed-point BS + erf + exp/ln/sqrt) | `margin/src/blackscholes.rs` | Q48.64 or Q32.32 fixed point; A&S erf poly scales exactly; degenerate branches keep semantics |
| `fees.rs` (FeeSchedule ladder, taker/maker fee, option caps, tier discount) | `economics/src/fees.rs:35-209` | HashMap volume ledger → static array or drop tiers to tier-0 (see §8.2) |
| `funding.rs` (FundingCalculator::perp_funding, payment_per_lot) | `economics/src/funding.rs:60-129` | ports 1:1 (pure integer) |
| `risk.rs` (check_order gates, reduce_only_cap, order_margin_increment) | `risk/src/pretrade.rs:164-325` | clone→copy of static account struct; BTreeMap marks→array |
| `liq.rs` (planner: penalized price, ranking; queue → selection sort) | `risk/src/liquidation.rs:88-165, 272-386` | BinaryHeap→sorted static array; bisection loop fixed 32 iterations |
| `sha256.rs` + `bookhash.rs` | `settlement/src/hash.rs`, `settlement/src/books.rs:38-134` | ports nearly 1:1; leaf layout identical (big-endian fields) so roots match the host implementation |
| `engine.rs` (plan_place/plan_match/apply_event/apply_trade, one instrument, no tick sweep or a single funding+liquidation stage) | `engine/src/engine.rs:1140-1526, 1616-1835, 2601-2731` | Vec<Event>→fixed event ring or direct apply (guest needs no journal: apply as you go, hash committed outputs) |

**Explicitly not ported** (host-side concerns): `poc-api` (JSON/FIX),
`poc-persist` (WAL), `poc-governance`, `poc-rfq`, `poc-settlement`
(merkle/por/batch/exit — except `hash.rs`+`books.rs`), `poc-volsurface`
(IV is a public input in the guest), oracle aggregation (mark/spot is a
public input), `poc-demo`/`poc-bench`.

### 8.2 Simplifications and why they preserve fidelity

1. **Limit + Market orders only** (drop stops/trailing/OCO/TWAP/iceberg).
   These add parked-order state machines orthogonal to the matching/margin
   core; the *matching*, *PnL*, *fee*, and *margin* semantics exercised by
   a proving workload are fully covered by limit/market + GTC/IOC/FOK +
   post-only/reduce-only + the 4 STP modes (STP is cheap — keep all four;
   it is the only intra-loop policy branch).
2. **Single instrument per guest** (or one perp + one option sharing an
   underlying). Cross-underlying margin summation is a trivial loop;
   keeping one underlying exercises the full SFPM netting logic (perp +
   option legs, SOMC, 18-scenario grid) without a symbol table.
3. **Marks as public input.** Oracle defense (median/quorum/quarantine) is
   host-side; the guest receives `(spot, iv_bps)` as verified public input
   and computes perp mark = spot, option mark = fixed-point BS — exactly
   what `build_marks` does with a healthy oracle. Funding index/mark TWAPs
   become public-input arrays (the guest just consumes the samples).
4. **No journal.** The engine's plan/apply split exists for *replay
   auditability*; in a zkVM the proof *is* the audit. Apply events
   immediately; commit outputs via the book hash chain + trade list +
   account digest (same leaf encodings as `settlement/books.rs` so host
   cross-checks bit-for-bit).
5. **Fee tiers → tier 0 (or a 2-tier ladder).** The volume ledger
   (day-bucketed HashMap) is bookkeeping, not trading semantics; one tier
   preserves the fee *formula* (`apply_bps` ceil/floor asymmetry) and the
   option premium caps. The 60/30/10 revenue router is 2 mul_divs — keep
   it; conservation is a nice proven invariant.
6. **Fixed-point BS replacing f64.** Keep the exact same formula *shape*
   (d1/d2, erf A&S 7.1.26 with the same 5 coefficients scaled to Q-format,
   degenerate collapses to intrinsic). IV solver → **bisection with a
   fixed 40-iteration budget** (drop Newton; deterministic iteration
   count, same bracket [1e-6, 5.0]). BAW (if the American guest is built)
   → fixed 64-iteration bisection on the same residual. Semantic fidelity:
   the host test suite asserts BAW ≡ BS at r=0 — so at the venue default
   the guest only needs European BS; set r=0 in the benchmark.
7. **u128 kept.** Prices ≤ 2^40 (quote-minor per base for BTC-scale), qty
   ≤ 2^32 lots, products fit u128 exactly like the host; `mul_div` carries
   its rounding modes verbatim, so every ledger value the guest computes
   equals the host's bit-for-bit.

### 8.3 Guest data layout (static capacity, no alloc)

```rust
// ---- capacities (see §8.5) ----
const MAX_ORDERS: usize    = 1024;  // resting orders on the book
const MAX_LEVELS: usize    = 128;   // price levels per side
const MAX_CHANNELS: usize  = 4;     // channels per level (32 orders/level)
const MAX_ACCOUNTS: usize  = 16;
const MAX_POSITIONS: usize = 4;     // per account
const MAX_BATCH: usize     = 256;   // orders in the public-input batch
const MAX_FILLS: usize     = 512;   // total fills emitted
const MAX_TRADES_OUT: usize = 512;

// ---- book ----
struct GuestChannel { slots: [u64; 8], live: u8, visible_total: u64 }        // orderbook lib.rs:93-97
struct GuestLevel  { price_ticks: u64, channels: [GuestChannel; MAX_CHANNELS],
                     n_channels: u8, total_visible: u64, live_orders: u64 }
struct GuestBook   { asks: [GuestLevel; MAX_LEVELS], n_asks: u16,           // ascending
                     bids: [GuestLevel; MAX_LEVELS], n_bids: u16,           // descending by price
                     orders: [GuestOrder; MAX_ORDERS], order_live: [bool; MAX_ORDERS], // dense id → slot (id-1)
                     next_slot: u16 }
struct GuestOrder  { id: u64, sub: u16, side: Side, price_ticks: u64,
                     qty_lots: u64, filled_lots: u64, tif: Tif,
                     post_only: bool, reduce_only: bool, stp: Stp }         // 40 B
// Level insert = shift-insert into the sorted side array (≤ 128 × sizeof(GuestLevel) —
// do it as index move, not memcpy of channels; or store level headers + separate
// channel pool). Orders array is indexed by id-1 for O(1) `orders.get(id)`.
// Channel::push/find-slot semantics copied verbatim (append after last live slot).
```

Matching walks `asks[0..n_asks]` (best first) exactly like the BTreeMap
iterator; bid side walks `bids[0..n_bids]` stored **descending** (the
inversion `INVERT − p` becomes "store descending"), preserving the
"first = best" property with zero key arithmetic.

### 8.4 Public input / output byte regions (flat guest memory)

All integers little-endian (native RV64), fixed offsets, no varints
inside the guest (varint decode costs instructions without benefit when
capacities are static):

```
INPUT  @ 0x1000
  [0x000] header: magic(4) | n_batch(u16) | n_seed_orders(u16) | n_accounts(u16) | flags(u16)
  [0x008] instrument: tick_size(u128) lot_size(u128) base_decimals(u32)
          band_bps(u64) imr_bps(u64) mmr_bps(u64) somc_bps(u64) liq_fee_bps(u64)
          taker_bps(i64) maker_bps(i64) opt_taker_cap_bps(u64) opt_maker_cap_bps(u64)
          scan_range_bps(u64) init_mult_pct(u64) vol_shift_pct(u64) is_option(u8) kind(u8)
          strike(u128) iv_bps(u32) tte_ms(u64) exercise_fee_bps(u64)
  [0x060] spot_quote_minor(u128) | index_twap(u128) | mark_twap(u128)   // oracle outputs
  [0x080] accounts[n_accounts]: { id(u16), cash(i128), order_margin(u128),
          positions[MAX_POSITIONS]: { sym(u8), signed_lots(i64), avg_entry(u128), realized(i128) } }
  [0x080 + A] seed book: asks_levels(u16) bids_levels(u16)
          per level: price_ticks(u64) n_orders(u16)
          per order:  sub(u16) side(u8) price_ticks(u64) qty(u64) visible(u64)
  [ ... ] order batch: n_batch × { sub(u16) side(u8) tif(u8) stp(u8) flags(u8)
          price_ticks(u64) qty_lots(u64) }                                 // 24 B each

OUTPUT @ 0x8000 (or write-back region)
  [0x000] trades_count(u16) × { seq(u32) taker_order(u64) maker_order(u64)
          taker_sub(u16) maker_sub(u16) maker_side(u8) price_ticks(u64)
          qty_lots(u64) notional(u128) taker_fee(i128) maker_fee(i128) }    // 64 B
  [ ... ] position deltas: per touched account × instrument:
          { sub(u16) signed_lots_after(i64) avg_entry_after(u128)
            realized_delta(i128) cash_after(i128) }
  [ ... ] margin flags: per account { equity(i128) maintenance(u128)
            initial(u128) health(u8 liq/restricted/healthy) }
  [ ... ] fills_digest: book hash-chain root [u8;32]   // settlement/books.rs layout
  [ ... ] account_digest + conservation check word (Σ cash + fees + funding == const)
```

The final book root uses the **exact leaf encoding of
`settlement/src/books.rs:56-70`** (big-endian fields, LEAF/LINK domain
strings) so the host `poc-settlement` verifier can cross-check the guest
commitment directly. `lattice-vm/state.rs` memory is a word-addressed
sparse map for the host simulator; the guest linker script should place
`.input`/`.output` at fixed addresses and the bench harness
(`lattice-bench`) copies regions in/out — same pattern as
`lattice-vm/src/state.rs:180` `load_program`.

### 8.5 Recommended capacities for a benchmark guest

| Parameter | Value | Rationale |
|---|---|---|
| Batch orders N | 256 | one zkVM "block"; ~10^7 trace steps at perp-only |
| Book depth D (resting) | 512 (16 levels/side × 32 orders/level = 4 channels) | matches the repo's own bench scenario shapes (200/2,000-order books) |
| Max levels/side | 128 | bounds level-array shifts; deep-enough for realistic books |
| Max orders on book | 1,024 | 2× batch + seed |
| Accounts | 16 | cross-margin netting visible (hedges across accounts) with tiny position tables |
| Positions/account | 4 | perp + call + put + one more |
| Max fills total | 512 | 2/order avg; cap partial-fill fragmentation |
| Channels/level | 4 (32 orders) | keeps per-step work = 8-slot probes × ≤4 |

Memory footprint: orders 1,024 × 40 B ≈ 40 KB + levels 256 × ~140 B ≈
36 KB + accounts 16 × ~160 B ≈ 3 KB + I/O regions ≈ 40 KB → **well under
256 KB of guest RAM**, cache-friendly for the memory argument.

---

## 9. Proving-strategy notes (lzx-specific)

**Where the cost lives, by argument type:**

- **Lookup / range-check heavy:**
  - Every `mul_div` = `u128×u128 → u128 /÷ u128` + rounding select. In a
    circuit-per-step RV64 model these are just more instructions (mul/mulh
    are cheap rows; `divu`/`remu` are single instructions in RV64M), but
    the *products and quotients* are the values to range-check if the
    constraint system proves arithmetic directly. Keeping operands ≤ 64
    bits where semantically safe (prices, lots) removes u128 div entirely.
  - Fixed-point BS: `exp`, `ln`, `sqrt`, `erf` are polynomial Horner +
    range reductions — either lookup tables (lattice-lookup) or extra
    ALU rows; the A&S erf polynomial is degree-5 → ~10 mul-adds.
  - Price-time priority comparisons: u64 compares at each level/slot —
    pure branch rows, no lookups.
- **ALU-heavy:** PnL multiplies (`(fill_per_lot − entry_per_lot) × lots`,
  i128), VWAP updates (2 mul + div), fee bps (2 mul_div per side per
  fill), the SFPM scan (18 scenarios × legs of mul-adds), funding
  (`lots × per_lot`), and **SHA-256** (book commitment: ~2 compressions =
    ~128 rounds ≈ 2 k ALU ops per resting order — the single largest ALU
    block if D is large; consider committing only touched levels or a
    cheaper digest for the first guest, but SHA-256 keeps host
    compatibility).
- **Memory-argument heavy:** book mutations. Each fill does random-access
  read+write into `orders[id-1]`, the maker's channel slot, level
  aggregates, two account records, and two position records. With lzx's
  `lattice-memory` (Twist/Shout one-hot checking, sparse memories —
  `lzx/crates/lattice-memory/src/{twist,shout,sparse}.rs`) this is the
  natural fit: one-hot or sparse read/write arguments over the flat
  static arrays; capacities in §8.5 keep table widths bounded (orders
  table = 1,024 rows × ~5 words — comfortable for one-hot; accounts =
  16 rows — trivial).
- **Branch divergence:** the match walk's depth depends on the batch;
  instruction-trace zkVMs (lzx proves RV64IMAC steps) absorb this
  natively — T is just the dynamic count. Avoid porting the *shape* of
  data-dependent-iteration analytics (Newton IV solver) — fixed-count
  bisection keeps T input-independent for the analytics phase.

**Expected T (trace steps = dynamic instructions, per §7.3):**

| Guest | Per order | Batch of 256 |
|---|---|---|
| Perp CLOB (match + fees + positions + margin gate, no SHA) | 15-40 k | 4-10 M |
| + SHA-256 book root over D=512 | ~2 k/resting order once | +1-2 M (once, or per block) |
| + funding settle + one liquidation pass | bursty | +0.5-5 M |
| Option-leg margin (fixed-pt BS, 36 evals/order) | +30-80 k | +8-20 M |

**Suggested staged plan:**
1. **Guest A (perp CLOB kernel):** book + match + fills + fees + positions
   + perp-only SFPM + book hash root. T ≈ 5-15 M. This proves the
   *matching engine* — the workload's defining loop — end to end.
2. **Guest B (options):** add fixed-point BS marks + option legs in the
   scan + SOMC + option fee caps + one expiry settlement. T ≈ 20-50 M.
3. **Guest C (cascade):** add funding interval + liquidation pass
   (penalized IOC cross + insurance/ADL). Stress-shaped input, T bursty.

Verification anchors for the port: the host repo's own tests
(`cargo test --workspace`, 370 tests) — port the orderbook unit tests
(price-time priority, iceberg requeue, STP matrix, auction uncross
differential) and the account PnL tests (`margin/src/account.rs:374-528`)
into the guest test harness; then differential-test guest-vs-host on
random batches (same seed → same trades, same position deltas, same book
root — mirroring `engine::tests::determinism_and_replay` and the fuzz
targets `fuzz/targets/fuzz_matching.rs`).

---

### Appendix: measured native anchors (2-vCPU container, rustc 1.98.1)

From `docs/BENCHMARKS.md` and repo README: taker vs 2,000-order deep book
485 ns p50 (lazy channel walk); crossing taker vs 200 resting 506 ns;
place/cancel churn 539 ns; tick sweep 700 ns-1.2 µs; auction uncross
10k orders/121 levels 9.1 ms; BAW American put 10.7 µs (r=3%) / 81 ns
(r=0, ≡ BS); European BSM 81 ns; Merton perpetual 23 ns; book commitment
1,000 resting orders 1.2 ms; settlement merkle root 2,000 accounts
1.77 ms; WAL replay ~7.7 M commands/s. 370 tests pass workspace-wide.
