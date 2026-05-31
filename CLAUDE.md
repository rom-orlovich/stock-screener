# CLAUDE.md — stock-screener

Context for any future Claude session working on this repo. Read this
first.

## What this is

A YAML-driven momentum/breakout/mean-reversion screener for US equities.
Backtests over the S&P 500, walks the calendar with no lookahead, and
self-tunes nightly with strict over-fit guardrails.

## Hard rules

1. **Strategies live in YAML, math lives in code.** When the user asks for
   a new strategy variant, the answer is almost always a new
   `formulas/*.yaml` file, NOT an edit to `engine/score.py`. Only add
   code when a genuinely new sub-score is needed (then add it + a weight
   key + a `bounds:` entry and document it in `README.md`).
2. **No lookahead, ever.** `engine/backtest.py` slices `df.loc[:d0]`
   before scoring. Any new indicator must respect this. Don't use
   centered windows, future-aligned resamples, or `ffill` past the
   evaluation bar.
3. **Don't tune the YAML by hand when the auto-tuner is running.**
   Manual edits invalidate the monthly baseline lock. If you must edit,
   delete the relevant entry in `runs/auto_tune_state.json` so the next
   nightly run re-establishes the baseline.
4. **Backups are sacred.** `auto_tune.py` writes `*.bak_<stamp>.yaml`
   before every accepted change. Don't delete them — they're the
   rollback trail.

## Architecture cheat sheet

```
run.py ──► engine.backtest.run() ──► engine.score.score_ticker()
                │                              │
                ├── _market_regime_ok()        └── reads YAML weights
                │   (only when YAML has         + timeframe sub-scores
                │    absolute_momentum block)
                │
                ├── _breakout_event_fired()
                │   (only when cfg.mode in {"event","managed"}:
                │    filters ranked picks to volume-confirmed breaks)
                │
                ├── _run_managed()  (only when cfg.mode=="managed":
                │   decoupled bar-by-bar hold; positions persist across
                │   rebalances; daily exits; slots refilled at rebalances)
                │
                └── _period_return_with_exits()  (rebalance/event only)
                    (stop / trail / tp / time / hold)

auto_tune.py ──► walk-forward 3-fold ──► accepts/rejects
                                            ├── writes backup
                                            ├── updates state.json
                                            └── logs to auto_tune.csv
```

## Key files (do not break)

- `engine/score.py` — central scoring math. The sub-score keys it
  produces (`momentum`, `trend`, `rsi`, `breakout`, `breakout_thrust`,
  `volatility`, `atr_contraction`, `volume_dryup`, `bb_squeeze`) do NOT
  need to be present in every YAML. The scorer reads each weight via
  `w.get(key, 0.0)` and `_norm` sums only the keys actually present, so a
  key absent from a formula contributes 0 and does not renormalize it. The
  YAMLs are already non-uniform (e.g. `momentum_v1`/`mean_reversion_v1`
  carry only `momentum/trend/rsi/breakout`). Add a new sub-score's weight
  key only to the formulas that use it. `breakout_thrust` lives only in the
  two breakout YAMLs.
- `engine/backtest.py` — walks the calendar; respects `absolute_momentum`
  block for regime rotation; exit logic priority is fixed (stop → trail
  → tp → time → hold). Don't reorder.
- `auto_tune.py` — over-fit guardrails (walk-forward, cooldown,
  anti-mirror, drift cap, baseline lock, max-1-per-run). Don't soften
  these without explicit user approval.
- `auto_tune_all.sh` — nightly wrapper invoked by the cron job
  `stock-auto-tune-nightly` at 03:00 Israel.

## Recent changes (history matters)

- **2026-05-31** — feat: **`mode="managed"`** (`engine/backtest._run_managed` +
  helpers `_resolve_exit_levels`/`_build_precompute`/`_ranked_at`) — decoupled
  bar-by-bar hold. Positions persist across W-FRI bars, managed on every daily bar
  (stop/trail/tp/time, arithmetic mirrored byte-for-byte from
  `_period_return_with_exits`) until an exit fires; freed slots refill from the
  event-confirmed ranking. Gated by `cfg.mode=="managed"` at the top of `run()` so
  rebalance/event stay **bit-identical** (golden re-checked; I deliberately did NOT
  refactor the shared exit/numba path — parity is sacred — managed *mirrors* it in
  a separate stepper, proven by `test_stepper_parity`). Produces **real round-trip
  trades** with a `bars_held` column (~1/3 the position-week count). Equity sampled
  weekly (n_per_year=52 unchanged); cost charged once per round-trip. Wired into
  `run.py --mode managed`. Tests: `scripts/test_managed_path.py`. **Validation
  verdict (HONEST, `scripts/validate_managed_path.py`, full sp500 × {2023→now,
  2018→now, bull_2021, bear_2022, ai_2023_2024} × 3 modes, parallel=3):** managed
  is the FIRST mode with real R:R asymmetry — **payoff 1.21–1.45** (highest of the
  3 modes in 9/10 cells; vs event ~1.0–1.18, reb ~0.9–1.18), losers cut ~8 bars vs
  winners ~12–14, and it **roughly halves event's drawdown** (base full_2023
  −0.134 vs event −0.278). BUT avg hold is only ~2.4–2.6 weeks (capped by
  `time_stop_bars=15`; exits time-dominated — base full_2023 time 709/stop 314/
  trail 92), event still wins **total return** outright (base full_2023 +1.03 vs
  +0.65), alpha stays negative on long windows, and sharpe is a wash (managed wins
  full windows + ai, loses clean bull_2021 to weekly reload). A better-shaped,
  lower-DD, lower-return system — not a free lunch. YAML `mode:` left at `event`.
  Next lever = raise/drop `time_stop_bars` to let winners truly run. See
  `/tmp/managed_RESULT.md`. Not pushed/merged.
- **2026-05-31** — feat: event-entry path. `BacktestConfig.mode`
  (`rebalance` default / `event`), `_breakout_event_fired` (causal,
  vectorized: `close > prior-N high` AND `volume >= k*avg`, trailing
  `event_window_bars`, W-FRI cadence per Q1), `config_from_formula` reading a
  YAML `backtest:` block. Wired into `run.py --mode` + `run_regimes.cfg_for`;
  `mode: event` set only in the two breakout YAMLs. Default path bit-identical
  (`scripts/test_event_path.py` golden). **Validation verdict (HONEST,
  `scripts/validate_event_path.py`, full sp500 + 6 regimes): it does NOT make
  these real breakout systems.** Payoff ratio stays ~1.0–1.1 in every mode,
  max-DD generally *worsens* (filter shrinks the book → concentration), and
  exits are ~89% `hold` / 0% `time` because the weekly rebalance caps each trade
  at ~5 bars so stops/trails can't manage the trade. The `rebalance_stop`
  ablation shows the stops alone do ~nothing. Event mode *does* lift
  sharpe/alpha on 2023→now (pre_breakout flips to +0.38 alpha) but loses badly
  in bull-2021/bear-2022. Fails the Q3 gate (sharpe+DD across regimes). Next
  lever = decouple the hold from the W-FRI grid (manage bar-by-bar across weeks)
  — out of scope here, changes `_stats`/cost parity. See `/tmp/event_RESULT.md`.
- **2026-05-30** — feat: `breakout_thrust` sub-score in `engine/score.py`
  (`_breakout_thrust_score`) + its vectorized mirror in `engine/score_vec.py`.
  Trapezoid on `px/pivot - 1`: 0 at/below the causal pivot (`rolling_high`,
  current bar excluded), peaks 1-3% above, fades to 0 by ~5% (don't chase).
  Wired into all three paths (`timeframe_score`, `_timeframe_score_at`,
  `_daily_score_vectorized`) via `w.get("breakout_thrust", 0.0)`. New opt-in
  YAML knob `trend_gate_quietness` zeros `atr_contraction`/`volume_dryup`/
  `bb_squeeze` unless `px > sma_slow` (Stage-2). Both gated to the two breakout
  YAMLs only (re-weighted toward direction, `bb_squeeze`→0). Tests:
  `scripts/parity_breakout_thrust.py` (shape + tri-path parity).
- **2026-05-30** — perf: `engine/bank.py` (new module) — shared per-ticker
  indicator bank. Computes the union of every formula's `(indicator, period)`
  ONCE per ticker (28 distinct vs 132 recomputed across 12 formulas);
  `assemble_precompute()` selects each formula's subset. Bit-identical to
  `precompute_indicators` (`scripts/parity_bank.py`). `bt.run()` gained an
  optional `bank=` param. Opt-in via `run_regimes.py --shared-bank`.
- **2026-05-30** — perf: `run_regimes.py` fork pool. Parent builds the panel
  once; `--mp-context fork` (default) lets workers inherit it via COW instead
  of re-reading ~504 pickles each. Added `--end` pin, `--limit`, `--shared-bank`.
  `backtest_regimes.sh` now runs `--parallel $(nproc) --mp-context fork
  --shared-bank`; `auto_tune_all.sh` exports `USE_VECTORIZED_SCORING=1`.
- **2026-05-30** — PROFILER FINDING (`scripts/profile_split.py`): per-d0
  scoring is **95.5%** of a vectorized backtest, precompute only 4.5%. So the
  bank is a ~7% win (not 2×), and a daily cross-section is near-worthless — the
  real bottleneck is the weekly/monthly per-d0 recompute (the adaptive-cap
  fallback at `score.py:444-462`). That's the next lever (high parity risk).
- **2026-05-27** — added `base_range_pct` / `base_pivot` to `engine/atr.py`
- **2026-05-27** — added `momentum_skip_recent` knob to `engine/score.py`
  (for 12-1 dual momentum)
- **2026-05-27** — added `_market_regime_ok()` + bond rotation to
  `engine/backtest.py` (only fires when YAML declares `absolute_momentum`)
- **2026-05-27** — added `BacktestConfig` exit fields:
  `trailing_stop_pct`, `trailing_activate_pct`, `atr_stop_mult`,
  `atr_stop_period`, `time_stop_bars`
- **2026-05-27** — created `dual_momentum_v1.yaml` and
  `base_breakout_v1.yaml`
- **2026-05-27** — rewrote `auto_tune.py` with walk-forward folds,
  cooldowns, anti-mirror, monthly drift cap, baseline lock

## Known issues / TODO

- **DONE (2026-05-31)** — decoupled hold (`mode="managed"`, `_run_managed`):
  the "manage bar-by-bar across weeks" lever the event-path note called out as
  out-of-scope. Positions persist across W-FRI bars, managed daily until a
  stop/trail/tp/time exit fires; freed slots refill from the event ranking.
  First mode to produce real R:R asymmetry (payoff ~1.3–1.5 in up regimes) +
  lower drawdown, at the cost of raw return. `scripts/validate_managed_path.py`
  (3-way), `scripts/test_managed_path.py`. See Recent changes / `/tmp/managed_RESULT.md`.
- **DONE (2026-05-31)** — event-driven entry is implemented:
  `BacktestConfig.mode` (`"rebalance"` default / `"event"`), gated so the
  rebalance path stays bit-identical. Per decision Q1 the event detector
  fires on the **weekly W-FRI bar** (not a daily scan — that would break
  `n_per_year=52`, per-rebalance cost, and weekly-resample parity):
  `_breakout_event_fired` requires `close > prior-N high` AND
  `volume >= k*avg_vol` within the trailing `event_window_bars`. It filters
  the ranked picks; it is NOT a separate detector replacing the ranker (the
  ranker still orders, the event gates entry). See README "Entry modes".
- The scoring sub-score `breakout` is still a continuous ranker. The event
  mode now supplies the missing binary trigger as an entry *filter* on top of
  the ranking + tight stops, which is what creates the R:R asymmetry.
- `pre_breakout_v1` has `corr(score, return) ≈ -0.03` — the score is not
  predictive in its current form. The auto-tuner may still find local
  improvements, but don't expect it to fix the fundamental signal.

## Decisions you can't read off the code

These exist because we burned cycles on the alternative. Don't undo them
without re-running the experiment that led here.

1. **Backtest `end` = yesterday, not today.** Today's bar isn't closed
   until ~23:00 IDT; running with `end=today` either fetches a half-bar
   or yfinance drops the day silently. `auto_tune_all.sh` and
   `backtest_all_uniform.sh` both use `$(date -Idate -d yesterday)`.

2. **Window changes invalidate the monthly baseline.** Because we made
   `--end` dynamic, the baseline lock (computed on day-1-of-month's
   window) drifts versus the candidate evaluations later that month. When
   you change the window mid-month, clear `monthly_baseline` per formula
   in `runs/auto_tune_state.json` (we keep a `.bak_<stamp>` next to it).

3. **Leaderboard dedup is by `summary.formula`, not filename.** Old
   `bt_summary_*.json` files lack the `_<universe>` suffix
   (e.g. `bt_summary_momentum_v1_20260522_*.json`); deduping by filename
   splits them into a phantom second row per strategy. `dashboard.py
   :: _index_summaries()` uses the in-file `formula` field instead.

4. **Regime backtests share one data fetch.** `run_regimes.py` does a
   single `get_universe()` for the widest window covering every regime,
   then calls `bt.run(price_data=…, start=R.start, end=R.end)` per
   `(formula, regime)`. Calling `run.py` per-regime would re-fetch 504
   tickers × N regimes — ~50× slower and yfinance will rate-limit you.

5. **Regime detection is heuristic, not a model.** SPY 60d return + 60d
   annualised vol + position vs MA200 → `bull` / `bear` / `choppy` /
   `crash`. Good enough to map to a regime kind; not a forecast. Don't
   stuff this into the live trading path.

6. **Long jobs run in `tmux`, never via `nohup &`.** Plain `nohup` orphans
   die when the parent Claude session ends (we've been bitten — see the
   May 28 `backtest_all_uniform_20260528_120638.log` that stopped after
   one line). Use `tmux new -d -s <name> 'exec ./script.sh'`. The session
   survives logout, supports `tmux attach` for live observation, and
   `tmux ls` tells you what's actually running.

7. **Sequencing via watcher, not parallel.** `scripts/chain_regime_after
   _uniform.sh` polls `tmux has-session -t stock-uniform` every 30s and
   launches `backtest_regimes.sh` when it ends. Running both at once
   trips yfinance rate limits.

8. **Daily picks come from `bt_trades_*.csv`, not a new engine path.**
   Every rebalance already records its top-N selections (the trades
   table). Don't add a separate "picks" emitter to `engine/backtest.py`
   — derive picks by grouping trades on `enter`, sorted by `score`
   descending. See `dashboard.py :: _picks_by_date()`.

9. **Mobile leaderboard uses a CSS sticky-first trick, not a JS hack.**
   Tables with class `sticky-first` get `position: sticky; left: 0` on
   their first column. The first column needs an explicit background so
   it covers cells scrolling under it (`background: #131826`).

## Conventions for tmux sessions used by this repo

Reserved session names — don't reuse for unrelated work:

| Session | Purpose | Started by |
|---|---|---|
| `stock-server` | `python -m http.server -d docs 8000` | `@reboot` cron, runs forever |
| `stock-uniform` | `backtest_all_uniform.sh` | manual or scheduled |
| `stock-regime` | `backtest_regimes.sh` | watcher, or manual |
| `stock-regime-watcher` | `chain_regime_after_uniform.sh` | manual when chaining |

## When the user asks for ...

- **"new strategy"** → new YAML, run baseline backtest, let the nightly
  tuner pick it up
- **"improve X"** → check `runs/auto_tune.csv` first to see what's been
  tried; check `runs/bt_summary_X_*.json` for current stats; THEN
  propose a change
- **"why is X losing to SPY"** → read the `bt_trades_*.csv`, group by
  `exit` reason, check `corr(score, ret)`, look at avg_win vs avg_loss.
  Don't guess.
- **"add a stop / TP / trailing"** → already wired in `BacktestConfig`,
  just pass the CLI flag in `run.py`
- **"rebalance differently"** → `--rebalance ME` for monthly, `W-FRI`
  for weekly. `n_per_year` mapping in `_stats()` may need an update for
  exotic frequencies.

## Don't do

- Don't add new modules to `engine/` without updating both `README.md`
  and this file
- Don't write to `runs/` from outside the backtest/tuner pipelines
- Don't introduce strategy-specific branching in `engine/score.py` —
  use YAML weights
- Don't use `pd.DataFrame.iterrows()` in hot loops (`score_ticker` is
  called ~88k times per backtest)
- Don't bypass the bounds check in `auto_tune.py` even "just to try
  something" — the user will lose monthly baseline integrity
