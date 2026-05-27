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
                └── _period_return_with_exits()
                    (stop / trail / tp / time / hold)

auto_tune.py ──► walk-forward 3-fold ──► accepts/rejects
                                            ├── writes backup
                                            ├── updates state.json
                                            └── logs to auto_tune.csv
```

## Key files (do not break)

- `engine/score.py` — central scoring math. The sub-score keys it
  produces (`momentum`, `trend`, `rsi`, `breakout`, `volatility`,
  `atr_contraction`, `volume_dryup`, `bb_squeeze`) MUST stay aligned
  with `timeframe_score_weights` in every YAML.
- `engine/backtest.py` — walks the calendar; respects `absolute_momentum`
  block for regime rotation; exit logic priority is fixed (stop → trail
  → tp → time → hold). Don't reorder.
- `auto_tune.py` — over-fit guardrails (walk-forward, cooldown,
  anti-mirror, drift cap, baseline lock, max-1-per-run). Don't soften
  these without explicit user approval.
- `auto_tune_all.sh` — nightly wrapper invoked by the cron job
  `stock-auto-tune-nightly` at 03:00 Israel.

## Recent changes (history matters)

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

- `base_breakout_v1` still runs in rebalance mode. True event-driven
  entry (enter on the day price > pivot AND volume > 2× avg, not at the
  next Friday) is not implemented. When implementing, add
  `BacktestConfig.mode: "event"` and gate the new path on that flag —
  don't break the rebalance path.
- The scoring sub-score `breakout` produces a continuous value, not a
  binary trigger. For genuine R:R asymmetry the engine needs a separate
  event detector (not a top-N ranker). Plan it before extending
  `base_breakout_v1`.
- `pre_breakout_v1` has `corr(score, return) ≈ -0.03` — the score is not
  predictive in its current form. The auto-tuner may still find local
  improvements, but don't expect it to fix the fundamental signal.

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
