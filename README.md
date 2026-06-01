# stock-screener

Momentum / breakout / mean-reversion screener for US equities. Strategies are
YAML-driven (the code is the math, the YAML is the numbers), backtests are
deterministic, and a nightly auto-tuner evolves each formula with strict
over-fit guardrails.

---

## Quick start

```bash
# Backtest a strategy on the S&P 500
python run.py backtest \
  --formula formulas/dual_momentum_v1.yaml \
  --universe sp500 \
  --start 2023-01-01 --end 2026-05-26 \
  --top-n 20 --atr-stop-mult 2.0 \
  --trailing-stop 0.06 --trailing-activate 0.05 \
  --time-stop-bars 15

# Rank the universe today (no backtest, just current scores)
python run.py scan --formula formulas/momentum_v1.yaml --tickers AAPL NVDA AMD

# Auto-tune a single formula (3 walk-forward trials, with guardrails)
python auto_tune.py --formula formulas/dual_momentum_v1.yaml --trials 3
```

Outputs land in `runs/`:
- `bt_summary_<formula>_<universe>_<stamp>.json` — high-level stats
- `bt_trades_<...>.csv` — every trade with score, return, exit reason
- `bt_equity_<...>.csv` / `bt_benchmark_<...>.csv` — equity curves
- `auto_tune.csv` — every tuning trial (accepted or rejected)
- `auto_tune_state.json` — per-formula state (cooldowns, drift counters, baselines)
- `auto_tune_nightly_<YYYYMMDD>.log` — full nightly run output

---

## How scoring works

Every formula in `formulas/*.yaml` produces a 0-1 score per ticker per day.
The score is a weighted sum of up to nine sub-scores, computed across three
timeframes (`daily`, `weekly`, `monthly`):

| Sub-score | What it measures |
|---|---|
| `momentum` | % change over `momentum_lookback` (with optional `momentum_skip_recent` for 12-1 dual momentum) |
| `trend` | SMA-fast > SMA-slow and price above both |
| `rsi` | Inside `rsi_band`, decaying outside |
| `breakout` | Proximity to N-day high (long) or low (`direction: reversion`) |
| `breakout_thrust` | `close` sitting just above the causal pivot — peaks 1-3% above, fades to 0 by ~5% (don't chase). Rewards the early break itself, not just nearness |
| `volatility` | Lowest stdev of daily returns → highest score |
| `atr_contraction` | ATR-% today vs N sessions ago — vol contraction = squeeze. Gated to `px > sma_slow` when the YAML sets `trend_gate_quietness: true` (Stage-2) |
| `volume_dryup` | 20d avg vol < 80% of 60d avg vol (also `trend_gate_quietness`-gated) |
| `bb_squeeze` | Current BB-width in bottom quintile of trailing window (also `trend_gate_quietness`-gated) |

Sub-scores are mixed by `timeframe_score_weights` per timeframe, then the
three timeframes are mixed by `timeframe_weights`, then an `alignment_bonus`
fires when all three timeframes agree.

Math lives in `engine/score.py`. **Never edit it to express a new strategy —
write a new YAML.**

---

## Strategies (current YAMLs)

| File | Idea | Best timeframe |
|---|---|---|
| `momentum_v1.yaml` | Classic momentum + trend, top-N ranked | weekly rebalance |
| `mean_reversion_v1.yaml` | `direction: reversion` — oversold rank-up | weekly |
| `quality_v1.yaml` | Trend + RSI band, lower risk profile | weekly |
| `pre_breakout_v1.yaml` | ATR contraction + BB squeeze + volume dry-up | daily-dominant |
| `dual_momentum_v1.yaml` | Antonacci 12-1 + absolute-momentum bond rotation (`SPY` vs `SHY`) | monthly-dominant |
| `dual_momentum_v2.yaml` | Same as v1 but 3M lookback — reacts faster, more whipsaw | weekly |
| `base_breakout_v1.yaml` | Stage-2 base breakout (Weinstein/IBD style) | daily-dominant |
| `momentum_lowvol_v1.yaml` | Momentum + volatility penalty (~cross-sectional Sharpe ranker) | weekly |
| `trend_strength_v1.yaml` | Trend-dominant (`trend` weight 0.50 + alignment bonus 0.20) | monthly-dominant |
| `momentum_quality_combo_v1.yaml` | Balanced blend: directional + stable | weekly |

`dual_momentum_v1` is the only strategy that uses the `absolute_momentum`
block — when `SPY` 12-1 return is below `SHY` 12-1 return, the backtest
rotates the entire allocation to `SHY` for that period and tags the row
with `exit: "regime_off"`.

---

## Backtest engine (`engine/backtest.py`)

- Walks the calendar at `rebalance` frequency (default `W-FRI`)
- On each rebalance: scores every ticker using data **only up to that date**
  (no lookahead), holds the top-N equal-weighted until the next rebalance
- Exit rules evaluated on each daily close, in priority order:
  1. **Hard stop** — `entry - atr_stop_mult * ATR` (else flat `stop_loss_pct`)
  2. **Trailing stop** — `peak * (1 - trailing_stop_pct)`, armed after
     `trailing_activate_pct` gain
  3. **Take profit** — `entry * (1 + take_profit_pct)`
  4. **Time stop** — N daily bars since entry
  5. **Fall-through** close-to-close → `exit: "hold"`

`exit_breakdown` in the summary tells you which exit dominates — a healthy
strategy mixes `trail` + `tp` + `hold`; if 95% is `hold` your exits are
inactive.

### Entry modes — `rebalance` (default) vs `event` vs `managed`

`BacktestConfig.mode` gates how each weekly scan picks names AND how long it holds:

- **`rebalance`** (default, unchanged) — take the top-N ranked names every
  `W-FRI` bar. Bit-identical to the legacy path.
- **`event`** — a volume-confirmed breakout filter on top of the ranking. A
  name only qualifies when, on (or within `event_window_bars` of) the scan bar,
  `close > prior-event_lookback-bar high` **AND**
  `volume >= vol_confirm_mult * volume.rolling(event_vol_lookback).mean()`.
  Cadence stays weekly `W-FRI` (no daily scan — preserves the `n_per_year=52`
  annualization, per-rebalance cost and the weekly-resample parity). Pair with a
  tight ATR stop + trailing + time stop to get the asymmetric R:R a real
  breakout system needs. Fully causal — `rolling_high` excludes the current bar,
  the volume mean is backward-looking.
- **`managed`** — event-confirmed entry **plus a hold decoupled from the W-FRI
  grid** (`_run_managed`). Positions persist across rebalance bars and are
  managed on *every daily bar* (stop / trail / tp / time, same priority +
  arithmetic as `_period_return_with_exits`) until an exit fires; freed slots
  refill at rebalance bars from the event-confirmed ranking. This is the fix
  `event` couldn't deliver: on the weekly grid every trade is force-closed at
  the next Friday (~5 bars) so trailing/time never fire — here they can.
  Produces **real round-trip trades** (a multi-week winner = ONE trade, with a
  `bars_held` column), so win-rate / payoff / avg-hold are honest. Two
  deliberate book-keeping differences (documented in `engine/backtest.py`):
  equity is still sampled weekly (`_stats` n_per_year=52 unchanged) and
  round-trip cost is charged once per entry, not on the whole book every week.

Carry mode + event params + exits in the formula YAML under a `backtest:` block
(`engine.backtest.config_from_formula` reads it; `run.py --mode` and
`run_regimes.py` honor it). Only the two breakout YAMLs set `mode: event`.
Validate event mode with `scripts/validate_event_path.py`; validate managed mode
(3-way `rebalance` / `event` / `managed`, full 2023→now + 2018→now + regimes,
reports W/L asymmetry + avg-hold) with `scripts/validate_managed_path.py`. Unit
tests: `scripts/test_event_path.py`, `scripts/test_managed_path.py`.

---

## Auto-tuner (`auto_tune.py`) — over-fit guardrails

Each trial:
1. Mutates one parameter by ±10%, clamped to `bounds`
2. Backtests on **3 walk-forward folds** (expanding train / rolling test)
3. Accepts ONLY if:
   - `composite_score(test) > monthly_baseline` on **all 3 folds**
   - `|test_dd| ≤ baseline_dd × 1.10` on all folds (risk guardrail)
   - `train_sharpe > 0` on all folds (sanity)
4. On accept: backs up the YAML (`*.bak_<stamp>.yaml`), writes new
5. On reject: 24h cooldown on that param + anti-mirror block on the
   opposite direction

Extra drift caps:
- **`MAX_ACCEPTED_PER_RUN = 1`** — at most one change per nightly run
- **`MONTHLY_DRIFT_CAP = 5`** — a param that flipped 5× this month gets
  frozen for 7 days
- **Monthly baseline lock** — the baseline used for acceptance is locked
  on the 1st of each month; intra-month gains can't shift it

State lives in `runs/auto_tune_state.json`. Every trial (accepted or
rejected) is appended to `runs/auto_tune.csv` so you can audit drift over
time.

---

## Nightly automation

Real OS crontab on the host (`crontab -l`):

- **`0 3 * * *`** — `auto_tune_all.sh` runs `auto_tune.py --trials 3` on every
  formula. The window end is `$(date -Idate -d yesterday)` so the tuner
  always evaluates against fresh data; start stays at `2023-01-01` to keep
  the training span stable across runs.
- **`@reboot`** — relaunches the dashboard HTTP server in a supervised
  `tmux` session named `stock-server` (port 8000, fronted by Tailscale).

Manual triggers (long-running, run inside `tmux` sessions):

```bash
# Re-baseline every formula on the same uniform window (10 formulas, ~3-4h).
tmux new -d -s stock-uniform -c $PWD 'exec ./backtest_all_uniform.sh'

# Cross-regime evaluation (10 formulas x 6 historical regimes, ~1h).
tmux new -d -s stock-regime  -c $PWD 'exec ./backtest_regimes.sh'

# Watch progress:
tmux attach -t stock-uniform   # Ctrl-b d to detach
tail -f runs/backtest_all_uniform_*.log
```

To chain `regime` after `uniform` (avoids yfinance rate-limit collision):

```bash
tmux new -d -s stock-regime-watcher -c $PWD 'exec ./scripts/chain_regime_after_uniform.sh'
```

---

## Stock advisor skill

`stock-advisor` is a Claude Code **skill** (not a CLI) that turns this
repo's output into an actionable weekly trade plan. It lives at
`~/.claude/skills/stock-advisor/SKILL.md` (user-global, not in this
repo).

What it does, in one Claude Code session:

1. Reads `runs/regime_current.json` (regenerates via `run_regimes.py`
   if older than 24h) → current SPY regime.
2. Picks the best strategy for that regime, filtered by
   `analysis.json` predictiveness (`corr_pearson > 0.05`,
   `trustworthiness_flags == []`).
3. Reads the latest `scan_<strategy>_sp500_*.csv` (triggers
   `scan_all.sh` if stale).
4. `WebSearch` news + sentiment for the top 5–10 tickers.
5. Cross-strategy consensus tiering across every trustworthy scan.
6. Outputs a phone-friendly plan: market view, top picks with sentiment
   and consensus tiers, entry/stops, and caveats.

Trigger from any Claude Code session by asking naturally
("what should I buy this week?", "today's picks", "market view +
top 10", "what about NVDA?") or explicitly with `/skill stock-advisor`.

A thin info wrapper lives at `scripts/quick_advisor.sh` — it just prints
how to invoke the skill.

---

## Live dashboard

`dashboard.py` regenerates `docs/index.html`, `docs/data.json`,
`docs/strategy.html`, and `docs/strategies/<formula>.json`. The HTML polls
`data.json` every 15 s — no rebuild needed for new data, just re-run the
Python script.

Served by a Python `http.server` running in `tmux:stock-server` and exposed
via Tailscale:

```
https://<host>.<tailnet>.ts.net/
https://<host>.<tailnet>.ts.net/strategy.html?f=<formula_name>
```

### Per-strategy archive (`strategy.html?f=<formula>`)

For every formula in the leaderboard, a dedicated page with:

- **Performance timeline** — Sharpe / Return / |Max DD| across every
  historical run (`runs/bt_summary_<formula>_*_<stamp>.json`)
- **Daily picks** — top-N per rebalance day (W-FRI by default) with
  per-trade scores, P&L until next rebalance, and exit reasons. Derived
  from `runs/bt_trades_<formula>_<universe>_<stamp>.csv`.
- **Accepted parameter changes** — from `runs/auto_tune_state.json`
  (`formulas.<name>.yaml.change_log`)
- **YAML param diffs** — between consecutive `formulas/<name>.bak_*.yaml`
  snapshots
- **All auto-tune trials** — accepted *and* rejected, from
  `runs/auto_tune.csv`

### Regime matrix

Section on the main page that compares every strategy across every named
regime defined in `regimes.json`:

| | 2018 vol | 2020 COVID | 2021 bull | 2022 bear | AI 2023-24 | 2025 |
|---|---|---|---|---|---|---|
| momentum_v1 | … | … | … | … | … | … |
| dual_momentum_v1 | … | … | … | … | … | … |
| … | | | | | | |

Best cell per column is highlighted. Metric switcher: `sharpe`,
`total_return`, `max_drawdown`, `alpha_vs_benchmark`.

A **current-regime banner** at the top classifies SPY's latest 60d state
(`bull` / `bear` / `choppy` / `crash`) using a simple heuristic
(`run_regimes.classify_current_regime()` — last 60d return, annualised
vol, position vs MA200) and recommends the historically-best strategy for
the matching regime kind.

To refresh the matrix:

```bash
python run_regimes.py        # one wide fetch, 60 backtests in-memory
python dashboard.py          # writes docs/regime_matrix into data.json
# or just:
bash backtest_regimes.sh
```

---

## Indicator library

- `engine/indicators.py` — RSI, SMA, momentum (with skip support),
  rolling high/low, stdev of returns, OHLCV resampling
- `engine/atr.py` — true range, Wilder ATR, ATR-%, Bollinger band width,
  OBV, avg volume, avg-dollar volume, **`base_range_pct`** (consolidation
  detector), **`base_pivot`** (breakout trigger level)
- `engine/bank.py` — shared per-ticker **indicator bank** (perf): computes the
  union of every formula's `(indicator, period)` once per ticker and reuses it
  across all formulas. `collect_specs` / `build_bank` / `assemble_precompute`.
  Output is bit-identical to `precompute_indicators`; opt-in via
  `run_regimes.py --shared-bank`.

All pure functions, no I/O, no globals.

---

## Layout

```
stock-screener/
├── engine/
│   ├── score.py                 # math — DO NOT touch when adding strategies
│   ├── backtest.py              # walk-forward backtester + exit logic + regime rotation
│   ├── indicators.py            # pure indicator functions
│   ├── atr.py                   # ATR + base detection helpers
│   ├── data.py                  # yfinance / parquet provider
│   └── universe.py              # S&P 500 list + liquidity filter
├── formulas/                    # YAML strategy configs (one per strategy)
├── runs/                        # all backtest output, auto-tune state, logs
│   ├── auto_tune.csv            # every tuning trial (accepted+rejected)
│   ├── auto_tune_state.json     # cooldowns, change_log, monthly baselines
│   ├── bt_summary_*.json        # one per backtest run
│   ├── bt_trades_*.csv          # per-trade detail (powers Daily picks)
│   ├── regime_matrix.json       # strategy × regime grid (from run_regimes.py)
│   └── regime_current.json      # current SPY regime classification
├── docs/                        # served by stock-server tmux (port 8000)
│   ├── index.html               # main dashboard (Tailwind + Chart.js, polls data.json)
│   ├── data.json                # gitignored, regenerated every dashboard.py call
│   ├── strategy.html            # per-strategy archive viewer (?f=<name>)
│   └── strategies/<name>.json   # per-strategy history payload
├── scripts/
│   └── chain_regime_after_uniform.sh   # tmux watcher: regime after uniform
├── auto_tune.py                 # walk-forward tuner with guardrails
├── auto_tune_all.sh             # nightly wrapper (cron 03:00), end=yesterday
├── backtest_all_uniform.sh      # all formulas, uniform window, end=yesterday
├── backtest_regimes.sh          # wrapper for run_regimes.py + dashboard refresh
├── run_regimes.py               # one wide fetch + 60 in-memory backtests
├── regimes.json                 # named historical regimes (edit to add/move)
├── dashboard.py                 # regenerates docs/* from runs/*
├── compare.py                   # side-by-side strategy comparison
├── run.py                       # CLI entry point (scan / backtest)
├── README.md                    # this file
└── CLAUDE.md                    # context for the agent maintaining this repo
```

---

## Adding a new strategy

1. Copy `formulas/momentum_v1.yaml` to `formulas/<your_name>_v1.yaml`
2. Adjust `timeframe_score_weights`, `timeframe_weights`, `rsi_band`, etc.
3. Add `bounds:` for every param you want the auto-tuner to touch
4. Run a baseline backtest:

   ```bash
   python run.py backtest --formula formulas/<your_name>_v1.yaml \
     --universe sp500 --start 2023-01-01 --end 2026-05-26 --top-n 20
   ```

5. The nightly cron will pick it up automatically tomorrow at 03:00 —
   no further wiring needed.

If your strategy needs a new sub-score, **add the math to `engine/score.py`
and expose a weight key**, then declare it in YAML. Never inline strategy
specifics in code.
