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
The score is a weighted sum of up to eight sub-scores, computed across three
timeframes (`daily`, `weekly`, `monthly`):

| Sub-score | What it measures |
|---|---|
| `momentum` | % change over `momentum_lookback` (with optional `momentum_skip_recent` for 12-1 dual momentum) |
| `trend` | SMA-fast > SMA-slow and price above both |
| `rsi` | Inside `rsi_band`, decaying outside |
| `breakout` | Proximity to N-day high (long) or low (`direction: reversion`) |
| `volatility` | Lowest stdev of daily returns → highest score |
| `atr_contraction` | ATR-% today vs N sessions ago — vol contraction = squeeze |
| `volume_dryup` | 20d avg vol < 80% of 60d avg vol |
| `bb_squeeze` | Current BB-width in bottom quintile of trailing window |

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
| `base_breakout_v1.yaml` | Stage-2 base breakout (Weinstein/IBD style) | daily-dominant |

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

Two cron jobs (managed by the outer agent, not OS crontab):

- **`stock-auto-tune-nightly`** — `0 3 * * *` (03:00 Israel)
  Runs `auto_tune_all.sh` which loops over every `formulas/*.yaml`
  (excluding `*.bak_*.yaml`) and runs `auto_tune.py --trials 3` on each.
  Sends a 4-6 line summary back to the chat.
- **`stock-leaderboard-weekly`** — `0 8 * * 0` (Sunday 08:00)
  Reads the last 7 days of `bt_summary_*.json` + `auto_tune.csv` and
  produces a markdown table sorted by composite (`sharpe - |dd|`).

To run the nightly loop manually:

```bash
bash auto_tune_all.sh
# then check the latest log:
ls -t runs/auto_tune_nightly_*.log | head -1 | xargs cat
```

---

## Indicator library

- `engine/indicators.py` — RSI, SMA, momentum (with skip support),
  rolling high/low, stdev of returns, OHLCV resampling
- `engine/atr.py` — true range, Wilder ATR, ATR-%, Bollinger band width,
  OBV, avg volume, avg-dollar volume, **`base_range_pct`** (consolidation
  detector), **`base_pivot`** (breakout trigger level)

All pure functions, no I/O, no globals.

---

## Layout

```
stock-screener/
├── engine/
│   ├── score.py          # math — DO NOT touch when adding strategies
│   ├── backtest.py       # walk-forward backtester + exit logic + regime rotation
│   ├── indicators.py     # pure indicator functions
│   ├── atr.py            # ATR + base detection helpers
│   ├── data.py           # yfinance / parquet provider
│   └── universe.py       # S&P 500 list + liquidity filter
├── formulas/             # YAML strategy configs (one per strategy)
├── runs/                 # all backtest output, auto-tune state, logs
├── auto_tune.py          # walk-forward tuner with guardrails
├── auto_tune_all.sh      # nightly wrapper, loops over all formulas
├── compare.py            # side-by-side strategy comparison
├── run.py                # CLI entry point (scan / backtest)
├── README.md             # this file
└── CLAUDE.md             # context for the agent maintaining this repo
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
