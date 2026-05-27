# Stock Screener — Design

Momentum-based US-stock screener that pulls maximum data, ranks tickers by tunable formulas across multiple timeframes, backtests fast, then validates forward on a paper account. Claude drives the judgment loop (retro + tuning); Python does everything that must be exact and repeatable.

## Locked decisions (brainstorming Q1–Q7)

| # | Decision | Implementation impact |
|---|----------|----------------------|
| 1 | Data source <= $10/mo | Candidates: Alpaca (free, data + paper), Tiingo (~$10), EODHD/FMP. Start on Alpaca free. |
| 2 | Technical momentum | Historical OHLCV -> RSI, moving averages, momentum, breakouts. |
| 3 | Daily + weekly + monthly | Pull EOD, resample to W/M, score each timeframe separately, then weight (trend-alignment across all three = higher score). |
| 4 | Whole US market + filter | ~6-8K tickers; liquidity/price prefilter before heavy scan. |
| 5 | Backtest + Paper | Backtest first for fast calibration; Alpaca paper for live forward validation. |
| 6 | Python + Claude skills | Deterministic engine + judgment brain, fully separated. |
| 7 | Auto-tune within bounds + cron | Claude tries formula variants inside allowed ranges; daily/weekly cron sends a WhatsApp summary. |

## Core principle — engine vs. brain

- **Deterministic code (Python):** data fetch, indicators, scoring, backtest, P&L. Same input -> same output, always.
- **Claude (skills):** reads results, decides what to tune, writes retro, proposes next experiment. Claude *runs* the engine, never replaces its math.
- **Formula = config (YAML), not code:** Claude edits numbers within allowed ranges, re-runs backtest, compares versions. Each version (v1, v2...) is saved with its results.

## Repo structure

```
stock-screener/
├── .claude/skills/
│   ├── screener/          # "run scan + ranking"
│   ├── backtest/          # "test a formula against history"
│   ├── retro/             # "analyze trades — what worked, what didn't, why"
│   └── tune-formula/      # "propose a formula improvement within bounds"
├── engine/
│   ├── data.py            # pull EOD from source (Alpaca first)
│   ├── indicators.py      # RSI, MAs, momentum, breakouts
│   ├── score.py           # the formula (daily+weekly+monthly weighted)
│   ├── backtest.py        # historical simulation
│   └── paper.py           # Alpaca paper connection
├── formulas/
│   └── momentum_v1.yaml   # weights + thresholds + allowed tuning ranges
├── runs/                  # every scan/backtest saved with timestamp
└── journal/               # trade journal + accumulated retros
```

## Self-improvement loop

backtest -> Claude retro (what succeeded/failed and why) -> Claude proposes v2 within bounds -> backtest again -> winner promoted to paper -> cron sends WhatsApp summary.

## Build order

1. **Phase 1 — engine + backtest** (no account dependency): data.py (Alpaca free), indicators.py, score.py, momentum_v1.yaml, backtest.py. See results within days.
2. **Phase 2 — paper forward**: open Alpaca account, paper.py, daily cron scan.
3. **Phase 3 — auto-tune + cron summaries**: tune-formula skill within bounds, scheduled WhatsApp retro.

## Open item

- Alpaca account: not yet created (user will open). Phase 1 needs no account — Alpaca free data key is enough; paper account only needed for Phase 2.
