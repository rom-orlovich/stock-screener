#!/usr/bin/env python3
"""Multi-period validation harness for leading_stock_v1.

Fetches the widest window ONCE for the chosen universe (CLAUDE.md decision #4:
one get_universe for the union, slice per period in-memory — NEVER re-fetch per
period or yfinance rate-limits you), applies the liquidity filter once, then runs
bt.run() for every (time_stop x period) cell on the shared panel via a fork pool
(parent builds the panel, workers inherit it COW). Honest metrics straight off
res.equity / res.trades / res.stats; the full grid is written to JSON for
re-verification.

Periods: 2018, 2020, 2021, 2022, 2023+ (2023-01-01 -> end), full (2018-01-01 -> end).
time_stops A/B'd: 40 (default, let winners run a while) vs 0 (no cap).

Usage:
  USE_VECTORIZED_SCORING=1 python scripts/validate_leading_stock.py \
      --universe sp500 --end 2026-05-28 --parallel 6
  USE_VECTORIZED_SCORING=1 python scripts/validate_leading_stock.py \
      --universe russell3000 --min-liquidity 10000000 --end 2026-05-28 --parallel 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.universe import get_universe_tickers, liquidity_filter  # noqa: E402

RUNS = ROOT / "runs"
FORMULA_FP = ROOT / "formulas" / "leading_stock_v1.yaml"

# (name, start, end-or-None=use --end)
PERIODS = [
    ("2018", "2018-01-01", "2018-12-31"),
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023+", "2023-01-01", None),
    ("full", "2018-01-01", None),
]

_PANEL: dict | None = None  # fork-inherited shared panel


def _metrics(res) -> dict:
    """Honest metrics from a BacktestResult. Monthly returns are calendar-month
    equity changes (equity is sampled weekly; resample ME .last() -> pct_change)."""
    eq = res.equity
    tr = res.trades
    st = res.stats
    monthly = eq.resample("ME").last().pct_change().dropna()
    m = {
        "total_return": st.get("total_return"),
        "benchmark_total_return": st.get("benchmark_total_return"),
        "alpha_vs_benchmark": st.get("alpha_vs_benchmark"),
        "sharpe_ann_weekly": st.get("sharpe"),
        "max_drawdown": st.get("max_drawdown"),
        "win_rate": st.get("win_rate"),
        "n_trades": st.get("n_trades"),
        "exit_breakdown": st.get("exit_breakdown"),
        "n_months": int(len(monthly)),
        "avg_monthly_return": round(float(monthly.mean()), 5) if len(monthly) else None,
        "median_monthly_return": round(float(monthly.median()), 5) if len(monthly) else None,
        "monthly_sharpe_ann": (round(float(monthly.mean() / monthly.std() * np.sqrt(12)), 3)
                               if len(monthly) > 1 and monthly.std() else None),
    }
    if tr is not None and not tr.empty and "ret" in tr.columns:
        rets = tr["ret"].astype(float)
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        m["expectancy_per_trade"] = round(float(rets.mean()), 5)
        m["avg_win"] = round(avg_win, 5)
        m["avg_loss"] = round(avg_loss, 5)
        m["payoff_ratio"] = round(abs(avg_win / avg_loss), 3) if avg_loss != 0 else None
        if "bars_held" in tr.columns:
            m["avg_bars_held"] = round(float(tr["bars_held"].astype(float).mean()), 1)
    return m


def _run_cell(payload):
    ts_name, time_stop, pname, start, end = payload
    f = Formula.load(FORMULA_FP)
    cfg = bt.config_from_formula(f, time_stop_bars=time_stop)
    try:
        res = bt.run(_PANEL, f, start=start, end=end, cfg=cfg)
        return (ts_name, pname, _metrics(res), None)
    except Exception as exc:  # noqa: BLE001
        return (ts_name, pname, None, f"{type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--min-liquidity", type=float, default=0.0)
    ap.add_argument("--end", default="2026-05-28")
    ap.add_argument("--fetch-start", default="2017-05-01")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--time-stops", default="40,0")
    ap.add_argument("--limit", type=int, default=0, help="cap universe (smoke test)")
    args = ap.parse_args()

    os.environ.setdefault("USE_VECTORIZED_SCORING", "1")
    time_stops = [int(x) for x in args.time_stops.split(",") if x.strip() != ""]

    tickers = get_universe_tickers(args.universe)
    if args.limit > 0:
        tickers = tickers[: args.limit]
    if "SPY" not in tickers:
        tickers.append("SPY")
    print(f"[{args.universe}] fetching {len(tickers)} tickers  "
          f"{args.fetch_start} -> {args.end}", flush=True)
    data = get_universe(tickers, start=args.fetch_start, end=args.end, provider="yf")
    print(f"  fetched {len(data)} non-empty", flush=True)
    if args.min_liquidity > 0:
        before = len(data)
        keep_spy = data.get("SPY")
        data = liquidity_filter(data, min_avg_dollar_vol=args.min_liquidity)
        if keep_spy is not None:
            data["SPY"] = keep_spy
        print(f"  liquidity filter (>= ${args.min_liquidity:,.0f}): "
              f"{before} -> {len(data)}", flush=True)

    global _PANEL
    _PANEL = data

    jobs = []
    for ts in time_stops:
        ts_name = f"time_stop_{ts}"
        for pname, start, pend in PERIODS:
            end = pend or args.end
            jobs.append((ts_name, ts, pname, start, end))

    print(f"  running {len(jobs)} cells, parallel={args.parallel}", flush=True)
    results: dict = {ts: {} for ts in (f"time_stop_{t}" for t in time_stops)}
    if args.parallel > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.parallel) as pool:
            for ts_name, pname, met, err in pool.imap_unordered(_run_cell, jobs):
                results[ts_name][pname] = met if met else {"error": err}
                tag = "OK" if met else f"ERR {err}"
                print(f"    {ts_name}/{pname}: {tag}", flush=True)
    else:
        for job in jobs:
            ts_name, pname, met, err = _run_cell(job)
            results[ts_name][pname] = met if met else {"error": err}
            print(f"    {ts_name}/{pname}: {'OK' if met else err}", flush=True)

    out = {
        "universe": args.universe,
        "universe_size": len(data),
        "min_liquidity": args.min_liquidity,
        "window": [args.fetch_start, args.end],
        "formula": "leading_stock_v1",
        "time_stops": time_stops,
        "results": results,
    }
    RUNS.mkdir(exist_ok=True)
    fp = RUNS / f"validate_leading_stock_{args.universe}.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {fp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
