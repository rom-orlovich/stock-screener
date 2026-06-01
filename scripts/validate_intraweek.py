#!/usr/bin/env python3
"""ON-vs-OFF validation for the additive intra-week managed entry.

Single process, sequential cells (zero forking -> zero OOM risk). Fetches ONE
wide panel (warm pickle cache) and slices every cell from it via bt.run, so the
universe is read once. For each cell runs leading_stock_v1 with intra-week entry
OFF (weekly-only, the parity default) and ON, and reports the full metric set the
brief asks for: total return, sharpe, max-DD, avg/median MONTHLY return, win-rate,
W/L payoff, n_trades, avg_bars_held, turnover (trades/month), alpha, exits.

    cd ~/sc-phaseA && USE_VECTORIZED_SCORING=1 \
      /home/madma/stock-screener/.venv/bin/python scripts/validate_intraweek.py \
      --universe sp500 --wide-start 2017-05-01 --wide-end 2026-05-28 [--min-liquidity 5e7]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.universe import get_universe_tickers, liquidity_filter  # noqa: E402

FORMULA = ROOT / "formulas" / "leading_stock_v1.yaml"

# (name, start, end) — end=None means run to the wide-end.
CELLS = [
    ("full_2018", "2018-01-01", None),
    ("full_2023", "2023-01-01", None),
    ("bull_2021", "2021-01-01", "2021-12-31"),
    ("bear_2022", "2022-01-01", "2022-12-31"),
    ("ai_2023_2024", "2023-01-01", "2024-12-31"),
]


def _cfg(raw: dict, intraweek: bool) -> bt.BacktestConfig:
    """leading_stock_v1's backtest block overlaid on the dataclass defaults, with
    intraweek_entry forced to the requested value."""
    cfg = bt.BacktestConfig(benchmark_ticker="SPY")
    blk = raw.get("backtest") or {}
    valid = {f.name for f in dataclasses.fields(bt.BacktestConfig)}
    for k, v in blk.items():
        if k in valid:
            setattr(cfg, k, v)
    cfg.intraweek_entry = intraweek
    return cfg


def _metrics(res: bt.BacktestResult, cfg: bt.BacktestConfig) -> dict:
    s = res.stats
    tr = res.trades
    eq = res.equity
    # Monthly returns from the (weekly-sampled) equity curve, resampled to ME.
    monthly = eq.resample("ME").last().pct_change().dropna()
    avg_m = float(monthly.mean()) if len(monthly) else 0.0
    med_m = float(monthly.median()) if len(monthly) else 0.0
    n_months = max(1, len(monthly))
    # Payoff = avg win / |avg loss| on realised trade returns.
    payoff = None
    avg_win = avg_loss = None
    avg_bars = None
    if not tr.empty:
        wins = tr.loc[tr["ret"] > 0, "ret"]
        losses = tr.loc[tr["ret"] < 0, "ret"]
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        payoff = round(avg_win / abs(avg_loss), 3) if avg_loss else None
        if "bars_held" in tr.columns:
            avg_bars = round(float(tr["bars_held"].mean()), 1)
    return {
        "total_return": s.get("total_return"),
        "sharpe": s.get("sharpe"),
        "max_drawdown": s.get("max_drawdown"),
        "win_rate": s.get("win_rate"),
        "payoff": payoff,
        "avg_win": round(avg_win, 4) if avg_win is not None else None,
        "avg_loss": round(avg_loss, 4) if avg_loss is not None else None,
        "n_trades": s.get("n_trades"),
        "avg_bars_held": avg_bars,
        "avg_monthly_ret": round(avg_m, 4),
        "median_monthly_ret": round(med_m, 4),
        "trades_per_month": round(s.get("n_trades", 0) / n_months, 2),
        "n_months": n_months,
        "benchmark_total_return": s.get("benchmark_total_return"),
        "alpha_vs_benchmark": s.get("alpha_vs_benchmark"),
        "exit_breakdown": s.get("exit_breakdown", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--wide-start", default="2017-05-01")
    ap.add_argument("--wide-end", default="2026-05-28")
    ap.add_argument("--min-liquidity", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = Formula.load(str(FORMULA)).raw
    tickers = get_universe_tickers(args.universe)
    if "SPY" not in tickers:
        tickers = list(tickers) + ["SPY"]
    print(f"loading {len(tickers)} tickers ({args.universe}) "
          f"{args.wide_start}->{args.wide_end} ...", flush=True)
    t0 = time.time()
    panel = get_universe(tickers, start=args.wide_start, end=args.wide_end, provider="yf")
    if args.min_liquidity > 0:
        before = len(panel)
        spy = panel.get("SPY")
        panel = liquidity_filter(panel, min_avg_dollar_vol=args.min_liquidity)
        if spy is not None and "SPY" not in panel:
            panel["SPY"] = spy
        print(f"liquidity filter: {before} -> {len(panel)} tickers "
              f"(>= ${args.min_liquidity:,.0f})", flush=True)
    print(f"panel ready: {len(panel)} tickers in {time.time()-t0:.0f}s", flush=True)

    # Build the shared indicator bank ONCE (formula-independent union of
    # indicators over the full panel). Every cell's bt.run then ASSEMBLES its
    # precompute from this instead of recomputing — ~10x fewer heavy passes than
    # rebuilding precompute per (cell, mode). Bit-identical (scripts/parity_bank).
    f = Formula(raw=raw)
    tb = time.time()
    from engine.bank import build_bank, collect_specs  # noqa: E402
    specs = collect_specs([f])
    bank = {t: build_bank(df, specs) for t, df in panel.items()}
    print(f"bank built: {len(bank)} tickers in {time.time()-tb:.0f}s", flush=True)

    results: dict[str, dict] = {}
    for name, start, end in CELLS:
        cell_end = end or args.wide_end
        results[name] = {}
        for mode_name, iw in (("OFF", False), ("ON", True)):
            cfg = _cfg(raw, iw)
            tc = time.time()
            res = bt.run(panel, f, start=start, end=cell_end, cfg=cfg, bank=bank)
            m = _metrics(res, cfg)
            m["_secs"] = round(time.time() - tc, 1)
            results[name][mode_name] = m
            print(f"[{name:14s} {mode_name:3s}] ret={m['total_return']} sharpe={m['sharpe']} "
                  f"dd={m['max_drawdown']} wr={m['win_rate']} payoff={m['payoff']} "
                  f"n={m['n_trades']} bars={m['avg_bars_held']} "
                  f"tpm={m['trades_per_month']} ({m['_secs']}s)", flush=True)

    out = Path(args.out) if args.out else (ROOT / "runs" /
          f"intraweek_validate_{args.universe}_{int(args.min_liquidity)}.json")
    out.write_text(json.dumps({"universe": args.universe,
                               "wide": [args.wide_start, args.wide_end],
                               "min_liquidity": args.min_liquidity,
                               "cells": results}, indent=2))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
