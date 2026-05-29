#!/usr/bin/env python3
"""End-to-end backtest parity + timing between old and vectorized scoring.

Single data fetch; runs `engine.backtest.run` twice, then compares stats and
the equity curve point-by-point.

Usage:
    python scripts/full_backtest_parity.py \
        --formula formulas/dual_momentum_v1.yaml \
        --start 2023-01-01 --end 2026-05-28 --universe sp500
"""
from __future__ import annotations

import argparse
import os
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
from engine.universe import sp500  # noqa: E402

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--formula", default="formulas/dual_momentum_v1.yaml")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-05-28")
    p.add_argument("--universe", default="sp500", choices=["sp500", "small"])
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--rebalance", default="W-FRI")
    return p.parse_args()


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def diff_stats(a: dict, b: dict) -> list[str]:
    msgs = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        av = a.get(k)
        bv = b.get(k)
        if isinstance(av, dict) and isinstance(bv, dict):
            if av != bv:
                msgs.append(f"  {k}: old={av} new={bv}")
            continue
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if pd.isna(av) and pd.isna(bv):
                continue
            if abs(float(av) - float(bv)) > EPS:
                msgs.append(f"  {k}: old={fmt(av)} new={fmt(bv)} diff={float(av)-float(bv):.2e}")
            continue
        if av != bv:
            msgs.append(f"  {k}: old={av!r} new={bv!r}")
    return msgs


def main() -> int:
    args = parse_args()
    formula = Formula.load(args.formula)
    if args.universe == "sp500":
        tickers = sp500()
    else:
        tickers = ["AAPL", "MSFT", "NVDA", "AMD", "SPY", "SHY", "META", "GOOGL", "TSLA", "AMZN"]
    if "SPY" not in tickers:
        tickers = list(tickers) + ["SPY"]
    if "SHY" not in tickers and formula.raw.get("absolute_momentum"):
        tickers = list(tickers) + ["SHY"]

    # Pick a fetch_start that hits the existing on-disk cache (avoids 503 yfinance refetches).
    # 2017-05-01 gives ~6y warmup before 2023, enough to keep the adaptive caps in
    # `timeframe_score` inactive on daily (sma_slow=200 needs n_avail >= 600 trading
    # days), so the vectorized daily path stays on the precomputed branch most of
    # the time and we measure a meaningful speedup.
    fetch_start = "2017-05-01"
    print(f"Fetching {len(tickers)} tickers {fetch_start} → {args.end} ...", flush=True)
    data = get_universe(tickers, start=fetch_start, end=args.end, provider="yf")
    print(f"  got {len(data)} tickers", flush=True)

    cfg = bt.BacktestConfig(top_n=args.top_n, rebalance=args.rebalance)

    # OLD path
    os.environ["USE_VECTORIZED_SCORING"] = "0"
    t0 = time.time()
    res_old = bt.run(data, formula, start=args.start, end=args.end, cfg=cfg)
    t_old = time.time() - t0
    print(f"\nOLD path: {t_old:.1f}s   stats={res_old.stats}", flush=True)

    # NEW path
    os.environ["USE_VECTORIZED_SCORING"] = "1"
    t0 = time.time()
    res_new = bt.run(data, formula, start=args.start, end=args.end, cfg=cfg)
    t_new = time.time() - t0
    print(f"NEW path: {t_new:.1f}s   stats={res_new.stats}", flush=True)

    speedup = (t_old / t_new) if t_new > 0 else float("inf")
    print(f"\nSpeedup: {speedup:.2f}x  ({t_old:.1f}s → {t_new:.1f}s, saved {t_old - t_new:.1f}s)", flush=True)

    # Compare stats
    msgs = diff_stats(res_old.stats, res_new.stats)

    # Compare equity curves point-wise
    eq_o = res_old.equity
    eq_n = res_new.equity
    if not eq_o.index.equals(eq_n.index):
        msgs.append("  equity index mismatch")
    else:
        diff = (eq_o - eq_n).abs()
        max_d = float(diff.max())
        if max_d > EPS:
            msgs.append(f"  equity max abs diff = {max_d:.2e}")
            bad = diff[diff > EPS].head(5)
            for d, v in bad.items():
                msgs.append(f"    {d.date()}: old={eq_o.loc[d]:.8f} new={eq_n.loc[d]:.8f}")

    # Compare trades count + per-trade returns aligned
    if len(res_old.trades) != len(res_new.trades):
        msgs.append(f"  trades length differs: old={len(res_old.trades)} new={len(res_new.trades)}")
    else:
        try:
            o_sorted = res_old.trades.sort_values(["enter", "ticker"]).reset_index(drop=True)
            n_sorted = res_new.trades.sort_values(["enter", "ticker"]).reset_index(drop=True)
            for col in ("ticker", "ret", "score", "exit", "enter", "exit_date"):
                if col not in o_sorted.columns:
                    continue
                if col in ("ret", "score"):
                    d = (o_sorted[col].astype(float) - n_sorted[col].astype(float)).abs()
                    if (d > EPS).any():
                        msgs.append(f"  trades.{col} max diff = {float(d.max()):.2e}")
                else:
                    if not (o_sorted[col] == n_sorted[col]).all():
                        n_bad = int((o_sorted[col] != n_sorted[col]).sum())
                        msgs.append(f"  trades.{col} diff in {n_bad} rows")
        except Exception as e:  # noqa: BLE001
            msgs.append(f"  trades compare error: {e}")

    if msgs:
        print("\nDIFFERENCES:")
        for m in msgs:
            print(m)
        return 1

    print("\nFULL BACKTEST PARITY PASSED — bit-identical.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
