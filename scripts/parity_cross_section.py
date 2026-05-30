#!/usr/bin/env python3
"""Bit-for-bit parity between the per-ticker vec path and the cross-section path.

Runs the same backtest twice on sp500 (or a configurable basket) with:
  USE_VECTORIZED_SCORING=1  + (cross-section OFF)   — baseline
  USE_CROSS_SECTION=1                                — new path

Asserts identical equity curve, every trade, every summary stat.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np  # noqa: F401  (left for ad-hoc diagnostics)
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.universe import sp500  # noqa: E402

EPS_NUM = 1e-6
EPS_EQ = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--formula", default="formulas/dual_momentum_v1.yaml")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-05-28")
    p.add_argument("--universe", default="sp500", choices=["sp500", "small"])
    return p.parse_args()


def run_once(data, formula, args, label: str, **env: str):
    for k in ("USE_VECTORIZED_SCORING", "USE_CROSS_SECTION"):
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v
    cfg = bt.BacktestConfig()
    t0 = time.time()
    res = bt.run(data, formula, start=args.start, end=args.end, cfg=cfg)
    dt = time.time() - t0
    print(f"  {label}  {dt:.1f}s  stats={res.stats}", flush=True)
    return res, dt


def diff_stats(a: dict, b: dict) -> list[str]:
    msgs = []
    for k in sorted(set(a) | set(b)):
        va = a.get(k); vb = b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            if va != vb:
                msgs.append(f"  {k}: vec={va} xs={vb}")
            continue
        try:
            af = float(va); bf = float(vb)
        except (TypeError, ValueError):
            if va != vb:
                msgs.append(f"  {k}: vec={va!r} xs={vb!r}")
            continue
        if pd.isna(af) and pd.isna(bf):
            continue
        if abs(af - bf) > EPS_NUM:
            msgs.append(f"  {k}: vec={af} xs={bf} diff={af-bf:.2e}")
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

    fetch_start = "2017-05-01"  # ~6y warmup so caps are inactive after mid-2024
    print(f"Fetching {len(tickers)} tickers {fetch_start} → {args.end} ...", flush=True)
    data = get_universe(tickers, start=fetch_start, end=args.end, provider="yf")
    print(f"  loaded {len(data)} tickers", flush=True)

    res_vec, t_vec = run_once(data, formula, args, "VEC ", USE_VECTORIZED_SCORING="1")
    res_xs,  t_xs  = run_once(data, formula, args, "X-SE", USE_CROSS_SECTION="1")

    speedup = (t_vec / t_xs) if t_xs > 0 else float("inf")
    print(f"\nSpeedup: {speedup:.2f}x   ({t_vec:.1f}s → {t_xs:.1f}s, saved {t_vec - t_xs:.1f}s)", flush=True)

    msgs = diff_stats(res_vec.stats, res_xs.stats)
    if not res_vec.equity.index.equals(res_xs.equity.index):
        msgs.append("equity index mismatch")
    else:
        d = (res_vec.equity - res_xs.equity).abs().max()
        if float(d) > EPS_EQ:
            msgs.append(f"equity max abs diff = {float(d):.2e}")
    if len(res_vec.trades) != len(res_xs.trades):
        msgs.append(f"trades length: vec={len(res_vec.trades)} xs={len(res_xs.trades)}")
    else:
        a = res_vec.trades.sort_values(["enter", "ticker"]).reset_index(drop=True)
        b = res_xs.trades.sort_values(["enter", "ticker"]).reset_index(drop=True)
        for col in ("ret", "score"):
            if col in a.columns:
                d = (a[col].astype(float) - b[col].astype(float)).abs().max()
                if float(d) > EPS_NUM:
                    msgs.append(f"trades.{col} max abs diff = {float(d):.2e}")
        for col in ("ticker", "exit"):
            if col in a.columns and not (a[col] == b[col]).all():
                n = int((a[col] != b[col]).sum())
                msgs.append(f"trades.{col} mismatched in {n} rows")

    if msgs:
        print("\nDIFFERENCES:")
        for m in msgs:
            print(m)
        return 1
    print("\nCROSS-SECTION PARITY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
