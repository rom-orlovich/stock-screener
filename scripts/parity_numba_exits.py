#!/usr/bin/env python3
"""Bit-for-bit parity between the Python and Numba exit-loop paths.

Runs the SAME backtest twice with different USE_NUMBA_EXITS values and asserts
the equity curve, every trade's (ret, exit), and the summary stats are
identical within EPS.

Usage:
    python scripts/parity_numba_exits.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.universe import sp500  # noqa: E402

EPS = 1e-9  # Numba math should match Python float to last bit.


def run_once(data, formula, start, end, cfg, use_numba: bool) -> bt.BacktestResult:
    os.environ["USE_NUMBA_EXITS"] = "1" if use_numba else "0"
    t0 = time.time()
    res = bt.run(data, formula, start=start, end=end, cfg=cfg)
    dt = time.time() - t0
    print(f"  numba={use_numba}  {dt:.1f}s  stats={res.stats}", flush=True)
    return res, dt


def compare(res_a, res_b, label_a, label_b) -> list[str]:
    msgs = []
    if not res_a.equity.index.equals(res_b.equity.index):
        msgs.append("equity index mismatch")
    else:
        d = (res_a.equity - res_b.equity).abs().max()
        if float(d) > EPS:
            msgs.append(f"equity max abs diff = {float(d):.2e}")
    if len(res_a.trades) != len(res_b.trades):
        msgs.append(f"trades length: {label_a}={len(res_a.trades)} {label_b}={len(res_b.trades)}")
    else:
        a = res_a.trades.sort_values(["enter", "ticker"]).reset_index(drop=True)
        b = res_b.trades.sort_values(["enter", "ticker"]).reset_index(drop=True)
        for col in ("ret", "score"):
            if col in a.columns:
                d = (a[col].astype(float) - b[col].astype(float)).abs().max()
                if float(d) > EPS:
                    msgs.append(f"trades.{col} max abs diff = {float(d):.2e}")
        for col in ("exit", "ticker"):
            if col in a.columns and not (a[col] == b[col]).all():
                n_bad = int((a[col] != b[col]).sum())
                msgs.append(f"trades.{col} mismatched in {n_bad} rows")
    for k in res_a.stats:
        va = res_a.stats.get(k)
        vb = res_b.stats.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            if va != vb:
                msgs.append(f"stats.{k}: {label_a}={va} {label_b}={vb}")
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if abs(float(va) - float(vb)) > EPS:
                msgs.append(f"stats.{k}: {label_a}={va} {label_b}={vb}")
        elif va != vb:
            msgs.append(f"stats.{k}: {label_a}={va} {label_b}={vb}")
    return msgs


def main() -> int:
    formula = Formula.load(ROOT / "formulas" / "dual_momentum_v1.yaml")
    # Use a small basket so the test runs in minutes, not hours.
    tickers = ["SPY", "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "TSLA", "AMZN", "SHY"]
    start = "2023-01-01"
    end = "2026-05-28"
    fetch_start = "2021-01-01"
    print(f"Fetching {len(tickers)} tickers {fetch_start} → {end} ...", flush=True)
    data = get_universe(tickers, start=fetch_start, end=end, provider="yf")
    print(f"  loaded {len(data)} tickers", flush=True)

    # Exercise multiple exit configurations so we hit each branch (stop, trail, tp, time, hold).
    configs = [
        ("default (atr+trail+time)", bt.BacktestConfig(top_n=5)),
        ("flat stop only", bt.BacktestConfig(top_n=5, stop_loss_pct=0.05,
                                              atr_stop_mult=0.0, trailing_stop_pct=0.0,
                                              time_stop_bars=0)),
        ("tp only", bt.BacktestConfig(top_n=5, take_profit_pct=0.04,
                                       atr_stop_mult=0.0, trailing_stop_pct=0.0,
                                       time_stop_bars=0)),
        ("trail only", bt.BacktestConfig(top_n=5, trailing_stop_pct=0.05,
                                          trailing_activate_pct=0.03,
                                          atr_stop_mult=0.0, time_stop_bars=0)),
        ("time stop only", bt.BacktestConfig(top_n=5, time_stop_bars=5,
                                              atr_stop_mult=0.0, trailing_stop_pct=0.0)),
        ("no exits (hold)", bt.BacktestConfig(top_n=5, atr_stop_mult=0.0,
                                               trailing_stop_pct=0.0, time_stop_bars=0)),
    ]

    overall_fail = False
    for name, cfg in configs:
        print(f"\n== {name} ==", flush=True)
        res_py, t_py = run_once(data, formula, start, end, cfg, use_numba=False)
        res_nb, t_nb = run_once(data, formula, start, end, cfg, use_numba=True)
        msgs = compare(res_py, res_nb, "py", "nb")
        if msgs:
            overall_fail = True
            print("  DIFFERENCES:")
            for m in msgs:
                print(f"    {m}")
        else:
            speedup = t_py / t_nb if t_nb > 0 else 0.0
            print(f"  PARITY OK   numba {speedup:.2f}x faster ({t_py:.1f}s → {t_nb:.1f}s)")

    if overall_fail:
        print("\nNUMBA PARITY FAILED")
        return 1
    print("\nNUMBA PARITY PASSED across all exit configurations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
