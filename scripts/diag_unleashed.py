#!/usr/bin/env python3
"""Diagnostic: is leading_stock_v1's SPY gap from STOCK SELECTION or from the
risk/money-management layer (under-investment + early time-stop)?

Compares, on the SAME panel, the live leading_stock_v1 (high52 already shipped):
  managed_default — current config: risk_per_trade=0.01 (caps size -> under-
                    invested), time_stop_bars=40 (cuts winners at ~8 weeks).
  unleashed       — risk_per_trade=0 (equal-weight ~100% invested) +
                    time_stop_bars=0 (let winners run); KEEPS the trailing stop
                    as the position-management control.

If `unleashed` closes most of the SPY gap, the selection is fine and the drag was
the money/position management. Single-process, sp500 only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.universe import get_universe_tickers  # noqa: E402

FORMULA_FP = ROOT / "formulas" / "leading_stock_v1.yaml"
PERIODS = [("full_2023", "2023-01-01", None), ("full_2018", "2018-01-01", None)]
CONFIGS = {
    "managed_default": {},
    "unleashed": {"risk_per_trade": 0.0, "time_stop_bars": 0},
}
_PANEL = None


def _m(res) -> dict:
    st = res.stats
    return {
        "total_return": st.get("total_return"),
        "benchmark_total_return": st.get("benchmark_total_return"),
        "alpha": st.get("alpha_vs_benchmark"),
        "sharpe": st.get("sharpe"),
        "max_dd": st.get("max_drawdown"),
        "n_trades": st.get("n_trades"),
        "win_rate": st.get("win_rate"),
        "avg_bars_held": (round(float(res.trades["bars_held"].astype(float).mean()), 1)
                          if res.trades is not None and not res.trades.empty
                          and "bars_held" in res.trades.columns else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--fetch-start", default="2017-05-01")
    ap.add_argument("--out", default="runs/diag_unleashed.json")
    args = ap.parse_args()
    os.environ.setdefault("USE_VECTORIZED_SCORING", "1")

    tickers = get_universe_tickers(args.universe)
    if "SPY" not in tickers:
        tickers.append("SPY")
    print(f"[{args.universe}] fetching {len(tickers)} tickers", flush=True)
    global _PANEL
    _PANEL = get_universe(tickers, start=args.fetch_start, end=args.end, provider="yf")
    print(f"  fetched {len(_PANEL)} non-empty", flush=True)

    f = Formula.load(FORMULA_FP)
    results: dict = {c: {} for c in CONFIGS}
    for cname, ov in CONFIGS.items():
        for pname, start, pend in PERIODS:
            cfg = bt.config_from_formula(f, **ov)
            res = bt.run(_PANEL, f, start=start, end=pend or args.end, cfg=cfg)
            results[cname][pname] = _m(res)
            print(f"    {cname}/{pname}: OK", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}", flush=True)
    # pretty
    for pname, _, _ in PERIODS:
        spy = results["managed_default"][pname]["benchmark_total_return"]
        print(f"\n## {pname}  (SPY {spy:+.3f})")
        for c in CONFIGS:
            x = results[c][pname]
            print(f"  {c:16s} ret={x['total_return']:+.3f} alpha={x['alpha']:+.3f} "
                  f"sharpe={x['sharpe']:.2f} DD={x['max_dd']:.3f} "
                  f"trd={x['n_trades']} hold={x['avg_bars_held']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
