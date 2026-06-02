#!/usr/bin/env python3
"""A/B harness — does adding the two research signals improve leading_stock_v1?

Compares four variants of leading_stock_v1 on the SAME shared panel:
  baseline   — the live YAML, unchanged.
  high52     — + high52_proximity weight (George & Hwang 52-week-high nearness).
  trend_tmpl — + trend_template weight (Minervini 8-rule objective screen).
  both       — + both, split.

Each new weight is ADDED on top of the baseline weights and renormalized by the
scorer (_norm) — the standard "marginal contribution of one signal" test. ONE
documented weight per variant (no sweep — the deep-research brief warns against
curve-fitting). Every other knob (managed mode, exits, time_stop) is identical
across variants, so any delta is attributable to the signal.

MACHINE DISCIPLINE: single-process by default (--parallel 1). Fetch the widest
window ONCE (CLAUDE.md #4), slice per period in-memory. sp500 only.

Usage (run ONLY when auto_tune is NOT running):
  USE_VECTORIZED_SCORING=1 python scripts/ab_research_signals.py \
      --universe sp500 --end 2026-06-01 --parallel 1
"""
from __future__ import annotations

import argparse
import copy
import json
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

# variant name -> extra timeframe_score_weights merged on top of baseline.
VARIANTS: dict[str, dict[str, float]] = {
    "baseline": {},
    "high52": {"high52_proximity": 0.20},
    "trend_tmpl": {"trend_template": 0.20},
    "both": {"high52_proximity": 0.15, "trend_template": 0.15},
}

# (name, start, end-or-None=use --end)
PERIODS = [
    ("full_2023", "2023-01-01", None),
    ("full_2018", "2018-01-01", None),
    ("bull_2021", "2021-01-01", "2021-12-31"),
    ("bear_2022", "2022-01-01", "2022-12-31"),
    ("ai_2023_2024", "2023-01-01", "2024-12-31"),
]

_PANEL: dict | None = None


def _variant_formula(extra: dict[str, float]) -> Formula:
    raw = copy.deepcopy(Formula.load(FORMULA_FP).raw)
    for k, v in extra.items():
        raw["timeframe_score_weights"][k] = v
    return Formula(raw)


def _metrics(res) -> dict:
    """Honest metrics off res.equity / res.trades / res.stats."""
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
    vname, extra, pname, start, end = payload
    f = _variant_formula(extra)
    cfg = bt.config_from_formula(f)
    try:
        res = bt.run(_PANEL, f, start=start, end=end, cfg=cfg)
        return (vname, pname, _metrics(res), None)
    except Exception as exc:  # noqa: BLE001
        return (vname, pname, None, f"{type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--min-liquidity", type=float, default=0.0)
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--fetch-start", default="2017-05-01")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap universe (smoke test)")
    ap.add_argument("--variants", default="", help="comma subset of variant names")
    ap.add_argument("--periods", default="", help="comma subset of period names")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ.setdefault("USE_VECTORIZED_SCORING", "1")

    variants = VARIANTS if not args.variants else {
        k: VARIANTS[k] for k in args.variants.split(",") if k in VARIANTS}
    periods = PERIODS if not args.periods else [
        p for p in PERIODS if p[0] in set(args.periods.split(","))]

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
    for vname, extra in variants.items():
        for pname, start, pend in periods:
            jobs.append((vname, extra, pname, start, pend or args.end))

    print(f"  running {len(jobs)} cells (parallel={args.parallel})", flush=True)
    results: dict = {v: {} for v in variants}
    for job in jobs:
        vname, pname, met, err = _run_cell(job)
        results[vname][pname] = met if met else {"error": err}
        print(f"    {vname}/{pname}: {'OK' if met else err}", flush=True)

    out = {
        "universe": args.universe,
        "universe_size": len(data),
        "min_liquidity": args.min_liquidity,
        "window": [args.fetch_start, args.end],
        "formula": "leading_stock_v1",
        "variants": {k: v for k, v in variants.items()},
        "results": results,
    }
    RUNS.mkdir(exist_ok=True)
    fp = Path(args.out) if args.out else RUNS / f"ab_research_signals_{args.universe}.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {fp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
