#!/usr/bin/env python3
"""10-year daily simulation on a broad (Russell) universe — the real test of how
much the strategies actually hit and profit, and how reliable they are.

Runs managed mode (daily selection at weekly anchors + daily stop/trail/tp exits)
over 2016->now. Two strategies:
  leading_high52 — the SHIPPED leading_stock_v1 (high52 on, managed default).
  vcp_ride       — the robust core (VCP weighting + fully-invested ride).

Rich honest metrics: return/alpha/sharpe/maxDD/win-rate/payoff/avg-hold/exit mix.
SINGLE-PROCESS (--parallel 1, no fork pool). Survivorship caveat: the Russell list
is today's members -> absolute returns optimistic; trust relative + the trade stats.
"""
from __future__ import annotations

import argparse
import copy
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
from engine.universe import get_universe_tickers, liquidity_filter  # noqa: E402

FORMULA_FP = ROOT / "formulas" / "leading_stock_v1.yaml"
VCP_WEIGHTS = {
    "momentum": 0.18, "trend": 0.15, "rsi": 0.05,
    "high52_proximity": 0.18, "breakout": 0.10, "breakout_thrust": 0.12,
    "atr_contraction": 0.12, "volume_dryup": 0.10, "bb_squeeze": 0.08,
    "gap": 0.05, "volatility": 0.00, "trend_template": 0.00,
}
RIDE = {"risk_per_trade": 0.0, "time_stop_bars": 0}
_PANEL = None


def _vcp() -> Formula:
    raw = copy.deepcopy(Formula.load(FORMULA_FP).raw)
    raw["timeframe_score_weights"] = dict(VCP_WEIGHTS)
    return Formula(raw)


def _metrics(res) -> dict:
    eq, tr, st = res.equity, res.trades, res.stats
    monthly = eq.resample("ME").last().pct_change().dropna()
    m = {
        "total_return": st.get("total_return"),
        "benchmark_total_return": st.get("benchmark_total_return"),
        "alpha": st.get("alpha_vs_benchmark"),
        "sharpe": st.get("sharpe"),
        "max_dd": st.get("max_drawdown"),
        "win_rate": st.get("win_rate"),
        "n_trades": st.get("n_trades"),
        "exit_breakdown": st.get("exit_breakdown"),
        "n_months": int(len(monthly)),
        "avg_monthly_pct": round(float(monthly.mean()) * 100, 3) if len(monthly) else None,
        "cagr_pct": (round(((1 + st.get("total_return", 0)) ** (12 / len(monthly)) - 1) * 100, 2)
                     if len(monthly) > 1 else None),
    }
    if tr is not None and not tr.empty and "ret" in tr.columns:
        r = tr["ret"].astype(float)
        w, l = r[r > 0], r[r <= 0]
        m["avg_win_pct"] = round(float(w.mean()) * 100, 2) if len(w) else 0.0
        m["avg_loss_pct"] = round(float(l.mean()) * 100, 2) if len(l) else 0.0
        m["payoff"] = round(abs(float(w.mean()) / float(l.mean())), 2) if len(l) and float(l.mean()) else None
        m["expectancy_pct"] = round(float(r.mean()) * 100, 3)
        if "bars_held" in tr.columns:
            m["avg_bars_held"] = round(float(tr["bars_held"].astype(float).mean()), 1)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="russell1000")
    ap.add_argument("--min-liquidity", type=float, default=0.0)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--fetch-start", default="2015-01-01")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.environ.setdefault("USE_VECTORIZED_SCORING", "1")

    tickers = get_universe_tickers(args.universe)
    if "SPY" not in tickers:
        tickers.append("SPY")
    print(f"[{args.universe}] fetching {len(tickers)} tickers {args.fetch_start}->{args.end}", flush=True)
    data = get_universe(tickers, start=args.fetch_start, end=args.end, provider="yf")
    print(f"  fetched {len(data)} non-empty", flush=True)
    if args.min_liquidity > 0:
        before = len(data); spy = data.get("SPY")
        data = liquidity_filter(data, min_avg_dollar_vol=args.min_liquidity)
        if spy is not None:
            data["SPY"] = spy
        print(f"  liquidity >= ${args.min_liquidity:,.0f}: {before} -> {len(data)}", flush=True)
    global _PANEL
    _PANEL = data

    configs = {"leading_high52": (Formula.load(FORMULA_FP), {}), "vcp_ride": (_vcp(), RIDE)}
    windows = [("full_10yr", args.start), ("bear_2022", "2022-01-01")]
    wend = {"full_10yr": args.end, "bear_2022": "2022-12-31"}

    results: dict = {}
    for cname, (f, ov) in configs.items():
        results[cname] = {}
        for wname, wstart in windows:
            cfg = bt.config_from_formula(f, **ov)
            res = bt.run(_PANEL, f, start=wstart, end=wend[wname], cfg=cfg)
            results[cname][wname] = _metrics(res)
            print(f"    {cname}/{wname}: OK", flush=True)

    out = {"universe": args.universe, "universe_size": len(data),
           "min_liquidity": args.min_liquidity, "window": [args.start, args.end],
           "results": results}
    fp = Path(args.out) if args.out else ROOT / "runs" / f"russell_sim_{args.universe}.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {fp}", flush=True)
    for wname, _ in windows:
        spy = results["leading_high52"][wname]["benchmark_total_return"]
        print(f"\n## {wname}  (SPY {spy:+.3f})  universe={len(data)}")
        for c in configs:
            x = results[c][wname]
            print(f"  {c:15s} ret={x['total_return']:+.3f} cagr={x.get('cagr_pct')}% "
                  f"alpha={x['alpha']:+.3f} sharpe={x['sharpe']:.2f} DD={x['max_dd']:.3f}")
            print(f"  {'':15s} win={x['win_rate']} payoff={x.get('payoff')} "
                  f"avgW={x.get('avg_win_pct')}% avgL={x.get('avg_loss_pct')}% "
                  f"hold={x.get('avg_bars_held')} trades={x['n_trades']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
