#!/usr/bin/env python3
"""How far can the FULL technical pre-breakout setup go? VCP + confirm + ride.

VCP = Minervini Volatility-Contraction-Pattern weighting: up-weight the
contraction terms (atr_contraction / volume_dryup / bb_squeeze, all gated to
Stage-2 by trend_gate_quietness) on top of leadership (momentum + high52 + trend)
and the volume-confirmed break (breakout + breakout_thrust). Combined with the
"ride" config (fully invested, no time-stop, trailing stop as the only exit
control) — i.e. don't predict the break, confirm it then let winners run.

Configs compared on the SAME sp500 panel (single-process):
  shipped     — live leading_stock_v1 (high52 on, managed default: risk 1%,
                time_stop 40). The reference of record.
  high52_ride — shipped weights + ride (risk 0 = equal-weight ~100% invested,
                time_stop 0). Isolates the RIDE effect.
  vcp_ride    — VCP weighting + ride. Isolates the full technical setup.

ONE documented weight set per config (no sweep — anti-overfit). Honest caveat:
the sp500 universe is survivorship-biased (today's members), so ABSOLUTE returns
are optimistic; trust the RELATIVE deltas between configs.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.universe import get_universe_tickers  # noqa: E402

FORMULA_FP = ROOT / "formulas" / "leading_stock_v1.yaml"
PERIODS = [("full_2023", "2023-01-01", None), ("full_2018", "2018-01-01", None)]

# VCP weighting: ~30% contraction coil, ~22% the break, ~36% leadership/trend.
VCP_WEIGHTS = {
    "momentum": 0.18, "trend": 0.15, "rsi": 0.05,
    "high52_proximity": 0.18, "breakout": 0.10, "breakout_thrust": 0.12,
    "atr_contraction": 0.12, "volume_dryup": 0.10, "bb_squeeze": 0.08,
    "gap": 0.05, "volatility": 0.00, "trend_template": 0.00,
}
RIDE = {"risk_per_trade": 0.0, "time_stop_bars": 0}
_PANEL = None


def _vcp_formula() -> Formula:
    raw = copy.deepcopy(Formula.load(FORMULA_FP).raw)
    raw["timeframe_score_weights"] = dict(VCP_WEIGHTS)
    return Formula(raw)


def _m(res) -> dict:
    st = res.stats
    tr = res.trades
    out = {
        "total_return": st.get("total_return"),
        "benchmark_total_return": st.get("benchmark_total_return"),
        "alpha": st.get("alpha_vs_benchmark"),
        "sharpe": st.get("sharpe"),
        "max_dd": st.get("max_drawdown"),
        "win_rate": st.get("win_rate"),
        "n_trades": st.get("n_trades"),
    }
    if tr is not None and not tr.empty and "ret" in tr.columns:
        r = tr["ret"].astype(float)
        w = r[r > 0]; l = r[r <= 0]
        out["avg_win"] = round(float(w.mean()), 4) if len(w) else 0.0
        out["avg_loss"] = round(float(l.mean()), 4) if len(l) else 0.0
        out["payoff"] = round(abs(out["avg_win"] / out["avg_loss"]), 2) if out["avg_loss"] else None
        if "bars_held" in tr.columns:
            out["avg_bars_held"] = round(float(tr["bars_held"].astype(float).mean()), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--fetch-start", default="2017-05-01")
    ap.add_argument("--out", default="runs/diag_vcp_ride.json")
    args = ap.parse_args()
    os.environ.setdefault("USE_VECTORIZED_SCORING", "1")

    tickers = get_universe_tickers(args.universe)
    if "SPY" not in tickers:
        tickers.append("SPY")
    print(f"[{args.universe}] fetching {len(tickers)} tickers", flush=True)
    global _PANEL
    _PANEL = get_universe(tickers, start=args.fetch_start, end=args.end, provider="yf")
    print(f"  fetched {len(_PANEL)} non-empty", flush=True)

    shipped = Formula.load(FORMULA_FP)
    vcp = _vcp_formula()
    configs = {
        "shipped": (shipped, {}),
        "high52_ride": (shipped, RIDE),
        "vcp_ride": (vcp, RIDE),
    }

    results: dict = {c: {} for c in configs}
    for cname, (f, ov) in configs.items():
        for pname, start, pend in PERIODS:
            cfg = bt.config_from_formula(f, **ov)
            res = bt.run(_PANEL, f, start=start, end=pend or args.end, cfg=cfg)
            results[cname][pname] = _m(res)
            print(f"    {cname}/{pname}: OK", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}", flush=True)
    for pname, _, _ in PERIODS:
        spy = results["shipped"][pname]["benchmark_total_return"]
        print(f"\n## {pname}  (SPY {spy:+.3f})")
        for c in configs:
            x = results[c][pname]
            print(f"  {c:12s} ret={x['total_return']:+.3f} alpha={x['alpha']:+.3f} "
                  f"sharpe={x['sharpe']:.2f} DD={x['max_dd']:.3f} "
                  f"payoff={x.get('payoff')} hold={x.get('avg_bars_held')} trd={x['n_trades']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
