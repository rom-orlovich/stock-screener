#!/usr/bin/env python3
"""The combination: VCP + confirm + ride + absolute-momentum CASH ROTATION.

Tests whether adding a regime filter (rotate to cash when SPY 12-1 momentum <=
SHY) to the VCP+ride setup removes the bear-market weakness while keeping the bull
alpha. The regime gate now works in managed mode (engine/backtest._run_managed,
gated on an `absolute_momentum` block).

Configs on the SAME panel (single-process):
  vcp_ride        — VCP weighting + ride (no regime filter).
  vcp_ride_regime — same + absolute_momentum (SPY vs SHY, 12-1) -> flatten to cash
                    while risk-off.

Windows include bear_2022 ISOLATED — the decisive test. Honest caveat: sp500 is
survivorship-biased, so trust RELATIVE deltas (regime on vs off), not absolutes.
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
PERIODS = [
    ("full_2023", "2023-01-01", None),
    ("full_2018", "2018-01-01", None),
    ("bull_2021", "2021-01-01", "2021-12-31"),
    ("bear_2022", "2022-01-01", "2022-12-31"),
]
VCP_WEIGHTS = {
    "momentum": 0.18, "trend": 0.15, "rsi": 0.05,
    "high52_proximity": 0.18, "breakout": 0.10, "breakout_thrust": 0.12,
    "atr_contraction": 0.12, "volume_dryup": 0.10, "bb_squeeze": 0.08,
    "gap": 0.05, "volatility": 0.00, "trend_template": 0.00,
}
ABS_MOM = {"benchmark": "SPY", "bond_proxy": "SHY", "lookback": 252,
           "skip_recent": 21, "cash_fallback": "SHY"}
RIDE = {"risk_per_trade": 0.0, "time_stop_bars": 0}
_PANEL = None


def _vcp(regime: bool) -> Formula:
    raw = copy.deepcopy(Formula.load(FORMULA_FP).raw)
    raw["timeframe_score_weights"] = dict(VCP_WEIGHTS)
    if regime:
        raw["absolute_momentum"] = dict(ABS_MOM)
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
        out["payoff"] = (round(abs(float(w.mean()) / float(l.mean())), 2)
                         if len(w) and len(l) and float(l.mean()) != 0 else None)
        if "exit" in tr.columns:
            out["regime_exits"] = int((tr["exit"] == "regime").sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--fetch-start", default="2016-05-01")
    ap.add_argument("--out", default="runs/diag_vcp_regime.json")
    args = ap.parse_args()
    os.environ.setdefault("USE_VECTORIZED_SCORING", "1")

    tickers = get_universe_tickers(args.universe)
    for extra in ("SPY", "SHY"):
        if extra not in tickers:
            tickers.append(extra)
    print(f"[{args.universe}] fetching {len(tickers)} tickers (+SPY/SHY)", flush=True)
    global _PANEL
    _PANEL = get_universe(tickers, start=args.fetch_start, end=args.end, provider="yf")
    print(f"  fetched {len(_PANEL)} non-empty; SHY present={'SHY' in _PANEL}", flush=True)

    configs = {"vcp_ride": _vcp(False), "vcp_ride_regime": _vcp(True)}
    results: dict = {c: {} for c in configs}
    for cname, f in configs.items():
        for pname, start, pend in PERIODS:
            cfg = bt.config_from_formula(f, **RIDE)
            res = bt.run(_PANEL, f, start=start, end=pend or args.end, cfg=cfg)
            results[cname][pname] = _m(res)
            print(f"    {cname}/{pname}: OK", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}", flush=True)
    for pname, _, _ in PERIODS:
        spy = results["vcp_ride"][pname]["benchmark_total_return"]
        print(f"\n## {pname}  (SPY {spy:+.3f})")
        for c in configs:
            x = results[c][pname]
            print(f"  {c:16s} ret={x['total_return']:+.3f} alpha={x['alpha']:+.3f} "
                  f"sharpe={x['sharpe']:.2f} DD={x['max_dd']:.3f} payoff={x.get('payoff')} "
                  f"trd={x['n_trades']} regime_exits={x.get('regime_exits')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
