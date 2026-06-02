#!/usr/bin/env python3
"""Regime-signal ROBUSTNESS sweep: is a faster regime filter a robust fix, or a
single overfit value?

The 12-1 (252/21) regime signal lags — it hurt the isolated bear_2022. This sweeps
the regime lookback from laggy to responsive on the SAME VCP+ride setup, across all
4 windows. Decision rule (anti-overfit): keep a faster signal ONLY if it improves
the isolated bear_2022 AND holds the full-window gains AND the trend is consistent
across the family (not one lucky value).

Configs (lookback_days / skip_days):
  none      — no regime filter (reference)
  L252_s21  — 12-1 momentum (current, laggy)
  L126_s21  — 6-month
  L63_s0    — 3-month, no skip (most responsive)

Single-process, sp500. Survivorship-biased -> trust RELATIVE deltas.
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
RIDE = {"risk_per_trade": 0.0, "time_stop_bars": 0}
# name -> (lookback, skip) or None for no regime
REGIMES = {
    "none": None,
    "L252_s21": (252, 21),
    "L126_s21": (126, 21),
    "L63_s0": (63, 0),
}
_PANEL = None


def _formula(reg) -> Formula:
    raw = copy.deepcopy(Formula.load(FORMULA_FP).raw)
    raw["timeframe_score_weights"] = dict(VCP_WEIGHTS)
    if reg is not None:
        lb, sk = reg
        raw["absolute_momentum"] = {"benchmark": "SPY", "bond_proxy": "SHY",
                                    "lookback": lb, "skip_recent": sk, "cash_fallback": "SHY"}
    return Formula(raw)


def _m(res) -> dict:
    st = res.stats
    return {"ret": st.get("total_return"), "bench": st.get("benchmark_total_return"),
            "alpha": st.get("alpha_vs_benchmark"), "sharpe": st.get("sharpe"),
            "dd": st.get("max_drawdown"), "trd": st.get("n_trades")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--fetch-start", default="2016-05-01")
    ap.add_argument("--out", default="runs/diag_regime_sweep.json")
    args = ap.parse_args()
    os.environ.setdefault("USE_VECTORIZED_SCORING", "1")

    tickers = get_universe_tickers(args.universe)
    for extra in ("SPY", "SHY"):
        if extra not in tickers:
            tickers.append(extra)
    print(f"[{args.universe}] fetching {len(tickers)} tickers", flush=True)
    global _PANEL
    _PANEL = get_universe(tickers, start=args.fetch_start, end=args.end, provider="yf")
    print(f"  fetched {len(_PANEL)} non-empty", flush=True)

    results: dict = {c: {} for c in REGIMES}
    for cname, reg in REGIMES.items():
        f = _formula(reg)
        for pname, start, pend in PERIODS:
            cfg = bt.config_from_formula(f, **RIDE)
            res = bt.run(_PANEL, f, start=start, end=pend or args.end, cfg=cfg)
            results[cname][pname] = _m(res)
            print(f"    {cname}/{pname}: OK", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}", flush=True)
    for pname, _, _ in PERIODS:
        spy = results["none"][pname]["bench"]
        print(f"\n## {pname}  (SPY {spy:+.3f})")
        for c in REGIMES:
            x = results[c][pname]
            print(f"  {c:10s} ret={x['ret']:+.3f} alpha={x['alpha']:+.3f} "
                  f"sharpe={x['sharpe']:.2f} DD={x['dd']:.3f} trd={x['trd']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
