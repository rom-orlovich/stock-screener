#!/usr/bin/env python3
"""Reliability dial: turn "nice CAGR" into measured confidence.

Two tests on the practical config (liquid russell1000, swing managed mode):

  1. OUT-OF-SAMPLE CONSISTENCY — run each strategy on NON-OVERLAPPING 2-year
     sub-periods. A real edge is consistent; one that lives in a single lucky
     window is fragile. (Note: weights are fixed, not fit to the data, so this is
     a stability check, not a train/test split.)

  2. BOOTSTRAP / Monte-Carlo on the realized results — resample the per-trade
     returns and the monthly returns thousands of times to get:
       * 95% CI on per-trade expectancy + P(expectancy > 0) + t-stat
       * distribution of 10y CAGR and max-drawdown (5th/50th/95th pct)
       * P(CAGR > 0) and P(CAGR > SPY)
     Answers "how much can we trust it" with numbers, not adjectives.

Caveats printed: survivorship-biased universe (optimistic), and the simple
(IID) bootstrap understates tail risk (real drawdowns cluster). Single-process.
"""
from __future__ import annotations

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

FORMULA_FP = ROOT / "formulas" / "leading_stock_v1.yaml"
VCP_WEIGHTS = {
    "momentum": 0.18, "trend": 0.15, "rsi": 0.05,
    "high52_proximity": 0.18, "breakout": 0.10, "breakout_thrust": 0.12,
    "atr_contraction": 0.12, "volume_dryup": 0.10, "bb_squeeze": 0.08,
    "gap": 0.05, "volatility": 0.00, "trend_template": 0.00,
}
RIDE = {"risk_per_trade": 0.0, "time_stop_bars": 0}
SUBPERIODS = [("2016-2017", "2016-01-01", "2017-12-31"),
              ("2018-2019", "2018-01-01", "2019-12-31"),
              ("2020-2021", "2020-01-01", "2021-12-31"),
              ("2022-2023", "2022-01-01", "2023-12-31"),
              ("2024-2026", "2024-01-01", "2026-06-01")]
N_BOOT = 5000
_PANEL = None


def _vcp():
    raw = copy.deepcopy(Formula.load(FORMULA_FP).raw)
    raw["timeframe_score_weights"] = dict(VCP_WEIGHTS)
    return Formula(raw)


def _bootstrap(res, spy_cagr: float, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    out: dict = {}
    # Per-trade expectancy
    tr = res.trades
    if tr is not None and not tr.empty and "ret" in tr.columns:
        r = tr["ret"].astype(float).to_numpy()
        n = len(r)
        means = np.array([rng.choice(r, size=n, replace=True).mean() for _ in range(N_BOOT)])
        out["n_trades"] = int(n)
        out["expectancy_pct"] = round(float(r.mean()) * 100, 3)
        out["expectancy_ci95_pct"] = [round(float(np.percentile(means, 2.5)) * 100, 3),
                                      round(float(np.percentile(means, 97.5)) * 100, 3)]
        out["p_expectancy_gt0"] = round(float((means > 0).mean()), 4)
        sd = float(r.std(ddof=1))
        out["t_stat"] = round(float(r.mean() / (sd / np.sqrt(n))), 2) if sd else None
    # Monthly-return path bootstrap -> CAGR + maxDD distribution
    m = res.equity.resample("ME").last().pct_change().dropna().to_numpy()
    if len(m) > 6:
        k = len(m)
        cagrs, dds = [], []
        for _ in range(N_BOOT):
            path = rng.choice(m, size=k, replace=True)
            eq = np.cumprod(1.0 + path)
            cagrs.append(eq[-1] ** (12.0 / k) - 1.0)
            peak = np.maximum.accumulate(eq)
            dds.append(float((eq / peak - 1.0).min()))
        cagrs = np.array(cagrs); dds = np.array(dds)
        out["cagr_median_pct"] = round(float(np.median(cagrs)) * 100, 2)
        out["cagr_ci95_pct"] = [round(float(np.percentile(cagrs, 2.5)) * 100, 2),
                                round(float(np.percentile(cagrs, 97.5)) * 100, 2)]
        out["p_cagr_gt0"] = round(float((cagrs > 0).mean()), 4)
        out["p_cagr_gt_spy"] = round(float((cagrs > spy_cagr).mean()), 4)
        out["maxdd_median_pct"] = round(float(np.median(dds)) * 100, 2)
        out["maxdd_p95_worst_pct"] = round(float(np.percentile(dds, 2.5)) * 100, 2)
    return out


def main() -> int:
    os.environ.setdefault("USE_VECTORIZED_SCORING", "1")
    tickers = get_universe_tickers("russell1000")
    if "SPY" not in tickers:
        tickers.append("SPY")
    print(f"fetching {len(tickers)} tickers ...", flush=True)
    data = get_universe(tickers, start="2015-01-01", end="2026-06-01", provider="yf")
    spy = data.get("SPY")
    data = liquidity_filter(data, min_avg_dollar_vol=20_000_000)
    if spy is not None:
        data["SPY"] = spy
    print(f"  tradeable universe: {len(data)}", flush=True)
    global _PANEL
    _PANEL = data

    strategies = {"leading_high52": (Formula.load(FORMULA_FP), {}), "vcp_ride": (_vcp(), RIDE)}
    report: dict = {}
    for sname, (f, ov) in strategies.items():
        print(f"\n=== {sname} ===", flush=True)
        report[sname] = {"subperiods": {}, "bootstrap_full": {}}
        # consistency across sub-periods
        for pname, ps, pe in SUBPERIODS:
            res = bt.run(_PANEL, f, start=ps, end=pe, cfg=bt.config_from_formula(f, **ov))
            st = res.stats
            report[sname]["subperiods"][pname] = {
                "alpha": st.get("alpha_vs_benchmark"), "ret": st.get("total_return"),
                "bench": st.get("benchmark_total_return"), "sharpe": st.get("sharpe"),
                "win_rate": st.get("win_rate"), "n_trades": st.get("n_trades")}
            print(f"  {pname}: ret={st.get('total_return'):+.3f} alpha={st.get('alpha_vs_benchmark'):+.3f} "
                  f"win={st.get('win_rate')} trd={st.get('n_trades')}", flush=True)
        # full-period bootstrap
        full = bt.run(_PANEL, f, start="2016-01-01", end="2026-06-01", cfg=bt.config_from_formula(f, **ov))
        spy_tr = full.stats.get("benchmark_total_return", 0.0)
        nmo = len(full.equity.resample("ME").last().pct_change().dropna())
        spy_cagr = (1 + spy_tr) ** (12.0 / nmo) - 1.0 if nmo else 0.0
        boot = _bootstrap(full, spy_cagr)
        report[sname]["bootstrap_full"] = boot
        report[sname]["spy_cagr_pct"] = round(spy_cagr * 100, 2)
        print(f"  [bootstrap] expectancy={boot.get('expectancy_pct')}% CI95={boot.get('expectancy_ci95_pct')} "
              f"P(exp>0)={boot.get('p_expectancy_gt0')} t={boot.get('t_stat')}", flush=True)
        print(f"  [bootstrap] CAGR med={boot.get('cagr_median_pct')}% CI95={boot.get('cagr_ci95_pct')} "
              f"P(>0)={boot.get('p_cagr_gt0')} P(>SPY {report[sname]['spy_cagr_pct']}%)={boot.get('p_cagr_gt_spy')}", flush=True)
        print(f"  [bootstrap] maxDD med={boot.get('maxdd_median_pct')}% worst5%={boot.get('maxdd_p95_worst_pct')}%", flush=True)

    (ROOT / "runs" / "validate_robustness.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nwrote runs/validate_robustness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
