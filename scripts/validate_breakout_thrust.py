#!/usr/bin/env python3
"""Before/after validation for the breakout_thrust + trend-gate change.

Fetches sp500 ONCE, then runs both breakout formulas in OLD (main weights) vs
NEW (this branch) form across the full window AND every named regime, reporting
sharpe, max-drawdown, alpha-vs-SPY and corr(score, ret). Per decision Q3 the gate
is sharpe + max-DD across regimes, not total return vs SPY.

OLD yamls are read from /tmp/OLD_<formula>.yaml (written via `git show main:...`).
Config matches run_regimes.cfg_for (top_n=20, atr_stop=2.0, trail=0.06, time=15)
so OLD/NEW and full/regime are directly comparable. Vectorized scoring on.

Run in tmux (CLAUDE.md decision #6):
    USE_VECTORIZED_SCORING=1 python scripts/validate_breakout_thrust.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("USE_VECTORIZED_SCORING", "1")

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.universe import sp500  # noqa: E402

YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
FULL_START = "2023-01-01"
FORMULAS = {
    "base_breakout_v1": (ROOT / "formulas/base_breakout_v1.yaml", Path("/tmp/OLD_base_breakout_v1.yaml")),
    "pre_breakout_v1": (ROOT / "formulas/pre_breakout_v1.yaml", Path("/tmp/OLD_pre_breakout_v1.yaml")),
}

_SHARED_DATA: dict | None = None


def cfg_for() -> bt.BacktestConfig:
    return bt.BacktestConfig(
        top_n=20, rebalance="W-FRI",
        atr_stop_mult=2.0, trailing_stop_pct=0.06,
        trailing_activate_pct=0.05, time_stop_bars=15,
        benchmark_ticker="SPY",
    )


def _windows() -> list[tuple[str, str, str]]:
    regimes = json.loads((ROOT / "regimes.json").read_text())["regimes"]
    out = [("full_2023_now", FULL_START, YESTERDAY)]
    for r in regimes:
        out.append((r["name"], r["start"], r["end"] or YESTERDAY))
    return out


def _run_job(payload):
    name, version, yaml_path, win_name, start, end = payload
    raw = yaml.safe_load(Path(yaml_path).read_text())
    f = Formula(raw=raw)
    data = _SHARED_DATA
    try:
        res = bt.run(data, f, start=start, end=end, cfg=cfg_for())
        s = res.stats
        tr = res.trades
        corr = None
        if not tr.empty and tr["score"].std() > 0 and tr["ret"].std() > 0:
            corr = round(float(tr["score"].corr(tr["ret"])), 4)
        return {
            "formula": name, "version": version, "window": win_name,
            "sharpe": s.get("sharpe"), "max_drawdown": s.get("max_drawdown"),
            "total_return": s.get("total_return"),
            "alpha_vs_benchmark": s.get("alpha_vs_benchmark"),
            "corr_score_ret": corr, "n_trades": s.get("n_trades"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"formula": name, "version": version, "window": win_name, "error": str(exc)}


def main() -> int:
    windows = _windows()
    widest_start = min(w[1] for w in windows)
    fetch_start = (pd.Timestamp(widest_start) - pd.DateOffset(months=8)).strftime("%Y-%m-%d")
    tickers = sp500()
    if "SPY" not in tickers:
        tickers.append("SPY")
    print(f"fetching {len(tickers)} tickers  {fetch_start} -> {YESTERDAY}", flush=True)
    data = get_universe(tickers, start=fetch_start, end=YESTERDAY, provider="yf")
    print(f"  loaded {len(data)} tickers", flush=True)

    global _SHARED_DATA
    _SHARED_DATA = data

    jobs = []
    for name, (new_fp, old_fp) in FORMULAS.items():
        for version, fp in (("OLD", old_fp), ("NEW", new_fp)):
            for win_name, start, end in windows:
                jobs.append((name, version, str(fp), win_name, start, end))

    print(f"running {len(jobs)} backtests (fork pool=8, vectorized)\n", flush=True)
    results = []
    ctx = get_context("fork")
    with ctx.Pool(8) as pool:
        for r in pool.imap_unordered(_run_job, jobs):
            results.append(r)
            tag = f"{r['formula']:18s} {r['version']:3s} {r['window']:16s}"
            if "error" in r:
                print(f"  {tag}  ERROR {r['error']}", flush=True)
            else:
                print(f"  {tag}  sharpe={r['sharpe']}  maxDD={r['max_drawdown']}  "
                      f"alpha={r['alpha_vs_benchmark']}  corr={r['corr_score_ret']}", flush=True)

    out = ROOT / "runs" / "validate_breakout_thrust.json"
    out.write_text(json.dumps(results, indent=2, default=str))

    # Pretty before/after table per formula × window.
    print("\n\n================ BEFORE / AFTER ================", flush=True)
    idx = {(r["formula"], r["version"], r["window"]): r for r in results}
    for name in FORMULAS:
        print(f"\n### {name}")
        print(f"{'window':16s} | {'sharpe O→N':>14s} | {'maxDD O→N':>16s} | "
              f"{'alpha O→N':>16s} | {'corr O→N':>16s}")
        for win_name, _, _ in windows:
            o = idx.get((name, "OLD", win_name), {})
            n = idx.get((name, "NEW", win_name), {})
            def pair(k):
                return f"{o.get(k)}→{n.get(k)}"
            print(f"{win_name:16s} | {pair('sharpe'):>14s} | {pair('max_drawdown'):>16s} | "
                  f"{pair('alpha_vs_benchmark'):>16s} | {pair('corr_score_ret'):>16s}")
    print(f"\nsaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
