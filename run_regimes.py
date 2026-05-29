#!/usr/bin/env python3
"""Backtest every formula across every named regime — one data fetch, many runs.

Reads:
  - regimes.json
  - formulas/*.yaml (excluding *.bak_*)
Writes:
  - runs/regime_matrix.json   (formula x regime x stats grid)
  - runs/regime_current.json  (current market regime classification)

Massively faster than calling run.py per (formula, regime): the universe is
fetched once on the widest window, then every backtest slices the in-memory
dict via bt.run(start=..., end=...).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.universe import sp500  # noqa: E402

RUNS = ROOT / "runs"
FORMULAS = ROOT / "formulas"
REGIMES_FP = ROOT / "regimes.json"


def load_regimes() -> list[dict]:
    cfg = json.loads(REGIMES_FP.read_text())
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    out = []
    for r in cfg.get("regimes", []):
        end = r.get("end") or yesterday
        out.append({**r, "end": end})
    return out


def cfg_for(formula_raw: dict) -> bt.BacktestConfig:
    """Match the BacktestConfig used in auto_tune / run.py defaults."""
    return bt.BacktestConfig(
        top_n=20, rebalance="W-FRI",
        atr_stop_mult=2.0, trailing_stop_pct=0.06,
        trailing_activate_pct=0.05, time_stop_bars=15,
        benchmark_ticker="SPY",
    )


def classify_current_regime(data: dict[str, pd.DataFrame]) -> dict:
    """Heuristic on SPY: classify the current market state from the last 90 days.

    Returns a dict with:
      - state: 'bull' | 'bear' | 'choppy' | 'crash'
      - spy_60d_return: float
      - spy_60d_vol_ann: float (annualised stdev of daily returns)
      - spy_above_ma200: bool
      - rationale: human-readable string
    """
    spy = data.get("SPY")
    if spy is None or spy.empty:
        return {"state": "unknown", "rationale": "no SPY data"}
    s = spy["close"]
    if len(s) < 200:
        return {"state": "unknown", "rationale": "<200 bars of SPY"}
    rets = s.pct_change().dropna()
    last60_ret = float(s.iloc[-1] / s.iloc[-60] - 1.0)
    last60_vol = float(rets.iloc[-60:].std() * (252 ** 0.5))
    ma200 = float(s.iloc[-200:].mean())
    above = s.iloc[-1] > ma200
    last20_ret = float(s.iloc[-1] / s.iloc[-20] - 1.0)

    if last20_ret <= -0.10:
        state = "crash"
    elif last60_ret > 0.05 and last60_vol < 0.20 and above:
        state = "bull"
    elif last60_ret < -0.05 and not above:
        state = "bear"
    else:
        state = "choppy"

    return {
        "state": state,
        "spy_60d_return": round(last60_ret, 4),
        "spy_60d_vol_ann": round(last60_vol, 4),
        "spy_20d_return": round(last20_ret, 4),
        "spy_above_ma200": bool(above),
        "ma200": round(ma200, 2),
        "spy_last": round(float(s.iloc[-1]), 2),
        "as_of": str(s.index[-1].date()),
        "rationale": (
            f"SPY 60d return {last60_ret*100:.1f}%, "
            f"vol(ann) {last60_vol*100:.1f}%, "
            f"{'above' if above else 'below'} MA200 — classified '{state}'."
        ),
    }


def _run_job(payload):
    """Worker entry point: re-loads data from the warm pickle cache, runs ONE
    (formula, regime) backtest, returns (formula_name, regime_name, stats_or_err).

    Workers inherit the parent's environment under spawn, so
    USE_VECTORIZED_SCORING propagates without extra plumbing.
    """
    fp_str, regime, tickers, fetch_start, widest_end = payload
    fp = Path(fp_str)
    raw = yaml.safe_load(fp.read_text())
    f = Formula(raw=raw)
    cfg = cfg_for(raw)
    # Warm pickle cache: this is fast (no network).
    data = get_universe(tickers, start=fetch_start, end=widest_end, provider="yf")
    try:
        res = bt.run(data, f, start=regime["start"], end=regime["end"], cfg=cfg)
        stats = res.stats
        return fp.stem, regime["name"], {
            "regime_label": regime["label"],
            "kind": regime["kind"],
            "start": regime["start"],
            "end": regime["end"],
            "total_return": stats.get("total_return"),
            "sharpe": stats.get("sharpe"),
            "max_drawdown": stats.get("max_drawdown"),
            "win_rate": stats.get("win_rate"),
            "n_trades": stats.get("n_trades"),
            "benchmark_total_return": stats.get("benchmark_total_return"),
            "alpha_vs_benchmark": stats.get("alpha_vs_benchmark"),
        }
    except Exception as exc:  # noqa: BLE001
        return fp.stem, regime["name"], {"error": str(exc)}


def _flush_matrix(matrix: dict, regimes: list[dict], current: dict) -> None:
    """Atomic incremental save so a kill mid-run never loses the partial grid."""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "regimes": regimes,
        "matrix": matrix,
        "current": current,
    }
    target = RUNS / "regime_matrix.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True))
    tmp.replace(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of (formula, regime) jobs to run concurrently "
                        "via spawn pool. 0 = auto (cpu_count // 2). 1 = sequential.")
    p.add_argument("--vectorized", action="store_true",
                   help="Set USE_VECTORIZED_SCORING=1 for this run (and inherited by workers).")
    args = p.parse_args()
    if args.vectorized:
        os.environ["USE_VECTORIZED_SCORING"] = "1"
    if args.parallel == 0:
        args.parallel = max(1, (os.cpu_count() or 2) // 2)

    RUNS.mkdir(exist_ok=True)
    regimes = load_regimes()
    if not regimes:
        print("no regimes defined in regimes.json")
        return

    # Fetch ONE wide window covering every regime + 8-month warmup.
    widest_start = min(r["start"] for r in regimes)
    widest_end = max(r["end"] for r in regimes)
    fetch_start = (pd.Timestamp(widest_start) - pd.DateOffset(months=8)).strftime("%Y-%m-%d")
    tickers = sp500()
    if "SPY" not in tickers:
        tickers.append("SPY")
    print(f"fetching {len(tickers)} tickers  window {fetch_start} -> {widest_end}", flush=True)
    data = get_universe(tickers, start=fetch_start, end=widest_end, provider="yf")

    # Classify current regime once (uses latest data).
    current = classify_current_regime(data)
    (RUNS / "regime_current.json").write_text(json.dumps(current, indent=2, default=str))
    print(f"current regime: {current.get('state')}  ({current.get('rationale')})")

    # Build matrix: formula x regime x stats
    matrix: dict[str, dict] = {}
    formulas = sorted(p for p in FORMULAS.glob("*.yaml") if ".bak_" not in p.name)
    n_jobs = len(formulas) * len(regimes)
    print(f"\nrunning {len(formulas)} formulas x {len(regimes)} regimes = {n_jobs} backtests"
          f"   parallel={args.parallel}   vectorized={os.environ.get('USE_VECTORIZED_SCORING', '0') == '1'}",
          flush=True)
    for fp in formulas:
        matrix[fp.stem] = {}

    if args.parallel == 1:
        # Sequential — preserves the original code path exactly. Useful for
        # debugging and for environments where multiprocessing.spawn is flaky.
        for fp in formulas:
            for r in regimes:
                name, rname, result = _run_job((str(fp), r, tickers, fetch_start, widest_end))
                matrix[name][rname] = result
                sh = result.get("sharpe") if "error" not in result else "ERR"
                ret = result.get("total_return") if "error" not in result else result["error"]
                print(f"  {name:35s} {rname:15s}  sharpe={sh}  return={ret}", flush=True)
                _flush_matrix(matrix, regimes, current)
    else:
        # spawn pool — each worker re-loads data from the warm pickle cache.
        # imap_unordered streams results as soon as each job finishes; we
        # update the matrix and flush to disk every result so a kill loses at
        # most one in-flight backtest.
        from multiprocessing import get_context
        jobs = [(str(fp), r, tickers, fetch_start, widest_end)
                for fp in formulas for r in regimes]
        ctx = get_context("spawn")
        done = 0
        with ctx.Pool(args.parallel) as pool:
            for name, rname, result in pool.imap_unordered(_run_job, jobs):
                matrix[name][rname] = result
                done += 1
                sh = result.get("sharpe") if "error" not in result else "ERR"
                ret = result.get("total_return") if "error" not in result else result["error"]
                print(f"  [{done:>3}/{n_jobs}] {name:35s} {rname:15s}  sharpe={sh}  return={ret}",
                      flush=True)
                _flush_matrix(matrix, regimes, current)

    _flush_matrix(matrix, regimes, current)
    print(f"\nwrote {RUNS/'regime_matrix.json'} and {RUNS/'regime_current.json'}")


if __name__ == "__main__":
    main()
