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
from engine.universe import get_universe_tickers, liquidity_filter  # noqa: E402

RUNS = ROOT / "runs"
FORMULAS = ROOT / "formulas"
REGIMES_FP = ROOT / "regimes.json"

# Parent-owned panel, shared with workers via fork copy-on-write. Set in main()
# BEFORE the pool is created so forked workers inherit it (one physical copy for
# all workers). Under spawn this stays None in the child and _run_job re-loads
# from the warm pickle cache (legacy path).
_SHARED_DATA: dict | None = None

# Parent-owned indicator bank {ticker: bank}, also fork-inherited. Built once over
# the union of all formulas' (indicator, period) specs so the formula-independent
# series (rsi/atr/bb/volume/monthly resample, ~28 distinct vs 132 recomputed) are
# computed once per ticker instead of once per formula. None = legacy per-formula
# precompute. Only used on the vectorized path.
_SHARED_BANK: dict | None = None


def load_regimes() -> list[dict]:
    cfg = json.loads(REGIMES_FP.read_text())
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    out = []
    for r in cfg.get("regimes", []):
        end = r.get("end") or yesterday
        out.append({**r, "end": end})
    return out


def cfg_for(formula_raw: dict) -> bt.BacktestConfig:
    """Regime-backtest defaults, with any formula `backtest:` block overlaid on
    top (so the breakout formulas run in event mode here too)."""
    import dataclasses
    cfg = bt.BacktestConfig(
        top_n=20, rebalance="W-FRI",
        atr_stop_mult=2.0, trailing_stop_pct=0.06,
        trailing_activate_pct=0.05, time_stop_bars=15,
        benchmark_ticker="SPY",
    )
    blk = formula_raw.get("backtest") or {}
    valid = {fld.name for fld in dataclasses.fields(bt.BacktestConfig)}
    for k, v in blk.items():
        if k in valid:
            setattr(cfg, k, v)
    return cfg


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
    """Worker entry point: runs ONE (formula, regime) backtest, returns
    (formula_name, regime_name, stats_or_err).

    Data source, in order:
      1. `_SHARED_DATA` — the parent's in-memory panel, inherited via fork COW.
         No deserialization, no extra RAM (read-only pages are shared).
      2. warm pickle cache via get_universe — the spawn fallback (fresh
         interpreter, so the global is None) and the sequential legacy path.
    Workers inherit the parent's environment under both fork and spawn, so
    USE_VECTORIZED_SCORING / USE_SHARED_BANK propagate without extra plumbing.
    """
    fp_str, regime, tickers, fetch_start, widest_end = payload
    fp = Path(fp_str)
    raw = yaml.safe_load(fp.read_text())
    f = Formula(raw=raw)
    cfg = cfg_for(raw)
    if _SHARED_DATA is not None:
        data = _SHARED_DATA  # fork-inherited panel, already liquidity-filtered
    else:
        # Warm pickle cache: this is fast (no network). The parent filtered the
        # fork panel; spawn workers reload raw, so apply the same filter here.
        data = get_universe(tickers, start=fetch_start, end=widest_end, provider="yf")
        min_liq = float(os.environ.get("MIN_LIQUIDITY", "0") or "0")
        if min_liq > 0:
            keep_spy = data.get("SPY")
            data = liquidity_filter(data, min_avg_dollar_vol=min_liq)
            if keep_spy is not None:
                data["SPY"] = keep_spy
    bank = _SHARED_BANK  # fork-inherited; None under spawn / when not requested
    if bank is None and os.environ.get("USE_SHARED_BANK", "0") == "1":
        # Spawn worker: the parent's bank wasn't inherited (fresh interpreter), so
        # build it here from the cache-hydrated panel. Still one bank per worker,
        # reused across this worker's formulas-less single job (no cross-formula
        # win under spawn, but parity is identical — the win is a fork concern).
        from engine.bank import build_bank, collect_specs
        specs = collect_specs([f])
        bank = {t: build_bank(df, specs) for t, df in data.items()}
    try:
        res = bt.run(data, f, start=regime["start"], end=regime["end"], cfg=cfg, bank=bank)
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
                   help="Number of (formula, regime) jobs to run concurrently. "
                        "0 = auto (cpu_count // 2). 1 = sequential.")
    p.add_argument("--vectorized", action="store_true",
                   help="Set USE_VECTORIZED_SCORING=1 for this run (and inherited by workers).")
    p.add_argument("--shared-bank", action="store_true",
                   help="Build the per-ticker indicator bank once in the parent and "
                        "share it across all formulas (fork-inherited). Computes each "
                        "(indicator, period) once instead of once per formula. Implies "
                        "--vectorized (the bank only feeds the vectorized path).")
    p.add_argument("--mp-context", choices=("fork", "spawn"), default="fork",
                   help="Multiprocessing start method. 'fork' (Linux default) lets "
                        "workers inherit the parent panel via COW — no per-worker "
                        "pickle reload, lower RAM, so more workers fit. 'spawn' is the "
                        "portable fallback (each worker re-loads from the cache).")
    p.add_argument("--end", default=None,
                   help="Pin the widest window end (YYYY-MM-DD). Default: each regime's "
                        "own end, dynamic ends -> yesterday. Use to reproduce a run "
                        "against a cached window.")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap the universe to the first N tickers (0 = full universe). "
                        "For fast parity/dev runs.")
    p.add_argument("--universe", default="sp500",
                   help="Named universe: sp500 (default), russell3000, russell1000.")
    p.add_argument("--min-liquidity", type=float, default=0.0,
                   help="Min 60d avg $-volume to keep a ticker (0 = disabled). "
                        "Recommended for russell3000, e.g. 10000000 ($10M).")
    args = p.parse_args()
    if args.shared_bank:
        args.vectorized = True  # the bank only feeds the vectorized scorer
        os.environ["USE_SHARED_BANK"] = "1"
    if args.vectorized:
        os.environ["USE_VECTORIZED_SCORING"] = "1"
    if args.parallel == 0:
        args.parallel = max(1, (os.cpu_count() or 2) // 2)
    if args.min_liquidity > 0:
        # Propagate to spawn workers (fork inherits the already-filtered panel).
        os.environ["MIN_LIQUIDITY"] = str(args.min_liquidity)

    RUNS.mkdir(exist_ok=True)
    regimes = load_regimes()
    if not regimes:
        print("no regimes defined in regimes.json")
        return
    # Pin: clamp any regime end past --end down to it (reproducible cached window).
    if args.end:
        for r in regimes:
            if r["end"] > args.end:
                r["end"] = args.end

    # Fetch ONE wide window covering every regime + 8-month warmup.
    widest_start = min(r["start"] for r in regimes)
    widest_end = max(r["end"] for r in regimes)
    fetch_start = (pd.Timestamp(widest_start) - pd.DateOffset(months=8)).strftime("%Y-%m-%d")
    tickers = get_universe_tickers(args.universe)
    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]
    if "SPY" not in tickers:
        tickers.append("SPY")
    print(f"fetching {len(tickers)} tickers  window {fetch_start} -> {widest_end}", flush=True)
    data = get_universe(tickers, start=fetch_start, end=widest_end, provider="yf")
    if args.min_liquidity > 0:
        before = len(data)
        keep_spy = data.get("SPY")
        data = liquidity_filter(data, min_avg_dollar_vol=args.min_liquidity)
        if keep_spy is not None:
            data["SPY"] = keep_spy
        print(f"  liquidity filter: {before} -> {len(data)} tickers "
              f"(>= ${args.min_liquidity:,.0f})", flush=True)

    # Publish the panel for fork workers to inherit (COW). Must precede pool init.
    global _SHARED_DATA
    _SHARED_DATA = data

    # Classify current regime once (uses latest data).
    current = classify_current_regime(data)
    (RUNS / "regime_current.json").write_text(json.dumps(current, indent=2, default=str))
    print(f"current regime: {current.get('state')}  ({current.get('rationale')})")

    # Build matrix: formula x regime x stats
    matrix: dict[str, dict] = {}
    formulas = sorted(p for p in FORMULAS.glob("*.yaml") if ".bak_" not in p.name)

    # Shared indicator bank: compute the union of every formula's (indicator, period)
    # ONCE per ticker here in the parent; fork workers inherit it (COW). Must precede
    # pool init. Only meaningful on the vectorized path.
    if os.environ.get("USE_SHARED_BANK", "0") == "1":
        from engine.bank import build_bank, collect_specs
        from engine.score import Formula as _F
        f_objs = [_F(raw=yaml.safe_load(fp.read_text())) for fp in formulas]
        specs = collect_specs(f_objs)
        print(f"building shared bank: {len(specs)} distinct (indicator, period) "
              f"specs over {len(data)} tickers ...", flush=True)
        global _SHARED_BANK
        _SHARED_BANK = {t: build_bank(df, specs) for t, df in data.items()}

    n_jobs = len(formulas) * len(regimes)
    mp = args.mp_context if args.parallel != 1 else "sequential"
    print(f"\nrunning {len(formulas)} formulas x {len(regimes)} regimes = {n_jobs} backtests"
          f"   parallel={args.parallel} ({mp})"
          f"   vectorized={os.environ.get('USE_VECTORIZED_SCORING', '0') == '1'}",
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
        # Process pool. Under fork (default) workers inherit _SHARED_DATA via COW —
        # no per-worker reload, so RAM stays flat and more workers fit. Under spawn
        # each worker re-loads from the warm pickle cache.
        # imap_unordered streams results as soon as each job finishes; we
        # update the matrix and flush to disk every result so a kill loses at
        # most one in-flight backtest.
        from multiprocessing import get_context
        jobs = [(str(fp), r, tickers, fetch_start, widest_end)
                for fp in formulas for r in regimes]
        ctx = get_context(args.mp_context)
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
