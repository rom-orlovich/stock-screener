#!/usr/bin/env python3
"""Run every formula on the same uniform window in ONE Python process.

Speedup vs the old `for f in formulas; python run.py backtest ...` loop:

  * Single Python interpreter startup (saves ~2s × N formulas).
  * Single yfinance/cache pass — the universe dict is fetched once and shared
    across every backtest in memory. Avoids 504 × N pickle deserialisations.
  * Optional `--parallel N` runs N formulas concurrently via a process pool.
    Each worker reuses the warm pickle cache so the per-worker fetch is fast.

Output is byte-compatible with `python run.py backtest` (same filenames + same
JSON/CSV layouts) so dashboard.py picks it up without changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.universe import get_universe_tickers, liquidity_filter  # noqa: E402

RUNS = ROOT / "runs"
FORMULAS = ROOT / "formulas"


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _cfg(args) -> bt.BacktestConfig:
    return bt.BacktestConfig(
        top_n=args.top, rebalance=args.rebalance,
        stop_loss_pct=args.stop_loss, take_profit_pct=args.take_profit,
        trailing_stop_pct=args.trailing_stop,
        trailing_activate_pct=args.trailing_activate,
        atr_stop_mult=args.atr_stop_mult,
        atr_stop_period=args.atr_stop_period,
        time_stop_bars=args.time_stop_bars,
        benchmark_ticker=args.benchmark,
    )


def _write_outputs(formula_version: str, args, res: bt.BacktestResult,
                   universe_size: int):
    """Mirror what `run.py backtest` writes — same filenames so the dashboard
    aggregator finds them without code changes."""
    RUNS.mkdir(exist_ok=True)
    stamp = _stamp()
    suffix = f"_{args.universe}" if args.universe else ""
    base = f"{formula_version}{suffix}_{stamp}"
    res.trades.to_csv(RUNS / f"bt_trades_{base}.csv", index=False)
    res.equity.to_csv(RUNS / f"bt_equity_{base}.csv")
    if res.benchmark_equity is not None:
        res.benchmark_equity.to_csv(RUNS / f"bt_benchmark_{base}.csv")
    meta = {
        "formula": formula_version,
        "start": args.start,
        "end": args.end,
        "universe_label": args.universe or "tickers",
        "config": vars(_cfg(args)),
        "stats": res.stats,
        "universe": universe_size,
    }
    (RUNS / f"bt_summary_{base}.json").write_text(json.dumps(meta, indent=2))
    return base


def _run_one(fp: Path, args, data: dict) -> tuple[str, dict]:
    """Run a single formula against the already-fetched data dict.
    Returns (formula_version, stats). Used both in-process (sequential) and
    inside the process-pool workers (which re-fetch from warm cache)."""
    f = Formula.load(fp)
    cfg = _cfg(args)
    if getattr(args, "min_liquidity", 0) > 0:
        # Keep the benchmark even if it would be filtered.
        keep_bench = data.get(args.benchmark)
        data = liquidity_filter(data, min_avg_dollar_vol=args.min_liquidity)
        if keep_bench is not None:
            data[args.benchmark] = keep_bench
    res = bt.run(data, f, start=args.start, end=args.end, cfg=cfg)
    base = _write_outputs(f.version, args, res, len(data))
    print(f"  {f.version:35s} sharpe={res.stats.get('sharpe')}  return={res.stats.get('total_return')}  -> bt_summary_{base}.json",
          flush=True)
    return f.version, res.stats


def _worker(payload):
    """Process-pool entry point — receives the args + formula path, re-loads
    data from the warm pickle cache (fast), then runs one backtest."""
    fp_str, args_dict, tickers, fetch_start = payload
    args = argparse.Namespace(**args_dict)
    # Each worker re-builds the data dict from the warm cache (no network).
    data = get_universe(tickers, start=fetch_start, end=args.end, provider="yf")
    return _run_one(Path(fp_str), args, data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=None,
                   help="Default = yesterday (full closing bar).")
    p.add_argument("--universe", default="sp500")
    p.add_argument("--min-liquidity", type=float, default=0.0,
                   help="Min 60d avg $-volume to keep a ticker (0 = disabled). "
                        "Recommended for russell3000, e.g. 10000000 ($10M).")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--rebalance", default="W-FRI")
    p.add_argument("--stop-loss", type=float, default=0.0)
    p.add_argument("--take-profit", type=float, default=0.0)
    p.add_argument("--trailing-stop", type=float, default=0.06)
    p.add_argument("--trailing-activate", type=float, default=0.05)
    p.add_argument("--atr-stop-mult", type=float, default=2.0)
    p.add_argument("--atr-stop-period", type=int, default=14)
    p.add_argument("--time-stop-bars", type=int, default=15)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of formulas to run concurrently (process pool). "
                        "Default 1 = sequential. Use 0 to auto-pick (cpu_count/2).")
    p.add_argument("--only", default=None,
                   help="Optional comma-separated list of formula basenames "
                        "(no .yaml) to restrict the run to a subset.")
    args = p.parse_args()

    if args.end is None:
        from datetime import timedelta
        args.end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if args.parallel == 0:
        args.parallel = max(1, (os.cpu_count() or 2) // 2)

    formulas = sorted(p for p in FORMULAS.glob("*.yaml") if ".bak_" not in p.name)
    if args.only:
        keep = set(args.only.split(","))
        formulas = [p for p in formulas if p.stem in keep]
    if not formulas:
        print("no formulas to run")
        return

    tickers = get_universe_tickers(args.universe)
    if args.benchmark and args.benchmark not in tickers:
        tickers = list(tickers) + [args.benchmark]
    fetch_start = (pd.Timestamp(args.start) - pd.DateOffset(months=8)).strftime("%Y-%m-%d")

    print(f"=== run_uniform: {len(formulas)} formula(s) | window {args.start} -> {args.end} | "
          f"universe={args.universe} top={args.top} | parallel={args.parallel} ===",
          flush=True)

    t_start = datetime.now()
    if args.parallel == 1:
        # Single fetch, single process. Fastest single-host path when CPU-bound
        # but RAM-tight, since we share the data dict across every backtest.
        print(f"fetching {len(tickers)} tickers (one-time)...", flush=True)
        data = get_universe(tickers, start=fetch_start, end=args.end, provider="yf")
        print(f"  loaded {len(data)} tickers", flush=True)
        for fp in formulas:
            _run_one(fp, args, data)
    else:
        # Process pool. Each worker re-loads from the warm pickle cache (fast).
        # The cache is populated by either a prior run or by the dry fetch below
        # so the first worker doesn't pay the network cost.
        print(f"warming cache for {len(tickers)} tickers (one-time)...", flush=True)
        _ = get_universe(tickers, start=fetch_start, end=args.end, provider="yf")
        from multiprocessing import get_context
        ctx = get_context("spawn")
        args_dict = vars(args)
        payloads = [(str(fp), args_dict, tickers, fetch_start) for fp in formulas]
        with ctx.Pool(args.parallel) as pool:
            for _ in pool.imap_unordered(_worker, payloads):
                pass

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"=== done in {elapsed/60:.1f} min ({elapsed:.0f}s) ===", flush=True)


if __name__ == "__main__":
    main()
