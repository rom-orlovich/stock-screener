#!/usr/bin/env python3
"""Live-rank every formula's universe — one fetch, all formulas in-memory.

Writes runs/scan_<formula>_<stamp>.csv per formula (same format as
`python run.py scan` produces today) so existing tooling keeps working.

Compared to a `for f in formulas; python run.py scan ...` loop, this:
  * Spawns Python once.
  * Fetches the universe once (warm pickle cache); shared across every scan.
  * Each scan = 504 score_ticker calls (one rebalance worth) → ~5-10s/formula
    after the data is loaded. The whole 10-formula loop runs in ~1-2 min.

A scan is much cheaper than a backtest because it scores the universe ONCE
(at "today"), not 176 times (one per rebalance over the backtest window).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.data import get_universe  # noqa: E402
from engine.score import Formula, rank  # noqa: E402
from engine.universe import liquidity_filter, sp500  # noqa: E402

RUNS = ROOT / "runs"
FORMULAS = ROOT / "formulas"


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="sp500")
    p.add_argument("--min-liquidity", type=float, default=0.0,
                   help="Min avg dollar-volume to keep a ticker.")
    p.add_argument("--end", default=None,
                   help="Trailing window end. Default = today; the engine "
                        "still scores using whatever bars exist up to this.")
    p.add_argument("--years-back", type=int, default=6,
                   help="How many years of history to pull (needs ~6y so "
                        "monthly resample has enough bars for sma_slow).")
    p.add_argument("--only", default=None,
                   help="Comma-separated list of formula basenames (no .yaml).")
    args = p.parse_args()

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.DateOffset(years=args.years_back)).strftime("%Y-%m-%d")

    formulas = sorted(p for p in FORMULAS.glob("*.yaml") if ".bak_" not in p.name)
    if args.only:
        keep = set(args.only.split(","))
        formulas = [p for p in formulas if p.stem in keep]
    if not formulas:
        print("no formulas to scan")
        return

    tickers = sp500() if args.universe == "sp500" else []
    print(f"=== run_scans: {len(formulas)} formula(s) | universe={args.universe} "
          f"({len(tickers)} tickers) | window {start} -> {end} ===", flush=True)

    t0 = datetime.now()
    print(f"fetching {len(tickers)} tickers (one-time)...", flush=True)
    data = get_universe(tickers, start=start, end=end, provider="yf")
    if args.min_liquidity > 0:
        before = len(data)
        data = liquidity_filter(data, min_avg_dollar_vol=args.min_liquidity)
        print(f"  liquidity filter: {before} -> {len(data)} tickers", flush=True)
    fetch_secs = (datetime.now() - t0).total_seconds()
    print(f"  loaded {len(data)} tickers in {fetch_secs:.1f}s", flush=True)

    RUNS.mkdir(exist_ok=True)
    for fp in formulas:
        ts = datetime.now()
        f = Formula.load(fp)
        table = rank(data, f)
        stamp = _stamp()
        out = RUNS / f"scan_{f.version}_{args.universe}_{stamp}.csv"
        table.to_csv(out, index=False)
        secs = (datetime.now() - ts).total_seconds()
        top3 = ", ".join(table["ticker"].head(3).tolist()) if not table.empty else "—"
        print(f"  {f.version:35s}  ranked {len(table)} in {secs:.1f}s  "
              f"top3=[{top3}]  -> {out.name}", flush=True)

    print(f"=== done in {(datetime.now() - t0).total_seconds():.1f}s ===")


if __name__ == "__main__":
    main()
