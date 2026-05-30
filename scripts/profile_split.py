#!/usr/bin/env python3
"""Where does a vectorized backtest actually spend time? precompute vs per-d0 scoring.

Decides whether the indicator bank (cuts precompute) and cross-section (cuts the
per-d0 daily loop) are worth it, or whether the weekly/monthly per-d0 recompute
dominates and bounds the achievable speedup. Prints a breakdown for one
(formula, window) over N tickers.

    python scripts/profile_split.py --limit 60 --formula momentum_v1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

os.environ["USE_VECTORIZED_SCORING"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.score import Formula, precompute_indicators, score_ticker, score_ticker_at  # noqa: E402
from engine.bank import build_bank, collect_specs, assemble_precompute  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.universe import sp500  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--formula", default="momentum_v1")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-05-28")
    args = ap.parse_args()

    tickers = sp500()[: args.limit] + ["SPY"]
    data = get_universe(tickers, start="2017-05-01", end=args.end, provider="yf")
    f = Formula(raw=yaml.safe_load((ROOT / "formulas" / f"{args.formula}.yaml").read_text()))
    dates = pd.date_range(args.start, args.end, freq="W-FRI")
    print(f"{len(data)} tickers x {len(dates)} rebalance dates, formula={args.formula}", flush=True)

    # 1) precompute (per-formula, legacy)
    t = time.perf_counter()
    pre = {tk: precompute_indicators(df, f) for tk, df in data.items()}
    t_pre = time.perf_counter() - t

    # 2) precompute via shared bank (build once + assemble)
    t = time.perf_counter()
    specs = collect_specs([f])
    bank = {tk: build_bank(df, specs) for tk, df in data.items()}
    t_bank_build = time.perf_counter() - t
    t = time.perf_counter()
    pre_bank = {tk: assemble_precompute(bank[tk], f) for tk in data}
    t_assemble = time.perf_counter() - t

    # 3) per-d0 scoring loop (vectorized path = score_ticker_at)
    t = time.perf_counter()
    n_scores = 0
    for d0 in dates:
        for tk in data:
            try:
                score_ticker_at(pre[tk], d0, f)
                n_scores += 1
            except Exception:  # noqa: BLE001
                pass
    t_score_vec = time.perf_counter() - t

    # 4) per-d0 scoring loop (legacy scalar = score_ticker on slice) — small sample
    sample_dates = dates[:: max(1, len(dates) // 6)]
    t = time.perf_counter()
    for d0 in sample_dates:
        for tk in data:
            try:
                score_ticker(data[tk].loc[:d0], f)
            except Exception:  # noqa: BLE001
                pass
    t_score_scalar_sample = time.perf_counter() - t
    t_score_scalar_est = t_score_scalar_sample * len(dates) / max(1, len(sample_dates))

    print("\n=== TIME BREAKDOWN (one formula, one window) ===")
    print(f"  precompute (legacy, per-formula) : {t_pre:8.2f}s")
    print(f"  bank build (shared)              : {t_bank_build:8.2f}s  (amortized across ALL formulas)")
    print(f"  assemble from bank               : {t_assemble:8.2f}s")
    print(f"  per-d0 scoring (vectorized)      : {t_score_vec:8.2f}s   <-- {n_scores} score calls")
    print(f"  per-d0 scoring (scalar, est.)    : {t_score_scalar_est:8.2f}s")
    total_vec = t_pre + t_score_vec
    print(f"\n  vectorized backtest total ~= precompute + scoring = {total_vec:.1f}s")
    print(f"  precompute share of total        : {100*t_pre/total_vec:5.1f}%   <-- bank can cut at most this")
    print(f"  scoring share of total           : {100*t_score_vec/total_vec:5.1f}%   <-- daily x-section cuts only PART of this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
