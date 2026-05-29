#!/usr/bin/env python3
"""Bit-for-bit parity between the fast monthly path and the legacy per-d0 resample.

Sweeps every Friday in 2023-01-01 → 2026-05-28 on a small basket × 2 formulas,
runs score_ticker_at twice (once with USE_LEGACY_MONTHLY=1, once with the new
default), and asserts the final score plus every sub-metric match within EPS.

The 'as of d0 mid-month' bucket is the hard case — exactly where the old path
was paying ~0.5ms per call and the new path returns in microseconds.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.data import get_universe  # noqa: E402
from engine.score import (  # noqa: E402
    Formula,
    precompute_indicators,
    score_ticker_at,
)

EPS = 1e-9


def compare(a: dict, b: dict, ctx: str) -> list[str]:
    msgs = []
    for k in a:
        av = a.get(k)
        bv = b.get(k)
        if isinstance(av, dict):
            msgs.extend(compare(av, bv or {}, f"{ctx}.{k}"))
            continue
        if isinstance(av, bool) or isinstance(bv, bool):
            if bool(av) != bool(bv):
                msgs.append(f"{ctx}.{k}: legacy={av} fast={bv}")
            continue
        try:
            af = float(av)
            bf = float(bv)
        except (TypeError, ValueError):
            if av != bv:
                msgs.append(f"{ctx}.{k}: legacy={av!r} fast={bv!r}")
            continue
        if pd.isna(af) and pd.isna(bf):
            continue
        if abs(af - bf) > EPS:
            msgs.append(f"{ctx}.{k}: legacy={af:.10f} fast={bf:.10f} diff={af-bf:.2e}")
    return msgs


def run_once(data, formula, use_legacy: bool):
    if use_legacy:
        os.environ["USE_LEGACY_MONTHLY"] = "1"
    else:
        os.environ.pop("USE_LEGACY_MONTHLY", None)
    pre_cache = {tkr: precompute_indicators(df, formula) for tkr, df in data.items()}
    fridays = pd.date_range("2023-01-01", "2026-05-28", freq="W-FRI")
    results = {}
    t0 = time.time()
    for d0 in fridays:
        for tkr, df in data.items():
            if len(df.loc[:d0]) < 60:
                continue
            r = score_ticker_at(pre_cache[tkr], d0, formula)
            results[(tkr, d0)] = r
    dt = time.time() - t0
    return results, dt


def main() -> int:
    formulas = [
        ROOT / "formulas" / "dual_momentum_v1.yaml",
        ROOT / "formulas" / "momentum_v1.yaml",
        ROOT / "formulas" / "mean_reversion_v1.yaml",
    ]
    tickers = ["SPY", "AAPL", "MSFT", "NVDA", "AMD"]
    print(f"Fetching {len(tickers)} tickers ...", flush=True)
    data = get_universe(tickers, start="2021-01-01", end="2026-05-28", provider="yf")
    print(f"  loaded {len(data)}", flush=True)

    overall_fail = False
    for fp in formulas:
        if not fp.exists():
            print(f"  SKIP {fp.name} (missing)")
            continue
        print(f"\n== {fp.name} ==", flush=True)
        f = Formula.load(fp)
        legacy, t_leg = run_once(data, f, use_legacy=True)
        fast, t_fast = run_once(data, f, use_legacy=False)
        matched = mismatched = 0
        first_mismatches = []
        for key, lv in legacy.items():
            fv = fast.get(key)
            if fv is None:
                continue
            diffs = compare(lv, fv, f"{key[0]}@{key[1].date()}")
            if diffs:
                mismatched += 1
                if len(first_mismatches) < 5:
                    first_mismatches.extend(diffs[:3])
            else:
                matched += 1
        speedup = t_leg / t_fast if t_fast > 0 else 0.0
        print(f"  matched={matched}  mismatched={mismatched}")
        print(f"  legacy {t_leg:.1f}s -> fast {t_fast:.1f}s   speedup {speedup:.2f}x")
        if mismatched:
            overall_fail = True
            for m in first_mismatches:
                print(f"    {m}")

    if overall_fail:
        print("\nFAST MONTHLY PARITY FAILED")
        return 1
    print("\nFAST MONTHLY PARITY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
