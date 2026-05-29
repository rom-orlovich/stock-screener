#!/usr/bin/env python3
"""Bit-for-bit parity check between score_ticker (old) and score_ticker_at (new).

Iterates a small basket of tickers across every Friday in 2023-01-01 → end and
compares the final score. Fails loudly on any mismatch > EPS so we never ship
a "looks faster" path that silently changes ranks.

Run from repo root:
    python scripts/parity_test_vectorized.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.data import get_universe  # noqa: E402
from engine.score import (  # noqa: E402
    Formula,
    precompute_indicators,
    score_ticker,
    score_ticker_at,
)

EPS = 1e-6
TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "AMD"]
FETCH_START = "2021-01-01"   # warmup for 252-bar momentum + monthly bars
FETCH_END = "2026-05-28"
PARITY_START = "2023-01-01"
FORMULAS = [
    ROOT / "formulas" / "dual_momentum_v1.yaml",
    ROOT / "formulas" / "momentum_v1.yaml",
]


def compare_dicts(old: dict, new: dict, ctx: str) -> list[str]:
    """Compare every numeric field; return list of mismatch descriptions."""
    msgs = []
    for k in old:
        ov = old.get(k)
        nv = new.get(k)
        if isinstance(ov, dict):
            msgs.extend(compare_dicts(ov, nv or {}, f"{ctx}.{k}"))
            continue
        if isinstance(ov, bool) or isinstance(nv, bool):
            if bool(ov) != bool(nv):
                msgs.append(f"{ctx}.{k}: old={ov} new={nv}")
            continue
        try:
            ofv = float(ov)
            nfv = float(nv)
        except (TypeError, ValueError):
            if ov != nv:
                msgs.append(f"{ctx}.{k}: old={ov!r} new={nv!r}")
            continue
        if pd.isna(ofv) and pd.isna(nfv):
            continue
        if abs(ofv - nfv) > EPS:
            msgs.append(f"{ctx}.{k}: old={ofv:.8f} new={nfv:.8f} diff={ofv - nfv:.2e}")
    return msgs


def run_parity(price_data: dict[str, pd.DataFrame], formula_path: Path) -> tuple[int, int, list[str]]:
    f = Formula.load(formula_path)
    fridays = pd.date_range(start=PARITY_START, end=FETCH_END, freq="W-FRI")
    total = 0
    matched = 0
    mismatches: list[str] = []
    pre_cache: dict[str, dict] = {}
    for tkr, df in price_data.items():
        try:
            pre_cache[tkr] = precompute_indicators(df, f)
        except Exception as e:  # noqa: BLE001
            mismatches.append(f"{tkr}: precompute failed: {e}")
            continue
    for d0 in fridays:
        for tkr, df in price_data.items():
            hist = df.loc[:d0]
            if len(hist) < 60:
                continue
            pre = pre_cache.get(tkr)
            if pre is None:
                continue
            try:
                old = score_ticker(hist, f)
            except Exception as e:  # noqa: BLE001
                mismatches.append(f"{tkr}@{d0.date()} old raised: {e}")
                continue
            try:
                new = score_ticker_at(pre, d0, f)
            except Exception as e:  # noqa: BLE001
                mismatches.append(f"{tkr}@{d0.date()} new raised: {e}")
                continue
            total += 1
            diffs = compare_dicts(old, new, f"{tkr}@{d0.date()}")
            if diffs:
                mismatches.extend(diffs[:6])
            else:
                matched += 1
    return matched, total, mismatches


def main() -> int:
    print(f"Fetching {len(TICKERS)} tickers {FETCH_START} → {FETCH_END} ...", flush=True)
    data = get_universe(TICKERS, start=FETCH_START, end=FETCH_END, provider="yf")
    print(f"  got {len(data)} tickers", flush=True)
    if not data:
        print("No data fetched — aborting.", file=sys.stderr)
        return 2

    overall_fail = False
    for fpath in FORMULAS:
        if not fpath.exists():
            print(f"  SKIP {fpath.name} (missing)", flush=True)
            continue
        print(f"\n== {fpath.name} ==", flush=True)
        matched, total, mismatches = run_parity(data, fpath)
        print(f"  {matched}/{total} samples matched within {EPS}", flush=True)
        if mismatches:
            overall_fail = True
            print(f"  FIRST {min(len(mismatches), 20)} MISMATCHES:")
            for m in mismatches[:20]:
                print(f"    {m}")
            if len(mismatches) > 20:
                print(f"    ... and {len(mismatches) - 20} more")
    if overall_fail:
        print("\nPARITY FAILED — investigate before flipping the flag.", flush=True)
        return 1
    print("\nPARITY PASSED — vectorized path is bit-identical (within tolerance).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
