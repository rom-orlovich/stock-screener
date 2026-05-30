#!/usr/bin/env python3
"""Phase 2 gate: the shared indicator bank reproduces precompute_indicators exactly.

For a basket of tickers x every formula, assert
    assemble_precompute(build_bank(daily, specs), f)  ==  precompute_indicators(daily, f)
key-for-key, series-for-series, bit-identical (NaN==NaN, same index/dtype). Any
divergence means the bank would change a backtest result — a regression.

Run from repo root:
    python scripts/parity_bank.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bank import assemble_precompute, build_bank, collect_specs  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula, precompute_indicators  # noqa: E402

TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "AMD"]
FETCH_START = "2017-05-01"
FETCH_END = "2026-05-28"
FORMULAS = sorted(fp for fp in (ROOT / "formulas").glob("*.yaml") if ".bak_" not in fp.name)


def _eq(a, b) -> tuple[bool, str]:
    """Bit-identical equality for Series/DataFrame (NaN==NaN). Returns (ok, detail)."""
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        if a.equals(b):
            return True, ""
        return False, "DataFrame differs"
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        if a.equals(b):
            return True, ""
        # numeric diagnostic
        try:
            d = (a.astype(float) - b.astype(float)).abs().max()
        except Exception:  # noqa: BLE001
            d = float("nan")
        return False, f"Series differs (max abs diff={d})"
    return (a == b), ("" if a == b else f"{a!r} != {b!r}")


def _compare_tf(old_tf: dict, new_tf: dict, ctx: str) -> list[str]:
    msgs = []
    keys = set(old_tf) | set(new_tf)
    for k in keys:
        if k not in old_tf:
            msgs.append(f"{ctx}.{k}: only in bank")
            continue
        if k not in new_tf:
            msgs.append(f"{ctx}.{k}: only in legacy")
            continue
        ok, detail = _eq(old_tf[k], new_tf[k])
        if not ok:
            msgs.append(f"{ctx}.{k}: {detail}")
    return msgs


def main() -> int:
    data = get_universe(TICKERS, start=FETCH_START, end=FETCH_END, provider="yf")
    formulas = [Formula(raw=__import__("yaml").safe_load(fp.read_text())) for fp in FORMULAS]
    specs = collect_specs(formulas)
    print(f"specs in union: {len(specs)}  ({sorted(specs)})", flush=True)

    total_mismatch = 0
    for fp, f in zip(FORMULAS, formulas):
        f_mismatch = 0
        for tkr in TICKERS:
            df = data.get(tkr)
            if df is None or df.empty:
                continue
            legacy = precompute_indicators(df, f)
            bank = build_bank(df, specs)
            new = assemble_precompute(bank, f)

            msgs = []
            msgs += _compare_tf(new["daily"], legacy["daily"], f"{tkr}.daily")
            msgs += _compare_tf(new["weekly"], legacy["weekly"], f"{tkr}.weekly")
            ok, detail = _eq(new["monthly_full"], legacy["monthly_full"])
            if not ok:
                msgs.append(f"{tkr}.monthly_full: {detail}")
            mp_keys = set(new["monthly_partial"]) | set(legacy["monthly_partial"])
            for k in mp_keys:
                ok, detail = _eq(new["monthly_partial"].get(k), legacy["monthly_partial"].get(k))
                if not ok:
                    msgs.append(f"{tkr}.monthly_partial.{k}: {detail}")

            if msgs:
                f_mismatch += len(msgs)
                for m in msgs[:8]:
                    print(f"  MISMATCH {fp.stem} {m}")
        status = "OK" if f_mismatch == 0 else f"{f_mismatch} MISMATCHES"
        print(f"== {fp.stem:32s} {status}", flush=True)
        total_mismatch += f_mismatch

    print()
    if total_mismatch == 0:
        print("BANK PARITY PASSED — assemble_precompute is bit-identical to precompute_indicators.")
        return 0
    print(f"BANK PARITY FAILED — {total_mismatch} mismatches.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
