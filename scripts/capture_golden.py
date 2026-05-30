#!/usr/bin/env python3
"""Golden-master capture for the regime matrix — the no-regression oracle.

Runs every (formula, regime) backtest on a fixed, fully-cached window and records
a deterministic fingerprint of each cell: the full stats dict + a sha256 digest of
the trades table and the equity curve. A later run (with perf flags ON) is compared
against this fingerprint by scripts/diff_golden.py; any byte-level difference in
picks, trades, or stats is a regression.

Design choices that make it reproducible:
  * --end pins the window (default 2026-05-28, which is present in the warm cache),
    so no network fetch and no day-to-day drift from the dynamic 'yesterday' end.
  * Reuses run_regimes.cfg_for and run_regimes.load_regimes verbatim so the config
    matches production exactly (top_n=20, W-FRI, atr stop, trailing, time stop).
  * --limit subsets the universe for fast iteration; the FULL run (--limit 0) is the
    final acceptance gate.

Perf flags are read from the environment (USE_VECTORIZED_SCORING, USE_SHARED_BANK,
USE_CROSS_SECTION, USE_NUMBA_EXITS) so the same script captures both the baseline
(all OFF) and the candidate (flags ON).

Usage:
    python scripts/capture_golden.py --limit 60 --out runs/_golden/baseline.json
    USE_VECTORIZED_SCORING=1 python scripts/capture_golden.py --limit 60 --out runs/_golden/cand.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.universe import sp500  # noqa: E402
import run_regimes as rr  # noqa: E402

FORMULAS_DIR = ROOT / "formulas"


def _digest_trades(tr: pd.DataFrame) -> dict:
    """Deterministic fingerprint of a trades table. Order-independent: sorted by the
    natural key before hashing so worker/iteration order can never change the digest.
    """
    if tr is None or tr.empty:
        return {"n": 0, "sha": "EMPTY"}
    cols = [c for c in ("enter", "exit_date", "ticker", "score", "ret", "exit") if c in tr.columns]
    t = tr[cols].copy()
    t["enter"] = t["enter"].astype(str)
    t["exit_date"] = t["exit_date"].astype(str)
    t = t.sort_values(["enter", "ticker", "exit"]).reset_index(drop=True)
    # Canonical CSV with fixed float formatting (scores/rets are already rounded in bt.run).
    payload = t.to_csv(index=False, float_format="%.6f")
    return {"n": int(len(t)), "sha": hashlib.sha256(payload.encode()).hexdigest()[:16]}


def _digest_equity(eq: pd.Series) -> str:
    if eq is None or len(eq) == 0:
        return "EMPTY"
    s = "|".join(f"{x:.8f}" for x in eq.to_numpy())
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=60,
                   help="Universe subset size for speed. 0 = full sp500.")
    p.add_argument("--end", default="2026-05-28",
                   help="Pin the widest window end (must be in the warm cache).")
    p.add_argument("--out", default="runs/_golden/baseline.json")
    args = p.parse_args()

    flags = {k: os.environ.get(k, "0") for k in
             ("USE_VECTORIZED_SCORING", "USE_SHARED_BANK", "USE_CROSS_SECTION", "USE_NUMBA_EXITS")}

    regimes = rr.load_regimes()
    # Clamp the dynamic 'yesterday' end down to the pinned, cached date.
    for r in regimes:
        if r["end"] > args.end:
            r["end"] = args.end
    widest_start = min(r["start"] for r in regimes)
    widest_end = max(r["end"] for r in regimes)
    fetch_start = (pd.Timestamp(widest_start) - pd.DateOffset(months=8)).strftime("%Y-%m-%d")

    tickers = sp500()
    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]
    if "SPY" not in tickers:
        tickers.append("SPY")

    formulas = sorted(fp for fp in FORMULAS_DIR.glob("*.yaml") if ".bak_" not in fp.name)

    print(f"golden: {len(tickers)} tickers x {len(formulas)} formulas x {len(regimes)} regimes"
          f"  window {fetch_start} -> {widest_end}", flush=True)
    print(f"flags: {flags}", flush=True)

    data = get_universe(tickers, start=fetch_start, end=widest_end, provider="yf")

    cells: dict[str, dict] = {}
    for fp in formulas:
        raw = yaml.safe_load(fp.read_text())
        f = Formula(raw=raw)
        cfg = rr.cfg_for(raw)
        for r in regimes:
            res = bt.run(data, f, start=r["start"], end=r["end"], cfg=cfg)
            key = f"{fp.stem}::{r['name']}"
            cells[key] = {
                "stats": res.stats,
                "trades": _digest_trades(res.trades),
                "equity": _digest_equity(res.equity),
            }
            print(f"  {key:48s} ret={res.stats.get('total_return')} "
                  f"sharpe={res.stats.get('sharpe')} n_tr={res.stats.get('n_trades')}", flush=True)

    out = {
        "meta": {"limit": args.limit, "end": args.end, "flags": flags,
                 "n_tickers": len(tickers), "n_formulas": len(formulas), "n_regimes": len(regimes)},
        "cells": cells,
    }
    out_fp = ROOT / args.out
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    print(f"\nwrote {out_fp}  ({len(cells)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
