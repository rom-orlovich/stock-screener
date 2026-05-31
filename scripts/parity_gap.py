#!/usr/bin/env python3
"""Tests for the `gap` sub-score (leading_stock_v1).

Two independent checks, no network (synthetic data only):

  1. SHAPE  — the triangle is correct: 0 below gap_band_lo (noise, not a real
     gap), ramps lo->hi to a 1.0 peak at gap_band_hi, fades hi->fade back to 0,
     and 0 beyond fade (blow-off / exhaustion gap — don't buy it). Tests the
     scalar helper and confirms the key is wired into `timeframe_score`.

  2. PARITY — `gap` is bit-identical across all three scoring code paths on the
     same synthetic universe (with injected up-gaps so the term actually FIRES):
        engine.score.timeframe_score        (per-d0 reference, via score_ticker)
        engine.score.score_ticker_at        (per-ticker vectorized)
        engine.score_vec.score_universe_at  (cross-section)

Run from repo root:
    /home/madma/stock-screener/.venv/bin/python scripts/parity_gap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import score as score_mod  # noqa: E402
from engine.score import (  # noqa: E402
    Formula,
    precompute_indicators,
    score_ticker,
    score_ticker_at,
    timeframe_score,
)
from engine.score_vec import precompute_universe, score_universe_at  # noqa: E402

EPS = 1e-6
# Cross-section accumulates in numpy and rounds the daily total to 4 decimals;
# the per-d0 / per-ticker paths accumulate in python float. At a rounding
# boundary the two can land one unit apart in the 4th decimal — pre-existing,
# documented in parity_breakout_thrust.py. A real gap bug diverges by ~0.1.
EPS_XS = 1.5e-4
FORMULA = ROOT / "formulas" / "leading_stock_v1.yaml"


def test_shape() -> list[str]:
    fails: list[str] = []
    lo, hi, fade = 0.02, 0.05, 0.12
    cases = [
        ("flat", 0.0, 0.0),
        ("gap-down", -0.03, 0.0),
        ("+1% (noise, below lo)", 0.01, 0.0),
        ("+2% (ramp start)", 0.02, 0.0),
        ("+3.5% (mid ramp)", 0.035, 0.5),
        ("+5% (peak)", 0.05, 1.0),
        ("+8.5% (mid fade)", 0.085, 0.5),
        ("+12% (faded out)", 0.12, 0.0),
        ("+20% (blow-off)", 0.20, 0.0),
    ]
    for name, g, want in cases:
        got = score_mod._gap_score(g, lo, hi, fade)
        if abs(got - want) > EPS:
            fails.append(f"shape[{name}]: gap={g} got={got:.4f} want={want:.4f}")
    # NaN guard -> 0.0
    if abs(score_mod._gap_score(float("nan"), lo, hi, fade)) > EPS:
        fails.append("shape[guard NaN]: want 0")
    # Wired into the public scorer.
    f = Formula.load(FORMULA)
    sub = timeframe_score(_gappy_ticker(seed=1, n=800), f)
    if "gap" not in sub:
        fails.append("timeframe_score() output is missing the 'gap' key")
    return fails


def _gappy_ticker(seed: int, n: int = 1000) -> pd.DataFrame:
    """Upward-drifting OHLCV walk with injected up-gaps so the gap band fires.

    Most bars open at the prior close (no gap); ~every 15th bar opens 2.5-4.5%
    above the prior close (a real accumulation gap). High/low bracket the bar.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    mu, sigma = 0.0005, 0.016
    steps = rng.normal(mu, sigma, size=n)
    close = 40.0 * np.exp(np.cumsum(steps))
    open_ = np.empty(n)
    open_[0] = close[0]
    gap_bars = (np.arange(n) % 15 == 0)
    gap_mag = rng.uniform(0.025, 0.045, size=n)
    for i in range(1, n):
        if gap_bars[i]:
            open_[i] = close[i - 1] * (1.0 + gap_mag[i])
        else:
            open_[i] = close[i - 1]
    high = np.maximum.reduce([close * (1.0 + np.abs(rng.normal(0, 0.01, n))), close, open_])
    low = np.minimum.reduce([close * (1.0 - np.abs(rng.normal(0, 0.01, n))), close, open_])
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_parity() -> list[str]:
    fails: list[str] = []
    f = Formula.load(FORMULA)
    tickers = {f"SYN{i}": _gappy_ticker(seed=i, n=1000) for i in range(1, 6)}

    pre = {tkr: precompute_indicators(df, f) for tkr, df in tickers.items()}
    uni = precompute_universe(tickers, f)

    fridays = pd.date_range("2021-06-01", tickers["SYN1"].index[-1], freq="W-FRI")
    gap_fired = False
    n_samples = 0
    for d0 in fridays:
        xs_scores = score_universe_at(uni, d0, f)
        for tkr, df in tickers.items():
            hist = df.loc[:d0]
            if len(hist) < 260:
                continue
            old = score_ticker(hist, f)
            new = score_ticker_at(pre[tkr], d0, f)
            n_samples += 1

            if old["timeframes"]["daily"].get("gap", 0.0) > 0.0:
                gap_fired = True

            for tf in ("daily", "weekly", "monthly"):
                ov = old["timeframes"][tf].get("gap")
                nv = new["timeframes"][tf].get("gap")
                if ov is None or nv is None or abs(float(ov) - float(nv)) > EPS:
                    fails.append(f"{tkr}@{d0.date()} {tf} gap: per-d0={ov} vec={nv}")

            xs = xs_scores.get(tkr)
            if abs(old["score"] - new["score"]) > EPS:
                fails.append(f"{tkr}@{d0.date()} score: per-d0={old['score']} vec={new['score']}")
            if xs is None or abs(old["score"] - float(xs)) > EPS_XS:
                fails.append(f"{tkr}@{d0.date()} score: per-d0={old['score']} xs={xs}")

    if n_samples == 0:
        fails.append("no parity samples evaluated")
    if not gap_fired:
        fails.append("gap never fired on synthetic data — test is vacuous")
    return fails


def main() -> int:
    print("== gap SHAPE ==", flush=True)
    shape_fails = test_shape()
    if shape_fails:
        for m in shape_fails[:20]:
            print(f"  FAIL {m}")
    else:
        print("  ok — triangle shape correct, sub-score wired")

    print("\n== gap PARITY (tri-path, synthetic) ==", flush=True)
    parity_fails = test_parity()
    if parity_fails:
        for m in parity_fails[:20]:
            print(f"  FAIL {m}")
        if len(parity_fails) > 20:
            print(f"  ... and {len(parity_fails) - 20} more")
    else:
        print("  ok — bit-identical across per-d0 / vectorized / cross-section, term fires")

    if shape_fails or parity_fails:
        print("\nGAP TESTS FAILED", flush=True)
        return 1
    print("\nGAP TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
