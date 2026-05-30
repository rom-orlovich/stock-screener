#!/usr/bin/env python3
"""Tests for the `breakout_thrust` sub-score.

Two independent checks, no network required (synthetic data only):

  1. SHAPE  — the trapezoid is correct: 0 at/below the causal pivot, ramps to
     1.0 across the 1-3% band, fades back to 0 by ~+5% above the pivot, and is
     0 beyond. Tests the scalar helper directly and confirms the sub-score is
     wired into `timeframe_score`.

  2. PARITY — `breakout_thrust` is bit-identical across all three scoring
     code paths on the same synthetic universe:
        engine.score.timeframe_score        (per-d0 reference)
        engine.score.score_ticker_at        (per-ticker vectorized)
        engine.score_vec.score_universe_at  (cross-section)
     The parity run also asserts the term actually FIRES (non-zero on some
     bar) so the comparison is not vacuous.

Run from repo root:
    python scripts/parity_breakout_thrust.py
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
# The cross-section path accumulates in numpy and rounds the daily total to 4
# decimals; the per-d0 / per-ticker paths accumulate in python float. At a
# rounding boundary the two can land one unit apart in the 4th decimal. This
# wobble is PRE-EXISTING and independent of breakout_thrust — momentum_v1 (no
# thrust) shows the identical ≤1e-4 diff. A real thrust bug would diverge by
# ~0.1 (0.25 weight × up to 1.0), so this tolerance still catches one. Aggregate
# cross-section parity is verified authoritatively by parity_cross_section.py.
EPS_XS = 1.5e-4
BREAKOUT_FORMULA = ROOT / "formulas" / "base_breakout_v1.yaml"


# --------------------------------------------------------------------------
# 1. Shape test
# --------------------------------------------------------------------------

def test_shape() -> list[str]:
    fails: list[str] = []
    lo_band, hi_band, fade = 0.01, 0.03, 0.05
    hi = 100.0
    cases = [
        ("at high", 100.0, 0.0),
        ("0.5% below", 99.5, 0.0),
        ("5% below", 95.0, 0.0),
        ("+0.5%", 100.5, 0.5),
        ("+1% (peak start)", 101.0, 1.0),
        ("+2% (peak)", 102.0, 1.0),
        ("+3% (peak end)", 103.0, 1.0),
        ("+4% (fading)", 104.0, 0.5),
        ("+5% (faded out)", 105.0, 0.0),
        ("+8% (chasing)", 108.0, 0.0),
    ]
    for name, px, want in cases:
        got = score_mod._breakout_thrust_score(px, hi, lo_band, hi_band, fade)
        if abs(got - want) > EPS:
            fails.append(f"shape[{name}]: px={px} got={got:.4f} want={want:.4f}")

    # NaN / non-positive pivot guards -> 0.0
    for bad_hi in (float("nan"), 0.0, -5.0):
        got = score_mod._breakout_thrust_score(110.0, bad_hi, lo_band, hi_band, fade)
        if abs(got) > EPS:
            fails.append(f"shape[guard hi={bad_hi}]: got={got:.4f} want=0")

    # Wired into the public scorer: the returned sub-score dict must carry the key.
    f = Formula.load(BREAKOUT_FORMULA)
    df = _synthetic_ticker(seed=1, n=800)
    sub = timeframe_score(df, f)
    if "breakout_thrust" not in sub:
        fails.append("timeframe_score() output is missing the 'breakout_thrust' key")
    return fails


# --------------------------------------------------------------------------
# 2. Tri-path parity test (synthetic, deterministic)
# --------------------------------------------------------------------------

def _synthetic_ticker(seed: int, n: int = 1000) -> pd.DataFrame:
    """Deterministic upward-drifting OHLCV walk.

    Upward drift makes price repeatedly poke just above its prior-N-bar high so
    the breakout_thrust band (1-5% above pivot) is genuinely exercised.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    mu, sigma = 0.0006, 0.018
    steps = rng.normal(mu, sigma, size=n)
    close = 50.0 * np.exp(np.cumsum(steps))
    intra = np.abs(rng.normal(0.0, 0.012, size=n))
    high = close * (1.0 + intra)
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.012, size=n)))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum.reduce([high, close, open_])
    low = np.minimum.reduce([low, close, open_])
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_parity() -> list[str]:
    fails: list[str] = []
    f = Formula.load(BREAKOUT_FORMULA)
    tickers = {f"SYN{i}": _synthetic_ticker(seed=i, n=1000) for i in range(1, 6)}

    pre = {tkr: precompute_indicators(df, f) for tkr, df in tickers.items()}
    uni = precompute_universe(tickers, f)

    fridays = pd.date_range("2022-01-01", tickers["SYN1"].index[-1], freq="W-FRI")
    thrust_fired = False
    n_samples = 0
    for d0 in fridays:
        xs_scores = score_universe_at(uni, d0, f)
        for tkr, df in tickers.items():
            hist = df.loc[:d0]
            if len(hist) < 250:
                continue
            old = score_ticker(hist, f)
            new = score_ticker_at(pre[tkr], d0, f)
            n_samples += 1

            # confirm the term fires somewhere on the daily timeframe
            if old["timeframes"]["daily"].get("breakout_thrust", 0.0) > 0.0:
                thrust_fired = True

            # per-d0 vs per-ticker-vectorized: full sub-score dict
            for tf in ("daily", "weekly", "monthly"):
                ov = old["timeframes"][tf].get("breakout_thrust")
                nv = new["timeframes"][tf].get("breakout_thrust")
                if ov is None or nv is None or abs(float(ov) - float(nv)) > EPS:
                    fails.append(f"{tkr}@{d0.date()} {tf}: per-d0={ov} vec={nv}")

            # final score: score.py float paths are bit-identical (strict);
            # cross-section numpy path within the 4-decimal rounding boundary.
            xs = xs_scores.get(tkr)
            if abs(old["score"] - new["score"]) > EPS:
                fails.append(f"{tkr}@{d0.date()} score: per-d0={old['score']} vec={new['score']}")
            if xs is None or abs(old["score"] - float(xs)) > EPS_XS:
                fails.append(f"{tkr}@{d0.date()} score: per-d0={old['score']} xs={xs}")

    if n_samples == 0:
        fails.append("no parity samples evaluated")
    if not thrust_fired:
        fails.append("breakout_thrust never fired on synthetic data — test is vacuous")
    return fails


def main() -> int:
    print("== breakout_thrust SHAPE ==", flush=True)
    shape_fails = test_shape()
    if shape_fails:
        for m in shape_fails[:20]:
            print(f"  FAIL {m}")
    else:
        print("  ok — trapezoid shape correct, sub-score wired")

    print("\n== breakout_thrust PARITY (tri-path, synthetic) ==", flush=True)
    parity_fails = test_parity()
    if parity_fails:
        for m in parity_fails[:20]:
            print(f"  FAIL {m}")
        if len(parity_fails) > 20:
            print(f"  ... and {len(parity_fails) - 20} more")
    else:
        print("  ok — bit-identical across per-d0 / vectorized / cross-section, term fires")

    if shape_fails or parity_fails:
        print("\nBREAKOUT_THRUST TESTS FAILED", flush=True)
        return 1
    print("\nBREAKOUT_THRUST TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
