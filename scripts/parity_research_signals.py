#!/usr/bin/env python3
"""Tests for the two research-backed signals in leading_stock_v1.

  1. high52_proximity — George & Hwang (2004): close / 52-week-high.
  2. trend_template    — Minervini SEPA objective 8-rule screen (fraction passed).

Three independent checks, no network required (synthetic data only):

  SHAPE   — the scalar helpers compute the documented values (nearness ramp,
            8-rule fraction with NaN/partial cases) and are wired into the public
            scorer (the sub-score keys appear in timeframe_score output).

  BASELINE PARITY — the CURRENT leading_stock_v1.yaml (which carries NO
            high52_proximity / trend_template weight) run through the NEW code is
            bit-identical to a golden captured from the OLD code
            (tests/fixtures/golden_research_parity.json). Proves the signals are
            additive: absent weights -> zero contribution -> no behaviour change.

  TRI-PATH PARITY — with BOTH signals weighted ON, the score is identical across
            engine.score.score_ticker        (per-d0 reference)
            engine.score.score_ticker_at     (per-ticker vectorized)
            engine.score_vec.score_universe_at (cross-section)
            and both terms actually FIRE (non-zero), so the test is not vacuous.

Run from repo root:
    python scripts/parity_research_signals.py
"""
from __future__ import annotations

import copy
import json
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
# Cross-section accumulates in numpy + rounds the daily total to 4 decimals; the
# float paths can land one unit apart in the 4th decimal at a rounding boundary.
# This wobble is PRE-EXISTING (see parity_breakout_thrust.py) — a real bug would
# diverge by ~0.1. Aggregate parity is owned by parity_cross_section.py.
EPS_XS = 1.5e-4
LEADING = ROOT / "formulas" / "leading_stock_v1.yaml"
GOLDEN = ROOT / "tests" / "fixtures" / "golden_research_parity.json"


# --------------------------------------------------------------------------
# 1. Shape
# --------------------------------------------------------------------------

def test_shape() -> list[str]:
    fails: list[str] = []

    # high52_proximity = clip01(px / hi)
    h52 = score_mod._high52_proximity_score
    for name, px, hi, want in [
        ("at high", 100.0, 100.0, 1.0),
        ("20% below", 80.0, 100.0, 0.8),
        ("above high (clip)", 120.0, 100.0, 1.0),
        ("half", 50.0, 100.0, 0.5),
        ("nan hi", 100.0, float("nan"), 0.0),
        ("zero hi", 100.0, 0.0, 0.0),
    ]:
        got = h52(px, hi)
        if abs(got - want) > EPS:
            fails.append(f"high52[{name}]: got={got:.4f} want={want:.4f}")

    # trend_template = fraction of 8 rules passed
    tt = score_mod._trend_template_score
    cases = [
        # all 8 pass
        ("all pass", dict(px=120, ma_s=110, ma_m=105, ma_l=100, ma_l_prev=99,
                          hi52=130, lo52=80, mom=10), 8 / 8),
        # break stacking only (ma_s < ma_m) -> 7/8
        ("no stack", dict(px=120, ma_s=104, ma_m=105, ma_l=100, ma_l_prev=99,
                          hi52=130, lo52=80, mom=10), 7 / 8),
        # below all MAs, far from high, near low, negative RS -> r4,r5 only
        ("weak", dict(px=90, ma_s=110, ma_m=105, ma_l=100, ma_l_prev=99,
                      hi52=130, lo52=80, mom=-5), 2 / 8),
        # NaN MAs -> only r6 (above low), r7 (near high), r8 (RS) can pass
        ("nan MAs", dict(px=120, ma_s=float("nan"), ma_m=float("nan"),
                         ma_l=float("nan"), ma_l_prev=float("nan"),
                         hi52=130, lo52=80, mom=10), 3 / 8),
    ]
    for name, kw, want in cases:
        got = tt(**kw)
        if abs(got - want) > EPS:
            fails.append(f"trend_template[{name}]: got={got:.4f} want={want:.4f}")

    # Wired into the public scorer (keys present even at zero weight).
    f = Formula.load(LEADING)
    df = _synthetic_ticker(seed=1, n=900)
    sub = timeframe_score(df, f)
    for key in ("high52_proximity", "trend_template"):
        if key not in sub:
            fails.append(f"timeframe_score() output missing '{key}'")
    return fails


# --------------------------------------------------------------------------
# 2. Baseline parity vs golden (additive: zero weight -> no change)
# --------------------------------------------------------------------------

def test_baseline_golden() -> list[str]:
    fails: list[str] = []
    if not GOLDEN.exists():
        return [f"golden missing: {GOLDEN}"]
    golden = json.loads(GOLDEN.read_text())
    # Strip the two research weights from the LIVE YAML: with the new signals
    # removed the scorer must reproduce the OLD code byte-for-byte, whatever
    # final weights leading_stock_v1.yaml ends up carrying. This is the true
    # "additive / zero-impact" property, independent of the A/B outcome.
    raw = copy.deepcopy(Formula.load(LEADING).raw)
    for k in ("high52_proximity", "trend_template"):
        raw["timeframe_score_weights"].pop(k, None)
    f = Formula(raw)
    tickers = {f"SYN{i}": _synthetic_ticker(seed=i, n=1000) for i in range(1, 6)}
    fridays = pd.date_range("2022-01-01", tickers["SYN1"].index[-1], freq="W-FRI")
    n = 0
    for tkr, df in tickers.items():
        for d0 in fridays:
            key = f"{tkr}|{d0.date()}"
            if key not in golden:
                continue
            hist = df.loc[:d0]
            r = score_ticker(hist, f)
            g = golden[key]
            n += 1
            if abs(r["score"] - g["score"]) > EPS:
                fails.append(f"{key} score: new={r['score']} golden={g['score']}")
            for tf, gk in (("daily", "d"), ("weekly", "w"), ("monthly", "m")):
                nv = r["timeframes"][tf]["total"]
                if abs(nv - g[gk]) > EPS:
                    fails.append(f"{key} {tf}: new={nv} golden={g[gk]}")
    if n == 0:
        fails.append("no golden rows compared")
    return fails


# --------------------------------------------------------------------------
# 3. Tri-path parity with the signals ON
# --------------------------------------------------------------------------

def _synthetic_ticker(seed: int, n: int = 1000) -> pd.DataFrame:
    """Deterministic upward-drifting OHLCV walk (mirrors parity_breakout_thrust)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    steps = rng.normal(0.0006, 0.018, size=n)
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


def _formula_with_signals() -> Formula:
    """leading_stock_v1 with BOTH research signals weighted on (for parity)."""
    f = Formula.load(LEADING)
    raw = copy.deepcopy(f.raw)
    raw["timeframe_score_weights"]["high52_proximity"] = 0.15
    raw["timeframe_score_weights"]["trend_template"] = 0.15
    return Formula(raw)


def test_tripath() -> list[str]:
    fails: list[str] = []
    f = _formula_with_signals()
    tickers = {f"SYN{i}": _synthetic_ticker(seed=i, n=1000) for i in range(1, 6)}
    pre = {tkr: precompute_indicators(df, f) for tkr, df in tickers.items()}
    uni = precompute_universe(tickers, f)

    fridays = pd.date_range("2022-01-01", tickers["SYN1"].index[-1], freq="W-FRI")
    high52_fired = tt_fired = False
    n = 0
    for d0 in fridays:
        xs = score_universe_at(uni, d0, f)
        for tkr, df in tickers.items():
            hist = df.loc[:d0]
            if len(hist) < 250:
                continue
            old = score_ticker(hist, f)
            new = score_ticker_at(pre[tkr], d0, f)
            n += 1
            for tf in ("daily", "weekly", "monthly"):
                for key in ("high52_proximity", "trend_template"):
                    ov = old["timeframes"][tf].get(key)
                    nv = new["timeframes"][tf].get(key)
                    if ov is None or nv is None or abs(float(ov) - float(nv)) > EPS:
                        fails.append(f"{tkr}@{d0.date()} {tf}.{key}: d0={ov} vec={nv}")
            if old["timeframes"]["daily"].get("high52_proximity", 0.0) > 0.0:
                high52_fired = True
            if old["timeframes"]["daily"].get("trend_template", 0.0) > 0.0:
                tt_fired = True
            if abs(old["score"] - new["score"]) > EPS:
                fails.append(f"{tkr}@{d0.date()} score: d0={old['score']} vec={new['score']}")
            xv = xs.get(tkr)
            if xv is None or abs(old["score"] - float(xv)) > EPS_XS:
                fails.append(f"{tkr}@{d0.date()} score: d0={old['score']} xs={xv}")
    if n == 0:
        fails.append("no tri-path samples evaluated")
    if not high52_fired:
        fails.append("high52_proximity never fired — test vacuous")
    if not tt_fired:
        fails.append("trend_template never fired — test vacuous")
    return fails


def main() -> int:
    rc = 0
    for title, fn in [
        ("SHAPE (helpers + wiring)", test_shape),
        ("BASELINE PARITY (golden, zero-weight)", test_baseline_golden),
        ("TRI-PATH PARITY (signals on)", test_tripath),
    ]:
        print(f"== {title} ==", flush=True)
        f = fn()
        if f:
            rc = 1
            for m in f[:20]:
                print(f"  FAIL {m}")
            if len(f) > 20:
                print(f"  ... and {len(f) - 20} more")
        else:
            print("  ok")
    print("\n" + ("RESEARCH SIGNAL TESTS PASSED" if rc == 0 else "RESEARCH SIGNAL TESTS FAILED"),
          flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
