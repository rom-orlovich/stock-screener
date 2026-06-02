#!/usr/bin/env python3
"""Daily decision-support signals from the ROBUST core (VCP selection + per-position
stops). NOT an auto-trader and NOT a profit guarantee — a research/decision tool
Claude can run each day to surface candidates and risk levels.

What it prints (as of the latest closed bar):
  * MARKET regime context (SPY vs SHY 6-month momentum) — informational only; the
    regime sweep showed market-timing is fragile, so this is a CONTEXT flag, not an
    auto-flatten trigger.
  * BUY candidates — universe ranked by the VCP-weighted score, filtered to names
    whose volume-confirmed breakout fired within the last `--event-window` bars.
    For each: score, last close, and the suggested initial ATR stop (entry-2.5*ATR).
  * SELL/manage — if you pass a held book via --holdings "TKR:entry,...", prints the
    current stop/trail level and whether price is below it (exit signal).

Honest limits printed every run: survivorship-biased universe, no live fills/slippage,
past performance != future. Single-process, sp500.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import atr as atr_mod  # noqa: E402
from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula, score_ticker  # noqa: E402
from engine.universe import get_universe_tickers  # noqa: E402

FORMULA_FP = ROOT / "formulas" / "leading_stock_v1.yaml"
# Robust core: VCP weighting (selection), managed per-position stops handle risk.
VCP_WEIGHTS = {
    "momentum": 0.18, "trend": 0.15, "rsi": 0.05,
    "high52_proximity": 0.18, "breakout": 0.10, "breakout_thrust": 0.12,
    "atr_contraction": 0.12, "volume_dryup": 0.10, "bb_squeeze": 0.08,
    "gap": 0.05, "volatility": 0.00, "trend_template": 0.00,
}


def _vcp_formula() -> Formula:
    raw = copy.deepcopy(Formula.load(FORMULA_FP).raw)
    raw["timeframe_score_weights"] = dict(VCP_WEIGHTS)
    return Formula(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--end", default=None, help="as-of date (default: latest available)")
    ap.add_argument("--fetch-start", default="2024-06-01")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--event-window", type=int, default=5)
    ap.add_argument("--holdings", default="", help='e.g. "NVDA:120.5,AAPL:210"')
    args = ap.parse_args()

    f = _vcp_formula()
    cfg = bt.config_from_formula(f)
    tickers = get_universe_tickers(args.universe)
    for x in ("SPY", "SHY"):
        if x not in tickers:
            tickers.append(x)
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"fetching {len(tickers)} tickers {args.fetch_start} -> {end} ...", flush=True)
    data = get_universe(tickers, start=args.fetch_start, end=end, provider="yf")
    spy = data.get("SPY")
    d0 = spy.index[-1] if spy is not None and not spy.empty else pd.Timestamp(end)
    print(f"\n{'='*64}\nDAILY SIGNALS  (as of {pd.Timestamp(d0).date()})  universe={args.universe}\n{'='*64}")

    # Market context (informational — timing is fragile, do not auto-act on it).
    abs_cfg = {"benchmark": "SPY", "bond_proxy": "SHY", "lookback": 126, "skip_recent": 21}
    risk_on = bt._market_regime_ok(data, abs_cfg, d0)
    print(f"\nMARKET CONTEXT: SPY 6-mo momentum is {'RISK-ON' if risk_on else 'RISK-OFF'} "
          f"vs SHY (context flag only — NOT an auto-trade trigger; timing is fragile).")

    # BUY candidates: VCP rank filtered to recent volume-confirmed breakouts.
    lb = int(cfg.event_lookback) or 50
    rows = []
    for tkr, df in data.items():
        if tkr in ("SPY", "SHY") or df is None or len(df) < 260:
            continue
        if not bt._breakout_event_fired(df, d0, lb, cfg.vol_confirm_mult,
                                        cfg.event_vol_lookback, args.event_window):
            continue
        try:
            sc = score_ticker(df.loc[:d0], f)["score"]
        except Exception:  # noqa: BLE001
            continue
        px = float(df["close"].loc[:d0].iloc[-1])
        try:
            a = float(atr_mod.atr(df.loc[:d0], cfg.atr_stop_period).iloc[-1])
        except Exception:  # noqa: BLE001
            a = float("nan")
        stop = px - cfg.atr_stop_mult * a if pd.notna(a) else float("nan")
        rows.append((tkr, sc, px, stop, (px - stop) / px if pd.notna(stop) else float("nan")))
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"\nBUY CANDIDATES (breakout fired within {args.event_window} bars, top {args.top} by VCP score):")
    if not rows:
        print("  (none fired today — no fresh setups)")
    else:
        print(f"  {'ticker':8s} {'score':>6s} {'close':>9s} {'ATR-stop':>9s} {'risk%':>6s}")
        for tkr, sc, px, stop, riskpct in rows[: args.top]:
            print(f"  {tkr:8s} {sc:>6.3f} {px:>9.2f} {stop:>9.2f} {riskpct*100:>5.1f}%")

    # Manage held book.
    if args.holdings.strip():
        print("\nHELD POSITIONS — manage:")
        for item in args.holdings.split(","):
            tkr, _, ent = item.partition(":")
            tkr = tkr.strip().upper()
            df = data.get(tkr)
            if df is None or df.empty:
                print(f"  {tkr}: no data")
                continue
            px = float(df["close"].loc[:d0].iloc[-1])
            entry = float(ent) if ent else px
            try:
                a = float(atr_mod.atr(df.loc[:d0], cfg.atr_stop_period).iloc[-1])
                stop = entry - cfg.atr_stop_mult * a
            except Exception:  # noqa: BLE001
                stop = float("nan")
            ret = px / entry - 1.0
            sig = "EXIT (below stop)" if (pd.notna(stop) and px <= stop) else "hold"
            print(f"  {tkr:8s} entry={entry:.2f} now={px:.2f} ({ret*100:+.1f}%) "
                  f"stop~{stop:.2f}  -> {sig}")

    print(f"\n{'-'*64}\nLIMITS: sp500 is survivorship-biased (today's members) -> backtested\n"
          "edge is optimistic. No live fills/slippage modeled. This is decision\n"
          "support, NOT financial advice or a profit guarantee. Verify before acting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
