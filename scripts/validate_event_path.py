#!/usr/bin/env python3
"""Validation: does volume-confirmed EVENT entry + stops beat the legacy
rebalance path on the two breakout formulas?  (feat/event-path)

Per-d0 scoring is ~95% of backtest cost (CLAUDE.md profiler) and is IDENTICAL
across entry modes, so we score each formula ONCE over the union of all window
dates (parallel across cores, sharing the precompute via fork COW), cache the
ranked lists + a vectorized event-fired mask per ticker, then REPLAY every
(window, mode) cheaply. The replay reuses bt._period_return_with_exits /
bt._benchmark_equity / bt._stats verbatim, so equity, exits and stats match
bt.run exactly — only the (cached) scoring + pick policy is factored out.

3-way ablation per (formula, window):
  rebalance_nostop : top_n of ranked, no stops        (today's behaviour)
  rebalance_stop   : top_n of ranked + stops          (isolates the stop effect)
  event_stop       : top_n of ranked & event-fired + stops (entry selection)

Windows: full 2023-01-01->end + every regime in regimes.json (ends clamped).

Run:
    cd ~/sc-event && USE_VECTORIZED_SCORING=1 USE_CROSS_SECTION=1 \
        /home/madma/stock-screener/.venv/bin/python scripts/validate_event_path.py --end 2026-05-29
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.score_vec import precompute_universe, score_universe_at  # noqa: E402
from engine.universe import sp500  # noqa: E402

TOP_N = 20
STOPS = dict(atr_stop_mult=2.0, atr_stop_period=14, trailing_stop_pct=0.06,
             trailing_activate_pct=0.05, time_stop_bars=15)

# Fork-shared globals (published before the pool is created).
_DATA: dict | None = None
_UPRE: dict | None = None
_F: Formula | None = None


# --------------------------------------------------------------------------
# Scoring (cached once per formula) + event mask
# --------------------------------------------------------------------------
def _ranked_at(d0: pd.Timestamp) -> tuple[str, list]:
    """Mirror bt.run's cross-section ranking at d0: every ticker with >=60 bars
    of history and score >= 0, sorted desc. Returns (iso_date, [[tkr, score], ...])."""
    xs = score_universe_at(_UPRE, d0, _F)
    ranked = []
    for tkr, df in _DATA.items():
        if len(df.loc[:d0]) < 60:
            continue
        sc = xs.get(tkr, 0.0)
        if sc >= 0.0:
            ranked.append([tkr, float(sc)])
    ranked.sort(key=lambda x: x[1], reverse=True)
    return d0.isoformat(), ranked


def _event_mask(blk: dict) -> dict[str, pd.Series]:
    """Per-ticker boolean Series: did a volume-confirmed break fire within the
    trailing event_window_bars (vectorized; same definition as
    bt._breakout_event_fired)."""
    lb = int(blk.get("event_lookback") or 30)
    k = float(blk.get("vol_confirm_mult", 1.5))
    vlb = int(blk.get("event_vol_lookback", 60))
    win = int(blk.get("event_window_bars", 5))
    out: dict[str, pd.Series] = {}
    for tkr, df in _DATA.items():
        if "volume" not in df.columns:
            continue
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        vol = df["volume"]
        roll_hi = high.shift(1).rolling(lb, min_periods=lb).max()
        avg_vol = vol.rolling(vlb, min_periods=vlb).mean()
        fired = (close > roll_hi) & (vol >= k * avg_vol)
        out[tkr] = fired.rolling(win, min_periods=1).max().fillna(0).astype(bool)
    return out


# --------------------------------------------------------------------------
# Replay (cheap) — reuses bt's exit + stats functions verbatim
# --------------------------------------------------------------------------
def _replay(window: dict, ranked_by_date: dict, mask: dict,
            mode: str, with_stops: bool) -> bt.BacktestResult:
    cfg = bt.BacktestConfig(top_n=TOP_N, rebalance="W-FRI", benchmark_ticker="SPY",
                            **(STOPS if with_stops else {}))
    dates = pd.date_range(window["start"], window["end"], freq="W-FRI")
    equity = [1.0]
    eq_index = [dates[0]]
    trades = []
    cost = cfg.cost_bps / 10000.0
    for d0, d1 in zip(dates[:-1], dates[1:]):
        ranked = ranked_by_date.get(d0.isoformat(), [])
        if mode == "event":
            picks = []
            for tkr, sc in ranked:
                m = mask.get(tkr)
                if m is None:
                    continue
                seg = m.loc[:d0]
                if not seg.empty and bool(seg.iloc[-1]):
                    picks.append((tkr, sc))
                    if len(picks) >= TOP_N:
                        break
        else:
            picks = [(t, s) for t, s in ranked[:TOP_N]]
        if not picks:
            equity.append(equity[-1]); eq_index.append(d1); continue
        rets = []
        for tkr, sc in picks:
            r, ex = bt._period_return_with_exits(_DATA[tkr], d0, d1, cfg)
            if not pd.isna(r):
                rets.append(r)
                trades.append({"enter": d0, "exit_date": d1, "ticker": tkr,
                               "score": round(sc, 4), "ret": round(r, 4), "exit": ex})
        period_ret = (sum(rets) / len(rets) if rets else 0.0) - cost
        equity.append(equity[-1] * (1 + period_ret)); eq_index.append(d1)
    eq = pd.Series(equity, index=pd.DatetimeIndex(eq_index), name="equity")
    tr = pd.DataFrame(trades)
    bench = bt._benchmark_equity(_DATA, cfg.benchmark_ticker, eq.index)
    stats = bt._stats(eq, tr, cfg, bench)
    return bt.BacktestResult(equity=eq, trades=tr, stats=stats, benchmark_equity=bench)


def _asymmetry(res: bt.BacktestResult) -> dict:
    tr = res.trades
    if tr.empty or "ret" not in tr.columns:
        return {"avg_win": None, "avg_loss": None, "payoff_ratio": None,
                "expectancy": None, "n_win": 0, "n_loss": 0}
    ret = tr["ret"].to_numpy(dtype=float)
    wins, losses = ret[ret > 0], ret[ret < 0]
    aw = float(wins.mean()) if wins.size else 0.0
    al = float(losses.mean()) if losses.size else 0.0
    return {"avg_win": round(aw, 4), "avg_loss": round(al, 4),
            "payoff_ratio": round(aw / abs(al), 2) if al != 0 else None,
            "expectancy": round(float(ret.mean()), 4),
            "n_win": int(wins.size), "n_loss": int(losses.size)}


def _row(res: bt.BacktestResult) -> dict:
    s, a = res.stats, _asymmetry(res)
    return {"total_return": s.get("total_return"), "sharpe": s.get("sharpe"),
            "max_drawdown": s.get("max_drawdown"), "alpha": s.get("alpha_vs_benchmark"),
            "bench_return": s.get("benchmark_total_return"), "win_rate": s.get("win_rate"),
            "n_trades": s.get("n_trades"), **a, "exit_breakdown": s.get("exit_breakdown")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-05-29")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "runs" / "event_validation.json"))
    args = ap.parse_args()
    os.environ.setdefault("USE_VECTORIZED_SCORING", "1")

    regimes = json.loads((ROOT / "regimes.json").read_text())["regimes"]
    windows = [{"name": "full_2023_to_now", "kind": "full", "start": "2023-01-01", "end": args.end}]
    for r in regimes:
        end = min(r.get("end") or args.end, args.end)
        windows.append({"name": r["name"], "kind": r["kind"], "start": r["start"], "end": end})

    # Union of all W-FRI scan dates across windows.
    all_dates = sorted({d for w in windows
                        for d in pd.date_range(w["start"], w["end"], freq="W-FRI")})
    print(f"{len(windows)} windows, {len(all_dates)} unique scan dates", flush=True)

    t0 = time.time()
    tk = sp500()
    if "SPY" not in tk:
        tk.append("SPY")
    data = get_universe(tk, start="2017-05-01", end=args.end, provider="yf")
    print(f"fetched {len(data)} tickers in {time.time()-t0:.1f}s", flush=True)

    global _DATA, _UPRE, _F
    _DATA = data
    results: dict = {}
    for fname in ["base_breakout_v1", "pre_breakout_v1"]:
        f = Formula.load(str(ROOT / "formulas" / f"{fname}.yaml"))
        blk = f.raw.get("backtest") or {}
        _F = f
        ts = time.time()
        _UPRE = precompute_universe(data, f)
        print(f"[{fname}] precompute_universe {time.time()-ts:.1f}s", flush=True)
        mask = _event_mask(blk)
        # Score every date once, in parallel (fork: workers inherit _DATA/_UPRE/_F).
        ts = time.time()
        ranked_by_date: dict = {}
        ctx = get_context("fork")
        with ctx.Pool(args.parallel) as pool:
            done = 0
            for iso, ranked in pool.imap_unordered(_ranked_at, all_dates, chunksize=4):
                ranked_by_date[iso] = ranked
                done += 1
                if done % 50 == 0:
                    print(f"  [{fname}] scored {done}/{len(all_dates)} dates "
                          f"({time.time()-ts:.0f}s)", flush=True)
        print(f"[{fname}] scored {len(all_dates)} dates in {time.time()-ts:.1f}s", flush=True)

        results[fname] = {}
        for w in windows:
            results[fname][w["name"]] = {"kind": w["kind"], "start": w["start"],
                                         "end": w["end"], "modes": {}}
            for mode_name, mode, stops in [("rebalance_nostop", "rebalance", False),
                                           ("rebalance_stop", "rebalance", True),
                                           ("event_stop", "event", True)]:
                res = _replay(w, ranked_by_date, mask, mode, stops)
                results[fname][w["name"]]["modes"][mode_name] = _row(res)
            r = results[fname][w["name"]]["modes"]
            print(f"  {fname:18s} {w['name']:18s} "
                  f"reb_nostop[shp={r['rebalance_nostop']['sharpe']},dd={r['rebalance_nostop']['max_drawdown']},pay={r['rebalance_nostop']['payoff_ratio']}] "
                  f"event[shp={r['event_stop']['sharpe']},dd={r['event_stop']['max_drawdown']},pay={r['event_stop']['payoff_ratio']},n={r['event_stop']['n_trades']},ex={r['event_stop']['exit_breakdown']}]",
                  flush=True)
            Path(args.out).write_text(json.dumps(results, indent=2, default=str))

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}  (total {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
