#!/usr/bin/env python3
"""Validation: does the DECOUPLED managed hold finally create the W/L asymmetry
that the weekly event path could not?  (feat/managed-mode)

3-way ablation per (formula, window) — identical stops + event params, only the
hold MECHANISM changes via cfg.mode:
  rebalance : plain top-N ranking, weekly force-close (+ within-week stops)
  event     : event-filtered entry,  weekly force-close (+ within-week stops)   [current system]
  managed   : event-filtered entry,  DECOUPLED daily-managed hold (positions persist)

  managed - event   = the pure effect of decoupling the hold from the W-FRI grid.
  event - rebalance = the entry filter (already measured in event_RESULT).

Windows: full 2023->now, full 2018->now, and every regime in regimes.json.

Calls the REAL bt.run for every (formula, window, mode) — no replay shim — so the
managed numbers come from the exact code path shipped in engine/backtest.py.
Parallel across the 48 jobs via a fork pool sharing the panel + indicator bank.

    cd ~/sc-managed && USE_VECTORIZED_SCORING=1 \\
        /home/madma/stock-screener/.venv/bin/python scripts/validate_managed_path.py --end 2026-05-29
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
from engine.universe import sp500  # noqa: E402

TOP_N = 20
FORMULAS = ["base_breakout_v1", "pre_breakout_v1"]
MODES = ["rebalance", "event", "managed"]

# Fork-shared globals (published before the pool is created).
_DATA: dict | None = None
_BANK: dict | None = None


def _asymmetry(tr: pd.DataFrame) -> dict:
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


def _holding(tr: pd.DataFrame) -> dict:
    """Avg holding period — only meaningful for managed (carries bars_held)."""
    if tr.empty or "bars_held" not in tr.columns:
        return {"avg_hold_bars": None, "avg_hold_weeks": None,
                "hold_win_bars": None, "hold_loss_bars": None}
    bh = tr["bars_held"].to_numpy(dtype=float)
    win = tr[tr["ret"] > 0]["bars_held"]
    los = tr[tr["ret"] < 0]["bars_held"]
    return {"avg_hold_bars": round(float(bh.mean()), 1),
            "avg_hold_weeks": round(float(bh.mean()) / 5.0, 1),
            "hold_win_bars": round(float(win.mean()), 1) if len(win) else None,
            "hold_loss_bars": round(float(los.mean()), 1) if len(los) else None}


def _row(res: bt.BacktestResult) -> dict:
    s = res.stats
    return {"total_return": s.get("total_return"), "sharpe": s.get("sharpe"),
            "max_drawdown": s.get("max_drawdown"), "alpha": s.get("alpha_vs_benchmark"),
            "bench_return": s.get("benchmark_total_return"), "win_rate": s.get("win_rate"),
            "n_trades": s.get("n_trades"), **_asymmetry(res.trades), **_holding(res.trades),
            "exit_breakdown": s.get("exit_breakdown")}


def _job(payload):
    fname, win, mode = payload
    f = Formula.load(str(ROOT / "formulas" / f"{fname}.yaml"))
    # Same stops + event params from the YAML; only the hold mechanism (mode) varies.
    cfg = bt.config_from_formula(f, top_n=TOP_N, rebalance="W-FRI",
                                 benchmark_ticker="SPY", mode=mode)
    try:
        res = bt.run(_DATA, f, start=win["start"], end=win["end"], cfg=cfg, bank=_BANK)
        return fname, win["name"], mode, _row(res)
    except Exception as exc:  # noqa: BLE001
        return fname, win["name"], mode, {"error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-05-29")
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "runs" / "managed_validation.json"))
    args = ap.parse_args()
    os.environ.setdefault("USE_VECTORIZED_SCORING", "1")

    regimes = json.loads((ROOT / "regimes.json").read_text())["regimes"]
    windows = [
        {"name": "full_2023_to_now", "kind": "full", "start": "2023-01-01", "end": args.end},
        {"name": "full_2018_to_now", "kind": "full", "start": "2018-01-01", "end": args.end},
    ]
    for r in regimes:
        end = min(r.get("end") or args.end, args.end)
        windows.append({"name": r["name"], "kind": r["kind"], "start": r["start"], "end": end})

    t0 = time.time()
    tk = sp500()
    if "SPY" not in tk:
        tk.append("SPY")
    data = get_universe(tk, start="2017-05-01", end=args.end, provider="yf")
    print(f"fetched {len(data)} tickers in {time.time()-t0:.1f}s", flush=True)

    # Build the shared indicator bank once (union of both formulas' specs).
    from engine.bank import build_bank, collect_specs
    f_objs = [Formula.load(str(ROOT / "formulas" / f"{n}.yaml")) for n in FORMULAS]
    specs = collect_specs(f_objs)
    tb = time.time()
    bank = {t: build_bank(df, specs) for t, df in data.items()}
    print(f"built bank: {len(specs)} specs over {len(data)} tickers in {time.time()-tb:.1f}s", flush=True)

    global _DATA, _BANK
    _DATA, _BANK = data, bank

    jobs = [(fn, w, m) for fn in FORMULAS for w in windows for m in MODES]
    print(f"running {len(jobs)} jobs ({len(FORMULAS)}f x {len(windows)}w x {len(MODES)}m) "
          f"parallel={args.parallel}", flush=True)

    results: dict = {fn: {w["name"]: {"kind": w["kind"], "start": w["start"],
                                      "end": w["end"], "modes": {}} for w in windows}
                     for fn in FORMULAS}
    ctx = get_context("fork")
    done = 0
    with ctx.Pool(args.parallel) as pool:
        for fname, wname, mode, row in pool.imap_unordered(_job, jobs):
            results[fname][wname]["modes"][mode] = row
            done += 1
            sh = row.get("sharpe", "ERR")
            pay = row.get("payoff_ratio")
            n = row.get("n_trades")
            hw = row.get("avg_hold_weeks")
            print(f"  [{done:>2}/{len(jobs)}] {fname:18s} {wname:18s} {mode:9s} "
                  f"sharpe={sh} dd={row.get('max_drawdown')} payoff={pay} "
                  f"n={n} hold_w={hw} ex={row.get('exit_breakdown')}", flush=True)
            Path(args.out).write_text(json.dumps(results, indent=2, default=str))

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}  (total {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
