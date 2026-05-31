#!/usr/bin/env python3
"""TDD harness for the volume-confirmed breakout EVENT path (feat/event-path).

Repo convention: tests are self-contained assert-scripts run with the project
interpreter (pytest is not installed in the venv) — see scripts/parity_*.py.

Run from the worktree root so it imports THIS worktree's engine:
    cd ~/sc-event && /home/madma/stock-screener/.venv/bin/python scripts/test_event_path.py
    ... --capture   # (re)write the golden fixture from the CURRENT code

Tests:
  1. REGRESSION GOLDEN — default cfg (mode="rebalance") reproduces a golden
     equity+trades bit-identical. Captured on the pre-change code; if the new
     mode field ever alters the default path this fails.
  2. DETECTOR — _breakout_event_fired fires on break+volume, NOT on break-only
     or volume-only or no-break.
  3. EVENT MODE — picks are a subset of the rebalance picks and only contain
     names whose event fired.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.score import Formula  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "golden_rebalance.json"


# --------------------------------------------------------------------------
# Deterministic synthetic panel — seeded, stable across runs.
# --------------------------------------------------------------------------
def _make_panel(n_days: int = 520, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "SPY"]
    drifts = {"AAA": 0.0008, "BBB": 0.0004, "CCC": 0.0002, "DDD": -0.0001,
              "EEE": 0.0006, "FFF": 0.0003, "SPY": 0.0003}
    data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        shocks = rng.normal(drifts[t], 0.015, n_days)
        close = 100.0 * np.exp(np.cumsum(shocks))
        # Inject periodic volume-confirmed pops so events + stops both fire.
        vol = rng.uniform(1.0e6, 2.0e6, n_days)
        for k in range(40, n_days, 55):
            close[k:] *= 1.06          # step-up breakout
            vol[k] *= 3.0              # volume thrust on the break bar
        intra = np.abs(rng.normal(0.0, 0.008, n_days))
        high = close * (1.0 + intra)
        low = close * (1.0 - intra)
        op = (high + low) / 2.0
        data[t] = pd.DataFrame(
            {"open": op, "high": high, "low": low, "close": close, "volume": vol},
            index=idx,
        )
    return data


def _eq_records(res: bt.BacktestResult) -> dict:
    return {
        "equity": [round(float(v), 10) for v in res.equity.to_numpy()],
        "eq_index": [str(d.date()) for d in res.equity.index],
        "trades": json.loads(res.trades.to_json(orient="records")) if not res.trades.empty else [],
        "stats": res.stats,
    }


# Two golden cfgs: plain (no exits, pure hold) and with-exits (stop path).
def _golden_cfgs() -> dict[str, bt.BacktestConfig]:
    return {
        "plain": bt.BacktestConfig(top_n=3, rebalance="W-FRI"),
        "with_exits": bt.BacktestConfig(
            top_n=3, rebalance="W-FRI",
            atr_stop_mult=2.0, trailing_stop_pct=0.06,
            trailing_activate_pct=0.05, time_stop_bars=15,
        ),
    }


def _formula() -> Formula:
    return Formula.load(str(ROOT / "formulas" / "base_breakout_v1.yaml"))


def capture() -> None:
    data, f = _make_panel(), _formula()
    out = {name: _eq_records(bt.run(data, f, "2022-06-01", "2023-12-29", cfg=cfg))
           for name, cfg in _golden_cfgs().items()}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(out, indent=2))
    print(f"[capture] wrote golden -> {FIXTURE}")
    for name, rec in out.items():
        print(f"  {name}: n_trades={len(rec['trades'])} "
              f"exits={rec['stats'].get('exit_breakdown')} "
              f"final_eq={rec['equity'][-1]}")


def test_regression_golden() -> None:
    assert FIXTURE.exists(), f"golden missing — run with --capture first ({FIXTURE})"
    golden = json.loads(FIXTURE.read_text())
    data, f = _make_panel(), _formula()
    for name, cfg in _golden_cfgs().items():
        got = _eq_records(bt.run(data, f, "2022-06-01", "2023-12-29", cfg=cfg))
        g = golden[name]
        assert got["equity"] == g["equity"], f"[{name}] equity drift vs golden"
        assert got["eq_index"] == g["eq_index"], f"[{name}] index drift"
        assert got["trades"] == g["trades"], f"[{name}] trades drift vs golden"
        assert got["stats"] == g["stats"], f"[{name}] stats drift vs golden"
    print("[PASS] regression golden — default mode bit-identical (plain + with_exits)")


def test_detector() -> None:
    fire = bt._breakout_event_fired
    n = 80
    idx = pd.bdate_range("2023-01-02", periods=n)
    base = np.linspace(100.0, 100.0, n)          # flat -> rolling high ~100
    vol = np.full(n, 1.0e6)
    d0 = idx[-1]

    def df_from(close, volume):
        c = np.asarray(close, dtype=float)
        return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999,
                             "close": c, "volume": np.asarray(volume, float)}, index=idx)

    lookback, k, vol_lb, window = 20, 1.5, 30, 1

    # (a) break + volume thrust on the last bar -> FIRE
    c = base.copy(); c[-1] = 105.0
    v = vol.copy(); v[-1] = 3.0e6
    assert fire(df_from(c, v), d0, lookback, k, vol_lb, window) is True, "should fire on break+vol"

    # (b) break but NO volume thrust -> NO fire
    c = base.copy(); c[-1] = 105.0
    assert fire(df_from(c, vol), d0, lookback, k, vol_lb, window) is False, "no fire: break w/o volume"

    # (c) volume thrust but NO break (price below rolling high) -> NO fire
    v = vol.copy(); v[-1] = 3.0e6
    assert fire(df_from(base, v), d0, lookback, k, vol_lb, window) is False, "no fire: volume w/o break"

    # (d) break+vol two bars ago, window=1 (d0 only) -> NO fire; window=3 -> FIRE
    c = base.copy(); c[-3] = 105.0
    v = vol.copy(); v[-3] = 3.0e6
    assert fire(df_from(c, v), d0, lookback, k, vol_lb, 1) is False, "window=1 misses older break"
    assert fire(df_from(c, v), d0, lookback, k, vol_lb, 3) is True, "window=3 catches trailing break"

    # (e) insufficient history -> NO fire (no lookahead / no crash)
    short = pd.bdate_range("2023-01-02", periods=10)
    cs = np.full(10, 100.0); cs[-1] = 200.0
    vs = np.full(10, 5.0e6)
    sdf = pd.DataFrame({"open": cs, "high": cs, "low": cs, "close": cs, "volume": vs}, index=short)
    assert fire(sdf, short[-1], lookback, k, vol_lb, window) is False, "no fire on thin history"
    print("[PASS] detector — fires only on confirmed break (5 cases)")


def test_event_mode_filter() -> None:
    data, f = _make_panel(), _formula()
    reb = bt.run(data, f, "2022-06-01", "2023-12-29", cfg=bt.BacktestConfig(
        top_n=4, rebalance="W-FRI",
        atr_stop_mult=2.0, trailing_stop_pct=0.06, time_stop_bars=15))
    lb, k, vlb, win = 30, 1.5, 60, 5
    evt = bt.run(data, f, "2022-06-01", "2023-12-29", cfg=bt.BacktestConfig(
        top_n=4, rebalance="W-FRI",
        atr_stop_mult=2.0, trailing_stop_pct=0.06, time_stop_bars=15,
        mode="event", vol_confirm_mult=k, event_window_bars=win,
        event_lookback=lb, event_vol_lookback=vlb))

    # Core invariant: EVERY event-mode trade had the breakout event fire on its
    # entry bar — the filter does exactly what it claims (no lookahead, no leak).
    for r in evt.trades.to_dict("records"):
        d0 = pd.Timestamp(r["enter"])
        assert bt._breakout_event_fired(data[r["ticker"]], d0, lb, k, vlb, win), \
            f"event trade {r['ticker']} @ {d0.date()} did NOT fire an event"

    # Per scan date the event book is still capped at top_n (no extra trades/date).
    for d0, grp in evt.trades.groupby("enter"):
        assert len(grp) <= 4, f"event book exceeded top_n on {d0}"

    # The filter must be selective: on this panel not every rebalance pick fired,
    # so event mode trades strictly fewer names overall.
    assert len(evt.trades) < len(reb.trades), "event filter had no effect — check wiring"
    print(f"[PASS] event mode filter — every event trade confirmed; "
          f"rebalance={len(reb.trades)} event={len(evt.trades)} trades")


def main() -> int:
    if "--capture" in sys.argv:
        capture()
        return 0
    test_regression_golden()
    test_detector()
    test_event_mode_filter()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
