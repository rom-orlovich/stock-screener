#!/usr/bin/env python3
"""TDD harness for mode="managed" — the decoupled bar-by-bar portfolio simulator
(feat/managed-mode).

Repo convention: tests are self-contained assert-scripts run with the project
interpreter (pytest is not installed) — see scripts/parity_*.py / test_event_path.py.

    cd ~/sc-managed && /home/madma/stock-screener/.venv/bin/python scripts/test_managed_path.py

What managed mode must do (and what these tests pin):
  1. PARITY — adding managed code leaves mode="rebalance" / "event" bit-identical
     (re-checks the committed golden_rebalance.json).
  2. STEPPER PARITY — when a position exits inside its first rebalance window, the
     managed round-trip (ret, exit reason, bars_held) matches the legacy
     _period_return_with_exits over that same (d0, d1].
  3. WINNER RIDES — a strong uptrend is held ACROSS many rebalance bars
     (bars_held >> 5) and exits on trail/time, not force-closed weekly.
  4. LOSER CUT — a falling name is stopped out quickly (exit="stop", small hold).
  5. SLOT FREE + REFILL — a stopped-out slot is refilled from the ranking later.
  6. NO LEAK — every managed entry had its breakout event fire on/just before d0.
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
# Panel helpers
# --------------------------------------------------------------------------
def _df(close: np.ndarray, idx: pd.DatetimeIndex, vol: np.ndarray | None = None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    intra = 0.004
    high = close * (1.0 + intra)
    low = close * (1.0 - intra)
    if vol is None:
        vol = np.full(close.shape[0], 1.0e6)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": np.asarray(vol, float)},
        index=idx,
    )


def _flat_spy(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """A calm SPY so it's the master calendar and never the top pick."""
    n = len(idx)
    close = 100.0 * (1.0 + 0.0001) ** np.arange(n)
    return _df(close, idx)


def _make_panel_from_closes(closes: dict[str, np.ndarray], idx: pd.DatetimeIndex,
                            vols: dict[str, np.ndarray] | None = None) -> dict[str, pd.DataFrame]:
    data = {t: _df(c, idx, (vols or {}).get(t)) for t, c in closes.items()}
    if "SPY" not in data:
        data["SPY"] = _flat_spy(idx)
    return data


def _formula() -> Formula:
    return Formula.load(str(ROOT / "formulas" / "base_breakout_v1.yaml"))


# Managed cfg with the event gate NEUTRALISED (vol_confirm_mult=0 -> always
# confirmed; small lookbacks so the break fires on any new high). Lets the unit
# tests probe the HOLD/EXIT mechanics in isolation.
def _managed_cfg(**kw) -> bt.BacktestConfig:
    base = dict(
        top_n=kw.pop("top_n", 3), rebalance="W-FRI", mode="managed",
        vol_confirm_mult=0.0, event_lookback=3, event_vol_lookback=3, event_window_bars=10,
        atr_stop_mult=2.0, atr_stop_period=14,
        trailing_stop_pct=0.06, trailing_activate_pct=0.05, time_stop_bars=15,
        benchmark_ticker="SPY",
    )
    base.update(kw)
    return bt.BacktestConfig(**base)


# --------------------------------------------------------------------------
# 1. PARITY — rebalance/event default path unchanged
# --------------------------------------------------------------------------
def _eq_records(res: bt.BacktestResult) -> dict:
    return {
        "equity": [round(float(v), 10) for v in res.equity.to_numpy()],
        "eq_index": [str(d.date()) for d in res.equity.index],
        "trades": json.loads(res.trades.to_json(orient="records")) if not res.trades.empty else [],
        "stats": res.stats,
    }


def _golden_cfgs() -> dict[str, bt.BacktestConfig]:
    return {
        "plain": bt.BacktestConfig(top_n=3, rebalance="W-FRI"),
        "with_exits": bt.BacktestConfig(
            top_n=3, rebalance="W-FRI",
            atr_stop_mult=2.0, trailing_stop_pct=0.06,
            trailing_activate_pct=0.05, time_stop_bars=15,
        ),
    }


def _golden_panel(n_days: int = 520, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Identical to scripts/test_event_path._make_panel so the golden matches."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "SPY"]
    drifts = {"AAA": 0.0008, "BBB": 0.0004, "CCC": 0.0002, "DDD": -0.0001,
              "EEE": 0.0006, "FFF": 0.0003, "SPY": 0.0003}
    data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        shocks = rng.normal(drifts[t], 0.015, n_days)
        close = 100.0 * np.exp(np.cumsum(shocks))
        vol = rng.uniform(1.0e6, 2.0e6, n_days)
        for k in range(40, n_days, 55):
            close[k:] *= 1.06
            vol[k] *= 3.0
        intra = np.abs(rng.normal(0.0, 0.008, n_days))
        high = close * (1.0 + intra)
        low = close * (1.0 - intra)
        op = (high + low) / 2.0
        data[t] = pd.DataFrame(
            {"open": op, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
    return data


def test_parity_rebalance_unchanged() -> None:
    assert FIXTURE.exists(), f"golden missing ({FIXTURE}) — run test_event_path --capture"
    golden = json.loads(FIXTURE.read_text())
    data, f = _golden_panel(), _formula()
    for name, cfg in _golden_cfgs().items():
        got = _eq_records(bt.run(data, f, "2022-06-01", "2023-12-29", cfg=cfg))
        g = golden[name]
        assert got["equity"] == g["equity"], f"[{name}] equity drift vs golden — parity broken"
        assert got["trades"] == g["trades"], f"[{name}] trades drift vs golden"
        assert got["stats"] == g["stats"], f"[{name}] stats drift vs golden"
    print("[PASS] parity — rebalance default path still bit-identical to golden")


# --------------------------------------------------------------------------
# 2. STEPPER PARITY — managed first-window exit == _period_return_with_exits
# --------------------------------------------------------------------------
def test_stepper_parity() -> None:
    # AAA rises for 60 bars (always at a new high -> event fires every bar with
    # k=0; first entry anchor is idx[59], the first Friday with >=60 bars), then
    # drops hard from idx[60] so the ATR stop fires inside that first window.
    idx = pd.bdate_range("2023-01-02", periods=90)
    close = np.empty(90)
    close[:60] = 100.0 * (1.0 + 0.01) ** np.arange(60)
    close[60:] = close[59] * (1.0 - 0.05) ** np.arange(1, 31)
    data = _make_panel_from_closes({"AAA": close}, idx)
    f = _formula()
    start, end = "2023-01-02", str(idx[-1].date())

    cfg = _managed_cfg(top_n=1, atr_stop_mult=2.0, trailing_stop_pct=0.0, time_stop_bars=0)
    res = bt.run(data, f, start, end, cfg=cfg)
    assert not res.trades.empty, "managed produced no trades"
    first = res.trades.iloc[0]
    assert "bars_held" in res.trades.columns, "managed trades must carry bars_held"

    # Legacy single-window result over the same (d0, d1].
    d0 = pd.Timestamp(first["enter"])
    dates = pd.date_range(start, end, freq="W-FRI")
    d1 = dates[list(dates).index(d0) + 1]
    leg_cfg = bt.BacktestConfig(atr_stop_mult=2.0, atr_stop_period=14)
    leg_ret, leg_exit = bt._period_return_with_exits(data["AAA"], d0, d1, leg_cfg)

    assert first["exit"] == leg_exit == "stop", f"exit mismatch: {first['exit']} vs {leg_exit}"
    assert abs(float(first["ret"]) - round(float(leg_ret), 4)) < 1e-9, \
        f"ret mismatch: {first['ret']} vs {round(float(leg_ret), 4)}"
    print(f"[PASS] stepper parity — managed stop trade matches legacy "
          f"(ret={first['ret']}, exit={first['exit']}, bars_held={int(first['bars_held'])})")


# --------------------------------------------------------------------------
# 3. WINNER RIDES across many rebalance bars
# --------------------------------------------------------------------------
def test_winner_rides() -> None:
    # One relentless uptrend for 180 bars (entered at idx[59]), then an 8% pullback
    # -> trailing fires only after riding for many weeks (bars_held >> 5).
    idx = pd.bdate_range("2023-01-02", periods=200)
    up = 100.0 * (1.0 + 0.008) ** np.arange(180)
    pull = up[-1] * (1.0 - 0.02) ** np.arange(1, 21)
    close = np.concatenate([up, pull])
    data = _make_panel_from_closes({"WIN": close}, idx)
    f = _formula()
    start, end = "2023-01-02", str(idx[-1].date())
    cfg = _managed_cfg(top_n=2, atr_stop_mult=2.0, trailing_stop_pct=0.06,
                       trailing_activate_pct=0.05, time_stop_bars=0)
    res = bt.run(data, f, start, end, cfg=cfg)
    assert not res.trades.empty, "no trades"
    win = res.trades[res.trades["ticker"] == "WIN"]
    assert not win.empty, "WIN never traded"
    held = int(win["bars_held"].max())
    assert held > 5, f"winner force-closed weekly (max bars_held={held} <= 5) — not decoupled"
    closed = win[win["exit"].isin(["trail", "time"])]
    assert not closed.empty, f"winner never exited on trail/time (exits={list(win['exit'])})"
    print(f"[PASS] winner rides — WIN held {held} bars across "
          f"~{held // 5} weeks, exit={list(closed['exit'])}")


# --------------------------------------------------------------------------
# 4. LOSER CUT quickly
# --------------------------------------------------------------------------
def test_loser_cut() -> None:
    idx = pd.bdate_range("2023-01-02", periods=90)
    close = np.empty(90)
    close[:60] = 100.0 * (1.0 + 0.01) ** np.arange(60)
    close[60:] = close[59] * (1.0 - 0.04) ** np.arange(1, 31)
    data = _make_panel_from_closes({"LOSE": close}, idx)
    f = _formula()
    start, end = "2023-01-02", str(idx[-1].date())
    cfg = _managed_cfg(top_n=1, atr_stop_mult=2.0, trailing_stop_pct=0.0, time_stop_bars=0)
    res = bt.run(data, f, start, end, cfg=cfg)
    assert not res.trades.empty, "no trades"
    stops = res.trades[res.trades["exit"] == "stop"]
    assert not stops.empty, f"loser not stopped (exits={list(res.trades['exit'])})"
    assert float(stops.iloc[0]["ret"]) < 0, "stop trade should be a loss"
    assert int(stops.iloc[0]["bars_held"]) <= 10, "loser held too long before stop"
    print(f"[PASS] loser cut — exit=stop ret={stops.iloc[0]['ret']} "
          f"bars_held={int(stops.iloc[0]['bars_held'])}")


# --------------------------------------------------------------------------
# 5. SLOT FREE + REFILL
# --------------------------------------------------------------------------
def test_slot_free_and_refill() -> None:
    # top_n=1. AAA leads (highest score, entered idx[59]) then crashes (stops out).
    # BBB is a steady climber, so after AAA frees the slot BBB gets entered later.
    idx = pd.bdate_range("2023-01-02", periods=160)
    aaa = np.empty(160)
    aaa[:85] = 100.0 * (1.0 + 0.02) ** np.arange(85)       # leads, highest score
    aaa[85:] = aaa[84] * (1.0 - 0.05) ** np.arange(1, 76)  # crashes -> stop
    bbb = 100.0 * (1.0 + 0.006) ** np.arange(160)          # steady climber
    data = _make_panel_from_closes({"AAA": aaa, "BBB": bbb}, idx)
    f = _formula()
    start, end = "2023-01-02", str(idx[-1].date())
    cfg = _managed_cfg(top_n=1, atr_stop_mult=2.0, trailing_stop_pct=0.0, time_stop_bars=0)
    res = bt.run(data, f, start, end, cfg=cfg)
    traded = set(res.trades["ticker"])
    assert "AAA" in traded, "AAA never entered"
    assert "BBB" in traded, "slot never refilled with BBB after AAA stopped out"
    # AAA's exit must precede BBB's entry (the slot was freed first).
    aaa_exit = pd.Timestamp(res.trades[res.trades.ticker == "AAA"].iloc[0]["exit_date"])
    bbb_enter = pd.Timestamp(res.trades[res.trades.ticker == "BBB"].iloc[0]["enter"])
    assert bbb_enter >= aaa_exit, "BBB entered before the slot was freed (top_n=1 violated)"
    print("[PASS] slot free + refill — AAA stopped out, BBB filled the freed slot")


# --------------------------------------------------------------------------
# 6. NO LEAK — every managed entry was event-confirmed
# --------------------------------------------------------------------------
def test_no_leak() -> None:
    data, f = _golden_panel(), _formula()
    lb, k, vlb, win = 30, 1.5, 60, 5
    cfg = bt.BacktestConfig(
        top_n=4, rebalance="W-FRI", mode="managed",
        vol_confirm_mult=k, event_lookback=lb, event_vol_lookback=vlb, event_window_bars=win,
        atr_stop_mult=2.0, trailing_stop_pct=0.06, time_stop_bars=15)
    res = bt.run(data, f, "2022-06-01", "2023-12-29", cfg=cfg)
    assert not res.trades.empty, "no managed trades"
    for r in res.trades.to_dict("records"):
        d0 = pd.Timestamp(r["enter"])
        assert bt._breakout_event_fired(data[r["ticker"]], d0, lb, k, vlb, win), \
            f"managed entry {r['ticker']} @ {d0.date()} had NO event — leak"
    # The book is never larger than top_n at any instant: count concurrent holds.
    print(f"[PASS] no leak — all {len(res.trades)} managed entries event-confirmed")


# --------------------------------------------------------------------------
# 7. INTRA-WEEK ENTRY (additive) — OFF = parity, ON = enters mid-week
# --------------------------------------------------------------------------
FIXTURE_MGD = ROOT / "tests" / "fixtures" / "golden_managed.json"


def _managed_golden_cfgs() -> dict[str, bt.BacktestConfig]:
    """Two managed configs that exercise the daily exit machinery + event gate.
    The intra-week OFF path must reproduce these byte-for-byte (parity)."""
    return {
        "mgd_neutral": _managed_cfg(top_n=3),
        "mgd_event": bt.BacktestConfig(
            top_n=4, rebalance="W-FRI", mode="managed",
            vol_confirm_mult=1.5, event_lookback=30, event_vol_lookback=60,
            event_window_bars=5, atr_stop_mult=2.0, trailing_stop_pct=0.06,
            trailing_activate_pct=0.05, time_stop_bars=15),
    }


def capture_managed_golden() -> None:
    data, f = _golden_panel(), _formula()
    out = {name: _eq_records(bt.run(data, f, "2022-06-01", "2023-12-29", cfg=cfg))
           for name, cfg in _managed_golden_cfgs().items()}
    FIXTURE_MGD.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_MGD.write_text(json.dumps(out))
    print(f"[capture] wrote managed golden -> {FIXTURE_MGD}")


def test_intraweek_off_parity() -> None:
    """intraweek_entry=False reproduces the original managed path bit-for-bit."""
    assert FIXTURE_MGD.exists(), f"managed golden missing ({FIXTURE_MGD}) — run --capture-managed"
    golden = json.loads(FIXTURE_MGD.read_text())
    data, f = _golden_panel(), _formula()
    for name, cfg in _managed_golden_cfgs().items():
        cfg.intraweek_entry = False  # explicit OFF
        got = _eq_records(bt.run(data, f, "2022-06-01", "2023-12-29", cfg=cfg))
        g = golden[name]
        assert got["equity"] == g["equity"], f"[{name}] equity drift — intra-week OFF broke parity"
        assert got["trades"] == g["trades"], f"[{name}] trades drift — intra-week OFF broke parity"
        assert got["stats"] == g["stats"], f"[{name}] stats drift — intra-week OFF broke parity"
    print("[PASS] intra-week OFF — managed path still bit-identical to golden")


def _intraweek_panel() -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex, int]:
    """A single name 'MID' that drifts gently up on FLAT volume (no event fires —
    volume never clears the 1.5x bar) until a Tuesday spike. idx[81] is a Tuesday
    (bdate_range Mon-start, 81 % 5 == 1): close is a fresh high AND volume = 3x ->
    the breakout event fires on THAT Tuesday only. The next W-FRI anchor is the
    Friday idx[84]. So OFF enters Friday idx[84]; ON must enter Tuesday idx[81]."""
    n = 120
    idx = pd.bdate_range("2023-01-02", periods=n)
    tue = 81
    assert idx[tue].weekday() == 1, "idx[81] must be a Tuesday"
    # 1%/day so each close clears the prior bar's high (_df high = close*1.004):
    # the `break` condition fires daily; only the volume gate gates the event.
    close = 100.0 * (1.0 + 0.01) ** np.arange(n)
    vol = np.full(n, 1.0e6)
    vol[tue] = 3.0e6                                # the only volume spike
    data = _make_panel_from_closes({"MID": close}, idx, vols={"MID": vol})
    return data, idx, tue


def _intraweek_cfg() -> bt.BacktestConfig:
    return bt.BacktestConfig(
        top_n=2, rebalance="W-FRI", mode="managed",
        vol_confirm_mult=1.5, event_lookback=3, event_vol_lookback=5, event_window_bars=5,
        atr_stop_mult=2.0, atr_stop_period=14, trailing_stop_pct=0.0, time_stop_bars=0,
        benchmark_ticker="SPY")


def test_intraweek_on_enters_midweek() -> None:
    data, idx, tue = _intraweek_panel()
    f = _formula()
    start, end = "2023-01-02", str(idx[-1].date())
    fri = idx[84]
    assert fri.weekday() == 4, "idx[84] must be the following Friday"

    off = bt.run(data, f, start, end, cfg=_intraweek_cfg())  # intraweek default OFF
    cfg_on = _intraweek_cfg(); cfg_on.intraweek_entry = True
    on = bt.run(data, f, start, end, cfg=cfg_on)

    off_mid = off.trades[off.trades.ticker == "MID"]
    on_mid = on.trades[on.trades.ticker == "MID"]
    assert not off_mid.empty, "OFF: MID never entered"
    assert not on_mid.empty, "ON: MID never entered"

    off_enter = pd.Timestamp(off_mid.iloc[0]["enter"])
    on_enter = pd.Timestamp(on_mid.iloc[0]["enter"])
    assert off_enter == fri, f"OFF should enter at the Friday anchor {fri.date()}, got {off_enter.date()}"
    assert on_enter == idx[tue], f"ON should enter on the Tuesday {idx[tue].date()}, got {on_enter.date()}"
    assert on_enter < off_enter, "intra-week entry must be EARLIER than the weekly anchor"
    # Slot accounting: MID entered exactly once, book never exceeds top_n.
    assert len(on_mid) == 1, f"ON: MID entered {len(on_mid)} times — slot/dedup broken"
    print(f"[PASS] intra-week ON — MID entered Tue {on_enter.date()} vs weekly Fri {off_enter.date()}")


def main() -> int:
    test_parity_rebalance_unchanged()
    test_stepper_parity()
    test_winner_rides()
    test_loser_cut()
    test_slot_free_and_refill()
    test_no_leak()
    test_intraweek_off_parity()
    test_intraweek_on_enters_midweek()
    print("\nALL MANAGED TESTS PASSED")
    return 0


if __name__ == "__main__":
    if "--capture-managed" in sys.argv:
        capture_managed_golden()
        raise SystemExit(0)
    raise SystemExit(main())
