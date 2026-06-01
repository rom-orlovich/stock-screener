"""Historical backtest: rebalance into the top-N ranked tickers, measure P&L.

Deterministic. Walks the calendar, and on each rebalance date recomputes the
formula score using ONLY data up to that date (no lookahead), holds the top-N
equal-weighted until the next rebalance (or until an exit rule fires), and
records realized returns.

Exit logic (priority order, evaluated on each daily close inside the period):
  1. hard stop (ATR-multiple if `atr_stop_mult > 0`, else flat `stop_loss_pct`)
  2. trailing stop (`trailing_stop_pct`, only armed after `trailing_activate_pct` gain)
  3. take profit (`take_profit_pct`)
  4. time stop (`time_stop_bars` daily closes since entry)
  5. fall-through close-to-close until d1 -> 'hold'
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import atr as atr_mod
from .score import Formula, precompute_indicators, score_ticker, score_ticker_at
from .score_vec import precompute_universe, score_universe_at


# ---------------------------------------------------------------------------
# Numba JIT exit loop (opt-in via USE_NUMBA_EXITS=1).
# Hot inner loop iterating ~5 daily bars per pick — 88k+ calls per backtest.
# Pure-arithmetic loop: ideal for JIT. Parity is bit-exact vs the Python loop.
# Import is lazy so the module still loads when numba isn't installed.
# ---------------------------------------------------------------------------
_EXIT_CODES = ("stop", "trail", "tp", "time", "hold")
_jit_exit_loop = None


def _load_jit():
    global _jit_exit_loop
    if _jit_exit_loop is not None:
        return _jit_exit_loop
    try:
        import numba  # noqa: F401
    except ImportError:
        return None

    @numba.njit(cache=True)
    def _impl(prices, entry, has_stop, stop_lvl, has_tp, tp_lvl,
              trail_stop_pct, trail_arm_lvl, time_stop_bars):
        n = prices.shape[0]
        if n == 0:
            return np.nan, 4
        peak = entry
        trail_armed = False
        has_trail = False
        trail_lvl = 0.0
        for i in range(n):
            px = prices[i]
            if trail_stop_pct > 0.0:
                if px > peak:
                    peak = px
                if not trail_armed and px >= trail_arm_lvl:
                    trail_armed = True
                if trail_armed:
                    trail_lvl = peak * (1.0 - trail_stop_pct)
                    has_trail = True
            if has_stop and px <= stop_lvl:
                return px / entry - 1.0, 0
            if has_trail and px <= trail_lvl:
                return px / entry - 1.0, 1
            if has_tp and px >= tp_lvl:
                return px / entry - 1.0, 2
            if time_stop_bars > 0 and (i + 1) >= time_stop_bars:
                return px / entry - 1.0, 3
        return prices[n - 1] / entry - 1.0, 4

    _jit_exit_loop = _impl
    return _impl


@dataclass
class BacktestConfig:
    top_n: int = 10
    rebalance: str = "W-FRI"     # pandas offset alias for rebalance dates
    min_score: float = 0.0       # only hold names scoring above this
    cost_bps: float = 5.0        # round-trip cost per rebalance, basis points

    # Flat stop / TP (legacy).
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0

    # New: trailing stop. Activated once price hits entry * (1 + activate_pct);
    # then trailing_stop_pct below the running peak triggers exit.
    trailing_stop_pct: float = 0.0
    trailing_activate_pct: float = 0.05

    # New: ATR-multiple hard stop. If > 0, overrides flat stop_loss_pct.
    # stop level = entry - atr_stop_mult * ATR(atr_stop_period) at d0.
    atr_stop_mult: float = 0.0
    atr_stop_period: int = 14

    # New: time stop in daily bars since entry (0 = disabled).
    time_stop_bars: int = 0

    # Managed-mode only: risk-based position sizing. 0 = equal-weight (legacy,
    # ~equity/top_n per name). >0 sizes each entry so the distance to its stop
    # (entry - stop_lvl, set by the ATR/flat stop) risks this fraction of current
    # equity; the size is still capped at the equal-weight target so the book
    # stays diversified. Rebalance/event paths ignore this (parity is sacred).
    risk_per_trade: float = 0.0

    benchmark_ticker: str = "SPY"  # buy-and-hold ref over the same window.

    # Event-entry mode. "rebalance" (default) is the legacy path and MUST stay
    # bit-identical. "event" filters each weekly scan's ranked names down to those
    # with a volume-confirmed breakout on (or just before) d0 — see
    # `_breakout_event_fired`. Cadence stays the existing W-FRI rebalance bar
    # (decision Q1: no daily scan — preserves n_per_year=52, per-rebalance cost,
    # and the weekly-resample parity). Pair with stops for the asymmetric R:R.
    mode: str = "rebalance"
    vol_confirm_mult: float = 1.5        # today's volume >= k * avg_vol_long (Q2: fixed)
    event_lookback: int = 0              # prior-N-bar high; 0 -> formula breakout_lookback
    event_vol_lookback: int = 60         # avg-volume window for the confirmation
    event_window_bars: int = 1           # fired on any of the trailing N bars <= d0

    # Managed-mode ADDITIVE intra-week entry. False (default) = entries fire ONLY
    # at the weekly W-FRI rebalance anchors -> behavior bit-identical to the
    # original managed path (parity preserved). True = ALSO scan every daily bar:
    # when a slot is free and the breakout event fires on THAT day's bar, enter
    # that day instead of waiting for Friday. Weekly equity sampling and the
    # weekly re-rank are unchanged; cost is charged per-fill as before. Strict
    # superset of the weekly behavior — never removes a weekly entry, only adds
    # earlier ones. Ignored outside managed mode.
    intraweek_entry: bool = False


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict = field(default_factory=dict)
    benchmark_equity: pd.Series | None = None


def _period_return_with_exits(df: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp,
                              cfg: BacktestConfig) -> tuple[float, str]:
    """Close-to-close return with full exit logic. Returns (return, exit_reason).

    exit_reason ∈ {'stop', 'trail', 'tp', 'time', 'hold'}.
    """
    s = df["close"]
    p0 = s.loc[:d0]
    if p0.empty:
        return np.nan, "hold"
    entry = float(p0.iloc[-1])
    if entry <= 0:
        return np.nan, "hold"

    # Resolve hard stop level: ATR-based if requested, else flat.
    stop_lvl: float | None = None
    if cfg.atr_stop_mult > 0.0:
        try:
            hist = df.loc[:d0]
            atr_v = atr_mod.atr(hist, cfg.atr_stop_period).iloc[-1]
            if pd.notna(atr_v) and atr_v > 0:
                stop_lvl = entry - float(cfg.atr_stop_mult) * float(atr_v)
        except Exception:  # noqa: BLE001
            stop_lvl = None
    if stop_lvl is None and cfg.stop_loss_pct > 0.0:
        stop_lvl = entry * (1.0 - cfg.stop_loss_pct)

    tp_lvl = entry * (1.0 + cfg.take_profit_pct) if cfg.take_profit_pct > 0.0 else None
    trail_arm_lvl = entry * (1.0 + cfg.trailing_activate_pct) if cfg.trailing_stop_pct > 0.0 else None

    has_exits = (stop_lvl is not None or tp_lvl is not None
                 or cfg.trailing_stop_pct > 0.0 or cfg.time_stop_bars > 0)
    if not has_exits:
        p1 = s.loc[:d1]
        if p1.empty:
            return np.nan, "hold"
        return float(p1.iloc[-1] / entry - 1.0), "hold"

    window = s.loc[(s.index > d0) & (s.index <= d1)]
    if window.empty:
        return np.nan, "hold"

    # JIT fast path — same priority order + arithmetic as the Python loop below.
    if os.environ.get("USE_NUMBA_EXITS", "0") == "1":
        jit = _load_jit()
        if jit is not None:
            ret, code = jit(
                window.to_numpy(dtype=np.float64),
                float(entry),
                bool(stop_lvl is not None),
                float(stop_lvl) if stop_lvl is not None else 0.0,
                bool(tp_lvl is not None),
                float(tp_lvl) if tp_lvl is not None else 0.0,
                float(cfg.trailing_stop_pct),
                float(trail_arm_lvl) if trail_arm_lvl is not None else 0.0,
                int(cfg.time_stop_bars),
            )
            return float(ret), _EXIT_CODES[code]

    peak = entry
    trail_armed = False
    trail_lvl: float | None = None
    for i, (_, px) in enumerate(window.items()):
        px = float(px)

        # Update trailing stop
        if cfg.trailing_stop_pct > 0.0:
            if px > peak:
                peak = px
            if not trail_armed and trail_arm_lvl is not None and px >= trail_arm_lvl:
                trail_armed = True
            if trail_armed:
                trail_lvl = peak * (1.0 - cfg.trailing_stop_pct)

        # Exits in priority order
        if stop_lvl is not None and px <= stop_lvl:
            return float(px / entry - 1.0), "stop"
        if trail_lvl is not None and px <= trail_lvl:
            return float(px / entry - 1.0), "trail"
        if tp_lvl is not None and px >= tp_lvl:
            return float(px / entry - 1.0), "tp"
        if cfg.time_stop_bars > 0 and (i + 1) >= cfg.time_stop_bars:
            return float(px / entry - 1.0), "time"

    return float(window.iloc[-1] / entry - 1.0), "hold"


def _benchmark_equity(price_data: dict[str, pd.DataFrame], ticker: str,
                      eq_index: pd.DatetimeIndex) -> pd.Series | None:
    """Buy-and-hold equity for `ticker`, sampled on `eq_index` (close-to-close,
    base=1.0 at eq_index[0]). Returns None if data unavailable.
    """
    df = price_data.get(ticker)
    if df is None or df.empty:
        return None
    s = df["close"]
    vals = []
    base = None
    for d in eq_index:
        p = s.loc[:d]
        if p.empty:
            vals.append(np.nan)
            continue
        v = float(p.iloc[-1])
        if base is None:
            base = v
            vals.append(1.0)
        else:
            vals.append(v / base if base > 0 else np.nan)
    return pd.Series(vals, index=eq_index, name=f"{ticker}_equity")


def _abs_momentum_return(s: pd.Series, d0: pd.Timestamp, lookback: int, skip: int) -> float | None:
    """Lookback-skip return ending at d0. Returns None if not enough history."""
    hist = s.loc[:d0]
    if len(hist) <= lookback + skip:
        return None
    end = float(hist.iloc[-1 - skip]) if skip > 0 else float(hist.iloc[-1])
    start = float(hist.iloc[-1 - skip - lookback])
    if start <= 0:
        return None
    return end / start - 1.0


def _market_regime_ok(price_data: dict[str, pd.DataFrame], abs_cfg: dict, d0: pd.Timestamp) -> bool:
    """Return True if benchmark momentum > bond proxy momentum (risk-on).

    When False, the caller should rotate to abs_cfg['cash_fallback'] for the period.
    Missing data is treated as risk-on (don't block trading on data gaps).
    """
    bench = abs_cfg.get("benchmark", "SPY")
    bond = abs_cfg.get("bond_proxy")
    lookback = int(abs_cfg.get("lookback", 252))
    skip = int(abs_cfg.get("skip_recent", 21))
    b_df = price_data.get(bench)
    if b_df is None or b_df.empty:
        return True
    b_ret = _abs_momentum_return(b_df["close"], d0, lookback, skip)
    if b_ret is None:
        return True
    if bond:
        bond_df = price_data.get(bond)
        if bond_df is not None and not bond_df.empty:
            bond_ret = _abs_momentum_return(bond_df["close"], d0, lookback, skip)
            if bond_ret is not None:
                return b_ret > bond_ret
    return b_ret > 0.0


def _breakout_event_fired(df: pd.DataFrame, d0: pd.Timestamp, lookback: int,
                          vol_mult: float, vol_lookback: int,
                          window_bars: int) -> bool:
    """True if a volume-confirmed breakout fired on any of the trailing
    `window_bars` daily bars ending at d0.

    Definition (fully causal — no lookahead):
      break    : close > prior-`lookback`-bar high   (high.shift(1).rolling(lb).max(),
                 the current bar excluded — same semantics as engine.indicators.rolling_high)
      confirm  : volume >= vol_mult * volume.rolling(vol_lookback).mean()
                 (today's volume is known at the close; the rolling mean is backward-looking)

    Only used when cfg.mode == "event"; the default path never calls this.
    """
    if "volume" not in df.columns:
        return False
    hist = df.loc[:d0]
    need = max(int(lookback), int(vol_lookback)) + 1
    if len(hist) < need:
        return False
    close = hist["close"]
    high = hist["high"] if "high" in hist.columns else close
    vol = hist["volume"]
    roll_hi = high.shift(1).rolling(window=int(lookback), min_periods=int(lookback)).max()
    avg_vol = vol.rolling(window=int(vol_lookback), min_periods=int(vol_lookback)).mean()
    # NaN comparisons (warmup bars) evaluate False — exactly what we want.
    broke = close.to_numpy() > roll_hi.to_numpy()
    confirmed = vol.to_numpy() >= (float(vol_mult) * avg_vol.to_numpy())
    fired = broke & confirmed
    if fired.size == 0:
        return False
    w = max(1, int(window_bars))
    return bool(fired[-w:].any())


def _fired_series(df: pd.DataFrame, lookback: int, vol_mult: float,
                  vol_lookback: int) -> np.ndarray:
    """Vectorized per-bar event-fired flags over the WHOLE series — bit-identical
    to the `broke & confirmed` array inside _breakout_event_fired (rolling is
    causal, so computing over the full df gives the same value at each bar as
    slicing to d0). Precomputed once per ticker in managed mode so _enter can
    cheaply pre-filter to event-firing candidates instead of scoring everything."""
    if "volume" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    vol = df["volume"]
    roll_hi = high.shift(1).rolling(window=int(lookback), min_periods=int(lookback)).max()
    avg_vol = vol.rolling(window=int(vol_lookback), min_periods=int(vol_lookback)).mean()
    broke = close.to_numpy() > roll_hi.to_numpy()
    confirmed = vol.to_numpy() >= (float(vol_mult) * avg_vol.to_numpy())
    return broke & confirmed


def config_from_formula(f: Formula, **overrides) -> BacktestConfig:
    """Build a BacktestConfig from an optional `backtest:` block in the formula
    YAML, with explicit (non-None) keyword overrides taking precedence.

    Keys not matching a BacktestConfig field are ignored. This is how the two
    breakout formulas carry their event-mode + exit settings without hardcoding
    them in run.py / run_regimes.py.
    """
    blk = dict(f.raw.get("backtest") or {})
    valid = {fld.name for fld in dataclasses.fields(BacktestConfig)}
    kwargs = {k: v for k, v in blk.items() if k in valid}
    for k, v in overrides.items():
        if v is not None:
            kwargs[k] = v
    return BacktestConfig(**kwargs)


def _resolve_exit_levels(df: pd.DataFrame, d0: pd.Timestamp, entry: float,
                         cfg: BacktestConfig) -> tuple[float | None, float | None, float | None]:
    """Resolve (stop_lvl, tp_lvl, trail_arm_lvl) at entry — byte-for-byte the same
    arithmetic as _period_return_with_exits lines 143-157, so managed-mode exits
    are identical in semantics to the single-window path."""
    stop_lvl: float | None = None
    if cfg.atr_stop_mult > 0.0:
        try:
            hist = df.loc[:d0]
            atr_v = atr_mod.atr(hist, cfg.atr_stop_period).iloc[-1]
            if pd.notna(atr_v) and atr_v > 0:
                stop_lvl = entry - float(cfg.atr_stop_mult) * float(atr_v)
        except Exception:  # noqa: BLE001
            stop_lvl = None
    if stop_lvl is None and cfg.stop_loss_pct > 0.0:
        stop_lvl = entry * (1.0 - cfg.stop_loss_pct)
    tp_lvl = entry * (1.0 + cfg.take_profit_pct) if cfg.take_profit_pct > 0.0 else None
    trail_arm_lvl = entry * (1.0 + cfg.trailing_activate_pct) if cfg.trailing_stop_pct > 0.0 else None
    return stop_lvl, tp_lvl, trail_arm_lvl


def _build_precompute(price_data: dict[str, pd.DataFrame], f: Formula,
                      cfg: BacktestConfig, bank: dict | None) -> dict:
    """Set up the scoring fast-path context (mirrors run()'s setup block). Used by
    the managed path so run() itself stays byte-identical."""
    use_vec = (os.environ.get("USE_VECTORIZED_SCORING", "0") == "1"
               and cfg.rebalance.endswith("FRI"))
    use_xs = (os.environ.get("USE_CROSS_SECTION", "0") == "1"
              and cfg.rebalance.endswith("FRI"))
    precomputed: dict[str, dict | None] = {}
    universe_pre: dict | None = None
    if use_xs:
        universe_pre = precompute_universe(price_data, f)
    elif use_vec:
        assemble = None
        if bank is not None:
            from .bank import assemble_precompute as assemble  # lazy
        for tkr, df in price_data.items():
            try:
                if assemble is not None and tkr in bank:
                    precomputed[tkr] = assemble(bank[tkr], f)
                else:
                    precomputed[tkr] = precompute_indicators(df, f)
            except Exception:  # noqa: BLE001
                precomputed[tkr] = None
    return {"use_vec": use_vec, "use_xs": use_xs,
            "precomputed": precomputed, "universe_pre": universe_pre}


def _ranked_at(price_data: dict[str, pd.DataFrame], f: Formula, d0: pd.Timestamp,
               cfg: BacktestConfig, ctx: dict,
               only: set[str] | None = None) -> list[tuple[str, float]]:
    """Score + rank every ticker at d0 (no lookahead) — mirrors run()'s ranking
    block exactly so managed picks come from the same scores as rebalance/event.

    `only` (managed perf): when given, score only these tickers. Used to restrict
    ranking to the event-firing candidates — non-firers can never enter (the event
    gate in _enter skips them), so the entered set / order is bit-identical while
    avoiding scoring the whole universe on every entry bar."""
    ranked: list[tuple[str, float]] = []
    if ctx["use_xs"] and ctx["universe_pre"] is not None:
        xs_scores = score_universe_at(ctx["universe_pre"], d0, f)
        for tkr, df in price_data.items():
            if only is not None and tkr not in only:
                continue
            if len(df.loc[:d0]) < 60:
                continue
            sc = xs_scores.get(tkr, 0.0)
            if sc >= cfg.min_score:
                ranked.append((tkr, sc))
    else:
        for tkr, df in price_data.items():
            if only is not None and tkr not in only:
                continue
            if len(df.loc[:d0]) < 60:
                continue
            try:
                pre = ctx["precomputed"].get(tkr) if ctx["use_vec"] else None
                if pre is not None:
                    sc = score_ticker_at(pre, d0, f)["score"]
                else:
                    sc = score_ticker(df.loc[:d0], f)["score"]
            except Exception:  # noqa: BLE001
                continue
            if sc >= cfg.min_score:
                ranked.append((tkr, sc))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _run_managed(price_data: dict[str, pd.DataFrame], f: Formula,
                 start: str, end: str, cfg: BacktestConfig,
                 bank: dict | None = None) -> BacktestResult:
    """mode="managed": bar-by-bar portfolio sim that DECOUPLES the holding period
    from the W-FRI rebalance grid. Positions persist across rebalance bars and are
    managed on EVERY daily bar (stop / trail / tp / time, same priority and
    arithmetic as _period_return_with_exits) until an exit fires; freed slots are
    refilled at rebalance bars from the event-confirmed ranking. This is the fix
    the event-path validation pointed to: on the weekly grid every trade is force-
    closed at the next Friday (~5 bars), so time/trailing never fire — here they can.

    Produces REAL round-trip trades (a multi-week winner = ONE trade, with a
    `bars_held` column), so win-rate / payoff / avg-hold are honest.

    Documented differences from the rebalance/event path (deliberate, not parity
    bugs — managed is a separate mode; rebalance/event stay bit-identical):
      - Equity is sampled at the rebalance anchors (weekly), so _stats annualizes
        identically (n_per_year=52); exits are still evaluated on every daily bar.
      - Each sleeve compounds independently (winners are never trimmed back to
        equal weight — that's the point); new entries deploy ~equity/top_n.
      - Cost is charged once per round-trip at entry (capital0 *= 1-cost), not on
        the whole book every week — a multi-week hold pays no weekly turnover.
      - Positions still open at the window end are recorded with exit="open"
        (marked-to-market) so the trade table accounts for all capital.
    """
    reb_dates = pd.date_range(start=start, end=end, freq=cfg.rebalance)
    if len(reb_dates) < 2:
        raise ValueError("backtest window too short for the rebalance frequency")

    ctx = _build_precompute(price_data, f, cfg, bank)

    # Master daily calendar: every market day (benchmark index), else the union.
    cal = price_data.get(cfg.benchmark_ticker)
    if cal is not None and not cal.empty:
        master = cal.index
    else:
        master = pd.DatetimeIndex(sorted(set().union(*[df.index for df in price_data.values()])))
    lo, hi = reb_dates[0], reb_dates[-1]
    master = master[(master >= lo) & (master <= hi)]
    if len(master) == 0:
        raise ValueError("no trading days in the requested window")

    # Rebalance anchor = last trading day <= each calendar Friday. Scoring at the
    # anchor == scoring at the Friday (.loc[:fri] == .loc[:anchor]).
    anchors: list[pd.Timestamp] = []
    seen: set = set()
    for fri in reb_dates:
        prior = master[master <= fri]
        if len(prior):
            a = prior[-1]
            if a not in seen:
                seen.add(a)
                anchors.append(a)
    anchor_set = set(anchors)
    lb = int(cfg.event_lookback) or int(f.raw.get("indicators", {}).get("breakout_lookback", 30))
    cost = cfg.cost_bps / 10000.0

    # Precompute per-bar event flags once per ticker (bit-identical to
    # _breakout_event_fired). _firers(t) then returns the names whose event fired
    # within the trailing window ending at t via an O(1) array slice — so _enter
    # scores only event-firing candidates, not the whole universe, on every bar.
    win = max(1, int(cfg.event_window_bars))
    fired_map: dict[str, np.ndarray] = {
        tkr: _fired_series(df, lb, cfg.vol_confirm_mult, cfg.event_vol_lookback)
        for tkr, df in price_data.items()
    }

    cash = 1.0
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity: list[float] = []
    eq_index: list[pd.Timestamp] = []

    def _mv(t: pd.Timestamp) -> float:
        tot = cash
        for tkr, p in positions.items():
            px = price_data[tkr]["close"].asof(t)
            if pd.notna(px):
                tot += p["capital0"] * float(px) / p["entry_price"]
        return tot

    def _close(tkr: str, t: pd.Timestamp, px: float, reason: str) -> None:
        nonlocal cash
        p = positions.pop(tkr)
        cash += p["capital0"] * px / p["entry_price"]
        trades.append({"enter": p["entry_date"], "exit_date": t, "ticker": tkr,
                       "score": round(p["score"], 4),
                       "ret": round(px / p["entry_price"] - 1.0, 4),
                       "exit": reason, "bars_held": int(p["bars_held"])})

    def _exits(t: pd.Timestamp) -> None:
        for tkr in list(positions):
            p = positions[tkr]
            if t <= p["entry_date"]:
                continue
            px = price_data[tkr]["close"].asof(t)
            if pd.isna(px):
                continue
            px = float(px)
            p["bars_held"] += 1
            if cfg.trailing_stop_pct > 0.0:
                if px > p["peak"]:
                    p["peak"] = px
                if not p["trail_armed"] and p["trail_arm_lvl"] is not None and px >= p["trail_arm_lvl"]:
                    p["trail_armed"] = True
                if p["trail_armed"]:
                    p["trail_lvl"] = p["peak"] * (1.0 - cfg.trailing_stop_pct)
            reason = None
            if p["stop_lvl"] is not None and px <= p["stop_lvl"]:
                reason = "stop"
            elif p["trail_lvl"] is not None and px <= p["trail_lvl"]:
                reason = "trail"
            elif p["tp_lvl"] is not None and px >= p["tp_lvl"]:
                reason = "tp"
            elif cfg.time_stop_bars > 0 and p["bars_held"] >= cfg.time_stop_bars:
                reason = "time"
            if reason is not None:
                _close(tkr, t, px, reason)

    def _firers(t: pd.Timestamp) -> set[str]:
        """Names (not currently held) whose breakout event fired within the
        trailing `win` bars ending at the last bar <= t. O(1) array slice per
        ticker — same condition as _breakout_event_fired, just precomputed."""
        out: set[str] = set()
        for tkr, df in price_data.items():
            if tkr in positions:
                continue
            pos = df.index.searchsorted(t, side="right") - 1
            if pos < 0:
                continue
            lo = pos - win + 1
            if lo < 0:
                lo = 0
            if fired_map[tkr][lo:pos + 1].any():
                out.add(tkr)
        return out

    def _enter(t: pd.Timestamp) -> None:
        nonlocal cash
        free = cfg.top_n - len(positions)
        if free <= 0:
            return
        firers = _firers(t)
        if not firers:
            return
        ranked = _ranked_at(price_data, f, t, cfg, ctx, only=firers)
        mv = _mv(t)
        target = (mv / cfg.top_n) if cfg.top_n else 0.0
        added = 0
        for tkr, sc in ranked:
            if added >= free:
                break
            if tkr in positions:
                continue
            if not _breakout_event_fired(price_data[tkr], t, lb, cfg.vol_confirm_mult,
                                         cfg.event_vol_lookback, cfg.event_window_bars):
                continue
            px = price_data[tkr]["close"].asof(t)
            if pd.isna(px) or float(px) <= 0:
                continue
            entry = float(px)
            stop_lvl, tp_lvl, trail_arm_lvl = _resolve_exit_levels(price_data[tkr], t, entry, cfg)
            # Risk-based sizing: budget cfg.risk_per_trade of equity to the stop
            # distance, capped at the equal-weight target. risk_per_trade=0 (or no
            # stop) -> equal weight, byte-identical to the legacy managed path.
            size = target
            if cfg.risk_per_trade > 0.0 and stop_lvl is not None and entry > stop_lvl:
                risk_frac = (entry - stop_lvl) / entry
                if risk_frac > 0.0:
                    size = min(target, mv * cfg.risk_per_trade / risk_frac)
            deploy = min(size, cash)
            if deploy <= 1e-12:
                break
            cash -= deploy
            positions[tkr] = {
                "entry_price": entry, "entry_date": t, "capital0": deploy * (1.0 - cost),
                "peak": entry, "bars_held": 0, "score": sc,
                "stop_lvl": stop_lvl, "tp_lvl": tp_lvl,
                "trail_arm_lvl": trail_arm_lvl, "trail_armed": False, "trail_lvl": None,
            }
            added += 1

    # First anchor: baseline equity 1.0, open the initial book.
    equity.append(1.0)
    eq_index.append(anchors[0])
    _enter(anchors[0])

    for t in master:
        if t <= anchors[0]:
            continue
        _exits(t)
        if t in anchor_set:
            _enter(t)
            equity.append(_mv(t))
            eq_index.append(t)
        elif cfg.intraweek_entry:
            # ADDITIVE: scan this mid-week bar too. _enter is a no-op when the
            # book is full (free <= 0) and only fills slots whose breakout event
            # fired on/just before t. Equity is NOT sampled here — weekly sampling
            # (n_per_year=52) and the weekly re-rank above are untouched.
            _enter(t)

    # Mark-to-market close any positions still open at the final bar.
    last = master[-1]
    for tkr in list(positions):
        px = price_data[tkr]["close"].asof(last)
        if pd.notna(px):
            _close(tkr, last, float(px), "open")

    eq = pd.Series(equity, index=pd.DatetimeIndex(eq_index), name="equity")
    tr = pd.DataFrame(trades)
    bench_eq = _benchmark_equity(price_data, cfg.benchmark_ticker, eq.index)
    stats = _stats(eq, tr, cfg, bench_eq)
    return BacktestResult(equity=eq, trades=tr, stats=stats, benchmark_equity=bench_eq)


def run(price_data: dict[str, pd.DataFrame], f: Formula,
        start: str, end: str, cfg: BacktestConfig | None = None,
        bank: dict | None = None) -> BacktestResult:
    """`bank`: optional {ticker: indicator-bank} from engine.bank.build_bank, shared
    across formulas (and across fork workers). When present, the vectorized path
    assembles each ticker's precompute from it instead of recomputing — bit-identical
    output (proven by scripts/parity_bank.py), far less CPU. Ignored unless the
    vectorized path is active.
    """
    cfg = cfg or BacktestConfig()
    if cfg.mode == "managed":
        # Decoupled bar-by-bar portfolio path. Isolated entirely so the
        # rebalance/event path below stays byte-identical (parity is sacred).
        return _run_managed(price_data, f, start, end, cfg, bank)
    dates = pd.date_range(start=start, end=end, freq=cfg.rebalance)
    if len(dates) < 2:
        raise ValueError("backtest window too short for the rebalance frequency")

    abs_mom = f.raw.get("absolute_momentum") or {}
    cash_fallback = abs_mom.get("cash_fallback") if abs_mom else None

    # Vectorized scoring path — opt-in via env flag. Parity is only proven for
    # W-FRI rebalances (weekly resample bucket containing d0=Friday matches
    # the per-d0 partial resample because Sat/Sun have no trading).
    use_vec = (os.environ.get("USE_VECTORIZED_SCORING", "0") == "1"
               and cfg.rebalance.endswith("FRI"))
    use_xs = (os.environ.get("USE_CROSS_SECTION", "0") == "1"
              and cfg.rebalance.endswith("FRI"))
    precomputed: dict[str, dict | None] = {}
    universe_pre: dict | None = None
    if use_xs:
        # Cross-section vectorization subsumes the per-ticker precompute.
        universe_pre = precompute_universe(price_data, f)
    elif use_vec:
        assemble = None
        if bank is not None:
            from .bank import assemble_precompute as assemble  # lazy: only when sharing
        for tkr, df in price_data.items():
            try:
                if assemble is not None and tkr in bank:
                    precomputed[tkr] = assemble(bank[tkr], f)
                else:
                    precomputed[tkr] = precompute_indicators(df, f)
            except Exception:  # noqa: BLE001
                precomputed[tkr] = None

    equity = [1.0]
    eq_index = [dates[0]]
    trades = []
    cost = cfg.cost_bps / 10000.0

    for d0, d1 in zip(dates[:-1], dates[1:]):
        # Absolute-momentum regime check: if risk-off, rotate to cash_fallback.
        if abs_mom and cash_fallback and not _market_regime_ok(price_data, abs_mom, d0):
            fb_df = price_data.get(cash_fallback)
            if fb_df is not None and not fb_df.empty:
                r, _ = _period_return_with_exits(fb_df, d0, d1, BacktestConfig(
                    benchmark_ticker=cfg.benchmark_ticker))
                if not np.isnan(r):
                    trades.append({"enter": d0, "exit_date": d1, "ticker": cash_fallback,
                                   "score": 0.0, "ret": round(r, 4), "exit": "regime_off"})
                    equity.append(equity[-1] * (1 + r - cost))
                    eq_index.append(d1)
                    continue

        # Score every ticker using data only up to d0 (no lookahead).
        ranked = []
        if use_xs and universe_pre is not None:
            # One batched call returns {ticker: score} for the whole universe.
            xs_scores = score_universe_at(universe_pre, d0, f)
            for tkr, df in price_data.items():
                hist = df.loc[:d0]
                if len(hist) < 60:
                    continue
                sc = xs_scores.get(tkr, 0.0)
                if sc >= cfg.min_score:
                    ranked.append((tkr, sc))
        else:
            for tkr, df in price_data.items():
                hist = df.loc[:d0]
                if len(hist) < 60:
                    continue
                try:
                    pre = precomputed.get(tkr) if use_vec else None
                    if pre is not None:
                        sc = score_ticker_at(pre, d0, f)["score"]
                    else:
                        sc = score_ticker(hist, f)["score"]
                except Exception:  # noqa: BLE001
                    continue
                if sc >= cfg.min_score:
                    ranked.append((tkr, sc))
        ranked.sort(key=lambda x: x[1], reverse=True)
        if cfg.mode == "event":
            # Keep only ranked names with a volume-confirmed breakout on/just
            # before d0. Walk the sorted list and stop once top_n slots fill —
            # so we test only as many names as needed (cheap), and event-fired
            # names fill the book rather than leaving it short.
            lb = int(cfg.event_lookback) or int(f.raw.get("indicators", {}).get("breakout_lookback", 30))
            picks = []
            for tkr, sc in ranked:
                if _breakout_event_fired(price_data[tkr], d0, lb, cfg.vol_confirm_mult,
                                         cfg.event_vol_lookback, cfg.event_window_bars):
                    picks.append((tkr, sc))
                    if len(picks) >= cfg.top_n:
                        break
        else:
            picks = ranked[: cfg.top_n]

        if not picks:
            equity.append(equity[-1])
            eq_index.append(d1)
            continue

        rets = []
        for tkr, sc in picks:
            r, exit_reason = _period_return_with_exits(price_data[tkr], d0, d1, cfg)
            if not np.isnan(r):
                rets.append(r)
                trades.append({"enter": d0, "exit_date": d1, "ticker": tkr,
                               "score": round(sc, 4), "ret": round(r, 4),
                               "exit": exit_reason})
        period_ret = (np.mean(rets) if rets else 0.0) - cost
        equity.append(equity[-1] * (1 + period_ret))
        eq_index.append(d1)

    eq = pd.Series(equity, index=pd.DatetimeIndex(eq_index), name="equity")
    tr = pd.DataFrame(trades)

    bench_eq = _benchmark_equity(price_data, cfg.benchmark_ticker, eq.index)
    stats = _stats(eq, tr, cfg, bench_eq)
    return BacktestResult(equity=eq, trades=tr, stats=stats, benchmark_equity=bench_eq)


def _stats(eq: pd.Series, tr: pd.DataFrame, cfg: BacktestConfig,
           bench_eq: pd.Series | None = None) -> dict:
    rets = eq.pct_change().dropna()
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    n_per_year = {"W-FRI": 52, "W": 52, "ME": 12, "M": 12}.get(cfg.rebalance, 52)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(n_per_year)) if rets.std() else 0.0
    dd = float((eq / eq.cummax() - 1.0).min())
    win = float((tr["ret"] > 0).mean()) if not tr.empty else 0.0
    out = {
        "total_return": round(total, 4),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(dd, 4),
        "win_rate": round(win, 4),
        "n_trades": int(len(tr)),
        "n_rebalances": int(len(eq) - 1),
    }
    if not tr.empty and "exit" in tr.columns:
        out["exit_breakdown"] = tr["exit"].value_counts().to_dict()
    if bench_eq is not None and bench_eq.dropna().size >= 2:
        b = bench_eq.dropna()
        bench_total = float(b.iloc[-1] / b.iloc[0] - 1.0)
        out["benchmark_total_return"] = round(bench_total, 4)
        out["alpha_vs_benchmark"] = round(total - bench_total, 4)
    else:
        out["benchmark_total_return"] = None
        out["alpha_vs_benchmark"] = None
    return out
