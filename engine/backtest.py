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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import atr as atr_mod
from .score import Formula, score_ticker


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

    benchmark_ticker: str = "SPY"  # buy-and-hold ref over the same window.


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


def run(price_data: dict[str, pd.DataFrame], f: Formula,
        start: str, end: str, cfg: BacktestConfig | None = None) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    dates = pd.date_range(start=start, end=end, freq=cfg.rebalance)
    if len(dates) < 2:
        raise ValueError("backtest window too short for the rebalance frequency")

    abs_mom = f.raw.get("absolute_momentum") or {}
    cash_fallback = abs_mom.get("cash_fallback") if abs_mom else None

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
        for tkr, df in price_data.items():
            hist = df.loc[:d0]
            if len(hist) < 60:
                continue
            try:
                sc = score_ticker(hist, f)["score"]
            except Exception:  # noqa: BLE001
                continue
            if sc >= cfg.min_score:
                ranked.append((tkr, sc))
        ranked.sort(key=lambda x: x[1], reverse=True)
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
