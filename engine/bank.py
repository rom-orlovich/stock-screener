"""Shared per-ticker indicator bank — compute each (indicator, period) ONCE,
reuse across every formula.

`precompute_indicators` in score.py is keyed by formula, but every series it
builds (rsi/sma/momentum/rolling_high/low/stdev/atr_pct/bb_width/avg_volume +
the monthly resample) is a pure function of (price series, indicator, period) —
none depend on the formula's weights. Running N formulas over the same ticker
therefore recomputes identical series N times. The bank computes the UNION of
every (indicator, period) any formula needs, once per ticker, and `assemble`
selects each formula's subset — yielding a dict bit-identical to
`precompute_indicators(daily, f)`.

Composes with the fork pool: build the bank once in the parent, let workers
inherit it via copy-on-write. Read-only after construction (no mutation in the
scoring path), so sharing is safe.

Parity: scripts/parity_bank.py asserts assemble(build(...), f) ==
precompute_indicators(daily, f) series-for-series (max abs diff 0.0).
"""
from __future__ import annotations

import pandas as pd

from . import atr as atr_mod
from . import indicators as ind
from .score import Formula

# Default periods mirror precompute_indicators' cfg.get(...) fallbacks exactly.
_VOL_DEFAULT = 90
_ATR_DEFAULT = 14
_BB_DEFAULT = 20
_VOL_SHORT_DEFAULT = 20
_VOL_LONG_DEFAULT = 60


def _specs_for(cfg: dict) -> set[tuple[str, int]]:
    """The (indicator, period) tuples one formula's indicator block needs.

    Mirrors the exact function+period pairs in score.precompute_indicators._compute.
    """
    return {
        ("rsi", cfg["rsi_period"]),
        ("sma", cfg["sma_fast"]),
        ("sma", cfg["sma_slow"]),
        ("momentum", cfg["momentum_lookback"]),
        ("rolling_high", cfg["breakout_lookback"]),
        ("rolling_low", cfg["breakout_lookback"]),
        ("stdev_returns", cfg.get("volatility_lookback", _VOL_DEFAULT)),
        ("atr_pct", cfg.get("atr_period", _ATR_DEFAULT)),
        ("bb_width", cfg.get("bb_period", _BB_DEFAULT)),
        ("avg_vol", cfg.get("vol_short_lookback", _VOL_SHORT_DEFAULT)),
        ("avg_vol", cfg.get("vol_long_lookback", _VOL_LONG_DEFAULT)),
    }


def collect_specs(formulas: list[Formula]) -> set[tuple[str, int]]:
    """Union of every (indicator, period) across all formulas."""
    specs: set[tuple[str, int]] = set()
    for f in formulas:
        specs |= _specs_for(f.raw["indicators"])
    return specs


def _build_tf(df: pd.DataFrame, specs: set[tuple[str, int]]) -> dict:
    """Per-timeframe bank: price columns + every spec series the union needs.

    Each series is computed with the SAME call score.precompute_indicators._compute
    uses, and the same OHLC/volume guards + per-indicator try/except (so a series
    that would be absent in the legacy dict is simply absent here too).
    """
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    has_ohlc = {"open", "high", "low", "close"}.issubset(df.columns)
    has_vol = "volume" in df.columns

    series: dict[tuple[str, int], pd.Series] = {}
    for name, period in specs:
        if name == "rsi":
            series[(name, period)] = ind.rsi(close, period)
        elif name == "sma":
            series[(name, period)] = ind.sma(close, period)
        elif name == "momentum":
            series[(name, period)] = ind.momentum(close, period)
        elif name == "rolling_high":
            series[(name, period)] = ind.rolling_high(high, period)
        elif name == "rolling_low":
            series[(name, period)] = ind.rolling_low(low, period)
        elif name == "stdev_returns":
            series[(name, period)] = ind.stdev_returns(close, period)
        elif name == "atr_pct":
            if has_ohlc:
                try:
                    series[(name, period)] = atr_mod.atr_pct(df, period)
                except Exception:  # noqa: BLE001 — match _compute's swallow
                    pass
        elif name == "bb_width":
            if has_ohlc:
                try:
                    series[(name, period)] = atr_mod.bollinger_band_width(close, period)
                except Exception:  # noqa: BLE001
                    pass
        elif name == "avg_vol":
            if has_ohlc and has_vol:
                try:
                    series[(name, period)] = atr_mod.avg_volume(df, period)
                except Exception:  # noqa: BLE001
                    pass
    return {"df": df, "close": close, "high": high, "low": low, "series": series}


def build_bank(daily: pd.DataFrame, specs: set[tuple[str, int]]) -> dict:
    """Per-ticker bank covering the daily + weekly timeframes and the monthly parts.

    Formula-independent. The monthly resample + cumulative buckets are identical
    to score.precompute_indicators (they never depended on the formula at all).
    """
    weekly = ind.resample_ohlcv(daily, "W")

    monthly_full = ind.resample_ohlcv(daily, "ME")
    monthly_partial: dict[str, pd.Series] = {}
    has_full_ohlc = {"open", "high", "low", "close"}.issubset(daily.columns)
    if has_full_ohlc and len(daily) > 0:
        gb = daily.groupby(daily.index.to_period("M"), sort=False)
        monthly_partial["open"] = gb["open"].transform("first")
        monthly_partial["high"] = gb["high"].cummax()
        monthly_partial["low"] = gb["low"].cummin()
        if "volume" in daily.columns:
            monthly_partial["volume"] = gb["volume"].cumsum()

    return {
        "daily": _build_tf(daily, specs),
        "weekly": _build_tf(weekly, specs),
        "monthly_full": monthly_full,
        "monthly_partial": monthly_partial,
        "_daily_raw": daily,
    }


def _assemble_tf(tf_bank: dict, cfg: dict) -> dict:
    """Select one formula's roles from a timeframe bank — same keys _compute emits."""
    s = tf_bank["series"]
    out: dict = {
        "df": tf_bank["df"],
        "close": tf_bank["close"],
        "high": tf_bank["high"],
        "low": tf_bank["low"],
        "rsi": s[("rsi", cfg["rsi_period"])],
        "sma_fast": s[("sma", cfg["sma_fast"])],
        "sma_slow": s[("sma", cfg["sma_slow"])],
        "momentum": s[("momentum", cfg["momentum_lookback"])],
        "rolling_high": s[("rolling_high", cfg["breakout_lookback"])],
        "rolling_low": s[("rolling_low", cfg["breakout_lookback"])],
        "stdev_returns": s[("stdev_returns", cfg.get("volatility_lookback", _VOL_DEFAULT))],
    }
    # Optional keys: present only if the bank computed them (OHLC/volume guards),
    # exactly mirroring _compute, where a failed/guarded indicator leaves the key out.
    ap = s.get(("atr_pct", cfg.get("atr_period", _ATR_DEFAULT)))
    if ap is not None:
        out["atr_pct"] = ap
    bw = s.get(("bb_width", cfg.get("bb_period", _BB_DEFAULT)))
    if bw is not None:
        out["bb_width"] = bw
    avs = s.get(("avg_vol", cfg.get("vol_short_lookback", _VOL_SHORT_DEFAULT)))
    if avs is not None:
        out["avg_vol_short"] = avs
    avl = s.get(("avg_vol", cfg.get("vol_long_lookback", _VOL_LONG_DEFAULT)))
    if avl is not None:
        out["avg_vol_long"] = avl
    return out


def assemble_precompute(bank: dict, f: Formula) -> dict:
    """Reconstruct the score.precompute_indicators(daily, f) dict from a ticker bank.

    Returns the exact same structure score_ticker_at consumes — selecting this
    formula's periods from the shared series.
    """
    cfg = f.raw["indicators"]
    return {
        "daily": _assemble_tf(bank["daily"], cfg),
        "weekly": _assemble_tf(bank["weekly"], cfg),
        "monthly_full": bank["monthly_full"],
        "monthly_partial": bank["monthly_partial"],
        "_daily_raw": bank["_daily_raw"],
    }
