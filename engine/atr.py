"""ATR + volatility/range helpers used by pre-breakout patterns and exits.

Pure, deterministic. Same input -> same output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range = max(high-low, |high - prev_close|, |low - prev_close|)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR — exponential moving average of true range."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as a percentage of close. Useful for cross-ticker comparison."""
    return atr(df, period) / df["close"]


def bollinger_band_width(close: pd.Series, period: int = 20,
                         num_std: float = 2.0) -> pd.Series:
    """(upper - lower) / mid. Lower = tighter = squeezed."""
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (upper - lower) / mid


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — cumulative volume signed by close-to-close direction."""
    close = df["close"]
    vol = df["volume"]
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * vol).cumsum()


def avg_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"].rolling(window=period, min_periods=period).mean()


def avg_dollar_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return (df["close"] * df["volume"]).rolling(window=period, min_periods=period).mean()


def base_range_pct(df: pd.DataFrame, lookback: int = 30) -> pd.Series:
    """Range of the trailing `lookback` window as a fraction of its midpoint.

    Used to detect consolidation bases: low values (e.g. <0.10) mean price
    has been confined to a tight range — a classic pre-breakout setup.
    Returns (rolling_high - rolling_low) / rolling_mid.
    """
    high = df["high"] if "high" in df.columns else df["close"]
    low = df["low"] if "low" in df.columns else df["close"]
    rh = high.rolling(window=lookback, min_periods=lookback).max()
    rl = low.rolling(window=lookback, min_periods=lookback).min()
    mid = (rh + rl) / 2.0
    return (rh - rl) / mid.replace(0.0, pd.NA)


def base_pivot(df: pd.DataFrame, lookback: int = 30) -> pd.Series:
    """Rolling high over `lookback` bars — the breakout trigger level."""
    high = df["high"] if "high" in df.columns else df["close"]
    return high.rolling(window=lookback, min_periods=lookback).max()
