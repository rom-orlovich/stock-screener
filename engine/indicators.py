"""Pure, deterministic technical indicators. Same input -> same output.

All functions take a pandas Series/DataFrame of prices and return aligned
Series. No I/O, no randomness, no global state.
"""
from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100.0)  # zero losses -> max strength


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


def momentum(close: pd.Series, lookback: int) -> pd.Series:
    """Percent change over `lookback` bars."""
    return close.pct_change(periods=lookback) * 100.0


def rolling_high(high: pd.Series, lookback: int) -> pd.Series:
    """Highest high over the prior `lookback` bars (excludes current bar)."""
    return high.shift(1).rolling(window=lookback, min_periods=lookback).max()


def rolling_low(low: pd.Series, lookback: int) -> pd.Series:
    """Lowest low over the prior `lookback` bars (excludes current bar)."""
    return low.shift(1).rolling(window=lookback, min_periods=lookback).min()


def stdev_returns(close: pd.Series, lookback: int = 60) -> pd.Series:
    """Rolling standard deviation of daily log-returns (volatility proxy)."""
    rets = close.pct_change()
    return rets.rolling(window=lookback, min_periods=lookback // 2).std()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample daily OHLCV to a coarser timeframe.

    `rule`: 'W' (weekly) or 'ME' (month-end). Expects columns
    open/high/low/close/volume and a DatetimeIndex.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = [c for c in agg if c in df.columns]
    out = df[cols].resample(rule).agg({c: agg[c] for c in cols})
    return out.dropna(how="all")
