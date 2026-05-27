"""Data fetch — pluggable provider. Daily OHLCV in, normalized DataFrame out.

Phase 1 default provider is yfinance (no API key) so the backtest runs today.
An Alpaca provider stub is included for when the free data key is added
(design Phase 1/2). All providers return the same normalized schema:

    DatetimeIndex (tz-naive, ascending) + columns open/high/low/close/volume
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "runs" / "_data_cache"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_yf(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True,
                      progress=False, multi_level_index=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    return _normalize(raw)


def fetch_alpaca(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Alpaca free-tier daily bars. Needs ALPACA_KEY_ID / ALPACA_SECRET env vars."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(os.environ["ALPACA_KEY_ID"],
                                       os.environ["ALPACA_SECRET"])
    req = StockBarsRequest(symbol_or_symbols=ticker, timeframe=TimeFrame.Day,
                           start=pd.Timestamp(start), end=pd.Timestamp(end) if end else None)
    bars = client.get_stock_bars(req).df
    if bars.empty:
        return pd.DataFrame()
    if "symbol" in bars.index.names:
        bars = bars.xs(ticker, level="symbol")
    return _normalize(bars)


PROVIDERS = {"yf": fetch_yf, "alpaca": fetch_alpaca}


def get_history(ticker: str, start: str, end: str | None = None,
                provider: str = "yf", use_cache: bool = True) -> pd.DataFrame:
    """Fetch one ticker's daily history, with on-disk parquet cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{provider}_{ticker}_{start}_{end or 'now'}.pkl"
    if use_cache and cache.exists():
        return pd.read_pickle(cache)
    df = PROVIDERS[provider](ticker, start, end)
    if use_cache and not df.empty:
        df.to_pickle(cache)
    return df


def get_universe(tickers: list[str], start: str, end: str | None = None,
                 provider: str = "yf") -> dict[str, pd.DataFrame]:
    """Fetch many tickers. Returns {ticker: df}, skipping empties."""
    out = {}
    for t in tickers:
        df = get_history(t, start, end, provider=provider)
        if not df.empty:
            out[t] = df
    return out
