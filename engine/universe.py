"""Fetch ticker universes. S&P 500 from Wikipedia, with liquidity prefilter."""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

CACHE = Path(__file__).resolve().parent.parent / "runs" / "_universe_cache"


def sp500() -> list[str]:
    """Return current S&P 500 tickers (cached on disk)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / "sp500.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file)["Symbol"].tolist()
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": UA}, timeout=30).text
    df = pd.read_html(StringIO(html))[0]
    # Wikipedia uses '.' but yfinance wants '-' (e.g. BRK.B -> BRK-B)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    df[["Symbol"]].to_csv(cache_file, index=False)
    return df["Symbol"].tolist()


def liquidity_filter(data: dict, min_avg_dollar_vol: float = 5_000_000) -> dict:
    """Keep only tickers whose 60-day average $-volume >= threshold."""
    out = {}
    for t, df in data.items():
        if len(df) < 60:
            continue
        recent = df.tail(60)
        avg_dollar_vol = (recent["close"] * recent["volume"]).mean()
        if avg_dollar_vol >= min_avg_dollar_vol:
            out[t] = df
    return out
