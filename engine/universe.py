"""Fetch ticker universes. S&P 500 from Wikipedia; Russell 3000/1000 from
iShares ETF holdings. All cached on disk, with a liquidity prefilter."""
from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

CACHE = Path(__file__).resolve().parent.parent / "runs" / "_universe_cache"

# Public iShares fund-document API. portfolioId selects the ETF; omitting
# asOfDate returns the latest published holdings, component=holdings the CSV.
# (The legacy `*.ajax?fileType=csv` URL now serves an SPA shell behind a bot
#  wall — this varnish-api host is the current, curl-reachable feed.)
_ISHARES_API = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/"
    "product-data/api/v1/get-fund-document"
)
_ISHARES_PARAMS = {
    "appType": "PRODUCT_PAGE",
    "appSubType": "ISHARES",
    "targetSite": "us-ishares",
    "locale": "en_US",
    "userType": "individual",
    "component": "holdings",
}
# Plausible US listing symbol after normalising '.' -> '-' (drops cash rows,
# blanks, futures and internal placeholders like 'P5N994').
_TICKER_RE = re.compile(r"[A-Z]{1,6}(-[A-Z]{1,3})?$")
# iShares concatenates dual-class share tickers (e.g. 'BRKB'); a handful of
# these are listed on Yahoo/yfinance with a dash. Most class shares already
# match (GOOGL, META, FOXA, NWSA, …) — only the dash-convention names need a
# remap. Verified against yfinance. Others not in this map (if any) simply
# fail the fetch and are dropped, like any delisted name.
_YF_OVERRIDES = {
    "BRKB": "BRK-B", "BFB": "BF-B", "BFA": "BF-A", "LENB": "LEN-B",
    "HEIA": "HEI-A", "MOGA": "MOG-A", "GEFB": "GEF-B",
}


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


def _ishares_holdings(portfolio_id: str, label: str, cache_name: str,
                      min_count: int, max_count: int) -> list[str]:
    """Return the equity tickers of an iShares ETF, cached on disk.

    Mirrors sp500(): cache-first, normalise '.' -> '-' for yfinance, keep only
    equity rows. Order is preserved from the CSV (largest weight first)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / cache_name
    if cache_file.exists():
        return pd.read_csv(cache_file)["Symbol"].tolist()

    params = {**_ISHARES_PARAMS, "portfolioId": portfolio_id}
    resp = requests.get(
        _ISHARES_API, params=params, timeout=60,
        headers={"User-Agent": UA,
                 "Referer": f"https://www.ishares.com/us/products/{portfolio_id}/",
                 "Accept": "text/csv,*/*"},
    )
    resp.raise_for_status()
    text = resp.text
    if text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        raise RuntimeError(
            f"iShares returned an HTML bot-wall page instead of the {label} "
            f"holdings CSV. Refresh the cache manually into {cache_file}.")

    # The CSV has a fund-info preamble; the table starts at the 'Ticker,' header.
    lines = text.splitlines()
    try:
        hi = next(i for i, ln in enumerate(lines[:30])
                  if ln.lstrip().startswith("Ticker,") and "Asset Class" in ln)
    except StopIteration:
        raise RuntimeError(f"Could not locate the holdings header in the {label} CSV.")
    df = pd.read_csv(StringIO("\n".join(lines[hi:])))

    eq = df[df["Asset Class"].astype(str).str.strip().str.lower() == "equity"]
    sym = (eq["Ticker"].astype(str).str.strip()
           .str.replace(".", "-", regex=False)
           .map(lambda t: _YF_OVERRIDES.get(t, t)))
    tickers = [t for t in dict.fromkeys(sym.tolist()) if _TICKER_RE.fullmatch(t)]
    if not (min_count <= len(tickers) <= max_count):
        raise RuntimeError(
            f"{label} holdings count {len(tickers)} outside the sane range "
            f"[{min_count}, {max_count}] — check the iShares feed.")
    pd.DataFrame({"Symbol": tickers}).to_csv(cache_file, index=False)
    return tickers


def russell3000() -> list[str]:
    """Return Russell 3000 tickers from the iShares IWV ETF (cached)."""
    return _ishares_holdings("239714", "iShares Russell 3000 ETF (IWV)",
                             "russell3000.csv", 2400, 3100)


def russell1000() -> list[str]:
    """Return Russell 1000 tickers from the iShares IWB ETF (cached)."""
    return _ishares_holdings("239707", "iShares Russell 1000 ETF (IWB)",
                             "russell1000.csv", 900, 1100)


_UNIVERSES = {
    "sp500": sp500,
    "russell3000": russell3000,
    "russell1000": russell1000,
}


def get_universe_tickers(name: str) -> list[str]:
    """Resolve a named universe ('sp500', 'russell3000', 'russell1000')."""
    try:
        return _UNIVERSES[name]()
    except KeyError:
        raise ValueError(
            f"unknown universe '{name}'. Choose from: {', '.join(_UNIVERSES)}")


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
