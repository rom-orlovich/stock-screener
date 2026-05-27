"""Scoring: turn OHLCV into a single tunable rank score per ticker.

Reads the formula from a YAML config (formulas/*.yaml). The config holds the
numbers; this code holds the math. Claude tunes the YAML, never this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import atr as atr_mod
from . import indicators as ind


@dataclass
class Formula:
    raw: dict

    @classmethod
    def load(cls, path: str | Path) -> "Formula":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    @property
    def version(self) -> str:
        return self.raw.get("version", "unknown")

    @property
    def direction(self) -> str:
        """'long' (default, momentum-style) or 'reversion' (mean-reversion).

        Reversion inverts momentum + trend sub-scores so oversold / downtrending
        names rank highest. Other components (rsi band, breakout/breakdown) are
        already symmetric and reused as-is via the YAML config.
        """
        return self.raw.get("direction", "long")


def _norm(weights: dict) -> dict:
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def timeframe_score(df: pd.DataFrame, f: Formula) -> dict:
    """Score a single timeframe's OHLCV. Returns sub-scores + trend flag.

    Uses only the latest bar's values. Returns 0-1 components.

    When `direction: reversion` is set in the YAML, the momentum and trend
    sub-scores are inverted (1 - x) so oversold / downtrending names rank
    highest; the rsi band and breakout proximity are reused as-is (they are
    already symmetric — the YAML controls which band / extreme is rewarded).
    Setting `breakout: 0` weight together with non-zero `breakdown` is the
    recommended way to express a downside breakout — but the simpler path is
    to leave breakout active and reuse rolling_low via the indicator section.
    """
    cfg = f.raw["indicators"]
    close = df["close"]
    high = df["high"] if "high" in df else close
    low = df["low"] if "low" in df else close
    direction = f.direction

    if len(close.dropna()) < cfg["sma_slow"] + 1:
        return {"momentum": 0.0, "trend": 0.0, "rsi": 0.0,
                "breakout": 0.0, "volatility": 0.0,
                "atr_contraction": 0.0, "volume_dryup": 0.0,
                "bb_squeeze": 0.0,
                "total": 0.0, "uptrend": False}

    rsi_v = ind.rsi(close, cfg["rsi_period"]).iloc[-1]
    sf = ind.sma(close, cfg["sma_fast"]).iloc[-1]
    ss = ind.sma(close, cfg["sma_slow"]).iloc[-1]
    mom_lb = cfg["momentum_lookback"]
    mom_skip = int(cfg.get("momentum_skip_recent", 0) or 0)
    if mom_skip > 0 and len(close) > mom_lb + mom_skip:
        end_val = float(close.iloc[-1 - mom_skip])
        start_val = float(close.iloc[-1 - mom_skip - mom_lb])
        mom = (end_val / start_val - 1) * 100 if start_val > 0 else float("nan")
    else:
        mom = ind.momentum(close, mom_lb).iloc[-1]
    hi = ind.rolling_high(high, cfg["breakout_lookback"]).iloc[-1]
    lo_n = ind.rolling_low(low, cfg["breakout_lookback"]).iloc[-1]
    px = close.iloc[-1]

    # raw momentum -> 0-1 (≈ +50% -> ~1.0). Reversion inverts it.
    mom_s = _clip01((mom + 10) / 60.0) if pd.notna(mom) else 0.0

    # trend: reward fast>slow and price above both
    uptrend = pd.notna(sf) and pd.notna(ss) and sf > ss and px > sf and px > ss
    trend_s = 1.0 if uptrend else (0.5 if pd.notna(sf) and px > sf else 0.0)

    if direction == "reversion":
        mom_s = 1.0 - mom_s
        trend_s = 1.0 - trend_s

    # rsi: 1.0 inside band, decaying outside
    lo, hicut = f.raw["rsi_band"]
    if pd.isna(rsi_v):
        rsi_s = 0.0
    elif lo <= rsi_v <= hicut:
        rsi_s = 1.0
    elif rsi_v < lo:
        rsi_s = _clip01(rsi_v / lo)
    else:  # outside band on the high side
        rsi_s = _clip01((100 - rsi_v) / (100 - hicut))

    # breakout / breakdown proximity to N-bar extreme
    if direction == "reversion":
        # proximity to N-bar low (price at/below low -> 1.0)
        if pd.notna(lo_n) and lo_n > 0:
            brk_s = _clip01(lo_n / px) if px >= lo_n else 1.0
        else:
            brk_s = 0.0
    else:
        if pd.notna(hi) and hi > 0:
            brk_s = _clip01(px / hi - 0.0) if px >= hi else _clip01(1 - (hi - px) / hi * 5)
        else:
            brk_s = 0.0

    # volatility sub-score: lowest 90-day stdev of returns -> highest score
    # Optional — only contributes if YAML weights include `volatility`.
    vol_lb = cfg.get("volatility_lookback", 90)
    vol = ind.stdev_returns(close, vol_lb).iloc[-1]
    if pd.notna(vol):
        # Typical daily stdev for liquid US equities ~0.005-0.05. Below 0.01
        # is very low-vol; above 0.05 is high-vol. Squash linearly.
        vol_s = _clip01((0.05 - vol) / 0.04)
    else:
        vol_s = 0.0

    # Pre-breakout components: ATR contraction, volume dry-up, BB squeeze.
    # Only contribute when the YAML weights mention them (default 0). Each
    # peaks at 1.0 when its threshold is met and decays linearly away from it.
    atr_s = vdry_s = bbsq_s = 0.0
    if {"open", "high", "low", "close"}.issubset(df.columns):
        # ATR-% contraction: today's atr_pct vs N sessions ago.
        ap_period = cfg.get("atr_period", 14)
        ap_lookback = cfg.get("atr_contraction_lookback", 20)
        ap_thresh = cfg.get("atr_contraction_threshold", 0.30)
        try:
            ap = atr_mod.atr_pct(df, ap_period)
            if len(ap) > ap_lookback and pd.notna(ap.iloc[-1]) and pd.notna(ap.iloc[-1 - ap_lookback]):
                prev = float(ap.iloc[-1 - ap_lookback])
                cur = float(ap.iloc[-1])
                if prev > 0:
                    ratio = cur / prev  # <1 means contracted
                    target = 1.0 - ap_thresh
                    # ratio at 1.0 -> 0, ratio at target -> 1, below target stays 1
                    atr_s = _clip01((1.0 - ratio) / max(ap_thresh, 1e-6))
        except Exception:  # noqa: BLE001
            atr_s = 0.0

        # Volume dry-up: short-window avg vol vs long-window avg vol.
        v_short = cfg.get("vol_short_lookback", 20)
        v_long = cfg.get("vol_long_lookback", 60)
        v_thresh = cfg.get("vol_dryup_threshold", 0.80)
        if "volume" in df.columns:
            try:
                short_av = atr_mod.avg_volume(df, v_short).iloc[-1]
                long_av = atr_mod.avg_volume(df, v_long).iloc[-1]
                if pd.notna(short_av) and pd.notna(long_av) and long_av > 0:
                    ratio = float(short_av) / float(long_av)
                    # ratio at 1.0 -> 0, ratio at v_thresh -> 1, below stays 1
                    vdry_s = _clip01((1.0 - ratio) / max(1.0 - v_thresh, 1e-6))
            except Exception:  # noqa: BLE001
                vdry_s = 0.0

        # Bollinger band-width squeeze: current width vs percentile of trailing window.
        bb_period = cfg.get("bb_period", 20)
        bb_lookback = cfg.get("bb_squeeze_lookback", 60)
        bb_pct = cfg.get("bb_squeeze_percentile", 0.20)
        try:
            bbw = atr_mod.bollinger_band_width(close, bb_period)
            window = bbw.dropna().iloc[-bb_lookback:]
            if len(window) >= max(10, bb_lookback // 2):
                cur_w = float(window.iloc[-1])
                # rank of current width within window, 0=lowest 1=highest
                rank = float((window <= cur_w).sum() - 1) / max(len(window) - 1, 1)
                if rank <= bb_pct:
                    bbsq_s = 1.0
                else:
                    bbsq_s = _clip01((1.0 - rank) / max(1.0 - bb_pct, 1e-6))
        except Exception:  # noqa: BLE001
            bbsq_s = 0.0

    w = _norm(f.raw["timeframe_score_weights"])
    total = (w.get("momentum", 0.0) * mom_s
             + w.get("trend", 0.0) * trend_s
             + w.get("rsi", 0.0) * rsi_s
             + w.get("breakout", 0.0) * brk_s
             + w.get("volatility", 0.0) * vol_s
             + w.get("atr_contraction", 0.0) * atr_s
             + w.get("volume_dryup", 0.0) * vdry_s
             + w.get("bb_squeeze", 0.0) * bbsq_s)
    return {"momentum": round(mom_s, 4), "trend": round(trend_s, 4),
            "rsi": round(rsi_s, 4), "breakout": round(brk_s, 4),
            "volatility": round(vol_s, 4),
            "atr_contraction": round(atr_s, 4),
            "volume_dryup": round(vdry_s, 4),
            "bb_squeeze": round(bbsq_s, 4),
            "total": round(total, 4), "uptrend": bool(uptrend)}


def score_ticker(daily: pd.DataFrame, f: Formula) -> dict:
    """Full multi-timeframe score for one ticker from its daily OHLCV."""
    weekly = ind.resample_ohlcv(daily, "W")
    monthly = ind.resample_ohlcv(daily, "ME")

    sub = {
        "daily": timeframe_score(daily, f),
        "weekly": timeframe_score(weekly, f),
        "monthly": timeframe_score(monthly, f),
    }
    tw = _norm(f.raw["timeframe_weights"])
    base = sum(tw[tf] * sub[tf]["total"] for tf in tw)

    # Alignment: long -> all uptrend; reversion -> none uptrend (i.e. consistent
    # downtrend / weak across timeframes).
    if f.direction == "reversion":
        aligned = not any(sub[tf]["uptrend"] for tf in ("daily", "weekly", "monthly"))
    else:
        aligned = all(sub[tf]["uptrend"] for tf in ("daily", "weekly", "monthly"))
    bonus = f.raw.get("alignment_bonus", 0.0) if aligned else 0.0
    final = round(_clip01(base + bonus), 4)

    return {"score": final, "aligned": aligned,
            "base": round(base, 4), "bonus": bonus, "timeframes": sub}


def rank(price_data: dict[str, pd.DataFrame], f: Formula) -> pd.DataFrame:
    """Rank a universe. `price_data`: {ticker: daily OHLCV df}."""
    rows = []
    for tkr, df in price_data.items():
        if df is None or df.empty:
            continue
        try:
            r = score_ticker(df, f)
        except Exception as e:  # noqa: BLE001 — one bad ticker shouldn't kill the scan
            rows.append({"ticker": tkr, "score": np.nan, "error": str(e)})
            continue
        rows.append({
            "ticker": tkr, "score": r["score"], "aligned": r["aligned"],
            "daily": r["timeframes"]["daily"]["total"],
            "weekly": r["timeframes"]["weekly"]["total"],
            "monthly": r["timeframes"]["monthly"]["total"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
