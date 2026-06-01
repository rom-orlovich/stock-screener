"""Scoring: turn OHLCV into a single tunable rank score per ticker.

Reads the formula from a YAML config (formulas/*.yaml). The config holds the
numbers; this code holds the math. Claude tunes the YAML, never this file.
"""
from __future__ import annotations

import os
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


def _breakout_thrust_score(px: float, hi: float,
                           lo_band: float, hi_band: float, fade: float) -> float:
    """Reward `close` sitting just above the causal pivot `hi`.

    `hi` is the prior-N-bar high (`rolling_high` excludes the current bar — no
    lookahead). Trapezoid in pct = px/hi - 1: 0 at/below the pivot, ramps 0->1
    across (0, lo_band], holds 1.0 on [lo_band, hi_band], fades 1->0 across
    (hi_band, fade), and 0 beyond `fade` (don't chase an extended break).
    """
    if not (pd.notna(hi) and hi > 0):
        return 0.0
    pct = px / hi - 1.0
    if pct <= 0.0:
        return 0.0
    if pct < lo_band:
        return _clip01(pct / lo_band)
    if pct <= hi_band:
        return 1.0
    if pct < fade:
        return _clip01((fade - pct) / (fade - hi_band))
    return 0.0


def _gap_score(gap_pct: float, lo_band: float, hi_band: float, fade: float) -> float:
    """Reward a recent accumulation gap-up sitting in a band.

    `gap_pct` is the largest single-bar up-gap (open vs prior close) over the
    lookback window — causal, only bars up to the evaluation bar. Triangle peaking
    at `hi_band`: 0 below `lo_band` (noise, not a real gap), ramps lo->hi to 1.0,
    fades hi->fade back to 0 (a >fade gap is a blow-off / exhaustion move — don't
    buy it). Below 0 (gap-down) scores 0; this is a long-only confirmation term.
    """
    if not (pd.notna(gap_pct)) or gap_pct <= 0.0 or gap_pct < lo_band:
        return 0.0
    if gap_pct < hi_band:
        return _clip01((gap_pct - lo_band) / max(hi_band - lo_band, 1e-9))
    if gap_pct < fade:
        return _clip01((fade - gap_pct) / max(fade - hi_band, 1e-9))
    return 0.0


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

    # Cap indicator windows to what the timeframe can actually supply. With
    # ~3y of monthly bars there are only ~41 samples, so YAML sma_slow=50/200
    # would otherwise trip the guard and zero out the entire monthly score.
    # Daily/weekly with thousands of bars are unaffected (min() returns the
    # YAML value).
    n_avail = int(close.dropna().shape[0])
    local_sma_slow = min(cfg["sma_slow"], max(20, n_avail // 3))
    # n_avail//6 is the floor so daily/weekly (thousands of bars) keep
    # cfg["sma_fast"] exactly — strict parity on the well-supplied path.
    local_sma_fast = min(cfg["sma_fast"], max(5, n_avail // 6))
    local_mom_lb = min(cfg["momentum_lookback"], max(3, n_avail // 4))
    local_breakout_lb = min(cfg["breakout_lookback"], max(10, n_avail // 4))

    if n_avail < local_sma_slow + 1:
        return {"momentum": 0.0, "trend": 0.0, "rsi": 0.0,
                "breakout": 0.0, "breakout_thrust": 0.0, "volatility": 0.0,
                "atr_contraction": 0.0, "volume_dryup": 0.0,
                "bb_squeeze": 0.0, "gap": 0.0,
                "total": 0.0, "uptrend": False}

    rsi_v = ind.rsi(close, cfg["rsi_period"]).iloc[-1]
    sf = ind.sma(close, local_sma_fast).iloc[-1]
    ss = ind.sma(close, local_sma_slow).iloc[-1]
    mom_lb = local_mom_lb
    mom_skip = int(cfg.get("momentum_skip_recent", 0) or 0)
    if mom_skip > 0 and len(close) > mom_lb + mom_skip:
        end_val = float(close.iloc[-1 - mom_skip])
        start_val = float(close.iloc[-1 - mom_skip - mom_lb])
        mom = (end_val / start_val - 1) * 100 if start_val > 0 else float("nan")
    else:
        mom = ind.momentum(close, mom_lb).iloc[-1]
    hi = ind.rolling_high(high, local_breakout_lb).iloc[-1]
    lo_n = ind.rolling_low(low, local_breakout_lb).iloc[-1]
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

    # breakout_thrust: reward the early break (close 1-3% above the causal pivot),
    # fading out beyond ~5% so we don't chase. Uses the same `hi` (rolling_high,
    # current bar excluded) — causal, no lookahead.
    t_lo = cfg.get("breakout_thrust_band_lo", 0.01)
    t_hi = cfg.get("breakout_thrust_band_hi", 0.03)
    t_fade = cfg.get("breakout_thrust_fade", 0.05)
    thrust_s = _breakout_thrust_score(px, hi, t_lo, t_hi, t_fade)

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

    # gap sub-score: largest single-bar up-gap (open vs prior close) over the
    # trailing gap_lookback bars. Optional — only contributes if YAML weights
    # mention `gap`. Causal: uses only bars up to the current one.
    gap_s = 0.0
    if "open" in df.columns and len(close) > 1:
        glb = int(cfg.get("gap_lookback", 10))
        g_lo = cfg.get("gap_band_lo", 0.02)
        g_hi = cfg.get("gap_band_hi", 0.05)
        g_fade = cfg.get("gap_fade", 0.12)
        prev_close = close.shift(1)
        gaps = (df["open"] - prev_close) / prev_close
        recent = gaps.iloc[-glb:].dropna()
        if not recent.empty:
            gap_s = _gap_score(float(recent.max()), g_lo, g_hi, g_fade)

    # Stage-2 trend gate (opt-in via YAML): the quietness terms only count when
    # price sits above the slow SMA, dropping "quiet downtrend / topping" false
    # positives. Causal — `ss` uses only data up to the current bar.
    if cfg.get("trend_gate_quietness", False) and not (pd.notna(ss) and px > ss):
        atr_s = vdry_s = bbsq_s = 0.0

    w = _norm(f.raw["timeframe_score_weights"])
    total = (w.get("momentum", 0.0) * mom_s
             + w.get("trend", 0.0) * trend_s
             + w.get("rsi", 0.0) * rsi_s
             + w.get("breakout", 0.0) * brk_s
             + w.get("breakout_thrust", 0.0) * thrust_s
             + w.get("volatility", 0.0) * vol_s
             + w.get("atr_contraction", 0.0) * atr_s
             + w.get("volume_dryup", 0.0) * vdry_s
             + w.get("bb_squeeze", 0.0) * bbsq_s
             + w.get("gap", 0.0) * gap_s)
    return {"momentum": round(mom_s, 4), "trend": round(trend_s, 4),
            "rsi": round(rsi_s, 4), "breakout": round(brk_s, 4),
            "breakout_thrust": round(thrust_s, 4),
            "volatility": round(vol_s, 4),
            "atr_contraction": round(atr_s, 4),
            "volume_dryup": round(vdry_s, 4),
            "bb_squeeze": round(bbsq_s, 4),
            "gap": round(gap_s, 4),
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


# ---------------------------------------------------------------------------
# Vectorized scoring path (precompute indicators once per ticker, reuse per d0).
# Gated by USE_VECTORIZED_SCORING in engine/backtest.py — default OFF.
#
# Parity guarantees (vs the per-d0 path above):
#   * Daily indicators are causal — value at position pos on full series equals
#     value at position -1 on the prior slice.
#   * Weekly indicators: SAFE only when the rebalance date d0 is a Friday
#     (W-FRI rebalance). The W (anchor=Sunday) bucket containing d0=Friday
#     holds Mon-Fri data only because there is no Sat/Sun trading; the same
#     bucket is what `resample_ohlcv(daily.loc[:Friday], 'W')` produces.
#     For any non-Friday d0 this would leak future bars.
#   * Monthly is NOT precomputed — the full-month bucket leaks future bars
#     when d0 falls inside the month. Monthly is rebuilt per d0 from
#     `daily_raw.loc[:d0]`, identical to the per-d0 path.
# ---------------------------------------------------------------------------


def _pos_at_daily(index: pd.DatetimeIndex, d0: pd.Timestamp) -> int:
    """Position of last bar at or before d0 in a daily index. -1 if none."""
    return int(index.searchsorted(d0, side="right")) - 1


def _pos_at_weekly(index: pd.DatetimeIndex, d0: pd.Timestamp) -> int:
    """Position of bucket containing d0 in a weekly W-anchor=SUN index.

    For d0 = Friday, the matching bucket label is the following Sunday; for
    d0 = Sunday the label equals d0 itself. Returns len(index) when d0 is past
    the last bucket label.
    """
    return int(index.searchsorted(d0, side="left"))


def precompute_indicators(daily: pd.DataFrame, f: Formula) -> dict:
    """Pre-compute every indicator the scorer needs on the FULL series.

    Returns a dict keyed by timeframe ('daily', 'weekly') plus '_daily_raw'
    (used by the monthly per-d0 path inside `score_ticker_at`).
    """
    cfg = f.raw["indicators"]

    def _compute(df: pd.DataFrame) -> dict:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close
        out: dict = {
            "df": df,
            "close": close,
            "high": high,
            "low": low,
            "rsi": ind.rsi(close, cfg["rsi_period"]),
            "sma_fast": ind.sma(close, cfg["sma_fast"]),
            "sma_slow": ind.sma(close, cfg["sma_slow"]),
            "momentum": ind.momentum(close, cfg["momentum_lookback"]),
            "rolling_high": ind.rolling_high(high, cfg["breakout_lookback"]),
            "rolling_low": ind.rolling_low(low, cfg["breakout_lookback"]),
            "stdev_returns": ind.stdev_returns(close, cfg.get("volatility_lookback", 90)),
        }
        if {"open", "high", "low", "close"}.issubset(df.columns):
            # Per-bar up-gap (open vs prior close); _timeframe_score_at takes a
            # trailing rolling max. Causal — open[i] and close[i-1] only.
            out["gap"] = (df["open"] - close.shift(1)) / close.shift(1)
            try:
                out["atr_pct"] = atr_mod.atr_pct(df, cfg.get("atr_period", 14))
            except Exception:  # noqa: BLE001
                pass
            try:
                out["bb_width"] = atr_mod.bollinger_band_width(close, cfg.get("bb_period", 20))
            except Exception:  # noqa: BLE001
                pass
            if "volume" in df.columns:
                try:
                    out["avg_vol_short"] = atr_mod.avg_volume(df, cfg.get("vol_short_lookback", 20))
                    out["avg_vol_long"] = atr_mod.avg_volume(df, cfg.get("vol_long_lookback", 60))
                except Exception:  # noqa: BLE001
                    pass
        return out

    weekly = ind.resample_ohlcv(daily, "W")

    # Monthly precompute: build the full-month resample once + month-to-date
    # cumulatives so `score_ticker_at` can construct the "as of d0" monthly
    # bucket in O(1) instead of resampling the daily slice on every call.
    # 88k calls × ~0.5ms resample = ~44s wasted per backtest in the old path.
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
        "daily": _compute(daily),
        "weekly": _compute(weekly),
        "monthly_full": monthly_full,
        "monthly_partial": monthly_partial,
        "_daily_raw": daily,
    }


def _build_monthly_at_d0(precomputed: dict, d0: pd.Timestamp) -> pd.DataFrame:
    """O(1) construction of the monthly OHLCV "as of d0" from precomputed parts.

    Replaces `ind.resample_ohlcv(daily.loc[:d0], 'ME')` — same result, no
    per-d0 resample. Parity tested in scripts/parity_fast_monthly.py.
    """
    monthly_full = precomputed["monthly_full"]
    monthly_partial = precomputed["monthly_partial"]
    daily = precomputed["_daily_raw"]
    if len(daily) == 0:
        return monthly_full.iloc[:0]

    daily_pos = _pos_at_daily(daily.index, d0)
    if daily_pos < 0:
        return monthly_full.iloc[:0]

    # Position of the month-end label whose bucket contains d0's month.
    # 'ME' labels are month-end timestamps; searchsorted(d0, 'left') returns the
    # index of the first label >= d0, which is the month-end of d0's month.
    month_idx = int(monthly_full.index.searchsorted(d0, side="left"))

    # Case A: d0 falls exactly on a month-end already represented in monthly_full
    # (rare — d0 must be a trading day equal to the resample label). Use the
    # full bucket as-is; no partial construction needed.
    if month_idx < len(monthly_full) and monthly_full.index[month_idx] == d0:
        return monthly_full.iloc[: month_idx + 1]

    # Case B: d0 is mid-month — synthesize the partial bucket from cumulatives.
    if not monthly_partial:
        # No OHLC; fall back to the per-d0 resample to preserve close-only paths.
        return ind.resample_ohlcv(daily.loc[:d0], "ME")

    last_daily_idx = daily.index[daily_pos]
    partial_close = float(daily["close"].iloc[daily_pos])
    partial_open = float(monthly_partial["open"].iloc[daily_pos])
    partial_high = float(monthly_partial["high"].iloc[daily_pos])
    partial_low = float(monthly_partial["low"].iloc[daily_pos])
    cols: dict[str, list] = {
        "open": [partial_open],
        "high": [partial_high],
        "low": [partial_low],
        "close": [partial_close],
    }
    if "volume" in monthly_partial:
        cols["volume"] = [float(monthly_partial["volume"].iloc[daily_pos])]

    # Bucket label matches the 'ME' label of d0's month — same label the old path
    # would produce when resampling daily.loc[:d0]. When monthly_full doesn't
    # cover that month (d0 sits past the last sampled month-end), compute the
    # month-end timestamp directly.
    if month_idx < len(monthly_full):
        label = monthly_full.index[month_idx]
    else:
        label = pd.Timestamp(last_daily_idx).to_period("M").to_timestamp(how="end").normalize()

    partial_row = pd.DataFrame(cols, index=pd.DatetimeIndex([label]))
    # Re-align columns to monthly_full when available so concat preserves order.
    if len(monthly_full) > 0:
        partial_row = partial_row.reindex(columns=monthly_full.columns)
    complete = monthly_full.iloc[:month_idx]
    out = pd.concat([complete, partial_row])
    # Mirror the old path's `dropna(how="all")` — partial row always has data,
    # so this is a no-op in practice, kept for byte-compatibility.
    return out.dropna(how="all")


_EMPTY_TF = {
    "momentum": 0.0, "trend": 0.0, "rsi": 0.0,
    "breakout": 0.0, "breakout_thrust": 0.0, "volatility": 0.0,
    "atr_contraction": 0.0, "volume_dryup": 0.0,
    "bb_squeeze": 0.0, "gap": 0.0,
    "total": 0.0, "uptrend": False,
}


def _timeframe_score_at(pre: dict, pos: int, f: Formula) -> dict:
    """Mirror of `timeframe_score` reading from precomputed series at position `pos`.

    `pos` is the integer position in the precomputed series that corresponds
    to the evaluation timestamp d0 — see `_pos_at_daily` / `_pos_at_weekly`.
    """
    cfg = f.raw["indicators"]
    close = pre["close"]
    high = pre["high"]
    low = pre["low"]
    df = pre["df"]
    direction = f.direction

    if pos < 0 or pos >= len(close):
        return dict(_EMPTY_TF)

    # Mirror the adaptive caps from `timeframe_score`. Caps depend on n_avail
    # at d0, so when any cap is active for this position the precomputed
    # series (built with the full YAML period) cannot be reused — fall back
    # to per-d0 evaluation on the slice for bit-exact parity.
    n_avail = int(close.iloc[: pos + 1].notna().sum())
    local_sma_slow = min(cfg["sma_slow"], max(20, n_avail // 3))
    local_sma_fast = min(cfg["sma_fast"], max(5, n_avail // 6))
    local_mom_lb = min(cfg["momentum_lookback"], max(3, n_avail // 4))
    local_breakout_lb = min(cfg["breakout_lookback"], max(10, n_avail // 4))

    caps_active = (local_sma_slow != cfg["sma_slow"]
                   or local_sma_fast != cfg["sma_fast"]
                   or local_mom_lb != cfg["momentum_lookback"]
                   or local_breakout_lb != cfg["breakout_lookback"])
    if caps_active:
        # `pre["df"].iloc[:pos+1]` matches the slice the per-d0 path would see:
        #   daily:  df.loc[:d0]               (causal, no Sat/Sun in df anyway)
        #   weekly: resample(daily.loc[:d0])  (same bucket set for d0=Friday)
        return timeframe_score(df.iloc[: pos + 1], f)

    if n_avail < local_sma_slow + 1:
        return dict(_EMPTY_TF)

    rsi_v = pre["rsi"].iloc[pos]
    sf = pre["sma_fast"].iloc[pos]
    ss = pre["sma_slow"].iloc[pos]

    mom_lb = cfg["momentum_lookback"]
    mom_skip = int(cfg.get("momentum_skip_recent", 0) or 0)
    n_close_total = pos + 1  # equals len(close.loc[:d0]) in old path
    if mom_skip > 0 and n_close_total > mom_lb + mom_skip:
        end_val = float(close.iloc[pos - mom_skip])
        start_val = float(close.iloc[pos - mom_skip - mom_lb])
        mom = (end_val / start_val - 1) * 100 if start_val > 0 else float("nan")
    else:
        mom = pre["momentum"].iloc[pos]

    hi = pre["rolling_high"].iloc[pos]
    lo_n = pre["rolling_low"].iloc[pos]
    px = close.iloc[pos]

    mom_s = _clip01((mom + 10) / 60.0) if pd.notna(mom) else 0.0

    uptrend = pd.notna(sf) and pd.notna(ss) and sf > ss and px > sf and px > ss
    trend_s = 1.0 if uptrend else (0.5 if pd.notna(sf) and px > sf else 0.0)

    if direction == "reversion":
        mom_s = 1.0 - mom_s
        trend_s = 1.0 - trend_s

    lo, hicut = f.raw["rsi_band"]
    if pd.isna(rsi_v):
        rsi_s = 0.0
    elif lo <= rsi_v <= hicut:
        rsi_s = 1.0
    elif rsi_v < lo:
        rsi_s = _clip01(rsi_v / lo)
    else:
        rsi_s = _clip01((100 - rsi_v) / (100 - hicut))

    if direction == "reversion":
        if pd.notna(lo_n) and lo_n > 0:
            brk_s = _clip01(lo_n / px) if px >= lo_n else 1.0
        else:
            brk_s = 0.0
    else:
        if pd.notna(hi) and hi > 0:
            brk_s = _clip01(px / hi - 0.0) if px >= hi else _clip01(1 - (hi - px) / hi * 5)
        else:
            brk_s = 0.0

    t_lo = cfg.get("breakout_thrust_band_lo", 0.01)
    t_hi = cfg.get("breakout_thrust_band_hi", 0.03)
    t_fade = cfg.get("breakout_thrust_fade", 0.05)
    thrust_s = _breakout_thrust_score(px, hi, t_lo, t_hi, t_fade)

    vol = pre["stdev_returns"].iloc[pos]
    if pd.notna(vol):
        vol_s = _clip01((0.05 - vol) / 0.04)
    else:
        vol_s = 0.0

    atr_s = vdry_s = bbsq_s = 0.0
    if {"open", "high", "low", "close"}.issubset(df.columns):
        ap_lookback = cfg.get("atr_contraction_lookback", 20)
        ap_thresh = cfg.get("atr_contraction_threshold", 0.30)
        ap = pre.get("atr_pct")
        try:
            # Old path: `len(ap) > ap_lookback` on ap computed from slice of length pos+1.
            if ap is not None and (pos + 1) > ap_lookback:
                cur_v = ap.iloc[pos]
                prev_v = ap.iloc[pos - ap_lookback]
                if pd.notna(cur_v) and pd.notna(prev_v):
                    prev = float(prev_v)
                    cur = float(cur_v)
                    if prev > 0:
                        ratio = cur / prev
                        atr_s = _clip01((1.0 - ratio) / max(ap_thresh, 1e-6))
        except Exception:  # noqa: BLE001
            atr_s = 0.0

        v_thresh = cfg.get("vol_dryup_threshold", 0.80)
        if "volume" in df.columns:
            try:
                short_av_s = pre.get("avg_vol_short")
                long_av_s = pre.get("avg_vol_long")
                if short_av_s is not None and long_av_s is not None:
                    short_av = short_av_s.iloc[pos]
                    long_av = long_av_s.iloc[pos]
                    if pd.notna(short_av) and pd.notna(long_av) and long_av > 0:
                        ratio = float(short_av) / float(long_av)
                        vdry_s = _clip01((1.0 - ratio) / max(1.0 - v_thresh, 1e-6))
            except Exception:  # noqa: BLE001
                vdry_s = 0.0

        bb_lookback = cfg.get("bb_squeeze_lookback", 60)
        bb_pct = cfg.get("bb_squeeze_percentile", 0.20)
        try:
            bbw = pre.get("bb_width")
            if bbw is not None:
                # Old: bbw computed on close.loc[:d0] then dropna().iloc[-bb_lookback:]
                bbw_slice = bbw.iloc[: pos + 1].dropna()
                window = bbw_slice.iloc[-bb_lookback:]
                if len(window) >= max(10, bb_lookback // 2):
                    cur_w = float(window.iloc[-1])
                    rank_v = float((window <= cur_w).sum() - 1) / max(len(window) - 1, 1)
                    if rank_v <= bb_pct:
                        bbsq_s = 1.0
                    else:
                        bbsq_s = _clip01((1.0 - rank_v) / max(1.0 - bb_pct, 1e-6))
        except Exception:  # noqa: BLE001
            bbsq_s = 0.0

    # gap sub-score — mirror of the per-d0 path. `pre["gap"]` is the per-bar
    # up-gap series; take the trailing rolling max over gap_lookback bars. The
    # slice [start:pos+1] reproduces `gaps.iloc[-glb:]` on the length-(pos+1) slice.
    gap_s = 0.0
    gap_series = pre.get("gap")
    if gap_series is not None and pos >= 1:
        glb = int(cfg.get("gap_lookback", 10))
        g_lo = cfg.get("gap_band_lo", 0.02)
        g_hi = cfg.get("gap_band_hi", 0.05)
        g_fade = cfg.get("gap_fade", 0.12)
        start = max(0, pos - glb + 1)
        recent = gap_series.iloc[start: pos + 1].dropna()
        if not recent.empty:
            gap_s = _gap_score(float(recent.max()), g_lo, g_hi, g_fade)

    # Stage-2 trend gate (opt-in via YAML) — mirror of the per-d0 path.
    if cfg.get("trend_gate_quietness", False) and not (pd.notna(ss) and px > ss):
        atr_s = vdry_s = bbsq_s = 0.0

    w = _norm(f.raw["timeframe_score_weights"])
    total = (w.get("momentum", 0.0) * mom_s
             + w.get("trend", 0.0) * trend_s
             + w.get("rsi", 0.0) * rsi_s
             + w.get("breakout", 0.0) * brk_s
             + w.get("breakout_thrust", 0.0) * thrust_s
             + w.get("volatility", 0.0) * vol_s
             + w.get("atr_contraction", 0.0) * atr_s
             + w.get("volume_dryup", 0.0) * vdry_s
             + w.get("bb_squeeze", 0.0) * bbsq_s
             + w.get("gap", 0.0) * gap_s)
    return {"momentum": round(mom_s, 4), "trend": round(trend_s, 4),
            "rsi": round(rsi_s, 4), "breakout": round(brk_s, 4),
            "breakout_thrust": round(thrust_s, 4),
            "volatility": round(vol_s, 4),
            "atr_contraction": round(atr_s, 4),
            "volume_dryup": round(vdry_s, 4),
            "bb_squeeze": round(bbsq_s, 4),
            "gap": round(gap_s, 4),
            "total": round(total, 4), "uptrend": bool(uptrend)}


def score_ticker_at(precomputed: dict, d0: pd.Timestamp, f: Formula) -> dict:
    """Vectorized counterpart to `score_ticker`. Same return shape & math.

    Monthly is recomputed per d0 from the raw daily series — using a full-month
    resample would leak future bars when d0 falls inside a month.
    """
    daily_pre = precomputed["daily"]
    weekly_pre = precomputed["weekly"]
    daily_raw = precomputed["_daily_raw"]

    d_pos = _pos_at_daily(daily_pre["close"].index, d0)
    w_pos = _pos_at_weekly(weekly_pre["close"].index, d0)

    daily_sub = _timeframe_score_at(daily_pre, d_pos, f)
    weekly_sub = _timeframe_score_at(weekly_pre, w_pos, f)
    # Fast monthly: O(1) from precomputed cumulatives. Set USE_LEGACY_MONTHLY=1
    # to fall back to the per-d0 resample for debugging.
    if os.environ.get("USE_LEGACY_MONTHLY", "0") == "1":
        monthly_slice = ind.resample_ohlcv(daily_raw.loc[:d0], "ME")
    else:
        monthly_slice = _build_monthly_at_d0(precomputed, d0)
    monthly_sub = timeframe_score(monthly_slice, f)

    sub = {"daily": daily_sub, "weekly": weekly_sub, "monthly": monthly_sub}
    tw = _norm(f.raw["timeframe_weights"])
    base = sum(tw[tf] * sub[tf]["total"] for tf in tw)

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
