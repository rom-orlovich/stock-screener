"""Cross-section vectorized scoring layer.

Goal: scale linearly in N_tickers without paying Python-per-ticker overhead.

Architecture
------------
* `precompute_universe(price_data, f)` aligns every ticker to a shared daily
  index and stacks the daily indicators into (N_tickers, N_dates) ndarrays.
  Heavier than the per-ticker precompute (one master reindex pass per
  ticker), but pays back at backtest time when the per-d0 cost drops from
  N * pandas_overhead to ~1 numpy op.

* `score_universe_at(universe_pre, d0, f) -> dict[ticker, score]`
  - For the daily timeframe, computes scores for every ticker in a single
    batched pass when the adaptive caps in `timeframe_score` are inactive
    for *all* tickers at d0. Each ticker that still has caps active falls
    back to per-ticker `score_ticker_at` so parity is preserved bit-exact.
  - Weekly + monthly stay on the per-ticker path (cap is essentially always
    active there with realistic data windows; cross-section would need
    a much bigger refactor to win). They reuse the per-ticker precompute
    that lives in `universe_pre["per_ticker"]`.

Gated by USE_CROSS_SECTION=1 in engine/backtest.py. Default OFF until the
sp500 parity test in scripts/parity_cross_section.py confirms zero drift.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

from . import indicators as ind  # noqa: F401  (kept for symmetry, used by score.py)
from .score import (  # noqa: F401
    Formula,
    _clip01,
    _norm,
    _pos_at_daily,
    _pos_at_weekly,
    _timeframe_score_at,
    _TT_HIGH_WITHIN,
    _TT_LONG_RISING_LB,
    _TT_LOW_ABOVE,
    precompute_indicators,
    score_ticker_at,
    timeframe_score,
    _build_monthly_at_d0,
)


# ---------------------------------------------------------------------------
# Universe precompute
# ---------------------------------------------------------------------------


def _to_array(series_or_none) -> np.ndarray | None:
    if series_or_none is None:
        return None
    return series_or_none.to_numpy(dtype=np.float64, copy=False)


def precompute_universe(price_data: Dict[str, pd.DataFrame], f: Formula) -> dict:
    """Build cross-section indicator matrices on a shared daily index.

    Returns a dict with:
      * tickers          — list of ticker symbols, the order used by every matrix
      * per_ticker       — per-ticker precompute (parity fallback + monthly path)
      * master_daily     — pd.DatetimeIndex shared across the daily matrices
      * daily_close, daily_high, daily_low — (N, D) ndarrays
      * daily_rsi, daily_sma_fast, daily_sma_slow,
        daily_momentum, daily_rolling_high, daily_rolling_low,
        daily_stdev_returns — (N, D) ndarrays (YAML-period precomputes)
      * daily_notna_cum  — (N, D) int ndarray, cumulative count of non-NaN closes
      * daily_cumsum_close — (N, D+1) float ndarray; cumsum prefixed with 0 so
        SMA-N at pos = (cs[pos+1] - cs[pos+1-N]) / N via O(1) gather
    """
    cfg = f.raw["indicators"]
    tickers = list(price_data.keys())
    per_ticker: dict[str, dict | None] = {}
    for tkr, df in price_data.items():
        if df is None or df.empty:
            per_ticker[tkr] = None
            continue
        try:
            per_ticker[tkr] = precompute_indicators(df, f)
        except Exception:  # noqa: BLE001 — one bad ticker shouldn't poison the rest
            per_ticker[tkr] = None

    valid = [tkr for tkr in tickers if per_ticker[tkr] is not None]
    if not valid:
        return {
            "tickers": tickers,
            "per_ticker": per_ticker,
            "master_daily": pd.DatetimeIndex([]),
            "daily": {},
        }

    # Master daily index = union of all valid tickers' daily indices.
    union: set = set()
    for tkr in valid:
        union.update(per_ticker[tkr]["daily"]["close"].index)
    master_daily = pd.DatetimeIndex(sorted(union))
    D = len(master_daily)
    N = len(tickers)

    keys = (
        "close", "high", "low",
        "rsi", "sma_fast", "sma_slow",
        "momentum", "rolling_high", "rolling_low", "stdev_returns",
        "tt_ma_s", "tt_ma_m", "tt_ma_l",
    )
    mats: dict[str, np.ndarray] = {k: np.full((N, D), np.nan, dtype=np.float64) for k in keys}

    # Optional OHLC-only indicators — leave NaN when the ticker lacks the source.
    has_ohlc = np.zeros(N, dtype=bool)
    atr_pct = np.full((N, D), np.nan, dtype=np.float64)
    gap = np.full((N, D), np.nan, dtype=np.float64)
    bb_width = np.full((N, D), np.nan, dtype=np.float64)
    has_volume = np.zeros(N, dtype=bool)
    avg_vol_short = np.full((N, D), np.nan, dtype=np.float64)
    avg_vol_long = np.full((N, D), np.nan, dtype=np.float64)

    for i, tkr in enumerate(tickers):
        pre = per_ticker[tkr]
        if pre is None:
            continue
        d = pre["daily"]
        for k in keys:
            s = d.get(k)
            if s is None:
                continue
            arr = s.reindex(master_daily).to_numpy(dtype=np.float64, copy=False)
            mats[k][i, :] = arr
        if "atr_pct" in d:
            has_ohlc[i] = True
            atr_pct[i, :] = d["atr_pct"].reindex(master_daily).to_numpy(dtype=np.float64, copy=False)
        if "gap" in d:
            gap[i, :] = d["gap"].reindex(master_daily).to_numpy(dtype=np.float64, copy=False)
        if "bb_width" in d:
            bb_width[i, :] = d["bb_width"].reindex(master_daily).to_numpy(dtype=np.float64, copy=False)
        if "avg_vol_short" in d and "avg_vol_long" in d:
            has_volume[i] = True
            avg_vol_short[i, :] = d["avg_vol_short"].reindex(master_daily).to_numpy(dtype=np.float64, copy=False)
            avg_vol_long[i, :] = d["avg_vol_long"].reindex(master_daily).to_numpy(dtype=np.float64, copy=False)

    notna_close = ~np.isnan(mats["close"])
    notna_cum = notna_close.cumsum(axis=1).astype(np.int32)
    # cumsum prefixed with a 0 column so SMA-N at pos = (cs[pos+1]-cs[pos+1-N])/N
    close_filled = np.where(notna_close, mats["close"], 0.0)
    cumsum_close = np.concatenate([np.zeros((N, 1)), close_filled.cumsum(axis=1)], axis=1)

    return {
        "tickers": tickers,
        "per_ticker": per_ticker,
        "master_daily": master_daily,
        "daily_close": mats["close"],
        "daily_high": mats["high"],
        "daily_low": mats["low"],
        "daily_rsi": mats["rsi"],
        "daily_sma_fast": mats["sma_fast"],
        "daily_sma_slow": mats["sma_slow"],
        "daily_momentum": mats["momentum"],
        "daily_rolling_high": mats["rolling_high"],
        "daily_rolling_low": mats["rolling_low"],
        "daily_stdev_returns": mats["stdev_returns"],
        "daily_tt_ma_s": mats["tt_ma_s"],
        "daily_tt_ma_m": mats["tt_ma_m"],
        "daily_tt_ma_l": mats["tt_ma_l"],
        "daily_atr_pct": atr_pct,
        "daily_gap": gap,
        "daily_bb_width": bb_width,
        "daily_avg_vol_short": avg_vol_short,
        "daily_avg_vol_long": avg_vol_long,
        "daily_has_ohlc": has_ohlc,
        "daily_has_volume": has_volume,
        "daily_notna_cum": notna_cum,
        "daily_cumsum_close": cumsum_close,
    }


# ---------------------------------------------------------------------------
# Vectorized daily timeframe score (caps-inactive subset only)
# ---------------------------------------------------------------------------


def _daily_score_vectorized(
    universe_pre: dict,
    d_pos: int,
    f: Formula,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the daily timeframe TOTAL and UPTREND arrays for the subset of
    tickers in `mask` using the YAML-period precomputed indicators.

    Returns (total[mask], uptrend[mask]) — same numbers `_timeframe_score_at`
    would return on each ticker individually, vectorized via numpy.
    """
    cfg = f.raw["indicators"]
    direction = f.direction

    close = universe_pre["daily_close"][mask, d_pos]
    rsi = universe_pre["daily_rsi"][mask, d_pos]
    sf = universe_pre["daily_sma_fast"][mask, d_pos]
    ss = universe_pre["daily_sma_slow"][mask, d_pos]
    hi = universe_pre["daily_rolling_high"][mask, d_pos]
    lo_n = universe_pre["daily_rolling_low"][mask, d_pos]
    vol = universe_pre["daily_stdev_returns"][mask, d_pos]

    # Momentum: honor momentum_skip_recent. cfg["momentum_lookback"] is the
    # period used on the precomputed (no cap by assumption here).
    mom_lb = cfg["momentum_lookback"]
    mom_skip = int(cfg.get("momentum_skip_recent", 0) or 0)
    if mom_skip > 0:
        # Per-ticker n_avail varies; the per-ticker path guards on
        # `len(close) > mom_lb + mom_skip`. With caps-inactive (n_avail >=
        # 3 * sma_slow) and a typical sma_slow >= 50, this guard is satisfied
        # so we can always compute the skip-recent momentum vectorized.
        close_full = universe_pre["daily_close"]
        end = close_full[mask, d_pos - mom_skip]
        start = close_full[mask, d_pos - mom_skip - mom_lb]
        with np.errstate(divide="ignore", invalid="ignore"):
            mom = np.where(start > 0, (end / start - 1.0) * 100.0, np.nan)
    else:
        mom = universe_pre["daily_momentum"][mask, d_pos]

    # mom_s
    mom_s = np.clip((mom + 10.0) / 60.0, 0.0, 1.0)
    mom_s = np.where(np.isnan(mom), 0.0, mom_s)

    # uptrend / trend_s
    finite_sf_ss = np.isfinite(sf) & np.isfinite(ss)
    uptrend = finite_sf_ss & (sf > ss) & (close > sf) & (close > ss)
    trend_s = np.where(uptrend, 1.0, np.where(np.isfinite(sf) & (close > sf), 0.5, 0.0))

    if direction == "reversion":
        mom_s = 1.0 - mom_s
        trend_s = 1.0 - trend_s

    # RSI band
    lo, hicut = f.raw["rsi_band"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi_low_branch = np.clip(rsi / float(lo) if lo else 0.0, 0.0, 1.0)
        rsi_high_branch = np.clip((100.0 - rsi) / (100.0 - float(hicut)), 0.0, 1.0)
    rsi_s = np.where(
        np.isnan(rsi), 0.0,
        np.where((rsi >= lo) & (rsi <= hicut), 1.0,
                 np.where(rsi < lo, rsi_low_branch, rsi_high_branch))
    )

    # Breakout / breakdown
    if direction == "reversion":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(close > 0, lo_n / close, 0.0)
        brk_s = np.where(
            ~np.isfinite(lo_n) | (lo_n <= 0), 0.0,
            np.where(close >= lo_n, np.clip(ratio, 0.0, 1.0), 1.0),
        )
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            near = 1.0 - np.where(hi > 0, (hi - close) / hi, 0.0) * 5.0
            push = np.where(hi > 0, close / hi, 0.0)
        brk_s = np.where(
            ~np.isfinite(hi) | (hi <= 0), 0.0,
            np.where(close >= hi, np.clip(push, 0.0, 1.0), np.clip(near, 0.0, 1.0)),
        )

    # breakout_thrust — vectorized mirror of score._breakout_thrust_score.
    t_lo = cfg.get("breakout_thrust_band_lo", 0.01)
    t_hi = cfg.get("breakout_thrust_band_hi", 0.03)
    t_fade = cfg.get("breakout_thrust_fade", 0.05)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(hi > 0, close / hi - 1.0, np.nan)
        ramp = np.clip(pct / t_lo, 0.0, 1.0)
        fade_v = np.clip((t_fade - pct) / (t_fade - t_hi), 0.0, 1.0)
    thrust_s = np.where(
        ~np.isfinite(hi) | (hi <= 0) | ~np.isfinite(pct) | (pct <= 0.0), 0.0,
        np.where(pct < t_lo, ramp,
                 np.where(pct <= t_hi, 1.0,
                          np.where(pct < t_fade, fade_v, 0.0))),
    )

    # Volatility
    vol_s = np.where(np.isnan(vol), 0.0, np.clip((0.05 - vol) / 0.04, 0.0, 1.0))

    # ATR contraction (only for tickers with OHLC)
    has_ohlc = universe_pre["daily_has_ohlc"][mask]
    atr_s = np.zeros_like(close)
    ap_lookback = cfg.get("atr_contraction_lookback", 20)
    ap_thresh = cfg.get("atr_contraction_threshold", 0.30)
    if d_pos > ap_lookback:
        cur_ap = universe_pre["daily_atr_pct"][mask, d_pos]
        prev_ap = universe_pre["daily_atr_pct"][mask, d_pos - ap_lookback]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(prev_ap > 0, cur_ap / prev_ap, np.nan)
        atr_s = np.where(
            has_ohlc & np.isfinite(cur_ap) & np.isfinite(prev_ap) & (prev_ap > 0),
            np.clip((1.0 - ratio) / max(ap_thresh, 1e-6), 0.0, 1.0),
            0.0,
        )

    # Volume dryup
    vdry_s = np.zeros_like(close)
    v_thresh = cfg.get("vol_dryup_threshold", 0.80)
    has_volume = universe_pre["daily_has_volume"][mask]
    short_av = universe_pre["daily_avg_vol_short"][mask, d_pos]
    long_av = universe_pre["daily_avg_vol_long"][mask, d_pos]
    with np.errstate(divide="ignore", invalid="ignore"):
        v_ratio = np.where(long_av > 0, short_av / long_av, np.nan)
    vdry_s = np.where(
        has_ohlc & has_volume & np.isfinite(short_av) & np.isfinite(long_av) & (long_av > 0),
        np.clip((1.0 - v_ratio) / max(1.0 - v_thresh, 1e-6), 0.0, 1.0),
        0.0,
    )

    # BB squeeze — per-ticker dropna+rank window. Compute only when weight > 0
    # in the YAML; otherwise it'd be wasted work and zero anyway.
    w = _norm(f.raw["timeframe_score_weights"])
    bbsq_s = np.zeros_like(close)
    if w.get("bb_squeeze", 0.0) > 0.0 and "daily_bb_width" in universe_pre:
        bb_lookback = cfg.get("bb_squeeze_lookback", 60)
        bb_pct = cfg.get("bb_squeeze_percentile", 0.20)
        bbw_mat = universe_pre["daily_bb_width"]
        # Per-ticker because dropna + ranking varies in length. Stay in numpy.
        idx_mask = np.where(mask)[0]
        for j, ti in enumerate(idx_mask):
            row = bbw_mat[ti, : d_pos + 1]
            row = row[~np.isnan(row)]
            if len(row) == 0:
                continue
            window = row[-bb_lookback:]
            if len(window) < max(10, bb_lookback // 2):
                continue
            cur_w = float(window[-1])
            rank = float((window <= cur_w).sum() - 1) / max(len(window) - 1, 1)
            if rank <= bb_pct:
                bbsq_s[j] = 1.0
            else:
                bbsq_s[j] = _clip01((1.0 - rank) / max(1.0 - bb_pct, 1e-6))

    # gap — vectorized mirror of score._gap_score: trailing rolling max of the
    # per-bar up-gap over gap_lookback bars, then the triangle band. gap survives
    # the trend gate (it's a confirmation, not a quietness term).
    gap_s = np.zeros_like(close)
    if w.get("gap", 0.0) > 0.0 and "daily_gap" in universe_pre:
        glb = int(cfg.get("gap_lookback", 10))
        g_lo = cfg.get("gap_band_lo", 0.02)
        g_hi = cfg.get("gap_band_hi", 0.05)
        g_fade = cfg.get("gap_fade", 0.12)
        start = max(0, d_pos - glb + 1)
        gw = universe_pre["daily_gap"][mask, start: d_pos + 1]  # (Nmask, win)
        gmax = np.nanmax(np.where(np.isnan(gw), -np.inf, gw), axis=1)
        gmax = np.where(np.isneginf(gmax), np.nan, gmax)
        with np.errstate(divide="ignore", invalid="ignore"):
            ramp = np.clip((gmax - g_lo) / max(g_hi - g_lo, 1e-9), 0.0, 1.0)
            fade_v = np.clip((g_fade - gmax) / max(g_fade - g_hi, 1e-9), 0.0, 1.0)
        gap_s = np.where(
            ~np.isfinite(gmax) | (gmax <= 0.0) | (gmax < g_lo), 0.0,
            np.where(gmax < g_hi, ramp, np.where(gmax < g_fade, fade_v, 0.0)),
        )

    # high52_proximity — George & Hwang nearness to the 52-week high, vectorized
    # mirror of score._high52_proximity_score (`hi` == breakout_lookback rolling
    # high == 252 == 52 weeks in leading_stock_v1). Not a quietness term — it
    # survives the trend gate.
    high52_s = np.zeros_like(close)
    if w.get("high52_proximity", 0.0) > 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            prox = np.where(hi > 0, close / hi, 0.0)
        high52_s = np.where(np.isfinite(hi) & (hi > 0), np.clip(prox, 0.0, 1.0), 0.0)

    # trend_template — Minervini 8-rule fraction, vectorized mirror of
    # score._trend_template_score. mom is the raw momentum (RS proxy, rule 8).
    tt_s = np.zeros_like(close)
    if w.get("trend_template", 0.0) > 0.0:
        ma_s_v = universe_pre["daily_tt_ma_s"][mask, d_pos]
        ma_m_v = universe_pre["daily_tt_ma_m"][mask, d_pos]
        ma_l_v = universe_pre["daily_tt_ma_l"][mask, d_pos]
        if d_pos >= _TT_LONG_RISING_LB:
            ma_l_prev_v = universe_pre["daily_tt_ma_l"][mask, d_pos - _TT_LONG_RISING_LB]
        else:
            ma_l_prev_v = np.full_like(ma_l_v, np.nan)
        r1 = np.isfinite(ma_s_v) & (close > ma_s_v)
        r2 = np.isfinite(ma_m_v) & (close > ma_m_v)
        r3 = np.isfinite(ma_l_v) & (close > ma_l_v)
        r4 = (np.isfinite(ma_s_v) & np.isfinite(ma_m_v) & np.isfinite(ma_l_v)
              & (ma_s_v > ma_m_v) & (ma_m_v > ma_l_v))
        r5 = np.isfinite(ma_l_v) & np.isfinite(ma_l_prev_v) & (ma_l_v > ma_l_prev_v)
        r6 = np.isfinite(lo_n) & (lo_n > 0) & (close >= lo_n * (1.0 + _TT_LOW_ABOVE))
        r7 = np.isfinite(hi) & (hi > 0) & (close >= hi * (1.0 - _TT_HIGH_WITHIN))
        r8 = np.isfinite(mom) & (mom > 0.0)
        tt_s = (r1.astype(np.float64) + r2 + r3 + r4 + r5 + r6 + r7 + r8) / 8.0

    # Stage-2 trend gate (opt-in via YAML) — mirror of the per-d0 path: the
    # quietness terms only count when price sits above the slow SMA.
    if cfg.get("trend_gate_quietness", False):
        gate = np.isfinite(ss) & (close > ss)
        atr_s = np.where(gate, atr_s, 0.0)
        vdry_s = np.where(gate, vdry_s, 0.0)
        bbsq_s = np.where(gate, bbsq_s, 0.0)

    # Match the per-ticker path: weighted sum of UNROUNDED sub-scores, then
    # round only the total. Pre-rounding each sub-score before summing would
    # introduce ~5e-5 drift per term — caught by parity_cross_section.py.
    total = (w.get("momentum", 0.0) * mom_s
             + w.get("trend", 0.0) * trend_s
             + w.get("rsi", 0.0) * rsi_s
             + w.get("breakout", 0.0) * brk_s
             + w.get("breakout_thrust", 0.0) * thrust_s
             + w.get("volatility", 0.0) * vol_s
             + w.get("atr_contraction", 0.0) * atr_s
             + w.get("volume_dryup", 0.0) * vdry_s
             + w.get("bb_squeeze", 0.0) * bbsq_s
             + w.get("gap", 0.0) * gap_s
             + w.get("high52_proximity", 0.0) * high52_s
             + w.get("trend_template", 0.0) * tt_s)

    return np.round(total, 4), uptrend


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_universe_at(universe_pre: dict, d0, f: Formula) -> Dict[str, float]:
    """Return {ticker: score} for every ticker in universe_pre.

    Splits the universe into:
      * caps-inactive subset (daily fast path eligible) → vectorized daily.
      * caps-active subset → per-ticker `score_ticker_at` fallback.
    Weekly + monthly always run per-ticker (cap-active for realistic windows).
    """
    tickers = universe_pre["tickers"]
    per_ticker = universe_pre["per_ticker"]

    md = universe_pre.get("master_daily")
    if md is None or len(md) == 0:
        return {tkr: _safe_score(per_ticker.get(tkr), d0, f) for tkr in tickers}

    master_daily = universe_pre["master_daily"]
    d_pos = _pos_at_daily(master_daily, d0)
    if d_pos < 0:
        return {tkr: _safe_score(per_ticker.get(tkr), d0, f) for tkr in tickers}

    cfg = f.raw["indicators"]
    n_avail = universe_pre["daily_notna_cum"][:, d_pos]
    # Caps-inactive = none of the four adaptive caps would clip below the YAML
    # value. Same formula as `_timeframe_score_at` (score.py).
    cap_thresh_slow = cfg["sma_slow"] * 3
    cap_thresh_fast = cfg["sma_fast"] * 6
    cap_thresh_mom = cfg["momentum_lookback"] * 4
    cap_thresh_brk = cfg["breakout_lookback"] * 4
    inactive_mask = (
        (n_avail >= cap_thresh_slow)
        & (n_avail >= cap_thresh_fast)
        & (n_avail >= cap_thresh_mom)
        & (n_avail >= cap_thresh_brk)
        & (n_avail >= cfg["sma_slow"] + 1)
    )
    # Skip tickers without a usable per_ticker (None precompute) — they fall
    # back below to a 0.0 score via the safe wrapper.
    has_pre = np.array([per_ticker.get(tkr) is not None for tkr in tickers])
    fast_mask = inactive_mask & has_pre

    scores: Dict[str, float] = {}

    if fast_mask.any():
        # Vectorized daily for the fast subset.
        daily_total, daily_uptrend = _daily_score_vectorized(universe_pre, d_pos, f, fast_mask)
        fast_idx = np.where(fast_mask)[0]
        tw = _norm(f.raw["timeframe_weights"])
        # Weekly + monthly stay per-ticker. Daily is taken straight from the
        # vectorized arrays — DO NOT call score_ticker_at here, or daily ends
        # up being computed twice per ticker and the whole speedup vanishes.
        for j, i in enumerate(fast_idx):
            tkr = tickers[i]
            pre = per_ticker[tkr]
            weekly_pre = pre["weekly"]
            w_pos = _pos_at_weekly(weekly_pre["close"].index, d0)
            weekly_sub = _timeframe_score_at(weekly_pre, w_pos, f)
            monthly_slice = _build_monthly_at_d0(pre, d0)
            monthly_sub = timeframe_score(monthly_slice, f)

            daily_t = float(daily_total[j])
            daily_up = bool(daily_uptrend[j])
            sub = {
                "daily": {"total": daily_t, "uptrend": daily_up},
                "weekly": weekly_sub,
                "monthly": monthly_sub,
            }
            base = (tw.get("daily", 0.0) * daily_t
                    + tw.get("weekly", 0.0) * weekly_sub["total"]
                    + tw.get("monthly", 0.0) * monthly_sub["total"])
            if f.direction == "reversion":
                aligned = not (daily_up or weekly_sub["uptrend"] or monthly_sub["uptrend"])
            else:
                aligned = daily_up and weekly_sub["uptrend"] and monthly_sub["uptrend"]
            bonus = f.raw.get("alignment_bonus", 0.0) if aligned else 0.0
            final = round(_clip01(base + bonus), 4)
            scores[tkr] = final

    # Caps-active or precompute-missing tickers: per-ticker fallback.
    slow_idx = np.where(~fast_mask)[0]
    for i in slow_idx:
        tkr = tickers[i]
        scores[tkr] = _safe_score(per_ticker.get(tkr), d0, f)
    return scores


def _safe_score(pre, d0, f: Formula) -> float:
    if pre is None:
        return 0.0
    try:
        return score_ticker_at(pre, d0, f)["score"]
    except Exception:  # noqa: BLE001
        return 0.0
