#!/usr/bin/env python3
"""Deep critical analysis of every strategy from runs/bt_*_*.{csv,json}.

Honest answer to: how trustworthy are these numbers, what biases exist,
how predictive is the score, and where would I lose money in production.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
STAMP_RE = re.compile(r"_(\d{8}_\d{6})\.json$")


def _latest_summary_per_formula() -> dict[str, Path]:
    """Latest bt_summary per formula (by stamp in filename)."""
    by_formula: dict[str, tuple[str, Path]] = {}
    for fp in RUNS.glob("bt_summary_*.json"):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        name = data.get("formula")
        if not name:
            continue
        m = STAMP_RE.search(fp.name)
        stamp = m.group(1) if m else ""
        if name not in by_formula or stamp > by_formula[name][0]:
            by_formula[name] = (stamp, fp)
    return {n: p for n, (_, p) in by_formula.items()}


def _trades_for(summary_fp: Path) -> pd.DataFrame:
    base = summary_fp.stem.replace("bt_summary_", "")
    fp = RUNS / f"bt_trades_{base}.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, parse_dates=["enter", "exit_date"])
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce")
    return df


def _equity_for(summary_fp: Path) -> pd.Series:
    base = summary_fp.stem.replace("bt_summary_", "")
    fp = RUNS / f"bt_equity_{base}.csv"
    if not fp.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(fp, parse_dates=[0])
    df.columns = [c.lower() for c in df.columns]
    s = df.set_index(df.columns[0])[df.columns[1]] if len(df.columns) >= 2 else pd.Series(dtype=float)
    return s


def _benchmark_for(summary_fp: Path) -> pd.Series:
    base = summary_fp.stem.replace("bt_summary_", "")
    fp = RUNS / f"bt_benchmark_{base}.csv"
    if not fp.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(fp, parse_dates=[0])
    df.columns = [c.lower() for c in df.columns]
    s = df.set_index(df.columns[0])[df.columns[1]] if len(df.columns) >= 2 else pd.Series(dtype=float)
    return s


def trade_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {}
    rets = t["ret"].dropna()
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf")
    return {
        "n_trades": int(len(rets)),
        "win_rate": round(float((rets > 0).mean()), 4),
        "avg_win_pct": round(float(wins.mean() * 100), 3) if len(wins) else 0.0,
        "avg_loss_pct": round(float(losses.mean() * 100), 3) if len(losses) else 0.0,
        "profit_factor": round(float(profit_factor), 3),
        "expectancy_pct": round(float(rets.mean() * 100), 4),
        "ret_std_pct": round(float(rets.std() * 100), 3),
        "skew": round(float(rets.skew()), 3),
        "kurt": round(float(rets.kurt()), 3),
        "best_trade_pct": round(float(rets.max() * 100), 2),
        "worst_trade_pct": round(float(rets.min() * 100), 2),
        "max_consecutive_losses": _max_streak(rets <= 0),
        "max_consecutive_wins": _max_streak(rets > 0),
    }


def _max_streak(mask: pd.Series) -> int:
    """Longest consecutive True run."""
    if mask.empty:
        return 0
    run = best = 0
    for v in mask.values:
        if v:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def score_predictiveness(t: pd.DataFrame) -> dict:
    """Does a higher score predict a better realised return? This is the
    central question — a strategy with strong Sharpe but corr(score, ret) ≈ 0
    is winning by luck, not by signal."""
    if t.empty or t["score"].nunique() < 5:
        return {}
    valid = t.dropna(subset=["score", "ret"])
    if len(valid) < 30:
        return {}
    pearson = float(valid["score"].corr(valid["ret"]))
    spearman = float(valid["score"].rank().corr(valid["ret"].rank()))
    # Decile analysis: sort by score, group into 10 buckets, mean return per bucket.
    valid = valid.copy()
    valid["bucket"] = pd.qcut(valid["score"], 10, labels=False, duplicates="drop")
    decile_ret = valid.groupby("bucket")["ret"].mean()
    top_minus_bottom = float(decile_ret.iloc[-1] - decile_ret.iloc[0]) if len(decile_ret) >= 2 else 0.0
    return {
        "corr_pearson": round(pearson, 4),
        "corr_spearman": round(spearman, 4),
        "decile_top_pct": round(float(decile_ret.iloc[-1] * 100), 3) if len(decile_ret) else None,
        "decile_bottom_pct": round(float(decile_ret.iloc[0] * 100), 3) if len(decile_ret) else None,
        "top_minus_bottom_decile_pct": round(top_minus_bottom * 100, 3),
        "monotonicity_corr": round(float(np.corrcoef(np.arange(len(decile_ret)), decile_ret.values)[0, 1]), 3) if len(decile_ret) >= 3 else None,
    }


def annual_breakdown(eq: pd.Series, bench: pd.Series) -> list[dict]:
    if eq.empty:
        return []
    years = []
    for year, grp in eq.groupby(eq.index.year):
        if len(grp) < 2:
            continue
        first, last = grp.iloc[0], grp.iloc[-1]
        ret = float(last / first - 1.0)
        dd = float((grp / grp.cummax() - 1.0).min())
        bench_grp = bench[bench.index.year == year] if not bench.empty else pd.Series(dtype=float)
        bench_ret = float(bench_grp.iloc[-1] / bench_grp.iloc[0] - 1.0) if len(bench_grp) >= 2 else None
        years.append({
            "year": int(year),
            "return_pct": round(ret * 100, 2),
            "max_dd_pct": round(dd * 100, 2),
            "bench_return_pct": round(bench_ret * 100, 2) if bench_ret is not None else None,
            "alpha_pct": round((ret - bench_ret) * 100, 2) if bench_ret is not None else None,
        })
    return years


def concentration(t: pd.DataFrame, n: int = 10) -> dict:
    if t.empty:
        return {}
    pnl_by_ticker = t.groupby("ticker")["ret"].sum().sort_values(ascending=False)
    total = float(pnl_by_ticker.sum())
    if total == 0:
        return {}
    top_n_share = float(pnl_by_ticker.head(n).sum() / total) if total != 0 else 0.0
    # HHI on positive contributors (concentration of the winning side)
    pos = pnl_by_ticker[pnl_by_ticker > 0]
    pos_share = pos / pos.sum() if pos.sum() > 0 else pd.Series(dtype=float)
    hhi = float((pos_share ** 2).sum()) if len(pos_share) else 0.0
    return {
        "top_10_tickers_share_of_pnl": round(top_n_share, 4),
        "n_winning_tickers": int((pnl_by_ticker > 0).sum()),
        "n_losing_tickers": int((pnl_by_ticker < 0).sum()),
        "winning_concentration_hhi": round(hhi, 4),
        "top_5_contributors": [(str(t), round(float(v), 3)) for t, v in pnl_by_ticker.head(5).items()],
        "bottom_5_contributors": [(str(t), round(float(v), 3)) for t, v in pnl_by_ticker.tail(5).items()],
    }


def trustworthiness_flags(stats: dict, ts: dict, pred: dict, conc: dict, annual: list[dict]) -> list[str]:
    """A short list of red/yellow flags the user should consider."""
    flags: list[str] = []
    sharpe = stats.get("sharpe") or 0
    n_trades = ts.get("n_trades") or 0
    if n_trades < 100:
        flags.append("LOW sample size (<100 trades) — wide confidence intervals")
    pearson = pred.get("corr_pearson")
    if pearson is not None and abs(pearson) < 0.03:
        flags.append(f"WEAK score→return correlation ({pearson}) — strategy may be winning by luck, not skill")
    elif pearson is not None and pearson < 0:
        flags.append(f"NEGATIVE score→return correlation ({pearson}) — the signal is INVERTED, top picks lose")
    pf = ts.get("profit_factor")
    if pf is not None and pf < 1.2:
        flags.append(f"LOW profit factor ({pf}) — wins barely cover losses")
    avg_win = ts.get("avg_win_pct") or 0
    avg_loss = ts.get("avg_loss_pct") or 0
    if avg_win and avg_loss and abs(avg_loss) > avg_win * 1.5:
        flags.append("ASYMMETRIC payoff — avg loss > 1.5× avg win (small-edge mean reversion?)")
    skew = ts.get("skew")
    if skew is not None and skew < -1:
        flags.append(f"NEGATIVE skew ({skew}) — left-tail-heavy, beware tail risk")
    top_share = conc.get("top_10_tickers_share_of_pnl") or 0
    if top_share > 0.6:
        flags.append(f"HIGH concentration: top 10 tickers contribute {top_share*100:.0f}% of P&L — strategy survives on a few names")
    monotonic = pred.get("monotonicity_corr")
    if monotonic is not None and monotonic < 0.3:
        flags.append(f"NON-MONOTONIC deciles ({monotonic}) — score doesn't cleanly rank reality")
    # Year stability
    if annual:
        years_positive = sum(1 for y in annual if y.get("return_pct", 0) > 0)
        if years_positive / len(annual) < 0.5:
            flags.append(f"YEAR STABILITY: only {years_positive}/{len(annual)} years positive")
        alphas = [y.get("alpha_pct") for y in annual if y.get("alpha_pct") is not None]
        if alphas:
            best_alpha = max(alphas)
            if sum(alphas) > 0 and best_alpha / sum(alphas) > 0.7:
                flags.append(f"SINGLE-YEAR alpha: one year contributes {best_alpha/sum(alphas)*100:.0f}% of total alpha — fragile to regime shift")
    return flags


def main():
    formulas = _latest_summary_per_formula()
    if not formulas:
        print("no bt_summary files found")
        return
    print(f"=== Analyzing {len(formulas)} strategies @ {datetime.now().isoformat(timespec='seconds')} ===\n")
    out = {}
    for name in sorted(formulas):
        fp = formulas[name]
        summary = json.loads(fp.read_text())
        stats = summary.get("stats", {})
        trades = _trades_for(fp)
        eq = _equity_for(fp)
        bench = _benchmark_for(fp)
        ts = trade_stats(trades)
        pred = score_predictiveness(trades)
        annual = annual_breakdown(eq, bench)
        conc = concentration(trades)
        flags = trustworthiness_flags(stats, ts, pred, conc, annual)
        out[name] = {
            "window": [summary.get("start"), summary.get("end")],
            "universe_size": summary.get("universe"),
            "stats": stats,
            "trade_stats": ts,
            "predictiveness": pred,
            "concentration": conc,
            "annual": annual,
            "trustworthiness_flags": flags,
        }

    (RUNS / "analysis.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {RUNS / 'analysis.json'}")

    # Pretty summary table
    print("\n" + "=" * 110)
    print(f"{'Strategy':<32} {'Sharpe':>7} {'Return':>9} {'DD':>8} {'WinR':>7} {'PF':>6} {'corr':>7} {'top-bot':>8} {'flags'}")
    print("=" * 110)
    for name, d in sorted(out.items(), key=lambda kv: -(kv[1]["stats"].get("sharpe") or 0)):
        s = d["stats"]
        ts_ = d["trade_stats"]
        p = d["predictiveness"]
        nflags = len(d["trustworthiness_flags"])
        print(f"{name:<32} {s.get('sharpe','—'):>7} {(s.get('total_return') or 0)*100:>8.1f}% "
              f"{(s.get('max_drawdown') or 0)*100:>7.1f}% {(s.get('win_rate') or 0)*100:>6.1f}% "
              f"{ts_.get('profit_factor','—'):>6} "
              f"{p.get('corr_pearson','—'):>7} "
              f"{(p.get('top_minus_bottom_decile_pct') or 0):>7.2f}% "
              f"{'⚠' * min(nflags, 5)}")
    print("\nFull details: runs/analysis.json")


if __name__ == "__main__":
    main()
