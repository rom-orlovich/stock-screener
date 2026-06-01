#!/usr/bin/env python3
"""Phase 1 CLI: scan (rank a universe today) or backtest (simulate history).

Examples:
    python run.py scan    --formula formulas/momentum_v1.yaml
    python run.py backtest --formula formulas/momentum_v1.yaml --start 2022-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine import backtest as bt
from engine.data import get_universe
from engine.score import Formula, rank
from engine.universe import get_universe_tickers, liquidity_filter

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

# Small liquid default universe for a fast, no-account Phase 1 run.
# Phase 4 replaces this with the full US-market list + liquidity prefilter.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO",
    "JPM", "V", "MA", "UNH", "XOM", "COST", "NFLX", "CRM", "ADBE", "PEP", "KO",
]


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_tickers(args) -> list[str]:
    """Resolve a named --universe (sp500/russell3000/russell1000) or --tickers."""
    if getattr(args, "universe", None):
        return get_universe_tickers(args.universe)
    return args.tickers


def cmd_scan(args):
    f = Formula.load(args.formula)
    tickers = _resolve_tickers(args)
    # ~6y so monthly resample has enough bars for sma_slow (50 monthly bars).
    start = (pd.Timestamp.today() - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    print(f"fetching {len(tickers)} tickers from {args.provider}...", flush=True)
    data = get_universe(tickers, start=start, provider=args.provider)
    if args.min_liquidity > 0:
        before = len(data)
        data = liquidity_filter(data, min_avg_dollar_vol=args.min_liquidity)
        print(f"liquidity filter: {before} -> {len(data)} tickers (>= ${args.min_liquidity:,.0f} avg daily $-vol)", flush=True)
    table = rank(data, f)
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"scan_{f.version}_{_stamp()}.csv"
    table.to_csv(out, index=False)
    print(f"\nScan — formula={f.version}  universe={len(data)} tickers")
    print(table.head(args.top).to_string(index=False))
    print(f"\nsaved -> {out}")


def cmd_backtest(args):
    f = Formula.load(args.formula)
    # Pull a little before start so indicators have warmup history.
    fetch_start = (pd.Timestamp(args.start) - pd.DateOffset(months=8)).strftime("%Y-%m-%d")
    tickers = _resolve_tickers(args)
    # Ensure benchmark is fetched alongside the universe so SPY equity is set.
    bench = args.benchmark
    if bench and bench not in tickers:
        tickers = list(tickers) + [bench]
    print(f"fetching {len(tickers)} tickers from {args.provider}...", flush=True)
    data = get_universe(tickers, start=fetch_start, end=args.end, provider=args.provider)
    if args.min_liquidity > 0:
        before = len(data)
        # Keep benchmark even if it would be filtered (it won't, but be safe).
        bench_df = data.get(bench)
        data = liquidity_filter(data, min_avg_dollar_vol=args.min_liquidity)
        if bench and bench_df is not None and bench not in data:
            data[bench] = bench_df
        print(f"liquidity filter: {before} -> {len(data)} tickers", flush=True)
    # Overlay any `backtest:` block in the formula YAML, then CLI flags win where
    # explicitly set (None = not set, so the YAML/dataclass default stands).
    cfg = bt.config_from_formula(
        f,
        top_n=args.top_n, rebalance=args.rebalance,
        stop_loss_pct=args.stop_loss, take_profit_pct=args.take_profit,
        trailing_stop_pct=args.trailing_stop,
        trailing_activate_pct=args.trailing_activate,
        atr_stop_mult=args.atr_stop_mult,
        atr_stop_period=args.atr_stop_period,
        time_stop_bars=args.time_stop_bars,
        benchmark_ticker=bench,
        mode=args.mode,
        intraweek_entry=args.intraweek_entry,
    )
    res = bt.run(data, f, start=args.start, end=args.end, cfg=cfg)

    RUNS.mkdir(exist_ok=True)
    stamp = _stamp()
    suffix = f"_{args.universe}" if args.universe else ""
    base = f"{f.version}{suffix}_{stamp}"
    res.trades.to_csv(RUNS / f"bt_trades_{base}.csv", index=False)
    res.equity.to_csv(RUNS / f"bt_equity_{base}.csv")
    if res.benchmark_equity is not None:
        res.benchmark_equity.to_csv(RUNS / f"bt_benchmark_{base}.csv")
    meta = {"formula": f.version, "start": args.start, "end": args.end,
            "universe_label": args.universe or "tickers",
            "config": vars(cfg), "stats": res.stats, "universe": len(data)}
    (RUNS / f"bt_summary_{base}.json").write_text(json.dumps(meta, indent=2))

    print(f"\nBacktest — formula={f.version}  {args.start} -> {args.end}  "
          f"universe={len(data)}  top_n={cfg.top_n}  rebalance={cfg.rebalance}")
    for k, v in res.stats.items():
        print(f"  {k:>14}: {v}")
    print(f"\nsaved -> runs/bt_summary_{base}.json")


def main():
    p = argparse.ArgumentParser(description="Stock screener Phase 1")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = dict()

    s = sub.add_parser("scan")
    s.add_argument("--formula", default="formulas/momentum_v1.yaml")
    s.add_argument("--provider", default="yf")
    s.add_argument("--tickers", nargs="*", default=DEFAULT_UNIVERSE)
    s.add_argument("--universe", default=None,
                   help="named universe (sp500/russell3000/russell1000); overrides --tickers")
    s.add_argument("--min-liquidity", type=float, default=0.0,
                   help="min 60d avg $-volume filter (0 = disabled)")
    s.add_argument("--top", type=int, default=15)
    s.set_defaults(func=cmd_scan)

    b = sub.add_parser("backtest")
    b.add_argument("--formula", default="formulas/momentum_v1.yaml")
    b.add_argument("--provider", default="yf")
    b.add_argument("--tickers", nargs="*", default=DEFAULT_UNIVERSE)
    b.add_argument("--universe", default=None,
                   help="named universe (sp500/russell3000/russell1000); overrides --tickers")
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--top-n", type=int, default=5)
    b.add_argument("--rebalance", default="W-FRI")
    b.add_argument("--mode", choices=("rebalance", "event", "managed"), default=None,
                   help="entry mode; overrides the formula's backtest.mode. "
                        "'managed' = decoupled bar-by-bar hold (positions persist "
                        "across rebalances, exits managed daily). Default: rebalance "
                        "unless the YAML sets one.")
    # Exit flags default to None so they don't clobber a formula's backtest:
    # block — pass one explicitly to override. Unset everywhere -> no stops.
    b.add_argument("--stop-loss", type=float, default=None,
                   help="exit if intra-period close falls this fraction below entry")
    b.add_argument("--take-profit", type=float, default=None)
    b.add_argument("--trailing-stop", type=float, default=None,
                   help="trailing stop fraction below running peak (0 = disabled)")
    b.add_argument("--trailing-activate", type=float, default=None,
                   help="activate trailing stop after this gain (default 5%%)")
    b.add_argument("--atr-stop-mult", type=float, default=None,
                   help="hard stop = entry - N*ATR (0 = disabled, overrides --stop-loss)")
    b.add_argument("--atr-stop-period", type=int, default=None)
    b.add_argument("--time-stop-bars", type=int, default=None,
                   help="exit after N daily bars since entry (0 = disabled)")
    b.add_argument("--intraweek-entry", action=argparse.BooleanOptionalAction, default=None,
                   help="managed mode only: ADDITIVE intra-week entries — also enter "
                        "on any daily bar whose breakout event fires, not just the "
                        "weekly anchor. Default (off) is bit-identical to weekly-only.")
    b.add_argument("--benchmark", default="SPY",
                   help="buy-and-hold benchmark ticker for alpha calc")
    b.add_argument("--min-liquidity", type=float, default=0.0,
                   help="min 60d avg $-volume filter (0 = disabled)")
    b.set_defaults(func=cmd_backtest)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
