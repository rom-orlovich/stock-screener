#!/usr/bin/env python3
"""Compare momentum / mean-reversion / quality strategies vs SPY on S&P 500.

Window:       2022-01-01 -> 2024-12-31
Universe:     S&P 500 (cached) + SPY benchmark
Rebalance:    weekly (W-FRI), equal-weight top 20
Risk exits:   stop_loss = 8%, take_profit = 20%

Each formula's backtest is run in a fresh child process to avoid OOM on
constrained hosts (~500 tickers x 156 weekly rebalances per formula). The
parent process reads each child's JSON output and writes runs/comparison.md.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine import backtest as bt
from engine.data import get_universe
from engine.score import Formula
from engine.universe import sp500

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
TMP_DIR = RUNS / "_compare_tmp"

START = "2022-01-01"
END = "2024-12-31"
TOP_N = 20
REBALANCE = "W-FRI"
STOP_LOSS = 0.08
TAKE_PROFIT = 0.20
BENCHMARK = "SPY"
FETCH_START = "2021-05-01"  # 8 months of warmup, matches existing cache key

FORMULAS = [
    ("momentum_v1", ROOT / "formulas" / "momentum_v1.yaml"),
    ("mean_reversion_v1", ROOT / "formulas" / "mean_reversion_v1.yaml"),
    ("quality_v1", ROOT / "formulas" / "quality_v1.yaml"),
]


def _pct(x):
    return f"{x * 100:+.2f}%" if x is not None else "n/a"


def _num(x, nd: int = 2):
    return f"{x:.{nd}f}" if x is not None else "n/a"


def _run_one(formula_name: str, formula_path: Path) -> dict:
    """Child-process entry: load data, run one backtest, return stats dict."""
    print(f"[child:{formula_name}] loading universe...", flush=True)
    tickers = sp500()
    if BENCHMARK not in tickers:
        tickers = tickers + [BENCHMARK]
    data = get_universe(tickers, start=FETCH_START, end=END, provider="yf")
    print(f"[child:{formula_name}] loaded {len(data)} tickers "
          f"(SPY present: {BENCHMARK in data})", flush=True)
    if BENCHMARK not in data:
        raise SystemExit(f"benchmark {BENCHMARK} not loaded")

    f = Formula.load(formula_path)
    cfg = bt.BacktestConfig(
        top_n=TOP_N, rebalance=REBALANCE,
        stop_loss_pct=STOP_LOSS, take_profit_pct=TAKE_PROFIT,
        benchmark_ticker=BENCHMARK,
    )
    print(f"[child:{formula_name}] backtesting...", flush=True)
    res = bt.run(data, f, start=START, end=END, cfg=cfg)
    stats = dict(res.stats)
    stats["formula"] = formula_name
    stats["universe_size"] = len(data)
    # Free the heavy stuff before returning.
    del data, res
    gc.collect()
    return stats


def _spawn_child(formula_name: str, formula_path: Path, out_json: Path) -> int:
    """Spawn ourselves as a child to run a single formula."""
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--child", "--formula-name", formula_name,
           "--formula-path", str(formula_path),
           "--out-json", str(out_json)]
    env = os.environ.copy()
    print(f"[parent] spawning child for {formula_name}...", flush=True)
    return subprocess.call(cmd, env=env)


def _child_main(formula_name: str, formula_path: Path, out_json: Path) -> int:
    stats = _run_one(formula_name, formula_path)
    out_json.write_text(json.dumps(stats, indent=2, default=str))
    print(f"[child:{formula_name}] wrote {out_json}", flush=True)
    return 0


def _parent_main() -> int:
    print(f"compare.py — {START} -> {END}  top_n={TOP_N}  rebalance={REBALANCE}", flush=True)
    print(f"  stop_loss={STOP_LOSS:.0%}  take_profit={TAKE_PROFIT:.0%}  benchmark={BENCHMARK}",
          flush=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name, fpath in FORMULAS:
        out_json = TMP_DIR / f"{name}.json"
        rc = _spawn_child(name, fpath, out_json)
        if rc != 0 or not out_json.exists():
            print(f"[parent] child for {name} failed (rc={rc}) — skipping", file=sys.stderr)
            continue
        stats = json.loads(out_json.read_text())
        print(f"[parent] {name}: {stats}", flush=True)
        rows.append(stats)

    if not rows:
        print("[parent] no formulas completed — aborting", file=sys.stderr)
        return 1

    bench_total = next((r.get("benchmark_total_return") for r in rows
                        if r.get("benchmark_total_return") is not None), None)
    universe_size = next((r.get("universe_size") for r in rows), None)

    RUNS.mkdir(exist_ok=True)
    out_path = RUNS / "comparison.md"
    lines: list[str] = []
    lines.append("# Strategy comparison")
    lines.append("")
    lines.append(f"- Window: **{START} -> {END}**")
    if universe_size is not None:
        lines.append(f"- Universe: S&P 500 ({universe_size} tickers loaded incl. {BENCHMARK})")
    lines.append(f"- Rebalance: `{REBALANCE}`, top-N **{TOP_N}**, equal-weight")
    lines.append(f"- Risk exits: stop-loss **{STOP_LOSS:.0%}**, take-profit **{TAKE_PROFIT:.0%}**")
    lines.append(f"- Benchmark: **{BENCHMARK}** buy-and-hold = {_pct(bench_total)}")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("| Formula | Total return | Sharpe | Max DD | Win rate | Alpha vs SPY | # trades |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r['formula']}` "
            f"| {_pct(r.get('total_return'))} "
            f"| {_num(r.get('sharpe'))} "
            f"| {_pct(r.get('max_drawdown'))} "
            f"| {_pct(r.get('win_rate'))} "
            f"| {_pct(r.get('alpha_vs_benchmark'))} "
            f"| {r.get('n_trades')} |"
        )
    lines.append(
        f"| `SPY buy-and-hold` | {_pct(bench_total)} | n/a | n/a | n/a | +0.00% | n/a |"
    )
    lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}", flush=True)
    # Echo the markdown for log capture
    print("\n--- runs/comparison.md ---")
    print(out_path.read_text())
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true")
    p.add_argument("--formula-name")
    p.add_argument("--formula-path")
    p.add_argument("--out-json")
    args = p.parse_args()
    if args.child:
        return _child_main(args.formula_name, Path(args.formula_path), Path(args.out_json))
    return _parent_main()


if __name__ == "__main__":
    sys.exit(main())
