#!/usr/bin/env python3
"""Generate a live, self-refreshing single-page dashboard from runs/ artifacts.

Reads:
  - runs/bt_summary_*.json     (latest per formula)
  - runs/bt_equity_*.csv       (equity curves)
  - runs/bt_benchmark_*.csv    (benchmark equity)
  - runs/bt_trades_*.csv       (per-trade detail, latest per formula)
  - runs/auto_tune.csv         (every tuning trial)
  - runs/auto_tune_state.json  (cooldowns, drift counters, baselines)

Writes:
  - docs/data.json   (the data the page reads, polled every N seconds)
  - docs/index.html  (loads data.json via fetch, re-renders live)

Run once with `python dashboard.py`; a cron (every 5 min) keeps data.json fresh.
Serve with `python -m http.server -d docs 8000` (or any static host).
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
DOCS = ROOT / "docs"
FORMULAS = ROOT / "formulas"

STAMP_RE = re.compile(r"_(\d{8}_\d{6})\.json$")
BAK_STAMP_RE = re.compile(r"\.bak_(\d{8}_\d{6})\.yaml$")
TRADES_PER_STRATEGY = 300  # cap to keep payload small but still useful


def _stamp_from_name(name: str) -> str:
    m = STAMP_RE.search(name)
    return m.group(1) if m else ""


def _read_summary(fp: Path) -> dict:
    try:
        return json.loads(fp.read_text())
    except Exception:
        return {}


def _read_equity(fp: Path) -> list[dict]:
    rows = []
    try:
        with fp.open() as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                if len(r) < 2:
                    continue
                try:
                    rows.append({"date": r[0][:10], "value": float(r[1])})
                except ValueError:
                    continue
    except Exception:
        pass
    return rows


def _read_trades(fp: Path) -> list[dict]:
    rows = []
    try:
        with fp.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
    except Exception:
        pass
    return rows


def _trade_summary(trades: list[dict]) -> dict:
    if not trades:
        return {"recent": [], "top_wins": [], "top_losses": [], "by_exit": {}}

    # parse rets once
    norm = []
    for t in trades:
        try:
            ret = float(t.get("ret", 0))
        except ValueError:
            ret = 0.0
        norm.append({
            "enter": (t.get("enter") or "")[:10],
            "exit_date": (t.get("exit_date") or "")[:10],
            "ticker": t.get("ticker", ""),
            "ret": ret,
            "exit": t.get("exit", ""),
        })

    # recent N (last by exit_date)
    norm.sort(key=lambda r: (r["exit_date"] or r["enter"]), reverse=True)
    recent = norm[:TRADES_PER_STRATEGY]

    # top winners / losers (across full set)
    wins = sorted(norm, key=lambda r: r["ret"], reverse=True)[:20]
    losses = sorted(norm, key=lambda r: r["ret"])[:20]

    by_exit: dict[str, int] = defaultdict(int)
    for r in norm:
        by_exit[r["exit"] or "hold"] += 1

    return {
        "recent": recent,
        "top_wins": wins,
        "top_losses": losses,
        "by_exit": dict(by_exit),
    }


def _index_summaries() -> dict[str, list[tuple[Path, dict]]]:
    """Group every bt_summary_*.json by the formula name *inside* the JSON,
    not by the filename — old filenames lack the universe suffix and would
    otherwise create phantom duplicate rows for the same formula."""
    by_formula: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for fp in RUNS.glob("bt_summary_*.json"):
        summary = _read_summary(fp)
        if not summary:
            continue
        name = summary.get("formula")
        if not name:
            # fallback: derive from filename stem
            stem = fp.stem.replace("bt_summary_", "")
            name = stem.rsplit("_", 2)[0]
        by_formula[name].append((fp, summary))
    for name in by_formula:
        by_formula[name].sort(key=lambda x: _stamp_from_name(x[0].name), reverse=True)
    return by_formula


def collect_strategies(indexed: dict | None = None) -> list[dict]:
    if indexed is None:
        indexed = _index_summaries()

    out = []
    for formula_key, items in indexed.items():
        latest, summary = items[0]
        files = [p for p, _ in items]
        if not summary:
            continue
        stamp = _stamp_from_name(latest.name)
        base = latest.stem.replace("bt_summary_", "")
        equity_fp = RUNS / f"bt_equity_{base}.csv"
        bench_fp = RUNS / f"bt_benchmark_{base}.csv"
        trades_fp = RUNS / f"bt_trades_{base}.csv"

        eq = _read_equity(equity_fp) if equity_fp.exists() else []
        bench = _read_equity(bench_fp) if bench_fp.exists() else []
        trades = _read_trades(trades_fp) if trades_fp.exists() else []
        tsum = _trade_summary(trades)

        stats = summary.get("stats", {}) or {}
        cfg = summary.get("config", {}) or {}
        out.append({
            "formula": summary.get("formula", formula_key),
            "universe": summary.get("universe_label", ""),
            "start": summary.get("start", ""),
            "end": summary.get("end", ""),
            "stamp": stamp,
            "stats": stats,
            "config": cfg,
            "equity": eq,
            "benchmark": bench,
            "exits": tsum["by_exit"],
            "trades_recent": tsum["recent"],
            "trades_top_wins": tsum["top_wins"],
            "trades_top_losses": tsum["top_losses"],
            "trades_total": len(trades),
            "history_count": len(files),
        })

    def composite(s):
        sh = s["stats"].get("sharpe") or 0
        dd = abs(s["stats"].get("max_drawdown") or 0)
        return sh - dd
    out.sort(key=composite, reverse=True)
    return out


def collect_auto_tune() -> list[dict]:
    fp = RUNS / "auto_tune.csv"
    if not fp.exists():
        return []
    rows = []
    try:
        with fp.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
    except Exception:
        return []
    rows.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return rows


def collect_state() -> dict:
    fp = RUNS / "auto_tune_state.json"
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text())
    except Exception:
        return {}


def _yaml_params(raw: dict) -> dict:
    """Flatten the tunable bits of a formula YAML into one dict for diffing.
    We only show keys that auto_tune treats as tunable (declared in `bounds`)
    plus the timeframe_score_weights and indicators sections."""
    flat = {}
    for k, v in (raw.get("indicators") or {}).items():
        flat[f"indicators.{k}"] = v
    for k, v in (raw.get("timeframe_score_weights") or {}).items():
        flat[f"timeframe_score_weights.{k}"] = v
    for k, v in raw.items():
        if k in ("indicators", "timeframe_score_weights", "bounds"):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            flat[k] = v
    return flat


def _load_yaml_snapshots(formula_basename: str) -> list[dict]:
    """Return YAML snapshots over time (oldest → newest), each a flat param dict.
    Sources: every formulas/<basename>.bak_*.yaml backup + the current file."""
    snaps = []
    base = FORMULAS / f"{formula_basename}.yaml"
    if not base.exists():
        return snaps
    # backups: formulas/<base>.bak_YYYYMMDD_HHMMSS.yaml
    baks = sorted(FORMULAS.glob(f"{formula_basename}.bak_*.yaml"),
                  key=lambda p: BAK_STAMP_RE.search(p.name).group(1) if BAK_STAMP_RE.search(p.name) else "")
    for p in baks:
        m = BAK_STAMP_RE.search(p.name)
        try:
            raw = yaml.safe_load(p.read_text()) or {}
        except Exception:
            continue
        snaps.append({
            "stamp": m.group(1) if m else "",
            "source": p.name,
            "params": _yaml_params(raw),
        })
    try:
        cur_raw = yaml.safe_load(base.read_text()) or {}
        snaps.append({
            "stamp": "current",
            "source": base.name,
            "params": _yaml_params(cur_raw),
        })
    except Exception:
        pass
    return snaps


def _param_diffs(snaps: list[dict]) -> list[dict]:
    """For each consecutive pair, what changed."""
    diffs = []
    for i in range(1, len(snaps)):
        prev = snaps[i - 1]["params"]
        cur = snaps[i]["params"]
        changes = []
        keys = sorted(set(prev) | set(cur))
        for k in keys:
            a, b = prev.get(k), cur.get(k)
            if a != b:
                changes.append({"key": k, "from": a, "to": b})
        if changes:
            diffs.append({
                "from_stamp": snaps[i - 1]["stamp"],
                "to_stamp": snaps[i]["stamp"],
                "changes": changes,
            })
    return diffs


def _picks_by_date(trades: list[dict]) -> list[dict]:
    """Group trades by rebalance day (`enter`), keep the top-N highest-scoring
    picks per day with their realised P&L and exit reason. Sorted newest day
    first so the latest week's picks land at the top of the UI."""
    by_day: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        d = (t.get("enter") or "")[:10]
        if not d:
            continue
        try:
            sc = float(t.get("score") or 0)
        except ValueError:
            sc = 0.0
        try:
            ret = float(t.get("ret") or 0)
        except ValueError:
            ret = 0.0
        by_day[d].append({
            "ticker": t.get("ticker", ""),
            "score": sc,
            "ret": ret,
            "exit": t.get("exit", ""),
            "exit_date": (t.get("exit_date") or "")[:10],
        })
    out = []
    for d in sorted(by_day.keys(), reverse=True):
        picks = sorted(by_day[d], key=lambda r: r["score"], reverse=True)
        out.append({
            "date": d,
            "picks": picks,
            "n": len(picks),
            "avg_ret": round(sum(p["ret"] for p in picks) / len(picks), 4) if picks else 0.0,
        })
    return out


def collect_strategy_histories(indexed: dict) -> dict[str, dict]:
    """For every formula seen in bt_summaries, build a full history payload."""
    # auto_tune trials keyed by formula filename (e.g. "momentum_v1.yaml")
    trials_by_formula: dict[str, list[dict]] = defaultdict(list)
    fp = RUNS / "auto_tune.csv"
    if fp.exists():
        try:
            with fp.open() as f:
                for r in csv.DictReader(f):
                    trials_by_formula[r.get("formula", "")].append(r)
        except Exception:
            pass
    for k in trials_by_formula:
        trials_by_formula[k].sort(key=lambda r: r.get("ts", ""), reverse=True)

    state = collect_state()
    state_formulas = (state or {}).get("formulas", {}) or {}

    out: dict[str, dict] = {}
    for formula, items in indexed.items():
        # Daily picks come from the LATEST run's trades CSV (operationally the
        # "what should I buy this week" view).
        latest_fp, _ = items[0]
        latest_base = latest_fp.stem.replace("bt_summary_", "")
        latest_trades_fp = RUNS / f"bt_trades_{latest_base}.csv"
        latest_trades = _read_trades(latest_trades_fp) if latest_trades_fp.exists() else []
        picks_by_date = _picks_by_date(latest_trades)

        # timeline of backtest runs (oldest → newest for chart left-to-right)
        timeline = []
        for fp_summary, summary in sorted(items, key=lambda x: _stamp_from_name(x[0].name)):
            stamp = _stamp_from_name(fp_summary.name)
            stats = summary.get("stats", {}) or {}
            timeline.append({
                "stamp": stamp,
                "iso": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}",
                "start": summary.get("start", ""),
                "end": summary.get("end", ""),
                "universe": summary.get("universe_label", ""),
                "total_return": stats.get("total_return"),
                "sharpe": stats.get("sharpe"),
                "max_drawdown": stats.get("max_drawdown"),
                "win_rate": stats.get("win_rate"),
                "alpha_vs_benchmark": stats.get("alpha_vs_benchmark"),
                "n_trades": stats.get("n_trades"),
            })

        # state per formula is keyed by "<basename>.yaml"
        basename = formula  # e.g. "momentum_v1"
        state_key = f"{basename}.yaml"
        fs = state_formulas.get(state_key, {}) or {}
        change_log = fs.get("change_log", []) or []
        monthly_baseline = fs.get("monthly_baseline") or {}

        snaps = _load_yaml_snapshots(basename)
        diffs = _param_diffs(snaps)

        out[basename] = {
            "formula": basename,
            "timeline": timeline,
            "change_log": change_log,
            "monthly_baseline": monthly_baseline,
            "trials": trials_by_formula.get(state_key, []),
            "yaml_snapshots": snaps,
            "param_diffs": diffs,
            "picks_by_date": picks_by_date,
            "picks_source": latest_trades_fp.name if latest_trades_fp.exists() else None,
        }
    return out


def collect_regime_matrix() -> dict:
    """Load runs/regime_matrix.json if present, else return an empty shell.
    Also re-derives the 'best strategy per regime' ranking so the dashboard
    doesn't have to compute it in JS."""
    fp = RUNS / "regime_matrix.json"
    if not fp.exists():
        return {}
    try:
        payload = json.loads(fp.read_text())
    except Exception:
        return {}
    matrix = payload.get("matrix") or {}
    regimes = payload.get("regimes") or []
    # For each regime, rank formulas by composite (sharpe - |dd|).
    best_per_regime: dict[str, list[dict]] = {}
    for r in regimes:
        rn = r["name"]
        scored = []
        for formula, by_regime in matrix.items():
            cell = by_regime.get(rn) or {}
            if "error" in cell:
                continue
            sh = cell.get("sharpe") or 0
            dd = abs(cell.get("max_drawdown") or 0)
            scored.append({
                "formula": formula,
                "composite": round(sh - dd, 3),
                "sharpe": sh,
                "total_return": cell.get("total_return"),
                "max_drawdown": cell.get("max_drawdown"),
                "alpha_vs_benchmark": cell.get("alpha_vs_benchmark"),
            })
        scored.sort(key=lambda x: x["composite"], reverse=True)
        best_per_regime[rn] = scored
    payload["best_per_regime"] = best_per_regime
    return payload


def collect_window_audit(strategies: list[dict]) -> dict:
    """Flag if strategies use different backtest windows (cherry-picking risk)."""
    windows = {(s.get("start", ""), s.get("end", "")) for s in strategies if s.get("start")}
    return {
        "distinct_windows": sorted(list(windows)),
        "uniform": len(windows) <= 1,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>stock-screener — live dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root { color-scheme: dark; }
  body { background: #0b0f1a; color: #d6dae3; font-family: ui-sans-serif, system-ui, -apple-system; }
  .card { background: #131826; border: 1px solid #1f2638; border-radius: 14px; }
  .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
  .pill-green { background: #0f3a25; color: #5fe6a2; }
  .pill-red   { background: #3a0f1a; color: #ff7a8a; }
  .pill-gray  { background: #1f2638; color: #9aa4bb; }
  .pill-amber { background: #3a2c08; color: #d4a408; }
  .num { font-variant-numeric: tabular-nums; }
  .leader { box-shadow: 0 0 0 2px #d4a40855 inset; }
  details > summary { cursor: pointer; list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  table.compact td, table.compact th { padding: 4px 8px; }
  /* Keep the Strategy column visible when the leaderboard overflows on mobile */
  table.sticky-first th:first-child,
  table.sticky-first td:first-child {
    position: sticky;
    left: 0;
    background: #131826;
    z-index: 2;
    box-shadow: 1px 0 0 #1f2638;
    min-width: 11rem;
  }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
  /* HARD CAP chart container so canvas can't grow forever */
  .chart-box { position: relative; height: 240px; max-height: 240px; width: 100%; }
  .trades-box { max-height: 320px; overflow-y: auto; }
  .live-dot { display:inline-block; width:8px; height:8px; border-radius:999px; background:#5fe6a2; box-shadow:0 0 0 0 rgba(95,230,162,.6); animation: pulse 2s infinite; vertical-align: middle; }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(95,230,162,.55); }
    70%  { box-shadow: 0 0 0 10px rgba(95,230,162,0); }
    100% { box-shadow: 0 0 0 0 rgba(95,230,162,0); }
  }
</style>
</head>
<body class="min-h-screen">
<header class="max-w-6xl mx-auto px-4 pt-8 pb-4">
  <div class="flex items-center justify-between flex-wrap gap-3">
    <div>
      <h1 class="text-3xl font-bold tracking-tight">stock-screener</h1>
      <p class="text-slate-400 mt-1">Live transparency: every backtest, every tuning trial, every exit reason, every trade.</p>
    </div>
    <div class="text-xs text-slate-400 text-right">
      <div><span class="live-dot"></span> <span id="live-label">LIVE</span> · refreshing every <span id="poll-secs">15</span>s</div>
      <div class="text-slate-500">data.json regenerated by cron · last <span id="gen-ts">—</span></div>
      <div class="text-slate-500">next refresh in <span id="countdown">15</span>s</div>
    </div>
  </div>
  <p class="text-xs text-slate-500 mt-3"><a class="underline hover:text-slate-300" href="https://github.com/rom-orlovich/stock-screener" target="_blank">repo on GitHub</a></p>
</header>

<main class="max-w-6xl mx-auto px-4 pb-16">

  <section id="window-audit" class="card p-4 mb-4 hidden">
    <div class="text-sm font-semibold mb-1">⚠️ Backtest windows are not uniform</div>
    <div id="window-audit-list" class="text-xs text-slate-400"></div>
  </section>

  <section class="card p-5 mb-6">
    <h2 class="text-lg font-semibold mb-3">Run it yourself</h2>
    <pre class="bg-black/40 border border-white/5 rounded-lg p-3 text-sm overflow-x-auto"><code># Backtest the leader
python run.py backtest \
  --formula formulas/dual_momentum_v1.yaml \
  --universe sp500 --start 2022-01-01 --end 2026-05-26 --top-n 20

# Regenerate this dashboard (data.json is what the page polls)
python dashboard.py && python -m http.server -d docs 8000</code></pre>
  </section>

  <section class="card p-5 mb-6">
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-lg font-semibold">Strategy leaderboard</h2>
      <span class="text-xs text-slate-500">Sorted by composite = sharpe − |dd|</span>
    </div>
    <div class="overflow-x-auto">
      <table class="w-full text-sm compact num sticky-first">
        <thead class="text-slate-400 text-left border-b border-white/5">
          <tr>
            <th>Strategy</th>
            <th class="text-right">Return</th>
            <th class="text-right">Sharpe</th>
            <th class="text-right">Max DD</th>
            <th class="text-right">Win%</th>
            <th class="text-right">Trades</th>
            <th class="text-right">vs SPY</th>
            <th class="text-right">Window</th>
          </tr>
        </thead>
        <tbody id="leaderboard"></tbody>
      </table>
    </div>
  </section>

  <section id="regime-card" class="card p-5 mb-6 hidden">
    <div class="flex items-baseline justify-between flex-wrap gap-3">
      <div>
        <h2 class="text-lg font-semibold">Regime matrix — strategy × market period</h2>
        <p class="text-xs text-slate-500 mt-1">Each cell is one backtest of that strategy over that historical regime. Best Sharpe per column highlighted.</p>
      </div>
      <div class="text-xs text-slate-400 text-right">
        <div>generated <span id="regime-gen">—</span></div>
        <div>Run <code>./backtest_regimes.sh</code> to refresh.</div>
      </div>
    </div>

    <div id="current-regime-banner" class="card p-3 mt-3 hidden" style="background:#1a2030">
      <div class="flex items-center justify-between flex-wrap gap-3">
        <div>
          <div class="text-xs uppercase text-slate-400">Current market regime</div>
          <div class="text-xl font-semibold mt-1"><span id="cur-state">—</span>
            <span class="pill pill-gray ml-2" id="cur-asof">as of —</span></div>
          <div class="text-xs text-slate-400 mt-1" id="cur-rationale">—</div>
        </div>
        <div>
          <div class="text-xs text-slate-400">Top strategy for this regime</div>
          <div class="text-xl font-semibold text-emerald-400 mt-1" id="cur-top">—</div>
          <div class="text-xs text-slate-500" id="cur-top-stats">—</div>
        </div>
      </div>
    </div>

    <div class="flex items-center gap-3 mt-4 flex-wrap text-xs">
      <span class="text-slate-400">Metric:</span>
      <select id="regime-metric" class="bg-[#0b0f1a] border border-white/10 px-2 py-1 rounded">
        <option value="sharpe">Sharpe</option>
        <option value="total_return">Total return</option>
        <option value="max_drawdown">Max DD</option>
        <option value="alpha_vs_benchmark">α vs SPY</option>
      </select>
    </div>

    <div class="overflow-x-auto mt-3">
      <table class="w-full text-sm compact num sticky-first" id="regime-table">
        <thead class="text-slate-400 text-left border-b border-white/5">
          <tr id="regime-head"></tr>
        </thead>
        <tbody id="regime-rows"></tbody>
      </table>
    </div>

    <details class="mt-4">
      <summary class="text-sm text-sky-300 hover:underline">Best strategy per regime (ranked)</summary>
      <div id="regime-rankings" class="grid-2 mt-3"></div>
    </details>
  </section>

  <section class="space-y-4" id="strategy-cards"></section>

  <section class="card p-5 mt-6">
    <h2 class="text-lg font-semibold mb-3">Auto-tune activity (latest 30 trials)</h2>
    <p class="text-xs text-slate-500 mb-3">Every nightly mutation — accepted or rejected — logged with the walk-forward result.</p>
    <div class="overflow-x-auto">
      <table class="w-full text-sm compact num">
        <thead class="text-slate-400 text-left border-b border-white/5">
          <tr>
            <th>When</th><th>Formula</th><th>Param</th><th>Change</th>
            <th class="text-right">Folds passed</th><th class="text-right">Score</th>
            <th class="text-right">Baseline</th><th>Result</th>
          </tr>
        </thead>
        <tbody id="autotune-log"></tbody>
      </table>
    </div>
  </section>

  <section class="card p-5 mt-6">
    <h2 class="text-lg font-semibold mb-3">Active guardrails</h2>
    <p class="text-xs text-slate-500 mb-3">Cooldowns, frozen params and monthly baselines per strategy.</p>
    <div id="state-blocks" class="grid-2"></div>
  </section>

</main>

<script>
const POLL_SECS = 15;
document.getElementById('poll-secs').textContent = POLL_SECS;

const fmtPct = v => (v === null || v === undefined) ? '—' : (Number(v) * 100).toFixed(2) + '%';
const fmtPctSigned = v => (v === null || v === undefined) ? '—' : (Number(v) >= 0 ? '+' : '') + (Number(v) * 100).toFixed(2) + '%';
const fmtNum = (v, d=2) => (v === null || v === undefined) ? '—' : Number(v).toFixed(d);

const charts = {};
let lastSig = '';
let countdown = POLL_SECS;

function composite(s) {
  const sh = (s.stats && s.stats.sharpe) || 0;
  const dd = Math.abs((s.stats && s.stats.max_drawdown) || 0);
  return sh - dd;
}

function renderLeaderboard(data) {
  const lb = document.getElementById('leaderboard');
  lb.innerHTML = '';
  data.strategies.forEach((s, i) => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-white/5 ' + (i === 0 ? 'leader' : '');
    const ret = s.stats.total_return;
    const alpha = s.stats.alpha_vs_benchmark;
    tr.innerHTML = `
      <td class="py-2"><a href="#${s.formula}" class="text-sky-300 hover:underline">${s.formula}</a>
        <a href="strategy.html?f=${s.formula}" class="text-xs text-slate-400 hover:text-sky-300 ml-2" title="full history + param diffs">history →</a>
        ${i === 0 ? '<span class="pill pill-amber" style="margin-left:6px">leader</span>' : ''}
        ${s.universe ? `<span class="pill pill-gray" style="margin-left:4px">${s.universe}</span>` : ''}</td>
      <td class="text-right ${ret > 0 ? 'text-emerald-400' : 'text-rose-400'}">${fmtPct(ret)}</td>
      <td class="text-right">${fmtNum(s.stats.sharpe)}</td>
      <td class="text-right text-rose-400">${fmtPct(s.stats.max_drawdown)}</td>
      <td class="text-right">${fmtPct(s.stats.win_rate)}</td>
      <td class="text-right text-slate-400">${s.stats.n_trades || s.trades_total || 0}</td>
      <td class="text-right ${alpha > 0 ? 'text-emerald-400' : 'text-rose-400'}">${fmtPct(alpha)}</td>
      <td class="text-right text-xs text-slate-500">${s.start || '—'}<br>${s.end || '—'}</td>`;
    lb.appendChild(tr);
  });
}

function tradeRowHTML(t) {
  const pos = t.ret >= 0;
  return `<tr class="border-b border-white/5">
    <td class="text-xs text-slate-400">${t.exit_date || ''}</td>
    <td class="text-xs">${t.ticker || ''}</td>
    <td class="text-xs text-slate-500">${t.enter || ''}</td>
    <td class="text-right text-xs ${pos ? 'text-emerald-400' : 'text-rose-400'}">${fmtPctSigned(t.ret)}</td>
    <td class="text-xs"><span class="pill pill-gray">${t.exit || 'hold'}</span></td></tr>`;
}

function renderStrategies(data) {
  const cards = document.getElementById('strategy-cards');
  cards.innerHTML = '';
  data.strategies.forEach((s, idx) => {
    const card = document.createElement('section');
    card.id = s.formula;
    card.className = 'card p-5';
    const exitsTxt = Object.entries(s.exits || {}).map(([k,v]) =>
      `<span class="pill pill-gray" style="margin-right:6px">${k}: ${v}</span>`).join('');
    const recent = (s.trades_recent || []).slice(0, 100);
    const wins = (s.trades_top_wins || []).slice(0, 10);
    const losses = (s.trades_top_losses || []).slice(0, 10);
    card.innerHTML = `
      <div class="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 class="text-xl font-semibold">${s.formula}</h3>
          <p class="text-xs text-slate-500">${s.start} → ${s.end} · ${s.universe || 'tickers'} · last run ${s.stamp || '—'}</p>
        </div>
        <div class="text-right text-sm">
          <div>Composite <span class="font-semibold">${fmtNum(composite(s))}</span></div>
          <div class="text-xs text-slate-500">${s.history_count} historical run${s.history_count===1?'':'s'} · ${s.trades_total || 0} trades total</div>
        </div>
      </div>

      <div class="grid-2 mt-4">
        <div>
          <div class="text-xs text-slate-400 mb-1">Equity vs SPY (base 1.0)</div>
          <div class="chart-box"><canvas id="chart-${idx}"></canvas></div>
        </div>
        <div class="text-sm space-y-1">
          <div class="flex justify-between"><span class="text-slate-400">Total return</span><span class="num">${fmtPct(s.stats.total_return)}</span></div>
          <div class="flex justify-between"><span class="text-slate-400">Sharpe</span><span class="num">${fmtNum(s.stats.sharpe)}</span></div>
          <div class="flex justify-between"><span class="text-slate-400">Max drawdown</span><span class="num text-rose-400">${fmtPct(s.stats.max_drawdown)}</span></div>
          <div class="flex justify-between"><span class="text-slate-400">Win rate</span><span class="num">${fmtPct(s.stats.win_rate)}</span></div>
          <div class="flex justify-between"><span class="text-slate-400">Trades</span><span class="num">${s.stats.n_trades || s.trades_total || 0}</span></div>
          <div class="flex justify-between"><span class="text-slate-400">Rebalances</span><span class="num">${s.stats.n_rebalances || 0}</span></div>
          <div class="flex justify-between"><span class="text-slate-400">Benchmark return</span><span class="num">${fmtPct(s.stats.benchmark_total_return)}</span></div>
          <div class="flex justify-between"><span class="text-slate-400">Alpha vs SPY</span><span class="num ${s.stats.alpha_vs_benchmark > 0 ? 'text-emerald-400' : 'text-rose-400'}">${fmtPct(s.stats.alpha_vs_benchmark)}</span></div>
          <div class="mt-3 text-xs text-slate-400">Exit breakdown</div>
          <div>${exitsTxt || '<span class="text-slate-500 text-xs">no trades yet</span>'}</div>
        </div>
      </div>

      <details class="mt-4" ${recent.length ? '' : 'hidden'}>
        <summary class="text-sm text-sky-300 hover:underline">Recent trades (last ${recent.length})</summary>
        <div class="trades-box mt-2 border border-white/5 rounded">
          <table class="w-full text-sm compact">
            <thead class="text-slate-400 text-left border-b border-white/5 sticky top-0 bg-[#131826]">
              <tr><th>Exit</th><th>Ticker</th><th>Enter</th><th class="text-right">PnL</th><th>Reason</th></tr>
            </thead>
            <tbody>${recent.map(tradeRowHTML).join('')}</tbody>
          </table>
        </div>
      </details>

      <details class="mt-2" ${wins.length ? '' : 'hidden'}>
        <summary class="text-sm text-sky-300 hover:underline">Top 10 wins / losses</summary>
        <div class="grid-2 mt-2">
          <div>
            <div class="text-xs text-emerald-400 mb-1">Best 10</div>
            <table class="w-full text-sm compact"><tbody>${wins.map(tradeRowHTML).join('')}</tbody></table>
          </div>
          <div>
            <div class="text-xs text-rose-400 mb-1">Worst 10</div>
            <table class="w-full text-sm compact"><tbody>${losses.map(tradeRowHTML).join('')}</tbody></table>
          </div>
        </div>
      </details>

      <details class="mt-2">
        <summary class="text-sm text-sky-300 hover:underline">Backtest config used</summary>
        <pre class="bg-black/40 border border-white/5 rounded p-3 text-xs mt-2 overflow-x-auto">${JSON.stringify(s.config, null, 2)}</pre>
      </details>`;
    cards.appendChild(card);

    if (charts[idx]) { try { charts[idx].destroy(); } catch(e){} }
    const ctx = card.querySelector(`#chart-${idx}`).getContext('2d');
    const eqLabels = s.equity.map(p => p.date);
    const eqData = s.equity.map(p => p.value);
    const bench = s.benchmark || [];
    const benchMap = new Map(bench.map(p => [p.date, p.value]));
    const benchData = eqLabels.map(d => benchMap.get(d) ?? null);
    charts[idx] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: eqLabels,
        datasets: [
          { label: s.formula, data: eqData, borderColor: '#60a5fa', backgroundColor: 'rgba(96,165,250,0.1)', tension: 0.2, pointRadius: 0, borderWidth: 2, fill: true },
          { label: 'SPY', data: benchData, borderColor: '#94a3b8', borderDash:[4,4], pointRadius: 0, borderWidth: 1.5 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { labels: { color: '#94a3b8' }}},
        scales: {
          x: { ticks: { color: '#64748b', maxTicksLimit: 6 }, grid: { color: 'rgba(255,255,255,0.04)' }},
          y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.04)' }},
        }
      }
    });
  });
}

function renderAutoTune(data) {
  const at = document.getElementById('autotune-log');
  at.innerHTML = '';
  (data.auto_tune || []).slice(0, 30).forEach(r => {
    const tr = document.createElement('tr');
    const accepted = String(r.accepted).toLowerCase() === 'true';
    tr.className = 'border-b border-white/5';
    tr.innerHTML = `
      <td class="py-1.5 text-xs text-slate-400">${r.ts || ''}</td>
      <td class="text-xs">${r.formula || ''}</td>
      <td class="text-xs">${r.param || ''}</td>
      <td class="text-xs">${r.old} → ${r.new} <span class="text-slate-500">(${r.direction || ''})</span></td>
      <td class="text-right text-xs">${r.folds_passed || 0}</td>
      <td class="text-right text-xs">${fmtNum(r.avg_composite || r.score, 3)}</td>
      <td class="text-right text-xs text-slate-400">${fmtNum(r.baseline_avg || r.baseline_score, 3)}</td>
      <td><span class="pill ${accepted ? 'pill-green' : 'pill-red'}">${accepted ? 'accepted' : 'rejected'}</span></td>`;
    at.appendChild(tr);
  });
  if (!data.auto_tune || data.auto_tune.length === 0) {
    at.innerHTML = '<tr><td colspan="8" class="text-slate-500 text-xs py-3">No auto-tune trials yet. The nightly cron runs at 03:00 Israel time.</td></tr>';
  }
}

function renderState(data) {
  const sb = document.getElementById('state-blocks');
  sb.innerHTML = '';
  const byFormula = (data.state && data.state.formulas) || {};
  const now = new Date();
  Object.entries(byFormula).forEach(([name, fs]) => {
    const cd = fs.cooldown_until || {};
    const fz = fs.frozen_until || {};
    const am = fs.anti_mirror || {};
    const cl = fs.change_log || [];
    const mb = fs.monthly_baseline || {};
    const active = obj => Object.entries(obj).filter(([k,v]) => v && new Date(v.until || v) > now);
    const cdRows = active(cd);
    const fzRows = active(fz);
    const amRows = Object.entries(am).filter(([k,v]) => v && v.until && new Date(v.until) > now);
    const div = document.createElement('div');
    div.className = 'card p-3';
    div.innerHTML = `
      <div class="text-sm font-semibold mb-2">${name}</div>
      <div class="text-xs text-slate-400">Baseline month: <span class="text-slate-200">${mb.month || '—'}</span> · composite avg: <span class="num">${fmtNum(mb.composite_avg, 3)}</span></div>
      <div class="text-xs text-slate-400 mt-1">Accepted changes (log): <span class="text-slate-200">${cl.length}</span></div>
      <div class="mt-2 text-xs">${cdRows.length === 0 ? '<span class="text-slate-500">No active cooldowns</span>' : cdRows.map(([k,_]) => `<span class="pill pill-gray" style="margin-right:4px">${k}</span>`).join('')}</div>
      ${fzRows.length ? `<div class="mt-1 text-xs">Frozen: ${fzRows.map(([k]) => `<span class="pill pill-red" style="margin-right:4px">${k}</span>`).join('')}</div>` : ''}
      ${amRows.length ? `<div class="mt-1 text-xs">Anti-mirror: ${amRows.map(([k,v]) => `<span class="pill pill-gray" style="margin-right:4px">${k}→${v.direction}</span>`).join('')}</div>` : ''}`;
    sb.appendChild(div);
  });
  if (Object.keys(byFormula).length === 0) {
    sb.innerHTML = '<div class="text-slate-500 text-xs col-span-2">No state yet. The first nightly run will lock the monthly baselines.</div>';
  }
}

function renderWindowAudit(data) {
  const wa = data.window_audit || {};
  const banner = document.getElementById('window-audit');
  const list = document.getElementById('window-audit-list');
  if (!wa.uniform && (wa.distinct_windows || []).length > 1) {
    banner.classList.remove('hidden');
    list.innerHTML = 'Distinct windows in play: ' + wa.distinct_windows.map(w => `<code>${w[0]} → ${w[1]}</code>`).join(' · ') + ' — re-run all formulas on the same window for fair comparison.';
  } else {
    banner.classList.add('hidden');
  }
}

function render(data) {
  document.getElementById('gen-ts').textContent = data.generated_at;
  renderWindowAudit(data);
  renderLeaderboard(data);
  renderRegimeMatrix(data);
  renderStrategies(data);
  renderAutoTune(data);
  renderState(data);
}

function renderRegimeMatrix(data) {
  const r = data.regime || {};
  const card = document.getElementById('regime-card');
  if (!r.matrix || Object.keys(r.matrix).length === 0) {
    card.classList.add('hidden');
    return;
  }
  card.classList.remove('hidden');
  document.getElementById('regime-gen').textContent = r.generated_at || '—';
  const regimes = r.regimes || [];
  const matrix = r.matrix;
  const best = r.best_per_regime || {};

  // Current regime banner.
  const cur = r.current || {};
  if (cur.state && cur.state !== 'unknown') {
    document.getElementById('current-regime-banner').classList.remove('hidden');
    document.getElementById('cur-state').textContent = cur.state;
    document.getElementById('cur-asof').textContent = 'as of ' + (cur.as_of || '—');
    document.getElementById('cur-rationale').textContent = cur.rationale || '';
    // Map current state -> closest historical regime kind.
    const stateToKind = { bull: 'bull', bear: 'bear', choppy: 'choppy', crash: 'crash_then_bull' };
    const wanted = stateToKind[cur.state] || cur.state;
    const matchingRegime = regimes.find(rr => rr.kind === wanted) || regimes.find(rr => rr.kind === 'narrow_bull');
    if (matchingRegime && best[matchingRegime.name] && best[matchingRegime.name][0]) {
      const top = best[matchingRegime.name][0];
      document.getElementById('cur-top').textContent = top.formula;
      document.getElementById('cur-top-stats').textContent =
        `(${matchingRegime.label}: Sharpe ${fmtNum(top.sharpe)}, return ${fmtPct(top.total_return)}, DD ${fmtPct(top.max_drawdown)})`;
    }
  }

  const metric = document.getElementById('regime-metric').value || 'sharpe';
  drawRegimeTable(matrix, regimes, metric);
  document.getElementById('regime-metric').onchange = e => drawRegimeTable(matrix, regimes, e.target.value);

  // Rankings per regime
  const rankWrap = document.getElementById('regime-rankings');
  rankWrap.innerHTML = regimes.map(rr => {
    const list = best[rr.name] || [];
    const rows = list.slice(0, 5).map((s, i) => `
      <tr class="border-b border-white/5">
        <td class="text-xs text-slate-500">${i + 1}</td>
        <td class="text-xs">${s.formula}</td>
        <td class="text-right text-xs">${fmtNum(s.sharpe)}</td>
        <td class="text-right text-xs ${s.total_return > 0 ? 'text-emerald-400' : 'text-rose-400'}">${fmtPct(s.total_return)}</td>
      </tr>`).join('');
    return `<div class="card p-3">
      <div class="text-sm font-semibold">${rr.label}</div>
      <div class="text-xs text-slate-500">${rr.start} → ${rr.end} · <span class="pill pill-gray">${rr.kind}</span></div>
      <table class="w-full text-sm compact num mt-2">
        <thead class="text-slate-400 text-left border-b border-white/5">
          <tr><th>#</th><th>Strategy</th><th class="text-right">Sharpe</th><th class="text-right">Return</th></tr>
        </thead>
        <tbody>${rows || '<tr><td colspan="4" class="text-slate-500 text-xs py-2">no data</td></tr>'}</tbody>
      </table>
    </div>`;
  }).join('');
}

function drawRegimeTable(matrix, regimes, metric) {
  const head = document.getElementById('regime-head');
  head.innerHTML = '<th>Strategy</th>' + regimes.map(r => `<th class="text-right" title="${r.note || ''}">${r.label}<br><span class="text-slate-500 text-xs font-normal">${r.start.slice(0,7)} → ${r.end.slice(0,7)}</span></th>`).join('');
  const formulas = Object.keys(matrix).sort();
  // Find best cell per regime for highlighting.
  const bestByCol = {};
  regimes.forEach(r => {
    let best = -Infinity, bestF = null;
    formulas.forEach(f => {
      const cell = matrix[f][r.name];
      if (!cell || cell.error) return;
      let v = cell[metric];
      if (v == null) return;
      if (metric === 'max_drawdown') v = -Math.abs(v);  // less DD = better
      if (v > best) { best = v; bestF = f; }
    });
    bestByCol[r.name] = bestF;
  });
  const body = document.getElementById('regime-rows');
  body.innerHTML = formulas.map(f => {
    const cells = regimes.map(r => {
      const cell = matrix[f][r.name];
      if (!cell) return '<td class="text-right text-slate-600">—</td>';
      if (cell.error) return `<td class="text-right text-rose-400 text-xs" title="${cell.error}">err</td>`;
      const v = cell[metric];
      const isBest = bestByCol[r.name] === f;
      const fmt = (metric === 'sharpe') ? fmtNum(v) : fmtPct(v);
      const cls = (metric === 'max_drawdown') ? 'text-rose-400'
        : (v > 0 ? 'text-emerald-400' : 'text-rose-400');
      const ring = isBest ? 'background:#0f3a25; color:#5fe6a2; font-weight:600;' : '';
      return `<td class="text-right num" style="${ring}">${fmt}</td>`;
    }).join('');
    return `<tr class="border-b border-white/5">
      <td><a href="strategy.html?f=${f}" class="text-sky-300 hover:underline">${f}</a></td>
      ${cells}
    </tr>`;
  }).join('');
}

async function fetchData() {
  try {
    const res = await fetch('data.json?_=' + Date.now(), { cache: 'no-store' });
    if (!res.ok) throw new Error('http ' + res.status);
    const data = await res.json();
    const sig = data.generated_at + '|' + (data.strategies||[]).length + '|' + (data.auto_tune||[]).length;
    if (sig !== lastSig) {
      lastSig = sig;
      render(data);
      document.getElementById('live-label').textContent = 'LIVE · updated';
      setTimeout(() => { document.getElementById('live-label').textContent = 'LIVE'; }, 2000);
    }
  } catch (e) {
    document.getElementById('live-label').textContent = 'OFFLINE';
  }
}

function tick() {
  countdown -= 1;
  if (countdown <= 0) {
    countdown = POLL_SECS;
    fetchData();
  }
  document.getElementById('countdown').textContent = countdown;
}

fetchData();
setInterval(tick, 1000);
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') fetchData();
});
</script>
</body>
</html>
"""


STRATEGY_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>strategy archive — stock-screener</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root { color-scheme: dark; }
  body { background: #0b0f1a; color: #d6dae3; font-family: ui-sans-serif, system-ui, -apple-system; }
  .card { background: #131826; border: 1px solid #1f2638; border-radius: 14px; }
  .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
  .pill-green { background: #0f3a25; color: #5fe6a2; }
  .pill-red   { background: #3a0f1a; color: #ff7a8a; }
  .pill-gray  { background: #1f2638; color: #9aa4bb; }
  .num { font-variant-numeric: tabular-nums; }
  table.compact td, table.compact th { padding: 4px 8px; }
  .chart-box { position: relative; height: 280px; max-height: 280px; width: 100%; }
  .scroll-box { max-height: 360px; overflow-y: auto; }
  .delta-up   { color: #5fe6a2; }
  .delta-down { color: #ff7a8a; }
  details > summary { cursor: pointer; list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
</style>
</head>
<body class="min-h-screen">
<header class="max-w-6xl mx-auto px-4 pt-8 pb-4">
  <div class="flex items-center justify-between flex-wrap gap-3">
    <div>
      <a href="index.html" class="text-xs text-slate-400 hover:text-sky-300">← back to leaderboard</a>
      <h1 class="text-3xl font-bold tracking-tight mt-1" id="title">strategy archive</h1>
      <p class="text-slate-400 mt-1" id="subtitle">Every backtest run, every accepted parameter change, every auto-tune trial.</p>
    </div>
  </div>
</header>

<main class="max-w-6xl mx-auto px-4 pb-16 space-y-6">

  <section id="missing" class="card p-5 hidden">
    <div class="text-rose-300 text-sm">No archive found for this formula. Pass <code>?f=&lt;formula_name&gt;</code> in the URL, e.g. <code>strategy.html?f=momentum_v1</code>.</div>
  </section>

  <section id="content" class="space-y-6 hidden">

    <div class="card p-5">
      <div class="flex items-baseline justify-between flex-wrap gap-3">
        <h2 class="text-lg font-semibold">Performance timeline</h2>
        <span class="text-xs text-slate-500">Each point = one backtest run. Hover the chart for the run timestamp.</span>
      </div>
      <div class="chart-box mt-3"><canvas id="timeline-chart"></canvas></div>
      <div class="overflow-x-auto mt-3">
        <table class="w-full text-sm compact num">
          <thead class="text-slate-400 text-left border-b border-white/5">
            <tr>
              <th>Run</th><th>Window</th>
              <th class="text-right">Return</th><th class="text-right">Sharpe</th>
              <th class="text-right">Max DD</th><th class="text-right">Win%</th>
              <th class="text-right">α vs SPY</th><th class="text-right">Trades</th>
            </tr>
          </thead>
          <tbody id="timeline-rows"></tbody>
        </table>
      </div>
    </div>

    <div class="card p-5">
      <div class="flex items-baseline justify-between flex-wrap gap-3">
        <h2 class="text-lg font-semibold">Daily picks (top-N per rebalance)</h2>
        <span class="text-xs text-slate-500">From the latest backtest run · <span id="picks-source">—</span></span>
      </div>
      <p class="text-xs text-slate-500 mt-1">Each rebalance day (default W-FRI) the strategy ranks every ticker and holds the top N until the next rebalance.</p>
      <div class="flex items-center gap-3 mt-3 flex-wrap">
        <label class="text-xs text-slate-400">Jump to date:</label>
        <select id="picks-date" class="bg-[#0b0f1a] border border-white/10 text-sm px-2 py-1 rounded"></select>
        <span class="text-xs text-slate-500" id="picks-meta">—</span>
      </div>
      <div class="overflow-x-auto mt-3">
        <table class="w-full text-sm compact num">
          <thead class="text-slate-400 text-left border-b border-white/5">
            <tr>
              <th>#</th><th>Ticker</th>
              <th class="text-right">Score</th>
              <th class="text-right">P&amp;L (until next rebal)</th>
              <th>Exit reason</th>
              <th>Exit date</th>
            </tr>
          </thead>
          <tbody id="picks-rows"></tbody>
        </table>
      </div>
      <details class="mt-4">
        <summary class="text-sm text-sky-300 hover:underline">All rebalance days (collapsed)</summary>
        <div class="scroll-box mt-2 border border-white/5 rounded">
          <table class="w-full text-sm compact num">
            <thead class="text-slate-400 text-left border-b border-white/5 sticky top-0 bg-[#131826]">
              <tr><th>Date</th><th class="text-right">Picks</th><th class="text-right">Avg P&amp;L</th><th>Top 3</th></tr>
            </thead>
            <tbody id="picks-summary-rows"></tbody>
          </table>
        </div>
      </details>
    </div>

    <div class="card p-5">
      <div class="flex items-baseline justify-between flex-wrap gap-3">
        <h2 class="text-lg font-semibold">Accepted parameter changes</h2>
        <span class="text-xs text-slate-500">From <code>auto_tune_state.json</code> — only mutations that passed walk-forward.</span>
      </div>
      <div class="overflow-x-auto mt-3">
        <table class="w-full text-sm compact num">
          <thead class="text-slate-400 text-left border-b border-white/5">
            <tr><th>When</th><th>Parameter</th><th>Direction</th><th class="text-right">From</th><th class="text-right">To</th><th class="text-right">Δ%</th></tr>
          </thead>
          <tbody id="changelog-rows"></tbody>
        </table>
      </div>
    </div>

    <div class="card p-5">
      <div class="flex items-baseline justify-between flex-wrap gap-3">
        <h2 class="text-lg font-semibold">YAML param diffs (snapshot → snapshot)</h2>
        <span class="text-xs text-slate-500">Derived from <code>formulas/&lt;name&gt;.bak_*.yaml</code> backups + current file.</span>
      </div>
      <div id="diff-blocks" class="mt-3 space-y-3 text-sm"></div>
    </div>

    <div class="card p-5">
      <div class="flex items-baseline justify-between flex-wrap gap-3">
        <h2 class="text-lg font-semibold">Auto-tune trials (all attempts)</h2>
        <span class="text-xs text-slate-500">Including rejected — useful to see what the tuner is exploring.</span>
      </div>
      <div class="overflow-x-auto mt-3 scroll-box">
        <table class="w-full text-sm compact num">
          <thead class="text-slate-400 text-left border-b border-white/5 sticky top-0 bg-[#131826]">
            <tr>
              <th>When</th><th>Param</th><th>Direction</th>
              <th class="text-right">From</th><th class="text-right">To</th>
              <th class="text-right">Folds</th>
              <th class="text-right">avg composite</th><th class="text-right">baseline</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody id="trials-rows"></tbody>
        </table>
      </div>
    </div>

  </section>
</main>

<script>
const fmtPct = v => (v === null || v === undefined || v === '') ? '—' : (Number(v) * 100).toFixed(2) + '%';
const fmtPctSigned = v => (v === null || v === undefined || v === '') ? '—' : (Number(v) >= 0 ? '+' : '') + (Number(v) * 100).toFixed(2) + '%';
const fmtNum = (v, d=2) => (v === null || v === undefined || v === '') ? '—' : Number(v).toFixed(d);

function qs(name) {
  return new URLSearchParams(location.search).get(name);
}

async function load() {
  const f = qs('f');
  if (!f) { document.getElementById('missing').classList.remove('hidden'); return; }
  let data;
  try {
    const res = await fetch(`strategies/${encodeURIComponent(f)}.json?_=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) throw new Error();
    data = await res.json();
  } catch {
    document.getElementById('missing').classList.remove('hidden');
    return;
  }
  document.title = `${data.formula} — archive`;
  document.getElementById('title').textContent = data.formula;
  const baseline = data.monthly_baseline || {};
  document.getElementById('subtitle').textContent =
    `${(data.timeline||[]).length} backtest runs · ${(data.picks_by_date||[]).length} rebalance days · `
    + `${(data.change_log||[]).length} accepted param changes · ${(data.trials||[]).length} auto-tune trials · `
    + `baseline month ${baseline.month || '—'} (avg composite ${fmtNum(baseline.composite_avg, 3)})`;
  document.getElementById('content').classList.remove('hidden');
  renderTimeline(data);
  renderPicks(data);
  renderChangelog(data);
  renderDiffs(data);
  renderTrials(data);
}

function renderPicks(data) {
  const picks = data.picks_by_date || [];
  const sel = document.getElementById('picks-date');
  const src = document.getElementById('picks-source');
  src.textContent = data.picks_source || 'no trades file';
  if (!picks.length) {
    sel.innerHTML = '<option>—</option>';
    document.getElementById('picks-rows').innerHTML = '<tr><td colspan="6" class="text-slate-500 text-xs py-3">No trades recorded for the latest run.</td></tr>';
    document.getElementById('picks-summary-rows').innerHTML = '';
    return;
  }
  sel.innerHTML = picks.map(d => `<option value="${d.date}">${d.date}  (${d.n} picks · avg ${(d.avg_ret*100).toFixed(2)}%)</option>`).join('');
  sel.onchange = () => renderPicksRows(picks, sel.value);
  renderPicksRows(picks, picks[0].date);

  const ssr = document.getElementById('picks-summary-rows');
  ssr.innerHTML = picks.map(d => {
    const top3 = d.picks.slice(0, 3).map(p => p.ticker).join(', ');
    const cls = d.avg_ret > 0 ? 'text-emerald-400' : 'text-rose-400';
    return `<tr class="border-b border-white/5 cursor-pointer hover:bg-white/5" data-date="${d.date}">
      <td class="text-xs text-slate-400">${d.date}</td>
      <td class="text-right text-xs">${d.n}</td>
      <td class="text-right text-xs ${cls}">${(d.avg_ret*100).toFixed(2)}%</td>
      <td class="text-xs text-slate-300">${top3}</td>
    </tr>`;
  }).join('');
  ssr.querySelectorAll('tr').forEach(tr => {
    tr.addEventListener('click', () => {
      const d = tr.dataset.date;
      sel.value = d;
      renderPicksRows(picks, d);
      document.getElementById('picks-rows').scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });
}

function renderPicksRows(picks, date) {
  const day = picks.find(d => d.date === date);
  const tbody = document.getElementById('picks-rows');
  const meta = document.getElementById('picks-meta');
  if (!day) { tbody.innerHTML = ''; meta.textContent = ''; return; }
  meta.textContent = `${day.n} picks · avg P&L ${(day.avg_ret*100).toFixed(2)}%`;
  tbody.innerHTML = day.picks.map((p, i) => {
    const cls = p.ret > 0 ? 'text-emerald-400' : 'text-rose-400';
    const sign = p.ret >= 0 ? '+' : '';
    return `<tr class="border-b border-white/5">
      <td class="text-xs text-slate-500">${i + 1}</td>
      <td class="text-sm font-medium">${p.ticker}</td>
      <td class="text-right text-xs">${p.score.toFixed(3)}</td>
      <td class="text-right text-xs ${cls}">${sign}${(p.ret*100).toFixed(2)}%</td>
      <td class="text-xs"><span class="pill pill-gray">${p.exit || 'hold'}</span></td>
      <td class="text-xs text-slate-500">${p.exit_date || ''}</td>
    </tr>`;
  }).join('');
}

function renderTimeline(data) {
  const tl = data.timeline || [];
  const tbody = document.getElementById('timeline-rows');
  tbody.innerHTML = tl.map(r => {
    const ret = r.total_return, alpha = r.alpha_vs_benchmark;
    return `<tr class="border-b border-white/5">
      <td class="text-xs text-slate-400">${r.iso || r.stamp}</td>
      <td class="text-xs text-slate-500">${r.start} → ${r.end}<br><span class="pill pill-gray">${r.universe || ''}</span></td>
      <td class="text-right ${ret > 0 ? 'text-emerald-400' : 'text-rose-400'}">${fmtPct(ret)}</td>
      <td class="text-right">${fmtNum(r.sharpe)}</td>
      <td class="text-right text-rose-400">${fmtPct(r.max_drawdown)}</td>
      <td class="text-right">${fmtPct(r.win_rate)}</td>
      <td class="text-right ${alpha > 0 ? 'text-emerald-400' : 'text-rose-400'}">${fmtPctSigned(alpha)}</td>
      <td class="text-right text-slate-400">${r.n_trades || 0}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="8" class="text-slate-500 text-xs py-3">No backtest runs found.</td></tr>';

  const ctx = document.getElementById('timeline-chart').getContext('2d');
  const labels = tl.map(r => r.iso || r.stamp);
  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Sharpe', data: tl.map(r => r.sharpe), borderColor: '#60a5fa', backgroundColor: 'rgba(96,165,250,0.1)', tension: 0.2, borderWidth: 2, yAxisID: 'y' },
        { label: 'Total return', data: tl.map(r => r.total_return), borderColor: '#5fe6a2', tension: 0.2, borderWidth: 2, yAxisID: 'y1' },
        { label: '|Max DD|', data: tl.map(r => Math.abs(r.max_drawdown || 0)), borderColor: '#ff7a8a', borderDash: [4,4], tension: 0.2, borderWidth: 1.5, yAxisID: 'y1' },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { labels: { color: '#94a3b8' }}},
      scales: {
        x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { color: 'rgba(255,255,255,0.04)' }},
        y:  { position: 'left',  ticks: { color: '#60a5fa' }, grid: { color: 'rgba(255,255,255,0.04)' }, title: { text: 'Sharpe', display: true, color: '#60a5fa' }},
        y1: { position: 'right', ticks: { color: '#5fe6a2', callback: v => (v*100).toFixed(0)+'%' }, grid: { drawOnChartArea: false }, title: { text: 'Return / |DD|', display: true, color: '#5fe6a2' }},
      }
    }
  });
}

function renderChangelog(data) {
  const tbody = document.getElementById('changelog-rows');
  const log = (data.change_log || []).slice().reverse();  // newest first
  tbody.innerHTML = log.map(c => {
    let pct = '';
    if (typeof c.old === 'number' && typeof c.new === 'number' && c.old !== 0) {
      pct = (((c.new - c.old) / Math.abs(c.old)) * 100).toFixed(1) + '%';
    }
    const cls = c.direction === 'up' ? 'delta-up' : 'delta-down';
    return `<tr class="border-b border-white/5">
      <td class="text-xs text-slate-400">${(c.ts || '').slice(0,19).replace('T',' ')}</td>
      <td class="text-xs">${c.param}</td>
      <td class="text-xs"><span class="pill pill-gray ${cls}">${c.direction || ''}</span></td>
      <td class="text-right text-xs">${c.old}</td>
      <td class="text-right text-xs">${c.new}</td>
      <td class="text-right text-xs ${cls}">${pct}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="6" class="text-slate-500 text-xs py-3">No accepted changes yet — tuner has not modified this strategy.</td></tr>';
}

function renderDiffs(data) {
  const wrap = document.getElementById('diff-blocks');
  const diffs = (data.param_diffs || []).slice().reverse();  // newest first
  if (!diffs.length) {
    wrap.innerHTML = '<div class="text-xs text-slate-500">No backups (.bak_*.yaml) recorded for this formula yet.</div>';
    return;
  }
  wrap.innerHTML = diffs.map(d => `
    <details class="card p-3" open>
      <summary class="font-semibold text-sm text-sky-300">${d.from_stamp} → ${d.to_stamp}  <span class="text-slate-500 text-xs">(${d.changes.length} change${d.changes.length===1?'':'s'})</span></summary>
      <table class="w-full text-sm compact num mt-2">
        <thead class="text-slate-400 text-left border-b border-white/5">
          <tr><th>Key</th><th class="text-right">From</th><th class="text-right">To</th></tr>
        </thead>
        <tbody>${d.changes.map(c => `
          <tr class="border-b border-white/5">
            <td class="text-xs">${c.key}</td>
            <td class="text-right text-xs text-slate-400">${JSON.stringify(c.from)}</td>
            <td class="text-right text-xs">${JSON.stringify(c.to)}</td>
          </tr>`).join('')}</tbody>
      </table>
    </details>`).join('');
}

function renderTrials(data) {
  const tbody = document.getElementById('trials-rows');
  const rows = data.trials || [];
  tbody.innerHTML = rows.map(r => {
    const accepted = String(r.accepted).toLowerCase() === 'true';
    return `<tr class="border-b border-white/5">
      <td class="text-xs text-slate-400">${(r.ts || '').slice(0,19).replace('T',' ')}</td>
      <td class="text-xs">${r.param || ''}</td>
      <td class="text-xs">${r.direction || ''}</td>
      <td class="text-right text-xs text-slate-400">${r.old}</td>
      <td class="text-right text-xs">${r.new}</td>
      <td class="text-right text-xs">${r.folds_passed || 0}</td>
      <td class="text-right text-xs">${fmtNum(r.avg_composite, 3)}</td>
      <td class="text-right text-xs text-slate-500">${fmtNum(r.baseline_avg, 3)}</td>
      <td><span class="pill ${accepted ? 'pill-green' : 'pill-red'}">${accepted ? 'accepted' : 'rejected'}</span></td>
    </tr>`;
  }).join('') || '<tr><td colspan="9" class="text-slate-500 text-xs py-3">No trials recorded.</td></tr>';
}

load();
</script>
</body>
</html>
"""


def main():
    DOCS.mkdir(exist_ok=True)
    indexed = _index_summaries()
    strategies = collect_strategies(indexed)
    histories = collect_strategy_histories(indexed)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategies": strategies,
        "auto_tune": collect_auto_tune(),
        "state": collect_state(),
        "window_audit": collect_window_audit(strategies),
        "regime": collect_regime_matrix(),
    }
    (DOCS / "data.json").write_text(json.dumps(payload, indent=2, default=str))
    (DOCS / "index.html").write_text(HTML_TEMPLATE)

    # Per-strategy archive: one JSON per formula + a shared HTML viewer.
    strat_dir = DOCS / "strategies"
    strat_dir.mkdir(exist_ok=True)
    for name, hist in histories.items():
        (strat_dir / f"{name}.json").write_text(json.dumps(hist, indent=2, default=str))
    (DOCS / "strategy.html").write_text(STRATEGY_HTML_TEMPLATE)

    print(f"wrote {DOCS/'index.html'} ({len(HTML_TEMPLATE)} bytes) + data.json")
    print(f"wrote {DOCS/'strategy.html'} + {len(histories)} per-strategy archives in strategies/")
    print(f"strategies: {len(payload['strategies'])}  trials: {len(payload['auto_tune'])}")
    print("serve with:  python -m http.server -d docs 8000")


if __name__ == "__main__":
    main()
