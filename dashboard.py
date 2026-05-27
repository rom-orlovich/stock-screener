#!/usr/bin/env python3
"""Generate a single-page static HTML dashboard from runs/ artifacts.

Reads:
  - runs/bt_summary_*.json     (latest per formula)
  - runs/bt_equity_*.csv       (equity curves)
  - runs/bt_benchmark_*.csv    (benchmark equity)
  - runs/bt_trades_*.csv       (per-trade detail, latest per formula)
  - runs/auto_tune.csv         (every tuning trial)
  - runs/auto_tune_state.json  (cooldowns, drift counters, baselines)

Writes:
  - docs/index.html  (single self-contained file, Chart.js + Tailwind via CDN)
  - docs/data.json   (the data the page reads; embedded too, but exposed)

No build step. Open docs/index.html in any browser, or serve via:
  python -m http.server -d docs 8000
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
DOCS = ROOT / "docs"

STAMP_RE = re.compile(r"_(\d{8}_\d{6})\.json$")


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
            header = next(reader, None)
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


def collect_strategies() -> list[dict]:
    """For each formula, find latest bt_summary_*.json and pair with equity/trades."""
    by_formula: dict[str, list[Path]] = defaultdict(list)
    for fp in RUNS.glob("bt_summary_*.json"):
        # name shape: bt_summary_<formula>[_<universe>]_<YYYYMMDD_HHMMSS>.json
        stem = fp.stem.replace("bt_summary_", "")
        # strip trailing stamp
        parts = stem.rsplit("_", 2)
        if len(parts) >= 2:
            formula_key = parts[0]
        else:
            formula_key = stem
        by_formula[formula_key].append(fp)

    out = []
    for formula_key, files in by_formula.items():
        files.sort(key=lambda p: _stamp_from_name(p.name), reverse=True)
        latest = files[0]
        summary = _read_summary(latest)
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

        # exit breakdown
        exits: dict[str, int] = defaultdict(int)
        for t in trades:
            exits[t.get("exit", "hold")] += 1

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
            "exits": dict(exits),
            "n_trades": len(trades),
            "history_count": len(files),
        })

    # Sort by composite score (sharpe - |dd|)
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


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>stock-screener — dashboard</title>
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
  .leader { box-shadow: 0 0 0 2px #d4a40855 inset; }
  details > summary { cursor: pointer; list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  table.compact td, table.compact th { padding: 4px 8px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
</style>
</head>
<body class="min-h-screen">
<header class="max-w-6xl mx-auto px-4 pt-8 pb-4">
  <h1 class="text-3xl font-bold tracking-tight">stock-screener</h1>
  <p class="text-slate-400 mt-1">Live transparency: every backtest, every tuning trial, every exit reason.</p>
  <p class="text-xs text-slate-500 mt-3">Generated <span id="gen-ts"></span> — <a class="underline hover:text-slate-300" href="https://github.com/rom-orlovich/stock-screener" target="_blank">repo on GitHub</a></p>
</header>

<main class="max-w-6xl mx-auto px-4 pb-16">

  <!-- Quick-start -->
  <section class="card p-5 mb-6">
    <h2 class="text-lg font-semibold mb-3">Run it yourself</h2>
    <pre class="bg-black/40 border border-white/5 rounded-lg p-3 text-sm overflow-x-auto"><code># Backtest the leader
python run.py backtest \
  --formula formulas/dual_momentum_v1.yaml \
  --universe sp500 --start 2023-01-01 --end 2026-05-26 --top-n 20

# Regenerate this dashboard
python dashboard.py && python -m http.server -d docs 8000</code></pre>
    <p class="text-xs text-slate-500 mt-2">Full guide in <a class="underline" href="https://github.com/rom-orlovich/stock-screener/blob/main/README.md" target="_blank">README.md</a>.</p>
  </section>

  <!-- Leaderboard -->
  <section class="card p-5 mb-6">
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-lg font-semibold">Strategy leaderboard</h2>
      <span class="text-xs text-slate-500">Sorted by composite = sharpe − |dd|</span>
    </div>
    <div class="overflow-x-auto">
      <table class="w-full text-sm compact num">
        <thead class="text-slate-400 text-left border-b border-white/5">
          <tr>
            <th>Strategy</th>
            <th class="text-right">Return</th>
            <th class="text-right">Sharpe</th>
            <th class="text-right">Max DD</th>
            <th class="text-right">Win%</th>
            <th class="text-right">Trades</th>
            <th class="text-right">vs SPY</th>
          </tr>
        </thead>
        <tbody id="leaderboard"></tbody>
      </table>
    </div>
  </section>

  <!-- Strategy cards -->
  <section class="space-y-4" id="strategy-cards"></section>

  <!-- Auto-tune log -->
  <section class="card p-5 mt-6">
    <h2 class="text-lg font-semibold mb-3">Auto-tune activity (latest 30 trials)</h2>
    <p class="text-xs text-slate-500 mb-3">Every nightly mutation — accepted or rejected — logged with the walk-forward result.</p>
    <div class="overflow-x-auto">
      <table class="w-full text-sm compact num">
        <thead class="text-slate-400 text-left border-b border-white/5">
          <tr>
            <th>When</th>
            <th>Formula</th>
            <th>Param</th>
            <th>Change</th>
            <th class="text-right">Folds passed</th>
            <th class="text-right">Score</th>
            <th class="text-right">Baseline</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody id="autotune-log"></tbody>
      </table>
    </div>
  </section>

  <!-- State / guardrails -->
  <section class="card p-5 mt-6">
    <h2 class="text-lg font-semibold mb-3">Active guardrails</h2>
    <p class="text-xs text-slate-500 mb-3">Cooldowns, frozen params and monthly baselines per strategy.</p>
    <div id="state-blocks" class="grid-2"></div>
  </section>

</main>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const fmtPct = v => (v === null || v === undefined) ? '—' : (Number(v) * 100).toFixed(2) + '%';
const fmtNum = (v, d=2) => (v === null || v === undefined) ? '—' : Number(v).toFixed(d);
const data = JSON.parse(document.getElementById('payload').textContent);
document.getElementById('gen-ts').textContent = data.generated_at;

function composite(s) {
  const sh = (s.stats && s.stats.sharpe) || 0;
  const dd = Math.abs((s.stats && s.stats.max_drawdown) || 0);
  return sh - dd;
}

// Leaderboard
const lb = document.getElementById('leaderboard');
data.strategies.forEach((s, i) => {
  const tr = document.createElement('tr');
  tr.className = 'border-b border-white/5 ' + (i === 0 ? 'leader' : '');
  const ret = s.stats.total_return;
  const alpha = s.stats.alpha_vs_benchmark;
  tr.innerHTML = `
    <td class="py-2"><a href="#${s.formula}" class="text-sky-300 hover:underline">${s.formula}</a>
      ${i === 0 ? '<span class="pill" style="background:#3a2c08;color:#d4a408;margin-left:6px">leader</span>' : ''}</td>
    <td class="text-right ${ret > 0 ? 'text-emerald-400' : 'text-rose-400'}">${fmtPct(ret)}</td>
    <td class="text-right">${fmtNum(s.stats.sharpe)}</td>
    <td class="text-right text-rose-400">${fmtPct(s.stats.max_drawdown)}</td>
    <td class="text-right">${fmtPct(s.stats.win_rate)}</td>
    <td class="text-right text-slate-400">${s.stats.n_trades || 0}</td>
    <td class="text-right ${alpha > 0 ? 'text-emerald-400' : 'text-rose-400'}">${fmtPct(alpha)}</td>`;
  lb.appendChild(tr);
});

// Strategy cards (with equity curve charts)
const cards = document.getElementById('strategy-cards');
data.strategies.forEach((s, idx) => {
  const card = document.createElement('section');
  card.id = s.formula;
  card.className = 'card p-5';
  const exitsTxt = Object.entries(s.exits || {}).map(([k,v]) =>
    `<span class="pill pill-gray" style="margin-right:6px">${k}: ${v}</span>`).join('');
  card.innerHTML = `
    <div class="flex items-start justify-between gap-3 flex-wrap">
      <div>
        <h3 class="text-xl font-semibold">${s.formula}</h3>
        <p class="text-xs text-slate-500">${s.start} → ${s.end} · ${s.universe} · last run ${s.stamp || '—'}</p>
      </div>
      <div class="text-right text-sm">
        <div>Composite <span class="font-semibold">${fmtNum(composite(s))}</span></div>
        <div class="text-xs text-slate-500">${s.history_count} historical run${s.history_count===1?'':'s'}</div>
      </div>
    </div>

    <div class="grid-2 mt-4">
      <div>
        <div class="text-xs text-slate-400 mb-1">Equity vs SPY (base 1.0)</div>
        <canvas id="chart-${idx}" height="160"></canvas>
      </div>
      <div class="text-sm space-y-1">
        <div class="flex justify-between"><span class="text-slate-400">Total return</span><span class="num">${fmtPct(s.stats.total_return)}</span></div>
        <div class="flex justify-between"><span class="text-slate-400">Sharpe</span><span class="num">${fmtNum(s.stats.sharpe)}</span></div>
        <div class="flex justify-between"><span class="text-slate-400">Max drawdown</span><span class="num text-rose-400">${fmtPct(s.stats.max_drawdown)}</span></div>
        <div class="flex justify-between"><span class="text-slate-400">Win rate</span><span class="num">${fmtPct(s.stats.win_rate)}</span></div>
        <div class="flex justify-between"><span class="text-slate-400">Trades</span><span class="num">${s.stats.n_trades || 0}</span></div>
        <div class="flex justify-between"><span class="text-slate-400">Rebalances</span><span class="num">${s.stats.n_rebalances || 0}</span></div>
        <div class="flex justify-between"><span class="text-slate-400">Benchmark return</span><span class="num">${fmtPct(s.stats.benchmark_total_return)}</span></div>
        <div class="flex justify-between"><span class="text-slate-400">Alpha vs SPY</span><span class="num ${s.stats.alpha_vs_benchmark > 0 ? 'text-emerald-400' : 'text-rose-400'}">${fmtPct(s.stats.alpha_vs_benchmark)}</span></div>
        <div class="mt-3 text-xs text-slate-400">Exit breakdown</div>
        <div>${exitsTxt || '<span class="text-slate-500 text-xs">no trades yet</span>'}</div>
      </div>
    </div>

    <details class="mt-4">
      <summary class="text-sm text-sky-300 hover:underline">Backtest config used</summary>
      <pre class="bg-black/40 border border-white/5 rounded p-3 text-xs mt-2 overflow-x-auto">${JSON.stringify(s.config, null, 2)}</pre>
    </details>`;
  cards.appendChild(card);

  // chart
  const ctx = card.querySelector(`#chart-${idx}`).getContext('2d');
  const eqLabels = s.equity.map(p => p.date);
  const eqData = s.equity.map(p => p.value);
  const bench = s.benchmark || [];
  const benchMap = new Map(bench.map(p => [p.date, p.value]));
  const benchData = eqLabels.map(d => benchMap.get(d) ?? null);
  new Chart(ctx, {
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
      plugins: { legend: { labels: { color: '#94a3b8' }}},
      scales: {
        x: { ticks: { color: '#64748b', maxTicksLimit: 6 }, grid: { color: 'rgba(255,255,255,0.04)' }},
        y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.04)' }},
      }
    }
  });
});

// Auto-tune log
const at = document.getElementById('autotune-log');
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

// State / guardrails
const sb = document.getElementById('state-blocks');
const byFormula = (data.state && data.state.formulas) || {};
Object.entries(byFormula).forEach(([name, fs]) => {
  const cd = fs.cooldown_until || {};
  const fz = fs.frozen_until || {};
  const am = fs.anti_mirror || {};
  const cl = fs.change_log || [];
  const mb = fs.monthly_baseline || {};
  const now = new Date();
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
</script>
</body>
</html>
"""


def main():
    DOCS.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategies": collect_strategies(),
        "auto_tune": collect_auto_tune(),
        "state": collect_state(),
    }
    (DOCS / "data.json").write_text(json.dumps(payload, indent=2, default=str))
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, default=str))
    (DOCS / "index.html").write_text(html)
    print(f"wrote {DOCS/'index.html'} ({len(html)} bytes)")
    print(f"strategies: {len(payload['strategies'])}  trials: {len(payload['auto_tune'])}")
    print("serve with:  python -m http.server -d docs 8000")


if __name__ == "__main__":
    main()
