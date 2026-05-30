#!/usr/bin/env bash
# WSL driver: prove no-regression (verify_fork) + benchmark OLD vs NEW on the real
# run_regimes path, from the warm cache. Logs progress to /tmp/wsl_bench.log.
#   OLD = spawn pool, parallel 4, vectorized   (pre-change production behaviour)
#   NEW = fork pool,  parallel 8, shared bank   (this branch)
set -u
cd ~/stock-screener
PY=./.venv/bin/python
END=2026-05-28
LIM="${1:-50}"
LOG=/tmp/wsl_bench.log
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ "${GATE:-0}" = "1" ]; then
  say "=== correctness gate: verify_fork (seq/fork/spawn/bank identical matrix) ==="
  $PY scripts/verify_fork.py >>"$LOG" 2>&1
  say "verify_fork exit=$?  (tail below)"
  grep -E "IDENTICAL|DIVERGENCE|PARITY" "$LOG" | tail -6 | sed 's/^/    /' | tee -a "$LOG"
fi
# The OLD-vs-NEW matrix diff below IS the no-regression proof on the real path:
# OLD = pre-change production behaviour (spawn+vectorized), NEW = fork+bank.

say "=== OLD: --parallel 4 --mp-context spawn --vectorized --limit $LIM ==="
t0=$(date +%s)
$PY run_regimes.py --parallel 4 --mp-context spawn --vectorized --limit "$LIM" --end "$END" >/tmp/old.log 2>&1
OLD=$(( $(date +%s) - t0 )); cp runs/regime_matrix.json /tmp/old_matrix.json
say "OLD wall=${OLD}s"

say "=== NEW: --parallel 8 --mp-context fork --shared-bank --limit $LIM ==="
t0=$(date +%s)
$PY run_regimes.py --parallel 8 --mp-context fork --shared-bank --limit "$LIM" --end "$END" >/tmp/new.log 2>&1
NEW=$(( $(date +%s) - t0 )); cp runs/regime_matrix.json /tmp/new_matrix.json
say "NEW wall=${NEW}s"

say "=== regression check: OLD matrix vs NEW matrix ==="
$PY -c "import json;a=json.load(open('/tmp/old_matrix.json'))['matrix'];b=json.load(open('/tmp/new_matrix.json'))['matrix'];print('MATRIX IDENTICAL — no regression' if json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True) else 'MATRIX DIFFERS — REGRESSION')" 2>&1 | tee -a "$LOG"

awk -v o="$OLD" -v n="$NEW" 'BEGIN{printf "[bench] OLD=%ds NEW=%ds speedup=%.2fx\n", o, n, (n>0? o/n : 0)}' | tee -a "$LOG"
say "DONE"
