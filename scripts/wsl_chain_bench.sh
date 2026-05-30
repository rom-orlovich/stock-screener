#!/usr/bin/env bash
# Full-universe head-to-head on WSL, without disturbing the in-flight OLD run.
# Waits for the running `run_regimes.py --parallel 6` (OLD: spawn-era code, no
# fork/bank) to finish, snapshots its matrix + wall time, then runs NEW
# (fork + shared bank + parallel 8) on the SAME window and compares.
set -u
cd ~/stock-screener
PY=./.venv/bin/python
LOG=/tmp/wsl_chain.log
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

OLD_START=$(date -d "2026-05-30 11:50:59" +%s)
say "waiting for OLD run (--parallel 6) to finish ..."
while pgrep -f "run_regimes.py --parallel 6" >/dev/null 2>&1; do sleep 20; done
OLD_WALL=$(( $(date +%s) - OLD_START ))
cp runs/regime_matrix.json /tmp/old_full_matrix.json
say "OLD done: wall=${OLD_WALL}s ($((OLD_WALL/60))m), $($PY -c 'import json;print(sum(len(v) for v in json.load(open("/tmp/old_full_matrix.json"))["matrix"].values()))') cells"

say "launching NEW: --parallel 8 --mp-context fork --shared-bank (same dynamic window) ..."
t0=$(date +%s)
$PY run_regimes.py --parallel 8 --mp-context fork --shared-bank >/tmp/new_full.log 2>&1
NEW_WALL=$(( $(date +%s) - t0 ))
cp runs/regime_matrix.json /tmp/new_full_matrix.json
say "NEW done: wall=${NEW_WALL}s ($((NEW_WALL/60))m)"

say "=== REGRESSION CHECK (full universe): OLD matrix vs NEW matrix ==="
$PY -c "import json;a=json.load(open('/tmp/old_full_matrix.json'))['matrix'];b=json.load(open('/tmp/new_full_matrix.json'))['matrix'];ka=set(a);kb=set(b);common={f:{r for r in a[f]} & {r for r in b[f]} for f in ka&kb};diffs=[(f,r) for f in ka&kb for r in (set(a[f])&set(b[f])) if a[f][r]!=b[f][r]];print('CELLS_OLD',sum(len(v) for v in a.values()),'CELLS_NEW',sum(len(v) for v in b.values()));print('IDENTICAL — no regression' if not diffs else ('REGRESSION in '+str(len(diffs))+' cells: '+str(diffs[:5])))" 2>&1 | tee -a "$LOG"
awk -v o=$OLD_WALL -v n=$NEW_WALL 'BEGIN{printf "[result] OLD=%ds (%.0fm)  NEW=%ds (%.0fm)  speedup=%.2fx\n",o,o/60,n,n/60,(n>0?o/n:0)}' | tee -a "$LOG"
say "CHAIN DONE"
