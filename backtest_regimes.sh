#!/usr/bin/env bash
# Wrapper: runs run_regimes.py, regenerates dashboard, logs everything.
set -u
cd "$(dirname "$0")"
mkdir -p runs
PY="${PYTHON:-$([ -x ./.venv/bin/python ] && echo ./.venv/bin/python || command -v python3 || command -v python)}"
LOG="runs/backtest_regimes_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"
# Fast path: fork pool (workers inherit the panel via COW — no per-worker pickle
# reload, low RAM so all cores fit) + shared indicator bank (--shared-bank implies
# --vectorized). Both parity-proven byte-identical (scripts/verify_fork.py,
# scripts/parity_bank.py). Override worker count with REGIME_PARALLEL.
REGIME_PARALLEL="${REGIME_PARALLEL:-$(nproc 2>/dev/null || echo 4)}"
echo "=== regime backtest @ $(date -Iseconds)  python=$PY  parallel=$REGIME_PARALLEL fork+bank ===" | tee -a "$LOG"
"$PY" run_regimes.py --parallel "$REGIME_PARALLEL" --mp-context fork --shared-bank 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "--- regenerating dashboard ---" | tee -a "$LOG"
"$PY" dashboard.py 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "=== done @ $(date -Iseconds) ===" | tee -a "$LOG"
