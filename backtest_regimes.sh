#!/usr/bin/env bash
# Wrapper: runs run_regimes.py, regenerates dashboard, logs everything.
set -u
cd "$(dirname "$0")"
mkdir -p runs
PY="${PYTHON:-$([ -x ./.venv/bin/python ] && echo ./.venv/bin/python || command -v python3 || command -v python)}"
LOG="runs/backtest_regimes_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"
echo "=== regime backtest @ $(date -Iseconds)  python=$PY ===" | tee -a "$LOG"
"$PY" run_regimes.py 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "--- regenerating dashboard ---" | tee -a "$LOG"
"$PY" dashboard.py 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "=== done @ $(date -Iseconds) ===" | tee -a "$LOG"
