#!/usr/bin/env bash
# Live-rank every formula's universe and write scan_<formula>_<stamp>.csv
# per strategy. Much cheaper than a backtest — one score per ticker (not
# 176). Designed to run daily ~07:00 IDT (post-US-close) so each strategy's
# "what's hot now" ranking lands in the dashboard.
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-$([ -x ./.venv/bin/python ] && echo ./.venv/bin/python || command -v python3 || command -v python)}"
LOG="runs/scan_all_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"
echo "=== scan_all @ $(date -Iseconds)  python=$PY ===" | tee -a "$LOG"

"$PY" run_scans.py --universe sp500 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- regenerating dashboard ---" | tee -a "$LOG"
"$PY" dashboard.py >> "$LOG" 2>&1 || echo "dashboard FAILED" >> "$LOG"
echo "=== done @ $(date -Iseconds) ===" | tee -a "$LOG"
