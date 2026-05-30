#!/usr/bin/env bash
# Nightly auto-tune wrapper: runs auto_tune.py on every formula in formulas/
# Excludes .bak_*.yaml backups. Logs everything to runs/auto_tune_nightly.log.
set -u
cd "$(dirname "$0")"
mkdir -p runs

# Portable python resolver: prefer a project-local venv if present, else fall
# back to whatever the shell finds. Hosts that need a different interpreter
# (e.g. PEP 668 systems running from a venv) just create ./.venv and we use it
# automatically — no per-host edits to this script.
PY="${PYTHON:-$([ -x ./.venv/bin/python ] && echo ./.venv/bin/python || command -v python3 || command -v python)}"

# Vectorized scoring for every backtest the tuner runs (~216/night). Parity-proven
# bit-identical for the W-FRI rebalance the tuner uses (scripts/parity_test_vectorized.py,
# full_backtest_parity.py); self-gates to the slow path for any non-FRI rebalance, so
# enabling it cannot change a result or invalidate the monthly baseline lock.
export USE_VECTORIZED_SCORING=1
LOG="runs/auto_tune_nightly_$(date +%Y%m%d).log"
: > "$LOG"
# Window end = yesterday (full closing bar). Start stays at 2023-01-01 so the
# baseline lock has a stable training span; only the end rolls forward.
END_DATE="$(date -Idate -d yesterday)"
echo "=== auto-tune nightly run @ $(date -Iseconds) | window 2023-01-01 -> ${END_DATE} ===" >> "$LOG"
for f in formulas/*.yaml; do
  case "$f" in
    *.bak_*.yaml) continue ;;
  esac
  echo "" >> "$LOG"
  echo "--- $f ---" >> "$LOG"
  "$PY" auto_tune.py --formula "$f" --start 2023-01-01 --end "$END_DATE" --universe sp500 --trials 3 >> "$LOG" 2>&1 || echo "FAILED on $f" >> "$LOG"
done
echo "" >> "$LOG"
echo "--- regenerating dashboard ---" >> "$LOG"
"$PY" dashboard.py >> "$LOG" 2>&1 || echo "dashboard FAILED" >> "$LOG"
echo "" >> "$LOG"
echo "=== done @ $(date -Iseconds) ===" >> "$LOG"
