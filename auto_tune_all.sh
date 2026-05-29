#!/usr/bin/env bash
# Nightly auto-tune wrapper: runs auto_tune.py on every formula in formulas/
# Excludes .bak_*.yaml backups. Logs everything to runs/auto_tune_nightly.log.
set -u
cd "$(dirname "$0")"
mkdir -p runs
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
  python auto_tune.py --formula "$f" --start 2023-01-01 --end "$END_DATE" --universe sp500 --trials 3 >> "$LOG" 2>&1 || echo "FAILED on $f" >> "$LOG"
done
echo "" >> "$LOG"
echo "--- regenerating dashboard ---" >> "$LOG"
python dashboard.py >> "$LOG" 2>&1 || echo "dashboard FAILED" >> "$LOG"
echo "" >> "$LOG"
echo "=== done @ $(date -Iseconds) ===" >> "$LOG"
