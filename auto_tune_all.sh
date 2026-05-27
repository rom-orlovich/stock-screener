#!/usr/bin/env bash
# Nightly auto-tune wrapper: runs auto_tune.py on every formula in formulas/
# Excludes .bak_*.yaml backups. Logs everything to runs/auto_tune_nightly.log.
set -u
cd "$(dirname "$0")"
mkdir -p runs
LOG="runs/auto_tune_nightly_$(date +%Y%m%d).log"
: > "$LOG"
echo "=== auto-tune nightly run @ $(date -Iseconds) ===" >> "$LOG"
for f in formulas/*.yaml; do
  case "$f" in
    *.bak_*.yaml) continue ;;
  esac
  echo "" >> "$LOG"
  echo "--- $f ---" >> "$LOG"
  python auto_tune.py --formula "$f" --start 2023-01-01 --end 2026-05-26 --universe sp500 --trials 3 >> "$LOG" 2>&1 || echo "FAILED on $f" >> "$LOG"
done
echo "" >> "$LOG"
echo "=== done @ $(date -Iseconds) ===" >> "$LOG"
