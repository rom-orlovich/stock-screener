#!/usr/bin/env bash
# Run a uniform backtest for every formula on the SAME window so the dashboard
# leaderboard becomes a fair, apples-to-apples comparison. Cache is warm so
# each run is ~30s. Dashboard is regenerated after every formula so the live
# UI shows each strategy appearing one by one.
set -u
cd "$(dirname "$0")"
START=2023-01-01
END="$(date -Idate -d yesterday)"
UNIVERSE=sp500
LOG="runs/backtest_all_uniform_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"
echo "=== uniform backtest @ $(date -Iseconds)  window $START -> $END  universe $UNIVERSE ===" | tee -a "$LOG"
for f in formulas/*.yaml; do
  case "$f" in *.bak_*.yaml) continue ;; esac
  echo "" | tee -a "$LOG"
  echo "--- $f @ $(date +%H:%M:%S) ---" | tee -a "$LOG"
  python run.py backtest --formula "$f" --universe "$UNIVERSE" \
      --start "$START" --end "$END" --top 20 >> "$LOG" 2>&1 \
      || echo "FAILED on $f" | tee -a "$LOG"
  python dashboard.py >> "$LOG" 2>&1 || true
done
echo "" | tee -a "$LOG"
echo "=== done @ $(date -Iseconds) ===" | tee -a "$LOG"
