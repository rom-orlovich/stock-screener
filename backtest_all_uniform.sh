#!/usr/bin/env bash
# Run a uniform backtest for every formula on the SAME window so the dashboard
# leaderboard becomes a fair, apples-to-apples comparison.
#
# Implementation: delegates to run_uniform.py which fetches the universe ONCE
# and runs every backtest in-memory (optionally in a process pool). Massively
# faster than the old `for f in formulas; python run.py backtest ...` loop
# (~25 min vs ~4 hours on this hardware; ~6-8 min with --parallel 4 on a
# 4-core machine).
#
# Environment overrides:
#   PARALLEL=4    Override the formula concurrency (default = cpu_count/2).
#   ONLY=foo,bar  Run only these formula basenames (no .yaml).
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-$([ -x ./.venv/bin/python ] && echo ./.venv/bin/python || command -v python3 || command -v python)}"
PARALLEL="${PARALLEL:-0}"   # 0 = auto (cpu_count/2)
START=2023-01-01
END="$(date -Idate -d yesterday)"
UNIVERSE=sp500
LOG="runs/backtest_all_uniform_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"
echo "=== uniform backtest @ $(date -Iseconds)  window $START -> $END  universe $UNIVERSE  python=$PY  parallel=$PARALLEL ===" | tee -a "$LOG"

ONLY_ARG=""
if [ -n "${ONLY:-}" ]; then
  ONLY_ARG="--only $ONLY"
fi

"$PY" run_uniform.py \
    --start "$START" --end "$END" \
    --universe "$UNIVERSE" --top 20 \
    --parallel "$PARALLEL" \
    $ONLY_ARG 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- regenerating dashboard ---" | tee -a "$LOG"
"$PY" dashboard.py >> "$LOG" 2>&1 || echo "dashboard FAILED" >> "$LOG"
echo "" | tee -a "$LOG"
echo "=== done @ $(date -Iseconds) ===" | tee -a "$LOG"
