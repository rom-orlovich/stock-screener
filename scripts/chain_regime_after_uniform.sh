#!/usr/bin/env bash
# Watch the uniform backtest tmux session; when it exits, fire the regime backtest.
# Started in its own tmux session ('stock-regime-watcher') so it survives logout.
set -u
cd "$(dirname "$0")/.."
LOG="runs/chain_regime_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"
echo "=== chain watcher start @ $(date -Iseconds) ===" >> "$LOG"

# Wait for the uniform tmux session to disappear.
until ! tmux has-session -t stock-uniform 2>/dev/null; do
  sleep 30
done

echo "uniform session ended @ $(date -Iseconds) -- launching regime backtest" >> "$LOG"

# Now run the regime backtest in its own supervised tmux session.
tmux kill-session -t stock-regime 2>/dev/null || true
tmux new-session -d -s stock-regime -c "$(pwd)" \
  "exec ./backtest_regimes.sh"

echo "regime tmux launched @ $(date -Iseconds)" >> "$LOG"
