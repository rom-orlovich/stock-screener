#!/usr/bin/env bash
# Wait for stock-uniform + stock-regime tmux sessions to finish on this host,
# then rsync runs/ to WSL (rom-pc-sec) and tear down stock-* tmux sessions.
# Periodic mid-run rsyncs every 10 min so WSL stays roughly up-to-date even
# while the laptop is still working.
set -u
cd "$(dirname "$0")/.."
LOG="runs/migrate_to_wsl_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"
log()  { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
sync() {
  log "rsync runs/ -> rom-pc-sec:stock-screener/runs/ ..."
  rsync -a --info=stats1 --exclude='*.log' --exclude='_compare_tmp' \
    runs/ rom-pc-sec:stock-screener/runs/ 2>&1 | tee -a "$LOG"
}

log "watcher start -- will sync every 10 min until stock-uniform + stock-regime are both gone."

# Periodic mid-run sync. Stop conditions: BOTH stock-uniform and stock-regime
# tmux sessions are absent (stock-regime is created by the chain watcher
# after stock-uniform ends, so we must see it appear AND disappear).
saw_regime=0
while true; do
  # Use exact-match (= prefix) so 'stock-regime' does NOT match 'stock-regime-watcher'.
  uniform_alive=$(tmux has-session -t =stock-uniform 2>/dev/null && echo 1 || echo 0)
  regime_alive=$(tmux has-session -t =stock-regime 2>/dev/null && echo 1 || echo 0)
  [ "$regime_alive" = "1" ] && saw_regime=1
  if [ "$uniform_alive" = "0" ] && [ "$regime_alive" = "0" ] && [ "$saw_regime" = "1" ]; then
    log "both sessions ended (regime was observed)."
    break
  fi
  log "still alive: uniform=$uniform_alive regime=$regime_alive saw_regime=$saw_regime"
  sync
  sleep 600
done

log "final sync"
sync

log "regenerating dashboard on WSL so per-strategy archives reflect new data"
ssh rom-pc-sec 'cd ~/stock-screener && ./.venv/bin/python dashboard.py' 2>&1 | tee -a "$LOG"

log "tearing down laptop stock-* tmux sessions"
for s in stock-server stock-regime-watcher; do
  tmux kill-session -t "$s" 2>/dev/null && log "killed $s" || log "$s already gone"
done

log "done. WSL is now the single source of truth."
