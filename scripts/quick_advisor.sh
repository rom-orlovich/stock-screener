#!/usr/bin/env bash
# Thin wrapper — the real work lives in the stock-advisor Claude Code skill
# (~/.claude/skills/stock-advisor/SKILL.md).
#
# Usage hints:
#   In a Claude Code session, just ask one of:
#     - "what should I buy this week?"
#     - "give me today's picks"
#     - "market view + top 10"
#     - "what about NVDA?"        (single-ticker mode)
#
#   The skill description triggers automatically. You can also force it:
#     /skill stock-advisor
#
# The skill will:
#   1. Read runs/regime_current.json (regenerates if stale)
#   2. Pick the best strategy for the current regime from analysis.json
#   3. Read the latest scan_<strategy>_sp500_*.csv (regenerates via scan_all.sh if stale)
#   4. WebSearch news + sentiment for the top 5-10 tickers
#   5. Cross-strategy consensus tiering across all trustworthy scans
#   6. Output a concise, phone-friendly plan with entry/stops/caveats

set -euo pipefail
cat <<'EOF'
stock-advisor is a Claude Code skill, not a standalone command.

Open a Claude Code session in this repo (or anywhere) and ask naturally:
  - "what should I buy this week?"
  - "today's picks"
  - "market view"
  - "what about <TICKER>?"

Or force the skill:
  /skill stock-advisor

Skill location: ~/.claude/skills/stock-advisor/SKILL.md
EOF
