#!/usr/bin/env python3
"""Compare two golden fingerprints (scripts/capture_golden.py output).

Exit 0 iff every cell's stats, trades digest, and equity digest are identical —
i.e. the candidate flags caused ZERO change to any backtest result. Exit 1 and
print the first divergences otherwise. This is the no-regression gate.

Usage:
    python scripts/diff_golden.py runs/_golden/baseline.json runs/_golden/cand.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: diff_golden.py <baseline.json> <candidate.json>")
        return 2
    base = json.loads(Path(sys.argv[1]).read_text())["cells"]
    cand = json.loads(Path(sys.argv[2]).read_text())["cells"]

    keys = sorted(set(base) | set(cand))
    mismatches: list[str] = []
    for k in keys:
        if k not in base:
            mismatches.append(f"{k}: only in candidate")
            continue
        if k not in cand:
            mismatches.append(f"{k}: only in baseline")
            continue
        b, c = base[k], cand[k]
        if b["trades"] != c["trades"]:
            mismatches.append(f"{k}: TRADES differ  base={b['trades']} cand={c['trades']}")
        if b["equity"] != c["equity"]:
            mismatches.append(f"{k}: EQUITY differ  base={b['equity']} cand={c['equity']}")
        if b["stats"] != c["stats"]:
            # Report the specific stat fields that diverged.
            bs, cs = b["stats"], c["stats"]
            for sk in sorted(set(bs) | set(cs)):
                if bs.get(sk) != cs.get(sk):
                    mismatches.append(f"{k}: stat[{sk}]  base={bs.get(sk)} cand={cs.get(sk)}")

    n = len(keys)
    if mismatches:
        print(f"REGRESSION: {len(mismatches)} divergence(s) across {n} cells:")
        for m in mismatches[:60]:
            print(f"  {m}")
        if len(mismatches) > 60:
            print(f"  ... and {len(mismatches) - 60} more")
        return 1
    print(f"IDENTICAL: all {n} cells match (stats + trades + equity). No regression.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
