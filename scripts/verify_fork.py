#!/usr/bin/env python3
"""Phase 1+2 end-to-end gate: the regime matrix must be byte-identical across
multiprocessing modes AND with the shared indicator bank.

  seq        --parallel 1                      (original sequential path = reference)
  fork       --parallel N --mp-context fork     (workers inherit panel via COW)
  spawn      --parallel N --mp-context spawn     (workers re-load from pickle cache)
  fork_bank  --shared-bank --mp-context fork     (bank computed once, shared, vec path)

All four must produce an identical 'matrix' (ignoring the 'generated_at' timestamp
and the 'current' regime block). fork/spawn only change how a worker obtains the
panel; the bank only changes how indicators are computed (once vs per-formula) —
neither may change a single result. This is the production-path proof complementing
the unit-level scripts/parity_bank.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LIMIT = "10"
END = "2026-05-28"

CONFIGS = {
    "seq":       ["--parallel", "1", "--vectorized"],
    "fork":      ["--parallel", "4", "--mp-context", "fork", "--vectorized"],
    "spawn":     ["--parallel", "4", "--mp-context", "spawn", "--vectorized"],
    "fork_bank": ["--parallel", "4", "--mp-context", "fork", "--shared-bank"],
}


def run(name: str, flags: list[str]) -> dict:
    cmd = [PY, "-u", "run_regimes.py", *flags, "--limit", LIMIT, "--end", END]
    print(f"[{name}] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{name}] FAILED rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        raise SystemExit(2)
    mat = json.loads((ROOT / "runs" / "regime_matrix.json").read_text())["matrix"]
    (ROOT / "runs" / "_golden" / f"p1_matrix_{name}.json").write_text(
        json.dumps(mat, indent=2, sort_keys=True))
    return mat


def main() -> int:
    (ROOT / "runs" / "_golden").mkdir(parents=True, exist_ok=True)
    mats = {name: run(name, flags) for name, flags in CONFIGS.items()}
    ref_name = "seq"
    ref = json.dumps(mats[ref_name], sort_keys=True)
    ok = True
    for name, mat in mats.items():
        if name == ref_name:
            continue
        if json.dumps(mat, sort_keys=True) == ref:
            print(f"  {name} == {ref_name}: IDENTICAL")
        else:
            ok = False
            print(f"  {name} != {ref_name}: DIVERGENCE")
            # show first differing cell
            for fk in sorted(set(mat) | set(mats[ref_name])):
                if mat.get(fk) != mats[ref_name].get(fk):
                    print(f"    first diff at formula '{fk}':")
                    a, b = mats[ref_name].get(fk, {}), mat.get(fk, {})
                    for rk in sorted(set(a) | set(b)):
                        if a.get(rk) != b.get(rk):
                            print(f"      {rk}: seq={a.get(rk)} {name}={b.get(rk)}")
                    break
    print("\nPHASE 1 PARITY: " + ("PASS — fork/spawn/seq identical" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
