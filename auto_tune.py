#!/usr/bin/env python3
"""Auto-tune a strategy YAML by mutating one parameter at a time.

Walk-forward 3-fold + overfit guardrails:
  1. Load formula + bounds (from formulas/*.yaml)
  2. Pick a tunable param (from `bounds`), honoring cooldowns + drift cap
  3. Mutate ±10% (clamped to bounds). Skip if anti-mirror block is active.
  4. Backtest across 3 walk-forward (train, test) folds
  5. Accept ONLY if composite_score(test) > monthly_baseline on ALL 3 folds
     AND |test_dd| <= baseline_dd * 1.10 on ALL folds
     AND train_sharpe > 0 on ALL folds
  6. On accept: backup YAML, write new, increment per-param change counter,
     set anti-mirror block, advance state.
  7. On reject: register cooldown (24h) + anti-mirror (24h opposite direction).
  8. At most 1 accepted change per run (cap drift).
  9. Monthly baseline lock: first run of each calendar month becomes the
     baseline used for the rest of that month.

State file: runs/auto_tune_state.json
Trial log:  runs/auto_tune.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import backtest as bt  # noqa: E402
from engine.data import get_universe  # noqa: E402
from engine.score import Formula  # noqa: E402
from engine.universe import sp500  # noqa: E402

RUNS = ROOT / "runs"
LOG = RUNS / "auto_tune.csv"
STATE = RUNS / "auto_tune_state.json"

COOLDOWN_HOURS = 24
MONTHLY_DRIFT_CAP = 5   # max accepted changes per param per calendar month
DRIFT_FREEZE_DAYS = 7
MAX_ACCEPTED_PER_RUN = 1


# ----------------------------- state ---------------------------------------

def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {"formulas": {}}


def save_state(state: dict):
    RUNS.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, default=str))


def fstate(state: dict, formula_name: str) -> dict:
    """Per-formula state. Initializes lazily."""
    fs = state["formulas"].setdefault(formula_name, {
        "cooldown_until": {},        # param -> iso ts
        "anti_mirror": {},            # param -> {direction: 'up'|'down', until: iso}
        "change_log": [],             # list of {ts, param, old, new}
        "frozen_until": {},           # param -> iso ts
        "monthly_baseline": None,     # {month: 'YYYY-MM', folds: [{sharpe, dd, score}], composite_avg: float}
    })
    return fs


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def is_active(ts_iso: str | None) -> bool:
    if not ts_iso:
        return False
    try:
        return parse_iso(ts_iso) > datetime.now()
    except Exception:  # noqa: BLE001
        return False


# ----------------------------- backtest ------------------------------------

def walk_forward_folds(start: str, end: str, k: int = 3):
    """Return k (train_start, train_end, test_start, test_end) tuples.

    Uses an expanding-train / rolling-test split:
      total span divided into k+1 chunks; fold i trains on [0..i+1] and tests
      on chunk i+1.
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    span = (e - s) / (k + 1)
    folds = []
    for i in range(k):
        train_s = s
        train_e = s + span * (i + 1)
        test_s = train_e
        test_e = s + span * (i + 2)
        folds.append((
            train_s.strftime("%Y-%m-%d"),
            train_e.strftime("%Y-%m-%d"),
            test_s.strftime("%Y-%m-%d"),
            test_e.strftime("%Y-%m-%d"),
        ))
    return folds


def run_bt(data, f, start, end):
    cfg = bt.BacktestConfig(
        top_n=20, rebalance="W-FRI",
        atr_stop_mult=2.0, trailing_stop_pct=0.06,
        trailing_activate_pct=0.05, time_stop_bars=15,
        benchmark_ticker="SPY",
    )
    return bt.run(data, f, start=start, end=end, cfg=cfg).stats


def composite(stats: dict) -> float:
    sharpe = float(stats.get("sharpe") or 0.0)
    dd = abs(float(stats.get("max_drawdown") or 0.0))
    return sharpe - dd


def eval_folds(data, raw, folds):
    """Run all folds. Returns list of {train, test, composite}."""
    out = []
    f = Formula(raw=raw)
    for ts, te, vs, ve in folds:
        tr = run_bt(data, f, ts, te)
        va = run_bt(data, f, vs, ve)
        out.append({
            "train": tr, "test": va,
            "composite": composite(va),
        })
    return out


# ----------------------------- params --------------------------------------

def tunable_params(raw: dict) -> list[tuple[str, str, float, float]]:
    bounds = raw.get("bounds", {}) or {}
    out = []
    for k, (lo, hi) in bounds.items():
        if k in raw.get("indicators", {}):
            out.append((k, "indicators", lo, hi))
        elif k in raw.get("timeframe_score_weights", {}):
            out.append((k, "timeframe_score_weights", lo, hi))
        elif k in raw:
            out.append((k, "root", lo, hi))
    return out


def get_param(raw, key, section):
    if section == "indicators":
        return raw["indicators"][key]
    if section == "timeframe_score_weights":
        return raw["timeframe_score_weights"][key]
    return raw[key]


def set_param(raw, key, value, section):
    if section == "indicators":
        raw["indicators"][key] = value
    elif section == "timeframe_score_weights":
        raw["timeframe_score_weights"][key] = value
    else:
        raw[key] = value


def mutate(value, lo, hi, force_direction: str | None = None):
    if force_direction == "up":
        step = 1.1
    elif force_direction == "down":
        step = 0.9
    else:
        step = random.choice([0.9, 1.1])
    direction = "up" if step > 1 else "down"
    new = value * step
    new = max(lo, min(hi, new))
    if isinstance(value, int):
        new = int(round(new))
        if new == value:
            new = value + (1 if step > 1 else -1)
            new = max(int(lo), min(int(hi), new))
    else:
        new = round(new, 4)
    return new, direction


# ----------------------------- guards --------------------------------------

def param_blocked(fs: dict, key: str, direction: str) -> str | None:
    if is_active(fs["cooldown_until"].get(key)):
        return "cooldown"
    if is_active(fs["frozen_until"].get(key)):
        return "frozen"
    am = fs["anti_mirror"].get(key)
    if am and is_active(am.get("until")) and am.get("direction") == direction:
        return "anti_mirror"
    return None


def changes_this_month(fs: dict, key: str) -> int:
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    return sum(1 for c in fs["change_log"]
               if c.get("param") == key and c.get("ts", "") >= cutoff)


# ----------------------------- logging -------------------------------------

def log_trial(row: dict):
    RUNS.mkdir(exist_ok=True)
    header = not LOG.exists()
    with LOG.open("a") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header:
            w.writeheader()
        w.writerow(row)


# ----------------------------- main ----------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--formula", required=True)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-05-26")
    p.add_argument("--universe", default="sp500")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    fp = Path(args.formula)
    raw = yaml.safe_load(fp.read_text())
    state = load_state()
    fs = fstate(state, fp.name)

    tickers = sp500() if args.universe == "sp500" else []
    if "SPY" not in tickers:
        tickers.append("SPY")
    fetch_start = (pd.Timestamp(args.start) - pd.DateOffset(months=8)).strftime("%Y-%m-%d")
    print(f"fetching {len(tickers)} tickers...", flush=True)
    data = get_universe(tickers, start=fetch_start, end=args.end, provider="yf")

    folds = walk_forward_folds(args.start, args.end, k=args.folds)
    print(f"walk-forward folds: {len(folds)}")
    for i, (ts, te, vs, ve) in enumerate(folds):
        print(f"  fold {i+1}: train {ts}->{te}  test {vs}->{ve}")

    # Monthly baseline lock
    current_month = datetime.now().strftime("%Y-%m")
    mb = fs.get("monthly_baseline")
    if not mb or mb.get("month") != current_month:
        print(f"\n=== monthly baseline lock for {current_month} ===")
        base_eval = eval_folds(data, raw, folds)
        fs["monthly_baseline"] = {
            "month": current_month,
            "folds": [{
                "train_sharpe": e["train"]["sharpe"],
                "train_dd": e["train"]["max_drawdown"],
                "test_sharpe": e["test"]["sharpe"],
                "test_dd": e["test"]["max_drawdown"],
                "composite": e["composite"],
            } for e in base_eval],
            "composite_avg": round(sum(e["composite"] for e in base_eval) / len(base_eval), 4),
        }
        save_state(state)
        print(f"baseline avg composite: {fs['monthly_baseline']['composite_avg']:.3f}")
    else:
        print(f"using locked baseline from {mb['month']} (avg composite {mb['composite_avg']:.3f})")

    baseline = fs["monthly_baseline"]
    baseline_composites = [f["composite"] for f in baseline["folds"]]
    baseline_dds = [abs(f["test_dd"]) for f in baseline["folds"]]

    params = tunable_params(raw)
    if not params:
        print("no tunable params declared in bounds")
        return

    accepted = []
    for trial in range(args.trials):
        if len(accepted) >= MAX_ACCEPTED_PER_RUN:
            print(f"trial {trial+1}: cap of {MAX_ACCEPTED_PER_RUN} accepted reached, stopping")
            break
        candidate_raw = yaml.safe_load(fp.read_text())
        random.shuffle(params)

        picked = None
        for key, section, lo, hi in params:
            # Drift cap
            if changes_this_month(fs, key) >= MONTHLY_DRIFT_CAP:
                fs["frozen_until"][key] = (datetime.now() + timedelta(days=DRIFT_FREEZE_DAYS)).isoformat()
                continue
            cur = get_param(candidate_raw, key, section)
            # Try both directions, pick first non-blocked
            for direction in random.sample(["up", "down"], 2):
                if not param_blocked(fs, key, direction):
                    picked = (key, section, lo, hi, cur, direction)
                    break
            if picked:
                break

        if not picked:
            print(f"trial {trial+1}: all params blocked, skipping")
            continue

        key, section, lo, hi, cur, direction = picked
        new, direction = mutate(cur, lo, hi, force_direction=direction)
        if new == cur:
            print(f"trial {trial+1}: {key} no-op (at bound)")
            continue
        set_param(candidate_raw, key, new, section)

        try:
            cand_eval = eval_folds(data, candidate_raw, folds)
        except Exception as exc:  # noqa: BLE001
            print(f"trial {trial+1}: error {exc}")
            continue

        # Acceptance: improves AND respects dd guard AND train sane on ALL folds
        improvements = [e["composite"] > b for e, b in zip(cand_eval, baseline_composites)]
        dd_ok = [abs(e["test"]["max_drawdown"]) <= b * 1.10 for e, b in zip(cand_eval, baseline_dds)]
        train_ok = [e["train"]["sharpe"] > 0 for e in cand_eval]
        accept = all(improvements) and all(dd_ok) and all(train_ok)

        avg_composite = round(sum(e["composite"] for e in cand_eval) / len(cand_eval), 4)
        avg_test_sharpe = round(sum(e["test"]["sharpe"] for e in cand_eval) / len(cand_eval), 3)
        avg_test_dd = round(sum(e["test"]["max_drawdown"] for e in cand_eval) / len(cand_eval), 3)

        row = {
            "ts": now_iso(), "formula": fp.name,
            "param": key, "direction": direction,
            "old": cur, "new": new,
            "folds_passed": sum(improvements),
            "avg_composite": avg_composite,
            "avg_test_sharpe": avg_test_sharpe,
            "avg_test_dd": avg_test_dd,
            "baseline_avg": baseline["composite_avg"],
            "accepted": accept,
        }
        log_trial(row)

        marker = "✓" if accept else "✗"
        print(f"trial {trial+1} {marker} {key} ({direction}): {cur} -> {new}  "
              f"folds_passed={sum(improvements)}/{len(folds)} "
              f"avg_composite={avg_composite:.3f} (base {baseline['composite_avg']:.3f})")

        if accept:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = fp.with_name(fp.stem + f".bak_{stamp}.yaml")
            shutil.copy(fp, backup)
            fp.write_text(yaml.safe_dump(candidate_raw, sort_keys=False))
            fs["change_log"].append({"ts": now_iso(), "param": key,
                                      "old": cur, "new": new, "direction": direction})
            opposite = "down" if direction == "up" else "up"
            fs["anti_mirror"][key] = {
                "direction": opposite,
                "until": (datetime.now() + timedelta(hours=COOLDOWN_HOURS)).isoformat(),
            }
            accepted.append(f"{key}:{cur}->{new}")
            print(f"  applied. backup -> {backup.name}")
        else:
            fs["cooldown_until"][key] = (datetime.now() + timedelta(hours=COOLDOWN_HOURS)).isoformat()
            fs["anti_mirror"][key] = {
                "direction": direction,
                "until": (datetime.now() + timedelta(hours=COOLDOWN_HOURS)).isoformat(),
            }

        save_state(state)

    print(f"\nfinal: accepted {len(accepted)} of {args.trials} trials")
    for a in accepted:
        print(f"  + {a}")


if __name__ == "__main__":
    main()
