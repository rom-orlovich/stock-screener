# Research Signals — Implementation + A/B Result

Branch `feat/research-signals` (worktree). Two research-backed ranking signals
added to `leading_stock_v1`, grounded in a deep-research factor review (academic,
out-of-sample evidence). **No other signals added** — the brief warns most
anomalies don't replicate; lean only on the survivors.

## What was added (grounded in the cited research)

### 1. `high52_proximity` — George & Hwang (2004)
*"Nearness to the 52-week high is a better predictor of future returns than past
returns; ranking by the 52-week high dominates and improves upon past-return
momentum."* (bauer.uh.edu/tgeorge/papers/gh4-paper.pdf)

`engine/score.py::_high52_proximity_score` = `clip01(close / rolling_high(252))`,
causal (rolling_high excludes the current bar), 1.0 at/above the 52-week high,
0.8 when 20% below. A **gentle** nearness measure — deliberately distinct from the
existing `breakout` sub-score, which applies a 5× penalty below the pivot. Reuses
the `breakout_lookback` rolling high (= 252 = 52 weeks in `leading_stock_v1`).

### 2. `trend_template` — Minervini SEPA 8-rule screen
*"Fully objective 8-rule screen, filters ~95% of stocks."* Implemented as a
**continuous score = fraction of the 8 rules passed (0-1)**, NOT a hard gate.
Rationale (documented in code): the engine already gates entries in managed mode
via the volume-confirmed breakout event, so a second hard 8/8 gate would starve
the book; a continuous fraction lets the ranker weight trend *quality* instead.

The 8 causal rules (`engine/score.py::_trend_template_score`):
1. px > 50d MA  2. px > 150d MA  3. px > 200d MA
4. 50d > 150d > 200d (stacking)
5. 200d MA rising vs ~1 month (21 bars) ago
6. px ≥ 30% above the 52-week low
7. px within 25% of the 52-week high
8. RS proxy — positive 6-month momentum.

**RS-rank (rule 8) decision:** true cross-sectional RS-rank (≥70th pct vs the
universe) is not available inside the per-ticker scorer without a two-pass
refactor (out of scope, high parity risk). The engine already enforces
cross-sectional RS at the *portfolio* level (it ranks names by total score and
takes top_n). Inside the per-ticker template, positive 6-month momentum is the
objective per-ticker proxy. This is an honest approximation, flagged here.

**Fixed periods (no overfitting surface):** 50/150/200/21/252 and the 30%/25%
bands are module-level constants in `engine/score.py`, deliberately NOT exposed to
the auto-tuner — the research explicitly warns against curve-fitting.

Both signals default to weight 0 via `w.get(key, 0.0)`, so every other formula is
unaffected and the zero-weight path is bit-identical to old code. Mirrored across
all three scoring paths (per-d0, per-ticker vectorized, cross-section) and the
shared indicator bank.

## Parity (all green, no network)

`scripts/parity_research_signals.py`:
- **SHAPE** — both scalar helpers compute the documented values (nearness ramp,
  8-rule fractions incl. NaN/partial cases); keys wired into `timeframe_score`.
- **BASELINE GOLDEN** — `leading_stock_v1` with the new weights stripped is
  bit-identical to a golden captured from the OLD code
  (`tests/fixtures/golden_research_parity.json`, 215 rows). Proves additivity.
- **TRI-PATH** — with both signals ON, score is identical across per-d0 /
  vectorized / cross-section (float paths strict; cross-section within the
  pre-existing 1.5e-4 rounding boundary), and both terms verifiably fire.

Existing suites unaffected: `parity_bank`, `parity_breakout_thrust`, `parity_gap`,
`parity_fast_monthly` all still bit-identical.

## A/B validation (single-process, sp500) — PARTIAL

Harness `scripts/ab_research_signals.py` (`--parallel 1`, sp500 only). Each new
weight is ADDED on top of baseline and renormalized by the scorer — the standard
"marginal contribution of one signal" test. ONE documented weight per variant
(no sweep — anti-overfit). All other knobs (managed mode, exits, time_stop=40)
identical across variants, so any delta is attributable to the signal.

### Window 1 — sp500, 2023-01-01 → 2026-06-01 (DONE)
`runs/ab_research_focused_2023.json`. Variant `both` = +high52(0.15) +tt(0.15).

| variant | total ret | alpha vs SPY | sharpe(wk) | max DD | win | trades | payoff | avg hold |
|---|---|---|---|---|---|---|---|---|
| baseline | **0.768** | −0.265 | **1.14** | **−0.188** | 0.48 | 276 | 1.87 | 28.3 |
| +both    | 0.721 | −0.312 | 1.08 | −0.234 | 0.48 | 276 | 1.88 | 28.3 |

SPY total return over the window: **1.033** (both variants trail buy-and-hold).

**Read:** adding both signals is **slightly negative** on this window — lower
return, lower sharpe, worse drawdown. The trade count is **identical (276)** and
the exit mix near-identical, meaning the ranking reshuffle barely changes the
managed-mode book: entry is gated by the volume-confirmed breakout event, and
among event-firing candidates these ranking signals move selection very little.
This matches the research caveat that 52wk/momentum effects are **weaker in
large-caps** (sp500 is all large-cap) and that 2023-now is a mega-cap-led bull.

### Window 2 — full 2018 → now + regimes (DEFERRED)
Deferred until `auto_tune` finishes (a background waiter handles it), then the
full 4-variant sweep (baseline / high52 / trend_tmpl / both) runs on
full_2018, bull_2021, bear_2022, ai_2023_2024 — which also **isolates each signal
separately** (the focused run only tested `both`). The 2018-now window is where
managed mode historically diverges most (per CLAUDE.md the managed path's edge is
on the long window), so it is the decisive test.

## Partial verdict

- **Implementation: solid and parity-clean.** Both signals are correct, causal,
  additive, and triple-path consistent. They are safe to keep in the codebase at
  weight 0 (zero impact) regardless of the A/B outcome.
- **On large-cap 2023-now: neither helps** (`both` is marginally negative). NOT a
  reason to ship them weighted yet.
- **Recommendation pending Window 2.** If full_2018 + the regimes also show no
  improvement, the honest call is to **keep the code (default OFF, weight 0) and
  do NOT weight them in `leading_stock_v1.yaml`** — exactly the anti-curve-fit
  discipline the brief demands. `leading_stock_v1.yaml` is left UNCHANGED for now.

_Updated after Window 2 completes._
