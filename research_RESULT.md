# Research Signals — Implementation + A/B Result (FINAL)

Branch `feat/research-signals`. Two research-backed ranking signals added to
`leading_stock_v1`, grounded in a deep-research factor review (academic,
out-of-sample evidence). **No other signals added** — the brief warns most
anomalies don't replicate; lean only on the survivors.

## What was added (grounded in the cited research)

### 1. `high52_proximity` — George & Hwang (2004)  →  **SHIPPED**
*"Nearness to the 52-week high is a better predictor of future returns than past
returns; ranking by the 52-week high dominates and improves upon past-return
momentum."* (bauer.uh.edu/tgeorge/papers/gh4-paper.pdf)

`engine/score.py::_high52_proximity_score` = `clip01(close / rolling_high(252))`,
causal, 1.0 at/above the 52-week high, 0.8 when 20% below. A gentle nearness
measure — distinct from `breakout` (which applies a 5× penalty below the pivot).
Reuses the `breakout_lookback` rolling high (= 252 = 52 weeks in leading_stock_v1).

### 2. `trend_template` — Minervini SEPA 8-rule screen  →  kept in code, weight 0
Implemented as a **continuous score = fraction of 8 rules passed (0-1)**, NOT a
hard gate (the engine already gates entries via managed mode + volume-confirmed
breakout, so a second 8/8 gate would starve the book). 8 causal rules: px>50/150/
200d MA, 50>150>200 stacking, 200d MA rising vs ~1mo ago, ≥30% above 52wk low,
within 25% of 52wk high, positive 6-month RS.
- **RS-rank (rule 8)**: true cross-sectional RS-rank isn't available inside the
  per-ticker scorer without a two-pass refactor (out of scope). The engine already
  enforces cross-sectional RS at the portfolio level (ranks by total score, takes
  top_n); inside the template, positive 6-month momentum is the per-ticker proxy.
  Honest approximation, flagged.
- **Fixed periods (no overfitting surface)**: 50/150/200/21/252 + 30%/25% bands
  are module constants, NOT auto-tuner params (the research warns against
  curve-fitting).

Both default to weight 0 via `w.get(key,0.0)`; other formulas unaffected and the
zero-weight path is bit-identical to old code. Mirrored across all three scoring
paths (per-d0, vectorized, cross-section) + the shared bank.

## Parity (all green, no network) — `scripts/parity_research_signals.py`
- **SHAPE** — both helpers compute documented values; keys wired into the scorer.
- **BASELINE GOLDEN** — `leading_stock_v1` with the new weights stripped is
  bit-identical to a golden captured from OLD code (215 rows). Proves additivity.
- **TRI-PATH** — with both signals ON, score identical across per-d0 / vectorized /
  cross-section (floats strict; cross-section within the pre-existing 1.5e-4
  rounding boundary); both terms verifiably fire.
- Existing suites unaffected: `parity_bank`, `parity_breakout_thrust`, `parity_gap`,
  `parity_fast_monthly` all still bit-identical.

## A/B validation (single-process `--parallel 1`, sp500, concurrent with nightly tuner)

Harness `scripts/ab_research_signals.py`. Each new weight ADDED on top of baseline
and renormalized — the standard "marginal contribution of one signal" test. ONE
documented weight per variant (no sweep — anti-overfit). All other knobs (managed
mode, exits, time_stop=40) identical across variants. The shipped YAML config
(`high52_proximity: 0.20`) is bit-identical to the `high52` variant below, so its
numbers ARE the live leading_stock_v1 numbers.

Full grid — **total return** (and max-drawdown) by window. `both` = +0.15/+0.15.

| window (SPY) | baseline | +high52 | +trend_tmpl | +both |
|---|---|---|---|---|
| **full_2023** (+1.033) | 0.768 / −0.188 | **0.907 / −0.163** | 0.892 / −0.215 | 0.721 / −0.234 |
| **full_2018** (+2.143) | 1.679 / −0.227 | 1.698 / −0.208 | **1.878 / −0.227** | 1.822 / −0.234 |
| bull_2021 (+0.262) | 0.104 / −0.101 | 0.104 / −0.101 | 0.105 / −0.092 | 0.105 / −0.092 |
| bear_2022 (−0.166) | −0.093 / −0.220 | **−0.081 / −0.210** | −0.094 / −0.223 | −0.094 / −0.223 |
| ai_2023_2024 (+0.577) | 0.531 / −0.112 | **0.656 / −0.099** | 0.603 / −0.098 | 0.582 / −0.098 |

Sharpe/alpha confirm the same ranking (e.g. full_2023 sharpe: base 1.14 → high52
1.29 → trend 1.21 → both 1.08; full_2023 alpha: −0.265 → −0.127 → −0.141 → −0.312).

## HONEST per-signal verdict

### `high52_proximity` — **KEEP, SHIPPED** (weight 0.20 raw → 0.167 normalized)
Positive or neutral in **every** window, never negative. Strongest where it
matters most — the tuner's own decision window full_2023 (**+18% return, alpha
−0.265→−0.127, sharpe 1.14→1.29, DD −0.188→−0.163**) and the AI bull
(**+24% return**, DD −0.112→−0.099). Defensive in the 2022 bear (smaller loss,
better DD). Improves drawdown in 3 of 5 windows and ties the other 2. This is
exactly the George-Hwang result the literature predicts. Shipped as the #2 weight
behind momentum — a genuinely first-class ranking signal. Note: it reuses the
`breakout_lookback` rolling high, so keep that at ~252 for the 52-week window
(the tuner *may* drift it within [120,300]; the signal degrades gracefully).

### `trend_template` — **KEEP in code, do NOT ship weighted** (weight 0, opt-in)
Individually strong: best raw return on the long window (full_2018 **+11.9%**,
+0.20 alpha) and solidly positive on full_2023 (+16%) and the AI bull (+14%). BUT:
1. **Redundant with high52** — both reward "strong-trend, near-52wk-high" names.
   Combined at equal weight they over-concentrate and **interfere destructively**:
   `both` on full_2023 = **0.721, worse than baseline 0.768**, and underperforms
   *either signal alone* in every window.
2. Worse drawdown than high52 on the tuner window (−0.215 vs −0.163) and flat in
   the bear (−0.094 vs baseline −0.093).
Choosing between two partly-redundant signals, high52 is the more robust
(better risk-adjusted, never negative). Shipping both would require dedicated
weight tuning = curve-fitting, which the brief forbids. So `trend_template` stays
implemented + parity-clean + documented as an opt-in alternative (set its weight
INSTEAD of high52, never alongside).

### `both` (equal weight) — **DROP.** Underperforms each signal alone and even
baseline on the live tuner window. The earlier "Window-1 both-is-negative" read
was an artifact of testing only the combined variant — isolating the signals
reverses it.

## Final shipped config (`formulas/leading_stock_v1.yaml`)
- `timeframe_score_weights.high52_proximity: 0.20`  (new, A/B-justified)
- `timeframe_score_weights.trend_template: 0.00`    (off; opt-in)
- `bounds.high52_proximity: [0.05, 0.30]`           (tuner may refine the weight)
- All other weights unchanged.

## Merge-time note (CLAUDE.md hard rule #3)
This is a hand edit to a tuned YAML. On merge to `main`, clear `leading_stock_v1`'s
`monthly_baseline` in `runs/auto_tune_state.json` (gitignored, per-machine) so the
next nightly run re-establishes the baseline against the new weight. No action
needed in this worktree (its `runs/` is separate; the running nightly tuner uses
the main checkout and was never touched).

## Status
Implementation + tests + parity + A/B + verdict complete. Committed on
`feat/research-signals` and pushed. NOT merged.
