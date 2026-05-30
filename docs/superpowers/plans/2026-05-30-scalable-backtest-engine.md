# Scalable Backtest Engine — Implementation Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the regime-matrix run (`run_regimes.py --parallel 4 --vectorized`, currently ~4h) to ~1h and make it scale to a *growing* universe (S&P 500 → more tickers), with **provably zero change to backtest results**.

**Architecture:** One unifying move — *compute everything reusable once in the parent process, then share it read-only with every worker via `fork` copy-on-write.* Today the parent fetches the panel and discards it; spawn workers re-read 504 pickle files each and every `bt.run` recomputes formula-independent indicators 12×–72×. We invert that: parent builds (a) the price panel and (b) a per-ticker **indicator bank** (every distinct `(indicator, period)` series + the monthly resample) once; workers inherit both and only assemble + score. A later phase replaces the per-ticker Python scoring loop with a **cross-sectional vectorized** scorer that scales with `numpy`, not with ticker count.

**Tech Stack:** Python 3.12, pandas, numpy, multiprocessing (`fork` context), yfinance. Optional numba (exit loop). No new deps required for Phases 1–3.

**Non-negotiable (user constraint):** *No liquidity prefilter — keep every stock, the universe is expanding.* Every change ships behind a flag, default OFF, with a parity script proving bit-identical results, and is only enabled after the **full regime matrix matches the golden master byte-for-byte.**

---

## 1. Baseline & context (what is already true)

| Fact | Evidence | Implication |
|---|---|---|
| Baseline run = `run_regimes.py --parallel 4 --vectorized` ≈ 4h | user-confirmed; matches commit `720fac7` | The 2.28× vectorized win and 4-way parallel are **already spent**. Next gains are architectural. |
| `--parallel 4`, not 8, on this box | this device = 8 cores / **7.6 GB RAM** | Parallelism is **RAM-bound**, not CPU-bound: each spawn worker holds a full panel copy. |
| WSL `rom-pc-sec` = 8 cores / 9.7 GB / **no numba**; user will raise WSL to **10 cores** | `ssh` probe | More cores only help if RAM per worker drops — i.e. only with `fork` COW sharing. |
| Spawn workers re-read ~504 pickles each | `run_regimes.py:122` inside `_run_job` | 72 jobs × 504 = **~36k redundant deserializations** per run; scales linearly with universe. |
| `precompute_indicators` is keyed by formula but computes **formula-independent** series (RSI/ATR/BB/volume/monthly-resample) | `engine/score.py:285-350` | Same indicator math redone per formula **and** per regime → up to **72× redundant** for the monthly resample, ~12× for single-period indicators. |
| Per-`d0` scoring loops over every ticker in Python | `engine/backtest.py:308` | Cost is `O(formulas × regimes × tickers × dates)` in **interpreted Python** — the part that scales worst as the universe grows. |
| Regime runs use `rebalance="W-FRI"` | `run_regimes.py:53` | The vectorized path's parity (proven only for W-FRI) **applies cleanly** here. |
| Cache key embeds `end=yesterday` | `engine/data.py:62` + nightly `--end` | Cache never hits across days → full re-download nightly; a bigger universe makes this a yfinance rate-limit hazard. |

**Cost model.** Total regime-matrix wall time ≈
`Σ_jobs [ data_load + Σ_tickers precompute + Σ_dates Σ_tickers score_lookup + Σ_picks exit_loop ] / workers`.
Every term except `data_load` scales **linearly with universe size T**. The levers below attack each term and improve its scaling in T.

---

## 2. The levers (mechanism · effect · scaling · regression)

Ordered by leverage-per-risk for the *4h regime run on a growing universe*.

### Lever 1 — `fork` pool + parent-owned panel (replaces spawn re-load)

**Mechanism.** Parent already builds `data` at `run_regimes.py:184`. Stash it in a module global *before* creating the pool; switch `get_context("spawn")` → `get_context("fork")` (`run_regimes.py:220`). Fork workers inherit `data` via copy-on-write — one physical copy of the panel for all 10 workers. `_run_job` reads the global instead of calling `get_universe` (`run_regimes.py:122`). Keep a `--mp-context spawn` escape hatch for non-fork platforms.

**Effect.** Eliminates ~36k pickle reads/run. Drops per-worker RAM from *(panel + derived)* to *(only the pages a worker dirties)* → makes **`--parallel 10` fit in WSL's 10 GB**, where spawn at 10 would OOM. Parallelism 4→10 ≈ **2.0–2.5×** (72 jobs divide well across 10 workers), plus the removed reload overhead.

**Scaling in T.** Spawn reload cost is `O(jobs × T)`; fork is `O(T)` once. The bigger the universe, the larger this win.

**Regression risk: NONE (results), LOW (mechanics).** Fork changes *how a worker obtains bytes*, not the bytes or the math — workers see the identical DataFrame. Risks are purely operational: (a) fork-safety — verify no open files/locks/threads exist at pool-creation time (parent does only closed-file I/O and no threading → safe); (b) env-var propagation — children inherit `os.environ` under fork, so `USE_VECTORIZED_SCORING` still flows (`run_regimes.py:113-114` comment already anticipates this); (c) numba's JIT cache is lazy + per-process → unaffected. **Proof:** results are independent of worker count and order (matrix keyed by formula/regime, JSON sorted) — the golden-master diff (§3) is byte-identical by construction.

### Lever 2 — Indicator bank: precompute once in the parent, inherit via fork

**Mechanism.** Every series in `precompute_indicators` (`score.py:285-350`) is a pure function of `(price_series, indicator_fn, period)` — **none depend on `timeframe_score_weights`**. Build, in the parent, a per-ticker bank of every *distinct* `(indicator, period)` across all formulas + the formula-independent monthly resample:

```python
# new: engine/bank.py  (computed once in the parent, before the fork pool)
def collect_specs(formulas: list[Formula]) -> set[tuple[str, int]]:
    """Union of every (indicator, period) any formula needs."""
    specs = set()
    for f in formulas:
        c = f.raw["indicators"]
        specs |= {
            ("rsi", c["rsi_period"]), ("sma", c["sma_fast"]), ("sma", c["sma_slow"]),
            ("momentum", c["momentum_lookback"]),
            ("rolling_high", c["breakout_lookback"]), ("rolling_low", c["breakout_lookback"]),
            ("stdev_returns", c.get("volatility_lookback", 90)),
            ("atr_pct", c.get("atr_period", 14)), ("bb_width", c.get("bb_period", 20)),
            ("avg_vol", c.get("vol_short_lookback", 20)), ("avg_vol", c.get("vol_long_lookback", 60)),
        }
    return specs

def build_bank(daily: pd.DataFrame, specs) -> dict:
    """Per-timeframe {(indicator,period): series} + monthly resample. Pure, causal."""
    # daily + weekly timeframes; reuses engine.indicators / engine.atr exactly.
    ...
```

Refactor `precompute_indicators(daily, f)` into `assemble_precompute(bank_for_ticker, f)` that **selects** the series the formula needs from the bank (zero new computation). Gate behind `USE_SHARED_BANK=1`; when off, fall through to today's `precompute_indicators`. Composes with Lever 1: the bank is built once in the parent and inherited COW by all workers.

**Effect.** Collapses indicator computation from `O(formulas × regimes × T)` to `O(distinct_specs × T)`. Measured redundancy (agent audit): monthly resample 72×→1×; RSI/ATR/BB/volume 12×→1×; SMA/momentum/breakout 12×→2–6×. Net ~**65–75% less indicator CPU**, which is the dominant per-ticker cost on the vectorized path (where `score_ticker_at` is O(1) lookups). Realistic **~1.6–2.0×** on top of Lever 1.

**Scaling.** Win grows with both #formulas and #regimes. As strategies are added, the bank amortizes harder.

**Regression risk: LOW.** The bank stores the *identical* series the per-formula path computes (same function, same period, same full causal series), so `score_ticker_at` reads identical values. The **adaptive-cap fallback** (`score.py:444-462`) is untouched: it recomputes from the raw slice via `timeframe_score(df.iloc[:pos+1], f)` and never reads the bank, so short-history behavior is preserved exactly. **Proof:** parity script asserts `build_bank`-fed precompute == current `precompute_indicators` series-for-series (max abs diff 0.0), then a full single-formula `full_backtest_parity.py`-style run asserts every trade identical.

### Lever 3 — Cross-sectional vectorized scorer (the `feat/cross-section-vec` payoff)

**Mechanism.** Replace the per-ticker Python loop (`backtest.py:308`) for the **daily** timeframe. Align all tickers onto a common date index → one 2D float array per indicator (`shape = (n_dates, n_tickers)`); per `d0`, a *single* `searchsorted` yields the row; compute all sub-scores as vectorized numpy over the length-T vector; combine with the weight vector via one matrix–vector product; `argsort` for top-N. Weekly/monthly stay on the per-ticker path (they are dominated by the adaptive-cap fallback — see Risk). Gate behind `USE_CROSS_SECTION=1`.

**Effect.** Turns the `O(T)` interpreted loop into `O(1)` numpy ops over a T-vector for the daily timeframe. ~**1.3–1.6×** at current T; **the win grows with T** — numpy over 2000 tickers is ~the same wall as 500, whereas the Python loop is strictly linear. This is *the* lever for "more stocks."

**Scaling.** Best scaling of all levers: near-flat in T for the daily score step.

**Regression risk: LOW (daily-only), HIGH (if extended to weekly).** Two concrete hazards, both must be covered by parity:
1. **Float summation order / ties.** The scalar path ranks with Python's *stable* sort on score alone — ties resolve to `price_data` insertion order (`backtest.py:322`). A vectorized `argsort` sums weights in a different order (tiny FP deltas) and has its own tie order. If a tie flips, top-N picks change → trades change → **regression**. *Mitigation:* (a) compute the cross-section weighted sum in float64 in the same conceptual order; (b) replicate the scalar tie-break exactly — stable argsort keyed `(-score, ticker_insertion_index)`; (c) **parity asserts identical top-N picks per `d0` and an identical trades table**, not merely close scores.
2. **Alignment / NaN / skip masking.** The `len(hist) < 60` skip (`backtest.py:310`), the `_EMPTY_TF` early-returns, and `searchsorted` side semantics (`_pos_at_daily`, `score.py:270-272`) must reproduce per-column. *Mitigation:* build a boolean validity mask per `d0`; masked tickers get `-inf` score so they never enter top-N — exactly mirroring the `continue`.

**Why weekly is deferred:** for any formula with `sma_slow ≈ 200` (the default), weekly `n_avail < 3×sma_slow` is *always* true → the adaptive cap fires on essentially every weekly call and routes to the slow per-slice path (`score.py:462`). Vectorizing only the *uncapped* weekly branch buys almost nothing; vectorizing the *capped* branch means reproducing an `n_avail`-dependent, per-(ticker,date) window — HIGH parity risk for little gain. Leave weekly/monthly per-ticker.

### Lever 4 — numba exit loop (optional, lowest priority)

**Mechanism.** Already implemented behind `USE_NUMBA_EXITS` (`backtest.py:37-77`); bit-exact arithmetic mirror of the Python exit loop (`backtest.py:174-199`).

**Effect.** Small — the exit loop is ~5 bars × picks, a minor share. **WSL has no numba** (would need `pip install numba`), and commit `07ac28b` marks its parity test *pending*.

**Regression risk: LOW once verified, currently UNVERIFIED.** Action: run `scripts/parity_numba_exits.py` to completion; only enable if bit-identical. *(A verification agent is confirming this now; fold its verdict into Phase 4.)*

### Lever 5 — Incremental data cache + batched fetch (enables "more stocks")

**Mechanism.** Re-key the disk cache on `(provider, ticker, start)` and store the max `end` fetched; on read, if cached `end ≥` requested, slice in-memory; else fetch only the missing tail (`engine/data.py:58-68`). Replace per-ticker `yf.download` (`data.py:30`) with batched multi-ticker download. Optionally store float32 prices.

**Effect.** Kills the nightly full re-download; makes fetch cost sub-linear in calendar days and robust to a 2–4× larger universe (fewer HTTP calls → less rate-limit exposure). Doesn't touch the 4h *compute*, but is prerequisite to scaling the universe.

**Regression risk: LOW.** Caching/fetch only — same OHLCV returned. float32 changes least-significant digits → keep prices float64 unless a parity run over the full matrix stays byte-identical; treat float32 as a separate, separately-proven step.

---

## 3. Regression-proof protocol (the part that matters most)

This repo's culture: *every* perf change ships behind a flag + a parity script (see `scripts/full_backtest_parity.py`: "1562 trades bit-identical"). We formalize it into a gate every phase must pass.

1. **Pin a reproducible window.** Parity and golden-master runs use a **fixed `--end` date** (not `yesterday`) so re-runs are comparable. Record it in the spec PR.
2. **Capture the golden master (once, before any change).** Run the current `run_regimes.py --parallel 4 --vectorized` on the pinned window; save `runs/regime_matrix.json` as `runs/_golden/regime_matrix.golden.json`. Also capture full trades+stats for 2 representative formulas (one momentum, one mean-reversion) via a `full_backtest_parity.py`-style dump.
3. **Per-lever parity script (the "test").** Each lever gets a script asserting fast-path == reference-path:
   - stats dict equal; equity curve equal; **every trade row identical**.
   - Cross-section additionally: identical top-N picks per `d0`.
   - Tolerance policy: **bit-identical for picks & trades**; scores may differ < 1e-9 **only if** they never change a pick.
4. **Integration gate (before flipping any default).** Re-run the **full** regime matrix with the new flags ON; diff against the golden master ignoring only the `generated_at` timestamp. Must be **byte-identical** (JSON is already `sort_keys=True`, so `diff` is meaningful).
5. **Flip defaults only after the matrix matches.** Update `backtest_regimes.sh` / `auto_tune_all.sh` to pass the proven flags only once the gate is green. Keep each flag's OFF path intact for rollback (mirrors `USE_LEGACY_MONTHLY`).

`make parity` target (or `scripts/parity_all.sh`) runs every parity script + the golden diff in sequence and exits non-zero on any mismatch. **No phase merges without it green.**

---

## 4. Implementation plan (phased, TDD = parity-as-test)

Each phase is independently shippable and leaves the engine correct. Worktree: `feat/cross-section-vec` (current branch).

### Phase 0 — Golden master & harness

**Files:** Create `scripts/capture_golden.py`, `scripts/parity_all.sh`; Create dir `runs/_golden/`.

- [ ] **Step 1** — Write `scripts/capture_golden.py`: runs `run_regimes` logic on a **pinned** window (`--end 2026-05-29`), copies `runs/regime_matrix.json` → `runs/_golden/regime_matrix.golden.json`, and dumps trades+stats for `momentum_v1` and `mean_reversion_v1` to `runs/_golden/`.
- [ ] **Step 2** — Run it; confirm golden files exist and are non-empty. Expected: 12×6 matrix populated.
- [ ] **Step 3** — Write `scripts/parity_all.sh` that runs each existing parity script (`parity_test_vectorized.py`, `parity_fast_monthly.py`, `parity_numba_exits.py`) and prints PASS/FAIL, exiting non-zero on any failure.
- [ ] **Step 4** — Run `scripts/parity_all.sh`; record the current PASS/FAIL state of each (numba expected FAIL/unverified).
- [ ] **Step 5** — Commit: `test(parity): golden master capture + parity_all harness`.

### Phase 1 — fork pool + parent-owned panel

**Files:** Modify `run_regimes.py` (`_run_job` ~109-140; pool ~217-230; main ~157-184).

- [ ] **Step 1** — Add `--mp-context {fork,spawn}` arg (default `fork`). Add a module global `_SHARED_DATA = None`.
- [ ] **Step 2** — In `main`, after building `data` (line 184), set `_SHARED_DATA = data`. Change `_run_job` to use `_SHARED_DATA` when present, else fall back to `get_universe` (so the spawn path still works). Shrink the payload to `(fp_str, regime)`.
- [ ] **Step 3** — Switch `get_context("spawn")` → `get_context(args.mp_context)` at line 220.
- [ ] **Step 4 (parity)** — Run `run_regimes.py --parallel 10 --vectorized --mp-context fork` on the pinned window; diff `runs/regime_matrix.json` vs golden ignoring `generated_at`. Expected: **byte-identical**.
- [ ] **Step 5** — Run again with `--parallel 1` and with `--mp-context spawn`; both must match golden (proves all three paths agree).
- [ ] **Step 6** — Commit: `perf(regimes): fork pool inherits parent panel — no per-worker reload`.

### Phase 2 — Shared indicator bank

**Files:** Create `engine/bank.py`; Modify `engine/score.py` (`precompute_indicators` → add `assemble_precompute`, gate on `USE_SHARED_BANK`); Modify `run_regimes.py` (build bank in parent, store global); Create `scripts/parity_bank.py`.

- [ ] **Step 1** — Write `scripts/parity_bank.py`: for 5 tickers × all 12 formulas, assert `assemble_precompute(build_bank(...), f)` returns series equal (max abs diff 0.0) to `precompute_indicators(daily, f)` for every key. (Failing test — function doesn't exist yet.)
- [ ] **Step 2** — Run it; expected FAIL (`ImportError: engine.bank`).
- [ ] **Step 3** — Implement `engine/bank.py` (`collect_specs`, `build_bank`) and `assemble_precompute` in `score.py`, gated so `USE_SHARED_BANK!=1` uses the current path verbatim.
- [ ] **Step 4** — Run `scripts/parity_bank.py`; expected **PASS** (diff 0.0 every key).
- [ ] **Step 5** — Wire `run_regimes.py`: parent calls `collect_specs(formulas)` + builds `bank[ticker]` once, stores in a global inherited by fork workers; workers pass the per-ticker bank into `bt.run` (new optional `bank=` arg threaded to `assemble_precompute`).
- [ ] **Step 6 (parity)** — Full regime matrix with `USE_SHARED_BANK=1`; diff vs golden → **byte-identical**.
- [ ] **Step 7** — Commit: `perf(engine): shared per-ticker indicator bank — kills cross-formula recompute`.

### Phase 3 — Cross-sectional daily scorer

**Files:** Create `engine/cross_section.py`; Modify `engine/backtest.py` (`run` loop ~292-323, gate on `USE_CROSS_SECTION`); Create `scripts/parity_cross_section.py`.

- [ ] **Step 1** — Write `scripts/parity_cross_section.py`: for the full sp500 on the pinned window, assert the cross-section path produces an **identical trades table and identical top-N picks per `d0`** vs the per-ticker `score_ticker_at` path. (Failing test.)
- [ ] **Step 2** — Run it; expected FAIL (no `engine.cross_section`).
- [ ] **Step 3** — Implement the daily cross-section panel + masked, tie-stable `argsort` ranker (tie key `(-score, insertion_index)`), weekly/monthly still per-ticker. Gate on `USE_CROSS_SECTION=1`.
- [ ] **Step 4** — Run `scripts/parity_cross_section.py`; expected **PASS** (identical picks + trades).
- [ ] **Step 5 (parity)** — Full regime matrix with all flags ON; diff vs golden → **byte-identical**.
- [ ] **Step 6** — Commit: `perf(engine): cross-sectional daily scorer — O(1) numpy over the universe`.

### Phase 4 — numba exits (optional) + default flip

**Files:** `scripts/parity_numba_exits.py` (run only); Modify `backtest_regimes.sh`, `auto_tune_all.sh`.

- [ ] **Step 1** — Run `scripts/parity_numba_exits.py` to completion; record bit-identical PASS/FAIL. If FAIL, leave `USE_NUMBA_EXITS` OFF and stop here for numba.
- [ ] **Step 2** — On WSL, `pip install numba` only if Step 1 passed and the exit loop is a measured bottleneck.
- [ ] **Step 3 (default flip)** — Update `backtest_regimes.sh` to call `run_regimes.py --parallel 10 --vectorized --mp-context fork` with `USE_SHARED_BANK=1 USE_CROSS_SECTION=1` exported; update `auto_tune_all.sh` to export `USE_VECTORIZED_SCORING=1` (+ bank). Only after §3 gate green.
- [ ] **Step 4** — Final full-matrix run via the updated wrapper; diff vs golden → byte-identical; record wall-time before/after.
- [ ] **Step 5** — Commit: `perf: enable fork+bank+cross-section by default — matrix byte-identical, <Xh`.

### Phase 5 (separate spec) — universe scaling: incremental cache + batched fetch

Lever 5. Carries its own golden re-baseline because expanding the universe legitimately changes results (new tickers can enter top-N). Spec separately so the "no regression" gate above stays meaningful (it proves *engine* invariance; *universe* changes are an explicit, approved product change per CLAUDE.md baseline-lock rules).

---

## 5. Expected outcome

| Stage | Mechanism | Wall (regime matrix, indicative) | Result invariance |
|---|---|---|---|
| Baseline | `--parallel 4 --vectorized` | ~4h | — |
| +Phase 1 | fork + parallel 10 | ~1.6–2.0h | byte-identical |
| +Phase 2 | shared indicator bank | ~0.9–1.2h | byte-identical |
| +Phase 3 | cross-section daily | ~0.7–1.0h | byte-identical |
| Scaling | universe ×2 tickers | Phases 1–3 keep it near-flat where Python loop would double | (new universe = approved product change) |

Numbers are directional (no profiler run yet); the **invariance column is the hard guarantee** — enforced by the §3 golden-master gate, not by estimate.

---

## 6. Risks & open questions

- **FP tie-flips in cross-section ranking** — the single most likely regression vector. Mitigated by tie-stable argsort + picks-level parity (Phase 3 Step 1). If a flip is found, the parity script catches it *before* merge.
- **fork unavailable off-Linux** — `--mp-context spawn` retained as fallback; both proven against golden (Phase 1 Step 5).
- **Bank memory for a large universe** — the bank holds several series per ticker. For ×3 universe, confirm parent RSS stays within WSL 10 GB *before* forking (workers add little under COW). If tight, build the bank lazily per ticker-shard.
- **numba parity** — unverified at spec time; Phase 4 gates on it, never assumed.
- **Auto-tuner** — same architecture (vectorized + bank, precompute-once-slice-folds) cuts the nightly 216 backtests; tracked as a follow-on once the regime path is proven, since the tuner currently runs the *slow* path entirely (`USE_VECTORIZED_SCORING` never set).

---

## 7. Self-review

- **Spec coverage:** Levers 1–5 each map to a phase (1→P1, 2→P2, 3→P3, 4→P4, 5→P5). Regression protocol → §3, enforced in every phase's parity step. ✔
- **Placeholders:** `build_bank` body is sketched, not full — flagged as the implementer's task with exact inputs/outputs and the parity test that pins its behavior (Phase 2 Step 1). No "TODO/handle edge cases" left in the gates. ✔
- **Consistency:** flag names (`USE_SHARED_BANK`, `USE_CROSS_SECTION`, `USE_NUMBA_EXITS`, `USE_VECTORIZED_SCORING`), function names (`collect_specs`, `build_bank`, `assemble_precompute`), and the golden-diff gate are used identically across §2 and §4. ✔
