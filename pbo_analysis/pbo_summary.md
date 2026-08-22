# Probability of Backtest Overfitting (PBO) — supplementary analysis

**Status: integrated into `submission_JAM/manuscript.tex`.** The PBO/CSCV results and the
regime-shift explanation below are now reported in the manuscript's "Sensitivity analysis:
cash-allocation ratio" subsection (Table 4, Figure 5), with supporting updates to "Related
literature" (Backtest robustness), "Research design," "Methodological limitations," and the
Discussion's limitations and future-research paragraphs. The manuscript's write-up is
condensed relative to this file, which retains the full investigative detail (including the
superseded dispersion hypothesis) for the record.

## What this fulfills

`submission_JAM/manuscript.tex` (Discussion and conclusions → future research) states:

> "Replacing the single in-sample/out-of-sample split with rolling walk-forward analysis,
> combined with formal overfitting-adjusted statistics such as the Deflated Sharpe Ratio
> \parencite{bailey2014} or the **Probability of Backtest Overfitting**
> \parencite{baileyetal2016pbo}, would provide a more rigorous statistical foundation than the
> smoothness-based diagnostic used here."

This analysis runs the PBO half of that promise, via Combinatorially Symmetric
Cross-Validation (CSCV), following Bailey, Borwein, López de Prado & Zhu (2016), *"The
Probability of Backtest Overfitting,"* *Journal of Computational Finance* 20(4):39–69 — the
exact citation already present in `references.bib` as `baileyetal2016pbo`.

## Methodology

- **Trial set (N=5):** the same five cash-allocation ratios already reported in the manuscript's
  Table 3 — 10% / 20% / 30% / 40% / 50%. No new parameter grid was introduced; this is a
  formal re-examination of an analysis the paper already performs, not a new source of
  overfitting risk in itself.
- **Data, constants, and backtest engine:** copied verbatim (constants and function bodies) from
  `tactical_cash_reserve_backtest.ipynb` (cells 2, 4, 7, 9, 11) — same correction threshold (15%),
  lookback (52wk), cash cap (20%), transaction cost (0.15%), same common Taiwan/US date window
  (2009-01-02 to 2026-08-03), read directly from the cached data files. See `pbo_cscv.py`.
- **Performance metric per trial:** weekly NAV return (`total_value / cumulative_contribution`,
  the same NAV definition the paper already uses for MDD), in excess of the concurrent
  cash-reserve rate — consistent with the paper's own Sortino-ratio MAR convention — annualized
  into a Sharpe ratio.
- **CSCV procedure:** partition the weekly excess-return series into `S` contiguous equal blocks;
  for every way of choosing `S/2` blocks as in-sample (IS) and the rest as out-of-sample (OOS),
  find the IS-best trial by Sharpe ratio, then find that trial's **rank** among the N=5 OOS Sharpe
  ratios. PBO = the fraction of combinations where the IS winner finishes at or below the OOS
  median rank. Run at S=16 (headline) and S=10/S=20 (stability check).

## Results

| Market | S | Combinations | Weeks used (dropped) | **PBO** | Mean λ | P(OOS loss) |
|---|---:|---:|---:|---:|---:|---:|
| Taiwan (0050) | 10 | 252 | 900 (5) | 49.2% | +0.073 | 1.6% |
| Taiwan (0050) | **16** | **12,870** | **896 (9)** | **47.9%** | **+0.164** | **0.0%** |
| Taiwan (0050) | 20 | 184,756 | 900 (5) | 59.0% | −0.091 | 0.8% |
| United States (SPY) | 10 | 252 | 910 (8) | 91.3% | −0.952 | 0.0% |
| United States (SPY) | **16** | **12,870** | **912 (6)** | **94.9%** | **−1.047** | **0.0%** |
| United States (SPY) | 20 | 184,756 | 900 (18) | 94.6% | −1.163 | 0.9% |

Full detail in `pbo_results.csv`. Diagnostic plots: `pbo_distribution_taiwan.pdf`,
`pbo_distribution_us.pdf` (S=16 headline; with N=5 trials the logit λ is inherently discrete —
exactly 5 possible values, one per possible OOS rank — plotted as a labelled bar chart rather
than a continuous histogram for clarity).

## Interpretation — the two markets tell different stories, and this should be reported honestly

**Taiwan: PBO ≈ 48–59% across S — statistically indistinguishable from a coin flip.** In-sample
Sharpe ranking among the five allocation ratios carries essentially no information about their
out-of-sample rank. This is **consistent with, and reinforces**, the manuscript's existing
"smooth curve, no isolated spike" conclusion (Section 4.2): if the five candidates are this close
to interchangeable, it follows naturally that picking the in-sample "best" one shouldn't reliably
predict anything out-of-sample either — which is exactly why the paper already declines to treat
any single ratio as "optimal."

**United States: PBO ≈ 91–95% across S — high and stable, not a coin flip.** The in-sample-preferred
allocation ratio finishes **below the OOS median 91–95% of the time**, and finishes dead last
(worst of 5) in 57.3% of all 12,870 combinations at S=16. This is a **genuine, robust overfitting
warning signal** for the US-side allocation-ratio sensitivity analysis specifically. It does not
mean the paper's headline US result (Table 2, Table 6) is wrong — that result does not depend on
selecting a "best" allocation ratio in-sample, it simply reports all five — but it does mean that
**if a reader were tempted to treat one particular US allocation ratio as the empirically
"validated" choice on the strength of the sensitivity analysis, CSCV says that impulse is not well
supported by the data.**

**A hypothesized mechanical contributor — since tested and superseded, see the follow-up section
directly below.** (Original note, kept for the record: in both markets the 50%-allocation trial is
chosen as the in-sample winner far more often than the other four combined — Taiwan: 8,154 of
12,870 combinations at S=16, ≈63%; United States: 5,928 of 12,870, ≈46% — well above the ~20% a
symmetric split would suggest, and it was initially guessed that this reflected higher block-level
Sharpe *dispersion* for the 50% trial. That guess turned out to be wrong; the actual mechanism is
described below.)

## Follow-up investigation: why is the 50% trial over-selected? (`diagnose_50pct_volatility.py`)

The "likely mechanical contributor" hypothesized above — that the 50% trial's block-level Sharpe
ratio is *more dispersed* across blocks, making it spike as the IS winner by chance — was
**tested directly and is not correct**. Per-block annualized Sharpe was computed for all five
trials over the same S=16 partition (`diagnostic_block_sharpe_{taiwan,us}.csv`,
`diagnostic_block_sharpe_{taiwan,us}.pdf`). The cross-block standard deviation of Sharpe is
essentially **identical across all five trials** in both markets:

- Taiwan: 1.025 (10%) vs. 1.023 (50%) — a 0.002 difference, not meaningfully different.
- United States: 0.574 (10%) vs. **0.569 (50%) — the 50% trial's dispersion is actually the
  *lowest* of the five**, the opposite of the original hypothesis.

**The real mechanism is a chronological regime shift, not noise-driven dispersion.** Ranking the
five trials by block-level Sharpe reveals a clean structural break in the United States data:

| United States, block-level Sharpe winner | Date range |
|---|---|
| 10% wins 7 of blocks 1–9 (blk2→30%, blk3→50% are the only exceptions) | 2009-01 to 2018-11 |
| **50% wins blocks 10–16 without exception** (blk12 ties 40%/50%) | **2018-11 to 2026-06** |

The break point — between block 9 (ending 2018-11-02) and block 10 (starting 2018-11-09) — falls
almost exactly on the manuscript's own design-period/out-of-sample boundary (2018-12-31 /
2019-01-01, chosen for entirely separate reasons in Section 3.1). Taiwan shows **no comparable
clean break**: its block-level winners interleave throughout the full sample (10%, 10%, 50%, 10%,
10%, 10%, 50%, 10%, 10%, 50%, 50%, 10%, 50%, 10%, 50%, 50% — blocks 1–16), which is the direct
explanation for why Taiwan's PBO sits near a coin flip while the United States' does not: CSCV
draws blocks into IS/OOS combinations at random, and whenever a combination happens to weight IS
toward the pre-2019 blocks, the "IS-best" trial (typically 10%) reflects a regime that does not
carry into an OOS half weighted toward the post-2019 blocks (where 50% dominates) — a genuine,
reproducible structural instability, not sampling noise.

This connects directly to a result the manuscript **already reports** in Table 4: the United
States' design-period CAGR gap (−0.56 pp) is roughly *28 times* the size of its out-of-sample gap
(−0.02 pp), by far the largest design/OOS swing of either market (Taiwan's corresponding gap only
narrows from −0.15 pp to −0.03 pp). The PBO finding gives a mechanistic account of *why*: the
pre-2019 US sample was a comparatively steady bull run in which holding a cash reserve was mostly
a drag, while 2019–2026 contained the COVID-19 crash and the 2022 rate-hike correction, both of
which rewarded a larger reserve — so the strategy's relative standing genuinely changed regime
around exactly the point the manuscript already splits its sample. Full per-block detail (Sharpe
ratios, trigger counts, deployed amounts, cash weights) is in `diagnostic_block_sharpe_us.csv` and
`diagnostic_block_deploys_us.csv` (and the Taiwan equivalents).

## Recommendation

The mechanism is now understood well enough to write up with confidence, not just flagged as an
open question. This is substantive enough that it should not be silently omitted if the paper's
overfitting-control narrative is revisited. Suggested framing (no changes made to
`manuscript.tex` by this analysis — this is the recommendation, not yet applied):

1. **Report the PBO figures for both markets transparently**, as an addition to Section 4.5 / the
   robustness discussion, alongside the existing smoothness check — framed as "necessary but not
   sufficient": the United States sensitivity curve already passes the smoothness check (no
   isolated spike, per the manuscript's own text) yet still carries a 94.9% probability of backtest
   overfitting under CSCV, which the smoothness heuristic alone cannot detect.
2. **Explain the United States result mechanistically, using the regime-shift finding above**,
   rather than leaving PBO=94.9% as an unexplained number: the pre-2019 sample favored low
   allocation, the 2019–2026 sample (COVID-19 crash, 2022 correction) favored high allocation, and
   this split falls almost exactly at the manuscript's own design/OOS boundary — which also
   explains, mechanistically, the large design-vs-OOS CAGR-gap swing the paper already reports for
   the United States in Table 4 (−0.56 pp → −0.02 pp). This turns a potentially alarming diagnostic
   number into a substantive, well-supported methodological insight: cross-sectional resampling
   (CSCV) and chronological out-of-sample validation are answering related but distinct questions,
   and combining both gives a fuller picture than either alone.
3. **Note the contrast with Taiwan explicitly** (PBO ≈ 48–59%, no comparable regime break, winners
   interleave throughout the sample) as evidence that the United States result is a genuine
   market-specific finding, not a general property of the strategy or the CSCV method itself.

## Files in this folder

- `pbo_cscv.py` — PBO/CSCV implementation (data loading, backtest engine, CSCV/PBO core, plotting).
- `pbo_results.csv` — all six (market × S) PBO result rows.
- `pbo_distribution_taiwan.pdf`, `pbo_distribution_us.pdf` — S=16 logit-distribution bar charts.
- `diagnose_50pct_volatility.py` — follow-up script investigating the 50%-trial over-selection.
- `diagnostic_block_sharpe_{taiwan,us}.csv` — per-block (S=16) annualized Sharpe ratio, all 5 trials.
- `diagnostic_block_sharpe_{taiwan,us}.pdf` — the same, plotted chronologically.
- `diagnostic_block_deploys_{taiwan,us}.csv` — per-block trigger count, $ deployed, avg. cash
  weight, all 5 trials (used to rule out the deployment-driven-jump hypothesis).
- `pbo_summary.md` — this file.
