# Cash Reserve Backtest: Taiwan, the United States and South Africa

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22084641.svg)](https://doi.org/10.5281/zenodo.22084641)

Reproducible backtest code and cached data for a study that prices a tactical cash reserve rule
against dollar-cost averaging in three markets chosen because their deposit rates differ by more
than an order of magnitude: Taiwan (0050), the United States (SPY) and South Africa (STX40).
This repository is the accompanying paper's data-and-code availability reference: every table and
figure in the manuscript can be regenerated from the files here.

The organising result is that holding a reserve costs the deposit spread applied to the idle cash
-- the gross price -- and the deployment trigger recovers most of it: 90% in Taiwan, 61% in the
United States and more than all of it in South Africa, where deposits pay 6.4% and the reserve
ends up adding return rather than costing it.

The manuscript itself is under double-blind peer review, so neither it nor the target journal
is named here; the submitted version is not distributed from this repository.

## What this strategy is

A "buy the dip" rule, formalized: instead of investing 100% of each periodic contribution
directly into a broad-market ETF (dollar-cost averaging, DCA), a fixed share of each contribution
is diverted into an interest-bearing cash reserve. When the ETF's price falls at least 15% from
its trailing 52-week high, the entire reserve is deployed into the ETF in a single lump-sum
purchase. The paper backtests this rule against 100% DCA in two markets under an identical
research design:

- **Taiwan**, via the Yuanta Taiwan Top 50 ETF (`0050.TW`), with a New Taiwan Dollar savings
  deposit rate (Bank of Taiwan) as the cash-reserve rate.
- **United States**, via the SPDR S&P 500 ETF Trust (`SPY`), with the FDIC's National Rate on
  Savings Deposits as the cash-reserve rate, extended back to 1993 with the Bankrate Monitor
  national savings account rate for the long-sample check.
- **South Africa**, via the Satrix 40 ETF (`STX40.JO`), with the IMF International Financial
  Statistics deposit rate for South Africa as the cash-reserve rate.

Every fixed parameter -- the correction threshold, lookback window, cash-reserve cap, deployment
style, transaction cost, and the design/out-of-sample date split -- is identical in both markets;
only the cash-allocation ratio is varied for sensitivity analysis. See the paper for the full
methodology and results.

## Repository structure

```
.
├── tactical_cash_reserve_backtest.ipynb   # Main notebook -- start here
├── data/                                  # Cached price and interest-rate series
│   ├── 0050_total_return_daily.csv        #   0050.TW, auto-adjusted daily close (Yahoo Finance)
│   ├── spy_total_return_daily.csv         #   SPY, auto-adjusted daily close (Yahoo Finance)
│   ├── twd_savings_rate.csv               #   Bank of Taiwan savings deposit rate, {date, annual_rate}
│   ├── usd_savings_rate.csv               #   FDIC National Rate on Savings Deposits, {date, annual_rate}
│   ├── a13rate.xls                        #   Raw source for the TWD rate (Central Bank of Taiwan)
│   └── fdic_national_rate_archive.xlsx    #   Raw source for the pre-2021 portion of the USD rate (FDIC)
├── figures/
│   ├── make_figures.py                    # Regenerates the publication-quality PDF figures below
│   └── fig1-fig7_*.pdf                    # The manuscript's figures, as produced by that script
├── robustness/                            # Analyses added for the three-market version
│   ├── common.py                          #   Shared engine, plus the static-cash benchmark
│   ├── south_africa.py                    #   Third market, incl. isolated-misprint repair
│   ├── run_static_benchmark.py            #   Static cash holding the same average weight
│   ├── run_bootstrap.py                   #   Stationary block bootstrap + deflated Sharpe
│   ├── run_us_long_sample.py              #   United States 1993-2026, incl. 2008
│   ├── build_all_results.py               #   Writes results.json, the single source of numbers
│   └── results.json                       #   Every figure the manuscript reports
├── pbo_analysis/
│   ├── pbo_cscv.py                        # Probability of backtest overfitting (CSCV), both markets
│   ├── pbo_results.csv                    # PBO at 10, 16 and 20 blocks -- the stability check
│   ├── pbo_summary.md                     # Full write-up, including the late-2018 US regime shift
│   └── diagnostic_block_*.{csv,pdf}       # Block-by-block Sharpe winners behind that explanation
├── requirements.txt
└── LICENSE
```

## Reproducing the results

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace tactical_cash_reserve_backtest.ipynb
```

The notebook is organized to follow the manuscript's own section order (data, methodology,
backtest engine, then each results table/figure in sequence) rather than a generic exploratory
order, and every code cell is documented with the design rationale behind it, not just what it
does -- the goal is that a reader can follow the notebook alongside the paper's *Data and
methodology* and *Results* sections and reproduce every number independently.

Data is cached under `data/`, so a normal run does not require a network connection or re-fetch
from Yahoo Finance/FDIC/FRED; delete a cache file and re-run to force a fresh download if you want
to check current data instead of the exact snapshot used in the paper (note that Yahoo Finance's
history is occasionally revised retroactively, so a fresh download will not necessarily match the
paper bit-for-bit).

To regenerate the publication-quality vector-PDF figures used in the manuscript specifically
(journal styling, print resolution) rather than the notebook's lighter-weight inline plots:

```bash
cd figures && python make_figures.py
```

## Design principle: one engine, two markets

Every backtest function in the notebook is written to be market-agnostic: it takes a price series
and a cash-rate series as arguments and is called once for Taiwan and once for the United States,
with no separate code path for either market. Setting the allocation ratio to 0 in that same
function reproduces the 100% DCA control exactly -- the control is not a separately implemented
strategy, it is the tactical strategy's own code evaluated at its zero-allocation endpoint. This
is what lets the paper claim that any reported performance gap reflects the allocation decision
itself, and that any cross-market agreement in the results is not an artifact of the two markets
being analyzed with different code.

## License

Code and cached data are released under the MIT License (see `LICENSE`). Price data originates
from Yahoo Finance; interest-rate data originates from the Central Bank of Taiwan / Bank of Taiwan
and the U.S. Federal Deposit Insurance Corporation (FDIC) / Federal Reserve Bank of St. Louis
(FRED). Please consult those sources' own terms if you redistribute the raw data independently of
this repository.

## Status

This paper is currently under peer review. The results and figures here match the submitted
manuscript; they may be revised following review. The sample runs from 2 January 2009 to
3 August 2026 in both markets.
