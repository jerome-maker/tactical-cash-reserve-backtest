# -*- coding: utf-8 -*-
"""
United States, 1993 to 2026: the sample the main analysis has to leave out.

The main results are confined to 2009 onwards because Taiwan's usable price
history starts there, and the US series is trimmed to match so the two markets
are compared over identical dates. That exclusion removes the dot-com bear
market and the 2008 global financial crisis -- precisely the episodes a rule
that waits for corrections is supposed to be built for.

SPY has traded since January 1993, so the US side can be run over its full
history as a supplementary analysis. The cash reserve still earns a published
savings-deposit rate, spliced from three official/industry sources:

  1993-01-29 to 2009-05-23   Bankrate Monitor national savings account APY
                             (FRED series BRMSA0102, semiannual, held constant
                             between observations)
  2009-05-24 to 2021-03-31   FDIC national rate on savings deposits, pre-2021
                             methodology (weekly)
  2021-04-01 onwards         FDIC national rate on savings deposits, Final Rule
                             methodology, via FRED series SNDR (monthly)

At the 2009 join the two sources agree closely (Bankrate 0.23 percent for the
second half of 2009 against the FDIC's 0.22 percent at 24 May 2009), so the
splice introduces no visible step.

Writes us_long_sample_results.csv and data/usd_savings_rate_long.csv.
"""
import os

import numpy as np
import pandas as pd

import common as C
from pbo_cscv import clean_price_series
from run_static_benchmark import match_average_cash

DATA = os.path.join(C.PROJECT_ROOT, "data")
END = "2026-08-03"          # same end date as the main analysis
SPLICE = pd.Timestamp("2009-05-24")


def build_long_rate_series():
    br = pd.read_csv(os.path.join(DATA, "bankrate_savings_apy_semiannual.csv"),
                     parse_dates=["observation_date"])
    br = br.rename(columns={"observation_date": "date", "BRMSA0102": "annual_rate"})
    br["annual_rate"] = br["annual_rate"] / 100.0
    br = br[br["date"] < SPLICE]

    fdic = pd.read_csv(os.path.join(DATA, "usd_savings_rate.csv"), parse_dates=["date"])

    out = pd.concat([br, fdic], ignore_index=True).sort_values("date")
    path = os.path.join(DATA, "usd_savings_rate_long.csv")
    out.to_csv(path, index=False)
    print(f"[saved] {path}: {out['date'].min().date()} to {out['date'].max().date()}, "
          f"{len(out)} observations ({len(br)} Bankrate + {len(fdic)} FDIC/FRED)")
    return path


def main():
    rate_path = build_long_rate_series()

    price_daily = pd.read_csv(os.path.join(DATA, "spy_total_return_daily_1993.csv"),
                              index_col=0, parse_dates=True)["price"]
    price_daily, corrections = clean_price_series(price_daily)
    if corrections:
        print(f"[data cleaning] {len(corrections)} jump(s) back-adjusted")
    price_daily = price_daily.loc[:END]
    price = price_daily.resample("W-FRI").last().dropna()
    price.name = "price"
    rate = C.load_cash_rate_series(price.index, rate_path)
    wc = C.WEEKLY_CONTRIBUTION_USD
    print(f"United States long sample: {price.index.min().date()} to {price.index.max().date()} "
          f"({len(price)} weeks, {len(price)/52:.1f} years)")

    bench = C.run_backtest(price, rate, alloc_ratio=0.0, weekly_contribution=wc)
    tact = C.run_backtest(price, rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                          weekly_contribution=wc)
    m_t_full = C.compute_metrics(tact, price, rate)
    w, avg, static = match_average_cash(price, rate, wc, m_t_full["Avg. Cash Weight"])
    print(f"matched constant-mix weight {w:.4%} -> realised average cash {avg:.4%}")

    runs = [("Invest every contribution", bench),
            ("Cash reserve, 20 percent", tact),
            ("Static cash, matched average weight", static)]

    periods = [("Full sample 1993-2026", None, None),
               ("Excluded era 1993-2008", None, "2008-12-31"),
               ("Main-analysis era 2009-2026", "2009-01-01", None)]

    rows = []
    for pname, start, end in periods:
        for name, bt in runs:
            sub = bt.loc[start:end]
            loc = bt.index.get_loc(sub.index[0])
            iv, idate = (bt.iloc[loc - 1]["total_value"], bt.index[loc - 1]) if loc else (None, None)
            m = C.compute_metrics(sub, price.loc[sub.index], rate.loc[sub.index], name,
                                  initial_value=iv, initial_date=idate)
            m["Period"] = pname
            rows.append(m)

    # allocation-ratio sweep over the long sample: is the trade-off still smooth?
    sweep = []
    for ratio in C.SENSITIVITY_ALLOC_RATIOS:
        bt = C.run_backtest(price, rate, alloc_ratio=ratio, weekly_contribution=wc)
        m = C.compute_metrics(bt, price, rate, f"{ratio:.0%}")
        sweep.append({"Reserve allocation ratio": f"{ratio:.0%}",
                      "CAGR (XIRR)": m["CAGR (XIRR)"], "MDD": m["MDD"],
                      "Sortino Ratio": m["Sortino Ratio"]})

    df = pd.DataFrame(rows)[["Period", "Strategy", "CAGR (XIRR)", "MDD", "Sortino Ratio",
                             "Avg. Cash Weight", "Trigger Count", "Terminal Value"]]
    df.to_csv("us_long_sample_results.csv", index=False)
    pd.DataFrame(sweep).to_csv("us_long_sample_sensitivity.csv", index=False)

    for pname in df["Period"].unique():
        g = df[df["Period"] == pname]
        b = g[g["Strategy"] == "Invest every contribution"].iloc[0]
        print(f"\n=== {pname} ===")
        for _, r in g.iterrows():
            gap = "" if r["Strategy"] == b["Strategy"] else (
                f" | gap CAGR {100*(r['CAGR (XIRR)']-b['CAGR (XIRR)']):+.3f}pp"
                f"  MDD {100*(r['MDD']-b['MDD']):+.3f}pp"
                f"  Sortino {r['Sortino Ratio']-b['Sortino Ratio']:+.4f}")
            print(f"  {r['Strategy']:36s} CAGR {r['CAGR (XIRR)']:7.4%}  MDD {r['MDD']:8.4%}  "
                  f"Sortino {r['Sortino Ratio']:.4f}  triggers {int(r['Trigger Count'])}{gap}")

    print("\n=== allocation-ratio sweep, full 1993-2026 sample ===")
    for s in sweep:
        print(f"  {s['Reserve allocation ratio']:>4s}  CAGR {s['CAGR (XIRR)']:.4%}  "
              f"MDD {s['MDD']:.4%}  Sortino {s['Sortino Ratio']:.4f}")


if __name__ == "__main__":
    main()
