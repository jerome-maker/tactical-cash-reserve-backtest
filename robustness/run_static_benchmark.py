# -*- coding: utf-8 -*-
"""
The static cash benchmark.

The tactical rule holds cash and releases it at a defined correction. The
obvious question a reader should ask is whether the release matters at all: a
portfolio that simply holds the same average amount of cash, and never times
anything, is the benchmark the timing element has to beat.

To make that comparison exact rather than approximate, the static portfolio's
target weight is solved so that its realised average cash weight equals the
tactical strategy's, market by market and period by period. A third line --
the same reserve accumulated but never deployed -- isolates what the trigger
itself contributes.

Writes static_benchmark_results.csv.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

import common as C


def realised_avg_cash(target, price, cash_rate, weekly_contribution):
    bt = C.run_constant_mix(price, cash_rate, target_cash_weight=target,
                            weekly_contribution=weekly_contribution)
    return (bt["cash_balance"] / bt["total_value"]).mean(), bt


def match_average_cash(price, cash_rate, weekly_contribution, target_avg):
    """Solve for the constant-mix weight whose realised average cash weight equals target_avg."""
    def f(w):
        return realised_avg_cash(w, price, cash_rate, weekly_contribution)[0] - target_avg
    lo, hi = 1e-6, min(0.9, target_avg * 4 + 0.05)
    w = brentq(f, lo, hi, xtol=1e-8)
    avg, bt = realised_avg_cash(w, price, cash_rate, weekly_contribution)
    return w, avg, bt


def main():
    markets = C.build_markets()
    rows = []
    for label, cfg in markets.items():
        price, rate, wc = cfg["price"], cfg["cash_rate"], cfg["weekly_contribution"]

        bench = C.run_backtest(price, rate, alloc_ratio=0.0, weekly_contribution=wc)
        tact = C.run_backtest(price, rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                              weekly_contribution=wc)
        # same reserve, never released: deploy_fraction=0 switches the trigger off
        hoard = C.run_backtest(price, rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                               weekly_contribution=wc, deploy_fraction=0.0)

        m_t = C.compute_metrics(tact, price, rate, "Cash reserve, 20 percent")
        target_avg = m_t["Avg. Cash Weight"]
        w, avg, static = match_average_cash(price, rate, wc, target_avg)
        print(f"{label}: matched constant-mix weight {w:.4%} -> realised average cash "
              f"{avg:.4%} (tactical {target_avg:.4%})")

        for name, bt in [("Invest every contribution", bench),
                         ("Cash reserve, 20 percent", tact),
                         ("Static cash, matched average weight", static),
                         ("Reserve accumulated, never deployed", hoard)]:
            m = C.compute_metrics(bt, price, rate, name)
            m["Market"] = label
            m["Period"] = "Full sample"
            rows.append(m)

        # design and held-out periods, same four lines
        for period, start, end in [("Design 2009-2018", None, C.DESIGN_PERIOD_END),
                                   ("Held out 2019-2026", C.OOS_PERIOD_START, None)]:
            for name, bt in [("Invest every contribution", bench),
                             ("Cash reserve, 20 percent", tact),
                             ("Static cash, matched average weight", static),
                             ("Reserve accumulated, never deployed", hoard)]:
                sub = bt.loc[start:end]
                loc = bt.index.get_loc(sub.index[0])
                iv, idate = (bt.iloc[loc - 1]["total_value"], bt.index[loc - 1]) if loc else (None, None)
                m = C.compute_metrics(sub, price.loc[sub.index], rate.loc[sub.index], name,
                                      initial_value=iv, initial_date=idate)
                m["Market"] = label
                m["Period"] = period
                rows.append(m)

    df = pd.DataFrame(rows)[["Market", "Period", "Strategy", "CAGR (XIRR)", "MDD",
                             "Sortino Ratio", "Avg. Cash Weight", "Trigger Count",
                             "Terminal Value"]]
    df.to_csv("static_benchmark_results.csv", index=False)

    pd.set_option("display.width", 200)
    for period in df["Period"].unique():
        print(f"\n=== {period} ===")
        sub = df[df["Period"] == period].set_index(["Market", "Strategy"])
        print(sub[["CAGR (XIRR)", "MDD", "Sortino Ratio", "Avg. Cash Weight"]]
              .apply(lambda c: c.map(lambda v: f"{v:.4%}" if abs(v) < 1 else f"{v:.4f}")))

    print("\n=== gaps against 'invest every contribution', percentage points ===")
    for (mk, per), g in df.groupby(["Market", "Period"], sort=False):
        b = g[g["Strategy"] == "Invest every contribution"].iloc[0]
        for _, r in g.iterrows():
            if r["Strategy"] == "Invest every contribution":
                continue
            print(f"{mk:20s} {per:18s} {r['Strategy']:36s} "
                  f"CAGR {100*(r['CAGR (XIRR)']-b['CAGR (XIRR)']):+.3f}  "
                  f"MDD {100*(r['MDD']-b['MDD']):+.3f}  "
                  f"Sortino {r['Sortino Ratio']-b['Sortino Ratio']:+.4f}")


if __name__ == "__main__":
    main()
