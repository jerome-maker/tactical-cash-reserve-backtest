# -*- coding: utf-8 -*-
"""
Where does the drawdown improvement come from?

The underwater chart (Fig. 3 of the paper) shows two lines that sit on top of
each other most of the time and separate in a few corrections. This script
measures the separation episode by episode, so the text describing the figure
reads its numbers from data rather than from the eye: for each correction
window, the deepest fall of the contribution-adjusted portfolio value below
its previous high, for the benchmark and for the 20 percent cash reserve.

Writes drawdown_episodes.csv.
"""
import os

import pandas as pd

import common as C
from south_africa import build_three_markets

EPISODES = [("2011-2012", "2011-01-01", "2012-12-31"),
            ("2015-2016", "2015-06-01", "2016-06-30"),
            ("Late 2018", "2018-09-01", "2019-01-31"),
            ("March 2020", "2020-02-01", "2020-06-30"),
            ("2022", "2022-01-01", "2022-12-31"),
            ("2025", "2025-01-01", "2025-06-30")]


def underwater(bt):
    nav = bt["total_value"] / bt["cumulative_contribution"]
    return nav / nav.cummax() - 1.0


def main():
    rows = []
    for label, cfg in build_three_markets().items():
        price, rate, wc = cfg["price"], cfg["cash_rate"], cfg["weekly_contribution"]
        bench = C.run_backtest(price, rate, alloc_ratio=0.0, weekly_contribution=wc)
        tact = C.run_backtest(price, rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                              weekly_contribution=wc)
        ub, ut = underwater(bench), underwater(tact)
        first = tact.index[tact["triggered"]]
        for name, start, end in EPISODES:
            b, t = 100 * ub[start:end].min(), 100 * ut[start:end].min()
            rows.append({"Market": label, "Episode": name, "Benchmark trough (pct)": b,
                         "Reserve trough (pct)": t, "Improvement (pp)": t - b,
                         "First trigger": str(first[0].date()) if len(first) else ""})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.HERE, "drawdown_episodes.csv"), index=False)
    print(df.to_string(index=False, float_format=lambda v: "{:.2f}".format(v)))


if __name__ == "__main__":
    main()
