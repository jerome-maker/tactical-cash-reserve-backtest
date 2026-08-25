# -*- coding: utf-8 -*-
"""
South Africa: the third market.

The Satrix 40 ETF (STX40.JO) tracks the FTSE/JSE Top 40 and plays the same role
in South Africa that 0050 plays in Taiwan and SPY plays in the United States --
the country's established broad-market index fund for retail investors. Its
usable price history begins on 2 January 2009, the same day Taiwan's does, so
the third market fits the existing common analysis window exactly and forces no
change to the sample period.

The cash reserve earns South Africa's published deposit rate: the IMF's
International Financial Statistics monthly deposit rate for South Africa
(series M.ZA.FIDR_PA), which is the same concept the Bank of Taiwan and FDIC
series capture for the other two markets. IFS publishes it through February
2025; the remainder of the sample is held at the World Bank's 2025 annual
deposit interest rate for South Africa (WDI series FR.INR.DPST, 7.42 percent),
which for overlapping years is the annual mean of the same IFS monthly series.

Unlike Taiwan and the United States, South African deposit rates are high --
they average 6.3 percent over the sample against 0.8 and 0.6 percent -- so this
market tests the rule where holding cash actually pays something.
"""
import os

import numpy as np
import pandas as pd

import common as C
from pbo_cscv import clean_price_series

DATA = os.path.join(C.PROJECT_ROOT, "data")
SA_PRICE = os.path.join(DATA, "stx40_total_return_daily.csv")
SA_RATE = os.path.join(DATA, "zar_deposit_rate.csv")
MONTHLY_CONTRIBUTION_ZAR = 10_000
WEEKLY_CONTRIBUTION_ZAR = MONTHLY_CONTRIBUTION_ZAR * 12 / 52


def repair_isolated_spikes(price, drop_threshold=-0.5, verbose=True):
    """
    Repair single-day vendor misprints of the round-trip kind: one observation
    collapses by more than `drop_threshold` and the very next observation undoes
    it almost exactly. clean_price_series() cannot handle these -- it treats a
    jump as a split and back-adjusts the whole history before it -- so they have
    to be repaired first.

    The bad observation is replaced by the geometric mean of its neighbours.
    Every repair is printed, because silently rewriting price data is exactly the
    sort of thing a reader is entitled to see.
    """
    price = price.copy()
    ret = price.pct_change()
    repairs = []
    for i in range(1, len(price) - 1):
        if ret.iloc[i] < drop_threshold and ret.iloc[i + 1] > 1.0:
            before, after = price.iloc[i - 1], price.iloc[i + 1]
            round_trip = (1 + ret.iloc[i]) * (1 + ret.iloc[i + 1]) - 1
            if abs(round_trip) < 0.10:          # the pair really does cancel
                fixed = float(np.sqrt(before * after))
                repairs.append((price.index[i], float(price.iloc[i]), fixed))
                price.iloc[i] = fixed
    if verbose and repairs:
        for d, old, new in repairs:
            print(f"[data repair] {d.date()}: isolated misprint {old:,.4f} -> {new:,.4f} "
                  f"(neighbours bracket it; ratio {new / old:,.1f}x)")
    return price, repairs


def load_sa_market(end=None):
    price_daily = pd.read_csv(SA_PRICE, index_col=0, parse_dates=True)["price"]
    price_daily, _ = repair_isolated_spikes(price_daily)
    price_daily, corrections = clean_price_series(price_daily)
    if corrections:
        print(f"[data cleaning] STX40.JO: {len(corrections)} jump(s) back-adjusted")
    if end is not None:
        price_daily = price_daily.loc[:end]
    price = price_daily.resample("W-FRI").last().dropna()
    price.name = "price"
    rate = C.load_cash_rate_series(price.index, SA_RATE)
    return dict(price=price, cash_rate=rate,
                weekly_contribution=WEEKLY_CONTRIBUTION_ZAR, currency="R", slug="south_africa")


def build_three_markets():
    """The two published markets plus South Africa, on the identical window."""
    markets = C.build_markets()
    end = max(m["price"].index.max() for m in markets.values())
    sa = load_sa_market(end=end)
    common_start = max(m["price"].index.min() for m in list(markets.values()) + [sa])
    for m in list(markets.values()) + [sa]:
        assert m["price"].index.min() <= common_start + pd.Timedelta(days=7), "window mismatch"
    markets["South Africa (STX40)"] = sa
    print(f"South Africa weekly series: {len(sa['price'])} weeks "
          f"({sa['price'].index.min().date()} to {sa['price'].index.max().date()}), "
          f"average deposit rate {sa['cash_rate'].mean():.2%}")
    return markets


if __name__ == "__main__":
    m = build_three_markets()
    sa = m["South Africa (STX40)"]
    for label, cfg in m.items():
        bench = C.run_backtest(cfg["price"], cfg["cash_rate"], alloc_ratio=0.0,
                               weekly_contribution=cfg["weekly_contribution"])
        tact = C.run_backtest(cfg["price"], cfg["cash_rate"], alloc_ratio=0.20,
                              weekly_contribution=cfg["weekly_contribution"])
        b = C.compute_metrics(bench, cfg["price"], cfg["cash_rate"])
        t = C.compute_metrics(tact, cfg["price"], cfg["cash_rate"])
        print(f"{label:22s} bench CAGR {b['CAGR (XIRR)']:.4%} MDD {b['MDD']:.4%} "
              f"Sortino {b['Sortino Ratio']:.4f} | tactical gap "
              f"CAGR {100*(t['CAGR (XIRR)']-b['CAGR (XIRR)']):+.3f}pp "
              f"MDD {100*(t['MDD']-b['MDD']):+.3f}pp "
              f"Sortino {t['Sortino Ratio']-b['Sortino Ratio']:+.4f} "
              f"| triggers {int(t['Trigger Count'])} cash {t['Avg. Cash Weight']:.2%}")
