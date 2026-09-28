# -*- coding: utf-8 -*-
"""
Within-market test of the mechanism: does the reserve cost more when deposits
pay less?

The paper's claim is that the price of holding a cash reserve is set by the
spread between what the equity earns and what the deposit pays. Across three
markets that claim rests on three data points, which is what the first editor
called descriptive. Deposit rates also move a great deal *within* each market --
South Africa's ranged from 3.7 to 11.3 percent over the sample and Taiwan's from
0.10 to 0.83 -- so the same mechanism can be tested without leaving the market,
and without the confounding a cross-country contrast carries.

Two designs, both on the weekly net asset value return difference between the
tactical strategy and the benchmark (the realised weekly cost of the reserve):

  1. Regime split. Weeks are divided at each market's median deposit rate. If
     the mechanism is real the deficit should be larger in the low-rate half,
     where the spread is wider. Tested with a Newey-West HAC t-test on the
     difference between the two halves.
  2. Regression. The weekly difference is regressed on the concurrent deposit
     rate, within market and then pooled with market fixed effects, with HAC
     (Newey-West) standard errors. A positive coefficient means the reserve
     costs less where deposits pay more, which is the paper's thesis stated as
     an estimated relationship rather than a contrast between cases.

Writes rate_regime_results.csv and rate_regime_regressions.csv.
"""
import numpy as np
import pandas as pd

import common as C
from run_inference import hac_cov, nw_lag, nav_excess
from south_africa import build_three_markets
from scipy.stats import t as student_t

ANNUALISE_PP = 5200.0      # weekly fraction -> percentage points a year


def hac_ols(y, X, names):
    """OLS with Newey-West standard errors. X must already include a constant."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    T, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    S = hac_cov(X * resid[:, None])          # HAC covariance of the score
    cov = XtX_inv @ (S * T) @ XtX_inv
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    p = 2.0 * (1.0 - student_t.cdf(np.abs(tstat), df=T - k))
    return pd.DataFrame({"term": names, "coefficient": beta, "std_error": se,
                         "t": tstat, "p_value": p})


def main():
    markets = build_three_markets()
    regime_rows, reg_rows = [], []
    pooled_y, pooled_rate, pooled_market = [], [], []

    for label, cfg in markets.items():
        price, rate, wc = cfg["price"], cfg["cash_rate"], cfg["weekly_contribution"]
        bench = C.run_backtest(price, rate, alloc_ratio=0.0, weekly_contribution=wc)
        tact = C.run_backtest(price, rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                              weekly_contribution=wc)

        d = (nav_excess(tact, rate) - nav_excess(bench, rate)).dropna()
        r = rate.reindex(d.index)
        cash_w = (tact["cash_balance"] / tact["total_value"]).reindex(d.index)

        med = float(r.median())
        low = r <= med                      # low deposit rate  = wide spread
        high = ~low

        for name, mask in (("Low deposit rate (wide spread)", low),
                           ("High deposit rate (narrow spread)", high)):
            sub = d[mask]
            regime_rows.append({
                "Market": label, "Regime": name,
                "Weeks": int(mask.sum()),
                "Mean deposit rate (%)": 100 * float(r[mask].mean()),
                "Mean cost (pp a year)": ANNUALISE_PP * float(sub.mean()),
                "Average cash weight (%)": 100 * float(cash_w[mask].mean()),
            })

        # HAC test of the gap between the two halves: cost_t = a + b * low_t
        lowd = low.astype(float).values
        X = np.column_stack([np.ones(len(d)), lowd])
        res = hac_ols(d.values * ANNUALISE_PP, X, ["constant", "low rate dummy"])
        res.insert(0, "Market", label)
        res.insert(1, "Specification", "Regime dummy")
        reg_rows.append(res)

        # cost_t = a + b * deposit_rate_t
        X = np.column_stack([np.ones(len(d)), r.values * 100.0])
        res = hac_ols(d.values * ANNUALISE_PP, X, ["constant", "deposit rate (pp)"])
        res.insert(0, "Market", label)
        res.insert(1, "Specification", "Deposit rate, within market")
        reg_rows.append(res)

        pooled_y.append(d.values * ANNUALISE_PP)
        pooled_rate.append(r.values * 100.0)
        pooled_market.append(np.full(len(d), label))

    # pooled regression with market fixed effects
    y = np.concatenate(pooled_y)
    rt = np.concatenate(pooled_rate)
    mk = np.concatenate(pooled_market)
    labels = sorted(set(mk))
    dummies = [(mk == m).astype(float) for m in labels[1:]]     # first market is the base
    X = np.column_stack([np.ones(len(y)), rt] + dummies)
    names = ["constant", "deposit rate (pp)"] + ["FE: " + m for m in labels[1:]]
    res = hac_ols(y, X, names)
    res.insert(0, "Market", "Pooled")
    res.insert(1, "Specification", "Deposit rate, market fixed effects")
    reg_rows.append(res)

    regimes = pd.DataFrame(regime_rows)
    regs = pd.concat(reg_rows, ignore_index=True)
    regimes.to_csv("rate_regime_results.csv", index=False)
    regs.to_csv("rate_regime_regressions.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== Regime split (weekly cost of the reserve, annualised) ===")
    print(regimes.to_string(index=False, float_format=lambda v: "{:.4f}".format(v)))
    print("\n=== Regressions (dependent variable: weekly cost in pp a year) ===")
    print(regs.to_string(index=False, float_format=lambda v: "{:.4f}".format(v)))
    print("\n[saved] rate_regime_results.csv, rate_regime_regressions.csv")


if __name__ == "__main__":
    main()
