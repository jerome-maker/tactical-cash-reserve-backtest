# -*- coding: utf-8 -*-
"""
Conventional significance tests for the reported differences.

The first editor to see this paper objected that the performance differences
were "largely descriptive and not supported by conventional statistical
significance testing". The block bootstrap in run_bootstrap.py already measured
sampling variation, but it reported the share of replicates preserving a sign,
which is not the object a referee looks for. This script produces the standard
ones: standard errors, confidence intervals, test statistics and p-values.

Two tests, both on the weekly net asset value return series the paper already
uses (portfolio value divided by cumulative contributions), in excess of the
concurrent deposit rate:

  1. Ledoit and Wolf (2008) test for the difference of two Sharpe ratios:
     a studentized circular block bootstrap around a HAC standard error obtained
     by the delta method. This is the right test here because the returns are
     serially dependent and fat-tailed, which the naive Jobson-Korkie test
     assumes away.
  2. A Newey-West HAC t-test on the mean weekly return difference between the
     strategy and the benchmark.

Writes inference_results.csv.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import common as C
from run_static_benchmark import match_average_cash
from south_africa import build_three_markets

N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 4999
BLOCK = 26          # weeks; matches the headline block length used elsewhere
SEED = 20260923


# ------------------------------------------------------------------
# HAC machinery
# ------------------------------------------------------------------
def nw_lag(T):
    """Newey-West lag truncation, the usual automatic rule."""
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def hac_cov(V, lag=None):
    """Newey-West HAC covariance of the sample mean of the rows of V (T x k)."""
    V = np.asarray(V, dtype=float)
    T = V.shape[0]
    if lag is None:
        lag = nw_lag(T)
    Vc = V - V.mean(axis=0)
    S = (Vc.T @ Vc) / T
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        G = (Vc[j:].T @ Vc[:-j]) / T
        S = S + w * (G + G.T)
    return S


def sharpe(r):
    r = np.asarray(r, dtype=float)
    return r.mean() / r.std(ddof=1)


def sharpe_diff_se(r1, r2):
    """
    HAC standard error of SR1 - SR2 by the delta method.

    With mu = E[r] and gamma = E[r^2], the Sharpe ratio is
    mu / sqrt(gamma - mu^2), so its gradient with respect to (mu, gamma) is
    (gamma / (gamma - mu^2)^{3/2}, -mu / (2 (gamma - mu^2)^{3/2})).
    Stacking the four moments (mu1, mu2, gamma1, gamma2) and sandwiching the HAC
    covariance of their sample means gives the variance of the difference.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    T = len(r1)
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1 ** 2).mean(), (r2 ** 2).mean()
    s1 = (g1 - mu1 ** 2) ** 1.5
    s2 = (g2 - mu2 ** 2) ** 1.5
    if s1 <= 0 or s2 <= 0:
        return np.nan
    grad = np.array([g1 / s1, -g2 / s2, -mu1 / (2 * s1), mu2 / (2 * s2)])
    V = np.column_stack([r1, r2, r1 ** 2, r2 ** 2])
    S = hac_cov(V)
    var = float(grad @ S @ grad) / T
    return np.sqrt(var) if var > 0 else np.nan


def circular_block_index(T, block, rng):
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    idx = np.concatenate([(s + np.arange(block)) % T for s in starts])
    return idx[:T]


def ledoit_wolf_sharpe_test(r1, r2, block=BLOCK, n_boot=N_BOOT, seed=SEED):
    """Studentized circular block bootstrap test of H0: SR1 = SR2."""
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    T = len(r1)
    d_hat = sharpe(r1) - sharpe(r2)
    se_hat = sharpe_diff_se(r1, r2)
    if not np.isfinite(se_hat):
        return {"diff": d_hat, "se": np.nan, "t": np.nan, "p": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "replicates": 0}
    rng = np.random.default_rng(seed)
    t_stats = []
    for _ in range(n_boot):
        idx = circular_block_index(T, block, rng)
        b1, b2 = r1[idx], r2[idx]
        se_b = sharpe_diff_se(b1, b2)
        if np.isfinite(se_b) and se_b > 0:
            t_stats.append(abs((sharpe(b1) - sharpe(b2) - d_hat) / se_b))
    t_stats = np.asarray(t_stats)
    t_obs = d_hat / se_hat
    p = float((t_stats >= abs(t_obs)).mean()) if len(t_stats) else np.nan
    q95 = float(np.quantile(t_stats, 0.95)) if len(t_stats) else np.nan
    return {"diff": d_hat, "se": se_hat, "t": t_obs, "p": p,
            "ci_low": d_hat - q95 * se_hat, "ci_high": d_hat + q95 * se_hat,
            "replicates": int(len(t_stats))}


def hac_mean_test(d):
    """Newey-West t-test on the mean of a single series."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    T = len(d)
    S = hac_cov(d.reshape(-1, 1))
    se = float(np.sqrt(S[0, 0] / T))
    tstat = d.mean() / se
    p = 2.0 * (1.0 - student_t.cdf(abs(tstat), df=T - 1))
    return {"mean": float(d.mean()), "se": se, "t": tstat, "p": p,
            "ci_low": float(d.mean()) - 1.96 * se,
            "ci_high": float(d.mean()) + 1.96 * se,
            "lag": nw_lag(T), "observations": T}


def nav_excess(bt, rate):
    """Weekly NAV return in excess of the concurrent deposit rate."""
    nav = bt["total_value"] / bt["cumulative_contribution"]
    return (nav.pct_change() - rate / 52.0).dropna()


def main():
    markets = build_three_markets()
    rows = []
    for label, cfg in markets.items():
        price, rate, wc = cfg["price"], cfg["cash_rate"], cfg["weekly_contribution"]
        bench = C.run_backtest(price, rate, alloc_ratio=0.0, weekly_contribution=wc)
        tact = C.run_backtest(price, rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                              weekly_contribution=wc)
        m_t = C.compute_metrics(tact, price, rate)
        static_w, _, static = match_average_cash(price, rate, wc, m_t["Avg. Cash Weight"])

        ex_b = nav_excess(bench, rate)
        ex_t = nav_excess(tact, rate)
        ex_s = nav_excess(static, rate)
        idx = ex_b.index.intersection(ex_t.index).intersection(ex_s.index)
        ex_b, ex_t, ex_s = ex_b[idx], ex_t[idx], ex_s[idx]

        # The third row answers the question the first two only imply: holding the
        # same average cash, does releasing it at a correction beat never releasing it?
        for name, ex, ref in (("Cash reserve vs benchmark", ex_t, ex_b),
                              ("Static cash vs benchmark", ex_s, ex_b),
                              ("Cash reserve vs static cash", ex_t, ex_s)):
            lw = ledoit_wolf_sharpe_test(ex.values, ref.values)
            rows.append({"Market": label, "Comparison": name,
                         "Test": "Ledoit-Wolf Sharpe difference (annualised)",
                         "Estimate": lw["diff"] * np.sqrt(52),
                         "Std. error": lw["se"] * np.sqrt(52),
                         "Statistic": lw["t"], "p-value": lw["p"],
                         "CI lower": lw["ci_low"] * np.sqrt(52),
                         "CI upper": lw["ci_high"] * np.sqrt(52),
                         "Observations": lw["replicates"]})
            ht = hac_mean_test((ex - ref).values)
            rows.append({"Market": label, "Comparison": name,
                         "Test": "Newey-West mean weekly return difference (pp a year)",
                         "Estimate": ht["mean"] * 5200,
                         "Std. error": ht["se"] * 5200,
                         "Statistic": ht["t"], "p-value": ht["p"],
                         "CI lower": ht["ci_low"] * 5200,
                         "CI upper": ht["ci_high"] * 5200,
                         "Observations": ht["observations"]})
        print("[done] {}".format(label), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("inference_results.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda v: "{:.4f}".format(v)))
    print("\n[saved] inference_results.csv")


if __name__ == "__main__":
    main()
