# -*- coding: utf-8 -*-
"""
Sampling uncertainty around the reported differences.

The paper reports differences of a tenth of a percentage point in growth and a
hundredth in the Sortino ratio. Those numbers mean nothing without a statement
of how much of them is sampling noise, so this script attaches interval
estimates to every headline difference, and computes the deflated Sharpe ratio
that the paper's own future-research paragraph promises.

Method: stationary block bootstrap (Politis & Romano, 1994) on the joint series
of weekly equity returns and weekly cash rates, preserving their alignment.
Each replicate rebuilds a synthetic price path, re-runs the full backtest for
every strategy, and recomputes every metric with the same functions the paper
uses. Reported intervals are percentile intervals over replicates.

Writes bootstrap_results.csv and deflated_sharpe.csv.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import common as C
from run_static_benchmark import match_average_cash
from south_africa import build_three_markets

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
BLOCK_LENGTHS = [13, 26, 52]
HEADLINE_BLOCK = 26
SEED = 20260825


def stationary_bootstrap_index(n, mean_block, rng):
    """Politis-Romano indices: geometric block lengths, uniform starts, circular wrap."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    i = rng.integers(0, n)
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = rng.integers(0, n)
        else:
            i = (i + 1) % n
    return idx


def replicate(price, rate, wc, static_weight, idx):
    """Rebuild a synthetic path from resampled returns and re-run every strategy."""
    ret = price.pct_change().fillna(0.0).values
    boot_ret = ret[idx]
    boot_price = pd.Series(price.iloc[0] * np.cumprod(1.0 + boot_ret),
                           index=price.index, name="price")
    boot_rate = pd.Series(rate.values[idx], index=rate.index)

    bench = C.run_backtest(boot_price, boot_rate, alloc_ratio=0.0, weekly_contribution=wc)
    tact = C.run_backtest(boot_price, boot_rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                          weekly_contribution=wc)
    static = C.run_constant_mix(boot_price, boot_rate, target_cash_weight=static_weight,
                                weekly_contribution=wc)
    out = {}
    m_b = C.compute_metrics(bench, boot_price, boot_rate)
    for name, bt in (("tactical", tact), ("static", static)):
        m = C.compute_metrics(bt, boot_price, boot_rate)
        out[name] = {
            "CAGR gap (pp)": 100.0 * (m["CAGR (XIRR)"] - m_b["CAGR (XIRR)"]),
            "MDD gap (pp)": 100.0 * (m["MDD"] - m_b["MDD"]),
            "Sortino gap": m["Sortino Ratio"] - m_b["Sortino Ratio"],
        }
    return out


def deflated_sharpe(nav_excess_weekly, sharpes_across_trials):
    """Bailey & Lopez de Prado (2014), using the five allocation ratios as the trial set."""
    r = np.asarray(nav_excess_weekly, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis()) + 3.0          # pandas gives excess kurtosis
    n_trials = len(sharpes_across_trials)
    var_sr = float(np.var(sharpes_across_trials, ddof=1))
    gamma = 0.5772156649015329
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_star = np.sqrt(var_sr) * ((1.0 - gamma) * z1 + gamma * z2)
    denom = np.sqrt(1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr ** 2)
    dsr = norm.cdf((sr - sr_star) * np.sqrt(T - 1) / denom)
    return {"Sharpe (weekly)": sr, "Sharpe (annualised)": sr * np.sqrt(52),
            "Expected max Sharpe under the null (weekly)": sr_star,
            "Skewness": skew, "Kurtosis": kurt, "Observations": T,
            "Number of trials": n_trials, "Deflated Sharpe ratio": dsr}


def main():
    markets = build_three_markets()
    rng = np.random.default_rng(SEED)
    boot_rows, dsr_rows = [], []

    for label, cfg in markets.items():
        price, rate, wc = cfg["price"], cfg["cash_rate"], cfg["weekly_contribution"]

        bench = C.run_backtest(price, rate, alloc_ratio=0.0, weekly_contribution=wc)
        tact = C.run_backtest(price, rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                              weekly_contribution=wc)
        m_b = C.compute_metrics(bench, price, rate)
        m_t = C.compute_metrics(tact, price, rate)
        static_w, _, static = match_average_cash(price, rate, wc, m_t["Avg. Cash Weight"])
        m_s = C.compute_metrics(static, price, rate)

        observed = {
            "tactical": {"CAGR gap (pp)": 100 * (m_t["CAGR (XIRR)"] - m_b["CAGR (XIRR)"]),
                         "MDD gap (pp)": 100 * (m_t["MDD"] - m_b["MDD"]),
                         "Sortino gap": m_t["Sortino Ratio"] - m_b["Sortino Ratio"]},
            "static": {"CAGR gap (pp)": 100 * (m_s["CAGR (XIRR)"] - m_b["CAGR (XIRR)"]),
                       "MDD gap (pp)": 100 * (m_s["MDD"] - m_b["MDD"]),
                       "Sortino gap": m_s["Sortino Ratio"] - m_b["Sortino Ratio"]},
        }

        for L in BLOCK_LENGTHS:
            reps = REPS if L == HEADLINE_BLOCK else max(REPS // 2, 200)
            draws = {"tactical": [], "static": []}
            for k in range(reps):
                idx = stationary_bootstrap_index(len(price), L, rng)
                try:
                    rep = replicate(price, rate, wc, static_w, idx)
                except Exception:
                    continue
                for kind in draws:
                    draws[kind].append(rep[kind])
                if (k + 1) % 100 == 0:
                    print(f"  {label} L={L}: {k + 1}/{reps}", flush=True)

            for kind, recs in draws.items():
                d = pd.DataFrame(recs)
                for metric in ["CAGR gap (pp)", "MDD gap (pp)", "Sortino gap"]:
                    v = d[metric].replace([np.inf, -np.inf], np.nan).dropna()
                    obs = observed[kind][metric]
                    share = float((v < 0).mean() if obs < 0 else (v > 0).mean())
                    boot_rows.append({
                        "Market": label, "Comparison": kind, "Metric": metric,
                        "Mean block length (weeks)": L, "Replicates": len(v),
                        "Observed": obs, "Bootstrap mean": v.mean(),
                        "CI lower (2.5%)": v.quantile(0.025),
                        "CI upper (97.5%)": v.quantile(0.975),
                        "Share of replicates with the observed sign": share,
                    })

        # deflated Sharpe ratio over the five allocation ratios
        sharpes, excess_by_ratio = [], {}
        for ratio in C.SENSITIVITY_ALLOC_RATIOS:
            bt = C.run_backtest(price, rate, alloc_ratio=ratio, weekly_contribution=wc)
            nav = bt["total_value"] / bt["cumulative_contribution"]
            ex = (nav.pct_change() - rate / 52.0).dropna()
            excess_by_ratio[ratio] = ex
            sharpes.append(ex.mean() / ex.std(ddof=1))
        d = deflated_sharpe(excess_by_ratio[C.DEFAULT_CASH_ALLOC_RATIO], sharpes)
        d["Market"] = label
        dsr_rows.append(d)
        print(f"{label}: DSR={d['Deflated Sharpe ratio']:.4f} "
              f"(annualised Sharpe {d['Sharpe (annualised)']:.3f})", flush=True)

    pd.DataFrame(boot_rows).to_csv("bootstrap_results.csv", index=False)
    pd.DataFrame(dsr_rows).to_csv("deflated_sharpe.csv", index=False)
    print("\n[saved] bootstrap_results.csv, deflated_sharpe.csv")


if __name__ == "__main__":
    main()
