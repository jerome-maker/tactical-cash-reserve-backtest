# -*- coding: utf-8 -*-
"""
Compute every number the manuscript reports, for all three markets, and write
results.json. The manuscript builder reads that file, so no figure in the paper
is ever typed by hand and the paper cannot drift away from the code.

The organising idea is the price of the reserve. Holding cash instead of equity
costs the spread between what the equity earned and what the deposit paid,
applied to the share of the portfolio sitting in cash: the paper's cash-drag
measure. That is the gross price. The trigger then buys the market after a fall
and recovers part of it. What is left is the net price, and it is the number an
investor actually pays. Because the gross price depends on the local deposit
rate, the same rule is cheap where deposits pay well and expensive where they
do not, which is what the three markets are here to show.
"""
import json
import os

import numpy as np
import pandas as pd

import common as C
from pbo_cscv import build_returns_matrix, cscv_pbo
from run_static_benchmark import match_average_cash
from south_africa import build_three_markets

OUT = os.path.join(C.HERE, "results.json")
PBO_BLOCKS = 16
THRESHOLD_ALT = 0.10


def num(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def clean(d):
    return {k: num(v) for k, v in d.items() if isinstance(v, (int, float))}


def metrics_of(bt, price, rate, label="", start=None, end=None):
    if start is None and end is None:
        return C.compute_metrics(bt, price, rate, label)
    sub = bt.loc[start:end]
    loc = bt.index.get_loc(sub.index[0])
    iv, idate = (bt.iloc[loc - 1]["total_value"], bt.index[loc - 1]) if loc else (None, None)
    return C.compute_metrics(sub, price.loc[sub.index], rate.loc[sub.index], label,
                             initial_value=iv, initial_date=idate)


def main():
    markets = build_three_markets()
    res = {"markets": {}, "meta": {}}

    for label, cfg in markets.items():
        price, rate, wc = cfg["price"], cfg["cash_rate"], cfg["weekly_contribution"]
        equity_cagr = (price.iloc[-1] / price.iloc[0]) ** (52 / len(price)) - 1
        m = {"currency": cfg["currency"], "slug": cfg["slug"], "weeks": int(len(price)),
             "start": str(price.index.min().date()), "end": str(price.index.max().date()),
             "mean_deposit_rate": num(rate.mean()), "equity_cagr": num(equity_cagr)}

        bench = C.run_backtest(price, rate, alloc_ratio=0.0, weekly_contribution=wc)
        tact = C.run_backtest(price, rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                              weekly_contribution=wc)
        hoard = C.run_backtest(price, rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                               weekly_contribution=wc, deploy_fraction=0.0)
        m_b = C.compute_metrics(bench, price, rate, "benchmark")
        m_t = C.compute_metrics(tact, price, rate, "tactical")
        static_w, static_avg, static = match_average_cash(price, rate, wc, m_t["Avg. Cash Weight"])
        m_s = C.compute_metrics(static, price, rate, "static")
        m_h = C.compute_metrics(hoard, price, rate, "hoard")

        m["static_cash_weight"] = num(static_w)
        m["static_realised_avg_cash"] = num(static_avg)
        m["main"] = {"benchmark": clean(m_b), "tactical": clean(m_t),
                     "static": clean(m_s), "hoard": clean(m_h)}

        gross = m_t["Cash Drag"]
        net = m_b["CAGR (XIRR)"] - m_t["CAGR (XIRR)"]
        m["price_of_reserve"] = {
            "gross_price_pp": num(100 * gross),
            "net_price_pp": num(100 * net),
            "recovered_pp": num(100 * (gross - net)),
            "recovered_share": num((gross - net) / gross) if gross else None,
            "avg_cash_weight": num(m_t["Avg. Cash Weight"]),
            "spread_pp": num(100 * (equity_cagr - rate.mean())),
        }

        def gaps(mm):
            return {"cagr_pp": num(100 * (mm["CAGR (XIRR)"] - m_b["CAGR (XIRR)"])),
                    "mdd_pp": num(100 * (mm["MDD"] - m_b["MDD"])),
                    "sortino": num(mm["Sortino Ratio"] - m_b["Sortino Ratio"])}

        m["gaps"] = {"tactical": gaps(m_t), "static": gaps(m_s), "hoard": gaps(m_h)}

        sweep = {}
        for ratio in C.SENSITIVITY_ALLOC_RATIOS:
            bt = C.run_backtest(price, rate, alloc_ratio=ratio, weekly_contribution=wc)
            key = "{:.0%}".format(ratio)
            sweep[key] = {
                "full": num(C.compute_metrics(bt, price, rate)["CAGR (XIRR)"]),
                "design": num(metrics_of(bt, price, rate, "", None, C.DESIGN_PERIOD_END)["CAGR (XIRR)"]),
                "heldout": num(metrics_of(bt, price, rate, "", C.OOS_PERIOD_START, None)["CAGR (XIRR)"]),
            }
        m["sweep"] = sweep
        m["sweep_step_pp"] = {}
        for per in ("full", "design", "heldout"):
            vals = [sweep["{:.0%}".format(r)][per] for r in C.SENSITIVITY_ALLOC_RATIOS]
            steps = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
            m["sweep_step_pp"][per] = num(100 * float(np.mean(steps)))

        m["periods"] = {}
        for pname, start, end in (("design", None, C.DESIGN_PERIOD_END),
                                  ("heldout", C.OOS_PERIOD_START, None)):
            mb = metrics_of(bench, price, rate, "benchmark", start, end)
            mt = metrics_of(tact, price, rate, "tactical", start, end)
            ms = metrics_of(static, price, rate, "static", start, end)
            m["periods"][pname] = {
                "benchmark": clean(mb), "tactical": clean(mt), "static": clean(ms),
                "gap_cagr_pp": num(100 * (mt["CAGR (XIRR)"] - mb["CAGR (XIRR)"])),
                "gap_mdd_pp": num(100 * (mt["MDD"] - mb["MDD"])),
                "gap_sortino": num(mt["Sortino Ratio"] - mb["Sortino Ratio"]),
            }

        alt = C.run_backtest(price, rate, correction_threshold=THRESHOLD_ALT,
                             alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO, weekly_contribution=wc)
        m_alt = C.compute_metrics(alt, price, rate, "alt threshold")
        m["threshold_10pct"] = clean(m_alt)
        m["threshold_10pct"]["gap_cagr_pp"] = num(100 * (m_alt["CAGR (XIRR)"] - m_b["CAGR (XIRR)"]))

        rm = build_returns_matrix(price, rate, wc)
        pbo = cscv_pbo(rm, S=PBO_BLOCKS)
        lam = np.asarray(pbo["lambdas"], dtype=float)
        omega = 1.0 / (1.0 + np.exp(-lam))
        ranks = np.round(omega * (pbo["N"] + 1)).astype(int)
        m["pbo"] = {"pbo": num(pbo["pbo"]), "mean_lambda": num(pbo["mean_lambda"]),
                    "n_combinations": int(pbo["n_combinations"]), "blocks": PBO_BLOCKS,
                    "n_trials": int(pbo["N"]),
                    "rank_shares": {str(r): num(100.0 * float((ranks == r).mean()))
                                    for r in range(1, pbo["N"] + 1)}}

        swings = {}
        grids = {"Deployment style, 100 percent down to 10 percent":
                     [dict(deploy_fraction=f) for f in (1.0, 0.5, 0.25, 0.10)],
                 "Correction threshold, 5 to 25 percent":
                     [dict(correction_threshold=t) for t in (0.05, 0.10, 0.15, 0.20, 0.25)],
                 "Lookback window, 13 to 104 weeks":
                     [dict(lookback=l) for l in (13, 26, 52, 104)],
                 "Cash reserve cap, 5 to 50 percent":
                     [dict(cash_cap_ratio=c) for c in (0.05, 0.10, 0.20, 0.35, 0.50)]}
        for name, grid in grids.items():
            gs = []
            for kw in grid:
                bt = C.run_backtest(price, rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                                    weekly_contribution=wc, **kw)
                gs.append(C.compute_metrics(bt, price, rate)["CAGR (XIRR)"] - m_b["CAGR (XIRR)"])
            swings[name] = num(100 * (max(gs) - min(gs)))
        m["parameter_leverage"] = swings

        res["markets"][label] = m
        p = m["price_of_reserve"]
        print("[done] {}: gross {:.3f}pp, net {:+.3f}pp, recovered {:.0%}, PBO {:.1f}%".format(
            label, p["gross_price_pp"], p["net_price_pp"], p["recovered_share"] or 0,
            100 * m["pbo"]["pbo"]), flush=True)

    for name, path in (("bootstrap", "bootstrap_results.csv"),
                       ("deflated_sharpe", "deflated_sharpe.csv"),
                       ("us_long_sample", "us_long_sample_results.csv"),
                       ("us_long_sensitivity", "us_long_sample_sensitivity.csv")):
        if os.path.exists(path):
            res[name] = pd.read_csv(path).to_dict(orient="records")

    res["meta"] = {"generated_from": "robustness/build_all_results.py",
                   "pbo_blocks": PBO_BLOCKS,
                   "alloc_ratios": ["{:.0%}".format(r) for r in C.SENSITIVITY_ALLOC_RATIOS],
                   "default_ratio": "{:.0%}".format(C.DEFAULT_CASH_ALLOC_RATIO),
                   "design_end": C.DESIGN_PERIOD_END, "oos_start": C.OOS_PERIOD_START,
                   "buy_cost_rate": C.BUY_COST_RATE, "cash_cap_ratio": C.CASH_CAP_RATIO,
                   "correction_threshold": C.CORRECTION_THRESHOLD,
                   "lookback_weeks": C.LOOKBACK_WEEKS}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("\n[saved] " + OUT)


if __name__ == "__main__":
    main()
