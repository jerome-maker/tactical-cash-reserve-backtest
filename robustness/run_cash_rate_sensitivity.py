# -*- coding: utf-8 -*-
"""
Does the price of the reserve depend on paying cash the retail savings rate?

The paper pays idle cash the rate a retail saver actually earns on a liquid
deposit. A referee will reasonably ask what happens if the reserve sits in
something that pays more -- a money-market fund or Treasury bills. This script
re-runs the headline comparison with each market's short-term rate in place of
its savings-deposit rate, holding every rule parameter fixed:

  Taiwan         Bank of Taiwan one-month time-deposit rate (floating), from the
                 same Central Bank of Taiwan table (data/a13rate.xls) that
                 supplies the savings rate. Column 4 of that table is the savings
                 rate already used in the paper; column 6 is the one-month rate.
  United States  three-month Treasury bill, secondary market (FRED TB3MS).
  South Africa   Treasury bill rate (IMF International Financial Statistics,
                 DBnomics IMF/IFS/M.ZA.FITB_PA), held at its last observation
                 (March 2025) to the end of the sample, the same treatment the
                 deposit-rate series receives after its own last observation.

Writes cash_rate_sensitivity.csv and the three short-rate series to data/.
"""
import io
import json
import os
import urllib.request

import numpy as np
import pandas as pd

import common as C
from run_inference import hac_mean_test, nav_excess
from run_static_benchmark import match_average_cash
from south_africa import build_three_markets

DATA = os.path.join(C.PROJECT_ROOT, "data")
TW_SHORT = os.path.join(DATA, "twd_one_month_deposit_rate.csv")
US_SHORT = os.path.join(DATA, "usd_tbill_3m_rate.csv")
ZA_SHORT = os.path.join(DATA, "zar_tbill_rate.csv")
DBNOMICS = "https://api.db.nomics.world/v22/series/IMF/IFS/M.ZA.FITB_PA?observations=1&format=json"


def roc_month(code):
    """'09001' (ROC year 90, month 01) -> 2001-01-01."""
    code = str(code).strip()
    return pd.Timestamp(year=int(code[:-2]) + 1911, month=int(code[-2:]), day=1)


def build_taiwan():
    raw = pd.read_excel(os.path.join(DATA, "a13rate.xls"), header=None)
    rows = raw.iloc[5:]
    rows = rows[rows[0].astype(str).str.strip().str.fullmatch(r"\d{4,5}")]
    dates = rows[0].map(roc_month)
    savings = pd.Series(rows[4].astype(float).values / 100, index=dates)
    one_month = pd.Series(rows[6].astype(float).values / 100, index=dates)
    # the savings column must reproduce the series the paper already uses
    used = pd.read_csv(os.path.join(DATA, "twd_savings_rate.csv"),
                       parse_dates=["date"]).set_index("date")["annual_rate"]
    common = used.index.intersection(savings.index)
    gap = float((used[common] - savings[common]).abs().max())
    assert gap < 1e-9, "column 4 does not reproduce twd_savings_rate.csv (max gap {})".format(gap)
    out = one_month.rename("annual_rate").rename_axis("date").to_frame()
    out.to_csv(TW_SHORT)
    return out


def build_us():
    text = open(os.path.join(DATA, "raw_fred_tb3ms_2008on.txt"), encoding="utf-8").read()
    df = pd.read_csv(io.StringIO(text.strip().replace(";", "\n")), parse_dates=["observation_date"])
    out = pd.DataFrame({"date": df["observation_date"], "annual_rate": df["TB3MS"] / 100})
    out.set_index("date").to_csv(US_SHORT)
    return out


def build_za():
    if not os.path.exists(ZA_SHORT):
        with urllib.request.urlopen(DBNOMICS, timeout=60) as fh:
            doc = json.load(fh)["series"]["docs"][0]
        s = pd.Series([v for v in doc["value"]], index=pd.to_datetime(doc["period"]))
        s = pd.to_numeric(s, errors="coerce").dropna() / 100
        s.rename("annual_rate").rename_axis("date").to_frame().to_csv(ZA_SHORT)
    return pd.read_csv(ZA_SHORT, parse_dates=["date"])


def price_of_reserve(price, rate, wc):
    equity_cagr = (price.iloc[-1] / price.iloc[0]) ** (52 / len(price)) - 1
    bench = C.run_backtest(price, rate, alloc_ratio=0.0, weekly_contribution=wc)
    tact = C.run_backtest(price, rate, alloc_ratio=C.DEFAULT_CASH_ALLOC_RATIO,
                          weekly_contribution=wc)
    m_b = C.compute_metrics(bench, price, rate)
    m_t = C.compute_metrics(tact, price, rate)
    _, _, static = match_average_cash(price, rate, wc, m_t["Avg. Cash Weight"])
    m_s = C.compute_metrics(static, price, rate)
    gross = m_t["Cash Drag"]
    net = m_b["CAGR (XIRR)"] - m_t["CAGR (XIRR)"]

    ex_b, ex_t, ex_s = (nav_excess(bt, rate) for bt in (bench, tact, static))
    idx = ex_b.index.intersection(ex_t.index).intersection(ex_s.index)
    tests = {name: hac_mean_test((a[idx] - b[idx]).values)
             for name, a, b in (("tactical_vs_benchmark", ex_t, ex_b),
                                ("static_vs_benchmark", ex_s, ex_b),
                                ("tactical_vs_static", ex_t, ex_s))}
    row = {"mean_cash_rate": rate.mean(), "spread_pp": 100 * (equity_cagr - rate.mean()),
           "avg_cash_weight": m_t["Avg. Cash Weight"],
           "gross_price_pp": 100 * gross, "net_price_pp": 100 * net,
           "recovered_share": (gross - net) / gross if gross else np.nan,
           "static_cost_pp": 100 * (m_b["CAGR (XIRR)"] - m_s["CAGR (XIRR)"]),
           "mdd_gap_pp": 100 * (m_t["MDD"] - m_b["MDD"])}
    for name, t in tests.items():
        row[name + "_pp"] = t["mean"] * 5200
        row[name + "_p"] = t["p"]
    return row


def main():
    build_taiwan()
    build_us()
    build_za()
    short = {"Taiwan (0050)": (TW_SHORT, "Bank of Taiwan one-month time deposit"),
             "United States (SPY)": (US_SHORT, "Three-month Treasury bill"),
             "South Africa (STX40)": (ZA_SHORT, "Treasury bill (IMF IFS)")}
    rows = []
    for label, cfg in build_three_markets().items():
        price, wc = cfg["price"], cfg["weekly_contribution"]
        alt_path, alt_name = short[label]
        alt = C.load_cash_rate_series(price.index, alt_path)
        for name, rate in (("Savings deposit (paper)", cfg["cash_rate"]), (alt_name, alt)):
            r = price_of_reserve(price, rate, wc)
            r.update({"Market": label, "Cash rate": name})
            rows.append(r)
        print("[done] " + label, flush=True)
    df = pd.DataFrame(rows)
    cols = ["Market", "Cash rate"] + [c for c in df.columns if c not in ("Market", "Cash rate")]
    df = df[cols]
    df.to_csv(os.path.join(C.HERE, "cash_rate_sensitivity.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda v: "{:.4f}".format(v)))


if __name__ == "__main__":
    main()
