# -*- coding: utf-8 -*-
"""
Shared engine for the robustness analyses that support the manuscript.

Everything here either imports from `pbo_analysis/pbo_cscv.py` or is copied
verbatim from `tactical_cash_reserve_backtest.ipynb` (cells 2, 11, 13), so the
robustness results cannot drift away from the results in the paper.

The one genuinely new piece is `run_constant_mix()`, the static cash benchmark:
a portfolio that simply holds a fixed share of its value in cash and rebalances
once a year, with no trigger and no timing. It answers the question the tactical
rule has to survive -- does releasing cash at a defined correction beat just
holding the same amount of cash all along?
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq, newton

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pbo_analysis"))

from pbo_cscv import (                                            # noqa: E402
    BUY_COST_RATE, CASH_CAP_RATIO, CORRECTION_THRESHOLD, LOOKBACK_WEEKS,
    SENSITIVITY_ALLOC_RATIOS, WEEKLY_CONTRIBUTION_TWD, WEEKLY_CONTRIBUTION_USD,
    build_markets, load_cash_rate_series, run_backtest,
)

DEFAULT_CASH_ALLOC_RATIO = 0.20
DESIGN_PERIOD_END = "2018-12-31"
OOS_PERIOD_START = "2019-01-01"
OUT_DIR = HERE


# ------------------------------------------------------------------
# metrics -- copied verbatim from notebook cell 13
# ------------------------------------------------------------------
def xirr(cashflows, dates, guess=0.1):
    dates = pd.to_datetime(pd.Index(dates))
    t0 = dates[0]
    days = np.array([(d - t0).days for d in dates], dtype=float)
    cfs = np.array(cashflows, dtype=float)

    def npv(r):
        return np.sum(cfs / (1.0 + r) ** (days / 365.0))

    for lo, hi in [(-0.99, 10.0), (-0.9999, 50.0)]:
        try:
            if npv(lo) * npv(hi) < 0:
                return brentq(npv, lo, hi, maxiter=1000)
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
    try:
        return newton(npv, x0=guess, maxiter=1000)
    except Exception:
        return np.nan


def compute_metrics(bt, price, cash_rate, label="", initial_value=None, initial_date=None):
    dates = list(bt.index)
    cashflows = list(-bt["contribution"].values)
    if initial_value is not None:
        dates = [initial_date] + dates
        cashflows = [-initial_value] + cashflows
    dates = dates + [bt.index[-1]]
    cashflows = cashflows + [bt["total_value"].iloc[-1]]
    cagr = xirr(cashflows, dates)

    nav = bt["total_value"] / bt["cumulative_contribution"]
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    mdd = dd.min()

    nav_ret = nav.pct_change().dropna()
    mar_weekly = cash_rate.mean() / 52.0
    diff = nav_ret - mar_weekly
    downside_sq = np.where(diff < 0, diff.values ** 2, 0.0)
    downside_std_annual = np.sqrt(downside_sq.mean()) * np.sqrt(52) if len(nav_ret) else np.nan
    mean_ret_annual = nav_ret.mean() * 52 if len(nav_ret) else np.nan
    sortino = (mean_ret_annual - cash_rate.mean()) / downside_std_annual if downside_std_annual else np.nan

    n_weeks = len(price)
    equity_cagr = (price.iloc[-1] / price.iloc[0]) ** (52 / n_weeks) - 1 if n_weeks > 1 else np.nan
    cash_cagr = cash_rate.mean()
    avg_cash_weight = (bt["cash_balance"] / bt["total_value"]).mean()
    cash_drag = avg_cash_weight * (equity_cagr - cash_cagr)

    return {
        "Strategy": label, "CAGR (XIRR)": cagr, "MDD": mdd, "Sortino Ratio": sortino,
        "Cash Drag": cash_drag, "Avg. Cash Weight": avg_cash_weight,
        "Trigger Count": int(bt["triggered"].sum()),
        "Terminal Value": bt["total_value"].iloc[-1],
    }


# ------------------------------------------------------------------
# the static cash benchmark
# ------------------------------------------------------------------
def run_constant_mix(price, cash_rate, target_cash_weight,
                     weekly_contribution=WEEKLY_CONTRIBUTION_TWD,
                     buy_cost_rate=BUY_COST_RATE, sell_cost_rate=BUY_COST_RATE,
                     rebalance="annual"):
    """
    Hold a fixed share of portfolio value in cash; never time the market.

    Each contribution is split at the target ratio, so the weight drifts only
    with relative performance between contributions. Once a year, in the first
    week of the calendar year, the portfolio is rebalanced back to the target.
    Purchases pay the same cost as everywhere else in the paper; rebalancing
    sales are charged at the same rate, which is conservative -- it is the
    static benchmark, not the tactical strategy, that pays it.

    The returned frame carries the same columns run_backtest() produces, so the
    identical compute_metrics() can be applied to both.
    """
    idx = price.index
    equity_units = 0.0
    cash_balance = 0.0
    cumulative_contribution = 0.0
    last_rebalance_year = None

    records = []
    for t in range(len(idx)):
        dt = idx[t]
        p = price.iloc[t]

        cash_balance *= (1.0 + cash_rate.iloc[t] / 52.0)

        cumulative_contribution += weekly_contribution
        to_cash = target_cash_weight * weekly_contribution
        buy_amount = weekly_contribution - to_cash
        cash_balance += to_cash

        sell_cost = 0.0
        if rebalance == "annual" and dt.year != last_rebalance_year:
            if last_rebalance_year is not None:
                # Rebalance on the portfolio including this week's contribution, part of
                # which is still earmarked for the equity purchase below.
                total_now = equity_units * p + cash_balance + buy_amount
                shortfall = target_cash_weight * total_now - cash_balance
                if shortfall > 0:
                    # Cash sleeve is light: hold back this week's purchase first, and only
                    # sell equity for whatever is still missing.
                    from_purchase = min(shortfall, buy_amount)
                    buy_amount -= from_purchase
                    cash_balance += from_purchase
                    still_needed = shortfall - from_purchase
                    if still_needed > 0:
                        proceeds = min(still_needed, equity_units * p)
                        sell_cost = proceeds * sell_cost_rate
                        equity_units -= proceeds / p
                        cash_balance += proceeds - sell_cost
                elif shortfall < 0:
                    # Cash sleeve is heavy: move the excess into this week's purchase.
                    excess = min(-shortfall, cash_balance)
                    cash_balance -= excess
                    buy_amount += excess
            last_rebalance_year = dt.year

        buy_cost = buy_amount * buy_cost_rate
        units_bought = (buy_amount - buy_cost) / p if buy_amount > 0 else 0.0
        equity_units += units_bought

        equity_value = equity_units * p
        total_value = equity_value + cash_balance

        records.append({
            "date": dt, "price": p, "contribution": weekly_contribution,
            "cumulative_contribution": cumulative_contribution,
            "to_cash": to_cash, "to_equity_direct": buy_amount,
            "deployed": 0.0, "triggered": False, "buy_cost": buy_cost + sell_cost,
            "equity_units": equity_units, "equity_value": equity_value,
            "cash_balance": cash_balance, "total_value": total_value,
        })

    return pd.DataFrame(records).set_index("date")
