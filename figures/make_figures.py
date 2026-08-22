"""
Generate all figures for the JAM submission from the REAL cached backtest
data (same source as the notebook), using the paper's current parameters:
correction_threshold=15%, lookback=52wk, cash_cap=20%, alloc_ratio=20%.

Both markets (Taiwan/0050 and United States/SPY) share the identical fixed
parameters and the identical run_backtest() logic -- only the price and
cash-rate series differ. This mirrors the paper's "same experimental
setup" design.

Outputs (vector PDF, journal prefers EPS/PDF/TIFF, sans-serif labels):
  fig1_mechanism.pdf    - trigger mechanism, real 2020 COVID window (Taiwan)
  fig2_equity_curve.pdf - full-sample equity curves, log scale, (a) TW (b) US
  fig3_drawdown.pdf     - underwater (drawdown) chart, (a) TW (b) US
  fig4_sensitivity.pdf  - allocation-ratio sensitivity, 2x3 grid (TW row / US row)
  fig5_tornado.pdf      - parameter-leverage tornado/bar chart (Taiwan only, diagnostic)
  fig6_cross_market.pdf - cross-market comparison of the strategy-vs-control gap
  fig7_pbo.pdf           - Probability of Backtest Overfitting (CSCV, S=16), (a) TW (b) US
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
OUT_DIR = SCRIPT_DIR

# ---------- style: sans-serif, journal-safe, colour + linestyle redundancy ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})
BLUE = "#2a5c9e"     # control
ORANGE = "#c9642a"   # strategy
GREY = "#888888"
AQUA = "#3c8a5e"      # United States accent (distinct from TW blue/orange pair)

# ---------- shared parameters (identical across both markets) ----------
BUY_COST_RATE = 0.0015
CORRECTION_THRESHOLD = 0.15
LOOKBACK_WEEKS = 52
CASH_CAP_RATIO = 0.20
DEFAULT_ALLOC = 0.20

MONTHLY_CONTRIBUTION_TWD = 10_000
WEEKLY_CONTRIBUTION_TWD = MONTHLY_CONTRIBUTION_TWD * 12 / 52
MONTHLY_CONTRIBUTION_USD = 1_000
WEEKLY_CONTRIBUTION_USD = MONTHLY_CONTRIBUTION_USD * 12 / 52


def clean_price_series(price, jump_threshold=0.30):
    price = price.copy()
    ret = price.pct_change()
    jump_dates = ret[ret.abs() > jump_threshold].index
    for d in jump_dates:
        loc = price.index.get_loc(d)
        ratio = price.iloc[loc] / price.iloc[loc - 1]
        price.iloc[:loc] = price.iloc[:loc] * ratio
    return price


def run_backtest(price, cash_rate, correction_threshold=CORRECTION_THRESHOLD, lookback=LOOKBACK_WEEKS,
                  cash_cap_ratio=CASH_CAP_RATIO, alloc_ratio=0.0, buy_cost_rate=BUY_COST_RATE,
                  weekly_contribution=WEEKLY_CONTRIBUTION_TWD, deploy_fraction=1.0):
    idx = price.index
    n = len(idx)
    rolling_high = price.rolling(window=lookback, min_periods=1).max()
    drawdown = price / rolling_high - 1.0

    equity_units = 0.0
    cash_balance = 0.0
    cumulative_contribution = 0.0
    armed = True
    records = []
    for t in range(n):
        p = price.iloc[t]
        wk_cash_rate = cash_rate.iloc[t]
        cash_balance *= (1.0 + wk_cash_rate / 52.0)
        cumulative_contribution += weekly_contribution
        cash_cap = cash_cap_ratio * cumulative_contribution
        to_cash = min(alloc_ratio * weekly_contribution, max(cash_cap - cash_balance, 0.0))
        to_equity_direct = weekly_contribution - to_cash
        cash_balance += to_cash

        triggered = False
        if drawdown.iloc[t] <= -correction_threshold:
            if armed:
                triggered = True
                armed = False
        else:
            armed = True

        deployed = 0.0
        if triggered and cash_balance > 0:
            deployed = cash_balance * deploy_fraction
            cash_balance -= deployed

        buy_amount = to_equity_direct + deployed
        buy_cost = buy_amount * buy_cost_rate
        net_invest = buy_amount - buy_cost
        units_bought = net_invest / p if buy_amount > 0 else 0.0
        equity_units += units_bought

        equity_value = equity_units * p
        total_value = equity_value + cash_balance
        records.append({
            "price": p, "rolling_high": rolling_high.iloc[t], "drawdown": drawdown.iloc[t],
            "cash_balance": cash_balance, "cumulative_contribution": cumulative_contribution,
            "contribution": weekly_contribution, "triggered": triggered,
            "equity_value": equity_value, "total_value": total_value,
        })
    return pd.DataFrame(records, index=idx)


def nav_drawdown(bt):
    nav = bt["total_value"] / bt["cumulative_contribution"]
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    return dd


def xirr(cashflows, dates, guess=0.1):
    from scipy.optimize import brentq, newton
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


def cagr_of(bt):
    dates = list(bt.index) + [bt.index[-1]]
    cashflows = list(-bt["contribution"].values) + [bt["total_value"].iloc[-1]]
    return xirr(cashflows, dates)


# ============================================================
# Load cached data (identical source to the notebook), both markets
# ============================================================
def load_price_daily(price_csv):
    price = pd.read_csv(os.path.join(DATA_DIR, price_csv), index_col=0, parse_dates=True)["price"]
    return clean_price_series(price)


def load_market(price_daily, rate_csv, weekly_contribution, common_start, common_end):
    # Trim to the common analysis window BEFORE weekly resampling -- this mirrors the
    # notebook's Sec. 10 "common window" logic exactly. SPY's cached history extends
    # back to 2003 (well before 0050's 2009 start and before the 2008 GFC), so without
    # this trim the United States backtest would silently run over a longer, unmatched
    # period and break the "same experimental setup" comparison with Taiwan.
    price_daily_trimmed = price_daily.loc[common_start:common_end]
    price_weekly = price_daily_trimmed.resample("W-FRI").last().dropna()
    price_weekly.name = "price"

    rate_df = pd.read_csv(os.path.join(DATA_DIR, rate_csv), parse_dates=["date"]).set_index("date").sort_index()
    cash_rate_weekly = rate_df["annual_rate"].reindex(price_weekly.index, method="ffill").bfill()

    bt_bench = run_backtest(price_weekly, cash_rate_weekly, alloc_ratio=0.0,
                             weekly_contribution=weekly_contribution)
    bt_strat = run_backtest(price_weekly, cash_rate_weekly, alloc_ratio=DEFAULT_ALLOC,
                             weekly_contribution=weekly_contribution)
    return price_weekly, cash_rate_weekly, bt_bench, bt_strat


# Common analysis window = intersection of both markets' cached data coverage. SPY's
# history fully covers and exceeds 0050's, so this intersection equals 0050's own
# range (2009-01-02 to the latest cached date) -- i.e. the United States backtest is
# trimmed down to Taiwan's window, not the other way around.
price_daily_tw_raw = load_price_daily("0050_total_return_daily.csv")
price_daily_us_raw = load_price_daily("spy_total_return_daily.csv")
common_start = max(price_daily_tw_raw.index.min(), price_daily_us_raw.index.min())
common_end = min(price_daily_tw_raw.index.max(), price_daily_us_raw.index.max())
print(f"Common analysis window: {common_start.date()} to {common_end.date()}")

price_weekly_tw, cash_rate_weekly_tw, bt_bench, bt_strat = load_market(
    price_daily_tw_raw, "twd_savings_rate.csv", WEEKLY_CONTRIBUTION_TWD, common_start, common_end)
price_weekly_us, cash_rate_weekly_us, bt_bench_us, bt_strat_us = load_market(
    price_daily_us_raw, "usd_savings_rate.csv", WEEKLY_CONTRIBUTION_USD, common_start, common_end)

trigger_dates = bt_strat.index[bt_strat["triggered"]]
trigger_dates_us = bt_strat_us.index[bt_strat_us["triggered"]]
print("TW trigger dates:", list(trigger_dates.strftime("%Y-%m-%d")))
print("US trigger dates:", list(trigger_dates_us.strftime("%Y-%m-%d")))

# ============================================================
# Figure 1: trigger mechanism, real window around the 2020 COVID-19 crash (Taiwan)
# ============================================================
covid_triggers = [d for d in trigger_dates if pd.Timestamp("2020-01-01") <= d <= pd.Timestamp("2020-12-31")]
anchor = covid_triggers[0] if covid_triggers else pd.Timestamp("2020-03-20")
win_start = anchor - pd.Timedelta(weeks=20)
win_end = anchor + pd.Timedelta(weeks=16)
win = bt_strat.loc[win_start:win_end]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})

ax1.plot(win.index, win["price"], color=BLUE, lw=1.4, label="0050 price (weekly close)")
ax1.plot(win.index, win["rolling_high"], color=GREY, lw=1.0, ls="--", label="Rolling 52-week high")
ax1.plot(win.index, win["rolling_high"] * (1 - CORRECTION_THRESHOLD), color=ORANGE, lw=1.0, ls=":",
          label=f"Correction threshold (-{CORRECTION_THRESHOLD:.0%} of high)")
for d in win.index[win["triggered"]]:
    ax1.axvline(d, color=ORANGE, lw=1.0, alpha=0.6)
    ax1.scatter([d], [win.loc[d, "price"]], color=ORANGE, zorder=5, s=28, marker="v")
ax1.set_ylabel("Price (TWD)")
ax1.legend(loc="upper right", fontsize=7, frameon=True, facecolor="white",
           framealpha=0.9, edgecolor="none")
ax1.set_title("(a) Price vs. rolling 52-week high and correction threshold", loc="left", fontsize=9)

ax2.fill_between(win.index, win["cash_balance"], 0, color=ORANGE, alpha=0.25, lw=0)
ax2.plot(win.index, win["cash_balance"], color=ORANGE, lw=1.4, label="Cash reserve balance")
for d in win.index[win["triggered"]]:
    ax2.axvline(d, color=ORANGE, lw=1.0, alpha=0.6)
ax2.set_ylabel("Cash reserve (TWD)")
ax2.set_title("(b) Cash reserve balance: accumulates, then deploys in full on trigger", loc="left", fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()

fig.suptitle("Illustration of the deployment trigger (real 2020 example, Taiwan)", fontsize=10, y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_mechanism.pdf"), bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 2: full-sample equity curves (log scale), (a) Taiwan (b) United States
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

panels = [
    (axes[0], bt_bench, bt_strat, trigger_dates, "TWD", "(a) Taiwan (0050)"),
    (axes[1], bt_bench_us, bt_strat_us, trigger_dates_us, "USD", "(b) United States (SPY)"),
]
for ax, bench, strat, trig, ccy, title in panels:
    ax.plot(bench.index, bench["total_value"], color=BLUE, lw=1.3, label="100% DCA (control)")
    ax.plot(strat.index, strat["total_value"], color=ORANGE, lw=1.3, ls="--", label="Tactical reserve (20%)")
    ax.scatter(trig, strat.loc[trig, "total_value"], color=ORANGE, s=18, zorder=5,
               edgecolor="white", linewidth=0.5, label="Deployment events")
    ax.set_yscale("log")
    ax.set_ylabel(f"Portfolio value ({ccy}, log scale)")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.set_title(title, loc="left", fontsize=9.5)

fig.suptitle("Portfolio value growth, 2009–2026: (a) Taiwan, (b) United States", fontsize=10, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_equity_curve.pdf"), bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 3: underwater (drawdown) chart, (a) Taiwan (b) United States
# ============================================================
dd_bench = nav_drawdown(bt_bench)
dd_strat = nav_drawdown(bt_strat)
dd_bench_us = nav_drawdown(bt_bench_us)
dd_strat_us = nav_drawdown(bt_strat_us)

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
panels = [
    (axes[0], dd_bench, dd_strat, "(a) Taiwan (0050)"),
    (axes[1], dd_bench_us, dd_strat_us, "(b) United States (SPY)"),
]
for ax, dd_b, dd_s, title in panels:
    ax.fill_between(dd_b.index, dd_b.values * 100, 0, color=BLUE, alpha=0.18, lw=0)
    ax.plot(dd_b.index, dd_b.values * 100, color=BLUE, lw=1.0, label="100% DCA (control)")
    ax.fill_between(dd_s.index, dd_s.values * 100, 0, color=ORANGE, alpha=0.18, lw=0)
    ax.plot(dd_s.index, dd_s.values * 100, color=ORANGE, lw=1.0, ls="--", label="Tactical reserve (20%)")
    ax.set_ylabel("Drawdown from prior high (%)")
    ax.set_xlabel("Year")
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.set_title(title, loc="left", fontsize=9.5)

fig.suptitle("Underwater chart: drawdown of contribution-adjusted net asset value", fontsize=10, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_drawdown.pdf"), bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 4: sensitivity analysis, 2x3 grid (row 1 = Taiwan, row 2 = United States)
# (verified values copied from Table 3 / Table 4 in the manuscript)
# ============================================================
ratios = [10, 20, 30, 40, 50]
tw_full = [20.03, 19.99, 19.96, 19.92, 19.86]
tw_design = [7.88, 7.80, 7.73, 7.65, 7.52]
tw_oos = [29.49, 29.47, 29.46, 29.44, 29.42]
us_full = [14.69, 14.57, 14.43, 14.37, 14.33]
us_design = [10.83, 10.54, 10.23, 10.07, 9.98]
us_oos = [17.42, 17.41, 17.40, 17.39, 17.39]

fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.6))

row_specs = [
    (axes[0], [tw_full, tw_design, tw_oos], BLUE,
     ["(a) Taiwan — full sample", "(b) Taiwan — design period (2009–2018)",
      "(c) Taiwan — out-of-sample (2019–2026)"]),
    (axes[1], [us_full, us_design, us_oos], AQUA,
     ["(d) United States — full sample", "(e) United States — design period (2009–2018)",
      "(f) United States — out-of-sample (2019–2026)"]),
]
markers = ["o", "s", "^"]
for row_axes, series_list, color, titles in row_specs:
    for ax, series, marker, title in zip(row_axes, series_list, markers, titles):
        ax.plot(ratios, series, marker=marker, color=color, lw=1.6, ms=4.5)
        ax.set_xlabel("Cash-allocation ratio (%)")
        ax.set_ylabel("CAGR (%)")
        ax.set_xticks(ratios)
        ax.set_title(title, loc="left", fontsize=8)
        lo, hi = min(series), max(series)
        pad = max((hi - lo) * 0.6, 0.05)
        ax.set_ylim(lo - pad, hi + pad)

fig.suptitle("CAGR vs. allocation ratio: smooth, monotonic decline in both markets and all periods\n"
             "(each panel on its own scale; top row Taiwan, bottom row United States)", fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_sensitivity.pdf"), bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 5: tornado / bar chart of parameter leverage, both markets (Sec. 4.6).
# Computed live (not hardcoded) via one-at-a-time grid sweeps, using the same
# run_backtest() as the rest of this script, on both the Taiwan and United States
# common-window series loaded above.
# ============================================================
def leverage_swing(price_weekly, cash_rate_weekly, weekly_contribution, control_cagr, param, grid):
    gaps = []
    for v in grid:
        kwargs = dict(alloc_ratio=DEFAULT_ALLOC, weekly_contribution=weekly_contribution)
        kwargs[param] = v
        bt = run_backtest(price_weekly, cash_rate_weekly, **kwargs)
        gaps.append(cagr_of(bt) - control_cagr)
    return max(gaps) - min(gaps)


PARAM_SPECS = [
    ("Deployment style\n(100%→ 10% of reserve)", "deploy_fraction", [1.0, 0.75, 0.50, 0.25, 0.10], True),
    ("Correction threshold\n(5%–25%)", "correction_threshold", [0.05, 0.10, 0.15, 0.20, 0.25], True),
    ("Lookback window\n(13–104 weeks)", "lookback", [13, 26, 52, 78, 104], False),
    ("Cash reserve cap\n(5%–50%)", "cash_cap_ratio", [0.05, 0.10, 0.20, 0.30, 0.40, 0.50], False),
]

control_cagr_tw = cagr_of(bt_bench)
control_cagr_us = cagr_of(bt_bench_us)

swing_tw = [leverage_swing(price_weekly_tw, cash_rate_weekly_tw, WEEKLY_CONTRIBUTION_TWD,
                            control_cagr_tw, param, grid) * 100
            for _, param, grid, _ in PARAM_SPECS]
swing_us = [leverage_swing(price_weekly_us, cash_rate_weekly_us, WEEKLY_CONTRIBUTION_USD,
                            control_cagr_us, param, grid) * 100
            for _, param, grid, _ in PARAM_SPECS]
print("Leverage swings (Taiwan, pp):", dict(zip([p[1] for p in PARAM_SPECS], swing_tw)))
print("Leverage swings (US, pp):", dict(zip([p[1] for p in PARAM_SPECS], swing_us)))

labels = [p[0] for p in PARAM_SPECS]
fixed_flag = [p[3] for p in PARAM_SPECS]
# order by combined (TW+US average) swing, ascending, so largest ends up at top in barh
avg_swing = [(a + b) / 2 for a, b in zip(swing_tw, swing_us)]
order = np.argsort(avg_swing)
labels_o = [labels[i] for i in order]
swing_tw_o = [swing_tw[i] for i in order]
swing_us_o = [swing_us[i] for i in order]

fig, ax = plt.subplots(figsize=(6.6, 3.2))
y = np.arange(len(labels_o))
h = 0.32
bars_tw = ax.barh(y + h / 2, swing_tw_o, height=h, color=BLUE, label="Taiwan (0050)")
bars_us = ax.barh(y - h / 2, swing_us_o, height=h, color=AQUA, label="United States (SPY)")
for b, v in zip(bars_tw, swing_tw_o):
    ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f} pp", va="center", fontsize=7.5)
for b, v in zip(bars_us, swing_us_o):
    ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f} pp", va="center", fontsize=7.5)
ax.set_yticks(y, labels_o, fontsize=8)
ax.set_xlabel("Swing in the CAGR gap (percentage points)")
ax.set_xlim(0, max(max(swing_tw_o), max(swing_us_o)) * 1.2)
ax.legend(loc="lower right", fontsize=8, frameon=False)
ax.set_title("Diagnostic: leverage of each fixed parameter over the CAGR gap, both markets",
             loc="left", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_tornado.pdf"), bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 6 (NEW): cross-market comparison of the strategy-vs-control gap
# ============================================================
# Verified gap values (strategy minus control), full sample, 20% allocation --
# copied from the manuscript's Table 2 (both markets' main results).
markets = ["Taiwan (0050)", "United States (SPY)"]
cagr_gap = [-0.08, -0.24]      # percentage points
mdd_gap = [0.17, 0.17]         # percentage points (positive = smaller drawdown)
sortino_gap = [0.039, 0.012]

fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
panel_data = [
    (cagr_gap, "CAGR gap (pp)", "CAGR"),
    (mdd_gap, "MDD gap (pp)\n(positive = smaller drawdown)", "MDD"),
    (sortino_gap, "Sortino ratio gap", "Sortino ratio"),
]
colors_bar = [BLUE, AQUA]
for ax, (vals, ylabel, title) in zip(axes, panel_data):
    bars = ax.bar(markets, vals, color=colors_bar, width=0.55)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.3f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=8)
    ax.tick_params(axis="x", labelsize=7.5)

fig.suptitle("Cross-market comparison: strategy-vs-control gap, Taiwan vs. United States\n"
             "(full sample, 20% allocation) — all three metrics point the same direction in both markets",
             fontsize=9.5, y=1.06)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig6_cross_market.pdf"), bbox_inches="tight")
plt.close(fig)

print("All figures written to", OUT_DIR)
print("TW Bench MDD:", dd_bench.min(), "TW Strat MDD:", dd_strat.min())
print("US Bench MDD:", dd_bench_us.min(), "US Strat MDD:", dd_strat_us.min())

# ============================================================
# Figure 7 (NEW): Probability of Backtest Overfitting (PBO) via Combinatorially
# Symmetric Cross-Validation (CSCV; Bailey, Borwein, Lopez de Prado & Zhu 2016),
# applied to the same N=5 allocation-ratio trials as Figure 4 / Table sens.
# Computed live at S=16 (the paper's headline block count) using the same
# run_backtest() and price/cash-rate series already loaded above; full
# stability check across S=10/16/20 and the follow-up regime-shift diagnostic
# live in pbo_analysis/ (not duplicated here to keep this script focused).
# ============================================================
import itertools
from math import log, sqrt

SENSITIVITY_ALLOC_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50]
PBO_S = 16


def annualized_sharpe(weekly_excess_returns):
    arr = np.asarray(weekly_excess_returns, dtype=float)
    mu = arr.mean() * 52.0
    sigma = arr.std(ddof=1) * sqrt(52.0)
    return mu / sigma if sigma > 0 else 0.0


def build_returns_matrix(price_weekly, cash_rate_weekly, weekly_contribution):
    cols = {}
    for r in SENSITIVITY_ALLOC_RATIOS:
        bt = run_backtest(price_weekly, cash_rate_weekly, alloc_ratio=r,
                           weekly_contribution=weekly_contribution)
        nav = bt["total_value"] / bt["cumulative_contribution"]
        weekly_excess = nav.pct_change() - (cash_rate_weekly / 52.0)
        cols[f"{r:.0%}"] = weekly_excess
    return pd.DataFrame(cols).dropna(how="any")


def cscv_pbo(returns_matrix, S):
    T, N = returns_matrix.shape
    block_size = T // S
    weeks_used = block_size * S
    values = returns_matrix.iloc[:weeks_used].to_numpy()
    blocks = [values[i * block_size:(i + 1) * block_size] for i in range(S)]

    lambdas = []
    for is_combo in itertools.combinations(range(S), S // 2):
        oos_combo = tuple(b for b in range(S) if b not in is_combo)
        is_data = np.concatenate([blocks[b] for b in is_combo], axis=0)
        oos_data = np.concatenate([blocks[b] for b in oos_combo], axis=0)
        sharpe_is = np.array([annualized_sharpe(is_data[:, n]) for n in range(N)])
        sharpe_oos = np.array([annualized_sharpe(oos_data[:, n]) for n in range(N)])
        n_star = int(np.argmax(sharpe_is))
        rank_oos = 1 + int(np.sum(sharpe_oos < sharpe_oos[n_star]))
        omega = rank_oos / (N + 1)
        lambdas.append(log(omega / (1 - omega)))

    lambdas = np.array(lambdas)
    return dict(S=S, N=N, n_combinations=len(lambdas), pbo=float(np.mean(lambdas <= 0)), lambdas=lambdas)


rm_tw = build_returns_matrix(price_weekly_tw, cash_rate_weekly_tw, WEEKLY_CONTRIBUTION_TWD)
rm_us = build_returns_matrix(price_weekly_us, cash_rate_weekly_us, WEEKLY_CONTRIBUTION_USD)
pbo_tw = cscv_pbo(rm_tw, S=PBO_S)
pbo_us = cscv_pbo(rm_us, S=PBO_S)
print(f"PBO (S={PBO_S}): Taiwan={pbo_tw['pbo']:.1%} ({pbo_tw['n_combinations']:,} combos), "
      f"United States={pbo_us['pbo']:.1%} ({pbo_us['n_combinations']:,} combos)")


def pbo_bars(ax, result, N):
    omega = 1.0 / (1.0 + np.exp(-result["lambdas"]))
    ranks = np.round(omega * (N + 1)).astype(int)
    rank_values = np.arange(1, N + 1)
    rank_lambdas = np.log((rank_values / (N + 1)) / (1 - rank_values / (N + 1)))
    counts = np.array([(ranks == r).sum() for r in rank_values])
    colors = [ORANGE if lam <= 0 else BLUE for lam in rank_lambdas]
    ax.bar(rank_lambdas, counts, width=0.28, color=colors, edgecolor="white", linewidth=0.5)
    for lam, c in zip(rank_lambdas, counts):
        ax.text(lam, c + counts.max() * 0.02, f"{c / len(result['lambdas']):.0%}",
                 ha="center", va="bottom", fontsize=6.5)
    ax.axvline(0, color="#222222", lw=1.1, ls="--")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_xlabel(r"Logit $\lambda$ (one value per possible OOS rank)")
    ax.set_ylabel("CSCV combinations")


fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
pbo_bars(axes[0], pbo_tw, N=5)
axes[0].set_title(f"(a) Taiwan (0050): PBO = {pbo_tw['pbo']:.1%}", loc="left", fontsize=9)
pbo_bars(axes[1], pbo_us, N=5)
axes[1].set_title(f"(b) United States (SPY): PBO = {pbo_us['pbo']:.1%}", loc="left", fontsize=9)
fig.suptitle(f"Probability of Backtest Overfitting (CSCV, S={PBO_S}, N=5 allocation-ratio trials)",
             fontsize=10, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig7_pbo.pdf"), bbox_inches="tight")
plt.close(fig)
print("[saved] fig7_pbo.pdf")
