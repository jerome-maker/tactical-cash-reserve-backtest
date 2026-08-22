"""
Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV), applied to the tactical-cash-reserve-strategy backtest.

Method: Bailey, D.H., Borwein, J., Lopez de Prado, M., & Zhu, Q.J. (2016).
"The Probability of Backtest Overfitting." Journal of Computational Finance,
20(4), 39-69. (Cited in submission_JAM/references.bib as `baileyetal2016pbo`.)

This script formalizes the exact promise made in the manuscript's own
future-research paragraph:

    "Replacing the single in-sample/out-of-sample split with rolling
    walk-forward analysis, combined with formal overfitting-adjusted
    statistics such as the Deflated Sharpe Ratio ... or the Probability
    of Backtest Overfitting ..., would provide a more rigorous statistical
    foundation than the smoothness-based diagnostic used here."

Trial set (N=5): the same SENSITIVITY_ALLOC_RATIOS already reported in the
manuscript's Table 3 (10/20/30/40/50% cash-allocation ratio) -- no new
parameter grid is introduced, so this is a genuine re-examination of an
analysis the paper already performs, not a new source of overfitting risk
in itself.

Data loading, cleaning, and the backtest engine below are copied verbatim
(constants and function bodies) from `tactical_cash_reserve_backtest.ipynb`
(cells 2, 4, 7, 9, 11) so the N=5 trials are byte-for-byte the same backtest
already reported in the paper. This script adds no new modeling assumptions
beyond the CSCV/PBO procedure itself; it reads directly from the project's
cached data files (no network calls), so it is fully deterministic.

Outputs (written next to this script, in pbo_analysis/):
    pbo_results.csv                 - one row per (market, S)
    pbo_distribution_taiwan.pdf     - lambda-distribution histogram, S=16
    pbo_distribution_us.pdf         - lambda-distribution histogram, S=16
    pbo_summary.md                  - written report
"""
import itertools
import os
from math import comb, log, sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# Constants -- copied verbatim from tactical_cash_reserve_backtest.ipynb, cell 2
# ============================================================
CORRECTION_THRESHOLD = 0.15
LOOKBACK_WEEKS = 52
CASH_CAP_RATIO = 0.20
BUY_COST_RATE = 0.0015

SENSITIVITY_ALLOC_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50]  # the N=5 trials

MONTHLY_CONTRIBUTION_TWD = 10_000
WEEKLY_CONTRIBUTION_TWD = MONTHLY_CONTRIBUTION_TWD * 12 / 52
MONTHLY_CONTRIBUTION_USD = 1_000
WEEKLY_CONTRIBUTION_USD = MONTHLY_CONTRIBUTION_USD * 12 / 52

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_DIR = SCRIPT_DIR

TW_PRICE_CACHE = os.path.join(DATA_DIR, "0050_total_return_daily.csv")
US_PRICE_CACHE = os.path.join(DATA_DIR, "spy_total_return_daily.csv")
TWD_RATE_CACHE = os.path.join(DATA_DIR, "twd_savings_rate.csv")
USD_RATE_CACHE = os.path.join(DATA_DIR, "usd_savings_rate.csv")

BLOCK_COUNTS = [10, 16, 20]  # S=16 is the headline; 10 and 20 are the stability check


# ============================================================
# Data loading -- copied verbatim (logic) from cells 4, 7, 9
# ============================================================
def clean_price_series(price, jump_threshold=0.30):
    """
    Detect single-day price jumps larger than `jump_threshold` in magnitude and back-adjust
    the price series before the jump by the same ratio, so the return series is continuous.
    This guards against unannounced stock splits or vendor data transitions being misread as
    real one-day returns; genuine returns of this size essentially never occur in a broad-market
    ETF, so any correction made here should be inspected.
    """
    price = price.copy()
    ret = price.pct_change()
    jump_dates = ret[ret.abs() > jump_threshold].index
    report = []
    for d in jump_dates:
        loc = price.index.get_loc(d)
        ratio = price.iloc[loc] / price.iloc[loc - 1]
        price.iloc[:loc] = price.iloc[:loc] * ratio
        report.append((d, ratio))
    return price, report


def load_price_daily(cache_path):
    """Load an already-cached {Date, price} CSV (produced by the notebook's
    fetch_total_return_price) and apply the same jump cleaning."""
    price = pd.read_csv(cache_path, index_col=0, parse_dates=True)["price"]
    price, corrections = clean_price_series(price)
    if corrections:
        print(f"[clean_price_series] {cache_path}: {len(corrections)} jump(s) back-adjusted")
    return price


def load_cash_rate_series(weekly_index, cache_path, fallback_annual_rate=0.005):
    """
    Load a cached {date, annual_rate} savings-deposit-rate CSV and forward-fill it onto a weekly
    price index. If the cache file is missing, fall back to a flat assumed rate (documented
    simplification; savings rates move little, so this has limited effect on conclusions). Used
    identically for the TWD and USD rate series.
    """
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date").sort_index()
        s = df["annual_rate"].reindex(weekly_index, method="ffill").bfill()
        print(f"[cash rate] loaded {cache_path}")
        return s
    print(f"[cash rate] {cache_path} not found -- falling back to a flat "
          f"{fallback_annual_rate:.2%} annual rate.")
    return pd.Series(fallback_annual_rate, index=weekly_index)


def build_markets():
    """Reproduces cells 2-9 of the notebook: load both markets' daily price series, apply the
    common-window trim (so Taiwan and United States cover an identical date range), resample to
    weekly, and load each market's cash-reserve rate. Returns the same MARKETS dict structure
    used throughout the notebook and manuscript."""
    price_daily_tw = load_price_daily(TW_PRICE_CACHE)
    price_daily_us = load_price_daily(US_PRICE_CACHE)

    # --- common-window trim, copied verbatim from cell 9 ---
    common_start = max(price_daily_tw.index.min(), price_daily_us.index.min())
    common_end = min(price_daily_tw.index.max(), price_daily_us.index.max())
    print(f"Common analysis window: {common_start.date()} to {common_end.date()}")

    tw_unaffected = (common_start == price_daily_tw.index.min()) and (common_end == price_daily_tw.index.max())
    print(f"Taiwan window unaffected by the trim: {'YES' if tw_unaffected else 'NO -- CHECK THIS'}")
    assert tw_unaffected, "Taiwan's window should never be affected by the US-side trim"

    price_daily_tw_trimmed = price_daily_tw.loc[common_start:common_end]
    price_daily_us_trimmed = price_daily_us.loc[common_start:common_end]

    price_weekly_tw = price_daily_tw_trimmed.resample("W-FRI").last().dropna()
    price_weekly_tw.name = "price"
    price_weekly_us = price_daily_us_trimmed.resample("W-FRI").last().dropna()
    price_weekly_us.name = "price"

    cash_rate_weekly_tw = load_cash_rate_series(price_weekly_tw.index, TWD_RATE_CACHE)
    cash_rate_weekly_us = load_cash_rate_series(price_weekly_us.index, USD_RATE_CACHE)

    print(f"Taiwan weekly series: {len(price_weekly_tw)} weeks ({len(price_weekly_tw) / 52:.1f} years)")
    print(f"United States weekly series: {len(price_weekly_us)} weeks ({len(price_weekly_us) / 52:.1f} years)")

    return {
        "Taiwan (0050)": dict(
            price=price_weekly_tw, cash_rate=cash_rate_weekly_tw,
            weekly_contribution=WEEKLY_CONTRIBUTION_TWD, currency="NT$", slug="taiwan",
        ),
        "United States (SPY)": dict(
            price=price_weekly_us, cash_rate=cash_rate_weekly_us,
            weekly_contribution=WEEKLY_CONTRIBUTION_USD, currency="US$", slug="us",
        ),
    }


# ============================================================
# Backtest engine -- copied verbatim from cell 11
# ============================================================
def run_backtest(price, cash_rate, correction_threshold=CORRECTION_THRESHOLD,
                  lookback=LOOKBACK_WEEKS, cash_cap_ratio=CASH_CAP_RATIO,
                  alloc_ratio=0.0, buy_cost_rate=BUY_COST_RATE,
                  weekly_contribution=WEEKLY_CONTRIBUTION_TWD, deploy_fraction=1.0):
    """
    Run the tactical cash reserve backtest (alloc_ratio=0 reproduces the 100% DCA control).

    price               : weekly closing price series (pd.Series, indexed by week-ending date)
    cash_rate           : matching weekly annualized cash-reserve interest rate series
    alloc_ratio         : fraction of each contribution diverted to the cash reserve (0-1)
    deploy_fraction     : fraction of the accumulated reserve released on each trigger
                          (1.0 = full lump-sum, the paper's default)

    Returns a DataFrame with one row per week.
    """
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
        dt = idx[t]
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
            "date": dt, "price": p, "contribution": weekly_contribution,
            "cumulative_contribution": cumulative_contribution,
            "to_cash": to_cash, "to_equity_direct": to_equity_direct,
            "deployed": deployed, "triggered": triggered, "buy_cost": buy_cost,
            "equity_units": equity_units, "equity_value": equity_value,
            "cash_balance": cash_balance, "total_value": total_value,
            "drawdown_from_high": drawdown.iloc[t], "rolling_high": rolling_high.iloc[t],
        })

    return pd.DataFrame(records).set_index("date")


# ============================================================
# Trial matrix: N=5 allocation-ratio trials -> weekly excess-return matrix
# ============================================================
def build_returns_matrix(price, cash_rate, weekly_contribution):
    """
    For each allocation ratio in SENSITIVITY_ALLOC_RATIOS, run the backtest and compute the
    weekly return of NAV = total_value / cumulative_contribution (the same NAV definition the
    notebook/manuscript already use for MDD), net of the concurrent cash-reserve rate -- i.e. a
    weekly *excess* return over the local risk-free/opportunity-cost proxy, consistent with the
    paper's own Sortino-ratio convention (cash rate as MAR). Returns a (T weeks x N=5) DataFrame,
    columns labelled by allocation ratio, aligned on a shared date index.
    """
    cols = {}
    for r in SENSITIVITY_ALLOC_RATIOS:
        bt = run_backtest(price, cash_rate, alloc_ratio=r, weekly_contribution=weekly_contribution)
        nav = bt["total_value"] / bt["cumulative_contribution"]
        weekly_ret = nav.pct_change()
        weekly_excess = weekly_ret - (cash_rate / 52.0)
        cols[f"{r:.0%}"] = weekly_excess
    df = pd.DataFrame(cols).dropna(how="any")
    return df


# ============================================================
# CSCV / PBO core
# ============================================================
def annualized_sharpe(weekly_excess_returns):
    """Accepts either a pandas Series or a numpy array of weekly excess returns."""
    arr = np.asarray(weekly_excess_returns, dtype=float)
    mu = arr.mean() * 52.0
    sigma = arr.std(ddof=1) * sqrt(52.0)
    return mu / sigma if sigma > 0 else 0.0


def cscv_pbo(returns_matrix: pd.DataFrame, S: int):
    """
    Combinatorially Symmetric Cross-Validation (Bailey et al. 2016).

    returns_matrix : (T x N) DataFrame of weekly excess returns, one column per trial.
    S              : number of contiguous, equal-size time blocks to partition T into (must be
                     even). Every combination of S/2 blocks is used as the in-sample (IS) half,
                     with the complementary S/2 blocks as out-of-sample (OOS).

    For each combination: rank all N trials by IS Sharpe ratio, take the IS winner n*, find n*'s
    rank among the N OOS Sharpe ratios, convert the relative rank to a logit lambda. PBO is the
    fraction of combinations where lambda <= 0 (the IS winner performed at/below the OOS median).
    """
    assert S % 2 == 0, "S must be even"
    T, N = returns_matrix.shape
    block_size = T // S
    weeks_used = block_size * S
    weeks_dropped = T - weeks_used

    values = returns_matrix.iloc[:weeks_used].to_numpy()
    blocks = [values[i * block_size:(i + 1) * block_size] for i in range(S)]

    lambdas = []
    n_star_choice_counts = np.zeros(N, dtype=int)
    loss_count = 0

    for is_combo in itertools.combinations(range(S), S // 2):
        oos_combo = tuple(b for b in range(S) if b not in is_combo)
        is_data = np.concatenate([blocks[b] for b in is_combo], axis=0)
        oos_data = np.concatenate([blocks[b] for b in oos_combo], axis=0)

        sharpe_is = np.array([annualized_sharpe(is_data[:, n]) for n in range(N)])
        sharpe_oos = np.array([annualized_sharpe(oos_data[:, n]) for n in range(N)])

        n_star = int(np.argmax(sharpe_is))
        n_star_choice_counts[n_star] += 1

        # rank of n*'s OOS Sharpe among all N OOS Sharpes: 1 = worst, N = best
        rank_oos = 1 + int(np.sum(sharpe_oos < sharpe_oos[n_star]))
        omega = rank_oos / (N + 1)
        lam = log(omega / (1 - omega))
        lambdas.append(lam)

        if sharpe_oos[n_star] < 0:
            loss_count += 1

    lambdas = np.array(lambdas)
    n_combinations = len(lambdas)
    pbo = float(np.mean(lambdas <= 0))
    prob_loss = loss_count / n_combinations

    return dict(
        S=S, N=N, n_combinations=n_combinations,
        weeks_total=T, weeks_used=weeks_used, weeks_dropped=weeks_dropped,
        pbo=pbo, mean_lambda=float(np.mean(lambdas)), prob_loss=prob_loss,
        lambdas=lambdas,
        n_star_choice_counts=dict(zip(returns_matrix.columns, n_star_choice_counts.tolist())),
    )


# ============================================================
# Plotting
# ============================================================
BLUE = "#2a5c9e"
ORANGE = "#c9642a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def plot_lambda_distribution(result, market_name, out_path):
    """
    With N trials, the OOS rank of the IS winner can only take N distinct values (1..N), so
    lambda is inherently discrete with exactly N possible values, regardless of how many CSCV
    combinations feed into it. We plot this explicitly as a categorical bar chart -- one bar per
    possible OOS rank of the IS-selected trial -- rather than an auto-binned histogram, and label
    each bar with its rank so the "IS winner finished Nth of N out-of-sample" reading is direct.
    """
    lambdas = result["lambdas"]
    N = result["N"]
    pbo = result["pbo"]

    # Recover, for each combination, which of the N possible OOS ranks it corresponds to.
    # omega = rank/(N+1)  =>  rank = round(omega * (N+1)); omega = 1/(1+exp(-lambda))
    omega = 1.0 / (1.0 + np.exp(-lambdas))
    ranks = np.round(omega * (N + 1)).astype(int)
    rank_values = np.arange(1, N + 1)
    rank_lambdas = np.log((rank_values / (N + 1)) / (1 - rank_values / (N + 1)))
    counts = np.array([(ranks == r).sum() for r in rank_values])

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    colors = [ORANGE if lam <= 0 else BLUE for lam in rank_lambdas]
    bars = ax.bar(rank_lambdas, counts, width=0.28, color=colors, edgecolor="white", linewidth=0.5)
    for lam, c, r in zip(rank_lambdas, counts, rank_values):
        ax.text(lam, c + counts.max() * 0.015, f"rank {r}/{N}\n{c/len(lambdas):.1%}",
                 ha="center", va="bottom", fontsize=6.5)
    ax.axvline(0, color="#222222", lw=1.2, ls="--")
    ax.set_xlabel(r"Logit $\lambda = \ln(\omega / (1-\omega))$, one value per possible OOS rank"
                  f" (N={N} trials)")
    ax.set_ylabel("Number of CSCV combinations")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_title(
        f"{market_name}: CSCV logit distribution (S={result['S']}, N={N})\n"
        f"PBO = P(λ ≤ 0) = {pbo:.1%}  (n = {result['n_combinations']:,} combinations)",
        loc="left", fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    markets = build_markets()
    rows = []
    headline_results = {}

    for market_name, cfg in markets.items():
        print(f"\n=== {market_name} ===")
        returns_matrix = build_returns_matrix(cfg["price"], cfg["cash_rate"], cfg["weekly_contribution"])
        print(f"Returns matrix: {returns_matrix.shape[0]} weeks x {returns_matrix.shape[1]} trials "
              f"(columns: {list(returns_matrix.columns)})")

        for S in BLOCK_COUNTS:
            result = cscv_pbo(returns_matrix, S=S)
            expected_combos = comb(S, S // 2)
            assert result["n_combinations"] == expected_combos, \
                f"combination count mismatch: got {result['n_combinations']}, expected {expected_combos}"
            rows.append({
                "market": market_name,
                "S_blocks": S,
                "N_trials": result["N"],
                "n_combinations": result["n_combinations"],
                "weeks_total": result["weeks_total"],
                "weeks_used": result["weeks_used"],
                "weeks_dropped": result["weeks_dropped"],
                "PBO": result["pbo"],
                "mean_lambda": result["mean_lambda"],
                "prob_loss": result["prob_loss"],
            })
            print(f"  S={S:>2}: n_combinations={result['n_combinations']:>7,}  "
                  f"weeks_used={result['weeks_used']}  weeks_dropped={result['weeks_dropped']}  "
                  f"PBO={result['pbo']:.1%}  mean_lambda={result['mean_lambda']:+.3f}  "
                  f"prob_loss={result['prob_loss']:.1%}")
            print(f"       IS-winner choice counts: {result['n_star_choice_counts']}")

            if S == 16:
                headline_results[market_name] = result

    results_df = pd.DataFrame(rows)
    results_path = os.path.join(OUT_DIR, "pbo_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[saved] {results_path}")

    plot_lambda_distribution(headline_results["Taiwan (0050)"], "Taiwan (0050)",
                              os.path.join(OUT_DIR, "pbo_distribution_taiwan.pdf"))
    plot_lambda_distribution(headline_results["United States (SPY)"], "United States (SPY)",
                              os.path.join(OUT_DIR, "pbo_distribution_us.pdf"))

    return results_df, headline_results


if __name__ == "__main__":
    main()
