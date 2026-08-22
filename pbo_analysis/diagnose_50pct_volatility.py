"""
Diagnostic follow-up to pbo_summary.md's flagged observation: the 50%-allocation trial is
chosen as the CSCV in-sample winner far more often than a 1-in-5 baseline would predict, in
both markets but more consequentially for the United States PBO result (94.9%). This script
investigates the mechanical cause directly, rather than speculating.

Hypothesis under test: the 50% trial's per-block (S=16) annualized Sharpe ratio is more
*dispersed* across blocks than the other four trials' -- i.e. it occasionally spikes high in a
given block by chance (making it the in-sample "winner" disproportionately often) without that
spike persisting into other blocks -- and this is mechanically tied to how much is deployed, and
when, relative to that trial's cash-reserve size.

Reuses build_markets(), run_backtest(), build_returns_matrix() from pbo_cscv.py unchanged (same
directory), so this is guaranteed to use the identical trial set and block partition as the PBO
analysis already reported.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pbo_cscv import (
    SENSITIVITY_ALLOC_RATIOS, BLUE, ORANGE,
    build_markets, run_backtest, annualized_sharpe,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
S = 16  # match the PBO headline block count


def block_partition(n_weeks, S):
    block_size = n_weeks // S
    weeks_used = block_size * S
    return block_size, weeks_used


def analyze_market(market_name, cfg):
    print(f"\n{'=' * 70}\n{market_name}\n{'=' * 70}")

    trial_bts = {}
    trial_excess = {}
    for r in SENSITIVITY_ALLOC_RATIOS:
        bt = run_backtest(cfg["price"], cfg["cash_rate"], alloc_ratio=r,
                           weekly_contribution=cfg["weekly_contribution"])
        nav = bt["total_value"] / bt["cumulative_contribution"]
        weekly_excess = nav.pct_change() - (cfg["cash_rate"] / 52.0)
        trial_bts[r] = bt
        trial_excess[r] = weekly_excess

    # Align everything on the same (post-dropna) index, exactly as build_returns_matrix does.
    returns_matrix = pd.DataFrame({f"{r:.0%}": trial_excess[r] for r in SENSITIVITY_ALLOC_RATIOS}).dropna(how="any")
    idx = returns_matrix.index
    n_weeks = len(idx)
    block_size, weeks_used = block_partition(n_weeks, S)
    print(f"{n_weeks} weeks available, {S} blocks of {block_size} weeks ({weeks_used} weeks used, "
          f"{n_weeks - weeks_used} dropped)")

    block_bounds = [(i * block_size, (i + 1) * block_size) for i in range(S)]
    block_dates = [(idx[a], idx[b - 1]) for a, b in block_bounds]

    # --- per-block annualized Sharpe, per trial ---
    sharpe_table = pd.DataFrame(
        {f"{r:.0%}": [annualized_sharpe(returns_matrix[f"{r:.0%}"].to_numpy()[a:b]) for a, b in block_bounds]
         for r in SENSITIVITY_ALLOC_RATIOS},
        index=[f"blk{i+1}" for i in range(S)],
    )

    print("\nPer-block annualized Sharpe ratio (rows=blocks, cols=trials):")
    print(sharpe_table.round(3).to_string())

    cross_block_std = sharpe_table.std(ddof=1).rename("cross_block_sharpe_std")
    cross_block_range = (sharpe_table.max() - sharpe_table.min()).rename("cross_block_sharpe_range")
    print("\nCross-block dispersion of each trial's Sharpe estimate (higher = more likely to spike as IS winner by chance):")
    print(pd.concat([cross_block_std, cross_block_range], axis=1).round(3).to_string())

    # --- deployment activity per block, per trial: does deployment intensity track Sharpe dispersion? ---
    deploy_rows = []
    for r in SENSITIVITY_ALLOC_RATIOS:
        bt = trial_bts[r].loc[idx[:weeks_used]]
        n_triggers = [int(bt["triggered"].to_numpy()[a:b].sum()) for a, b in block_bounds]
        total_deployed = [float(bt["deployed"].to_numpy()[a:b].sum()) for a, b in block_bounds]
        avg_cash_weight = [float((bt["cash_balance"] / bt["total_value"]).to_numpy()[a:b].mean()) for a, b in block_bounds]
        deploy_rows.append(pd.Series(n_triggers, name=f"{r:.0%}_n_triggers"))
        deploy_rows.append(pd.Series(total_deployed, name=f"{r:.0%}_deployed"))
        deploy_rows.append(pd.Series(avg_cash_weight, name=f"{r:.0%}_avg_cash_wt"))
    deploy_table = pd.concat(deploy_rows, axis=1)
    deploy_table.index = [f"blk{i+1}" for i in range(S)]

    print("\nTrigger count per block, all trials share the same trigger weeks (deploy_fraction=1.0,\n"
          "identical price series -- only the *amount* deployed differs by allocation ratio):")
    trig_cols = [c for c in deploy_table.columns if c.endswith("_n_triggers")]
    print(deploy_table[trig_cols].to_string())

    print("\nAverage cash weight per block, 10% vs 50% trial (mechanical exposure difference):")
    print(deploy_table[["10%_avg_cash_wt", "50%_avg_cash_wt"]].round(3).to_string())

    # --- correlation: does a block's 50% deployed-dollar amount predict how extreme its 50% Sharpe is? ---
    dep_50 = deploy_table["50%_deployed"].to_numpy()
    sharpe_50 = sharpe_table["50%"].to_numpy()
    sharpe_50_abs_dev = np.abs(sharpe_50 - sharpe_50.mean())
    if dep_50.std() > 0:
        corr = np.corrcoef(dep_50, sharpe_50_abs_dev)[0, 1]
        print(f"\nCorrelation between block's 50%-trial deployed $ and |block Sharpe - mean Sharpe| "
              f"for the 50% trial: {corr:+.3f}")

    # Identify the single most extreme block for the 50% trial and describe it.
    extreme_blk = int(np.argmax(np.abs(sharpe_50 - sharpe_50.mean())))
    d0, d1 = block_dates[extreme_blk]
    print(f"\nMost extreme block for the 50% trial: blk{extreme_blk+1} "
          f"({d0.date()} to {d1.date()}), Sharpe={sharpe_50[extreme_blk]:.3f} "
          f"(market mean across blocks: {sharpe_50.mean():.3f})")
    print(f"  Triggers in that block (10%..50%): "
          f"{[int(deploy_table.loc[f'blk{extreme_blk+1}', f'{r:.0%}_n_triggers']) for r in SENSITIVITY_ALLOC_RATIOS]}")
    print(f"  $ deployed in that block (10%..50%): "
          f"{[round(deploy_table.loc[f'blk{extreme_blk+1}', f'{r:.0%}_deployed'], 0) for r in SENSITIVITY_ALLOC_RATIOS]}")

    return dict(sharpe_table=sharpe_table, deploy_table=deploy_table, cross_block_std=cross_block_std,
                block_dates=block_dates, extreme_blk=extreme_blk)


def plot_block_sharpe(market_name, result, out_path):
    sharpe_table = result["sharpe_table"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(SENSITIVITY_ALLOC_RATIOS)))
    for r, color in zip(SENSITIVITY_ALLOC_RATIOS, colors):
        col = f"{r:.0%}"
        lw = 2.4 if col == "50%" else 1.1
        ax.plot(range(1, len(sharpe_table) + 1), sharpe_table[col], marker="o", ms=3.5,
                lw=lw, color=(ORANGE if col == "50%" else color), label=col,
                zorder=5 if col == "50%" else 3)
    ax.axhline(0, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel("Block index (S=16, chronological)")
    ax.set_ylabel("Block-level annualized Sharpe ratio")
    ax.set_title(f"{market_name}: per-block Sharpe ratio by allocation-ratio trial\n"
                 f"(50% highlighted -- its cross-block std is "
                 f"{result['cross_block_std']['50%']:.2f} vs.\n"
                 f"{result['cross_block_std'].drop('50%').mean():.2f} average for the other four)",
                 loc="left", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out_path}")


def main():
    markets = build_markets()
    results = {}
    for market_name, cfg in markets.items():
        results[market_name] = analyze_market(market_name, cfg)

    plot_block_sharpe("Taiwan (0050)", results["Taiwan (0050)"],
                       os.path.join(OUT_DIR, "diagnostic_block_sharpe_taiwan.pdf"))
    plot_block_sharpe("United States (SPY)", results["United States (SPY)"],
                       os.path.join(OUT_DIR, "diagnostic_block_sharpe_us.pdf"))

    # Save the raw per-block tables for the record.
    for market_name, result in results.items():
        slug = "taiwan" if "Taiwan" in market_name else "us"
        result["sharpe_table"].to_csv(os.path.join(OUT_DIR, f"diagnostic_block_sharpe_{slug}.csv"))
        result["deploy_table"].to_csv(os.path.join(OUT_DIR, f"diagnostic_block_deploys_{slug}.csv"))
    print("\n[saved] diagnostic_block_sharpe_{taiwan,us}.csv, diagnostic_block_deploys_{taiwan,us}.csv")

    return results


if __name__ == "__main__":
    main()
