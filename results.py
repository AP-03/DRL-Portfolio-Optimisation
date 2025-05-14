import os
import pandas as pd
import numpy as np
from utils import build_env_windows
from PerformanceAnalysis import analyze_performance
import matplotlib.pyplot as plt
import seaborn as sns
from backtest import backtest
from MVO_final import run_mvo

def results():
    combined_monthly_percentage,combined_annual_percentage, annual_df = backtest()
    combined_monthly_mvo, annual_df_mvo, portfolio_value_mvo, metrics_mvo = run_mvo()

    # === Annual Return Bar Plot ===
    plt.figure(figsize=(10, 5))
    sns.barplot(x="year", y="return", data=combined_annual_percentage, color="skyblue")
    plt.axhline(combined_annual_percentage['return'].mean(), color='black', linestyle='--', label='Mean')
    plt.title("Annual Returns Across All Test Years")
    plt.ylabel("Return")
    plt.xlabel("Year")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./backtest_results/combined_annual_returns.png")
    plt.show()

    # === Monthly Heatmap ===
    heatmap_df = combined_monthly_percentage.to_frame(name="Return")
    heatmap_df['Year'] = heatmap_df.index.year
    heatmap_df['Month'] = heatmap_df.index.month
    pivot_table = heatmap_df.pivot(index="Year", columns="Month", values="Return")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
    plt.title("Monthly Returns Heatmap Across All Windows")
    plt.tight_layout()
    plt.savefig("./backtest_results/combined_monthly_heatmap.png")
    plt.show()

    # === Monthly Return Distribution ===
    plt.figure(figsize=(10, 5))
    combined_monthly_percentage.hist(bins=50, edgecolor='black')
    plt.axvline(combined_monthly_percentage.mean(), color='yellow', linestyle='--', label='Mean')
    plt.title("Distribution of Monthly Returns (All Windows)")
    plt.xlabel("Monthly Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./backtest_results/combined_monthly_distribution.png")
    plt.show()

    # === DRL Metrics Line Plots ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.lineplot(ax=axes[0], x="year", y="sharpe", data=annual_df, marker="o", color="orangered", label="DRL")
    axes[0].axhline(0, linestyle="--", color="dodgerblue")
    axes[0].set_title("Sharpe Ratio")
    axes[0].set_ylabel("Sharpe")
    axes[0].set_xlabel("Backtest Year")
    axes[0].grid(True)

    sns.lineplot(ax=axes[1], x="year", y="drawdown", data=annual_df, marker="o", color="orangered", label="DRL")
    axes[1].set_title("Maximum Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Backtest Year")
    axes[1].grid(True)

    sns.lineplot(ax=axes[2], x="year", y="turnover", data=annual_df, marker="o", color="orangered", label="DRL")
    axes[2].set_title("Avg. Daily Change in Portfolio Weights")
    axes[2].set_ylabel("Δp_w")
    axes[2].set_xlabel("Backtest Year")
    axes[2].grid(True)

    plt.suptitle("Backtest Performance: Deep RL Portfolio Allocation", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("./backtest_results/backtest_drl_metrics.png")
    plt.show()





    # MVO





    # --- Annual Return Horizontal Bar Plot ---
    plt.figure(figsize=(8, 6))
    sns.barplot(
        y="year",
        x="return",
        data=annual_df_mvo,
        color="skyblue",
        orient="h"          # horizontal orientation
    )
    # mean line now vertical at the mean return
    mean_ret = annual_df_mvo['return'].mean()
    plt.axvline(mean_ret, color='black', linestyle='--', label=f'Mean = {mean_ret:.1%}')

    plt.title("Annual Returns Across All Test Years")
    plt.xlabel("Return")
    plt.ylabel("Year")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("./backtest_results/mvo_combined_annual_returns_horizontal.png")
    plt.show()

    # --- 3) Monthly Heatmap ---
    # Recompute monthly portfolio values
    monthly_value = portfolio_value_mvo.resample('M').last()
    monthly_growth = monthly_value.pct_change()

    # Add year and month
    heatmap_value = monthly_growth.to_frame(name="growth")
    heatmap_value['Year']  = heatmap_value.index.year
    heatmap_value['Month'] = heatmap_value.index.month
    pivot_growth = heatmap_value.pivot(index="Year", columns="Month", values="growth")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_growth, annot=True, fmt=".2%", cmap="RdYlGn", center=0)
    plt.title("Monthly Portfolio Value Growth")
    plt.tight_layout()
    plt.savefig("./backtest_results/mvo_portfolio_growth_heatmap.png")
    plt.show()


    # --- 4) Monthly Return Distribution ---
    plt.figure(figsize=(10, 5))
    combined_monthly_mvo.hist(bins=50, edgecolor='black')
    plt.axvline(combined_monthly_mvo.mean(), color='yellow', linestyle='--', label='Mean')
    plt.title("Distribution of Monthly Returns (All Windows)")
    plt.xlabel("Monthly Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./backtest_results/mvo_combined_monthly_distribution.png")
    plt.show()

    # --- MVO Metrics Line Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Sharpe
    sns.lineplot(ax=axes[0], x="year", y="sharpe", data=annual_df_mvo, marker="o",
                color="blue", label="MVO")
    axes[0].axhline(0, linestyle="--", color="dodgerblue")
    axes[0].set_title("Sharpe Ratio")
    axes[0].set_ylabel("Sharpe")
    axes[0].set_xlabel("Backtest Year")
    axes[0].grid(True)

    # Drawdown
    sns.lineplot(ax=axes[1], x="year", y="drawdown", data=annual_df_mvo, marker="o",
                color="blue", label="MVO")
    axes[1].set_title("Maximum Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Backtest Year")
    axes[1].grid(True)

    # Turnover
    sns.lineplot(ax=axes[2], x="year", y="turnover", data=annual_df_mvo, marker="o",
                color="blue", label="MVO")
    axes[2].set_title("Avg. Daily Change in Portfolio Weights")
    axes[2].set_ylabel("Δp_w")
    axes[2].set_xlabel("Backtest Year")
    axes[2].grid(True)

    plt.suptitle("Backtest Performance: MVO Portfolio Allocation", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("./backtest_results/backtest_mvo_metrics.png")
    plt.show()


    plt.figure(figsize=(10, 5))
    portfolio_value_mvo.plot(color="dodgerblue", linewidth=2)
    plt.title("Portfolio Value Over Time (Initial Balance: $100,000)")
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./backtest_results/mvo_portfolio_value_over_time.png")
    plt.show()

    # =========================
    # 7. Summary Metrics
    # =========================

    # 2) Plot it as a Matplotlib table
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')  # no axes

    # Create the table: one column of Values, rows = Metric names
    table = ax.table(
        cellText=metrics_mvo[['Value']].values,
        rowLabels=metrics_mvo['Metric'],
        colLabels=['Value'],
        cellLoc='center',
        rowLoc='center',
        loc='center'
    )

    # Tweak appearance
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)  # cell width, height multiplier

    plt.title("Backtest Summary Metrics", pad=20)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    results()