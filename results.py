import os
import pandas as pd
import numpy as np
from utils import build_env_windows
from PerformanceAnalysis import analyze_performance
import matplotlib.pyplot as plt
import seaborn as sns
from backtest import backtest
from metrics import compute_metrics

def results():
    combined_monthly_percentage,combined_annual_percentage, annual_df,cum_portfolio,combined_returns = backtest()

    drl_metrics = compute_metrics(combined_returns)

    metrics_df = pd.DataFrame({
    "Metric": drl_metrics.keys(),
    "DRL": drl_metrics.values()
   # "MVO": [mvo_metrics[k] for k in drl_metrics.keys()]
    })

    pd.set_option("display.float_format", "{:.4f}".format)
    print(metrics_df)



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

        # === Plot portfolio value over time ===
    plt.figure(figsize=(12, 6))
    plt.plot(cum_portfolio, label="Cumulative Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig("./backtest_results/portfolio_value_over_time.png")
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



if __name__ == "__main__":
    results()