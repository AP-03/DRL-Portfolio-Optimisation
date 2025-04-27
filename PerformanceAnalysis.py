# PerformanceAnalysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Main Analysis Function ===
def analyze_performance(portfolio_values, weights=None, title_prefix=""):
    """
    Args:
        portfolio_values (pd.Series): Daily portfolio value (indexed by Date)
        weights (pd.DataFrame): Optional, daily portfolio weights for turnover calc
        title_prefix (str): Prefix for plot titles
    """

    # --- Calculate Daily Returns ---
    returns = portfolio_values.pct_change().dropna()

    # --- Calculate Monthly Returns ---
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # --- Calculate Annual Returns ---
    annual_returns = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)

    # --- Plot Portfolio Value ---
    plt.figure(figsize=(12, 5))
    portfolio_values.plot()
    plt.title(f"{title_prefix} Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Monthly Returns Heatmap ---
    monthly_returns_df = monthly_returns.to_frame(name="Return")
    monthly_returns_df['Year'] = monthly_returns_df.index.year
    monthly_returns_df['Month'] = monthly_returns_df.index.month
    pivot_table = monthly_returns_df.pivot(index="Year", columns="Month", values="Return")


    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
    plt.title(f"{title_prefix} Monthly Returns Heatmap")
    plt.tight_layout()
    plt.show()

    # --- Annual Returns Bar Plot ---
    plt.figure(figsize=(10, 5))
    annual_returns.index = annual_returns.index.year
    annual_returns.plot(kind='bar')
    plt.axhline(annual_returns.mean(), color='black', linestyle='--', label='Mean')
    plt.title(f"{title_prefix} Annual Returns")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Distribution of Monthly Returns ---
    plt.figure(figsize=(10, 5))
    monthly_returns.hist(bins=50)
    plt.axvline(monthly_returns.mean(), color='yellow', linestyle='--', label='Mean')
    plt.title(f"{title_prefix} Distribution of Monthly Returns")
    plt.xlabel("Monthly Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Sharpe Ratio and Max Drawdown (Overall) ---
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    max_drawdown = (portfolio_values / portfolio_values.cummax() - 1).min()

    print("=== Performance Metrics ===")
    print(f"Sharpe Ratio     : {sharpe_ratio:.4f}")
    print(f"Max Drawdown     : {max_drawdown:.4f}")
    print(f"Annualized Return: {((1 + returns.mean()) ** 252 - 1):.4f}")
    print(f"Annual Volatility: {returns.std() * np.sqrt(252):.4f}")

    # --- Optional: Average Turnover ---
    if weights is not None:
        turnover = weights.diff().abs().sum(axis=1)
        avg_turnover = turnover.mean()
        print(f"Average Daily Turnover: {avg_turnover:.4f}")

    return {
        "daily_returns": returns,
        "monthly_returns": monthly_returns,
        "annual_returns": annual_returns,
        "sharpe": sharpe_ratio,
        "drawdown": max_drawdown
    }
