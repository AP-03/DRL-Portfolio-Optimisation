import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from portfolioEnv import PortfolioEnv
from utils import build_env_windows
from PerformanceAnalysis import analyze_performance
import matplotlib.pyplot as plt
import seaborn as sns

# === Load full feature/return data ===
features_df = pd.read_csv('./data/features_df.csv', index_col='Date', parse_dates=True)
returns_df = pd.read_csv('./data/returns_df.csv', index_col='Date', parse_dates=True)

# === Build the rolling windows (same as training setup) ===
env_windows = build_env_windows(features_df, returns_df)

# === Output directory ===
os.makedirs("./backtest_results", exist_ok=True)

# === Store full-period series ===
all_returns = []
all_monthly = []
annual_returns = []

for i, window in enumerate(env_windows):
    f_test, r_test = window['test']
    model_path = f"./models/ppo_window_{i}_best.zip"

    if not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}, skipping...")
        continue

    model = PPO.load(model_path)

    def make_env():
        return PortfolioEnv(f_test, r_test)

    env = DummyVecEnv([make_env])

    obs = env.reset()
    portfolio_values = [10000.0]
    weights_list = []

    for _ in range(len(f_test) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        log_r = info[0]["raw_log_return"]
        new_value = portfolio_values[-1] * np.exp(log_r)
        portfolio_values.append(new_value)
        weights_list.append(info[0]["weights"][:-1])  # exclude cash if turnover is based on assets only

    portfolio_series = pd.Series(portfolio_values, index=f_test.index)
    weights_df = pd.DataFrame(weights_list, index=f_test.index[:-1])

    # Compute returns and metrics for aggregation
    daily_returns = portfolio_series.pct_change().dropna()
    monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    annual_return = (1 + daily_returns.mean()) ** 252 - 1
    sharpe = ((daily_returns.mean()) / (daily_returns.std())* np.sqrt(252))
    drawdown = (portfolio_series / portfolio_series.cummax() - 1).min()
    turnover = weights_df.diff().abs().sum(axis=1).mean()
  #  print(f"Window {i}: Test dates from {f_test.index[0].date()} to {f_test.index[-1].date()} ({len(f_test)} days)")


    all_returns.append(daily_returns)
    all_monthly.append(monthly_returns)
    annual_returns.append({
        "year": window['years'][1],
        "return": annual_return,
        "sharpe": sharpe,
        "drawdown": drawdown,
        "turnover": turnover
    })

print(weights_df.sum(axis=1).describe())
# === Combine all for aggregate performance analysis ===
combined_returns = pd.concat(all_returns).sort_index()
combined_monthly = pd.concat(all_monthly).sort_index()
annual_df = pd.DataFrame(annual_returns)

combined_monthly_percentage = combined_monthly * 100
combined_annual_percentage = annual_df.copy()
combined_annual_percentage['return']=combined_annual_percentage['return'] * 100

# === Save summary metrics for reuse ===
annual_df.to_csv("./backtest_results/summary_metrics.csv", index=False)


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