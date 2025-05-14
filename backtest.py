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


def backtest():
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
    all_portfolios = []

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
        daily_returns_cum = pd.Series(np.diff(np.log(portfolio_values)), index=f_test.index[1:])
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

        # Combine and compound the log returns to get a continuous portfolio value
    combined_returns = pd.concat(all_returns).sort_index()
    cumulative_portfolio = 10000 * np.exp(combined_returns.cumsum())


    # === Save summary metrics for reuse ===
    annual_df.to_csv("./backtest_results/summary_metrics.csv", index=False)

    return combined_monthly_percentage, combined_annual_percentage, annual_df, cumulative_portfolio