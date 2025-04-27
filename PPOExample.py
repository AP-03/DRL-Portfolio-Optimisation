import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from portfolioEnv import PortfolioEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from PerformanceAnalysis import analyze_performance

# === Load Data ===
features_df = pd.read_csv('./data/features_df.csv', index_col='Date', parse_dates=True)
returns_df = pd.read_csv('./data/returns_df.csv', index_col='Date', parse_dates=True)

# === Create Environment ===
def make_env():
    return PortfolioEnv(features_df, returns_df)

env = DummyVecEnv([make_env])

# === Load Trained Model ===
model = PPO.load("./models/ppo_portfolio")

# === Simulate the Portfolio ===
obs = env.reset()
portfolio_values = [10000.0]  # Start with 10K
weights_list = []

for _ in range(len(features_df) - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    new_value = portfolio_values[-1] * np.exp(reward[0])
    portfolio_values.append(new_value)

    # Record the action (portfolio weight)
    weights_list.append(action.flatten())

# === Create a Series for Analysis ===
portfolio_series = pd.Series(portfolio_values, index=features_df.index)

# === Create a DataFrame for Weights ===
weights_df = pd.DataFrame(weights_list, index=features_df.index[:-1])  # align dimensions

# === Run Full Performance Analysis ===
analyze_performance(portfolio_series, weights=weights_df, title_prefix="PPO Training Data")
