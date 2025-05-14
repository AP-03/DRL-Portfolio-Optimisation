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
portfolio_values = [10000.0]  # Initial value
weights_list = []

for _ in range(len(features_df) - 2):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # Get portfolio value from env info (which uses whole shares and prices)
    portfolio_value = info[0]["portfolio_value"]
    portfolio_values.append(portfolio_value)
    
    weights_list.append(info[0]["weights"])  # store actual weights used


# === Create a Series for Analysis ===
portfolio_series = pd.Series(portfolio_values, index=features_df.index[:-1])

# === Create a DataFrame for Weights ===
weights_df = pd.DataFrame(weights_list, index=features_df.index[:-2])  # align dimensions

# === Run Full Performance Analysis ===
analyze_performance(portfolio_series, weights=weights_df, title_prefix="PPO Training Data")
