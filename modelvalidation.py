import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from portfolioEnv import PortfolioEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# === Load the data used for training ===
features_df = pd.read_csv('./data/features_df.csv', index_col='Date', parse_dates=True)
returns_df = pd.read_csv('./data/returns_df.csv', index_col='Date', parse_dates=True)

# === Create the same environment ===
def make_env():
    return PortfolioEnv(features_df, returns_df)

env = DummyVecEnv([make_env])

# === Load the trained model ===
model = PPO.load("./models/ppo_portfolio")

# === Evaluate on training set ===
obs = env.reset()
portfolio_values = [10000]  # Start with $1 initial capital

for _ in range(len(features_df) - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # Grow portfolio value by exp(reward) because reward = daily log return
    new_value = portfolio_values[-1] * np.exp(reward[0])
    portfolio_values.append(new_value)

# === Create a Series for easy plotting ===
portfolio_series = pd.Series(portfolio_values, index=features_df.index)

# === Plot the Portfolio Performance ===
plt.figure(figsize=(14, 6))
portfolio_series.plot(label='PPO Portfolio (Training Data)', linewidth=2)
plt.title('PPO Portfolio Growth During Training')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
