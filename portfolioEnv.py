import gym
from gym import spaces
import numpy as np
import pandas as pd
from collections import deque
import random
import torch
from scipy.special import softmax

class DifferentialSharpeReward:
    def __init__(self, eta: float = 1 / 252, clip: float = 50.0, eps: float = 1e-12):
        self.eta, self.clip, self.eps = eta, clip, eps
        self.reset()

    def reset(self):
        self.A = 0.0
        self.B = 0.0

    def _finite(self, x: float) -> float:
        return 0.0 if not np.isfinite(x) else x

    def compute(self, Rt: float) -> float:
        Rt = self._finite(Rt)
        dA, dB = Rt - self.A, Rt**2 - self.B
        var = self.B - self.A**2
        if (not np.isfinite(var)) or var < self.eps:
            Dt = 0.0
        else:
            Dt = (self.B * dA - 0.5 * self.A * dB) / np.power(var, 1.5)
            Dt = np.clip(Dt, -self.clip, self.clip)
            Dt = self._finite(Dt)
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(Dt)

LOOKBACK = 60
REGIME_COLS = ["SP500_logret", "vol20", "vol_ratio", "VIX"]

class PortfolioEnv(gym.Env):
    def __init__(self, features_df: pd.DataFrame, returns_df: pd.DataFrame):
        super().__init__()
        assert len(features_df) == len(returns_df)
        self.features_df = features_df.sort_index().reset_index(drop=True)
        self.returns_df = returns_df.sort_index().reset_index(drop=True)
        self.asset_cols = returns_df.columns.tolist()
        self.price_df = self.features_df[self.asset_cols].copy()
        self.n_assets = len(self.asset_cols)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        obs_len = (self.n_assets * LOOKBACK) + (self.n_assets + 1) + len(REGIME_COLS)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        self.ret_buffer = deque(maxlen=LOOKBACK)
        self.diff_sharpe = DifferentialSharpeReward()
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.diff_sharpe.reset()
        self.ret_buffer.clear()
        for _ in range(LOOKBACK):
            self.ret_buffer.append(np.zeros(self.n_assets, dtype=np.float32))
        self.prev_weights = np.zeros(self.n_assets + 1, dtype=np.float32)  # last one is cash
        self.prev_weights[-1] = 1.0  # 100% cash initially
        self.portfolio_value = 10000.0
        self.cash = self.portfolio_value
        self.current_prices = self.price_df.iloc[0].values.astype(np.float32)
        return self._obs(), {}

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        return [seed]

    def step(self, action):
        weights = softmax(action)
        prices_today = self.current_prices
        prices_next = self.price_df.iloc[self.current_step + 1].values.astype(np.float32)

        capital = self.portfolio_value
        dollar_alloc = weights * capital
        shares = np.floor(dollar_alloc / prices_today)
        invested = shares * prices_today
        cash = capital - invested.sum()

        new_value = float(np.dot(shares, prices_next) + cash)
        Rt = np.log(new_value / capital + 1e-12)
        reward = self.diff_sharpe.compute(Rt)

        self.portfolio_value = new_value
        self.cash = cash
        self.current_prices = prices_next
        self.ret_buffer.append(self.returns_df.iloc[self.current_step].values.astype(np.float32))

        # record current weights including cash
        weights_with_cash = invested / new_value
        cash_weight = cash / new_value
        self.prev_weights = np.append(weights_with_cash, cash_weight).astype(np.float32)

     #   if self.current_step < 5 or self.current_step > len(self.features_df) - 5:
     #   print(f"[DEBUG] Step {self.current_step}")
     #   print(f"  prices_today: {prices_today}")
     #   print(f"  shares: {shares}")
     #   print(f"  prices_next: {prices_next}")
     #   print(f"  portfolio_value: {new_value}")


        self.current_step += 1
        done = self.current_step >= len(self.features_df) - 2
        terminated = done
        truncated = False

        obs = self._obs()
        info = {
            "raw_log_return": Rt,
            "weights": self.prev_weights,
            "shares": shares,
            "cash": cash,
            "portfolio_value": new_value
        }

        return obs, reward, terminated, truncated, info

    def _obs(self):
        hist = np.concatenate(list(self.ret_buffer), dtype=np.float32)
        regime = self.features_df.loc[self.current_step, REGIME_COLS].values.astype(np.float32)
        return np.concatenate([hist, self.prev_weights, regime]).astype(np.float32)
