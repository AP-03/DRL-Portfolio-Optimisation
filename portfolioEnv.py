import gym
from gym import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    def __init__(self, features_df, returns_df, initial_cash=1.0):
        super(PortfolioEnv, self).__init__()
        
        self.features_df = features_df
        self.returns_df = returns_df
        self.initial_cash = initial_cash
        
        self.current_step = 0
        
        # Observation space: vector of features (normalized)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(features_df.shape[1],), dtype=np.float32
        )
        
        # Action space: Portfolio weights (bounded between 0 and 1 for each asset)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(returns_df.shape[1],), dtype=np.float32
        )
        
    def reset(self):
        self.current_step = 0
        return self._next_observation()
    
    def step(self, action):
        # Normalize action to sum to 1 (portfolio weights)
        action = action / (action.sum() + 1e-8)
        
        # Calculate portfolio return
        portfolio_return = np.dot(self.returns_df.iloc[self.current_step].values, action)
        
        # Reward = daily portfolio log return
        reward = portfolio_return
        
        self.current_step += 1
        done = self.current_step >= len(self.features_df) - 1
        
        next_obs = self._next_observation()
        
        info = {"step": self.current_step}
        
        return next_obs, reward, done, info
    
    def _next_observation(self):
        return self.features_df.iloc[self.current_step].values.astype(np.float32)
