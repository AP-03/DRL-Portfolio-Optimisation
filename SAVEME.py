# portfolioEnv.py
import gym
from gym import spaces
import numpy as np
import pandas as pd
from collections import deque
import random
import torch
from scipy.special import softmax

# ─────────────────────────────────────────────────────────
# 1.  Differential Sharpe with full numeric guards
# ─────────────────────────────────────────────────────────
class DifferentialSharpeReward:
    def __init__(self, eta: float = 1 / 252, clip: float = 50.0, eps: float = 1e-12):
        self.eta, self.clip, self.eps = eta, clip, eps
        self.reset()

    def reset(self):
        self.A = 0.0   # EMA of returns
        self.B = 0.0   # EMA of squared returns

    def _finite(self, x: float) -> float:
        """Replace nan / inf with 0 (neutral)."""
        return 0.0 if not np.isfinite(x) else x

    def compute(self, Rt: float) -> float:
        Rt = self._finite(Rt)

        dA, dB = Rt - self.A, Rt**2 - self.B
        var    = self.B - self.A**2         # sample variance estimate

        # Guard-1: warm-up + tiny-negative round-off
        if (not np.isfinite(var)) or var < self.eps:
            Dt = 0.0
        else:
            Dt = (self.B * dA - 0.5 * self.A * dB) / np.power(var, 1.5)
            Dt = np.clip(Dt, -self.clip, self.clip)  # Guard-2
            Dt = self._finite(Dt)

        # EMA updates
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(Dt)


# ─────────────────────────────────────────────────────────
# 2.  Portfolio environment
# ─────────────────────────────────────────────────────────
LOOKBACK = 60        # days of return history
REGIME_COLS = ["SP500_logret", "vol20", "vol_ratio", "VIX"]

class PortfolioEnv(gym.Env):
    """
    Observation  = [ 60×N log-return history | current weights w | 4 regime feats ]
    Action       = raw vector in [0,1]^N  → softmax-like normalisation  (long-only)
    Reward       = Differential Sharpe ratio of portfolio log return
    """

    def __init__(self,
                 features_df: pd.DataFrame,
                 returns_df : pd.DataFrame):
        super().__init__()
        assert len(features_df) == len(returns_df)

        self.features_df = features_df.sort_index().reset_index(drop=True)
        self.returns_df  = returns_df .sort_index().reset_index(drop=True)
        self.n_assets    = returns_df.shape[1]

        # --- action & observation spaces -------------------------------------
        self.action_space = spaces.Box(low=0.0, high=1.0,
                                       shape=(self.n_assets,), dtype=np.float32)

        obs_len = (self.n_assets * LOOKBACK) + self.n_assets + len(REGIME_COLS)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_len,), dtype=np.float32)

        # --- helpers ----------------------------------------------------------
        self.ret_buffer   = deque(maxlen=LOOKBACK)
        self.prev_weights = np.zeros(self.n_assets, dtype=np.float32)
        self.diff_sharpe  = DifferentialSharpeReward()
        self.current_step = 0

    # ───────── gym API ───────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.diff_sharpe.reset()

        self.ret_buffer.clear()
        for _ in range(LOOKBACK):
            self.ret_buffer.append(np.zeros(self.n_assets, dtype=np.float32))

        self.prev_weights[:] = 0.0
        return self._obs(), {}

    
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        return [seed]

    def step(self, action):
        # 1. convert raw action → long-only weights  Σw = 1
        w = softmax(action)

        # 2. fetch today’s per-asset log-returns (already finite from preprocessing)
        r_log = self.returns_df.iloc[self.current_step].values.astype(np.float32)

        # 3. exact portfolio log-return  log(Σ w_i e^{r_i})
        Rt = float(np.logaddexp.reduce(r_log + np.log(w + 1e-12)))

        # 4. reward
        reward = self.diff_sharpe.compute(Rt)

        # 5. update history buffer & weights
        self.ret_buffer.append(r_log)
        self.prev_weights = w.astype(np.float32)
        self.current_step += 1

        done = self.current_step >= len(self.features_df) - 1
        terminated = done
        truncated = False  # You can later add timeouts if needed

        obs = self._obs()
        info = {"raw_log_return": Rt, "weights": w}

        return obs, reward, terminated, truncated, info



    # ───────── helpers ───────────────────────────────────────────────────────
    def _obs(self):
        hist   = np.concatenate(list(self.ret_buffer), dtype=np.float32)
        regime = self.features_df.loc[self.current_step, REGIME_COLS].values.astype(np.float32)
        return np.concatenate([hist, self.prev_weights, regime]).astype(np.float32)
