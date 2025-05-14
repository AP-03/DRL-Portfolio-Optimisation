import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from portfolioEnv import PortfolioEnv
from utils import build_env_windows
import torch
import warnings
warnings.filterwarnings("ignore")


def make_env(features, returns):
    def _init():
        return PortfolioEnv(features, returns)
    return _init


def evaluate_agent(model, env, n_episodes=10):
    """Run the agent in the env and return average episode reward."""
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    return np.mean(rewards)


if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)

    features_df = pd.read_csv('./data/features_df.csv', index_col='Date', parse_dates=True)
    returns_df = pd.read_csv('./data/returns_df.csv', index_col='Date', parse_dates=True)
    env_windows = build_env_windows(features_df, returns_df)

    prev_best_model_path = None

    for i, window in enumerate(env_windows):
        print(f"\n🔁 Training window {i+1}/10 ({window['years'][0]}–{window['years'][1]})")
        f_train, r_train = window['train']
        f_val, r_val     = window['val']

        best_reward = -np.inf
        best_model = None

        for seed in range(5):
            print(f"  🚀 Training agent {seed+1}/5")

            # Setup training env
            train_env = SubprocVecEnv([make_env(f_train, r_train) for _ in range(10)])
            train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

            # PPO policy setup
            policy_kwargs = dict(
                activation_fn=torch.nn.Tanh,
                net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                log_std_init=-1
            )

            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                verbose=1,
                tensorboard_log="./ppo_logs/",
                n_epochs=16,
                learning_rate=3e-4,
                n_steps=756,
                batch_size=1260,
                gae_lambda=0.9,
                gamma=0.9,
                clip_range=0.25,
                policy_kwargs=policy_kwargs,
                seed=seed
            )

            # If not the first window, load previous best model weights
            if prev_best_model_path:
                model.set_parameters(prev_best_model_path)

            model.learn(total_timesteps=7_500_000)

            # Validation env (unwrapped, no learning)
            val_env = DummyVecEnv([make_env(f_val, r_val)])
            val_env = VecNormalize(val_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
            val_env.training = False
            val_env.norm_reward = False

            mean_val_reward = evaluate_agent(model, val_env, n_episodes=10)
            print(f"     🔎 Val Reward: {mean_val_reward:.4f}")

            if mean_val_reward > best_reward:
                best_reward = mean_val_reward
                best_model = model

        # Save best model from this window
        model_path = f"./models/ppo_window_{i}_best.zip"
        best_model.save(model_path)
        prev_best_model_path = model_path

        print(f"✅ Best agent for window {i} saved: {model_path}")
