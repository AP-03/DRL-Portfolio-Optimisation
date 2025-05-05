import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from portfolioEnv import PortfolioEnv
import torch

# === Create environment ===
def make_env():
    def _init():
        return PortfolioEnv(features_df, returns_df)
    return _init


if __name__ == "__main__":
    features_df = pd.read_csv('./data/features_df.csv', index_col='Date', parse_dates=True)
    returns_df = pd.read_csv('./data/returns_df.csv', index_col='Date', parse_dates=True)

    n_envs = 10
    env = VecNormalize(SubprocVecEnv([make_env() for _ in range(n_envs)]),norm_obs=False, norm_reward=True, clip_reward=10.0)

    policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,
    net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # pi=policy network, vf=value network
    log_std_init=-1
    )

    # === Create PPO model ===
    model = PPO(
        policy="MlpPolicy",    # Use Multi-Layer Perceptron policy
        env=env,
        verbose=1,             # Print training info
        tensorboard_log="./ppo_logs/",
        n_epochs=16,           # Number of epochs per update
        learning_rate=3e-4,    # Standard PPO learning rate
        n_steps=756,          # Number of steps to collect before each policy update
        batch_size=1260,         # Minibatch size for SGD
        gae_lambda=0.9,        # GAE lambda for advantage estimation
        gamma=0.9,            # Discount factor
        clip_range=0.25,         # Clipping range for PPO stability
        policy_kwargs=policy_kwargs
    )

    # === Train PPO ===
    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps)

    # === Save model ===
    model.save("./models/ppo_portfolio")

    print("✅ Training complete and model saved!")
