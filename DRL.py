import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from portfolioEnv import PortfolioEnv

# === Create environment ===
def make_env():
    def _init():
        return PortfolioEnv(features_df, returns_df)
    return _init

if __name__ == "__main__":
    features_df = pd.read_csv('./data/features_df.csv', index_col='Date', parse_dates=True)
    returns_df = pd.read_csv('./data/returns_df.csv', index_col='Date', parse_dates=True)

    n_envs = 10
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    # === Create PPO model ===
    model = PPO(
        policy="MlpPolicy",    # Use Multi-Layer Perceptron policy
        env=env,
        verbose=1,             # Print training info
        learning_rate=3e-4,    # Standard PPO learning rate
        n_steps=2048 // n_envs,          # Number of steps to collect before each policy update
        batch_size=64,         # Minibatch size for SGD
        ent_coef=0.0,          # Entropy coefficient (exploration vs exploitation)
        gamma=0.99,            # Discount factor
        clip_range=0.2         # Clipping range for PPO stability
    )

    # === Train PPO ===
    total_timesteps = 500_000   # 500K steps (adjust depending on your compute)
    model.learn(total_timesteps=total_timesteps)

    # === Save model ===
    model.save("./models/ppo_portfolio")

    print("✅ Training complete and model saved!")
