import os
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from gymnasium.wrappers import FlattenObservation

# Flattened Atari environment for MLP policy
def make_env(seed=None):
    def _init():
        env = gym.make("AssaultNoFrameskip-v4", render_mode=None, full_action_space=True)
        env = FlattenObservation(env)
        if seed is not None:
            env.seed(seed)
        return env
    return _init

# Custom training logger callback
class TrainingLogger(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.all_metrics = []

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done = self.locals.get("dones", [False])[0]

        self.current_episode_reward += reward
        self.current_episode_length += 1

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.all_metrics.append({
                "timestep": self.num_timesteps,
                "episode": len(self.episode_rewards),
                "reward": self.current_episode_reward,
                "length": self.current_episode_length,
                "mean_reward": np.mean(self.episode_rewards[-100:]),
                "mean_length": np.mean(self.episode_lengths[-100:])
            })
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

    def _on_training_end(self) -> None:
        pd.DataFrame(self.all_metrics).to_csv(self.log_path, index=False)

# Training function
def train_dqn(hyperparams, policy_type):
    os.makedirs("results1", exist_ok=True)
    os.makedirs("logs1", exist_ok=True)
    os.makedirs("models1", exist_ok=True)

    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([make_env(seed=42)])
    eval_env = DummyVecEnv([make_env(seed=42)])

    model = DQN(
        policy=f"{policy_type}Policy",
        env=env,
        verbose=1,
        tensorboard_log=f"logs1/{policy_type.lower()}_assault",
        device="cuda" if torch.cuda.is_available() else "cpu",  # GPU support
        **hyperparams
    )

    print("Using device:", model.device)

    train_logger = TrainingLogger(f"results1/{policy_type.lower()}_metrics.csv")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models1/best_{policy_type.lower()}",
        log_path=f"logs1/{policy_type.lower()}_eval",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    print(f"Training {policy_type} policy on Assault with MLP...")
    model.learn(
        total_timesteps=200_000,
        callback=[train_logger, eval_callback],
        progress_bar=True
    )
    model.save(f"models1/dqn_{policy_type.lower()}_assault_final")

    env.close()
    eval_env.close()

# MLP hyperparameters
hyperparam_configs = [
    {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 16,
        "buffer_size": 10000,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "exploration_fraction": 0.1,
        "learning_starts": 1000,
        "train_freq": 4,
        "target_update_interval": 1000
    }
]

if __name__ == "__main__":
    train_dqn(hyperparam_configs[0], policy_type="Mlp")