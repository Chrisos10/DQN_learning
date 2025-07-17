import os
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation


def make_env(seed=None, flatten=False):
    def _init():
        env = gym.make("AssaultNoFrameskip-v4", render_mode=None, full_action_space=True)
        env = AtariWrapper(env)
        if flatten:
            env = FlattenObservation(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def train_dqn(hyperparams, policy_type="Cnn"):
    log_dir = f"logs/{policy_type.lower()}_assault"
    os.makedirs(log_dir, exist_ok=True)
    model_path = f"models1/dqn_{policy_type.lower()}_final.zip"

    flatten = policy_type == "Mlp"

    env = DummyVecEnv([make_env(seed=42, flatten=flatten)])
    eval_env = DummyVecEnv([make_env(seed=100, flatten=flatten)])

    model = DQN(
        policy=f"{policy_type}Policy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        learning_rate=hyperparams["learning_rate"],
        gamma=hyperparams["gamma"],
        batch_size=hyperparams["batch_size"],
        exploration_initial_eps=hyperparams["epsilon_start"],
        exploration_final_eps=hyperparams["epsilon_end"],
        exploration_fraction=hyperparams["epsilon_decay"]
    )

    model.learn(total_timesteps=200_000)
    model.save(model_path)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"{policy_type} Policy - Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    os.makedirs("models1", exist_ok=True)

    hyperparam_configs = [
        {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.1  # Fraction of total timesteps over which exploration decays
        }
    ]

    train_dqn(hyperparam_configs[0], policy_type="Mlp")