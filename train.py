import os
import gym
from gym.wrappers import ResizeObservation, GrayScaleObservation, TransformObservation, FlattenObservation
import torch
import json
import numpy as np
import pandas as pd
import random
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


"""
Trains DQN agents on Atari Assault using CNN/MLP policies with hyperparameter tuning.

Features:
- Multiple hyperparameter configurations for optimization
- Supports both CNN (pixel input) and MLP (preprocessed) policies
- Saves models, metrics, and training logs
- Reproducible through seed control and config saving

Usage Examples:
  python train.py --run-all               # Train all configurations
  python train.py --hparam-set 0 --policy CnnPolicy  # Specific experiment
"""
# Ensure numpy boolean type compatibility
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Configuration of the hyperparameter sets
# These sets are to explore different training dynamics and finetuning
HPARAM_SETS = [
    #  Set 1: Initially starting with a low learning rate and higher gamma for exploration
    {
        "id": "set1",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 100000,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.1,
        "learning_starts": 10000,
        "train_freq": 4,
        "target_update_interval": 1000
    },
    # Set 2: Increased learning rate and gamma, larger batch size to see if it improves performance through exploitation
    {
        "id": "set2", 
        "learning_rate": 2.5e-4,
        "gamma": 0.40,
        "batch_size": 64,
        "buffer_size": 50000,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "exploration_fraction": 0.2,
        "learning_starts": 5000,
        "train_freq": 4,
        "target_update_interval": 500
    },
    # Set 3: Further reduced buffer size and batch size, with a different exploration strategy 
    {
        "id": "set3",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 50000,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.1,
        "learning_starts": 10000,
        "train_freq": 4,
        "target_update_interval": 1000
    },
    # Set 4: Changed the batch size and buffer size, eps decay, and learning starts
    {
        "id": "set4",
        "learning_rate": 1e-3,
        "gamma": 0.9,
        "batch_size": 16,
        "buffer_size": 25000,
        "exploration_initial_eps": 0.5,
        "exploration_final_eps": 0.02,
        "exploration_fraction": 0.05,
        "learning_starts": 1000,
        "train_freq": 1,
        "target_update_interval": 100
    }
]

class ComprehensiveLogger(BaseCallback):
    """
        Enhanced training logger callback that tracks and saves episode rewards and metrics.
        
        This callback extends Stable Baselines3's BaseCallback to:
        - Track per-episode rewards
        - Calculate running mean rewards
        - Save training metrics to CSV for analysis
        - Maintain full training history
        
        Args:
            log_path (str): Path to save the training metrics CSV file
            verbose (int, optional): Verbosity level. Defaults to 0.
        
        Attributes:
            log_path (str): Path where metrics will be saved
            episode_rewards (list): List of total rewards for each completed episode
            current_episode_reward (float): Accumulated reward for the current episode
            all_metrics (list): List of dictionaries containing all recorded metrics
    """
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.all_metrics = []

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done = self.locals.get("dones", [False])[0]

        self.current_episode_reward += reward

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.all_metrics.append({
                "timestep": self.num_timesteps,
                "episode": len(self.episode_rewards),
                "reward": self.current_episode_reward,
                "mean_reward": np.mean(self.episode_rewards[-100:])
            })
            self.current_episode_reward = 0

        return True

    def _on_training_end(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        pd.DataFrame(self.all_metrics).to_csv(self.log_path, index=False)

def train(hparam_idx, policy_type):
    seed = 42  # Defined seed for reproducibility

    # Set global seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    assert 0 <= hparam_idx < len(HPARAM_SETS), "Invalid hyperparameter index"
    hparams = HPARAM_SETS[hparam_idx].copy()
    
    # Adjust hyperparameters for MLP
    if policy_type == "MlpPolicy":
        hparams.update({
            "buffer_size": min(10000, hparams["buffer_size"]),  # Further reduced buffer to avoid memory issues
            "batch_size": min(16, hparams["batch_size"]),  # Smaller batches
            "learning_starts": max(2500, hparams["learning_starts"] // 4)  # Earlier learning
        })
    
    # Directory setup
    base_name = f"{hparams['id']}_{policy_type.lower()}"
    model_dir = f"models/{base_name}"
    log_dir = f"logs/{base_name}"
    results_dir = f"results/{base_name}"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save hyperparameters
    with open(f"{results_dir}/hparams.json", "w") as f:
        json.dump(hparams, f, indent=4)

    # Environment setup
    if policy_type == "CnnPolicy":
        env = make_atari_env(
            "AssaultNoFrameskip-v4",
            n_envs=1,
            seed=seed,
            wrapper_kwargs={'clip_reward': False}
        )
    else:
        def make_env():
            env = gym.make("AssaultNoFrameskip-v4",
                         render_mode="rgb_array",
                         full_action_space=True)
            # preprocessing pipeline
            env = ResizeObservation(env, 28)  # Smaller to avoid memory issues
            env = GrayScaleObservation(env)
            env = TransformObservation(
                env, 
                lambda obs: obs.astype(np.float32) / 255.0 # Normalize pixel values
            )
            env = FlattenObservation(env)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return env
        
        base_env = DummyVecEnv([make_env])
        env = VecFrameStack(base_env, n_stack=1)

    # Model setup
    model_kwargs = {
        "policy": policy_type,
        "env": env,
        "device": "auto",
        "tensorboard_log": log_dir,
        **{k: v for k, v in hparams.items() 
           if k not in ["id", "policy_kwargs"]},
        **({"policy_kwargs": {"net_arch": [64, 64]}}  # Smaller network for computational efficiency
           if policy_type == "MlpPolicy" else {})
    }

    model = DQN(**model_kwargs)

    # Callbacks
    train_logger = ComprehensiveLogger(f"{results_dir}/training_metrics.csv")
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{model_dir}/best",
        log_path=f"{results_dir}/eval",
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Training with adjusted timesteps
    total_timesteps = 100000 if policy_type == "MlpPolicy" else 200000
    
    print(f"\nTraining {policy_type} with {hparams['id']}...")
    print(f"Using buffer_size={hparams['buffer_size']}, batch_size={hparams['batch_size']}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[train_logger, eval_callback],
        progress_bar=True,
        tb_log_name=f"train_{base_name}"
    )
    
    # Final saves
    model.save(f"{model_dir}/final_model")
    env.close()
    print(f"=== Training complete: {base_name} ===")

def run_all_experiments():
    print("=== Starting all experiments ===")
    
    # Run all hyperparameter sets for CnnPolicy
    print("\nRunning all hyperparameter sets with CnnPolicy")
    for hparam_idx in range(len(HPARAM_SETS)):
        train(hparam_idx, "CnnPolicy")
    
    # Run only first hyperparameter set for MlpPolicy
    print("\nRunning first hyperparameter set with MlpPolicy")
    train(0, "MlpPolicy")
    
    print("\n=== All experiments completed ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparam-set", type=int, required=False,
                       help="Hyperparameter set index (0, 1, ...)")
    parser.add_argument("--policy", type=str, required=False,
                       choices=["CnnPolicy", "MlpPolicy"],
                       help="Policy type to train")
    parser.add_argument("--run-all", action="store_true",
                       help="Run all experiments automatically")
    args = parser.parse_args()

    if args.run_all:
        run_all_experiments()
    elif args.hparam_set is not None and args.policy is not None:
        train(args.hparam_set, args.policy)
    else:
        print("Please specify either:")
        print("  --run-all to run all experiments")
        print("  OR")
        print("  Both --hparam-set and --policy to run a specific experiment")
        print("\nAvailable hyperparameter sets: 0-3")
        print("Available policies: CnnPolicy, MlpPolicy")

        # To automatically run all experiments, use 'python train.py --run-all'
        # To run a specific experiment for finetuning, use: 'python train.py --hparam-set 2 --policy CnnPolicy/MlpPolicy'