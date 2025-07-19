import gymnasium as gym
import torch
from stable_baselines3 import DQN
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv
import time

# Create the environment (same preprocessing as training)
def make_env():
    def _init():
        env = gym.make("AssaultNoFrameskip-v4", render_mode="human", full_action_space=True)
        env = FlattenObservation(env)
        return env
    return _init

# Load environment
env = DummyVecEnv([make_env()])

# Load trained MLP model
model_path = "models_assault_mlp/dqn_mlp_assault_final"
model = DQN.load(model_path, env=env)

print("Loaded MLP model:", model_path)
print("Running agent... Press Ctrl+C to exit.")

# Run the agent
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)  # slow down slightly for visibility

    if done:
        obs = env.reset()
