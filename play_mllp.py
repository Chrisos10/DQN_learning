import gymnasium as gym
import torch
from stable_baselines3 import DQN
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, TransformObservation, FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import time
import ale_py
from ale_py import ALEInterface
import numpy as np

# === Settings ===
ale = ALEInterface()
gym.register_envs(ale_py)


def make_env():
    def _init():
        # using ALE and full action space
        env = gym.make("ALE/Assault-v5", render_mode="human", full_action_space=True)

        env = ResizeObservation(env, (28, 28))
        env = GrayscaleObservation(env)
        env = FlattenObservation(env)

        return env
    return _init


env = DummyVecEnv([make_env()])
env = VecFrameStack(env, n_stack=1)

model_path = "models/set1_mlppolicy/best/best_model"
model = DQN.load(model_path, env=env, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

print("Loaded MLP model:", model_path)
print("Running agent... Press Ctrl+C to exit.")

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)

    if done:
        obs = env.reset()