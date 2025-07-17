# play.py
import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def make_env():
    def _init():
        env = gym.make("AssaultNoFrameskip-v4", render_mode="human")
        return AtariWrapper(env)
    return _init

env = DummyVecEnv([make_env()])
env = VecFrameStack(env, n_stack=4)

model = DQN.load("models/dqn_cnn_final", env=env, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

obs = env.reset()
done = False

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
