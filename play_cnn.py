# play.py
import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import ale_py
from ale_py import ALEInterface

# === Settings and Environment Setup ===
ale = ALEInterface()
gym.register_envs(ale_py)


def make_env():
    def _init():
        env = gym.make("AssaultNoFrameskip-v4", render_mode="human")
        return AtariWrapper(env)
    return _init

env = DummyVecEnv([make_env()])
env = VecFrameStack(env, n_stack=1)

# === Load the pre-trained DQN model ===
model = DQN.load(
    "models/set1_cnnpolicy/best/final_model",
    env=env,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    custom_objects={
        "lr_schedule": lambda _: 0.0001,
        "exploration_schedule": None,
        "replay_buffer": None
    }
)
obs = env.reset()
done = False

# === Main loop for playing the game ===
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
