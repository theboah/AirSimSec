import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from gymnasium.envs.registration import register

with open("config.json", "r") as f:
    config = json.load(f)

register(
    id="AirSim",
    entry_point="env_wrapper:AirSimEnv",
    kwargs={"config": config},
)

env = make_vec_env("AirSim", n_envs=1, seed=42)
env = VecTransposeImage(env)

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/", device="cuda")
model.learn(total_timesteps=75_600, progress_bar=True)
model.save("airsimbase_ppo_HOUR_noATTACK_randomised")