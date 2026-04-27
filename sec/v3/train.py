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

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/", device="cuda", n_steps=16000, batch_size=16000)
model.learn(total_timesteps=500_000, progress_bar=True)
model.save("a_none_goal_check_t1_500k.zip")