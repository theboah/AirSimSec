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

model = PPO.load("airsimbase_ppo_HOUR_noATTACK_randomised.zip", env=env, device="cuda")

obs = env.reset()
for _ in range(20_000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)

    # Render using the underlying gym env, bypassing SB3 VecEnv tile_images logic
    env.env_method("render")

    if dones.any():
        obs = env.reset()

env.close()