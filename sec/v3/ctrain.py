import json
import os

import commentjson
import copy
from projectairsim import ProjectAirSimClient, World
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from env_wrapper import AirSimEnv


def _parse_xyz_string(xyz: str):
    values = [float(part) for part in str(xyz).strip().split()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values for xyz, got: {xyz}")
    return values


def _format_xyz_string(x: float, y: float, z: float):
    return f"{x:.6f} {y:.6f} {z:.6f}"


def _clone_actor_with_spacing(actor_template, actor_index, spacing_m):
    actor = copy.deepcopy(actor_template)
    actor["name"] = f"Drone{actor_index + 1}"

    origin = actor.setdefault("origin", {})
    xyz = _parse_xyz_string(origin.get("xyz", "0 0 0"))
    xyz[1] += actor_index * float(spacing_m)
    origin["xyz"] = _format_xyz_string(xyz[0], xyz[1], xyz[2])
    return actor

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

env_num = int(config.get("env_num", 1))
if env_num <= 0:
    raise ValueError("env_num must be >= 1")

scene_dir = os.path.abspath(config.get("scene_config_path", script_dir))
scene_name = config.get("scene_filename", "scene_basic_drone.jsonc")
actor_spacing_m = float(config.get("parallel_actor_spacing_m", 4.0))

scene_path = os.path.join(scene_dir, scene_name)
with open(scene_path, "r", encoding="utf-8") as f:
    scene_config = commentjson.load(f)
actors = scene_config.get("actors", [])
if not actors:
    raise ValueError("Scene config must contain at least one actor")

template_actor = actors[0]
generated_actors = []
for idx in range(env_num):
    if idx < len(actors):
        actor = copy.deepcopy(actors[idx])
        actor["name"] = f"Drone{idx + 1}"
    else:
        actor = _clone_actor_with_spacing(template_actor, idx, actor_spacing_m)
    generated_actors.append(actor)

scene_config["actors"] = generated_actors

tmp_scene_name = "sec_scene_config_tmp.json"
tmp_scene_path = os.path.join(scene_dir, tmp_scene_name)
with open(tmp_scene_path, "w", encoding="utf-8") as f:
    json.dump(scene_config, f, indent=2)

client = ProjectAirSimClient()
client.connect()

world = World(
    client,
    scene_config_name=tmp_scene_name,
    delay_after_load_sec=4,
    sim_config_path=scene_dir,
)

drone_name = config.get("drone_name", "Drone1")
env = DummyVecEnv(
    [
        lambda: AirSimEnv(
            config=config,
            client=client,
            world=world,
            drone_name=drone_name,
        )
    ]
)

monitor_dir = config.get("monitor_log_dir")
monitor_file = None
if monitor_dir:
    os.makedirs(monitor_dir, exist_ok=True)
    monitor_file = os.path.join(monitor_dir, "vec_monitor.csv")
env = VecMonitor(env, filename=monitor_file)

env = VecTransposeImage(env)

# Load existing model instead of creating a new one
model = PPO.load("a_gps_jam_h1_t1.zip", env=env, device="cuda")

try:
    # Keep training from where it left off
    model.learn(total_timesteps=75_600, progress_bar=True, reset_num_timesteps=False)

    # Save back (same name to overwrite, or new name for versioning)
    model.save("a_gps_jam_h1_t1")
finally:
    env.close()
    client.disconnect()