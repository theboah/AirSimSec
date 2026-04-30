import asyncio
import copy
import json
import os
from copy import deepcopy

import commentjson
import numpy as np
from projectairsim import ProjectAirSimClient, World
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from env_wrapper import AirSimEnv


class AirSimParallelVecEnv(DummyVecEnv):
    """DummyVecEnv variant that dispatches per-env async steps concurrently."""

    def __init__(self, env_fns):
        super().__init__(env_fns)
        self._step_loop = asyncio.new_event_loop()

    async def _step_wait_async(self):
        coroutines = [
            env.step_async(self.actions[env_idx])
            for env_idx, env in enumerate(self.envs)
        ]
        return await asyncio.gather(*coroutines)

    def step_wait(self):
        results = self._step_loop.run_until_complete(self._step_wait_async())

        for env_idx, (obs, rew, terminated, truncated, info) in enumerate(results):
            self.buf_rews[env_idx] = rew
            self.buf_infos[env_idx] = info
            self.buf_dones[env_idx] = bool(terminated or truncated)
            self.buf_infos[env_idx]["TimeLimit.truncated"] = bool(truncated and not terminated)

            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()

            self._save_obs(env_idx, obs)

        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def close(self):
        super().close()
        if self._step_loop is not None and not self._step_loop.is_closed():
            self._step_loop.close()
            self._step_loop = None


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

    # Spread cloned drones along East/Y to avoid immediate overlap collisions.
    xyz[1] += actor_index * float(spacing_m)
    origin["xyz"] = _format_xyz_string(xyz[0], xyz[1], xyz[2])

    return actor


def main():
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

    drone_names = [f"Drone{i + 1}" for i in range(env_num)]
    env_fns = [
        (
            lambda drone_name=drone_name: AirSimEnv(
                config=config,
                client=client,
                world=world,
                drone_name=drone_name,
            )
        )
        for drone_name in drone_names
    ]

    env = AirSimParallelVecEnv(env_fns)

    # Expose per-episode stats (`episode` info) so PPO logs ep_len_mean/ep_rew_mean.
    monitor_dir = config.get("monitor_log_dir")
    monitor_file = None
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        monitor_file = os.path.join(monitor_dir, "vec_monitor.csv")
    env = VecMonitor(env, filename=monitor_file)

    env = VecTransposeImage(env)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        device="cuda"
    )

    try:
        model.learn(total_timesteps=config.get("total_timesteps"), progress_bar=True)
        model.save("a_jam_parra_t1_10p_1m.zip")
    finally:
        env.close()
        client.disconnect()


if __name__ == "__main__":
    main()
