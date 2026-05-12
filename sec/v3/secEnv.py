import asyncio
import numpy as np
from copy import deepcopy
from stable_baselines3.common.vec_env import DummyVecEnv
from projectairsim import ProjectAirSimClient, World
from SingleDroneEnv import SecSingleDroneEnv


class SecEnvAll(DummyVecEnv):
    def __init__(self, config, scene_config_name, num_envs):
        self.config = config
        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, scene_config_name, delay_after_load_sec=7)

        list_of_env_fns = [
            lambda i=i: SecSingleDroneEnv(self.client, self.world, f"Drone{i}", self.config)
            for i in range(num_envs)
        ]
        super().__init__(list_of_env_fns)
        self._step_loop = asyncio.new_event_loop()  # one persistent loop
        asyncio.set_event_loop(self._step_loop)

    def step_wait(self):
        # Run all env steps concurrently on the persistent loop
        asyncio.set_event_loop(self._step_loop)
        results = self._step_loop.run_until_complete(
            asyncio.gather(*[
                env.step_async(self.actions[i])
                for i, env in enumerate(self.envs)
            ])
        )

        for i, (obs, rew, terminated, truncated, info) in enumerate(results):
            self.buf_rews[i] = rew
            self.buf_infos[i] = info
            self.buf_dones[i] = bool(terminated or truncated)
            self.buf_infos[i]["TimeLimit.truncated"] = bool(truncated and not terminated)

            if self.buf_dones[i]:
                self.buf_infos[i]["terminal_observation"] = obs
                obs, self.reset_infos[i] = self.envs[i].reset()

            self._save_obs(i, obs)

        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)

    def close(self):
        super().close()
        if self._step_loop and not self._step_loop.is_closed():
            self._step_loop.close()
        self.client.disconnect()