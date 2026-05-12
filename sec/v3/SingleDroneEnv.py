import numpy as np
import asyncio
import time

import gymnasium as gym
from gymnasium import spaces
from projectairsim.drone import YawControlMode

from SecDrone import SecDrone
from SecUtils import get_random_start_pose, get_random_goal, distance_to_goal, is_goal_reached, is_outside_arena, is_outside_step_limit

import time

MAX_WAIT_S = 5.0
POLL_INTERVAL_S = 0.00001

class SecSingleDroneEnv(gym.Env):
    def __init__(self, client, world, name, config):
        super().__init__()
        self.drone = SecDrone(client, world, name)
        self.config = config
        self.event_loop = None
        self.action_duration = config.get("action_duration_s")
        self.step_penalty = config.get("step_penalty")
        self.goal_reached_bonus = config.get("goal_reached_bonus")
        self.collision_penalty = config.get("collision_penalty")
        self.ep_step_count = 0
        self.action_shape = (4,)
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=self.action_shape,
            dtype=np.float32,
        )
        obs_shape = config.get("observation_image_shape", [84, 84, 3])
        self.obs_image_height = int(obs_shape[0])
        self.obs_image_width = int(obs_shape[1])
        self.obs_image_channels = int(obs_shape[2])
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.obs_image_height,
                        self.obs_image_width,
                        self.obs_image_channels,
                    ),
                    dtype=np.uint8,
                ),
                "gps": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(9,),
                    dtype=np.float32,
                ),
                "goal_vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )
    


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.drone.collided = False
        self.drone.img_msg = None  # ← explicitly clear stale frame
        self.ep_step_count = 0

        self.drone.enable_api_control()
        self.drone.arm()

        self.episode_start_pose = get_random_start_pose(self.config)
        self.goal_coordinates = get_random_goal(self.config)

        self.drone.set_pose(self.episode_start_pose, reset_kinematics=True)

        # Wait until the camera publishes a fresh frame after the pose reset
        deadline = time.monotonic() + MAX_WAIT_S
        while self.drone.img_msg is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"[{self.drone.name}] Timed out waiting for first camera frame after reset"
                )
            time.sleep(POLL_INTERVAL_S)

        state = self.drone.get_obs()
        self.previous_goal_distance = distance_to_goal(state.get("pose"), self.goal_coordinates)
        return self._build_observation(state), {}
    
    def step(self, action):
        # Keep for compatibility but it should never be called by SecEnvAll
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.action(action))
    
    async def step_async(self, action):
        return await self.action(action)
    
    async def action(self, action):
        v_north = float(action[0])
        v_east = float(action[1])
        v_down = float(action[2])
        yaw_rate = float(action[3])
        move_task = await self.drone.move_by_velocity_async(
            v_north,
            v_east,
            v_down,
            duration=(self.action_duration),
            yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
            yaw_is_rate=True,
            yaw=yaw_rate,
        )
        await move_task
        self.ep_step_count += 1
        
        return self._get_state()
    
    def _get_state(self):
        state = self.drone.get_obs()
        info = {}
        
        current_goal_distance = distance_to_goal(state.get("pose"), self.goal_coordinates)
        
        goal_reached = is_goal_reached(self.config, state.get("pose"), self.goal_coordinates)
        
        out_of_arena = is_outside_arena(self.config, state.get("pose"))
        
        timeout = is_outside_step_limit(self.config, self.ep_step_count)
        if self.ep_step_count != 0 and state.get("collided"):
            print("COLLISION DETECTED")
        terminated = goal_reached or (self.ep_step_count != 5 and state.get("collided")) or out_of_arena
        truncated = timeout
        print(terminated, truncated)
            
        reward = self._get_reward(
            previous_goal_distance = self.previous_goal_distance,
            current_goal_distance = current_goal_distance,
            goal_reached=goal_reached,
            collision_detected=state.get("collided"),
            out_of_arena=out_of_arena,
            timeout=timeout
        )
        
        self.previous_goal_distance = current_goal_distance
        
        return self._build_observation(state), reward, terminated, truncated, info

    def _build_observation(self, state):
        pose = state.get("pose") or {}
        position = {}
        if isinstance(pose, dict):
            position = pose.get("position") or pose.get("translation") or {}
        try:
            pos_x = float(position.get("x", position.get("X", 0.0)))
            pos_y = float(position.get("y", position.get("Y", 0.0)))
            pos_z = float(position.get("z", position.get("Z", 0.0)))
        except Exception:
            pos_x = 0.0
            pos_y = 0.0
            pos_z = 0.0
        gps_msg = state.get("gps")
        goal_n = self.goal_coordinates.get("n", self.goal_coordinates.get("x", 0.0))
        goal_e = self.goal_coordinates.get("e", self.goal_coordinates.get("y", 0.0))
        goal_d = self.goal_coordinates.get("d", self.goal_coordinates.get("z", 0.0))
        goal_vector = np.array([
            goal_n - pos_x,
            goal_e - pos_y,
            goal_d - pos_z,
        ], dtype=np.float32)
        gps_vector = np.array([
            float(gps_msg.get("latitude", 0.0)) if gps_msg is not None else 0.0,
            float(gps_msg.get("longitude", 0.0)) if gps_msg is not None else 0.0,
            float(gps_msg.get("altitude", 0.0)) if gps_msg is not None else 0.0,
            float((gps_msg.get("velocity") or {}).get("x", 0.0)) if gps_msg is not None else 0.0,
            float((gps_msg.get("velocity") or {}).get("y", 0.0)) if gps_msg is not None else 0.0,
            float((gps_msg.get("velocity") or {}).get("z", 0.0)) if gps_msg is not None else 0.0,
            float(gps_msg.get("eph", 0.0)) if gps_msg is not None else 0.0,
            float(gps_msg.get("epv", 0.0)) if gps_msg is not None else 0.0,
            float(gps_msg.get("fix_type", 0.0)) if gps_msg is not None else 0.0,
        ], dtype=np.float32)
        return {"image": state.get("image"), "gps": gps_vector, "goal_vector": goal_vector}
    
    def _get_reward(self, previous_goal_distance: float, current_goal_distance: float,
                   goal_reached: bool, collision_detected: bool, out_of_arena: bool, timeout: bool):
        
        delta_distance = previous_goal_distance - current_goal_distance
        max_distance_per_step = (3 ** 0.5) * self.action_duration

        reward = 0
        reward += self.step_penalty
        
        progress_reward = np.clip(delta_distance / max_distance_per_step, -1.0, 1.0)
        reward += progress_reward

        if goal_reached:
            reward += self.goal_reached_bonus
        if collision_detected:
            reward += self.collision_penalty
        if out_of_arena:
            reward += self.collision_penalty
        if timeout:
            reward += self.collision_penalty

        return reward