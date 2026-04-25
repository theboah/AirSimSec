import copy
import os
import sys
from typing import Any, Dict, List
import asyncio
import time
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.drone import YawControlMode
from projectairsim.utils import (
    quaternion_to_rpy,
    rpy_to_quaternion,
    load_scene_config_as_dict,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

def get_image_data(image_msg):
    """Get decoded image data array from image messages"""
    if image_msg is not None:
        nparr = np.frombuffer(image_msg["data"], dtype="uint8")
        img_np = np.reshape(nparr, [image_msg["height"], image_msg["width"], 3])
        return img_np
    raise ValueError("image_msg is None")

class AirSimEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    EARTH_RADIUS_M = 6378137.0

    def __init__(self, config, render_mode=None, **kwargs):
        # Load config
        scene_path = config.get("scene_config_path")
        dt = config.get("dt", 0.1)
        goal_tolerance = config.get("goal_tolerance", 0.5)
        goal_coords = config.get("goal_coords", [0.0, 0.0, 0.0])
        goal_gps_coords = config.get("goal_gps_coords")
        self.randomization_config = (
            config.get("randomization", {}) if isinstance(config.get("randomization"), dict) else {}
        )
        self.randomization_enabled = bool(self.randomization_config.get("enabled", False))
        self.randomization_seed = self.randomization_config.get("seed")
        self.goal_randomization_config = (
            self.randomization_config.get("goal", {})
            if isinstance(self.randomization_config.get("goal"), dict)
            else {}
        )
        self.start_randomization_config = (
            self.randomization_config.get("start", {})
            if isinstance(self.randomization_config.get("start"), dict)
            else {}
        )
        self.limit_x, self.limit_y, self.limit_z = config.get("limit_xyz", [50.0, 50.0, 50.0])
        self.step_limit = config.get("step_limit", 2000)

        self.scene_path = os.path.abspath(scene_path)
        self.scene_dir = os.path.dirname(self.scene_path)
        self.scene_name = config.get("scene_filename") or os.path.basename(self.scene_path)
        self.render_mode = render_mode
        self.env_config = config

        self.action_shape = (4,)  # V_north, V_east, V_down, yaw_rate_cmd
        obs_shape = config.get("observation_image_shape", [128, 128, 3])
        self.obs_image_height = int(obs_shape[0])
        self.obs_image_width = int(obs_shape[1])
        self.obs_image_channels = int(obs_shape[2])
        self.observation_space = spaces.Dict(
            {
                "rgb_image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.obs_image_height,
                        self.obs_image_width,
                        self.obs_image_channels,
                    ),
                    dtype=np.uint8,
                ),
                "pose": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4,),
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
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=self.action_shape,
            dtype=np.float32,
        )
        self.yaw_rate_max_rad_s = float(config.get("max_yaw_rate_rad_s", np.pi / 2.0))

        self.save_obs_example = bool(config.get("save_obs_example", True))
        self.obs_example_path = config.get(
            "obs_example_path",
            os.path.join(self.scene_dir, "obs_example.png"),
        )
        self._saved_obs_example = False

        self.annotation_msg = {}
        self.img_msg = None
        self.gps_msg = None
        self.actor_pose = None
        self.state = None
        self.collision_info = None
        self.accept_collision_events = False
        self.collision_accept_after = 0.0
        self.collision_ignore_duration_sec = float(
            config.get("collision_ignore_duration_sec", 1.0)
        )
        self.takeoff_ack_timeout_sec = float(
            config.get("takeoff_ack_timeout_sec", 4.0 * 0.02)
        )
        self.reset_takeoff_wait_sec = float(
            config.get("reset_takeoff_wait_sec", 1.5)
        )
        self.reset_land_wait_sec = float(
            config.get("reset_land_wait_sec", 1.0)
        )
        self.pose_reset_position_tol_m = float(
            config.get("pose_reset_position_tol_m", 0.05)
        )
        self.pose_reset_quat_tol = float(config.get("pose_reset_quat_tol", 1e-3))
        self.pose_reset_max_retries = int(config.get("pose_reset_max_retries", 3))
        self.subscribed_topics = []
        self.step_num = 0
        
        
        
        

        client = ProjectAirSimClient()
        client.connect()
        scene_config, _ = load_scene_config_as_dict(self.scene_name, self.scene_dir)
        world = World(
            client,
            scene_config_name=self.scene_name,
            delay_after_load_sec=4,
            sim_config_path=self.scene_dir,
        )
        drone = Drone(client, world, "Drone1")
        self.client, self.world, self.actor = client, world, drone
        self.initial_actor_pose = self.actor.get_ground_truth_pose()
        self.initially_landed = self._is_currently_landed()
        self.done = False

        self.dt = dt
        self.goal_tolerance = goal_tolerance
        self.fixed_goal_gps_coords = (
            goal_gps_coords if goal_gps_coords is not None else goal_coords
        )
        self.goal_gps_coords = self.fixed_goal_gps_coords
        self.goal_coords = self.goal_gps_coords
        self.home_geo_point = self._get_home_geo_point(scene_config)
        self.goal_local_ned = [0.0, 0.0, 0.0]
        self.prev_goal_distance = None
        self.episode_index = 0
        self.episode_start_pose = None
        self.episode_goal_gps_coords = self.goal_gps_coords

        self.progress_reward_scale = float(config.get("progress_reward_scale", 1.0))
        self.step_penalty = float(config.get("step_penalty", 0.01))
        self.goal_reached_bonus = float(config.get("goal_reached_bonus", 10.0))
        self.collision_penalty = float(config.get("collision_penalty", 15.0))
        self.out_of_arena_penalty = float(config.get("out_of_arena_penalty", 10.0))
        self.timeout_penalty = float(config.get("timeout_penalty", 2.0))

        self.arena_norm_factor = np.sqrt(
            self.limit_x**2 + self.limit_y**2 + self.limit_z**2
        )

    def reset(self, *, seed=None, options=None):
        reset_t0 = time.perf_counter()
        print("[RESET] start")
        effective_seed = seed
        if effective_seed is None and self.randomization_seed is not None:
            effective_seed = int(self.randomization_seed) + int(self.episode_index)
        super().reset(seed=effective_seed)
        self.done = False
        self.step_num = 0
        self.actor_pose = None
        self.state = None
        self.collision_info = None
        self.img_msg = None
        self.gps_msg = None
        self.annotation_msg = {}
        self.accept_collision_events = False
        self.collision_accept_after = time.time() + self.collision_ignore_duration_sec
        
        t = time.perf_counter()
        self.world.pause()
        print(f"[RESET] world.pause() took {time.perf_counter() - t:.3f}s")

        t = time.perf_counter()
        self._cancel_active_task()
        print(f"[RESET] _cancel_active_task() took {time.perf_counter() - t:.3f}s")

        self.episode_goal_gps_coords, self.goal_local_ned, self.episode_start_pose = (
            self._sample_episode_setup()
        )
        self.goal_gps_coords = self.episode_goal_gps_coords
        self.goal_coords = self.goal_gps_coords

        t = time.perf_counter()
        self._reset_actor_pose(self.episode_start_pose)
        print(f"[RESET] _reset_actor_pose() took {time.perf_counter() - t:.3f}s")

        t = time.perf_counter()
        self._ensure_subscriptions()
        print(f"[RESET] _ensure_subscriptions() took {time.perf_counter() - t:.3f}s")

        t = time.perf_counter()
        self.actor.enable_api_control()
        self.actor.arm()
        print(f"[RESET] enable_api_control()+arm() took {time.perf_counter() - t:.3f}s")

        if self.initially_landed:
            t = time.perf_counter()
            self._run_coroutine_sync(self._takeoff_and_wait_async())
            print(f"[RESET] _takeoff_and_wait_async() took {time.perf_counter() - t:.3f}s")

        # Advance Sim by a dt step to get initial observation
        t = time.perf_counter()
        self.world.continue_for_sim_time(self.dt * 1e9, wait_until_complete=True)
        print(f"[RESET] continue_for_sim_time(dt) took {time.perf_counter() - t:.3f}s")

        # Wait until an initial image observation is received
        t = time.perf_counter()
        image_wait_loops = 0
        while self.img_msg is None:
            image_wait_loops += 1
            time.sleep(self.dt)
        print(
            f"[RESET] initial image wait took {time.perf_counter() - t:.3f}s "
            f"({image_wait_loops} loop(s))"
        )

        self.accept_collision_events = True
        t = time.perf_counter()
        state = self.get_state()
        print(f"[RESET] get_state() took {time.perf_counter() - t:.3f}s")
        self.prev_goal_distance = self._goal_distance(state)
        obs = {
            "rgb_image": state["rgb_image"],
            "pose": np.asarray(state["pose"], dtype=np.float32),
            "goal_vector": np.asarray(state["goal_vector"], dtype=np.float32),
        }
        print(f"[RESET] done in {time.perf_counter() - reset_t0:.3f}s")
        self.episode_index += 1
        return obs, {}

    def step(self, action):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    
        return loop.run_until_complete(self.step_async(action))

    def close(self):
        self.client.disconnect()

    async def _take_action(self,action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        # Action:{Vx: float, Vy: float, Vz: float, Vyaw: float}
        elif isinstance(action, Dict):
            action = [
                action["Vx"],
                action["Vy"],
                action["Vz"],
                action.get("Vyaw", action.get("yaw_rate", 0.0)),
            ]
        v_north = action[0]
        v_east = action[1]
        v_down = action[2]
        yaw_rate_cmd = action[3]
        yaw_rate = float(np.clip(yaw_rate_cmd, -1.0, 1.0)) * self.yaw_rate_max_rad_s

        move_task = await self.actor.move_by_velocity_async(
            v_north,
            v_east,
            v_down,
            duration=(self.dt),
            yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
            yaw_is_rate=True,
            yaw=yaw_rate,
        )
        self.world.continue_for_sim_time(self.dt * 1e9, wait_until_complete=True)

    async def step_async(self, action):
        timeout = False
        goal_reached = False
        out_of_arena = False
        
        self._take_action(action)
        
        next_state = self.get_state()
        ground_truth_info = next_state
        observed_info = next_state
        
        #OBSERVED INFO
        #attack observed info here
        
        
        #TRUE INFO
        true_current_goal_distance = self._goal_distance(ground_truth_info)
        previous_goal_distance = (
            true_current_goal_distance
            if self.prev_goal_distance is None
            else self.prev_goal_distance
        )
        
        collision_detected = bool(self.done and self._is_collision_event(self.collision_info))
        if collision_detected:
            print("COLLISION")
        elif true_current_goal_distance <= self.goal_tolerance:
            self.done = True
            goal_reached = True
            print("GOAL REACHED")
        elif self.is_outside_arena(ground_truth_info.get("pose")):
            self.done = True
            out_of_arena = True
            print("OUT OF ARENA")
        elif self.step_num >= self.step_limit:
            self.done = True
            timeout = True
            print("EPISODE TIMEOUT")

        reward = self.get_reward(
            previous_goal_distance=previous_goal_distance,
            current_goal_distance=true_current_goal_distance,
            goal_reached=goal_reached,
            collision_detected=collision_detected,
            out_of_arena=out_of_arena,
            timeout=timeout,
        )
        terminated = self.done and not timeout
        self.step_num += 1
        self.prev_goal_distance = true_current_goal_distance
        info = {
            "status": "Running",
            "gps_position": observed_info.get("gps_position"),
            "gps_displacement_to_goal": observed_info.get("gps_displacement_to_goal"),
            "goal_coords": self.goal_gps_coords,
        }
        next_obs = {
            "rgb_image": observed_info["rgb_image"],
            "pose": np.asarray(observed_info["pose"], dtype=np.float32),
            "goal_vector": np.asarray(observed_info["goal_vector"], dtype=np.float32),
        }
        return (next_obs, reward, terminated, timeout, info)

    def render(self, mode="human", close=False):
        if mode is None:
            mode = self.render_mode

        if self.img_msg is not None and mode == "human":
            self.display_debug_info(self.state, self.annotation_msg)

    def get_state(self):
        # Get pose of the Robot/Drone
        if self.actor_pose is None:
            self.actor_pose = self.get_vec_from_pose(
                self.world.get_object_pose(self.actor.name)
            )
        (actor_x, actor_y, actor_z, _, _, actor_yaw) = self.actor_pose
        sensor_gps_position = self.get_gps_position(self.gps_msg)
        gps_position = sensor_gps_position
        if gps_position is None and self.home_geo_point is not None:
            gps_position = self._local_ned_to_geo(
                [actor_x, actor_y, actor_z],
                self.home_geo_point,
            )
        displacement_to_goal = [
            actor_x - self.goal_local_ned[0],
            actor_y - self.goal_local_ned[1],
            actor_z - self.goal_local_ned[2],
        ]

        drone_pose: List = [actor_x, actor_y, actor_z, actor_yaw]
        # Populate the observation for the Agent
        self.state = {
            "pose": drone_pose,
            "rgb_image": self._get_processed_rgb_image(self.img_msg),
            "gps_position": gps_position,
            "displacement_to_goal": displacement_to_goal,
            "goal_vector": displacement_to_goal,
        }
        if self.goal_gps_coords is not None and self.state["gps_position"] is not None:
            self.state["gps_displacement_to_goal"] = [
                self.state["gps_position"][0] - self.goal_gps_coords[0],
                self.state["gps_position"][1] - self.goal_gps_coords[1],
                self.state["gps_position"][2] - self.goal_gps_coords[2],
            ]
        else:
            self.state["gps_displacement_to_goal"] = None
            
        return self.state

    def get_reward(
        self,
        previous_goal_distance: float,
        current_goal_distance: float,
        goal_reached: bool,
        collision_detected: bool,
        out_of_arena: bool,
        timeout: bool,
    ):
        """Progress-shaped reward with terminal event shaping."""
        reward = (
            self.progress_reward_scale
            * (previous_goal_distance - current_goal_distance)
            - self.step_penalty
        )

        if goal_reached:
            reward += self.goal_reached_bonus
        if collision_detected:
            reward -= self.collision_penalty
        if out_of_arena:
            reward -= self.out_of_arena_penalty
        if timeout:
            reward -= self.timeout_penalty

        return reward

    #Callbacks
    def _ensure_subscriptions(self):
        if self.subscribed_topics:
            return

        pose_topic = self.actor.robot_info["actual_pose"]
        self.client.subscribe(
            pose_topic,
            lambda _, pose: self.callback_pose(pose),
        )
        self.subscribed_topics.append(pose_topic)

        camera_topic = self.actor.sensors["ForwardViewCamera"]["scene_camera"]
        self.client.subscribe(
            camera_topic,
            self.callback_camera,
        )
        self.subscribed_topics.append(camera_topic)

        gps_topic = self.get_gps_topic()
        if gps_topic is not None:
            self.client.subscribe(gps_topic, self.callback_gps)
            self.subscribed_topics.append(gps_topic)

        collision_topic = self.actor.robot_info["collision_info"]
        self.client.subscribe(
            collision_topic,
            lambda _, collision_info: self.callback_collision(collision_info),
        )
        self.subscribed_topics.append(collision_topic)
    
    def callback_pose(self, pose):
        (
            actor_x,
            actor_y,
            actor_z,
            actor_roll,
            actor_pitch,
            actor_yaw,
        ) = self.get_vec_from_pose(pose)
        self.actor_pose = [
            actor_x,
            actor_y,
            actor_z,
            actor_roll,
            actor_pitch,
            actor_yaw,
        ]

    def callback_camera(self, topic, image_msg):
        self.img_msg = image_msg

    def _get_processed_rgb_image(self, image_msg):
        image = get_image_data(image_msg)
        if self.obs_image_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(
                image,
                (self.obs_image_width, self.obs_image_height),
                interpolation=cv2.INTER_AREA,
            )
            image = np.expand_dims(image, axis=-1)
        else:
            image = cv2.resize(
                image,
                (self.obs_image_width, self.obs_image_height),
                interpolation=cv2.INTER_AREA,
            )

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image
    
    def callback_gps(self, topic, gps_msg):
        self.gps_msg = gps_msg

    def get_gps_topic(self):
        for sensor in self.actor.sensors.values():
            topic = sensor.get("gps")
            if topic is not None:
                return topic
        return None

    def get_gps_position(self, gps_msg):
        if gps_msg is None:
            return None
        latitude = gps_msg.get("latitude")
        longitude = gps_msg.get("longitude")
        altitude = gps_msg.get("altitude")
        if latitude is None or longitude is None or altitude is None:
            return None
        return [latitude, longitude, altitude]
        
    def callback_collision(self, collision_info):
        self.collision_info = collision_info
        if not self.accept_collision_events:
            return
        if time.time() < self.collision_accept_after:
            return
        if not self._is_collision_event(collision_info):
            return
        self.done = True
        print("COLLISION")

    def _is_collision_event(self, collision_info):
        if isinstance(collision_info, dict):
            for key in ("has_collided", "hasCollided", "is_collision", "collision"):
                if key in collision_info:
                    return bool(collision_info[key])

            collision_msg_keys = {
                "time_stamp",
                "object_name",
                "segmentation_id",
                "position",
                "impact_point",
                "normal",
                "penetration_depth",
            }
            if collision_msg_keys.issubset(set(collision_info.keys())):
                return True

            flat_keys = {
                "impact_point_x",
                "impact_point_y",
                "impact_point_z",
                "position_x",
                "position_y",
                "position_z",
                "normal_x",
                "normal_y",
                "normal_z",
                "penetration_depth",
                "time_stamp",
            }
            if flat_keys.issubset(set(collision_info.keys())):
                return True

            return False

        return bool(collision_info)

    #Helpers

    def get_vec_from_pose(self, pose: Dict[str, Any]):
        """Get [x, y, z, roll, pitch, yaw] from Pose dict"""

        if "position" in pose:
            position = pose.get("position")
        else:
            position = pose.get("translation")

        x = position.get("x", 0.0)
        y = position.get("y", 0.0)
        z = position.get("z", 0.0)
        if "rotation" in pose:
            rotation = pose.get("rotation")
        else:
            rotation = pose.get("orientation")
        rot_w = rotation.get("w", 1.0)
        rot_x = rotation.get("x", 0.0)
        rot_y = rotation.get("y", 0.0)
        rot_z = rotation.get("z", 0.0)
        roll, pitch, yaw = quaternion_to_rpy(rot_w, rot_x, rot_y, rot_z)

        return (x, y, z, roll, pitch, yaw)

    def is_outside_arena(self, pose: List):
        x, y, z, _ = pose
        max_x = self.goal_local_ned[0] + self.limit_x
        min_x = self.goal_local_ned[0] - self.limit_x
        max_y = self.goal_local_ned[1] + self.limit_y
        min_y = self.goal_local_ned[1] - self.limit_y
        max_z = self.goal_local_ned[2] + self.limit_z
        min_z = self.goal_local_ned[2] - self.limit_z

        if (
            x <= min_x
            or x >= max_x
            or y <= min_y
            or y >= max_y
            or z <= min_z
            or z >= max_z
        ):
            return True
        return False

    def display_debug_info(
        self, state, annotation_msg, win_name="ProjectAirSimDetectAvoid"
    ):
        if state["rgb_image"] is not None:
            img_np = state["rgb_image"]
            for annotation in annotation_msg.get("annotations", []):
                bbox_center = annotation["bbox2d"]["center"]
                bbox_size = annotation["bbox2d"]["size"]
                v1 = (
                    int(bbox_center["x"] - (bbox_size["x"] / 2.0)),
                    int(bbox_center["y"] - (bbox_size["y"] / 2.0)),
                )
                v2 = (
                    int(bbox_center["x"] + (bbox_size["x"] / 2.0)),
                    int(bbox_center["y"] + (bbox_size["y"] / 2.0)),
                )
                cv2.rectangle(img_np, v1, v2, (0, 255, 0), 3)
            cv2.imshow(win_name, img_np)
            cv2.waitKey(15)
            global key
            key = cv2.waitKeyEx(20)

    def _cancel_active_task(self):
        try:
            self.actor.cancel_last_task()
        except Exception as exc:
            print(f"WARN: failed to cancel last task: {exc}")

    def _reset_actor_pose(self, target_pose=None):
        """Reset drone pose and kinematics to initial spawn pose."""
        try:
            target_pose = target_pose or self.initial_actor_pose
            current_pose = None
            diff = None
            for _ in range(max(1, self.pose_reset_max_retries)):
                self.actor.set_pose(target_pose, reset_kinematics=True)
                current_pose = self.actor.get_ground_truth_pose()
                is_valid, diff = self._is_pose_reset_valid(
                    current_pose,
                    target_pose,
                )
                if is_valid:
                    return

            print("BIG BAD ERROR: pose reset validation failed")
            print(f"target_pose: {target_pose}")
            print(f"current_pose: {current_pose}")
            print(f"pose_diff: {diff}")
        except Exception as exc:
            print(f"WARN: failed to reset actor pose: {exc}")

    def _is_pose_reset_valid(self, current_pose, target_pose):
        """Validate reset pose by translation and quaternion tolerances."""
        current_translation = (current_pose or {}).get("translation", {})
        target_translation = (target_pose or {}).get("translation", {})
        current_rotation = (current_pose or {}).get("rotation", {})
        target_rotation = (target_pose or {}).get("rotation", {})

        current_xyz = np.array(
            [
                float(current_translation.get("x", 0.0)),
                float(current_translation.get("y", 0.0)),
                float(current_translation.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
        target_xyz = np.array(
            [
                float(target_translation.get("x", 0.0)),
                float(target_translation.get("y", 0.0)),
                float(target_translation.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
        position_error_m = float(np.linalg.norm(current_xyz - target_xyz, ord=2))

        current_quat = np.array(
            [
                float(current_rotation.get("w", 1.0)),
                float(current_rotation.get("x", 0.0)),
                float(current_rotation.get("y", 0.0)),
                float(current_rotation.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
        target_quat = np.array(
            [
                float(target_rotation.get("w", 1.0)),
                float(target_rotation.get("x", 0.0)),
                float(target_rotation.get("y", 0.0)),
                float(target_rotation.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
        quat_error = float(
            min(
                np.linalg.norm(current_quat - target_quat, ord=2),
                np.linalg.norm(current_quat + target_quat, ord=2),
            )
        )

        frame_id_match = (current_pose or {}).get("frame_id") == (target_pose or {}).get(
            "frame_id"
        )
        is_valid = (
            position_error_m <= self.pose_reset_position_tol_m
            and quat_error <= self.pose_reset_quat_tol
            and frame_id_match
        )
        diff = {
            "position_error_m": position_error_m,
            "quat_error": quat_error,
            "frame_id_match": frame_id_match,
            "position_tol_m": self.pose_reset_position_tol_m,
            "quat_tol": self.pose_reset_quat_tol,
        }
        return is_valid, diff

    def _get_home_geo_point(self, scene_config):
        home = scene_config.get("home-geo-point") if isinstance(scene_config, dict) else None
        if not isinstance(home, dict):
            return None
        latitude = home.get("latitude")
        longitude = home.get("longitude")
        altitude = home.get("altitude")
        if latitude is None or longitude is None or altitude is None:
            return None
        return [float(latitude), float(longitude), float(altitude)]

    def _geo_to_local_ned(self, geo_coords, reference_geo_coords):
        """Convert [lat, lon, alt] to local [north, east, down] in meters."""
        lat, lon, alt = geo_coords
        ref_lat, ref_lon, ref_alt = reference_geo_coords
        d_lat = np.radians(float(lat) - float(ref_lat))
        d_lon = np.radians(float(lon) - float(ref_lon))
        ref_lat_rad = np.radians(float(ref_lat))
        north = self.EARTH_RADIUS_M * d_lat
        east = self.EARTH_RADIUS_M * np.cos(ref_lat_rad) * d_lon
        down = float(ref_alt) - float(alt)
        return [north, east, down]

    def _local_ned_to_geo(self, local_ned, reference_geo_coords):
        """Convert local [north, east, down] in meters to [lat, lon, alt]."""
        north, east, down = local_ned
        ref_lat, ref_lon, ref_alt = reference_geo_coords
        ref_lat_rad = np.radians(float(ref_lat))
        lat = float(ref_lat) + np.degrees(float(north) / self.EARTH_RADIUS_M)
        cos_lat = np.cos(ref_lat_rad)
        if abs(cos_lat) < 1e-9:
            lon = float(ref_lon)
        else:
            lon = float(ref_lon) + np.degrees(float(east) / (self.EARTH_RADIUS_M * cos_lat))
        alt = float(ref_alt) - float(down)
        return [lat, lon, alt]

    def _run_coroutine_sync(self, coroutine):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coroutine)

    def _sample_uniform(self, low: float, high: float) -> float:
        return float(self.np_random.uniform(low, high))

    def _goal_local_from_coords(self, goal_coords):
        if self.home_geo_point is None:
            return None
        return self._geo_to_local_ned(goal_coords, self.home_geo_point)

    def _pose_translation(self, pose):
        if pose is None:
            return None
        translation = pose.get("translation", pose.get("position", {}))
        if not isinstance(translation, dict):
            return None
        return [
            float(translation.get("x", 0.0)),
            float(translation.get("y", 0.0)),
            float(translation.get("z", 0.0)),
        ]

    def _sample_goal_coords(self):
        if not (self.randomization_enabled and self.goal_randomization_config.get("enabled", False)):
            return copy.deepcopy(self.fixed_goal_gps_coords)

        if self.goal_randomization_config.get("mode", "gps_box") != "gps_box":
            return copy.deepcopy(self.fixed_goal_gps_coords)

        gps_box = self.goal_randomization_config.get("gps_box", {})
        required_keys = (
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
            "alt_min",
            "alt_max",
        )
        if not all(key in gps_box for key in required_keys):
            return copy.deepcopy(self.fixed_goal_gps_coords)

        return [
            self._sample_uniform(float(gps_box["lat_min"]), float(gps_box["lat_max"])),
            self._sample_uniform(float(gps_box["lon_min"]), float(gps_box["lon_max"])),
            self._sample_uniform(float(gps_box["alt_min"]), float(gps_box["alt_max"])),
        ]

    def _sample_start_pose(self):
        pose = copy.deepcopy(self.initial_actor_pose)
        if not (self.randomization_enabled and self.start_randomization_config.get("enabled", False)):
            return pose

        if self.start_randomization_config.get("mode", "local_ned_box") != "local_ned_box":
            return pose

        local_box = self.start_randomization_config.get("local_ned_box", {})
        required_keys = (
            "north_min",
            "north_max",
            "east_min",
            "east_max",
            "down_min",
            "down_max",
        )
        if not all(key in local_box for key in required_keys):
            return pose

        translation = pose.get("translation", pose.get("position", {}))
        if not isinstance(translation, dict):
            translation = {}
            pose["translation"] = translation

        translation["x"] = float(translation.get("x", 0.0)) + self._sample_uniform(
            float(local_box["north_min"]), float(local_box["north_max"])
        )
        translation["y"] = float(translation.get("y", 0.0)) + self._sample_uniform(
            float(local_box["east_min"]), float(local_box["east_max"])
        )
        translation["z"] = float(translation.get("z", 0.0)) + self._sample_uniform(
            float(local_box["down_min"]), float(local_box["down_max"])
        )

        if bool(self.start_randomization_config.get("randomize_yaw", False)):
            rotation = pose.get("rotation", pose.get("orientation", {}))
            if not isinstance(rotation, dict):
                rotation = {}
                pose["rotation"] = rotation

            roll, pitch, _ = quaternion_to_rpy(
                float(rotation.get("w", 1.0)),
                float(rotation.get("x", 0.0)),
                float(rotation.get("y", 0.0)),
                float(rotation.get("z", 0.0)),
            )
            yaw = self._sample_uniform(
                float(self.start_randomization_config.get("yaw_min_rad", -np.pi)),
                float(self.start_randomization_config.get("yaw_max_rad", np.pi)),
            )
            rot_w, rot_x, rot_y, rot_z = rpy_to_quaternion(roll, pitch, yaw)
            pose["rotation"] = {"w": rot_w, "x": rot_x, "y": rot_y, "z": rot_z}

        return pose

    def _sample_episode_setup(self):
        goal_coords = copy.deepcopy(self.fixed_goal_gps_coords)
        goal_local_ned = copy.deepcopy(self.goal_local_ned)
        start_pose = copy.deepcopy(self.initial_actor_pose)

        max_attempts = max(1, int(self.randomization_config.get("max_episode_sampling_attempts", 25)))
        min_goal_start_distance = float(
            self.goal_randomization_config.get("min_start_to_goal_distance_m", 0.0)
        )
        max_goal_start_distance = float(
            self.goal_randomization_config.get("max_start_to_goal_distance_m", np.inf)
        )

        for _ in range(max_attempts):
            candidate_goal_coords = self._sample_goal_coords()
            candidate_goal_local = self._goal_local_from_coords(candidate_goal_coords)
            candidate_start_pose = self._sample_start_pose()

            if candidate_goal_local is None:
                return candidate_goal_coords, goal_local_ned, candidate_start_pose

            start_xyz = self._pose_translation(candidate_start_pose)
            if start_xyz is None:
                return candidate_goal_coords, candidate_goal_local, candidate_start_pose

            start_to_goal_distance = float(
                np.linalg.norm(
                    np.asarray(start_xyz, dtype=np.float64) - np.asarray(candidate_goal_local, dtype=np.float64),
                    ord=2,
                )
            )
            if min_goal_start_distance <= start_to_goal_distance <= max_goal_start_distance:
                return candidate_goal_coords, candidate_goal_local, candidate_start_pose

            goal_coords = candidate_goal_coords
            goal_local_ned = candidate_goal_local
            start_pose = candidate_start_pose

        return goal_coords, goal_local_ned, start_pose

    async def _takeoff_and_wait_async(self):
        print("[RESET] takeoff_async: issuing command")
        try:
            takeoff_task = await self.actor.takeoff_async(
                timeout_sec=self.takeoff_ack_timeout_sec
            )
            deadline = time.monotonic() + self.reset_takeoff_wait_sec
            sim_step_ns = max(1, int(self.dt * 1e9))

            while time.monotonic() < deadline:
                if not self._is_currently_landed():
                    print("[RESET] takeoff_async: airborne state reached")
                    return

                self.world.continue_for_sim_time(sim_step_ns, wait_until_complete=True)
                await asyncio.sleep(0)

                if takeoff_task.done():
                    break

            print(
                "[RESET] takeoff_async: still landed after quick wait "
                f"({self.reset_takeoff_wait_sec:.3f}s); continuing reset"
            )
        except Exception as exc:
            print(f"[RESET] takeoff_async failed: {exc}")

    async def _land_and_wait_async(self):
        try:
            if self._is_currently_landed():
                return
            land_task = await self.actor.land_async(timeout_sec=self.takeoff_ack_timeout_sec)
            deadline = time.monotonic() + self.reset_land_wait_sec
            sim_step_ns = max(1, int(self.dt * 1e9))
            while time.monotonic() < deadline:
                if self._is_currently_landed():
                    return
                self.world.continue_for_sim_time(sim_step_ns, wait_until_complete=True)
                await asyncio.sleep(0)
                if land_task.done():
                    break
            print(
                "[RESET] land_async: still not landed after quick wait "
                f"({self.reset_land_wait_sec:.3f}s); continuing reset"
            )
        except Exception as exc:
            print(f"[RESET] land_async failed: {exc}")
    
    def _is_currently_landed(self):
        try:
            landed_state = int(self.actor.get_landed_state())
            return landed_state == 0
        except Exception:
            return False

