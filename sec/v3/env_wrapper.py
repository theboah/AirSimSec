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
from attacks import load_attacks_from_config
from secState import Pose, GPSPosition, State

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
        self.scene_dir = os.path.abspath(config.get("scene_config_path"))
        self.scene_name = config.get("scene_filename")
        scene_config, _ = load_scene_config_as_dict(self.scene_name, self.scene_dir)
        
        self.render_mode = render_mode
        self.dt = config.get("dt", 0.1)
        self.subscribed_topics = []
        
        #Randomisation
        self.randomization_config = (config.get("randomization", {}))
        self.randomization_seed = self.randomization_config.get("seed")
        
            #Start
        self.start_randomization_config = (self.randomization_config.get("start", {}))
        self.episode_start_pose = None
        
            #Goal
        self.goal_randomization_config = (self.randomization_config.get("goal", {}))
        self.goal_tolerance = config.get("goal_tolerance", 0.5)
        self.home_geo_point = self._get_home_geo_point(scene_config)
        self.goal_local_ned = [0.0, 0.0, 0.0]
        self.prev_goal_distance = None
        self.episode_goal_gps_coords = [0.0, 0.0, 0.0]
        
        # AirSim plugin setup
        self.drone_name = kwargs.get("drone_name") or config.get("drone_name", "Drone1")

        external_client = kwargs.get("client")
        self.client = external_client if external_client is not None else ProjectAirSimClient()
        self._owns_client = external_client is None
        if self._owns_client:
            self.client.connect()

        external_world = kwargs.get("world")
        self.world = external_world if external_world is not None else World(
            self.client,
            scene_config_name=self.scene_name,
            delay_after_load_sec=float(config.get("world_delay_after_load_sec", 4.0)),
            sim_config_path=config.get("scene_config_path"),
        )
        
        
        self.actor = Drone(self.client, self.world, self.drone_name)
        self.initial_actor_pose = self.actor.get_ground_truth_pose()
        self.initially_landed = self._is_currently_landed()
        self.yaw_rate_max_rad_s = float(config.get("max_yaw_rate_rad_s", np.pi / 2.0))

        #Training
        self.limit_x, self.limit_y, self.limit_z = config.get("limit_xyz", [50.0, 50.0, 50.0])
        self.done = False
        self.episode_index = 0
        self.state = None
        self.collision_info = None
        self.step_num = 0
        self.step_limit = config.get("step_limit", 2000)
        self.reset_num = 0
        self.annotation_msg = {}
        self.img_msg = None
        self.gps_msg = None
        self.action_shape = (4,)
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=self.action_shape,
            dtype=np.float32,
        )
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
        self.accept_collision_events = False
        self.collision_accept_after = 0.0

        self._event_loop = self._setup_event_loop()
        
        #Attacks
        self.min_reset_before_attacks = int(config.get("min_reset_before_attacks", 5000))
        self.attacks = load_attacks_from_config(config.get("attacks", []))
        #self.attacks = []
        
        #Reward and penalties
        self.progress_reward_scale = float(config.get("progress_reward_scale", 1.0))
        self.step_penalty = float(config.get("step_penalty", 0.01))
        self.reward_reference_dt = float(config.get("reward_reference_dt", 0.2))
        if self.reward_reference_dt <= 0.0:
            raise ValueError("reward_reference_dt must be > 0")
        self.reward_dt_scale = self.dt / self.reward_reference_dt
        self.goal_reached_bonus = float(config.get("goal_reached_bonus", 10.0))
        self.collision_penalty = float(config.get("collision_penalty", 15.0))
        self.out_of_arena_penalty = float(config.get("out_of_arena_penalty", 10.0))
        self.timeout_penalty = float(config.get("timeout_penalty", 2.0))
        self.missing_goal_distance_m = float(config.get("missing_goal_distance_m", 1e6))
        
        

    def reset(self, *, seed=None, options=None):
        self.reset_num += 1
        effective_seed = seed
        if effective_seed is None and self.randomization_seed is not None:
            effective_seed = int(self.randomization_seed) + int(self.episode_index)
        super().reset(seed=effective_seed)
        self.done = False
        self.step_num = 0
        self.state = None
        self.collision_info = None
        self.img_msg = None
        self.gps_msg = None
        self.annotation_msg = {}
        self.accept_collision_events = False
        self.collision_accept_after = time.time() + 1

        self._cancel_active_task()

        self.episode_goal_gps_coords, self.goal_local_ned, self.episode_start_pose = (
            self._sample_episode_setup()
        )
        self.goal_gps_coords = self.episode_goal_gps_coords

        self.actor.set_pose(self.episode_start_pose, reset_kinematics=True)

        self._ensure_subscriptions()

        self.actor.enable_api_control()
        self.actor.arm()

        if self.initially_landed:
            self._run_coroutine_sync(self._takeoff_and_wait_async())


        # Wait until an initial image observation is received
        image_wait_loops = 0
        while self.img_msg is None:
            image_wait_loops += 1
            time.sleep(self.dt)

        self.accept_collision_events = True
        state = self.get_state()
        self.prev_goal_distance = self._goal_distance(state)
        pose = [0.0, 0.0, 0.0, 0.0] if state.pose is None else state.pose.to_list()
        goal_vector = self._goal_vector_as_list(state.goal_vector)
        obs = {
            "rgb_image": state.img,
            "pose": np.asarray(pose, dtype=np.float32),
            "goal_vector": np.asarray(goal_vector, dtype=np.float32),
        }
        self.episode_index += 1
        return obs, {}

    def step(self, action):
        """Execute one step of the environment."""
        loop = self._get_event_loop()
        return loop.run_until_complete(self.step_async(action))

    def close(self):
        """Close the environment and clean up resources."""
        try:
            if self._owns_client:
                self.client.disconnect()
        finally:
            if self._event_loop is not None and not self._event_loop.is_closed():
                self._event_loop.close()
                self._event_loop = None

    async def _take_action(self,action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        # Action:{Vx: float, Vy: float, Vz: float, Vyaw: float}
        elif isinstance(action, dict):
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
        
        
        
        await self.actor.move_by_velocity_async(
            v_north,
            v_east,
            v_down,
            duration=(self.dt),
            yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
            yaw_is_rate=True,
            yaw=yaw_rate,
        )

    async def step_async(self, action):
        timeout = False
        goal_reached = False
        out_of_arena = False
        
        await self._take_action(action)

         
        ground_truth_info = self.get_state()
        observed_info = copy.deepcopy(ground_truth_info)
        
        #OBSERVED INFO
        #attack observed info here
        if self.reset_num >= self.min_reset_before_attacks:
            for attack in self.attacks:
                observed_info = attack.attack_sim(self.step_num, observed_info)  
        
        #TRUE INFO
        true_current_goal_distance = self._goal_distance(ground_truth_info)
        previous_goal_distance = (
            true_current_goal_distance
            if self.prev_goal_distance is None or not np.isfinite(self.prev_goal_distance)
            else self.prev_goal_distance
        )
        
        collision_detected = bool(self.done and self._is_collision_event(self.collision_info))
        if collision_detected:
            print("COLLISION")
        elif true_current_goal_distance <= self.goal_tolerance:
            self.done = True
            goal_reached = True
            print("GOAL REACHED")
        elif self.is_outside_arena(ground_truth_info.pose):
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
        self.prev_goal_distance = (
            true_current_goal_distance
            if np.isfinite(true_current_goal_distance)
            else previous_goal_distance
        )
        info = {
            "status": "Running",
            "gps_position": observed_info.gps_position,
            "gps_displacement_to_goal": observed_info.goal_vector,
            "goal_coords": self.goal_gps_coords,
        }
        pose = [0.0, 0.0, 0.0, 0.0] if observed_info.pose is None else observed_info.pose.to_list()
        goal_vector = self._goal_vector_as_list(observed_info.goal_vector)
        next_obs = {
            "rgb_image": observed_info.img,
            "pose": np.asarray(pose, dtype=np.float32),
            "goal_vector": np.asarray(goal_vector, dtype=np.float32),
        }
        return (next_obs, reward, terminated, timeout, info)

    def render(self, mode="human", close=False):
        if mode is None:
            mode = self.render_mode

        if self.img_msg is not None and mode == "human":
            self.display_debug_info(self.state, self.annotation_msg)

    def get_state(self):
        (actor_x, actor_y, actor_z, _, _, actor_yaw) = self.get_vec_from_pose(self.actor.get_ground_truth_pose())
        drone_pose = Pose(actor_x, actor_y, actor_z, actor_yaw)
        
        gps_position = self.get_gps_position(self.gps_msg)

        # Calculate goal_vector in local NED coordinates (meters)
        if self.goal_local_ned is None:
            goal_vector = None
        else:
            goal_vector = [actor_x - self.goal_local_ned[0], actor_y - self.goal_local_ned[1],actor_z - self.goal_local_ned[2]]

        self.state = State(drone_pose, self._get_processed_rgb_image(self.img_msg), gps_position, goal_vector)
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
        delta_distance = previous_goal_distance - current_goal_distance
        if not np.isfinite(delta_distance):
            delta_distance = 0.0

        reward = (
            self.progress_reward_scale
            * delta_distance
            - (self.step_penalty * self.reward_dt_scale)
        )

        if goal_reached:
            reward += self.goal_reached_bonus
        if collision_detected:
            reward -= self.collision_penalty
        if out_of_arena:
            reward -= self.out_of_arena_penalty
        if timeout:
            reward -= self.timeout_penalty

        if not np.isfinite(reward):
            return -(self.step_penalty * self.reward_dt_scale)

        return reward

    #Returns goal_distance in meters
    def _goal_distance(self, state):
        goal_vector_array = np.asarray(state.goal_vector, dtype=np.float64)
        distance = float(np.linalg.norm(goal_vector_array, ord=2))
        if not np.isfinite(distance):
            return self.missing_goal_distance_m
        return distance

    def _goal_vector_as_list(self, goal_vector):
        if goal_vector is None:
            return [0.0, 0.0, 0.0]
        if hasattr(goal_vector, "to_list"):
            return goal_vector.to_list()
        if isinstance(goal_vector, np.ndarray):
            return goal_vector.tolist()
        if isinstance(goal_vector, (list, tuple)):
            return list(goal_vector)
        return [float(goal_vector[0]), float(goal_vector[1]), float(goal_vector[2])]

    #Callbacks
    def _ensure_subscriptions(self):
        if self.subscribed_topics:
            return

        camera_topic = self.actor.sensors["ForwardViewCamera"]["scene_camera"]
        self.client.subscribe(
            camera_topic,
            self.callback_camera,
        )
        self.subscribed_topics.append(camera_topic)

        gps_topic = self.actor.sensors["GPS"]["gps"]
        self.client.subscribe(gps_topic, self.callback_gps)
        self.subscribed_topics.append(gps_topic)

        collision_topic = self.actor.robot_info["collision_info"]
        self.client.subscribe(
            collision_topic,
            lambda _, collision_info: self.callback_collision(collision_info),
        )
        self.subscribed_topics.append(collision_topic)

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
        
        pos = GPSPosition(latitude, longitude, altitude)
        
        return pos
        
    def callback_collision(self, collision_info):
        self.collision_info = collision_info
        if not self.accept_collision_events:
            return
        if time.time() < self.collision_accept_after:
            return
        if not self._is_collision_event(collision_info):
            return
        self.done = True

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

    def is_outside_arena(self, pose):
        if pose is None:
            return False
        x, y, z, _ = pose.to_list()
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
        if state is not None and state.img is not None:
            img_np = state.img
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
            position_error_m <= 0.05
            and quat_error <= 1e-3 and frame_id_match
        )
        diff = {
            "position_error_m": position_error_m,
            "quat_error": quat_error,
            "frame_id_match": frame_id_match,
            "position_tol_m": 0.05,
            "quat_tol": 1e-3,
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

    def _setup_event_loop(self):
        """Create and return a new event loop for this environment."""
        return asyncio.new_event_loop()

    def _get_event_loop(self):
        """Get the cached event loop, creating one if necessary."""
        if self._event_loop is None or self._event_loop.is_closed():
            self._event_loop = self._setup_event_loop()
        return self._event_loop

    def _run_coroutine_sync(self, coroutine):
        """Run an async coroutine synchronously using the cached event loop."""
        loop = self._get_event_loop()
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
        gps_box = self.goal_randomization_config.get("gps_box", {})
        required_keys = (
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
            "alt_min",
            "alt_max",
        )

        return [
            self._sample_uniform(float(gps_box["lat_min"]), float(gps_box["lat_max"])),
            self._sample_uniform(float(gps_box["lon_min"]), float(gps_box["lon_max"])),
            self._sample_uniform(float(gps_box["alt_min"]), float(gps_box["alt_max"])),
        ]

    def _sample_start_pose(self):
        pose = copy.deepcopy(self.initial_actor_pose)

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
        takeoff_timeout_sec = float(self.randomization_config.get("takeoff_timeout_sec", 4.0))
        await self.actor.takeoff_async(timeout_sec=takeoff_timeout_sec)
        
    
    def _is_currently_landed(self):
        try:
            landed_state = int(self.actor.get_landed_state())
            return landed_state == 0
        except Exception:
            return False

