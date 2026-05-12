import random
import json
import copy

import numpy as np

from projectairsim.utils import rpy_to_quaternion
from projectairsim.types import Pose,Vector3,Quaternion

def get_random_start_pose(config):
    start_config = config.get("start_settings")
    pose_x = random.uniform(float(start_config ["n_min"]), float(start_config ["n_max"]))
    pose_y = random.uniform(float(start_config ["e_min"]), float(start_config ["e_max"]))
    pose_z = float(start_config ["d"])
    if start_config.get("randomize_yaw"):
        yaw = random.uniform(float(start_config.get("yaw_min_rad", -np.pi)),float(start_config.get("yaw_max_rad", np.pi)))
    else:
        yaw = 0.0
    roll = 0.0
    pitch = 0.0
    rot_w, rot_x, rot_y, rot_z = rpy_to_quaternion(roll, pitch, yaw)
    return Pose({"translation":Vector3({"x":pose_x, "y":pose_y, "z":pose_z}), "rotation":Quaternion({"w": rot_w, "x": rot_x, "y": rot_y, "z": rot_z})})

def get_random_goal(config):
    sel = random.randrange(0, len(config.get("goals")))
    return config.get("goals")[sel]
    
def generate_tmp_scene_config(num_envs, scene_config_name):
    with open(f"sim_config/{scene_config_name}", "r", encoding="utf-8") as f:
        scene_config = json.load(f)
    template = scene_config["actors"][0]
    new_actors = []
    for i in range(num_envs):
        new_actor = copy.deepcopy(template)
        new_actor["name"] = f"Drone{i}"
        x,y,z =_parse_xyz_string(new_actor["origin"]["xyz"])
        new_actor["origin"]["xyz"] = _format_xyz_string(x, y-2*i, z)
        new_actors.append(new_actor)
    scene_config["actors"] = new_actors
    with open(f"sim_config/tmp_{scene_config_name}", "w", encoding="utf-8") as f:
        json.dump(scene_config, f, indent=4)
    return f"tmp_{scene_config_name}"

def _parse_xyz_string(xyz: str):
    values = [float(part) for part in str(xyz).strip().split()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values for xyz, got: {xyz}")
    return values


def _format_xyz_string(x: float, y: float, z: float):
    return f"{x:.6f} {y:.6f} {z:.6f}"

def is_outside_arena(config, pose):
    limits = config.get("limits") or {}
    north = limits.get("north")
    east = limits.get("east")
    down = limits.get("down")
    if not (north and east and down) or len(north) != 2 or len(east) != 2 or len(down) != 2:
        return False

    try:
        min_x = min(float(north[0]), float(north[1]))
        max_x = max(float(north[0]), float(north[1]))
        min_y = min(float(east[0]), float(east[1]))
        max_y = max(float(east[0]), float(east[1]))
        min_z = min(float(down[0]), float(down[1]))
        max_z = max(float(down[0]), float(down[1]))
    except (TypeError, ValueError):
        return False

    coords = _pose_to_xyz(pose)
    if coords is None:
        return False

    x, y, z = coords

    if x < min_x or x > max_x or y < min_y or y > max_y or z < min_z or z > max_z:
        return True

    return False

def is_goal_reached(config, pose, goal_coordinates):
    goal_n, goal_e, goal_d = _goal_coordinates_to_ned(goal_coordinates)
    coords = _pose_to_xyz(pose)
    if coords is None:
        return False
    x, y, z = coords
    dif_x = x - goal_n
    dif_y = y - goal_e
    dif_z = z - goal_d
    distance = np.sqrt(dif_x**2 + dif_y**2 + dif_z**2)
    return distance < config.get("goal_tolerance")

def is_outside_step_limit(config, step):
    step_limit = config.get("step_limit")
    return step >= step_limit

def distance_to_goal(pose, goal_coordinates):
    goal_n, goal_e, goal_d = _goal_coordinates_to_ned(goal_coordinates)
    coords = _pose_to_xyz(pose)
    if coords is None:
        return float("inf")
    x, y, z = coords
    dif_x = x - goal_n
    dif_y = y - goal_e
    dif_z = z - goal_d
    return np.sqrt(dif_x**2 + dif_y**2 + dif_z**2)


def _goal_coordinates_to_ned(goal_coordinates):
    if all(key in goal_coordinates for key in ("n", "e", "d")):
        return goal_coordinates["n"], goal_coordinates["e"], goal_coordinates["d"]
    if all(key in goal_coordinates for key in ("x", "y", "z")):
        return goal_coordinates["x"], goal_coordinates["y"], goal_coordinates["z"]
    raise KeyError("goal_coordinates must contain either n/e/d or x/y/z keys")


def _pose_to_xyz(pose):
    if pose is None:
        return None

    # canonical nested keys
    pos = None
    if isinstance(pose, dict):
        if "position" in pose and isinstance(pose["position"], dict):
            pos = pose["position"]
        elif "translation" in pose and isinstance(pose["translation"], dict):
            pos = pose["translation"]
        elif all(k in pose for k in ("x", "y", "z")):
            try:
                return float(pose["x"]), float(pose["y"]), float(pose["z"])
            except Exception:
                return None
    else:
        return None

    if pos is None:
        return None

    try:
        x = float(pos.get("x", pos.get("X")))
        y = float(pos.get("y", pos.get("Y")))
        z = float(pos.get("z", pos.get("Z")))
        return x, y, z
    except Exception:
        return None
