import math
import random

from secState import GPSPosition


def load_attacks_from_config(attacks_config) -> list["AirSimAttack"]:
    """Create configured attack objects from config.

    Accepts either:
    - a full config dict containing an "attacks" list, or
    - the attacks list directly.
    """
    if attacks_config is None:
        return []

    if isinstance(attacks_config, dict):
        attacks_list = attacks_config.get("attacks", [])
    elif isinstance(attacks_config, list):
        attacks_list = attacks_config
    else:
        raise ValueError("attacks_config must be a dict or list")

    attacks = []
    for idx, attack_cfg in enumerate(attacks_list):
        if not isinstance(attack_cfg, dict):
            raise ValueError(f"Attack entry at index {idx} must be an object")

        attack_type = attack_cfg.get("type")
        if not attack_type:
            raise ValueError(f"Attack entry at index {idx} is missing 'type'")

        try:
            probability = attack_cfg["probability"]
            duration_min = attack_cfg["duration_min"]
            duration_max = attack_cfg["duration_max"]
            start_ep_min = attack_cfg["start_ep_min"]
        except KeyError as exc:
            missing_key = exc.args[0]
            raise ValueError(
                f"Attack '{attack_type}' at index {idx} is missing required key '{missing_key}'"
            ) from exc

        start_ep_max = attack_cfg.get("start_ep_max", start_ep_min)

        if attack_type == "gps_disable":
            attack = GPSDisableAttack(
                probability,
                duration_min,
                duration_max,
                start_ep_min,
                start_ep_max,
            )
        elif attack_type == "gps_jam_noise":
            try:
                attack_pos_cfg = attack_cfg["attack_pos"]
                attack_range_m = attack_cfg["attack_range_m"]
                eph_min_3d = attack_cfg["eph_min_3d"]
                eph_min_2d = attack_cfg["eph_min_2d"]
                default = attack_cfg["default"]
            except KeyError as exc:
                missing_key = exc.args[0]
                raise ValueError(
                    f"Attack '{attack_type}' at index {idx} is missing required key '{missing_key}'"
                ) from exc

            attack = GPSJamNoiseAttack(
                probability,
                duration_min,
                duration_max,
                start_ep_min,
                start_ep_max,
                attack_range_m,
                GPSPosition(
                    latitude=attack_pos_cfg["lat"],
                    longitude=attack_pos_cfg["long"],
                    altitude=attack_pos_cfg["alt"],
                ),
                eph_min_3d,
                eph_min_2d,
                default,
            )
        else:
            raise ValueError(
                f"Unknown attack type '{attack_type}' at index {idx}. "
                "Supported types: gps_disable, gps_jam_noise"
            )

        attacks.append(attack)

    return attacks


class AirSimAttack:
    # duration min and max are steps
    def __init__(self, probability, duration_min, duration_max, start_ep_min, start_ep_max=None):
        if start_ep_max is None:
            start_ep_max = start_ep_min

        if not (0.0 <= probability <= 1.0):
            raise ValueError("Probability must be between 0 and 1")
        if duration_min < 0 or duration_max < 0:
            raise ValueError("Duration min and max must be non-negative")
        if duration_min > duration_max:
            raise ValueError("Duration min cannot be greater than duration max")
        if start_ep_min < 0 or start_ep_max < 0:
            raise ValueError("Start episode min and max must be non-negative")
        if start_ep_min > start_ep_max:
            raise ValueError("Start episode min cannot be greater than start episode max")

        self.probability = probability
        self.duration_min = duration_min
        self.duration_max = duration_max
        self.start_ep_min = start_ep_min
        self.start_ep_max = start_ep_max
        self.active = False
        self.duration = 0

    def roll(self):
        return random.random() < self.probability

    def attack_sim(self, ep_num, state):
        if state is None:
            return state

        if not self.active:
            if ep_num < self.start_ep_min:
                return state
            if not self.roll():
                return state
            self.active = True
            self.duration = random.randint(self.duration_min, self.duration_max)

        self.duration -= 1
        if self.duration <= 0:
            self.active = False

        return self.attack(state)

    # Override this method to implement attack logic.
    def attack(self, state):
        return state


class GPSDisableAttack(AirSimAttack):
    def __init__(self, probability, duration_min, duration_max, start_ep_min, start_ep_max=None):
        super().__init__(probability, duration_min, duration_max, start_ep_min, start_ep_max)

    def attack(self, state):
        state.pose = None
        state.gps_msg = None
        state.goal_vector = None
        return state


class GPSJamNoiseAttack(AirSimAttack):
    def __init__(
        self,
        probability,
        duration_min,
        duration_max,
        start_ep_min,
        start_ep_max,
        attack_range_m,
        attack_pos,
        eph_min_3d,
        eph_min_2d,
        default,
    ):
        super().__init__(probability, duration_min, duration_max, start_ep_min, start_ep_max)
        self.attack_range_m = attack_range_m
        self.attack_pos = attack_pos
        self.eph_min_3d = eph_min_3d
        self.eph_min_2d = eph_min_2d
        self.default = default

    def dilution_level(self, distance, default):
        if distance >= self.attack_range_m:
            return default

        distance_factor = 1.0 - (distance / self.attack_range_m)
        return default + (self.attack_range_m * distance_factor)

    def distance_to_attack(self, gps_msg):
        if gps_msg is None:
            return math.inf

        latitude = gps_msg.get("latitude")
        longitude = gps_msg.get("longitude")
        altitude = gps_msg.get("altitude")
        if latitude is None or longitude is None or altitude is None:
            return math.inf

        latitude_scale = 111320.0
        longitude_scale = 111320.0 * math.cos(math.radians(latitude))

        dx = (latitude - self.attack_pos.lat) * latitude_scale
        dy = (longitude - self.attack_pos.long) * longitude_scale
        dz = altitude - self.attack_pos.alt
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def attack(self, state):
        if state.gps_msg is None:
            return state

        distance_to_attack = self.distance_to_attack(state.gps_msg)
        if distance_to_attack > self.attack_range_m:
            return state

        epv = eph = self.dilution_level(distance_to_attack, self.default)

        if self.eph_min_3d > self.eph_min_2d:
            raise ValueError("eph_min_3d must be <= eph_min_2d")

        fix = 3
        if eph <= self.eph_min_3d:
            fix = 0
        elif eph <= self.eph_min_2d:
            fix = 2

        gps_msg = state.gps_msg
        velocity = gps_msg.setdefault("velocity", {"x": 0.0, "y": 0.0, "z": 0.0})

        noise_degrees_horizontal = random.gauss(0.0, eph / 111320.0)
        noise_degrees_vertical = random.gauss(0.0, epv)

        gps_msg["latitude"] += noise_degrees_horizontal
        gps_msg["longitude"] += noise_degrees_horizontal
        gps_msg["altitude"] += noise_degrees_vertical
        gps_msg["epv"] = epv
        gps_msg["eph"] = eph
        gps_msg["fix_type"] = fix
        velocity["x"] += random.gauss(0.0, eph / 111320.0)
        velocity["y"] += random.gauss(0.0, eph / 111320.0)
        velocity["z"] += random.gauss(0.0, epv)

        return state


# TODO: GPS spoofing attacks
        