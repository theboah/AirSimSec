import random

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
            start_step_min = attack_cfg["start_step_min"]
            start_step_max = attack_cfg["start_step_max"]
        except KeyError as exc:
            missing_key = exc.args[0]
            raise ValueError(
                f"Attack '{attack_type}' at index {idx} is missing required key '{missing_key}'"
            ) from exc

        if attack_type == "gps_disable":
            attack = GPSDisableAttack(
                probability,
                duration_min,
                duration_max,
                start_step_min,
                start_step_max,
            )
        elif attack_type == "gps_jam_noise":
            if "noise_range" not in attack_cfg:
                raise ValueError(
                    f"Attack '{attack_type}' at index {idx} is missing required key 'noise_range'"
                )
            attack = GPSJamNoiseAttack(
                probability,
                duration_min,
                duration_max,
                start_step_min,
                start_step_max,
                attack_cfg["noise_range"],
            )
        else:
            raise ValueError(
                f"Unknown attack type '{attack_type}' at index {idx}. "
                "Supported types: gps_disable, gps_jam_noise, gps_spoof_towards"
            )

        attacks.append(attack)

    return attacks


class AirSimAttack:
    
    #duration min and max are steps
    def __init__(self, probability, duration_min, duration_max, start_step_min, start_step_max):
        
        if not (0.0 <= probability <= 1.0):
            raise ValueError("Probability must be between 0 and 1")
        if duration_min < 0 or duration_max < 0:
            raise ValueError("Duration min and max must be non-negative")
        if duration_min > duration_max:
            raise ValueError("Duration min cannot be greater than duration max")
        if start_step_min < 0 or start_step_max < 0:
            raise ValueError("Start step min and max must be non-negative")
        if start_step_min > start_step_max:
            raise ValueError("Start step min cannot be greater than start step max")
        
        
        self.probability = probability
        self.duration_min = duration_min
        self.duration_max = duration_max
        self.start_step_min = start_step_min
        self.start_step_max = start_step_max
        self.active = False
        self.duration = 0
        
    def roll(self):
        if random.random() < self.probability:
            return True
        return False
    
    def attack_sim(self, step, state):
        if not self.active:
            if step < self.start_step_min or step > self.start_step_max:
                return state
            if not self.roll():
                return state
            self.active = True
            self.duration = random.randint(self.duration_min, self.duration_max)
        
        self.duration -= 1
        if self.duration <= 0:
            self.active = False  
        return self.attack(state)
        
    
    #Overide this method to implement attack logic
    def attack(self,state):
        return state
        
    
class GPSDisableAttack(AirSimAttack):
    def __init__(self, probability,duration_min, duration_max, start_step_min, start_step_max):
        super().__init__(probability, duration_min, duration_max, start_step_min, start_step_max)
        
    def attack(self,state):
        state.pose = None
        state.gps_position = None
        state.goal_vector = None
        return state
    
class GPSJamNoiseAttack(AirSimAttack):
    def __init__(self, probability,duration_min, duration_max, start_step_min, start_step_max, noise_range):
        super().__init__(probability, duration_min, duration_max, start_step_min, start_step_max)
        self.noise_range = noise_range
        
    def attack(self,state):
        noise = random.uniform(-self.noise_range, self.noise_range)
        state.pose = None

        if state.gps_position is not None:
            state.gps_position.lat += noise
            state.gps_position.long += noise
            state.gps_position.alt += noise

        if state.goal_vector is not None:
            state.goal_vector = [
                float(state.goal_vector[0]) + noise,
                float(state.goal_vector[1]) + noise,
                float(state.goal_vector[2]) + noise,
            ]
        return state
    
#TODO: GPS Spoofing attacks
        