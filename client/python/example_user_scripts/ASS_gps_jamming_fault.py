from projectairsim.validate import FaultInjectionModule
from projectairsim import Drone, World
from projectairsim.types import ImageType

def inject_faults(drone: Drone, world: World):
    fault_injection_module = FaultInjectionModule(drone, world)

    def gps_jamming_fault():
        noise_level = 1000 if 1000 > 0 else 100.0
        drone.apply_noise_gps(sensor_name="GPS", noise_level=noise_level)
    
    fault_injection_module.add_fault_injection_at_simtime(gps_jamming_fault, 10*(10**9))
    fault_injection_module.start()
    return fault_injection_module