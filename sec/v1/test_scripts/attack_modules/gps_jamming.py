from projectairsim.validate import FaultInjectionModule
from projectairsim import Drone, World

def gps_disable_fault(drone: Drone, world: World, time):
    fault_injection_module = FaultInjectionModule(drone, world)

    def gps_jamming_fault():
        drone.ASS_disable_gps_sensor_config(sensor_name="GPS")

    fault_injection_module.add_fault_injection_at_simtime(gps_jamming_fault, time)

    fault_injection_module.start()
     
def gps_jamming_fault(drone: Drone, world: World,time, level):
    fault_injection_module = FaultInjectionModule(drone, world)

    def gps_jamming_fault():
        noise_level = level if level > 0 else 100.0
        drone.apply_noise_gps(sensor_name="GPS", noise_level=noise_level)
    
    fault_injection_module.add_fault_injection_at_simtime(gps_jamming_fault, time)
    fault_injection_module.start()