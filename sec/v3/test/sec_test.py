import asyncio
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log
from projectairsim.image_utils import ImageDisplay

class testDrone(Drone):
    def __init__(self, client, world, name):
        super().__init__(client, world, name)
        self.img_msg = None
        self.gps_msg = None
        self.collided = False
        self.pose = None
        self.episode_start_pose = None
        self.goal_coordinates = None
        self._setup_subscriptions()
    
    def _img_callback(self, img_msg):
        self.img_msg = img_msg
    
    def _gps_callback(self, gps_msg):
        self.gps_msg = gps_msg
    
    def _collision_callback(self, collision_msg):
        self.collided = True
    
    def _pose_callback(self, pose_msg):
        self.pose = pose_msg    
        
    def _setup_subscriptions(self):
        self.client.subscribe(
            self.sensors["ForwardViewCamera"]["scene_camera"],
            lambda _, img_msg: self._img_callback(img_msg),
        )
        self.client.subscribe(
            self.sensors["GPS"]["gps"],
            lambda _, gps_msg: self._gps_callback(gps_msg),
        )
        self.client.subscribe(
            self.robot_info["collision_info"],
            lambda _, collision_msg: self._collision_callback(collision_msg),
        )
        self.client.subscribe(
            self.robot_info["actual_pose"]
            , lambda _, pose_msg: self._pose_callback(pose_msg)
        )


# Async main function to wrap async drone commands
async def main():
    # Create a Project AirSim client
    client = ProjectAirSimClient()

    try:
        # Connect to simulation environment
        client.connect()

        # Create a World object to interact with the sim world and load a scene
        world = World(client, "scene_sec.jsonc", delay_after_load_sec=2)

        # Create a Drone object to interact with a drone in the loaded sim world
        drone1 = testDrone(client, world, "Drone1")
        drone2 = testDrone(client, world, "Drone2")
        drone3 = testDrone(client, world, "Drone3")
        drone4 = testDrone(client, world, "Drone4")
        drone5 = testDrone(client, world, "Drone5")
        drones = [drone1, drone2, drone3, drone4, drone5]
        
        # ------------------------------------------------------------------------------

        # Set the drone to be ready to fly
        projectairsim_log().info("Invoking enable_api_control")
        for drone in drones:
            drone.enable_api_control()
            
        projectairsim_log().info("Invoking Arm")
        for drone in drones:
            drone.arm()

        # ------------------------------------------------------------------------------
        for drone in drones:
            takeoff_task = await drone.takeoff_async()
            await takeoff_task
        
        await asyncio.gather(
                await drone1.move_by_velocity_async(v_north=0.0, v_east=1.0, v_down=-1.0, duration=1),
                await drone2.move_by_velocity_async(v_north=0.0, v_east=1.0, v_down=-1.0, duration=1),
                await drone3.move_by_velocity_async(v_north=0.0, v_east=1.0, v_down=-1.0, duration=1),
                await drone4.move_by_velocity_async(v_north=0.0, v_east=1.0, v_down=-1.0, duration=1),
                await drone5.move_by_velocity_async(v_north=0.0, v_east=1.0, v_down=-1.0, duration=1)
        )
        # ------------------------------------------------------------------------------

    except Exception as err:
        projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    finally:
        # Always disconnect from the simulation environment to allow next connection
        for drone in drones:
            drone.disarm()
            drone.disable_api_control()
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
