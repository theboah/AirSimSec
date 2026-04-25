import asyncio
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log
from projectairsim.image_utils import ImageDisplay

# Async main function to wrap async drone commands
async def main():
    # Create a Project AirSim client
    client = ProjectAirSimClient()
    fault_injection_module = None

    try:
        # Connect to simulation environment
        client.connect()

        # Create a World object to interact with the sim world and load a scene
        world = World(client, "scene_px4_sitl.jsonc", delay_after_load_sec=10)

        # Create a Drone object to interact with a drone in the loaded sim world
        drone = Drone(client, world, "Drone1")
        
        # ------------------------------------------------------------------------------

        # Set the drone to be ready to fly
        projectairsim_log().info("Invoking enable_api_control")
        drone.enable_api_control()
        projectairsim_log().info("Invoking Arm")
        drone.arm()

        # ------------------------------------------------------------------------------

        takeoff_task = await drone.takeoff_async()
        projectairsim_log().info("takeoff_async invoked")
        await takeoff_task
        projectairsim_log().info("takeoff_async completed")

        # Command the drone to move up and east in NED coordinate system for 5 seconds
        move_up_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=1.0, v_down=-1.0, duration=5.0
        )
        projectairsim_log().info("Move-Up invoked")
        await move_up_task
        projectairsim_log().info("Move-Up completed")

        # Command the Drone to move down and west in NED coordinate system for 5 seconds
        move_down_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=-1.0, v_down=1.0, duration=5.0
        )
        projectairsim_log().info("Move-Down invoked")
        await move_down_task
        projectairsim_log().info("Move-Down completed")

        land_task = await drone.land_async()
        projectairsim_log().info("land_async invoked")
        await land_task
        projectairsim_log().info("land_async completed")

        # ------------------------------------------------------------------------------

        # Shut down the drone
        projectairsim_log().info("Invoking Disarm")
        drone.disarm()
        projectairsim_log().info("Invoking disable_api_control")
        drone.disable_api_control()

        # ------------------------------------------------------------------------------

    except Exception as err:
        projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    finally:
        if fault_injection_module:
            fault_injection_module.stop()
        # Always disconnect from the simulation environment to allow next connection
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
