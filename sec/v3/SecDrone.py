from projectairsim import Drone
from projectairsim.utils import unpack_image


class SecDrone(Drone):
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
        
    def get_obs(self):
        obs = {}
        obs["image"] = unpack_image(self.img_msg)
        obs["gps"] = self.gps_msg
        obs["pose"] = self.pose
        obs["collided"] = self.collided
        return obs
    

    
