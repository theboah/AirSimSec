class Pose:
    def __init__(self, x, y, z, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw

    def to_list(self):
        return [self.x, self.y, self.z, self.yaw]

class GPSPosition:
    def __init__(self, latitude, longitude, altitude):
        self.lat = latitude
        self.long = longitude
        self.alt = altitude

    def to_list(self):
        return [self.lat, self.long, self.alt]

class State:
    def __init__(self, drone_pose, img, gps_position, goal_vector):
        self.pose = drone_pose
        self.img = img
        self.gps_position = gps_position
        self.goal_vector = goal_vector
    