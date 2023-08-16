import numpy as np

class Transform:
    def __init__(self, lidar_range_lim, lidar_angle_lim):
        # robot pose
        self.x = 0
        self.y = 0
        self.theta = 0
        # lidar sensor condition
        self.range_lim = lidar_range_lim
        self.angle_lim = lidar_angle_lim
    
    def setPose(self, pose):
        self.x, self.y, self.theta = pose.values()

    def scanToPoint(self, lidar_scan):
        '''
        Transform laser scan (range and angle) to points
        '''
        lidar_points = []
        rmin, rmax = self.range_lim
        for i in range(lidar_scan.shape[0]):
            r = lidar_scan[i]
            if rmin < r < rmax:
                rad = np.deg2rad(-135 + (i / (lidar_scan.shape[0]-1)) * 270)
                lidar_points.append([r * np.cos(rad), r * np.sin(rad), 0])
        return np.array(lidar_points)

    def laserToWorld(self, laser_point, pose):
        '''
        Transform laser points from laser frame to world frame
        '''
        self.setPose(pose)
        T_laser = np.vstack([laser_point.T, np.ones((laser_point.shape[0], 1)).T])
        laser_point_w = self.chain('wTb', 'bTl') @ T_laser # (4, # of lasers)
        
        return laser_point_w[:3,:].T

    @property
    def worldTobody(self):
        R = self.yaw()
        P = np.array([[self.x, self.y, 0.127]]).T
        return np.vstack([np.hstack([R,P]), np.array([0,0,0,1])])
    
    @property
    def bodyToLidar(self):
        return np.array([[1, 0, 0, 0.150915],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0.51435],
                         [0, 0, 0, 1]])

    @property
    def bTl(self):
        return self.bodyToLidar

    @property
    def wTb(self):
        return self.worldTobody

    def chain(self, *trans):
        ret = np.eye(4)
        for t in trans:
            ret = ret @ getattr(self, t)
        return ret

    def yaw(self):
        rad = self.theta
        return np.array([[np.cos(rad), -np.sin(rad), 0],
                        [np.sin(rad), np.cos(rad), 0],
                        [0, 0, 1]])
