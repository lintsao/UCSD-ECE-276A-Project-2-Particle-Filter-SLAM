import numpy as np
import itertools

def lidarScanToXYPoints(lidar_scan, lidar_scan_rlim=(1, 30)):
    lidarScanXYPoints = []
    rmin, rmax = lidar_scan_rlim
    for i in range(1081):
        r = lidar_scan[i]

        if rmin < r < rmax:
            rad = np.deg2rad(-135 + (i / 1081) * 270)
            lidarScanXYPoints.append([r * np.cos(rad), r * np.sin(rad), 0])
    return np.array(lidarScanXYPoints)

def yaw(rad):
    return np.array([[np.cos(rad), -np.sin(rad), 0],
                     [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])


def pitch(rad):
    return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0],
                     [-np.sin(rad), 0, np.cos(rad)]])


def roll(rad):
    return np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)],
                     [0, np.sin(rad), np.cos(rad)]])

class Transform:

    def __init__(self,
                 x=None,
                 y=None,
                 theta=None,
                 *args,
                 **kwargs):
        self._x = x
        self._y = y
        self._theta = theta

    @property
    def bodyTlidar(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        P = np.array([[0.150915, 0, 0.51435]]).T # Obtained from the doc.
        return np.vstack([np.hstack([R, P]), np.array([0, 0, 0, 1])])

    @property
    def worldTbody(self):
        R = yaw(self._theta)
        P = np.array([[self._x, self._y, 0.127]]).T
        return np.vstack([np.hstack([R, P]), np.array([0, 0, 0, 1])])

    @property
    def bodyTworld(self):
        return np.linalg.inv(self.worldTbody)

    @property
    def opticalTregular(self):
        return np.array(
            [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64)

    @property
    def regularToptical(self):
        return np.linalg.inv(self.opticalTregular)

    @property
    def oTr(self):
        return self.opticalTregular

    @property
    def rTo(self):
        return self.regularToptical

    @property
    def bTl(self):
        return self.bodyTlidar

    @property
    def wTb(self):
        return self.worldTbody

    @property
    def bTw(self):
        return self.bodyTworld

    def chain(self, *transforms):
        ret = np.eye(4)
        for t in transforms:
            ret = ret @ getattr(self, t)
        return ret