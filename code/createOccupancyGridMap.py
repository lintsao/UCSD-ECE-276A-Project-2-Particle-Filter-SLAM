import numpy as np
from lidarProcess import *
from map import *
from pr2_utils import *
import cv2
import matplotlib
from collections import defaultdict
from collections import OrderedDict
from load_data import *

def createOccupancyGridMap(MAP_XLIM, MAP_YLIM, MAP_RESOLUTION, MAP_LOGODDS_FREE_DIFF, lidarData):
    logodds = Map2D(xlim=MAP_XLIM, ylim=MAP_YLIM, resolution=MAP_RESOLUTION)
    t = Transform(0, 0, 0)
    
    lidarScanXYPoints = lidarScanToXYPoints(lidarData["lidar_ranges"].T[0], (lidarData["lidar_range_min"], lidarData["lidar_range_max"]))

    lidarScanPointsWorld = np.array([0, 0, 0, 0])
    for x, y, z in lidarScanXYPoints:
        tmp = t.chain('wTb', 'bTl') @ np.array([x, y, z, 1]).T
        lidarScanPointsWorld = np.vstack([lidarScanPointsWorld, tmp])

    lidarScanPointsWorld = lidarScanPointsWorld[1:]
    lidarScanPointsWorld = lidarScanPointsWorld[:, :3]
    lidarScanPointsWorld = lidarScanPointsWorld[logodds.in_map(lidarScanPointsWorld), :2]
    lidarScanIndices = logodds.coordinate_to_index(lidarScanPointsWorld)

    mask = cv2.drawContours(image=np.zeros_like(logodds.data), 
                            contours=[lidarScanIndices.reshape((-1,1,2)).astype(np.int32)], 
                            contourIdx =-1, 
                            color = (MAP_LOGODDS_FREE_DIFF), 
                            thickness=-1)
    return mask