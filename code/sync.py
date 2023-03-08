import numpy as np
import sys
from lidarProcess import *
from map import *
from pr2_utils import *
import cv2
import matplotlib
from collections import defaultdict
from collections import OrderedDict
from load_data import *
from createOccupancyGridMap import *

def sync(encoderData, lidarData, imuData, kinectData):
    syncData = OrderedDict(dict())

    for i in range(len(lidarData["lidar_stamps"])):
        lidarTime = lidarData["lidar_stamps"][i]
        imuTimeList = imuData["imu_stamps"]
        idx = np.argmin(abs(lidarTime - imuTimeList))
        syncData[lidarTime] = {"lidar": lidarData["lidar_ranges"].T[i], "imu": [imuData["imu_angular_velocity"][0][idx], imuData["imu_angular_velocity"][1][idx], imuData["imu_angular_velocity"][2][idx]]}

    #sync imu, lidar ,and encoder
    for i in range(len(lidarData["lidar_stamps"])):
        lidarTime = lidarData["lidar_stamps"][i]
        encoderTimeList = encoderData["encoder_stamps"]
        idx = np.argmin(abs(lidarTime - encoderTimeList))
        syncData[lidarTime]["encoder"] = [encoderData["encoder_counts"][0][idx], encoderData["encoder_counts"][1][idx], encoderData["encoder_counts"][2][idx], encoderData["encoder_counts"][3][idx]]

    #sync imu, lidar ,encoder, and kinect
    # for i in range(len(lidarData["lidar_stamps"])):
    #     lidarTime = lidarData["lidar_stamps"][i]
    #     kinectTimeList = kinectData["kinect_stamps"]
    #     idx = np.argmin(abs(lidarTime - kinectTimeList))
    #     syncData[lidarTime]["kinect"] = kinectData["rgb_stamps"][idx]

    return syncData