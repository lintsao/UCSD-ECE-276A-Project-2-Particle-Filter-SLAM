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
from sync import *

def getMotion(syncData):
    state = [[0, 0, 0]]
    index = 0
    # print(len(syncData))

    for key, value in syncData.items():
        if (index == 0):
            index += 1
            prevTime = key
            continue

        timeInterval = key - prevTime
        fr = value["encoder"][0]
        fl = value["encoder"][1]
        rr = value["encoder"][2]
        rl = value["encoder"][3]
        w = value["imu"][2]
        vr = (fr + rr) / 2 * 0.0022 * 40
        vl = (fl + rl) / 2 * 0.0022 * 40
        v = (vr + vl) / 2

        state.append([
        state[-1][0] + timeInterval * v * np.cos(state[-1][2]),
        state[-1][1] + timeInterval * v * np.sin(state[-1][2]),
        state[-1][2] + timeInterval * w,
        ])

        index += 1
        prevTime = key

    return state