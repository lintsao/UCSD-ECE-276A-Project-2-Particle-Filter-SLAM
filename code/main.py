import numpy as np
from lidarProcess import *

from map import *
from pr2_utils import *
from load_data import *
from createOccupancyGridMap import *
from sync import *
from motionUtils import *
from getMotion import *
from lidarProcess import *



def main():
    # Set the data path and information.
    DATASET      = 21
    ENCODER_PATH = "../data/Encoders%d.npz"%DATASET
    LIDAR_PATH   = "../data/Hokuyo%d.npz"%DATASET
    IMU_PATH     = "../data/Imu%d.npz"%DATASET
    KINECT_PATH  = "../data/Kinect%d.npz"%DATASET

    FIRST_OCCUPANCY_GRID_MAP_PATH = "result/firstOccupancyMap%d.png"%DATASET
    THETA_PATH                    = "result/Theta%d.png"%DATASET
    TRAJECTORY_PATH               = "result/Trajectory.%d.png"%DATASET
    PARTICLE_PATH                 = "result/Particle.%d.png"%DATASET

    MAP_XLIM       = (-40, 40)
    MAP_YLIM       = (-40, 40)
    MAP_RESOLUTION = 0.05

    BELIEF                    = 0.8
    MAP_LOGODDS_FREE_DIFF     = np.log((1- BELIEF)/BELIEF)*0.5

    # Load data.
    encoderData, lidarData, imuData, kinectData = loadData(ENCODER_PATH, LIDAR_PATH, IMU_PATH, KINECT_PATH)

    # Use first LiDAR scan to create an occupancy grid map.
    mask = createOccupancyGridMap(MAP_XLIM, MAP_YLIM, MAP_RESOLUTION, MAP_LOGODDS_FREE_DIFF, lidarData)
    plot_map(1-mask, save_fig_name=FIRST_OCCUPANCY_GRID_MAP_PATH)

    # Sync data.
    syncData = sync(encoderData, lidarData, imuData, kinectData)

     # Get motion model.
    state = getMotion(syncData)
    plotTheta([i for i in range(len(state))], [x[2] for x in state], THETA_PATH)
    print([x[1] for x in state])
    plotTrajectory([x[0] for x in state], [x[1] for x in state], TRAJECTORY_PATH)

    # particles init
    P = particleInit(num=30)

    # grid map init
    MAP = mapInit()

    # init parameters
    step    = 20
    trajectory   = np.empty(shape=(1,2))
    scale    = np.array([0.001, 0.001, 0.01*np.pi/180])

    for i in range(0, len(lidarData['lidar_stamps']), step):
        if(i%100==0): 
            print(i)
        # if(i >= 1500): 
        #     break
        
        # Predict Step.
        deltaPose = getDeltaMotion(state, i, step)
        P['states'] = motionModelPredict(P['states'], deltaPose, scale)

        # Update Step.
        bestParticle = measureModelUpdate(MAP, P, lidarData, i)
        trajectory = np.vstack((trajectory, [int(bestParticle[0]/MAP['res']) + MAP['sizex']//2, int(bestParticle[1]/MAP['res']) + MAP['sizey']//2]))
        
        # Mapping Step.
        bestPose = {'x': bestParticle[0], 'y': bestParticle[1], 'theta': bestParticle[2]}
        MAP['map'] = mapping(MAP['map'], lidarData, i, MAP['res'], bestPose)
        
        # Resampling Step.
        N_eff = 1 / np.sum(P['weight']**2)
        if N_eff < 0.1 * P['number']:
            print("Resampling")
            P = resampling_wheel(P)

    # Plot. 
    plot(MAP['map'], MAP['res'], state, trajectory, PARTICLE_PATH)

if __name__ == "__main__":
    main()