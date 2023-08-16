import numpy as np
from utils import *
from map import OccupancyMap
from transform import *
import matplotlib.pyplot as plt
from motion import *
from observation import *
from particle import *

if __name__ == "__main__":
    print("Init ...")

    # Get data.
    dataset = 20
    root = "../data/"
    joint_data, lidar_range_lim, lidar_angle_lim = GetData(root, dataset) # Get the sync data from the encoder, imu, lidar.

    # Initializethe robot pose and the map.
    pose = { 'x': 0.0, 'y': 0.0, 'theta': 0.0 }
    xlim = (-30,30)
    ylim = (-30,30)
    T = Transform(lidar_range_lim, lidar_angle_lim)
    Map = OccupancyMap(T, xlim=xlim, ylim=ylim, res=0.2)

    # Dead-reckoning.
    robot_pose = GetOdometry(joint_data)

    # Initialize the particles.
    num_particle = 200
    P = Particle(num=num_particle)

    # Initialize the noise for the motion model.
    x_sigma = 1e-4 * 5
    y_sigma = 1e-4 * 5
    omega_sigma = 1e-5 * 1
    noise_var_scale = np.diag([x_sigma, y_sigma, omega_sigma])
    trajectory = np.empty(shape=(1,2))
    step_size = 4

    # Implement the particle filter slam.
    print("Start particle filter slam ...")
    for i in range(0, len(joint_data), step_size):
        if i == 0:
            continue

        data = joint_data[i]

        # Predict
        delta_pose = GetRelativeMotion(robot_pose, i, step_size)
        P.state = MotionModelPrediction(P.state , delta_pose, noise_var_scale)

        # Update
        best_particle = UpdateParticle(Map, P, data['lidar_ranges'], T)
        trajectory = np.vstack((trajectory, [int(best_particle[0] / Map.res) + Map.xsize // 2, int(best_particle[1] / Map.res) + Map.ysize // 2]))

        # Mapping
        best_pose = { 'x': best_particle[0], 'y': best_particle[1], 'theta': best_particle[2] }
        Map.mapping(data['lidar_ranges'], best_pose)

        # Resample
        N_eff = 1 / np.sum(P.weight ** 2)
        if N_eff < 0.1 * P.num:
            print("Resampling...")
            P.resampling()
        
        if i % 100 == 0:
            print(f"Step {i}")
            Map.plot(trajectory, data, best_pose, f"../frames_{dataset}/frame_{i}.png", i)
        
    print("Save frames to gifs ...")
    GetGif(f"../frames_{dataset}/", f"../gif/test_{dataset}.gif")
