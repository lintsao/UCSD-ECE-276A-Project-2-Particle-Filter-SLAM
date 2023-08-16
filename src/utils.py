import os
import glob
import numpy as np
from PIL import Image

def filtering(scan, pose, min_dist=0.1, max_dist=25):
    '''
    filter out scans that are too close and too far
    '''
    post_scan = np.empty([1,3])

    for i in range(len(scan)):
        x = scan[i][0] - pose['x']
        y = scan[i][1] - pose['y']
        distance = np.sqrt(x**2 + y**2)
        if (distance > min_dist and distance < max_dist):
            post_scan = np.vstack((post_scan, scan[i]))
        
    post_scan = post_scan[1:]
    return post_scan

def t_align(ref_time, target_time_list):
    '''
    Time stamp alignment
    
    Inputs:
        lidar_time - current lidar scan time stamp
        joint_time - list of joint time stamp
    Outputs:
        idx - corresponding joint time stamp index
    '''
    idx = np.argmin(abs(ref_time - target_time_list))
    return idx

def SyncData(data):
    '''
    Sync data with time stamps using lidar time stamps as reference
    
    Inputs:
        data - contains data read from files
    Outputs:
        joint_data - data align with time
    '''
    encoder_counts = data["encoder_counts"]
    encoder_stamps = data["encoder_stamps"]
    lidar_ranges = data["lidar_ranges"]
    lidar_stamps = data["lidar_stamps"]
    imu_angular_velocity = data["imu_angular_velocity"]
    imu_linear_acceleration = data["imu_linear_acceleration"]
    imu_stamps = data["imu_stamps"]

    joint_data = []
    for idx, t in enumerate(lidar_stamps):
        idx_encoder = t_align(t, encoder_stamps)
        idx_imu = t_align(t, imu_stamps)
        temp = {"encoder_counts" : encoder_counts[:, idx_encoder],
                "encoder_stamps" : encoder_stamps[idx_encoder],
                "lidar_ranges" : lidar_ranges[:, idx],
                "lidar_stamps" : lidar_stamps[idx],
                "imu_angular_velocity" : imu_angular_velocity[:, idx_imu],
                "imu_linear_acceleration" : imu_linear_acceleration[:, idx_imu],
                "imu_stamps" : imu_stamps[idx_imu]
                }
        joint_data.append(temp)
    return joint_data

def GetData(root, dataset):
    '''
    Load data from dataset and sync all data with time stamp
    '''
    
    with np.load(os.path.join(root, f"Encoders{dataset}.npz")) as data:
        encoder_counts = data["counts"] # 4 x n encoder counts
        encoder_stamps = data["time_stamps"] # encoder time stamps
    
    with np.load(os.path.join(root, f"Hokuyo{dataset}.npz")) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        lidar_range_min = data["range_min"] # minimum range value [m]
        lidar_range_max = data["range_max"] # maximum range value [m]
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
                                            # (# of laser, time)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

    with np.load(os.path.join(root, f"Imu{dataset}.npz")) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

    with np.load("../data/Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

    lidar_range_lim = (lidar_range_min.item(), lidar_range_max.item())
    lidar_angle_lim = (lidar_angle_min.item(), lidar_angle_max.item())

    data = {"encoder_counts" : encoder_counts,
            "encoder_stamps" : encoder_stamps,
            "lidar_ranges" : lidar_ranges,
            "lidar_stamps" : lidar_stamps,
            "imu_angular_velocity" : imu_angular_velocity,
            "imu_linear_acceleration" : imu_linear_acceleration,
            "imu_stamps" : imu_stamps}

    joint_data_sync = SyncData(data)

    return joint_data_sync, lidar_range_lim, lidar_angle_lim

def GetGif(frame_folder, save_path):
    fn = glob.glob(f"{frame_folder}/*.png")
    fn_for = sorted(fn, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    fn_rev = sorted(fn, key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)[1:]
    frames = [Image.open(image) for image in fn_for] + [Image.open(image) for image in fn_rev]
    # frames = [Image.open(image) for image in fn_for]
    frame_one = frames[0]
    frame_one.save(save_path, format="GIF", append_images=frames,
                save_all=True, duration=100, loop=0)
    
