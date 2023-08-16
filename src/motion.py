import numpy as np

def GetVelocityFromEncoder(encoder_data):
    '''
    Compute linear velocity from encoder data

    Inputs: 
        encoder_data:       reading of front/rear and left/right wheel

    Outputs:
        linear velocity:    average of left and right velocity
    '''
    fr, fl, rr, rl = encoder_data
    vl = (fl + rl) / 2 * 0.0022 * 40
    vr = (fr + rr) / 2 * 0.0022 * 40
    return (vl + vr) / 2

def GetOdometry(joint_data):
    '''
    Compute odometry for differential-drive robot with encoder (linear velocity) and imu data (angular velocity)

    Inputs: 
        encoder_data:       reading of front/rear and left/right wheel

    Outputs:
        linear velocity:    average of left and right velocity
    '''
    robot_pose = np.zeros((len(joint_data), 3)) # [len of data, (x, y, theta)]
    # init pose
    curr_x = 0
    curr_y = 0
    curr_angle = 0

    for i in range(len(joint_data)):
        if i == 0:
            continue
        
        tau = joint_data[i]['lidar_stamps'] - joint_data[i-1]['lidar_stamps'] # Time difference.
        omega = joint_data[i]['imu_angular_velocity'][2] # yaw
        vt = GetVelocityFromEncoder(joint_data[i]['encoder_counts'])

        curr_x += tau * vt * np.cos(curr_angle)
        curr_y += tau * vt * np.sin(curr_angle)
        curr_angle += tau * omega
        robot_pose[i][0] = curr_x
        robot_pose[i][1] = curr_y
        robot_pose[i][2] = curr_angle

    return robot_pose

def MotionModelPrediction(p_state, motion, noise_var_scale):
    '''
    Apply prediction step to the particles

    Inputs:
        p_state:            states of particles
        motion:             motion applied in delta time step
        noise_var_scale:    variance of motion model noise
    
    Outputs:
        p_state:            states of particles
    '''
    # motion_noise = np.random.randn(p_state.shape[0], 3) * noise_var_scale
    u = np.random.multivariate_normal(np.array(motion), noise_var_scale,
                                          len(p_state))
    # p_state = p_state + motion + motion_noise
    p_state = p_state + u
    p_state[:,2] = p_state[:,2] % (2 * np.pi)
    return p_state

def GetRelativeMotion(robot_pose, idx, step_size):
    '''
    Get delta motion within delta time step

    Inputs:
        robot_pose:     robot odometry (with enocder and IMU)
        idx:            current index of joint data
        step_size:      step_size for time step
    
    Outputs:
        delta_pose:     delta pose based on step_size
    '''

    if idx >= step_size: 
        delta_pose = robot_pose[idx] - robot_pose[idx-step_size]
    else:
        # first iteration delta can only get from previous pose
        delta_pose = robot_pose[idx] - robot_pose[idx-1]
        
    return delta_pose