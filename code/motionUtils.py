import numpy as np

def getOdom(lidarData):
    robotPose = np.zeros((len(lidarData),3))
    currX = 0
    currY = 0
    currAngle = 0

    for idx, data in enumerate(lidarData):
        currX += data['delta_pose'][0][0]
        currY += data['delta_pose'][0][1]
        currAngle += data['delta_pose'][0][2]
        robotPose[idx][0] = currX
        robotPose[idx][1] = currY
        robotPose[idx][2] = currAngle

    return robotPose

def getDeltaMotion(robotPose, idx, step):
    if(idx >= step): 
        deltaPose = [robotPose[idx][0] - robotPose[idx-step][0], robotPose[idx][1] - robotPose[idx-step][1], robotPose[idx][2] - robotPose[idx-step][2]]
    else: 
        deltaPose = [robotPose[idx][0] - robotPose[idx-1][0], robotPose[idx][1] - robotPose[idx-1][1], robotPose[idx][2] - robotPose[idx-1][2]]
        
    return deltaPose

def motionModelPredict(particles, motion, scale):
    motionNoise = np.random.randn(particles.shape[0],3) * scale
    particles = particles + motion + motionNoise
    particles[:,2] = particles[:,2] % (2*np.pi)
    return particles