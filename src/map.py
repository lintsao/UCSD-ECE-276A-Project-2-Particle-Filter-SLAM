import numpy as np
import matplotlib.pyplot as plt
from utils import filtering
from transform import *

class OccupancyMap:
    def __init__(self,
                 T,
                 xlim=(-30,30),
                 ylim=(-30,30),
                 res=0.2,
                 belief=0.8,
                 ):
        self.res = res
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim
        self.xsize = int(np.ceil((self.xmax - self.xmin) / self.res)) # Split the grids.
        self.ysize = int(np.ceil((self.ymax - self.ymin) / self.res)) # Split the grids.
        self.grid = np.zeros((self.xsize, self.ysize))
        self.logodd = np.log10(belief / (1 - belief))
        self.T = T
    
    def mapping(self, lidar_scan, pose):
        '''
        Update map based on current lidar_scan and pose

        Inputs:
            lidar_scan:     current lidar_scan
            pose:           current robot pose

        Outputs:
            None       
        '''
        laser_point_l = self.T.scanToPoint(lidar_scan) # (3, # of lasers) # Transform laser scan (range and angle) to points.
        laser_point_w = self.T.laserToWorld(laser_point_l, pose) # (# lasers, 3) # Transform laser points from laser frame to world frame.
        laser_point_w = filtering(laser_point_w, pose) # Remove unconfident points. 
        
        xi, yi = (laser_point_w[:, 0]/self.res).astype(int), (laser_point_w[:, 1]/self.res).astype(int) # Convert to grid.
        for (a, b) in zip(xi, yi):
            line = bresenham2D(int(pose['x'] / self.res), int(pose['y'] / self.res), a, b).astype(np.int16) # Check all grid points that the ray passes.
            x = a + self.grid.shape[0] // 2  # offset to center
            y = b + self.grid.shape[1] // 2  # offset to center

            if 0 <= x and x <= self.grid.shape[0] and 0 <= y and y <= self.grid.shape[1]:
                self.grid[x, y] += self.logodd # Increase the log-odds if the cell was observed occupied

            for lx in line[0][:-1] + self.grid.shape[0] // 2:
                for ly in line[1][:-1] + self.grid.shape[1] // 2:
                    if 0 <= lx and lx <= self.grid.shape[0] and 0 <= ly and ly <= self.grid.shape[1]:
                        self.grid[lx, ly] -= self.logodd # Decrease the log-odds if the cell was observed free.
        
        # clip
        self.grid[self.grid >= 100]  = 100
        self.grid[self.grid <= -100] = -100

    def plot(self, trajectory, data, pose, save_path, frame):
        '''
        Plot map and the trajectory
        '''
        laser_point_l = self.T.scanToPoint(data['lidar_ranges']) # (3, # of lasers)
        laser_point_w = self.T.laserToWorld(laser_point_l, pose) # (# lasers, 3)
        xi, yi = (laser_point_w[:, 0]/self.res).astype(int) + self.xsize // 2, (laser_point_w[:, 1]/self.res).astype(int) + self.ysize // 2

        fig = plt.figure(figsize=(6,6))
        plt.axis('off')
        plt.imshow(self.grid, cmap='Greys', origin='lower')
        plt.scatter(trajectory[1:].T[1], trajectory[1:].T[0], s=0.05, c='b', marker='.')
        plt.scatter(yi, xi, s=0.05, c='r', marker='.')
        plt.title(f"Frame #{frame}")
        plt.savefig(save_path, bbox_inches='tight')
        # plt.show()

def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    
    Inputs:
        (sx, sy):           start point of ray
        (ex, ey):           end point of ray
    Outputs:
        np.vstack((x,y)):   all grid points that the ray passes
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap 

    if dy == 0:
        q = np.zeros((dx+1,1))
    else:
        q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))

    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)

    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)

    return np.vstack((x,y))
