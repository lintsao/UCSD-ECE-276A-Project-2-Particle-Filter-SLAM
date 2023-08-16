import numpy as np
from numpy import unravel_index
from utils import filtering

def Softmax(x, temp):
    '''
    Softmax with temperature

    Inputs:
        x:      particle weights
        temp:   temperature for relieving sharpness
    
    Outputs:
       results of softmax 
    '''
    y = np.exp(x / temp)
    return y / y.sum()

def GetMapCorrelation(im, x_im, y_im, vp, xs, ys):  
    '''
    Compute correlation of lidar scans from positions with the actual map

    Inputs:
        im:         the map 
        x_im,y_im:  physical x,y positions of the grid map cells
        vp[0:2,:]:  occupied x,y positions from range sensor (in physical unit)  
        xs, ys:     physical x,y,positions you want to evaluate "correlation" 

    Outputs:
        c:          sum of the cell values of all the positions hit by range sensor
    '''
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076
            ix = np.int16(np.round((x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                                    np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]]) + 1
    return cpr

def UpdateParticle(Map, P, lidar_data, T):
    '''
    Update particle weights with lidar scan

    Inputs:
        Map:            Current Map
        P:              Particles
        lidar_data:     Current lidar_data
        T:              Transform helper
    
    Outputs:
        best_particle:  particle with highest correlation with the map
    '''
    l = 2
    corrs = []
    res = Map.res
    particles = P.state

    grid_tmp = np.zeros_like(Map.grid)  # for calculate correlation
    grid_tmp[Map.grid > 0] = 1          # occupied
    grid_tmp[Map.grid < 0] = 0          # free

    laser_point_l = T.scanToPoint(lidar_data) # (3, # of lasers)
    x_im, y_im = np.arange(Map.xmin, Map.xmax + res, res), np.arange(Map.ymin, Map.ymax + res, res)
    x_range, y_range = np.arange(-res * l, res * l + res, res), np.arange(-res * l, res * l + res, res)

    for i in range(len(particles)):
        particle_state = {'x':particles[i][0], 'y':particles[i][1], 'theta':particles[i][2]}
        scan_w = T.laserToWorld(laser_point_l, particle_state)
        # scan_w = LaserToWorld(laser_point_l, particle_state)
        scan_w = filtering(scan_w, particle_state)
        x, y = scan_w[:,0], scan_w[:,1]
        corr = GetMapCorrelation(grid_tmp, x_im, y_im, np.vstack((x,y)), particles[i][0] + x_range, particles[i][1] + y_range)
        corrs.append(np.max(corr))
        # bias_x, bias_y = unravel_index(corr.argmax(), corr.shape)
        # # print(bias_x, bias_y)
        # P.state[i][0] += (bias_x - l) * res
        # P.state[i][1] += (bias_y - l) * res

    # get the particle with largest weight
    corrs = np.array(corrs)
    
    P.weight = P.weight * (corrs / np.linalg.norm(corrs))
    # P.weight = Softmax(P.weight * corrs, 7)

    best_idx = np.where(P.weight==np.max(P.weight))[0][0]
    best_particle = particles[best_idx]
    return best_particle

