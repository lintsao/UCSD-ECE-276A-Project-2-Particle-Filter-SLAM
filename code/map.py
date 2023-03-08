import numpy as np
from pr2_utils import *
from lidarProcess import *
import sys

def mapInit(xmin=-40, xmax=40, ymin=-40, ymax=40, res=0.05):
    MAP = {}
    MAP['res']   = res  #meters
    MAP['xmin']  = xmin #meters
    MAP['ymin']  = ymin
    MAP['xmax']  = xmax
    MAP['ymax']  = ymax
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res']))
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res']))
    MAP['map']   = np.zeros((MAP['sizex'], MAP['sizey']))
    return MAP

def softmax(x):
    x = np.exp(x-np.max(x))
    return x / x.sum()

def filtering(scan, pose, min_dist=0.1, max_dist=25):
    postScan = np.empty([1,3])

    for i in range(len(scan)):
        x = scan[i][0] - pose['x']
        y = scan[i][1] - pose['y']
        distance = np.sqrt(x**2 + y**2)
        if (distance > min_dist and distance < max_dist):
            postScan = np.vstack((postScan, scan[i][:3]))
        
    postScan = postScan[1:]
    return postScan

def measureModelUpdate(MAP, P, lidarData, i):
    # calculate map correlation for each particle
    l = 2
    corrs = []
    res = MAP['res']
    particles = P['states']

    grid_tmp = np.zeros_like(MAP['map'])  # for calculate correlation
    # print(MAP)
    grid_tmp[MAP['map'] > 0] = 1          # occupied
    grid_tmp[MAP['map'] < 0] = 0          # free
    lidarScanXYPoints = lidarScanToXYPoints(lidarData["lidar_ranges"].T[i], (lidarData["lidar_range_min"], lidarData["lidar_range_max"]))
    xIm, yIm = np.arange(MAP['xmin'], MAP['xmax']+res, res), np.arange(MAP['ymin'], MAP['ymax']+res, res)
    xRange, yRange = np.arange(-res*l, res*l+res, res), np.arange(-res*l, res*l+res, res)

    for i in range(len(particles)):
        particle_state = {'x':particles[i][0], 'y':particles[i][1], 'theta':particles[i][2]}
        t = Transform(particles[i][0], particles[i][1], particles[i][2])

        lidarScanPointsWorld = np.array([0, 0, 0, 0])
        for x, y, z in lidarScanXYPoints:
            tmp = t.chain('wTb', 'bTl') @ np.array([x, y, z, 1]).T
            lidarScanPointsWorld = np.vstack([lidarScanPointsWorld, tmp])

        lidarScanPointsWorld = lidarScanPointsWorld[1:]
        lidarScanPointsWorld = filtering(lidarScanPointsWorld, particle_state)
        x, y = lidarScanPointsWorld[:,0], lidarScanPointsWorld[:,1]
        corr = mapCorrelation(grid_tmp, xIm, yIm, np.vstack((x,y)), particles[i][0]+xRange, particles[i][1]+yRange)
        corrs.append(np.max(corr))
    
    # get the particle with largest weight
    corrs = np.array(corrs)
    P['weight'] = softmax(P['weight'] * corrs)
    bestIdx = np.where(P['weight']==np.max(P['weight']))[0][0]
    bestParticle = particles[bestIdx]

    return bestParticle

def mapCorrelation(im, x_im, y_im, vp, xs, ys):
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
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr

class Map2D:

    def __init__(self,
                 xlim=(-20, 20),
                 ylim=(-20, 20),
                 resolution=0.05,
                 dtype=np.float64):
        self.resolution = resolution
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim
        self.xsize = int(np.ceil((self.xmax - self.xmin) / self.resolution + 1))
        self.ysize = int(np.ceil((self.ymax - self.ymin) / self.resolution + 1))
        self._map = np.zeros((self.xsize, self.ysize), dtype=dtype)

    @property
    def data(self):
        return self._map

    @data.setter
    def data(self, d):
        assert d.shape == self._map.shape
        self._map = d

    def in_map(self, coordinates):
        return np.logical_and(
            np.logical_and(self.xmin <= coordinates[:, 0],
                           coordinates[:, 0] <= self.xmax),
            np.logical_and(self.ymin <= coordinates[:, 1],
                           coordinates[:, 1] <= self.ymax))

    def coordinate_to_index(self, coordinates):
        coordinates = np.array(coordinates)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)

        return np.hstack([
            np.ceil((coordinates[:, 0] - self.xmin) / self.resolution).reshape(
                -1, 1),
            np.ceil((coordinates[:, 1] - self.ymin) / self.resolution).reshape(
                -1, 1),
        ]).astype(np.int32)
    
def resampling_wheel(P):
    '''
        Resampling step of particle filter
    '''
    N = P['number']
    beta = 0
    choseIdx = []
    index = int(np.random.choice(np.arange(N), 1, p=[1/N]*N))  # choose an index uniformly

    for _ in range(N):
        beta = beta + np.random.uniform(low=0, high=2*np.max(P['weight']), size=1)
        while(P['weight'][index] < beta):
            beta  = beta - P['weight'][index]
            index = (index+1) % N
        choseIdx.append(index)
    
    P['states'] = P['states'][choseIdx]
    P['weight'].fill(1/N)

    return P
    
def mapping(grid, data, id, res, pose):
    freeOdd = np.log(9)/4
    occuOdd = np.log(9)

    # from range to points
    lidarScanXYPoints = lidarScanToXYPoints(data["lidar_ranges"].T[id], (data["lidar_range_min"], data["lidar_range_max"]))
    # from laser frame to world frame
    lidarScanPointsWorld = np.array([0, 0, 0, 0])
    t = Transform(pose['x'], pose['y'], pose['theta'])
    for x, y, z in lidarScanXYPoints:
        tmp = t.chain('wTb', 'bTl') @ np.array([x, y, z, 1]).T
        lidarScanPointsWorld = np.vstack([lidarScanPointsWorld, tmp])

    lidarScanPointsWorld = lidarScanPointsWorld[1:]
    # filter out too close or far points
    lidarScanPointsWorld = filtering(lidarScanPointsWorld, pose)

    xi, yi = (lidarScanPointsWorld[:, 0]/res).astype(int), (lidarScanPointsWorld[:, 1]/res).astype(int)

    for (a, b) in zip(xi, yi):
        line = bresenham2D(int(pose['x']/res), int(pose['y']/res), a, b).astype(np.int16)
        x    = a + grid.shape[0]//2  # offset to center
        y    = b + grid.shape[1]//2  # offset to center
        grid[x, y] += occuOdd
        grid[line[0] + grid.shape[0]//2, line[1] + grid.shape[1]//2] -= freeOdd

    # clip
    grid[grid >= 100]  = 100
    grid[grid <= -100] = -100

    return grid

def plot(grid, res, robot_pose, trajectory, path):
    print("Plot...")
    fig = plt.figure(figsize=(12,6))

    ax1 = fig.add_subplot(121)
    plt.plot(np.array(robot_pose).T[0], np.array(robot_pose).T[1], label="Odom of Lidar")

    plt.scatter((trajectory[1:].T[0] - grid.shape[0]//2)*res, 
                (trajectory[1:].T[1] - grid.shape[1]//2)*res, 
                label="Odom of Particle Filter", s=2, c='r')
    plt.legend(loc='upper left')
    
    ax2 = fig.add_subplot(122)
    plt.imshow(grid, cmap='gray', vmin=-100, vmax=100, origin='lower')
    plt.scatter(trajectory[1:][1], trajectory[1:][0], s=1, c='r')

    plt.title("Occupancy grid")
    plt.savefig(path)
    plt.show()