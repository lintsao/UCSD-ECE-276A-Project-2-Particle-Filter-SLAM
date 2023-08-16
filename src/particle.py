import numpy as np

class Particle:
    def __init__(self, num=16):
        self.num = num
        self.weight = np.ones(num) / num
        # add perturbation for randomness, np.randon.randn ~ N(0,1)
        # self.state = np.zeros((num, 3)) + \
        #     np.random.randn(num, 3) * np.array([0.1, 0.1, 0.1 * np.pi/180])
        self.state = np.zeros((num, 3))
        
    def resampling(self):
        '''
        Implementation of Stratified Resampling algorithm
        '''
        N = self.num
        beta = 0
        chose_idx = []
        index = int(np.random.choice(np.arange(N), 1, p=[1/N]*N))  # choose an index uniformly

        for _ in range(N):
            beta = beta + np.random.uniform(low=0, high=2*np.max(self.weight), size=1)
            while(self.weight[index] < beta):
                beta  = beta - self.weight[index]
                index = (index+1) % N
            chose_idx.append(index)
        
        self.state = self.state[chose_idx]
        self.weight.fill(1/self.num)