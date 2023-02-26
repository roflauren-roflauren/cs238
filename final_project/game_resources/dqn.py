import numpy  as np
import pandas as pd 
import random as rd 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

## constants: 
# dimensionality of input, output: 
NUM_INPUTS, NUM_ACTIONS = 42, 16


class DeepQNN(nn.Module): 
    """ A 4-layer neural network for deep Q-learning. """
    def __init__(self, num_hn_layer1, num_hn_layer2):
        super().__init__()
        # create layers:
        self.ll1 = nn.Linear(NUM_INPUTS   , num_hn_layer1)
        self.ll2 = nn.Linear(num_hn_layer1, num_hn_layer2)
        self.ll3 = nn.Linear(num_hn_layer2, NUM_ACTIONS  )
        # initialize layers:
        nn.init.xavier_uniform_(self.ll1.weight)
        nn.init.xavier_uniform_(self.ll2.weight)
        nn.init.xavier_uniform_(self.ll3.weight)
        
    def forward(self, episode_input): 
        x1  = F.relu(self.ll1(episode_input))
        x2  = F.relu(self.ll2(x1))
        out = self.ll3(x2)    
        return out 
    
class Buffer():
    """ A buffer to store and sample episodes for NN training. """
    def __init__(self, device="cpu", seed = 238): 
        self.buffer = []
        self.device = device
        self.seed = seed
    
    def __len__(self): 
        return len(self.buffer)
    
    def reset(self):
        self.buffer = []
    
    def add(self, s, a, r, sp, t): 
        self.buffer.append((s, a, r, sp, t))
    
    def sample(self, batch_size): 
        # prepare containers for sampled (s, a, r, sp, t) values:
        s, a, r, sp, t = [], [], [], [], []
        # set random seed to ensure replicability of sampling and training: 
        rd.seed(self.seed)
        # generate indices to sample:
        sample_idxs = np.random.choice(len(self.buffer), batch_size)
        # populate s, a, r, sp, t containers with sampled values: 
        for idx in sample_idxs: 
            si, ai, ri, spi, ti = self.buffer[idx]
            s.append( np.array(si,  copy = False)), a.append( np.array(ai,  copy = False)), r.append( np.array(ri,  copy = False)), \
                sp.append(np.array(spi, copy = False)),  t.append( np.array(ti,  copy = False))
        # convert sampled lists to tensors: 
        s, a, r, sp, t = torch.tensor(np.array(s, dtype=np.float32), device=self.device), \
            torch.tensor(np.array(a, dtype='int64'), device=self.device), torch.tensor(np.array(r, dtype=np.float32), device=self.device), \
                torch.tensor(np.array(sp), device=self.device), torch.tensor(np.array(t, dtype=np.float32), device=self.device)  
        # return the sampled episode information: 
        return s, a, r, sp, t