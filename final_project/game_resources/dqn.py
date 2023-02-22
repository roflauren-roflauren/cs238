import numpy  as np
import pandas as pd 
import random as rd 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

## constants: 
# dimensionality of input, output: 
NUM_INPUTS, NUM_ACTIONS = 10, 16

## hyperparameters: 
# number of nodes in hidden layer: 
NUM_HIDDEN_NODES = 12

class DeepQNN(nn.Module): 
    """A 4-layer neural network for deep Q-learning."""
    def __init__(self):
        super().__init__()
        # create layers:
        self.ll1 = nn.Linear(NUM_INPUTS, NUM_HIDDEN_NODES)
        self.ll2 = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES)
        self.ll3 = nn.Linear(NUM_HIDDEN_NODES, NUM_ACTIONS)
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
    """ A buffer to store and sample episodes for NN training."""
    def __init__(self, device="cpu"): 
        self.buffer = []
        self.device = device
    
    def __len__(self): 
        return len(self.buffer)
    
    def reset(self):
        self.buffer = []
    
    def add(self, s, a, r, sp, t): 
        self.buffer.append((s, a, r, sp, t))
    
    def sample(self, batch_size): 
        # prepare containers for sampled (s, a, r, sp, t) values:
        s, a, r, sp, t = [], [], [], [], []
        # generate indices to sample:sta
        sample_idxs = np.random.choice(len(self.buffer), batch_size)
        # populate s, a, r, sp, t containers with sampled values: 
        for idx in sample_idxs: 
            si, ai, ri, spi, ti = self.buffer[idx]
            s.append(si), a.append(ai), r.append(ri), sp.append(spi), t.append(ti)
        # convert sampled lists to tensors: 
        s, a, r, sp, t = torch.Tensor(s, device=self.device), torch.Tensor(a, device=self.device), \
            torch.Tensor(r, device=self.device), torch.Tensor(sp, device=self.device), torch.Tensor(t, device=self.device)  
        # return the sampled episode information: 
        return s, a, r, sp, t