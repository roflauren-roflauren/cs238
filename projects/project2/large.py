## imports: 
import time 
import numpy  as np 
import pandas as pd 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

## environment specs: 
# actions: mysteries! 
ACTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
# discount rate:
lmb = 0.95

## hyperparameters: 
# number of nodes in hidden layer: 
NUM_HIDDEN_NODES = 12
# learning rate: 
lr = 1e-4
# number of times to loop through training data:
num_replays = 5
# number of episodes to process before transferring
# training NN weights to target NN: 
num_episodes_per_copy = 2000

## other constants: 
NUM_INPUTS = 6
KEY_STATES = [150000, 230000, 270000, 290000, 300000]


class DeepQNN(nn.Module):
    """ Neural network for deep Q-learning."""
    def __init__(self): 
        super().__init__()
        # create layers:
        self.ll1 = nn.Linear(NUM_INPUTS, NUM_HIDDEN_NODES)
        self.ll2 = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES)
        self.ll3 = nn.Linear(NUM_HIDDEN_NODES, len(ACTIONS))
        # initialize layers:
        nn.init.xavier_uniform_(self.ll1.weight)
        nn.init.xavier_uniform_(self.ll2.weight)
        nn.init.xavier_uniform_(self.ll3.weight)
        
    def forward(self, episode_input): 
        x1  = F.relu(self.ll1(episode_input))
        x2  = F.relu(self.ll2(x1))
        out = self.ll3(x2)    
        return out 


def construct_state_nn_input_tensor(s): 
    ret = [s]
    for state_val in KEY_STATES: 
        ret.append(s - state_val)
    return torch.Tensor(ret)


def main(): 
    # program time bookkeeping:
    program_start_time = time.time()
    
    # take over any available gpus:
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    # instantiate trainer & target deep q-learning networks: 
    train_nn, target_nn = DeepQNN().to(device), DeepQNN().to(device)
    
    # instantiate optimizer and loss function: 
    optimizer = torch.optim.Adam(train_nn.parameters(), lr = lr)
    loss_fn   = nn.MSELoss()
    
    # intake training episodes: 
    episodes = pd.read_csv('./data/large.csv', header=0)
    
    # log for recent episode training losses:
    ep_losses = []
    
    # training time bookkeeping: 
    training_start_time = time.time()
    
    # for each training data loop: 
    for replay_idx in range(num_replays):
        # bookkeeping: report which training data replay loop we're on.
        print("\nI'm chugging on episode set replay number: {replay_idx}\n".format(replay_idx = replay_idx))
        # for each training episode: 
        for idx, episode in enumerate(episodes.values):     
            # if sufficient episodes have elapsed, copy training weights to target network:
            if idx % num_episodes_per_copy == 0:
                target_nn.load_state_dict(train_nn.state_dict())
                
            # extract episode information: 
            s, a, r, sp = episode[0], episode[1], episode[2], episode[3]

            # construct the input tensors for train_nn and target_nn: 
            s_nn_rep  = construct_state_nn_input_tensor(s).to(device)
            sp_nn_rep = construct_state_nn_input_tensor(sp).to(device)
            
            # compute the estimated q-value of the given episode (q_opt(s,a)):
            q_val = train_nn(s_nn_rep)[a-1]
            # compute the optimal (action) state-action value for sp: 
            v_opt_sp = target_nn(sp_nn_rep).max(-1).values 
            # compute the deep q-learning target (r + lmb & v_opt(sp)): 
            target = r + lmb * v_opt_sp 
            
            # learn from this episode: 
            loss = loss_fn(q_val, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record episode loss:
            ep_losses.append(loss.item())
            # print average loss every 5000 episodes:
            if len(ep_losses) % 5000 == 0: 
                print("Average loss in last 5000 episodes: {avg_loss}".format(avg_loss = np.mean(ep_losses)))
                # bookkeeping: reset loss log:
                ep_losses = []

    # TODO: print different times taken
    # TODO: extract policy from target nn, write to file
    
    
    
if __name__ == "__main__":
    main()