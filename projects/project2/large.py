## imports: 
import time 
import numpy  as np 
import pandas as pd 
import random as rd 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

## environment specs: 
# number of states: 
NUM_STATES = 312020
# number of digits in max index state:
MAX_STATES_DIGITS = 6
# actions: mysteries! 
ACTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
# discount rate:
lmb = 0.95

## hyperparameters: 
# number of nodes in hidden layer: 
NUM_HIDDEN_NODES = 12
# learning rate: 
lr = 6e-4
# number of times to loop through training data:
num_replays = 20
# number of episodes to process before transferring
# training NN weights to target NN: 
num_episodes_per_copy = 1000

## other constants: 
NUM_INPUTS = 12
KEY_STATES = [150000, 230000, 270000, 290000, 300000]
rd.seed(238)


class DeepQNN(nn.Module):
    """ A simple 3-layer neural network for deep Q-learning."""
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
    # nn input vector consists of state idx: 
    ret = [s]
    # distance to key states with high/low rewards:
    for state_val in KEY_STATES: 
        ret.append(s - state_val)
    digits = [int(d) for d in str(s)]
    # digits of state index as individual features: 
    if len(digits) < MAX_STATES_DIGITS: 
        # pad with 0's if necessary
        ret = ret + ([0] * (MAX_STATES_DIGITS - len(digits))) + digits
    else: 
        ret = ret + digits
    return torch.Tensor(ret)


def main(): 
    ## initializations and data:
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
    
    ## kick off training!
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
    
    # report network training time: 
    print("Deep Q-learning took: --- %s seconds ---"
          % round((time.time() - training_start_time)))
    
    # save training and target networks weights: 
    torch.save(train_nn.state_dict(),  "./large_model_weights/train_nn.params")
    torch.save(target_nn.state_dict(), "./large_model_weights/target_nn.params")
    
    # policy extraction/eval. time bookkeeping: 
    policy_extraction_start_time = time.time()
    
    ## extract and write policy to file: 
    # create a file writer: 
    with open('./policy_files/large.policy', 'w') as f: 
        # iterate through states: 
        for state_num in range(NUM_STATES): 
            # increment state number (since Python is 0-index'd):
            state_num += 1 
            # construct state vector for this state: 
            s_nn_rep = construct_state_nn_input_tensor(state_num).to(device)
            # compute the state-action values for this state: 
            q_vals = target_nn(s_nn_rep)
            # init. the max Q-val idx:
            max_action_idx = 0
            try: 
                # extract the index of the action with the maximum q_val: 
                max_action_idx = q_vals.detach().cpu().numpy().argmax() + 1 # increment by 1 b/c Python is 0-index'd
            except:
                # in case anything goes wrong, just gen. a random action: 
                max_action_idx = rd.choice(ACTIONS)
            # write extracted action to policy file: 
            f.write(str(max_action_idx)+'\n')
    
    ## program time bookkeeping:
    # report policy evaluation time: 
    print("Policy extraction took: --- %s seconds ---"
          % round(time.time() - policy_extraction_start_time))
        
    # report program runtime: 
    print("Overall program took: --- %s seconds ---"
          % round(time.time() - program_start_time))

if __name__ == "__main__":
    main()