## imports: 
import time 
import numpy as np 
import pandas as pd 
import random as rd 

## environment specs: 
# number of states: 
NUM_STATES = 312020
# actions: mysteries! 
ACTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
# discount rate:
lmb = 0.95

## hyperparameters: 
# default value for states: 
st_val_dft = 0
# number of times to loop through training data:
num_replays = 5
# interpolation rate for q-learning  
eta = (num_replays + 1) * 0.05

## other constants:
# random seed: 
rd.seed(238)
# replay buffer size: 
batch_size = 32


def q_learning(data): 
    # init. state-action value and state value matrices: 
    q_opt = np.full(shape=(len(ACTIONS), NUM_STATES), fill_value=st_val_dft) 
    v_opt = np.full(shape=(NUM_STATES), fill_value=st_val_dft)

    for rn in range(num_replays):
        # report replay progress:
        print("Working on replay number: {rn}".format(rn=rn))
        # decrement and report learning rate: 
        global eta 
        eta -= 0.05
        print("Learning rate for this replay: {eta}".format(eta=eta))
        # re-init. replay buffer: 
        buffer = []    
    
        # iterate through episodes and conduct q-learning! 
        for idx, episode in enumerate(data.values):
            # report episode progress:
            if (idx % 10000 == 0): print("Now working on episode: {idx}".format(idx=idx))
            # extract episode information:
            s, a, r, sp = episode[0], episode[1], episode[2], episode[3]
            # add episode to buffer: 
            buffer.append((s, a, r, sp))
            # if buffer has enough episodes:
            if len(buffer) > batch_size: 
                # sample a batch:
                samples = rd.sample(buffer, batch_size)
                # conduct q-learning for each sample in batch: 
                for (s, a, r, sp) in samples: 
                    # apply q-learning update (i.e., updating the state-action value matrix): 
                    # equ: q_opt(s,a) <- (1 - eta) * q_opt(s,a) + eta * (r + lmb * v_opt(sp))
                    q_opt[a-1][s] = (1 - eta) * q_opt[a-1][s] + eta * (r + lmb * v_opt[sp])
                # update the state value matrix after all Q-updates for this batch: 
                v_opt = q_opt.max(axis=0)
            
    return q_opt


def extract_policy(q_opt, filename):
    # create a file writer: 
    with open(filename, 'w') as f: 
        # iterate through states: 
        for i in range(NUM_STATES): 
            # initialize the policy action for this state:
            action_num = 0
            # retrieve the state-action values for this corresponding state: 
            sa_vals = q_opt[:,i]
            # determine the maximum action value for this state: 
            max_action_val = max(list(sa_vals))
            # if the max action value is the same as the default, no meaningful learning occurred. 
            if max_action_val == st_val_dft:
                # so, just execute a random action as the policy: 
                action_num = rd.randint(1, len(ACTIONS))
            # otherwise, if the max action value exceeds the default value: 
            else: 
                # select the action which achieves the max action value as the policy: 
                action_num = list(sa_vals).index(max_action_val) + 1 # plus 1 because Python is 0-indexed
                
            # write the extracted action for this state to our policy file:        
            f.write(str(action_num)+'\n')
                

def main(): 
    # program time bookkeeping:
    program_start_time = time.time()
    
    # intake data: 
    data = pd.read_csv('./data/large.csv', header=0)
    
    # conduct vanilla q_learning with the data (report q-learning runtime): 
    q_learning_start_time = time.time() 
    q_opt = q_learning(data)
    print("Q-learning took: --- %s seconds ---" 
          % round((time.time() - q_learning_start_time), 3))
    
    # extract and write policy to file: 
    extract_policy(q_opt, './policy_files/large.policy')
    
    # report program runtime: 
    print("Overall program took: --- %s seconds ---" 
          % round((time.time() - program_start_time), 3))
    

if __name__ == "__main__":
    main()