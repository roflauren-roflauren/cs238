## imports: 
import time 
import numpy as np 
import pandas as pd 
import random as rd 

## environment specs: 
# num vals for vel, pos: 
NUM_VEL_VALS = 100
NUM_POS_VALS = 500
# actions - indicating amounts of acceleration.
ACTIONS = [1, 2, 3, 4, 5, 6, 7] 
# discount rate:
lmb = 1 

## hyperparameters: 
# default value for states: 
st_val_dft = -300
# interpolation rate for q-learning  
eta = 0.05
# number of times to loop through training data:
num_replays = 3

## other constants:
# random seed: 
rd.seed(238)


def q_learning(data): 
    # init. state-action value and state value matrices: 
    q_opt = np.full(shape=(len(ACTIONS), NUM_VEL_VALS, NUM_POS_VALS), fill_value=st_val_dft) 
    v_opt = np.full(shape=(NUM_VEL_VALS, NUM_POS_VALS), fill_value=st_val_dft)
    
    for _ in range(num_replays):
        # iterate through episodes and conduct q-learning! 
        for episode in data.values:
            # extract episode information:
            s, a, r, sp = episode[0], episode[1], episode[2], episode[3]

            # convert s, sp to indices into state-value & value matrices 
            (s_r, s_c)   = np.unravel_index(s,  shape=(NUM_VEL_VALS, NUM_POS_VALS))
            (sp_r, sp_c) = np.unravel_index(sp, shape=(NUM_VEL_VALS, NUM_POS_VALS))

            # *manual* reward shaping - subtract from 'r' distance to goal: 
            r -= (NUM_POS_VALS - s_c)

            # apply q-learning update (i.e., updating the state-action value matrix): 
            # equ: q_opt(s,a) <- (1 - eta) * q_opt(s,a) + eta * (r + lmb * v_opt(sp))
            q_opt[a-1][s_r][s_c] = (1 - eta) * q_opt[a-1][s_r][s_c] + eta * (r + lmb * v_opt[sp_r][sp_c])
            
            # update the state value matrix: 
            v_opt = q_opt.max(axis=0)
            
    return q_opt


def extract_policy(q_opt, filename):
    # create a file writer: 
    with open(filename, 'w') as f: 
        # iterate through states: 
        for i in range(NUM_VEL_VALS): 
            for j in range(NUM_POS_VALS):
                # initialize the policy action for this state:
                action_num = 0
                
                # retrieve the state-action values for this corresponding state: 
                sa_vals = q_opt[:,i,j]
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
    data = pd.read_csv('./data/medium.csv', header=0)
    
    # conduct vanilla q_learning with the data (report q-learning runtime): 
    q_learning_start_time = time.time() 
    q_opt = q_learning(data)
    print("Q-learning took: --- %s seconds ---" 
          % round((time.time() - q_learning_start_time), 3))
    
    # extract and write policy to file: 
    extract_policy(q_opt, './policy_files/medium.policy')
    
    # report program runtime: 
    print("Overall program took: --- %s seconds ---" 
          % round((time.time() - program_start_time), 3))
    

if __name__ == "__main__":
    main()