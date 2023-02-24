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
# max iterations for value iteration: 
MAX_ITERS = 1000000

## other constants:
# random seed: 
rd.seed(238)

def normalize(d, target=1.0):
   raw = sum(d.values())
   factor = target/raw
   return {key:round(value*factor,6) for key,value in d.items()}            

def main(): 
    # program time bookkeeping:
    program_start_time = time.time()
    
    # intake data: 
    data = pd.read_csv('./data/large.csv', header=0)
    
    # init. transition prob. dictionaries: 
    action_prob_dict = {
        1 : {}, 4: {}, 7: {},
        2 : {}, 5: {}, 8: {},
        3 : {}, 6: {}, 9: {}
    }
    # populate transition prob. dictionaries: 
    for idx, episode in enumerate(data.values): 
        s, a, r, sp = episode[0], episode[1], episode[2], episode[3]
        if sp - s in action_prob_dict[a]: 
            action_prob_dict[a][sp-s] += 1
        else: 
            action_prob_dict[a][sp-s] = 1
    
    # init. state values: 
    state_vals = np.full(shape=(NUM_STATES, 1), fill_value=st_val_dft)
    
    # conduct asynchronous value iteration: 
    vi_start_time = time.time()
    for idx in range(MAX_ITERS): 
        if idx % 10000 == 0: print("Working on iteration {idx}".format(idx=idx))
        
        state_to_update = idx % NUM_STATES
        action_values = []
        
        for a in ACTIONS: 
            action_value = 0
            
            for state_chg in action_prob_dict[a].keys(): 
                r = 0
                sp = min(state_to_update + state_chg, 0)
                if sp == 151211 and state_to_update != 151211:
                    r = 100 
                elif sp == 151312 and state_to_update != 151312:
                    r = 100 
                action_value += lmb * state_vals[sp] * action_prob_dict[a][state_chg]
                action_value += r 
    
            action_values.append(action_value)

        state_vals[state_to_update] = max(action_values)
    
    print("Asynch VI took: --- %s seconds ---" 
        % round((time.time() - vi_start_time), 3))
    
    # extract policy:
    pe_start_time = time.time()
    with open('./policy_files/large_vi.policy', 'w') as f: 
        # iterate through states: 
        for i in range(NUM_STATES): 
            # initialize the policy action for this state:
            action_num = 0
            
            action_values = []
        
            for a in ACTIONS: 
                action_value = 0
                
                for state_chg in action_prob_dict[a].keys(): 
                    r = 0
                    sp = min(state_to_update + state_chg, 0)
                    if sp == 151211 and state_to_update != 151211:
                        r = 100 
                    elif sp == 151312 and state_to_update != 151312:
                        r = 100 
                    action_value += lmb * state_vals[sp] * action_prob_dict[a][state_chg]
                    action_value += r 

                action_values.append(action_value)
                opt_val = max(action_values)
                action_idx = action_values.index(opt_val) + 1
            
            f.write(str(action_num)+'\n')
    
    print("PE took: --- %s seconds ---" 
        % round((time.time() - pe_start_time), 3))
    print("Overall program took: --- %s seconds ---" 
        % round((time.time() - program_start_time), 3))

    return 0

if __name__ == "__main__":
    main()