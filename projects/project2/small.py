## imports: 
import time 
import numpy as np 
import pandas as pd 

## environment specs:
# grid world dims:
NUM_GRID_ROWS = 10 
NUM_GRID_COLS = 10 
# actions - 1: left, 2: right, 3: up, 4: down.
ACTIONS = [1, 2, 3, 4] 
# discount rate:
gamma = 0.95 

## hyperparameters:
# interpolation rate for q-learning:
eta = 0.05
# minimum val. s.t. the extracted policy will prescribe taking the action which
# produces this value instead of just moving toward the maximum value state:
min_state_val = 1

def get_state(grid_pos): 
    s = grid_pos - 1
    row_num = s // NUM_GRID_ROWS 
    col_num = s - (NUM_GRID_ROWS * row_num)
    return (row_num, col_num)


def q_learning(data): 
    # init. state-action value and state value matrices: 
    q_opt = np.zeros(shape=(len(ACTIONS), NUM_GRID_ROWS, NUM_GRID_COLS)) 
    v_opt = np.zeros(shape=(NUM_GRID_ROWS, NUM_GRID_COLS))
    
    # iterate through episodes and conduct q-learning! 
    for episode in data.values:
        # extract episode information:
        s, a, r, sp = get_state(episode[0]), episode[1], episode[2], get_state(episode[3])

        # apply q-learning update (i.e., updating the state-action value matrix): 
        # equ: q_opt(s,a) <- (1 - eta) * q_opt(s,a) + eta * (r + gamma * v_opt(sp))
        q_opt[a-1][s[0]][s[1]] = (1 - eta) * q_opt[a-1][s[0]][s[1]] + eta * (r + gamma * v_opt[sp[0]][sp[1]])
        
        # update the state value matrix: 
        v_opt = q_opt.max(axis=0)
            
    return q_opt, v_opt


def get_direction(i, j, max_val_state): 
    # uncouple row and column coordinates of max value state
    row_max_state, col_max_state = max_val_state[0], max_val_state[1]
    
    # if current pos (i, j) is closer to the max_val_state row-wise, move along rows: 
    if abs(i - row_max_state) <= abs(j - col_max_state): 
        # if in same row, move along columns:
        if i - row_max_state == 0: 
            # if to right of max state: 
            if j - col_max_state >= 0: 
                return 1 # go left 
            # if to left of max state: 
            else: 
                return 2 # go right 
        # if in higher row:
        elif i - row_max_state < 0: 
            return 4 # go down 
        # if in lower row: 
        elif i - row_max_state > 0: 
            return 3 # go up
    # if closer column-wise:
    else: 
        # if in same column, move along rows:
        if j - col_max_state == 0: 
            # if in lower row:  
            if i - row_max_state >= 0: 
                return 3 # go up 
            # if in higher row:
            else: 
                return 4 # go down 
        # if to left of max state:
        elif j - col_max_state < 0: 
            return 2 # go right 
        # if to right of max state
        elif j - col_max_state > 0: 
            return 1 # go left


def extract_policy(q_opt, v_opt, filename):
    # determine the maximum state value and the corresponding state: 
    v_max = v_opt.max()
    array = np.asarray(v_opt)
    idx   = (np.abs(array - v_max)).argmin()
    max_val_state = get_state(idx + 1)    
    
    # create a file writer: 
    with open(filename, 'w') as f: 
        # iterate through v_opt matrix: 
        for i in range(NUM_GRID_ROWS): 
            for j in range(NUM_GRID_COLS):
                # initialize the action for this state:
                action_num = 0
                
                # if there is a positive maximum value associated with this state: 
                if v_opt[i][j] > 0 and v_opt[i][j] > min_state_val: 
                    # determine which action generates this value: 
                    action_num = list(q_opt[:,i,j]).index(v_opt[i][j]) + 1 # add 1 b/c Python is 0-index'd
                # if not, just determine which action (when taken) will advance us toward the max value state:  
                else: 
                    action_num = get_direction(i, j, max_val_state)
                    
                # write that action to our policy for this state:        
                f.write(str(action_num)+'\n')
                

def main(): 
    # program time bookkeeping:
    program_start_time = time.time()
    
    # intake data: 
    data = pd.read_csv('./data/small.csv', header=0)
    
    # conduct vanilla q_learning with the data (report q-learning runtime): 
    q_learning_start_time = time.time() 
    q_opt, v_opt = q_learning(data)
    print("Q-learning took: --- %s seconds ---" 
          % round((time.time() - q_learning_start_time), 3))
    
    # extract and write policy to file: 
    extract_policy(q_opt, v_opt, './policy_files/small.policy')
    
    # report program runtime: 
    print("Overall program took: --- %s seconds ---" 
          % round((time.time() - program_start_time), 3))
    

if __name__ == "__main__":
    main()