import time 

import numpy  as np
import pandas as pd 
import random as rd 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

# custom classes:
import dqn 

class Trainer():
    """ A class for training the gameplay agent. """
    def __init__(self, 
        device : str = 'cpu', 
        load_prev_params : bool = False, 
        params_path : str = None
    ):
        """Initializes the Trainer object."""
        super().__init__()
        # register the device to be used for training: 
        self.device = device 
        # initialize the training and target NN on the desired device: 
        self.train_nn  = dqn.DeepQNN().to(self.device)
        self.target_nn = dqn.DeepQNN().to(self.device)
        # check if there are any parameter files to initialize our models from:
        if load_prev_params == True: 
            if params_path is None: 
                raise ValueError("Please provide a path to the file containing the network parameters you wish to load.")
            else: 
                self.train_nn.load_state_dict(torch.load(params_path))
                self.target_nn.load_state_dict(torch.load(params_path))
        # initialize a experience replay buffer: 
        self.replay_buffer = dqn.Buffer(self.device)
     
    def train(self, 
        lr = 6e-4, 
        loss_fn = nn.MSELoss(),
        optimizer_type = "Adam", 
        num_train_games = int(1e6),
        discount_factor = 0.95,
        replay_batch_size = 32,
        num_episodes_per_rp   = 1000,
        num_episodes_per_copy = 2000, 
        num_episodes_per_save = 10000,
        model_params_save_dir = "../model_params/"
    ):
        """This function actually performs the NN training.""" 
        # initialize the optimizer: 
        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.train_nn.parameters(), lr = lr)
        elif optimizer_type == "SGD": 
            self.optimizer = torch.optim.SGD(self.train_nn.parameters(), lr = lr)
        else: 
            raise ValueError("Only \"Adam\" or \"SGD\" are accepted as supported optimizer types at this point in time.")
        # initialize the loss function: 
        self.loss_fn = loss_fn
        
        ## MAIN TRAINING LOOP STARTS HERE: 
        # track total training time and total episode count: 
        train_time_start, episode_count = time.time(), 0
        
        # for each training game: 
        for game in num_train_games: 
            # TODO: instantiate a new game with all necessary variables reset
            # TODO: reset all necessary class variables 
            self.replay_buffer.reset() 
        
            # sentinel which captures whether the current training game is ongoing or over: 
            game_over = False 
            # vars. for tracking cumulative fighter rewards: 
            f1_rewards, f2_rewards = [], []
            
            # while this training game is ongoing: 
            while not game_over: 
                
                # TODO - for both fighters:
                    # retrieve their states (s)
                    # select their actions (a)  
                    # execute the actions 
                    # retrieve the following state (sp), rewards (r), game_terminated (t) sentinel
                
                # TODO: 
                # save the two experiences to the replay buffer 
                # append episode rewards to respective reward logs 
                
                # increment the training episodes count by 2 (one for each agent): 
                episode_count += 2
                
                # check if we need to...
                if episode_count % num_episodes_per_copy == 0: 
                    # copy the training weights over to the target network: 
                    self.target_nn.load_state_dict(self.train_nn.state_dict())
                if episode_count % num_episodes_per_rp == 0: 
                    # report the latest average episodes reward: 
                    print("Average reward in last 1000 episodes for fighter 1: {f1_avg}; and fighter 2: {f2_avg}".format(
                        f1_avg = np.mean(f1_rewards), f2_avg = np.mean(f2_rewards)
                    ))
                    # and clear the reward logs:
                    f1_rewards = [], f2_rewards = []
                if episode_count % num_episodes_per_save == 0: 
                    # save the model parameters (just in case of a catastropic failure): 
                    torch.save(self.train_nn.state_dict(),  model_params_save_dir + "train_nn"  + "_" + str(episode_count) + ".params")
                    torch.save(self.target_nn.state_dict(), model_params_save_dir + "target_nn" + "_" + str(episode_count) + ".params")
                    
                # train the NN (if there are sufficient episodes in the replay buffer):
                if len(self.replay_buffer) > replay_batch_size: 
                    s, a, r, sp, t, = self.replay_buffer.sample(replay_batch_size)
                    # TODO: TRAIN THE NN (write train_step function for this?)
                        # if the train_step function, pass in the 'discount_factor' parameter for target calculation!
                
        # report the total time elapsed during training: 
        print("Agent training using {num_games} training games took: {train_time} seconds".format(
            num_games = num_train_games, train_time = round((time.time() - train_time_start), 3)
        ))