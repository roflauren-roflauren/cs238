import time 

import numpy  as np
import pandas as pd 
import random as rd 
import pygame as pg

import torch 
import torch.nn as nn 
import torch.nn.functional as F

# custom classes:
import dqn 
from   game import Game

class Trainer():
    """ A class for training the gameplay agent. """
    def __init__(self, 
        device : str = 'cpu', 
        load_prev_params : bool = False, 
        params_path : str = None
    ):
        """ Initializes the Trainer object. """
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
    
    
    def stable_softmax(q_vals, softmax_precision):
        # TODO: IMPLEMENT ME!
        pass
    
    def train(self, 
        lr = 6e-4, 
        loss_fn = nn.MSELoss(),
        optimizer_type = "Adam", 
        num_train_games = int(1e6),
        discount_factor = 0.95,
        softmax_precision = 1, 
        replay_batch_size = 32,
        num_episodes_per_rp   = 1000,
        num_episodes_per_copy = 2000, 
        num_episodes_per_save = 10000,
        model_params_save_dir = "./model_params/"
    ):
        """ This function actually performs the NN training. """ 
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
        
        # instantiate a game object: 
        training_game = Game("AI_TRAINER", "AI_TRAINER")
        
        # for each training game: 
        for game_idx in num_train_games: 
            # reset the game environment and Trainer's replay buffer:
            training_game.reset_fighters(), self.replay_buffer.reset() 
        
            # sentinel which captures whether the current training game is ongoing or over: 
            game_over = False 
            # vars. for tracking cumulative fighter rewards: 
            f1_rewards, f2_rewards = [], []
            
            # while this training game is ongoing: 
            while not game_over: 
                ## STATES: 
                # retrieve the state for each fighter (as numpy ndarray): 
                f1_state_np, f2_state_np = training_game.fighter_1.get_state(training_game.fighter_2), \
                    training_game.fighter_2.get_state(training_game.fighter_1)
                # convert these states into Torch tensors: 
                f1_state, f2_state = torch.from_numpy(f1_state_np).to(self.device), \
                    torch.from_numpy(f2_state_np).to(self.device)
                
                ## ACTIONS: 
                # generate q-values of actions for each fighter: 
                f1_qs, f2_qs = self.target_nn(f1_state), self.target_nn(f2_state)
                # apply the stable softmax to generate an action for each fighter: 
                f1_action_idx, f2_action_idx = \
                    self.stable_softmax(q_vals=f1_qs.cpu().data.numpy(), softmax_precision=softmax_precision), \
                    self.stable_softmax(q_vals=f2_qs.cpu().data.numpy(), softmax_precision=softmax_precision) 
                
                ## REWARDS, NEXT STATE, GAME_TERMINATED: 
                # step the game environment using the sample actions: 
                [f1_data, f2_data] = training_game.step(
                    prv_rd_is_end=game_over,
                    f1_action_idx=f1_action_idx,
                    f2_action_idx=f2_action_idx
                )
                # unpack each fighter's data
                f1_next_state, f1_reward, f1_terminal = f1_data[0], f1_data[1], f1_data[2]
                f2_next_state, f2_reward, f2_terminal = f2_data[0], f2_data[1], f2_data[2]
                
                ## REPLAY BUFFER AND REWARD LOGGING: 
                # save the each fighter's experience to the replay buffer: 
                self.replay_buffer.add(f1_state_np, f1_action_idx, f1_reward, f1_next_state, f1_terminal)
                self.replay_buffer.add(f2_state_np, f2_action_idx, f2_reward, f2_next_state, f2_terminal)
                # append episode rewards to respective fighter's reward logs: 
                f1_rewards.append(f1_reward), f2_rewards.append(f2_reward)
                
                ## TRAINING BOOKKEEPING:
                # increment the training episodes count by 2 (one for each agent): 
                episode_count += 2
                
                # check if we need to...
                if episode_count % num_episodes_per_copy == 0: 
                    # copy the training weights over to the target network: 
                    self.target_nn.load_state_dict(self.train_nn.state_dict())
                if episode_count % num_episodes_per_rp == 0: 
                    # report the latest average episodes reward: 
                    print("Average reward in last {num_episodes_per_rp} episodes for fighter 1: {f1_avg}; and fighter 2: {f2_avg}".format(
                        num_episodes_per_rp = num_episodes_per_rp, f1_avg = np.mean(f1_rewards), f2_avg = np.mean(f2_rewards)
                    ))
                    # and clear the reward logs:
                    f1_rewards = [], f2_rewards = []
                if episode_count % num_episodes_per_save == 0: 
                    # save the model parameters (just in case of a catastropic failure): 
                    torch.save(self.train_nn.state_dict(),  model_params_save_dir + "train_nn"  + "_" + str(episode_count) + ".params")
                    torch.save(self.target_nn.state_dict(), model_params_save_dir + "target_nn" + "_" + str(episode_count) + ".params")

                ## NEURAL NETWORK TRAINING:
                # if there are sufficient episodes in the replay buffer:
                if len(self.replay_buffer) > replay_batch_size: 
                    # sample s, a, r, sp, t lists from the replay buffer:
                    s, a, r, sp, t, = self.replay_buffer.sample(replay_batch_size)
                    # TODO: TRAIN THE NN (write train_step function for this?)
                        # if the train_step function, pass in the 'discount_factor' parameter for target calculation!
        
        # kill the game: 
        del training_game
        
        # report the total time elapsed during training: 
        print("Agent training using {num_games} training games took: {train_time} seconds".format(
            num_games = num_train_games, train_time = round((time.time() - train_time_start), 3)
        ))