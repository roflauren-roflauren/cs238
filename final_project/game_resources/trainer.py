import time 

import numpy  as np
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
        num_hidden_nodes : tuple = (34, 26),
        device : str = 'cpu', 
        seed : int  = 238, 
        load_prev_params : bool = False, 
        params_path : str = None
    ):
        """ Initializes the Trainer object. """
        super().__init__()
        # register the random seed to be used for training:
        self.seed = rd.seed(seed)
        # register the device to be used for training: 
        self.device = device 
        # initialize the training and target NN on the desired device: 
        self.train_nn  = dqn.DeepQNN(num_hidden_nodes[0], num_hidden_nodes[1]).to(self.device)
        self.target_nn = dqn.DeepQNN(num_hidden_nodes[0], num_hidden_nodes[1]).to(self.device)
        # check if there are any parameter files to initialize our models from:
        if load_prev_params == True: 
            if params_path is None: 
                raise ValueError("Please provide a path to the file containing the network parameters you wish to load.")
            else: 
                self.train_nn.load_state_dict(torch.load(params_path))
                self.target_nn.load_state_dict(torch.load(params_path))
        # initialize a experience replay buffer: 
        self.replay_buffer = dqn.Buffer(device = self.device, seed = self.seed)
 
    
    ## NOTE: stable_softmax() CURRENTLY *NOT* IN USE.
    def stable_softmax(self, q_vals, softmax_precision):
        """ A utility function which performs the stable softmax operation and samples an index from the resulting probability vector."""
        # apply the stable softmax to convert Q-values to probs.:
        probs = (
            np.exp(softmax_precision * (q_vals - np.max(q_vals))) / 
            np.exp(softmax_precision * (q_vals - np.max(q_vals))).sum()
        )
        # error guarding:  
        has_nan_prob = np.isnan(probs).any()
        # if the softmax produces a 'nan' prob, select a random action with uniform prob:
        if has_nan_prob == True:  probs = np.full(probs.shape, 1 / len(probs))
        # enforce random seed to ensure replicability across training trials: 
        rd.seed(self.seed)
        return np.random.choice(len(probs), p=probs) 
  
    
    ## NOTE: epsilon_greedy() CURRENTLY IN USE.
    def epsilon_greedy(self, q_vals, epsilon): 
        # initialize the ret. value (an action index):
        action_idx = 0
        # sample a probability: 
        prob = np.random.uniform()
        # use the epsilon-greedy action selection approach:
        if prob < epsilon: 
            action_idx = np.random.choice(len(q_vals))
        else:
            action_idx = np.argmax(q_vals)
        return action_idx
   
    
    def q_update(self, s, a, r, sp, t, discount_factor): 
        # compute the Q-values for each action, for each episode: 
        q_vals = self.train_nn(s)
        # compute the maximum Q-vals for the next states: 
        max_sp_qs = self.target_nn(sp).max(-1).values
        # compute the Q-update target value: 
        targets = r + (1 - t) * discount_factor * max_sp_qs
        # update the episode Q-values to only retain those corresponding to the executed action: 
        action_masks = F.one_hot(a, num_classes = 16)
        q_vals = (action_masks * q_vals).sum(-1) # collapse the tensor for scalar comparisons 
        # compute loss and backprop! 
        loss = self.loss_fn(q_vals, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # return the loss for logging purposes: 
        return loss
    
    
    def train_aivai(self, 
        lr = 6e-4, 
        loss_fn = nn.MSELoss(),
        optimizer_type = "Adam", 
        num_train_games = int(1e6),
        game_max_frames = 10000,
        discount_factor = 0.95,
        epsilon = 1.0,
        replay_batch_size = 64,
        num_episodes_per_rp   = 5000,
        num_episodes_per_copy = 20000, 
        num_episodes_per_save = 50000,
        model_params_save_dir = "./model_params/", 
        reward_save_file = "./reward.txt",
        loss_save_file   = "./loss.txt",
        time_save_file   = "./time.txt",
        save_file_start_game_num : int = None, 
        save_file_start_ep_count : int = None
    ):
        """ This function actually performs the NN training for an AI agent trained versus itself. """ 
        # initialize the optimizer: 
        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.train_nn.parameters(), lr = lr)
        elif optimizer_type == "SGD": 
            self.optimizer = torch.optim.SGD(self.train_nn.parameters(), lr = lr)
        else: 
            raise ValueError("Only \"Adam\" or \"SGD\" are accepted as supported optimizer types currently.")
        # initialize the loss function: 
        self.loss_fn = loss_fn
        
        # track total episode count: 
        episode_count = 0
           
        # add the prior episode count if we're continuing training from a prior session: 
        episode_count = episode_count + save_file_start_ep_count \
            if save_file_start_ep_count is not None else episode_count
        
        # extract the epsilon parameter into a local var. for updates: 
        e = epsilon
        
        # decrement the number of already-trained games from the num_train_games: 
        if save_file_start_game_num is not None:
            num_games = num_train_games - save_file_start_game_num
        else:
            num_games = num_train_games
        
        ## MAIN TRAINING LOOP STARTS HERE: 
        # for each training game: 
        for game_idx in range(num_games): 
            # log the training game start time: 
            game_time_start = time.time()
            
            # report which training game this is: 
            print("Training with game number {game_num}".format(
                game_num = game_idx + save_file_start_game_num if save_file_start_game_num is not None else game_idx))
            
            # (re-)instantiate the game environment:
            training_game = Game("AI_TRAINER", "AI_TRAINER", game_max_frames = game_max_frames) 
            
            # reset the experience replay buffer:
            self.replay_buffer.reset() 
        
            # sentinel which captures whether the current training game is ongoing or over: 
            game_over = False 
            # vars. for tracking cumulative fighter rewards: 
            f1_rewards, f2_rewards = [], []
            
            # var. for tracking NN loss: 
            loss_log = []
            
            # while this training game is ongoing: 
            while not game_over: 
                ## STATES: 
                ## tell fighters what frame they're in (from game class variable): 
                training_game.fighter_1.receive_frame_info(training_game.frame)
                training_game.fighter_2.receive_frame_info(training_game.frame)
                
                # retrieve the state for each fighter (as numpy ndarray): 
                f1_state_np, f2_state_np = training_game.fighter_1.get_state(training_game.fighter_2), \
                    training_game.fighter_2.get_state(training_game.fighter_1)
                                    
                # convert these states into Torch tensors: 
                f1_state, f2_state = torch.from_numpy(f1_state_np).to(self.device), \
                    torch.from_numpy(f2_state_np).to(self.device)
                
                ## ACTIONS: 
                # generate q-values of actions for each fighter: 
                f1_qs, f2_qs = self.target_nn(f1_state), self.target_nn(f2_state)
                
                # apply epsilon greedy to generate an action for each fighter: 
                f1_action_idx, f2_action_idx = \
                    self.epsilon_greedy(q_vals = f1_qs.cpu().data.numpy(), epsilon = e), \
                    self.epsilon_greedy(q_vals = f2_qs.cpu().data.numpy(), epsilon = e) 
                
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
                # retrieve the game_over sentinel from the terminal condition assessed by fighters: 
                game_over = f1_terminal # f2_terminal could work too, doesn't matter! 
            
                ## REPLAY BUFFER AND REWARD LOGGING: 
                # save the each fighter's experience to the replay buffer: 
                self.replay_buffer.add(f1_state_np, f1_action_idx, f1_reward, f1_next_state, f1_terminal)
                self.replay_buffer.add(f2_state_np, f2_action_idx, f2_reward, f2_next_state, f2_terminal)
                # append episode rewards to respective fighter's reward logs: 
                f1_rewards.append(f1_reward), f2_rewards.append(f2_reward)
                
                ## TRAINING BOOKKEEPING:
                # increment the training episodes count by 2 (one for each agent): 
                episode_count += 2
                
                ## CHECK IF WE NEED TO...
                # copy the training weights over to the target network: 
                if episode_count % num_episodes_per_copy == 0: 
                    # info message:
                    print("COPYING WEIGHTS FROM TRAIN NN TO TARGET NN...") 
                    self.target_nn.load_state_dict(self.train_nn.state_dict())
                    
                # report the latest average episodes reward and loss: 
                if episode_count % num_episodes_per_rp == 0: 
                    print("Average reward in last {num_episodes_per_rp} episodes for fighters 1, 2: {f1_avg}, {f2_avg}".format(
                        num_episodes_per_rp = num_episodes_per_rp, f1_avg = np.mean(f1_rewards), f2_avg = np.mean(f2_rewards)
                    ))
                    print("Average loss in last {num_episodes_per_rp} episodes: {avg_loss}".format(
                        num_episodes_per_rp = num_episodes_per_rp, avg_loss = np.mean(loss_log)
                    ))
                    # write these figures to file: 
                    with open(reward_save_file, 'a+') as f: 
                        for line in f:
                            if line.isspace(): break
                        f.write(str(np.mean(f1_rewards)) + '\n')
                        f.write(str(np.mean(f2_rewards)) + '\n')
                            
                        
                    with open(loss_save_file, 'a+') as f: 
                        for line in f:
                            if line.isspace(): break
                        f.write(str(np.mean(loss_log)) + '\n')
                                
                    # if here, don't forget to clear the reward and loss logs:
                    f1_rewards, f2_rewards, loss_log = [], [], []
                    
                # save the model parameters (just in case of a catastropic failure): 
                if episode_count % num_episodes_per_save == 0: 
                    # add the prior game count if we're continuing training from a prior session: 
                    game_count = game_idx + save_file_start_game_num \
                        if save_file_start_game_num is not None else game_idx
                    # info message:
                    print("SAVING MODEL PARAMETERS...")
                    torch.save(
                        self.target_nn.state_dict(), 
                        model_params_save_dir + "target_nn" + "_" \
                            + str(game_count) + "_" + str(episode_count) + ".params"
                        )
                    
                ## NEURAL NETWORK TRAINING:
                # if there are sufficient episodes in the replay buffer:
                if len(self.replay_buffer) > replay_batch_size: 
                    # sample s, a, r, sp, t lists from the replay buffer:
                    s, a, r, sp, t, = self.replay_buffer.sample(replay_batch_size)
                    # conduct the Q-update! 
                    loss = self.q_update(s, a, r, sp, t, discount_factor)
                    # log loss: 
                    loss_log.append(loss.cpu().data.numpy())
            
            # log the time for this training game: 
            game_time_end = time.time()
            with open(time_save_file, 'a+') as f: 
                        for line in f:
                            if line.isspace(): break
                        f.write(str(round(game_time_end - game_time_start, 3)) + '\n')
            
            # decay epsilon for next game if above some threshold: 
            if e > 0.05: e -= 0.001
        
        # save model parameters one last time after all training games are done! 
        game_count = game_idx + save_file_start_game_num \
                        if save_file_start_game_num is not None else game_idx
        # info message:
        print("SAVING MODEL PARAMETERS...")
        torch.save(
            self.target_nn.state_dict(), 
            model_params_save_dir + "target_nn" + "_" \
                + str(game_count) + "_" + str(episode_count) + ".params"
            )
        
        # quit pygame once all training games are done: 
        pg.quit()
    
    
    def train_aivrand(self, 
        lr = 6e-4, 
        loss_fn = nn.MSELoss(),
        optimizer_type = "Adam", 
        num_train_games = 1000,
        game_max_frames = 10000,
        discount_factor = 0.99,
        epsilon = 1.0,
        replay_batch_size = 64,
        num_episodes_per_rp   = 10000,
        num_episodes_per_copy = 20000, 
        num_episodes_per_save = 200000,
        model_params_save_dir = "./model_params/ai_v_rand/", 
        reward_save_file = "./reward_aivrand.txt",
        loss_save_file   = "./loss_aivrand.txt",
        time_save_file   = "./time_aivrand.txt",
        save_file_start_game_num : int = None, 
        save_file_start_ep_count : int = None
    ):
        """ This function actually performs the NN trained versus a random bot. """ 
        # initialize the optimizer: 
        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.train_nn.parameters(), lr = lr)
        elif optimizer_type == "SGD": 
            self.optimizer = torch.optim.SGD(self.train_nn.parameters(), lr = lr)
        else: 
            raise ValueError("Only \"Adam\" or \"SGD\" are accepted as supported optimizer types currently.")
        # initialize the loss function: 
        self.loss_fn = loss_fn
        
        # track total episode count: 
        episode_count = 0
           
        # add the prior episode count if we're continuing training from a prior session: 
        episode_count = episode_count + save_file_start_ep_count \
            if save_file_start_ep_count is not None else episode_count
        
        # extract the epsilon parameter into a local var. for updates: 
        e = epsilon
        
        # decrement the number of already-trained games from the num_train_games: 
        if save_file_start_game_num is not None:
            num_games = num_train_games - save_file_start_game_num
        else:
            num_games = num_train_games
        
        ## MAIN TRAINING LOOP STARTS HERE: 
        # for each training game: 
        for game_idx in range(num_games): 
            # log the training game start time: 
            game_time_start = time.time()
            
            # report which training game this is: 
            print("Training with game number {game_num}".format(
                game_num = game_idx + save_file_start_game_num if save_file_start_game_num is not None else game_idx))
            
            # (re-)instantiate the game environment:
            training_game = Game("AI_TRAINER", "AI_RANDOM", game_max_frames = game_max_frames) 
            
            # reset the experience replay buffer:
            self.replay_buffer.reset() 
        
            # sentinel which captures whether the current training game is ongoing or over: 
            game_over = False 
            # var. for tracking cumulative fighter reward: 
            f1_rewards = []
            # var. for tracking NN loss: 
            loss_log = []
            
            # while this training game is ongoing: 
            while not game_over: 
                ## STATES: 
                ## tell AI fighter what frame they're in (from game class variable): 
                training_game.fighter_1.receive_frame_info(training_game.frame)
                
                # retrieve the state for AI fighter (as numpy ndarray): 
                f1_state_np = training_game.fighter_1.get_state(training_game.fighter_2)
                
                # convert state into Torch tensors: 
                f1_state = torch.from_numpy(f1_state_np).to(self.device)
                
                ## ACTIONS: 
                # generate q-values of actions for AI fighter: 
                f1_qs = self.target_nn(f1_state)
                
                # apply epsilon greedy to generate an action for each fighter: 
                f1_action_idx = self.epsilon_greedy(q_vals = f1_qs.cpu().data.numpy(), epsilon = e)
                
                ## REWARDS, NEXT STATE, GAME_TERMINATED: 
                # step the game environment using the sample action: 
                [ f1_data, _ ] = training_game.step(
                    prv_rd_is_end=game_over,
                    f1_action_idx=f1_action_idx
                )
                # unpack fighter data:
                f1_next_state, f1_reward, f1_terminal = f1_data[0], f1_data[1], f1_data[2]

                # retrieve the game_over sentinel from the terminal condition assessed by fighters: 
                game_over = f1_terminal # f2_terminal could work too, doesn't matter! 
            
                ## REPLAY BUFFER AND REWARD LOGGING: 
                # save the each fighter's experience to the replay buffer: 
                self.replay_buffer.add(f1_state_np, f1_action_idx, f1_reward, f1_next_state, f1_terminal)

                # append episode rewards to respective fighter's reward logs: 
                f1_rewards.append(f1_reward)
                
                ## TRAINING BOOKKEEPING:
                # increment the training episodes count by 1 (only one since other agent is random): 
                episode_count += 1
                
                ## CHECK IF WE NEED TO...
                # copy the training weights over to the target network: 
                if episode_count % num_episodes_per_copy == 0: 
                    # info message:
                    print("COPYING WEIGHTS FROM TRAIN NN TO TARGET NN...") 
                    self.target_nn.load_state_dict(self.train_nn.state_dict())
                    
                # report the latest average episodes reward and loss: 
                if episode_count % num_episodes_per_rp == 0: 
                    print("Average reward in last {num_episodes_per_rp} episodes for fighter 1: {f1_avg}".format(
                        num_episodes_per_rp = num_episodes_per_rp, f1_avg = np.mean(f1_rewards)
                    ))
                    print("Average loss in last {num_episodes_per_rp} episodes: {avg_loss}".format(
                        num_episodes_per_rp = num_episodes_per_rp, avg_loss = np.mean(loss_log)
                    ))
                    # write these figures to file: 
                    with open(reward_save_file, 'a+') as f: 
                        for line in f:
                            if line.isspace(): break
                        f.write(str(np.mean(f1_rewards)) + '\n')
                        
                    with open(loss_save_file, 'a+') as f: 
                        for line in f:
                            if line.isspace(): break
                        f.write(str(np.mean(loss_log)) + '\n')
                                
                    # if here, don't forget to clear the reward and loss logs:
                    f1_rewards, loss_log = [], []
                    
                # save the model parameters (just in case of a catastropic failure): 
                if episode_count % num_episodes_per_save == 0: 
                    # add the prior game count if we're continuing training from a prior session: 
                    game_count = game_idx + save_file_start_game_num \
                        if save_file_start_game_num is not None else game_idx
                    # info message:
                    print("SAVING MODEL PARAMETERS...")
                    torch.save(
                        self.target_nn.state_dict(), 
                        model_params_save_dir + "target_nn" + "_" \
                            + str(game_count) + "_" + str(episode_count) + ".params"
                        )
                    
                ## NEURAL NETWORK TRAINING:
                # if there are sufficient episodes in the replay buffer:
                if len(self.replay_buffer) > replay_batch_size: 
                    # sample s, a, r, sp, t lists from the replay buffer:
                    s, a, r, sp, t, = self.replay_buffer.sample(replay_batch_size)
                    # conduct the Q-update! 
                    loss = self.q_update(s, a, r, sp, t, discount_factor)
                    # log loss: 
                    loss_log.append(loss.cpu().data.numpy())
            
            # log the time for this training game: 
            game_time_end = time.time()
            with open(time_save_file, 'a+') as f: 
                        for line in f:
                            if line.isspace(): break
                        f.write(str(round(game_time_end - game_time_start, 3)) + '\n')
            
            # decay epsilon for next game if above some threshold: 
            if e > 0.05: e -= 0.001
        
        # save model parameters one last time after all training games are done! 
        game_count = game_idx + save_file_start_game_num \
                        if save_file_start_game_num is not None else game_idx
        # info message:
        print("SAVING MODEL PARAMETERS...")
        torch.save(
            self.target_nn.state_dict(), 
            model_params_save_dir + "target_nn" + "_" \
                + str(game_count) + "_" + str(episode_count) + ".params"
            )
        
        # quit pygame once all training games are done: 
        pg.quit()