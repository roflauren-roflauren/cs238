from trainer import Trainer

import torch
import torch.nn as nn 

def main():
    # take over any available GPUs: 
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    # instantiate trainer:    
    trainer = Trainer(
        num_hidden_nodes = (34, 26),
        device = device, 
        seed = 238,
        load_prev_params = True,
        params_path = "./model_params/ai_v_rand/target_nn_770_6600000.params"
    )

    # train ai vs. rand!
    trainer.train_aivrand(
        lr                       = 6e-4,
        loss_fn                  = nn.MSELoss(),
        optimizer_type           = "Adam",
        num_train_games          = 1000,
        game_max_frames          = 10000,
        discount_factor          = 0.99, 
        epsilon                  = 0.229, 
        replay_batch_size        = 64,
        num_episodes_per_rp      = 10000,
        num_episodes_per_copy    = 20000, 
        num_episodes_per_save    = 200000,
        model_params_save_dir    = "./model_params/ai_v_rand/",
        reward_save_file         = "./reward_aivrand.txt",
        loss_save_file           = "./loss_aivrand.txt",
        time_save_file           = "./time_aivrand.txt",
        save_file_start_game_num = 771,
        save_file_start_ep_count = 6600000
    )


    # train ai vs. ai!
    trainer.train_aivai(
        lr                       = 6e-4,
        loss_fn                  = nn.MSELoss(),
        optimizer_type           = "Adam",
        num_train_games          = 1000,
        game_max_frames          = 10000,
        discount_factor          = 0.99, 
        epsilon                  = 0.058, 
        replay_batch_size        = 64,
        num_episodes_per_rp      = 10000,
        num_episodes_per_copy    = 20000, 
        num_episodes_per_save    = 200000,
        model_params_save_dir    = "./model_params/ai_v_ai/",
        reward_save_file         = "./reward_aivai.txt",
        loss_save_file           = "./loss_aivai.txt",
        time_save_file           = "./time_aivai.txt",
        save_file_start_game_num = 942,
        save_file_start_ep_count = 13800000
    )

if __name__ == '__main__':
    main()