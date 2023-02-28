from trainer import Trainer

import torch
import torch.nn as nn 

def main():
    # take over any available GPUs: 
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    trainer = Trainer(
        num_hidden_nodes = (34, 25),
        device = device, 
        seed = 238,
        load_prev_params = True,
        params_path = "./model_params/target_nn_1193_16300000.params"
    )
    
    trainer.train(
        lr = 6e-4,
        loss_fn = nn.MSELoss(),
        optimizer_type = "Adam",
        num_train_games = 10000,
        game_max_frames = 10000,
        discount_factor = 0.95, 
        epsilon = 0.05, 
        replay_batch_size = 64,
        num_episodes_per_rp = 10000,
        num_episodes_per_copy = 20000, 
        num_episodes_per_save = 100000,
        reward_save_file = "./reward.txt",
        loss_save_file = "./loss.txt",
        save_file_start_game_num = 1193,
        save_file_start_ep_count = 16300000
    )

if __name__ == '__main__':
    main()