from trainer import Trainer

import torch
import torch.nn as nn 

def main():
    # take over any available GPUs: 
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    trainer = Trainer(
        num_hidden_nodes = (34, 26),
        device = device, 
        seed = 238,
        load_prev_params = False,
        params_path = None
    )
    
    trainer.train(
        lr = 6e-4,
        loss_fn = nn.MSELoss(),
        optimizer_type = "Adam",
        num_train_games = 1000,
        game_max_frames = 10000,
        discount_factor = 0.95, 
        epsilon = 1.0, 
        replay_batch_size = 64,
        num_episodes_per_rp = 5000,
        num_episodes_per_copy = 20000, 
        num_episodes_per_save = 50000,
        reward_save_file = "./reward.txt",
        loss_save_file = "./loss.txt",
        save_file_start_game_num = 0,
        save_file_start_ep_count = 0
    )

if __name__ == '__main__':
    main()