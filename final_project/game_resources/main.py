from trainer import Trainer

import torch
import torch.nn as nn 

def main():
    # take over any available GPUs: 
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    trainer = Trainer(
        device = device, 
        seed = 238,
        load_prev_params = False,
        params_path = NotImplemented
    )
    
    trainer.train(
        lr = 0.0006,
        loss_fn = nn.MSELoss(),
        optimizer_type = "Adam",
        num_train_games = 100,
        game_max_frames = 10000,
        discount_factor = 0.95, 
        softmax_precision = 1e-4,
        softmax_precision_multiplier = 1.1,
        replay_batch_size = 64,
    )

if __name__ == '__main__':
    main()