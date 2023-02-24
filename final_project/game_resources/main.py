from trainer import Trainer

import torch.nn as nn 

def main():
    trainer = Trainer()
    
    trainer.train(
        lr = 0.0006,
        loss_fn = nn.MSELoss(),
        optimizer_type = "Adam",
        num_train_games = 1000,
        discount_factor = 0.95, 
        softmax_precision = 1,
        replay_batch_size = 32,
    )

if __name__ == '__main__':
    main()