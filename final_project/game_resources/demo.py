import argparse

import torch 
import numpy as np

import pygame 

from game import Game
from dqn  import DeepQNN

## parse demo 'mode' argument:
argp = argparse.ArgumentParser()
argp.add_argument(
    '--mode', 
    help = "Choose HvH, HvRand, HvAI",
    default = None, 
    type = str
)
argp.add_argument(
    '--params',
    help = "Specify the path to the parameters file for the deep Q-learning model which generates intelligent AI actions.", 
    default = None, 
    type = str    
)
args = argp.parse_args()

## game flow control variables: 
intro_count = 2
last_count_update = pygame.time.get_ticks()
score = [0, 0] # player scores: [P1, P2]
ROUND_OVER_COOLDOWN = 1000 # millseconds 

def main():
    # set fighter types: 
    if args.mode == "HvH":
        fighter_1_type, fighter_2_type = "HUMAN", "HUMAN"
    elif args.mode == "HvRand":
        fighter_1_type, fighter_2_type = "HUMAN", "AI_RANDOM"
    elif args.mode == "HvAI":
        fighter_1_type, fighter_2_type = "HUMAN", "AI_TRAINER"
        # if 'fighter_2_type' == "AI_TRAINER", make sure a path to model parameters was provided: 
        if argp.params is None: 
            raise FileNotFoundError("Please specify a path for the AI action generation model parameters using the \"--params\" argument.")
    else: 
        raise ValueError("Invalid demo mode provided.") 
    
    # take over any available GPUs: 
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    # instantiate a model to evaluate states and generate actions for the AI_TRAINER agent: 
    # NOTE: this model will only be used if '--mode' == "HvAI".
    model = DeepQNN().to(device)
    # load model parameters: 
    if fighter_2_type == "AI_TRAINER":
        model.load_State_dict(torch.load(argp.params))
    
    # create game object: 
    game = Game(fighter_1_type, fighter_2_type)

    # main demo loop and sentinels to control loop breaking, round ending: 
    game_is_running, round_over = True, False
    while game_is_running: 
        if intro_count > 0:
            # lock game framerate:
            game.clock.tick(game.FPS)
            # render fighter healthbars normally handled by 'game.step()':  
            game.draw_health_bar(game.fighter_1.health, 20, 20), game.draw_health_bar(game.fighter_2.health, 580, 20)
            game.draw_text("P1: " + str(game.score[0]), game.score_font, game.COLORS["RED"], 20,  60)
            game.draw_text("P2: " + str(game.score[1]), game.score_font, game.COLORS["RED"], 925, 60)
            # display countdown timer: 
            game.draw_text(str(intro_count), game.count_font, game.COLORS["RED"], 
                           game.SCREEN_WIDTH / 2 - 12, game.SCREEN_HEIGHT / 10)
            # decrement countdown
            if pygame.time.get_ticks() - last_count_update >= 1000: 
                intro_count -= 1
                last_count_update = pygame.time.get_ticks()
        elif intro_count <= 0:      
            # if the round is ongoing: 
            if round_over == False: 
                # if applicable, generate fighter greedy action from model:
                if fighter_2_type == "AI_TRAINER": 
                    qs = model(game.fighter_2.get_state()).cpu().data.numpy()
                    f2_action_idx = np.argmax(qs)
                else: 
                    f2_action_idx = None 
                
                # advance the game according to the registered and/or generated actions: 
                f1_data, f2_data = game.step(round_over, f2_action_idx = f2_action_idx)
                
                # set the 'round_over' sentinel according to the effects of the latest env step: 
                round_over = f1_data[2]
            
            # if the round is over: 
            if round_over == True: 
                # score-keeping: 
                winner = 1 if game.fighter_1.alive == False else 0
                score[winner] += 1
                round_over_time = pygame.time.get_ticks()
                # reset game: 
                if pygame.time.get_ticks() - round_over_time > ROUND_OVER_COOLDOWN: 
                    round_over = False
                    intro_count = 3  # reset the round start countdown
                    game.reset_fighters() # reset the fighters (by re-instantiating them)
            
            # event handler: manual quit ('x' button in top-right corner)
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT:
                    game_is_running = False

if __name__ == "__main__": 
    main() 