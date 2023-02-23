import pygame
from fighter import Fighter


class Game(): 
    """ This class represents the brawler game environment. """
    def __init__(self, fighter_1_type: str, fighter_2_type : str):
        """ Constructor for game environment. """
        super().__init__()
        # verify goodness of fighter types: 
        self.fighter_1_type, self.fighter_2_type = fighter_1_type, fighter_2_type
        if self.fighter_1_type not in ["HUMAN", "AI_RANDOM", "AI_TRAINER"] or self.fighter_2_type not in ["HUMAN", "AI_RANDOM", "AI_TRAINER"]: 
            raise ValueError("Please provide \"HUMAN\", \"AI_RANDOM\", or \"AI_TRAINER\", as the fighter types.")
        # initiate pygame: 
        pygame.init() 
        # game window dimensions: 
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 1000, 600 
        # frame rate control: 
        self.clock, self.FPS = pygame.time.Clock(), 120 
        # colors: 
        self.COLORS = {
            "RED" : (255, 0, 0),       "YELLOW" : (255, 255, 0),
            "WHITE" : (255, 255, 255), "GREEN" : (0, 255, 0)
        }
        # game widow object:
        self.screen = pygame.display.set_mode(size=(self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("MORTAL (Q)OMBAT")
        # background image: 
        self.bg_image = pygame.image.load("./game_resources/visual_assets/oak_woods_background/collated_forest_bg.png").convert_alpha()
        # spritesheets: 
        self.knight_sheet     = pygame.image.load("./game_resources/visual_assets/final_spritesheet.png").convert_alpha()
        self.inv_knight_sheet = pygame.image.load("./game_resources/visual_assets/spritesheet_inverted.png").convert_alpha()
        # fonts: 
        self.count_font = pygame.font.Font("./game_resources/fonts/turok.ttf", 80)
        self.score_font = pygame.font.Font("./game_resources/fonts/turok.ttf", 30)
        # fighter variables: 
        self.fighter_1 = Fighter(1, self.fighter_1_type, 200, 264, False, self.knight_sheet)
        self.fighter_2 = Fighter(2, self.fighter_2_type, 700, 264, True,  self.inv_knight_sheet)
    
    def __del__(self): 
        """ Destructor for game environment. """
        # exit pygame: 
        pygame.quit()
        
    def reset_fighters(self): 
        """ Reset the fighters for another round. """
        # reset the fighters (by re-instantiating them):
        self.fighter_1 = Fighter(1, self.fighter_1_type, 200, 264, False, self.knight_sheet)
        self.fighter_2 = Fighter(2, self.fighter_2_type, 700, 264, True,  self.inv_knight_sheet)
    
    def draw_text(self, text, font, text_col, x, y): 
        """ Draws text on the screen. """
        img = font.render(text, True, text_col)
        self.screen.blit(img, (x, y))
        
    def draw_bg(self):
        """ Draws the gameplay background on the screen. """
        scaled_bg = pygame.transform.scale(self.bg_image, size=(self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.screen.blit(scaled_bg, dest=(0, 0)) # (0, 0) is top-left corner
    
    def draw_health_bar(self, health, x, y):
        """ Draws the character health bars on the screen. """ 
        # ratio of remaining health (100 is max):
        ratio = health / 100
        pygame.draw.rect(self.screen, self.COLORS["WHITE"], (x - 3, y - 3, 406, 36)) # outline of health bars
        pygame.draw.rect(self.screen, self. COLORS["RED"], (x, y, 400, 30))          # underlying full (red) health bar for comp.
        pygame.draw.rect(self.screen, self.COLORS["GREEN"], (x, y, 400 * ratio, 30)) # remaining health bar
        
    def step(self, 
        prv_rd_is_end : bool, 
        f1_action_idx : int = None, 
        f2_action_idx : int = None 
    ): 
        """ Accepts fighter actions and advances the environment after conducting them."""
        # force frame rate to self.FPS: 
        self.clock.tick(self.FPS)
        # draw background 
        self.draw_bg()
        
        # display player states (health and score):
        self.draw_health_bar(self.fighter_1.health, 20, 20)
        self.draw_health_bar(self.fighter_2.health, 580, 20)
        self.draw_text("P1: " + str(self.score[0]), self.score_font, self.COLORS["RED"], 20,  60)
        self.draw_text("P2: " + str(self.score[1]), self.score_font, self.COLORS["RED"], 925, 60)
        
        # move fighters according to received actions:
        self.fighter_1.move(
            self.SCREEN_WIDTH, self.SCREEN_HEIGHT, 
            self.fighter_2, 
            prv_rd_is_end, 
            f1_action_idx
        )
        self.fighter_2.move(
            self.SCREEN_WIDTH, self.SCREEN_HEIGHT, 
            self.fighter_1,
            prv_rd_is_end, 
            f2_action_idx
        )     

        # process joint dependent actions:
        self.fighter_1.process_joint_actions(self.fighter_2)
        # update fighter images and animations:
        self.fighter_1.update(), self.fighter_2.update()
        # draw fighters:
        self.fighter_1.draw(self.screen), self.fighter_2.draw(self.screen)

        # update display with all changes: 
        pygame.display.update()
        
        # if the round is over (i.e., one of the fighters is dead):
        if self.fighter_1.alive == False or self.fighter_2 == False: 
            f1_terminal, f2_terminal = True, True 
            f1_reward = 10000 if self.fighter_1.alive == True else -10000
            f2_reward = 10000 if self.fighter_2.alive == True else -10000     
        else: 
            f1_terminal, f2_terminal = False, False 
            f1_reward = self.fighter_1.health - self.fighter_2.health
            f2_reward = self.fighter_2.health - self.fighter_1.health
        # retrieve the (current) next state for each fighter: 
        f1_next_state = self.fighter_1.get_state(self.fighter_2)
        f2_next_state = self.fighter_2.get_state(self.fighter_1) 
                
        return [(f1_next_state, f1_reward, f1_terminal), 
                (f2_next_state, f2_reward, f2_terminal)]