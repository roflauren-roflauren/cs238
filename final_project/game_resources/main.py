import pygame
from fighter import Fighter

# initiate pygame for all other parts of program:
pygame.init() 

#######################
###   GAME WINDOW   ###
#######################

# dimensions:
SCREEN_WIDTH = 1000 
SCREEN_HEIGHT = 600

# frame rate:
clock = pygame.time.Clock()
FPS = 120

# colors:
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# define fighter variables: 
KNIGHT_SIZE = 162 # number of pixels in one frame of the spritesheet (assumed to be a square)
KNIGHT_SCALE = 4  # how much to scale up individual images
KNIGHT_OFFSET = [72, 56] # how much to offset the sprites by so they're centered on their underlying rectangles
KNIGHT_DATA = [KNIGHT_SIZE, KNIGHT_SCALE, KNIGHT_OFFSET]

# instantiate window object and window name:
screen = pygame.display.set_mode(size=(SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Mortal (Q)ombat")

# load background image:
bg_image = pygame.image.load("./game_resources/visual_assets/oak_woods_background/collated_forest_bg.png").convert_alpha()

# TO-DO: REPLACE WITH DIFF. SPRITE SHEET

# load spritesheets: 
knight_sheet = pygame.image.load("./game_resources/visual_assets/sample_warrior_spritesheet.png").convert_alpha()

# define number of steps/frames in each animation: 
KNIGHT_ANIMATION_STEPS = [10, 8, 1, 7, 7, 3, 7]

# function which actually draws the background: 
def draw_bg():
    """_summary_
    Utility function which actually draws the background image on the game screen.
    """
    scaled_bg = pygame.transform.scale(bg_image, size=(SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.blit(scaled_bg, dest=(0, 0)) # (0, 0) is top-left corner
    
# function which draws fighter healthbars: 
def draw_health_bar(health, x, y): 
    """_summary_

    Args:
        health (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_d
    """
    # ratio of remaining health (100 is max):
    ratio = health / 100
    pygame.draw.rect(screen, WHITE, (x - 3, y - 3, 406, 36)) # outline of health bars
    pygame.draw.rect(screen, RED, (x, y, 400, 30)) # underlying full (red) health for comparison
    pygame.draw.rect(screen, GREEN, (x, y, 400 * ratio, 30)) # remaining health rect 

#######################
###    GAME VARS    ###
#######################

# create two instances of fighters 
fighter_1 = Fighter(200, 264, False, KNIGHT_DATA, knight_sheet, KNIGHT_ANIMATION_STEPS) 
fighter_2 = Fighter(700, 264, True, KNIGHT_DATA, knight_sheet, KNIGHT_ANIMATION_STEPS)


#######################
###    GAME LOOP    ###
#######################

# sentinel to control game loop flow
game_is_running = True  

while game_is_running:
    # forces frame rate to 'FPS':
    clock.tick(FPS)
    
    # draw background:
    draw_bg()
    
    # show player stats: 
    draw_health_bar(fighter_1.health, 20, 20)
    draw_health_bar(fighter_2.health, 580, 20)
    
    # move fighters according to key presses: 
    fighter_1.move(SCREEN_WIDTH, SCREEN_HEIGHT, screen, fighter_2)
    
    # draw fighters:
    fighter_1.draw(screen)
    fighter_2.draw(screen)

    # event handler: manual quit ('x' button in top-right corner)
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT:
            game_is_running = False
            
    # update display with all changes:
    pygame.display.update()

# exit pygame after game loop terminates: 
pygame.quit