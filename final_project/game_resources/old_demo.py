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

# define countdown & scoreboard variables: 
intro_count = 2
last_count_update = pygame.time.get_ticks()
score = [0, 0] # player scores. [P1, P2]
round_over = False
ROUND_OVER_COOLDOWN = 1000 # millseconds

# instantiate window object and window name:
screen = pygame.display.set_mode(size=(SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("MORTAL (Q)OMBAT")

# load background image:
bg_image = pygame.image.load("../game_resources/visual_assets/oak_woods_background/collated_forest_bg.png").convert_alpha()

# load spritesheets: 
knight_sheet = pygame.image.load("../game_resources/visual_assets/final_spritesheet.png").convert_alpha()
inverted_knight_sheet = pygame.image.load("../game_resources/visual_assets/spritesheet_inverted.png").convert_alpha()

# define fonts: 
count_font = pygame.font.Font("../game_resources/fonts/turok.ttf", 80)
score_font = pygame.font.Font("../game_resources/fonts/turok.ttf", 30)

# function for drawing text on the game screen: 
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

# function which actually draws the background: 
def draw_bg():
    scaled_bg = pygame.transform.scale(bg_image, size=(SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.blit(scaled_bg, dest=(0, 0)) # (0, 0) is top-left corner
    
# function which draws fighter healthbars: 
def draw_health_bar(health, x, y): 
    # ratio of remaining health (100 is max):
    ratio = health / 100
    pygame.draw.rect(screen, WHITE, (x - 3, y - 3, 406, 36)) # outline of health bars
    pygame.draw.rect(screen, RED, (x, y, 400, 30)) # underlying full (red) health for comparison
    pygame.draw.rect(screen, GREEN, (x, y, 400 * ratio, 30)) # remaining health rect d

#######################
###    GAME VARS    ###
#######################

# create two instances of fighters 
fighter_1 = Fighter(1, "HUMAN", 200, 264, False, knight_sheet) 
fighter_2 = Fighter(2, "HUMAN", 700, 264, True,  inverted_knight_sheet)

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
    
    # show player stats (health and score):
    draw_health_bar(fighter_1.health, 20, 20)
    draw_health_bar(fighter_2.health, 580, 20)
    draw_text("P1: " + str(score[0]), score_font, RED, 20, 60)
    draw_text("P2: " + str(score[1]), score_font, RED, 925, 60)
    
    # if the round countdown is over: 
    if intro_count <= 0: 
        # move fighters according to key presses: 
        fighter_1.move(SCREEN_WIDTH, SCREEN_HEIGHT, fighter_2, round_over)
        fighter_2.move(SCREEN_WIDTH, SCREEN_HEIGHT, fighter_1, round_over)
    else: # summary: decrement round start countdown timer 
        # display countdown timer: 
        draw_text(str(intro_count), count_font, RED, SCREEN_WIDTH / 2 - 12, SCREEN_HEIGHT / 10)
        # decrement countdown
        if pygame.time.get_ticks() - last_count_update >= 1000: 
            intro_count -= 1
            last_count_update = pygame.time.get_ticks()
    
    # process joint dependent actions: 
    fighter_1.process_joint_actions(fighter_2)
    
    # update fighter images/animations: 
    fighter_1.update()
    fighter_2.update()
    
    # draw fighters:
    fighter_1.draw(screen)
    fighter_2.draw(screen)
    
    # check for player defeat: 
    if round_over == False: 
        if fighter_1.alive == False or fighter_2.alive == False: 
            if fighter_1.alive == False: 
                score[1] += 1
            else: 
                score[0] += 1
            round_over = True
            round_over_time = pygame.time.get_ticks()
    else: # round over is true
        if pygame.time.get_ticks() - round_over_time > ROUND_OVER_COOLDOWN: 
            round_over = False
            intro_count = 3
            # reset the fighters (by re-instantiating them): 
            fighter_1 = Fighter(1, "HUMAN", 200, 264, False, knight_sheet) 
            fighter_2 = Fighter(2, "HUMAN", 700, 264, True,  inverted_knight_sheet)

    # event handler: manual quit ('x' button in top-right corner)
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT:
            game_is_running = False
            
    # update display with all changes:
    pygame.display.update()

# exit pygame after game loop terminates: 
pygame.quit()