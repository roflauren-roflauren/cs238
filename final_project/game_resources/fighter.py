import pygame
import random as rd

CHAR_WIDTH  = 80
CHAR_HEIGHT = 180

class Fighter():
    def __init__(self, player, player_type, x, y, start_flip, data, sprite_sheet, animation_steps):
        """_summary_

        Args:
            x (int): left-coord where fighter will be instantiated. 
            y (int): top-coord where fighter will be instantiated. 
        """
        # player 1 or player 2: 
        self.player = player
        
        # controller type (human or AI): 
        self.player_type = player_type
        # key press idxs for random AI input generation: 
        self.keypress_idxs = {
            "movement" : [(4, 80), (7, 79), (20, 82)],    # pygame.key.get_pressed() ScancodeWrapper()'s idxs for: [(a, l_arrow), (d, r_arrow), (w, up_arrow)]
            "weapon"   : [(13, 89), (14, 90), (15, 91)]   #                                                        [(j, KP1),     (k, KP2),     (l, KP3)     ]                                 
        }
        
        # retrieve the character size, scaling factor, and offset:
        self.size = data[0]
        self.image_scale = data[1]
        self.offset = data[2]
        
        # ensures that character faces target:
        self.flip = start_flip
        
        # list of sprite images for different actions
        self.animation_list = self.load_images(sprite_sheet, animation_steps)
        
        # maintains what action the character is doing 
        # 0: idle, 1: run, 2: jump, 3: attack1, 4: attack2, 5: hit, 6: death
        self.action = 0
        
        # cooldown on any action (for tracking stun lock penalty from successful enemy parry):
        self.action_cooldown = 0
        
        # animation variables: 
        self.frame_index = 0 # which frame of the animation we're on
        self.image = self.animation_list[self.action][self.frame_index]
        self.update_time = pygame.time.get_ticks()
        
        # player position as represented by a rectangle:
        self.rect = pygame.Rect(
            x, y, CHAR_WIDTH, CHAR_HEIGHT
        )
        
        # y-velocity: 
        self.vel_y = 0
        
        # track whether character is moving L/R or not: 
        self.running = False
        
        # track whether character is jumping or not: 
        self.jump = False
        
        # track whether character is attacking or not: 
        self.attacking = False
        
        # attack type during the current move: 
        self.attack_type = 0 # valid values: 0 (inactive), 1, 2
        
        # cooldown on attacking (prevents endless attacking):
        self.attack_cooldown = 0
        
        # character's attack rect (used to check for an effective parry by enemy):
        self.attacking_rect = pygame.Rect(0,0,0,0)
        
        # track whether character is parrying or not: 
        self.parrying = False 
        
        # cooldown on parrying (prevents endless parrying):
        self.parry_cooldown = 0
        
        # character's parrying rect (used to check for an effective parry against enemy):
        self.parrying_rect = pygame.Rect(0,0,0,0)
                
        # track whether character has been hit: 
        self.hit = False
        
        # health pool: 
        self.health = 100
        
        # tracks dead or alive: 
        self.alive = True
        
    
    def load_images(self, sprite_sheet, animation_steps):
        """_summary_

        Args:
            sprite_sheet (_type_): _description_
            animation_steps (_type_): _description_
        """
        # extract images from spritesheet:
        animation_list = []
        for y, animation in enumerate(animation_steps):
            temp_img_list = []
            for x in range(animation):
                temp_img = sprite_sheet.subsurface(x * self.size, y * self.size, self.size, self.size) 
                # scale the image before appending to the animation list:
                temp_img_list.append(
                    pygame.transform.scale(temp_img, 
                                           (self.size * self.image_scale, self.size * self.image_scale)))
            animation_list.append(temp_img_list)
            
        return animation_list
            
    def move(self, screen_width, screen_height, target, round_over):
        """_summary_

        Args:
            screen_width (_type_): _description_
            screen_height (_type_): _description_
        """
        SPEED = 10     # controls how quickly characters are moving on the screen
        GRAVITY = 2    # counters jumping & ensures player doesn't fly off screen
        dx, dy = 0, 0  # change in x, y coordinate of character
        
        # reset key presses/character condition: 
        self.running = False
        self.attack_type = 0
        
        # get keypresses:
        key = pygame.key.get_pressed()
        # if player is human, just collect provided keypresses:
        if self.player_type == "HUMAN":
            key = pygame.key.get_pressed()  
        # if player is AI (self.player_type == "AI_X"), follow policy to gen. keypresses:
        else: 
            if self.player_type == "AI_RANDOM": 
                # list with bool for keypresses; will be 'False' for every keyboard input since agent is AI:
                l_key = list(pygame.key.get_pressed())
                
                # generate random inputs for movement and weapon action (both inputs have an optional, null value; i.e., -1): 
                movement_input, weapon_input = rd.randint(-1, 2), rd.randint(-1, 2)
                
                # map random ints to keypresses and update 'key' (i.e., keypress dictionary):
                if movement_input != -1: 
                    movement_key = self.keypress_idxs["movement"][movement_input][self.player - 1]
                    l_key[movement_key] = True
                if weapon_input != -1: 
                    weapon_key = self.keypress_idxs["weapon"][weapon_input][self.player - 1]
                    l_key[weapon_key] = True
                    
                # convert the keypress bool list back to pygame's ScancodeWrapper type:
                key = pygame.key.ScancodeWrapper(l_key)
                
            else:
                # intelligent AI decision making to go here! 
                pass  
        
        # move update notes: 
        # can only perform movement actions if:
        #   - not currently attacking;
        #   - not currently parrying; 
        #   - not stun-locked;
        #   - not dead;
        #   - round isn't over & resetting: 
        if self.attacking == False and self.parrying == False and self.action_cooldown == 0 and self.alive == True and round_over == False:
            
            # check player 1 controls: 
            if self.player == 1: 
                # movement keypresses: 
                if key[pygame.K_a]: # left
                    dx = -SPEED 
                    self.running = True
                if key[pygame.K_d]: # right 
                    dx = SPEED 
                    self.running = True
                if key[pygame.K_w] and self.jump == False : # jump, 2nd cond. ensures no double-jump allowed! 
                    self.vel_y = -35
                    self.jump = True
                    
                if key[pygame.K_j] or key[pygame.K_k]: 
                    # determine attack type: 
                    if key[pygame.K_j]: 
                        self.attack_type = 1 # attack 1: thrust mid-level, 4 frames, low damage
                    else: 
                        self.attack_type = 2 # attack 2: sweeps low-to-high, medium damage
                    # call attack helper method: 
                    self.attack()
                    
                if key[pygame.K_l]: # parry
                    # call parry helper method:
                    self.parry()

            # check player 2 controls: 
            if self.player == 2: 
                # movement keypresses: 
                if key[pygame.K_LEFT]: # left
                    dx = -SPEED 
                    self.running = True
                if key[pygame.K_RIGHT]: # right 
                    dx = SPEED 
                    self.running = True
                if key[pygame.K_UP] and self.jump == False : # jump, 2nd cond. ensures no double-jump allowed! 
                    self.vel_y = -35
                    self.jump = True
                    
                if key[pygame.K_KP1] or key[pygame.K_KP2]: 
                    # determine attack type: 
                    if key[pygame.K_KP1]: 
                        self.attack_type = 1 # attack 1: thrust mid-level, 4 frames, low damage
                    else: 
                        self.attack_type = 2 # attack 2: sweeps low-to-high, medium damage
                    # call attack helper method: 
                    self.attack()
                    
                if key[pygame.K_KP3]: # parry
                    self.parry()
            
            
        # apply gravity: 
        self.vel_y += GRAVITY
        dy += self.vel_y
            
        # ensure player stays on screen: 
        if self.rect.left + dx < 0: # if movement would go off LHS of screen: 
            dx = 0 - self.rect.left
        if self.rect.right + dx > screen_width: # if movement would go off RHS of screen: 
            dx = screen_width - self.rect.right 
        if self.rect.bottom + dy > screen_height - 156: # if movement would go through bottom of bg's floor:
            self.vel_y = 0
            self.jump = False 
            dy = screen_height - 156 - self.rect.bottom # note: 156 == dist. in pixels from bottom of screen to bg's "floor"
    
        # make sure characters face each other:
        if target.rect.centerx > self.rect.centerx: 
            self.flip = False
        else: 
            self.flip = True
            
        # apply attack cooldown: 
        if self.attack_cooldown > 0: 
            self.attack_cooldown -= 1
            
        # apply parry cooldown: 
        if self.parry_cooldown > 0: 
            self.parry_cooldown -= 1
            
        # apply action cooldown: 
        if self.action_cooldown > 0: 
            self.action_cooldown -= 1
    
        # update player positions: 
        self.rect.x += dx
        self.rect.y += dy
       
    
    def update(self):
        # check what action the player is performing: 
        if self.health <= 0: 
            self.health = 0
            self.alive = False
            self.update_action(6) # 6: death
        elif self.parrying == True:
            self.update_action(7) # 7: parrying 
        elif self.hit == True:
            self.update_action(5) # 5: received hit
        elif self.attacking == True: 
            if self.attack_type == 1:
                self.update_action(3) # 3: attack1
            elif self.attack_type == 2:
                self.update_action(4) # 4: attack2
        elif self.jump == True: 
            self.update_action(2) # 2: jump 
        elif self.running == True: 
            self.update_action(1) # 1: run 
        else: 
            self.update_action(0) # 0: idle
        
        animation_cooldown = 50 # in milliseconds, how long each frame takes
        self.image = self.animation_list[self.action][self.frame_index]

        # check if enough time has passed since the last update to refresh animation: 
        if pygame.time.get_ticks() - self.update_time > animation_cooldown: 
            self.frame_index += 1
            self.update_time = pygame.time.get_ticks()
            
        # check if the animation has finished: 
        if self.frame_index >= len(self.animation_list[self.action]): 
            # if the player is dead, then end the animation: 
            if self.alive == False: 
                self.frame_index = len(self.animation_list[self.action]) - 1
            else: 
                self.frame_index = 0                
                # check if an attack was executed: 
                if self.action == 3 or self.action == 4: 
                    self.attacking = False
                    self.attack_cooldown = 40
                    self.attacking_rect = pygame.Rect(0,0,0,0)
                # check if attack was received: 
                if self.action == 5: 
                    self.hit = False
                    # if the player was in the middle of an attack, then their attack is cancelled: 
                    self.attacking = False 
                    self.attack_cooldown = 40 
                    self.attacking_rect = pygame.Rect(0,0,0,0)
                # check if parry was executed:
                if self.action == 7: 
                    self.parrying = False
                    self.parry_cooldown = 40
                    self.parrying_rect = pygame.Rect(0,0,0,0)
        
    
    def update_action(self, new_action): 
        # check if the new action is different from the previous one: 
        if new_action != self.action: 
            self.action = new_action
            # reset the animation index too:
            self.frame_index = 0
            self.update_time = pygame.time.get_ticks()
    

    def parry(self): 
        # check if the character can parry: 
        if self.parry_cooldown == 0:
            self.parrying = True
            
            # draw the rectangle defining the parry hitbox:
            self.parrying_rect = pygame.Rect(
                self.rect.centerx - (2 * self.rect.width * self.flip), # self.flip ensures attack faces enemy
                self.rect.y, 
                2 * self.rect.width, # make parry rect pretty small to make it more skillful
                self.rect.height
            )


    def attack(self): 
        # check if the character can attack: 
        if self.attack_cooldown == 0:
            self.attacking = True
            # change the attack hitbox according to attack type:
            if self.attack_type == 1: # (stronger, higher damage attack - like a top-down swing)
                self.attacking_rect = pygame.Rect(
                    self.rect.centerx - (2 * self.rect.width * self.flip), # self.flip ensures attack faces enemy
                    self.rect.y, 
                    2.5 * self.rect.width, 
                    self.rect.height
                )
            else: # self.attack_type == 2 (longer range, less damage attack - like a spear jab)
                self.attacking_rect = pygame.Rect(
                    self.rect.centerx - (3 * self.rect.width * self.flip), # self.flip ensures attack faces enemy
                    self.rect.centery, 
                    3.25 * self.rect.width, 
                    0.25 * self.rect.height 
                )
    
    
    def process_joint_actions(self, target):
        # if self attacks: 
        if self.attacking == True:
            # if target successfully parries: 
            if target.parrying == True and self.attacking_rect.colliderect(target.parrying_rect): 
                self.action_cooldown = 150
            # else (if) target either did not parry, or they did but unsuccessfully: 
            else:
                if self.attacking_rect.colliderect(target.rect):
                    if self.attack_type == 1: 
                        target.health -= .75
                    else: # self.attack_type == 2: 
                        target.health -= .20
                    target.hit = True
                    
        # if opponent is attacking: 
        if target.attacking == True: 
            # if self successfully parries: 
            if self.parrying == True and self.parrying_rect.colliderect(target.attacking_rect): 
                target.action_cooldown = 150
            # else (if) self either did not parry, or they did but unsuccessfully: 
            else:
                if target.attacking_rect.colliderect(self.rect):
                    if target.attack_type == 1: 
                        self.health -= .75
                    else: # self.attack_type == 2: 
                        self.health -= .20
                    self.hit = True
            
    
    def draw(self, surface):
        """_summary_
        Args:
            surface (_type_): Game window we are drawing the character/fighter onto. 
        """
        img = pygame.transform.flip(self.image, self.flip, False)
        
        # uncomment to show rectangle character is drawn on top of: 
        # pygame.draw.rect(surface, (255, 0, 0), self.rect)
        surface.blit(img, 
                        (self.rect.x - (self.offset[0] * self.image_scale), 
                         self.rect.y - (self.offset[1] * self.image_scale)
                         ))