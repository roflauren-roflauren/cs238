import pygame

CHAR_WIDTH  = 80
CHAR_HEIGHT = 180

class Fighter():
    def __init__(self, x, y, start_flip, data, sprite_sheet, animation_steps):
        """_summary_

        Args:
            x (int): left-coord where fighter will be instantiated. 
            y (int): top-coord where fighter will be instantiated. 
        """
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
        
        # animation variables: 
        self.frame_index = 0 # which frame of the animation we're on
        self.image = self.animation_list[self.action][self.frame_index]
        
        # player position as represented by a rectangle:
        self.rect = pygame.Rect(
            x, y, CHAR_WIDTH, CHAR_HEIGHT
        )
        
        # y-velocity: 
        self.vel_y = 0
        
        # track whether character is jumping or not: 
        self.jump = False
        
        # track whether character is attacking or not: 
        self.attacking = False
        
        # attack type during the current move: 
        self.attack_type = 0 # valid values: 0 (inactive), 1, 2
        
        # health pool: 
        self.health = 100
    
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
            
    def move(self, screen_width, screen_height, surface, target):
        """_summary_

        Args:
            screen_width (_type_): _description_
            screen_height (_type_): _description_
        """
        SPEED = 10     # controls how quickly characters are moving on the screen
        GRAVITY = 2    # counters jumping & ensures player doesn't fly off screen
        dx, dy = 0, 0  # change in x, y coordinate of character
        
        # get keypresses: 
        key = pygame.key.get_pressed()
        
        
        # TO-DO: ADD IN BLOCKING FUNCTIONALITY
        # TO-DO: PROBABLY DELETE THIS, LET CHARACTERS ATTACK IN MIDAIR AND WHATNOT: 
        # can only perform other actions if not currently attacking: 
        if self.attacking == False:
        
            # movement keypresses: 
            if key[pygame.K_a]: # left
                dx = -SPEED 
            if key[pygame.K_d]: # right 
                dx = SPEED 
            if key[pygame.K_w] and self.jump == False : # jump, 2nd cond. ensures no double-jump allowed! 
                self.vel_y = -30 
                self.jump = True
            if key[pygame.K_j] or key[pygame.K_k]: 
                
                self.attack(surface, target)
                
                # determine attack type: 
                if key[pygame.K_j]: 
                    self.attack_type = 1 # attack 1: sweeps low to high, 5 frames, medium damage
                else: 
                    self.attack_type = 2 # attack 2: thrust mid-level, 4 frames, low damage
            
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
    
        # update player positions: 
        self.rect.x += dx
        self.rect.y += dy
    
    def attack(self, surface, target): 
        """_summary_
        """
        self.attacking = True
        
        # TO-DO: RESET self.attacking TO ALLOW OTHER MOVEMENTS 
        # TO-DO: MAKE attacking_rect DEPEND ON ATTACK TYPE AND CHANGE DIMS.
        
        attacking_rect = pygame.Rect(
            self.rect.centerx - (2 * self.rect.width * self.flip), # self.flip ensures attack faces enemy
            self.rect.y, 
            2 * self.rect.width, 
            self.rect.height
        )
        
        if attacking_rect.colliderect(target.rect):
           target.health -= 10
        
        # TO-DO: DELETE
        pygame.draw.rect(surface, (0, 255, 0), attacking_rect)
        
    def draw(self, surface):
        """_summary_
        Args:
            surface (_type_): Game window we are drawing the character/fighter onto. 
        """
        img = pygame.transform.flip(self.image, self.flip, False)
        pygame.draw.rect(surface, (255, 0, 0), self.rect)
        surface.blit(img, 
                        (self.rect.x - (self.offset[0] * self.image_scale), 
                         self.rect.y - (self.offset[1] * self.image_scale)
                         ))