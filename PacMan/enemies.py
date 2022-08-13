
import pygame
import random
from math import dist


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 576

# Define some colors
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (0,0,255)
GREEN = (0,255,0)
RED = (255,0,0)

def manhattan(x, y):
    return sum(abs(val1-val2) for val1, val2 in zip(x,y))

def euclidean(x,y):
    return dist(x,y)


class Block(pygame.sprite.Sprite):
    def __init__(self,x,y,color,width,height):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self)
        # Set the background color and set it to be transparent
        self.image = pygame.Surface([width,height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x,y)


class Ellipse(pygame.sprite.Sprite):
    def __init__(self,x,y,color,width,height):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self)
        # Set the background color and set it to be transparent
        self.image = pygame.Surface([width,height])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)
        # Draw the ellipse
        pygame.draw.ellipse(self.image,color,[0,0,width,height])
        self.rect = self.image.get_rect()
        self.rect.topleft = (x,y)

        
class SlimeMH(pygame.sprite.Sprite):
    def __init__(self,x,y,change_x,change_y,player):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self)
        # Set the direction of the slime
        self.change_x = change_x
        self.change_y = change_y
        #player is the player object
        self.player = player
        # Load image
        self.image = pygame.image.load("Pacman\media\slime.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.topleft = (x,y)
        #Get intersections
        self.intersections = get_intersection_position()
 

    def update(self,horizontal_blocks,vertical_blocks):
        self.rect.x += self.change_x
        self.rect.y += self.change_y
        if self.rect.right < 0:
            self.rect.left = SCREEN_WIDTH
        elif self.rect.left > SCREEN_WIDTH:
            self.rect.right = 0
        if self.rect.bottom < 0:
            self.rect.top = SCREEN_HEIGHT
        elif self.rect.top > SCREEN_HEIGHT:
            self.rect.bottom = 0

        if self.rect.topleft in self.intersections:
            direction = self.best_action()
            if direction == "left" and self.change_x == 0:
                self.change_x = -2
                self.change_y = 0
            elif direction == "right" and self.change_x == 0:
                self.change_x = 2
                self.change_y = 0
            elif direction == "up" and self.change_y == 0:
                self.change_x = 0
                self.change_y = -2
            elif direction == "down" and self.change_y == 0:
                self.change_x = 0
                self.change_y = 2
    
    #Choose best action for slime taking into account the player position
    def best_action(self):  
        #Calculate manhattan distance to player
        dist_left = ("left",manhattan((self.rect.centerx - 2, self.rect.centery), self.player.rect.center))
        dist_right = ("right",manhattan((self.rect.centerx + 2, self.rect.centery), self.player.rect.center))
        dist_up = ("up", manhattan((self.rect.centerx, self.rect.centery - 2), self.player.rect.center))
        dist_down = ("down", manhattan((self.rect.centerx, self.rect.centery + 2), self.player.rect.center))
        
        #Choose min distance
        min_dist = min(dist_left, dist_right, dist_up, dist_down, key=lambda x: x[1])
        return min_dist[0]

class SlimeEURev(pygame.sprite.Sprite):
    def __init__(self,x,y,change_x,change_y,player):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self)
        # Set the direction of the slime
        self.change_x = change_x
        self.change_y = change_y
        #player is the player object
        self.player = player
        # Load image
        self.image = pygame.image.load("Pacman\media\slime.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.topleft = (x,y)
        #Get intersections
        self.intersections = get_intersection_position()
 

    def update(self,horizontal_blocks,vertical_blocks):
        self.rect.x += self.change_x
        self.rect.y += self.change_y
        if self.rect.right < 0:
            self.rect.left = SCREEN_WIDTH
        elif self.rect.left > SCREEN_WIDTH:
            self.rect.right = 0
        if self.rect.bottom < 0:
            self.rect.top = SCREEN_HEIGHT
        elif self.rect.top > SCREEN_HEIGHT:
            self.rect.bottom = 0

        if self.rect.topleft in self.intersections:
            direction = self.best_action()
            if direction == "left" and self.change_x == 0:
                self.change_x = -2
                self.change_y = 0
            elif direction == "right" and self.change_x == 0:
                self.change_x = 2
                self.change_y = 0
            elif direction == "up" and self.change_y == 0:
                self.change_x = 0
                self.change_y = -2
            elif direction == "down" and self.change_y == 0:
                self.change_x = 0
                self.change_y = 2
    
    #Choose best action for slime taking into account the player position
    def best_action(self):  
        #Calculate manhattan distance to player
        dist_left = ("right",euclidean((self.rect.centerx - 2, self.rect.centery), self.player.rect.center))
        dist_right = ("left",euclidean((self.rect.centerx + 2, self.rect.centery), self.player.rect.center))
        dist_up = ("down", euclidean((self.rect.centerx, self.rect.centery - 2), self.player.rect.center))
        dist_down = ("up", euclidean((self.rect.centerx, self.rect.centery + 2), self.player.rect.center))
        
        #Choose min distance
        min_dist = min(dist_left, dist_right, dist_up, dist_down, key=lambda x: x[1])
        return min_dist[0]

        
        
       
        
                

def get_intersection_position():
    items = []
    for i,row in enumerate(enviroment()):
        for j,item in enumerate(row):
            if item == 3:
                items.append((j*32,i*32))

    return items

def enviroment():
    grid = ((0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1,3,1,1,1),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0),
            (0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0))

    return grid

def draw_enviroment(screen):
    for i,row in enumerate(enviroment()):
        for j,item in enumerate(row):
            if item == 1:
                pygame.draw.line(screen, BLUE , [j*32, i*32], [j*32+32,i*32], 3)
                pygame.draw.line(screen, BLUE , [j*32, i*32+32], [j*32+32,i*32+32], 3)
            elif item == 2:
                pygame.draw.line(screen, BLUE , [j*32, i*32], [j*32,i*32+32], 3)
                pygame.draw.line(screen, BLUE , [j*32+32, i*32], [j*32+32,i*32+32], 3)
