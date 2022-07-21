import pygame
from game import Game
import neat
import time
from itertools import chain
import pickle
from math import dist

FPS = 600
TIME_WEIGHT = 0.2
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 576

def manhattan(x, y):
    return sum(abs(val1-val2) for val1, val2 in zip(x,y))

def euclidean(x,y):
    return dist(x,y)

def calc_corridor(x,y):
    if y == 80:
        return 1
    elif y == 208:
        return 2
    elif y == 336:
        return 3
    elif y == 464:
        return 4
    elif x == 48:
        return 5
    elif x == 176:
        return 6
    elif x == 304:
        return 7
    elif x == 432:
        return 8
    elif x == 560:
        return 9
    elif x == 688:
        return 10
    else: 
        return 0
          
class PacmanGame:

    def __init__(self, screen, clock):
        self.game = Game()
        self.screen = screen
        self.clock = clock

    def test_ai(self, net, config):
        """
        Train the AI by passing a neural networks and the NEAt config object.

        """
        run = True

     
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            self.game.loop(self.screen, self.clock)

            #Store the positions of player and enemies to pass them as inputs
            
            distances = []
            if self.game.enemies.__len__() <= 8:
                for enemy in self.game.enemies:
                    distances.append(euclidean(enemy.rect.center, self.game.player.rect.center))
                for i in range(8 - self.game.enemies.__len__()):
                    distances.append(9999999999)
                
            corridors = []
            if self.game.enemies.__len__() <= 8:
                for enemy in self.game.enemies:
                    corridors.append(calc_corridor(enemy.rect.center[0], enemy.rect.center[1]))
                for i in range(8 - self.game.enemies.__len__()):
                    distances.append(0)

            #Calculate corridors of ghosts and player
            corridors.append(calc_corridor(self.game.player.rect.center[0], self.game.player.rect.center[1]))

            #Calculate distance and corridor of nearest dot
            distance_nd, nd_coor = self.find_nearest_dot()
            corridor_nd = calc_corridor(nd_coor[0], nd_coor[1])

            #Calculate if Pacman is in an intersection
            in_intersection = self.game.player.in_intersection()

            #Calculate previous direction of Pacman
            moving_up = 1 if self.game.player.change_y == -3 else 0
            moving_down = 1 if self.game.player.change_y == 3 else 0
            moving_right = 1 if self.game.player.change_x == 3 else 0
            moving_left = 1 if self.game.player.change_x == -3 else 0

            #inputs = tuple(distances) + tuple(corridors) + (distance_nd,) + (corridor_nd,) + (in_intersection,) + (moving_up,) + (moving_down,) + (moving_right,) + (moving_left,)
            inputs = tuple(distances) + tuple(corridors) + (corridor_nd, distance_nd, in_intersection, moving_up, moving_down, moving_right, moving_left)

            self.move_ai(net, inputs)

            pygame.display.update()

        return False

    def train_ai(self, genome, config):
        """
        Train the AI by passing a neural networks and the NEAt config object.

        """
        run = True
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #net = neat.nn.RecurrentNetwork.create(genome, config)
     
        self.genome = genome

        prev_score = 1
        duration_no_score = 0
        count = 0

        initial_len = self.game.dots_group.__len__()
      
        start_time = time.time()

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            score = self.game.loop(self.screen, self.clock)

            if score == prev_score:
                if count == 0:
                    start_time_no_score = time.time()
                    count += 1
                else:
                    duration_no_score = time.time() - start_time_no_score

            if score > prev_score:
                prev_score = score
                count = 0

                

            #Store the positions of player and enemies to pass them as inputs
            
            # positions = [x.rect.center for x in self.game.enemies]
            # positions.append(self.game.player.rect.center)
            # inputs = tuple(chain(*positions)) + (score_increased,)

            #Store euclid distance of player and enemies to pass them as inputs and corridors for enemies and players and score_increased

            #Calculate inputs
            #Calculate distance of ghosts to the player
            distances = []
            if self.game.enemies.__len__() <= 8:
                for enemy in self.game.enemies:
                    distances.append(euclidean(enemy.rect.center, self.game.player.rect.center))
                for i in range(8 - self.game.enemies.__len__()):
                    distances.append(9999999999)
                
            corridors = []
            if self.game.enemies.__len__() <= 8:
                for enemy in self.game.enemies:
                    corridors.append(calc_corridor(enemy.rect.center[0], enemy.rect.center[1]))
                for i in range(8 - self.game.enemies.__len__()):
                    distances.append(0)

            #Calculate corridors of ghosts and player
            corridors.append(calc_corridor(self.game.player.rect.center[0], self.game.player.rect.center[1]))

            #Calculate distance and corridor of nearest dot
            distance_nd, nd_coor = self.find_nearest_dot()
            corridor_nd = calc_corridor(nd_coor[0], nd_coor[1])

            #Calculate if Pacman is in an intersection
            in_intersection = self.game.player.in_intersection()

            #Calculate previous direction of Pacman
            moving_up = 1 if self.game.player.change_y == -3 else 0
            moving_down = 1 if self.game.player.change_y == 3 else 0
            moving_right = 1 if self.game.player.change_x == 3 else 0
            moving_left = 1 if self.game.player.change_x == -3 else 0

            #inputs = tuple(distances) + tuple(corridors) + (distance_nd,) + (corridor_nd,) + (in_intersection,) + (moving_up,) + (moving_down,) + (moving_right,) + (moving_left,)
            inputs = tuple(distances) + tuple(corridors) + (corridor_nd, distance_nd, in_intersection, moving_up, moving_down, moving_right, moving_left)
            
            self.move_ai(net, inputs)

            pygame.display.update()

            duration = time.time() - start_time
            if score >= initial_len or self.game.game_over or duration_no_score > 1.5:
                self.calculate_fitness(score, duration)
                break

        return False


    def move_ai(self, net, positions):
        
        output = net.activate(positions)
        decision = output.index(max(output))

        if decision == 0:  # Move up
            self.game.player.move_up()
        elif decision == 1:  # Move down
            self.game.player.move_down()
        elif decision == 2:  # Move right
            self.game.player.move_right()
        elif decision == 3: #Move left
            self.game.player.move_left()


    def calculate_fitness(self, score, duration):
        #self.genome.fitness = score - TIME_WEIGHT * (duration * (self.clock.get_fps()/30))
        # self.genome.fitness = self.genome.raw_fitness / self.genome.evaluations
        self.genome.fitness = score

    #Find the nearest dot to the player
    def find_nearest_dot(self):
        nearest_dot = None
        nearest_distance = 9999999
        if self.game.dots_group.__len__() == 0:
            return (0, (0,0))
        else:
            for dot in self.game.dots_group:
                distance = euclidean(self.game.player.rect.center,dot.rect.center)
                if distance < nearest_distance:
                    nearest_dot = dot
                    nearest_distance = distance
            return (nearest_distance,nearest_dot.rect.center)

 
            