import pygame
from game import Game
import neat
import time
from itertools import chain
import pickle
from math import dist

FPS = 300
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
            
            positions = [x.rect.center for x in self.game.enemies]
            positions.append(self.game.player.rect.center)
            input_positions = tuple(chain(*positions))

            self.move_ai(net, input_positions)

            pygame.display.update()

        return False

    def train_ai(self, genome, config):
        """
        Train the AI by passing a neural networks and the NEAt config object.

        """
        run = True
        
        #net = neat.nn.FeedForwardNetwork.create(genome, config)
        net = neat.nn.RecurrentNetwork.create(genome, config)
     
        self.genome = genome
      
        prev_score = 1

        score_increased = 0

        start_time = time.time()

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            score = self.game.loop(self.screen, self.clock)

            if(score > prev_score):
                prev_score = score
                score_increased = 1
            else:
                score_increased = 0

            #Store the positions of player and enemies to pass them as inputs
            
            # positions = [x.rect.center for x in self.game.enemies]
            # positions.append(self.game.player.rect.center)
            # inputs = tuple(chain(*positions)) + (score_increased,)

            #Store euclid distance of player and enemies to pass them as inputs and corridors for enemies and players and score_increased

            distances = [euclidean(x.rect.center, self.game.player.rect.center) for x in self.game.enemies]
            corridors = [calc_corridor(x.rect.center[0], x.rect.center[1]) for x in self.game.enemies]
            corridors.append(calc_corridor(self.game.player.rect.center[0], self.game.player.rect.center[1]))
            
            inputs = tuple(distances) + tuple(corridors) + (score_increased,)

            self.move_ai(net, inputs)

            pygame.display.update()

            duration = time.time() - start_time
            if score >= 206 or self.game.game_over:
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
        self.genome.raw_fitness += score - TIME_WEIGHT * (duration * (FPS/30))
        self.genome.fitness = self.genome.raw_fitness / self.genome.evaluations

 
            