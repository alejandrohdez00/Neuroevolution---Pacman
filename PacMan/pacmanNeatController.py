import pygame
from game import Game
import neat
import time
from itertools import chain
import pickle

MAX_TIME = 60
TIME_WEIGHT = 0.2
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 576

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
        
        start_time = time.time()

        net = neat.nn.FeedForwardNetwork.create(genome, config)
     
        self.genome = genome
      

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            score = self.game.loop(self.screen, self.clock)

            #Store the positions of player and enemies to pass them as inputs
            
            positions = [x.rect.center for x in self.game.enemies]
            positions.append(self.game.player.rect.center)
            input_positions = tuple(chain(*positions))

            self.move_ai(net, input_positions)

            pygame.display.update()

            duration = time.time() - start_time
            if score >= 206 or duration >= MAX_TIME or self.game.game_over:
                self.calculate_fitness(score, duration)
                break

        return False


    def move_ai(self, net, positions):
        
        output = net.activate(positions)
        decision = output.index(max(output))

        if decision == 0:  # Don't move
            self.genome.fitness -= 0.01  # we want to discourage this
        elif decision == 1:  # Move up
            self.game.player.move_up()
        elif decision == 2:  # Move down
            self.game.player.move_down()
        elif decision == 3: #Move right
            self.game.player.move_right()
        else:
            self.game.player.move_left()


    def calculate_fitness(self, score, duration):
        if duration == MAX_TIME:
            self.genome.fitness += score 
        else:
            self.genome.fitness += score - TIME_WEIGHT * duration



    
