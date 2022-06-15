import pygame
from game import Game
import neat
import time
from itertools import chain

MAX_TIME = 60

class PacmanGame:

    def __init__(self, screen, clock):
        self.game = Game()
        self.screen = screen
        self.clock = clock

    def train_ai(self, genome, config):
        """
        Train the AI by passing a neural networks and the NEAt config object.

        """
        run = True

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
            if score >= 156 or duration >= MAX_TIME or self.game.game_over:
                self.calculate_fitness(score, duration)
                break

        return False

    def move_ai(self, net, positions):
        
        output = net.activate(positions)
        decision = output.index(max(output))

        #valid = True
        if decision == 0:  # Don't move
            self.genome.fitness -= 0.01  # we want to discourage this
        elif decision == 1:  # Move up
            self.game.player.move_up()
        elif decision == 2:  # Move down
            self.game.player.move_down()
        elif decision == 3: #Move right
            self.game.player.move_right()
        elif decision == 4:
            self.game.player.move_left()
        elif decision == 5:
            self.game.player.stop_move_up()
        elif decision == 6:
            self.game.player.stop_move_down()
        elif decision == 7:
            self.game.player.stop_move_right()
        else:
            self.game.player.stop_move_left()

        # if not valid:  # If the movement makes the paddle go off the screen punish the AI
        #     self.genome.fitness -= 1

    def calculate_fitness(self, score, duration):
        if duration == MAX_TIME:
            self.genome.fitness += score 
        else:
            self.genome.fitness += score - 0.2 * duration