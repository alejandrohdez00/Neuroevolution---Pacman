from turtle import position
import pygame
from game import Game
import os 
import neat
import time
from itertools import chain

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 576

class PacmanGame:

    def __init__(self, screen, clock):
        self.game = Game()
        self.screen = screen
        self.clock = clock

    def main():

        screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        game = Game()
        inf = game.loop(screen)
        print(inf)


    def train_ai(self, genome, config):
        """
        Train the AI by passing two NEAT neural networks and the NEAt config object.
        These AI's will play against eachother to determine their fitness.
        """
        run = True
        
        start_time = time.time()

        net = neat.nn.FeedForwardNetwork.create(genome, config)
     
        self.genome = genome
      

        max_time = 180

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
            if score >= 156 or duration >= 180 or self.game.game_over:
                self.calculate_fitness(score, duration)
                break

        return False

    def move_ai(self, net, positions):
        
        output = net.activate(positions)
        decision = output.index(max(output))

        #valid = True
        if decision == 0:  # Don't move
            pass
            #self.genome.fitness -= 0.01  # we want to discourage this
        elif decision == 1:  # Move up
            self.game.player.move_up()
        elif decision == 2:  # Move down
            self.game.player.move_down()
        elif decision == 3: #Move right
            self.game.player.move_right()
        else:
            self.game.player.move_left()

        # if not valid:  # If the movement makes the paddle go off the screen punish the AI
        #     self.genome.fitness -= 1

    def calculate_fitness(self, score, duration):
        self.genome.fitness += score - duration
    




def eval_genomes(genomes, config):
    """
    Run each genome against eachother one time to determine the fitness.
    """
    screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
    pygame.display.set_caption("PACMAN")
    clock = pygame.time.Clock()

    for i, (genome_id, genome) in enumerate(genomes):
        print(round(i/len(genomes) * 100), end=" ")
        genome.fitness = 0
        pacman = PacmanGame(screen, clock)

        force_quit = pacman.train_ai(genome, config)
        if force_quit:
            quit()


def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-85')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    #test_best_network(config)
