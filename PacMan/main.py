
import pygame
from pacmanNeatController import PacmanGame
import os 
import neat

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 576

def eval_genomes(genomes, config):
    """
    Run each genome against eachother one time to determine the fitness.
    """
    screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
    pygame.display.set_caption("PACMAN")
    clock = pygame.time.Clock()

    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0
        pacman = PacmanGame(screen, clock)

        force_quit = pacman.train_ai(genome, config)
        if force_quit:
            quit()


def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-10')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, filename_prefix = ".\Second Model\checkpoint - "))

    winner = p.run(eval_genomes, 50)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    #test_best_network(config)
