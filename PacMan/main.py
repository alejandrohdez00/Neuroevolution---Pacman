
import pygame
from pacmanNeatController import PacmanGame
import os 
import neat
import pickle
import visualization as vis
import neuralNetworkVis as nnvis
from manim import *

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
    #p = neat.Checkpointer.restore_checkpoint('.\Third model\checkpoint - 199')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=10, time_interval_seconds=None, filename_prefix = ".\Fourth model\checkpoint - "))

    winner = p.run(eval_genomes, 1)

    node_names = {-1: "GHOST1 X", -2: "GHOST1 Y", -3: "GHOST2 X", -4: "GHOST2 Y", -5: "GHOST3 X", -6: "GHOST3 Y", -7: "GHOST4 X", -8: "GHOST4 Y", -9: "GHOST5 X", -10: "GHOST5 Y", 
    -11: "GHOST6 X", -12: "GHOST6 Y", -13: "GHOST7 X", -14: "GHOST7 Y", -15: "GHOST8 X", -16: "GHOST8 Y", -17: "PACMAN X", -18: "PACMAN Y", 0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}
    
    #Visualize winner
    vis.draw_net(config, winner, True, prune_unused=True, node_names = node_names)
    vis.plot_stats(stats, ylog=False, view=True)
    vis.plot_species(stats, view=True)

    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

def test_best_network(config):
        with open(".\First model\\best.pickle", "rb") as f:
            winner = pickle.load(f)
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        pygame.display.set_caption("PACMAN")
        clock = pygame.time.Clock()

        pacman = PacmanGame(screen, clock)
        pacman.test_ai(winner_net, config)

    
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    print(config_path)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    #test_best_network(config)
