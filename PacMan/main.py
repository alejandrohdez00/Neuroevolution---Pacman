
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
        genome.evaluations += 1
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

    winner = p.run(eval_genomes, 200)

    node_names = {-1: "GHOST1 DIST", -2: "GHOST2 DIST", -3: "GHOST3 DIST", -4: "GHOST4 DIST", -5: "GHOST5 DIST", -6: "GHOST6 DIST", -7: "GHOST7 DIST", -8: "GHOST8 DIST", -9: "GHOST1 CORR", -10: "GHOST2 CORR", 
    -11: "GHOST3 CORR", -12: "GHOST4 CORR", -13: "GHOST5 CORR", -14: "GHOST6 CORR", -15: "GHOST7 CORR", -16: "GHOST8 CORR", -17: "PACMAN CORR", -18: "S_INCREASED", 0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}
    
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
