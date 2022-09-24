
import pygame
from pacmanNeatController import PacmanGame
import os 
import neat
import pickle
import visualization as vis
#from manim import *

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 576

#node names for dist_corr inputs
# node_names = {-1: "GHOST1 DIST", -2: "GHOST2 DIST", -3: "GHOST3 DIST", -4: "GHOST4 DIST", -5: "GHOST5 DIST", -6: "GHOST6 DIST", -7: "GHOST7 DIST", -8: "GHOST8 DIST", -9: "GHOST1 CORR", -10: "GHOST2 CORR", 
# -11: "GHOST3 CORR", -12: "GHOST4 CORR", -13: "GHOST5 CORR", -14: "GHOST6 CORR", -15: "GHOST7 CORR", -16: "GHOST8 CORR", -17: "PACMAN CORR", -18: "ND CORR", -19: "ND DIST", -20: "IN_INTERS", -21: "MOVING UP", 
# -22: "MOVING DOWN", -23: "MOVING RIGHT", -24: "MOVING LEFT", 0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}

#node names for x_y inputs
# node_names = {-1: "GHOST1 X", -2: "GHOST1 Y", -3: "GHOST2 X", -4: "GHOST2 Y", -5: "GHOST3 X", -6: "GHOST3 Y", -7: "GHOST4 X", -8: "GHOST4 Y", -9: "GHOST5 X", -10: "GHOST5 Y", -11: "GHOST6 X", -12: "GHOST6 Y",
# -13: "GHOST7 X", -14: "GHOST7 Y", -15: "GHOST8 X", -16: "GHOST8 Y", -17: "PACMAN X", -18: "PACMAN Y", -19: "ND X", -20: "ND Y", -21: "IN_INTERS", -22: "MOVING UP", -23: "MOVING DOWN", -24: "MOVING RIGHT", -25: "MOVING LEFT", 0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}
    
node_names = {-1: "GHOST1 X", -2: "GHOST1 Y", -3: "GHOST2 X", -4: "GHOST2 Y", -5: "GHOST3 X", -6: "GHOST3 Y", -7: "GHOST4 X", -8: "GHOST4 Y", -9: "GHOST5 X", -10: "GHOST5 Y", -11: "GHOST6 X", -12: "GHOST6 Y",
-13: "GHOST7 X", -14: "GHOST7 Y", -15: "GHOST8 X", -16: "GHOST8 Y", -17: "PACMAN X", -18: "PACMAN Y", -19: "ND X", -20: "ND Y", -21: "IN_INTERS", 0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}

def eval_genomes(genomes, config):
    """
    Run each genome against eachother one time to determine the fitness.
    """
    screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
    pygame.display.set_caption("PACMAN")
    clock = pygame.time.Clock()

    for i, (genome_id, genome) in enumerate(genomes):
        #genome.evaluations += 1
        genome.fitness = 0
        pacman = PacmanGame(screen, clock)

        force_quit = pacman.train_ai(genome, config)
        if force_quit:
            quit()


def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('.\\Model x_y_det\\Model with 8 ghosts with no moving knowledge\\1000 gen\\checkpoints\\checkpoint - 149')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=10, time_interval_seconds=None, filename_prefix = ".\\Model x_y_det\\Model with 8 ghosts with no moving knowledge\\1000 genv4\\checkpoints\\checkpoint - "))

    winner = p.run(eval_genomes, 800)
    
    #Visualize winner
    vis.draw_net(config, winner, True, prune_unused=True, node_names = node_names)
    vis.plot_stats(stats, ylog=False, view=True)
    vis.plot_species(stats, view=True)

    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

def test_best_network(config):
        with open("best.pickle", "rb") as f:
            winner = pickle.load(f)
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        pygame.display.set_caption("PACMAN")
        clock = pygame.time.Clock()

        pacman = PacmanGame(screen, clock)
        score = pacman.test_ai(winner_net, config)
        return score

def evaluate_checkpoint(config, checkpoint):
    p = neat.Checkpointer.restore_checkpoint(checkpoint, config)
    winner = p.run(eval_genomes, 1)
    
    #Visualize winner
    vis.draw_net(config, winner, True, prune_unused=True, node_names = node_names)
    
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    print(config_path)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    # scores = []
    # for i in range(1):
    #     scores.append(test_best_network(config))
    # print(scores)
    # print(sum(scores)/len(scores))
