
import pygame
from pacmanNeatController import PacmanGame
import os 
import neat
import pickle
import visualization as vis

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
    p = neat.Checkpointer.restore_checkpoint('.\Third Model - Recurrent\checkpoint - 154')
    #p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=10, filename_prefix = ".\Third Model - Recurrent\checkpoint - "))
    #vis.plot_species(stats)

    winner = p.run(eval_genomes, 500)
    
    #Visualize winner
    vis.draw_net(config, winner, True)
    vis.draw_net(config, winner, True, prune_unused=True)
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
        pacman.test_ai(winner_net)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    print(config_path)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    #test_best_network(config)
