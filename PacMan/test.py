import pygame
from game import Game
import os 
import neat
import time
from itertools import chain

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 576

def main():

        screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        pygame.display.set_caption("PACMAN")
        clock = pygame.time.Clock()
        game = Game()
        start = time.time()
        while not game.game_over and game.score < 156:
            inf = game.loop(screen,clock)
              
            positions = [x.rect.center for x in game.enemies]
            positions.append(game.player.rect.center)
            p = tuple(chain(*positions))
            print(p)
            for x in game.enemies:
                print(f"{x} position: {x.rect.center}")
            print(f"Position pacman: {game.player.rect.center}")

        duration = time.time() - start
        print(game.score, duration)

if __name__ == '__main__':
    main()