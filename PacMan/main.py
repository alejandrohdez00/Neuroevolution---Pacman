import pygame
from game import Game

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 576

def main():

    screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
    game = Game()
    inf = game.loop(screen)
    print(inf)

if __name__ == '__main__':
    main()
