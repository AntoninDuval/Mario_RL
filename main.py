import os
import sys
import time

from pyboy import PyBoy, WindowEvent

from agents.random_agent import RandomAgent



def main():

    # Makes us able to import PyBoy from the directory below
    file_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, file_path + "/..")

    # Check if the ROM is given through argv
    filename = './Super_Mario_Land_World.gb'

    quiet = False
    pyboy = PyBoy(filename, window_type="headless" if quiet else "SDL2", window_scale=3, debug=not quiet,
                  game_wrapper=True)
    pyboy.set_emulation_speed(0)
    assert pyboy.cartridge_title() == "SUPER MARIOLAN"

    mario = pyboy.game_wrapper()
    mario.start_game()

    assert mario.score == 0
    assert mario.lives_left == 2
    assert mario.time_left == 400
    assert mario.world == (1, 1)
    assert mario.fitness == 0  # A built-in fitness score for AI development
    last_fitness = 0

    agent = RandomAgent()

    for _ in range(1000):

        if mario.lives_left == 0:
            mario.reset_game()
            mario.start_game()

        action, end_action = agent.get_action()

        # Exectute action during 5 frame
        for mini_steps in range(10):
            pyboy.send_input(action)
            pyboy.tick()
        # Realease the button
        pyboy.send_input(end_action)
        pyboy.tick()

    mario.reset_game()
    pyboy.stop()

if __name__ == '__main__':
    main()