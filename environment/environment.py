from pyboy import PyBoy, WindowEvent
import numpy as np

class Environment:

    def __init__(self, filename, max_steps, visualize=False):

        self.pyboy = PyBoy(filename, window_type="headless" if not visualize else "SDL2", window_scale=3, debug=visualize,
                  game_wrapper=True)
        assert self.pyboy.cartridge_title() == "SUPER MARIOLAN"

        self.pyboy.set_emulation_speed(0)
        self.mario = self.pyboy.game_wrapper()
        self.mario_lives = None
        self.fitness_score = 0
        self.previous_fitness_score = 0
        self._level_progress_max = 0

        self.observation_space = (16, 20)
        self.action_space = [4]

        self.max_steps = max_steps

        self.pair_actions = {"5": 13,
                             "3": 11,
                             "4": 12,
                             "0": 0}

        self.actions = [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A,
                        0]

    def start(self):
        self.mario.start_game()
        self.mario_lives = self.mario.lives_left
        self.fitness_score = 0


    def reset(self):
        self.mario.reset_game()
        self.mario_lives = self.mario.lives_left
        self.fitness_score = 0
        self._level_progress_max = 0
        # We start with a fitness of 2710 every game
        self.previous_fitness_score = self.compute_reward()

    def stop(self):
        self.pyboy.stop()

    def obs(self):
        return np.asarray(self.mario.game_area()).astype('int64')

    def compute_reward(self):
        self._level_progress_max = max(self.mario.level_progress, self._level_progress_max)
        fitness = self.mario.score + self._level_progress_max * 5 + self.mario.lives_left * 100
        return fitness


    def step(self, action, s):
        action = self.actions[action]

        for mini_steps in range(9):
            self.pyboy.send_input(action)
            self.pyboy.tick()

        # Get the corresponding release action
        end_action = self.pair_actions[str(action)]
        self.pyboy.send_input(end_action)
        self.pyboy.tick()

        #Compute reward

        self.fitness_score = self.compute_reward()

        reward = self.fitness_score - self.previous_fitness_score
        self.previous_fitness_score = self.fitness_score

        # Update fitness score
        self.fitness_score = self.mario.fitness

        if self.mario_lives != self.mario.lives_left or s == self.max_steps-1:
            done = True
        else:
            done = False

        new_state = self.obs()

        return new_state, reward, done
