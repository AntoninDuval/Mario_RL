from pyboy import PyBoy, WindowEvent
import torch
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
        # Compute first fitness score
        self.previous_fitness_score = self.compute_reward()

    def stop(self):
        self.pyboy.stop()

    def obs(self):
        game_area = self.normalize_input(self.mario.game_area())
        return game_area

    def normalize_input(self, game_area):

        game_area = np.asarray(game_area).astype('int64')

        mario = np.arange(70)
        mario = np.delete(mario, [15,31])

        background = [336, 87, 89, 88,145,128, 247,248, 254,300, 305,306,307, 308, 310,338, 339,320,321,322,323,324,325,326,327,329,330,338, 339,350]
        floor = [352,353, 232]
        block_bonus = 129
        bonus = [131, 244]
        ennemy = [144]
        obstacle = [368, 130, 369, 355, 370, 371, 383]

        game_area[np.isin(game_area, mario)] = 1
        game_area[np.isin(game_area, background)] = 0
        game_area[np.isin(game_area, ennemy)] = 2
        game_area[np.isin(game_area, obstacle)] = 3
        game_area[game_area == block_bonus] = 4
        game_area[np.isin(game_area, floor)] = 5
        game_area[np.isin(game_area, bonus)] = 6
        return game_area



    def compute_reward(self):
        self._level_progress_max = max(self.mario.level_progress, self._level_progress_max)
        fitness = self.mario.score + self.mario.time_left * 10 + self._level_progress_max * 5 + self.mario.lives_left * 100
        return fitness

    def print_obs(self, game_area):

        print("\n".join(
            [
                f"{i: <3}| " + "".join([str(tile).ljust(4) for tile in line])
                for i, line in enumerate(game_area)
            ]
        ), "\n")

    def step(self, action, s):
        action = self.actions[action]

        # Do the same action during 8 frames (allows to do a big jump)
        for mini_steps in range(8):
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

        new_state = torch.Tensor(self.obs())

        if 15 in new_state and 31 in new_state: #titles used for dead mario
            reward = -100
            done = True
        else:
            done = False

        return new_state, reward, done