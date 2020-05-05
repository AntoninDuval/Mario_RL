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
        self.mario_size = 0

        self.fitness_score = 0
        self.previous_fitness_score = 0
        self.previous_direction = 0
        self._level_progress_max = 0

        self.observation_space = (16, 20)
        self.action_space = [4]

        self.max_steps = max_steps

        self.pair_actions = {"5": 13,
                             "3": 11,
                             "4": 12,
                             "0": 0}

        self.action_jump = [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A]

        self.action_direction = [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT, 0]

    def start(self):
        self.mario.start_game()
        self.mario_lives = self.mario.lives_left
        self.fitness_score = 0


    def reset(self):
        self.mario.reset_game()
        self.mario_lives = self.mario.lives_left
        self.mario_size = 0

        self.fitness_score = 0
        self.level_progress_max = 0
        # Compute first fitness score
        self.previous_fitness_score = self.compute_reward()
        self.previous_direction = 0

    def stop(self):
        self.pyboy.stop()

    def obs(self):
        game_area = self.normalize_input(self.mario.game_area())
        return game_area

    def normalize_input(self, game_area):

        game_area = np.asarray(game_area).astype('int64')

        if 32 in game_area and 49 in game_area:
            self.mario_size = torch.ones(1)
        else:
            self.mario_size = torch.zeros(1)

        mario = np.arange(70)
        mario = np.delete(mario, [15, 31])

        background = [336, 87, 89, 88, 91,145,168,169,128, 247,248, 254,300, 305,306,307, 308, 310,316,328, 331, 332, 338, 339,320,321,322,323,324,325,326,327,329,330,338, 339,350]
        floor = [142, 143, 239, 352,353, 232]
        block_bonus = 129
        bonus = [131,132, 244, 246]
        ennemy = [144, 150, 151, 152, 153, 160, 161, 162, 163, 176, 177, 178, 179]
        obstacle = [368, 130, 369, 355, 370, 371, 383]

        game_area[np.isin(game_area, mario)] = 1
        game_area[np.isin(game_area, mario)] = 1
        game_area[np.isin(game_area, background)] = 0
        game_area[np.isin(game_area, ennemy)] = 3
        game_area[np.isin(game_area, obstacle)] = 4
        game_area[np.isin(game_area, floor)] = 4
        game_area[game_area == block_bonus] = 5
        game_area[np.isin(game_area, bonus)] = 6
        return game_area



    def compute_reward(self):
        self.level_progress_max = max(self.mario.level_progress, self.level_progress_max)
        fitness = self.mario.score + self.mario.time_left * 10 + self.level_progress_max * 5 + self.mario.lives_left * 100
        return fitness

    def print_obs(self, game_area):

        print("\n".join(
            [
                f"{i: <3}| " + "".join([str(tile).ljust(4) for tile in line])
                for i, line in enumerate(game_area)
            ]
        ), "\n")

    def step(self, actions=None):
        if actions is not None:
            direction = self.action_direction[actions[0]]
            jump = self.action_jump[actions[1]]

            if direction != self.previous_direction:
                # Release the previous direction
                end_action = self.pair_actions[str(self.previous_direction)]
                self.pyboy.send_input(end_action)
                self.previous_direction = direction

            self.pyboy.send_input(direction)
            self.pyboy.send_input(jump)
            for ministeps in range(6):
                self.pyboy.tick()

        else:
            for ministeps in range(5):
                self.pyboy.tick()

        # Compute reward
        self.fitness_score = self.compute_reward()

        reward = min((self.fitness_score - self.previous_fitness_score), 100) / 100
        self.previous_fitness_score = self.fitness_score

        # Update fitness score
        self.fitness_score = self.mario.fitness

        new_state = torch.Tensor(self.obs())

        if (15 in new_state and 31 in new_state) or self.mario.lives_left == 1:  # titles used for dead mario
            reward = -1
            done = True
        else:
            done = False

        return new_state, reward, done