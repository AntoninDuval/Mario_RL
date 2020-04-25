from abc import ABC
from pyboy import WindowEvent


import numpy as np
from numpy import ndarray


class Agent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.actions = [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PASS]

    def train(self, done):
        pass

    def update_replay_memory(self, state, action, reward, new_state, done):
        pass

    def get_action(self, state: ndarray, training : bool):
        pass

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
