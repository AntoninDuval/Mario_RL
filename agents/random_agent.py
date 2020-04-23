from agents.agent import Agent
from pyboy import WindowEvent
import random

class RandomAgent(Agent):
    def __init__(self):
        super(RandomAgent, self).__init__("random_agent")
        self.actions = [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A]
        self.end_actions = [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A]

    def get_action(self):
        # Get a random action
        index = random.randint(0, len(self.actions)-1)
        return self.actions[index], self.end_actions[index]
