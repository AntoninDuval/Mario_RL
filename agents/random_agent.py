from agents.agent import Agent
import random

class RandomAgent(Agent):
    def __init__(self):
        super(RandomAgent, self).__init__("random_agent")


    def get_action(self, state):
        # Get a random action
        index = random.randint(0, len(self.actions)-1)
        return self.actions[index]
