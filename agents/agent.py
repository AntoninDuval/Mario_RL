from abc import ABC

class Agent(ABC):
    def __init__(self, name: str):
        self.name = name

    def train(self):
        pass

    def update_replay_memory(self):
        pass

    def get_action(self):
        pass

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
