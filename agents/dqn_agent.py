from agents.agent import Agent
import random
import numpy as np
from collections import deque
from numpy import ndarray

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from agents.replay_memory import ReplayMemory

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 500
MINIBATCH_SIZE = 256

UPDATE_TARGET_EVERY = 1


class DQNModel(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQNModel, self).__init__()

        self.input_size = 1
        for dim in observation_space:
            self.input_size *= dim

        self.layer1 = nn.Linear(self.input_size, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 4)

    def forward(self, state):
        x = torch.relu(self.layer1(state.view(-1, self.input_size)))
        x = torch.relu(self.layer2(x))
        output = self.layer3(x)
        return output

class DQN_Agent(Agent):
    def __init__(self, epsilon=0.1, discount = 0.99, learning_rate=1e-4):
        super(DQN_Agent, self).__init__("dqn_agent")
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.model = None
        self.target_model = None
        self.criterion = None
        self.optimizer = None

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        self.state = None

    def setup(self, observation_space, action_space, trained_model=None):

        self.model = DQNModel(observation_space, action_space)
        self.target_model = DQNModel(observation_space, action_space)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if trained_model is not None:
            self.load_model(trained_model)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def update_replay_memory(self, state,
                             action, reward,
                             new_state, done):
        self.replay_memory.append((state, action, reward, new_state, done))

    def train(self, done):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 0

        # Get a minibatch from replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        mem_states = torch.Tensor([s[0] for s in minibatch])
        mem_actions = torch.Tensor([s[1] for s in minibatch])
        mem_rewards = torch.Tensor([s[2] for s in minibatch])
        mem_new_states = torch.Tensor([s[3] for s in minibatch])
        mem_done = torch.Tensor([s[4] for s in minibatch]).type(torch.BoolTensor)

        with torch.no_grad():
            # Predict next action
            next_qs = self.target_model(mem_new_states)
            target_qs = self.model(mem_states)
            max_future_qs, _ = next_qs.max(1)
            new_qs = mem_rewards + self.discount * max_future_qs * ~mem_done
            target_qs[:, mem_actions.tolist()] = new_qs

        output = self.model(mem_states)
        loss = self.criterion(output, target_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network counter every episode
        if done:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            self.target_update_counter = 0

        return loss.item()

    def get_action(self, state: ndarray, training : bool):
        if random.random() > self.epsilon or not training:
            # Ask network for next action
            with torch.no_grad():
                qs_action = self.target_model(torch.Tensor(state))
                action_value, action_index = qs_action.max(1)
        else:
            # Get a random action
            action_index = random.randint(0, len(self.actions)-1)

        return action_index

    def save_model(self, file_name: str):
        torch.save(self.model.state_dict(), './agents/models/' + file_name)

    def load_model(self, file_name: str):
        self.model.load_state_dict(torch.load('./agents/models/' + file_name))
        self.target_model.load_state_dict(torch.load('./agents/models/' + file_name))