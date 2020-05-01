from agents.agent import Agent
import random
import numpy as np
from collections import deque
from numpy import ndarray

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from agents.replay_memory import Replay_Buffer

REPLAY_MEMORY_SIZE = 128
MIN_REPLAY_MEMORY_SIZE = 128
MINIBATCH_SIZE = 128
SEED=420

UPDATE_TARGET_EVERY = 1


class DQNModel(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQNModel, self).__init__()

        self.input_size = 1
        for dim in observation_space:
            self.input_size *= dim

        self.convlayer1 = nn.Conv2d(3, 64, 2)
        self.convlayer2 = nn.Conv2d(64, 64, 2)
        self.convlayer3 = nn.Conv2d(64, 32, 2)

        self.layer1 = nn.Linear(7072, 4)

    def forward(self, state):
        x = F.leaky_relu(self.convlayer1(state.view(-1, 3, 16, 20)))
        x = F.leaky_relu(self.convlayer2(x))
        x = F.leaky_relu(self.convlayer3(x))
        output = self.layer1(x.view(-1, 7072))
        return output

class DQN_Agent(Agent):
    def __init__(self, epsilon=0.1, discount = 0.90, learning_rate=1e-4):
        super(DQN_Agent, self).__init__("dqn_agent")
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount

        self.replay_memory = Replay_Buffer(REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, SEED)


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
        self.replay_memory.add_experience(state, action, reward, new_state, done)

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.replay_memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones

    def train(self, done):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 0

        # Get a minibatch from replay memory
        states, actions, rewards, next_states, dones = self.sample_experiences()

        with torch.no_grad():
            # Predict next action
            next_qs = self.target_model(next_states).max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.discount * next_qs * (1 - dones))
        Q_expected = self.model(states).gather(1, actions.long())

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.01)
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

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
                qs_action = self.target_model(state)
                action_value, action_index = qs_action.max(1)
        else:
            # Get a random action
            action_index = random.randint(0, len(self.actions)-1)
        return action_index

    def save_model(self, file_name: str):
        torch.save(self.model.state_dict(), './' + file_name)

    def load_model(self, file_name: str):
        self.model.load_state_dict(torch.load('./agents/models/' + file_name, map_location=torch.device('cpu')))
        self.target_model.load_state_dict(torch.load('./agents/models/' + file_name, map_location=torch.device('cpu')))