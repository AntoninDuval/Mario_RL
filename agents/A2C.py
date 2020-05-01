from agents.agent import Agent

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from numpy import ndarray
import torch.optim as optim


class ActorCritic(nn.Module):
    def __init__(self, observation_space, num_actions, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.input_size = 1
        for dim in observation_space:
            self.input_size *= dim

        self.num_actions = num_actions[0]
        self.critic_linear1 = nn.Linear(self.input_size, 256)
        self.critic_linear2 = nn.Linear(256, 1)

        self.actor_linear1 = nn.Linear(self.input_size, 256)
        self.actor_linear2 = nn.Linear(256, self.num_actions)

    def forward(self, state):
        value = F.relu(self.critic_linear1(state.view(-1, self.input_size)))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state.view(-1, self.input_size)))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

class A2C_Agent(Agent):
    def __init__(self, epsilon=0.1, discount = 0.99, learning_rate=1e-4):
        super(A2C_Agent, self).__init__("a2C_agent")
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount

        self.model = None
        self.optimizer = None

        self.state = None

    def setup(self, observation_space, action_space, trained_model=None):
        self.model = ActorCritic(observation_space, action_space)
        self.action_space = action_space

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if trained_model is not None:
            self.load_model(trained_model)


    def get_action(self, state: ndarray, training : bool):
        value, policy_dist = self.model.forward(torch.Tensor(state))

        print('Policy dist :', policy_dist.detach().numpy())

        cat = Categorical(policy_dist)
        action = cat.sample()

        return action, cat.log_prob(action), cat.entropy().mean(), value

    def train(self, values, rewards, log_probs, Qval, entropy_term):
        print("Values :", values)
        print("log_probs", log_probs)
        print("Qval", Qval)
        print("entropty_term", entropy_term)
        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + self.discount * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
        ac_loss.requires_grad = True

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()

        return ac_loss.item()
