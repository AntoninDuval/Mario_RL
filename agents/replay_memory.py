import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class ReplayMemory(Dataset):
    """Rolling replay memory: arrays are initialized at full length but elements are inserted at current head cursor.
    Writing is rolling, meaning that when head reaches the maximum length of arrays, it cycles back to the beginning
    and overwrites old elements."""

    def __init__(self, max_len, observation_space, action_space):
        self.max_len = max_len
        self.observation_space = observation_space
        self.action_space = action_space

        # Initializing zero-arrays of full length
        self.states = torch.zeros(max_len, dtype=torch.float32)
        self.actions = torch.zeros(max_len, dtype=torch.float32)
        self.rewards = torch.zeros(max_len, dtype=torch.float32)
        self.new_states = torch.zeros(max_len, dtype=torch.float32)
        self.dones = torch.zeros(max_len, dtype=bool)

        # None of them require gradient computation, save some resources:
        self.states.requires_grad = False
        self.actions.requires_grad = False
        self.rewards.requires_grad = False
        self.new_states.requires_grad = False
        self.dones.requires_grad = False

        # Current writing head in memory
        self.head = 0

        # Current quantity of elements added to the memory ; will saturate at 'max_len'
        self.fill = 0

    def __len__(self):
        """Retrieves the length of the replay memory (only available entries)"""
        return self.fill

    def __getitem__(self, idx):
        """Retrieves item at given index or list of indices.
        :param idx: integer, list of integers or integer tensor
        :return tuple containing the slices of replay memory arrays at given indices"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.states[idx], self.actions[idx], self.rewards[idx], self.new_states[idx], self.dones[idx]

    def random_access(self, n):
        """Retrieves n random elements from replay memory.
        :param n: number of elements to retrieve
        :return tuple : states, agent_states, actions, rewards, new_states, new_agent_states, dones ; random corresponding slices of replay memory arrays"""
        indices = random.sample(range(len(self)), n)
        return self[indices]

    def _extend_unsafe(self, state, action, reward, new_state, done):
        """PRIVATE | Extends current memory with given arrays, up to 'add' elements.
        Unsafe: no length check. This function shouldn't be called from outside this class.
        :param states: world observations
        :param actions: taken actions
        :param rewards: received rewards
        :param new_states: world observations after taking action
        :param done: is episode done?
        :param add: number of elements to add, from the beginning of passed arrays
        """
        begin = self.head
        end = begin + 1
        self.states[begin:end] = torch.from_numpy(state)
        self.actions[begin:end] = torch.from_numpy(action)
        self.rewards[begin:end] = torch.from_numpy(reward)
        self.new_states[begin:end] = torch.from_numpy(new_state)
        self.dones[begin:end] = done

    def extend(self, state, action, reward, new_state, done):
        """ Extends the replay memory with all given entries. Memory writing is rolling: when reaching
        saturation, old elements are overwritten automatically.
        :param states: 				ndarray[ _ , observation_space]		world observations
        :param actions: 			ndarray[ _ , action_space]			taken actions
        :param rewards: 			ndarray[ _ ]						received rewards
        :param new_states: 			ndarray[ _ , observation_space] 	world observations after taking action
        :param done: 				bool								is episode done?
        """

        # Add those elements
        self._extend_unsafe(state, action, reward, new_state, done)

        # Updating fill (how much space is left in the replay memory, before saturation)
        self.fill = max(self.fill, min(self.max_len, self.head + 1))

        # Updating head position, putting it back to 0 if reached max length
        self.head = (self.head + 1) % self.max_len
