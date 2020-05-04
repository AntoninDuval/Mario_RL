import os
import sys
import time
import datetime
import math
import torch
import numpy as np

from environment.environment import Environment
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQN_Agent
from agents.A2C import A2C_Agent

TRAINING = False
SAVE_MODEL = False
VISUALIZE = True
N_EPOCHS = 200
N_STEPS = 10000

use_model = None
use_model = "MVP_model.h5"

def main():
    # Check if the ROM is given through argv
    filename = './Super_Mario_Land_World.gb'
    env = Environment(filename, max_steps=N_STEPS, visualize=VISUALIZE)
    env.start()
    agent = DQN_Agent(discount=0.9, epsilon=0.9, learning_rate=1e-5)
    avg_loss = None
    agent_is_setup = False
    min_epsilon = 0.05
    max_epsilon = 0.05


    for episode in range(N_EPOCHS):
        print("\n ", "=" * 50)
        env.reset()
        state = torch.Tensor(env.obs())
        old_state = state
        old_old_state = state
        is_a_released = torch.ones(1)
        states = [torch.cat((state, old_state, old_old_state), 0).view(3, 16, 20), is_a_released]
        episode_reward = 0

        if not agent_is_setup:
            agent.setup(env.observation_space, env.action_space, use_model)
            agent_is_setup = True

        for steps in range(N_STEPS):
            # Get action from agent
            actions = agent.get_action(states, TRAINING)
            new_state, reward, done = env.step(actions)

            #env.print_obs(new_state.numpy().astype(int))

            if actions[1] == 0:
                is_a_released = torch.zeros(1)
            else:
                is_a_released = torch.ones(1)

            if steps+1 == N_STEPS:
                done = True

            episode_reward += reward

            new_states = [torch.cat((new_state, states[0][0,:,:], states[0][1,:,:]),0).view(3,16,20), is_a_released]

            agent.update_replay_memory(states, actions, reward, new_states, done)

            # Train the neural network
            if TRAINING:
                loss = agent.train(done)
                if avg_loss is None:
                    avg_loss = loss
                else:
                    avg_loss = 0.99 * avg_loss + 0.01 * loss
            else:
                avg_loss = 0

            states = new_states

            if (steps + 1) % 20 == 0:
                print("\rAverage loss : {:.5f} --".format(avg_loss), "Episode rewards: {} --".format(episode_reward),
                      "epochs {}/{} --".format(episode, N_EPOCHS),
                      "steps {}/{}".format(steps + 1, N_STEPS),
                      end="")
            if done:
                print("\n", env.level_progress_max)
                break

        agent.epsilon = max(min_epsilon, min(max_epsilon, 1.0 - math.log10((episode + 1) / 5)))
    if SAVE_MODEL and TRAINING:
        date = datetime.datetime.now()
        model_name = str(date.day) + '_' + str(date.month) + '_' + str(date.hour) + '_' + agent.name + '.h5'
        agent.save_model(model_name)

    env.stop()

if __name__ == '__main__':
    main()
