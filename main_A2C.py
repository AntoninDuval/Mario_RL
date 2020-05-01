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

TRAINING = True
SAVE_MODEL = False
VISUALIZE = True
N_EPOCHS = 10
N_STEPS = 200

use_model = None
#use_model = "20_4_13_collect_agent_memory.h5"


def main():

    # Check if the ROM is given through argv
    filename = './Super_Mario_Land_World.gb'

    env = Environment(filename, max_steps=N_STEPS, visualize=VISUALIZE)
    env.start()
    agent = A2C_Agent(discount=0.99, epsilon=0.9, learning_rate=1e-3)

    agent_is_setup = False

    entropy_term = 0
    all_rewards = []
    all_lengths = []
    average_lengths = []

    for episode in range(N_EPOCHS):
        print("\n ", "=" * 50)
        print("Epoch {}/{}".format(episode + 1, N_EPOCHS))
        env.reset()
        state = env.obs()

        log_probs = []
        values = []
        rewards = []

        if not agent_is_setup:
            agent.setup(env.observation_space, env.action_space, use_model)
            agent_is_setup = True

        for steps in range(N_STEPS):
            # Get action from agent
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action(state, TRAINING)

            value = value.detach().numpy()[0, 0]

            new_state, reward, done = env.step(action, steps)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy

            # Set obs to the new state
            state = new_state

            if done or steps == N_STEPS - 1:
                Qval, _ = agent.model.forward(torch.Tensor(new_state))
                Qval = Qval.detach().numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode,
                                                                                                               np.sum(
                                                                                                                   rewards),
                                                                                                               steps,
                                                                                                               average_lengths[
                                                                                                                   -1]))
                break

        print("Loss :", agent.train(values, rewards, log_probs, Qval, entropy_term))

    if SAVE_MODEL and TRAINING:
        date = datetime.datetime.now()
        model_name = str(date.day) + '_' + str(date.month) + '_' + str(date.hour) + '_' + agent.name + '.h5'
        agent.save_model(model_name)

    env.stop()

if __name__ == '__main__':
    main()
