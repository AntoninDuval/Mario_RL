import os
import sys
import time
import datetime
import math
import numpy as np

from environment.environment import Environment
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQN_Agent



TRAINING = True
SAVE_MODEL = False
VISUALIZE = True
N_EPOCHS = 200
N_STEPS = 200

use_model = None
#use_model = "20_4_13_collect_agent_memory.h5"


def main():

    # Check if the ROM is given through argv
    filename = './Super_Mario_Land_World.gb'

    env = Environment(filename, max_steps=N_STEPS, visualize=VISUALIZE)
    env.start()
    agent = DQN_Agent(discount=0.9, epsilon=0.9, learning_rate=1e-4)

    avg_loss = None
    agent_is_setup = False

    min_epsilon = 0.1
    max_epsilon = 1

    for episode in range(N_EPOCHS):
        print("\n ", "=" * 50)
        env.reset()
        state = env.obs()
        episode_reward = 0

        if not agent_is_setup:
            agent.setup(env.observation_space, env.action_space, use_model)
            agent_is_setup = True

        for steps in range(N_STEPS):
            # Get action from agent
            action = agent.get_action(state, TRAINING)
            new_state, reward, done = env.step(action, steps)
            print(reward)
            episode_reward += reward
            agent.update_replay_memory(state, action, reward, new_state, done)

            # Train the neural network
            if TRAINING:
                loss = agent.train(done)
                if avg_loss is None:
                    avg_loss = loss
                else:
                    avg_loss = 0.99 * avg_loss + 0.01 * loss
            else:
                avg_loss = 0
            # Set obs to the new state
            state = new_state

            if (steps + 1) % 20 == 0:
                print("\rAverage loss : {:.5f} --".format(avg_loss), "Episode rewards: {} --".format(episode_reward),
                      "epochs {}/{} --".format(episode, N_EPOCHS),
                      "steps {}/{}".format(steps+1, N_STEPS),
                      end="")
            if done:
                print("")
                print("Mario died")
                break
        agent.epsilon = max(min_epsilon, min(max_epsilon, 1.0 - math.log10((episode + 1) / 2)))
        print('\n', 'Epsilon :', agent.epsilon)
    if SAVE_MODEL and TRAINING:
        date = datetime.datetime.now()
        model_name = str(date.day) + '_' + str(date.month) + '_' + str(date.hour) + '_' + agent.name + '.h5'
        agent.save_model(model_name)

    env.stop()

if __name__ == '__main__':
    main()