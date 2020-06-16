import numpy as np
from tqdm import tqdm
import gym
import gym_snake
from gym_snake.envs.core_agents import QFunction, Policy, EpsilonRegime
from gym_snake.envs.agents import DQNAgent, QNet
from gym.utils import play
import matplotlib
import pickle
matplotlib.use('TkAgg')

NUM_EPISODES = int(5e3)
BATCH_SIZE = 32
NUM_EPOCHS = 1
penalty = 0.1
env = gym.make('snake-v0', height=20, width=20, num_players=2)

def epsilon_decay(n):
    return 1 / (n / 100 + 1)


regime = EpsilonRegime(1, epsilon_decay)

agent=DQNAgent(env, QNet(env, 128))
print(agent.evaluate(100,100))
play(env)

