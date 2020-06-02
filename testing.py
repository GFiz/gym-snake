import numpy as np
import pandas as pd
import seaborn
import tkinter
from pathlib import Path
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import gym
import gym_snake
from gym_snake.envs.core_agents import QFunction, Policy, EpsilonRegime
from gym_snake.envs.agents import DQNAgent, QNet
import matplotlib
import pickle
matplotlib.use('TkAgg')

NUM_EPISODES = int(5e3)
BATCH_SIZE = 32
NUM_EPOCHS = 1
penalty = 0.1
env = gym.make('snake-v0', height=6, width=6)
agent = DQNAgent(env, QNet(env, 32), mem_cap=int(1e5))


def epsilon_decay(n):
    return 1 / (n / 100 + 1)


regime = EpsilonRegime(1, epsilon_decay)
# losses, scores = agent.train(
#     NUM_EPISODES,
#     penalty,
#     BATCH_SIZE,
#     NUM_EPOCHS,
#     regime)
# agent.save('dqnagent2.pickle')

with open('dqnagent2.pickle', 'rb') as handle:
    agent2 = pickle.load(handle)

print(agent2.evaluate(1000, 50))
agent2.playthrough(100)