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
env=gym.make('snake-v0', height=6, width=6, num_players=2)
def epsilon_decay(n):
    return 1 / (n / 100 + 1)


regime = EpsilonRegime(1, epsilon_decay)

agent=DQNAgent(env, QNet(env, 128))
agent.train(20, 0.1, 64, 1, regime)
print(agent.evaluate(100,100))
agent.playthrough(50)

