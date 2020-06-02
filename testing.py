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
from gym.utils.play import play
import matplotlib
import pickle
matplotlib.use('TkAgg')

NUM_EPISODES = int(5e3)
BATCH_SIZE = 32
NUM_EPOCHS = 1
penalty = 0.1
env=gym.make('snake-v0', height=50, width=50, num_players=2)
play(env, fps=10, zoom=20.)
