import tkinter
from pathlib import Path
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import gym
import gym_snake
from gym_snake.envs.core_agents import QFunction, Policy
from gym_snake.envs.agents import DQNAgent, QNet
import matplotlib
matplotlib.use('TkAgg')


NUM_EPISODES = int(1e2)
BATCH_SIZE=8
NUM_EPOCHS = 1
penalty = 0.1
env = gym.make('snake-v0', height=5, width=5)
agent=DQNAgent(env, QNet(env, 32),mem_cap=int(1e5))
agent.Qpred.neural_net.summary()

agent.train(NUM_EPISODES, penalty, BATCH_SIZE, NUM_EPOCHS)
agent.playthrough(200)
