import tkinter
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import gym
import gym_snake
from gym_snake.envs.core_agents import QFunction, Policy
from gym_snake.envs.agents import DQNAgent, QNet, ExperienceMemory, Experience
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

def playthrough(agent:Policy):
    env = gym.make('snake-v0', height=6, width=6)
    env.reset()
    game = env.game_object
    data = [game.get_board()]
    done = False
    counter = 0

    while (not done and counter<=200):
        action = agent.follow(game.observation(),0)
        _, _, done, _ = env.step(action)
        data.append(game.get_board())
        counter+=1
    
    fig = plt.figure()
    plot = plt.matshow(data[0], fignum=0)
    def init():
        plot.set_data(data[0])
        return [plot]
    def update(j):
        plot.set_data(data[j])
        return [plot]
    anim = FuncAnimation(fig, update, init_func = init, frames=len(data), interval = 300, blit=True)
    plt.draw()
    plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.close(fig)

NUM_EPISODES = int(1e5)
BATCH_SIZE = 8
penalty = 0.1
env = gym.make('snake-v0', height=6, width=6)
agent=DQNAgent(env, QNet(env, 32),mem_cap=int(1e5))
agent.Qpred.neural_net.summary()
memory=agent.memory
scores = []
for i in tqdm(range(NUM_EPISODES)):
    env.reset()
    done = False
    counter = 0
    while not done:
        observation = env.game_object.observation()
        action = agent.follow(observation, 1/(i/300+1))
        next_observation, reward, done, info = env.step(action)
        memory.store(Experience(observation, action, reward-penalty, next_observation, done))
        if len(memory.memory) >= BATCH_SIZE:
            agent.update(BATCH_SIZE, 1)
    scores.append(env.game_object.score)
    if (i + 1) % 1000 == 0:
        print(np.mean(scores))
        playthrough(agent)
        scores = []

       

