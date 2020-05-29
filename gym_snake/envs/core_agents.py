from gym_snake.envs.snake_env import SnakeEnv
from gym_snake.envs.snake_game import SnakeGame
from gym import Env
from collections import namedtuple
import random
import numpy as np


class QFunction:
    def __init__(self, env: Env, **kwargs):
        self.enviroment = env
        ...

    def evaluate(self, observation, action):
        raise NotImplementedError

    def evaluate_max(self, observation):
        evaluations = [self.evaluate(observation, action)
                       for action in self.enviroment.action_space]
        return np.max(evaluations)

    def update(self, **kwargs):
        raise NotImplementedError


class Policy:
    def __init__(self, env: Env):
        self.environment = env
        self.actions = env.action_space
        ...

    def follow(self, observation, epsilon: float):
        p = random.uniform(0, 1)
        if p >= epsilon:
            raise NotImplementedError
        else:
            random_action = self.environment.action_space.sample()
            return random_action

    def update(self, **kwargs):
        raise NotImplementedError


class EpsilonRegime:
    def __init__(self, initial_value: float, decay_function):
        self.initial_value = initial_value
        self.func = decay_function

    def get_epsilon(self, iter_num: int):
        return self.func(iter_num) * self.initial_value


Experience = namedtuple(
    'Experience', ['state','action', 'reward', 'successor_state', 'done'])


class ExperienceMemory:
    def __init__(self, capacity=int(1e6)):
        assert type(capacity) == int 
        self._capacity = capacity
        self.idx = int(0)
        self.memory = []

    def capacity(self):
        return self._capacity

    @property
    def idx(self)-> int:
        return self._idx

    @idx.setter
    def idx(self, value:int):
        self._idx = value

    def store(self, exp: Experience):
        if len(self.memory) < self.capacity():
            self.memory.append(exp)
        else:
            self.memory[self.idx] = exp
        self.idx += 1
        self.idx %= self.capacity()
    
    def sample(self, batch_size):
        assert 0 <= batch_size <= len(self.memory)
        return random.sample(self.memory, batch_size)
    def reset(self):
        self.memory = []
        self.idx = 0