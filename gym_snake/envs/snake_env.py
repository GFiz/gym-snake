import gym
import numpy as np
from gym_snake.envs.snake_game import SnakeGame
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Discrete, Box

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, height, width):
        self.game_object = SnakeGame(height, width)
        self.game_object.insert_pill()
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=-1, high=1, shape=self.game_object.observation().shape)

    def get_keys_to_action(self):
        keys_to_action = {(ord('w'),):0, (ord('s'),): 1, (ord('a'),): 2, (ord('d'),): 3}
        return keys_to_action

    def observation(self):
        return self.game_object.observation()
        
    def step(self, a):
        game = self.game_object
        action = game.ACTIONS[a]
        old_score = game.score
        game.step(action)
        info = {'ACTION_NAME': action}
        return game.observation(), game.score - old_score, not game.active_game, info

    def reset(self):
        self.game_object.reset()
        return self.game_object.observation()

    def render(self, mode='human'):
        game = self.game_object
        if mode == 'human':
            board = np.zeros(game.board_shape)
            board[:, :] = game.board
            head = game.snake.body[-1]
            board[head[0], head[1]] = 2
            return board
        if mode == 'rgb_array':
            height, width = game.board_shape
            board_rgb = np.zeros((height, width, 3))
            head = game.snake.body[-1]
            for i in range(height):
                for j in range(width):
                    if game.board[i, j] == -1:
                        board_rgb[i, j, :] = [255, 0, 0]
                    if game.board[i, j] == 0:
                        board_rgb[i, j, :] = [0, 255, 0]
                    if game.board[i, j] == 1:
                        board_rgb[i, j,:] = [0, 0, 255]
                    if i == head[0] and j == head[1]:
                        board_rgb[i, j,:] = [255, 255, 0]
            return board_rgb.astype(np.uint8)

    def close(self):
        ...
