import gym
import numpy as np
from gym_snake.envs.snake_game import SnakeGame
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Discrete, Box

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, height, width, num_players):
        self.game_object = SnakeGame(height, width, num_players)
        self.game_object.insert_pill()
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=-1, high=1, shape=self.game_object.observation().shape)

    def get_keys_to_action(self):
        keys_to_action = {(ord('w'),):0, (ord('s'),): 1, (ord('a'),): 2, (ord('d'),): 3}
        return keys_to_action

    def observation(self):
        return self.game_object.observation()

    def step(self, a, playerID = 0):
        game = self.game_object
        action = game.ACTIONS[a]
        old_score = game.snakes[playerID].score
        game.step(action,playerID)
        new_score = game.snakes[playerID].score
        info = {'ACTION_NAME': action}
        return game.observation(), new_score - old_score, not game.active_game, info

    def reset(self):
        self.game_object.reset()
        return self.game_object.observation()

    def render(self, mode='human'):
        game = self.game_object
        if mode == 'human':
            board = np.zeros(game.board_shape)
            board[:, :] = game.board
            for snake in game.snakes:
                head = snake.body[-1]
                board[head[0], head[1]] = (snake.playerID+1)*11
            return board
        if mode == 'rgb_array':
            height, width = game.board_shape
            board_rgb = np.zeros((height, width, 3))
            heads = [ snake.body[-1] for snake in game.snakes]
            for i in range(height):
                for j in range(width):
                    if game.board[i, j] == -1:
                        board_rgb[i, j, :] = [255, 0, 0]
                    if game.board[i, j] == 0:
                        board_rgb[i, j, :] = [0, 255, 0]
                    if game.board[i, j] == 1:
                        board_rgb[i, j,:] = [0, 0, 255]
            for head in heads:
                board_rgb[head[0],head[1]] = [255,255,0]
            return board_rgb.astype(np.uint8)

    def close(self):
        ...
