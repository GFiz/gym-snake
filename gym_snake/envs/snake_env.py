import gym
import numpy as np
from gym_snake.envs.snake_game import SnakeGame
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Discrete, Box, Tuple


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, height, width, num_players):
        self.game_object = SnakeGame(height, width, num_players)
        self.game_object.insert_pill()
        self.action_space = Tuple((Discrete(4), Discrete(num_players)))
        self.observation_space = Box(
            low=-2, high=2, shape=self.game_object.observation(0).shape)

    def observation(self,playerID):
        return self.game_object.observation(playerID)

    def step(self, a_tuple):
        game = self.game_object
        action = game.ACTIONS[a_tuple[0]]
        playerID = a_tuple[1]
        snake = game.snakes[playerID]
        old_score = snake.score
        game.step((action, playerID))
        new_score = snake.score
        info = {'ACTION_NAME': (action, playerID)}
        return game.observation(playerID), new_score - old_score, snake.done, info

    def reset(self):
        self.game_object.reset()
    

    def render(self, mode='human'):
        game = self.game_object
        if mode == 'human':
            board = game.get_board()
            for snake in game.snakes:
                head = snake.body[-1]
                board[head[0], head[1]] = (snake.playerID+1)*11
            return board
        if mode == 'rgb_array':
            height, width = game.board_shape
            board_rgb = np.zeros((height, width, 3))
            heads = [snake.body[-1] for snake in game.snakes]
            for i in range(height):
                for j in range(width):
                    if game.board[i, j] == -1:
                        board_rgb[i, j, :] = [255, 0, 0]
                    if game.board[i, j] == 0:
                        board_rgb[i, j, :] = [0, 255, 0]
                    if game.board[i, j] == 1:
                        board_rgb[i, j, :] = [0, 0, 255]
            for head in heads:
                board_rgb[head[0], head[1]] = [255, 255, 0]
            return board_rgb.astype(np.uint8)

    def close(self):
        ...
