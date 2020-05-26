import gym
import numpy as np
from gym_snake.envs.snake_game import SnakeGame
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Discrete

class SnakeEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self,height,width):
      self.game_object = SnakeGame(height, width)
      self.game_object.insert_pill()
      self.action_space = Discrete(4)
      self.observation_space = self.game_object.observation()
  def step(self, a):
      game = self.game_object
      action = game.ACTIONS[a]
      old_score = game.score
      game.step(action)
      return game.observation(), old_score-game.score, game.active_game
  def reset(self):
      self.game_object.reset()
      return self.game_object.observation()
  def render(self, mode='human'):
    game = self.game_object
    if mode == 'human':
        board = np.zeros(game.board_shape)
        board[:,:] = game.board
        head = game.snake[-1]
        board[head[0], head[1]] = 3
        print(board)
    ...
  def close(self):
    ...