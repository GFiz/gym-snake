from gym.spaces import Box
import numpy as np
import random
import os


class Snake:
    def __init__(self, start):
        self.body = [start]

    def move(self, action):
        head = self.body[-1]
        head_x, head_y = head[0], head[1]
        if action == 'UP':
            new_head = [head_x-1, head_y]
            self.body.append(new_head)
            self.body.pop(0)
        if action == 'DOWN':
            new_head = [head_x+1, head_y]
            self.body.append(new_head)
            self.body.pop(0)
        if action == 'LEFT':
            new_head = [head_x, head_y-1]
            self.body.append(new_head)
            self.body.pop(0)
        if action == 'RIGHT':
            new_head = [head_x, head_y+1]
            self.body.append(new_head)
            self.body.pop(0)
        


class SnakeGame:
    """Snake game representation.  
    """
    ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def __init__(self, height: int, width: int, start=None):
        """Initialize SnakeGame on a a board of given size with a start postion. A zero in the matrix indicates a "free position", a "-1" indicates a forbidden postion and "1" will indicate a pill position.

        Args:
            start ([int,int]): start position of snake head
            width (int): Width of board
            height (int): Height of board
        """
        if start == None:
            start = [random.randint(1, height - 2),
                     random.randint(1, width - 2)]

        self.board_shape = (height, width)
        self.score = 0
        self.snake = Snake(start)
        self.board = np.zeros((height, width))
        self.board[0, :] = -np.ones(width)
        self.board[height - 1, :] = -np.ones(width)
        self.board[:, 0] = -np.ones(height)
        self.board[:, width - 1] = -np.ones(height)
        self.board[start[0], start[1]] = -1
        self.active_game = True

    def step(self, action):
        tail = self.snake.body[0]
        self.snake.move(action)
        head = self.snake.body[-1]
        reward = self.board[head[0], head[1]]
        self.score += reward
        if reward == 0:
            self.board[tail[0], tail[1]] = 0
            self.board[head[0], head[1]] = -1
        if reward == -1:
            self.active_game = False
        if reward == 1:
            self.board[head[0], head[1]] = -1
            self.snake.body.insert(0,tail)
            self.insert_pill()
    
    def insert_pill(self):
        height, width = self.board_shape
        board = self.board
        if np.any(board == 0):
            x = random.randint(0, height-1)
            y = random.randint(0, width-1)
            while (board[x, y] == -1):
                x = random.randint(0, height-1)
                y = random.randint(0, width-1)
            board[x, y] = 1
        else:
            self.active_game = False

    def reset(self):
        self.__init__(*self.board_shape)
        self.insert_pill()

    def observation(self):
        board_features = self.board
        head = self.snake.body[-1]
        head_features = np.zeros(self.board_shape)
        head_features[head[0], head[1]] = 1
        combined_features = np.concatenate((board_features, head_features))
        return combined_features
