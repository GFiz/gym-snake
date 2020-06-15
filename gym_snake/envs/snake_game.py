from gym.spaces import Box
import numpy as np
import random
import os


class Snake:
    def __init__(self, start, playerID):
        self.body = [start]
        self.score = 0
        self.playerID = playerID
        self.done = False

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

    def __init__(self, height: int, width: int, num_players: int):
        """Initialize SnakeGame on a a board of given size with a start postion. A zero in the matrix indicates a "free position", a "-1" indicates a forbidden postion and "1" will indicate a pill position.

        Args:
            start ([int,int]): start position of snake head
            width (int): Width of board
            height (int): Height of board
        """
        start_points = random.sample([(i, j) for i in range(
            1, height-1) for j in range(1, width-1)], num_players)
        self.board_shape = (height, width)
        self.snakes = [Snake(start, i) for i, start in enumerate(start_points)]
        self.board = np.zeros((height, width))
        self.board[0, :] = -np.ones(width)
        self.board[height - 1, :] = -np.ones(width)
        self.board[:, 0] = -np.ones(height)
        self.board[:, width - 1] = -np.ones(height)
        for start in start_points:
            self.board[start[0], start[1]] = -1
        self.done = False

    def step(self, action_tuple):
        action = action_tuple[0]
        playerID = action_tuple[1]
        snake = self.snakes[playerID]
        tail = snake.body[0]
        snake.move(action)
        head = snake.body[-1]
        reward = self.board[head[0], head[1]]
        snake.score += reward
        if reward == 0:
            self.board[tail[0], tail[1]] = 0
            self.board[head[0], head[1]] = -1
        if reward == -1:
            snake.done = True
            self.board[tail[0], tail[1]] = 0
        if reward == 1:
            self.board[head[0], head[1]] = -1
            snake.body.insert(0, tail)
            self.insert_pill()
        self.done = np.all([snake.done for snake in self.snakes])

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
        self.__init__(*self.board_shape, num_players=len(self.snakes))
        self.insert_pill()

    def get_scores(self):
        return np.array([snake.score for snake in self.snakes])

    def get_board(self):
        array = np.zeros(self.board_shape)
        array[:, :] = self.board
        return array

    def observation(self, playerID):
        all_features = self.get_board()
        for snake in self.snakes:
            if snake.playerID == playerID:
                head = snake.body[-1]
                all_features[head[0], head[1]] = 2
            else:
                head = snake.body[-1]
                all_features[head[0], head[1]] = -2
        return all_features

    
        
