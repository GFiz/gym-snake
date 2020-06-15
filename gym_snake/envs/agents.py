from gym import Env
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import random
import pickle
from gym_snake.envs.core_agents import Policy, QFunction, Experience, ExperienceMemory, EpsilonRegime
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten
from keras.callbacks import TensorBoard
import tensorflow as tf


class QNet(QFunction):
    def __init__(self, env: Env, hidden_units: int, modelpath=None):
        if modelpath != None:
            self.neural_net = load_model(modelpath)
        else:
            inpTensor = Input(env.observation_space.shape)
            flatTensor = Flatten()(inpTensor)
            hiddenOut = Dense(hidden_units, activation='relu')(flatTensor)
            out = Dense(env.action_space[0].n)(hiddenOut)
            model = Model(inpTensor, out)
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.mean_squared_error)
            self.neural_net = model

    def evaluate(self, observation, done=False):
        if done:
            return 0
        evaluation = self.neural_net.predict(observation[np.newaxis, :])
        return evaluation

    def evaluate_max(self, observation, done: bool):
        evaluation = self.evaluate(observation, done)
        return np.max(evaluation)

    def update(self, X, Y, num_epochs, verbose=0):
        model = self.neural_net
        model.fit(x=X, y=Y, epochs=num_epochs, verbose=verbose)
        return model.history.history['loss']


class DQNAgent(Policy):
    def __init__(self, env: Env, Q: QNet, mem_cap=int(10e5)):
        super().__init__(env)
        self.Qpred = Q
        self.memory = ExperienceMemory(mem_cap)

    def follow(self, observation, epsilon: float):
        p = random.uniform(0, 1)
        if p >= epsilon:
            evaluation = self.Qpred.evaluate(observation)
            return np.argmax(evaluation)
        else:
            random_action = self.environment.action_space[0].sample()
            return random_action

    def update(self, batch_size, num_epochs, verbose=0):
        Q = self.Qpred
        batch = self.memory.sample(batch_size)
        X = np.concatenate([exp.state[np.newaxis, :] for exp in batch])
        Y = []
        for exp in batch:
            observation = exp.state
            next_obsevation = exp.successor_state
            action = exp.action
            done = exp.done
            target = Q.evaluate(observation)
            target[0, action] = exp.reward + \
                Q.evaluate_max(next_obsevation, done)
            Y.append(target[np.newaxis, :])
        Y = np.concatenate(Y).reshape((X.shape[0], 4))
        loss = Q.update(X, Y, num_epochs, verbose)
        return loss[0]

    def train(self, num_episodes, penalty, batch_size, num_epochs, regime: EpsilonRegime):
        env = self.environment
        memory = self.memory
        scores = []
        losses = []
        for i in tqdm(range(num_episodes)):
            env.reset()
            game = env.game_object
            game_over = False
            loss = []
            epsilon = regime.get_epsilon(i)
            while not game_over:
                for snake in game.snakes:
                    if not snake.done:
                        observation = game.observation(snake.playerID)
                        action = self.follow(observation, epsilon)
                        next_observation, reward, done, info = env.step((action,snake.playerID))
                        memory.store(Experience(observation, action,
                                                reward-penalty, next_observation, done))
                        if len(memory.memory) >= batch_size:
                            loss.append(self.update(batch_size, 1))
                    if len(loss) > 0:
                        losses.append(np.mean(loss))
                    else:
                        losses.append(np.inf)
                    scores.append(env.game_object.get_scores())
                game_over = game.done
        return losses, scores

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, -1)
