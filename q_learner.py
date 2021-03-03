import gym
import pybullet_envs
import pybullet as p

import time, math

# Reference: https://medium.com/@flomay/using-q-learning-to-solve-the-cartpole-balancing-problem-c0a7f47d3f9d
# https://github.com/isaac-223/CartPole-v0-using-Q-learning-SARSA-and-DNN/blob/master/Qlearning_for_cartpole.py

import numpy as np

class CartPoleQAgent():
    def __init__(self, buckets=(3, 3, 6, 6),
                 num_episodes=500, min_lr=0.1,
                 min_epsilon=0.1,
                 discount=1.0, decay=25,
                 environment='CartPole-v0'):
        self.buckets      = buckets
        self.num_episodes = num_episodes
        self.min_lr       = min_lr
        self.min_epsilon  = min_epsilon
        self.discount     = discount
        self.decay        = decay
        self.env          = gym.make(environment)

        # [position, velocity, angle, angular velocity]
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]

        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize(self, obs):
        disc = []
        for i, o in enumerate(obs):
            l = self.lower_bounds[i]
            u = self.upper_bounds[i]
            b = self.buckets[i]
            scaling = ((o + abs(l)) / (u - l))
            new_o = int(round((b - 1) * scaling))
            new_o = min(b - 1, max(0, new_o))
            disc.append(new_o)
        return tuple(disc)

    def choose_action(self, state):
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def update_q(self, state, action, reward, new_state):
        self.Q[state][action] += self.learning_rate * (reward + self.discount * np.max(self.Q[new_state]) - self.Q[state][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self):
        for e in range(self.num_episodes):
            current_state = self.discretize(self.env.reset())

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state

        print('Finished training!')

    def run(self):
        self.env = gym.wrappers.Monitor(self.env,'cartpole')
        t = 0
        done = False
        current_state = self.discretize(self.env.reset())
        while not done:
                self.env.render()
                t = t+1
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)
                current_state = new_state

        return t

if __name__ == "__main__":
    agent = CartPoleQAgent()
    agent.train()
    t = agent.run()
    print("Time", t)

# p.connect(p.DIRECT)
#
# for _ in range(10):
#     env.reset()
#     # for _ in range(1000000):
#     for _ in range(1000):
#         env.render()
#         action = env.action_space.sample()
#         obs, reward, done, _ = env.step(action)
#         time.sleep(1/60)
#         if done:
#             break
#     print(reward)
