import gym
import pybullet_envs
import pybullet as p

import time

p.connect(p.DIRECT)

# Possible Environments

# https://github.com/bulletphysics/bullet3/releases
# multi-agent envs: https://github.com/koulanurag/ma-gym
# env = gym.make('AntBulletEnv-v0')
# env = gym.make('AsteroidsNoFrameskip-v4')
# env = gym.make('ma_gym:Combat-v0')

env = gym.make('CartPole-v0')

for _ in range(10):
    env.reset()
    # for _ in range(1000000):
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        time.sleep(1/60)
        if done:
            break
    print(reward)
