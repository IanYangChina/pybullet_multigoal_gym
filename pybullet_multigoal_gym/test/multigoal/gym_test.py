import os
import gym
import numpy as np
import pybullet_multigoal_gym
import matplotlib.pyplot as plt
print(pybullet_multigoal_gym.envs.get_id())

env = gym.make('KukaSlideRenderSparseEnv-v0')
obs = env.reset()
t = 0
while True:
    t += 1
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
