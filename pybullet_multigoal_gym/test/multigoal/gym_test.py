import os
import gym
import numpy as np
import pybullet_multigoal_gym
import matplotlib.pyplot as plt
print(pybullet_multigoal_gym.envs.get_id())

env = gym.make('KukaReachRenderSparseEnv-v0')
obs = env.reset()
t = 0
while True:
    t += 1
    action = env.action_space.sample()*0
    action[0] = -1.0
    action[1] = 1.0
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
