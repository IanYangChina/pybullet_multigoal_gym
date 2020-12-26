import os
import gym
import numpy as np
import pybullet_multigoal_gym
import matplotlib.pyplot as plt
# print(pybullet_multigoal_gym.envs.get_id())

env = gym.make('KukaHierPickAndPlaceRenderSparseEnv-v0')
obs = env.reset()
t = 0
while True:
    t += 1
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # plt.imshow(obs['desired_sub_goal_image'])
    # plt.pause(0.00001)
    # plt.imshow(obs['achieved_sub_goal_image'])
    # plt.pause(0.00001)
    if done:
        env.reset()
