import os
import gym
import numpy as np
import pybullet_multigoal_gym
import matplotlib.pyplot as plt
# print(pybullet_multigoal_gym.envs.get_id())

env = gym.make('KukaSlideRenderDenseEnv-v0')
obs = env.reset()
time_done = False
while True:
    action = env.action_space.sample()
    obs, reward, time_done, info = env.step(action)
    # plt.imshow(obs['observation'])
    # plt.pause(0.00001)
    if time_done:
        env.reset()
