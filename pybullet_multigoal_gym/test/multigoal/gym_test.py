import os
import numpy as np
import pybullet_multigoal_gym as pmg
import matplotlib.pyplot as plt
# pmg.envs.print_id()

env = pmg.make('KukaParallelGripPickAndPlaceRenderDenseImageObsEnv-v0')
obs = env.reset()
time_done = False
while True:
    action = env.action_space.sample()
    action *= 0
    action -= 1
    action[2] = 1
    obs, reward, time_done, info = env.step(action)
    plt.imshow(obs['observation'])
    plt.pause(0.00001)
    if time_done:
        env.reset()
