import os
import numpy as np
import pybullet_multigoal_gym as pmg
import matplotlib.pyplot as plt
# pmg.envs.print_id()

env = pmg.make('KukaParallelGripPushRenderSparseDepthImageObsImageGoalEnv-v0')
obs = env.reset()
time_done = False
while True:
    action = env.action_space.sample()
    obs, reward, time_done, info = env.step(action)
    plt.imshow(obs['achieved_goal_img'].transpose(-1, 0, 1)[-1], cmap='Greys')
    plt.pause(0.00001)
    if time_done:
        env.reset()
