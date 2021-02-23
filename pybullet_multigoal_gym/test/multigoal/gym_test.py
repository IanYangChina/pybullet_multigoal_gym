import os
import numpy as np
import pybullet_multigoal_gym as pmg
import matplotlib.pyplot as plt
# pmg.envs.print_id()

env = pmg.make('KukaParallelGripPushRenderSparseDepthImageObsImageGoalEnv-v0')
env.env.visualize_target = False
obs = env.reset()
time_done = False
f, axarr = plt.subplots(1, 2)
while True:
    action = env.action_space.sample()
    obs, reward, time_done, info = env.step(action)
    axarr[0].imshow(obs['desired_goal_img'][:, :, :-1])
    axarr[1].imshow(obs['achieved_goal_img'][:, :, :-1])
    plt.pause(0.00001)
    if time_done:
        env.reset()