import os
import numpy as np
import pybullet_multigoal_gym as pmg
import matplotlib.pyplot as plt
# pmg.envs.print_id()

env = pmg.make('KukaRobotiqGripHierPickAndPlaceRenderSparseImageObsEnv-v0')
obs = env.reset()
while True:
    time_done = False
    while not time_done:
        high_level_action = env.high_level_action_space.sample()
        env.set_sub_goal(high_level_action)
        sub_goal_done = False
        while not sub_goal_done and not time_done:
            action = env.low_level_action_space.sample()
            obs, reward, time_done, info = env.step(action)
            sub_goal_done = info['sub_goal_achieved']
            # plt.imshow(obs['desired_sub_goal_image'])
            # plt.pause(0.00001)
            # plt.imshow(obs['achieved_sub_goal_image'])
            # plt.pause(0.00001)
        if time_done:
            obs = env.reset()
            print(obs['desired_final_goal_ind'])
