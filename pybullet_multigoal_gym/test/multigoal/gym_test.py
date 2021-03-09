import os
import numpy as np
import time
import pybullet_multigoal_gym as pmg

# env = pmg.make_env(task='slide',
#                    gripper='parallel_jaw',
#                    render=True,
#                    binary_reward=True,
#                    max_episode_steps=50,
#                    image_observation=False,
#                    depth_image=False,
#                    goal_image=False)

env = pmg.make('KukaParallelGripBlockStackRenderSparseEnv-v0')
obs = env.reset()
time_done = False
while True:
    time.sleep(1)
    # action = env.action_space.sample()
    # obs, reward, time_done, info = env.step(action)
    # if time_done:
    #     env.reset()
    env.reset()
