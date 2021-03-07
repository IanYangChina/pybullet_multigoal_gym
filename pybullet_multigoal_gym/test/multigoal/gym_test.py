import os
import numpy as np
import pybullet_multigoal_gym as pmg
# pmg.envs.print_id()

env = pmg.make('KukaParallelGripSlideRenderSparseEnv-v0')
obs = env.reset()
time_done = False
while True:
    # action = env.action_space.sample()
    # obs, reward, time_done, info = env.step(action)
    # if time_done:
    #     env.reset()
    env.reset()