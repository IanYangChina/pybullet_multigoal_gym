import os
import numpy as np
import time
import pybullet_multigoal_gym as pmg

num_episodes = 160
env = pmg.make_env(task='block_rearrange',
                   gripper='parallel_jaw',
                   num_block=4,
                   render=True,
                   binary_reward=True,
                   joint_control=True,
                   max_episode_steps=50,
                   image_observation=False,
                   use_curriculum=True,
                   num_goals_to_generate=num_episodes)

# env = pmg.make('KukaParallelGripBlockStackRenderSparseEnv-v0')
# env.activate_curriculum_update()
obs = env.reset()
time_done = False
while True:
    time.sleep(0.05)
    action = env.action_space.sample()
    obs, reward, time_done, info = env.step(action)
    if time_done:
        env.reset()
    # env.reset()
