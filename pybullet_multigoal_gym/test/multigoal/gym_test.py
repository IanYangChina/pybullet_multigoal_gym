import os
import numpy as np
import time
import pybullet_multigoal_gym as pmg

num_episodes = 300
env = pmg.make_env(task='chest_pick_and_place',
                   gripper='parallel_jaw',
                   num_block=2,
                   render=True,
                   visualize_target=True,
                   binary_reward=True,
                   joint_control=False,
                   task_decomposition=True,
                   abstract_demonstration=True,
                   max_episode_steps=1000,
                   image_observation=False,
                   use_curriculum=False,
                   num_goals_to_generate=num_episodes)

# env = pmg.make('KukaParallelGripBlockStackRenderSparseEnv-v0')
# env.activate_curriculum_update()
obs = env.reset()
time_done = False
while not time_done:
    time.sleep(0.05)
    action = env.action_space.sample() * 0.0
    action = np.array([-0.1, 0.1, 0.1, 1.0])
    obs, reward, time_done, info = env.step(action)
    # if time_done:
    #     env.reset()
    # env.set_sub_goal(0)
    # print(env.desired_goal)
    # env.set_sub_goal(1)
    # print(env.desired_goal)
    # env.set_sub_goal(2)
    # print(env.desired_goal)
    # env.set_sub_goal(3)
    # print(env.desired_goal)
    # env.set_sub_goal(4)
    # print(env.desired_goal)
    # env.set_sub_goal(5)
    # print(env.desired_goal)
    # env.set_sub_goal(6)
    # print(env.desired_goal)
    # env.reset()
