import os
import gym
import numpy as np
import pybullet_multigoal_gym
import matplotlib.pyplot as plt


env = gym.make('MGKuka2ObjPyBulletEnv-v0')
# print(env.env.robot.ordered_joint_names)
state = env.reset()  # should return a state vector if everything worked
i = 0
while True:
    i += 1
    action = env.action_space.sample()
    action[:-1] *= 0
    # action[2] = -0.1
    # rgb = env.render(mode='rgb_array')
    # plt.imshow(rgb)
    # plt.show()
    env.step(action)
    if i % 1000 == 0:
        env.reset()
