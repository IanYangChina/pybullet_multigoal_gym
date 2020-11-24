import os
import gym
import numpy as np
import pybullet_multigoal_gym
import matplotlib.pyplot as plt


env = gym.make('MGKuka2ObjPyBulletEnv-v0')
# print(env.env.robot.ordered_joint_names)
obs = env.reset()  # should return a state vector if everything worked
t = 0
while True:
    t += 1
    action = env.action_space.sample() * 0
    # action[2] = -0.1
    # rgb = env.render(mode='rgb_array')
    # plt.imshow(rgb)
    # plt.show()
    obs, reward, done, info = env.step(action)
    if t % 100 == 0:
        print("reset")
        env.reset()
