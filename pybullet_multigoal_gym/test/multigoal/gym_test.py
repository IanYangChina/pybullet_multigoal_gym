import os
import gym
import pybullet_multigoal_gym


env = gym.make('MGKuka2ObjPyBulletEnv-v0')
state = env.reset()  # should return a state vector if everything worked

while True:
    action = env.action_space.sample() * 0
    env.step(action)
