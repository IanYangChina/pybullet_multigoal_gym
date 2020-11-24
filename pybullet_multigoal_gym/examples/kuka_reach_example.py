import pybullet_multigoal_gym


env = pybullet_multigoal_gym.make('KukaPickAndPlaceRenderSparseEnv-v0')
obs = env.reset()
t = 0
while True:
    t += 1
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
