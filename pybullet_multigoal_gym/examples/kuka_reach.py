import pybullet_multigoal_gym as pmg


env = pmg.make_env(task='reach',
                   gripper='parallel_jaw',
                   render=True,
                   binary_reward=True,
                   max_episode_steps=50,
                   image_observation=False,
                   depth_image=False,
                   goal_image=False)
obs = env.reset()
t = 0
while True:
    t += 1
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
