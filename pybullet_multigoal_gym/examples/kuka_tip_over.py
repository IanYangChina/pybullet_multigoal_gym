import pybullet_multigoal_gym
import pybullet_multigoal_gym as pmg
import matplotlib.pyplot as plt

f, axarr = plt.subplots(1, 2)

env = pmg.make_env(task='tip_over',
                   gripper='parallel_jaw',
                   render=True,
                   binary_reward=True,
                   joint_control=True,
                   max_episode_steps=50,
                   image_observation=False,
                   depth_image=False,
                   goal_image=False,
                   visualize_target=True,
                   camera_setup=None,
                   observation_cam_id=[0],
                   goal_cam_id=0,
                   )
obs = env.reset()
t = 0
while True:
    t += 1
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # axarr[0].imshow(obs['desired_goal_img'])
    # axarr[1].imshow(obs['achieved_goal_img'])
    plt.pause(0.00001)
    if done:
        env.reset()
