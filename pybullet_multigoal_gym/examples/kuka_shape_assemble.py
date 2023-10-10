import os
import open3d as o3d
import numpy as np
import pybullet_multigoal_gym as pmg
import matplotlib.pyplot as plt

camera_setup = [
    # {
    #     'cameraEyePosition': [-0.58, 0.0, 0.327],
    #     'cameraTargetPosition': [-0.58, 0.0, 0.02],
    #     'cameraUpVector': [1, 0, 0],
    #     # resolution: 0.0015625 meters per pixel for the workspace of 0.35 x 0.35 meters
    #     'render_width': 224,
    #     'render_height': 224
    # },
    # {
    #     'cameraEyePosition': [-0.9, -0.0, 0.4],
    #     'cameraTargetPosition': [-0.45, -0.0, 0.0],
    #     'cameraUpVector': [0, 0, 1],
    #     'render_width': 224,
    #     'render_height': 224
    # }
]

env = pmg.make_env(
    task='primitive_push_reach',
    primitive='discrete_push',
    render=True,
    binary_reward=True,
    distance_threshold=0.05,
    image_observation=True,
    depth_image=True,
    goal_image=True,
    point_cloud=False,
    state_noise=True,
    visualize_target=False,
    camera_setup=camera_setup,
    observation_cam_id=[1],
    goal_cam_id=-1,
    gripper='parallel_jaw',
    max_episode_steps=500)

obs = env.reset()
time_done = False
f, axarr = plt.subplots(2, 2)
while True:
    action = env.action_space.sample()
    obs, reward, time_done, info = env.step(action)
    axarr[0][0].imshow(obs['desired_goal_img'][:, :, :3])
    axarr[0][1].imshow(obs['desired_goal_img'][:, :, 3])
    axarr[1][0].imshow(obs['observation'][:, :, :3])
    axarr[1][1].imshow(obs['observation'][:, :, 3])
    plt.pause(0.00001)
    if time_done:
        obs = env.reset()
