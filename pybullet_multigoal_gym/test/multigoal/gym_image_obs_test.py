import os
import open3d as o3d
import numpy as np
import pybullet_multigoal_gym as pmg
import matplotlib.pyplot as plt

camera_setup = [
    {
        'cameraEyePosition': [-0.55, 0.0, 0.282],
        'cameraTargetPosition': [-0.55, 0.0, 0.02],
        'cameraUpVector': [1, 0, 0],
        # resolution: 0.002 meters per pixel for the workspace of 0.4 x 0.3 meters
        'render_width': 200,
        'render_height': 150
    },
    {
        'cameraEyePosition': [-1.0, -0.25, 0.6],
        'cameraTargetPosition': [-0.6, -0.05, 0.2],
        'cameraUpVector': [0, 0, 1],
        'render_width': 128,
        'render_height': 128
    }
]

env = pmg.make_env(
    task='primitive_push_reach',
    render=True,
    primitive='discrete_push',
    binary_reward=True,
    distance_threshold=0.05,
    image_observation=True,
    depth_image=True,
    goal_image=True,
    point_cloud=True,
    visualize_target=True,
    camera_setup=camera_setup,
    observation_cam_id=1,
    goal_cam_id=0,
    gripper='parallel_jaw',
    max_episode_steps=10000)

obs = env.reset()
time_done = False
f, axarr = plt.subplots(3, 2)
# env.set_sub_goal(0)
# print(env.desired_goal)
# t = 0
while True:
    action = env.action_space.sample()
    # action[0] = 20
    # action[1] = 74
    # action[2] = 99
    obs, reward, time_done, info = env.step(action)
    axarr[0][0].imshow(obs['desired_goal_img'][:, :, :3])
    axarr[0][1].imshow(obs['desired_goal_img'][:, :, 3])
    axarr[1][0].imshow(obs['achieved_goal_img'][:, :, :3])
    axarr[1][1].imshow(obs['achieved_goal_img'][:, :, 3])
    axarr[2][0].imshow(obs['observation'][:, :, :3])
    axarr[2][1].imshow(obs['observation'][:, :, 3])
    plt.pause(0.00001)
    # new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    # pcd_raw = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(obs['pcd']))
    # o3d.visualization.draw_geometries([pcd_raw, new_frame])
    # obs = env.reset()
    # t += 1
    # if t == 3:
    #     env.set_sub_goal(1)
    # if t == 6:
    #     env.set_sub_goal(2)
    # if t == 9:
    #     env.set_sub_goal(3)
    # if t == 12:
    #     env.set_sub_goal(4)
    if time_done:
        obs = env.reset()
