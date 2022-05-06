import pybullet_multigoal_gym as pmg
import matplotlib.pyplot as plt

f, axarr = plt.subplots(1, 2)

camera_setup = [
    {
        'cameraEyePosition': [-0.9, -0.0, 0.4],
        'cameraTargetPosition': [-0.45, -0.0, 0.0],
        'cameraUpVector': [0, 0, 1],
        'render_width': 224,
        'render_height': 224
    },
    {
        'cameraEyePosition': [-1.0, -0.25, 0.6],
        'cameraTargetPosition': [-0.6, -0.05, 0.2],
        'cameraUpVector': [0, 0, 1],
        'render_width': 224,
        'render_height': 224
    },
]

env = pmg.make_env(task='pick_and_place',
                   gripper='parallel_jaw',
                   render=True,
                   binary_reward=True,
                   max_episode_steps=5,
                   image_observation=True,
                   depth_image=False,
                   goal_image=True,
                   visualize_target=True,
                   camera_setup=camera_setup,
                   observation_cam_id=[0],
                   goal_cam_id=1,
                   )
obs = env.reset()
t = 0
while True:
    t += 1
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    axarr[0].imshow(obs['desired_goal_img'])
    axarr[1].imshow(obs['achieved_goal_img'])
    plt.pause(0.00001)
    if done:
        env.reset()
