#### Warning (2022.05.06): 
```diff
@@ I will stop further development on this repo as Mujoco is now free. @@
@@ However I will keep maintaining the repo and respond to issues. @@
@@ Feel free to keep using the package and contact me whenever necessary. @@
@@ Thanks for you interests! @@
```

### Pybullet-implementation of the multi-goal robotics environment originally from Open AI Gym.

This version uses a kuka iiwa14 7DoF arm, equipped with a robotiq85 two finger gripper or a simple parallel jaw.

The basic four tasks are basically the same as the ones in the OpenAI Gym: **Reach, Push, Pick and Place, Slide**.
However, it may look funny when the robot picks up a block with the robotiq85 gripper,
since it's under-actuated and thus makes it hard to fine-tune the simulation. 
You can use the parallel jaw gripper, which is effectively the same as the OpenAI one.

I have implemented some goal-conditioned RL algos in my another repo, using the 
original Gym environment. There are DDPG-HER, SAC-HER, and others.
<a href="https://github.com/IanYangChina/DRL_Implementation.git" target="_blank">DRL_Implementation.</a>
Expired mujoco license got me here. I will also retrain those agents and pose performance ASAP.

This package also provides some harder tasks for long-horizon sparse reward robotic arm manipulation tasks
on this package as well. All the environments have been summarised in a paper.
The newest release is the most recommended. There are still on-going updates for this package, the [v1.0 release] was 
the version published on Taros 2021 and [ArXiv](https://arxiv.org/abs/2105.05985).
Due to further development, the description in this paper may not be exactly the same with the master branch.

The following tasks are supported in the v1.3 branch:
1. Reach, push, pick-and-place, slide as the gym-robotics tasks;
2. Four Multi-step tasks described in the [Taros paper](https://arxiv.org/abs/2105.05985)
3. Two shape-assemble tasks (block-fitting & reaching) with pushing primitive actions (continuous & discrete)
4. An insertion task with 6 DoF gripper frame control

```
@InProceedings{yang2021pmg,
author="Yang, Xintong and Ji, Ze and Wu, Jing and Lai, Yu-Kun",
title="An Open-Source Multi-goal Reinforcement Learning Environment for Robotic Manipulation with Pybullet",
booktitle="Towards Autonomous Robotic Systems",
year="2021",
publisher="Springer International Publishing",
pages="14--24",
isbn="978-3-030-89177-0"
}
```

### Installation

```
git clone https://github.com/IanYangChina/pybullet_multigoal_gym.git
cd pybullet_multigoal_gym
pip install -r requirements.txt
pip install .
```

### Some info

Observation, state, action, goal and reward are all setup to be the same as the original environment.

Observation is a dictionary, containing the state, desired goal, achieved goal and other sensory data.

Use the `make_env(...)` method to make your environments. Due to backend differences, the `render()` method 
should not need to be called by users. 
To run experiment headless, make environment with `render=False`. 
To run experiment with image observations, make environment with `image_observation=True`. 
Only the Reach, PickAndPlace and Push envs support image observation. Users can define their
own camera for observation and goal images. The camera id `-1` stands for a on-hand camera. 
See examples below.

### Try it out

See the [examples folder](https://github.com/IanYangChina/pybullet_multigoal_gym/tree/master/pybullet_multigoal_gym/examples)
for more scripts to play with.

```python
# Single-stage manipulation environments
# Reach, Push, PickAndPlace, Slide
import pybullet_multigoal_gym as pmg
# Install matplotlib if you want to use imshow to view the goal images
import matplotlib.pyplot as plt

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

env = pmg.make_env(
    # task args ['reach', 'push', 'slide', 'pick_and_place', 
    #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
    task='block_stack',
    gripper='parallel_jaw',
    num_block=4,  # only meaningful for multi-block tasks, up to 5 blocks
    render=True,
    binary_reward=True,
    max_episode_steps=50,
    # image observation args
    image_observation=True,
    depth_image=False,
    goal_image=True,
    visualize_target=True,
    camera_setup=camera_setup,
    observation_cam_id=[0],
    goal_cam_id=1)

f, axarr = plt.subplots(1, 2)
obs = env.reset()
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print('state: ', obs['state'], '\n',
          'desired_goal: ', obs['desired_goal'], '\n',
          'achieved_goal: ', obs['achieved_goal'], '\n',
          'reward: ', reward, '\n')
    axarr[0].imshow(obs['desired_goal_img'])
    axarr[1].imshow(obs['achieved_goal_img'])
    plt.pause(0.00001)
    if done:
        env.reset()
```

### Scenes

<img src="src/HERBenchmark.png" width="800"/>

<img src="/src/MultiStepBenchmark.png" width="800"/>

<img src="/src/AssembleTasks.png" width="800"/>

### Update log

2020.12.26 --- Add hierarchical environments for a pick and place task, with image observation and goal supported. 
See the above example.

2020.12.28 --- Add image observation to non-hierarchical environments.

2021.01.14 --- Add parallel jaw gripper.

2021.03.08 --- Add a make_env(...) method to replace the pre-registration codes.

2021.03.09 --- Add multi-block stacking and re-arranging tasks

2021.03.12 --- Add multi-block tasks with a chest

2021.03.17 --- Joint space control support

2021.03.18 --- Finish curriculum; add on-hand camera observation

2021.11.11 --- Finish task decomposition, subgoal generation codes and some compatibility issues, new release

2021.12.03 --- Add shape assembly task with push primitive support (discrete & continuous)

2022.05.06 --- Clean up stuff; add insertion task; remove hierarchical env codes.
