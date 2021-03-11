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

Our team is preparing some harder tasks for long-horizon sparse reward robotic arm manipulation tasks
on this package as well. All the environments will be summarised in a paper for references ASAP. We will
release a stable version when everything is set, but feel free to play with the current branch.

### Installation

```
git clone https://github.com/IanYangChina/pybullet_multigoal_gym.git
cd pybullet_multigoal_gym
pip install -r requirements.txt
pip install .
```

### Some info

Observation, state, action, goal and reward are all setup to be the same as the original environment.

Observation is a dictionary, containing the state, desired goal and achieved goal.

Since no rotation is involved, states contain the end-effector Cartesian position, 
linear and angular velocities; and block Cartesian position, linear and angular velocities 
(if the task involves a block).

Goals are either EE or block Cartesian positions, in the world frame.

Actions are 3 dimensional for the Reach, Push and Slide tasks, which are xyz movements in the 
EE space. For the Pick and Place task, there is an extra dimension related to the closing and opening
of the gripper fingers. All of them are within [-1, 1].

Rewards are set to negatively proportional to the goal distance. For sparse, 
it's -1 and 0 rewards, where 0 stands for goal being achieved. For dense reward,
it equals to the negative goal distance (achieved & desired goals).

Use the `make_env(...)` method to make your environments. Due to backend differences, the `render()` method 
should not need to be called by users. 
To run experiment headless, make environment with `render=False`. 
To run experiment with image observations, make environment with `image_observation=True`. 
Only the Reach, PickAndPlace and Push envs support image observation. See examples below.

### Try it out

```python
# Single-stage manipulation environments
# Reach, Push, PickAndPlace, Slide
import pybullet_multigoal_gym as pmg
# Install matplotlib if you want to use imshow to view the goal images
import matplotlib.pyplot as plt

env = pmg.make_env(task='reach',    
                   # task is in ['reach', 'push', 'pick_and_place', 'slide']
                   gripper='parallel_jaw', 
                   # gripper is in ['parallel_jaw', 'robotiq85']
                   render=True,
                   binary_reward=True,
                   max_episode_steps=50,
                   image_observation=True,
                   depth_image=False,
                   goal_image=False)
obs = env.reset()
t = 0
while True:
    t += 1
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print('state: ', obs['state'], '\n',
          'desired_goal: ', obs['desired_goal'], '\n',
          'achieved_goal: ', obs['achieved_goal'], '\n',
          'reward: ', reward, '\n')
    plt.imshow(obs['observation']) # only works for environments with image observation
    plt.pause(0.00001)      
    if done:
        env.reset()
```

```python
# Hierarchical environments: only a simple pick-and-place task is available
import pybullet_multigoal_gym as pmg
print("Existing hierarchical envs: ", pmg.ids)
# Install matplotlib if you want to use imshow to view the goal images
import matplotlib.pyplot as plt


env = pmg.make('KukaParallelGripHierPickAndPlaceSparseImageObsEnv-v0')
obs = env.reset()
time_done = False
while True:
    high_level_action = env.high_level_action_space.sample()
    env.set_sub_goal(high_level_action)
    sub_goal_done = False
    while not sub_goal_done and not time_done:
        action = env.low_level_action_space.sample()
        obs, reward, time_done, info = env.step(action)
        sub_goal_done = info['sub_goal_achieved']
        print('state: ', obs['state'], '\n',
              'desired_sub_goal: ', obs['desired_sub_goal'], '\n',
              'achieved_sub_goal: ', obs['achieved_sub_goal'], '\n',
              'sub_reward: ', reward['sub_reward'], '\n',
              'final_reward: ', reward['final_reward'], '\n',)       
        plt.imshow(obs['desired_sub_goal_image'])
        plt.pause(0.00001)
        plt.imshow(obs['achieved_sub_goal_image'])
        plt.pause(0.00001)
    if time_done:
        env.reset()
```

### Scenes

<img src="src/HERBenchmark.png" width="800"/>

<img src="/src/MultiStepBenchmark.png" width="800"/>

### Updates

2020.12.26 --- Add hierarchical environments for a pick and place task, with image observation and goal supported. 
See the above example.

2020.12.28 --- Add image observation to non-hierarchical environments.

2021.01.14 --- Add parallel jaw gripper.

2021.03.08 --- Add a make_env(...) method to replace the pre-registration codes.

2021.03.09 --- Add multi-block stacking and re-arranging tasks
