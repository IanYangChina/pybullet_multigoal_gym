### Pybullet-implementation of the multi-goal robotics environment originally from Open AI Gym.

This version uses a kuka iiwa14 7DoF arm, equipped with a robotiq85 two finger gripper.

The four tasks are basically the same as the ones in the OpenAI Gym: **Reach, Push, Pick and Place, Slide**.
However, it may look funny when the robot picks up a block with the robotiq85 gripper,
since it's under-actuated and thus makes it hard to fine-tune the simulation. 
I will make a parallel-jaw gripper ASAP if I have time.

I have implemented some goal-conditioned RL algos in my another repo, using the 
original Gym environment. There are DDPG-HER, SAC-HER, and others.
<a href="https://github.com/IanYangChina/DRL_Implementation.git" target="_blank">DRL_Implementation.</a>
Expired mujoco license got me here. I will also 
retrain those agents and pose performance ASAP.

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

Due to backend differences, the `render()` method should only be called when you need an image observation. To run experiment headless, make environment without the word `'Render'` in the id.

Here's how you view all the env ids:
```python
import pybullet_multigoal_gym as pmg
print(pmg.envs.get_id())
```
```commandline
>>Sparse reward, render
'KukaReachRenderSparseEnv-v0', 'KukaPushRenderSparseEnv-v0', 
'KukaPickAndPlaceRenderSparseEnv-v0', 'KukaSlideRenderSparseEnv-v0', 

>>Dense reward, render
'KukaReachRenderDenseEnv-v0', 'KukaPushRenderDenseEnv-v0', 
'KukaPickAndPlaceRenderDenseEnv-v0', 'KukaSlideRenderDenseEnv-v0', 

>>Sparse reward, headless
'KukaReachSparseEnv-v0', 'KukaPushSparseEnv-v0', 
'KukaPickAndPlaceSparseEnv-v0', 'KukaSlideSparseEnv-v0', 

>>Dense reward, headless
'KukaReachDenseEnv-v0', 'KukaPushDenseEnv-v0', 
'KukaPickAndPlaceDenseEnv-v0', 'KukaSlideDenseEnv-v0'
```

### Try it out

```python
import pybullet_multigoal_gym as pmg


env = pmg.make('KukaPickAndPlaceRenderSparseEnv-v0')
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
    if done:
        env.reset()
```

### Scenes

<img src="src/01.jpeg" width="300"/>

<img src="src/02.jpeg" width="300"/>
