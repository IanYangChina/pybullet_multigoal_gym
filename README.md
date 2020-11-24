### Pybullet-implementation of the multi-goal robotics environment originally from Open AI Gym.

This version uses a kuka iiwa14 7DoF arm, equipped with a robotiq85 two finger gripper.

The four tasks are basically the same as the ones in the OpenAI Gym: Reach, Push, Pick and Place, Slide.
However, it may looks funny when the robot picks up a block with the robotiq85 gripper,
since it's under-actuated and thus hard to fine-tune the simulation. 
I will make a parallel-jaw gripper ASAP if I have time.

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
### Installation

```
git clone https://github.com/IanYangChina/pybullet_multigoal_gym.git
cd pybullet_multigoal_gym
pip install -r requirements.txt
pip install .
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
    if done:
        env.reset()
```

### Scenes

<img src="src/01.jpeg" width="300"/>

<img src="src/02.jpeg" width="300"/>
