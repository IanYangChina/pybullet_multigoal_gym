from gym.envs.registration import register, make
ids = []
# multi-goal envs
renders = [True, False]
for i in range(2):
    if renders[i]:
        tag = 'Render'
    else:
        tag = ''

    ids.append('KukaParallelGripHierPickAndPlace' + tag + 'SparseEnv-v0')
    register(
        id='KukaParallelGripHierPickAndPlace' + tag + 'SparseEnv-v0',
        entry_point='pybullet_multigoal_gym.envs.kuka.kuka_hierarchical_pick_and_place:HierarchicalKukaPickAndPlaceEnv',
        kwargs={
            'render': renders[i],
            'binary_reward': True,
            'image_observation': False,
            'gripper_type': 'parallel_jaw'
        },
        max_episode_steps=50,
    )

    ids.append('KukaParallelGripHierPickAndPlace' + tag + 'SparseImageObsEnv-v0')
    register(
        id='KukaParallelGripHierPickAndPlace' + tag + 'SparseImageObsEnv-v0',
        entry_point='pybullet_multigoal_gym.envs.kuka.kuka_hierarchical_pick_and_place:HierarchicalKukaPickAndPlaceEnv',
        kwargs={
            'render': renders[i],
            'binary_reward': True,
            'image_observation': True,
            'gripper_type': 'parallel_jaw'
        },
        max_episode_steps=50,
    )

    ids.append('KukaParallelGripBlockStack' + tag + 'SparseEnv-v0')
    register(
        id='KukaParallelGripBlockStack' + tag + 'SparseEnv-v0',
        entry_point='pybullet_multigoal_gym.envs.kuka.kuka_multi_block_env:KukaBulletMultiBlockEnv',
        kwargs={
            'grasping': True,
            'render': renders[i],
            'binary_reward': True,
            'image_observation': False,
            'gripper_type': 'parallel_jaw'
        },
        max_episode_steps=200,
    )

    ids.append('KukaParallelGripBlockRearrange' + tag + 'SparseEnv-v0')
    register(
        id='KukaParallelGripBlockRearrange' + tag + 'SparseEnv-v0',
        entry_point='pybullet_multigoal_gym.envs.kuka.kuka_multi_block_env:KukaBulletMultiBlockEnv',
        kwargs={
            'grasping': False,
            'render': renders[i],
            'binary_reward': True,
            'image_observation': False,
            'gripper_type': 'parallel_jaw'
        },
        max_episode_steps=200,
    )


def print_id():
    for env_id in ids:
        print(env_id)


def make_env(task='reach', gripper='parallel_jaw', render=False, binary_reward=True, max_episode_steps=50,
             image_observation=False, depth_image=False, goal_image=False):
    tasks = ['push', 'reach', 'slide', 'pick_and_place']
    assert task in tasks, 'invalid task name: {}, only support: {}'.format(task, tasks)
    grippers = ['robotiq85', 'parallel_jaw']
    assert gripper in grippers, 'invalid gripper: {}, only support: {}'.format(gripper, grippers)
    if task == 'reach':
        task_tag = 'Reach'
        entry = 'pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaReachEnv'
    elif task == 'push':
        task_tag = 'Push'
        entry = 'pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaPushEnv'
    elif task == 'pick_and_place':
        task_tag = 'PickAndPlace'
        entry = 'pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaPickAndPlaceEnv'
    elif task == 'slide':
        task_tag = 'Slide'
        image_observation = depth_image = goal_image = False
        entry = 'pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaSlideEnv'
    else:
        raise ValueError('something\'s wrong')
    env_id = task_tag
    if gripper == 'parallel_jaw':
        env_id += 'ParallelGrip'
    else:
        env_id += 'Robotiq85Grip'
    if render:
        env_id += 'Render'
    if binary_reward:
        env_id += 'SparseReward'
    else:
        env_id += 'DenseReward'
    if image_observation:
        if depth_image:
            env_id += 'DepthImgObs'
        else:
            env_id += 'ImgObs'
        if goal_image:
            env_id += 'ImgGoal'
    env_id += '-v0'
    print('Task id: %s' % env_id)
    register(
            id=env_id,
            entry_point=entry,
            kwargs={
                'render': render,
                'binary_reward': binary_reward,
                'image_observation': image_observation,
                'depth_image': depth_image,
                'goal_image': goal_image,
                'gripper_type': gripper
            },
            max_episode_steps=max_episode_steps,
        )

    return make(env_id)
