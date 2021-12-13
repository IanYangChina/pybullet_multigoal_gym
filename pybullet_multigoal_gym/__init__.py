from gym.envs.registration import register, make, registry

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


def print_id():
    for env_id in ids:
        print(env_id)


def make_env(task='reach', gripper='parallel_jaw', num_block=5, render=False, binary_reward=True,
             grip_informed_goal=False, task_decomposition=False,
             joint_control=False, max_episode_steps=50, distance_threshold=0.05,
             primitive=None,
             image_observation=False, depth_image=False, goal_image=False, point_cloud=False,
             visualize_target=True,
             camera_setup=None, observation_cam_id=0, goal_cam_id=0,
             use_curriculum=False, num_goals_to_generate=1e6):
    tasks = ['push', 'reach', 'slide', 'pick_and_place',
             'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push',
             'shape_assemble', 'primitive_push_reach']
    grippers = ['robotiq85', 'parallel_jaw']
    assert gripper in grippers, 'invalid gripper: {}, only support: {}'.format(gripper, grippers)
    if task == 'reach':
        task_tag = 'Reach'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs:KukaReachEnv'
    elif task == 'push':
        task_tag = 'Push'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs:KukaPushEnv'
    elif task == 'pick_and_place':
        task_tag = 'PickAndPlace'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs:KukaPickAndPlaceEnv'
    elif task == 'slide':
        task_tag = 'Slide'
        image_observation = depth_image = goal_image = False
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs:KukaSlideEnv'
    elif task == 'block_stack':
        task_tag = 'BlockStack'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_multi_step_envs:KukaBlockStackEnv'
    elif task == 'block_rearrange':
        task_tag = 'BlockRearrangeEnv'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_multi_step_envs:KukaBlockRearrangeEnv'
    elif task == 'chest_pick_and_place':
        task_tag = 'ChestPickAndPlace'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_multi_step_envs:KukaChestPickAndPlaceEnv'
    elif task == 'chest_push':
        task_tag = 'ChestPush'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_multi_step_envs:KukaChestPushEnv'
    elif task == 'shape_assemble':
        task_tag = 'ShapeAssemble'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_shape_assemble_envs:KukaPushAssembleEnv'
    elif task == 'primitive_push_reach':
        task_tag = 'PrimPushReach'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_shape_assemble_envs:KukaPushReachEnv'
    else:
        raise ValueError('invalid task name: {}, only support: {}'.format(task, tasks))
    env_id = 'Kuka' + task_tag
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
    if joint_control:
        env_id += 'JointCtrl'
    if image_observation:
        if depth_image:
            env_id += 'DepthImgObs'
        else:
            env_id += 'ImgObs'
        if goal_image:
            env_id += 'ImgGoal'
        if camera_setup is not None:
            assert observation_cam_id <= len(camera_setup) - 1, 'invalid observation camera id'
            assert goal_cam_id <= len(camera_setup) - 1, 'invalid goal camera id'
            print('Received %i cameras, cam %i for observation, cam %i for goal image' %
                  (len(camera_setup), observation_cam_id, goal_cam_id))
        else:
            print('Using default camera for observation and goal image')
    env_id += '-v0'
    print('Task id: %s' % env_id)
    if env_id not in registry.env_specs:
        # register and make env instance
        if task in ['push', 'reach', 'slide', 'pick_and_place']:
            register(
                id=env_id,
                entry_point=entry,
                kwargs={
                    'render': render,
                    'binary_reward': binary_reward,
                    'joint_control': joint_control,
                    'distance_threshold': distance_threshold,
                    'image_observation': image_observation,
                    'depth_image': depth_image,
                    'goal_image': goal_image,
                    'visualize_target': visualize_target,
                    'camera_setup': camera_setup,
                    'observation_cam_id': observation_cam_id,
                    'goal_cam_id': goal_cam_id,
                    'gripper_type': gripper,
                },
                max_episode_steps=max_episode_steps,
            )
        elif task in ['block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']:
            assert num_block <= 5, "only support up to 5 blocks"
            register(
                id=env_id,
                entry_point=entry,
                kwargs={
                    'render': render,
                    'binary_reward': binary_reward,
                    'joint_control': joint_control,
                    'distance_threshold': distance_threshold,
                    'task_decomposition': task_decomposition,
                    'image_observation': image_observation,
                    'depth_image': depth_image,
                    'goal_image': goal_image,
                    'visualize_target': visualize_target,
                    'camera_setup': camera_setup,
                    'observation_cam_id': observation_cam_id,
                    'goal_cam_id': goal_cam_id,
                    'gripper_type': gripper,
                    'grip_informed_goal': grip_informed_goal,
                    'num_block': num_block,
                    'use_curriculum': use_curriculum,
                    'num_goals_to_generate': int(num_goals_to_generate)
                },
                max_episode_steps=max_episode_steps,
            )
        else:
            assert task in ['shape_assemble', 'primitive_push_reach']
            assert primitive in ['discrete_push', 'continuous_push']
            register(
                id=env_id,
                entry_point=entry,
                kwargs={
                    'render': render,
                    'binary_reward': binary_reward,
                    'distance_threshold': distance_threshold,
                    'image_observation': image_observation,
                    'depth_image': depth_image,
                    'pcd': point_cloud,
                    'goal_image': goal_image,
                    'visualize_target': visualize_target,
                    'camera_setup': camera_setup,
                    'observation_cam_id': observation_cam_id,
                    'goal_cam_id': goal_cam_id,
                    'gripper_type': gripper,
                    'primitive': primitive
                },
                max_episode_steps=max_episode_steps,
            )

    return make(env_id)
