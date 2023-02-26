from gym.envs.registration import register, make, registry


def make_env(task='reach', gripper='parallel_jaw', num_block=5, render=False, binary_reward=True,
             grip_informed_goal=False, task_decomposition=False,
             joint_control=False, max_episode_steps=50, distance_threshold=0.05,
             primitive=None,
             image_observation=False, depth_image=False, goal_image=False, point_cloud=False, state_noise=False,
             visualize_target=True,
             camera_setup=None, observation_cam_id=None, goal_cam_id=0,
             use_curriculum=False, num_goals_to_generate=1e6):
    if observation_cam_id is None:
        observation_cam_id = [0]
    tasks = ['push', 'reach', 'slide', 'pick_and_place',
             'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push',
             'primitive_push_assemble', 'primitive_push_reach', 'insertion']
    grippers = ['robotiq85', 'parallel_jaw']
    assert gripper in grippers, 'invalid gripper: {}, only support: {}'.format(gripper, grippers)
    if task == 'reach':
        task_tag = 'Reach'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs:KukaReachEnv'
    elif task == 'tip_over':
        task_tag = 'TipOver'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs:KukaTipOverEnv'
    elif task == 'push':
        task_tag = 'Push'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs:KukaPushEnv'
    elif task == 'pick_and_place':
        task_tag = 'PickAndPlace'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs:KukaPickAndPlaceEnv'
    elif task == 'slide':
        task_tag = 'Slide'
        assert not image_observation, "slide task doesn't support image observation well."
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
    elif task == 'primitive_push_assemble':
        task_tag = 'ShapeAssemble'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_shape_assemble_envs:KukaPushAssembleEnv'
    elif task == 'primitive_push_reach':
        task_tag = 'PrimPushReach'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_shape_assemble_envs:KukaPushReachEnv'
    elif task == 'insertion':
        task_tag = 'Insertion'
        entry = 'pybullet_multigoal_gym.envs.task_envs.kuka_insertion_envs:KukaInsertionEnv'
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
            assert len(observation_cam_id) <= len(camera_setup) + 1, 'invalid observation camera id list'
            assert goal_cam_id <= len(camera_setup) - 1, 'invalid goal camera id'
            print('Received %i cameras, cam {} for observation, cam %i for goal image'.format(observation_cam_id) %
                  (len(camera_setup), goal_cam_id))
        else:
            print('Using default camera for observation and goal image')
    env_id += '-v0'
    print('Task id: %s' % env_id)
    if env_id not in registry.env_specs:
        # register and make env instance
        if task in ['push', 'reach', 'tip_over', 'slide', 'pick_and_place']:
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
        elif task in ['primitive_push_assemble', 'primitive_push_reach']:
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
        else:
            assert task in ['insertion']
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
                    'state_noise': state_noise,
                    'visualize_target': visualize_target,
                    'camera_setup': camera_setup,
                    'observation_cam_id': observation_cam_id,
                    'goal_cam_id': goal_cam_id,
                    'gripper_type': gripper,
                },
                max_episode_steps=max_episode_steps,
            )

    return make(env_id)
