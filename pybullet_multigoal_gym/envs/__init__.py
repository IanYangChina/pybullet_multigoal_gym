from gym.envs.registration import register

ids = []
# multi-goal envs
renders = [True, False]
sparse = [True, False]
image_obs = [True, False]
gripper_type = ['robotiq85', 'parallel_jaw']
for grip in gripper_type:
    if grip == 'robotiq85':
        grip_tag = 'RobotiqGrip'
    else:
        grip_tag = 'ParallelGrip'
    for i in range(2):
        if renders[i]:
            tag = 'Render'
        else:
            tag = ''
        for j in range(2):
            if sparse[j]:
                reward_tag = 'Sparse'
            else:
                reward_tag = 'Dense'
            for k in range(2):
                if image_obs[k]:
                    obs_tag = 'ImageObs'
                else:
                    obs_tag = ''

                ids.append('Kuka'+grip_tag+'Reach' + tag + reward_tag + obs_tag + 'Env-v0')
                register(
                    id='Kuka'+grip_tag+'Reach' + tag + reward_tag + obs_tag + 'Env-v0',
                    entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaReachEnv',
                    kwargs={
                        'render': renders[i],
                        'binary_reward': sparse[j],
                        'image_observation': image_obs[k],
                        'gripper_type': grip
                    },
                    max_episode_steps=50,
                )

                ids.append('Kuka'+grip_tag+'Push' + tag + reward_tag + obs_tag + 'Env-v0')
                register(
                    id='Kuka'+grip_tag+'Push' + tag + reward_tag + obs_tag + 'Env-v0',
                    entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaPushEnv',
                    kwargs={
                        'render': renders[i],
                        'binary_reward': sparse[j],
                        'image_observation': image_obs[k],
                        'gripper_type': grip
                    },
                    max_episode_steps=50,
                )

                ids.append('Kuka'+grip_tag+'PickAndPlace' + tag + reward_tag + obs_tag + 'Env-v0')
                register(
                    id='Kuka'+grip_tag+'PickAndPlace' + tag + reward_tag + obs_tag + 'Env-v0',
                    entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaPickAndPlaceEnv',
                    kwargs={
                        'render': renders[i],
                        'binary_reward': sparse[j],
                        'image_observation': image_obs[k],
                        'gripper_type': grip
                    },
                    max_episode_steps=50,
                )

            ids.append('Kuka'+grip_tag+'Slide' + tag + reward_tag + 'Env-v0')
            register(
                id='Kuka'+grip_tag+'Slide' + tag + reward_tag + 'Env-v0',
                entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaSlideEnv',
                kwargs={
                    'render': renders[i],
                    'binary_reward': sparse[j],
                    'gripper_type': grip
                },
                max_episode_steps=50,
            )

        ids.append('Kuka'+grip_tag+'HierPickAndPlace' + tag + 'SparseEnv-v0')
        register(
            id='Kuka'+grip_tag+'HierPickAndPlace' + tag + 'SparseEnv-v0',
            entry_point='pybullet_multigoal_gym.envs.kuka.kuka_hierarchical_pick_and_place:HierarchicalKukaPickAndPlaceEnv',
            kwargs={
                'render': tag,
                'binary_reward': True,
                'image_observation': False,
                'gripper_type': grip
            },
            max_episode_steps=50,
        )

        ids.append('Kuka'+grip_tag+'HierPickAndPlace' + tag + 'SparseImageObsEnv-v0')
        register(
            id='Kuka'+grip_tag+'HierPickAndPlace' + tag + 'SparseImageObsEnv-v0',
            entry_point='pybullet_multigoal_gym.envs.kuka.kuka_hierarchical_pick_and_place:HierarchicalKukaPickAndPlaceEnv',
            kwargs={
                'render': tag,
                'binary_reward': True,
                'image_observation': True,
                'gripper_type': grip
            },
            max_episode_steps=50,
        )


def print_id():
    for env_id in ids:
        print(env_id)
