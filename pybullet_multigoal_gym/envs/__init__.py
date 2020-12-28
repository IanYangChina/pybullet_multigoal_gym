from gym.envs.registration import register

ids = []
# multi-goal envs
renders = [True, False]
sparse = [True, False]
image_obs = [True, False]
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

            ids.append('KukaReach' + tag + reward_tag + obs_tag + 'Env-v0')
            register(
                id='KukaReach' + tag + reward_tag + obs_tag + 'Env-v0',
                entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaReachEnv',
                kwargs={
                    'render': renders[i],
                    'binary_reward': sparse[j],
                    'image_observation': image_obs[k]
                },
                max_episode_steps=50,
            )

            ids.append('KukaPush' + tag + reward_tag + obs_tag + 'Env-v0')
            register(
                id='KukaPush' + tag + reward_tag + obs_tag + 'Env-v0',
                entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaPushEnv',
                kwargs={
                    'render': renders[i],
                    'binary_reward': sparse[j],
                    'image_observation': image_obs[k]
                },
                max_episode_steps=50,
            )

            ids.append('KukaPickAndPlace' + tag + reward_tag + obs_tag + 'Env-v0')
            register(
                id='KukaPickAndPlace' + tag + reward_tag + obs_tag + 'Env-v0',
                entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaPickAndPlaceEnv',
                kwargs={
                    'render': renders[i],
                    'binary_reward': sparse[j],
                    'image_observation': image_obs[k]
                },
                max_episode_steps=50,
            )

        ids.append('KukaSlide' + tag + reward_tag + 'Env-v0')
        register(
            id='KukaSlide' + tag + reward_tag + 'Env-v0',
            entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaSlideEnv',
            kwargs={
                'render': renders[i],
                'binary_reward': sparse[j]
            },
            max_episode_steps=50,
        )

    ids.append('KukaHierPickAndPlace' + tag + 'SparseEnv-v0')
    register(
        id='KukaHierPickAndPlace' + tag + 'SparseEnv-v0',
        entry_point='pybullet_multigoal_gym.envs.kuka.kuka_hierarchical_pick_and_place:HierarchicalKukaPickAndPlaceEnv',
        kwargs={
            'render': tag,
            'binary_reward': True
        },
        max_episode_steps=50,
    )


def print_id():
    for env_id in ids:
        print(env_id)
