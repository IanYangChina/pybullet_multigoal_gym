from gym.envs.registration import register

ids = []
# multi-goal envs
renders = [True, False]
sparse = [True, False]
for i in range(2):
    if renders[i]:
        tag = 'Render'
    else:
        tag = ''
    for j in range(2):
        if sparse[j]:
            tag_ = 'Sparse'
        else:
            tag_ = 'Dense'
        ids.append('KukaReach' + tag + tag_ + 'Env-v0')
        register(
            id='KukaReach' + tag + tag_ + 'Env-v0',
            entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaReachEnv',
            kwargs={
                'render': renders[i],
                'binary_reward': sparse[j]
            },
            max_episode_steps=50,
        )

        ids.append('KukaPush' + tag + tag_ + 'Env-v0')
        register(
            id='KukaPush' + tag + tag_ + 'Env-v0',
            entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaPushEnv',
            kwargs={
                'render': renders[i],
                'binary_reward': sparse[j]
            },
            max_episode_steps=50,
        )

        ids.append('KukaPickAndPlace' + tag + tag_ + 'Env-v0')
        register(
            id='KukaPickAndPlace' + tag + tag_ + 'Env-v0',
            entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaPickAndPlaceEnv',
            kwargs={
                'render': renders[i],
                'binary_reward': sparse[j]
            },
            max_episode_steps=50,
        )

        ids.append('KukaSlide' + tag + tag_ + 'Env-v0')
        register(
            id='KukaSlide' + tag + tag_ + 'Env-v0',
            entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaSlideEnv',
            kwargs={
                'render': renders[i],
                'binary_reward': sparse[j]
            },
            max_episode_steps=50,
        )

    ids.append('KukaReach' + tag + 'Env-v0')
    register(
        id='KukaReach' + tag + 'DenseImageObsEnv-v0',
        entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaReachEnv',
        kwargs={
            'render': renders[i],
            'binary_reward': False,
            'image_observation': True
        },
        max_episode_steps=50,
    )

    ids.append('KukaReach' + tag + 'Env-v0')
    register(
        id='KukaPickAndPlace' + tag + 'DenseImageObsEnv-v0',
        entry_point='pybullet_multigoal_gym.envs.kuka.kuka_envs:KukaPickAndPlaceEnv',
        kwargs={
            'render': renders[i],
            'binary_reward': False,
            'image_observation': True
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


def get_id():
    return ids
