from pybullet_multigoal_gym.envs.kuka.kuka_env_base import KukaBulletMGEnv


class KukaPushEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 distance_threshold=0.02, target_on_table=True,
                                 grasping=False, has_obj=True, randomized_obj_pos=True, obj_range=0.15)
