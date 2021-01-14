from pybullet_multigoal_gym.envs.kuka.kuka_env_base import KukaBulletMGEnv


class KukaPickAndPlaceEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, image_observation=False, gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=image_observation, gripper_type=gripper_type,
                                 distance_threshold=0.02,
                                 grasping=True, has_obj=True, randomized_obj_pos=True, obj_range=0.15)


class KukaPushEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, image_observation=False, gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=image_observation, gripper_type=gripper_type,
                                 target_on_table=True, distance_threshold=0.02,
                                 grasping=False, has_obj=True, randomized_obj_pos=True, obj_range=0.15)


class KukaReachEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, image_observation=False, gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=image_observation, gripper_type=gripper_type,
                                 distance_threshold=0.02,
                                 grasping=False, has_obj=False, randomized_obj_pos=True, obj_range=0.15)


class KukaSlideEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=False, gripper_type=gripper_type,
                                 table_type='long_table', target_on_table=True, distance_threshold=0.02,
                                 grasping=False, has_obj=True, randomized_obj_pos=False, obj_range=0.15)
