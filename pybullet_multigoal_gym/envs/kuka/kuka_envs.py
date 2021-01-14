from pybullet_multigoal_gym.envs.kuka.kuka_env_base import KukaBulletMGEnv


class KukaPickAndPlaceEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, image_observation=False, gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=image_observation, gripper_type=gripper_type,
                                 grasping=True, has_obj=True, randomized_obj_pos=True)


class KukaPushEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, image_observation=False, gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=image_observation, gripper_type=gripper_type,
                                 target_on_table=True,
                                 grasping=False, has_obj=True, randomized_obj_pos=True)


class KukaReachEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, image_observation=False, gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=image_observation, gripper_type=gripper_type,
                                 grasping=False, has_obj=False, randomized_obj_pos=True)


class KukaSlideEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=False, gripper_type=gripper_type,
                                 table_type='long_table', target_on_table=True,
                                 grasping=False, has_obj=True, randomized_obj_pos=False)
