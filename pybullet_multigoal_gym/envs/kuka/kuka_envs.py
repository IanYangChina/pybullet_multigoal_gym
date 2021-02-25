from pybullet_multigoal_gym.envs.kuka.kuka_env_base import KukaBulletMGEnv


class KukaPickAndPlaceEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False,
                 gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                 gripper_type=gripper_type,
                                 target_in_the_air=False,
                                 grasping=True, has_obj=True, randomized_obj_pos=True)


class KukaPushEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False, depth_only=False,
                 gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=image_observation, goal_image=goal_image,
                                 depth_image=depth_image, depth_only=depth_only,
                                 gripper_type=gripper_type,
                                 target_in_the_air=False, end_effector_start_on_table=True,
                                 grasping=False, has_obj=True, randomized_obj_pos=True)


class KukaReachEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False,
                 gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                 gripper_type=gripper_type,
                                 grasping=False, has_obj=False, randomized_obj_pos=True)


class KukaSlideEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                 image_observation=False, gripper_type=gripper_type,
                                 table_type='long_table', target_in_the_air=False, end_effector_start_on_table=True,
                                 grasping=False, has_obj=True, randomized_obj_pos=True)
