from pybullet_multigoal_gym.envs.base_envs.kuka_shape_assemble_base_env import KukaBulletShapeAssembleEnv


class KukaPushAssembleEnv(KukaBulletShapeAssembleEnv):
    def __init__(self, render=True, binary_reward=True, distance_threshold=0.05,
                 primitive='discrete_push',
                 image_observation=False, goal_image=False, depth_image=False, pcd=False, visualize_target=False,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw'):
        KukaBulletShapeAssembleEnv.__init__(self, render=render, binary_reward=binary_reward,
                                            distance_threshold=distance_threshold,
                                            image_observation=image_observation, goal_image=goal_image,
                                            depth_image=depth_image, pcd=pcd,
                                            visualize_target=visualize_target,
                                            camera_setup=camera_setup, observation_cam_id=observation_cam_id,
                                            goal_cam_id=goal_cam_id,
                                            gripper_type=gripper_type, obj_range=0.1, target_range=0.15,
                                            end_effector_start_on_table=False,
                                            grasping=False, primitive=primitive)
