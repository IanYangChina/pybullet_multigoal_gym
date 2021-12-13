import numpy as np
import quaternion as quat
from pybullet_multigoal_gym.envs.base_envs.kuka_shape_assemble_base_env import KukaBulletPrimitiveEnv


class KukaPushAssembleEnv(KukaBulletPrimitiveEnv):
    def __init__(self, render=True, binary_reward=True, distance_threshold=0.05,
                 primitive='discrete_push',
                 image_observation=False, goal_image=False, depth_image=False, pcd=False, visualize_target=False,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw'):
        KukaBulletPrimitiveEnv.__init__(self, render=render, binary_reward=binary_reward,
                                        distance_threshold=distance_threshold,
                                        image_observation=image_observation, goal_image=goal_image,
                                        depth_image=depth_image, pcd=pcd,
                                        visualize_target=visualize_target, regenerate_goal_when_step=True,
                                        camera_setup=camera_setup, observation_cam_id=observation_cam_id,
                                        goal_cam_id=goal_cam_id,
                                        gripper_type=gripper_type, obj_range=0.1, target_range=0.15,
                                        end_effector_start_on_table=False,
                                        grasping=False, primitive=primitive,
                                        manipulated_object_keys=['slot', 'cube'], goal_object_key='cube',
                                        orientation_informed_goal=True)

    def _generate_goal(self):
        (x, y, z), (a, b, c, w), _, _, _, _ = self._p.getLinkState(self.object_bodies['slot'], 2)
        target_obj_quat = np.array([a, b, c, w])
        orientation_euler = quat.as_euler_angles(quat.as_quat_array([w, a, b, c]))

        self.desired_goal = np.concatenate([np.array([x, y, z]), orientation_euler], axis=-1)

        if self.visualize_target:
            self.set_object_pose(self.object_bodies[self.goal_object_key+'_target'],
                                 self.desired_goal[:3],
                                 target_obj_quat)

    def _generate_goal_image(self):
        self.robot.set_kuka_joint_state(self.robot.kuka_away_pose)

        # Push task
        original_obj_pos, original_obj_quat = self._p.getBasePositionAndOrientation(self.object_bodies[self.goal_object_key])
        target_obj_pos = self.desired_goal.copy()[:3]
        target_obj_euler = self.desired_goal.copy()[3:]
        target_obj_quat = quat.as_float_array(quat.from_euler_angles(target_obj_euler))
        target_obj_quat = np.concatenate([target_obj_quat[1:], [target_obj_quat[0]]], axis=-1)
        self.set_object_pose(self.object_bodies[self.goal_object_key],
                             target_obj_pos,
                             target_obj_quat)
        self.desired_goal_image = self.render(mode=self.render_mode, camera_id=self.goal_cam_id)
        self.set_object_pose(self.object_bodies[self.goal_object_key],
                             original_obj_pos,
                             original_obj_quat)

        self.robot.set_kuka_joint_state(self.robot.kuka_rest_pose)


class KukaPushReachEnv(KukaBulletPrimitiveEnv):
    def __init__(self, render=True, binary_reward=True, distance_threshold=0.05,
                 primitive='discrete_push',
                 image_observation=False, goal_image=False, depth_image=False, pcd=False, visualize_target=False,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw'):
        KukaBulletPrimitiveEnv.__init__(self, render=render, binary_reward=binary_reward,
                                        distance_threshold=distance_threshold,
                                        image_observation=image_observation, goal_image=goal_image,
                                        depth_image=depth_image, pcd=pcd,
                                        visualize_target=visualize_target, regenerate_goal_when_step=False,
                                        camera_setup=camera_setup, observation_cam_id=observation_cam_id,
                                        goal_cam_id=goal_cam_id,
                                        gripper_type=gripper_type, obj_range=0.1, target_range=0.15,
                                        end_effector_start_on_table=False,
                                        grasping=False, primitive=primitive,
                                        manipulated_object_keys=['rectangle'], goal_object_key='rectangle',
                                        orientation_informed_goal=True)

    def _generate_goal(self):
        original_obj_pos, original_obj_quat = self._p.getBasePositionAndOrientation(self.object_bodies[self.goal_object_key])

        while True:
            target_xyz = self.np_random.uniform(self.robot.target_bound_lower,
                                                       self.robot.target_bound_upper)
            # on the table
            target_xyz[-1] = self.object_initial_pos[self.goal_object_key][2]
            if np.linalg.norm(target_xyz - original_obj_pos) > 0.06:
                break
        target_obj_euler = quat.as_euler_angles(quat.as_quat_array([1., 0., 0., 0.]))
        target_obj_euler[-1] = self.np_random.uniform(-1.0, 1.0) * np.pi
        target_obj_quat = quat.as_float_array(quat.from_euler_angles(target_obj_euler))
        target_obj_quat = np.concatenate([target_obj_quat[1:], [target_obj_quat[0]]], axis=-1)

        self.desired_goal = np.concatenate((target_xyz, target_obj_euler))

        if self.visualize_target:
            self.set_object_pose(self.object_bodies[self.goal_object_key+'_target'],
                                 target_xyz,
                                 target_obj_quat)

    def _generate_goal_image(self):
        self.robot.set_kuka_joint_state(self.robot.kuka_away_pose)

        # Push task
        original_obj_pos, original_obj_quat = self._p.getBasePositionAndOrientation(self.object_bodies[self.goal_object_key])
        target_obj_pos = self.desired_goal.copy()[:3]
        target_obj_euler = self.desired_goal.copy()[3:]
        target_obj_quat = quat.as_float_array(quat.from_euler_angles(target_obj_euler))
        target_obj_quat = np.concatenate([target_obj_quat[1:], [target_obj_quat[0]]], axis=-1)
        self.set_object_pose(self.object_bodies[self.goal_object_key],
                             target_obj_pos,
                             target_obj_quat)
        self.desired_goal_image = self.render(mode=self.render_mode, camera_id=self.goal_cam_id)
        self.set_object_pose(self.object_bodies[self.goal_object_key],
                             original_obj_pos,
                             original_obj_quat)

        self.robot.set_kuka_joint_state(self.robot.kuka_rest_pose)