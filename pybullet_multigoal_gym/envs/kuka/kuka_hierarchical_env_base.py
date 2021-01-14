import os
import numpy as np
from pybullet_multigoal_gym.envs.hierarchical_env_bases import HierarchicalBaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka


class HierarchicalKukaBulletMGEnv(HierarchicalBaseBulletMGEnv):
    """
    Base class for hierarchical multi-goal RL task with a Kuka iiwa 14 robot
    """

    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, gripper_type='parallel_jaw',
                 num_steps=2,
                 table_type='table', target_on_table=False,
                 distance_threshold=0.02, grasping=False, has_obj=False, randomized_obj_pos=True, obj_range=0.15):
        self.binary_reward = binary_reward
        self.image_observation = image_observation

        self.table_type = table_type
        assert self.table_type == 'table', 'currently only support normal table'
        self.target_one_table = target_on_table
        self.distance_threshold = distance_threshold
        self.grasping = grasping
        self.has_obj = has_obj
        self.randomized_obj_pos = randomized_obj_pos
        self.obj_range = obj_range

        self.object_assets_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "objects")
        self.objects_urdf_loaded = False
        self.object_bodies = {
            'table': None,
            'block': None,
            'block_target': None,
            'grip_target': None,
        }
        self.object_initial_pos = {
            'table': [-0.45, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0],
            'block': [-0.45, 0.0, 0.175, 0.0, 0.0, 0.0, 1.0],
            'block_target': [-0.45, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
            'grip_target': [-0.45, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
        }

        self.num_steps = num_steps
        self.sub_goal_space, self.final_goal_space, self.goal_images = None, None, None
        self.sub_goal_strings, self.final_goal_strings = None, None
        self.desired_sub_goal = None
        self.desired_sub_goal_ind = None
        self.desired_sub_goal_image = None
        self.desired_final_goal = None
        self.desired_final_goal_ind = None
        self.desired_final_goal_image = None
        HierarchicalBaseBulletMGEnv.__init__(self, robot=Kuka(grasping=grasping, gripper_type=gripper_type),
                                             render=render, image_observation=image_observation,
                                             num_steps=self.num_steps,
                                             seed=0, gravity=9.81, timestep=0.002, frame_skip=20)

    def task_reset(self):
        # Load objects
        if not self.objects_urdf_loaded:
            self.objects_urdf_loaded = True
            self.object_bodies['table'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, self.table_type + ".urdf"),
                basePosition=self.object_initial_pos['table'][:3],
                baseOrientation=self.object_initial_pos['table'][3:])
            self.object_bodies['block_target'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "target.urdf"),
                basePosition=self.object_initial_pos['block_target'][:3],
                baseOrientation=self.object_initial_pos['block_target'][3:])
            self.object_bodies['grip_target'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "grip_target.urdf"),
                basePosition=self.object_initial_pos['grip_target'][:3],
                baseOrientation=self.object_initial_pos['grip_target'][3:])
            self.object_bodies['block'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "block.urdf"),
                basePosition=self.object_initial_pos['block'][:3],
                baseOrientation=self.object_initial_pos['block'][3:])

        # Randomize object poses
        if self.randomized_obj_pos:
            end_effector_tip_initial_position = self.robot.end_effector_tip_initial_position.copy()
            object_xy_1 = end_effector_tip_initial_position[:2]
            object_xy_2 = end_effector_tip_initial_position[:2]
            while (np.linalg.norm(object_xy_1 - end_effector_tip_initial_position[:2]) < 0.1) or \
                    (np.linalg.norm(object_xy_1 - object_xy_2[:2]) < 0.1):
                object_xy_1 = end_effector_tip_initial_position[:2] + \
                              self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_xyz_1 = self.object_initial_pos['block'][:3].copy()
            object_xyz_1[:2] = object_xy_1
            self._set_object_pose(self.object_bodies['block'],
                                  object_xyz_1,
                                  self.object_initial_pos['block'][3:])
        else:
            self._set_object_pose(self.object_bodies['block'],
                                  self.object_initial_pos['block'][:3],
                                  self.object_initial_pos['block'][3:])

        # Generate goal spaces
        self.sub_goal_space, self.final_goal_space, self.goal_images = self._generate_goal()
        self.sub_goal_strings = list(self.sub_goal_space.keys())
        self.final_goal_strings = list(self.final_goal_space.keys())
        self._sample_goal()

    def _sample_goal(self):
        goal_ind = self.np_random.random_integers(0, len(self.sub_goal_space) - 1)
        key = self.sub_goal_strings[goal_ind]
        self.desired_sub_goal = self.sub_goal_space[key].copy()
        self.desired_sub_goal_ind = goal_ind
        self.desired_final_goal = self.final_goal_space[key].copy()
        self.desired_final_goal_ind = goal_ind
        if self.image_observation:
            self.desired_sub_goal_image = self.goal_images[key].copy()
            self.desired_final_goal_image = self.goal_images[key].copy()
        self._update_target_objects()

    def _update_target_objects(self):
        # set target poses
        self._set_object_pose(self.object_bodies['block_target'],
                              self.desired_sub_goal[-3:],
                              self.object_initial_pos['block_target'][3:])
        self._set_object_pose(self.object_bodies['grip_target'],
                              self.desired_sub_goal[:3],
                              self.object_initial_pos['grip_target'][3:])

    def set_sub_goal(self, sub_goal, index=True):
        if index:
            key = self.sub_goal_strings[sub_goal]
            self.desired_sub_goal = self.sub_goal_space[key].copy()
            self.desired_sub_goal_ind = sub_goal
            if self.image_observation:
                self.desired_sub_goal_image = self.goal_images[key].copy()
        else:
            assert sub_goal.shape == self.observation_space['desired_sub_goal'].shape
            self.desired_sub_goal = sub_goal.copy()
        self._update_target_objects()
        return self.desired_sub_goal

    def set_final_goal(self, final_goal, index=True):
        if index:
            key = self.final_goal_strings[final_goal]
            self.desired_final_goal = self.final_goal_space[key].copy()
            self.desired_final_goal_ind = final_goal
            if self.image_observation:
                self.desired_final_goal_image = self.goal_images[key].copy()
        else:
            assert final_goal.shape == self.observation_space['desired_final_goal'].shape
            self.desired_final_goal = final_goal.copy()
        return self.desired_final_goal

    def _step_callback(self):
        pass

    def _get_obs(self):
        # robot state, shape=(7,), contains gripper xyz coordinates, orientation (and finger width)
        gripper_xyz, gripper_rpy, gripper_finger_closeness = self.robot.calc_robot_state()
        assert self.desired_sub_goal is not None
        block_xyz, block_quat = self._p.getBasePositionAndOrientation(self.object_bodies['block'])
        block_rpy = self._p.getEulerFromQuaternion(block_quat)
        # block_vel_xyz, block_vel_rpy = self._p.getBaseVelocity(self.object_bodies['block'])
        block_rel_xyz = gripper_xyz - np.array(block_xyz)
        block_rel_rpy = gripper_rpy - np.array(block_rpy)

        state = np.concatenate((gripper_xyz, gripper_finger_closeness, block_rel_xyz, block_rel_rpy))

        achieved_goal = np.concatenate((gripper_xyz.copy(),
                                        gripper_finger_closeness.copy(),
                                        np.array(block_xyz).copy()))
        if not self.image_observation:
            return {
                'state': state.copy(),
                'achieved_sub_goal': achieved_goal.copy(),
                'desired_sub_goal': self.desired_sub_goal.copy(),
                'desired_sub_goal_ind': self.desired_sub_goal_ind,
                'achieved_final_goal': achieved_goal.copy(),
                'desired_final_goal': self.desired_final_goal.copy(),
                'desired_final_goal_ind': self.desired_final_goal_ind,
            }
        else:
            observation = self.render(mode='rgb_array')

            return {
                'observation': observation.copy(),
                'state': state.copy(),
                'achieved_sub_goal': achieved_goal.copy(),
                'achieved_sub_goal_image': observation.copy(),
                'desired_sub_goal': self.desired_sub_goal.copy(),
                'desired_sub_goal_ind': self.desired_sub_goal_ind,
                'desired_sub_goal_image': self.desired_sub_goal_image.copy(),
                'achieved_final_goal': achieved_goal.copy(),
                'desired_final_goal': self.desired_final_goal.copy(),
                'desired_final_goal_ind': self.desired_final_goal_ind,
                'desired_final_goal_image': self.desired_final_goal_image.copy()
            }

    def _compute_reward(self, achieved_goal, desired_goal):
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        not_achieved = (d > self.distance_threshold)
        if self.binary_reward:
            return -not_achieved.astype(np.float32), ~not_achieved
        else:
            return -d, ~not_achieved

    def _set_object_pose(self, body_id, position, orientation=None):
        if orientation is None:
            orientation = self.object_initial_pos['table'][3:]
        self._p.resetBasePositionAndOrientation(body_id, position, orientation)

    def _generate_goal(self):
        raise NotImplementedError()

    def _generate_goal_image(self, target_finger_status, gripper_target_pos, block_target_pos):
        raise NotImplementedError()
