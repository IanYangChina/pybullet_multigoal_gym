import os
import numpy as np
from pybullet_multigoal_gym.envs.env_bases import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka


class KukaBulletMGEnv(BaseBulletMGEnv):
    """
    Base class for non-hierarchical multi-goal RL task with a Kuka iiwa 14 robot
    """

    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, gripper_type='parallel_jaw',
                 table_type='table', target_on_table=False, end_effector_start_on_table=False,
                 distance_threshold=0.01, grasping=False, has_obj=False, randomized_obj_pos=True, obj_range=0.15):
        self.binary_reward = binary_reward
        self.image_observation = image_observation

        self.table_type = table_type
        assert self.table_type in ['table', 'long_table']
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
            'target': None
        }
        self.object_initial_pos = {
            'table': [-0.45, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0],
            'block': [-0.45, 0.0, 0.175, 0.0, 0.0, 0.0, 1.0],
            'target': [-0.45, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0]
        }
        if self.table_type == 'long_table':
            self.object_initial_pos['table'][0] = -0.90
            self.object_initial_pos['block'][0] = -0.50
            self.object_initial_pos['block'][2] = 0.170

        self.desired_goal = None
        BaseBulletMGEnv.__init__(self,
                                 robot=Kuka(grasping=grasping,
                                            gripper_type=gripper_type,
                                            end_effector_start_on_table=end_effector_start_on_table),
                                 render=render, image_observation=image_observation,
                                 seed=0, timestep=0.002, frame_skip=20)
        self.object_bound_lower = self.robot.end_effector_xyz_lower.copy()
        self.object_bound_lower[0] += 0.03
        self.object_bound_upper = self.robot.end_effector_xyz_upper.copy()
        self.object_bound_upper[0] -= 0.03

    def task_reset(self):
        if not self.objects_urdf_loaded:
            self.objects_urdf_loaded = True
            self.object_bodies['table'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, self.table_type + ".urdf"),
                basePosition=self.object_initial_pos['table'][:3],
                baseOrientation=self.object_initial_pos['table'][3:])
            self.object_bodies['target'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "target.urdf"),
                basePosition=self.object_initial_pos['target'][:3],
                baseOrientation=self.object_initial_pos['target'][3:])
            if self.has_obj:
                if self.table_type == 'long_table':
                    self.object_bodies['block'] = self._p.loadURDF(
                        os.path.join(self.object_assets_path, "cylinder_bulk.urdf"),
                        basePosition=self.object_initial_pos['block'][:3],
                        baseOrientation=self.object_initial_pos['block'][3:])
                else:
                    self.object_bodies['block'] = self._p.loadURDF(
                        os.path.join(self.object_assets_path, "block.urdf"),
                        basePosition=self.object_initial_pos['block'][:3],
                        baseOrientation=self.object_initial_pos['block'][3:])

        object_xyz_1 = None
        if self.has_obj:
            if self.randomized_obj_pos:
                end_effector_tip_initial_position = self.robot.end_effector_tip_initial_position.copy()
                object_xy_1 = end_effector_tip_initial_position[:2]
                object_xy_2 = end_effector_tip_initial_position[:2]
                while (np.linalg.norm(object_xy_1 - end_effector_tip_initial_position[:2]) < 0.02) or \
                        (np.linalg.norm(object_xy_1 - object_xy_2[:2]) < 0.02):
                    object_xy_1 = self.np_random.uniform(self.robot.object_bound_lower[:-1],
                                                         self.robot.object_bound_upper[:-1])

                object_xyz_1 = np.append(object_xy_1, self.object_initial_pos['block'][2])
                self.set_object_pose(self.object_bodies['block'],
                                     object_xyz_1,
                                     self.object_initial_pos['block'][3:])
            else:
                self.set_object_pose(self.object_bodies['block'],
                                     self.object_initial_pos['block'][:3],
                                     self.object_initial_pos['block'][3:])

        self._generate_goal(current_obj_pos=object_xyz_1)

    def _generate_goal(self, current_obj_pos=None):
        if current_obj_pos is None:
            center = self.robot.end_effector_tip_initial_position[:2].copy()
        else:
            center = current_obj_pos[:2]
        self.desired_goal = self.np_random.uniform(self.robot.object_bound_lower,
                                                   self.robot.object_bound_upper)
        while np.linalg.norm(self.desired_goal[:2] - center) < 0.02:
            self.desired_goal = self.np_random.uniform(self.robot.object_bound_lower,
                                                       self.robot.object_bound_upper)
        if self.table_type == 'long_table':
            x = self.np_random.uniform(self.robot.object_bound_lower[0]-0.4,
                                       self.robot.object_bound_upper[0]-0.4,
                                       size=1)
            self.desired_goal[0] = x
        if self.target_one_table:
            self.desired_goal[2] = self.object_initial_pos['block'][2]

        self.set_object_pose(self.object_bodies['target'],
                             self.desired_goal,
                             self.object_initial_pos['target'][3:])

    def _step_callback(self):
        pass

    def _get_obs(self):
        # robot state contains gripper xyz coordinates, orientation (and finger width)
        gripper_xyz, gripper_rpy, gripper_finger_closeness, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_vel = self.robot.calc_robot_state()
        assert self.desired_goal is not None
        policy_state = state = gripper_xyz
        achieved_goal = gripper_xyz.copy()
        if self.has_obj:
            block_xyz, _ = self._p.getBasePositionAndOrientation(self.object_bodies['block'])
            block_rel_xyz = gripper_xyz - np.array(block_xyz)
            block_vel_xyz, block_vel_rpy = self._p.getBaseVelocity(self.object_bodies['block'])
            block_rel_vel_xyz = gripper_vel_xyz - np.array(block_vel_xyz)
            block_rel_vel_rpy = gripper_vel_rpy - np.array(block_vel_rpy)
            achieved_goal = np.array(block_xyz).copy()
            # the HER paper use different state observations for the policy and critic network
            # critic further takes the velocities as input
            state = np.concatenate((gripper_xyz, gripper_finger_closeness, block_rel_xyz,
                                    gripper_vel_xyz, gripper_finger_vel, block_rel_vel_xyz, block_rel_vel_rpy))
            policy_state = np.concatenate((gripper_xyz, gripper_finger_closeness, block_rel_xyz))
        else:
            assert not self.grasping, "grasping should not be true when there is no objects"

        if not self.image_observation:
            return {
                'state': state.copy(),
                'policy_state': policy_state.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.desired_goal.copy(),
            }
        else:
            return {
                'observation': self.render(mode='rgb_array'),
                'state': state.copy(),
                'policy_state': policy_state.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.desired_goal.copy(),
            }

    def _compute_reward(self, achieved_goal, desired_goal):
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        not_achieved = (d > self.distance_threshold)
        if self.binary_reward:
            return -not_achieved.astype(np.float32), ~not_achieved
        else:
            return -d, ~not_achieved

    def set_object_pose(self, body_id, position, orientation=None):
        if orientation is None:
            orientation = self.object_initial_pos['table'][3:]
        self._p.resetBasePositionAndOrientation(body_id, position, orientation)
