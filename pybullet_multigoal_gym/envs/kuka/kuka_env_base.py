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
                 table_type='table', target_on_table=False,
                 distance_threshold=0.02, grasping=False, has_obj=False, randomized_obj_pos=True, obj_range=0.15):
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
            self.object_initial_pos['block'][0] = -0.65

        self.desired_goal = None
        BaseBulletMGEnv.__init__(self, robot=Kuka(grasping=grasping, gripper_type=gripper_type),
                                 render=render, image_observation=image_observation,
                                 seed=0, timestep=0.002, frame_skip=20)

    def task_reset(self):
        if not self.objects_urdf_loaded:
            self.objects_urdf_loaded = True
            self.object_bodies['table'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, self.table_type+".urdf"),
                basePosition=self.object_initial_pos['table'][:3],
                baseOrientation=self.object_initial_pos['table'][3:])
            self.object_bodies['target'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "target.urdf"),
                basePosition=self.object_initial_pos['target'][:3],
                baseOrientation=self.object_initial_pos['target'][3:])
            if self.has_obj:
                self.object_bodies['block'] = self._p.loadURDF(
                    os.path.join(self.object_assets_path, "block.urdf"),
                    basePosition=self.object_initial_pos['block'][:3],
                    baseOrientation=self.object_initial_pos['block'][3:])

        if self.has_obj:
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
                self.set_object_pose(self.object_bodies['block'],
                                     object_xyz_1,
                                     self.object_initial_pos['block'][3:])
            else:
                self.set_object_pose(self.object_bodies['block'],
                                     self.object_initial_pos['block'][:3],
                                     self.object_initial_pos['block'][3:])

        self._generate_goal()

    def _generate_goal(self):
        end_effector_tip_initial_position = self.robot.end_effector_tip_initial_position.copy()
        self.desired_goal = end_effector_tip_initial_position + \
                            self.np_random.uniform(-self.obj_range, self.obj_range, size=3)
        if self.table_type == 'long_table':
            self.desired_goal[0] -= 0.60
        if self.target_one_table:
            self.desired_goal[2] = self.object_initial_pos['block'][2]
        self.set_object_pose(self.object_bodies['target'],
                             self.desired_goal,
                             self.object_initial_pos['target'][3:])

    def _step_callback(self):
        pass

    def _get_obs(self):
        # robot state contains gripper xyz coordinates, orientation (and finger width)
        state = self.robot.calc_robot_state()
        assert self.desired_goal is not None
        if self.has_obj:
            block_pos, block_quat = self._p.getBasePositionAndOrientation(self.object_bodies['block'])
            block_linear_vel, block_angular_vel = self._p.getBaseVelocity(self.object_bodies['block'])
            state = np.concatenate((state, block_pos, block_quat, block_linear_vel, block_angular_vel))
            achieved_goal = np.array(block_pos).copy()
        else:
            achieved_goal = state[:3].copy()

        if not self.image_observation:
            return {
                'state': state,
                'achieved_goal': achieved_goal,
                'desired_goal': self.desired_goal.copy(),
            }
        else:
            return {
                'observation': self.render(mode='rgb_array'),
                'state': state,
                'achieved_goal': achieved_goal,
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
