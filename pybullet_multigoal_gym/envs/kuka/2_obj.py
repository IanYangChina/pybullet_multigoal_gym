import os
import numpy as np
from pybullet_multigoal_gym.envs.env_bases import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka


class Kuka2ObjEnv(BaseBulletMGEnv):
    def __init__(self, render=True,
                 binary_reward=True, distance_threshold=0.02,
                 randomized_obj_pos=True, obj_range=0.15):
        robot = Kuka()
        self.object_assets_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "objects")
        self.objects_urdf_loaded = False
        self.binary_reward = binary_reward
        self.distance_threshold = distance_threshold
        self.desired_goal = None
        self.randomized_obj_pos = randomized_obj_pos
        self.obj_range = obj_range
        self.object_bodies = {
            'table': None,
            'red_block': None,
            'blue_block': None
        }
        self.object_initial_pos = {
            'table': [-0.42, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0],
            'red_block': [-0.42, 0.1, 0.186, 0.0, 0.0, 0.0, 1.0],
            'blue_block': [-0.42, -0.1, 0.186, 0.0, 0.0, 0.0, 1.0]
        }

        BaseBulletMGEnv.__init__(self, robot=robot, render=render, seed=0,
                                 use_real_time_simulation=False,
                                 gravity=9.81, timestep=0.002, frame_skip=20, num_solver_iterations=5)

    def task_reset(self):
        if not self.objects_urdf_loaded:
            self.objects_urdf_loaded = True
            self.object_bodies['table'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "table.urdf"),
                basePosition=self.object_initial_pos['table'][:3],
                baseOrientation=self.object_initial_pos['table'][3:])
            self.object_bodies['red_block'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "red_block.urdf"),
                basePosition=self.object_initial_pos['red_block'][:3],
                baseOrientation=self.object_initial_pos['red_block'][3:])
            self.object_bodies['blue_block'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "blue_block.urdf"),
                basePosition=self.object_initial_pos['blue_block'][:3],
                baseOrientation=self.object_initial_pos['blue_block'][3:])

        if self.randomized_obj_pos:
            end_effector_tip_initial_position = self.robot.end_effector_tip_initial_position.copy()
            object_xy_1 = end_effector_tip_initial_position[:2]
            object_xy_2 = end_effector_tip_initial_position[:2]
            while (np.linalg.norm(object_xy_1 - end_effector_tip_initial_position[:2]) < 0.1) or \
                    (np.linalg.norm(object_xy_1 - object_xy_2[:2]) < 0.1):
                object_xy_1 = end_effector_tip_initial_position[:2] + \
                              self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                object_xy_2 = end_effector_tip_initial_position[:2] + \
                              self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_xyz_1 = self.object_initial_pos['red_block'][:3].copy()
            object_xyz_1[:2] = object_xy_1
            object_xyz_2 = self.object_initial_pos['blue_block'][:3].copy()
            object_xyz_2[:2] = object_xy_2
            self.set_object_pose(self.object_bodies['red_block'],
                                 object_xyz_1,
                                 self.object_initial_pos['red_block'][3:])
            self.set_object_pose(self.object_bodies['blue_block'],
                                 object_xyz_2,
                                 self.object_initial_pos['blue_block'][3:])
        else:
            self.set_object_pose(self.object_bodies['red_block'],
                                 self.object_initial_pos['red_block'][:3],
                                 self.object_initial_pos['red_block'][3:])
            self.set_object_pose(self.object_bodies['blue_block'],
                                 self.object_initial_pos['blue_block'][:3],
                                 self.object_initial_pos['blue_block'][3:])

        self._generate_goal()

    def _generate_goal(self):
        self.desired_goal, _ = self._p.getBasePositionAndOrientation(self.object_bodies['red_block'])[:3]
        self.desired_goal = np.array(self.desired_goal)
        self.desired_goal[2] += 0.1

    def _step_callback(self):
        pass

    def _get_obs(self):
        # robot state contains gripper xyz coordinates & finger width
        robot_state = self.robot.calc_robot_state()
        assert self.desired_goal is not None
        return {
            'state': robot_state,
            'achieved_goal': robot_state[:-1],
            'desired_goal': self.desired_goal,
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
        # todo: reset causes object to disappear
        if orientation is None:
            orientation = self.object_initial_pos['table'][3:]
        self._p.resetBasePositionAndOrientation(body_id, position, orientation)
