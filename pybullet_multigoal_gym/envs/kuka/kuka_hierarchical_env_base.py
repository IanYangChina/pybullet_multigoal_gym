import os
import numpy as np
from pybullet_multigoal_gym.envs.hierarchical_env_bases import HierarchicalBaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka


class HierarchicalKukaBulletMGEnv(HierarchicalBaseBulletMGEnv):
    def __init__(self, render=True,
                 binary_reward=True, distance_threshold=0.02, table_type='table', target_on_table=False,
                 grasping=False, has_obj=False, randomized_obj_pos=True, obj_range=0.15):
        self.object_assets_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "objects")
        self.objects_urdf_loaded = False
        self.binary_reward = binary_reward
        self.distance_threshold = distance_threshold
        self.desired_sub_goal = None
        self.desired_sub_goal_image = None
        self.desired_final_goal = None
        self.desired_final_goal_image = None
        self.sub_goal_space, self.final_goal_space, self.goal_images = None, None, None
        self.grasping = grasping
        self.gripper_tip_offset = 0.015
        self.has_obj = has_obj
        self.randomized_obj_pos = randomized_obj_pos
        self.obj_range = obj_range
        self.object_bodies = {
            'table': None,
            'block': None,
            'block_target': None,
            'grip_target': None,
        }
        self.object_initial_pos = {
            'table': [-0.42, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0],
            'block': [-0.42, 0.0, 0.175, 0.0, 0.0, 0.0, 1.0],
            'block_target': [-0.42, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
            'grip_target': [-0.42, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
        }
        self.table_type = table_type
        assert self.table_type == 'table', 'currently only support normal table'
        self.target_one_table = target_on_table

        HierarchicalBaseBulletMGEnv.__init__(self, robot=Kuka(grasping=grasping), render=render, seed=0,
                                             use_real_time_simulation=False,
                                             gravity=9.81, timestep=0.002, frame_skip=20, num_solver_iterations=5)

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
            self.set_object_pose(self.object_bodies['block'],
                                 object_xyz_1,
                                 self.object_initial_pos['block'][3:])
        else:
            self.set_object_pose(self.object_bodies['block'],
                                 self.object_initial_pos['block'][:3],
                                 self.object_initial_pos['block'][3:])

        # Generate goal spaces
        self.sub_goal_space, self.final_goal_space, self.goal_images = self._generate_goal()
        self._sample_goal()

    def _generate_goal(self):
        raise NotImplementedError()

    def _sample_goal(self):
        sub_goal_ind = self.np_random.random_integers(0, len(self.sub_goal_space) - 1)
        key = list(self.sub_goal_space.keys())[sub_goal_ind]
        self.desired_sub_goal = self.sub_goal_space[key].copy()
        self.desired_sub_goal_image = self.goal_images[key].copy()
        self.desired_final_goal = self.final_goal_space['place'].copy()
        self.desired_final_goal_image = self.goal_images['place'].copy()

        # set target poses
        self.set_object_pose(self.object_bodies['block_target'],
                             self.desired_sub_goal[-3:],
                             self.object_initial_pos['block_target'][3:])
        self.set_object_pose(self.object_bodies['grip_target'],
                             self.desired_sub_goal[:3],
                             self.object_initial_pos['grip_target'][3:])

    def _step_callback(self):
        pass

    def _get_obs(self):
        # robot state contains gripper xyz coordinates, orientation (and finger width)
        state = self.robot.calc_robot_state()
        assert self.desired_sub_goal is not None
        block_pos, block_quat = self._p.getBasePositionAndOrientation(self.object_bodies['block'])
        block_linear_vel, block_angular_vel = self._p.getBaseVelocity(self.object_bodies['block'])
        state = np.concatenate((state, block_pos, block_quat, block_linear_vel, block_angular_vel))

        achieved_goal = np.concatenate((state[:3].copy(), [state[6].copy()], np.array(block_pos).copy()))

        observation = self.render(mode='rgb_array')

        return {
            'observation': observation.copy(),
            'state': state.copy(),
            'achieved_sub_goal': achieved_goal.copy(),
            'achieved_sub_goal_image': observation.copy(),
            'desired_sub_goal': self.desired_sub_goal.copy(),
            'desired_sub_goal_image': self.desired_sub_goal_image.copy(),
            'achieved_final_goal': achieved_goal.copy(),
            'desired_final_goal': self.desired_final_goal.copy(),
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

    def set_object_pose(self, body_id, position, orientation=None):
        if orientation is None:
            orientation = self.object_initial_pos['table'][3:]
        self._p.resetBasePositionAndOrientation(body_id, position, orientation)

    def _generate_goal_image(self, target_finger_status, gripper_target_pos, block_target_pos):
        # set target poses
        self.set_object_pose(self.object_bodies['block_target'],
                             block_target_pos,
                             self.object_initial_pos['block_target'][3:])
        self.set_object_pose(self.object_bodies['grip_target'],
                             gripper_target_pos,
                             self.object_initial_pos['grip_target'][3:])
        # record current poses
        kuka_joint_pos, kuka_joint_vel = self.robot.get_kuka_joint_state()
        finger_joint_pos, finger_joint_vel = self.robot.get_finger_joint_state()
        block_pos, block_quat = self._p.getBasePositionAndOrientation(self.object_bodies['block'])
        # set system to target states
        target_kuka_joint_pos = self.robot.compute_ik(self._p, gripper_target_pos)
        self.robot.set_finger_joint_state(target_finger_status)
        self.robot.set_kuka_joint_state(target_kuka_joint_pos, np.zeros(len(target_kuka_joint_pos)))
        self.set_object_pose(self.object_bodies['block'], block_target_pos)

        # codes for testing reward function
        # block_pos_, _ = self._p.getBasePositionAndOrientation(self.object_bodies['block'])
        # state = self.robot.calc_robot_state()
        # achieved_goal = np.concatenate((state[:3].copy(), [state[-1].copy()], np.array(block_pos_).copy()))
        # desired_goal = np.concatenate((gripper_target_pos, [0.03], block_target_pos))
        # sub_reward, sub_goal_achieved = self._compute_reward(achieved_goal, desired_goal)

        # render an image
        goal_img = self.render(mode='rgb_array')
        # set system state back
        self.robot.set_finger_joint_state(finger_joint_pos[0], finger_joint_vel)
        self.robot.set_kuka_joint_state(kuka_joint_pos, kuka_joint_vel)
        self.set_object_pose(self.object_bodies['block'], block_pos, block_quat)

        return goal_img
