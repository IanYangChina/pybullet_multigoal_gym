import numpy as np
from copy import deepcopy as dcp
from pybullet_multigoal_gym.utils.demonstrator import StepDemonstrator
from pybullet_multigoal_gym.envs.kuka.kuka_hierarchical_env_base import HierarchicalKukaBulletMGEnv


class HierarchicalKukaPickAndPlaceEnv(HierarchicalKukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, image_observation=False, gripper_type='parallel_jaw'):
        self.step_demonstrator = StepDemonstrator([
            [0],
            [0, 1]
        ])
        HierarchicalKukaBulletMGEnv.__init__(self,
                                             render=render,
                                             binary_reward=binary_reward,
                                             image_observation=image_observation,
                                             gripper_type=gripper_type,
                                             num_steps=self.step_demonstrator.demon_num,
                                             distance_threshold=0.02,
                                             grasping=True, has_obj=True, randomized_obj_pos=True)

    def _generate_goal(self):
        block_pos, _ = self._p.getBasePositionAndOrientation(self.object_bodies['block'])
        block_pos = np.array(block_pos)
        end_effector_tip_initial_position = self.robot.end_effector_tip_initial_position.copy()
        block_target_position = end_effector_tip_initial_position + \
                                self.np_random.uniform(-self.obj_range, self.obj_range, size=3)
        if self.target_one_table:
            block_target_position = self.object_initial_pos['block'][2]

        picking_grip_pos = block_pos.copy()
        picking_grip_pos[-1] += self.robot.gripper_tip_offset
        placing_grip_pos = block_target_position.copy()
        placing_grip_pos[-1] += self.robot.gripper_tip_offset
        sub_goals = {
            "pick": np.concatenate([
                # gripper xyz & finger width
                picking_grip_pos, [0.03],
                # absolute positions of blocks
                block_pos,
            ]),
            "place": np.concatenate([
                # gripper xyz & finger width
                placing_grip_pos, [0.03],
                # absolute positions of blocks
                block_target_position,
            ]),
        }
        final_goals = dcp(sub_goals)
        if not self.image_observation:
            return sub_goals, final_goals, None
        else:
            goal_images = {
                "pick": self._generate_goal_image(self.robot.gripper_grasp_block_state, picking_grip_pos, block_pos),
                "place": self._generate_goal_image(self.robot.gripper_grasp_block_state, placing_grip_pos, block_target_position),
            }
            return sub_goals, final_goals, goal_images

    def _generate_goal_image(self, target_finger_status, gripper_target_pos, block_target_pos):
        # set target poses
        self._set_object_pose(self.object_bodies['block_target'],
                              block_target_pos,
                              self.object_initial_pos['block_target'][3:])
        self._set_object_pose(self.object_bodies['grip_target'],
                              gripper_target_pos,
                              self.object_initial_pos['grip_target'][3:])
        # record current poses
        kuka_joint_pos, kuka_joint_vel = self.robot.get_kuka_joint_state()
        finger_joint_pos, finger_joint_vel = self.robot.get_finger_joint_state()
        block_pos, block_quat = self._p.getBasePositionAndOrientation(self.object_bodies['block'])
        # set system to target states
        target_kuka_joint_pos = self.robot.compute_ik(self._p, gripper_target_pos)
        self.robot.set_finger_joint_state(target_finger_status)
        self.robot.set_kuka_joint_state(target_kuka_joint_pos)
        self._set_object_pose(self.object_bodies['block'], block_target_pos)

        # codes for testing reward function
        # block_pos_, _ = self._p.getBasePositionAndOrientation(self.object_bodies['block'])
        # gripper_xyz, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_closeness, gripper_finger_vel = self.robot.calc_robot_state()
        # achieved_goal = np.concatenate((gripper_xyz.copy(), gripper_finger_closeness, np.array(block_pos_).copy()))
        # desired_goal = np.concatenate((gripper_target_pos, [0.03], block_target_pos))
        # sub_reward, sub_goal_achieved = self._compute_reward(achieved_goal, desired_goal)

        # render an image
        goal_img = self.render(mode='rgb_array')
        # set system state back
        self.robot.set_finger_joint_state(finger_joint_pos[0], finger_joint_vel)
        self.robot.set_kuka_joint_state(kuka_joint_pos, kuka_joint_vel)
        self._set_object_pose(self.object_bodies['block'], block_pos, block_quat)

        return goal_img
