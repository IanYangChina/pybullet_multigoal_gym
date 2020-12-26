import numpy as np
from copy import deepcopy as dcp
from pybullet_multigoal_gym.utils.demonstrator import StepDemonstrator
from pybullet_multigoal_gym.envs.kuka.kuka_hierarchical_env_base import HierarchicalKukaBulletMGEnv


class HierarchicalKukaPickAndPlaceEnv(HierarchicalKukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True):
        self.step_demonstrator = StepDemonstrator([
            [0],
            [0, 1]
        ])
        HierarchicalKukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward,
                                             distance_threshold=0.02,
                                             grasping=True, has_obj=True, randomized_obj_pos=True, obj_range=0.15)

    def _generate_goal(self):
        block_pos, _ = self._p.getBasePositionAndOrientation(self.object_bodies['block'])
        block_pos = np.array(block_pos)
        end_effector_tip_initial_position = self.robot.end_effector_tip_initial_position.copy()
        block_target_position = end_effector_tip_initial_position + \
                                self.np_random.uniform(-self.obj_range, self.obj_range, size=3)
        if self.target_one_table:
            block_target_position = self.object_initial_pos['block'][2]

        picking_grip_pos = block_pos.copy()
        picking_grip_pos[-1] += self.gripper_tip_offset
        placing_grip_pos = block_target_position.copy()
        placing_grip_pos[-1] += self.gripper_tip_offset
        sub_goals = {
            "pick": np.concatenate([
                # gripper state & position
                picking_grip_pos, [0.03],
                # absolute positions of blocks
                block_pos,
            ]),
            "place": np.concatenate([
                # gripper state & position
                placing_grip_pos, [0.03],
                # absolute positions of blocks
                block_target_position,
            ]),
        }
        final_goals = dcp(sub_goals)
        goal_images = {
            "pick": self._generate_goal_image(0.55, picking_grip_pos, block_pos),
            "place": self._generate_goal_image(0.55, placing_grip_pos, block_target_position),
        }

        return sub_goals, final_goals, goal_images
