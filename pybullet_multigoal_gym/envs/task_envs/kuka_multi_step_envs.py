import numpy as np
from pybullet_multigoal_gym.envs.base_envs.kuka_multi_step_base_env import KukaBulletMultiBlockEnv


class KukaBlockStackEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', num_block=5,
                 use_curriculum=False, num_goals_to_generate=1e5):
        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=False,
                                         num_block=num_block, grasping=True, chest=False,
                                         obj_range=0.15, target_range=0.15,
                                         use_curriculum=use_curriculum,
                                         num_curriculum=num_block-1,
                                         num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses):
        desired_goal = []

        # generate a random order of blocks to be stacked
        new_order = np.arange(self.num_block, dtype=np.int)
        self.np_random.shuffle(new_order)
        new_order = new_order.tolist()
        # generate a random base block position
        base_target_xyz = self.np_random.uniform(self.robot.target_bound_lower,
                                                 self.robot.target_bound_upper)
        # make sure the block is on the table surface
        base_target_xyz[-1] = 0.175
        # generate the stacked target poses
        target_xyzs = [base_target_xyz]
        for _ in range(self.num_block - 1):
            next_target_xyz = base_target_xyz.copy()
            next_target_xyz[-1] = 0.175 + self.block_size * (_ + 1)
            target_xyzs.append(next_target_xyz.copy())
        # generate goal and set target poses according to the order
        for _ in range(self.num_block):
            desired_goal.append(target_xyzs[new_order.index(_)])
            if self.visualize_target:
                self.set_object_pose(self.object_bodies[self.target_keys[_]],
                                     desired_goal[-1],
                                     self.object_initial_pos[self.target_keys[_]][3:])

        self.desired_goal = np.concatenate(desired_goal)


class KukaBlockRearrangeEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', num_block=5,
                 use_curriculum=False, num_goals_to_generate=1e5):
        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=True,
                                         num_block=num_block, grasping=False, chest=False,
                                         obj_range=0.15, target_range=0.15,
                                         use_curriculum=use_curriculum,
                                         num_curriculum=num_block,
                                         num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses):
        desired_goal = []

        if not self.curriculum:
            for _ in range(self.num_block):
                done = False
                while not done:
                    new_target_xy = self.np_random.uniform(self.robot.target_bound_lower[:-1],
                                                           self.robot.target_bound_upper[:-1])
                    target_not_overlap = []
                    for pos in desired_goal + block_poses:
                        target_not_overlap.append((np.linalg.norm(new_target_xy - pos[:-1]) > 0.06))
                    if all(target_not_overlap):
                        desired_goal.append(np.concatenate((new_target_xy.copy(), [0.175])))
                        done = True
        else:
            curriculum_level = self.np_random.choice(self.num_block, p=self.curriculum_prob)
            self.curriculum_goal_step = curriculum_level * 25 + self.base_curriculum_episode_steps
            ind_block_to_move = np.sort(self.np_random.choice(np.arange(self.num_block), size=curriculum_level+1),
                                        kind='stable')
            targets = []
            for _ in range(curriculum_level + 1):
                done = False
                while not done:
                    new_target_xy = self.np_random.uniform(self.robot.target_bound_lower[:-1],
                                                           self.robot.target_bound_upper[:-1])
                    target_not_overlap = []
                    for pos in targets + block_poses:
                        target_not_overlap.append((np.linalg.norm(new_target_xy - pos[:-1]) > 0.08))
                    if all(target_not_overlap):
                        targets.append(np.concatenate((new_target_xy.copy(), [0.175])))
                        done = True

            for i in range(self.num_block):
                if i in ind_block_to_move:
                    desired_goal.append(targets[0])
                    del targets[0]
                else:
                    desired_goal.append(block_poses[i])

            if self.curriculum_update:
                self.num_generated_goals_per_curriculum[curriculum_level] += 1
                self.update_curriculum_prob()
                # print(self.curriculum_prob)
                # print(self.num_generated_goals_per_curriculum)

        if self.visualize_target:
            for _ in range(self.num_block):
                self.set_object_pose(self.object_bodies[self.target_keys[_]],
                                     desired_goal[_],
                                     self.object_initial_pos[self.target_keys[_]][3:])

        self.desired_goal = np.concatenate(desired_goal)


class KukaChestPickAndPlaceEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', num_block=5,
                 use_curriculum=False, num_goals_to_generate=1e5):
        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=False,
                                         num_block=num_block, grasping=True, chest=True, chest_door='up_sliding',
                                         obj_range=0.1,
                                         use_curriculum=use_curriculum,
                                         num_curriculum=num_block+1,
                                         num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses):
        desired_goal = []

        # chest pick and place
        chest_center_xyz, _ = self.chest_robot.get_base_pos(self._p)
        chest_center_xyz = np.array(chest_center_xyz)
        chest_center_xyz[0] += 0.05
        chest_center_xyz[2] = 0.175
        if self.visualize_target:
            self.set_object_pose(self.object_bodies['target_chest'],
                                 chest_center_xyz,
                                 self.object_initial_pos['target_chest'][3:])
        for _ in range(self.num_block):
            desired_goal.append(chest_center_xyz)

        self.desired_goal = np.concatenate(desired_goal)


class KukaChestPushEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', num_block=5,
                 use_curriculum=False, num_goals_to_generate=1e5):
        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=True,
                                         num_block=num_block, grasping=False, chest=True, chest_door='front_sliding',
                                         obj_range=0.1,
                                         use_curriculum=use_curriculum,
                                         num_curriculum=num_block+1,
                                         num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses):
        desired_goal = []

        # chest pick and place
        chest_center_xyz, _ = self.chest_robot.get_base_pos(self._p)
        chest_center_xyz = np.array(chest_center_xyz)
        chest_center_xyz[0] += 0.05
        chest_center_xyz[2] = 0.175
        if self.visualize_target:
            self.set_object_pose(self.object_bodies['target_chest'],
                                 chest_center_xyz,
                                 self.object_initial_pos['target_chest'][3:])
        for _ in range(self.num_block):
            desired_goal.append(chest_center_xyz)

        self.desired_goal = np.concatenate(desired_goal)
