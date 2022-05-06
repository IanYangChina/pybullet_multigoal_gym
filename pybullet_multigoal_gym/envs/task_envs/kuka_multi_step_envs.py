import numpy as np
from pybullet_multigoal_gym.utils.demonstrator import StepDemonstrator
from pybullet_multigoal_gym.envs.base_envs.kuka_multi_step_base_env import KukaBulletMultiBlockEnv


class KukaBlockStackEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True, distance_threshold=0.05, random_order=True,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', grip_informed_goal=False, num_block=5, joint_control=False,
                 task_decomposition=False, use_curriculum=False, num_goals_to_generate=1e5):

        self.grip_informed_goal = grip_informed_goal
        if self.grip_informed_goal:
            self.num_steps = num_block*2
        else:
            self.num_steps = num_block

        self.random_order = random_order
        self.last_order = None
        self.last_target_poses = None

        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                         grip_informed_goal=self.grip_informed_goal,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=False,
                                         num_block=num_block, obj_range=0.15, target_range=0.15,
                                         joint_control=joint_control, grasping=True, chest=False,
                                         task_decomposition=task_decomposition, use_curriculum=use_curriculum,
                                         num_curriculum=num_block, num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses, new_target=True):
        if new_target:
            # generate a random order of blocks to be stacked
            new_order = np.arange(self.num_block, dtype=int)
            if self.random_order:
                self.np_random.shuffle(new_order)
            new_order = new_order.tolist()

            base_target_xyz = None
            done = False
            while not done:
                base_target_xy = self.np_random.uniform(self.robot.target_bound_lower[:-1],
                                                        self.robot.target_bound_upper[:-1])
                target_not_overlap = []
                for pos in block_poses:
                    target_not_overlap.append((np.linalg.norm(base_target_xy - pos[:-1]) > 0.08))
                if all(target_not_overlap):
                    # put the block is on the table surface
                    base_target_xyz = np.concatenate((base_target_xy, [0.175]))
                    done = True

            # generate the stacked target poses
            target_xyzs = [base_target_xyz]
            for _ in range(self.num_block - 1):
                next_target_xyz = base_target_xyz.copy()
                next_target_xyz[-1] = 0.175 + self.block_size * (_ + 1)
                target_xyzs.append(next_target_xyz.copy())

            self.last_order = new_order
            self.last_target_poses = target_xyzs
        else:
            new_order = self.last_order
            target_xyzs = self.last_target_poses

        if self.curriculum:
            desired_goal = self._generate_curriculum(block_poses, target_xyzs, new_order, new_target)
        else:
            # generate goal and set target poses according to the order
            desired_goal = [None for _ in range(self.num_block)]
            for _ in range(self.num_block):
                desired_goal[new_order[_]] = target_xyzs[_]
            if self.grip_informed_goal:
                desired_goal.append(target_xyzs[-1].copy())
                desired_goal.append([0.03])

            if self.task_decomposition:
                self.sub_goals = self._generate_subgoals(block_poses, target_xyzs, new_order, new_target)

        if self.visualize_target:
            self._update_block_target(desired_goal)
            if self.grip_informed_goal:
                self._update_gripper_target(desired_goal[-2])

        self.desired_goal = np.concatenate(desired_goal)

    def _generate_subgoals(self, block_poses, target_xyzs, new_order, new_target=False):
        sub_goals = []
        if self.grip_informed_goal:
            for _ in range(self.num_block):
                sub_goal_pick = [None for _ in range(self.num_block)]
                for i in range(self.num_block):
                    if i < _:
                        sub_goal_pick[new_order[i]] = target_xyzs[i].copy()
                    else:
                        sub_goal_pick[new_order[i]] = block_poses[new_order[i]].copy()
                sub_goal_pick.append(block_poses[new_order[_]].copy())
                sub_goal_pick.append([0.03])
                sub_goals.append(np.concatenate(sub_goal_pick))

                sub_goal_place = [None for _ in range(self.num_block)]
                for i in range(self.num_block):
                    if i <= _:
                        sub_goal_place[new_order[i]] = target_xyzs[i].copy()
                    else:
                        sub_goal_place[new_order[i]] = block_poses[new_order[i]].copy()
                sub_goal_place.append(target_xyzs[_].copy())
                sub_goal_place.append([0.03])
                sub_goals.append(np.concatenate(sub_goal_place))
        else:
            for _ in range(self.num_block):
                sub_goal = [None for _ in range(self.num_block)]
                for i in range(self.num_block):
                    if i <= _:
                        sub_goal[new_order[i]] = target_xyzs[i].copy()
                    else:
                        sub_goal[new_order[i]] = block_poses[new_order[i]].copy()
                sub_goals.append(np.concatenate(sub_goal))

        return sub_goals

    def _generate_curriculum(self, block_poses, target_xyzs, new_order, new_target=False):
        desired_goal = [None for _ in range(self.num_block)]

        if new_target:
            curriculum_level = self.np_random.choice(self.num_curriculum, p=self.curriculum_prob)
            self.curriculum_goal_step = curriculum_level * 25 + self.base_curriculum_episode_steps
            self.last_curriculum_level = curriculum_level

            if self.curriculum_update:
                self.num_generated_goals_per_curriculum[curriculum_level] += 1
                self._update_curriculum_prob()
        else:
            curriculum_level = self.last_curriculum_level

        for i in range(self.num_block):
            if i <= curriculum_level:
                desired_goal[new_order[i]] = target_xyzs[i].copy()
            else:
                desired_goal[new_order[i]] = block_poses[new_order[i]].copy()

        if self.grip_informed_goal:
            desired_goal.append(target_xyzs[curriculum_level].copy())
            desired_goal.append([0.03])

        return desired_goal


class KukaBlockRearrangeEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True, distance_threshold=0.05,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', grip_informed_goal=False, num_block=5, joint_control=False,
                 task_decomposition=False, use_curriculum=False, num_goals_to_generate=1e5):

        assert not grip_informed_goal, "Block rearranging task does not support gripper informed goal representation."
        assert not task_decomposition, "Block rearranging task does not support task decomposition."

        self.last_target_poses = None

        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                         grip_informed_goal=False,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=True,
                                         num_block=num_block, obj_range=0.15, target_range=0.15,
                                         joint_control=joint_control, grasping=False, chest=False,
                                         task_decomposition=task_decomposition, use_curriculum=use_curriculum,
                                         num_curriculum=num_block, num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses, new_target=True):
        if new_target:
            target_xyzs = []
            for _ in range(self.num_block):
                done = False
                while not done:
                    new_target_xy = self.np_random.uniform(self.robot.target_bound_lower[:-1],
                                                           self.robot.target_bound_upper[:-1])
                    target_not_overlap = []
                    for pos in target_xyzs + block_poses:
                        target_not_overlap.append((np.linalg.norm(new_target_xy - pos[:-1]) > 0.06))
                    if all(target_not_overlap):
                        target_xyzs.append(np.concatenate((new_target_xy.copy(), [0.175])))
                        done = True
            self.last_target_poses = target_xyzs.copy()
        else:
            target_xyzs = self.last_target_poses.copy()

        if not self.curriculum:
            desired_goal = target_xyzs
        else:
            desired_goal = self._generate_curriculum(block_poses, target_xyzs, new_target)

        if self.visualize_target:
            self._update_block_target(desired_goal)

        self.desired_goal = np.concatenate(desired_goal)

    def _generate_curriculum(self, block_poses, target_xyzs, new_target=False):
        desired_goal = []

        if new_target:
            curriculum_level = self.np_random.choice(self.num_curriculum, p=self.curriculum_prob)
            self.curriculum_goal_step = curriculum_level * 25 + self.base_curriculum_episode_steps
            ind_block_to_move = np.sort(self.np_random.choice(np.arange(self.num_block),
                                                              size=curriculum_level + 1,
                                                              replace=False),
                                        kind='stable').tolist()
            self.last_ind_block_to_move = ind_block_to_move

            if self.curriculum_update:
                self.num_generated_goals_per_curriculum[curriculum_level] += 1
                self._update_curriculum_prob()
        else:
            ind_block_to_move = self.last_ind_block_to_move

        for i in range(self.num_block):
            if i in ind_block_to_move:
                desired_goal.append(target_xyzs[0].copy())
                del target_xyzs[0]
            else:
                desired_goal.append(block_poses[i].copy())

        return desired_goal


class KukaChestPickAndPlaceEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True, distance_threshold=0.05,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', grip_informed_goal=False, num_block=5, joint_control=False,
                 task_decomposition=False, use_curriculum=False, num_goals_to_generate=1e5):

        assert not goal_image, "Chest tasks do not support goal images well at the moment."

        self.grip_informed_goal = grip_informed_goal
        if self.grip_informed_goal:
            self.num_steps = num_block*3+1
        else:
            self.num_steps = num_block + 1

        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                         grip_informed_goal=self.grip_informed_goal,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=False,
                                         num_block=num_block, obj_range=0.1,
                                         joint_control=joint_control, grasping=True, chest=True, chest_door='up_sliding',
                                         task_decomposition=task_decomposition, use_curriculum=use_curriculum,
                                         num_curriculum=num_block+1, num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses, new_target=True):
        # all blocks should go into the sphere of 0.05 radius centred at the chest centre
        chest_center_xyz, _ = self.chest_robot.get_base_pos()
        chest_center_xyz = np.array(chest_center_xyz)
        chest_center_xyz[0] += 0.05
        chest_center_xyz[2] = 0.175
        chest_top_xyz = chest_center_xyz.copy()
        chest_top_xyz[-1] = 0.3

        # setup desired goal or subgoals
        if self.curriculum:
            desired_goal = self._generate_curriculum(block_poses, chest_center_xyz, chest_top_xyz, new_target)
        else:
            # the first element is the largest openness of the door (equal to the door joint pose upper limit)
            desired_goal = [[0.10]]
            for _ in range(self.num_block):
                desired_goal.append(chest_center_xyz)
            if self.grip_informed_goal:
                desired_goal.append(chest_top_xyz.copy())
                desired_goal.append([0.06])

            if self.task_decomposition:
                self.sub_goals = self._generate_subgoals(block_poses, chest_center_xyz, chest_top_xyz, new_target)

        if self.visualize_target:
            self._update_block_target(desired_goal, index_offset=1)
            if self.grip_informed_goal:
                self._update_gripper_target(desired_goal[-2])

        self.desired_goal = np.concatenate(desired_goal)

    def _generate_subgoals(self, block_poses, chest_center_xyz, chest_top_xyz, new_target=False):
        sub_goals = []

        sub_goal_open_door = [[0.10]]
        sub_goal_open_door = sub_goal_open_door + block_poses
        sub_goal_open_door.append(self.robot.parts['iiwa_gripper_tip'].get_position().copy())
        sub_goal_open_door.append(self.robot.get_finger_closeness())
        sub_goals.append(np.concatenate(sub_goal_open_door))

        if self.grip_informed_goal:
            for _ in range(self.num_block):
                # # block positions
                sub_goal_pick = block_poses.copy()
                # previous blocks should already be within the chest
                for i in range(self.num_block):
                    if i < _:
                        sub_goal_pick[i] = chest_center_xyz.copy()
                # gripper position
                sub_goal_pick.append(block_poses[_].copy())
                # finger width
                sub_goal_pick.append([0.03])
                # chest door joint state
                sub_goal_pick = [[0.10]] + sub_goal_pick
                sub_goals.append(np.concatenate(sub_goal_pick))

                sub_goal_move_to_chest_top = block_poses.copy()
                for i in range(self.num_block):
                    if i < _:
                        sub_goal_move_to_chest_top[i] = chest_center_xyz.copy()
                sub_goal_move_to_chest_top[_] = chest_top_xyz.copy()
                sub_goal_move_to_chest_top.append(chest_top_xyz.copy())
                sub_goal_move_to_chest_top.append([0.03])
                sub_goal_move_to_chest_top = [[0.10]] + sub_goal_move_to_chest_top
                sub_goals.append(np.concatenate(sub_goal_move_to_chest_top))

                sub_goal_drop = block_poses.copy()
                for i in range(self.num_block):
                    if i < _:
                        sub_goal_drop[i] = chest_center_xyz.copy()
                sub_goal_drop[_] = chest_center_xyz.copy()
                sub_goal_drop.append(chest_top_xyz.copy())
                sub_goal_drop.append([0.06])
                sub_goal_drop = [[0.10]] + sub_goal_drop
                sub_goals.append(np.concatenate(sub_goal_drop))
        else:
            for _ in range(self.num_block):
                sub_goal = block_poses.copy()
                for i in range(self.num_block):
                    if i <= _:
                        sub_goal[i] = chest_center_xyz.copy()
                sub_goal = [[0.10]] + sub_goal
                sub_goals.append(np.concatenate(sub_goal))

        return sub_goals

    def _generate_curriculum(self, block_poses, chest_center_xyz, chest_top_xyz, new_target=False):
        # the first element is the largest openness of the door (equal to the door joint pose upper limit)
        desired_goal = [[0.10]]

        if new_target:
            curriculum_level = self.np_random.choice(self.num_curriculum, p=self.curriculum_prob)
            self.curriculum_goal_step = curriculum_level * 25 + self.base_curriculum_episode_steps
            ind_block_to_move = np.sort(self.np_random.choice(np.arange(self.num_block),
                                                              size=curriculum_level,
                                                              replace=False),
                                        kind='stable').tolist()
            self.last_curriculum_level = curriculum_level
            self.last_ind_block_to_move = ind_block_to_move

            if self.curriculum_update:
                self.num_generated_goals_per_curriculum[curriculum_level] += 1
                self._update_curriculum_prob()
        else:
            ind_block_to_move = self.last_ind_block_to_move
            curriculum_level = self.last_curriculum_level

        for i in range(self.num_block):
            if i in ind_block_to_move:
                desired_goal.append(chest_center_xyz)
            else:
                desired_goal.append(block_poses[i])

        if self.grip_informed_goal:
            if curriculum_level == 0:
                desired_goal.append(self.robot.parts['iiwa_gripper_tip'].get_position())
                desired_goal.append(self.robot.get_finger_closeness())
            else:
                desired_goal.append(chest_top_xyz.copy())
                desired_goal.append([0.06])

        return desired_goal


class KukaChestPushEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True, distance_threshold=0.05, grip_informed_goal=False,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', num_block=5, joint_control=False,
                 task_decomposition=False, use_curriculum=False, num_goals_to_generate=1e5):

        assert not goal_image, "Chest tasks do not support goal images well at the moment."

        self.grip_informed_goal = grip_informed_goal
        if self.grip_informed_goal:
            self.num_steps = num_block*2+1
        else:
            self.num_steps = num_block + 1

        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target, grip_informed_goal=self.grip_informed_goal,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=True,
                                         num_block=num_block, obj_range=0.1,
                                         joint_control=joint_control, grasping=False, chest=True, chest_door='front_sliding',
                                         task_decomposition=task_decomposition, use_curriculum=use_curriculum,
                                         num_curriculum=num_block+1, num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses, new_target=True):
        # all blocks should go into the sphere of 0.05 radius centred at the chest bottom centre
        chest_center_xyz, _ = self.chest_robot.get_base_pos()
        chest_center_xyz = np.array(chest_center_xyz)
        chest_center_xyz[0] += 0.05
        chest_center_xyz[2] = 0.175

        # setup desired goal or subgoals
        if self.curriculum:
            desired_goal = self._generate_curriculum(block_poses, chest_center_xyz, new_target)
        else:
            # the first element is the largest openness of the door (equal to the door joint pose upper limit)
            desired_goal = [[0.12]]
            # the final goal is when all the blocks are in the chest
            for _ in range(self.num_block):
                desired_goal.append(chest_center_xyz)
            if self.grip_informed_goal:
                desired_goal.append(chest_center_xyz.copy())
                desired_goal[-1][0] += 0.03

            if self.task_decomposition:
                self.sub_goals = self._generate_subgoals(block_poses, chest_center_xyz, new_target)

        if self.visualize_target:
            self._update_block_target(desired_goal, index_offset=1)
            if self.grip_informed_goal:
                self._update_gripper_target(desired_goal[-1])

        self.desired_goal = np.concatenate(desired_goal)

    def _generate_subgoals(self, block_poses, chest_center_xyz, new_target=False):
        # empty subgoal list
        sub_goals = []

        # the first subgoal is to open the door
        # the first element is the largest openness of the door (equal to the door joint pose upper limit)
        sub_goal_open_door = [[0.12]]
        sub_goal_open_door = sub_goal_open_door + block_poses
        sub_goal_open_door.append(self.robot.parts['iiwa_gripper_tip'].get_position())
        sub_goals.append(np.concatenate(sub_goal_open_door))

        if self.grip_informed_goal:
            for _ in range(self.num_block):
                sub_goal_reach = block_poses.copy()
                # block goal state
                for i in range(self.num_block):
                    if i < _:
                        sub_goal_reach[i] = chest_center_xyz.copy()
                # gripper goal state
                sub_goal_reach.append(block_poses[_].copy())
                sub_goal_reach[-1][0] += 0.03
                # chest door goal state
                sub_goal_reach = [[0.12]] + sub_goal_reach
                sub_goals.append(np.concatenate(sub_goal_reach))

                sub_goal_push = block_poses.copy()
                # block goal state
                for i in range(self.num_block):
                    if i <= _:
                        sub_goal_push[i] = chest_center_xyz.copy()
                # gripper goal state
                sub_goal_push.append(chest_center_xyz.copy())
                sub_goal_push[-1][0] += 0.03
                # chest door goal state
                sub_goal_push = [[0.12]] + sub_goal_push
                sub_goals.append(np.concatenate(sub_goal_push))
        else:
            for _ in range(self.num_block):
                sub_goal = block_poses.copy()
                # block goal state
                for i in range(self.num_block):
                    if i <= _:
                        sub_goal[i] = chest_center_xyz.copy()
                sub_goal = [[0.12]] + sub_goal
                sub_goals.append(np.concatenate(sub_goal))

        return sub_goals

    def _generate_curriculum(self, block_poses, chest_center_xyz, new_target=False):
        # the first element is the largest openness of the door (equal to the door joint pose upper limit)
        desired_goal = [[0.12]]

        if new_target:
            curriculum_level = self.np_random.choice(self.num_curriculum, p=self.curriculum_prob)
            self.curriculum_goal_step = curriculum_level * 25 + self.base_curriculum_episode_steps
            ind_block_to_move = np.sort(self.np_random.choice(np.arange(self.num_block),
                                                              size=curriculum_level,
                                                              replace=False),
                                        kind='stable').tolist()
            self.last_curriculum_level = curriculum_level
            self.last_ind_block_to_move = ind_block_to_move

            if self.curriculum_update:
                self.num_generated_goals_per_curriculum[curriculum_level] += 1
                self._update_curriculum_prob()
        else:
            ind_block_to_move = self.last_ind_block_to_move
            curriculum_level = self.last_curriculum_level

        for i in range(self.num_block):
            if i in ind_block_to_move:
                desired_goal.append(chest_center_xyz)
            else:
                desired_goal.append(block_poses[i])

        if self.grip_informed_goal:
            if curriculum_level == 0:
                desired_goal.append(self.robot.parts['iiwa_gripper_tip'].get_position())
            else:
                desired_goal.append(chest_center_xyz.copy())
                desired_goal[-1][0] += 0.03

        return desired_goal
