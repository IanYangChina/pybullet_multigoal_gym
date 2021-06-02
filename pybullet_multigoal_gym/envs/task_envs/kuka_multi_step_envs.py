import numpy as np
from pybullet_multigoal_gym.utils.demonstrator import StepDemonstrator
from pybullet_multigoal_gym.envs.base_envs.kuka_multi_step_base_env import KukaBulletMultiBlockEnv


class KukaBlockStackEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True, distance_threshold=0.05, random_order=False,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', num_block=5, joint_control=False,
                 task_decomposition=False, abstract_demonstration=False,
                 use_curriculum=False, num_goals_to_generate=1e5):
        self.task_decomposition = task_decomposition
        self.num_steps = num_block
        if self.task_decomposition:
            self.grip_informed_goal = True
            self.num_steps = num_block * 2
        self.abstract_demonstration = abstract_demonstration
        if self.abstract_demonstration:
            demonstrations = []
            for i in range(self.num_steps):
                demonstrations.append([_ for _ in range(i+1)])
            self.step_demonstrator = StepDemonstrator(demonstrations)
        self.random_order = random_order
        self.last_order = None
        self.last_target_poses = None
        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                         grip_informed_goal=self.grip_informed_goal,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=True,
                                         num_block=num_block,joint_control=joint_control,
                                         grasping=True, chest=False,
                                         obj_range=0.15, target_range=0.15,
                                         use_curriculum=use_curriculum,
                                         num_curriculum=num_block,
                                         num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses, new_target=True):
        desired_goal = [None for _ in range(self.num_block)]

        if new_target:
            # generate a random order of blocks to be stacked
            new_order = np.arange(self.num_block, dtype=np.int)
            if self.random_order:
                self.np_random.shuffle(new_order)
            new_order = new_order.tolist()

            # generate a random base block position
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

        if not self.curriculum:
            # generate goal and set target poses according to the order
            for _ in range(self.num_block):
                desired_goal[new_order[_]] = target_xyzs[_]
            if self.grip_informed_goal:
                desired_goal.append(target_xyzs[-1].copy())
                desired_goal.append([0.03])

            if self.task_decomposition:
                self.sub_goals = []
                for _ in range(int(self.num_steps/2)):
                    sub_goal_pick = [None for _ in range(self.num_block)]
                    for i in range(self.num_block):
                        if i <= (_-1):
                            sub_goal_pick[new_order[i]] = target_xyzs[i]
                        else:
                            sub_goal_pick[new_order[i]] = block_poses[new_order[i]]
                    sub_goal_pick.append(block_poses[new_order[_]].copy())
                    sub_goal_pick.append([0.03])
                    self.sub_goals.append(np.concatenate(sub_goal_pick))

                    sub_goal_place = [None for _ in range(self.num_block)]
                    for i in range(self.num_block):
                        if i <= _:
                            sub_goal_place[new_order[i]] = target_xyzs[i]
                        else:
                            sub_goal_place[new_order[i]] = block_poses[new_order[i]]
                    sub_goal_place.append(target_xyzs[_].copy())
                    sub_goal_place.append([0.03])
                    self.sub_goals.append(np.concatenate(sub_goal_place))
        else:
            curriculum_level = self.np_random.choice(self.num_curriculum, p=self.curriculum_prob)
            self.curriculum_goal_step = curriculum_level * 25 + self.base_curriculum_episode_steps

            for _ in range(self.num_block):
                if _ <= curriculum_level:
                    desired_goal[new_order[_]] = target_xyzs[_]
                else:
                    desired_goal[new_order[_]] = block_poses[new_order[_]]

            if self.curriculum_update:
                self.num_generated_goals_per_curriculum[curriculum_level] += 1
                self.update_curriculum_prob()

        if self.visualize_target:
            self._update_block_target(desired_goal)

        self.desired_goal = np.concatenate(desired_goal)


class KukaBlockRearrangeEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True, distance_threshold=0.05,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', num_block=5, joint_control=False,
                 use_curriculum=False, num_goals_to_generate=1e5):
        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=True,
                                         num_block=num_block, joint_control=joint_control, grasping=False, chest=False,
                                         obj_range=0.15, target_range=0.15,
                                         use_curriculum=use_curriculum,
                                         num_curriculum=num_block,
                                         num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses, new_target=True):
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
            curriculum_level = self.np_random.choice(self.num_curriculum, p=self.curriculum_prob)
            self.curriculum_goal_step = curriculum_level * 25 + self.base_curriculum_episode_steps
            ind_block_to_move = np.sort(self.np_random.choice(np.arange(self.num_block), size=curriculum_level+1),
                                        kind='stable').tolist()
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

        if self.visualize_target:
            self._update_block_target(desired_goal)

        self.desired_goal = np.concatenate(desired_goal)


class KukaChestPickAndPlaceEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True, distance_threshold=0.05,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', num_block=5, joint_control=False,
                 task_decomposition=False, abstract_demonstration=False,
                 use_curriculum=False, num_goals_to_generate=1e5):
        self.task_decomposition = task_decomposition
        self.num_steps = num_block+1
        if self.task_decomposition:
            # self.grip_informed_goal = True,
            self.num_steps = num_block * 2 + 1
        self.abstract_demonstration = abstract_demonstration
        if self.abstract_demonstration:
            demonstrations = []
            for i in range(self.num_steps):
                demonstrations.append([_ for _ in range(i+1)])
            self.step_demonstrator = StepDemonstrator(demonstrations)
        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                         # grip_informed_goal=self.grip_informed_goal,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=True,
                                         num_block=num_block, joint_control=joint_control, grasping=True, chest=True, chest_door='up_sliding',
                                         obj_range=0.1,
                                         use_curriculum=use_curriculum,
                                         num_curriculum=num_block+1,
                                         num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses, new_target=True):
        # the first element is the largest openness of the door (equal to the door joint pose upper limit)
        desired_goal = [[0.10]]

        # all blocks should go into the sphere of 0.05 radius centred at the chest centre
        chest_center_xyz, _ = self.chest_robot.get_base_pos(self._p)
        chest_center_xyz = np.array(chest_center_xyz)
        chest_center_xyz[0] += 0.05
        chest_center_xyz[2] = 0.175

        chest_top_xyz = chest_center_xyz.copy()
        chest_top_xyz[-1] = 0.3

        if not self.curriculum:
            current_gripper_tip_xyz = self.robot.parts['iiwa_gripper_tip'].get_position()

            for _ in range(self.num_block):
                desired_goal.append(chest_center_xyz)
                # if self.grip_informed_goal:
                #     desired_goal.append(current_gripper_tip_xyz.copy())
                #     desired_goal.append([0.03])

            if self.task_decomposition:
                self.sub_goals = []

                sub_goal_open_door = [[0.10]]
                sub_goal_open_door = sub_goal_open_door + block_poses
                # sub_goal_open_door.append(current_gripper_tip_xyz.copy())
                # sub_goal_open_door.append([0.03])
                self.sub_goals.append(np.concatenate(sub_goal_open_door))

                for _ in range(self.num_block):
                    # # block positions
                    # sub_goal_pick = block_poses.copy()
                    # # previous blocks should already be within the chest
                    # for i in range(self.num_block):
                    #     if i < _:
                    #         sub_goal_pick[i] = chest_center_xyz.copy()
                    # # gripper position
                    # sub_goal_pick.append(block_poses[_].copy())
                    # # finger width
                    # sub_goal_pick.append([0.03])
                    # # chest door joint state
                    # sub_goal_pick = [[0.10]] + sub_goal_pick
                    # self.sub_goals.append(np.concatenate(sub_goal_pick))

                    sub_goal_move_to_chest_top = block_poses.copy()
                    for i in range(self.num_block):
                        if i < _:
                            sub_goal_move_to_chest_top[i] = chest_center_xyz.copy()
                    sub_goal_move_to_chest_top[_] = chest_top_xyz.copy()
                    # sub_goal_move_to_chest_top.append(chest_top_xyz.copy())
                    # sub_goal_move_to_chest_top.append([0.03])
                    sub_goal_move_to_chest_top = [[0.10]] + sub_goal_move_to_chest_top
                    self.sub_goals.append(np.concatenate(sub_goal_move_to_chest_top))

                    sub_goal_drop = block_poses.copy()
                    for i in range(self.num_block):
                        if i < _:
                            sub_goal_drop[i] = chest_center_xyz.copy()
                    sub_goal_drop[_] = chest_center_xyz.copy()
                    # sub_goal_drop.append(current_gripper_tip_xyz.copy())
                    # sub_goal_drop.append([0.05])
                    sub_goal_drop = [[0.10]] + sub_goal_drop
                    self.sub_goals.append(np.concatenate(sub_goal_drop))
        else:
            curriculum_level = self.np_random.choice(self.num_curriculum, p=self.curriculum_prob)
            self.curriculum_goal_step = curriculum_level * 25 + self.base_curriculum_episode_steps
            ind_block_to_move = np.sort(self.np_random.choice(np.arange(self.num_block), size=curriculum_level),
                                        kind='stable').tolist()

            for i in range(self.num_block):
                if i in ind_block_to_move:
                    desired_goal.append(chest_center_xyz)
                else:
                    desired_goal.append(block_poses[i])

            if self.curriculum_update:
                self.num_generated_goals_per_curriculum[curriculum_level] += 1
                self.update_curriculum_prob()

        if self.visualize_target:
            self._update_block_target(desired_goal, index_offset=1)
            # if self.grip_informed_goal:
            #     self._update_gripper_target(desired_goal[-2])

        self.desired_goal = np.concatenate(desired_goal)


class KukaChestPushEnv(KukaBulletMultiBlockEnv):
    def __init__(self, render=True, binary_reward=True, distance_threshold=0.05,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', num_block=5, joint_control=False,
                 task_decomposition=False, abstract_demonstration=False,
                 use_curriculum=False, num_goals_to_generate=1e5):
        self.task_decomposition = task_decomposition
        self.num_steps = num_block+1
        self.abstract_demonstration = abstract_demonstration
        if self.abstract_demonstration:
            demonstrations = []
            for i in range(self.num_steps):
                demonstrations.append([_ for _ in range(i+1)])
            self.step_demonstrator = StepDemonstrator(demonstrations)
        KukaBulletMultiBlockEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                         image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                         visualize_target=visualize_target,
                                         camera_setup=camera_setup, observation_cam_id=observation_cam_id, goal_cam_id=goal_cam_id,
                                         gripper_type=gripper_type, end_effector_start_on_table=True,
                                         num_block=num_block, joint_control=joint_control, grasping=False, chest=True, chest_door='front_sliding',
                                         obj_range=0.1,
                                         use_curriculum=use_curriculum,
                                         num_curriculum=num_block+1,
                                         num_goals_to_generate=num_goals_to_generate)

    def _generate_goal(self, block_poses, new_target=True):
        # the first element is the largest openness of the door (equal to the door joint pose upper limit)
        desired_goal = [[0.12]]

        # all blocks should go into the sphere of 0.05 radius centred at the chest centre
        chest_center_xyz, _ = self.chest_robot.get_base_pos(self._p)
        chest_center_xyz = np.array(chest_center_xyz)
        chest_center_xyz[0] += 0.05
        chest_center_xyz[2] = 0.175

        if not self.curriculum:
            for _ in range(self.num_block):
                desired_goal.append(chest_center_xyz)

            if self.task_decomposition:
                self.sub_goals = []
                for _ in range(self.num_steps):
                    sub_goal = [[0.12]]
                    for i in range(self.num_block):
                        if i < _:
                            sub_goal.append(chest_center_xyz)
                        else:
                            sub_goal.append(block_poses[i])
                    self.sub_goals.append(np.concatenate(sub_goal))
        else:
            curriculum_level = self.np_random.choice(self.num_curriculum, p=self.curriculum_prob)
            self.curriculum_goal_step = curriculum_level * 25 + self.base_curriculum_episode_steps
            ind_block_to_move = np.sort(self.np_random.choice(np.arange(self.num_block), size=curriculum_level),
                                        kind='stable').tolist()

            for i in range(self.num_block):
                if i in ind_block_to_move:
                    desired_goal.append(chest_center_xyz)
                else:
                    desired_goal.append(block_poses[i])

            if self.curriculum_update:
                self.num_generated_goals_per_curriculum[curriculum_level] += 1
                self.update_curriculum_prob()

        if self.visualize_target:
            self._update_block_target(desired_goal, index_offset=1)

        self.desired_goal = np.concatenate(desired_goal)
