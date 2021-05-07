import os
import numpy as np
from pybullet_multigoal_gym.envs.base_envs.base_env import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka
from pybullet_multigoal_gym.robots.chest import Chest


class KukaBulletMultiBlockEnv(BaseBulletMGEnv):
    """
    Base class for multi-block long-horizon manipulation tasks with a Kuka iiwa 14 robot
    """

    def __init__(self, render=True, binary_reward=True, grip_informed_goal=False,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', end_effector_start_on_table=False,
                 num_block=3, joint_control=False, grasping=False, chest=False, chest_door='front_sliding',
                 obj_range=0.15, target_range=0.15, distance_threshold=0.05,
                 use_curriculum=False, num_curriculum=5, base_curriculum_episode_steps=50, num_goals_to_generate=1e5):
        self.binary_reward = binary_reward
        self.grip_informed_goal = grip_informed_goal
        self.image_observation = image_observation
        self.goal_image = goal_image
        if depth_image:
            self.render_mode = 'rgbd_array'
        else:
            self.render_mode = 'rgb_array'
        self.visualize_target = visualize_target
        self.observation_cam_id = observation_cam_id
        self.goal_cam_id = goal_cam_id

        self.num_block = num_block
        self.joint_control = joint_control
        self.grasping = grasping
        self.chest = chest
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold

        self.object_assets_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "objects")
        self.objects_urdf_loaded = False
        self.object_bodies = {
            'table': None,
            'chest': None,
            'block_blue': None,
            'block_green': None,
            'block_purple': None,
            'block_red': None,
            'block_yellow': None,
            'target_blue': None,
            'target_green': None,
            'target_purple': None,
            'target_red': None,
            'target_yellow': None
        }
        self.object_initial_pos = {
            'table': [-0.52, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0],
            'chest': [-0.695, 0.0, 0.21, 0.0, 0.0, 0.0, 1.0],
            'target_chest': [-0.695, 0.18, 0.175, 0.0, 0.0, 0.0, 1.0],

            'block_blue': [-0.52, 0.0, 0.175, 0.0, 0.0, 0.0, 1.0],
            'block_green': [-0.52, 0.08, 0.175, 0.0, 0.0, 0.0, 1.0],
            'block_purple': [-0.52, 0.16, 0.175, 0.0, 0.0, 0.0, 1.0],
            'block_red': [-0.52, -0.08, 0.175, 0.0, 0.0, 0.0, 1.0],
            'block_yellow': [-0.52, -0.16, 0.175, 0.0, 0.0, 0.0, 1.0],

            'target_blue': [-0.52, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
            'target_green': [-0.52, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
            'target_purple': [-0.52, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
            'target_red': [-0.52, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
            'target_yellow': [-0.52, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
        }
        self.block_size = 0.03
        self.block_keys = ['block_blue', 'block_green', 'block_purple', 'block_red', 'block_yellow']
        self.target_keys = ['target_blue', 'target_green', 'target_purple', 'target_red', 'target_yellow']

        self.sub_goals = None
        self.desired_goal = None
        self.desired_goal_image = None

        robot = Kuka(grasping=self.grasping,
                     joint_control=joint_control,
                     gripper_type=gripper_type,
                     end_effector_start_on_table=end_effector_start_on_table,
                     obj_range=self.obj_range, target_range=self.target_range)
        if self.chest:
            self.chest_robot = Chest(base_position=self.object_initial_pos['chest'][:3],
                                     door=chest_door, rest_door_state=0)
            # self.distance_threshold = 0.1
            self.chest_pos_y_range = 0.15
            robot.object_bound_lower[0] += 0.05
            robot.object_bound_upper[0] += 0.05
            robot.object_bound_lower[1] -= 0.05
            robot.object_bound_upper[1] += 0.05

        self.curriculum = use_curriculum
        self.curriculum_update = False
        self.num_curriculum = num_curriculum
        # start with the easiest goal being the only possible goal
        self.curriculum_prob = np.concatenate([[1.0], np.zeros(self.num_curriculum-1)])
        self.base_curriculum_episode_steps = base_curriculum_episode_steps
        # the number of episode steps increases as goals become harder to achieve
        self.curriculum_goal_step = 0 * 25 + self.base_curriculum_episode_steps
        # the number of goals to generate for each curriculum
        #       commonly equal to the total number of episodes / the number of curriculum
        self.num_goals_per_curriculum = num_goals_to_generate / self.num_curriculum
        # record the number of generated goals per curriculum
        self.num_generated_goals_per_curriculum = np.zeros(self.num_curriculum)

        BaseBulletMGEnv.__init__(self, robot=robot, render=render,
                                 image_observation=image_observation, goal_image=goal_image,
                                 camera_setup=camera_setup,
                                 seed=0, timestep=0.002, frame_skip=20)

    def _task_reset(self):
        if not self.objects_urdf_loaded:
            # don't reload object urdf
            self.objects_urdf_loaded = True
            self.object_bodies['table'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "table.urdf"),
                basePosition=self.object_initial_pos['table'][:3],
                baseOrientation=self.object_initial_pos['table'][3:])
            if self.chest:
                self.chest_robot.reset(bullet_client=self._p)

            for n in range(self.num_block):
                block_name = self.block_keys[n]
                self.object_bodies[block_name] = self._p.loadURDF(
                    os.path.join(self.object_assets_path, block_name + ".urdf"),
                    basePosition=self.object_initial_pos[block_name][:3],
                    baseOrientation=self.object_initial_pos[block_name][3:])
                target_name = self.target_keys[n]

                self.object_bodies[target_name] = self._p.loadURDF(
                    os.path.join(self.object_assets_path, target_name + ".urdf"),
                    basePosition=self.object_initial_pos[target_name][:3],
                    baseOrientation=self.object_initial_pos[target_name][3:])
                if not self.visualize_target:
                    self.set_object_pose(self.object_bodies[target_name],
                                         [0.0, 0.0, -3.0],
                                         self.object_initial_pos[target_name][3:])

        # randomize object positions
        block_poses = []
        for _ in range(self.num_block):
            done = False
            while not done:
                new_object_xy = self.np_random.uniform(self.robot.object_bound_lower[:-1],
                                                       self.robot.object_bound_upper[:-1])
                object_not_overlap = []
                for pos in block_poses + [self.robot.end_effector_tip_initial_position]:
                    object_not_overlap.append((np.linalg.norm(new_object_xy - pos[:-1]) > 0.06))
                if all(object_not_overlap):
                    block_poses.append(np.concatenate((new_object_xy.copy(), [0.175])))
                    done = True

        for i in range(self.num_block):
            self.set_object_pose(self.object_bodies[self.block_keys[i]],
                                 block_poses[i],
                                 self.object_initial_pos[self.block_keys[i]][3:])

        if self.chest:
            self.chest_robot.robot_specific_reset(self._p)
            new_y = self.np_random.uniform(-self.chest_pos_y_range, self.chest_pos_y_range)
            chest_xyz = self.object_initial_pos['chest'][:3].copy()
            chest_xyz[1] = new_y
            self.chest_robot.set_base_pos(self._p, position=chest_xyz)

        # generate goals & images
        self._generate_goal(block_poses)
        if self.goal_image:
            self._generate_goal_image(block_poses)

    def activate_curriculum_update(self):
        self.curriculum_update = True

    def deactivate_curriculum_update(self):
        self.curriculum_update = False

    def update_curriculum_prob(self):
        # array of boolean masks
        mask_finished = self.num_generated_goals_per_curriculum == self.num_goals_per_curriculum
        mask_half = self.num_generated_goals_per_curriculum >= (self.num_goals_per_curriculum / 2)
        # set finished curriculum prob to 0.0
        self.curriculum_prob[mask_finished] = 0.0
        # process the first curriculum separately
        if mask_half[0] and not mask_finished[0]:
            self.curriculum_prob[0] = 0.5
            self.curriculum_prob[1] = 0.5

        # process the second to the second-last curriculums
        for i in range(1, self.num_curriculum-1):
            if mask_finished[i-1] and not mask_finished[i]:

                if mask_half[i]:
                    # set the next curriculum prob to 0.5
                    #       if the current one has been trained for half the total number
                    #       and the last one has been trained completely
                    self.curriculum_prob[i] = 0.5
                    self.curriculum_prob[i+1] = 0.5
                else:
                    # set the next curriculum prob to 1.0
                    #       if the current one has not yet been trained for half the total number
                    #       and the last one has been trained completely
                    self.curriculum_prob[i] = 1.0

        # process the last curriculum separately
        if mask_finished[-2]:
            self.curriculum_prob[-1] = 1.0

    def set_sub_goal(self, sub_goal_ind):
        self.desired_goal = self.sub_goals[sub_goal_ind].copy()
        if self.visualize_target:
            index_offset = 0
            if self.chest:
                index_offset = 1
            block_target_pos = []
            for _ in range(self.num_block):
                block_target_pos.append(
                    self.desired_goal[index_offset+_*3:index_offset+_*3+3]
                )
            self._update_block_target(block_target_pos)
        return self.desired_goal

    def _generate_goal(self, block_poses):
        raise NotImplementedError

    def _update_block_target(self, desired_goal, index_offset=0):
        for _ in range(self.num_block):
            self.set_object_pose(self.object_bodies[self.target_keys[_]],
                                 desired_goal[_+index_offset],
                                 self.object_initial_pos[self.target_keys[_]][3:])

    def _generate_goal_image(self, block_poses):
        target_obj_pos = self.desired_goal.copy()
        if self.chest:
            # this is tricky...
            self.desired_goal_image = self.render(mode=self.render_mode, camera_id=self.goal_cam_id)
        elif self.grasping:
            # block stacking
            self.robot.set_finger_joint_state(self.robot.gripper_grasp_block_state)
            target_gripper_pos = self.desired_goal[-3:].copy()
            target_gripper_pos[-1] = 0.175 + self.block_size * (self.num_block - 1)
            target_kuka_joint_pos = self.robot.compute_ik(self._p, target_gripper_pos)
            self.robot.set_kuka_joint_state(target_kuka_joint_pos)

            for i in range(self.num_block):
                self.set_object_pose(self.object_bodies[self.block_keys[i]],
                                     target_obj_pos[i * 3:i * 3 + 3],
                                     self.object_initial_pos[self.block_keys[i]][3:])

            self.desired_goal_image = self.render(mode=self.render_mode, camera_id=self.goal_cam_id)

            for i in range(self.num_block):
                self.set_object_pose(self.object_bodies[self.block_keys[i]],
                                     block_poses[i],
                                     self.object_initial_pos[self.block_keys[i]][3:])

            self.robot.set_kuka_joint_state(self.robot.kuka_rest_pose, np.zeros(7))
            self.robot.set_finger_joint_state(self.robot.gripper_abs_joint_limit)
        else:
            for i in range(self.num_block):
                self.set_object_pose(self.object_bodies[self.block_keys[i]],
                                     target_obj_pos[i * 3:i * 3 + 3],
                                     self.object_initial_pos[self.block_keys[i]][3:])

            self.desired_goal_image = self.render(mode=self.render_mode, camera_id=self.goal_cam_id)

            for i in range(self.num_block):
                self.set_object_pose(self.object_bodies[self.block_keys[i]],
                                     block_poses[i],
                                     self.object_initial_pos[self.block_keys[i]][3:])

    def _step_callback(self):
        pass

    def _get_obs(self):
        # robot state contains gripper xyz coordinates, orientation (and finger width)
        gripper_xyz, gripper_rpy, gripper_finger_closeness, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_vel, joint_poses = self.robot.calc_robot_state()
        assert self.desired_goal is not None

        block_states = []
        policy_block_states = []
        achieved_goal = []
        for n in range(self.num_block):
            block_name = self.block_keys[n]
            block_xyz, block_rpy = self._p.getBasePositionAndOrientation(self.object_bodies[block_name])
            block_rel_xyz = gripper_xyz - np.array(block_xyz)
            block_vel_xyz, block_vel_rpy = self._p.getBaseVelocity(self.object_bodies[block_name])
            block_rel_vel_xyz = gripper_vel_xyz - np.array(block_vel_xyz)
            block_rel_vel_rpy = gripper_vel_rpy - np.array(block_vel_rpy)
            # a block state for critic network contains:
            #   World frame position & euler orientation
            #   Relative position (w.r.t. gripper tip)
            #   Relative linear & euler-angular velocities (w.r.t. gripper tip)
            block_states = block_states + [block_xyz, block_rel_xyz, block_rpy, block_rel_vel_xyz, block_rel_vel_rpy]
            # a block state for policy network contains the relative position w.r.t. gripper tip
            policy_block_states = policy_block_states + [block_rel_xyz]
            # an achieved goal contains the current block positions in world frame
            achieved_goal.append(np.array(block_xyz).copy())

        state = [gripper_xyz, gripper_finger_closeness, gripper_vel_xyz, gripper_finger_vel] + block_states
        policy_state = [gripper_xyz, gripper_finger_closeness] + policy_block_states

        if self.joint_control:
            state = [joint_poses] + state
            policy_state = [joint_poses] + policy_state

        if self.chest:
            # door joint state represents the openness and velocity of the door
            door_joint_pos, door_joint_vel, keypoint_state = self.chest_robot.calc_robot_state()
            state = state + [[door_joint_pos], [door_joint_vel]] + keypoint_state
            policy_state = policy_state + [door_joint_pos] + keypoint_state
            achieved_goal.insert(0, [door_joint_pos])

        if self.grip_informed_goal:
            # gripper informed goals in addition indicates that goal states of the gripper (coordinates & finger width)
            achieved_goal.append(gripper_xyz)
            achieved_goal.append(gripper_finger_closeness)

        state = np.concatenate(state)
        policy_state = np.concatenate(policy_state)
        achieved_goal = np.concatenate(achieved_goal)

        assert achieved_goal.shape == self.desired_goal.shape

        if not self.image_observation:
            return {
                'observation': state.copy(),
                'policy_state': policy_state.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.desired_goal.copy(),
            }
        elif not self.goal_image:
            return {
                'observation': self.render(mode=self.render_mode, camera_id=self.observation_cam_id),
                'state': state.copy(),
                'policy_state': policy_state.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.desired_goal.copy(),
            }
        else:
            observation = self.render(mode=self.render_mode, camera_id=self.observation_cam_id)
            return {
                'observation': observation.copy(),
                'state': state.copy(),
                'policy_state': policy_state.copy(),
                'achieved_goal': achieved_goal.copy(),
                'achieved_goal_img': observation.copy(),
                'desired_goal': self.desired_goal.copy(),
                'desired_goal_img': self.desired_goal_image.copy(),
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
