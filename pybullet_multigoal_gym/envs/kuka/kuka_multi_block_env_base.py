import os
import numpy as np
from pybullet_multigoal_gym.envs.env_bases import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka


class KukaBulletMultiBlockEnv(BaseBulletMGEnv):
    """
    Base class for multi-block long-horizon manipulation tasks with a Kuka iiwa 14 robot
    """

    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw',
                 num_block=3, grasping=False, chest=False,
                 obj_range=0.15, target_range=0.15,
                 distance_threshold=0.05, randomized_obj_pos=True):
        self.binary_reward = binary_reward
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
        self.grasping = grasping
        self.chest = chest
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.randomized_obj_pos = randomized_obj_pos

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
            'table': [-0.45, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0],
            'chest': [-0.65, 0.0, 0.21, 0.0, 0.0, 0.0, 1.0],
            'target_chest': [-0.65, 0.18, 0.21, 0.0, 0.0, 0.0, 1.0],

            'block_blue': [-0.45, 0.0, 0.175, 0.0, 0.0, 0.0, 1.0],
            'block_green': [-0.45, 0.08, 0.175, 0.0, 0.0, 0.0, 1.0],
            'block_purple': [-0.45, 0.16, 0.175, 0.0, 0.0, 0.0, 1.0],
            'block_red': [-0.45, -0.08, 0.175, 0.0, 0.0, 0.0, 1.0],
            'block_yellow': [-0.45, -0.16, 0.175, 0.0, 0.0, 0.0, 1.0],

            'target_blue': [-0.45, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
            'target_green': [-0.45, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
            'target_purple': [-0.45, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
            'target_red': [-0.45, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
            'target_yellow': [-0.45, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0],
        }
        self.block_size = 0.03
        self.block_keys = ['block_blue', 'block_green', 'block_purple', 'block_red', 'block_yellow']
        self.target_keys = ['target_blue', 'target_green', 'target_purple', 'target_red', 'target_yellow']

        self.desired_goal = None
        self.desired_goal_image = None

        robot = Kuka(grasping=self.grasping,
                     gripper_type=gripper_type,
                     end_effector_start_on_table=False,
                     obj_range=self.obj_range, target_range=self.target_range)
        if self.chest:
            self.distance_threshold = 0.08
            self.chest_pos_y_range = 0.15
            robot.object_bound_lower[0] += 0.05
            robot.object_bound_upper[0] += 0.05
            robot.object_bound_lower[1] -= 0.05
            robot.object_bound_upper[1] += 0.05

        BaseBulletMGEnv.__init__(self, robot=robot, render=render,
                                 image_observation=image_observation, goal_image=goal_image,
                                 camera_setup=camera_setup,
                                 seed=0, timestep=0.002, frame_skip=20)

    def task_reset(self):
        if not self.objects_urdf_loaded:
            self.objects_urdf_loaded = True
            self.object_bodies['table'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "table.urdf"),
                basePosition=self.object_initial_pos['table'][:3],
                baseOrientation=self.object_initial_pos['table'][3:])
            if self.chest:
                self.object_bodies['chest'] = self._p.loadURDF(
                    os.path.join(self.object_assets_path, "chest.urdf"),
                    basePosition=self.object_initial_pos['chest'][:3],
                    baseOrientation=self.object_initial_pos['chest'][3:])
                self.object_bodies['target_chest'] = self._p.loadURDF(
                    os.path.join(self.object_assets_path, 'target_chest' + ".urdf"),
                    basePosition=self.object_initial_pos['target_chest'][:3],
                    baseOrientation=self.object_initial_pos['target_chest'][3:])
                if not self.visualize_target:
                    self.set_object_pose(self.object_bodies['target_chest'],
                                         [0.0, 0.0, -3.0],
                                         self.object_initial_pos['target_chest'][3:])

            for n in range(self.num_block):
                block_name = self.block_keys[n]
                self.object_bodies[block_name] = self._p.loadURDF(
                    os.path.join(self.object_assets_path, block_name + ".urdf"),
                    basePosition=self.object_initial_pos[block_name][:3],
                    baseOrientation=self.object_initial_pos[block_name][3:])
                target_name = self.target_keys[n]
                if not self.chest:
                    self.object_bodies[target_name] = self._p.loadURDF(
                        os.path.join(self.object_assets_path, target_name + ".urdf"),
                        basePosition=self.object_initial_pos[target_name][:3],
                        baseOrientation=self.object_initial_pos[target_name][3:])
                    if not self.visualize_target:
                        self.set_object_pose(self.object_bodies[target_name],
                                             [0.0, 0.0, -3.0],
                                             self.object_initial_pos[target_name][3:])

        block_poses = []
        if self.randomized_obj_pos:
            new_object_xy = self.np_random.uniform(self.robot.object_bound_lower[:-1],
                                                   self.robot.object_bound_upper[:-1])
            block_poses.append(np.concatenate((new_object_xy, [0.175])))
            for _ in range(self.num_block - 1):
                done = False
                while not done:
                    new_object_xy = self.np_random.uniform(self.robot.object_bound_lower[:-1],
                                                           self.robot.object_bound_upper[:-1])
                    object_not_overlap = []
                    for pos in block_poses:
                        object_not_overlap.append((np.linalg.norm(new_object_xy - pos[:-1]) > 0.06))
                    if all(object_not_overlap):
                        block_poses.append(np.concatenate((new_object_xy.copy(), [0.175])))
                        done = True

            for i in range(self.num_block):
                self.set_object_pose(self.object_bodies[self.block_keys[i]],
                                     block_poses[i],
                                     self.object_initial_pos[self.block_keys[i]][3:])
        else:
            for i in range(self.num_block):
                object_name = self.block_keys[i]
                self.set_object_pose(self.object_bodies[object_name],
                                     self.object_initial_pos[object_name][:3],
                                     self.object_initial_pos[object_name][3:])

        if self.chest:
            new_y = self.np_random.uniform(-self.chest_pos_y_range, self.chest_pos_y_range)
            chest_xyz = self.object_initial_pos['chest'][:3].copy()
            chest_xyz[1] = new_y
            self.set_object_pose(self.object_bodies['chest'],
                                 chest_xyz,
                                 self.object_initial_pos['chest'][3:])

        self._generate_goal()
        if self.goal_image:
            self._generate_goal_image(block_poses)

    def _generate_goal(self):
        desired_goal = []
        if self.chest:
            # chest pick and place
            chest_center_xyz, _ = self._p.getBasePositionAndOrientation(self.object_bodies['chest'])
            chest_center_xyz = np.array(chest_center_xyz)
            chest_center_xyz[0] += 0.05
            if self.visualize_target:
                self.set_object_pose(self.object_bodies['target_chest'],
                                     chest_center_xyz,
                                     self.object_initial_pos['target_chest'][3:])
            for _ in range(self.num_block):
                desired_goal.append(chest_center_xyz)
        elif self.grasping:
            # block stacking
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
        else:
            # block rearranging
            new_target_xy = self.np_random.uniform(self.robot.target_bound_lower[:-1],
                                                   self.robot.target_bound_upper[:-1])
            desired_goal.append(np.concatenate((new_target_xy, [0.175])))
            for _ in range(self.num_block - 1):
                done = False
                while not done:
                    new_target_xy = self.np_random.uniform(self.robot.target_bound_lower[:-1],
                                                           self.robot.target_bound_upper[:-1])
                    target_not_overlap = []
                    for pos in desired_goal:
                        target_not_overlap.append((np.linalg.norm(new_target_xy - pos[:-1]) > 0.06))
                    if all(target_not_overlap):
                        desired_goal.append(np.concatenate((new_target_xy.copy(), [0.175])))
                        done = True
            if self.visualize_target:
                for _ in range(self.num_block):
                    self.set_object_pose(self.object_bodies[self.target_keys[_]],
                                         desired_goal[_],
                                         self.object_initial_pos[self.target_keys[_]][3:])

        self.desired_goal = np.concatenate(desired_goal)

    def _generate_goal_image(self, block_poses):
        target_obj_pos = self.desired_goal.copy()
        if self.grasping:
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
        gripper_xyz, gripper_rpy, gripper_finger_closeness, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_vel = self.robot.calc_robot_state()
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

        if self.chest:
            pass

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
