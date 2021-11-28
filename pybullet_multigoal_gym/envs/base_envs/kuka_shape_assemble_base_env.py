import os
import numpy as np
from pybullet_multigoal_gym.envs.base_envs.base_env import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka


class KukaBulletShapeAssembleEnv(BaseBulletMGEnv):
    """
    Base class for the shape assemble manipulation tasks with a Kuka iiwa 14 robot
    """

    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False, pcd=False,
                 visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', obj_range=0.15, target_range=0.15,
                 end_effector_start_on_table=False,
                 distance_threshold=0.05, grasping=False,
                 primitive=None):
        self.binary_reward = binary_reward
        self.image_observation = image_observation
        self.goal_image = goal_image
        self.render_pcd = pcd
        if depth_image:
            self.render_mode = 'rgbd_array'
        else:
            self.render_mode = 'rgb_array'
        self.visualize_target = visualize_target
        self.observation_cam_id = observation_cam_id
        self.goal_cam_id = goal_cam_id

        self.distance_threshold = distance_threshold
        self.grasping = grasping
        self.obj_range = obj_range
        self.target_range = target_range

        self.object_assets_path = os.path.join(os.path.dirname(__file__),
                                               "..", "..", "assets", "objects", "assembling_shape")
        self.objects_urdf_loaded = False
        self.object_bodies = {
            'workspace': None,
            'cube': None,
            'target': None
        }
        self.object_initial_pos = {
            'workspace': [-0.55, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0],
            'cube': [-0.60, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0],
            'target': [-0.55, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0]
        }

        self.desired_goal = None
        self.desired_goal_image = None

        robot = Kuka(gripper_type=gripper_type, grasping=grasping,
                     primitive=primitive, workspace_range={"upper_xy": (-0.4, 0.2),
                                                           "lower_xy": (-0.7, -0.2)},
                     end_effector_start_on_table=end_effector_start_on_table, table_surface_z=0.04,
                     obj_range=self.obj_range, target_range=self.target_range)

        BaseBulletMGEnv.__init__(self, robot=robot, render=render,
                                 image_observation=image_observation, goal_image=goal_image,
                                 camera_setup=camera_setup,
                                 seed=0, timestep=0.002, frame_skip=20)

    def _task_reset(self, test=False):
        if not self.objects_urdf_loaded:
            # don't reload object urdf
            self.objects_urdf_loaded = True
            self.object_bodies['workspace'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "workspace.urdf"),
                basePosition=self.object_initial_pos['workspace'][:3],
                baseOrientation=self.object_initial_pos['workspace'][3:])
            self.object_bodies['target'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "target.urdf"),
                basePosition=self.object_initial_pos['target'][:3],
                baseOrientation=self.object_initial_pos['target'][3:])
            self.object_bodies['cube'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "cube.urdf"),
                basePosition=self.object_initial_pos['cube'][:3],
                baseOrientation=self.object_initial_pos['cube'][3:])
            if not self.visualize_target:
                self.set_object_pose(self.object_bodies['target'],
                                     [0.0, 0.0, -3.0],
                                     self.object_initial_pos['target'][3:])

        # randomize object positions
        end_effector_tip_initial_position = self.robot.end_effector_tip_initial_position.copy()
        object_xy_1 = end_effector_tip_initial_position[:2]
        while np.linalg.norm(object_xy_1 - end_effector_tip_initial_position[:2]) < 0.1:
            object_xy_1 = self.np_random.uniform(self.robot.object_bound_lower[:-1],
                                                 self.robot.object_bound_upper[:-1])

        object_xyz = np.concatenate((object_xy_1.copy(), [0.035]))
        self.set_object_pose(self.object_bodies['cube'],
                             object_xyz,
                             self.object_initial_pos['cube'][3:])

        # generate goals & images
        self._generate_goal(current_obj_pos=object_xyz)
        if self.goal_image:
            self._generate_goal_image(current_obj_pos=object_xyz)

    def _generate_goal(self, current_obj_pos=None):
        if current_obj_pos is None:
            # generate a goal around the gripper if no object is involved
            center = self.robot.end_effector_tip_initial_position.copy()
        else:
            center = current_obj_pos

        # generate the 3DoF goal within a 3D bounding box such that,
        #       it is at least 0.02m away from the gripper or the object
        while True:
            self.desired_goal = self.np_random.uniform(self.robot.target_bound_lower,
                                                       self.robot.target_bound_upper)
            if np.linalg.norm(self.desired_goal - center) > 0.1:
                break

        self.desired_goal[2] = self.object_initial_pos['cube'][2]

        if self.visualize_target:
            self.set_object_pose(self.object_bodies['target'],
                                 self.desired_goal,
                                 self.object_initial_pos['target'][3:])

    def _generate_goal_image(self, current_obj_pos=None):
        away_pos = self.robot.compute_ik(np.array([0.0, -0.52, 0.045]))
        self.robot.set_kuka_joint_state(away_pos)

        # Push task
        original_obj_pos = current_obj_pos.copy()
        target_obj_pos = self.desired_goal.copy()
        self.set_object_pose(self.object_bodies['cube'],
                             target_obj_pos,
                             self.object_initial_pos['cube'][3:])
        self.desired_goal_image = self.render(mode=self.render_mode, camera_id=self.goal_cam_id)
        self.set_object_pose(self.object_bodies['cube'],
                             original_obj_pos,
                             self.object_initial_pos['cube'][3:])

        self.robot.set_kuka_joint_state(self.robot.kuka_rest_pose)

    def _step_callback(self):
        self.robot.set_kuka_joint_state(self.robot.kuka_rest_pose)

    def _get_obs(self):

        # robot state contains gripper xyz coordinates, orientation (and finger width)
        gripper_xyz, gripper_rpy, gripper_finger_closeness, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_vel, joint_poses = self.robot.calc_robot_state()
        assert self.desired_goal is not None

        block_xyz, _ = self._p.getBasePositionAndOrientation(self.object_bodies['cube'])
        block_rel_xyz = gripper_xyz - np.array(block_xyz)
        block_vel_xyz, block_vel_rpy = self._p.getBaseVelocity(self.object_bodies['cube'])
        block_rel_vel_xyz = gripper_vel_xyz - np.array(block_vel_xyz)
        block_rel_vel_rpy = gripper_vel_rpy - np.array(block_vel_rpy)
        achieved_goal = np.array(block_xyz).copy()
        # the HER paper use different state observations for the policy and critic network
        # critic further takes the velocities as input
        state = np.concatenate((gripper_xyz, block_xyz, gripper_finger_closeness, block_rel_xyz,
                                gripper_vel_xyz, gripper_finger_vel, block_rel_vel_xyz, block_rel_vel_rpy))
        policy_state = np.concatenate((gripper_xyz, gripper_finger_closeness, block_rel_xyz))

        obs_dict = {
            'observation': state.copy(),
            'policy_state': policy_state.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy(),
        }

        if self.image_observation:
            away_pos = self.robot.compute_ik(np.array([0.0, -0.52, 0.045]))
            self.robot.set_kuka_joint_state(away_pos)

            observation = self.render(mode=self.render_mode, camera_id=self.observation_cam_id)
            obs_dict['observation'] = observation.copy()
            obs_dict.update({'state': state.copy()})

            if self.goal_image:
                obs_dict.update({
                    'achieved_goal_img': observation.copy(),
                    'desired_goal_img': self.desired_goal_image.copy(),
                })

            if self.render_pcd:
                pcd = self.render(mode='pcd', camera_id=self.observation_cam_id)
                obs_dict.update({'pcd': pcd.copy()})

            self.robot.set_kuka_joint_state(self.robot.kuka_rest_pose)

        return obs_dict

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
            orientation = self.object_initial_pos['workspace'][3:]
        self._p.resetBasePositionAndOrientation(body_id, position, orientation)
