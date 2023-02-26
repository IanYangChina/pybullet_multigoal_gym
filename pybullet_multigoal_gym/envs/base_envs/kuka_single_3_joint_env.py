from ntpath import join
import os
import numpy as np
from pybullet_multigoal_gym.envs.base_envs.base_env import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka
from pybullet_multigoal_gym.robots.kuka_with_box import KukaBox


class KukaBullet3Env(BaseBulletMGEnv):
    """
    Base class for the OpenAI multi-goal manipulation tasks with a Kuka iiwa 14 robot
    """

    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=None, goal_cam_id=0,
                 gripper_type='parallel_jaw', obj_range=0.15, target_range=0.15,
                 target_in_the_air=True, end_effector_start_on_table=False,
                 distance_threshold=0.05, joint_control=True, grasping=False, has_obj=False, tip_penalty=100):
        if observation_cam_id is None:
            observation_cam_id = [0]
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

        self.target_in_the_air = target_in_the_air
        self.distance_threshold = distance_threshold
        self.joint_control = joint_control
        self.grasping = grasping
        self.has_obj = has_obj
        self.obj_range = obj_range
        self.target_range = target_range

        self.object_assets_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "objects")
        self.objects_urdf_loaded = False
        self.object_bodies = {
            'block': None,
            'target': None
        }
        self.object_initial_pos = {
            'block': [-0.52, 0.0, 0.175, 0.0, 0.0, 0.0, 1.0],
            'target': [-0.52, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0]
        }

        self.tip_penalty = tip_penalty

        self.desired_goal = None
        self.desired_goal_image = None

        robot = KukaBox (grasping=grasping,
                     joint_control=joint_control,
                     gripper_type=gripper_type,
                     end_effector_start_on_table=end_effector_start_on_table,
                     obj_range=self.obj_range, target_range=self.target_range)

        BaseBulletMGEnv.__init__(self, robot=robot, render=render,
                                 image_observation=image_observation, goal_image=goal_image,
                                 camera_setup=camera_setup,
                                 seed=0, timestep=0.002, frame_skip=20)

    def _task_reset(self, test=False):
        if not self.objects_urdf_loaded:
            # don't reload object urdf
            self.objects_urdf_loaded = True
            self.object_bodies['target'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "target.urdf"),
                basePosition=self.object_initial_pos['target'][:3],
                baseOrientation=self.object_initial_pos['target'][3:])
            if not self.visualize_target:
                self.set_object_pose(self.object_bodies['target'],
                                     [0.0, 0.0, -3.0],
                                     self.object_initial_pos['target'][3:])

        # randomize object positions
        object_xyz_1 = None
        if self.has_obj:
            end_effector_tip_initial_position = self.robot.end_effector_tip_initial_position.copy()
            object_xy_1 = end_effector_tip_initial_position[:2]
            while np.linalg.norm(object_xy_1 - end_effector_tip_initial_position[:2]) < 0.1:
                object_xy_1 = self.np_random.uniform(self.robot.object_bound_lower[:-1],
                                                     self.robot.object_bound_upper[:-1])

            object_xyz_1 = np.append(object_xy_1, self.object_initial_pos['block'][2])
            self.set_object_pose(self.object_bodies['block'],
                                 object_xyz_1,
                                 self.object_initial_pos['block'][3:])

        # generate goals & images
        self._generate_goal(current_obj_pos=object_xyz_1)
        if self.goal_image:
            self._generate_goal_image(current_obj_pos=object_xyz_1)

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

        if not self.target_in_the_air:
            self.desired_goal[2] = self.object_initial_pos['block'][2]
        elif self.grasping:
            # with .5 probability, set the pick-and-place target on the table
            if self.np_random.uniform(0, 1) >= 0.5:
                self.desired_goal[2] = self.object_initial_pos['block'][2]

        if self.visualize_target:
            self.set_object_pose(self.object_bodies['target'],
                                 self.desired_goal,
                                 self.object_initial_pos['target'][3:])
        self.desired_joint_goal = np.array(self.robot.compute_ik(self.desired_goal))


    def _step_callback(self):
        pass

    def _get_obs(self):
        # robot state contains gripper xyz coordinates, orientation (and finger width)
        gripper_xyz, gripper_rpy, gripper_finger_closeness, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_vel, joint_poses = self.robot.calc_robot_state()
        assert self.desired_goal is not None
        policy_state = state = gripper_xyz
        achieved_goal = gripper_xyz.copy()
        if self.has_obj:
            block_xyz, _ = self._p.getBasePositionAndOrientation(self.object_bodies['block'])
            block_rel_xyz = gripper_xyz - np.array(block_xyz)
            block_vel_xyz, block_vel_rpy = self._p.getBaseVelocity(self.object_bodies['block'])
            block_rel_vel_xyz = gripper_vel_xyz - np.array(block_vel_xyz)
            block_rel_vel_rpy = gripper_vel_rpy - np.array(block_vel_rpy)
            achieved_goal = np.array(block_xyz).copy()
            # the HER paper use different state observations for the policy and critic network
            # critic further takes the velocities as input
            state = np.concatenate((gripper_xyz, block_xyz, gripper_finger_closeness, block_rel_xyz,
                                    gripper_vel_xyz, gripper_finger_vel, block_rel_vel_xyz, block_rel_vel_rpy))
            policy_state = np.concatenate((gripper_xyz, gripper_finger_closeness, block_rel_xyz))
        else:
            assert not self.grasping, "grasping should not be true when there is no objects"

        if self.joint_control:
            state = np.concatenate((joint_poses, state))
            policy_state = np.concatenate((joint_poses, policy_state))

        obs_dict = {'observation': state.copy(),
                    'policy_state': policy_state.copy(),
                    'achieved_goal': achieved_goal.copy(),
                    'desired_goal': self.desired_goal.copy(),
                    'desired_joint_goal': self.desired_joint_goal.copy(),
                    'COM': self.getCenterOfMass(),
                    'tipped_over': self.tipped_over()
                    }
        if self.image_observation:
            images = []
            for cam_id in self.observation_cam_id:
                images.append(self.render(mode=self.render_mode, camera_id=cam_id))
            obs_dict['observation'] = images[0].copy()
            obs_dict['images'] = images
            obs_dict.update({'state': state.copy()})
            if self.goal_image:
                achieved_goal_img = self.render(mode=self.render_mode, camera_id=self.goal_cam_id)
                obs_dict.update({
                    'achieved_goal_img': achieved_goal_img.copy(),
                    'desired_goal_img': self.desired_goal_image.copy(),
                })
        return obs_dict

    def _compute_reward(self, achieved_goal, desired_goal, tipped_over):
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        not_achieved = (d > self.distance_threshold)
        if tipped_over:
            d += self.tip_penalty
        if self.binary_reward:
            return -not_achieved.astype(np.float32), ~not_achieved
        else:
            return -d, ~not_achieved

    def set_object_pose(self, body_id, position, orientation=None):
        if orientation is None:
            orientation = self.object_initial_pos['table'][3:]
        self._p.resetBasePositionAndOrientation(body_id, position, orientation)

    def tipped_over(self):
        #TODO implement tipped over
        return False

    def getCenterOfMass(self):
        # Calculate the center of mass of the robot_id
        com_position = [0, 0, 0]
        joint_poses, joint_vels = self.robot.get_kuka_joint_state()
        for link_idx in range(len(joint_poses)):
            link_mass = list(self._p.getDynamicsInfo(self.robot.robot_id, link_idx))[0]
            link_com = self._p.getLinkState(self.robot.robot_id, link_idx)[0]
            link_com_position = [link_com[i] for i in range(3)]
            link_com_offset = [(link_mass/self.robot.total_mass) * link_com_position[i] for i in range(3)]
            com_position = [com_position[i] + link_com_offset[i] for i in range(3)]

        #draw com_position
        # self.drawPoint(com_position, [0,0,1])

                
        # Calculate the center of mass of the robot_id
        # com_position = [0, 0, 0]
        # print("Total mass: ", self.total_mass)
        # for link_idx in range(self.pgui.getNumJoints(self.robot_id)):
        #     print("Link mass: ",list(self.pgui.getDynamicsInfo(self.robot_id, link_idx, physicsClientId=2))[0])
        #     link_com = self.pgui.getLinkState(self.robot_id, link_idx, physicsClientId=2)[0]
        #     self.drawPoint(link_com, [0,0,1])


        return com_position