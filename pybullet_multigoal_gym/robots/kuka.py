from pybullet_multigoal_gym.robots.robot_bases import URDFBasedRobot
from gym import spaces
import numpy as np
import quaternion as quat


class Kuka(URDFBasedRobot):
    def __init__(self, bullet_client=None, gripper_type='parallel_jaw',
                 joint_control=False, grasping=False, end_effector_rotation_control=False, end_effector_force_sensor=False,
                 primitive=None, workspace_range=None, resolution=0.002,
                 end_effector_start_on_table=False, table_surface_z=0.175,
                 obj_range=0.15, target_range=0.15):
        self.gripper_type = gripper_type
        if self.gripper_type == 'robotiq85':
            model_urdf = 'robots/kuka/iiwa14_robotiq85.urdf'
        else:
            model_urdf = 'robots/kuka/iiwa14_parallel_jaw.urdf'
        URDFBasedRobot.__init__(self,
                                bullet_client=bullet_client,
                                model_urdf=model_urdf,
                                robot_name='iiwa14',
                                self_collision=False,
                                fixed_base=True)
        self.kuka_body_index = None
        self.kuka_joint_index = None
        # initial robot joint states
        self.kuka_rest_pose = [0, -0.5592432, 0, 1.733180, 0, -0.8501557, 0]
        self.kuka_away_pose = [0, 0.5467089, 0, 4.518901, 0, 0.828478, 0]
        self.joint_state_target = None
        self.end_effector_force_sensor = end_effector_force_sensor
        self.end_effector_force_sensor_enabled = False
        self.end_effector_tip_joint_index = None
        self.end_effector_target = None
        self.end_effector_target_rot = None
        self.end_effector_tip_initial_position = np.array([-0.52, 0.0, 0.25])
        self.table_surface_z = table_surface_z
        if end_effector_start_on_table:
            self.end_effector_tip_initial_position[-1] = self.table_surface_z + 0.001

        self.end_effector_xyz_upper = np.array([-0.37, 0.20, 0.55])
        self.end_effector_xyz_lower = np.array([-0.67, -0.20, self.table_surface_z])
        self.end_effector_fixed_quaternion = [0, -1, 0, 0]
        self.object_bound_lower = self.end_effector_tip_initial_position.copy() - obj_range
        self.object_bound_lower[0] += 0.03
        self.object_bound_upper = self.end_effector_tip_initial_position.copy() + obj_range
        self.object_bound_upper[0] -= 0.03
        self.target_bound_lower = self.end_effector_tip_initial_position.copy() - target_range
        self.target_bound_lower[0] += 0.03
        self.target_bound_lower[-1] = self.end_effector_xyz_lower[-1]
        self.target_bound_upper = self.end_effector_tip_initial_position.copy() + target_range
        self.target_bound_upper[0] -= 0.03

        self.gripper_joint_index = None
        if self.gripper_type == 'robotiq85':
            self.gripper_joint_name = [
                'iiwa_gripper_finger1_joint',
                'iiwa_gripper_finger2_joint',
                'iiwa_gripper_finger1_inner_knuckle_joint',
                'iiwa_gripper_finger1_finger_tip_joint',
                'iiwa_gripper_finger2_inner_knuckle_joint',
                'iiwa_gripper_finger2_finger_tip_joint'
            ]
            self.gripper_abs_joint_limit = 0.804
            self.gripper_grasp_block_state = 0.545
            self.gripper_mmic_joint_multiplier = np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
        else:
            self.gripper_joint_name = [
                'iiwa_gripper_finger1_joint',
                'iiwa_gripper_finger2_joint'
            ]
            self.gripper_abs_joint_limit = 0.035
            self.gripper_grasp_block_state = 0.02
            self.gripper_mmic_joint_multiplier = np.array([1.0, 1.0])
        self.gripper_num_joint = len(self.gripper_joint_name)
        self.gripper_tip_offset = 0.0

        # action space
        self.primitive = primitive
        self.joint_control = joint_control
        self.grasping = grasping
        self.end_effector_rotation_control = end_effector_rotation_control
        if self.primitive is not None:
            assert workspace_range is not None, "please define workspace range"
            self.workspace_range_upper = np.array(workspace_range['upper_xy'])
            self.workspace_range_lower = np.array(workspace_range['lower_xy'])
            self.workspace_range_range = self.workspace_range_upper - self.workspace_range_lower
            self.push_length = 0.1  # meters

            if self.primitive == 'discrete_push':
                self.num_angles = 20
                self.resolution = resolution  # meters per pixel
                # plus 0.0001 to bypass numpy precision issues
                self.action_map_width = int((self.workspace_range_range[0] + 0.0001) // self.resolution)
                self.action_map_height = int((self.workspace_range_range[1] + 0.0001) // self.resolution)
                self.action_space = spaces.MultiDiscrete([self.num_angles,
                                                          self.action_map_width,
                                                          self.action_map_height])
            elif self.primitive == 'continuous_push':
                # action = (x, y, angle)
                self.action_space = spaces.Box(-np.ones([3]), np.ones([3]))
            else:
                raise ValueError("Other primitives not supported atm.")
        elif self.joint_control:
            if self.grasping:
                self.action_space = spaces.Box(-np.ones([8]), np.ones([8]))
            else:
                self.action_space = spaces.Box(-np.ones([7]), np.ones([7]))
        else:
            if self.grasping:
                if self.end_effector_rotation_control:
                    self.action_space = spaces.Box(-np.ones([7]), np.ones([7]))
                else:
                    self.action_space = spaces.Box(-np.ones([4]), np.ones([4]))
            else:
                if self.end_effector_rotation_control:
                    self.action_space = spaces.Box(-np.ones([6]), np.ones([6]))
                else:
                    self.action_space = spaces.Box(-np.ones([3]), np.ones([3]))

    def robot_specific_reset(self):
        if self.kuka_body_index is None:
            self.kuka_body_index = self.jdict['plane_iiwa_joint'].bodies[self.jdict['plane_iiwa_joint'].bodyIndex]
        if self.kuka_joint_index is None:
            # The 0-th joint is the one that connects the world frame and the kuka base, so skip it
            self.kuka_joint_index = [
                self.jdict['iiwa_joint_1'].jointIndex,
                self.jdict['iiwa_joint_2'].jointIndex,
                self.jdict['iiwa_joint_3'].jointIndex,
                self.jdict['iiwa_joint_4'].jointIndex,
                self.jdict['iiwa_joint_5'].jointIndex,
                self.jdict['iiwa_joint_6'].jointIndex,
                self.jdict['iiwa_joint_7'].jointIndex,
            ]
        if self.end_effector_tip_joint_index is None:
            self.end_effector_tip_joint_index = self.jdict['iiwa_gripper_tip_joint'].jointIndex
        if self.gripper_joint_index is None:
            if self.gripper_type == 'robotiq85':
                self.gripper_joint_index = [
                    self.jdict['iiwa_gripper_finger1_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger2_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger1_inner_knuckle_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger1_finger_tip_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger2_inner_knuckle_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger2_finger_tip_joint'].jointIndex,
                ]
            else:
                self.gripper_joint_index = [
                    self.jdict['iiwa_gripper_finger1_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger2_joint'].jointIndex,
                ]
        if not self.end_effector_force_sensor_enabled:
            self._p.enableJointForceTorqueSensor(bodyUniqueId=self.kuka_body_index,
                                                 jointIndex=self.jdict['iiwa_joint_7'].jointIndex,
                                                 enableSensor=self.end_effector_force_sensor)
            self.end_effector_force_sensor_enabled = True

        # reset arm poses
        self.set_kuka_joint_state(self.kuka_rest_pose)
        self.kuka_rest_pose = self.compute_ik(self.end_effector_tip_initial_position)
        self.set_kuka_joint_state(self.kuka_rest_pose)
        self.set_finger_joint_state(self.gripper_abs_joint_limit)
        self.move_finger(grip_ctrl=self.gripper_abs_joint_limit)
        self.end_effector_target = self.parts['iiwa_gripper_tip'].get_position()
        self.end_effector_target_rot = self.parts['iiwa_gripper_tip'].get_orientation_eular()
        self.joint_state_target, _ = self.get_kuka_joint_state()

    def apply_action(self, a):
        assert self.action_space.contains(a)
        if self.grasping:
            # map action in [-1, 1] to gripper joint range
            grip_ctrl = (a[-1] + 1.0) * (self.gripper_abs_joint_limit / 2)
            self.move_finger(grip_ctrl=grip_ctrl)
        if self.primitive is not None:
            if self.primitive == 'discrete_push':
                push_start_x = (a[1] * self.resolution) + self.workspace_range_lower[0]
                push_start_y = (a[2] * self.resolution) + self.workspace_range_lower[1]
                push_angle = 2 * np.pi / self.num_angles * a[0]
            elif self.primitive == 'continuous_push':
                push_start_x = self.workspace_range_lower[0] + (self.workspace_range_range[0] * (a[0] + 1) / 2)
                push_start_y = self.workspace_range_lower[1] + (self.workspace_range_range[1] * (a[1] + 1) / 2)
                push_angle = a[2] * np.pi / 2
            else:
                raise ValueError("Other primitives not supported atm.")
            # calculate waypoints
            delta_x = np.cos(push_angle) * self.push_length
            delta_y = np.sin(push_angle) * self.push_length
            push_end_x = np.clip(push_start_x + delta_x, self.workspace_range_lower[0], self.workspace_range_upper[0])
            push_end_y = np.clip(push_start_y + delta_y, self.workspace_range_lower[1], self.workspace_range_upper[1])
            primitive_ee_waypoints = [
                np.array([push_start_x, push_start_y, self.table_surface_z + 0.1]),
                np.array([push_start_x, push_start_y, self.table_surface_z + 0.01]),
                np.array([push_end_x, push_end_y, self.table_surface_z + 0.01]),
                np.array([push_end_x, push_end_y, self.table_surface_z + 0.1])
            ]
            # waypoint visualisation for debugging
            # for i in range(len(primitive_ee_waypoints)):
            #     target_name = self.target_keys[i]
            #     self.set_object_pose(self.target_bodies[target_name], primitive_ee_waypoints[i])
            # delta_d = np.linalg.norm(np.array([push_start_x, push_start_y, self.table_surface_z + 0.025]) -
            #                          np.array([push_end_x, push_end_y, self.table_surface_z + 0.025]))
            # assert (delta_d - self.push_length) <= 0.02
            self.execute_primitive(primitive_ee_waypoints)
        else:
            if self.joint_control:
                self.joint_state_target = (a[:7] * 0.05) + self.joint_state_target
                joint_poses = self.joint_state_target.copy()
            else:
                # actions alter the ee target pose
                self.end_effector_target += (a[:3] * 0.01)
                self.end_effector_target = np.clip(self.end_effector_target,
                                                   self.end_effector_xyz_lower,
                                                   self.end_effector_xyz_upper)
                if not self.end_effector_rotation_control:
                    joint_poses = self.compute_ik(target_ee_pos=self.end_effector_target)
                else:
                    self.end_effector_target_rot += (a[3:6] * 0.05)
                    # quat.from_euler_angles --> (a, b, c, w)
                    target_ee_quat = quat.as_float_array(quat.from_euler_angles(self.end_effector_target_rot))
                    joint_poses = self.compute_ik(target_ee_pos=self.end_effector_target,
                                                  target_ee_quat=target_ee_quat)

            self.move_arm(joint_poses=joint_poses)
            for _ in range(5):
                # ensure the action is finished
                self._p.stepSimulation()

    def calc_robot_state(self):
        # gripper tip states in the world frame
        gripper_xyz = self.parts['iiwa_gripper_tip'].get_position()
        gripper_rpy = self.parts['iiwa_gripper_tip'].get_orientation_eular()
        gripper_vel_xyz = self.parts['iiwa_gripper_tip'].get_linear_velocity()
        gripper_vel_rpy = self.parts['iiwa_gripper_tip'].get_angular_velocity()
        if self.grasping:
            # calculate distance between the gripper finger tabs
            gripper_finger1_tab_xyz = np.array(self.parts['iiwa_gripper_finger1_finger_tab_link'].get_position())
            gripper_finger2_tab_xyz = np.array(self.parts['iiwa_gripper_finger2_finger_tab_link'].get_position())
            gripper_finger_closeness = np.sqrt(
                np.sum(np.square(gripper_finger1_tab_xyz - gripper_finger2_tab_xyz))).ravel()
            # calculate finger joint velocity instead of using a get() method due to compatibility among grippers
            grip_base_vel = self.parts['iiwa_gripper_base_link'].get_linear_velocity()
            grip_finger_vel = self.parts['iiwa_gripper_finger1_finger_tab_link'].get_linear_velocity()
            gripper_finger_vel = (grip_base_vel - grip_finger_vel)[1].ravel()
        else:
            # symmetric gripper
            gripper_finger_closeness = np.array([0.0])
            gripper_finger_vel = np.array([0.0])

        joint_poses, _ = self.get_kuka_joint_state()

        if self.end_effector_force_sensor:
            (fx, fy, fz, mx, my, mz) = self.jdict['iiwa_joint_7'].get_force()
            # fz += 22.10853  # compensate gravity force
            ee_joint_fx = np.clip(np.array([fx, fy, fz]), -50.0, 50.0)
            return gripper_xyz, gripper_rpy, gripper_finger_closeness, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_vel, joint_poses, ee_joint_fx
        else:
            return gripper_xyz, gripper_rpy, gripper_finger_closeness, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_vel, joint_poses

    def compute_ik(self, target_ee_pos, target_ee_quat=None):
        assert target_ee_pos.shape == (3,)
        if target_ee_quat is None:
            target_ee_quat = self.end_effector_fixed_quaternion
        else:
            assert target_ee_quat.shape == (4,)
        # kuka-specific values for ik computation using null space dumping method,
        #   obtained from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
        joint_poses = self._p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_body_index,
            endEffectorLinkIndex=self.end_effector_tip_joint_index,
            targetPosition=target_ee_pos,
            targetOrientation=target_ee_quat,
            # lower limits for null space
            lowerLimits=[-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05],
            # upper limits for null space
            upperLimits=[.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05],
            # joint ranges for null space
            jointRanges=[5.8, 4, 5.8, 4, 5.8, 4, 6],
            restPoses=self.kuka_rest_pose,
            maxNumIterations=40,
            residualThreshold=0.00001)
        return joint_poses[:7]

    def move_arm(self, joint_poses):
        self._p.setJointMotorControlArray(bodyUniqueId=self.kuka_body_index,
                                          jointIndices=self.kuka_joint_index,
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPositions=joint_poses,
                                          targetVelocities=np.zeros((7,)),
                                          forces=np.ones((7,)) * 200,
                                          positionGains=np.ones((7,)) * 0.03,
                                          velocityGains=np.ones((7,)))

    def move_finger(self, grip_ctrl):
        target_joint_poses = self.gripper_mmic_joint_multiplier * grip_ctrl
        self._p.setJointMotorControlArray(bodyUniqueId=self.kuka_body_index,
                                          jointIndices=self.gripper_joint_index,
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPositions=target_joint_poses,
                                          targetVelocities=np.zeros((self.gripper_num_joint,)),
                                          forces=np.ones((self.gripper_num_joint,)) * 50,
                                          positionGains=np.ones((self.gripper_num_joint,)) * 0.03,
                                          velocityGains=np.ones((self.gripper_num_joint,)))

    def execute_primitive(self, ee_waypoints):
        # execute primitive
        self.set_kuka_joint_state(self.kuka_rest_pose)
        for ee_waypoint in ee_waypoints:
            joint_waypoint = self.compute_ik(ee_waypoint)
            self.move_arm(joint_waypoint)
            for _ in range(20):
                # ensure the action is finished
                self._p.stepSimulation()
        self.set_kuka_joint_state(self.kuka_rest_pose)

    def set_object_pose(self, body_id, position, orientation=None):
        if orientation is None:
            orientation = [0.0, 0.0, 0.0, 1.0]
        self._p.resetBasePositionAndOrientation(body_id, position, orientation)

    def get_kuka_joint_state(self):
        kuka_joint_pos = []
        kuka_joint_vel = []
        for i in range(len(self.kuka_joint_index)):
            x, vx, _ = self.jdict['iiwa_joint_' + str(self.kuka_joint_index[i])].get_state()
            kuka_joint_pos.append(x)
            kuka_joint_vel.append(vx)
        return kuka_joint_pos, kuka_joint_vel

    def set_kuka_joint_state(self, pos=None, vel=None, gripper_tip_pos=None):
        if gripper_tip_pos is not None:
            pos = self.compute_ik(target_ee_pos=gripper_tip_pos)
        pos = np.array(pos)
        if vel is None:
            vel = np.zeros(pos.shape[0])
        for i in range(len(pos)):
            self.jdict['iiwa_joint_' + str(self.kuka_joint_index[i])].reset_position(pos[i], vel[i])

    def get_finger_joint_state(self):
        finger_joint_pos = []
        finger_joint_vel = []
        for name in self.gripper_joint_name:
            x, vx = self.jdict[name].get_state()
            finger_joint_pos.append(x)
            finger_joint_vel.append(vx)
        return finger_joint_pos, finger_joint_vel

    def set_finger_joint_state(self, pos, vel=None):
        pos = pos * self.gripper_mmic_joint_multiplier
        if vel is None:
            vel = np.zeros(pos.shape[0])
        for i in range(pos.shape[0]):
            self.jdict[self.gripper_joint_name[i]].reset_position(pos[i], vel[i])

    def get_finger_closeness(self):
        # calculate distance between the gripper finger tabs
        gripper_finger1_tab_xyz = np.array(self.parts['iiwa_gripper_finger1_finger_tab_link'].get_position())
        gripper_finger2_tab_xyz = np.array(self.parts['iiwa_gripper_finger2_finger_tab_link'].get_position())
        gripper_finger_closeness = np.sqrt(
            np.sum(np.square(gripper_finger1_tab_xyz - gripper_finger2_tab_xyz))).ravel()
        return gripper_finger_closeness
