from pybullet_multigoal_gym.robots.robot_bases import URDFBasedRobot
from gym import spaces
import numpy as np
import math


class Kuka(URDFBasedRobot):
    def __init__(self, grasping=False):
        URDFBasedRobot.__init__(self,
                                model_urdf='kuka/iiwa14_robotiq85.urdf',
                                robot_name='iiwa14',
                                self_collision=False)
        self.kuka_body_index = None
        self.kuka_joint_index = None
        # rest poses for null space, setting the end effector to point downward
        self.kuka_rest_pose = [0, -0.1 * math.pi, 0, 0.5 * math.pi, 0, -math.pi * 0.4, 0]
        self.end_effector_tip_joint_index = None
        self.end_effector_target = None
        self.end_effector_tip_initial_position = np.array([-0.42, 0.0, 0.40])
        self.end_effector_fixed_quaternion = [0, -1, 0, 0]
        self.robotiq_85_joint_index = None
        self.robotiq_85_joint_name = [
            'iiwa_gripper_finger1_joint',
            'iiwa_gripper_finger2_joint',
            'iiwa_gripper_finger1_inner_knuckle_joint',
            'iiwa_gripper_finger1_finger_tip_joint',
            'iiwa_gripper_finger2_inner_knuckle_joint',
            'iiwa_gripper_finger2_finger_tip_joint'
        ]
        self.robotiq_85_abs_joint_limit = 0.804
        self.robotiq_85_mimic_joint_multiplier = np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
        self.grasping = grasping
        if self.grasping:
            self.action_space = spaces.Box(-np.ones([4]), np.ones([4]))
        else:
            self.action_space = spaces.Box(-np.ones([3]), np.ones([3]))

    def robot_specific_reset(self, bullet_client):
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
        if self.robotiq_85_joint_index is None:
            self.robotiq_85_joint_index = [
                self.jdict['iiwa_gripper_finger1_joint'].jointIndex,
                self.jdict['iiwa_gripper_finger2_joint'].jointIndex,
                self.jdict['iiwa_gripper_finger1_inner_knuckle_joint'].jointIndex,
                self.jdict['iiwa_gripper_finger1_finger_tip_joint'].jointIndex,
                self.jdict['iiwa_gripper_finger2_inner_knuckle_joint'].jointIndex,
                self.jdict['iiwa_gripper_finger2_finger_tip_joint'].jointIndex,
            ]
        # reset arm poses
        for i in range(len(self.kuka_joint_index)):
            self.jdict['iiwa_joint_' + str(self.kuka_joint_index[i])].reset_position(self.kuka_rest_pose[i], 0)
        initial_joint_poses = self.compute_ik(bullet_client=bullet_client,
                                              target_ee_pos=self.end_effector_tip_initial_position)
        self.move_arm(bullet_client=bullet_client, joint_poses=initial_joint_poses)
        self.move_finger(bullet_client=bullet_client, grip_ctrl=self.robotiq_85_abs_joint_limit)
        for _ in range(20):
            bullet_client.stepSimulation()
        # obtain initial end effector coordinates in the world frame
        self.end_effector_target = self.parts['iiwa_gripper_tip'].get_position()

    def apply_action(self, a, bullet_client):
        if self.grasping:
            assert a.shape == (4,)
            grip_ctrl = (a[-1] + 1.0) * (self.robotiq_85_abs_joint_limit / 2)
            self.move_finger(bullet_client=bullet_client,
                             grip_ctrl=grip_ctrl)
        self.end_effector_target += (a[:3] * 0.01)
        joint_poses = self.compute_ik(bullet_client=bullet_client,
                                      target_ee_pos=self.end_effector_target)
        self.move_arm(bullet_client=bullet_client,
                      joint_poses=joint_poses)

    def calc_robot_state(self):
        # gripper tip coordinates in the world frame
        gripper_xyz = self.parts['iiwa_gripper_tip'].get_position()
        gripper_vel_xyz = self.parts['iiwa_gripper_tip'].get_linear_velocity()
        robot_state = np.concatenate((gripper_xyz, gripper_vel_xyz))
        if self.grasping:
            # calculate distance between the gripper finger tabs
            gripper_finger1_tab_xyz = np.array(self.parts['iiwa_gripper_finger1_finger_tab_link'].get_position())
            gripper_finger2_tab_xyz = np.array(self.parts['iiwa_gripper_finger2_finger_tab_link'].get_position())
            gripper_finger_closeness = np.sqrt(
                np.sum(np.square(gripper_finger1_tab_xyz - gripper_finger2_tab_xyz))).reshape(1, )
            robot_state = np.concatenate((robot_state, gripper_finger_closeness), axis=0)
        return robot_state

    def compute_ik(self, bullet_client, target_ee_pos, target_ee_quat=None):
        assert target_ee_pos.shape == (3,)
        if target_ee_quat is None:
            target_ee_quat = self.end_effector_fixed_quaternion
        else:
            assert target_ee_quat.shape == (4,)
        # kuka-specific values for ik computation using null space dumping method,
        #   obtained from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
        joint_poses = bullet_client.calculateInverseKinematics(
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
            restPoses=self.kuka_rest_pose)
        return joint_poses[:7]

    def move_arm(self, bullet_client, joint_poses):
        bullet_client.setJointMotorControlArray(bodyUniqueId=self.kuka_body_index,
                                                jointIndices=self.kuka_joint_index,
                                                controlMode=bullet_client.POSITION_CONTROL,
                                                targetPositions=joint_poses,
                                                targetVelocities=np.zeros((7,)),
                                                forces=np.ones((7,)) * 500,
                                                positionGains=np.ones((7,)) * 0.03,
                                                velocityGains=np.ones((7,)))

    def move_finger(self, bullet_client, grip_ctrl):
        target_joint_poses = self.robotiq_85_mimic_joint_multiplier * grip_ctrl
        bullet_client.setJointMotorControlArray(bodyUniqueId=self.kuka_body_index,
                                                jointIndices=self.robotiq_85_joint_index,
                                                controlMode=bullet_client.POSITION_CONTROL,
                                                targetPositions=target_joint_poses,
                                                targetVelocities=np.zeros((6,)),
                                                forces=np.ones((6,)) * 50,
                                                positionGains=np.ones((6,)) * 0.03,
                                                velocityGains=np.ones((6,)))

    def get_kuka_joint_state(self):
        kuka_joint_pos = []
        kuka_joint_vel = []
        for i in range(len(self.kuka_joint_index)):
            x, vx = self.jdict['iiwa_joint_' + str(self.kuka_joint_index[i])].get_state()
            kuka_joint_pos.append(x)
            kuka_joint_vel.append(vx)
        return kuka_joint_pos, kuka_joint_vel

    def set_kuka_joint_state(self, pos, vel):
        assert len(pos) == len(vel) == len(self.kuka_joint_index)
        for i in range(len(pos)):
            self.jdict['iiwa_joint_' + str(self.kuka_joint_index[i])].reset_position(pos[i], vel[i])

    def get_finger_joint_state(self):
        finger_joint_pos = []
        finger_joint_vel = []
        for name in self.robotiq_85_joint_name:
            x, vx = self.jdict[name].get_state()
            finger_joint_pos.append(x)
            finger_joint_vel.append(vx)
        return finger_joint_pos, finger_joint_vel

    def set_finger_joint_state(self, pos, vel=None):
        pos *= self.robotiq_85_mimic_joint_multiplier
        if vel is None:
            vel = np.zeros(pos.shape[0])
        for i in range(len(pos)):
            self.jdict[self.robotiq_85_joint_name[i]].reset_position(pos[i], vel[i])
