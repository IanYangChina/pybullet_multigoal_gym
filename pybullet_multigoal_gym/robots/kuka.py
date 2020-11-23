from pybullet_multigoal_gym.robots.robot_bases import URDFBasedRobot
import numpy as np
import math


class Kuka(URDFBasedRobot):
    def __init__(self):
        URDFBasedRobot.__init__(self,
                                model_urdf='kuka/iiwa14_robotiq85.urdf',
                                robot_name='iiwa14',
                                action_dim=4,
                                obs_dim=55,
                                self_collision=False)
        self.kuka_body_index = None
        self.kuka_joint_index = None
        # rest poses for null space, setting the end effector to point downward
        self.kuka_rest_pose = [0, -0.1 * math.pi, 0, 0.5 * math.pi, 0, -math.pi * 0.4, 0]
        self.end_effector_index = None
        self.end_effector_target = None
        self.robotiq_85_joint_index = None
        self.robotiq_85_abs_joint_limit = 0.804

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
        if self.end_effector_index is None:
            self.end_effector_index = self.jdict['iiwa_gripper_tip_joint'].jointIndex
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
        # obtain initial end effector coordinates in the world frame
        self.end_effector_target = self.parts['iiwa_gripper_tip'].get_position()

    def apply_action(self, a, bullet_client):
        assert a.shape == (4,)
        p = bullet_client
        self.end_effector_target += (a[:-1] * 0.01)
        # self.parts['target'].reset_position(self.end_effector_target)
        joint_poses = self.compute_ik(self.end_effector_target, p)
        p.setJointMotorControlArray(bodyUniqueId=self.kuka_body_index,
                                    jointIndices=self.kuka_joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_poses[:7],
                                    targetVelocities=np.zeros((7,)),
                                    forces=np.ones((7,)) * 500,
                                    positionGains=np.ones((7,)) * 0.03,
                                    velocityGains=np.ones((7,)))
        grip_ctrl = (a[-1] + 1.0) * (self.robotiq_85_abs_joint_limit / 2)
        grip_ctrl = np.array([grip_ctrl, grip_ctrl, grip_ctrl, -grip_ctrl, grip_ctrl, -grip_ctrl])
        p.setJointMotorControlArray(bodyUniqueId=self.kuka_body_index,
                                    jointIndices=self.robotiq_85_joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=grip_ctrl,
                                    targetVelocities=np.zeros((6,)),
                                    forces=np.ones((6,)) * 500,
                                    positionGains=np.ones((6,)) * 0.03,
                                    velocityGains=np.ones((6,)))

    def calc_state(self):
        # gripper tip coordinates in the world frame
        gripper_tip_xyz = self.parts['iiwa_gripper_tip'].get_position()
        # calculate distance between the gripper finger tabs
        gripper_finger1_tab_xyz = np.array(self.parts['iiwa_gripper_finger1_finger_tab_link'].get_position())
        gripper_finger2_tab_xyz = np.array(self.parts['iiwa_gripper_finger2_finger_tab_link'].get_position())
        gripper_finger_closeness = np.sqrt(
            np.square(np.sum(gripper_finger1_tab_xyz - gripper_finger2_tab_xyz))).reshape(1, )
        return np.concatenate((gripper_tip_xyz, gripper_finger_closeness), axis=0)

    def calc_reward(self):
        # method to override, purposed to compute task-specific rewards
        # raise NotImplementedError
        pass

    def compute_ik(self, target_ee_pos, bullet_client):
        assert target_ee_pos.shape == (3,)
        # kuka-specific values for ik computation using null space dumping method,
        #   obtained from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
        joint_poses = bullet_client.calculateInverseKinematics(
            bodyUniqueId=self.kuka_body_index,
            endEffectorLinkIndex=self.end_effector_index,
            targetPosition=target_ee_pos,
            targetOrientation=bullet_client.getQuaternionFromEuler([0, -math.pi, 0]),
            # lower limits for null space
            lowerLimits=[-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05],
            # upper limits for null space
            upperLimits=[.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05],
            # joint ranges for null space
            jointRanges=[5.8, 4, 5.8, 4, 5.8, 4, 6],
            restPoses=self.kuka_rest_pose)
        return joint_poses[:7]
