from pybullet_multigoal_gym.robots.robot_bases import MJCFBasedRobot, URDFBasedRobot
import numpy as np
import math

# kuka-specific values for ik computation and reset robot pose,
#   obtained from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
# lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space, setting the end effector to point downward
rp = [0, -0.1 * math.pi, 0, 0.5 * math.pi, 0, -math.pi * 0.4, 0]
# joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


class Kuka(URDFBasedRobot):
    def __init__(self):
        URDFBasedRobot.__init__(self,
                                'kuka/iiwa14_robotiq85.urdf',
                                'body0',
                                action_dim=4,
                                obs_dim=55,
                                self_collision=False)
        self.kuka_joint_index = None
        self.end_effector_index = None
        self.end_effector_target = None
        self.robotiq_85_joint_index = None
        self.robotiq_85_abs_joint_limit = 0.804

    def robot_specific_reset(self, bullet_client):
        if self.kuka_joint_index is None:
            # The 0-th joint is the one that connects the world frame and the kuka base, so skip it
            self.kuka_joint_index = [self.jdict['iiwa_joint_'+str(i)].jointIndex for i in [1, 2, 3, 4, 5, 6, 7]]
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
        for i in range(len(self.kuka_joint_index)):
            self.jdict['iiwa_joint_'+str(self.kuka_joint_index[i])].reset_position(rp[i], 0)
        self.end_effector_target = self.parts['iiwa_gripper_tip'].get_position()

    def apply_action(self, a):
        assert a.shape == (4,)
        self.end_effector_target += (a[:-1] * 0.01)
        # self.parts['target'].reset_position(self.end_effector_target)
        joint_poses = self.compute_ik(self.end_effector_target)
        self._p.setJointMotorControlArray(bodyUniqueId=self.jdict['iiwa_joint_7'].bodies[self.jdict['iiwa_joint_7'].bodyIndex],
                                          jointIndices=self.kuka_joint_index,
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPositions=joint_poses[:7],
                                          targetVelocities=np.zeros((7,)),
                                          forces=np.ones((7,))*500,
                                          positionGains=np.ones((7,))*0.03,
                                          velocityGains=np.ones((7,)))
        grip_ctrl = (a[-1] + 1.0) * (self.robotiq_85_abs_joint_limit / 2)
        self._p.setJointMotorControlArray(bodyUniqueId=self.jdict['iiwa_joint_7'].bodies[self.jdict['iiwa_joint_7'].bodyIndex],
                                    jointIndices=self.robotiq_85_joint_index,
                                    controlMode=self._p.POSITION_CONTROL,
                                    targetPositions=np.array([grip_ctrl, grip_ctrl, grip_ctrl, -grip_ctrl, grip_ctrl, -grip_ctrl]),
                                    targetVelocities=np.zeros((6,)),
                                    forces=np.ones((6,))*500,
                                    positionGains=np.ones((6,))*0.03,
                                    velocityGains=np.ones((6,)))

    def calc_state(self):
        return np.array([0.0])

    def calc_reward(self):
        return 0

    def compute_ik(self, target_ee_pos):
        assert target_ee_pos.shape == (3,)
        joint_poses = self._p.calculateInverseKinematics(
            bodyUniqueId=self.jdict['iiwa_joint_7'].bodies[self.jdict['iiwa_joint_7'].bodyIndex],
            endEffectorLinkIndex=self.end_effector_index,
            targetPosition=target_ee_pos,
            targetOrientation=self._p.getQuaternionFromEuler([0, -math.pi, 0]),
            lowerLimits=ll,
            upperLimits=ul,
            jointRanges=jr,
            restPoses=rp)
        return joint_poses[:7]
