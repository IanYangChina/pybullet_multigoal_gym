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
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
# joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


class Kuka(URDFBasedRobot):
    def __init__(self):
        URDFBasedRobot.__init__(self,
                                'kuka/iiwa14_robotiq85.urdf',
                                'body0',
                                action_dim=4,
                                obs_dim=55,
                                self_collision=True)
        # The 0-th joint is the one that connects the world frame and the kuka base, so skip it
        self.joint_index = [1, 2, 3, 4, 5, 6, 7]

    def robot_specific_reset(self, bullet_client):
        for i in range(len(self.joint_index)):
            self.jdict['iiwa_joint_'+str(self.joint_index[i])].reset_position(rp[i], 0)
        print("sth")
        pass

    def apply_action(self, a):
        assert a.shape == (4,)
        current_gripper_pose = self.parts['iiwa_link_7'].get_position()
        target_gripper_pose = current_gripper_pose + a[:-1]
        joint_poses = self.compute_ik(target_gripper_pose)
        for i in range(len(self.joint_index)):
            self._p.setJointMotorControl2(bodyIndex=0,
                                          jointIndex=self.joint_index[i],
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPosition=joint_poses[i],
                                          targetVelocity=0,
                                          force=500,
                                          positionGain=0.03,
                                          velocityGain=1)

    def calc_state(self):
        return np.array([0.0])

    def calc_reward(self):
        return 0

    def compute_ik(self, target_ee_pos):
        assert target_ee_pos.shape == (3,)
        return self._p.calculateInverseKinematics(0,
                                                  7,
                                                  target_ee_pos,
                                                  self._p.getQuaternionFromEuler([0, -math.pi, 0]),
                                                  ll, ul, jr, rp)
