import pybullet, pybullet_data
import gym, gym.spaces, gym.utils
import numpy as np
import os


class XmlBasedRobot:
    """Base class for .xml based agents."""

    def __init__(self, robot_name, action_dim, obs_dim, self_collision=True, add_ignored_joints=False):
        self.objects = None
        self.parts = {}
        self.jdict = {}
        self.ordered_joint_names = []
        self.add_ignored_joints = add_ignored_joints

        self.action_space = gym.spaces.Box(-np.ones([action_dim]), np.ones([action_dim]))
        self.observation_space = gym.spaces.Box(-np.inf * np.ones([obs_dim]), np.inf * np.ones([obs_dim]))

        self.robot_name = robot_name
        self.self_collision = self_collision

    def addToScene(self, bullet_client, bodies):
        p = bullet_client
        if np.isscalar(bodies):  # streamline the case where bodies is actually just one body
            bodies = [bodies]
        for i in range(len(bodies)):
            for j in range(p.getNumJoints(bodies[i])):
                # p.setJointMotorControl2(bodies[i], j, pybullet.POSITION_CONTROL, positionGain=0.1,
                #                               velocityGain=0.1, force=0)
                joint_info = p.getJointInfo(bodies[i], j)
                joint_name = joint_info[1].decode("utf8")
                self.ordered_joint_names.append(joint_name)
                part_name = joint_info[12].decode("utf8")
                self.parts[part_name] = BodyPart(p, part_name, bodies, i, j)
                self.jdict[joint_name] = Joint(p, bodies, i, j, joint_info)


class URDFBasedRobot(XmlBasedRobot):
    """Base class for URDF .xml based robots."""
    def __init__(self, model_urdf, robot_name, action_dim, obs_dim, base_position=None,
                 base_orientation=None, fixed_base=False, self_collision=False):
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)
        if base_position is None:
            base_position = [0, 0, 0]
        if base_orientation is None:
            base_orientation = [0, 0, 0, 1]
        self.model_urdf = model_urdf
        self.base_position = base_position
        self.base_orientation = base_orientation
        self.fixed_base = fixed_base
        self.done_loading = 0

    def reset(self, bullet_client):
        p = bullet_client
        # load urdf if it's first time that reset() gets called
        if self.done_loading == 0:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            full_path = os.path.join(os.path.dirname(__file__), "..", "assets", "robots", self.model_urdf)
            self.done_loading = 1
            if self.self_collision:
                self.addToScene(p, p.loadURDF(full_path,
                                              basePosition=self.base_position,
                                              baseOrientation=self.base_orientation,
                                              useFixedBase=self.fixed_base,
                                              flags=pybullet.URDF_USE_SELF_COLLISION))
            else:
                self.addToScene(p, p.loadURDF(full_path,
                                              basePosition=self.base_position,
                                              baseOrientation=self.base_orientation,
                                              useFixedBase=self.fixed_base))
        # reset robot-specific configuration
        self.robot_specific_reset(p)
        # return robot-specific states
        return self.calc_state()

    def robot_specific_reset(self, bullet_client):
        # method to override, purposed to reset robot-specific configuration
        raise NotImplementedError

    def calc_state(self):
        # method to override, purposed to obtain robot-specific states
        raise NotImplementedError


class PoseHelper:  # dummy class to comply to original interface
    def __init__(self, body_part):
        self.body_part = body_part

    def xyz(self):
        return self.body_part.current_position()

    def rpy(self):
        return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

    def orientation(self):
        return self.body_part.current_orientation()

    def speed(self):
        return self.body_part.speed()


class BodyPart:
    def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
        self.bodies = bodies
        self._p = bullet_client
        self.body_name = body_name
        self.bodyIndex = bodyIndex
        self.bodyPartIndex = bodyPartIndex
        self.initialPosition = self.current_position()
        self.initialOrientation = self.current_orientation()
        self.bp_pose = PoseHelper(self)

    def state_fields_of_pose_of(self, body_id,
                                link_id=-1):  # a method you will most probably need a lot to get pose and orientation
        if link_id == -1:
            (x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
        else:
            (x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(body_id, link_id)
        return np.array([x, y, z, a, b, c, d])

    def get_pose(self):
        return self.state_fields_of_pose_of(self.bodies[self.bodyIndex], self.bodyPartIndex)

    def speed(self):
        if self.bodyPartIndex == -1:
            (vx, vy, vz), _ = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
        else:
            (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (vr, vp, vy) = self._p.getLinkState(
                self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1)
        return np.array([vx, vy, vz])

    def current_position(self):
        return self.get_pose()[:3]

    def current_orientation(self):
        return self.get_pose()[3:]

    def get_position(self):
        return self.current_position()

    def get_orientation(self):
        return self.current_orientation()

    def get_velocity(self):
        return self._p.getBaseVelocity(self.bodies[self.bodyIndex])

    def reset_position(self, position):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, self.get_orientation())

    def reset_orientation(self, orientation):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], self.get_position(), orientation)

    def reset_velocity(self, linearVelocity=None, angularVelocity=None):
        if angularVelocity is None:
            angularVelocity = [0, 0, 0]
        if linearVelocity is None:
            linearVelocity = [0, 0, 0]
        self._p.resetBaseVelocity(self.bodies[self.bodyIndex], linearVelocity, angularVelocity)

    def reset_pose(self, position, orientation):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, orientation)

    def pose(self):
        return self.bp_pose

    def contact_list(self):
        return self._p.getContactPoints(self.bodies[self.bodyIndex], -1, self.bodyPartIndex, -1)


class Joint:
    def __init__(self, bullet_client, bodies, bodyIndex, jointIndex, joint_info):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.jointIndex = jointIndex
        self.joint_name = joint_info[1].decode("utf8")
        self.jointType = joint_info[2]
        self.lowerLimit = joint_info[8]
        self.upperLimit = joint_info[9]
        self.jointHasLimits = self.lowerLimit < self.upperLimit
        self.jointMaxVelocity = joint_info[11]

    def set_state(self, x, vx):
        self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

    def current_position(self):  # just some synonym method
        return self.get_state()

    def current_relative_position(self):
        pos, vel = self.get_state()
        if self.jointHasLimits:
            pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
            pos = 2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit)

        if self.jointMaxVelocity > 0:
            vel /= self.jointMaxVelocity
        elif self.jointType == 0:  # JOINT_REVOLUTE_TYPE
            vel *= 0.1
        else:
            vel *= 0.5
        return (
            pos,
            vel
        )

    def get_state(self):
        x, vx, _, _ = self._p.getJointState(self.bodies[self.bodyIndex], self.jointIndex)
        return x, vx

    def get_position(self):
        x, _ = self.get_state()
        return x

    def get_orientation(self):
        _, r = self.get_state()
        return r

    def get_velocity(self):
        _, vx = self.get_state()
        return vx

    def set_position(self, position):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex], self.jointIndex, pybullet.POSITION_CONTROL,
                                      targetPosition=position)

    def set_velocity(self, velocity):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex], self.jointIndex, pybullet.VELOCITY_CONTROL,
                                      targetVelocity=velocity)

    def set_motor_torque(self, torque):  # just some synonym method
        self.set_torque(torque)

    def set_torque(self, torque):
        self._p.setJointMotorControl2(bodyIndex=self.bodies[self.bodyIndex], jointIndex=self.jointIndex,
                                      controlMode=pybullet.TORQUE_CONTROL,
                                      force=torque)  # , positionGain=0.1, velocityGain=0.1)

    def reset_current_position(self, position, velocity):  # just some synonym method
        self.reset_position(position, velocity)

    def reset_position(self, position, velocity):
        self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, targetValue=position,
                                targetVelocity=velocity)
        self.disable_motor()

    def disable_motor(self):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex], self.jointIndex,
                                      controlMode=pybullet.POSITION_CONTROL, targetPosition=0, targetVelocity=0,
                                      positionGain=0.1, velocityGain=0.1, force=0)
